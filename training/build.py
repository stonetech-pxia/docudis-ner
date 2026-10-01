"""Builds the NER training set from templates, hand-written documents and registry records.

    python training/build.py [--chunks 30000] [--seed 1] [--out training/out]

Inputs (see training/GUIDE.md): training/inventories/<COUNTRY>.json, training/templates/<lang>/*.txt,
training/documents/<lang>/*.txt (inline markup), training/out/registry.jsonl (training/fetch_registry.py).
Output: <out>/train.jsonl, <out>/dev.jsonl, <out>/build_report.md. One record per document:
{"id", "lang", "country", "source", "doctype", "text", "spans": [{"start", "end", "label"}]}; the trainer cuts
records into 256-token chunks. Mix, in estimated chunks (900 characters each): templates 60 %, registry 25 %,
hand-written documents 15 % (oversampled with name substitution), languages equal, and within a language
GB 60 / US 20 / IE 20, FR 60 / BE 20 / CH 20, ES 100. 10 % of template files, document files and registry
records go to dev, whole. Every record passes training/check_isolation.py or is dropped.
"""
import argparse
import collections
import glob
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, 'tool')]
import check_isolation  # noqa: E402
import markup  # noqa: E402
import valid_ids  # noqa: E402

CHARS_PER_CHUNK = 900
MIX = {'template': 0.60, 'registry': 0.25, 'document': 0.15}
COUNTRY_SHARE = {'en': {'GB': .6, 'US': .2, 'IE': .2}, 'fr': {'FR': .6, 'BE': .2, 'CH': .2}, 'es': {'ES': 1.0}}
LANG_OF = {c: l for l, cs in COUNTRY_SHARE.items() for c in cs}

MONTHS = {
    'en': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
    'fr': ['janvier', 'février', 'mars', 'avril', 'mai', 'juin', 'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre'],
    'es': ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre'],
}
SHORT = {'en': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'],
         'fr': ['janv.', 'févr.', 'mars', 'avr.', 'mai', 'juin', 'juil.', 'août', 'sept.', 'oct.', 'nov.', 'déc.'],
         'es': ['ene.', 'feb.', 'mar.', 'abr.', 'may.', 'jun.', 'jul.', 'ago.', 'sep.', 'oct.', 'nov.', 'dic.']}
COUNTRY_NAME = {'en': {'GB': ['United Kingdom', 'UK', 'England', 'Scotland', 'Wales'], 'IE': ['Ireland'], 'US': ['USA', 'United States']},
                'fr': {'FR': ['France'], 'BE': ['Belgique'], 'CH': ['Suisse']}, 'es': {'ES': ['España']}}
US_STATES = {'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA', 'Colorado': 'CO',
             'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI',
             'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA',
             'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
             'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
             'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
             'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
             'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
             'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'}
TLD = {'GB': 'co.uk', 'IE': 'ie', 'US': 'com', 'FR': 'fr', 'BE': 'be', 'CH': 'ch', 'ES': 'es'}


def fold(s):
    import unicodedata
    s = unicodedata.normalize('NFKD', s)
    return re.sub(r'[^a-z0-9]+', '', ''.join(c for c in s if not unicodedata.combining(c)).lower())


# ---------------------------------------------------------------- identifiers (O)

def mod97_pair(body):
    return f'{97 - int(body) % 97:02d}'


class Numbers:
    def __init__(self, rng, country):
        self.r, self.c = rng, country

    def d(self, n):
        return ''.join(self.r.choice('0123456789') for _ in range(n))

    def phone(self, mobile):
        r, c, d = self.r, self.c, self.d
        intl = r.random() < 0.15
        if c == 'GB':
            return (f'+44 7{d(3)} {d(6)}' if intl else f'07{d(3)} {d(6)}') if mobile else \
                r.choice([f'01{d(3)} {d(6)}', f'020 {d(4)} {d(4)}', f'0{r.choice("12")}{d(2)} {d(3)} {d(4)}'])
        if c == 'IE':
            return (f'+353 8{r.choice("35679")} {d(3)} {d(4)}' if intl else f'08{r.choice("35679")} {d(3)} {d(4)}') if mobile else \
                r.choice([f'01 {d(3)} {d(4)}', f'0{r.choice("245679")}{d(1)} {d(3)} {d(4)}'])
        if c == 'US':
            a = f'{r.randint(201, 989)}'
            return r.choice([f'({a}) {r.randint(200, 999)}-{d(4)}', f'{a}-{r.randint(200, 999)}-{d(4)}',
                             f'+1 {a} {r.randint(200, 999)} {d(4)}', f'{a}.{r.randint(200, 999)}.{d(4)}'])
        if c == 'FR':
            first = r.choice('67') if mobile else r.choice('12345')
            sep = r.choice([' ', ' ', '.', ''])
            return f'+33 {first} {d(2)} {d(2)} {d(2)} {d(2)}' if intl else sep.join([f'0{first}'] + [d(2) for _ in range(4)])
        if c == 'BE':
            return (f'+32 4{d(2)} {d(2)} {d(2)} {d(2)}' if intl else f'04{d(2)} {d(2)} {d(2)} {d(2)}') if mobile else \
                f'0{r.choice(["2", "4", "10", "65", "81"])} {d(3)} {d(2)} {d(2)}'
        if c == 'CH':
            return (f'+41 7{r.choice("6789")} {d(3)} {d(2)} {d(2)}' if intl else f'07{r.choice("6789")} {d(3)} {d(2)} {d(2)}') \
                if mobile else f'0{r.choice(["21", "22", "24", "26", "27", "32"])} {d(3)} {d(2)} {d(2)}'
        first = r.choice('67') if mobile else r.choice('89')
        body = f'{first}{d(2)} {d(2)} {d(2)} {d(2)}' if r.random() < .6 else f'{first}{d(2)} {d(3)} {d(3)}'
        return f'+34 {body}' if intl else body

    def iban(self):
        r, c, d = self.r, self.c, self.d
        if c == 'GB':
            return valid_ids.gb_iban(r)
        if c == 'FR':
            return valid_ids.fr_iban(r)
        if c == 'ES':
            return valid_ids.es_iban(r)
        if c == 'IE':
            return valid_ids.grouped(valid_ids.iban('IE', r.choice(['AIBK', 'BOFI', 'IPBS', 'ULSB']) + d(6) + d(8)))
        if c == 'BE':
            body = d(10)
            return valid_ids.grouped(valid_ids.iban('BE', body + f'{int(body) % 97 or 97:02d}'))
        if c == 'CH':
            return valid_ids.grouped(valid_ids.iban('CH', d(5) + d(12)))
        return None  # US: no IBAN

    def bic(self):
        r = self.r
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        base = ''.join(r.choice(letters) for _ in range(4)) + self.c + r.choice(letters) + r.choice('0123456789' + letters)
        return base + (''.join(r.choice(letters + '0123456789') for _ in range(3)) if r.random() < .4 else '')

    def natid(self):
        r, c, d = self.r, self.c, self.d
        if c == 'GB':
            return r.choice([valid_ids.nino(r), valid_ids.nhs(r)])
        if c == 'IE':
            digits = d(7)
            second = r.choice(['', 'A', 'H', 'W'])
            total = sum(int(x) * w for x, w in zip(digits, range(8, 1, -1))) + (ord(second) - 64) * 9 if second else \
                sum(int(x) * w for x, w in zip(digits, range(8, 1, -1)))
            return f'{digits}{"WABCDEFGHIJKLMNOPQRSTUV"[total % 23]}{second}'
        if c == 'US':
            return valid_ids.us_ssn(r)
        if c == 'FR':
            return valid_ids.nir(r)
        if c == 'BE':
            yy, mm, dd, seq = r.randint(40, 99), r.randint(1, 12), r.randint(1, 28), r.randint(1, 997)
            body = f'{yy:02d}{mm:02d}{dd:02d}{seq:03d}'
            return f'{body[0:2]}.{body[2:4]}.{body[4:6]}-{body[6:9]}.{mod97_pair(body)}'
        if c == 'CH':
            body = '756' + d(9)
            check = (10 - sum(int(x) * (3 if i % 2 else 1) for i, x in enumerate(body)) % 10) % 10
            return f'756.{body[3:7]}.{body[7:11]}.{body[11]}{check}'
        return r.choice([valid_ids.dni(r), valid_ids.nie(r), valid_ids.es_nss(r)])

    def companyid(self):
        r, c, d = self.r, self.c, self.d
        if c == 'GB':
            return r.choice(['0', 'SC', 'NI']) + d(6 if r.random() < .5 else 7)
        if c == 'IE':
            return d(6)
        if c == 'US':
            return f'{d(2)}-{d(7)}'
        if c == 'FR':
            s = valid_ids.siren(r)
            return r.choice([f'{s[:3]} {s[3:6]} {s[6:]}', s, valid_ids.siret(r)])
        if c == 'BE':
            body = '0' + d(7)
            return f'{body[:4]}.{body[4:7]}.{body[7]}{mod97_pair(body)}'
        if c == 'CH':
            body = d(8)
            check = (11 - sum(int(x) * w for x, w in zip(body, [5, 4, 3, 2, 7, 6, 5, 4])) % 11) % 11
            if check == 10:
                return self.companyid()
            full = body + str(check)
            return f'CHE-{full[0:3]}.{full[3:6]}.{full[6:9]}'
        letter = r.choice('ABEFGHJ')
        digits = d(7)
        even = sum(int(x) for x in digits[1::2])
        odd = sum(sum(divmod(int(x) * 2, 10)) for x in digits[0::2])
        return f'{letter}{digits}{(10 - (even + odd) % 10) % 10}'

    def vat(self):
        c = self.c
        if c == 'GB':
            return f'GB {self.d(3)} {self.d(4)} {self.d(2)}'
        if c == 'IE':
            return f'IE{self.d(7)}{self.r.choice("ABCDEFGHW")}'
        if c == 'FR':
            s = valid_ids.siren(self.r)
            return f'FR{(12 + 3 * (int(s) % 97)) % 97:02d}{s}'
        if c == 'BE':
            return 'BE' + self.companyid().replace('.', '')
        if c == 'CH':
            return self.companyid() + self.r.choice([' TVA', ' MWST', ' IVA'])
        if c == 'ES':
            return 'ES' + self.companyid()
        return self.companyid()

    def taxid(self):
        c = self.c
        if c == 'GB':
            return self.d(10)
        if c == 'FR':
            return f'{self.r.choice("0123")}{self.d(12)}'
        if c == 'ES':
            return valid_ids.dni(self.r)
        return self.natid()

    def plate(self):
        r, d = self.r, self.d
        L = lambda n: ''.join(r.choice('ABCDEFGHJKLMNPRSTVWXYZ') for _ in range(n))  # noqa: E731
        return {'GB': f'{L(2)}{d(2)} {L(3)}', 'IE': f'{r.randint(12, 26)}{r.choice("12")}-{r.choice(["D", "C", "G", "KY", "L"])}-{r.randint(1, 99999)}',
                'US': f'{d(1)}{L(3)}{d(3)}', 'FR': f'{L(2)}-{d(3)}-{L(2)}', 'BE': f'1-{L(3)}-{d(3)}',
                'CH': f'{r.choice(["VD", "GE", "FR", "NE", "VS"])} {r.randint(1000, 999999)}', 'ES': f'{d(4)} {L(3)}'}[self.c]

    def ref(self):
        r, d = self.r, self.d
        pre = r.choice(['INV', 'REF', 'ORD', 'CL', 'DOS', 'FAC', 'EXP', 'SIN', 'CT', 'N', 'PO', 'TKT', ''])
        return r.choice([f'{pre}-{r.randint(2019, 2026)}-{d(5)}', f'{pre}{d(7)}', f'{d(3)}/{d(4)}/{d(2)}', f'{pre}/{d(2)}/{d(6)}',
                         f'{d(10)}']).lstrip('-/')

    def account(self):
        return self.d({'GB': 8, 'IE': 8, 'US': self.r.randint(9, 12)}.get(self.c, 11))

    def amount(self, big=False):
        r = self.r
        v = r.uniform(5, 250000) if big else r.choice([r.uniform(2, 200), r.uniform(20, 3000)])
        cents = r.random() < .85
        whole = int(v)
        s = f'{v:,.2f}' if cents else f'{whole:,}'
        c = self.c
        if c in ('GB', 'US', 'IE'):
            sym = {'GB': '£', 'US': '$', 'IE': '€'}[c]
            return r.choice([f'{sym}{s}', f'{sym}{s}', f'{s} {"GBP" if c == "GB" else "USD" if c == "US" else "EUR"}'])
        euro = s.replace(',', '§').replace('.', ',')
        if c == 'FR':
            return f'{euro.replace("§", r.choice([" ", " ", "."]))} {r.choice(["€", "€", "euros", "EUR"])}'
        if c == 'BE':
            return f'{euro.replace("§", ".")} {r.choice(["€", "EUR"])}'
        if c == 'CH':
            ch = s.replace(',', "'")
            return f'CHF {ch}' if r.random() < .6 else f'{ch} CHF'
        return f'{euro.replace("§", ".")} {r.choice(["€", "€", "euros", "EUR"])}'


# ---------------------------------------------------------------- entities (labelled)

class Entities:
    def __init__(self, inv, rng, lang, country):
        self.inv, self.r, self.lang, self.c = inv, rng, lang, country
        self.people, self.orgs, self.locs = {}, {}, {}
        self.org_seen = collections.Counter()
        self.used = set()

    def pick(self, seq):
        return self.r.choice(seq)

    # people
    def person(self, key):
        if key not in self.people:
            r = self.r
            g = self.pick(self.inv['given_names'])
            surname = self.pick(self.inv['surnames'])['name']
            if self.c == 'ES' or (self.lang == 'es') or (self.c in ('US',) and r.random() < .05):
                surname = f"{surname} {self.pick(self.inv['surnames'])['name']}"
            elif r.random() < .06:
                surname = f"{surname}-{self.pick(self.inv['surnames'])['name']}"
            given = g['name']
            if r.random() < .12 and self.c != 'ES':
                given2 = self.pick([x for x in self.inv['given_names'] if x['gender'] == g['gender']] or [g])['name']
                if given2 != given:
                    given = f'{given}-{given2}' if self.lang == 'fr' else f'{given} {given2}'
            style = r.choices(['normal', 'caps_surname', 'comma', 'caps', 'middle'], [60, 14, 8, 12, 6])[0]
            self.people[key] = {'given': given, 'surname': surname, 'gender': g['gender'], 'style': style,
                                'middle': r.choice('ABCDEFGHJKLMNPRSTW')}
        return self.people[key]

    def person_form(self, key, form):
        p = self.person(key)
        g, s, st = p['given'], p['surname'], p['style']
        if form in (None, 'full', 'signature'):
            if st == 'caps_surname':
                return f'{s.upper()} {g}'
            if st == 'comma':
                return f'{s}, {g}'
            if st == 'caps':
                return f'{g} {s}'.upper()
            if st == 'middle' and self.lang == 'en':
                return f"{g} {p['middle']}. {s}"
            return f'{g} {s}'
        if form == 'given':
            return g
        if form == 'surname':
            return s.upper() if st in ('caps_surname', 'caps') else s
        if form == 'initial_surname':
            initials = ''.join(part[0] + '.' for part in re.split(r'[ -]', g) if part)
            return f'{initials} {s}'
        if form == 'surname_given':
            return f'{s} {g}'.upper() if self.lang in ('es', 'fr') or self.r.random() < .5 else f'{s.upper()}, {g}'
        raise ValueError(form)

    def title(self, key):
        p = self.person(key)
        titles = self.inv['titles'].get(p['gender']) or self.inv['titles']['m']
        return self.pick(titles)

    # organisations
    def org(self, key, sector):
        if key not in self.orgs:
            names = self.inv['org_names'].get(sector) or [n for v in self.inv['org_names'].values() for n in v]
            name = self.pick(names)
            if self.r.random() < .12:  # a firm nobody spells out: its trade name is the acronym
                name = self.pick(self.inv['acronyms'])
            initials = ''.join(w[0] for w in re.findall(r"[A-Za-zÀ-ÿ][\w'’-]*", name) if w[0].isupper())
            acronym = initials
            if len(acronym) < 2 or self.r.random() < .5:
                acronym = self.pick(self.inv['acronyms'])
            legal = self.pick(self.inv['legal_forms'])
            prefix = self.lang == 'fr' and self.c == 'FR' and self.r.random() < .35
            self.orgs[key] = {'name': name, 'acronym': acronym, 'initials': initials if len(initials) > 1 else '',
                              'legal': f'{legal} {name}' if prefix else f'{name} {legal}' if self.lang != 'es' or self.r.random() < .8
                              else f'{name}, {legal}',
                              'caps': self.r.random() < .15}
        return self.orgs[key]

    def org_form(self, key, form, sector):
        o = self.org(key, sector)
        self.org_seen[key] += 1
        # A letter names the firm in full once and calls it by its initials after that. Across all 154
        # templates `ORG:acronym` is asked for once, so without this the model only ever sees full names.
        if form not in ('acronym', 'legal') and self.org_seen[key] > 1 and o['initials'] and self.r.random() < .45:
            return o['initials']
        v = o['acronym'] if form == 'acronym' else o['legal'] if form == 'legal' else o['name']
        return v.upper() if o['caps'] and form != 'acronym' else v

    # places
    def place(self, key):
        if key not in self.locs:
            r = self.r
            p = self.pick(self.inv['places'])
            n = str(r.choice([r.randint(1, 30), r.randint(1, 30), r.randint(1, 250), r.randint(100, 9999)] if self.c == 'US'
                             else [r.randint(1, 30), r.randint(1, 30), r.randint(1, 180)]))
            if self.c == 'FR' and r.random() < .08:
                n += r.choice([' bis', ' ter', 'B'])
            unit = self.pick(self.inv['units']).replace('{n}', str(r.randint(1, 42))) if r.random() < .35 else None
            if self.c == 'BE' and r.random() < .25:
                n += f' bte {r.randint(1, 12)}'
            self.locs[key] = {'street': self.pick(self.inv['streets']), 'n': n, 'unit': unit, 'town': p['town'],
                              'postcode': p.get('postcode', ''), 'region': p.get('region') or p['town'],
                              'po_box': self.pick(self.inv['po_box']).replace('{n}', str(r.randint(10, 9999))),
                              'caps': r.random() < .12}
        return self.locs[key]

    def loc_form(self, key, form):
        a = self.place(key)
        c = self.c
        street = {'GB': f"{a['n']} {a['street']}", 'IE': f"{a['n']} {a['street']}", 'US': f"{a['n']} {a['street']}",
                  'FR': f"{a['n']} {a['street']}", 'BE': f"{a['street']} {a['n']}", 'CH': f"{a['street']} {a['n']}",
                  'ES': f"{a['street']}, {a['n']}"}[c]
        if c == 'US':
            state = US_STATES.get(a['region'], a['region'][:2].upper())
            pt = f"{a['town']}, {state} {a['postcode']}".strip()
        elif c in ('GB',):
            pt = f"{a['town']} {a['postcode']}".strip()
        elif c == 'IE':
            pt = f"{a['town']}, {a['region']} {a['postcode']}".strip() if a['region'] != a['town'] else f"{a['town']} {a['postcode']}"
        else:
            pt = f"{a['postcode']} {a['town']}".strip()
        if form == 'line':
            parts = ([a['unit']] if a['unit'] and c not in ('ES',) else []) + [street + (f", {a['unit']}" if a['unit'] and c == 'ES' else '')] + [pt]
            v = ', '.join(parts) if c != 'US' or self.r.random() < .8 else ' '.join(parts)
        elif form == 'street':
            v = street
        elif form == 'unit':
            v = a['unit'] or self.pick(self.inv['units']).replace('{n}', str(self.r.randint(1, 42)))
        elif form == 'postcode_town':
            v = pt
        elif form in ('town', None):
            v = a['town']
        elif form == 'region':
            v = a['region']
        elif form == 'country':
            v = self.pick(COUNTRY_NAME[self.lang][c])
        elif form == 'po_box':
            v = a['po_box']
        else:
            raise ValueError(form)
        return v.upper() if a['caps'] and form in ('line', 'street', 'postcode_town', 'town') else v

    def date(self, form):
        r, lang = self.r, self.lang
        year = r.randint(1938, 2006) if form == 'birth' else r.choice([r.randint(2019, 2027), 2025, 2026, 2026])
        m, d = r.randint(1, 12), r.randint(1, 28)
        if form == 'birth':
            form = r.choice(['long', 'numeric', 'numeric'])
        if form == 'iso':
            return f'{year}-{m:02d}-{d:02d}'
        if form == 'numeric':
            sep = '.' if self.c == 'CH' and r.random() < .7 else r.choice(['/', '/', '/', '-', '.'])
            y = str(year) if r.random() < .85 else str(year)[2:]
            return f'{m:02d}/{d:02d}/{y}' if self.c == 'US' else f'{d:02d}{sep}{m:02d}{sep}{y}'
        month = MONTHS[lang][m - 1]
        if form == 'short':
            month = SHORT[lang][m - 1]
        if lang == 'en':
            day = f'{d}{"th" if 10 <= d % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(d % 10, "th")}' if r.random() < .15 else str(d)
            if self.c == 'US':
                return f'{month} {d}' if form == 'day_month' else f'{month} {d}, {year}'
            return f'{day} {month}' if form == 'day_month' else f'{day} {month} {year}'
        if lang == 'fr':
            day = '1er' if d == 1 else str(d)
            return f'{day} {month}' if form == 'day_month' else f'{day} {month} {year}'
        return f'{d} de {month}' if form == 'day_month' else (f'{d} {month} {year}' if form == 'short' else f'{d} de {month} de {year}')


# ---------------------------------------------------------------- rendering

SLOT = markup.SLOT
TOKEN = re.compile(SLOT.pattern + '|' + markup.ENTITY.pattern)


def render_template(tpl, inv, rng, numbers):
    lang, country = tpl['lang'], tpl['country']
    ents = Entities(inv, rng, lang, country)
    pieces = []  # [text, label]
    body = tpl['body']
    pos = 0
    for m in TOKEN.finditer(body):
        pieces.append([body[pos:m.start()], None])
        pos = m.end()
        if m.group(0).startswith('[['):
            pieces.append([m.group(8), m.group(7)])
            continue
        kind, n, form, sector, ref_kind, ref_n = m.groups()[:6]
        key = n or f'anon{m.start()}'
        if kind == 'PER':
            pieces.append([ents.person_form(key, form), 'PER'])
        elif kind == 'TITLE':
            pieces.append([ents.title(key), None])
        elif kind == 'ORG':
            pieces.append([ents.org_form(key, form, sector or 'employer'), 'ORG'])
        elif kind == 'LOC':
            pieces.append([ents.loc_form(key, form), 'LOC'])
            # Letterheads finish the address with the country on its own line. Only when the slot already
            # ends the line, so a town named mid-sentence is left alone.
            if form in ('line', 'postcode_town') and body[m.end():m.end() + 1] == '\n' and rng.random() < .3:
                pieces += [['\n', None], [ents.loc_form(key, 'country'), 'LOC']]
        elif kind == 'DATE':
            pieces.append([ents.date(form or 'long'), 'DATE'])
        elif kind == 'PUBLIC':
            body_ = rng.choice(inv['public_bodies'])
            if body_['local']:
                town = ents.loc_form(key, 'town')
                before, after = body_['name'].split('{town}', 1)
                pieces += [[before, None], [town, 'LOC'], [after, None]]
            else:
                pieces.append([body_['name'], None])
        elif kind == 'DEPT':
            pieces.append([rng.choice(inv['departments']), None])
        elif kind == 'JOB':
            pieces.append([rng.choice(inv['jobs']), None])
        elif kind in ('PHONE', 'MOBILE'):
            pieces.append([numbers.phone(kind == 'MOBILE' or rng.random() < .3), None])
        elif kind == 'EMAIL':
            domain = rng.choice(inv['email_domains'])
            if ref_kind == 'PER':
                p = ents.person(ref_n)
                g, s = fold(p['given'].split()[0].split('-')[0]), fold(p['surname'].split()[0])
                local = rng.choice([f'{g}.{s}', f'{g}{s}', f'{g[0]}{s}', f'{g}.{s}{rng.randint(1, 99)}', f'{s}.{g}', f'{g}_{s}'])
                if rng.random() < .3 and ents.orgs:
                    domain = fold(next(iter(ents.orgs.values()))['name'])[:18] + '.' + TLD[country]
            elif ref_kind == 'ORG':
                o = ents.org(ref_n, 'employer')
                local, domain = rng.choice(['info', 'contact', 'accounts', 'hello', 'admin', 'office', 'rrhh', 'compta']), \
                    fold(o['name'])[:20] + '.' + TLD[country]
            else:
                local = fold(rng.choice(inv['surnames'])['name']) + str(rng.randint(1, 999))
            pieces.append([f'{local}@{domain}', None])
        elif kind == 'URL':
            o = ents.org(ref_n or key, 'employer')
            pieces.append([rng.choice(['www.', 'https://www.', 'https://', '']) + fold(o['name'])[:20] + '.' + TLD[country], None])
        elif kind == 'IBAN':
            pieces.append([numbers.iban() or numbers.account(), None])
        elif kind == 'BIC':
            pieces.append([numbers.bic(), None])
        elif kind == 'CARD4':
            last = valid_ids.card(rng)[-4:]
            pieces.append([rng.choice([f'**** **** **** {last}', f'XXXX-{last}', f'ending {last}' if lang == 'en' else last]), None])
        elif kind == 'ACCOUNT':
            pieces.append([numbers.account(), None])
        elif kind == 'SORTCODE':
            pieces.append([f'{rng.randint(10, 99)}-{rng.randint(10, 99)}-{rng.randint(10, 99)}' if country in ('GB', 'IE')
                           else f'{rng.randint(10000000, 99999999)}', None])
        elif kind == 'NATID':
            pieces.append([numbers.natid(), None])
        elif kind == 'TAXID':
            pieces.append([numbers.taxid(), None])
        elif kind == 'COMPANYID':
            pieces.append([numbers.companyid(), None])
        elif kind == 'VAT':
            pieces.append([numbers.vat(), None])
        elif kind == 'PLATE':
            pieces.append([numbers.plate(), None])
        elif kind == 'REF':
            pieces.append([numbers.ref(), None])
        elif kind == 'NUM':
            pieces.append([str(rng.randint(1, 99)), None])
        elif kind == 'AMOUNT':
            pieces.append([numbers.amount(form == 'big'), None])
        else:
            raise ValueError(kind)
    pieces.append([body[pos:], None])
    return pieces


def pieces_from_markup(text):
    pieces, pos = [], 0
    for m in markup.ENTITY.finditer(text):
        pieces.append([text[pos:m.start()], None])
        pieces.append([m.group(2), m.group(1)])
        pos = m.end()
    pieces.append([text[pos:], None])
    return pieces


def substitute_names(pieces, inv, rng, lang, country):
    """Hand-written document variant: every person word and organisation value replaced consistently."""
    ents = Entities(inv, rng, lang, country)
    words, orgs = {}, {}

    def new_word(w, first):
        k = w.casefold()
        if k not in words:
            pool = inv['given_names'] if first else inv['surnames']
            words[k] = rng.choice(pool)['name'].split()[0]
        v = words[k]
        return v.upper() if w.isupper() and len(w) > 1 else v.lower() if w.islower() else v

    out = []
    for text, label in pieces:
        if label == 'PER':
            tokens = re.split(r'([\s,.-]+)', text)
            real = [t for t in tokens if t and not re.fullmatch(r'[\s,.-]+', t)]
            first_given = not (',' in text or (len(real) > 1 and real[0].isupper() and not real[-1].isupper()))
            res, seen_first = [], False
            for t in tokens:
                if not t or re.fullmatch(r'[\s,.-]+', t) or len(t) == 1:
                    res.append(t)
                    continue
                is_first = not seen_first
                seen_first = True
                res.append(new_word(t, is_first == first_given if len(real) > 1 else True))
            text = ''.join(res)
        elif label == 'ORG' and len(text) > 4 and not text.isupper():
            if text not in orgs:
                orgs[text] = ents.org_form(f'o{len(orgs)}', 'legal' if re.search(r'\b(Ltd|LLP|Inc|SARL|SAS|S\.L|S\.A|SA|SRL|Sàrl)\b', text)
                                           else 'name', rng.choice(list(inv['org_names'])))
            text = orgs[text]
        out.append([text, label])
    return out


OCR_SWAPS = [('l', '1'), ('I', '1'), ('O', '0'), ('o', '0'), ('rn', 'm'), ('é', 'e'), ('S', '5')]


def augment(pieces, rng, doctype):
    """Document-level noise: lowercase chat names, whole-document capitals, OCR errors, wrapped long entities."""
    if doctype == 'chat':
        for p in pieces:
            if p[1] == 'PER' and rng.random() < .5:
                p[0] = p[0].lower()
    if rng.random() < .04:
        for p in pieces:
            p[0] = p[0].upper()
    if rng.random() < .10:
        for p in pieces:
            if rng.random() < .15:
                a, b = rng.choice(OCR_SWAPS)
                i = p[0].find(a)
                if i >= 0:
                    p[0] = p[0][:i] + b + p[0][i + len(a):]
            if p[1] is None and '@' in p[0] and rng.random() < .5:
                p[0] = p[0].replace('@', rng.choice(['@ ', ' @', ' @ ']), 1)
    out = []
    for text, label in pieces:
        words = text.split(' ')
        if label == 'LOC' and text.count(',') >= 2 and rng.random() < .2:
            parts = [x.strip() for x in text.split(',') if x.strip()]
            for i, part in enumerate(parts):
                if i:
                    out.append(['\n', None])
                out.append([part, 'LOC'])
        elif label in ('LOC', 'ORG') and len(words) >= 3 and rng.random() < .05:
            cut = rng.randint(1, len(words) - 1)
            first, second = ' '.join(words[:cut]).rstrip(','), ' '.join(words[cut:])
            out += [[first, label], ['\n', None], [second, label]]
        else:
            out.append([text, label])
    return out


def to_record(pieces):
    text, spans = '', []
    for t, label in pieces:
        if label and t.strip():
            lead = len(t) - len(t.lstrip())
            core = t.strip().rstrip(',;:')
            start = len(text) + lead
            spans.append({'start': start, 'end': start + len(core), 'label': label})
        text += t
    return text, spans


# ---------------------------------------------------------------- driver

def load_templates():
    out = []
    for path in sorted(glob.glob(os.path.join(HERE, 'templates', '*', '*.txt'))):
        raw = open(path, encoding='utf-8').read()
        header = dict(re.findall(r'^#(\w+):\s*(\S+)\s*$', raw, re.M))
        body = '\n'.join(l for l in raw.split('\n') if not l.startswith('#')).strip('\n')
        lang = os.path.basename(os.path.dirname(path))
        out.append({'id': f"tpl-{lang}-{os.path.basename(path)[:-4]}", 'path': path, 'lang': lang, 'country': header['country'],
                    'doctype': header['doctype'], 'body': body})
    return out


def load_documents():
    out = []
    for path in sorted(glob.glob(os.path.join(HERE, 'documents', '*', '*.txt'))):
        name = os.path.basename(path)[:-4]
        lang = os.path.basename(os.path.dirname(path))
        country = name.split('-')[0] if name.split('-')[0] in LANG_OF else {'en': 'GB', 'fr': 'FR', 'es': 'ES'}[lang]
        out.append({'id': f'doc-{lang}-{name}', 'lang': lang, 'country': country,
                    'doctype': 'negatives' if name.startswith('neg') else name.split('-')[1] if '-' in name else 'document',
                    'text': open(path, encoding='utf-8').read()})
    return out


def load_inventories(report):
    blocked = check_isolation.blocked_inventory_values()
    invs = {}
    for path in sorted(glob.glob(os.path.join(HERE, 'inventories', '*.json'))):
        inv = json.load(open(path, encoding='utf-8'))
        dropped = 0
        for key in ('given_names', 'surnames'):
            keep = [x for x in inv[key] if x['name'].casefold() not in blocked]
            dropped += len(inv[key]) - len(keep)
            inv[key] = keep
        for sector in inv['org_names']:
            keep = [x for x in inv['org_names'][sector] if x.casefold() not in blocked]
            dropped += len(inv['org_names'][sector]) - len(keep)
            inv['org_names'][sector] = keep
        keep = [x for x in inv['streets'] if x.casefold() not in blocked]
        dropped += len(inv['streets']) - len(keep)
        inv['streets'] = keep
        invs[inv['country']] = inv
        report.append(f"- inventory {inv['country']}: {dropped} entries dropped (in a test set or the synthetic generator)")
    return invs


def split_dev(ids, rng, share=0.1):
    ids = sorted(ids)
    rng.shuffle(ids)
    return set(ids[:max(1, round(len(ids) * share))]) if len(ids) > 3 else set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chunks', type=int, default=30000)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default=os.path.join(HERE, 'out'))
    ap.add_argument('--registry', default=os.path.join(HERE, 'out', 'registry.jsonl'))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    report = [f'# Training set build (seed {args.seed}, target {args.chunks} chunks of ~{CHARS_PER_CHUNK} characters)', '']
    invs = load_inventories(report)
    templates, documents = load_templates(), load_documents()
    registry = []
    if os.path.exists(args.registry):
        registry = [json.loads(l) for l in open(args.registry, encoding='utf-8')]

    dropped = collections.Counter()
    ok_templates = []
    for t in templates:
        if check_isolation.item_violations(t['id'], SLOT.sub(' ⁣ ', t['body']), check_isolation.WRITTEN_MAX_SHARED):
            dropped['template (isolation)'] += 1
        elif t['country'] not in invs:
            dropped['template (no inventory)'] += 1
        else:
            ok_templates.append(t)
    ok_docs = []
    for d in documents:
        plain, _ = markup.parse(d['text'])
        if check_isolation.item_violations(d['id'], plain, check_isolation.WRITTEN_MAX_SHARED):
            dropped['document (isolation)'] += 1
        else:
            ok_docs.append(d)
    dev_ids = split_dev([t['id'] for t in ok_templates], rng) | split_dev([d['id'] for d in ok_docs], rng) | \
        split_dev([r['id'] for r in registry], rng)

    # Per language: the registry share is capped by what was actually fetched, and its shortfall goes to the templates.
    per_lang = args.chunks / 3
    goals = {}
    for lang in COUNTRY_SHARE:
        available = sum(len(r['text']) for r in registry if r['lang'] == lang) / CHARS_PER_CHUNK
        reg = min(per_lang * MIX['registry'], available)
        if reg < per_lang * MIX['registry'] - 1:
            report.append(f'- registry {lang}: {reg:.0f} of {per_lang * MIX["registry"]:.0f} chunks available; '
                          f'the shortfall goes to the templates')
        goals[lang] = {'registry': reg, 'document': per_lang * MIX['document'],
                       'template': per_lang - reg - per_lang * MIX['document']}
    records = {'train': [], 'dev': []}
    made = collections.Counter()

    def emit(rec, source):
        # A registry announcement shares its legally fixed skeleton with every other announcement (fetch_registry.py).
        margin = check_isolation.REGISTRY_MAX_SHARED if source == 'registry' else check_isolation.WRITTEN_MAX_SHARED
        if check_isolation.item_violations(rec['id'], rec['text'], margin):
            dropped[f'{source} output (isolation)'] += 1
            return
        split = 'dev' if rec['base'] in dev_ids else 'train'
        records[split].append(rec)
        made[(source, rec['lang'], rec['country'], split)] += len(rec['text']) / CHARS_PER_CHUNK

    # templates: per language and country, round-robin over template files until the share is reached
    for lang, countries in COUNTRY_SHARE.items():
        for country, share in countries.items():
            pool = [t for t in ok_templates if t['lang'] == lang and t['country'] == country]
            if not pool:
                pool = [dict(t, country=country) for t in ok_templates if t['lang'] == lang]
            if not pool:
                continue
            goal = goals[lang]['template'] * share
            got, i = 0.0, 0
            while got < goal:
                t = pool[i % len(pool)]
                inv = invs.get(country) or invs[t['country']]
                pieces = augment(render_template(t, inv, rng, Numbers(rng, country)), rng, t['doctype'])
                text, spans = to_record(pieces)
                emit({'id': f"{t['id']}-{country}-{i:05d}", 'base': t['id'], 'lang': lang, 'country': country, 'source': 'template',
                      'doctype': t['doctype'], 'text': text, 'spans': spans}, 'template')
                got += len(text) / CHARS_PER_CHUNK
                i += 1

    # hand-written documents: the original once, then name-substituted variants until the share is reached
    for lang in COUNTRY_SHARE:
        pool = [d for d in ok_docs if d['lang'] == lang]
        if not pool:
            continue
        goal, got, i = goals[lang]['document'], 0.0, 0
        while got < goal:
            d = pool[i % len(pool)]
            pieces = pieces_from_markup(d['text'])
            if i >= len(pool) and d['doctype'] != 'negatives':
                pieces = augment(substitute_names(pieces, invs[d['country']], rng, lang, d['country']), rng, d['doctype'])
            text, spans = to_record(pieces)
            emit({'id': f"{d['id']}-{i:05d}", 'base': d['id'], 'lang': lang, 'country': d['country'], 'source': 'document',
                  'doctype': d['doctype'], 'text': text, 'spans': spans}, 'document')
            got += len(text) / CHARS_PER_CHUNK
            i += 1

    # registry: sampled to its share per language
    for lang in COUNTRY_SHARE:
        pool = [r for r in registry if r['lang'] == lang]
        rng.shuffle(pool)
        goal, got = goals[lang]['registry'], 0.0
        for r in pool:
            if got >= goal:
                break
            emit(dict(r, base=r['id'], source='registry'), 'registry')
            got += len(r['text']) / CHARS_PER_CHUNK
        if pool and got < goal:
            report.append(f'- registry {lang}: only {got:.0f} of {goal:.0f} chunks available')

    os.makedirs(args.out, exist_ok=True)
    for split, recs in records.items():
        rng.shuffle(recs)
        with open(os.path.join(args.out, f'{split}.jsonl'), 'w', encoding='utf-8', newline='\n') as f:
            for r in recs:
                r = {k: v for k, v in r.items() if k != 'base'}
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    report += ['', '| source | lang | country | train chunks | dev chunks |', '|---|---|---|---|---|']
    keys = sorted({k[:3] for k in made})
    for k in keys:
        report.append(f'| {k[0]} | {k[1]} | {k[2]} | {made[k + ("train",)]:.0f} | {made[k + ("dev",)]:.0f} |')
    totals = collections.Counter()
    for k, v in made.items():
        totals[k[0]] += v
    all_ = sum(totals.values())
    report += ['', 'Share by source: ' + ', '.join(f'{s} {100 * v / all_:.0f} %' for s, v in totals.items()),
               f"Records: {len(records['train'])} train, {len(records['dev'])} dev; templates {len(ok_templates)}, "
               f"documents {len(ok_docs)}, registry records {len(registry)}", '',
               'Dropped: ' + (', '.join(f'{k} {v}' for k, v in dropped.items()) or 'none')]
    labels = collections.Counter(s['label'] for recs in records.values() for r in recs for s in r['spans'])
    report.append('Spans: ' + ', '.join(f'{k} {v}' for k, v in labels.most_common()))
    open(os.path.join(args.out, 'build_report.md'), 'w', encoding='utf-8', newline='\n').write('\n'.join(report) + '\n')
    print('\n'.join(report))


if __name__ == '__main__':
    main()
