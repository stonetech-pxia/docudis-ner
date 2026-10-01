"""Real registry announcements with labels aligned from their structured fields.

    python training/fetch_registry.py [--out training/out/registry.jsonl] [--bodacc 1200] [--borme 40] [--gazette 500]

  fr  BODACC   announcements from the opendatasoft API (Licence Ouverte 2.0). The text is rendered from the
               structured record, so every span is exact, not aligned by string search.
  es  BORME    the gazette's own PDFs (boe.es, reuse allowed with attribution), parsed with the fixed grammar
               of a registry act ("Nombramientos. Apoderado: X;Y. Datos registrales...").
  en  Gazette  corporate insolvency notices (Open Government Licence v3.0); the notice text is aligned with the
               values of its JSON-LD (company, company number, insolvency practitioner, address, dates).

Only dates that the test set does not use (2024-09-12 for BORME, autumn 2026 for BODACC and the Gazette).
An announcement whose text still contains an unlabelled name-shaped sequence is dropped, so the silver labels
stay clean; --report prints how many were dropped and why. Output records feed training/build.py.
"""
import argparse
import collections
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, 'tool')]
import check_isolation  # noqa: E402
from html_text import strip_tags  # noqa: E402

H = {'User-Agent': 'docudis-training-builder/1.0'}
PAUSE = 0.6  # between requests to the same source


def get(url, **kw):
    """One polite GET, backing off when the source rate-limits us."""
    kw.setdefault('headers', H)
    r = None
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=30, **kw)
        except requests.RequestException:  # a slow source drops the connection; retrying is cheaper than losing it
            time.sleep(2 ** attempt * 2)
            continue
        if r.status_code not in (429, 500, 502, 503, 504):
            time.sleep(PAUSE)
            return r
        time.sleep(2 ** attempt * 2)
    if r is None:
        raise requests.RequestException(f'no response from {url}')
    return r
LEGAL = re.compile(r'\b(S\.?A\.?R\.?L|SAS[U]?|SCI|SELARL|SELAS|EURL|SNC|SCP|SASU|S\.?A\.?|SOCIEDAD|LIMITADA|AN[ÓO]NIMA|S\.?L\.?U?|'
                   r'S\.?COOP|LTD|LIMITED|LLP|PLC|INC|ASBL|SPRL|SRL|GMBH|COOPERATIVA|MUTUA|ASOCIACI[ÓO]N|ASSOCIATION|CABINET|'
                   r'ETUDE|SCM|SISP|AARPI)\b', re.I)
DROP = collections.Counter()

# Sequences that look like a name but are boilerplate; they stay unlabelled and must not drop a record.
BOILER = [
    r'BODACC|BOLET[ÍI]N OFICIAL|REGISTRO MERCANTIL|Actos inscritos|Datos registrales|Fe de erratas|Otros conceptos|'
    r'Resultante|Suscrito|Desembolsado|Otras Causas|Objeto social|Declaraci[óo]n de unipersonalidad|Sociedad unipersonal|'
    r'Comienzo de operaciones|Nombramientos|Ceses|Dimisiones|Revocaciones|Reelecciones|Constituci[óo]n|Disoluci[óo]n|'
    r'Extinci[óo]n|Cambio de domicilio|Modificaciones estatutarias|Ampliaci[óo]n de capital|Reducci[óo]n de capital|'
    r'Fusi[óo]n|Escisi[óo]n|Apoderamientos|Situaci[óo]n concursal|Liquidaci[óo]n|Capital|Domicilio|Duraci[óo]n|'
    r'Cierre provisional|Adaptada|Ley \d+|Art[íi]culo \d+|REPR\.\d+|RRM|R\.M\.|I/A \d+|Num\.\d+',
    r'(?:In the )?High Court of Justice|Business and Property Courts?|Insolvency (?:and|&) Companies List|County Court|'
    r'Companies House|The Gazette|Insolvency Act|Insolvency \(England and Wales\) Rules|Companies Act|Rule \d|Section \d|'
    r'Office Holder|Joint Liquidator|Joint Administrator|Liquidator|Administrator|Notice|Registered Office|'
    r'Principal Trading Address|Company Number|Date of Appointment|For further details|IP No|Further details|'
    r'Alternative Contact|Chartered Accountants|Licensed Insolvency Practitioner|Statement of Affairs|Creditors|'
    r'Members Voluntary Liquidation|Voluntary Liquidation|Ag [A-Z]{2}\d+|Trading Name|Company Type|Holding Company|'
    r'Full Name|Case Number|Contact Details|General Meeting|Chancery Division|Previous Name|Registered Company|'
    r'Northern Ireland|High Court|Court of Session|The Insolvency Practitioners?|The Petitioner\'?s Solicitors?|'
    r'The Creditors?|The Members?|The Company|The Directors?|The Court|The Liquidators?|The Administrators?|'
    r'The Joint (?:Liquidators?|Administrators?)|Companies Court|Petition|Winding[- ]Up|Final Meeting|'
    r'Capacity|Telephone|Email|Registered Number|Trading As|Date of Birth|Last Known Address|Court Number|'
    r'Nature of Business:[^\n]*|Type of Liquidation:[^\n]*|By whom Appointed:[^\n]*|Principal Business Activity:[^\n]*|'
    r'PURSUANT TO (?:SECTION|ARTICLE|PARAGRAPH|RULE)[^\n]*|NOTICE IS HEREBY GIVEN|IN THE MATTER OF|'
    r'THE INSOLVENCY ACT[^\n]*|THE COMPANIES ACT[^\n]*|IN THE HIGH COURT OF JUSTICE|IN THE COUNTY COURT[^\n]*|'
    r'CREDITORS? VOLUNTARY LIQUIDATION|MEMBERS VOLUNTARY LIQUIDATION|COMPULSORY LIQUIDATION|'
    r'FINAL MEETING|GENERAL MEETING|WINDING UP',
    r'Greffe|Tribunal|RCS|Registre du Commerce|Annonce|Avis|Jugement|Nom commercial|Forme juridique|Capital|Adresse|'
    r'Si[èe]ge social|[ÉE]tablissement|Origine du fonds|Activit[ée]|Date d|Immatriculation|Nationalit[ée]|Non inscrit|'
    r'Mandataire|Administrateur|Liquidateur|Commissaire|SARL|SAS|SCI|EURL|SASU|SELARL|SNC',
    r'(?:January|February|March|April|May|June|July|August|September|October|November|December)',
]
BOILER_RE = re.compile('|'.join(BOILER))
NAME_SHAPE = re.compile(r"(?:\b[A-ZÀ-ÖØ-Þ][\wÀ-ÿ'’.-]{1,}\b[ -]){1,}\b[A-ZÀ-ÖØ-Þ][\wÀ-ÿ'’.-]{1,}\b")


def residual_names(pieces):
    """Name-shaped sequences left unlabelled; a record with any of them is dropped."""
    out = []
    for text, label in pieces:
        if label:
            continue
        clean = BOILER_RE.sub(' ', text)
        for m in NAME_SHAPE.finditer(clean):
            words = m.group(0).split()
            if len(words) >= 2 and sum(len(w) > 2 for w in words) >= 2 and not all(w.isupper() and len(w) <= 3 for w in words):
                out.append(m.group(0))
    return out


def record(doc_id, lang, country, text, spans, source, category):
    return {'id': doc_id, 'lang': lang, 'country': country, 'source': 'registry', 'doctype': category,
            'origin': source, 'text': text, 'spans': spans}


def assemble(doc_id, lang, country, pieces, source, category):
    residual = residual_names(pieces)
    if residual:
        DROP[f'{source}: unlabelled name ({residual[0][:40]!r})'] += 1
        return None
    text, spans = '', []
    for t, label in pieces:
        if label and t.strip():
            lead = len(t) - len(t.lstrip())
            core = t.strip()
            spans.append({'start': len(text) + lead, 'end': len(text) + lead + len(core), 'label': label})
        text += t
    text = text.strip('\n')
    if len(text) < 250 or not spans:
        DROP[f'{source}: too short'] += 1
        return None
    shift = 0 if not text.startswith(' ') else 0
    problems = check_isolation.item_violations(doc_id, text, max_shared=25)
    if problems:
        DROP[f'{source}: isolation ({problems[0][:40]})'] += 1
        return None
    return record(doc_id, lang, country, text, [{**s, 'start': s['start'] - shift, 'end': s['end'] - shift} for s in spans],
                  source, category)


# ---------------------------------------------------------------- BODACC (fr)

TOWN_AFTER = re.compile(r"^(.*?\b(?:de |du |d'|des |Mixte de Commerce de |))([A-ZÀ-Þ][^,]*)$")


def fr_date(iso):
    m = re.fullmatch(r'(\d{4})-(\d{2})-(\d{2})', str(iso or ''))
    return f'{m.group(3)}/{m.group(2)}/{m.group(1)}' if m else None


def court_pieces(name):
    """'Greffe du Tribunal de Commerce de Reims' -> body O + town LOC."""
    m = re.match(r'^(.*\b(?:de|du|d\'|des)\s*)(.+)$', name.strip())
    if m and m.group(2)[:1].isupper():
        return [[m.group(1), None], [m.group(2), 'LOC']]
    return [[name, None]]


def same_name(a, b):
    """'Rani Lebar' and 'LEBAR Rani' are the same string to a model: the same words in any order and case."""
    words = lambda s: {w.casefold() for w in re.findall(r"[^\W\d_]{2,}", s)}  # noqa: E731
    wa, wb = words(a), words(b)
    return bool(wa) and wa == wb


def bodacc_person(p, out, label=None):
    if label:
        out.append([label + '\n', None])
    imm = p.get('numeroImmatriculation') or {}
    if imm.get('numeroIdentification'):
        out.append([f"{imm.get('codeRCS', 'RCS')} ", None])
        if imm.get('nomGreffeImmat'):
            greffe = re.sub(r"^(?:de |du |d'|des )", '', txt(imm['nomGreffeImmat']).strip())
            out += [[txt(imm['nomGreffeImmat'])[:len(txt(imm['nomGreffeImmat'])) - len(greffe)], None],
                    [greffe, 'LOC'], [' ', None]]
        out.append([imm['numeroIdentification'] + '\n', None])
    elif p.get('nonInscrit'):
        out.append(['Non inscrit au RCS\n', None])
    if p.get('denomination'):
        out += [[txt(p['denomination']), 'ORG'], ['\n', None]]
    if p.get('nom') or p.get('prenom'):
        out += [[' '.join(x for x in [txt(p.get('nom')), txt(p.get('prenom'))] if x), 'PER'], ['\n', None]]
    if p.get('nomUsage'):
        out += [["Nom d'usage : ", None], [txt(p['nomUsage']), 'PER'], ['\n', None]]
    person_name = ' '.join(x for x in [txt(p.get('nom')), txt(p.get('prenom'))] if x)
    for key, lab, label_ in [('sigle', 'Sigle', 'ORG'), ('nomCommercial', 'Nom commercial', 'ORG'),
                             ('formeJuridique', 'Forme juridique', None), ('nationalite', 'Nationalité', None),
                             ('activite', 'Activité', None)]:
        if p.get(key):
            # A trade name that is just the owner's name would be labelled ORG here and PER three lines above,
            # so the same string would carry two labels in one announcement: leave it unlabelled instead.
            value, kind = txt(p[key]), label_
            if kind == 'ORG' and person_name and same_name(value, person_name):
                kind = None
            out += [[f'{lab} : ', None], [value, kind], ['\n', None]]
    cap = p.get('capital') or {}
    if cap.get('montantCapital'):
        out.append([f"Capital : {cap['montantCapital']} {cap.get('devise', '')}".strip() + '\n', None])
    for key, lab in [('adresseSiegeSocial', 'Siège social'), ('adresse', 'Adresse'), ('adressePP', 'Adresse')]:
        a = p.get(key)
        if isinstance(a, dict):
            a = a.get('france') or a.get('etranger') or a
            street = ' '.join(x for x in [txt(a.get('numeroVoie')), txt(a.get('typeVoie')), txt(a.get('nomVoie'))] if x)
            line = ', '.join(x for x in [txt(a.get('complGeographique')), street, txt(a.get('BP')),
                                         ' '.join(y for y in [txt(a.get('codePostal')), txt(a.get('ville'))] if y),
                                         txt(a.get('pays'))] if x)
            if line:
                out += [[f'{lab} : ', None], [line, 'LOC'], ['\n', None]]


def as_list(x):
    return x if isinstance(x, list) else [x] if x else []


def txt(x):
    """A structured field is usually a string, sometimes a list or a number."""
    if isinstance(x, (list, tuple)):
        return ' '.join(txt(i) for i in x if i)
    return '' if x is None else str(x)


def bodacc_docs(rng_dates, limit):
    docs, seen = [], set()
    for family in ['creation', 'vente', 'collective', 'modification', 'radiation']:
        for start, end in rng_dates:
            if len(docs) >= limit:
                break
            r = get('https://bodacc-datadila.opendatasoft.com/api/explore/v2.1/catalog/datasets/annonces-commerciales/records',
                    params={'limit': 100, 'where': f'dateparution>=date\'{start}\' and dateparution<=date\'{end}\' '
                                                   f'and familleavis="{family}"'})
            if r.status_code != 200:
                DROP[f'bodacc: HTTP {r.status_code}'] += 1
                continue
            for rec in r.json()['results']:
                key = (rec.get('numeroannonce'), rec.get('dateparution'))
                if key in seen or len(docs) >= limit:
                    continue
                seen.add(key)
                pieces = [[f"BODACC {rec.get('publicationavis', '')} n° {rec.get('parution', '')} du ", None],
                          [fr_date(rec.get('dateparution')) or '', 'DATE'],
                          [f" - Annonce n° {rec.get('numeroannonce')}\n{rec.get('familleavis_lib', '')} - "
                           f"{rec.get('typeavis_lib', '')}\n", None]]
                pieces += court_pieces(rec.get('tribunal') or '') + [['\n\n', None]]
                for p in as_list(json.loads(rec['listepersonnes']).get('personne')) if rec.get('listepersonnes') else []:
                    bodacc_person(p, pieces)
                    pieces.append(['\n', None])
                for e in as_list(json.loads(rec['listeetablissements']).get('etablissement')) if rec.get('listeetablissements') else []:
                    pieces.append(['Établissement :\n', None])
                    for key_, lab, label_ in [('origineFonds', 'Origine du fonds', None), ('enseigne', 'Enseigne', 'ORG'),
                                              ('activite', 'Activité', None)]:
                        if e.get(key_):
                            pieces += [[f'{lab} : ', None], [txt(e[key_]), label_], ['\n', None]]
                    a = e.get('adresse')
                    if isinstance(a, dict):
                        a = a.get('france') or a
                        line = ', '.join(x for x in [' '.join(y for y in [txt(a.get('numeroVoie')), txt(a.get('typeVoie')),
                                                                          txt(a.get('nomVoie'))] if y),
                                                     ' '.join(y for y in [txt(a.get('codePostal')), txt(a.get('ville'))] if y)] if x)
                        if line:
                            pieces += [['Adresse : ', None], [line, 'LOC'], ['\n', None]]
                    pieces.append(['\n', None])
                for key_, lab in [('listeprecedentproprietaire', 'Précédent propriétaire :'),
                                  ('listeprecedentexploitant', 'Précédent exploitant :')]:
                    if rec.get(key_):
                        for p in as_list(json.loads(rec[key_]).get('personne')):
                            bodacc_person(p, pieces, lab)
                        pieces.append(['\n', None])
                if rec.get('acte'):
                    acte = json.loads(rec['acte'])
                    for key_, lab in [('dateImmatriculation', "Date d'immatriculation"),
                                      ('dateCommencementActivite', "Date de commencement d'activité"), ('dateEffet', "Date d'effet")]:
                        if acte.get(key_):
                            pieces += [[f'{lab} : ', None], [fr_date(acte[key_]) or acte[key_], 'DATE'], ['\n', None]]
                    # "Descriptif" is free text naming people, firms and addresses that no field carries; keeping it
                    # would teach the model that those names are not entities, so the field is left out entirely.
                if rec.get('jugement'):
                    j = json.loads(rec['jugement'])
                    pieces += [[f"{j.get('nature', 'Jugement')} du ", None], [fr_date(j.get('date')) or '', 'DATE'], ['\n', None]]
                doc = assemble(f"reg-fr-bodacc-{rec.get('dateparution')}-{rec.get('numeroannonce')}", 'fr', 'FR', pieces,
                               'bodacc', family)
                if doc:
                    docs.append(doc)
            time.sleep(0.4)
    return docs


# ----------------------------------------------------------------- BORME (es)

# A role key introduces the people or companies that take that role. Each registry writes it its own way:
# "Adm. Unico:", "ADM.UNICO:", "Apo.Sol.:", "APODERAD.SOL:", "Cons.Del.Sol:", "LiqSolid:", "AUDIT.CUENT.:",
# "REPR.143 RRM:". Matching is case-insensitive and tolerates the abbreviation's dots, so a source that
# writes the keys in capitals does not silently lose every person in it.
ROLE_STEM = (r'Adm|Administrador[ae]?s?|Apoderad[oa]?s?|Apo|Liquidador(?:es)?|Liq|Auditor(?:es)?|Audit|'
             r'Consejer[oa]s?|Cons|Presidente|Vicepresidente|Secretari[oa]|Vicesecretari[oa]|Soci[oa]s?|'
             r'Repr|Representante|Comisi[óo]n de control|Entidad Dominante|Sociedad absorbente|Sociedad absorbida')
ROLE = rf'(?:{ROLE_STEM})\b[\w./ -]{{0,22}}?'
ACT_PART = re.compile(rf'(?P<role>{ROLE}):\s*(?P<names>[^.]+?)\.(?=\s|$)'
                      r'|(?:Domicilio:|Cambio de domicilio social\.)\s*(?P<addr>[^.]+(?:\.[^ ][^.]*)*?)\.(?=\s+[A-ZÁÉÍÓÚ]|$)'
                      r'|(?P<date>\b\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}\b)', re.I)
ES_DATE = re.compile(r'\b(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4})\b')
BOILER_RE = re.compile('|'.join(BOILER + [ROLE]))  # the role words themselves stay unlabelled
BORME_STOP = re.compile(r'^(SOCIEDAD|LIMITADA|AN[ÓO]NIMA|UNIPERSONAL|SL|SA|SLU|SAU|EN LIQUIDACI[ÓO]N|SOCIEDAD LIMITADA)$', re.I)


def borme_act_pieces(act):
    """'115535 - JAVIVES VINOTECA SL. Constitución. ...', with or without the older '(R.M. <place>)' tag."""
    m = re.match(r'^(\d{5,6}) - (.+?)\s*(?:\(R\.M\.\s*([^)]+)\))?\.\s+(?=[A-ZÁÉÍÓÚÑ])(.*)$', act, re.S)
    if not m:
        return None
    name, rest_after_name = m.group(2), m.group(4)
    # "ANDREYSKKA CORP. SOCIEDAD LIMITADA": the legal form sits after a period, so the name looks finished.
    tail = re.match(r'^(SOCIEDAD (?:LIMITADA|AN[ÓO]NIMA)(?: UNIPERSONAL| PROFESIONAL| LABORAL)?|S\.?L\.?U?|S\.?A\.?U?|'
                    r'SLNE|S\.?L\.?P|S\.?C\.?P|A\.?V\.?(?: SA)?|SICAV|SGR)\b\.?\s*', rest_after_name, re.I)
    if tail:
        name = f'{name}. {tail.group(1)}'
        rest_after_name = rest_after_name[tail.end():]
    pieces = [[f'{m.group(1)} - ', None], [name, 'ORG']]
    if m.group(3):
        pieces += [[' (R.M. ', None], [m.group(3), 'LOC'], [')', None]]
    pieces.append(['. ', None])
    rest = rest_after_name
    pos = 0
    for part in ACT_PART.finditer(rest):  # roles, addresses and dates in the order they are written
        pieces.append([rest[pos:part.start()], None])
        pos = part.end()
        if part.group('date'):
            pieces.append([part.group('date'), 'DATE'])
        elif part.group('addr') is not None:
            pieces += [['Domicilio: ', None], [part.group('addr'), 'LOC'], ['.', None]]
        else:
            pieces.append([f"{part.group('role')}: ", None])
            for i, name in enumerate(part.group('names').split(';')):
                if i:
                    pieces.append([';', None])
                value = name.strip()
                if value:
                    pieces += [[' ' * (len(name) - len(name.lstrip())), None],
                               [value, 'ORG' if LEGAL.search(value) else 'PER']]
            pieces.append(['.', None])
    pieces.append([rest[pos:], None])
    return pieces


ES_WEEKDAYS = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
ES_MONTHS = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio', 'agosto', 'septiembre', 'octubre',
             'noviembre', 'diciembre']


def es_weekday(yyyymmdd):
    import datetime
    d = datetime.date(int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8]))
    return ES_WEEKDAYS[d.weekday()]


def es_long_date(yyyymmdd):
    return f'{int(yyyymmdd[6:8])} de {ES_MONTHS[int(yyyymmdd[4:6]) - 1]} de {yyyymmdd[:4]}'


BORME_PROVINCE = {'01': 'ÁLAVA', '03': 'ALICANTE', '04': 'ALMERÍA', '07': 'BALEARES', '08': 'BARCELONA', '11': 'CÁDIZ',
                  '12': 'CASTELLÓN', '14': 'CÓRDOBA', '15': 'A CORUÑA', '17': 'GIRONA', '18': 'GRANADA', '20': 'GIPUZKOA',
                  '21': 'HUELVA', '23': 'JAÉN', '24': 'LEÓN', '28': 'MADRID', '29': 'MÁLAGA', '30': 'MURCIA', '33': 'ASTURIAS',
                  '35': 'LAS PALMAS', '36': 'PONTEVEDRA', '37': 'SALAMANCA', '38': 'SANTA CRUZ DE TENERIFE', '39': 'CANTABRIA',
                  '41': 'SEVILLA', '43': 'TARRAGONA', '45': 'TOLEDO', '46': 'VALENCIA', '47': 'VALLADOLID', '48': 'BIZKAIA',
                  '50': 'ZARAGOZA'}


def borme_docs(dates, provinces, per_doc=5, limit=200):
    docs = []
    for date in dates:
        if len(docs) >= limit:
            break
        r = get(f'https://www.boe.es/datosabiertos/api/borme/sumario/{date}', headers={**H, 'Accept': 'application/json'})
        if r.status_code != 200:
            DROP[f'borme: sumario HTTP {r.status_code}'] += 1
            continue
        body = r.text.replace('\\/', '/')
        urls = dict.fromkeys(re.findall(r'https://www\.boe\.es/borme/dias/[^"]+?BORME-A-[^"]+?\.pdf', body))
        items = [(BORME_PROVINCE.get(u.rsplit('-', 1)[1][:2], u.rsplit('-', 1)[1][:2]), u) for u in urls]
        chosen = [i for p in provinces for i in items if p == i[0]][:3]
        for title, url in chosen:
            pdf = get(url).content
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, 'b.pdf')
                open(path, 'wb').write(pdf)
                text = subprocess.run(['pdftotext', '-enc', 'UTF-8', '-l', '4', path, '-'],
                                      capture_output=True).stdout.decode('utf-8', 'replace')
            lines = [l for l in text.splitlines()
                     if l.strip() and not re.search(r'^cve:|Verificable en|BOLET[ÍI]N OFICIAL DEL REGISTRO|^N[úu]m\. \d+|'
                                                    r'^P[áa]g\. \d+|^https?://|D\.L\.:|ISSN|^\s*(Lunes|Martes|Miércoles|Jueves|'
                                                    r'Viernes|Sábado) ', l.strip())]
            joined = []
            for line in lines:
                if joined and not re.match(r'\d{5,6} - ', line.strip()):
                    joined[-1] = joined[-1][:-1] + line.strip() if joined[-1].endswith('-') else joined[-1] + ' ' + line.strip()
                else:
                    joined.append(line.strip())
            acts, province = [], title
            for line in joined:  # since 2025 the province is a section heading instead of an "(R.M. X)" tag
                if re.match(r'\d{5,6} - ', line):
                    acts.append((line, province))
                elif line.isupper() and not any(c.isdigit() for c in line) and 3 < len(line) < 40 \
                        and line not in ('SECCIÓN PRIMERA', 'ACTOS INSCRITOS', 'EMPRESARIOS'):
                    province = line.split('/')[0].title() if line.isupper() else line
            group, n = [], 0
            for act, act_province in acts:
                pieces = borme_act_pieces(act)
                if not pieces:
                    DROP['borme: act grammar'] += 1
                    continue
                group.append((pieces, act_province))
                if len(group) == per_doc:
                    n += 1
                    # The real gazette prints "Jueves 12 de septiembre de 2024" under the title. The weekday is
                    # not an entity and the date is; dropping the line taught the model nothing about either,
                    # and a benchmark run then labelled "Jueves" ADDRESS.
                    head = [['BOLETÍN OFICIAL DEL REGISTRO MERCANTIL\n', None], [f'{es_weekday(date)} ', None],
                            [es_long_date(date), 'DATE'], ['\nActos inscritos\n', None],
                            [group[0][1], 'LOC'], ['\n\n', None]]
                    flat = head + [p for g, _ in group for p in g + [['\n\n', None]]]
                    doc = assemble(f'reg-es-borme-{date}-{title[:6]}-{n}', 'es', 'ES', flat, 'borme', 'registry')
                    if doc:
                        # Regression check: a role key always introduces a person or a company, so a record that
                        # has one and no PER span means the key spelling was not recognised (it happened once).
                        if re.search(ROLE + ':', doc['text'], re.I) and not any(s['label'] == 'PER' for s in doc['spans']):
                            DROP['borme: role key but no person (check ROLE_STEM)'] += 1
                            continue
                        docs.append(doc)
                    group = []
            time.sleep(0.5)
    return docs


# --------------------------------------------------------------- Gazette (en)

def gazette_values(graph):
    """Values worth labelling from a notice's JSON-LD, longest first."""
    def strings(v):  # a field can be a string, a list, or a {"@value": ...} object
        if isinstance(v, str):
            return [v]
        if isinstance(v, dict):
            return strings(v.get('@value'))
        if isinstance(v, list):
            return [s for x in v for s in strings(x)]
        return []

    values = []
    for node in graph:
        node_id = str(node.get('@id', ''))
        if '/court/' in node_id or '/edition/' in node_id or '/issue/' in node_id:
            continue  # courts and the gazette's own editions are not entities (training/GUIDE.md section 1)
        types = strings(node.get('@type'))
        names = strings(node.get('foaf:name')) + strings(node.get('name'))
        if any('Person' in t or 'Practitioner' in t for t in types):
            values += [(n, 'PER') for n in names]
        elif any('Organisation' in t or 'Company' in t or 'Firm' in t for t in types):
            values += [(n, 'ORG') for n in names]
        elif 'administrative-area' in str(node.get('@id')):
            values += [(l, 'LOC') for l in strings(node.get('label'))]
        for key in ('street-address', 'locality', 'region', 'postal-code', 'extended-address'):
            values += [(v, 'LOC') for v in strings(node.get(key) or node.get('vcard:' + key)) if len(v) > 3]
    skip = {'the gazette', 'london gazette', 'edinburgh gazette', 'belfast gazette', 'justice', 'county hall',
            'companies house', 'the insolvency service'}
    return [(v, label) for v, label in values if isinstance(v, str) and v.strip() and v.strip().casefold() not in skip]


UK_POSTCODE = re.compile(r'\b[A-Z]{1,2}\d[\dA-Z]?\s?\d[A-Z]{2}\b')
UK_COMPANY = re.compile(r"\b[A-Z0-9][\w&'’.-]*(?: (?:[A-Z0-9][\w&'’.-]*|&|and|of|the)){0,6} "
                        r"(?:LIMITED|Limited|LTD|Ltd|PLC|Plc|LLP|CIC)\b\.?")
# The notice grammar names people in two fixed places.
GAZ_PERSON = re.compile(r"([A-Z][\w'’-]+(?: [A-Z][\w'’.-]+){1,3})\s*\n?\s*\(IP (?:number|No\.?) ?\d+\)")
GAZ_CONTACT = re.compile(r"contact ([A-Z][\w'’-]+(?: [A-Z][\w'’.-]+){1,3}) (?:on|at|by|via)")


def uk_extra_spans(text, known=()):
    """People, company names and postal addresses the JSON-LD does not carry, found by the notice's grammar."""
    spans = []
    known_orgs = [s for s in known if s['label'] == 'ORG']
    for rx in (GAZ_PERSON, GAZ_CONTACT):
        for m in rx.finditer(text):
            spans.append({'start': m.start(1), 'end': m.end(1), 'label': 'PER'})
    companies = [{'start': m.start(), 'end': m.end(), 'label': 'ORG'} for m in UK_COMPANY.finditer(text)]
    spans += companies
    for m in UK_POSTCODE.finditer(text):
        line_start = text.rfind('\n', 0, m.start()) + 1
        head = text[line_start:m.start()]
        cut = None
        for sep in re.finditer(r':\s|\bof\s|\bat\s|\.\s', head):  # the address starts after the field label or "of"
            cut = sep
        start = line_start + (cut.end() if cut else 0)
        after_of = bool(cut) and cut.group(0).startswith('of')
        for c in list(companies) + list(known_orgs):  # "of <Firm>, <address>": the firm is its own span
            if start <= c['start'] and c['end'] < m.start():
                start = c['end']
        while start < m.start() and (text[start].isspace() or text[start] in ',:'):
            start += 1
        if after_of:  # a firm without a legal form is not caught by UK_COMPANY; it is the part before the first comma
            head, sep, _ = text[start:m.end()].partition(', ')
            if sep and not any(ch.isdigit() for ch in head) and len(head) < 60:
                spans.append({'start': start, 'end': start + len(head), 'label': 'ORG'})
                start += len(head) + 2
        if m.start() - start < 200:
            # "601 High Road, London, E11 4PA formerly 4A Roman Road, E6 3RX" is two addresses, not one.
            former = re.search(r'\b(?:formerly|previously|trading from)\b', text[start:m.end()])
            if former:
                spans.append({'start': start, 'end': start + former.start(), 'label': 'LOC'})
                start += former.end() + 1
            spans.append({'start': start, 'end': m.end(), 'label': 'LOC'})
    return spans


def label_by_values(text, values):
    spans = []
    for value, label in sorted([v for v in values if isinstance(v[0], str) and v[0].strip()], key=lambda v: -len(v[0])):
        for m in re.finditer(r'(?<![^\W_])' + re.escape(value.strip()) + r'(?![^\W_])', text):
            if not any(s['start'] < m.end() and s['end'] > m.start() for s in spans):
                spans.append({'start': m.start(), 'end': m.end(), 'label': label})
    for extra in uk_extra_spans(text, spans):  # the JSON-LD is authoritative; the grammar fills the gaps
        if not any(s['start'] < extra['end'] and s['end'] > extra['start'] for s in spans):
            spans.append(extra)
    for m in re.finditer(r'\b(\d{1,2} (?:January|February|March|April|May|June|July|August|September|October|November|December)'
                         r' \d{4}|\d{2}/\d{2}/\d{4})\b', text):
        if not any(s['start'] < m.end() and s['end'] > m.start() for s in spans):
            spans.append({'start': m.start(), 'end': m.end(), 'label': 'DATE'})
    spans.sort(key=lambda s: s['start'])
    pieces, pos = [], 0
    for s in spans:
        pieces += [[text[pos:s['start']], None], [text[s['start']:s['end']], s['label']]]
        pos = s['end']
    pieces.append([text[pos:], None])
    return pieces


def gazette_docs(windows, limit):
    """`limit` counts accepted notices, so the loop is also capped by how many it may LOOK at: a notice costs two
    requests whether it is kept or dropped, and a run that only counted the kept ones once ran for ten hours."""
    docs, seen = [], set()
    budget = limit * 3
    for code in ['2443', '2441', '2442', '2445', '2450', '2451']:
        for start, end in windows:
            if len(docs) >= limit or len(seen) >= budget:
                break
            r = get('https://www.thegazette.co.uk/insolvency/notice/data.json',
                    params={'noticetypes': code, 'results-page-size': 50, 'start-publish-date': start,
                            'end-publish-date': end})
            if r.status_code != 200:
                DROP[f'gazette: feed HTTP {r.status_code}'] += 1
                continue
            for e in r.json().get('entry', []):
                nid = e['id'].rsplit('/', 1)[-1]
                if nid in seen or len(docs) >= limit or len(seen) >= budget:
                    continue
                seen.add(nid)
                try:
                    ld = get(f'https://www.thegazette.co.uk/notice/{nid}/data.jsonld').json()
                    page = get(f'https://www.thegazette.co.uk/notice/{nid}').text
                except Exception as exc:
                    DROP[f'gazette: {type(exc).__name__}'] += 1
                    continue
                m = re.search(r'<article[\s\S]*?</article>', page)
                body = strip_tags(m.group(0) if m else '')
                body = re.sub(r'^Notice category:[\s\S]*?Notice code:\n\d+\n', '', body)
                values = gazette_values(ld.get('@graph') or [])
                if e.get('title'):
                    values.append((e['title'], 'ORG'))
                if e.get('f:name') and e.get('f:familyName'):
                    values.append((f"{e['f:name']} {e['f:familyName']}", 'PER'))
                doc = assemble(f'reg-en-gazette-{nid}', 'en', 'GB', label_by_values(body, values), 'gazette', 'notice')
                if doc:
                    docs.append(doc)
                time.sleep(0.3)
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'out', 'registry.jsonl'))
    ap.add_argument('--bodacc', type=int, default=1200)
    ap.add_argument('--borme', type=int, default=200)
    ap.add_argument('--gazette', type=int, default=500)
    ap.add_argument('--years', nargs='*', type=int, default=[2025, 2024],
                    help='which years to take announcements from; the test set uses 2024-09-12 (BORME) and autumn 2026')
    ap.add_argument('--days', nargs='*', type=int, default=[4, 11, 18, 25], help='days of the month for BORME')
    args = ap.parse_args()
    fr_windows = [(f'{y}-{m:02d}-01', f'{y}-{m:02d}-07') for y in args.years for m in range(1, 13)] + \
                 [(f'{y}-{m:02d}-10', f'{y}-{m:02d}-16') for y in args.years for m in range(1, 13)]
    es_dates = [f'{y}{m:02d}{d:02d}' for y in args.years for m in range(1, 13) for d in args.days
                if not (y == 2024 and m == 9 and d == 12)]
    en_windows = [(f'{y}-{m:02d}-01', f'{y}-{m:02d}-14') for y in args.years for m in range(1, 13)] + \
                 [(f'{y}-{m:02d}-15', f'{y}-{m:02d}-28') for y in args.years for m in range(1, 13)]
    # Each source appends as soon as it finishes, so stopping the run keeps what the earlier sources fetched.
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    open(args.out, 'w', encoding='utf-8').close()
    docs = []
    print(f'windows: fr {len(fr_windows)}, es {len(es_dates)} days, en {len(en_windows)}; '
          f'caps: bodacc {args.bodacc}, borme {args.borme}, gazette {args.gazette} (looks at {args.gazette * 3})',
          flush=True)
    for name, fn in [('BODACC', lambda: bodacc_docs(fr_windows, args.bodacc)),
                     ('BORME', lambda: borme_docs(es_dates, ['MADRID', 'BARCELONA', 'VALENCIA', 'SEVILLA', 'MÁLAGA', 'MURCIA',
                                                             'ZARAGOZA', 'ALICANTE', 'BIZKAIA', 'A CORUÑA'], limit=args.borme)),
                     ('Gazette', lambda: gazette_docs(en_windows, args.gazette))]:
        try:
            got = fn()
        except Exception as exc:  # one source failing must not lose the others
            print(f'{name}: FAILED {exc!r}')
            continue
        print(f'{name}: {len(got)} documents, {sum(len(d["text"]) for d in got)} chars, '
              f'{sum(len(d["spans"]) for d in got)} spans', flush=True)
        docs += got
        with open(args.out, 'a', encoding='utf-8', newline='\n') as f:  # keep what is already fetched
            for d in got:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')
    print(f'written {args.out}: {len(docs)} records')
    for reason, n in DROP.most_common(25):
        print(f'  dropped {n}x {reason}')


if __name__ == '__main__':
    main()
