"""Inline markup parser and checker for the training material (see training/GUIDE.md).

    python training/markup.py --check <file> [<file> ...]

.txt under documents/: inline markup `[[PER|...]]` only.
.txt under templates/: header lines, inline markup and `{{SLOT}}` slots.
.json under inventories/: inventory schema and minimum sizes.
Exit code 1 when any file has a problem.
"""
import argparse
import json
import os
import re
import sys

LABELS = ('PER', 'ORG', 'LOC', 'DATE')
ENTITY = re.compile(r'\[\[(PER|ORG|LOC|DATE)\|([^\[\]|\n]+)\]\]')
SLOT = re.compile(r'\{\{([A-Z0-9]+)(?:#(\d+))?(?::([a-z_]+)(?:@([a-z]+))?)?(?::(PER|ORG)#(\d+))?\}\}')
COUNTRIES = ('GB', 'IE', 'US', 'FR', 'BE', 'CH', 'ES')
DOCTYPES = ('invoice', 'payslip', 'bank', 'insurance', 'lease', 'medical', 'letter', 'cv', 'support', 'chat', 'form')
SECTORS = ('trades', 'retail', 'health', 'finance', 'insurance', 'property', 'employer', 'school', 'utility', 'telecom',
           'hospitality', 'transport', 'legal')
SLOT_FORMS = {
    'PER': {'full', 'given', 'surname', 'initial_surname', 'surname_given', 'signature'},
    'TITLE': {None}, 'ORG': {'legal', 'name', 'acronym'},
    'LOC': {'line', 'street', 'unit', 'postcode_town', 'town', 'region', 'country', 'po_box'},
    'DATE': {'long', 'short', 'numeric', 'birth', 'day_month', 'iso'},
    'PUBLIC': {None}, 'DEPT': {None}, 'JOB': {None}, 'PHONE': {None}, 'MOBILE': {None},
    'EMAIL': {None}, 'URL': {None}, 'IBAN': {None}, 'BIC': {None}, 'CARD4': {None}, 'ACCOUNT': {None}, 'SORTCODE': {None},
    'NATID': {None}, 'TAXID': {None}, 'COMPANYID': {None}, 'VAT': {None}, 'PLATE': {None}, 'REF': {None}, 'NUM': {None},
    'AMOUNT': {None, 'big'},
}
MIN_SIZES = {'given_names': 200, 'surnames': 300, 'acronyms': 40, 'streets': 250, 'places': 150, 'public_bodies': 40,
             'departments': 40, 'jobs': 80}


def parse(text):
    """Inline markup -> (plain text, spans [{start, end, label}]). Raises ValueError on stray markup."""
    out, spans, pos = [], [], 0
    for m in ENTITY.finditer(text):
        out.append(text[pos:m.start()])
        start = sum(len(p) for p in out)
        value = m.group(2)
        if value != value.strip():
            raise ValueError(f'untrimmed span {value!r}')
        out.append(value)
        spans.append({'start': start, 'end': start + len(value), 'label': m.group(1)})
        pos = m.end()
    out.append(text[pos:])
    plain = ''.join(out)
    for bad in ('[[', ']]'):
        if bad in plain:
            i = plain.index(bad)
            raise ValueError(f'stray {bad!r} near {plain[max(0, i - 30):i + 30]!r}')
    return plain, spans


def check_document(text, negatives=False):
    plain, spans = parse(text)
    problems = []
    if not spans and not negatives:  # a neg-*.txt file is allowed to have none: that is its point
        problems.append('no entity')
    if '{{' in plain or '}}' in plain:
        problems.append('slot syntax in a document')
    return problems


def check_template(text):
    problems, header = [], {}
    body = []
    for line in text.split('\n'):
        m = re.match(r'#(\w+):\s*(\S+)\s*$', line)
        if m:
            header[m.group(1)] = m.group(2)
        elif not line.startswith('#'):
            body.append(line)
    if header.get('country') not in COUNTRIES:
        problems.append(f"bad or missing #country: {header.get('country')!r}")
    if header.get('doctype') not in DOCTYPES:
        problems.append(f"bad or missing #doctype: {header.get('doctype')!r}")
    body = '\n'.join(body)
    bound = set()
    for m in SLOT.finditer(body):
        kind, n, form, sector, ref_kind, ref_n = m.groups()
        if kind not in SLOT_FORMS:
            problems.append(f'unknown slot {m.group(0)}')
            continue
        if form not in SLOT_FORMS[kind]:
            problems.append(f'unknown form in {m.group(0)}')
        if sector and (kind != 'ORG' or sector not in SECTORS):
            problems.append(f'bad sector in {m.group(0)}')
        if kind in ('PER', 'ORG', 'LOC') and n:
            bound.add(f'{kind}#{n}')
        if ref_kind:
            if kind not in ('EMAIL', 'URL'):
                problems.append(f'only EMAIL / URL take a reference: {m.group(0)}')
            bound_ref = f'{ref_kind}#{ref_n}'
            if bound_ref not in {f'{k}#{x}' for k, x in re.findall(r'\{\{(PER|ORG)#(\d+)', body)}:
                problems.append(f'{m.group(0)} refers to an entity that is not in the template')
        if kind in ('TITLE',) and not n:
            problems.append(f'{m.group(0)} needs #n bound to a PER')
        if kind == 'PUBLIC' and not n:
            problems.append(f'{m.group(0)} needs #n bound to a LOC')
    rest = SLOT.sub('', body)
    if '{{' in rest or '}}' in rest:
        i = rest.find('{{') if '{{' in rest else rest.find('}}')
        problems.append(f'malformed slot near {rest[max(0, i - 30):i + 30]!r}')
    try:
        _, spans = parse(rest)
    except ValueError as e:
        problems.append(str(e))
        spans = []
    if len(SLOT.findall(body)) + len(spans) < 8:
        problems.append('fewer than 8 slots and entities')
    if not any(k in bound for k in bound if k.startswith('PER')):
        problems.append('no person')
    return problems


def check_inventory(data):
    problems = []
    for key in ('country', 'lang', 'given_names', 'surnames', 'titles', 'legal_forms', 'org_names', 'acronyms', 'streets',
                'units', 'places', 'po_box', 'public_bodies', 'departments', 'jobs', 'email_domains'):
        if key not in data:
            problems.append(f'missing {key}')
    if problems:
        return problems
    if data['country'] not in COUNTRIES:
        problems.append(f"bad country {data['country']!r}")
    for key, n in MIN_SIZES.items():
        if len(data[key]) < n:
            problems.append(f'{key}: {len(data[key])} < {n}')
    for sector in SECTORS:
        got = len(data['org_names'].get(sector, []))
        if got < 25:
            problems.append(f'org_names.{sector}: {got} < 25')
    for g in data['given_names']:
        if not isinstance(g, dict) or g.get('gender') not in ('m', 'f', 'x') or not g.get('name'):
            problems.append(f'bad given name {g!r}')
            break
    for p in data['places']:
        if not isinstance(p, dict) or not p.get('town') or 'postcode' not in p:
            problems.append(f'bad place {p!r}')
            break
    for b in data['public_bodies']:
        if not isinstance(b, dict) or 'name' not in b or b.get('local') not in (True, False) \
                or b['local'] != ('{town}' in b['name']):
            problems.append(f'bad public body {b!r} (local must be true exactly when the name contains {{town}})')
            break
    for u in data['units'] + data['po_box']:
        if '{n}' not in u:
            problems.append(f'unit / po box without {{n}}: {u!r}')
            break
    for key in ('given_names', 'surnames'):
        names = [x['name'] for x in data[key]]
        dup = len(names) - len(set(names))
        if dup:
            problems.append(f'{key}: {dup} duplicates')
    return problems


def check_file(path):
    norm = path.replace('\\', '/')
    raw = open(path, 'rb').read()
    if b'\r' in raw:
        return ['CR line endings']
    text = raw.decode('utf-8')
    if norm.endswith('.json'):
        return check_inventory(json.loads(text))
    if '/templates/' in norm:
        return check_template(text)
    return check_document(text, negatives=os.path.basename(norm).startswith('neg-'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', nargs='+', required=True)
    args = ap.parse_args()
    bad = 0
    for path in args.check:
        try:
            problems = check_file(path)
        except Exception as e:  # malformed JSON, bad UTF-8, stray markup
            problems = [f'{type(e).__name__}: {e}']
        for p in problems:
            print(f'{path}: {p}')
        bad += bool(problems)
    print(f'{len(args.check) - bad} of {len(args.check)} files ok')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
