# Copyright 2026 the Docudis contributors. Licensed under Apache-2.0.
"""Keeps the NER training material apart from every test set.

    python training/check_isolation.py [--jsonl training/out/train.jsonl ...]

Test side, in the docudis-android checkout named by DOCUDIS_APP_ROOT: benchmark/public_cases.json, synthetic_cases.json, consumer_cases.json, regression_cases.json,
ner_cases.json, hard_negatives.json, ocr_cases.json, and the name lists of tool/generate_synthetic_cases.py.
Training side: training/documents, training/templates, training/inventories and generated JSONL files.

A training item fails when it
  - has the id of a test case;
  - contains a test person or company name of two words or more (whole words, case-insensitive);
  - shares more than `max_shared` distinct runs of 8 words with a test text (0 for written material;
    registry announcements share their legally fixed skeleton with every other announcement, so
    fetch_registry.py allows a margin).
An inventory entry fails when it equals a name, company or street of the synthetic generator, or a test
person / company name. Failures are printed and written to training/out/isolation.json; exit code 1 if any.
build.py imports blocked_inventory_values() and item_violations() and drops what fails.
"""
import argparse
import ast
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_ROOT = os.environ.get('DOCUDIS_APP_ROOT')
TEST_SETS = ['public_cases.json', 'synthetic_cases.json', 'consumer_cases.json', 'regression_cases.json', 'ner_cases.json', 'hard_negatives.json',
             'ocr_cases.json']
SHINGLE = 8
WRITTEN_MAX_SHARED = 3  # statutory headings and legal boilerplate are shared by every real document of a kind
REGISTRY_MAX_SHARED = 25  # an announcement's skeleton is fixed by law, so it is shared with every other announcement
WORD = re.compile(r'[^\W_]+')


GENERIC_WORDS = set('the of and a an y de del la le les des du bank banco banque insurance seguros assurance assurances group groupe '
                    'grupo services service servicios dental medical health care clinic clinique clínica centre center practice '
                    'solutions systems partners associates associés asociados limited ltd llp plc inc sa sl srl sarl sas company '
                    'compagnie compañía société sociedad management gestion consulting energy energía énergie telecom mobile '
                    'university college school hospital trust council'.split())


def _app_path(*parts):
    """A file of the docudis-android checkout, which owns the test sets. Never skipped when missing."""
    if not APP_ROOT or not os.path.isdir(os.path.join(APP_ROOT, 'benchmark')):
        sys.exit('set DOCUDIS_APP_ROOT to a docudis-android checkout: the isolation check needs its test sets')
    return os.path.join(APP_ROOT, *parts)


def _real_organisations():
    """Large real organisations (bundled Wikidata list): naming one is not test contamination."""
    path = _app_path('packages', 'docudis_engine', 'lists', 'companies.json')
    names = {n.casefold() for row in json.load(open(path, encoding='utf-8')) for n in row['names']}
    extra = os.path.join(ROOT, 'training', 'real_organisations.txt')
    if os.path.exists(extra):
        names |= {l.strip().casefold() for l in open(extra, encoding='utf-8') if l.strip() and not l.startswith('#')}
    return names


def _load_tests():
    ids, texts, names = set(), [], set()
    real = _real_organisations()
    for name in TEST_SETS:
        data = json.load(open(_app_path('benchmark', name), encoding='utf-8'))
        for case in data['cases']:
            ids.add(case['id'])
            text = case.get('text') or case.get('expectedText') or ''
            texts.append(text)
            expected = case.get('expected') or case.get('criticalSpans') or []
            for e in expected:
                if e['type'] not in ('PERSON', 'COMPANY'):
                    continue
                value = e['value'].casefold()
                words = [w.casefold() for w in WORD.findall(value)]
                if len(words) < 2 or value in real or all(w in GENERIC_WORDS for w in words):
                    continue  # too short, a real organisation, or a generic phrase ("dental care")
                names.add(value)
    return ids, texts, names


def _generator_values():
    tree = ast.parse(open(_app_path('tool', 'generate_synthetic_cases.py'), encoding='utf-8').read())
    values = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and \
                node.targets[0].id in ('PEOPLE', 'COMPANIES', 'KNOWN_COMPANIES', 'SCHOOLS', 'ADDRESSES'):
            def walk(v):
                if isinstance(v, str):
                    values.add(v)
                elif isinstance(v, dict):
                    for x in v.values():
                        walk(x)
                elif isinstance(v, (list, tuple)):
                    for x in v:
                        walk(x)
            walk(ast.literal_eval(node.value))
    parts = set()
    for v in values:
        parts.add(v.casefold())
        if node_is_name(v):
            parts.update(w.casefold() for w in v.split())
        street = re.sub(r'^\d+\s+', '', v)
        if street != v:
            parts.add(street.casefold())
    return parts


def node_is_name(v):
    """A person name of the generator: two to four capitalised words, no digits."""
    words = v.split()
    return 2 <= len(words) <= 4 and all(w[:1].isupper() for w in words) and not any(c.isdigit() for c in v)


def _shingles(text):
    words = [w.casefold() for w in WORD.findall(text)]
    return {hash(tuple(words[i:i + SHINGLE])) for i in range(len(words) - SHINGLE + 1)}


_CACHE = {}


def _state():
    if not _CACHE:
        ids, texts, names = _load_tests()
        shingles = set()
        for t in texts:
            shingles |= _shingles(t)
        _CACHE.update(ids=ids, names=names, shingles=shingles, generator=_generator_values(),
                      name_re=[(n, re.compile(r'(?<![^\W_])' + re.escape(n) + r'(?![^\W_])', re.I)) for n in sorted(names)])
    return _CACHE


def item_violations(item_id, text, max_shared=0):
    s = _state()
    out = []
    if item_id in s['ids']:
        out.append(f'id {item_id} is a test case id')
    low = text.casefold()
    for name, rx in s['name_re']:
        if name in low and rx.search(text):
            out.append(f'test name {name!r}')
    words = [w.casefold() for w in WORD.findall(text)]
    shared = {}
    for i in range(len(words) - SHINGLE + 1):
        run = tuple(words[i:i + SHINGLE])
        if hash(run) in s['shingles']:
            shared[run] = ' '.join(run)
    if len(shared) > max_shared:
        out.append(f'shares {len(shared)} runs of 8 words with a test text: ' + next(iter(shared.values())))
    return out


def blocked_inventory_values():
    s = _state()
    return s['generator'] | s['names']


def _inventory_entries(data):
    for key in ('given_names', 'surnames'):
        for x in data.get(key, []):
            yield key, x['name'] if isinstance(x, dict) else x
    for sector, names in data.get('org_names', {}).items():
        for n in names:
            yield f'org_names.{sector}', n
    for key in ('acronyms', 'streets'):
        for x in data.get(key, []):
            yield key, x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', nargs='*', default=[])
    args = ap.parse_args()
    violations = []
    slot = re.compile(r'\{\{[^}]*\}\}')
    for path in sorted(glob.glob(os.path.join(ROOT, 'training', 'documents', '*', '*.txt')) +
                       glob.glob(os.path.join(ROOT, 'training', 'templates', '*', '*.txt'))):
        text = open(path, encoding='utf-8').read()
        text = re.sub(r'\[\[(?:PER|ORG|LOC|DATE)\|', '', text).replace(']]', '')
        text = slot.sub(' ⁣ ', text)  # a slot breaks word runs
        rel = os.path.relpath(path, ROOT).replace('\\', '/')
        violations += [{'item': rel, 'problem': p}
                       for p in item_violations(os.path.basename(path)[:-4], text, WRITTEN_MAX_SHARED)]
    blocked = blocked_inventory_values()
    for path in sorted(glob.glob(os.path.join(ROOT, 'training', 'inventories', '*.json'))):
        rel = os.path.relpath(path, ROOT).replace('\\', '/')
        for key, value in _inventory_entries(json.load(open(path, encoding='utf-8'))):
            if value.casefold() in blocked:
                violations.append({'item': f'{rel}:{key}', 'problem': f'{value!r} is in a test set or the synthetic generator'})
    for path in args.jsonl:
        for line in open(path, encoding='utf-8'):
            rec = json.loads(line)
            margin = REGISTRY_MAX_SHARED if rec.get('source') == 'registry' else WRITTEN_MAX_SHARED
            violations += [{'item': f"{path}:{rec['id']}", 'problem': p}
                           for p in item_violations(rec['id'], rec['text'], margin)]
    os.makedirs(os.path.join(ROOT, 'training', 'out'), exist_ok=True)
    json.dump(violations, open(os.path.join(ROOT, 'training', 'out', 'isolation.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    for v in violations[:200]:
        print(f"{v['item']}: {v['problem']}")
    print(f'{len(violations)} violations')
    sys.exit(1 if violations else 0)


if __name__ == '__main__':
    main()
