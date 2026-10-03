# Copyright 2026 the Docudis contributors. Licensed under Apache-2.0.
"""Checks cases.jsonl: line format, unique ids, and every `expect` against schema.json."""

import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

HERE = Path(__file__).parent
CASE_KEYS = {"id", "lang", "instruction", "expect", "tags"}
LANGS = {"zh", "en", "fr", "es", "de", "it", "mixed"}

validator = Draft202012Validator(json.loads((HERE.parent / "schema.json").read_text("utf-8")))
errors, ids = [], set()
path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "cases.jsonl"
lines = path.read_text("utf-8").splitlines()
for n, line in enumerate(lines, 1):
    try:
        case = json.loads(line)
    except json.JSONDecodeError as e:
        errors.append(f"line {n}: {e}")
        continue
    if not CASE_KEYS <= case.keys() or not case.keys() <= CASE_KEYS | {"note"}:
        errors.append(f"line {n}: keys {sorted(case)}")
        continue
    if case["id"] in ids:
        errors.append(f"line {n}: duplicate id {case['id']}")
    ids.add(case["id"])
    if case["lang"] not in LANGS:
        errors.append(f"line {n}: lang {case['lang']}")
    for e in validator.iter_errors(case["expect"]):
        errors.append(f"line {n} ({case['id']}): {e.message}")
    for key in ("dictionary", "never_hide"):
        for term in case["expect"].get(key, []):
            if term not in case["instruction"]:
                errors.append(f"line {n} ({case['id']}): {key} term {term!r} not in instruction")

print("\n".join(errors) or f"{len(lines)} cases OK")
sys.exit(1 if errors else 0)
