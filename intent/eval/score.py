# Copyright 2026 the Docudis contributors. Licensed under Apache-2.0.
"""Scores prediction files from run.py against cases.jsonl, following spec.md "## Scoring".

    python intent/eval/score.py intent/eval/results/*.jsonl [--failures]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from jsonschema import Draft202012Validator

HERE = Path(__file__).parent
FIELDS = ("types", "regions", "verticals", "dictionary", "never_hide", "unsupported")
validator = Draft202012Validator(json.loads((HERE.parent / "schema.json").read_text("utf-8")))


def normalize(intent):
    """Drops types that repeat the "*" action; lists compare as sets."""
    out = {}
    types = dict(intent.get("types", {}))
    if "*" in types:
        types = {k: v for k, v in types.items() if k == "*" or v != types["*"]}
    if types:
        out["types"] = types
    for key in ("regions", "verticals", "dictionary", "never_hide"):
        if intent.get(key):
            out[key] = frozenset(intent[key])
    if intent.get("unsupported"):
        out["unsupported"] = True
    return out


def parse(raw):
    try:
        intent = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(intent, dict):
        return None
    # The host drops empty fields and `"unsupported": false` before validating.
    intent = {k: v for k, v in intent.items() if v not in ([], {}, False)}
    if any(validator.iter_errors(intent)):
        return None
    return intent


def pct(hits, total):
    return f"{100 * hits / total:5.1f}% ({hits}/{total})"


def score(path, cases, show_failures):
    preds = {p["id"]: p for p in map(json.loads, path.read_text("utf-8").splitlines())}
    exact, invalid, ms = 0, 0, []
    field_hits = defaultdict(int)
    by = {"lang": defaultdict(lambda: [0, 0]), "tag": defaultdict(lambda: [0, 0])}
    failures = []
    for case in cases:
        pred = preds[case["id"]]
        ms.append(pred["ms"])
        intent = parse(pred["raw"])
        if intent is None:
            invalid += 1
        want = normalize(case["expect"])
        got = normalize(intent) if intent is not None else None
        ok = got == want
        exact += ok
        for field in FIELDS:
            field_hits[field] += got is not None and got.get(field) == want.get(field)
        for group, key in [("lang", case["lang"])] + [("tag", t) for t in case["tags"]]:
            by[group][key][0] += ok
            by[group][key][1] += 1
        if not ok:
            failures.append((case, pred["raw"]))

    n = len(cases)
    ms.sort()
    print(f"== {path.stem}")
    print(f"exact match  {pct(exact, n)}   invalid {invalid}   latency p50 {ms[n // 2]} ms, p90 {ms[n * 9 // 10]} ms")
    print("fields       " + "  ".join(f"{f} {100 * field_hits[f] / n:.0f}%" for f in FIELDS))
    for group in ("lang", "tag"):
        rows = sorted(by[group].items(), key=lambda kv: kv[1][0] / kv[1][1])
        print(f"by {group:<4}   " + "  ".join(f"{k} {100 * h / t:.0f}%/{t}" for k, (h, t) in rows))
    if show_failures:
        for case, raw in failures:
            print(f"  {case['id']}: {case['instruction']}\n    want {json.dumps(case['expect'], ensure_ascii=False)}\n    got  {raw}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", type=Path, nargs="+")
    ap.add_argument("--failures", action="store_true")
    ap.add_argument("--cases", type=Path, default=HERE / "cases.jsonl")
    args = ap.parse_args()
    cases = [json.loads(l) for l in args.cases.read_text("utf-8").splitlines()]
    for path in args.predictions:
        score(path, cases, args.failures)


if __name__ == "__main__":
    main()
