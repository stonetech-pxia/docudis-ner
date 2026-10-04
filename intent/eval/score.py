# Copyright 2026 the Docudis contributors. Licensed under Apache-2.0.
"""Scores prediction files from run.py against cases.jsonl, following spec.md "## Scoring".

    python intent/eval/score.py intent/eval/results/*.jsonl [--failures] [--complete]

With --complete, each intent first goes through ../postprocess.py, as the host
would.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from jsonschema import Draft202012Validator

HERE = Path(__file__).parent
FIELDS = ("types", "regions", "verticals", "dictionary", "never_hide", "unsupported")
SCHEMA = json.loads((HERE.parent / "schema.json").read_text("utf-8"))
validator = Draft202012Validator(SCHEMA)
TYPES = [t for t in SCHEMA["properties"]["types"]["propertyNames"]["enum"] if t != "*"]
# Core detects these but leaves them visible unless the policy says hide.
SHOWN_BY_DEFAULT = {"DATE", "AMOUNT"}


def hidden(intent, entity_type):
    types = intent.get("types", {})
    action = types.get(entity_type, types.get("*"))
    return entity_type not in SHOWN_BY_DEFAULT if action is None else action == "hide"


def severity(want, got):
    """What a miss costs. `leak`: something the user wanted hidden stays visible
    (a type, a dictionary term, or a term wrongly kept by never_hide).
    `unwarned`: an unsupported request the user is not told about.
    `overhide`: a type hidden that the user wanted visible. `minor`: the rest
    (regions, verticals, keep vs off)."""
    labels = set()
    if any(hidden(want, t) and not hidden(got, t) for t in TYPES):
        labels.add("leak")
    if set(want.get("dictionary", [])) - set(got.get("dictionary", [])):
        labels.add("leak")
    if set(got.get("never_hide", [])) - set(want.get("never_hide", [])):
        labels.add("leak")
    if want.get("unsupported") and not got.get("unsupported"):
        labels.add("unwarned")
    if any(hidden(got, t) and not hidden(want, t) for t in TYPES):
        labels.add("overhide")
    return labels or {"minor"}


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


def score(path, cases, show_failures, complete=None):
    preds = {p["id"]: p for p in map(json.loads, path.read_text("utf-8").splitlines())}
    exact, invalid, ms = 0, 0, []
    field_hits = defaultdict(int)
    by = {"lang": defaultdict(lambda: [0, 0]), "tag": defaultdict(lambda: [0, 0])}
    failures = []
    costs = defaultdict(list)
    for case in cases:
        pred = preds[case["id"]]
        ms.append(pred["ms"])
        intent = parse(pred["raw"])
        if intent is None:
            invalid += 1
        elif complete:
            intent = complete(case["instruction"], intent)
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
            # The host falls back to the defaults when the output is invalid.
            labels = severity(case["expect"], intent if intent is not None else {})
            for label in labels:
                costs[label].append(case["id"])
            failures.append((case, pred["raw"], labels))

    n = len(cases)
    ms.sort()
    print(f"== {path.stem}" + (" + keywords" if complete else ""))
    print(f"exact match  {pct(exact, n)}   invalid {invalid}   latency p50 {ms[n // 2]} ms, p90 {ms[n * 9 // 10]} ms")
    print("fields       " + "  ".join(f"{f} {100 * field_hits[f] / n:.0f}%" for f in FIELDS))
    print("misses       " + "  ".join(f"{label} {len(costs[label])}" for label in ("leak", "unwarned", "overhide", "minor")))
    if costs["leak"]:
        print("leaks        " + " ".join(costs["leak"]))
    for group in ("lang", "tag"):
        rows = sorted(by[group].items(), key=lambda kv: kv[1][0] / kv[1][1])
        print(f"by {group:<4}   " + "  ".join(f"{k} {100 * h / t:.0f}%/{t}" for k, (h, t) in rows))
    if show_failures:
        for case, raw, labels in failures:
            print(f"  {case['id']} [{', '.join(sorted(labels))}]: {case['instruction']}")
            print(f"    want {json.dumps(case['expect'], ensure_ascii=False)}\n    got  {raw}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", type=Path, nargs="+")
    ap.add_argument("--failures", action="store_true")
    ap.add_argument("--cases", type=Path, default=HERE / "cases.jsonl")
    ap.add_argument("--complete", action="store_true", help="add keyword regions/verticals as the host does")
    args = ap.parse_args()
    cases = [json.loads(l) for l in args.cases.read_text("utf-8").splitlines()]
    complete = None
    if args.complete:
        sys.path.insert(0, str(HERE.parent))
        from postprocess import complete  # noqa: PLC0415
    for path in args.predictions:
        score(path, cases, args.failures, complete)


if __name__ == "__main__":
    main()
