# Copyright 2026 the Docudis contributors. Licensed under Apache-2.0.
"""Builds the intent fine-tuning set from raw/*.jsonl.

    python intent/train/build.py

Drops cases that duplicate another training case or come close to a dev
(eval), few-shot or test case, writes the targets in canonical form (schema key order, types that
repeat "*" removed), and splits off a dev set. Output goes to out/ as chat
messages with the short system prompt in ../system_prompt.txt.
"""

import json
import random
import re
import subprocess
import sys
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).parent
INTENT = HERE.parent
SCHEMA = json.loads((INTENT / "schema.json").read_text("utf-8"))
FIELDS = list(SCHEMA["properties"])
TYPE_ORDER = SCHEMA["properties"]["types"]["propertyNames"]["enum"]
LEAK_RATIO = 0.8
DEV_SHARE = 0.1


def canonical(intent):
    out = {}
    for key in FIELDS:
        if key not in intent:
            continue
        value = intent[key]
        if key == "types":
            star = value.get("*")
            value = {k: value[k] for k in TYPE_ORDER if k in value and (k == "*" or value[k] != star)}
        out[key] = value
    return out


def norm(text):
    return re.sub(r"[\W_]+", " ", text.lower()).strip()


def close(a, b):
    m = SequenceMatcher(None, a, b, autojunk=False)
    return m.real_quick_ratio() >= LEAK_RATIO and m.quick_ratio() >= LEAK_RATIO and m.ratio() >= LEAK_RATIO


def main():
    raw = sorted((HERE / "raw").glob("*.jsonl"))
    for path in raw:
        if subprocess.run([sys.executable, str(INTENT / "eval" / "validate.py"), str(path)]).returncode:
            sys.exit(f"{path.name} does not validate")
    cases = [json.loads(l) for p in raw for l in p.read_text("utf-8").splitlines()]
    held_out = [INTENT / "eval" / "cases.jsonl", INTENT / "eval" / "shots.jsonl", INTENT / "test" / "cases.jsonl"]
    evals = [norm(json.loads(l)["instruction"]) for p in held_out if p.exists() for l in p.read_text("utf-8").splitlines()]

    kept, seen, leaks, dups = [], set(), [], []
    for case in cases:
        key = norm(case["instruction"])
        if key in seen:
            dups.append(case["id"])
            continue
        hit = next((e for e in evals if close(key, e)), None)
        if hit is not None:
            leaks.append(f"{case['id']}: {case['instruction']!r} ~ eval {hit!r}")
            continue
        seen.add(key)
        kept.append(case)

    system = (INTENT / "system_prompt.txt").read_text("utf-8").strip()
    random.Random(0).shuffle(kept)
    dev_ids = set()
    for lang in {c["lang"] for c in kept}:
        group = [c["id"] for c in kept if c["lang"] == lang]
        dev_ids.update(group[: max(1, round(len(group) * DEV_SHARE))])

    out = HERE / "out"
    out.mkdir(exist_ok=True)
    for split in ("train", "dev"):
        rows = [c for c in kept if (c["id"] in dev_ids) == (split == "dev")]
        with (out / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for c in rows:
                target = json.dumps(canonical(c["expect"]), ensure_ascii=False, separators=(",", ":"))
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": c["instruction"]},
                    {"role": "assistant", "content": target},
                ]
                f.write(json.dumps({"id": c["id"], "messages": messages}, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)}")
    print(f"dropped {len(dups)} duplicates, {len(leaks)} close to eval")
    for line in leaks:
        print("  " + line)


if __name__ == "__main__":
    main()
