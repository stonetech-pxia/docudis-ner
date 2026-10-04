# Copyright 2026 the Docudis contributors. Licensed under Apache-2.0.
"""Runs the eval cases through a llama-server (OpenAI-compatible) and writes predictions.

    llama-server -m model.gguf --port 8080 -ngl 99 -c 8192
    python intent/eval/run.py --label qwen3.5-2b-q4 [--shots] [--no-schema]

The system prompt is spec.md up to "## Scoring", or with --short-prompt the
one-line prompt the fine-tune is trained with. With --shots, the examples in
shots.jsonl (none of them in cases.jsonl) go in as prior chat turns. Without
--no-schema, decoding is constrained by ../intent.gbnf (see ../grammar.py).
"""

import argparse
import json
import time
from pathlib import Path

from openai import OpenAI

HERE = Path(__file__).parent
INTENT = HERE.parent


def system_prompt():
    spec = (INTENT / "spec.md").read_text("utf-8").split("## Scoring")[0].strip()
    return (
        "You convert a user's anonymisation instruction into one intent JSON object. "
        "Reply with the JSON object only.\n\n" + spec
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="names the output file")
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--shots", action="store_true")
    ap.add_argument("--no-schema", action="store_true")
    ap.add_argument("--short-prompt", action="store_true", help="system_prompt.txt, as in fine-tuning")
    ap.add_argument("--cases", type=Path, default=HERE / "cases.jsonl")
    args = ap.parse_args()

    client = OpenAI(base_url=args.url, api_key="none")
    system = (INTENT / "system_prompt.txt").read_text("utf-8").strip() if args.short_prompt else system_prompt()
    prefix = [{"role": "system", "content": system}]
    if args.shots:
        for line in (HERE / "shots.jsonl").read_text("utf-8").splitlines():
            shot = json.loads(line)
            prefix += [
                {"role": "user", "content": shot["instruction"]},
                {"role": "assistant", "content": json.dumps(shot["expect"], ensure_ascii=False)},
            ]
    extra = {"chat_template_kwargs": {"enable_thinking": False}}
    if not args.no_schema:
        # A bare grammar, not response_format: see ../grammar.py.
        extra["grammar"] = (INTENT / "intent.gbnf").read_text("utf-8")

    tag = args.label + ("-shots" if args.shots else "") + ("-noschema" if args.no_schema else "")
    tag += "-short" if args.short_prompt else ""
    tag += "" if args.cases == HERE / "cases.jsonl" else f"-{args.cases.parent.name}"
    out = HERE / "results" / f"{tag}.jsonl"
    out.parent.mkdir(exist_ok=True)
    cases = [json.loads(l) for l in args.cases.read_text("utf-8").splitlines()]
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for i, case in enumerate(cases, 1):
            start = time.perf_counter()
            reply = client.chat.completions.create(
                model=args.label,
                messages=prefix + [{"role": "user", "content": case["instruction"]}],
                temperature=0,
                max_tokens=256,
                extra_body=extra,
            )
            ms = round((time.perf_counter() - start) * 1000)
            raw = reply.choices[0].message.content or ""
            f.write(json.dumps({"id": case["id"], "raw": raw, "ms": ms}, ensure_ascii=False) + "\n")
            print(f"\r{i}/{len(cases)}", end="", flush=True)
    print(f"\n{out}")


if __name__ == "__main__":
    main()
