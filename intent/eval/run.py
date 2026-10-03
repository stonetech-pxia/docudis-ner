# Copyright 2026 the Docudis contributors. Licensed under Apache-2.0.
"""Runs the eval cases through a llama-server (OpenAI-compatible) and writes predictions.

    llama-server -m model.gguf --port 8080 -ngl 99 -c 8192
    python intent/eval/run.py --label qwen3.5-2b-q4 [--shots] [--no-schema]

The system prompt is spec.md up to "## Scoring". With --shots, the examples in
shots.jsonl (none of them in cases.jsonl) go in as prior chat turns. Without
--no-schema, decoding is constrained to schema.json.
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


def grammar_schema():
    # llama.cpp's grammar converter ignores propertyNames, so list the type keys
    # as optional properties instead. Empty lists and `"unsupported": false` are
    # allowed (the scorer drops them, as the host would): with minItems the
    # grammar forces a model that wants `[]` to invent an item.
    schema = json.loads((INTENT / "schema.json").read_text("utf-8"))
    props = schema["properties"]
    types = props["types"]
    action = types.pop("additionalProperties")
    keys = types.pop("propertyNames")["enum"]
    types.pop("minProperties")
    types["properties"] = {k: action for k in keys}
    types["additionalProperties"] = False
    for key in ("regions", "verticals", "dictionary", "never_hide"):
        props[key].pop("minItems")
    props["unsupported"] = {"type": "boolean"}
    return schema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="names the output file")
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--shots", action="store_true")
    ap.add_argument("--no-schema", action="store_true")
    ap.add_argument("--cases", type=Path, default=HERE / "cases.jsonl")
    args = ap.parse_args()

    client = OpenAI(base_url=args.url, api_key="none")
    prefix = [{"role": "system", "content": system_prompt()}]
    if args.shots:
        for line in (HERE / "shots.jsonl").read_text("utf-8").splitlines():
            shot = json.loads(line)
            prefix += [
                {"role": "user", "content": shot["instruction"]},
                {"role": "assistant", "content": json.dumps(shot["expect"], ensure_ascii=False)},
            ]
    extra = {"chat_template_kwargs": {"enable_thinking": False}}
    response_format = None
    if not args.no_schema:
        response_format = {"type": "json_schema", "json_schema": {"name": "intent", "schema": grammar_schema()}}

    tag = args.label + ("-shots" if args.shots else "") + ("-noschema" if args.no_schema else "")
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
                response_format=response_format,
                extra_body=extra,
            )
            ms = round((time.perf_counter() - start) * 1000)
            raw = reply.choices[0].message.content or ""
            f.write(json.dumps({"id": case["id"], "raw": raw, "ms": ms}, ensure_ascii=False) + "\n")
            print(f"\r{i}/{len(cases)}", end="", flush=True)
    print(f"\n{out}")


if __name__ == "__main__":
    main()
