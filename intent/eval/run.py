# Copyright 2026 the Docudis contributors. Licensed under Apache-2.0.
"""Runs the eval cases through a llama-server (OpenAI-compatible) and writes predictions.

    llama-server -m model.gguf --port 8080 -ngl 99 -c 8192
    python intent/eval/run.py --label qwen3.5-2b-q4 [--shots] [--no-schema]

The system prompt is spec.md up to "## Scoring", or with --short-prompt the
one-line prompt the fine-tune is trained with. With --shots, the examples in
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


def gbnf():
    """GBNF for schema.json without whitespace, as the fine-tune is trained.

    Written here rather than converted by llama.cpp, whose converter treats a
    property named "*" as its additional-properties marker and forces one key
    order. Keys may come in any order: the model does not always keep the
    training order, and a grammar that does forbids the key it wanted next. A
    repeated key is possible; JSON parsing keeps the last one. Empty lists and
    `"unsupported": false` are allowed (the scorer drops them, as the host
    would): forbidding them forces a model that wants `[]` to invent an item.
    """
    schema = json.loads((INTENT / "schema.json").read_text("utf-8"))
    props = schema["properties"]
    lit = lambda s: json.dumps(json.dumps(s))  # GBNF literal of a JSON string
    alt = lambda values: " | ".join(lit(v) for v in values)

    def obj(name, kvs):
        kv = f"( {' | '.join(kvs)} )"
        return f'{name} ::= "{{" ( {kv} ( "," {kv} )* )? "}}"'

    def array(item):
        return f'"[" ( {item} ( "," {item} )* )? "]"'

    rules = [
        'string ::= "\\"" ( [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" ( ["\\\\/bfnrt] | "u" [0-9a-fA-F]{4} ) )* "\\""',
        f"action ::= {alt(props['types']['additionalProperties']['enum'])}",
    ]
    type_kvs = []
    for i, key in enumerate(props["types"]["propertyNames"]["enum"]):
        rules.append(f'type{i} ::= {lit(key)} ":" action')
        type_kvs.append(f"type{i}")
    rules.append(obj("types", type_kvs))
    values = {
        "types": "types",
        "regions": array(f"( {alt(props['regions']['items']['enum'])} )"),
        "verticals": array(f"( {alt(props['verticals']['items']['enum'])} )"),
        "dictionary": array("string"),
        "never_hide": array("string"),
        "unsupported": '( "true" | "false" )',
    }
    field_kvs = []
    for key in props:
        rules.append(f'f-{key.replace("_", "-")} ::= {lit(key)} ":" {values[key]}')
        field_kvs.append(f'f-{key.replace("_", "-")}')
    rules.append(obj("root", field_kvs))
    return "\n".join(rules)


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
        # A bare grammar, not response_format: with response_format llama-server
        # changes the prompt, and the fine-tune then starts "thinking".
        extra["grammar"] = gbnf()

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
