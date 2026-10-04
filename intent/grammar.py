# Copyright 2026 the Docudis contributors. Licensed under Apache-2.0.
"""Writes intent.gbnf, the grammar that constrains the model's output, from schema.json.

    python intent/grammar.py          # rewrite intent.gbnf
    python intent/grammar.py --check  # fail if intent.gbnf is stale

The grammar has no whitespace, as the fine-tune is trained. It is written here
rather than converted by llama.cpp, whose converter treats a property named "*"
as its additional-properties marker and forces one key order. Keys may come in
any order: the model does not always keep the training order, and a grammar
that does forbids the key it wanted next. A repeated key is possible; JSON
parsing keeps the last one. Empty lists and `"unsupported": false` are allowed
(the host drops them): forbidding them forces a model that wants `[]` to invent
an item.

Pass the grammar as llama-server's `grammar`, not as `response_format`: with
`response_format` llama-server changes the prompt and the model starts
"thinking".
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
GRAMMAR = HERE / "intent.gbnf"


def gbnf():
    schema = json.loads((HERE / "schema.json").read_text("utf-8"))
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
    return "\n".join(rules) + "\n"


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(0 if GRAMMAR.read_text("utf-8") == gbnf() else "intent.gbnf is stale; run python intent/grammar.py")
    GRAMMAR.write_text(gbnf(), encoding="utf-8", newline="\n")
    print(GRAMMAR)
