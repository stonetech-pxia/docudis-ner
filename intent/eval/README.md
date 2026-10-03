# Intent evaluation

- `cases.jsonl`: 200 instructions with the expected intent (see `../spec.md`).
- `shots.jsonl`: few-shot examples for prompting; none of them is in the eval set.
- `validate.py`: checks `cases.jsonl` against `../schema.json`.
- `run.py`: sends every case to a llama-server and writes `results/<label>.jsonl`.
- `score.py`: exact match per `../spec.md` "Scoring", with breakdowns by field,
  language and tag; `--failures` prints each miss.

```sh
llama-server -m <model>.gguf --port 8080 -ngl 99 -c 8192 --jinja
python intent/eval/run.py --label <name> [--shots] [--no-schema]
python intent/eval/score.py intent/eval/results/*.jsonl
```

Without `--no-schema`, decoding is constrained to the schema. The grammar
allows empty lists and `"unsupported": false`, which the scorer drops as the
host would: with `minItems`, a model that wants `[]` is forced to invent an
item.

## Baseline (2026-10-03, before fine-tuning)

llama.cpp b11379, Q4_K_M, temperature 0, thinking off, RTX 3080. The system
prompt is `spec.md` up to "Scoring".

| Model | Prompt | Constrained | Exact match | Invalid | p50 latency |
|---|---|---|---|---|---|
| Gemma 4 E2B-it (unsloth GGUF) | spec + 6 shots | yes | **43.5%** | 0 | 197 ms |
| Gemma 4 E2B-it | spec + 6 shots | no | 43.5% | 12 | 121 ms |
| Gemma 4 E2B-it | spec | yes | 32.0% | 0 | 227 ms |
| Qwen3.5-2B (unsloth GGUF) | spec + 6 shots | yes | 20.0% | 1 | 135 ms |
| Qwen3.5-2B | spec | yes | 23.0% | 1 | 315 ms |
| Qwen3.5-2B | spec + 6 shots | no | 23.0% | 7 | 125 ms |

Both models score 0% on `only` (they leave out `"*": "off"`) and miss most
`region`, `vertical` and `unsupported` cases, which is what fine-tuning has
to teach. The Q4_K_M files are 1.2 GB (Qwen) and 3.0 GB (Gemma).
