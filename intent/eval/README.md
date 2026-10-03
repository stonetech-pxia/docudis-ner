# Intent evaluation

- `cases.jsonl`: 200 instructions with the expected intent (see `../spec.md`).
- `shots.jsonl`: few-shot examples for prompting; none of them is in the eval set.
- `validate.py`: checks `cases.jsonl` against `../schema.json`.
- `run.py`: sends every case to a llama-server and writes `results/<label>.jsonl`.
- `score.py`: exact match per `../spec.md` "Scoring", with breakdowns by field,
  language and tag; `--failures` prints each miss.

```sh
llama-server -m <model>.gguf --port 8080 -ngl 99 -c 8192 --jinja --reasoning off --reasoning-budget 0
python intent/eval/run.py --label <name> [--shots] [--no-schema] [--short-prompt]
python intent/eval/score.py intent/eval/results/*.jsonl
```

Without `--no-schema`, decoding is constrained by the GBNF grammar `run.py`
writes for `../schema.json`, passed as llama-server's `grammar`:

- Not `response_format`: with it llama-server changes the prompt, and the
  fine-tuned model starts with a reasoning block (250 tokens, sometimes no
  answer). `--reasoning off` is needed for the same reason.
- Not llama.cpp's schema converter: it treats a property named `"*"` as its
  additional-properties marker and forces a key order the model does not
  always follow.
- Keys may come in any order, so a key can repeat (seen twice in 200 cases).
  Empty lists and `"unsupported": false` are allowed and dropped by the
  scorer, as the host would: forbidding them forces a model that wants `[]`
  to invent an item.

## Results

llama.cpp b11379, Q4_K_M, temperature 0, RTX 3080.

| Model | Prompt | Constrained | Exact match | Invalid | p50 latency |
|---|---|---|---|---|---|
| Gemma 4 E2B fine-tune, round 2 (1234 cases) | one line | yes | **83.0%** | 0 | 108 ms |
| Gemma 4 E2B fine-tune, round 1 (882 cases) | one line | yes | 67.0% | 0 | 100 ms |
| Gemma 4 E2B fine-tune, round 1 | one line | no | 67.5% | 3 | 107 ms |
| Gemma 4 E2B-it (unsloth GGUF) | spec + 6 shots | yes* | 43.5% | 0 | 197 ms |
| Gemma 4 E2B-it | spec + 6 shots | no | 43.5% | 12 | 121 ms |
| Gemma 4 E2B-it | spec | yes* | 32.0% | 0 | 227 ms |
| Qwen3.5-2B (unsloth GGUF) | spec + 6 shots | yes* | 20.0% | 1 | 135 ms |
| Qwen3.5-2B | spec | yes* | 23.0% | 1 | 315 ms |
| Qwen3.5-2B | spec + 6 shots | no | 23.0% | 7 | 125 ms |

\* Baselines were constrained with `response_format` and llama.cpp's schema
converter, before the grammar above.

Round 2 data targeted what round 1 got wrong on this set, so these 200 cases
now act as a dev set: the scores overstate how the model does on instructions
nobody has looked at. Remaining misses after round 2: a vertical left out
when a region is also set, injected JSON obeyed or flagged `unsupported`,
unsupported requests (translate, asterisks, fake names) not flagged, and a
name to keep put in `dictionary` (zh-009).
