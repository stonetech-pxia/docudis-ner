# Intent model

A small on-device LLM (Gemma 4 E2B, fine-tuned) reads the user's instruction,
never the document, and outputs an intent JSON that the host maps onto Core's
v1 `DetectRequest`.

- `spec.md`: the intent fields, how each maps to Core, annotation rules.
- `schema.json`: the JSON Schema. `eval/run.py` writes the GBNF grammar used
  for constrained decoding from it.
- `system_prompt.txt`: the one-line system prompt the model is trained and run
  with.
- `keywords.json`, `postprocess.py`: host post-processing (below).
- `eval/`: dev set (200), runner and scorer. `test/`: frozen test set (300).
  `train/`: training data, fine-tuning and export.

## Running the model

```sh
llama-server -m intent-mixq8.gguf -ngl 99 -c 2048 --jinja --reasoning off --reasoning-budget 0
```

Send the system prompt and the instruction as chat messages with
temperature 0 and the GBNF grammar from `eval/run.py` as `grammar`, not as
`response_format` (see `eval/README.md`). The reply is one JSON object; drop
empty lists and `"unsupported": false`, then validate against `schema.json`.
If it does not validate, use `{}`.

## Host post-processing

`keywords.json` lists words that name a country (rule 6) or a document kind
(rule 7). The host adds every region and vertical they find to the model's
lists, and never removes what the model set. On the test set this takes exact
match from 84.7% to 89.0%; the misses it removes are regions and verticals the
model left out. `postprocess.py` is the reference; hosts port `keyword_hits`
with the same matching:

1. Fold the instruction: curly apostrophe to `'`, `ß` to `ss`, lowercase,
   Unicode NFKD, drop combining marks (so `relevé` matches `releve`).
2. For each group (`regions.<code>`, `verticals.<name>`), remove from the
   folded text every phrase in `except.regions` (for regions) or
   `except.<name>` (for that vertical), each replaced by a space.
3. A keyword made only of ASCII characters matches when it occurs with no
   letter or digit right before or after it; any other keyword (Chinese,
   Japanese) matches as a substring. Keywords are folded the same way.
4. Keywords in `case_sensitive` (`NIE`, `EIN`, `CPR`, which are also common
   words) are matched against the instruction folded without lowercasing.

Known misses, by design: a country that is only where the document is sent
("for my cousin in Germany") still adds that region, which loads one more rule
pack; a document quoted inside another (a medical record in a judgment) adds
both verticals.
