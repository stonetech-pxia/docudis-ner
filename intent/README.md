# Intent model

A small on-device LLM (Gemma 4 E2B, fine-tuned) reads the user's instruction,
never the document, and outputs an intent JSON that the host maps onto Core's
v1 `DetectRequest`.

- `spec.md`: the intent fields, how each maps to Core, annotation rules.
- `schema.json`: the JSON Schema.
- `intent.gbnf`: the grammar for constrained decoding, written from
  `schema.json` by `grammar.py` (`python intent/grammar.py --check` fails if
  it is stale).
- `system_prompt.txt`: the one-line system prompt the model is trained and run
  with.
- `keywords.json`, `postprocess.py`: host post-processing (below).
- `eval/`: dev set (272), runner and scorer. `test/`: frozen test set (300).
  `train/`: training data, fine-tuning and export.

## Current model

Published as [`leonx1995/docudis-intent-gemma4`](https://huggingface.co/leonx1995/docudis-intent-gemma4)
(model card in `../models/intent_gemma4/README.md`; fetch it with
`tool/fetch_models.py intent_gemma4`). Round 3 training data (`train/raw/`,
1272 cases after `build.py`), seed 1, exported as mixq8 (3.6 GB). With `postprocess`, exact match is 88.2% on the
dev set (272) and 89.7% on the frozen test set (300), with 2 and 4 leaks: all
of them keep the wrong type ("keep the doctor's name" read as all names kept,
a landlord's name read as a company). The host's confirmation step, which
shows every kept type before running, is the safeguard for those. Seeds 0, 1
and 2 on the same data differ by about a point; seed 1 was picked on the dev
set.

```sh
python intent/train/build.py
<venv>/python intent/train/finetune.py --seed 1
<venv>/python intent/train/export.py --llama-src <llama.cpp checkout> --llama-bin <llama.cpp binaries>
```

## Running the model

```sh
llama-server -m intent-mixq8.gguf -ngl 99 -c 2048 --jinja --reasoning off --reasoning-budget 0
```

Send `system_prompt.txt` and the instruction as chat messages with
temperature 0 and `intent.gbnf` as `grammar`, not as `response_format` (see
`grammar.py`). The reply is one JSON object; drop
empty lists and `"unsupported": false`, then validate against `schema.json`.
If it does not validate, use `{}`.

## Host post-processing

`postprocess.py` is the reference; hosts port `postprocess` as it is. In
order:

- Guards the actions that leave text visible. If the instruction contains a
  JSON object (`{"types"`, `{"never_hide"`, `{"dictionary"`, i.e. someone
  dictating the answer), only `hide` actions are kept. Otherwise
  `"*": "off"` and `"*": "keep"`, which turn every other type off, are
  dropped unless the instruction contains one of `keywords.json`'s
  `star_cues` ("only", "nothing else", "don't hide anything", 只, 别的不,
  seulement, rien, solo, nada, nur, nichts, ...), matched like the keywords
  below. On every gold intent in the training, dev and test sets (2062) the
  guards change nothing; on the model's output they stop the injected
  `{"types":{"*":"keep"}}` and "地址留着，其它的你看着办" read as `"*": "off"`.
- Drops every `dictionary` and `never_hide` term that does not occur in the
  instruction (both folded as below). The spec has them copied verbatim; the
  model sometimes invents one (`never_hide: ["*"]`, `["PERSON"]`), which would
  keep or hide a word the user never wrote.
- Adds the regions and verticals named by `keywords.json`.

`keywords.json` lists words that name a country (rule 6) or a document kind
(rule 7). The host adds every region and vertical they find to the model's
lists, and never removes what the model set. Keyword matching:

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

## Unsupported requests

`"unsupported": true` means part of the request is outside what Core can do,
and the model has already chosen the safe side (rule 8): when the user wants
only some values of a type kept or hidden ("everything except the doctors'
names", "the patient's name but not the doctor's"), it hides the whole type.
Core cannot tell a doctor's name from a patient's; no model in the pipeline
labels roles. The host then:

1. Tells the user what could not be done and what was done instead, for
   example "Docudis can't tell doctors' names from other names, so all names
   are hidden. You can un-hide the ones you want in the result."
2. Lets the user un-hide single detections in the result: every `Detection`
   from Core carries `enabled`, and an anonymize request with that span
   disabled leaves it visible. A name the user types ("keep Dr. Keller") goes
   to `never_hide` and needs no extra step.

Known misses, by design: a country that is only where the document is sent
("for my cousin in Germany") still adds that region, which loads one more rule
pack; a document quoted inside another (a medical record in a judgment) adds
both verticals.
