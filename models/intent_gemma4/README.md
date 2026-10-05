---
license: apache-2.0
language:
- zh
- en
- fr
- es
- de
- it
base_model: google/gemma-4-E2B-it
pipeline_tag: text-generation
library_name: gguf
tags:
- gguf
- gemma4
- anonymization
- pii
- structured-output
---

# Docudis intent (docudis-intent-gemma4)

The instruction-reading model of [Docudis](https://docudis.com), which replaces personal details in
documents with labels on the user's device. The user says in their own words what to hide or keep
("hide names and phone numbers, the amounts can stay, it's a French payslip"); this model turns that
into a small JSON object that Docudis maps onto its detection settings. **It reads only the
instruction, never the document.**

```text
这是法国的工资单，遮住名字和电话，金额留着
→ {"types":{"PERSON":"hide","PHONE":"hide","AMOUNT":"keep"},"regions":["fr"],"verticals":["employment"]}
```

- **Format:** GGUF for llama.cpp, `intent-mixq8.gguf` (3.6 GB): Q8_0, with the per-layer embeddings at
  Q4_K and the token embeddings at Q8_0.
- **Also here:** `system_prompt.txt` (the one line the model was trained with), `intent.gbnf` (the
  grammar for constrained decoding) and `model.json` (run settings).
- **Code:** the output format, post-processing, evaluation and training pipeline, at the commit this model was built from, are in
  [docudis-ner/intent](https://github.com/stonetech-pxia/docudis-ner/tree/7592d8d/intent) (Apache-2.0).

## Output

One JSON object, every field optional; `{}` means "use the defaults".

| Field | Meaning |
|---|---|
| `types` | Entity type (`PERSON EMAIL PHONE ID NUMBER CARD IBAN DATE BIRTH_DATE AMOUNT IP URL ADDRESS COMPANY SECRET API_KEY`, or `*` for every other type) to `hide`, `keep` (detect, leave visible) or `off` (do not detect) |
| `regions` | Country rule packs: `at be ch cn de dk es fi fr gb ie it jp nl no pl pt se us` |
| `verticals` | Document domain: `healthcare legal finance employment insurance technology utilities` |
| `dictionary` | Literal terms to hide, copied from the instruction |
| `never_hide` | Literal terms to leave visible, copied from the instruction |
| `unsupported` | `true` when part of the request cannot be done (fake names, partial masking, translation, keeping only some values of a type: one person's name, "my employer's name", "the hospital", "the rent"); the model has then hidden the whole type, the safe side |

The full rules are in [intent/spec.md](https://github.com/stonetech-pxia/docudis-ner/blob/7592d8d/intent/spec.md).

## How to run it

```sh
llama-server -m intent-mixq8.gguf -ngl 99 -c 2048 --jinja --reasoning off --reasoning-budget 0
```

Send `system_prompt.txt` as the system message and the user's instruction as the user message, with
temperature 0 and the contents of `intent.gbnf` as the request's `grammar`. Do not use
`response_format`: llama-server then changes the prompt and the model starts with a reasoning block.
`--reasoning off` is needed for the same reason.

The reply is one JSON object. Drop empty lists and `"unsupported": false`, validate it against
[intent/schema.json](https://github.com/stonetech-pxia/docudis-ner/blob/7592d8d/intent/schema.json) (use
`{}` if it does not validate), then run the host post-processing in
[intent/postprocess.py](https://github.com/stonetech-pxia/docudis-ner/blob/7592d8d/intent/postprocess.py):
it drops `keep`, `off` and `*` that the instruction does not support (prompt injection, "do whatever
with the rest"), drops literal terms that are not in the instruction, and adds regions and verticals
from a keyword list. The scores below include it.

## How it was made

LoRA fine-tune of [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) with
[Unsloth](https://github.com/unslothai/unsloth): base loaded in 4 bits (QLoRA), rank 16, alpha 16, no
dropout, on the language model's attention and MLP layers; 5 epochs, learning rate 2e-4 with a cosine
schedule and 5% warm-up, weight decay 0.01, effective batch 16, sequences up to 512 tokens, loss on
the reply only, seed 2. The adapter was merged into the 16-bit base and converted with llama.cpp
b11379.

The training data is 1,309 instruction → JSON pairs (145 more held out for validation) in Chinese,
English, French, Spanish, German, Italian and mixed-language messages. They are **synthetic**: written
with an LLM (Claude) against the spec, from short commands to long, informal or misspelled messages
and attempts to override the model, then reviewed and corrected case by case by a second LLM pass.
They contain no real personal data; every name and identifier is invented. Cases close to any
evaluation instruction were removed before training. The data is in
[intent/train/raw](https://github.com/stonetech-pxia/docudis-ner/tree/7592d8d/intent/train/raw).

## Evaluation

Exact match of the whole JSON object, temperature 0, constrained decoding, with the host
post-processing:

| Set | Cases | Exact match | Leaks |
|---|---|---|---|
| Test, frozen: written independently of the training data and never used to tune it | 300 | **90.3%** | 7 (2.3%) |
| Dev, used to steer the training data | 272 | 90.4% | 4 |

A *leak* is a miss that leaves visible something the user wanted hidden. The four on dev: "they only
need the rent amount and the dates" read as keeping all amounts, a named hospital kept along with
every company, "the rest whatever" read as keeping every other type, and one reply with a repeated
key (see Limitations). Without post-processing the test score is 87.7%; most of that difference is
regions and verticals the model leaves out. Two training seeds on the same data differ by about one
point (dev 89.3% and 90.4%). For comparison, the base model prompted with the full spec and six
examples scored 43.5% on the first 200 dev cases.

Both sets are scored on the labels as revised on 2026-10-05: a keep narrowed to some values of a type
("my employer's name", "the rent") counts as unsupported, not as keeping the whole type. The previous
release scores 87.3% with 13 leaks on test under these labels (89.7% with 4 under the old ones).

Speed for one instruction (output 15–30 tokens): about 0.12 s on an RTX 3080, about 3 s on a desktop
CPU (Intel i7-13700KF, no GPU).

## Limitations

- It reads instructions in the six languages above; other languages are untested.
- Docudis cannot tell one value of a type from another (a doctor's name from a patient's, the rent
  from the deposit). Such requests come back as the whole type hidden plus `unsupported`, and
  occasionally as the whole type kept: most of the leaks above. A host should show the user what will
  be kept visible before running.
- The grammar does not stop a key from repeating, and the model has written replies such as
  `{"types":{"*":"off",…,"*":"hide"}}`. JSON parsers disagree on which value wins, so a host should
  treat a reply with a repeated key as invalid.
- PERSON and COMPANY are sometimes confused for roles like "landlord" or "employer", and `API_KEY`
  sometimes comes back as `SECRET`.
- The training and test instructions are synthetic; real users phrase things differently, and the
  scores on their messages may be lower.
- It depends on the host post-processing for prompt-injection and keyword checks. Used alone, it can
  follow an instruction that dictates its JSON.

## Versions

| Revision | Date | Test | Change |
|---|---|---|---|
| `e25d293` | 2026-10-05 | 90.3%, 7 leaks | A keep of only part of a type ("my employer's name", "the rent") is now `unsupported`; 41 new training instructions; 5 epochs |
| `f73e207` | 2026-10-04 | 87.3%, 13 leaks (89.7%, 4 leaks on the old answers) | First release |

Load an older version with its revision, for example `hf_hub_download(..., revision="f73e207")`.
The full history, with the reason for each change, is in
[intent/HISTORY.md](https://github.com/stonetech-pxia/docudis-ner/blob/main/intent/HISTORY.md).

## License

Copyright 2026 Pengda Xia (stonetech). Licensed under the [Apache License 2.0](LICENSE).

This model is a derivative of google/gemma-4-E2B-it (Apache License 2.0, Copyright Google LLC),
modified by stonetech by fine-tuning it on the data above, merging the adapter and quantizing it.
