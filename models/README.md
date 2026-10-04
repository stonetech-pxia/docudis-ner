# On-device NER models

The model binaries are **not** committed (see `.gitignore`); only each folder's
`model.json` (the spec the app reads) is tracked. `manifest.json` pins every
binary to a Hugging Face repo, revision and SHA-256. On a fresh clone:

```bash
pip install huggingface_hub
"$PYTHON" tool/fetch_models.py                       # the shipped model
"$PYTHON" tool/fetch_models.py --all                 # plus the stock models and openai_privacy_filter (~950 MB)
```

| Folder | Hugging Face repo | Use |
|---|---|---|
| `xlmr_ner_docudis` | [`leonx1995/docudis-ner-xlmr`](https://huggingface.co/leonx1995/docudis-ner-xlmr) | **Shipped since 2026-09-21.** Our fine-tune of `Davlan/xlm-roberta-base-ner-hrl` on en / fr / es material, AFL-3.0; model card in `xlmr_ner_docudis/README.md` |
| `xlmr_ner_hrl` | `tjruesch/xlm-roberta-base-ner-hrl-onnx` | The stock model it was fine-tuned from, for A/B benchmarks only; not shipped |
| `distilbert_ner_hrl` | `Xenova/distilbert-base-multilingual-cased-ner-hrl` | Smaller/faster alternative, for A/B benchmarks only; not shipped |
| `openai_privacy_filter` | `openai/privacy-filter` | Desktop only (Windows, macOS), not for Android: the q4 ONNX export is 917 MB and needs ~1.6 GB after load. Apache-2.0 |
| `intent_gemma4` | [`leonx1995/docudis-intent-gemma4`](https://huggingface.co/leonx1995/docudis-intent-gemma4) | Not a NER model: reads the user's instruction (never the document) and returns the intent JSON of `../intent/spec.md`. GGUF for llama.cpp, 3.6 GB, plus its system prompt and grammar. Our fine-tune of `google/gemma-4-E2B-it`, Apache-2.0; model card in `intent_gemma4/README.md`. Fetch it by name (`tool/fetch_models.py intent_gemma4`); `model.json` holds its run settings instead of NER fields |

`openai_privacy_filter` labels only what it judges *private* (a person, a private
address or date, e-mail, phone, URL, account number, secret), mostly in English;
it has no organisation label. Its `account_number` label is deliberately left
out of `labelMap` until its Docudis type is decided, so those spans are dropped.
The ONNX export's attention memory grows with the square of the input length
(4096 tokens peak at ~8.7 GB), so it runs in 512-token windows like the others.

**Retraining `xlmr_ner_docudis`** happens on the Windows machine only (it needs
the NVIDIA GPU): `../training/README.md` has the pipeline, and
`docs/HANDOFF-ner-finetune-2026-09-19.md` in docudis-android §3.5a what each run
scored. After exporting a new model to `models/xlmr_ner_docudis`, benchmark it in
docudis-android, then publish it and move the pin:

```bash
huggingface-cli upload <repo> models/xlmr_ner_docudis . --include "model_quantized.onnx" "tokenizer.json" "model.json" "README.md" "LICENSE"
```

then put the commit id it prints and the new SHA-256 of both files into
`manifest.json`, and commit.

`model.json` fields: `tokenizer.kind` (`wordpiece`, `sentencepiece` or
`bytelevel`) and `tokenizer.file`, `labels` in model output order, `labelMap`
from model labels to Docudis entity types, `maxTokens` (model inputs per window,
special tokens included), `stride` (overlap between windows), `threshold` (minimum
mean probability of a span), `padId`, the ONNX input names (`inputs.ids`,
`inputs.mask`) and output name (`output`). Optional: `titleCase` (default `true`:
ALL-CAPS words are title-cased before tokenizing) and `decoder`, either
`{"kind": "bio"}` (the default: argmax per token, BIO labels) or
`{"kind": "bioes-viterbi", "biases": {...}}` (constrained Viterbi over BIOES
labels; `biases` holds the six transition biases of a `viterbi_calibration.json`
operating point, in camelCase: `backgroundStay`, `backgroundToStart`,
`endToBackground`, `endToStart`, `insideToContinue`, `insideToEnd`).
A tokenizer that adds no start and end tokens (byte-level BPE) gets none in its
windows either.

Apps do not keep their own copy. docudis-android pins a revision of this
repository in `tool/docudis_ner_version.json`, and its `tool/fetch_models.sh`
runs `tool/fetch_models.py --dest <app>/assets/models` at that revision, which
writes each `model.json` and its verified binaries into the app's git-ignored
model directory. Packaging (Play Asset Delivery pack `android/model_pack`,
`ModelLocator`) stays in the app.

Licenses: both XLM-R models are Academic Free License 3.0 (Davlan);
openai/privacy-filter is Apache-2.0.
