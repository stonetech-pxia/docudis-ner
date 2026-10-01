# On-device NER models

The model binaries are **not** committed (see `.gitignore`); only each folder's
`model.json` (the spec the app reads) is tracked. `manifest.json` pins every
binary to a Hugging Face repo, revision and SHA-256. On a fresh clone:

```bash
pip install huggingface_hub
huggingface-cli login                                # once: xlmr_ner_docudis is a private repo
"$PYTHON" tool/fetch_models.py                       # the shipped model
"$PYTHON" tool/fetch_models.py --all                 # plus the two stock models, for A/B benchmarks
```

| Folder | Hugging Face repo | Use |
|---|---|---|
| `xlmr_ner_docudis` | private, see `manifest.json` | **Shipped since 2026-09-21.** Our fine-tune of `Davlan/xlm-roberta-base-ner-hrl` on en / fr / es material, AFL-3.0 |
| `xlmr_ner_hrl` | `tjruesch/xlm-roberta-base-ner-hrl-onnx` | The stock model it was fine-tuned from, for A/B benchmarks |
| `distilbert_ner_hrl` | `Xenova/distilbert-base-multilingual-cased-ner-hrl` | Smaller/faster alternative, for A/B benchmarks |

**Retraining `xlmr_ner_docudis`** happens on the Windows machine only (it needs
the NVIDIA GPU): `../training/README.md` has the pipeline, and
`docs/HANDOFF-ner-finetune-2026-09-19.md` in docudis-android §3.5a what each run
scored. After exporting a new model to `models/xlmr_ner_docudis`, benchmark it in
docudis-android, then publish it and move the pin:

```bash
huggingface-cli upload <repo> models/xlmr_ner_docudis . --include "model_quantized.onnx" "tokenizer.json" "model.json"
```

then put the commit id it prints and the new SHA-256 of both files into
`manifest.json`, and commit.

`model.json` fields: `tokenizer.kind` (`wordpiece` or `sentencepiece`) and
`tokenizer.file`, `labels` in model output order (BIO), `labelMap` from model
labels to Docudis entity types, `maxTokens`, `stride`, `threshold`, `padId`, the
ONNX input names (`inputs.ids`, `inputs.mask`) and output name (`output`).

How the Android app gets the files is owned by docudis-android (Play Asset
Delivery pack `android/model_pack`, `ModelLocator`). Until it switches to this
repository (see its `docs/HANDOFF-rust-ner-2026-10-01.md`), the app keeps its
own copy of these specs under `assets/models`.

Licenses: both XLM-R models are Academic Free License 3.0 (Davlan).
