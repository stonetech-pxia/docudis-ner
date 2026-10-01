# NER fine-tuning material

Everything needed to turn `Davlan/xlm-roberta-base-ner-hrl` into the model the app ships, for English,
French and Spanish. Scope and the ten weaknesses this data targets: [docs/ner-finetune-scope.md](../docs/ner-finetune-scope.md).
Labels and file formats: [GUIDE.md](GUIDE.md).

**The test sets are off limits here.** `benchmark/` in docudis-android (public records, synthetic documents, the frozen consumer
set, hand-written cases, hard negatives) is never read when writing training material, and
`check_isolation.py` drops anything that repeats a test entity or a test text.

## Pipeline

```bash
py=C:/Users/Xia/anaconda3/python.exe
export DOCUDIS_APP_ROOT=../docudis-android      # the test sets live there; the isolation check refuses to run without it
$py training/fetch_registry.py                 # real announcements -> training/out/registry.jsonl (network)
$py training/check_isolation.py                # written material vs every test set
$py training/build.py --chunks 30000           # -> training/out/{train,dev}.jsonl + build_report.md
$py training/check_isolation.py --jsonl training/out/train.jsonl training/out/dev.jsonl
conda run -n docudis-ner python training/train.py        # the GPU environment, see below
conda run -n docudis-ner python training/export_onnx.py  # -> models/xlmr_ner_docudis
```

The training environment is `docudis-ner` (created 2026-09-20, Anaconda at `C:/Users/Xia/anaconda3`):

```bash
conda create -y -n docudis-ner python=3.11
conda run -n docudis-ner python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
conda run -n docudis-ner python -m pip install transformers datasets seqeval accelerate sentencepiece protobuf "optimum[onnxruntime]"
```

It has torch 2.6.0+cu124, transformers 4.57.6, datasets 5.0.1, accelerate 1.15.0, onnxruntime 1.30 (the version
the phone needs) on an RTX 3080 (10 GB). `sentencepiece` is not optional: without it the XLM-R tokenizer fails to
load with a confusing `'NoneType' object has no attribute 'endswith'`. Batch 16 at length 256 fits in 10 GB;
use `--batch 8 --accum 2` if something else is using the GPU.

| File | What it is |
|---|---|
| `GUIDE.md` | The label definitions, the inline markup, the template slots and the inventory schema. The only document the writing subagents read. |
| `inventories/<COUNTRY>.json` | Names, organisations, streets, towns, public bodies, departments and jobs for GB, IE, US, FR, BE, CH, ES. |
| `templates/<lang>/<COUNTRY>-<doctype>-NN.txt` | Documents with slots (`{{PER#1:full}}`, `{{LOC#1:line}}`, `{{IBAN}}`…). 60 % of the training set. |
| `documents/<lang>/<COUNTRY>-<focus>-NNN.txt` | Complete documents written by hand with inline markup (`[[PER|…]]`), plus `neg-*.txt` where words that are also names are used as ordinary words. 15 %. |
| `markup.py` | Parser and checker for all three formats. Every writer runs `--check` until it reports 0 problems. |
| `build.py` | Fills the templates (per-country identifiers, phones, dates, amounts with valid check digits), augments (name casing, registry order, all-caps documents, OCR noise, wrapped entities), oversamples the hand-written documents with substituted names, keeps the mix and the country shares, and splits 10 % of source files into dev. |
| `fetch_registry.py` | BODACC, BORME and The Gazette announcements whose spans come from their structured fields or their fixed grammar, not from a guess. 25 %. |
| `check_isolation.py` | Test-set isolation: ids, person and company names, and runs of 8 words. Two deliberate margins: `real_organisations.txt` (plus the bundled Wikidata list) lets both sides name a real bank or insurer, and shared runs are allowed up to 3 for written material and 25 for registry announcements, whose skeleton is fixed by law. Anything above that is dropped whole. |
| `train.py`, `export_onnx.py` | Full fine-tune (fp16, length 256, label order unchanged) and the ONNX int8 export the app loads. |

## Mix

Templates 60 %, registry 25 %, hand-written documents 15 %, in estimated chunks of 256 tokens; the three
languages are equal, and within a language GB 60 / US 20 / IE 20, FR 60 / BE 20 / CH 20, ES 100.
`build_report.md` prints what was actually produced, what was dropped and why.

## Who wrote what

The material was written by isolated subagents: one per country group for the inventories, one per language
and doctype group for the templates, three per language for the documents (formal, informal, hard formats),
and one reviewer per language over a sample. None of them saw the detection pipeline or any test set.
