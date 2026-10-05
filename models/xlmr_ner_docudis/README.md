---
license: afl-3.0
language:
- en
- fr
- es
base_model: Davlan/xlm-roberta-base-ner-hrl
pipeline_tag: token-classification
library_name: onnx
tags:
- ner
- pii
- anonymization
---

# Docudis NER (xlmr_ner_docudis)

The name-recognition model of [Docudis](https://docudis.com), which replaces personal details in documents
with labels on the user's device. It finds people (`PER`), organisations (`ORG`), places and addresses
(`LOC`) and dates (`DATE`) in English, French and Spanish text.

- **Format:** ONNX, int8-quantized (`model_quantized.onnx`), with its SentencePiece tokenizer (`tokenizer.json`).
- **Inputs:** windows of up to 256 tokens; Docudis runs overlapping windows with a stride of 32 and keeps
  spans whose mean probability is at least 0.5. `model.json` describes the labels and settings.
- **Code:** inference runs through [docudis-ner](https://github.com/stonetech-pxia/docudis-ner) (Apache-2.0).

## How it was made

This model is a full fine-tune of [Davlan/xlm-roberta-base-ner-hrl](https://huggingface.co/Davlan/xlm-roberta-base-ner-hrl),
which is itself fine-tuned from [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base).
The label set and order are unchanged.

The fine-tuning data, in English, French and Spanish in equal shares:

| Share | Material |
|---|---|
| 60 % | Document templates written for this purpose, filled with generated names, addresses and identifiers |
| 25 % | Official company announcements: BODACC (France), BORME (Spain), The Gazette (United Kingdom) |
| 15 % | Complete documents written for this purpose, with hand-marked entities |

The pipeline is in [docudis-ner/training](https://github.com/stonetech-pxia/docudis-ner/tree/main/training).
The training text is not distributed with the model.

### Data from official gazettes

- BODACC: Direction de l'information légale et administrative (DILA), under the Licence Ouverte 2.0.
- BORME: Agencia Estatal Boletín Oficial del Estado (www.boe.es).
- The Gazette (www.thegazette.co.uk): contains public sector information licensed under the Open Government
  Licence v3.0, Crown copyright.

These announcements name company officers and insolvency practitioners, as published by law in those
gazettes.

### The base model's training data

The model card of Davlan/xlm-roberta-base-ner-hrl lists its training corpora: CoNLL-2003 (English, German),
CoNLL-2002 (Spanish, Dutch), Europeana Newspapers (French), ANERcorp (Arabic), MSRA (Chinese), I-CAB (Italian),
the Latvian NER dataset and Paramopama + Second HAREM (Portuguese). Some of these corpora are distributed under research-only terms. This
model inherits the base model's weights, so evaluate those terms for your use.

## Limitations

No automatic detection finds every personal detail, and this model is no exception. It is tuned for
business and personal documents in English, French and Spanish; other languages fall back to what the base
model learned. Always review its output before sharing a document.

## License

Copyright 2026 Pengda Xia (stonetech). Licensed under the [Academic Free License 3.0](LICENSE).

This model is a Derivative Work of Davlan/xlm-roberta-base-ner-hrl (Academic Free License 3.0), modified by
stonetech by fine-tuning it on the data above and quantizing it. XLM-RoBERTa is licensed under the MIT
License, Copyright (c) Facebook, Inc. and its affiliates.
