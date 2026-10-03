# Tokenizer test fixtures

These Hugging Face tokenizer JSON files are test-only inputs for WordPiece,
SentencePiece and byte-level BPE alignment. They do not contain model weights
and do not execute inference. The production NER model and ONNX Runtime remain
outside Core.

| File | Source | License |
|---|---|---|
| `sentencepiece.json` | The tokenizer of `xlmr_ner_docudis`, which is XLM-RoBERTa's SentencePiece vocabulary (FacebookAI/xlm-roberta-base) | MIT, Copyright (c) Facebook, Inc. and its affiliates |
| `wordpiece.json` | The tokenizer of Xenova/distilbert-base-multilingual-cased-ner-hrl, whose WordPiece vocabulary is multilingual BERT's (google-research/bert) | Apache-2.0, Copyright 2018 The Google AI Language Team Authors |
| `bytelevel.json` | The normalizer, pre-tokenizer, post-processor and decoder of openai/privacy-filter's o200k tokenizer around an 800-token BPE trained here on a sample of `training/documents`, so CJK and emoji split into single bytes | Apache-2.0 (openai/privacy-filter); the trained BPE is this repository's |

The MIT license of the XLM-RoBERTa vocabulary:

```text
MIT License

Copyright (c) Facebook, Inc. and its affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The Apache-2.0 license text is in the repository's `LICENSE`.
