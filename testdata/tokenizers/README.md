# Tokenizer test fixtures

These Hugging Face tokenizer JSON files are test-only inputs for WordPiece,
SentencePiece and byte-level BPE alignment. `bytelevel.json` has the
normalizer, pre-tokenizer, post-processor and decoder of openai/privacy-filter's
o200k tokenizer around an 800-token BPE trained on a sample of
`training/documents`, so CJK and emoji split into single bytes. They do not
contain model weights and do not execute inference. The production NER model and ONNX Runtime remain outside Core.

