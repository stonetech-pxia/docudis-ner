# Docudis NER

Model-side code and material for Docudis: NER tokenizer alignment, windowing,
BIO decoding, the model manifest, and the training pipeline. Rules,
anonymization, restoration, and the C ABI live in
[docudis-core](https://github.com/stonetech-pxia/docudis-core).

## Dependency direction

```text
docudis-ner  ->  docudis-core (Detection, DetectionSource, EntityType)
docudis-core ->  nothing model-related
```

This crate produces `Detection`s with `source: Model`; hosts pass them to Core
in the `detections` field of the v1 JSON requests, together with detections
from any other model. Core is pinned by commit in the root `Cargo.toml`.

## Repository layout

- `crates/docudis-ner`: Hugging Face tokenizer wrapper (WordPiece and
  SentencePiece realignment), window construction, softmax selection, window
  merge, and BIO decoding. No inference runtime; no C ABI yet.
- `testdata/tokenizers`: tokenizer-only fixtures, no model weights.
- `models`: `manifest.json` (Hugging Face repo, revision, SHA-256 per binary)
  and each model's `model.json`. Binaries are fetched, not committed.
- `tool/fetch_models.py`: downloads and verifies the pinned binaries.
- `training`: fine-tuning material and pipeline for `xlmr_ner_docudis`.
- `docs/ner-finetune-scope.md`: what the fine-tune targets.

The training isolation check reads the test sets of a docudis-android checkout
named by `DOCUDIS_APP_ROOT` and refuses to run without it; see
[training/README.md](training/README.md).

## Local verification

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo build --release --workspace
```

## Provenance

`crates/docudis-ner` and `testdata/tokenizers` were moved from docudis-core
commit `8743fd84bc16ba90e5dae70ec97b580ab5171a89` (`crates/docudis-core/src/ner.rs`,
`crates/docudis-core/tests/ner_tokenizers.rs`). `models`, `training`,
`tool/fetch_models.py`, `tool/valid_ids.py`, and `docs/ner-finetune-scope.md`
were moved from docudis-android commit
`26984e5e3e621d762edb5bc0bfdc0e695295bda6`; `tool/html_text.py` copies `strip_tags` from its
`tool/fetch_public_samples.py`.

## License

Apache-2.0. The DocCloak.Core attribution is retained in
`LICENSE-DocCloak.Core` and `NOTICE-DocCloak.Core`.
