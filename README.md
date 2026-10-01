# Docudis NER

Model-side code and material for Docudis: NER inference through ONNX
Runtime, tokenizer alignment, windowing, BIO decoding, a versioned C ABI with
Dart bindings, the model manifest, and the training pipeline. Rules,
anonymization, and restoration live in
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

- `crates/docudis-ner`: `model.json` parsing (`ModelSpec`), Hugging Face
  tokenizer wrapper (WordPiece and SentencePiece realignment), window
  construction, softmax selection, window merge, and BIO decoding. With the
  `onnxruntime` feature, `NerModel` runs the whole pipeline through an ONNX
  Runtime library loaded at run time; its output matches the Dart
  `NerDetector` the app used before, span for span and bit for bit.
- `crates/docudis-ner-capi`: versioned `docudis_ner_v1_*` C ABI
  (`include/docudis_ner.h`), built as `libdocudis_ner_capi`.
- `bindings/dart`: `docudis_ner_ffi`, Dart FFI with ABI validation, UTF-16
  offsets, and model handles that can cross isolates.
- `scripts/build-android.sh`: Android libraries for `arm64-v8a`,
  `armeabi-v7a`, and `x86_64` (API 26), with exported-symbol checks.
- `testdata/tokenizers`: tokenizer-only fixtures, no model weights.
- `models`: `manifest.json` (Hugging Face repo, revision, SHA-256 per binary)
  and each model's `model.json`. Binaries are fetched, not committed.
- `tool/fetch_models.py`: downloads and verifies the pinned binaries.
- `training`: fine-tuning material and pipeline for `xlmr_ner_docudis`.
- `docs/ner-finetune-scope.md`: what the fine-tune targets.

The training isolation check reads the test sets of a docudis-android checkout
named by `DOCUDIS_APP_ROOT` and refuses to run without it; see
[training/README.md](training/README.md).

## ONNX Runtime

The library is not linked: hosts name it when they load a model. Android apps
ship `libonnxruntime.so` from `com.microsoft.onnxruntime:onnxruntime-android`
(1.28 or later; 1.23 crashes with SIGILL on some SoCs) and pass the bare file
name. `ort` is pinned to `2.0.0-rc.13`, which has two consequences:

- The first load decides for the whole process. A failed load cannot be
  retried (ort marks its library slot initialised although it is empty), so
  `NerModel::load` returns the first error again instead of calling ort.
- A command-line process that loads the library itself aborts at exit on
  Android ("destroyed mutex"): the library's C++ destructors run before ort
  releases its environment. App processes are killed rather than exited.

## Local verification

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo build --release --workspace
./scripts/test-c-header.sh

cd bindings/dart
dart pub get
dart format --output=none --set-exit-if-changed lib test
dart analyze
dart test
```

Real inference tests run when `DOCUDIS_NER_TEST_ORT` names an ONNX Runtime
library (for instance from the `onnxruntime-osx-arm64` release archive) and
`DOCUDIS_NER_TEST_MODEL` a folder with `model.json` and its files.

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
