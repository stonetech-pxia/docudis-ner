#![cfg(feature = "onnxruntime")]
// Its own test binary: ONNX Runtime initialisation is once per process.

use std::path::Path;

use docudis_ner::{ModelSpec, NerModel};

#[test]
fn a_missing_runtime_library_is_an_error_not_a_crash() {
    let spec =
        ModelSpec::from_json(include_str!("../../../models/xlmr_ner_docudis/model.json")).unwrap();
    let error = NerModel::load(
        "/definitely/missing/libonnxruntime.so",
        spec,
        Path::new("/missing/model.onnx"),
        Path::new("/missing/tokenizer.json"),
    )
    .err()
    .unwrap();
    assert!(
        error.to_string().contains("cannot load ONNX Runtime"),
        "{error}"
    );
}
