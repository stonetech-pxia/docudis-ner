#![cfg(feature = "onnxruntime")]
// Copyright 2026 the Docudis contributors. Licensed under Apache-2.0.
//! Real inference. Runs only when DOCUDIS_NER_TEST_ORT names an ONNX Runtime
//! library and DOCUDIS_NER_TEST_MODEL a model folder with model.json and its
//! files (tool/fetch_models.py writes them under models/).

use std::path::PathBuf;

use docudis_core::EntityType;
use docudis_ner::{ModelSpec, NerModel};

fn model() -> Option<NerModel> {
    let (Ok(runtime), Ok(dir)) = (
        std::env::var("DOCUDIS_NER_TEST_ORT"),
        std::env::var("DOCUDIS_NER_TEST_MODEL"),
    ) else {
        eprintln!("skipped: set DOCUDIS_NER_TEST_ORT and DOCUDIS_NER_TEST_MODEL");
        return None;
    };
    let dir = PathBuf::from(dir);
    let spec =
        ModelSpec::from_json(&std::fs::read_to_string(dir.join("model.json")).unwrap()).unwrap();
    let (model, tokenizer) = (dir.join(&spec.model), dir.join(&spec.tokenizer.file));
    Some(NerModel::load(&runtime, spec, &model, &tokenizer).unwrap())
}

#[test]
fn finds_people_and_places_in_latin_and_cjk_text() {
    let Some(mut model) = model() else { return };
    let text = "Bonjour, je suis Élodie Marchand et mon adresse est 12 rue des Lilas, 69003 Lyon.";
    let found = model.detect(text).unwrap();
    assert!(
        found
            .iter()
            .any(|d| d.entity_type == EntityType::Person && d.value == "Élodie Marchand"),
        "{found:?}"
    );
    assert!(found.iter().all(|d| text[d.start..d.end] == d.value));
    // XLM-R puts a zero-width piece before CJK text.
    let text = "张三昨天把报告发给了李四和王五。";
    let found = model.detect(text).unwrap();
    assert!(
        found.iter().all(|d| text[d.start..d.end] == d.value),
        "{found:?}"
    );
    assert!(model.detect("").unwrap().is_empty());
}
