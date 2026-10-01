// Copyright 2026 the Docudis contributors. Licensed under Apache-2.0.

use std::collections::HashMap;

use docudis_core::EntityType;
use serde::Deserialize;

use crate::{NerDecodeConfig, TokenizerKind};

/// A model's `model.json`, as published in `models/<name>/model.json`.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelSpec {
    pub name: String,
    /// ONNX file name next to `model.json`.
    pub model: String,
    pub tokenizer: TokenizerSpec,
    /// BIO labels in model output order.
    pub labels: Vec<String>,
    /// Model entity names (`PER`) to Docudis entity types.
    pub label_map: HashMap<String, EntityType>,
    pub max_tokens: usize,
    pub stride: usize,
    pub threshold: f64,
    pub inputs: ModelInputs,
    pub output: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TokenizerSpec {
    /// `wordpiece` or `sentencepiece`.
    pub kind: String,
    pub file: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelInputs {
    pub ids: String,
    pub mask: String,
}

impl ModelSpec {
    pub fn from_json(source: &str) -> Result<Self, String> {
        let spec: Self =
            serde_json::from_str(source).map_err(|error| format!("invalid model.json: {error}"))?;
        spec.tokenizer_kind()?;
        if spec.max_tokens < 3 {
            return Err(format!(
                "maxTokens must be at least 3, got {}",
                spec.max_tokens
            ));
        }
        if spec.labels.is_empty() {
            return Err("model.json has no labels".into());
        }
        Ok(spec)
    }

    pub fn tokenizer_kind(&self) -> Result<TokenizerKind, String> {
        match self.tokenizer.kind.as_str() {
            "wordpiece" => Ok(TokenizerKind::WordPiece),
            "sentencepiece" => Ok(TokenizerKind::SentencePiece),
            other => Err(format!("unknown tokenizer kind {other:?}")),
        }
    }

    pub fn decode_config(&self) -> NerDecodeConfig {
        NerDecodeConfig {
            model_name: self.name.clone(),
            labels: self.labels.clone(),
            label_map: self.label_map.clone(),
            threshold: self.threshold,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const XLMR: &str = include_str!("../../../models/xlmr_ner_docudis/model.json");

    #[test]
    fn parses_every_published_model_json() {
        for source in [
            XLMR,
            include_str!("../../../models/xlmr_ner_hrl/model.json"),
            include_str!("../../../models/distilbert_ner_hrl/model.json"),
        ] {
            let spec = ModelSpec::from_json(source).unwrap();
            assert!(spec.label_map.values().all(|t| *t != EntityType::Other));
            assert_eq!(spec.decode_config().labels, spec.labels);
        }
        let spec = ModelSpec::from_json(XLMR).unwrap();
        assert_eq!(spec.tokenizer_kind().unwrap(), TokenizerKind::SentencePiece);
        assert_eq!(spec.label_map["PER"], EntityType::Person);
        assert_eq!((spec.max_tokens, spec.stride), (256, 32));
    }

    #[test]
    fn rejects_an_unknown_tokenizer_kind() {
        let source = XLMR.replace("\"sentencepiece\"", "\"bpe\"");
        assert!(ModelSpec::from_json(&source).unwrap_err().contains("bpe"));
    }
}
