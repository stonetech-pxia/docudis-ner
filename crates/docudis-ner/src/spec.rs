// Copyright 2026 the Docudis contributors. Licensed under Apache-2.0.

use std::collections::HashMap;

use docudis_core::EntityType;
use serde::Deserialize;

use crate::{bioes_tags, NerDecodeConfig, TokenizerKind};

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
    /// Title-case ALL-CAPS words before tokenizing (see `title_cased`).
    #[serde(default = "default_title_case")]
    pub title_case: bool,
    #[serde(default)]
    pub decoder: DecoderSpec,
    pub inputs: ModelInputs,
    pub output: String,
}

fn default_title_case() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
pub struct TokenizerSpec {
    /// `wordpiece`, `sentencepiece` or `bytelevel`.
    pub kind: String,
    pub file: String,
}

/// How token logits become spans.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum DecoderSpec {
    /// Argmax per token, BIO spans, first sub-token of a word decides.
    #[default]
    Bio,
    /// Constrained Viterbi over BIOES labels (`decode_bioes_viterbi`).
    BioesViterbi {
        #[serde(default)]
        biases: TransitionBiases,
    },
}

/// Additive transition scores for the BIOES Viterbi: the six
/// `transition_bias_*` values of an `operating_points` entry in
/// openai/privacy-filter's `viterbi_calibration.json`. Positive values make
/// a transition more likely.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub struct TransitionBiases {
    pub background_stay: f64,
    pub background_to_start: f64,
    pub end_to_background: f64,
    pub end_to_start: f64,
    pub inside_to_continue: f64,
    pub inside_to_end: f64,
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
        if spec.decoder != DecoderSpec::Bio {
            bioes_tags(&spec.labels)?;
        }
        Ok(spec)
    }

    pub fn tokenizer_kind(&self) -> Result<TokenizerKind, String> {
        match self.tokenizer.kind.as_str() {
            "wordpiece" => Ok(TokenizerKind::WordPiece),
            "sentencepiece" => Ok(TokenizerKind::SentencePiece),
            "bytelevel" => Ok(TokenizerKind::ByteLevel),
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
    const OPF: &str = include_str!("../../../models/openai_privacy_filter/model.json");

    #[test]
    fn parses_every_published_model_json() {
        for source in [
            XLMR,
            include_str!("../../../models/xlmr_ner_hrl/model.json"),
            include_str!("../../../models/distilbert_ner_hrl/model.json"),
            OPF,
        ] {
            let spec = ModelSpec::from_json(source).unwrap();
            assert!(spec.label_map.values().all(|t| *t != EntityType::Other));
            assert_eq!(spec.decode_config().labels, spec.labels);
        }
        let spec = ModelSpec::from_json(XLMR).unwrap();
        assert_eq!(spec.tokenizer_kind().unwrap(), TokenizerKind::SentencePiece);
        assert_eq!(spec.label_map["PER"], EntityType::Person);
        assert_eq!((spec.max_tokens, spec.stride), (256, 32));
        assert!(spec.title_case);
        assert_eq!(spec.decoder, DecoderSpec::Bio);
    }

    #[test]
    fn parses_the_privacy_filter_spec() {
        let spec = ModelSpec::from_json(OPF).unwrap();
        assert_eq!(spec.tokenizer_kind().unwrap(), TokenizerKind::ByteLevel);
        assert_eq!(spec.labels.len(), 33);
        assert!(!spec.title_case);
        assert_eq!(
            spec.decoder,
            DecoderSpec::BioesViterbi {
                biases: TransitionBiases::default()
            }
        );
        // Account numbers stay unmapped until their Docudis type is decided.
        assert!(!spec.label_map.contains_key("account_number"));
    }

    #[test]
    fn rejects_a_viterbi_decoder_over_non_bioes_labels_and_unknown_biases() {
        let source = XLMR.replace("\"B-DATE\"", "\"DATE\"").replace(
            "\"output\": \"logits\"",
            "\"output\": \"logits\", \"decoder\": {\"kind\": \"bioes-viterbi\"}",
        );
        assert!(ModelSpec::from_json(&source)
            .unwrap_err()
            .contains("not a BIOES label"));
        let source = OPF.replace("backgroundStay", "backgroundStays");
        assert!(ModelSpec::from_json(&source)
            .unwrap_err()
            .contains("backgroundStays"));
    }

    #[test]
    fn rejects_an_unknown_tokenizer_kind() {
        let source = XLMR.replace("\"sentencepiece\"", "\"bpe\"");
        assert!(ModelSpec::from_json(&source).unwrap_err().contains("bpe"));
    }
}
