// Copyright 2026 the Docudis contributors. Licensed under Apache-2.0.
//
// Pure tokenizer-alignment, windowing and BIO decoding only. This module has
// no model runtime dependency.

use docudis_core::{Detection, DetectionSource, EntityType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerKind {
    WordPiece,
    SentencePiece,
}

pub trait NerTokenizer {
    type Error;
    fn encode(&self, text: &str) -> Result<NerEncoding, Self::Error>;
    fn start_id(&self) -> i64;
    fn end_id(&self) -> i64;
}

pub struct HuggingFaceNerTokenizer {
    tokenizer: Tokenizer,
    kind: TokenizerKind,
    start_id: i64,
    end_id: i64,
}
impl HuggingFaceNerTokenizer {
    pub fn from_json(source: &str, kind: TokenizerKind) -> Result<Self, tokenizers::Error> {
        let tokenizer = Tokenizer::from_bytes(source.as_bytes())?;
        let special = tokenizer.encode("a", true)?;
        let ids = special.get_ids();
        let start_id = i64::from(*ids.first().ok_or("tokenizer emitted no start token")?);
        let end_id = i64::from(*ids.last().ok_or("tokenizer emitted no end token")?);
        Ok(Self {
            tokenizer,
            kind,
            start_id,
            end_id,
        })
    }
}
impl NerTokenizer for HuggingFaceNerTokenizer {
    type Error = tokenizers::Error;
    fn encode(&self, text: &str) -> Result<NerEncoding, Self::Error> {
        let encoding = self.tokenizer.encode(text, false)?;
        let (starts, ends) = if self.kind == TokenizerKind::SentencePiece {
            let tokens = encoding.get_tokens().to_vec();
            realign_sentencepiece(text, &tokens)
        } else {
            encoding.get_offsets().iter().copied().unzip()
        };
        let word_ids = if self.kind == TokenizerKind::SentencePiece {
            sentencepiece_word_ids(text, &starts, &ends)
        } else {
            encoding
                .get_word_ids()
                .iter()
                .map(|id| id.map(|value| value as usize))
                .collect()
        };
        Ok(NerEncoding {
            ids: encoding.get_ids().iter().map(|id| i64::from(*id)).collect(),
            starts,
            ends,
            word_ids,
        })
    }
    fn start_id(&self) -> i64 {
        self.start_id
    }
    fn end_id(&self) -> i64 {
        self.end_id
    }
}

pub fn sentencepiece_word_ids(text: &str, starts: &[usize], ends: &[usize]) -> Vec<Option<usize>> {
    let mut out = Vec::with_capacity(starts.len());
    let mut word = 0_usize;
    let mut first = true;
    let mut previous_standalone = false;
    let mut previous_end = None;
    for (&start, &end) in starts.iter().zip(ends) {
        let piece = &text[start..end];
        let standalone = piece.chars().any(|c| {
            matches!(c, '\u{3040}'..='\u{30ff}' | '\u{3400}'..='\u{9fff}' | '\u{ac00}'..='\u{d7af}')
        }) || !piece.chars().any(char::is_alphanumeric);
        let starts_space = piece.chars().next().is_some_and(char::is_whitespace);
        let gap = previous_end.is_some_and(|value| start > value);
        if !first && (starts_space || standalone || previous_standalone || gap) {
            word += 1;
        }
        out.push(Some(word));
        first = false;
        previous_standalone = standalone;
        previous_end = Some(end);
    }
    out
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NerEncoding {
    pub ids: Vec<i64>,
    pub starts: Vec<usize>,
    pub ends: Vec<usize>,
    pub word_ids: Vec<Option<usize>>,
}
impl NerEncoding {
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
    pub fn validate(&self, text: &str) -> bool {
        let n = self.len();
        self.starts.len() == n
            && self.ends.len() == n
            && self.word_ids.len() == n
            && self.starts.iter().zip(&self.ends).all(|(s, e)| {
                s < e && *e <= text.len() && text.is_char_boundary(*s) && text.is_char_boundary(*e)
            })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NerWindow {
    pub token_start: usize,
    pub token_end: usize,
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
}
pub fn build_windows(
    enc: &NerEncoding,
    max_tokens: usize,
    stride: usize,
    start_id: i64,
    end_id: i64,
) -> Vec<NerWindow> {
    assert!(max_tokens >= 3);
    let width = max_tokens - 2;
    let step = (width.saturating_sub(stride)).max(1);
    let mut out = Vec::new();
    let mut start = 0;
    while start < enc.len() {
        let end = (start + width).min(enc.len());
        let mut ids = Vec::with_capacity(end - start + 2);
        ids.push(start_id);
        ids.extend_from_slice(&enc.ids[start..end]);
        ids.push(end_id);
        out.push(NerWindow {
            token_start: start,
            token_end: end,
            attention_mask: vec![1; ids.len()],
            input_ids: ids,
        });
        if end == enc.len() {
            break;
        }
        start += step
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenPrediction {
    pub label: usize,
    pub probability: f64,
}
pub fn prediction_from_logits(logits: &[f64]) -> Option<TokenPrediction> {
    let (&max, rest) = logits.split_first()?;
    let (mut idx, mut value) = (0, max);
    for (i, v) in rest.iter().enumerate() {
        if *v > value {
            value = *v;
            idx = i + 1
        }
    }
    let sum = logits.iter().map(|v| (*v - value).exp()).sum::<f64>();
    Some(TokenPrediction {
        label: idx,
        probability: 1. / sum,
    })
}
pub fn merge_window_predictions(
    token_count: usize,
    windows: &[NerWindow],
    logits: &[Vec<Vec<f64>>],
) -> Result<Vec<Option<TokenPrediction>>, String> {
    if windows.len() != logits.len() {
        return Err("window/logit count mismatch".into());
    }
    let mut best = vec![None; token_count];
    let mut distance = vec![None; token_count];
    for (window, rows) in windows.iter().zip(logits) {
        if rows.len() != window.input_ids.len() {
            return Err("logit rows do not match window length".into());
        }
        for token in window.token_start..window.token_end {
            let d = (token - window.token_start).min(window.token_end - 1 - token);
            if distance[token].is_some_and(|old| d <= old) {
                continue;
            }
            distance[token] = Some(d);
            best[token] = prediction_from_logits(&rows[token - window.token_start + 1]);
        }
    }
    Ok(best)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NerDecodeConfig {
    pub model_name: String,
    pub labels: Vec<String>,
    pub label_map: HashMap<String, EntityType>,
    pub threshold: f64,
}
pub fn decode_bio(
    text: &str,
    enc: &NerEncoding,
    predictions: &[Option<TokenPrediction>],
    config: &NerDecodeConfig,
) -> Result<Vec<Detection>, String> {
    if !enc.validate(text) || predictions.len() != enc.len() {
        return Err("invalid NER encoding or prediction length".into());
    }
    let mut out = Vec::new();
    let (mut entity, mut start, mut end, mut sum, mut count): (
        Option<String>,
        usize,
        usize,
        f64,
        usize,
    ) = (None, 0, 0, 0., 0);
    let close = |out: &mut Vec<Detection>,
                 entity: &mut Option<String>,
                 start: usize,
                 end: usize,
                 sum: f64,
                 count: usize| {
        let Some(name) = entity.take() else { return };
        let conf = if count == 0 { 0. } else { sum / count as f64 };
        let Some(kind) = config.label_map.get(&name).copied() else {
            return;
        };
        if conf < config.threshold {
            return;
        }
        let mut s = start;
        let mut e = end;
        while s < e && text[s..].chars().next().is_some_and(char::is_whitespace) {
            s += text[s..].chars().next().unwrap().len_utf8()
        }
        while e > s
            && text[..e]
                .chars()
                .next_back()
                .is_some_and(char::is_whitespace)
        {
            e -= text[..e].chars().next_back().unwrap().len_utf8()
        }
        if e > s {
            out.push(Detection {
                entity_type: kind,
                value: text[s..e].to_owned(),
                start: s,
                end: e,
                confidence: conf,
                detector: format!("ner:{}", config.model_name),
                source: DetectionSource::Model,
                enabled: true,
            })
        }
    };
    let (mut last_word, mut word_label): (Option<usize>, Option<String>) = (None, None);
    for (i, prediction) in predictions.iter().copied().enumerate() {
        let continuation = enc.word_ids[i].is_some() && enc.word_ids[i] == last_word;
        last_word = enc.word_ids[i];
        let label = if continuation {
            word_label.clone().unwrap_or_else(|| "O".into())
        } else {
            let label = prediction
                .and_then(|p| config.labels.get(p.label))
                .cloned()
                .unwrap_or_else(|| "O".into());
            word_label = Some(label.clone());
            label
        };
        if label == "O" {
            close(&mut out, &mut entity, start, end, sum, count);
            sum = 0.;
            count = 0;
            continue;
        }
        let (prefix, name) = label.split_once('-').unwrap_or(("B", label.as_str()));
        let continues =
            entity.as_deref() == Some(name) && (matches!(prefix, "I" | "E") || continuation);
        if !continues {
            close(&mut out, &mut entity, start, end, sum, count);
            entity = Some(name.to_owned());
            start = enc.starts[i];
            end = enc.ends[i];
            sum = 0.;
            count = 0
        } else {
            end = enc.ends[i]
        }
        if !continuation {
            if let Some(p) = prediction {
                sum += p.probability;
                count += 1
            }
        }
    }
    close(&mut out, &mut entity, start, end, sum, count);
    Ok(out)
}

pub fn title_cased(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut cursor = 0;
    for (start, ch) in text.char_indices() {
        if !ch.is_alphabetic() {
            out.push_str(&text[cursor..start + ch.len_utf8()]);
            cursor = start + ch.len_utf8();
            continue;
        }
        if start > 0
            && text[..start]
                .chars()
                .next_back()
                .is_some_and(char::is_alphabetic)
        {
            continue;
        }
        let end = text[start..]
            .char_indices()
            .find(|(_, c)| !c.is_alphabetic())
            .map_or(text.len(), |(i, _)| start + i);
        let word = &text[start..end];
        if word.chars().count() >= 4 && word.chars().all(|c| !c.is_lowercase()) {
            let first = word.chars().next().unwrap();
            let rest = &word[first.len_utf8()..];
            let lowered = rest.to_lowercase();
            if lowered.len() == rest.len() {
                out.push_str(&text[cursor..start]);
                out.push(first);
                out.push_str(&lowered);
                cursor = end
            }
        }
    }
    out.push_str(&text[cursor..]);
    out
}

pub fn realign_sentencepiece(text: &str, tokens: &[String]) -> (Vec<usize>, Vec<usize>) {
    let mut starts = vec![0; tokens.len()];
    let mut ends = vec![0; tokens.len()];
    let mut pending = Vec::new();
    let mut pending_chars = 0;
    let mut cursor = 0;
    fn place(
        text: &str,
        tokens: &[String],
        pending: &mut Vec<usize>,
        starts: &mut [usize],
        ends: &mut [usize],
        from: usize,
        to: usize,
    ) {
        let mut at = from;
        let total = pending.len();
        for (k, i) in pending.drain(..).enumerate() {
            let mut content = at;
            while content < to
                && text[content..]
                    .chars()
                    .next()
                    .is_some_and(char::is_whitespace)
            {
                content += text[content..].chars().next().unwrap().len_utf8()
            }
            let start = if tokens[i].starts_with('▁') {
                at
            } else {
                content
            };
            let wanted = tokens[i].trim_start_matches('▁').chars().count();
            let mut end = content;
            for _ in 0..wanted {
                if end >= to {
                    break;
                }
                end += text[end..].chars().next().unwrap().len_utf8()
            }
            if k + 1 == total || end > to {
                end = to
            }
            while end > content
                && text[..end]
                    .chars()
                    .next_back()
                    .is_some_and(char::is_whitespace)
            {
                end -= text[..end].chars().next_back().unwrap().len_utf8()
            }
            starts[i] = start;
            ends[i] = end.max(start);
            at = ends[i]
        }
    }
    for (i, token) in tokens.iter().enumerate() {
        let piece = token.trim_start_matches('▁');
        let found = find_piece(text, piece, cursor, pending_chars);
        let Some(found) = found else {
            pending.push(i);
            pending_chars += piece.chars().count();
            continue;
        };
        let mut start = found;
        if token.starts_with('▁') {
            while start > cursor
                && text[..start]
                    .chars()
                    .next_back()
                    .is_some_and(char::is_whitespace)
            {
                start -= text[..start].chars().next_back().unwrap().len_utf8()
            }
        }
        place(
            text,
            tokens,
            &mut pending,
            &mut starts,
            &mut ends,
            cursor,
            start,
        );
        pending_chars = 0;
        starts[i] = start;
        ends[i] = found + piece.len();
        cursor = ends[i]
    }
    place(
        text,
        tokens,
        &mut pending,
        &mut starts,
        &mut ends,
        cursor,
        text.len(),
    );
    (starts, ends)
}
fn find_piece(text: &str, piece: &str, cursor: usize, pending: usize) -> Option<usize> {
    if piece.is_empty() {
        return None;
    }
    let mut skipped = 0;
    for (at, _) in text[cursor..].char_indices() {
        let at = cursor + at;
        if text[at..].starts_with(piece) {
            return Some(at);
        }
        if !text[at..].chars().next().unwrap().is_whitespace() {
            skipped += 1;
            if skipped > pending + 2 {
                return None;
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn windows_overlap_and_keep_edge_distance() {
        let e = NerEncoding {
            ids: (0..10).collect(),
            starts: (0..10).collect(),
            ends: (1..11).collect(),
            word_ids: (0..10).map(Some).collect(),
        };
        let w = build_windows(&e, 6, 2, 100, 101);
        assert_eq!(
            w.iter()
                .map(|w| (w.token_start, w.token_end))
                .collect::<Vec<_>>(),
            [(0, 4), (2, 6), (4, 8), (6, 10)]
        )
    }
    #[test]
    fn sentencepiece_realigns_changed_character() {
        let text = "37,8\r\nºC y Dr. Tomás";
        let tokens = ["▁3", "7,8", "▁o", "C", "▁y", "▁Dr", ".", "▁Tomás"].map(str::to_owned);
        let (s, e) = realign_sentencepiece(text, &tokens);
        let got: Vec<_> = (0..tokens.len()).map(|i| &text[s[i]..e[i]]).collect();
        assert_eq!(got, ["3", "7,8", "\r\nº", "C", " y", " Dr", ".", " Tomás"])
    }
    #[test]
    fn uppercase_titlecase_preserves_offsets() {
        assert_eq!(title_cased("LOPEZ BAY ÉLODIE"), "Lopez BAY Élodie")
    }
    #[test]
    fn bio_decode_uses_utf8_offsets_without_a_runtime() {
        let text = "张三 Paris";
        let enc = NerEncoding {
            ids: vec![1, 2, 3],
            starts: vec![0, 3, 7],
            ends: vec![3, 6, 12],
            word_ids: vec![Some(0), Some(1), Some(2)],
        };
        let predictions = vec![
            Some(TokenPrediction {
                label: 1,
                probability: 0.9,
            }),
            Some(TokenPrediction {
                label: 2,
                probability: 0.8,
            }),
            Some(TokenPrediction {
                label: 3,
                probability: 0.95,
            }),
        ];
        let config = NerDecodeConfig {
            model_name: "fixture".into(),
            labels: vec!["O".into(), "B-PER".into(), "I-PER".into(), "B-LOC".into()],
            label_map: HashMap::from([
                ("PER".into(), EntityType::Person),
                ("LOC".into(), EntityType::Address),
            ]),
            threshold: 0.5,
        };
        let found = decode_bio(text, &enc, &predictions, &config).unwrap();
        assert_eq!(
            found.iter().map(|d| d.value.as_str()).collect::<Vec<_>>(),
            ["张三", "Paris"]
        );
    }
}
