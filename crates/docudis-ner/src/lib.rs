// Copyright 2026 the Docudis contributors. Licensed under Apache-2.0.
//
// Tokenizer alignment, windowing, BIO decoding and BIOES Viterbi decoding are
// pure and need no model runtime. Inference through ONNX Runtime lives behind the `onnxruntime`
// feature.

#[cfg(feature = "onnxruntime")]
mod inference;
mod spec;

#[cfg(feature = "onnxruntime")]
pub use inference::{NerError, NerModel};
pub use spec::{DecoderSpec, ModelInputs, ModelSpec, TokenizerSpec, TransitionBiases};

use docudis_core::{Detection, DetectionSource, EntityType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerKind {
    WordPiece,
    SentencePiece,
    /// Byte-level BPE (GPT-style, such as o200k): offsets come straight from
    /// the tokenizer.
    ByteLevel,
}

pub trait NerTokenizer {
    type Error;
    fn encode(&self, text: &str) -> Result<NerEncoding, Self::Error>;
    /// The start and end tokens that wrap every window, or `None` for a
    /// tokenizer that adds none (byte-level BPE).
    fn special_ids(&self) -> Option<(i64, i64)>;
}

pub struct HuggingFaceNerTokenizer {
    tokenizer: Tokenizer,
    kind: TokenizerKind,
    special_ids: Option<(i64, i64)>,
}
impl HuggingFaceNerTokenizer {
    pub fn from_json(source: &str, kind: TokenizerKind) -> Result<Self, tokenizers::Error> {
        let tokenizer = Tokenizer::from_bytes(source.as_bytes())?;
        let special = tokenizer.encode("a", true)?;
        let plain = tokenizer.encode("a", false)?;
        let special_ids = if special.get_ids() == plain.get_ids() {
            None
        } else {
            let ids = special.get_ids();
            let start_id = i64::from(*ids.first().ok_or("tokenizer emitted no start token")?);
            let end_id = i64::from(*ids.last().ok_or("tokenizer emitted no end token")?);
            Some((start_id, end_id))
        };
        Ok(Self {
            tokenizer,
            kind,
            special_ids,
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
    fn special_ids(&self) -> Option<(i64, i64)> {
        self.special_ids
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
                s <= e && *e <= text.len() && text.is_char_boundary(*s) && text.is_char_boundary(*e)
            })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NerWindow {
    pub token_start: usize,
    pub token_end: usize,
    /// Position of the window's first text token in `input_ids` (1 after a
    /// start token, 0 without special tokens).
    pub offset: usize,
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
}
/// Overlapping windows of at most `max_tokens` model inputs, each wrapped in
/// `special` start and end tokens when the tokenizer has them. `stride` is
/// the overlap between consecutive windows.
pub fn build_windows(
    enc: &NerEncoding,
    max_tokens: usize,
    stride: usize,
    special: Option<(i64, i64)>,
) -> Vec<NerWindow> {
    assert!(max_tokens >= 3);
    let width = if special.is_some() {
        max_tokens - 2
    } else {
        max_tokens
    };
    let step = (width.saturating_sub(stride)).max(1);
    let mut out = Vec::new();
    let mut start = 0;
    while start < enc.len() {
        let end = (start + width).min(enc.len());
        let mut ids = Vec::with_capacity(end - start + 2);
        if let Some((start_id, _)) = special {
            ids.push(start_id);
        }
        ids.extend_from_slice(&enc.ids[start..end]);
        if let Some((_, end_id)) = special {
            ids.push(end_id);
        }
        out.push(NerWindow {
            token_start: start,
            token_end: end,
            offset: usize::from(special.is_some()),
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
/// For each token, the logit row of the window where it sits farthest from
/// an edge (the first such window on ties).
fn best_rows<'a>(
    token_count: usize,
    windows: &[NerWindow],
    logits: &'a [Vec<Vec<f64>>],
) -> Result<Vec<Option<&'a [f64]>>, String> {
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
            best[token] = Some(rows[token - window.token_start + window.offset].as_slice());
        }
    }
    Ok(best)
}
pub fn merge_window_predictions(
    token_count: usize,
    windows: &[NerWindow],
    logits: &[Vec<Vec<f64>>],
) -> Result<Vec<Option<TokenPrediction>>, String> {
    Ok(best_rows(token_count, windows, logits)?
        .into_iter()
        .map(|row| row.and_then(prediction_from_logits))
        .collect())
}
/// One logit row per token, for decoders that need every label's score.
pub fn merge_window_logits(
    token_count: usize,
    windows: &[NerWindow],
    logits: &[Vec<Vec<f64>>],
) -> Result<Vec<Vec<f64>>, String> {
    best_rows(token_count, windows, logits)?
        .into_iter()
        .map(|row| {
            row.map(<[f64]>::to_vec)
                .ok_or_else(|| "a token is in no window".to_owned())
        })
        .collect()
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

/// Splits BIOES labels (`O`, `B-PER`, `I-PER`, `E-PER`, `S-PER`) into their
/// tag and entity name.
pub fn bioes_tags(labels: &[String]) -> Result<Vec<(char, Option<&str>)>, String> {
    labels
        .iter()
        .map(|label| match label.split_once('-') {
            _ if label == "O" => Ok(('O', None)),
            Some((tag @ ("B" | "I" | "E" | "S"), name)) if !name.is_empty() => {
                Ok((tag.chars().next().unwrap(), Some(name)))
            }
            _ => Err(format!("label {label:?} is not a BIOES label")),
        })
        .collect()
}

/// Decodes BIOES logits with a constrained Viterbi: paths start and end
/// outside an entity, an entity opens with `B` or `S`, `I` and `E` continue
/// only their own entity, and `biases` add to the allowed transitions. The
/// best path then decodes token by token through [`decode_bio`], each token
/// scored with its label's softmax probability.
pub fn decode_bioes_viterbi(
    text: &str,
    enc: &NerEncoding,
    logits: &[Vec<f64>],
    config: &NerDecodeConfig,
    biases: &TransitionBiases,
) -> Result<Vec<Detection>, String> {
    let tags = bioes_tags(&config.labels)?;
    let k = tags.len();
    if logits.len() != enc.len() || logits.iter().any(|row| row.len() != k) {
        return Err("logits do not match the encoding and labels".into());
    }
    if logits.is_empty() {
        return Ok(Vec::new());
    }
    let transition = |from: usize, to: usize| -> Option<f64> {
        let ((a, from_name), (b, to_name)) = (tags[from], tags[to]);
        match (a, b) {
            ('O', 'O') => Some(biases.background_stay),
            ('E' | 'S', 'O') => Some(biases.end_to_background),
            ('O', 'B' | 'S') => Some(biases.background_to_start),
            ('E' | 'S', 'B' | 'S') => Some(biases.end_to_start),
            ('B' | 'I', 'I') if from_name == to_name => Some(biases.inside_to_continue),
            ('B' | 'I', 'E') if from_name == to_name => Some(biases.inside_to_end),
            _ => None,
        }
    };
    let log_probs: Vec<Vec<f64>> = logits
        .iter()
        .map(|row| {
            let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let log_sum = row.iter().map(|v| (v - max).exp()).sum::<f64>().ln();
            row.iter().map(|v| v - max - log_sum).collect()
        })
        .collect();
    let mut score: Vec<f64> = (0..k)
        .map(|j| match tags[j].0 {
            'O' | 'B' | 'S' => log_probs[0][j],
            _ => f64::NEG_INFINITY,
        })
        .collect();
    let mut back = vec![vec![0_usize; k]; logits.len()];
    for t in 1..logits.len() {
        let mut next = vec![f64::NEG_INFINITY; k];
        for j in 0..k {
            let mut best = (f64::NEG_INFINITY, 0);
            for (i, previous) in score.iter().enumerate() {
                if let Some(bias) = transition(i, j) {
                    if previous + bias > best.0 {
                        best = (previous + bias, i)
                    }
                }
            }
            next[j] = best.0 + log_probs[t][j];
            back[t][j] = best.1
        }
        score = next
    }
    let mut last = (f64::NEG_INFINITY, 0);
    for (j, value) in score.iter().enumerate() {
        if matches!(tags[j].0, 'O' | 'E' | 'S') && *value > last.0 {
            last = (*value, j)
        }
    }
    let mut path = vec![last.1; logits.len()];
    for t in (1..logits.len()).rev() {
        path[t - 1] = back[t][path[t]]
    }
    let predictions: Vec<_> = path
        .iter()
        .zip(&log_probs)
        .map(|(&label, row)| {
            Some(TokenPrediction {
                label,
                probability: row[label].exp(),
            })
        })
        .collect();
    let tokens = NerEncoding {
        word_ids: vec![None; enc.len()],
        ..enc.clone()
    };
    decode_bio(text, &tokens, &predictions, config)
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
        let w = build_windows(&e, 6, 2, Some((100, 101)));
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
    #[test]
    fn windows_without_special_tokens_use_the_full_width() {
        let e = NerEncoding {
            ids: (0..10).collect(),
            starts: (0..10).collect(),
            ends: (1..11).collect(),
            word_ids: (0..10).map(Some).collect(),
        };
        let w = build_windows(&e, 6, 2, None);
        assert_eq!(
            w.iter()
                .map(|w| (w.token_start, w.token_end, w.offset, w.input_ids.len()))
                .collect::<Vec<_>>(),
            [(0, 6, 0, 6), (4, 10, 0, 6)]
        );
        // Token 5 sits 0 from the first window's edge and 1 from the second's.
        let logits: Vec<Vec<Vec<f64>>> = w
            .iter()
            .map(|w| {
                (0..w.input_ids.len())
                    .map(|i| vec![(w.token_start + i) as f64, w.token_start as f64])
                    .collect()
            })
            .collect();
        let merged = merge_window_logits(10, &w, &logits).unwrap();
        assert_eq!(merged[5], [5., 4.]);
        assert_eq!(merged[2], [2., 0.]);
    }

    fn bioes_config() -> NerDecodeConfig {
        NerDecodeConfig {
            model_name: "fixture".into(),
            labels: [
                "O", "B-person", "I-person", "E-person", "S-person", "B-secret", "I-secret",
                "E-secret", "S-secret",
            ]
            .map(str::to_owned)
            .to_vec(),
            label_map: HashMap::from([("person".into(), EntityType::Person)]),
            threshold: 0.,
        }
    }

    fn one_hot(label: usize, k: usize) -> Vec<f64> {
        (0..k).map(|i| if i == label { 4. } else { 0. }).collect()
    }

    fn words(text: &str) -> NerEncoding {
        let mut starts = Vec::new();
        let mut ends = Vec::new();
        let mut at = 0;
        for word in text.split(' ') {
            starts.push(at);
            ends.push(at + word.len());
            at += word.len() + 1;
        }
        NerEncoding {
            ids: (0..starts.len() as i64).collect(),
            word_ids: (0..starts.len()).map(Some).collect(),
            starts,
            ends,
        }
    }

    #[test]
    fn viterbi_decodes_bioes_spans_and_drops_unmapped_entities() {
        let text = "Ana María Ruiz y Bo usan sk-1";
        let enc = words(text);
        // O is never an entity's content; B I E and S follow the labels.
        let labels = [1, 2, 3, 0, 4, 0, 8];
        let logits: Vec<_> = labels.iter().map(|&l| one_hot(l, 9)).collect();
        let found = decode_bioes_viterbi(
            text,
            &enc,
            &logits,
            &bioes_config(),
            &TransitionBiases::default(),
        )
        .unwrap();
        assert_eq!(
            found
                .iter()
                .map(|d| (d.entity_type, d.value.as_str()))
                .collect::<Vec<_>>(),
            [
                (EntityType::Person, "Ana María Ruiz"),
                (EntityType::Person, "Bo")
            ]
        );
        assert!(found.iter().all(|d| text[d.start..d.end] == d.value));
    }

    #[test]
    fn viterbi_repairs_impossible_argmax_sequences() {
        let text = "x Ana Ruiz y";
        let enc = words(text);
        // Argmax says O I O O: an inside token with no beginning, which the
        // constraints forbid. The best allowed path reads it as S-person.
        let mut logits: Vec<_> = [0, 2, 0, 0].iter().map(|&l| one_hot(l, 9)).collect();
        logits[1][4] = 3.5;
        let found = decode_bioes_viterbi(
            text,
            &enc,
            &logits,
            &bioes_config(),
            &TransitionBiases::default(),
        )
        .unwrap();
        assert_eq!(
            found.iter().map(|d| d.value.as_str()).collect::<Vec<_>>(),
            ["Ana"]
        );
    }

    #[test]
    fn viterbi_biases_move_the_operating_point() {
        let text = "x Ana y";
        let enc = words(text);
        // "Ana" leans O (1.0) over S-person (0.6).
        let mut logits = vec![one_hot(0, 9), one_hot(0, 9), one_hot(0, 9)];
        logits[1] = (0..9)
            .map(|i| match i {
                0 => 1.,
                4 => 0.6,
                _ => -5.,
            })
            .collect();
        let config = bioes_config();
        let neutral = TransitionBiases::default();
        assert!(decode_bioes_viterbi(text, &enc, &logits, &config, &neutral)
            .unwrap()
            .is_empty());
        let eager = TransitionBiases {
            background_to_start: 1.,
            ..TransitionBiases::default()
        };
        let found = decode_bioes_viterbi(text, &enc, &logits, &config, &eager).unwrap();
        assert_eq!(
            found.iter().map(|d| d.value.as_str()).collect::<Vec<_>>(),
            ["Ana"]
        );
    }

    #[test]
    fn bioes_labels_are_validated() {
        assert!(bioes_tags(&["O".into(), "S-x".into()]).is_ok());
        assert!(bioes_tags(&["O".into(), "PER".into()]).is_err());
        assert!(bioes_tags(&["O".into(), "B-".into()]).is_err());
    }
}
