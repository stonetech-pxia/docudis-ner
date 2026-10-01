// Copyright 2026 the Docudis contributors. Licensed under Apache-2.0.

use docudis_ner::{HuggingFaceNerTokenizer, NerTokenizer, TokenizerKind};

#[test]
fn bundled_wordpiece_tokenizer_returns_utf8_spans() {
    let tokenizer = HuggingFaceNerTokenizer::from_json(
        include_str!("../../../testdata/tokenizers/wordpiece.json"),
        TokenizerKind::WordPiece,
    )
    .unwrap();
    let text = "张三 😀 met Élodie";
    let enc = tokenizer.encode(text).unwrap();
    assert!(!enc.is_empty());
    assert!(enc.validate(text));
    assert!(tokenizer.start_id() >= 0 && tokenizer.end_id() >= 0);
    for (start, end) in enc.starts.iter().zip(&enc.ends) {
        assert!(!text[*start..*end].trim().is_empty())
    }
}

#[test]
fn bundled_sentencepiece_tokenizer_realigns_normalized_characters() {
    let tokenizer = HuggingFaceNerTokenizer::from_json(
        include_str!("../../../testdata/tokenizers/sentencepiece.json"),
        TokenizerKind::SentencePiece,
    )
    .unwrap();
    let text = "Fiebre 37,8\r\nºC. Control por Dr. Tomás Garrido Lucena.";
    let enc = tokenizer.encode(text).unwrap();
    assert!(!enc.is_empty());
    assert!(enc.validate(text));
    for pair in enc
        .starts
        .iter()
        .zip(&enc.ends)
        .collect::<Vec<_>>()
        .windows(2)
    {
        assert!(pair[1].0 >= pair[0].1)
    }
    assert_eq!(*enc.ends.last().unwrap(), text.len());
}

#[test]
fn sentencepiece_space_piece_before_cjk_is_a_valid_zero_width_token() {
    // XLM-R emits a standalone "▁" before CJK text. It covers no character,
    // and the Dart reference keeps it in the model input as an empty span.
    let tokenizer = HuggingFaceNerTokenizer::from_json(
        include_str!("../../../testdata/tokenizers/sentencepiece.json"),
        TokenizerKind::SentencePiece,
    )
    .unwrap();
    let text = "张三昨天把报告发给了李四和王五。";
    let enc = tokenizer.encode(text).unwrap();
    assert_eq!((enc.starts[0], enc.ends[0]), (0, 0));
    assert!(enc.validate(text));
}
