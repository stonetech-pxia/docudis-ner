// Copyright 2026 the Docudis contributors. Licensed under Apache-2.0.

use std::fmt;
use std::path::Path;
use std::sync::OnceLock;

use docudis_core::Detection;
use ort::session::Session;
use ort::value::Tensor;

use crate::{
    build_windows, decode_bio, decode_bioes_viterbi, merge_window_logits, merge_window_predictions,
    title_cased, DecoderSpec, HuggingFaceNerTokenizer, ModelSpec, NerDecodeConfig, NerTokenizer,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NerError(String);

impl fmt::Display for NerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for NerError {}

fn error(context: &str, cause: impl fmt::Display) -> NerError {
    NerError(format!("{context}: {cause}"))
}

/// The outcome of the first load, kept for the life of the process. ONNX
/// Runtime accepts one environment per process, and ort 2.0.0-rc.13 cannot
/// retry a failed load: its library slot is marked initialised although it
/// is empty, and the next call panics on a null handle.
static RUNTIME: OnceLock<Result<(), NerError>> = OnceLock::new();

fn init_runtime(library: &str) -> Result<(), NerError> {
    RUNTIME
        .get_or_init(|| {
            // The official macOS build starts Microsoft's 1DS telemetry while
            // it creates the environment, before with_telemetry(false) can
            // turn it off: it saves a device ID and queues events under
            // Application Support/Microsoft, then keeps trying to upload
            // them. ONNX Runtime reads this variable first and never starts it.
            std::env::set_var("ORT_DISABLE_TELEMETRY", "1");
            // ort turns ONNX Runtime's telemetry on unless told otherwise.
            ort::init_from(library)
                .map_err(|e| error(&format!("cannot load ONNX Runtime from {library:?}"), e))?
                .with_telemetry(false)
                .commit();
            Ok(())
        })
        .clone()
}

/// A token-classification model run through ONNX Runtime.
///
/// With the `bio` decoder it mirrors the Dart `NerDetector`: the model reads
/// the title-cased text in overlapping windows, each token keeps the
/// prediction from the window where it sits farthest from an edge, and BIO
/// tags decode into spans of the original text (title casing keeps byte
/// lengths). The `bioes-viterbi` decoder keeps the same windows and merge but
/// decodes the merged logits with a constrained Viterbi; title casing follows
/// `titleCase` in either case.
pub struct NerModel {
    session: Session,
    tokenizer: HuggingFaceNerTokenizer,
    spec: ModelSpec,
    config: NerDecodeConfig,
}

impl NerModel {
    /// `runtime_library` is a path or, on Android, the bare file name
    /// `libonnxruntime.so` that the app ships.
    pub fn load(
        runtime_library: &str,
        spec: ModelSpec,
        model_path: &Path,
        tokenizer_path: &Path,
    ) -> Result<Self, NerError> {
        init_runtime(runtime_library)?;
        let kind = spec.tokenizer_kind().map_err(NerError)?;
        let source = std::fs::read_to_string(tokenizer_path)
            .map_err(|e| error(&format!("cannot read {}", tokenizer_path.display()), e))?;
        let tokenizer = HuggingFaceNerTokenizer::from_json(&source, kind)
            .map_err(|e| error("invalid tokenizer", e))?;
        drop(source);
        release_free_memory();
        let session = Session::builder()
            .and_then(|mut builder| builder.commit_from_file(model_path))
            .map_err(|e| error(&format!("cannot load {}", model_path.display()), e))?;
        let config = spec.decode_config();
        Ok(Self {
            session,
            tokenizer,
            spec,
            config,
        })
    }

    pub fn name(&self) -> &str {
        &self.spec.name
    }

    /// Detections in half-open UTF-8 byte offsets of `text`.
    pub fn detect(&mut self, text: &str) -> Result<Vec<Detection>, NerError> {
        let enc = if self.spec.title_case {
            self.tokenizer.encode(&title_cased(text))
        } else {
            self.tokenizer.encode(text)
        }
        .map_err(|e| error("tokenization failed", e))?;
        if enc.is_empty() {
            return Ok(Vec::new());
        }
        let windows = build_windows(
            &enc,
            self.spec.max_tokens,
            self.spec.stride,
            self.tokenizer.special_ids(),
        );
        let width = self.spec.labels.len();
        let mut logits = Vec::with_capacity(windows.len());
        for window in &windows {
            let n = window.input_ids.len();
            let ids = Tensor::from_array(([1usize, n], window.input_ids.clone()))
                .map_err(|e| error("input tensor", e))?;
            let mask = Tensor::from_array(([1usize, n], window.attention_mask.clone()))
                .map_err(|e| error("input tensor", e))?;
            let outputs = self
                .session
                .run(ort::inputs![
                    self.spec.inputs.ids.as_str() => ids,
                    self.spec.inputs.mask.as_str() => mask,
                ])
                .map_err(|e| error("inference failed", e))?;
            let output = outputs
                .get(&self.spec.output)
                .ok_or_else(|| NerError(format!("model has no output {:?}", self.spec.output)))?;
            let (_, data) = output
                .try_extract_tensor::<f32>()
                .map_err(|e| error("unexpected output tensor", e))?;
            if data.len() != n * width {
                return Err(NerError(format!(
                    "logits shape mismatch: {} != {n} x {width}",
                    data.len()
                )));
            }
            logits.push(
                data.chunks(width)
                    .map(|row| row.iter().map(|v| f64::from(*v)).collect())
                    .collect::<Vec<Vec<f64>>>(),
            );
        }
        match &self.spec.decoder {
            DecoderSpec::Bio => {
                let predictions =
                    merge_window_predictions(enc.len(), &windows, &logits).map_err(NerError)?;
                decode_bio(text, &enc, &predictions, &self.config)
            }
            DecoderSpec::BioesViterbi { biases } => {
                let merged = merge_window_logits(enc.len(), &windows, &logits).map_err(NerError)?;
                decode_bioes_viterbi(text, &enc, &merged, &self.config, biases)
            }
        }
        .map_err(NerError)
    }
}

/// Parsing a large tokenizer.json leaves ~90 MB of freed pages that the
/// Android allocator keeps; hand them back before the model is loaded.
fn release_free_memory() {
    #[cfg(target_os = "android")]
    {
        extern "C" {
            fn mallopt(param: i32, value: i32) -> i32;
        }
        const M_PURGE: i32 = -101;
        const M_PURGE_ALL: i32 = -104;
        // SAFETY: mallopt only adjusts the allocator; unknown parameters
        // return 0 on older Android versions.
        unsafe {
            if mallopt(M_PURGE_ALL, 0) == 0 {
                mallopt(M_PURGE, 0);
            }
        }
    }
}
