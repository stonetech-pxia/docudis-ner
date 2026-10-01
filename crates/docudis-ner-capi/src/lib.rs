// Copyright 2026 the Docudis contributors. Licensed under Apache-2.0.

use docudis_core::Detection;
use docudis_ner::{ModelSpec, NerModel};
use serde::{Deserialize, Serialize};
use std::{
    cell::RefCell,
    ffi::{c_char, CString},
    panic::{catch_unwind, AssertUnwindSafe},
    path::PathBuf,
    ptr, slice,
    sync::Mutex,
};

pub const ABI_VERSION: u32 = 1;
const VERSION: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocudisNerV1Status {
    Ok = 0,
    InvalidArgument = 1,
    InvalidUtf8 = 2,
    InvalidJson = 3,
    NerError = 4,
    Panic = 255,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DocudisNerV1Buffer {
    pub ptr: *mut u8,
    pub len: usize,
    pub capacity: usize,
}

impl DocudisNerV1Buffer {
    const EMPTY: Self = Self {
        ptr: ptr::null_mut(),
        len: 0,
        capacity: 0,
    };

    fn from_vec(mut bytes: Vec<u8>) -> Self {
        let buffer = Self {
            ptr: bytes.as_mut_ptr(),
            len: bytes.len(),
            capacity: bytes.capacity(),
        };
        std::mem::forget(bytes);
        buffer
    }
}

/// Opaque loaded model. Calls on one model are serialised.
pub struct DocudisNerV1Model(Mutex<NerModel>);

#[derive(Debug, Deserialize)]
struct LoadRequest {
    schema_version: u32,
    /// Path or, on Android, the bare `libonnxruntime.so` the app ships.
    onnxruntime_library: String,
    /// The model's `model.json` object.
    spec: serde_json::Value,
    model_path: PathBuf,
    tokenizer_path: PathBuf,
}

#[derive(Debug, Deserialize)]
struct DetectRequest {
    schema_version: u32,
    text: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct DetectResponse {
    schema_version: u32,
    detections: Vec<Detection>,
}

struct ApiFailure(DocudisNerV1Status, String);

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_error(message: impl AsRef<str>) {
    let sanitized = message.as_ref().replace('\0', "�");
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = CString::new(sanitized).unwrap_or_default();
    });
}

fn clear_error() {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = CString::default());
}

#[no_mangle]
pub extern "C" fn docudis_ner_v1_abi_version() -> u32 {
    ABI_VERSION
}

#[no_mangle]
pub extern "C" fn docudis_ner_v1_version() -> *const c_char {
    VERSION.as_ptr().cast()
}

#[no_mangle]
pub extern "C" fn docudis_ner_v1_last_error_message() -> *const c_char {
    LAST_ERROR.with(|slot| slot.borrow().as_ptr())
}

fn parse_request<T: for<'de> Deserialize<'de>>(input: &str) -> Result<T, ApiFailure> {
    serde_json::from_str(input).map_err(|error| {
        ApiFailure(
            DocudisNerV1Status::InvalidJson,
            format!("invalid request JSON: {error}"),
        )
    })
}

fn validate_schema(version: u32) -> Result<(), ApiFailure> {
    if version == 1 {
        Ok(())
    } else {
        Err(ApiFailure(
            DocudisNerV1Status::InvalidArgument,
            format!("unsupported schema_version {version}; expected 1"),
        ))
    }
}

/// Runs `operation` on the UTF-8 input behind the panic and error boundary.
/// `reset_out` clears the caller's output before anything else can fail.
unsafe fn invoke(
    input: *const u8,
    input_len: usize,
    out_is_null: bool,
    reset_out: impl FnOnce(),
    operation: impl FnOnce(&str) -> Result<(), ApiFailure>,
) -> DocudisNerV1Status {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if out_is_null {
            set_error("out must not be NULL");
            return DocudisNerV1Status::InvalidArgument;
        }
        reset_out();
        clear_error();
        if input.is_null() {
            set_error("input must not be NULL");
            return DocudisNerV1Status::InvalidArgument;
        }
        // SAFETY: the public C contract requires a readable input range.
        let bytes = unsafe { slice::from_raw_parts(input, input_len) };
        let input = match std::str::from_utf8(bytes) {
            Ok(value) => value,
            Err(error) => {
                set_error(format!("input is not valid UTF-8: {error}"));
                return DocudisNerV1Status::InvalidUtf8;
            }
        };
        match operation(input) {
            Ok(()) => DocudisNerV1Status::Ok,
            Err(ApiFailure(status, message)) => {
                set_error(message);
                status
            }
        }
    }));
    result.unwrap_or_else(|_| {
        set_error("Rust panic caught at the Docudis NER C ABI boundary");
        DocudisNerV1Status::Panic
    })
}

#[no_mangle]
/// Loads ONNX Runtime (once per process) and one model.
///
/// # Safety
///
/// `input` must point to `input_len` readable bytes. `out_model` must point
/// to writable storage; it receives NULL on failure. A model is released only
/// with [`docudis_ner_v1_model_free`].
pub unsafe extern "C" fn docudis_ner_v1_load_json(
    input: *const u8,
    input_len: usize,
    out_model: *mut *mut DocudisNerV1Model,
) -> DocudisNerV1Status {
    // SAFETY: `out_model` is checked for NULL before either closure writes it.
    unsafe {
        invoke(
            input,
            input_len,
            out_model.is_null(),
            || out_model.write(ptr::null_mut()),
            |input| {
                let request: LoadRequest = parse_request(input)?;
                validate_schema(request.schema_version)?;
                let spec = ModelSpec::from_json(&request.spec.to_string())
                    .map_err(|error| ApiFailure(DocudisNerV1Status::InvalidArgument, error))?;
                let model = NerModel::load(
                    &request.onnxruntime_library,
                    spec,
                    &request.model_path,
                    &request.tokenizer_path,
                )
                .map_err(|error| ApiFailure(DocudisNerV1Status::NerError, error.to_string()))?;
                let handle = Box::new(DocudisNerV1Model(Mutex::new(model)));
                out_model.write(Box::into_raw(handle));
                Ok(())
            },
        )
    }
}

#[no_mangle]
/// Detects entities in `{"schema_version":1,"text":"..."}`.
///
/// # Safety
///
/// `model` must be a live model from [`docudis_ner_v1_load_json`]. `input`
/// must point to `input_len` readable bytes. `out` must point to writable
/// storage; a successful output is released only with
/// [`docudis_ner_v1_buffer_free`].
pub unsafe extern "C" fn docudis_ner_v1_detect_json(
    model: *const DocudisNerV1Model,
    input: *const u8,
    input_len: usize,
    out: *mut DocudisNerV1Buffer,
) -> DocudisNerV1Status {
    // SAFETY: `out` is checked for NULL before either closure writes it.
    unsafe {
        invoke(
            input,
            input_len,
            out.is_null(),
            || out.write(DocudisNerV1Buffer::EMPTY),
            |input| {
                if model.is_null() {
                    return Err(ApiFailure(
                        DocudisNerV1Status::InvalidArgument,
                        "model must not be NULL".into(),
                    ));
                }
                let request: DetectRequest = parse_request(input)?;
                validate_schema(request.schema_version)?;
                // SAFETY: the caller promises a live model.
                let model = &(*model).0;
                let mut model = model
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let detections = model
                    .detect(&request.text)
                    .map_err(|error| ApiFailure(DocudisNerV1Status::NerError, error.to_string()))?;
                let bytes = serde_json::to_vec(&DetectResponse {
                    schema_version: 1,
                    detections,
                })
                .map_err(|error| {
                    ApiFailure(
                        DocudisNerV1Status::NerError,
                        format!("could not serialize response JSON: {error}"),
                    )
                })?;
                out.write(DocudisNerV1Buffer::from_vec(bytes));
                Ok(())
            },
        )
    }
}

#[no_mangle]
/// Releases a model. NULL is accepted.
///
/// # Safety
///
/// `model` must be NULL or a live model from [`docudis_ner_v1_load_json`]
/// that no other thread is using; it may be released only once.
pub unsafe extern "C" fn docudis_ner_v1_model_free(model: *mut DocudisNerV1Model) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if !model.is_null() {
            // SAFETY: created by Box::into_raw in docudis_ner_v1_load_json.
            drop(unsafe { Box::from_raw(model) });
        }
    }));
}

#[no_mangle]
/// Releases a successful output buffer and zeroes it. NULL is accepted.
///
/// # Safety
///
/// `buffer` must be NULL or point to writable storage holding an empty buffer
/// or a live buffer returned by this same loaded library, released only once.
pub unsafe extern "C" fn docudis_ner_v1_buffer_free(buffer: *mut DocudisNerV1Buffer) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if buffer.is_null() {
            return;
        }
        // SAFETY: the caller promises a buffer structure from this library.
        let owned = unsafe { buffer.read() };
        if !owned.ptr.is_null() {
            // SAFETY: outputs are created from a Vec with exactly these fields.
            unsafe { drop(Vec::from_raw_parts(owned.ptr, owned.len, owned.capacity)) };
        }
        // SAFETY: the caller provided writable storage for the structure.
        unsafe { buffer.write(DocudisNerV1Buffer::EMPTY) };
    }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    fn last_error() -> String {
        unsafe { CStr::from_ptr(docudis_ner_v1_last_error_message()) }
            .to_str()
            .unwrap()
            .to_owned()
    }

    #[test]
    fn invalid_load_json_has_a_stable_status_and_readable_error() {
        let mut model = ptr::dangling_mut();
        let request = b"not json";
        let status =
            unsafe { docudis_ner_v1_load_json(request.as_ptr(), request.len(), &mut model) };
        assert_eq!(status, DocudisNerV1Status::InvalidJson);
        assert!(model.is_null());
        assert!(last_error().contains("invalid request JSON"));
    }

    #[test]
    fn an_invalid_spec_is_rejected_before_onnx_runtime_loads() {
        let mut model = ptr::null_mut();
        let request = br#"{"schema_version":1,"onnxruntime_library":"x","spec":{"name":"m"},"model_path":"m","tokenizer_path":"t"}"#;
        let status =
            unsafe { docudis_ner_v1_load_json(request.as_ptr(), request.len(), &mut model) };
        assert_eq!(status, DocudisNerV1Status::InvalidArgument);
        assert!(last_error().contains("invalid model.json"));
    }

    #[test]
    fn detect_needs_a_model_and_frees_are_null_safe() {
        let mut output = DocudisNerV1Buffer::EMPTY;
        let request = br#"{"schema_version":1,"text":"Alice"}"#;
        let status = unsafe {
            docudis_ner_v1_detect_json(ptr::null(), request.as_ptr(), request.len(), &mut output)
        };
        assert_eq!(status, DocudisNerV1Status::InvalidArgument);
        assert!(output.ptr.is_null());
        unsafe {
            docudis_ner_v1_buffer_free(ptr::null_mut());
            docudis_ner_v1_buffer_free(&mut output);
            docudis_ner_v1_model_free(ptr::null_mut());
        }
        assert_eq!(docudis_ner_v1_abi_version(), ABI_VERSION);
    }
}
