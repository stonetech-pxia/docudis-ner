/* Copyright 2026 the Docudis contributors. Licensed under Apache-2.0. */

#ifndef DOCUDIS_NER_H
#define DOCUDIS_NER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  if defined(DOCUDIS_NER_BUILD_SHARED)
#    define DOCUDIS_NER_API __declspec(dllexport)
#  else
#    define DOCUDIS_NER_API
#  endif
#else
#  define DOCUDIS_NER_API __attribute__((visibility("default")))
#endif

#define DOCUDIS_NER_V1_ABI_VERSION 1u

typedef enum DocudisNerV1Status {
  DOCUDIS_NER_V1_OK = 0,
  DOCUDIS_NER_V1_INVALID_ARGUMENT = 1,
  DOCUDIS_NER_V1_INVALID_UTF8 = 2,
  DOCUDIS_NER_V1_INVALID_JSON = 3,
  DOCUDIS_NER_V1_NER_ERROR = 4,
  DOCUDIS_NER_V1_PANIC = 255
} DocudisNerV1Status;

/* Owned output bytes. Treat fields as read-only and release with
 * docudis_ner_v1_buffer_free on the same library that allocated them. */
typedef struct DocudisNerV1Buffer {
  uint8_t *ptr;
  size_t len;
  size_t capacity;
} DocudisNerV1Buffer;

/* A loaded model. Calls on one model are serialised; distinct models may be
 * used from different threads. */
typedef struct DocudisNerV1Model DocudisNerV1Model;

DOCUDIS_NER_API uint32_t docudis_ner_v1_abi_version(void);

/* Static UTF-8/NUL-terminated library version. Never free this pointer. */
DOCUDIS_NER_API const char *docudis_ner_v1_version(void);

/*
 * Loads ONNX Runtime (once per process, the first library named wins) and a
 * model. Input is UTF-8 JSON:
 *   {"schema_version":1,"onnxruntime_library":"libonnxruntime.so",
 *    "spec":{...model.json...},"model_path":"...","tokenizer_path":"..."}
 * `onnxruntime_library` is a path, or a bare file name the platform loader
 * resolves (on Android, the libonnxruntime.so the app ships). On success
 * `*out_model` owns the model; on failure it is NULL.
 */
DOCUDIS_NER_API DocudisNerV1Status docudis_ner_v1_load_json(
    const uint8_t *input,
    size_t input_len,
    DocudisNerV1Model **out_model);

/*
 * Input:  {"schema_version":1,"text":"..."}
 * Output: {"schema_version":1,"detections":[...]} with half-open UTF-8 byte
 * offsets, `source` "model" and `detector` "ner:<model name>". The output is
 * not NUL-terminated; on failure `out` is reset to an empty buffer.
 */
DOCUDIS_NER_API DocudisNerV1Status docudis_ner_v1_detect_json(
    const DocudisNerV1Model *model,
    const uint8_t *input,
    size_t input_len,
    DocudisNerV1Buffer *out);

/* Releases a model no other thread is using. NULL is accepted. */
DOCUDIS_NER_API void docudis_ner_v1_model_free(DocudisNerV1Model *model);

/* Releases a successful output buffer and zeroes it. NULL is accepted. */
DOCUDIS_NER_API void docudis_ner_v1_buffer_free(DocudisNerV1Buffer *buffer);

/* Thread-local, static, NUL-terminated detail for the most recent failure on
 * the current thread. Valid until the next C API call on that thread. Never
 * free this pointer. The empty string means there is no current error. */
DOCUDIS_NER_API const char *docudis_ner_v1_last_error_message(void);

#ifdef __cplusplus
}
#endif

#endif /* DOCUDIS_NER_H */
