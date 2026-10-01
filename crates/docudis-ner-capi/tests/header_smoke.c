/* Copyright 2026 the Docudis contributors. Licensed under Apache-2.0. */

#include <string.h>

#include "docudis_ner.h"

int main(void) {
  DocudisNerV1Model *model = (DocudisNerV1Model *)1;
  DocudisNerV1Buffer output = {0};
  if (docudis_ner_v1_abi_version() != DOCUDIS_NER_V1_ABI_VERSION) return 1;
  if (docudis_ner_v1_version() == NULL) return 2;

  const char invalid[] = "not json";
  if (docudis_ner_v1_load_json((const uint8_t *)invalid, sizeof(invalid) - 1,
                               &model) != DOCUDIS_NER_V1_INVALID_JSON) return 3;
  if (model != NULL) return 4;
  if (strlen(docudis_ner_v1_last_error_message()) == 0) return 5;

  const char missing[] =
      "{\"schema_version\":1,"
      "\"onnxruntime_library\":\"/definitely/missing/libonnxruntime.so\","
      "\"spec\":{\"name\":\"m\",\"model\":\"m.onnx\","
      "\"tokenizer\":{\"kind\":\"wordpiece\",\"file\":\"t.json\"},"
      "\"labels\":[\"O\"],\"labelMap\":{},\"maxTokens\":8,\"stride\":2,"
      "\"threshold\":0.5,\"inputs\":{\"ids\":\"input_ids\","
      "\"mask\":\"attention_mask\"},\"output\":\"logits\"},"
      "\"model_path\":\"m.onnx\",\"tokenizer_path\":\"t.json\"}";
  if (docudis_ner_v1_load_json((const uint8_t *)missing, sizeof(missing) - 1,
                               &model) != DOCUDIS_NER_V1_NER_ERROR) return 6;
  if (model != NULL) return 7;

  const char detect[] = "{\"schema_version\":1,\"text\":\"Alice\"}";
  if (docudis_ner_v1_detect_json(NULL, (const uint8_t *)detect,
                                 sizeof(detect) - 1,
                                 &output) != DOCUDIS_NER_V1_INVALID_ARGUMENT) {
    return 8;
  }
  if (output.ptr != NULL || output.len != 0) return 9;
  docudis_ner_v1_buffer_free(&output);
  docudis_ner_v1_buffer_free(NULL);
  docudis_ner_v1_model_free(NULL);
  return 0;
}
