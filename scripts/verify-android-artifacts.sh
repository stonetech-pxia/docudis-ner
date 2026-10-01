#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 JNI_LIBS_DIR ABI [ABI ...]" >&2
  exit 2
fi

directory="$1"
shift

nm_tool="${ANDROID_NM:-}"
ndk_roots=()
[[ -n "${ANDROID_NDK_HOME:-}" ]] && ndk_roots+=("$ANDROID_NDK_HOME")
[[ -n "${ANDROID_NDK_ROOT:-}" ]] && ndk_roots+=("$ANDROID_NDK_ROOT")

for sdk_root in "${ANDROID_SDK_ROOT:-}" "${ANDROID_HOME:-}"; do
  [[ -d "$sdk_root/ndk" ]] || continue
  for candidate in "$sdk_root"/ndk/*; do
    [[ -d "$candidate" ]] && ndk_roots+=("$candidate")
  done
done

if [[ -z "$nm_tool" ]]; then
  for ndk_root in "${ndk_roots[@]}"; do
    nm_tool="$(find "$ndk_root/toolchains/llvm/prebuilt" -type f -name llvm-nm -print -quit)"
    [[ -n "$nm_tool" ]] && break
  done
fi
if [[ -z "$nm_tool" ]]; then nm_tool="$(command -v llvm-nm || true)"; fi
[[ -n "$nm_tool" ]] || {
  echo "unable to find llvm-nm; set ANDROID_NM, ANDROID_NDK_HOME, ANDROID_NDK_ROOT, ANDROID_SDK_ROOT, or ANDROID_HOME" >&2
  exit 1
}

required=(
  docudis_ner_v1_abi_version
  docudis_ner_v1_version
  docudis_ner_v1_load_json
  docudis_ner_v1_detect_json
  docudis_ner_v1_model_free
  docudis_ner_v1_buffer_free
  docudis_ner_v1_last_error_message
)

for abi in "$@"; do
  library="$directory/$abi/libdocudis_ner_capi.so"
  [[ -s "$library" ]] || { echo "missing $library" >&2; exit 1; }
  symbols="$($nm_tool -D --defined-only "$library")"
  for symbol in "${required[@]}"; do
    grep -q " $symbol$" <<< "$symbols" || {
      echo "$library does not export $symbol" >&2
      exit 1
    }
  done
done
