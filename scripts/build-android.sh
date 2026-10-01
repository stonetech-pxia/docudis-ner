#!/usr/bin/env bash
set -euo pipefail

profile="${1:-release}"
case "$profile" in
  debug|release) ;;
  *) echo "usage: $0 [debug|release]" >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
abis="${DOCUDIS_ANDROID_ABIS:-arm64-v8a,armeabi-v7a,x86_64}"
# The app's minSdk; mallopt, used to release tokenizer memory, needs 26.
platform="${DOCUDIS_ANDROID_PLATFORM:-26}"
output="$repo_root/dist/android/$profile/jniLibs"

command -v cargo >/dev/null
command -v cargo-ndk >/dev/null || {
  echo "cargo-ndk is required: cargo install cargo-ndk --locked" >&2
  exit 1
}

args=()
IFS=',' read -r -a requested <<< "$abis"
for abi in "${requested[@]}"; do
  case "$abi" in
    arm64-v8a|armeabi-v7a|x86_64) args+=("-t" "$abi") ;;
    *) echo "unsupported Android ABI: $abi" >&2; exit 2 ;;
  esac
done

mkdir -p "$output"
build_args=(build -p docudis-ner-capi)
if [[ "$profile" == release ]]; then build_args+=(--release); fi

(cd "$repo_root" && cargo ndk "${args[@]}" --platform "$platform" -o "$output" "${build_args[@]}")

"$repo_root/scripts/verify-android-artifacts.sh" "$output" "${requested[@]}"
