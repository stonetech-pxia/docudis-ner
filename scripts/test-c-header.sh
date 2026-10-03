#!/usr/bin/env bash
# Copyright 2026 the Docudis contributors. Licensed under Apache-2.0.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"
cargo build -p docudis-ner-capi

case "$(uname -s)" in
  Darwin) library="libdocudis_ner_capi.dylib" ;;
  Linux) library="libdocudis_ner_capi.so" ;;
  *) echo "C header smoke script supports macOS and Linux" >&2; exit 2 ;;
esac

cc crates/docudis-ner-capi/tests/header_smoke.c \
  -Icrates/docudis-ner-capi/include \
  -Ltarget/debug -ldocudis_ner_capi "-Wl,-rpath,$repo_root/target/debug" \
  -o target/debug/docudis_ner_header_smoke
test -s "target/debug/$library"
target/debug/docudis_ner_header_smoke
