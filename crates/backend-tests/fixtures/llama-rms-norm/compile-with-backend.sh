#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

case "$(uname -s)" in
  Darwin)
    BACKEND="${ROOT_DIR}/target/debug/libstair_rust.dylib"
    ;;
  Linux)
    BACKEND="${ROOT_DIR}/target/debug/libstair_rust.so"
    ;;
  MINGW*|MSYS*|CYGWIN*)
    BACKEND="${ROOT_DIR}/target/debug/stair_rust.dll"
    ;;
  *)
    echo "unsupported host OS: $(uname -s)" >&2
    exit 1
    ;;
esac

cargo build --manifest-path "${ROOT_DIR}/Cargo.toml" -p stair-rust

TARGET_DIR="${ROOT_DIR}/target/stair-backend-tests-llama-rms-norm"
rm -rf "${TARGET_DIR}"

CARGO_TARGET_DIR="${TARGET_DIR}" \
cargo rustc --manifest-path "${SCRIPT_DIR}/Cargo.toml" --bin stair-llama-rms-norm -- \
  --emit=obj \
  "-Zcodegen-backend=${BACKEND}" \
  -Coverflow-checks=off

find "${TARGET_DIR}" -name '*.o' -size +0 -print
find "${TARGET_DIR}" -name '*.stair' -size +0 -print
