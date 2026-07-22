# crabbit

An experimental [rustc codegen backend](https://rustc-dev-guide.rust-lang.org/backend/backend-agnostic.html)
that lowers rustc MIR straight to native machine code and a Mach-O object
file, without going through LLVM. It's built on top of
[`pliron`](https://github.com/pliron-org/pliron), an MLIR-style,
multi-level IR framework in Rust.

> **Status: research / work in progress.** APIs, IR dialects and CLI
> surfaces change without notice, only macOS (`aarch64-apple-darwin` and
> `x86_64-apple-darwin`) is supported today, and correctness is tracked by a
> growing set of fixture programs rather than a stability guarantee. Expect
> gaps against arbitrary Rust input.

## How it fits together

rustc hands `crabbit` a crate's MIR through the `CodegenBackend` trait.
From there the crate is progressively imported and lowered through a stack
of [pliron](https://github.com/pliron-org/pliron) dialects, each owned by
its own crate in this workspace:

```
rustc MIR
   │  crates/crabbit (importer)
   ▼
mir dialect            (crabbit-mir)
   │  ConvertMirToLLVMPass
   ▼
llvm dialect            (llvm-compat)
   │  inline / mem2reg / SROA / simplify-cfg / phi<->block-args passes
   ▼
aarch64 / x86_64 dialect (pliron-ll)
   │  instruction selection, register allocation, encoding
   ▼
Mach-O object file
```

`pliron` supplies the underlying IR data structures (operations, types,
attributes, regions, the pass manager, dialect-conversion framework, ...);
everything above is dialects and passes defined on top of it.

## Crates

| Crate | Kind | Description |
| --- | --- | --- |
| [`crabbit`](crates/crabbit) | `dylib` | The rustc codegen backend itself (`#![feature(rustc_private)]`); imports MIR and drives the lowering pipeline to an object file. |
| [`crabbit-mir`](crates/mir) | lib | The `mir` dialect (rustc MIR represented as pliron IR) and its lowering to the `llvm` dialect. |
| [`llvm-compat`](crates/llvm-compat) | lib | The `llvm` dialect, the dialect-conversion framework, and LLVM-level cleanup passes (inline, mem2reg, SROA, simplify, ...). |
| [`pliron-ll`](crates/pliron-ll) | lib | Native machine-level dialects and backend pipelines: aarch64/x86_64 instruction selection, register allocation, encoding, and Mach-O emission. |
| [`crabbit-inspect-driver`](crates/inspect-driver) | bin | A [`pliron-inspect`](https://github.com/montaglue/pliron-inspect) driver wiring up the crabbit dialect stack, for stepping through the pass pipeline on hand-written IR. |
| [`backend-tests`](crates/backend-tests) | tests (unpublished) | Integration tests that build the `crabbit` dylib and use it as `rustc`'s codegen backend against small fixture crates under `fixtures/`. |

The crate was previously developed under the name **STAIR**; a few internal
symbols, trace/log strings and test fixtures still use that name while the
migration to `crabbit`/`pliron` finishes.

## Requirements

- The pinned nightly toolchain in [`rust-toolchain.toml`](rust-toolchain.toml),
  with the `rustc-dev` and `llvm-tools-preview` components (rustup installs
  these automatically from the toolchain file).
- macOS. Both `aarch64-apple-darwin` and `x86_64-apple-darwin` are supported
  as compilation targets; the backend itself only builds/runs on macOS.

## Building

```sh
cargo build --workspace
```

## Using it as a codegen backend

Point `rustc`/`cargo` at the built dylib with `-Zcodegen-backend`:

```sh
cargo build -p crabbit
BACKEND="target/debug/libcrabbit.dylib"   # libcrabbit.so on other Unixes, crabbit.dll on Windows (untested)

RUSTFLAGS="-Zcodegen-backend=${BACKEND}" \
cargo +nightly rustc --manifest-path path/to/some/crate/Cargo.toml
```

See `crates/backend-tests/fixtures/*/compile-with-backend.sh` for worked
examples, and `crates/backend-tests/tests/` for the integration tests that
run them.

## Inspecting IR

`crabbit-inspect-driver` links the full dialect stack (mir, llvm, ll,
aarch64, x86_64, macho) plus the mid-level pass pipeline behind the
[`pliron-inspect`](https://github.com/montaglue/pliron-inspect) stdio
protocol, so passes can be run and inspected step by step against
hand-written IR:

```sh
cargo run -p crabbit-inspect-driver -- path/to/module.mlir
```

## Testing

```sh
cargo test --workspace
```

`backend-tests` additionally builds and runs the fixture crates under
`crates/backend-tests/fixtures/` through the real codegen backend; see the
`README.md` in individual fixture directories for how to run them by hand.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
