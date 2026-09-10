# crabbit

A [rustc codegen backend](https://rustc-dev-guide.rust-lang.org/backend/backend-agnostic.html)
written entirely in Rust: MIR is imported into
[`pliron`](https://github.com/pliron-org/pliron) (an MLIR-style IR framework)
dialects and lowered — mid-end optimizations, instruction selection, register
allocation, encoding — down to native **aarch64 ELF** and **Mach-O** objects,
with **zero LLVM in the binary path**. The same dialect stack also emits
**native NVPTX**: `#[no_mangle] __stair_kernel_*` functions become PTX that
runs on real GPUs, again without LLVM or NVVM.

Around the compiler sits a toolchain built for *studying* compilation as much
as performing it:

- **`crabbit-analysisd`** — a resident analysis server: load printed IR, run
  the pipeline under different configurations on a worker pool, fetch the IR
  after any pass, get hover/def/references answers about it, download the
  object — over stdio or loopback HTTP, with a web UI in the sibling
  [`pliron-inspect`](https://github.com/montaglue/pliron-inspect) project.
- **Backward profile attribution** — every emitted instruction carries
  provenance through all lowering levels, so `perf` samples (CPU) and Nsight
  Compute stalls (GPU) are lifted *back through the pipeline* to the source
  op — and to the optimization decision — that produced the code. See
  [docs/BACKWARD-PGO.md](docs/BACKWARD-PGO.md); we believe this framing of
  PGO is new.
- **Swappable research register allocators** (cargo features `eregalloc`,
  `cmt`) and an A/B perf harness, so allocation policies are compared on
  real programs, not synthetic corpora.

> **Status: research project.** Correctness is enforced by an executable
> fixture suite and a 14-kernel CPU/GPU corpus with checksum gates, not by a
> stability guarantee. Expect gaps against arbitrary Rust input.

## Highlights (measured, reproducible in-repo)

- **Real crates compile.** All four `apache/arrow-rs` core crates
  (`arrow-schema`, `arrow-buffer`, `arrow-data`, `arrow-array`) compile
  through crabbit (`scripts/perf-harness/` drives this; corpus B).
- **GPU code within reach of nvcc.** On a 14-kernel corpus (each kernel as
  Rust-via-crabbit, Rust-via-LLVM from *identical IR*, and CUDA C++ via
  nvcc), crabbit's native PTX is geomean **1.13× nvcc**, with **10/14 kernels
  within 3%** and zero correctness failures
  (results: `kernel-corpus/results/gpu/20260828-104400/gpu_analysis.md`).
- **Optimizations that pay measurably.** Adding constant-divisor strength
  reduction took the `seidel_2d` GPU kernel from **3.93 ms → 2.06 ms**;
  a target-profile-aware sinking rule (don't add control dependence under
  SIMT divergence) recovered a measured 10% regression on `gemm_tiled`
  (`docs/MIDEND-PLAN.md` records each finding with its mechanism).
- **Attribution that survives audits.** Synthetic samples injected on a
  machine-instruction range provably lift to the correct source-level op —
  including a 50/50 split through a GVN merge
  (`crates/pliron-ll/tests/opmap_audit.rs`). On real hardware, 72% of
  `seidel_2d`'s GPU samples attribute to one source op (the consumer of the
  recurrence load), matching the PTX analysis
  (results: `kernel-corpus/results/gpu-attribution/20260831-090227/ANALYSIS.md`).
- **Honest experiments, nulls included.** Measured-cost feedback into the
  register allocator: null on this corpus. Analytic (spectral) block
  frequencies vs. measured: median Spearman **0.002** — the uniform-prior
  model cannot represent iteration counts
  (E1: `kernel-corpus/results/cpu-feedback/20260831/E1-ANALYSIS.md`,
  E4: `kernel-corpus/results/cpu-feedback/20260831/E4-ANALYSIS.md`).

[SCREENSHOT: arrow-schema module IR loaded in the pliron-inspect UI]
<!-- capture: CRABBIT_EMIT_IR=/tmp/ir cargo rustc -p arrow-schema (in an
     arrow-rs checkout, with -Zcodegen-backend), then crabbit-analysisd
     --http 127.0.0.1:8177 & pliron-inspect --server 127.0.0.1:8177; paste
     the .plir into the module panel; screenshot the IR view. -->

[SCREENSHOT: per-pass pipeline run with live progress in the UI]
<!-- capture: same session; start a run (target aarch64-linux, config
     CRABBIT_REGALLOC=eregalloc); screenshot the pass list mid-run. -->

[SCREENSHOT: run_costs heat view — real perf samples attributed per source op]
<!-- capture: profile a corpus binary per scripts/perf-harness/README.md
     ("Profile feedback" section), POST run_costs with the op_costs.json,
     screenshot the per-op cost table / IR panel. -->

## Platform matrix

| Component | Where it runs |
| --- | --- |
| Native backend (ELF) | **aarch64-linux** (primary; developed & tested on a DGX Spark) |
| Native backend (Mach-O) | aarch64/x86_64 macOS (earlier target; less exercised today) |
| NVPTX kernel path | any host; execution needs an NVIDIA GPU (tested: GB10, `sm_121`, CUDA 13) |
| Analysis server, IR tools, LSP stack | anywhere Rust runs |
| Profile attribution (CPU) | Linux `perf` (`kernel.perf_event_paranoid ≤ 2`) |
| Profile attribution (GPU) | Nsight Compute (`ncu`) |

## Quickstart

**Toolchain**: pinned by [`rust-toolchain.toml`](rust-toolchain.toml)
(nightly + `rustc-dev`); rustup picks it up automatically.

> **TODO(deps):** two dialect crates (`dialect-mir`, `mir-lower`) and the
> inspect driver library currently resolve from sibling checkouts
> (`../cuda-oxide`, `../pliron-inspect`); the research engines
> (`../eregalloc`, `../combinatorial-matrix-theory`) sit behind the
> `eregalloc`/`cmt` features. Relocation to pinned git dependencies is in
> progress — until it lands, clone those siblings next to this repo.

### Mode 1 — codegen backend

```sh
cargo build -p crabbit
BACKEND="$PWD/target/debug/libcrabbit.so"     # .dylib on macOS

# compile any crate with crabbit instead of LLVM:
cargo rustc --manifest-path path/to/crate/Cargo.toml --release -- \
  -Zcodegen-backend="$BACKEND" -Coverflow-checks=off
```

The fixture crates under `crates/backend-tests/fixtures/` (hello-world,
sudoku solver, HashMap, i128, FP, a GPU kernel, ...) are worked examples;
`cargo test -p backend-tests` compiles and *executes* all of them through
the backend and asserts their output.

### Mode 2 — analysis server

```sh
cargo build -p crabbit-inspect-driver
target/debug/crabbit-analysisd --workers 4 --http 127.0.0.1:8177
```

Then over plain HTTP (same JSON over stdio):

```sh
# get a module: any crabbit compile with CRABBIT_EMIT_IR=<dir> writes .plir
curl -d '{"cmd":"load_module","name":"m","text":"<contents of the .plir>"}' 127.0.0.1:8177
curl -d '{"cmd":"start_run","moduleId":1,"target":"aarch64-linux",
          "config":{"CRABBIT_REGALLOC":"eregalloc","CRABBIT_REGALLOC_ORACLE":"c2"}}' 127.0.0.1:8177
curl -d '{"cmd":"run_status","runId":1}' 127.0.0.1:8177      # per-pass progress
curl -d '{"cmd":"run_ir","runId":1,"pass":"aarch64-register-allocate"}' 127.0.0.1:8177
curl -d '{"cmd":"run_artifact","runId":1}' 127.0.0.1:8177    # the ELF object
```

Full command set: `server_health`, `list_targets`, `load_module`,
`start_run`, `run_status`, `cancel_run`, `run_ir`, `run_artifact`,
`run_costs` (profile lifting), and the language queries `ir_hover`,
`ir_definition`, `ir_references`, `ir_diagnostics`. The browser UI lives in
[`pliron-inspect`](https://github.com/montaglue/pliron-inspect)
(`pliron-inspect --server 127.0.0.1:8177`).

## Research knobs

Every policy is an environment variable read at compile time, so A/B needs
no rebuild (`crates/pliron-ll/src/codegen_opts.rs`,
`crates/research-config/`):

| Variable | Values | What it selects |
| --- | --- | --- |
| `CRABBIT_REGALLOC` | `linear`, `eregalloc` | baseline linear scan vs. the e-graph availability-oracle allocator |
| `CRABBIT_REGALLOC_ORACLE` | `c0`, `c2` | syntactic vs. saturated-e-graph restore oracle |
| `CRABBIT_BLOCK_FREQ` | `uniform`, `spectral`, `profile`, `cmt-provider` | spill-weight frequencies: none / analytic Perron–Frobenius / **measured perf profile** |
| `CRABBIT_MIDEND_DISABLE` | `gvn,licm,divmagic,dse,adce,sink` | ablate individual mid-end passes |
| `CRABBIT_PROFILE`, `CRABBIT_MEASURED_COSTS` | paths | measured frequencies / per-decision costs (see backward PGO) |
| `CRABBIT_EMIT_IR`, `CRABBIT_PTX_OUT`, `CRABBIT_LL_OUT` | paths | printed IR / PTX / textual LLVM IR side outputs |
| `CRABBIT_BLOCKMAP`, `CRABBIT_PTX_LINEMAP` | `1` | provenance sidecars for profile attribution |

`scripts/perf-harness/` runs the full config matrix over the fixtures, the
arrow crates, and the CPU kernel corpus, with miscompile gates and spill
metrics.

## Repository tour

| Crate | Kind | Description |
| --- | --- | --- |
| [`crabbit`](crates/crabbit) | `dylib`+`rlib` | The rustc backend: MIR import, pipeline driving, object + PTX + IR emission. |
| [`crabbit-mir`](crates/mir) | lib | The `mir` dialect and its lowering into the `llvm` dialect. |
| [`pliron-ll`](crates/pliron-ll) | lib | The heart: LLVM-dialect mid-end (GVN, LICM, DSE, ADCE, sink, div strength reduction, `TargetProfile`), aarch64/x86_64 machine dialects and pipelines, ELF/Mach-O writers, native NVPTX emission, profile/blockmap infrastructure. |
| [`research-config`](crates/research-config) | lib | Engine/flag parsing shared by the backend and the server (features `eregalloc`, `cmt`). |
| [`crabbit-inspect-driver`](crates/inspect-driver) | bins | `crabbit-inspect-driver` (one-shot pipeline CLI) and `crabbit-analysisd` (resident server). |
| [`backend-tests`](crates/backend-tests) | tests | Builds the dylib and compiles+**runs** fixture crates through it; round-trip, kernel-on-GPU, and server-equivalence gates. |

Design documents live in [`docs/`](docs) — notably
[`ARCHITECTURE.md`](docs/ARCHITECTURE.md),
[`BACKWARD-PGO.md`](docs/BACKWARD-PGO.md), and
[`MIDEND-PLAN.md`](docs/MIDEND-PLAN.md) (each optimization with its measured
effect, regressions included). The sibling `kernel-corpus` repository holds
the 14-kernel CPU/GPU corpus, the three-arm comparison pipeline, and all
result CSVs.

The project was previously developed under the name **STAIR**; a few
internal symbols and trace strings still carry that name.

## Testing

```sh
cargo test --workspace          # ~230 tests
CRABBIT_REGALLOC=eregalloc cargo test -p backend-tests   # fixtures under the research allocator
```

Every fixture executes and is output-checked; the kernel corpus adds
checksum gates for both CPU and GPU artifacts.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
