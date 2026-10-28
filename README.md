# crabbit

[![CI](https://github.com/montaglue/crabbit/actions/workflows/ci.yml/badge.svg)](https://github.com/montaglue/crabbit/actions/workflows/ci.yml)
[![docs](https://github.com/montaglue/crabbit/actions/workflows/docs.yml/badge.svg)](https://github.com/montaglue/crabbit/actions/workflows/docs.yml)
[![coverage](https://codecov.io/gh/montaglue/crabbit/graph/badge.svg)](https://codecov.io/gh/montaglue/crabbit)
[![dependencies](https://deps.rs/repo/github/montaglue/crabbit/status.svg)](https://deps.rs/repo/github/montaglue/crabbit)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![rustc](https://img.shields.io/badge/rustc-1.99.1-orange.svg)](rust-toolchain.toml)
[![last commit](https://img.shields.io/github/last-commit/montaglue/crabbit)](https://github.com/montaglue/crabbit/commits/main)

**An LLVM-free Rust compiler backend — pure Rust from MIR to machine code —
that compiles real crates: all four `apache/arrow-rs` core crates build
through it today.**

`crabbit` is a [rustc codegen backend](https://rustc-dev-guide.rust-lang.org/backend/backend-agnostic.html):
`cargo rustc -- -Zcodegen-backend=libcrabbit.so` and your crate is compiled
by ~45 passes of hand-written Rust — MIR imported into
[`pliron`](https://github.com/pliron-org/pliron) (an MLIR-style IR
framework) dialects, then mid-end optimization, instruction selection,
linear-scan register allocation, encoding — down to native **aarch64 ELF**
(and Mach-O) objects. **No LLVM anywhere in the binary path.**

```text
   host: MIR ──► mir dialect (our importer) ──► llvm dialect ──► aarch64 dialect ──► ELF/Mach-O
                                 │  gvn · licm · div-magic · unroll │
                                 │  dse · adce · sink · vectorize   └──► (registers incl. NEON,
                                 │                                        frames, encodings: all ours)
 device: MIR ──► mir dialect (our importer; cuda_device calls ──► nvvm ops) ──► llvm dialect
                                 └────────► native NVPTX ──► PTX ──► runs on real GPUs
```

Kernels are written in [cuda-oxide](https://github.com/NVlabs/cuda-oxide)'s
own device conventions (`#[kernel]`, `cuda_device::{thread, SharedArray,
warp, atomic}`); crabbit's importer translates them (the `cuda_device` API
becomes `nvvm` dialect ops) and NVIDIA's `mir-lower` takes them to the
LLVM dialect (input verification on), then crabbit's mid-end and NVPTX
translator finish the job — `docs/KERNEL-ABI.md`. A crate can hold its
kernels and the host program that launches them through cuda-oxide's host
API (`#[cuda_module]`, `kernels::load`, the typed launch methods on
`cuda-core` buffers): crabbit compiles both sides in one compilation and
embeds the PTX in the binary the way cuda-oxide's own backend does.

![A compiled function's control-flow graph in pliron-inspect's CFG view:
per-block IR, laid out and browsable per pass](docs/images/inspect-cfg.png)

![mem2reg before/after in the IR-diff view: allocas and stores deleted
(red), SSA values taking their place (green)](docs/images/inspect-ir-diff.png)

The same dialect stack emits **native NVPTX**: `#[kernel]` Rust functions
(cuda-oxide's device API) become PTX — no LLVM, no NVVM — measured at
geomean **1.01× nvcc** across a 14-kernel corpus (dead tie on tiled gemm at
35 vs 35 registers), and racing 17 production kernels sourced from real
codebases at parity, ahead of nvcc on 6 of them.

And underneath sits what it was built to carry — **backward PGO**:
every emitted instruction carries provenance through every lowering level,
so `perf` samples and Nsight stalls flow *backward through the same pipeline
the program flowed forward through*, each pass interpreting measured cost
across its own transformation, all the way to the source op — and the
optimization decision — that produced the code. We believe this framing of
profile-guided optimization is new. → [docs/BACKWARD-PGO.md](docs/BACKWARD-PGO.md)

Around the compiler:

- **`crabbit-analysisd`** — a resident analysis server: load printed IR, run
  the pipeline under different configurations on a worker pool, fetch the IR
  after any pass, ask hover/definition/references questions about it — over
  stdio or loopback HTTP, with a web UI in the sibling
  [`pliron-inspect`](https://github.com/montaglue/pliron-inspect) project.
- **Extensible pipelines** — every pass has a name and every pipeline stage
  is an order of names; a backend built with extensions registers its own
  passes and edits crabbit's orders: insert, remove, reorder or replace any
  pass, crabbit's own included ([docs/EXTENDING.md](docs/EXTENDING.md)).
- **An A/B perf harness** — policies compared on real programs, not
  synthetic corpora.

> **Status: experimental.** Correctness is enforced by an executable
> fixture suite and a 14-kernel CPU/GPU corpus with checksum gates, not by a
> stability guarantee. Expect gaps against arbitrary Rust input.

## Highlights (measured, reproducible in-repo)

- **Real crates compile.** All four `apache/arrow-rs` core crates
  (`arrow-schema`, `arrow-buffer`, `arrow-data`, `arrow-array`) compile
  through crabbit (`scripts/perf-harness/` drives this; corpus B).
- **GPU code at parity with nvcc.** On a 14-kernel corpus (each kernel as
  Rust-via-crabbit, Rust-via-LLVM from *identical IR*, and CUDA C++ via
  nvcc), crabbit's native PTX is geomean **1.01× nvcc**, **12/14 kernels
  within 2%**, tiled gemm a dead tie at 35 vs 35 registers, zero
  correctness failures
  (results: `kernel-corpus/results/gpu/20260920-final/`).
- **Real production kernels, sourced not written.** A sibling corpus of
  **17 kernels extracted verbatim** from real codebases (an MoE inference
  engine, llama.cpp, llm.c — dequantization, attention pooling, warp-level
  reductions, Mamba decode) runs correct through crabbit; crabbit is
  **ahead of nvcc on 6** (bf16-decode attention 0.77×, argsort 0.90×,
  q4_0 dequant 0.98×), at parity on most, and every remaining loss carries
  a SASS-level mechanism analysis. Porting it surfaced and fixed **three
  real miscompiles**, including a latent scalar-transmute bug on both
  targets (now fixture-guarded).
- **Optimizations that pay measurably.** Adding constant-divisor strength
  reduction took the `seidel_2d` GPU kernel from **3.93 ms → 2.06 ms**;
  shared-memory addressing + 128-bit shared loads took tiled gemm from
  1.49× nvcc to a tie; full unrolling of geometric-induction warp
  reductions turned two rms-norm kernels from losses into wins; a NEON
  loop vectorizer took the worst CPU kernel **4.5× faster**; a
  target-profile-aware sinking rule (don't add control dependence under
  SIMT divergence) recovered a measured 10% regression on `gemm_tiled`
  (`docs/MIDEND-PLAN.md` records each finding with its mechanism).
- **Attribution that survives audits.** Synthetic samples injected on a
  machine-instruction range provably lift to the correct source-level op —
  including a 50/50 split through a GVN merge
  (`crates/pliron-ll/tests/opmap_audit.rs`). On real hardware, 72% of
  `seidel_2d`'s GPU samples attribute to one source op (the consumer of the
  recurrence load), matching the PTX analysis
  (results: `kernel-corpus/results/gpu-attribution/20260831-090227/ANALYSIS.md`).
- **Honest experiments, nulls included.** Analytic (spectral) block
  frequencies under a uniform prior vs. measured: median Spearman
  **0.002** — the uniform walk cannot represent iteration counts; a
  loop-aware branch prior lifts it to **0.742** (better on 14/14 kernels)
  (`kernel-corpus/results/cpu-feedback/20260831/FREQ-MODEL-EVAL.md`).
  With that and a repaired cost model, measured profile frequencies became
  the best-performing linear-allocator configuration — winning exactly
  where the analytic model mis-predicts
  (`kernel-corpus/results/cpu-feedback/20260920-pgo/`).

![arrow-schema IR at the llvm-gvn boundary in the pliron-inspect UI: the
24 MB module compiled by crabbit, replayed per pass, browsable with
position→op hover/definition/references](docs/images/analysisd-arrow-gvn-ir.png)

![arrow-schema running through the 43-pass pipeline on crabbit-analysisd,
live progress in the UI](docs/images/analysisd-arrow-run.png)

Real measured attribution, end to end (perf samples on the running
binary, lifted through the machine and mid-end adjoints to source-level
op ids — `elementwise_chain`, Cortex-X925, 2 kHz cycle sampling):

```text
$ python3 scripts/perf-harness/profile_ingest.py --perf-script samples.txt \
    --blockmap elementwise_chain.blockmap.json -o profile.json
profile_ingest: 8948 samples attributed across 1 functions
$ jq '.["..._elementwise_chain_cpu6kernel"].source' profile.op_costs.json \
    | sort -rn -k2 | head -5        # top source ops by measured cycles
"200": 1888        # the hot chain's multiply
"174": 1055
"208": 1043
"186": 821
"166": 796
```

(The UI's heat-map rendering over `run_costs` is tracked follow-up work;
the server command and the data path above are live today.)

## Prerequisites

- Rust **1.99.1** (the stable release cuda-oxide pins) with `rustc-dev` +
  `rust-src`, installed automatically via `rust-toolchain.toml`;
  `.cargo/config.toml` sets `RUSTC_BOOTSTRAP=1` so `#![feature(rustc_private)]`
  and `-Zcodegen-backend` work on the stable release, exactly as upstream
  cuda-oxide builds.
- A C++ toolchain for the CFG-layout bindings the inspect driver links
  (`triskel`): g++ ≥ 13 (C++23), libffi-dev, and **fmt ≥ 10** — distro
  packages are often v9; build fmt 11 into /usr/local (the CI workflow
  shows the exact recipe).

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
(stable 1.99.1 + `rustc-dev`, with `RUSTC_BOOTSTRAP=1` from
`.cargo/config.toml`); rustup picks it up automatically.

> **Dependencies:** everything a plain `cargo build` needs resolves from
> crates.io and pinned public git revisions — no sibling checkouts. The
> `mir` dialect, its verifier and the `mir` → LLVM-dialect lowering are
> NVIDIA's [cuda-oxide](https://github.com/NVlabs/cuda-oxide) crates
> (`dialect-mir`, `mir-lower`, `llvm-export`), consumed unpatched from the
> upstream repository at one pinned revision; crabbit's own front end
> (`crates/crabbit/src/importer/`) produces the dialect and must satisfy
> its verifier, which `mir-lower` runs before lowering anything (see
> `docs/MIR-DIALECT.md`). The analysis tooling comes from
> [pliron-inspect](https://github.com/montaglue/pliron-inspect).

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
          "config":{"CRABBIT_SPILL_POLICY":"weighted","CRABBIT_BLOCK_FREQ":"spectral-loop"}}' 127.0.0.1:8177
curl -d '{"cmd":"run_status","runId":1}' 127.0.0.1:8177      # per-pass progress
curl -d '{"cmd":"run_ir","runId":1,"pass":"aarch64-register-allocate"}' 127.0.0.1:8177
curl -d '{"cmd":"run_artifact","runId":1}' 127.0.0.1:8177    # the ELF object
```

Full command set: `server_health`, `list_targets`, `load_module`,
`start_run`, `run_status`, `cancel_run`, `run_ir`, `run_artifact`,
`run_costs` (profile lifting), and the language queries `ir_hover`,
`ir_definition`, `ir_references`, `ir_diagnostics`.

The browser UI, [`pliron-inspect`](https://github.com/montaglue/pliron-inspect),
opens this workspace as a project: start `pliron-inspect`, open
`/analysis`, enter this directory as the project root. It builds and runs
`crabbit-analysisd` and `crabbit-inspect-driver` from here (named in
`[workspace.metadata.pliron-inspect]`), lists the traces and emitted
modules under `.crabbit-traces/` (`CRABBIT_TRACE=1` and `CRABBIT_EMIT_IR=1`
write there), and gets hover, definition into the dialects' Rust source,
references and diagnostics from a pliron-lsp engine built from this
workspace's dialect crates. `pliron-inspect --server 127.0.0.1:8177`
still works against a hand-started analysisd.

## Configuration knobs

Every policy is an environment variable read at compile time, so A/B needs
no rebuild (`crates/pliron-ll/src/codegen_opts.rs` and the passes
themselves):

| Variable | Values | What it selects |
| --- | --- | --- |
| `CRABBIT_SPILL_POLICY` | `furthest`, `weighted` | linear-scan victim choice: Poletto–Sarkar furthest use / cheapest frequency-weighted restore |
| `CRABBIT_RESTORE_ESTIMATE` | `reload`, `remat` | restores as reloads / constant defs rematerialized |
| `CRABBIT_BLOCK_FREQ` | `uniform`, `spectral`, `spectral-loop`, `profile` | spill-weight frequencies: none / analytic Perron–Frobenius (optionally with the loop-aware 7/8 branch prior) / **measured perf profile** |
| `CRABBIT_MIDEND_DISABLE` | `gvn,licm,divmagic,unroll,lcfwd,dse,adce,sink,vectorize` | ablate individual mid-end passes |
| `CRABBIT_MIR_CENSUS` | path | append a per-module JSON census of the imported `mir` dialect (op kinds, callee families, allocas, aggregates, loops) to that file, before lowering (`docs/MIR-CENSUS.md`) |
| `CRABBIT_MIR_DISABLE` | `census` | skip individual `mir`-level passes (the stage between import and `lower-dialect-mir`) |
| `CRABBIT_PROFILE` | path | measured block frequencies (see backward PGO) |
| `CRABBIT_EMIT_IR`, `CRABBIT_PTX_OUT`, `CRABBIT_LL_OUT` | paths | printed IR / PTX / textual LLVM IR side outputs (`CRABBIT_EMIT_IR=1`: under the project's `.crabbit-traces/ir/`) |
| `CRABBIT_TRACE`, `CRABBIT_TRACE_DIR` | `1`, path | per-pass IR traces (`.crt`), by default under the compiled project's `.crabbit-traces/` |
| `CRABBIT_BLOCKMAP`, `CRABBIT_PTX_LINEMAP` | `1` | provenance sidecars for profile attribution |

`scripts/perf-harness/` runs the full config matrix over the fixtures, the
arrow crates, and the CPU kernel corpus, with miscompile gates and spill
metrics.

## Repository tour

| Crate | Kind | Description |
| --- | --- | --- |
| [`crabbit-core`](crates/crabbit) | rlib | The rustc backend: MIR import, pipeline driving, object + PTX + IR emission. |
| [`crabbit`](crates/crabbit-backend) | `dylib` | Thin loadable wrapper over `crabbit-core`: `cargo build -p crabbit` → `target/debug/libcrabbit.so`. |
| [`crabbit-mir`](crates/mir) | lib | crabbit's passes on NVIDIA's `mir` dialect and around its lowering to the `llvm` dialect (census, linkage, function addresses). |
| [`crabbit-config`](crates/config) | lib | Compilation configuration (host and device targets) and the pipeline API: passes by name, stages as editable orders ([docs/EXTENDING.md](docs/EXTENDING.md)). |
| [`pliron-ll`](crates/pliron-ll) | lib | The heart: LLVM-dialect mid-end (GVN, LICM, DSE, ADCE, sink, div strength reduction, `TargetProfile`), aarch64/x86_64 machine dialects and pipelines, ELF/Mach-O writers, native NVPTX emission, profile/blockmap infrastructure. |
| [`crabbit-inspect-driver`](crates/inspect-driver) | bins | `crabbit-inspect-driver` (one-shot pipeline CLI) and `crabbit-analysisd` (resident server). |
| [`backend-tests`](crates/backend-tests) | tests | Builds the dylib and compiles+**runs** fixture crates through it; round-trip, kernel-on-GPU, and server-equivalence gates. |

Design documents live in [`docs/`](docs) — notably
[`ARCHITECTURE.md`](docs/ARCHITECTURE.md),
[`BACKWARD-PGO.md`](docs/BACKWARD-PGO.md), and
[`MIDEND-PLAN.md`](docs/MIDEND-PLAN.md) (each optimization with its measured
effect, regressions included). The sibling `kernel-corpus` repository holds
the 14-kernel CPU/GPU corpus, the three-arm comparison pipeline, and all
result CSVs.

## Testing

```sh
cargo test --workspace          # ~330 tests
# the fixtures under another backend dylib (e.g. one built with extensions):
CRABBIT_TEST_BACKEND=/path/to/backend.so cargo test -p backend-tests --test backend_smoke
```

Every fixture executes and is output-checked; the kernel corpus adds
checksum gates for both CPU and GPU artifacts.

## Extending crabbit

Every pass has a name and every pipeline stage — `mir` to LLVM dialect,
mid-end, machine, and the program stage that sees a crate's host and
kernel modules together — is an order of names with a default. A backend built
with extensions is a small dylib crate that links crabbit's library,
registers its passes (`crabbit_config::register_pass`) and one order
function (`crabbit_config::register_order`, which receives crabbit's
default order and may insert, remove, reorder or replace passes), then
returns crabbit's backend; rustc loads it with `-Zcodegen-backend`.
[docs/EXTENDING.md](docs/EXTENDING.md) has the complete recipe.

## Engines

Three closed-source engines are built on that API. Gains are against
crabbit's own default pipeline (linear-scan allocation), measured on the
14-kernel CPU corpus (Cortex-X925, pinned core, 2026-09-20):

| Engine | Direction | Gain |
| --- | --- | --- |
| Register allocation as e-graph materialization | spilling decided by an availability-aware oracle over each value's derivations: a restore is the cheapest of a reload or a recomputation | **1.6% faster** geomean with the loop-aware frequency prior below (histogram −14%, transpose −9%, worst dot_product +4%); 1.9% slower without it |
| Analytical models (combinatorial matrix theory) | spectral block frequencies with a loop-aware branch prior; max-plus scheduling bounds; unimodular loop transforms | the frequency prior inside the allocator above: **3.4% faster** geomean (gemm_tiled −25%); its frequencies' rank correlation with measured profiles 0.002 → 0.742 |
| Kernel fusion | fusing a program's back-to-back launches of its own kernels into one launch, kernels and host launch code rewritten together, with legality proven on affine access maps | on three real host programs (MoE decode tails of an inference engine) launches cut 2 → 1, 2 → 1 and 3 → 2 with outputs bit-identical: **2.0×** and **1.33×** faster per step, 1.03× on the third (whose step is mostly host routing) |

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
