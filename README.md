# crabbit

**An LLVM-free Rust compiler backend — pure Rust from MIR to machine code —
that compiles real crates: all four `apache/arrow-rs` core crates build
through it today.**

`crabbit` is a [rustc codegen backend](https://rustc-dev-guide.rust-lang.org/backend/backend-agnostic.html):
`cargo rustc -- -Zcodegen-backend=libcrabbit.so` and your crate is compiled
by ~40 passes of hand-written Rust — MIR imported into
[`pliron`](https://github.com/pliron-org/pliron) (an MLIR-style IR
framework) dialects, then mid-end optimization, instruction selection,
linear-scan register allocation, encoding — down to native **aarch64 ELF**
(and Mach-O) objects. **No LLVM anywhere in the binary path.**

```text
   MIR ──► mir dialect ──► llvm dialect ──► aarch64 dialect ──► ELF/Mach-O
                │  gvn · licm · div-magic         │
                │  dse · adce · sink              └──► (registers, frames,
                │                                       encodings: all ours)
                └────────► native NVPTX ──► PTX ──► runs on real GPUs
```

The same dialect stack emits **native NVPTX**: `#[no_mangle]
__stair_kernel_*` Rust functions become PTX — no LLVM, no NVVM — measured at
geomean **1.13× nvcc** across a 14-kernel corpus, 10/14 within 3%.

And underneath sits the research it was built to carry — **backward PGO**:
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
- **Swappable research register allocators** (an availability-aware e-graph
  oracle among them), composed by the workspace-excluded
  [`crates/crabbit-research`](crates/crabbit-research) dylib, and an A/B
  perf harness — policies compared on real programs, not synthetic corpora.

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

- Rust **nightly-2026-04-03** with `rustc-dev` + `rust-src` (installed
  automatically via `rust-toolchain.toml`).
- A C++ toolchain with libfmt for the CFG-layout bindings the inspect
  driver links (`triskel`): `apt install g++-14 libfmt-dev libffi-dev`
  (or any g++ ≥ 13 with fmt headers on the include path).

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

> **Dependencies:** everything a plain `cargo build` needs resolves from
> crates.io and pinned public git revisions (our
> [cuda-oxide fork](https://github.com/montaglue/cuda-oxide)'s
> `crabbit-patches` branch and
> [pliron-inspect](https://github.com/montaglue/pliron-inspect)) — no
> sibling checkouts. The research *engines* are private and live outside
> this workspace entirely: see **Research build** below.

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
| [`crabbit-core`](crates/crabbit) | rlib | The rustc backend: MIR import, pipeline driving, object + PTX + IR emission. |
| [`crabbit`](crates/crabbit-backend) | `dylib` | Thin loadable wrapper over `crabbit-core`: `cargo build -p crabbit` → `target/debug/libcrabbit.so`. |
| [`crabbit-mir`](crates/mir) | lib | The `mir` dialect and its lowering into the `llvm` dialect. |
| [`pliron-ll`](crates/pliron-ll) | lib | The heart: LLVM-dialect mid-end (GVN, LICM, DSE, ADCE, sink, div strength reduction, `TargetProfile`), aarch64/x86_64 machine dialects and pipelines, ELF/Mach-O writers, native NVPTX emission, profile/blockmap infrastructure. |
| [`research-config`](crates/research-config) | lib | Engine/flag parsing + the engine registry, shared by the backend and the server. |
| [`crabbit-research`](crates/crabbit-research) | `dylib`, excluded | The backend + the private research engines (eregalloc, CMT) in one `libcrabbit_research.so`; needs the sibling research checkouts. |
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
# fixtures under the research allocator (research build, see below):
CRABBIT_TEST_BACKEND=$PWD/crates/crabbit-research/target/debug/libcrabbit_research.so \
  CRABBIT_REGALLOC=eregalloc cargo test -p backend-tests --test backend_smoke
```

### Research build

The research engines live in private repositories, composed in by the
workspace-excluded `crates/crabbit-research` (path-deps on
`../eregalloc` and `../combinatorial-matrix-theory` checkouts):

```sh
cd crates/crabbit-research && cargo build   # -> target/debug/libcrabbit_research.so
```

Point `-Zcodegen-backend` (or the perf harness, which prefers it
automatically) at that dylib and the `CRABBIT_REGALLOC=eregalloc` /
`CRABBIT_BLOCK_FREQ=cmt-provider` axes come alive; the public
`libcrabbit.so` reports them as not linked.

Every fixture executes and is output-checked; the kernel corpus adds
checksum gates for both CPU and GPU artifacts.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
