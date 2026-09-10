# Agent brief: kernel corpus + research-engine experiments on real kernels

You are working in ~/projects/montaglue on a DGX Spark (aarch64-unknown-linux-gnu,
20 ARM cores: 10× Cortex-X925 perf @3.9GHz cpus 5-9,15-19 + 10× A725, NVIDIA GB10
GPU sm_121, CUDA 13.0 at /usr/local/cuda). Toolchain everywhere:
RUSTUP_TOOLCHAIN=nightly-2026-04-03. Do not commit anywhere unless the user asks.

## The repos and their state (all verified green)

- **crabbit** (~/projects/montaglue/crabbit): rustc codegen backend (`-Zcodegen-backend`,
  dylib at target/debug/libcrabbit.so) lowering MIR → cuda-oxide `dialect-mir` →
  pliron-llvm dialect → hand-written aarch64 backend → ELF. pliron 0.17.0 (crates.io).
  ~168 tests. All four apache/arrow-rs crates (schema/buffer/data/array) COMPILE
  through it. Native NVPTX kernel emission exists: `crates/pliron-ll/src/nvptx/`
  translates llvm-dialect funcs directly to PTX text (no LLVM/NVVM), verified by
  ptxas AND executed correctly on the GB10 — but it is NOT yet wired into
  `emit_object` (`crates/crabbit/src/lib.rs` still hard-errors when `#[kernel]`
  functions exist; the imported `rust_kernels` module is populated but unconsumed).
  Env-driven research options exist in `crates/pliron-ll/src/codegen_opts.rs`
  (CRABBIT_SPILL_POLICY / CRABBIT_RESTORE_ESTIMATE / CRABBIT_BLOCK_FREQ), read at
  register-allocation time. Per-pass IR tracing is opt-in via CRABBIT_TRACE.
  A/B perf harness: `scripts/perf-harness/` (python; config matrix over env vars,
  fixture runtimes with miscompile gates, objdump spill metrics on arrow objects).
- **cuda-oxide** (~/projects/montaglue/cuda-oxide, branch `crabbit-patches`):
  local clone of NVlabs/cuda-oxide, rev b099f64c + 2 committed patches (linkage
  propagation; pliron 0.17 port). crabbit consumes `dialect-mir`/`mir-lower` by
  path. Also contains `llvm-export` (pure-Rust textual .ll exporter for the
  llvm dialect — the bridge to the LLVM toolchain), `dialect-nvvm`,
  `cuda-intrinsics` (catalog in intrinsics/catalog.json; device intrinsics are
  symbols like `int_nvvm_read_ptx_sreg_tid_x`), `rustc-codegen-cuda` (their own
  backend), `cuda-host`/`cuda-device` (runtime API), `cargo-oxide` (build driver).
- **eregalloc** (~/projects/montaglue/eregalloc): now a workspace, 43 tests green.
  `eregalloc-passes` crate exports `EregallocRegisterAllocatePass::new(OracleKind
  {Syntactic|EGraph}, BlockFreqSource {Uniform|Spectral|Provider(fn)})`
  implementing crabbit's pass contract (same postconditions as the baseline
  aarch64 allocator). **`INTEGRATION-CONTRACT.md` at the repo root specifies the
  exact crabbit wiring** — read it first.
- **combinatorial-matrix-theory** (~/projects/montaglue/combinatorial-matrix-theory,
  workspace at compilers/): 178 tests green. `spectral-cfg` exports
  `block_frequencies(ctx, region) -> Vec<f64>` (dialect-agnostic, proven).
  `results-spark/` holds a completed M3-vs-Spark replication campaign
  (RESULTS-SPARK.md) — the throughput objective transfers across machines; the
  cache-model phase terms do not. Its kernel-bench dev corpus (seidel-2d,
  hw-kernel-light/heavy, gemm) defines kernel families worth reusing.

## End goal (the user's words, paraphrased)

Run ALL experiments through crabbit's infrastructure — CMT and eregalloc today,
GPU-kernel experiments next. Immediate objective: test both research projects on
**real Rust-written kernels**, and compare against **existing CUDA C++ kernels via
the llvm-export bridge**, so Rust and C++ implementations of the same kernels are
comparable — building a corpus large enough for solid analysis.

## Phase 1 — Wire the research engines into crabbit

Follow eregalloc/INTEGRATION-CONTRACT.md exactly: add path deps
(eregalloc-passes, cmt spectral-cfg with `package=` renames as specified), add the
`pipeline_with_allocator` hook in `crates/pliron-ll/src/passes/aarch64/mod.rs`,
put the `CRABBIT_REGALLOC=linear|eregalloc` + `CRABBIT_REGALLOC_ORACLE=c0|c2`
branch in the `crabbit` root crate (pliron-ll cannot dep on eregalloc-passes —
cycle), route CRABBIT_BLOCK_FREQ=spectral through the Provider hook (mind the
contract's fallthrough caveat). Acceptance: full crabbit test suite green under
BOTH engines (run backend-tests fixtures with CRABBIT_REGALLOC=eregalloc — every
fixture must still execute correctly; any output difference is a miscompile,
report loudly); harness matrix gains the engine axis; arrow crates compile under
the eregalloc engine (compile time may be worse — measure and report, their docs
warn ~3.2ms/block for C2).

## Phase 2 — Finish the GPU kernel path in crabbit

Consume the `rust_kernels` module in `emit_object`: run the same LLVM-level
mid-end pass pipeline over it (inline/simplify/sroa/mem2reg — see `pipeline()`),
then `pliron_ll::nvptx::write_ptx_from_ir` (PtxTarget default is sm_121/ISA 8.8),
and write the `.ptx` next to the object (embedding + launch shim is a later
phase; a sidecar file plus a documented naming convention is enough now).
Remove the "does not support kernels yet" hard error and the "direct host calls
to #[kernel]" limitation only as far as tests prove. The nvptx translator's
intrinsic map normalizes callee names (dots→underscores, strips llvm_/int_
prefixes) and covers the sreg family + barrier0 — extend it as corpus kernels
demand (shared memory, more barriers, math intrinsics), erroring precisely on
what's unsupported. How `#[kernel]` functions reach the importer: kernels are
detected by symbol prefix `__crabbit_kernel_` (see KERNEL_EXPORT_PREFIX in
importer_oxide.rs) — check how cuda-oxide's `#[kernel]` macro + cuda-device
intrinsics surface in MIR and make the corpus kernels use whatever attribute/
prefix actually round-trips; if cuda-oxide's macros don't fit crabbit's current
importer, a minimal `#[no_mangle] pub extern "C" fn __crabbit_kernel_*` convention
with a tiny device-intrinsics shim crate is acceptable for the corpus — document
the choice. Acceptance: at least one Rust kernel compiled BY CRABBIT from a
.rs file (not hand-built IR) into PTX, loaded via the CUDA driver API, runs
correctly on the GB10 (there is a prior harness pattern: cuCtxCreate takes 4
args on CUDA 13).

## Phase 3 — The kernel corpus

Create a sibling repo directory ~/projects/montaglue/kernel-corpus. Per kernel,
four artifacts: (1) Rust device kernel (for crabbit's PTX path), (2) Rust scalar
CPU variant (same loop nest, for the CPU RA experiments), (3) CUDA C++ reference
(.cu, same semantics), (4) a spec: input generator, expected-output checksum,
launch geometry, workload sizes (small = L1/L2-resident, large = memory-bound).
Start set (~12, deliberately overlapping CMT's families so CPU and GPU
experiments share shapes): vector-add/saxpy, dot-product reduction, prefix-sum,
histogram, matrix transpose, tiled gemm (+ the gemm-control shape from CMT),
jacobi-2d and seidel-2d stencils, rms-norm (crabbit has a llama-rms-norm fixture
to mine), softmax, an elementwise-chain (fusion-shaped), and one
division-recurrence kernel (CMT's hw-kernel2-heavy shape — the MII-bound case).
Integer-first where possible; FP kernels are fine on CPU (f32/f64 supported) but
note the nvptx translator is INTEGER-ONLY today — extending it to f32/f64
(ld/st/arith/cvt on .f32/.f64 regs) is in scope for the kernels that need it and
is straightforward given PTX has typed virtual registers.

## Phase 4 — The comparison pipeline (three arms per kernel)

a. **rust-crabbit**: crabbit → native PTX → ptxas → run.
b. **rust-llvm**: the SAME Rust kernel through the LLVM toolchain via the
   llvm-export bridge. Two candidate routes — investigate and pick what works,
   documenting why: (i) cuda-oxide's own pipeline (cargo-oxide /
   rustc-codegen-cuda → NVVM → PTX) if it runs on this machine; (ii) crabbit's
   llvm-dialect kernel module exported as textual .ll via cuda-oxide's
   `llvm-export`, then compiled with LLVM (llc -march=nvptx64 or clang) to PTX.
   Route (ii) is the purest apples-to-apples (identical front half, different
   back half) — prefer it if the .ll round-trips.
c. **cuda-cpp**: the C++ reference via nvcc (and optionally clang++ --cuda) — the
   "existing CUDA kernels" baseline.
Measure per arm: correctness (checksums — a mismatch is a loud finding, never a
silent skip), kernel wall time via CUDA events (median of N≥20 launches after
warmup, geometry per spec), and the static regalloc-relevant numbers from
`ptxas -v`: registers per thread, spill stores/loads bytes, shared mem,
occupancy (these are the GPU-side analogue of the CPU spill metrics — report
them side by side per kernel).

## Phase 5 — CPU-side experiments on the same corpus (and the full sweep re-run)

IMPORTANT — inherited state: the FIRST harness sweep was deliberately aborted
mid-run. `scripts/perf-harness/results/20260828-full/` holds PARTIAL results
from a shakedown run whose backend was rebuilt mid-flight (mixed backend
versions) — treat it as a harness smoke test, never as data. What it did
establish: the harness works end-to-end, the correctness gate passed on all 11
fixtures × 4 configs, and the static metrics show the intended remat signature
(ldr_sp −37.5% / mov_imm +10% under weighted+remat on pure-rust-aarch64). Your
job here includes the CLEAN re-run: after Phases 1–2 land and the backend is
final, re-run the ENTIRE sweep from scratch (fixtures + arrow crates + the
corpus CPU variants) into a fresh results directory.

Extend scripts/perf-harness to take the corpus's CPU variants as targets and run
the full matrix: CRABBIT_REGALLOC {linear, eregalloc} × ORACLE {c0, c2} ×
CRABBIT_BLOCK_FREQ {uniform, spectral} × (legacy CRABBIT_SPILL_POLICY axes for
the linear engine). Runtime medians pinned to a Cortex-X925 core (taskset -c 7),
plus objdump spill-traffic and code-size metrics. This is where eregalloc and CMT
actually bind today — the kernels give them register-pressure-rich, loop-shaped
real code, which the arrow crates (compile-only) cannot.

## Phase 6 — Analysis deliverable

One report (in kernel-corpus/ANALYSIS.md + CSVs): per kernel × arm × config
tables; the questions to answer: (1) does the eregalloc engine (c0 vs c2 vs
linear baseline) change executed spill traffic / runtime on real kernels, and
where; (2) does spectral frequency change decisions; (3) rust-crabbit vs
rust-llvm PTX gap (registers, spills, runtime) — how far is the native backend
from LLVM; (4) rust vs C++ gap per kernel; (5) which kernel families
discriminate between policies (pressure-rich vs memory-bound). Honesty rules:
report ties and losses as prominently as wins; flag any result that depends on a
single kernel; note compile-time costs per engine.

## Working style

Phases are checkpoints: build + full test suite green after each, report interim
results per phase (especially Phase 1's dual-engine fixture verification and
Phase 2's first real kernel-on-GPU), and stop-and-report rather than hack around
anything that looks like a design decision (e.g. kernel ABI conventions,
llvm-export round-trip failures). Long compiles are normal (arrow-array ~10min).
The scratchpad for this session may be cleaned — keep everything durable in the
repos. Nothing in any repo may be committed without the user's say-so.
