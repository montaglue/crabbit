# crabbit perf/metrics A/B harness

`harness.py` builds the crabbit backend once (`cargo build -p crabbit`), then
compiles a corpus under each entry of a config matrix (`configs.json`: a list
of `{name, env}` objects — the env vars, e.g. `CRABBIT_SPILL_POLICY`,
`CRABBIT_RESTORE_ESTIMATE`, `CRABBIT_BLOCK_FREQ`, are read at *target-crate*
compile time, so new research knobs are just more keys). Corpus A is the
runnable fixtures under `crates/backend-tests/fixtures/` (compiled `--release`
with the smoke-test recipe, run N times, median+min wall time; stdout is
checked against the expectations lifted from `backend_smoke.rs` — or against
the baseline config's output — and any divergence is flagged as a
**MISCOMPILE** and excluded from perf numbers). Corpus B is compile-only arrow
crates from a local arrow-rs checkout (`--arrow-rs`): each crate's `src/lib.rs`
is touched to force a leaf recompile (cargo does not track env vars), compile
wall time is recorded, and the fresh `.o` files (`-Csave-temps`) are analyzed
with `objdump` for text size, instruction count, and spill-traffic proxies
(sp-relative `ldr`/`str`/`ldp`/`stp` counts, `mov`-immediate counts as a remat
proxy); the same static metrics are collected from fixture objects. Results go
to `results/<timestamp>/` as long-format `results.csv` plus a rendered
`summary.md` with per-config deltas vs the first (baseline) config, also
printed to stdout. Uses `hyperfine` when installed, else a python monotonic
timer. Typical use: `python3 scripts/perf-harness/harness.py` (add
`--skip-backend-build`, `--runs 20`, `--fixtures sudoku-solver`,
`--arrow-crates arrow-schema`, `--skip-arrow` etc. as needed; build scratch
lives outside the repo in `--workdir`, default `~/.cache/crabbit-perf-harness`).


## Engine axis, corpus targets, pinning (2026-08-28)

- `configs.json` now spans the register-allocation *engine* too:
  `CRABBIT_REGALLOC=linear|eregalloc`, `CRABBIT_REGALLOC_ORACLE=c0|c2`, with
  `CRABBIT_BLOCK_FREQ` meaningful under both (see
  `crates/crabbit/src/regalloc_engine.rs`). `configs-engines.json` is the
  three-config engine-only matrix used for compile-time measurements.
- Corpus C: the kernel-corpus CPU variants (`--corpus <path>`, default
  `~/projects/montaglue/kernel-corpus`; `--kernels a,b`; `--skip-corpus`).
  Each `kernels/<k>/cpu` bin runs `--size large` (timed, exact `checksum …`
  stdout from `spec.json` when present, else gated on the baseline config's
  output) and `--size small`; static metrics come from its `.o`.
- `--pin-cpu 7` wraps every timed run in `taskset -c 7` (a Cortex-X925 core
  on the DGX Spark). `--backend <libcrabbit.so>` uses a specific dylib.
- The backend dylib is **snapshotted into the results directory** (sha256
  in `meta.json`), so rebuilding crabbit during a sweep cannot mix backend
  versions into one run. Compile failures keep their full stderr in
  `results/<run>/compile-fail-<config>-<target>.log`.
- arrow-rs lives durably at `~/.cache/crabbit-perf-harness/arrow-rs`.
