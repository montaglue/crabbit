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

## Profile feedback: measured block frequencies (2026-08-31)

`profile_ingest.py` closes the perf → codegen loop of
`docs/PROFILE-FEEDBACK-PLAN.md`: measured per-block execution frequencies
become a third `CRABBIT_BLOCK_FREQ` source next to `uniform` and
`spectral`, under both allocator engines.

Workflow (all offsets/ids are RA-time region order, stable across builds
with identical compiler/flags/sources — the profile build must use the
same config as the rebuild except the freq source):

```sh
# 1. profile build: same config as the final build, plus the blockmap
CRABBIT_BLOCKMAP=1 CARGO_TARGET_DIR=target-profile \
  cargo rustc -p elementwise-chain-cpu --release -- \
    -Zcodegen-backend=.../libcrabbit.so -Coverflow-checks=off
# -> next to each object: <object stem>.blockmap.json
#    {symbol: [{id, start, end}, ...]} in final .text offsets;
#    id -1 marks post-RA blocks (their samples are dropped)

# 2. record + fold (repeat for as many runs as wanted)
perf record -e cycles:u -o perf.data -- ./elementwise-chain-cpu --size large
perf script -i perf.data -F ip,sym,symoff --no-demangle > samples.txt

# 3. ingest (multiple --perf-script runs merge additively;
#    multiple --blockmap files cover multi-object programs)
python3 scripts/perf-harness/profile_ingest.py \
  --blockmap path/to/foo.blockmap.json \
  --perf-script samples.txt -o profile.json

# 4. rebuild with measured frequencies (either engine)
CRABBIT_BLOCK_FREQ=profile CRABBIT_PROFILE=$PWD/profile.json \
  CRABBIT_SPILL_POLICY=weighted cargo rustc ...              # linear
CRABBIT_REGALLOC=eregalloc CRABBIT_BLOCK_FREQ=profile \
  CRABBIT_PROFILE=$PWD/profile.json cargo rustc ...          # eregalloc
```

- `profile.json` is `{symbol: [freq per RA-order block]}`, normalized like
  the spectral model: `freq[b] = samples[b]/samples[entry]`, entry forced
  ≥ 1 sample, unsampled blocks 0.0 (entry is always 1.0).
- Robustness contract: a stale/missing profile, an absent symbol, or a
  wrong-length vector silently falls back to uniform for that function —
  never an error. The linear engine consumes frequencies only through
  `CRABBIT_SPILL_POLICY=weighted` (a one-time note is printed otherwise).
- Fallback for perf builds without sym/symoff formatting:
  `--nm-binary <bin>` (+ `--load-bias`) resolves bare-ip lines via `nm`;
  only valid when recorded addresses match link addresses plus the bias
  (non-PIE binaries).
- Requires `kernel.perf_event_paranoid` ≤ 2 (e.g.
  `sudo sysctl kernel.perf_event_paranoid=1`); check with
  `perf record -e cycles:u -- true`.
- Unit tests: `python3 scripts/perf-harness/test_profile_ingest.py`
  (fixtures in `fixtures/`), also run by `cargo test -p backend-tests`
  (`tests/profile_ingest.rs`).

- Cargo features (2026-08-31): the research engines are feature-gated —
  `eregalloc` and `cmt`, both ON by default in `research-config`, `crabbit`
  and `crabbit-inspect-driver`, so the harness needs nothing. A backend
  built `--no-default-features` answers `CRABBIT_REGALLOC=eregalloc` /
  `CRABBIT_BLOCK_FREQ=cmt-provider` with "crabbit was built without the
  `<feature>` feature".

## Mid-end backward-dataflow round (2026-08-31)

Three more gated mid-end passes: `llvm-global-dse`, `llvm-adce`,
`llvm-sink` (`CRABBIT_MIDEND_DISABLE=dse,adce,sink`). **`sink` is a
recorded experiment axis**: it moves pure ops toward their uses and so
changes register pressure — the RA experiments' independent variable.
Every sweep's meta.json should note whether `sink` was disabled; "policy
ranking with vs without sinking" is itself an experiment worth running.

## Backward attribution round 2 (2026-08-31)

- The mid-end now stamps source-level op ids (`llvm-op-ids` at the head of
  the shared mid-end) and every merging/creating pass declares its adjoint
  (GVN merges become multi-parent `derived_from`; inlined ops carry
  `ll.inlined_from` = the call-site id). The blockmap sidecar gains a
  top-level `__midend__` table (fresh RA-boundary id → source parents);
  `profile_ingest.py` emits a third `source` level in `op_costs.json`
  (equal-weight splits for merges).
- `CRABBIT_MEASURED_COSTS=<op_costs.json>` (under
  `CRABBIT_REGALLOC=eregalloc`): per-function measured restore costs from
  the `lifted` level replace the oracle's estimates (experiment E1's
  knob). Missing files/symbols fall back to estimates, never error.
- The analysis server answers `run_costs` (heat-map join of an op_costs
  payload with the ops at the `source`/`ra` boundary, plus per-root
  semantic accounting).
