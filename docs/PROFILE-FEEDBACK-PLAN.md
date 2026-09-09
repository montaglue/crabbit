# Profile feedback substrate: perf traces → codegen decisions

NOTE: this document describes the SUBSTRATE (sampling, blockmap, block
frequencies). The actual v1 design — backward interpretation through the
dialect stack — is docs/PROFILE-FEEDBACK-BACKWARD.md; the "v1"/"v1.5"
labels below are historical.

User's research idea (2026-08-31): take perf traces of the running program
and feed the measured metrics back into optimization. (Not backward
dataflow — feedback across runs; the consuming passes are ordinary.)

Why it fits: `CRABBIT_BLOCK_FREQ` is already an axis with `uniform` (no
information) and `spectral` (CMT's analytic Perron–Frobenius model).
Measured frequencies add the ground truth as a third source, giving the
experiment: how close is the analytic model to reality, and do the
allocators decide differently when told the truth?

## v1 scope: measured block frequencies

1. **Stable block IDs + block map.** At the RA pipeline position (where
   frequencies are consumed) each machine block gets a `blockmap_id`
   attribute: function symbol + block index in RA-time region order.
   Block placement / branch relaxation / encode preserve the attribute.
   The ELF writer emits a sidecar `<object stem>.blockmap.json`:
   `{symbol: [{id, start, end}, …]}` with final .text offsets (a block
   split/merged after RA maps every resulting range to its source id;
   blocks created post-RA get id -1 and their samples are dropped).
   Gated by `CRABBIT_BLOCKMAP=1` (cheap, but off by default).
2. **Ingestion tool** `scripts/perf-harness/profile_ingest.py`:
   `perf record -e cycles:u [-c PERIOD] -- <bin> …` then `perf script -F
   ip,sym,symoff` (fallback: addresses resolved via nm on the binary);
   aggregates samples per block id via the blockmap(s), normalizes per
   function like spectral does (`freq[b] = samples[b]/samples[entry_id]`,
   entry forced ≥ 1 sample; additive smoothing alpha=1 so a sampled zero is a floor, never a literal 0.0 — a literal zero marks uses as free-to-spill and measurably backfires; see profile_ingest.normalize), writes
   `profile.json: {symbol: [freq per RA-order block index]}`. Multiple
   runs merge additively.
3. **Consumption.** `CRABBIT_BLOCK_FREQ=profile` + `CRABBIT_PROFILE=
   <profile.json>`:
   - linear engine: `BlockFreqModel::Profile` in codegen_opts, looked up
     by function symbol in the allocator (fall back to uniform when a
     function or block is missing — a stale profile must never error).
   - eregalloc engine: `BlockFreqSource::Provider` fn that reads the same
     JSON (lazy static keyed by env var) and returns the per-block vector
     for the FuncOp's symbol; wrong-length vectors already fall back to
     uniform per the contract.
   Determinism makes IDs stable across the profile build and the rebuild
   (same compiler, flags, source ⇒ same RA-time CFG). The profile build
   should use the SAME config as the rebuild except the freq source.
4. **Experiment** (harness): corpus CPU variants × {uniform, spectral,
   profile} × {linear-weighted, eregalloc-c0, eregalloc-c2}; the harness
   grows a profile-collection step (run the baseline-freq binary once
   under perf, ingest, then compile the `profile` configs). Report also
   the per-function correlation (Spearman) of spectral vs measured
   frequencies — the direct CMT-model validation number.

## Extensions (explicitly later)

- ARM SPE / cache-miss counters → CMT cache-model line; branch counters →
  profile-guided `Aarch64BlockPlacementPass`; sample-driven spill-slot
  and remat decisions in eregalloc's oracle (freq is already the hook).

## Gates

Unit: blockmap ids survive placement/relax (test asserts contiguous
coverage of .text per function and monotone offsets); ingestion on a
synthetic perf script fixture; provider fallback on stale/missing
profiles. End to end: one corpus kernel (elementwise_chain — it showed
the largest eregalloc runtime delta, +34%) profiled and rebuilt with
`profile` under both engines, checksums unchanged, and the freq vectors
logged. No commits anywhere.

## v1.5: CUDA profiler feedback (user direction 2026-08-31)

v1 stays perf-only (CPU). v1.5 ingests the CUDA profiling tools — both
are installed (ncu = Nsight Compute CLI 2025, nsys 2025.3.2, plus CUPTI
under /usr/local/cuda/extras):

1. **Ingestion** `kernel-corpus/tools/ncu_ingest.py`: run
   `ncu --set full --csv` (or `--metrics` with a curated list) on a
   runner launch; parse per-kernel: achieved occupancy, registers,
   memory throughput %, warp-stall breakdown (long scoreboard / MIO /
   barrier / not-selected …), L1/L2 hit rates, and — the key one —
   **PC-sampling stall counts per PTX/SASS line** (`ncu --set source`,
   or CUPTI pcsampling). nsys stays out of scope (timeline tool; the
   runner's CUDA events already time launches).
2. **Attribution back to IR**: crabbit owns PTX emission, so the NVPTX
   translator gets an optional line map (`CRABBIT_PTX_LINEMAP=1`): PTX
   line → source llvm-dialect op id (module-order index), sidecar JSON
   next to the .ptx. ncu's per-line stalls then map to IR ops the same
   way v1's blockmap maps PCs to machine blocks. (SASS lines map to PTX
   lines via ncu's own correlation columns; take what it gives, don't
   chase perfect SASS attribution in v1.5.)
3. **First consumers** (analysis before optimization, same discipline as
   v1): per-kernel stall attribution tables for the corpus — e.g. prove
   directly that seidel's remaining gap is long-scoreboard stalls on the
   recurrence load, or that generic-vs-.shared loads show as MIO
   throttle on gemm_tiled — feeding ANALYSIS.md and ranking translator
   work. Optimization consumers (metric-driven unroll/remat/state-space
   choices in the kernel pipeline) come after the attribution proves
   trustworthy, as v2.
4. Runner integration: `tools/run_all.py --profile-gpu` runs the ncu leg
   per kernel×arm (ncu serializes launches — never mix its timings with
   the CUDA-event timings; it contributes counters only).

Gate for v1.5: attribution round-trip test on one kernel (a PTX line the
map says is the recurrence load must be the top stall line in seidel_2d's
rust-crabbit arm) and the gemm_tiled generic-load hypothesis confirmed or
refuted with the MIO/long-scoreboard split. No timings from under ncu.
