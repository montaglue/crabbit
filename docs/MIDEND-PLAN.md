# Mid-end plan: the LLVM optimizations crabbit measurably lacks

Motivation (2026-08-31): the kernel corpus quantified what the missing
mid-end costs. GPU, identical IR through `llc` vs our translator:
seidel_2d 2.31× vs 1.16×, gemm_tiled 1.81× vs 1.57×, gemm_control 1.28×
vs 1.05× (kernel-corpus/ANALYSIS.md Part A). CPU: crabbit-built corpus
binaries are 3–57× slower than rustc/LLVM builds (gemm_tiled 21 s vs
0.37 s). Beyond raw speed, the gaps *confound the RA experiments*: junk
memory traffic and recomputed addresses inflate live ranges and spill
counts that eregalloc/CMT are then measured against.

Current mid-end (crates/pliron-ll/src/passes/llvm/): inline, simplify
(constant folding, insert/extract forwarding, BLOCK-LOCAL store-to-load
forwarding, DSE, DCE), simplify-cfg, sroa, pin-type-punned-slots, plus
pliron's mem2reg. All local; nothing crosses a block except simplify-cfg's
merging.

## Passes to add, in evidence order

1. **gvn** — dominator-scoped global value numbering + redundant-load
   elimination + store-to-load forwarding across blocks, including the
   loop back edge (the seidel_2d reload; the gemm address recomputation).
   Conservative memory model: a load is redundant with a prior load/store
   at a *syntactically identical* address (same SSA value or identical
   GEP-of-same-base chain) with no intervening store/call that may write
   memory (any store to an unproven-distinct address, any call, kills).
   Pure-op CSE has no such caveat. Loop-carried forwarding: handle the
   single-latch natural-loop case (store in latch path, load at header
   iteration n+1) — this alone fixes seidel.
2. **div-strength-reduce** — sdiv/udiv/srem/urem by non-zero constants →
   shifts (powers of two, with the LLVM sign-fix sequence for sdiv) and
   magic-number multiply-high sequences (Hacker's Delight 10-*; LLVM's
   TargetLowering::BuildSDIV/BuildUDIV). i32 and i64; i8/i16 via widening.
   Kernels: seidel `/3`, plus pervasive Rust idioms (checked arithmetic,
   slice indexing). div_recurrence's data-dependent divisor must remain a
   real divide — good negative test.
3. **licm** — hoist loop-invariant pure ops (arith, GEP, casts, constants)
   to the preheader (create one when absent). No speculation of loads in
   v1 (no dereferenceability model): pure ops only. Needs dominators +
   natural-loop detection (below).
4. **instcombine extras** (extend simplify.rs, no new pass): reassociate
   `(a+c1)+c2`, `x*2^k → shl`, GEP-of-GEP flattening, compare
   canonicalization. Only what the corpus PTX/asm actually shows.
5. **NOT now**: loop unrolling and vectorization — they change register
   pressure, i.e. the *independent variable* of the RA experiments; add
   later as an explicitly flagged axis. Inliner heuristics: revisit only
   if profiles show call overhead.

## Infrastructure

- **Dominance & loops**: check what pliron 0.17 exports (mem2reg needs
  dominance internally); else implement Cooper–Harvey–Kennedy (~100
  lines) in `pliron-ll/src/passes/llvm/analysis.rs` with natural-loop
  detection (back edge t→h where h dominates t; loop body by reverse
  reachability). Differential-test dominators against
  `cmt-boolean-dataflow::DomMatrix` as a *dev-dependency* only
  (pliron-ll is published; a path dep would break `cargo publish`, a
  dev-dep is stripped).
- **Placement**: the shared `add_midend_passes` list in
  `crates/crabbit/src/lib.rs` — host and kernel pipelines both gain the
  passes (GPU arms improve too). Order: inline → simplify-cfg → sroa/
  mem2reg rounds (as today) → **gvn → licm → gvn** → div-strength-reduce
  → simplify → simplify-cfg.
- **Ablation**: `CRABBIT_MIDEND_DISABLE=gvn,licm,divmagic` (comma list,
  parsed like CodegenOpts, unknown names error). Default all-on. This
  gives the harness a mid-end axis for free.
- **Research note**: landing these *changes the RA experiments' baseline*.
  The 20260831-clean-full sweep is the pre-midend reference (backend
  df2122f6…); after landing, re-sweep and ask a genuinely new question:
  does the engine/policy ranking survive a real mid-end? (Less junk
  pressure → different spill decisions.)

## Gates (every pass, before it lands)

- `cargo test --workspace` green; backend-tests under
  `CRABBIT_REGALLOC=eregalloc` too.
- All 14 corpus CPU variants: crabbit-built checksums == spec (both sizes).
- GPU: `tools/run_all.py --arms rust-crabbit,rust-llvm` all-correct, and
  record the before/after ratio table (expect seidel & gemm to move).
- Per-pass unit tests with hand-built IR including the negative cases
  (may-alias store kills forwarding; data-dependent divisor untouched;
  no hoisting past a store that may alias).

## Backward dataflow passes (added 2026-08-31, user direction)

The existing mid-end is forward-only except block-local DSE/DCE in
simplify.rs. Backward-analysis passes to add, on top of M1's analysis.rs
(shared CFG utilities; generic backward bitset worklist over the region
CFG, differential-tested against cmt-boolean-dataflow, whose boolean-matrix
formulation is exactly the A^T of the forward problems — the CMT tie-in):

6. **global-dse** — backward memory liveness: delete a store when every
   path to exit hits another store to the same (syntactic-model) address
   first, or the address is a non-escaping alloca that is never loaded
   after it. Value for the experiments: dead stores inflate str_sp counts
   the RA sweep attributes to spilling.
7. **sink** — liveness-driven code sinking: move a pure op into the block
   of its uses (or the nearest common post-dominated block) when that
   shortens live ranges. NOTE: this changes register pressure — the RA
   experiments' independent variable — so unlike the rest of the mid-end
   it lands as an EXPLICIT axis (`CRABBIT_MIDEND_DISABLE=sink`, default ON
   but every sweep must record it), and "policy ranking with vs without
   sinking" is itself an experiment worth running.
8. **adce** — aggressive DCE: mark control-flow/side-effect roots, walk
   uses backward, delete everything unmarked, rewrite branches whose
   targets became empty. Subsumes the local DCE for cross-block junk.
9. **pre/lcm** (later, biggest) — lazy code motion: backward
   anticipability + forward availability, subsumes most of LICM and
   cross-diamond CSE. Only after 6–8 prove out; it reorders computation
   enough to need its own verification round.

Post-dominators: analysis.rs grows them next to dominators (same CHK on
the reversed CFG; exit-block handling for multiple returns — synthetic
exit). Sequencing: M3 starts when M1's analysis.rs + gvn land (same files;
no concurrent edits).

## Sink cross-target finding (2026-08-31, measured)

gemm_tiled, sink on vs off, SAME 128 registers: GPU 1.492→1.643 ms
(+10.1%); CPU kfn_ldr_sp −5.8%, runtime −2.0%. Root cause (PTX diff = two
`add.s32` moved): sinking placed the adds under divergent tile-bound
guards inside the k-loop. On SIMT the warp executes both sides of a
divergent guard, so the "skipped" work isn't skipped, and the sunk chain
(add→setp→bra→guarded global ld→st.shared→bar.sync) lengthens the
barrier-critical path every iteration, unhidden at 33% occupancy. On CPU
the same move genuinely skips work and shortens live ranges.
Follow-up: target-aware sink policy — kernel pipeline forbids sinking
that adds control dependence (or requires warp-uniform guards); host
pipeline keeps the aggressive form. Until then the recorded axis
(CRABBIT_MIDEND_DISABLE=sink) is the control.

## Tracked: consolidate the op-safety classification (review finding, 2026-09-10)

Five hand-maintained OpId sets encode op safety across peer passes —
simplify's `pure_op_ids`, licm's `hoistable_op_ids`, sink's
`sinkable_op_ids`, gvn's `bin_op_ids`/`cast_op_ids`/`benign_op_ids`, and
dse's reuse of gvn's — and all of them omit FP arithmetic
(FAdd/FSub/FMul/FDiv/FNeg), `SelectOp`, and the FP casts the pipeline
demonstrably carries. Consequence: every FP op is conservatively treated
as an unknown-effect op (GVN kills load-forwarding windows across an
`fadd`; ADCE roots dead FP math; LICM/sink skip it) — silent, divergent
per pass, and every new dialect op must be remembered in up to five lists
in four files. Plan: one shared classification table (op → {pure,
speculatable, memory effects}) in `passes/llvm/analysis.rs` that every
pass derives its set from, with the FP ops classified; a missed op then
fails in ONE place. Not done in the review-fix round: touching the safety
sets changes what every pass may do and deserves its own measured commit.
