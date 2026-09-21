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

DONE (2026-09-20): `passes/llvm/analysis.rs` now owns the table
(`OpEffect` + `op_effects()`), with FP arithmetic, fcmp, select and the
FP casts classified Pure (no FP exceptions modeled; nothing derived from
the table reassociates). simplify/adce consume `deletable_op_ids`,
gvn/dse `memory_benign_op_ids`, licm/sink derive via `op_ids_in`. gvn
additionally CSEs FP bins/fneg/fcmp/select on bit-identical keys that
include the fast-math flags (equality, like the nsw/nuw rule). Bycatch
fixed by the consolidation: AShr was missing from gvn's bin set and from
licm's hoistable set; InsertValue was missing from licm's.

## Full unroll (2026-09-20): the deferred item 5, revisited for gemm_tiled

The GPU corpus finally produced the evidence item 5 waited for: after the
state-space + immediate/GEP translator rounds, gemm_tiled sits at 1.49×
nvcc (56 regs, 66.7% occupancy) and the remaining PTX/SASS delta is the
un-unrolled 16-iteration inner tile loop — nvcc fully unrolls it; our
16-instruction PTX inner loop stalls on per-iteration index math + loop
control that straight-line code would constant-fold away.

`passes/llvm/unroll.rs` (llvm-unroll), gated by
`CRABBIT_MIDEND_DISABLE=unroll`:

- FULL unroll only. 1- or 2-block natural loops, one latch, one exit
  edge: `header→body→header` with the exiting cond_br in the header
  (rustc's canonical `Range` while-shape) or `header⇄header` (do-while).
  Preheader required (same restriction as licm).
- Trip count by direct simulation of the canonical induction shape:
  block-arg iv starting at constant C0, backedge update `add iv, C1`
  (constant), exiting `icmp` against constant C2 (either operand order,
  any predicate, the compare may test iv or iv-next). Simulation uses the
  dialect's wrapping semantics at the iv's width, so every predicate/step
  combination is exact, not pattern-matched.
- Bounds: 2 ≤ trip ≤ 32, ≤ 40 non-terminator ops across the loop blocks
  (worst-case growth ~1300 ops). Any region-free op clones (loads,
  stores, calls included): cloning preserves execution counts exactly, so
  no speculation argument is needed.
- Rewrite: clones are laid straight-line into the preheader; the iv is
  substituted by its per-iteration constant; loop-carried block args
  thread through the clones; uses of loop-defined values outside the loop
  are rewired to the final iteration's clone; the preheader branch is
  retargeted to the exit block with the exit edge's operands mapped; the
  loop blocks are deleted. Dead residue (cloned icmps, final iv adds) is
  left for adce/simplify.
- ADJOINT: 1→N cloning; clones are stamped `derived_from` the original
  op's effective sources after stripping the copied identity attrs (a
  clone must not duplicate `ll.op_id`); materialized iv constants derive
  from the iv update op; the new preheader→exit branch derives from the
  cond_br it replaces.
- Placement: after gvn/divmagic/licm/gvn (loop body already minimal,
  invariants hoisted), before the backward round — dse/adce erase the
  dead per-iteration control clones and the final simplify folds the
  per-iteration address math. Run for all targets; thresholds currently
  target-independent (TargetProfile is threaded for a future split).

### Results (2026-09-20, same-session A/B, gate off vs on)

GPU corpus (kernel-corpus results/gpu/unroll-baseline vs unroll-after,
14×2 arms×2 sizes, all correct, zero spills): **a well-diagnosed null on
gemm_tiled.** The pass fires on gemm_tiled's 16-iter tile loop and on
transpose; 12/14 kernels' PTX is byte-identical. gemm_tiled PTX loses its
inner loop (140→304 lines, 32 straight-line ld.shared with per-iteration
constant offsets folded into the index adds) — but runtime is exactly
unchanged (1.2941 → 1.2942 ms, 1.494× nvcc), regs 56, occupancy 66.7%,
because **ptxas was already fully unrolling this loop in SASS**: the
baseline and after cubins disassemble to the same 218 instructions modulo
register names (32 LDS, 16 IMAD MACs, no inner branch in either).
Mechanism of the remaining 1.49× gap, from the SASS diff vs nvcc:
- ptxas pre-hoists all 32 per-iteration LDS addresses into live registers
  (R2–R47: 15 IADD3 `+1..+0xf` then 32 LEA) — that address file IS the
  56-reg pressure capping occupancy at 66.7%; nvcc holds 2 base registers
  and uses immediate-offset/vectorized shared loads.
- nvcc vectorizes one operand's tile loads as 4× LDS.128 vs our 16× LDS.
- integer-op totals per tile: crabbit 35 LEA + 34 IADD3 + 18 IMAD vs nvcc
  19 IMAD + 5 LEA + 2 IADD3.
The fix is therefore NOT more mid-end unrolling: it is shared-memory
addressing form — emit `ld.shared [base+imm]` (reassociate
`(tx + 16k)<<2` to `base_tx + 64k` so the constant lands in the memory
operand) and consider LDS vectorization — translator/addressing work,
tracked for the nvptx backlog.
The optional second gvn after unroll was NOT added: per-iteration address
expressions differ in their constants (nothing for CSE to merge — the
iv-independent subexpressions were already hoisted by licm, which runs
before), and the SASS evidence shows ptxas already performs the cleanup
downstream on GPU.

Transpose also unrolls (regs 22→26, runtime 0.5904→0.5860 ms, within
noise, still ≤ nvcc). gemm_control/seidel_2d loops don't match (trip
count above the bound / shape), PTX identical. Geomean crabbit/nvcc
1.046→1.051 — entirely session noise on code-identical kernels (the two
changed kernels moved 0.000/−0.7%).

CPU corpus (perf-harness results/unroll-cpu-ab, runs=5, pinned CPU 7,
same-session unroll-off vs unroll-on, all correct): **no kernel regresses
>3% — and the unroller WINS where GPU was a null**, because the aarch64
backend has no ptxas downstream to do the unroll for it:

| kernel (run_median_s) | off | on | on/off |
|---|---|---|---|
| gemm_tiled | 10.198 | 8.525 | **0.836** |
| transpose | 1.141 | 1.020 | **0.894** |
| vector_add | 1.014 | 1.003 | 0.989 |
| rms_norm | 1.514 | 1.492 | 0.986 |
| seidel_2d | 0.992 | 1.000 | 1.008 (worst) |
| the other 9 | — | — | 0.997–1.002 |

Cross-session sanity vs results/session-after `linear`: same picture
(gemm_tiled 0.837, transpose 0.890, rest ±1%). No target gating needed:
thresholds stay target-independent.
