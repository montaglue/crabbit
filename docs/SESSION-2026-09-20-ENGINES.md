# Session 2026-09-20: making the research engines genuinely good

Goal (user): make eregalloc (especially) and combinatorial-matrix-theory
genuinely good — beat LLVM or the CUDA compiler on real code — expanding
what crabbit can compile as the means. Everything below is uncommitted
working-tree state across four repos (crabbit, eregalloc,
combinatorial-matrix-theory, kernel-corpus tooling/results).

## Headline results

**GPU (native PTX vs nvcc, 14-kernel corpus, same-session ratios):**
geomean crabbit/nvcc **1.13× → 1.04×**. gemm_control 1.39 → **1.07**
(regs 36→30); registers down on 13/14 kernels, most at 100% occupancy;
gemm_tiled 128 → 56 regs (occupancy 33% → 67%, runtime −13%). 14/14
kernels checksum-correct throughout. Remaining gaps: gemm_tiled 1.49×
(nvcc fully unrolls the 16-iteration tile loop — mid-end unroller in
flight), seidel_2d ~1.15× (memory-latency bound per earlier ncu
attribution).

**CPU (14-kernel corpus, Cortex-X925 pinned, runs=10):** the session's
changes cut runtime **~36% geomean** vs the pre-session backend (every
kernel faster; dot_product −52%, gemm_tiled −46%). The best configuration
is now **eregalloc-c2 + spectral-loop (geomean 0.627 vs before-linear),
beating the baseline linear allocator (0.638)** — the research engine with
the CMT-derived loop prior wins on real kernels: histogram −14% and
transpose −9% vs linear, gemm_tiled tied-best. The loop prior is decisive
through the engine: c2-uniform → c2-spectral-loop takes gemm_tiled
13.44 s → 10.13 s (−25%) and histogram −16%, no losses.
Results: `scripts/perf-harness/results/{session-before,session-after}`.

**CPU vs LLVM:** crabbit-best is now ~2–3× LLVM on the scalar kernels
(was 3–10×). The remaining outliers (elementwise_chain 9×, gemm_tiled
27×) are LLVM auto-vectorization/blocking — the CPU frontier is SIMD,
not register allocation.

**Frequency model (CMT):** the loop-aware prior lifts median Spearman vs
measured profiles from **0.002 to 0.742** (better on 14/14 kernels) and
cuts hot-block mass error from ~15× to ~2.6×
(`kernel-corpus/results/cpu-feedback/20260831/FREQ-MODEL-EVAL.md`).

## What landed

1. **Callee-saved registers (crabbit backend + eregalloc-passes mirror).**
   Pools grew 4→14 GPRs (x9–x12 caller + x19–x28 callee) and 4→21 FPRs
   (d16–d19, d23–d31 caller + d8–d15 callee). Values live across calls
   allocate callee-saved instead of the old unconditional force-spill
   (now only the overflow fallback). The allocator records used
   callee-saved registers in a new `aarch64_saved_regs` bitmask FuncOp
   attribute (printed/parsed); frame lowering emits the save area above
   the spill area. Isolated static effect on arrow-schema: **sp-loads
   −26.8%, sp-stores −24.0%, instructions −14.8%** — and the static
   counts still include the new prologue saves.

2. **eregalloc: typed strict FP + wider rematerialization.** FP arithmetic
   translates to typed strict OpKinds (real congruence/CSE in both
   oracles; reassociation stays opt-in per the FP licensing doc); FP
   `fmov #imm` constants, `adrp`+`add_lo12` pairs, and `add_sp_offset`
   frame addresses rematerialize via re-emission recipes (the adrp pair
   fits one scratch register); remat cost gate moved to a pass-side plan
   price so the engine's deliberate opaque over-pricing can't block
   always-cheaper address remat. Width-unsafe casts deliberately kept as
   stable opaques (the documented past miscompile shape).

3. **eregalloc: live-range splitting v1 (call-boundary).** Eligible
   call-crossing intervals (single def, straight-line block chain) split
   into open segments (caller-saved) and merged cross-call segments
   (callee-saved or one store/reload per call region). Position-keyed
   rewrite, parallel-move cycle breaking, per-segment H5 safe form.
   Corpus: 42/42 correct, runtime-neutral; arrow: spill-traffic neutral,
   +0.65% instructions from boundary moves (mechanism understood; v2 =
   segment register affinity). The end-to-end gate caught a real ICE the
   unit tests missed.

4. **CMT: loop-aware spectral frequencies.** Not the literal backedge
   rule — measured to be a no-op (machine CFGs rotate loops; backedges
   sit on unconditional latches) — but LLVM's loop-membership convention
   (in-loop successors share q=7/8), built on boolean-dataflow dominance
   backedges. Exported from spectral-cfg (`compute_with` prior seam +
   `block_frequencies_loop_aware`) and ported into crabbit's
   `spectral_freq.rs`; wired as `CRABBIT_BLOCK_FREQ=spectral-loop` in both
   engines. Trip constant N=8 validated by sweep (higher N hurts
   rms_norm, doesn't help jacobi — the residual error is which block in
   the nest is hot, not the constant).

5. **NVPTX: state spaces, immediates, coverage.** Pointer-provenance
   state-space inference (ld/st/atom `.global`/`.shared`; shared values
   keep raw addresses, cvta only at escapes), fma.rn contraction,
   immediate folding + GEP `mad.lo` rewrite. `ld.global.nc` skipped with
   a soundness argument (raw-pointer kernel ABI carries no aliasing
   promise; nvcc emits none here either — parity). Coverage: **struct
   GEPs + aggregate SSA values, `.local` allocas, device `.func` calls
   with the `.param` ABI (recursion included)** — real Rust kernel shapes
   compile and run on the GB10 (`backend-tests/fixtures/kernel-coverage`).

6. **Mid-end FP classification.** One shared `OpEffect` table
   (`passes/llvm/analysis.rs`); FP arith/compares/select/FP-casts now
   participate in GVN/LICM/sink/ADCE/DSE (strict identity only — GVN keys
   include fast-math flags). Honest null on `-O`-precleaned fixtures;
   real effect on unoptimized MIR (dead fdiv deleted, invariant fmul
   hoisted).

7. **Mid-end loop unroller (`llvm-unroll`).** Conservative full unroll of
   canonical constant-trip loops (trip ≤ 32, body ≤ 40 ops, 1–2 block
   shapes, exact trip by simulating the induction), with correct 1→N
   backward-PGO attribution; `CRABBIT_MIDEND_DISABLE=unroll` ablation.
   GPU gemm_tiled: **well-diagnosed null** — ptxas already fully unrolls
   that loop in SASS (proven by cubin diff); the true remaining 1.49×
   mechanism is addressing form: ptxas pre-hoists 32 per-iteration
   shared-memory addresses into registers (that address file IS the
   56-reg pressure), while nvcc keeps 2 base registers with
   immediate-offset `ld.shared` and 4× `LDS.128` vectorization —
   recorded as the concrete nvptx follow-up. CPU: unplanned wins —
   **gemm_tiled −16.4% (10.20 → 8.53 s), transpose −10.6%**, no kernel
   regressing on either target.

## Gates (all green at session end)

- crabbit workspace: 190 pliron-ll lib tests + all executing fixtures
  (backend_smoke 9 + 2 ignored AMDGPU) + nvptx suites (219 pliron-ll
  total after round 2).
- eregalloc: 31 (root) + 21+1 (eregalloc-passes, incl. splitting).
- CMT: spectral-cfg 11+ tests.
- Corpus: 14 kernels × {linear, eregalloc-c0, eregalloc-c2} × both sizes,
  checksum-correct (results/engines-gate-cs, results/splitting-gate);
  GPU 14/14 correct through every nvptx change.

## Honest nulls and open items

- Fixture wall-times are sub-millisecond — noise, not evidence.
- Splitting is runtime-neutral on this corpus (its win case needs
  callee-pool exhaustion, now rarer with 10 callee GPRs); the capability
  matters for higher-pressure code and enables v2 affinity work.
- arrow under eregalloc: reloads −1.6–2.4% vs linear (static); dynamic
  placement quality is not visible in static counts.
- CPU SIMD/vectorization is the dominant remaining gap vs LLVM.
- jacobi_2d/rms_norm frequency hot-mass remains below uniform —
  model-structure, not trip-count (see FREQ-MODEL-EVAL addendum).
- x86_64-darwin has no swappable allocator/callee-saved work (unchanged).

## Continuation results (backward PGO + GPU final)

**E1-v2 (backward PGO on the finished backend)** — 14 kernels × 8 cells,
all checksum-correct (`kernel-corpus/results/cpu-feedback/20260920-pgo`):
measured profile frequencies are now the best linear configuration
(geomean 0.988 vs linear; softmax −5% exactly where the spectral-loop
model mis-predicts and loses 9.4%; histogram −12%). Under the engine,
profile ties its own model overall and wins histogram (best cell 1.694,
−14.4% vs linear); measured oracle costs are now marginally positive
(0.9886) — E1's "pure null" is superseded. The `marginal` cost shape is
REFUTED as a default (gemm_tiled +15% under both frequency sources: the
distance divisor is load-bearing there); `enriched` stays the default and
`marginal` remains an A/B knob.

**GPU final** (`kernel-corpus/results/gpu/20260920-final`): geomean
crabbit/nvcc **1.0125**, 14/14 correct; gemm_tiled a dead tie (1.000,
35/35 registers); six kernels at or below 0.997. Remaining losses:
seidel_2d 1.139, gemm_control 1.069 (30 vs 40 regs — nvcc buys ILP with
registers; diggable).

**seidel lcfwd GPU A/B: null** (1.924 on vs 1.916 off). The pass fires
(static proof) but moves nothing on the GB10; either the bottleneck is
the serial recurrence chain itself (register RAW is still serial) or the
old 72%-long-scoreboard attribution is stale post-state-spaces. Deciding
needs fresh ncu (blocked on the profiling-permission modprobe).

**kernel-corpus-real** (new sibling repo, uncommitted): seven kernels
sourced verbatim from real codebases (sunbird: iq3_xxs_dequant,
comp_pool, mamba_ssm_step; llama.cpp: q4_0_dequant, argsort; llm.c:
softmax_forward, adamw) with provenance headers, Rust device ports, and
corpus-convention specs — the next-generation arena where compiler
quality actually differentiates. First crabbit-vs-nvcc measurements in
progress.

## Real-kernel corpus: first results and a miscompile

`kernel-corpus-real` (new sibling repo, git-initialized, zero commits) —
seven kernels sourced verbatim (provenance headers, MIT notices): sunbird
iq3_xxs_dequant / comp_pool / mamba_ssm_step, llama.cpp q4_0_dequant /
argsort, llm.c adamw / softmax_forward. All seven compile through crabbit
and run correct on the GB10.

crabbit/nvcc (large size, 20 runs, same session): argsort **0.90 (win)**,
adamw 1.00, softmax_forward 1.02, mamba_ssm_step 1.04, q4_0_dequant 1.07,
iq3_xxs_dequant 1.16, comp_pool 1.40. Loss mechanisms read from the PTX:
comp_pool — nvcc unrolls the slot loops ~8× to batch independent loads on
a latency-bound decode launch (needs runtime-trip partial unroll);
iq3 — the 32-entry chunk buffer stays in `.local` because raw-pointer
access defeats mem2reg (needs promotion of non-escaping raw-pointer local
arrays). The dominant coverage blocker across all three source codebases
is missing `shfl.sync` lowering (every warp-level reduction/GEMV).

**The corpus caught a real miscompile on its first run**: MIR
`CastKind::Transmute` was imported with structural cast classification, so
`f32::from_bits` (whose core body is `mem::transmute`) became a NUMERIC
u32→f32 conversion. Confirmed latent on the CPU path of committed HEAD as
well (a runtime-value probe aborts under the pre-fix backend; const
arguments had masked it — `from_bits` is `const fn`). Fixed by pinning the
MIR cast kind through import (bitcast lowering; PTX `mov.b32` alias);
regression fixture `backend-tests/fixtures/transmute-aarch64` added and
green in both profiles. 287 workspace tests green with the fix.
