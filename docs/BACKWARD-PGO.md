# Backward PGO: profiles as data flowing back through the compiler

*The profile-feedback design implemented in crabbit. Everything below is
implemented and measured in this repository; every claim links to code or
recorded results.*

## Thesis

Classical PGO treats a profile as a **flat annotation on the program**:
samples are mapped to source lines or basic blocks once, and a handful of
fixed consumers (inliner, block layout, spill weights) read that one map.
Every other level of the compiler that wants the information must re-derive
meaning from it heuristically — which is why PGO pipelines accumulate
scaling fudge factors and mismatch repair.

crabbit's design instead says: **PGO is not recompilation with more
information; it is step-by-step interpretation of accepted information.**
Measurements exist only at the lowest level (PC samples over machine
instructions; GPU warp-stall samples over SASS/PTX). Decisions live at
every level above. So profile data flows *backward* through the same
pipeline the program flowed forward through: in reverse pass order, each
pass accepts costs expressed on its **output** dialect's ops, interprets
them through the transformation only it knows it performed, and re-expresses
them on its **input** dialect's ops. Each pass is the **adjoint
interpreter of its own lowering** — the analogy to backpropagation is
exact, and it is why the design was originally called "backward passes".

Two properties fall out that a flat map cannot have:

1. **Per-level semantics.** At each boundary the lifted costs are stated in
   that dialect's vocabulary. "18% of cycles are spill reloads", "this much
   is division lowering", "this much is the frame prologue" is not a
   feature — it is just what the costs look like after lifting past the
   pass that created that code.
2. **Per-decision provenance.** Costs attach not only to program points but
   to the *optimization decision* that produced the instructions (this
   spill store belongs to allocation decision X; this multiply-high chain
   to strength-reduction site Y). The consumer of a measurement can be the
   exact component that made the decision being measured.

## Mechanism (as implemented)

- **Forward stamping.** A gated pass stamps every mid-end-input op with
  `ll.op_id` (`crates/pliron-ll/src/passes/llvm/`, pass `llvm-op-ids`).
  Each transforming pass declares its adjoint rule next to its own code:
  expansions carry a single parent (`ll.derived_from` — e.g. every op of a
  magic-number division chain derives from the original `sdiv`); merges
  carry weighted multi-parent sets (`ll.derived_from_many` — a GVN-deduped
  survivor lists all originals); inlined ops keep their callee-local id
  and gain the call-site id (`ll.inlined_from`). Deletions need nothing;
  moves ride along. Passes with no declared adjoint surface loudly as a
  synthetic root instead of misattributing.
- **The machine boundary.** Instruction selection brackets every machine op
  it emits with the source op being lowered
  (`crates/pliron-ll/src/passes/aarch64/opmap.rs`); the register allocator
  stamps spill stores with the spilled def and reloads with the use they
  feed; frame lowering and block placement stamp their code or named roots
  (`isel:abi`, `regalloc`, `frame`, `placement`). The encoder then emits a
  sidecar mapping final byte ranges to op provenance
  (`<obj>.blockmap.json`). The GPU analogue maps every emitted PTX line to
  its source op (`<ptx>.linemap.json`,
  `crates/pliron-ll/src/nvptx/mod.rs`) — verified byte-identical PTX with
  the map on or off.
- **Ingest and lift.** `scripts/perf-harness/profile_ingest.py` (Linux
  `perf`) and `kernel-corpus/tools/ncu_ingest.py` (Nsight Compute,
  including per-line warp-stall reasons joined via `nvdisasm` lineinfo)
  produce one `op_costs.json` shape with costs at the machine, lifted, and
  source levels — the backward walk applied at ingest, with equal-weight
  splits at multi-parent merges.
- **Consumers.** The eregalloc register allocator's restore oracle accepts
  measured per-decision costs (`CRABBIT_MEASURED_COSTS`), replacing its
  estimates where a measurement exists. The analysis server's `run_costs`
  command replays a compilation and returns per-op cost tables (with text
  positions) at any supported boundary, plus per-root semantic accounting —
  hot-line views at every level of the compiler, from one measurement.

## Evidence

**Fidelity is audited, not assumed**
(`crates/pliron-ll/tests/opmap_audit.rs`): synthetic samples injected on a
known machine-instruction range must lift to the correct source op through
the *real* pipeline and the *real* ingest tool. The two-hop audit covers a
GVN merge (samples split 50/50 onto both original adds) and a
strength-reduced division (samples land on the original `sdiv`). The first
draft of this audit failed — constant folding had moved the code — which is
the point: the audit catches attribution drift.

**Real hardware, GPU**
(`kernel-corpus/results/gpu-attribution/20260831-090227/ANALYSIS.md`):
on the `seidel_2d` kernel, 72% of 190k Nsight samples attribute to a single
source op — the *consumer* of the loop-carried load (137,290 of its 137,669
samples are long-scoreboard stalls), refining the pre-registered hypothesis
(PC sampling charges the wait to the consumer, not the load) and refuting
another (the division-replacement chain contributes nothing measurable).
On `gemm_tiled`, the attribution located the stall mass exactly on the four
generic tile-access ops and *inverted* the expected signature (MIO
throttle belongs to nvcc's `ld.shared` version at 97% occupancy; crabbit's
generic loads ride the L1TEX path at 33% occupancy).

**Real hardware, CPU, including the honest nulls**
(`kernel-corpus/results/cpu-feedback/20260831/E1-ANALYSIS.md`,
`E4-ANALYSIS.md`): measured per-decision costs currently change *no*
allocation decision on the 14-kernel corpus (the estimates already rank
victims identically at this function scale — a statement about the corpus,
recorded as such). Measured *frequencies* change decisions on 13/14 kernels
but do not improve runtime — and the backward machinery itself diagnosed
why: pricing every reload placement by measured execution showed both
configurations tie on the allocator's modeled objective (114,094 vs 115,212
weighted reload cost on `gemm_tiled`), so the losses live in terms the
objective omits (paired spill stores, code growth). **The measurement
system found a defect in the cost model of the optimizer consuming it** —
decision-level feedback doing exactly its job. Separately, E4: analytic
(uniform-prior spectral) frequencies correlate with measured ones at median
Spearman 0.002 and underestimate hot-block mass 5–50× — the model
structurally cannot represent iteration counts.

## Relation to existing systems

| System | What it attributes to | Consumers | What's different here |
| --- | --- | --- | --- |
| AutoFDO / sample-based PGO | source lines → basic-block counts, one flat map | inliner, layout, spill weights | no per-decision provenance; every IR level re-derives meaning from the source-line map; mismatch after transformation is repaired heuristically |
| BOLT / Propeller | final binary basic blocks | post-link layout | operates at one level (the bottom); no lift into the compiler's decision space |
| Nsight Compute source view | SASS ↔ source lines via debug info | a human | display-only; crabbit lifts the same samples into IR levels where *passes* consume them |

The genuinely new elements are (a) the profile as **level-typed data with
per-pass adjoint semantics** declared where the transformation lives, and
(b) **decision-level feedback**: costs addressed to the specific
optimization choices that produced the code, with the audit apparatus to
trust them.

## Honest limitations

- **Sampling resolution is a semantic hazard.** A block executing tens of
  thousands of times can catch zero samples; an early version fed those
  zeros to the allocator as "free to spill here" and made code worse.
  Ingest now applies additive smoothing (floor = one sample), but
  magnitude uncertainty below the sampling floor remains, and blending
  with structural priors is open work.
- **Merge attribution is a modeling choice.** Multi-parent splits are
  equal-weight; nothing validates the weights themselves, only the id
  sets (the audits pin the sets).
- **Corpus scale.** All experiment numbers come from 14 small kernels with
  hot loops that mostly fit the (deliberately tiny) 4-register allocatable
  pool; the decision-feedback experiments on pressure-rich host code
  (e.g. the arrow crates) are the next step, not a done one.
- **Cross-function lift** is stamped (`ll.inlined_from`) but not yet
  consumed by any analysis view.
