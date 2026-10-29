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

- **Forward stamping.** With `CRABBIT_PROFILE_MAP=1`, the mid-end's first
  pass stamps every op with `ll.op_id` (`llvm-op-ids`,
  `crates/pliron-ll/src/passes/llvm/op_ids.rs`). Each transforming pass
  stamps what it creates, next to its own code: expansions name a single
  parent (`ll.derived_from` — every op of a magic-number division chain
  derives from the original `sdiv`); a GVN survivor lists the ops it
  absorbed (`ll.derived_from_many`); inlined ops keep their callee-local
  id and name the call site (`ll.inlined_from`); unrolled clones name
  their original's sources. Deletions need nothing; moves ride along.
- **The machine boundary.** `aarch64-op-ids` keeps the ids of surviving
  ops and gives fresh ones to ops the mid-end created; their parents stay
  on the ops. Instruction selection brackets every machine op it emits
  with the source op being lowered
  (`crates/pliron-ll/src/passes/aarch64/opmap.rs`); the register allocator
  stamps spill stores with the spilled def and reloads with the use they
  feed; frame lowering and block placement stamp their code or named roots
  (`isel:abi`, `regalloc`, `frame`, `placement`). Machine blocks get their
  position at the allocator's input (`aarch64_blockmap_id`).
- **Traces.** With `CRABBIT_TRACE=1` the compilation leaves its IR after
  every pass in a trace (`.crt`, format in `crates/pliron-trace`). The
  stamps are pliron outlined attributes, so every snapshot carries them.
- **Backward passes** (`crates/pliron-bpgo`). A backward pass is paired
  with its forward pass by name: it gets the forward pass's input and its
  output with a profile on the ops, and puts the profile on the input —
  reading back the stamps its forward pass wrote. `walk` puts the
  measurement on the last snapshot, then goes through the passes from the
  last: a function a pass did not change keeps its profile; a changed one
  goes through the pass's backward pass; a function stops at the first
  pass that changed it and has no backward pass. Only a long enough
  suffix of the pipeline needs backward passes, and it grows pass by
  pass. crabbit's are next to their forward passes:
  `crates/pliron-ll/src/passes/aarch64/backward.rs` (instruction
  selection and every machine pass) and
  `crates/pliron-ll/src/passes/llvm/backward.rs` (the mid-end).
- **What a profile holds.** Not only cost: `cycles` (cycles spent in the
  op), `executions` (how often it ran), and `<pass>.cycles` — cycles of
  code a pass created as its own decision (`regalloc.cycles` for spill
  code charged to the value it spilled, `frame.cycles`, `abi.cycles`,
  `placement.cycles`, `unattributed.cycles`). Cycles add up where ops
  merge; an expansion's ops each ran as often as their source; unrolled
  copies add their executions to the original's.
- **Measuring.** `crabbit-bpgo run --trace <version>.crt -- <program>
  <args>` (`crates/bpgo`) runs the program under `perf record`, pinned to
  the fastest core with that core's own counters (`cycles`,
  `instructions`), puts the samples on the machine ops of the last
  snapshot by their encoded byte ranges, walks the trace, and writes
  `<version>.bpgo.crt`: the same snapshots, with the profile of every
  snapshot reached in the header, keyed by each op's line and column. The
  snapshot text is never re-printed — pliron renames values and blocks
  when it prints parsed IR. The raw recording stays next to it as
  `<version>.perf.data`; `crabbit-bpgo lift` walks such a recording again.
  [pliron-inspect](https://github.com/montaglue/pliron-inspect) shows the
  profile on every snapshot it reaches, as text and as a CFG.
- **Consumers.** `CRABBIT_BLOCK_FREQ=profile` with
  `CRABBIT_PROFILE=<profiled trace>` gives the allocator measured block
  frequencies (`passes/profile_freq.rs`). An allocator registered as an
  extension (docs/EXTENDING.md) can read measured costs per source op of
  the allocator's input from the same trace
  (`aarch64::backward::profiled_snapshot`, `samples_by_derived`).
- **GPU.** The NVPTX translator maps every emitted PTX line to its source
  op (`CRABBIT_PTX_LINEMAP`, `<ptx>.linemap.json`,
  `crates/pliron-ll/src/nvptx/mod.rs`; byte-identical PTX with the map on
  or off). The kernel path has no backward passes yet; the corpus's
  Nsight Compute tooling attributes samples through the line map.

## Evidence

**Fidelity is tested, not assumed**
(`crates/pliron-ll/tests/backward_pgo.rs`): a function with a GVN-merged
pair of adds and a constant division goes through the real mid-end and
machine pipeline, traced; synthetic cycles on the machine code of the
merged add and of the division's magic-number expansion must come back at
the source level as a 50/50 split across both original adds and onto the
original `sdiv`, and the total must hold at every snapshot.

**End to end, CPU** (2026-10-29, `gemm_tiled` CPU from the kernel corpus,
`--size large --repeat 5`, Cortex-X925): 331,416 of 331,618 cycle samples
land on the traced code; the profile reaches every snapshot from
`erase-mir-types` to the encoder, with the same 3.31e10 cycles at each.
At the source level one multiply holds 31% of the run (10.4G cycles,
1.83G executions, 485M of them its spill code).

The results below were measured on 2026-08-31 with the first
implementation of the lift, which ran outside the compiler; it was
replaced by the backward passes above.

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
  The block-frequency consumer now floors unmeasured blocks at half the
  least measured one, but magnitude uncertainty below the sampling floor
  remains, and blending with structural priors is open work.
- **Merge attribution is a modeling choice.** Multi-parent splits are
  equal-weight; nothing validates the weights themselves, only the id
  sets (the tests pin the sets).
- **Corpus scale.** All experiment numbers come from 14 small kernels with
  hot loops that mostly fit the (deliberately tiny) 4-register allocatable
  pool; the decision-feedback experiments on pressure-rich host code
  (e.g. the arrow crates) are the next step, not a done one.
- **Coverage is a suffix.** The MIR-level passes (`erase-mir-types` and
  before) have no backward passes yet, so profiles stop at the LLVM
  level.
- **Inlined code** returns its profile to the call site; the callee's
  own body gets only the profile of its out-of-line calls.
