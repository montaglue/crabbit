# Profile feedback v1 (as it should be): backward interpretation through the dialect stack

STATUS NOTE (user, 2026-08-31): this is NOT a v2 — it is what v1 was
always meant to be. The block-frequency work that landed earlier is the
substrate (sampling, blockmap, freq consumption), not the design.

Thesis (the user's recovered design — the reason the idea was
always called "backward passes"): **PGO is not recompilation with more
information; it is step-by-step interpretation of accepted information.**
Measurements exist only at the lowest level (PC samples over machine
instructions). Decisions live at every level above. So profile data must
flow BACKWARD through the same pipeline the program flows forward through:
in reverse pass order, each pass accepts the costs expressed on its OUTPUT
dialect's ops, interprets them through the transformation only it knows it
performed, and re-expresses them on its INPUT dialect's ops. Each pass is
the adjoint interpreter of its own lowering. Contrast: classical PGO keeps
one flat block/source-level map consumed at fixed points, and every other
level re-derives meaning from it heuristically.

## Mechanism

As implemented (details and file pointers: [BACKWARD-PGO.md](BACKWARD-PGO.md)):

1. **Forward stamping**: passes stamp the ops they create with the ops
   they came from (`ll.op_id`, `ll.derived_from`,
   `ll.derived_from_many`, `ll.inlined_from`; at the machine boundary
   `aarch64_op_id` / `aarch64_derived_from`, plus named roots for code
   created from nothing). Local knowledge only; ops that survive a pass
   keep their stamps. The stamps live on the ops, nowhere else.
2. **Traces**: a traced compilation keeps the IR after every pass, stamps
   included.
3. **Measurement**: `crabbit-bpgo` runs the program under `perf` and puts
   the samples on the machine ops of the last snapshot, by their encoded
   byte ranges.
4. **Backward interpretation**: walking the passes in REVERSE order, each
   pass's backward pass — code paired with the forward pass — takes the
   profile on its output and puts it on its input, reading back where its
   forward pass put what it needs. A function stops at the first pass
   that changed it and has no backward pass. Every snapshot reached gets
   its profile, in that level's vocabulary: `cycles`, `executions`, and
   per-pass decisions (`regalloc.cycles`, `frame.cycles`, …).
5. **Consumers, each at its own level**:
   - register allocation: measured block frequencies
     (`CRABBIT_BLOCK_FREQ=profile`), and measured costs of the
     allocator's own PAST decisions per source op, for an allocator that
     reads them — decision-level feedback, the first experiment.
   - mid-end passes (llvm level): e.g. divmagic consults measured cost of
     its previous expansion vs a hardware divide per site (settles the
     X925-vs-GPU per-target question empirically, per site).
   - frequency-model calibration: spectral-vs-measured error attributed
     per CFG shape and per level.
   - UI / hot lines: pliron-inspect shows the profile on every snapshot
     reached — source-construct cost for crabbit binaries with no DWARF.

## Merges and dedups (decided per pass, next to the pass)

Expansion (1→N) is trivial: every op of the expansion ran as often as its
source, and their cycles add up on it. N→1 needs a rule, and each rule is
the pass's own backward semantics, never a global one: a GVN survivor
shares its profile equally with the ops it absorbed; an unrolled clone is
another instance of its original (executions add); inlined code returns
its profile to the call site. Reordering passes (block placement) are
neutral. Code created from nothing (constants, prologue, spill code) is
the creating pass's own decision: `<pass>.cycles` on the function, or on
the value it serves (spill code on the value spilled).

## Experiments (in order)

E1: measured vs estimated allocator restore costs ({estimate, measured})
on the corpus CPU variants — runtime + spill traffic.
E2: per-site divmagic decisions from measured costs vs always-on/off.
E3: attribution fidelity: known synthetic cycles on machine code must
come back onto the right source ops through the full stack, conserved at
every snapshot (the correctness gate for the whole mechanism;
`crates/pliron-ll/tests/backward_pgo.rs`).
E4: spectral-model error maps.

## Experiment results (E4, 2026-08-31, real perf samples)

**E4 — spectral model vs measured frequencies: the analytic model
cannot see iteration counts.** Median Spearman 0.002, median Pearson
−0.209 over the 14 kernel functions; the measured-hottest block's
probability share is underestimated 5–50× (hot-ratio 0.008–0.20;
uniform null 0.025–0.25 — spectral is not better than uniform here and
often ranks the loop HEADER above the BODY, hence the negative
correlations: uniform branch priors give a backedge probability 0.5, so
loop mass saturates at O(1) instead of O(N)). Scope caveat: this judges
the uniform-prior Perron model as crabbit consumes it, on small
loop-dominated kernel CFGs; relative within-loop weighting is a weaker
claim not tested here. Full tables:
kernel-corpus/results/cpu-feedback/20260831/E4-ANALYSIS.md +
results.csv; profiles under profiles/. Support added:
`CRABBIT_DUMP_FREQS=<path>` (JSONL, RA order) in the aarch64 allocator.

## Status (2026-10-29)

- Backward passes cover instruction selection, every aarch64 machine pass
  and the whole mid-end; profiles reach every snapshot from
  `apply-module-linkage` to the encoder. The MIR-level passes have none yet.
- E3 holds (`backward_pgo.rs`); end to end on `gemm_tiled` CPU, 99.9% of
  the cycle samples land on traced code and the total is the same at
  every snapshot reached.
- E1 and E4 were measured on 2026-08-31 with the first implementation,
  which lifted the samples outside the compiler; it was replaced by the
  backward passes on 2026-10-29. E2 is not started.
- The GPU path maps PTX lines to source ops (`CRABBIT_PTX_LINEMAP`); it
  has no backward passes yet.
