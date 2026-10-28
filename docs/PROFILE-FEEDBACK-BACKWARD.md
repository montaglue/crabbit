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

1. **Forward stamping**: every pass stamps ops it creates with a
   `derived_from` attribute carrying the id(s) of the input op(s) they
   lower/replace (local knowledge only; ids are per-function dense, like
   `ll.blockmap_id`). Ops that survive a pass keep their id. v1's blockmap
   generalizes to an **op-map** sidecar: final `.text` ranges → machine op
   ids (encoder already knows per-op byte lengths).
2. **Ingest**: perf samples → machine-op costs (extend
   `profile_ingest.py`), in cycles.
3. **Backward interpretation**: the analysis server replays the pipeline
   (it already materializes IR at every pass boundary deterministically);
   walking passes in REVERSE order, costs are aggregated by
   `derived_from`: cost(input op) = Σ cost(output ops derived from it).
   At every boundary the costs are cached as attributes ON THAT LEVEL'S
   IR, in that dialect's vocabulary — semantic accounting (spill cost,
   division-lowering cost, bounds-check cost) is not a feature, it is
   what the costs look like at each level.
4. **Consumers, each at its own level**:
   - register allocation (RA boundary): measured restore/spill cost of
     the allocator's own PAST decisions replaces its estimate where
     present — decision-level feedback, the first experiment.
   - mid-end passes (llvm level): e.g. divmagic consults measured cost of
     its previous expansion vs a hardware divide per site (settles the
     X925-vs-GPU per-target question empirically, per site).
   - frequency-model calibration: spectral-vs-measured error attributed
     per CFG shape and per level.
   - UI / hot lines: the top of the lift is source-construct cost for
     crabbit binaries with no DWARF needed by our own tooling.

## The open design question (decide per pass kind)

Expansion (1→N) aggregates trivially. **Merges and dedups (N→1: GVN CSE,
simplify-cfg block merges, inlining's many-callers→one-body) need an
attribution rule** — proportional split by static count, dominant-user,
or multi-parent `derived_from` sets. The rule is the pass's own adjoint
semantics and is declared next to the pass, never globally. Reordering
passes (block placement) are id-neutral. Passes that create ops from
nothing (constants, prologue) attribute to a per-pass synthetic root so
overhead is visible as that pass's own cost.

## Experiments (in order)

E1: measured vs estimated allocator restore costs ({estimate, measured})
on the corpus CPU variants — runtime + spill traffic.
E2: per-site divmagic decisions from measured costs vs always-on/off.
E3: attribution-fidelity audit: inject a known-cost synthetic op at the
llvm level, verify the backward lift recovers its cost within noise
through the full stack (the correctness gate for the whole mechanism).
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

## Prerequisites and status

Substrate (block frequencies) + analysis server replay: landed. Real
sampling still blocked on `kernel.perf_event_paranoid=4` (user sudo
needed).

**Machine-level foundation — landed 2026-08-31** (the first backward hop,
machine → LLVM level):
- `ll.op_id` on LLVM ops (pass `aarch64-op-ids`, immediately before isel)
  and `ll.derived_from` on machine ops, gated on `CRABBIT_BLOCKMAP` /
  `CRABBIT_PROFILE_MAP`. Isel stamps via a bracket around its per-source
  dispatch (`opmap::IselStamper`) — one hook covers every lowering arm,
  edge blocks included; ABI glue is rooted `isel:abi`. RA restores inherit
  the use op's source, spill stores the def op's; frame rewrites inherit
  the rewritten op, prologue/epilogue root `frame`; placement's
  materialized branches root `placement`. Legalize/asm-lower/relax create
  nothing (verified). Audit: `opmap::unstamped_op_count` == 0 through the
  full pipeline (tested).
- Sidecar: the blockmap's block ranges gain
  `"ops": [{index, derived_from, start, end}]` (final layout order, byte
  offsets). Backward compatible — block-frequency-only consumers ignore it.
- Ingest: `profile_ingest.py` additionally writes
  `<output stem>.op_costs.json`:
  `{symbol: {machine: {op index: samples}, lifted: {llvm op id | root
  name: samples}}}` — `lifted` IS the machine→LLVM hop. Roots render as
  `isel:abi`/`regalloc`/`frame`/`placement`/`unattributed`.
- E3-lite audit test (`crates/pliron-ll/tests/opmap_audit.rs`): synthetic
  samples on a known machine range lift onto the correct LLVM op id
  through the real pipeline + real ingest.
- Consumer seam: the `lifted` costs are keyed by the def op's
  `derived_from` id at the RA boundary, so an allocator (registered as an
  extension, docs/EXTENDING.md) can replace its restore estimates with
  them where present.

**Mid-end adjoints + server lift — landed 2026-08-31 (round 2):**
- `llvm-op-ids` (head of the shared mid-end) stamps the SOURCE numbering;
  every merging/creating mid-end pass declares its adjoint next to its
  code: simplify/div-strength-reduce replacements via the
  `replace_op_with_value` chain walk (1→N, single parent), GVN merges →
  multi-parent `ll.derived_from_many` (equal weights), SROA slot allocas +
  access-rewrite brackets, simplify-cfg terminator rewrites (1→1),
  inline → `ll.inlined_from` = call-site id (callee-local ids kept for
  future cross-function lift); licm/sink move (id rides), dse/adce delete
  (nothing). Undeclared creations surface as root `midend`.
- RA boundary (`aarch64-op-ids`) PRESERVES surviving source ids, gives
  mid-end-created/inlined ops fresh ids, and serializes the fresh-id →
  source-parents adjoint table into the module (emitted as the sidecar's
  top-level `__midend__` object). `profile_ingest.py` adds the `source`
  level to `op_costs.json` (equal-weight splits).
- E3 proper (`opmap_audit.rs::two_hop_lift_recovers_source_ops_with_gvn_split`):
  synthetic samples on a merged add and on a divide's magic expansion lift
  through TWO hops onto a 50/50 split across both original adds and onto
  the original sdiv, via the real pipeline + real ingest tool.
- Server `run_costs` (protocol + `DriverHooks::attribution_ops`): joins an
  op_costs payload with the ops at the `source` (default) or `ra`
  boundary — per-op cost + line + snippet for heat-mapping any pass view,
  plus per-synthetic-root semantic accounting; cached per (run, level).

Still not started (at the time of this round): experiments E2/E4; real
sampling was blocked on `kernel.perf_event_paranoid=4`; intermediate
(per-mid-end-pass) boundary views — ids exist only at the two numbered
boundaries by design (eager collapse); cross-function lift INTO callee
bodies (callee-local ids are preserved but unconsumed).
