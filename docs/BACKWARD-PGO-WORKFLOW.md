# Backward PGO on the existing kernels: the laptop ↔ Spark workflow

*2026-10-26. Companion to [BACKWARD-PGO.md](BACKWARD-PGO.md) (the design)
and [PROFILE-FEEDBACK-BACKWARD.md](PROFILE-FEEDBACK-BACKWARD.md) (the
mechanism). This note answers one practical question: how to run the
backward profile loop over the two kernel corpora when the person works
from a laptop and the hardware, the profilers and the backends live on the
DGX Spark.*

## The split

Everything that needs the machine stays on the Spark; the laptop is a
browser and an editor over SSH. There is no step a laptop can do better:

| step | needs | where |
|---|---|---|
| build with provenance (`ll.op_id`, blockmap, PTX linemap) | the backend dylib for the Spark's targets | Spark |
| measure | `perf` (CPU, `perf_event_paranoid=1`) and `ncu` (GPU; counters are unlocked for non-admin via `/etc/modprobe.d/nvidia-profiler.conf`, verified 2026-10-26) | Spark |
| ingest and lift (`profile.json`, `op_costs.json`) | the sidecars of that exact build | Spark |
| consume: rebuild under `CRABBIT_PROFILE` / `CRABBIT_MEASURED_COSTS`, time | the same hardware the profile came from | Spark |
| look: hot ops at every level, decisions changed, A/B tables | a browser | laptop |

Copying profiles to the laptop and replaying there would need the same
backend build and the same module text to reproduce the op ids — fragile,
and it buys nothing because the rebuild must run on the Spark anyway.

## The two campaign drivers (Spark side)

Both are stdlib Python, one command per corpus, every artifact under the
corpus's `results/`, nothing committed by the tools.

**GPU — `kernel-corpus/tools/attribution_campaign.py`** (both corpora by
default, 31 kernels, ~5 minutes end to end):

```sh
cd ~/projects/montaglue/kernel-corpus
python3 tools/attribution_campaign.py                 # --size large, nvcc contrast on
python3 tools/attribution_campaign.py --corpus ../kernel-corpus-real --kernels argsort,gemv_f32
```

Per kernel it builds the device crate with `CRABBIT_PROFILE_MAP=1
CRABBIT_PTX_LINEMAP=1` (plus `CRABBIT_EMIT_IR=1 CRABBIT_TRACE=1`, see
below), assembles a lineinfo cubin, checks correctness, collects ncu
counters and per-SASS-line warp-stall samples, lifts them through the
linemap to source-level op ids (`<out>/<kernel>/crabbit.op_costs.json`),
and takes the same counters from the nvcc build of `ref.cu`. The
`summary.md` has the counters side by side and the hottest source ops with
their PTX text and dominant stall reason. Counters and samples only: ncu
serializes launches, so timings stay with `run_all.py`.

**CPU — `kernel-corpus/tools/cpu_pgo_campaign.py`** (the E1-v2 protocol as
a reusable tool; `--backend` is the backend dylib whose configurations it
compares):

```sh
python3 tools/cpu_pgo_campaign.py \
    --backend <backend dylib> \
    --corpus ../kernel-corpus-real --runs 10 --cpu 7
```

Per kernel: profile build (blockmap) → perf record pinned → ingest with
smoothing → every cell rebuilt from one profile (`linear`,
`linear-w-specloop`, `linear-w-prof`, `c2-specloop`, `c2-prof`,
`c2-prof-meas`), correctness-gated on both sizes, kernel-function
disassembly hashed (so "did the profile change a decision" is a fact, not a
guess), static spill metrics, timed pinned runs with the load average
recorded. Run it on an idle machine; the timings are the point.

## Where the results become visible from the laptop

The uncommitted crabbit/pliron-inspect work of October makes the UI open a
cargo project and list, under its `.crabbit-traces/`, the per-pass traces,
the emitted modules and any `*op_costs*.json` profile; the **costs** button
of a run lifts a profile onto that run's IR (`run_costs`). The campaign
drivers feed exactly that:

- GPU: each device crate is its own cargo project (`kernels/<k>/device`),
  so the build with `CRABBIT_EMIT_IR=1 CRABBIT_TRACE=1` leaves
  `kernels/<k>/device/.crabbit-traces/ir/<stem>-kernels.plir` (the lowered
  kernel module the server replays) and the driver copies the lifted
  profile to `kernels/<k>/device/.crabbit-traces/profiles/<k>.crabbit.op_costs.json`.
- CPU: the CPU crates are workspace members, so the corpus root is the
  project; the driver publishes `<corpus>/.crabbit-traces/profiles/<k>.cpu.profile.json`
  and `<k>.cpu.profile.op_costs.json`.

From the laptop, two equivalent setups:

1. **UI on the Spark, browser on the laptop.** On the Spark:
   `pliron-inspect` (it builds and runs `crabbit-analysisd` and the
   pliron-lsp engine for the opened project). On the laptop:
   `ssh -L 8080:localhost:8080 spark` and open `http://localhost:8080/analysis`,
   project root `~/projects/montaglue/kernel-corpus/kernels/<k>/device`
   (GPU) or `~/projects/montaglue/kernel-corpus` (CPU).
2. **Editor on the laptop.** VS Code Remote-SSH onto the Spark with the
   pliron-lsp extension: the `.plir` modules and traces open with the
   dialects' hover/definition; the UI link above for the cost views.

The results directories themselves (`results/gpu-attribution/<ts>-campaign/summary.md`,
`results/cpu-feedback/<ts>-pgo/summary.md`) are plain Markdown and read
fine over SSH or after an `rsync` of just those directories (they contain
no binaries except the snapshotted backend dylib, which can be excluded).

## What the loop can and cannot feed today

Consumers that exist: measured per-decision costs for allocators that
read them, block frequencies (`CRABBIT_BLOCK_FREQ=profile`), and
the hot-line views at the `source`/`ra` boundaries. On GPU no pass consumes
a profile yet (ptxas allocates registers; the mid-end decisions that matter
there — unroll factors, sinking under divergence, state spaces — are
analytic), so the GPU campaign is a *diagnosis* instrument: it says which
source op the stall mass lands on and why, per kernel, against nvcc's
counters. That is what located the state-space and addressing mechanisms
in September; the 2026-10-26 run is the first over all 31 kernels.

## Next step that would make the loop one click

A `run_profile` command on `crabbit-analysisd`: given a run of an opened
project, build the matching binary or cubin with the sidecars, run `perf`
or `ncu` on this host, ingest, and return the `op_costs` payload the UI
already knows how to lift. Then the laptop never runs a campaign script;
it presses a button on the Spark's server. The two drivers above are the
reference implementation of what that command has to do.
