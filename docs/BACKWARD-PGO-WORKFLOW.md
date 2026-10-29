# Backward PGO on the existing kernels: the laptop ↔ Spark workflow

*2026-10-26, updated 2026-10-29. Companion to
[BACKWARD-PGO.md](BACKWARD-PGO.md) (the design and its mechanism). This
note answers one practical question: how to run the backward profile loop
over the two kernel corpora when the person works from a laptop and the
hardware, the profilers and the backends live on the DGX Spark.*

## The split

Everything that needs the machine stays on the Spark; the laptop is a
browser and an editor over SSH. There is no step a laptop can do better:

| step | needs | where |
|---|---|---|
| build traced, with provenance (`CRABBIT_TRACE=1 CRABBIT_PROFILE_MAP=1`; PTX linemap on the GPU) | the backend dylib for the Spark's targets | Spark |
| measure | `perf` (CPU, `perf_event_paranoid=1`) and `ncu` (GPU; counters are unlocked for non-admin via `/etc/modprobe.d/nvidia-profiler.conf`, verified 2026-10-26) | Spark |
| walk back (`crabbit-bpgo`, CPU) | the trace of that exact build, and its binary while `perf` resolves the samples | Spark |
| consume: rebuild under `CRABBIT_PROFILE=<profiled trace>`, time | the same hardware the profile came from | Spark |
| look: hot ops at every level, decisions changed, A/B tables | a browser | laptop |

The profiled trace (`<version>.bpgo.crt`) stands on its own: it can be
copied anywhere and opened in pliron-inspect. Rebuilding from it must
still happen on the Spark, on the hardware it measured.

## CPU: one program, measured and walked back

Build the program through crabbit with a trace and provenance, run it
under `crabbit-bpgo`, open the result:

```sh
cd ~/projects/montaglue/kernel-corpus
RUSTUP_TOOLCHAIN=1.99.1 RUSTC_BOOTSTRAP=1 CRABBIT_TRACE=1 CRABBIT_PROFILE_MAP=1 \
    cargo rustc -p gemm-tiled-cpu --release -- \
    -Zcodegen-backend=<crabbit backend dylib> -Coverflow-checks=off
crabbit-bpgo run --trace .crabbit-traces/gemm_tiled_cpu/<version>.crt -- \
    target/release/gemm-tiled-cpu --size large --repeat 5
```

`crabbit-bpgo run` pins the program to the fastest core (`--cpu` picks
another), records `cycles` and `instructions` with that core's counters
(`--period`, default 100003), and writes `<version>.bpgo.crt` and the
recording `<version>.perf.data` next to the trace. It reports how many
samples landed on the traced code and where each function's profile
stops. `crabbit-bpgo lift --trace <version>.crt --perf-data <file>
--period <n>` walks a recording made elsewhere (or again, after a
backward pass changed) — `perf` needs the program's binary to resolve it.

## The two campaign drivers (Spark side)

Both are stdlib Python in the kernel corpus, one command per corpus, every
artifact under the corpus's `results/`, nothing committed by the tools.

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

Per kernel: profile build → perf record pinned → ingest with
smoothing → every cell rebuilt from one profile (`linear`,
`linear-w-specloop`, `linear-w-prof`, `c2-specloop`, `c2-prof`,
`c2-prof-meas`), correctness-gated on both sizes, kernel-function
disassembly hashed (so "did the profile change a decision" is a fact, not a
guess), static spill metrics, timed pinned runs with the load average
recorded. Run it on an idle machine; the timings are the point. It still
calls crabbit's former ingest script, removed on 2026-10-29: it needs
porting to `crabbit-bpgo` before it runs again.

## Where the results become visible from the laptop

Traces land in `.crabbit-traces/` at the root of the compiled project:
each GPU device crate is its own cargo project (`kernels/<k>/device`);
the CPU crates are workspace members, so the corpus root is the project.
The profiled traces sit next to the traces they came from.

From the laptop, two setups:

1. **pliron-inspect on the Spark, browser on the laptop.** On the Spark:
   `pliron-inspect --no-open` and, on the laptop,
   `ssh -L 3000:localhost:3000 spark` then `http://localhost:3000`; or
   `pliron-inspect --listen-all` on a trusted network. Dialects folder:
   the crabbit checkout; traces folder: the project's `.crabbit-traces/`.
   A profiled trace shows its profile on every snapshot it reaches —
   heat and values per op in the text, per block in the CFG.
2. **Editor on the laptop.** VS Code Remote-SSH onto the Spark with the
   pliron-lsp extension: the `.plir` modules and traces open with the
   dialects' hover/definition.

The results directories themselves (`results/gpu-attribution/<ts>-campaign/summary.md`,
`results/cpu-feedback/<ts>-pgo/summary.md`) are plain Markdown and read
fine over SSH or after an `rsync` of just those directories (they contain
no binaries except the snapshotted backend dylib, which can be excluded).

## What the loop can and cannot feed today

Consumers that exist: measured per-decision costs for allocators that
read them, and block frequencies (`CRABBIT_BLOCK_FREQ=profile`). On GPU no pass consumes
a profile yet (ptxas allocates registers; the mid-end decisions that matter
there — unroll factors, sinking under divergence, state spaces — are
analytic), so the GPU campaign is a *diagnosis* instrument: it says which
source op the stall mass lands on and why, per kernel, against nvcc's
counters. That is what located the state-space and addressing mechanisms
in September; the 2026-10-26 run is the first over all 31 kernels.
