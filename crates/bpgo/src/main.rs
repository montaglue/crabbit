//! `crabbit-bpgo`: backward PGO for a crabbit compilation.
//!
//! A compilation traced with `CRABBIT_TRACE=1 CRABBIT_PROFILE_MAP=1` leaves
//! its IR after every pass, with the metadata each pass stamped, in a trace
//! (`.crt`). This tool runs the compiled program under `perf`, puts the
//! samples on the ops of the trace's last snapshot, and walks the trace
//! backward through crabbit's backward passes, so every snapshot they reach
//! gets a profile. It writes the trace again, with the profiles, next to
//! the original as `<version>.bpgo.crt`.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, bail};
use clap::{Parser, Subcommand};
use pliron_bpgo::BackwardPasses;
use pliron_trace::Trace;

#[derive(Parser)]
#[command(
    name = "crabbit-bpgo",
    about = "Backward PGO for a crabbit compilation"
)]
struct Args {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run a program under perf, then walk its profile back through the
    /// trace of the compilation that built it.
    Run {
        /// The trace (`.crt`) of the compilation that built the program.
        #[arg(long)]
        trace: PathBuf,
        /// Events per sample.
        #[arg(long, default_value_t = 100_003)]
        period: u64,
        /// The CPU to run the program on (default: the fastest one); the
        /// counters are that CPU's own.
        #[arg(long)]
        cpu: Option<usize>,
        /// Where to write the trace with profiles (default: next to the
        /// trace, as `<version>.bpgo.crt`).
        #[arg(long)]
        out: Option<PathBuf>,
        /// The program and its arguments.
        #[arg(last = true, required = true)]
        command: Vec<String>,
    },
    /// Walk a recorded profile (`perf record -e cycles:u -e
    /// instructions:u -c PERIOD`) back through the trace.
    Lift {
        #[arg(long)]
        trace: PathBuf,
        #[arg(long)]
        perf_data: PathBuf,
        /// The period the profile was recorded with.
        #[arg(long)]
        period: u64,
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

fn main() -> anyhow::Result<()> {
    match Args::parse().command {
        Cmd::Run {
            trace,
            period,
            cpu,
            out,
            command,
        } => {
            let perf_data = trace.with_extension("perf.data");
            let cpu = cpu.unwrap_or_else(fastest_cpu);
            let (cycles, instructions) = match pmu_of(cpu) {
                Some(pmu) => (format!("{pmu}/cycles/u"), format!("{pmu}/instructions/u")),
                None => ("cycles:u".to_string(), "instructions:u".to_string()),
            };
            let status = Command::new("taskset")
                .args([
                    "-c",
                    &cpu.to_string(),
                    "perf",
                    "record",
                    "-q",
                    "-e",
                    &cycles,
                    "-e",
                    &instructions,
                    "-c",
                ])
                .arg(period.to_string())
                .arg("-o")
                .arg(&perf_data)
                .arg("--")
                .args(&command)
                .status()
                .context("cannot run taskset/perf")?;
            if !status.success() {
                bail!("perf record failed ({status})");
            }
            lift(&trace, &perf_data, period, out, Some(&command))
        }
        Cmd::Lift {
            trace,
            perf_data,
            period,
            out,
        } => lift(&trace, &perf_data, period, out, None),
    }
}

/// The CPU with the highest capacity (the big cores of a big.LITTLE chip),
/// else CPU 0.
fn fastest_cpu() -> usize {
    (0..1024)
        .map_while(|cpu| {
            std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{cpu}/cpu_capacity"))
                .ok()
                .map(|text| (cpu, text.trim().parse::<u64>().unwrap_or(0)))
        })
        .max_by_key(|&(cpu, capacity)| (capacity, std::cmp::Reverse(cpu)))
        .map_or(0, |(cpu, _)| cpu)
}

/// The PMU counting on `cpu`, when the chip has several (one per core
/// type): its name in `/sys/bus/event_source/devices`.
fn pmu_of(cpu: usize) -> Option<String> {
    let devices = std::fs::read_dir("/sys/bus/event_source/devices").ok()?;
    let mut pmus: Vec<_> = devices
        .flatten()
        .filter_map(|entry| {
            let cpus = std::fs::read_to_string(entry.path().join("cpus")).ok()?;
            let name = entry.file_name().to_string_lossy().into_owned();
            name.contains("pmu").then_some((name, cpus))
        })
        .collect();
    if pmus.len() < 2 {
        return None;
    }
    pmus.sort();
    pmus.into_iter()
        .find(|(_, cpus)| cpu_list_contains(cpus.trim(), cpu))
        .map(|(name, _)| name)
}

/// Whether a `0-4,10-14` style list holds `cpu`.
fn cpu_list_contains(list: &str, cpu: usize) -> bool {
    list.split(',').any(|range| match range.split_once('-') {
        Some((a, b)) => {
            a.parse::<usize>().is_ok_and(|a| a <= cpu) && b.parse::<usize>().is_ok_and(|b| cpu <= b)
        }
        None => range.parse::<usize>() == Ok(cpu),
    })
}

/// crabbit's backward passes.
fn backward_passes() -> BackwardPasses {
    let mut passes = BackwardPasses::new();
    pliron_ll::passes::aarch64::backward::register(&mut passes);
    pliron_ll::passes::llvm::backward::register(&mut passes);
    passes
}

fn lift(
    trace_path: &Path,
    perf_data: &Path,
    period: u64,
    out: Option<PathBuf>,
    command: Option<&[String]>,
) -> anyhow::Result<()> {
    // Link the dialects every snapshot may contain.
    let _ = crabbit_mir::dialects::dialect_mir::register;
    let script = Command::new("perf")
        .args([
            "script",
            "--no-demangle",
            "--ns",
            "-F",
            "time,event,ip,sym,symoff",
            "-i",
        ])
        .arg(perf_data)
        .output()
        .context("cannot run perf script")?;
    if !script.status.success() {
        bail!(
            "perf script failed: {}",
            String::from_utf8_lossy(&script.stderr)
        );
    }
    let samples = pliron_ll::passes::aarch64::backward::Samples::from_perf_script(
        &String::from_utf8_lossy(&script.stdout),
        period,
    );
    let trace = Trace::read(trace_path)?;
    let mut attached = None;
    let mut timeline = None;
    let walk = pliron_bpgo::walk(&trace, &backward_passes(), &mut |ctx, root| {
        attached = Some(pliron_ll::passes::aarch64::backward::attach(
            ctx, root, &samples,
        ));
        timeline = Some(pliron_ll::passes::aarch64::backward::timeline(
            ctx, root, &samples,
        ));
        Ok(())
    })
    .map_err(|e| anyhow::anyhow!(e))?;
    let attached = attached.unwrap_or_default();

    let mut annotated = pliron_bpgo::annotate(
        &trace,
        &walk,
        serde_json::json!({
            "tool": "perf",
            "events": ["cycles", "instructions"],
            "period": period,
            "command": command,
            "samples": attached.total,
            "matched": attached.matched,
        }),
    );
    // The cycle samples one by one, on the ops of the last snapshot.
    if let (Some(timeline), Some(bpgo)) = (timeline, annotated.meta.extra.get_mut("bpgo")) {
        let mut timeline = serde_json::to_value(timeline)?;
        timeline["snapshot"] = serde_json::json!(trace.snapshots.len().saturating_sub(1));
        timeline["event"] = serde_json::json!("cycles");
        bpgo["timeline"] = timeline;
    }
    annotated.meta.note = Some(format!(
        "backward PGO of {}",
        trace.meta.note.as_deref().unwrap_or("a compilation")
    ));
    let out = out.unwrap_or_else(|| {
        let stem = trace_path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        trace_path.with_file_name(format!("{stem}.bpgo.crt"))
    });
    annotated.write(&out)?;

    let reached = walk
        .snapshots
        .keys()
        .next()
        .copied()
        .unwrap_or(trace.snapshots.len());
    eprintln!(
        "crabbit-bpgo: {} of {} cycle samples on the traced code; profiles on snapshots {}..{} ({} .. {})",
        attached.matched.get("cycles").copied().unwrap_or(0),
        attached.total.get("cycles").copied().unwrap_or(0),
        reached,
        trace.snapshots.len() - 1,
        trace
            .snapshots
            .get(reached)
            .map_or("?", |s| s.label.as_str()),
        trace.snapshots.last().map_or("?", |s| s.label.as_str()),
    );
    for (symbol, stop) in &walk.stops {
        if stop.pass.is_some() {
            eprintln!(
                "  {symbol}: stops at snapshot {}: {}",
                stop.reached, stop.reason
            );
        }
    }
    eprintln!("crabbit-bpgo: wrote {}", out.display());
    Ok(())
}
