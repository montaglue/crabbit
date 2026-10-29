//! Backward PGO for the aarch64 pipeline (`pliron-bpgo`): the measurement
//! it starts from — `perf` samples on the encoded functions — and the
//! backward passes from the encoder back to instruction selection.
//!
//! Every machine op carries the `derived_from` its forward pass stamped
//! ([opmap]): the id of the LLVM op it was selected from, or
//! a negative root for code a pass made from nothing. A pass between
//! instruction selection and the encoder keeps most ops as they are
//! (matched by opcode and `derived_from`, in order); an op it created
//! stands in place of what it took out at the same spot, or belongs to the
//! op it was stamped from; code from nothing is the pass's own cost, on the
//! function (`frame.cycles`, `abi.cycles`, ...).

use std::collections::{BTreeMap, HashMap};

use pliron_bpgo::{
    BackwardPass, BackwardPasses, CYCLES, EXECUTIONS, ProfileAttr, align, profile, walk::ops_under,
};

use super::opmap::{self, roots};
use crate::{
    context::{Context, Ptr},
    dialects::{
        aarch64::op_interfaces::BinarySerializableOpInterface, aarch64::ops::FuncOp,
        builtin::op_interfaces::SymbolOpInterface,
    },
    ir::{op::op_cast, operation::Operation},
    linked_list::ContainsLinkedList,
};

/// `perf` samples of one run: per event (`cycles`, `instructions`), per
/// function symbol, per byte offset in the function, the sample count; and
/// the cycle samples one by one, in time order, when perf gave their times.
#[derive(Debug, Default)]
pub struct Samples {
    /// Events per sample (`perf record -c`).
    pub period: u64,
    pub events: BTreeMap<String, HashMap<String, BTreeMap<u64, u64>>>,
    /// Samples with no symbol.
    pub unknown: u64,
    /// The cycle samples in time order: `(time in ns, (symbol, offset))`,
    /// with no symbol for a sample perf could not place.
    pub timed: Vec<(u64, Option<(String, u64)>)>,
}

impl Samples {
    /// Read `perf script --no-demangle -F [time,]event,ip,sym,symoff [--ns]`
    /// output.
    pub fn from_perf_script(text: &str, period: u64) -> Samples {
        let mut samples = Samples {
            period,
            ..Samples::default()
        };
        for line in text.lines() {
            let mut tokens = line.split_whitespace().peekable();
            let time = tokens.peek().and_then(|first| perf_time(first));
            if time.is_some() {
                tokens.next();
            }
            let Some(event) = tokens.next() else {
                continue;
            };
            // `cycles:u:` or `armv8_pmuv3_1/cycles/u:` → `cycles`
            let event = event.trim_end_matches(':');
            let event = match event.split('/').nth(1) {
                Some(name) => name.to_string(),
                None => event.split(':').next().unwrap_or(event).to_string(),
            };
            let at = tokens.find_map(|token| {
                let (symbol, offset) = token.rsplit_once("+0x")?;
                Some((symbol.to_string(), u64::from_str_radix(offset, 16).ok()?))
            });
            if let Some(time) = time
                && event == CYCLES
            {
                samples.timed.push((time, at.clone()));
            }
            let Some((symbol, offset)) = at else {
                samples.unknown += 1;
                continue;
            };
            *samples
                .events
                .entry(event)
                .or_default()
                .entry(symbol)
                .or_default()
                .entry(offset)
                .or_default() += 1;
        }
        samples
    }

    fn count(&self, event: &str) -> u64 {
        self.events.get(event).map_or(0, |funcs| {
            funcs.values().flat_map(|offs| offs.values()).sum()
        })
    }
}

/// A perf time stamp (`seconds.fraction:`) in ns, if `token` is one.
fn perf_time(token: &str) -> Option<u64> {
    let (seconds, fraction) = token.strip_suffix(':')?.split_once('.')?;
    if fraction.is_empty() || fraction.len() > 9 || !fraction.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let scale = 10u64.pow(9 - fraction.len() as u32);
    Some(seconds.parse::<u64>().ok()? * 1_000_000_000 + fraction.parse::<u64>().ok()? * scale)
}

/// What [attach] matched.
#[derive(Debug, Default, serde::Serialize)]
pub struct Attached {
    /// Samples per event on the module's functions.
    pub matched: BTreeMap<String, u64>,
    /// Samples per event in total (other code of the program included).
    pub total: BTreeMap<String, u64>,
}

/// An op and its `[start, end)` byte range in its function.
type OpBytes = (Ptr<Operation>, u64, u64);

/// A function's blocks as the encoder laid them out: per block, its ops with
/// their bytes, and its size.
fn layout(ctx: &Context, func: Ptr<Operation>) -> Vec<(Vec<OpBytes>, u64)> {
    let Some(region) = func.deref(ctx).regions().next() else {
        return Vec::new();
    };
    let mut offset = 0u64;
    let mut blocks = Vec::new();
    for block in region.deref(ctx).iter(ctx) {
        let mut ops = Vec::new();
        let mut bytes_in_block = 0;
        for op in block.deref(ctx).iter(ctx) {
            let obj = Operation::get_op_dyn(op, ctx);
            let len = op_cast::<dyn BinarySerializableOpInterface>(&*obj)
                .map_or(0, |serializable| serializable.byte_len(ctx));
            ops.push((op, offset, offset + len));
            offset += len;
            bytes_in_block += len;
        }
        blocks.push((ops, bytes_in_block));
    }
    blocks
}

/// The block (by index) and op whose bytes hold byte `at` of the function.
fn find_op(blocks: &[(Vec<OpBytes>, u64)], at: u64) -> Option<(usize, Ptr<Operation>)> {
    blocks.iter().enumerate().find_map(|(bi, (ops, _))| {
        ops.iter()
            .find(|&&(_, start, end)| start <= at && at < end)
            .map(|&(op, _, _)| (bi, op))
    })
}

/// The cycle samples one by one, on the ops of a snapshot: when each was
/// taken and the line of the op it hit.
#[derive(Debug, Default, serde::Serialize)]
pub struct Timeline {
    /// Events per sample.
    pub period: u64,
    /// The first sample's time, in ns, by the clock perf records.
    pub start: u64,
    /// Each sample's time after the one before (the first's: 0), in ns.
    pub time: Vec<u64>,
    /// The line of the op each sample hit; 0 for code outside the module.
    pub line: Vec<u32>,
}

/// The timeline of `samples` on the ops of `root`, a module after the
/// encoder (whose ops carry their positions in the snapshot).
pub fn timeline(ctx: &Context, root: Ptr<Operation>, samples: &Samples) -> Timeline {
    let mut layouts: HashMap<String, Vec<(Vec<OpBytes>, u64)>> = HashMap::new();
    for func in ops_under(ctx, root) {
        let obj = Operation::get_op_dyn(func, ctx);
        if let Some(func_op) = obj.downcast_ref::<FuncOp>() {
            layouts.insert(func_op.get_symbol_name(ctx).to_string(), layout(ctx, func));
        }
    }
    let mut timeline = Timeline {
        period: samples.period,
        start: samples.timed.first().map_or(0, |&(time, _)| time),
        ..Timeline::default()
    };
    let mut previous = timeline.start;
    for (time, at) in &samples.timed {
        timeline.time.push(time.saturating_sub(previous));
        previous = *time;
        let line = at
            .as_ref()
            .and_then(|(symbol, offset)| find_op(layouts.get(symbol)?, *offset))
            .and_then(|(_, op)| pliron_bpgo::walk::position(ctx, op))
            .map_or(0, |(line, _)| line);
        timeline.line.push(line);
    }
    timeline
}

/// Put `samples` on the ops of `root`, a module after the encoder:
/// `cycles` on each op from the cycle samples at its bytes, and
/// `executions` of each block (on its ops) from the instruction samples —
/// a block of `n` instructions run `e` times retires `n · e` of them.
pub fn attach(ctx: &Context, root: Ptr<Operation>, samples: &Samples) -> Attached {
    let mut attached = Attached::default();
    for event in samples.events.keys() {
        attached.total.insert(event.clone(), samples.count(event));
    }
    let period = samples.period as f64;
    for func in ops_under(ctx, root) {
        let op_obj = Operation::get_op_dyn(func, ctx);
        let Some(func_op) = op_obj.downcast_ref::<FuncOp>() else {
            continue;
        };
        let symbol = func_op.get_symbol_name(ctx).to_string();
        let blocks = layout(ctx, func);
        let find = |at: u64| find_op(&blocks, at);
        if let Some(cycles) = samples
            .events
            .get("cycles")
            .and_then(|funcs| funcs.get(&symbol))
        {
            for (&at, &count) in cycles {
                if let Some((_, op)) = find(at) {
                    profile::add(ctx, op, CYCLES, count as f64 * period);
                    *attached.matched.entry("cycles".into()).or_default() += count;
                }
            }
        }
        if let Some(instructions) = samples
            .events
            .get("instructions")
            .and_then(|funcs| funcs.get(&symbol))
        {
            let mut per_block = vec![0u64; blocks.len()];
            for (&at, &count) in instructions {
                if let Some((bi, _)) = find(at) {
                    per_block[bi] += count;
                    *attached.matched.entry("instructions".into()).or_default() += count;
                }
            }
            for ((ops, bytes), count) in blocks.iter().zip(per_block) {
                // aarch64 instructions are 4 bytes.
                let instructions = (*bytes / 4).max(1) as f64;
                let executions = count as f64 * period / instructions;
                for &(op, _, _) in ops {
                    profile::add(ctx, op, EXECUTIONS, executions);
                }
            }
        }
    }
    attached
}

fn machine_key(ctx: &Context, op: Ptr<Operation>) -> String {
    let derived = opmap::derived_from(ctx, op).map_or("-".to_string(), |d| d.to_string());
    format!("{}|{derived}", Operation::get_opid(op, ctx))
}

/// The metric a root's cost goes under, on the function.
fn root_metric(root: i64) -> &'static str {
    match root {
        roots::ISEL_ABI => "abi",
        roots::REGALLOC => "regalloc",
        roots::FRAME => "frame",
        roots::MIDEND => "midend",
        _ => "unattributed",
    }
}

/// `what.cycles` of `from` on the function `func`.
fn function_cost(ctx: &Context, func: Ptr<Operation>, what: &str, from: &ProfileAttr) {
    if let Some(cycles) = from.0.get(CYCLES) {
        profile::add(ctx, func, &format!("{what}.cycles"), *cycles);
    }
}

/// The backward pass of a machine pass between instruction selection and
/// the encoder.
pub struct MachineBackward {
    name: &'static str,
    /// For a pass whose created code is a decision about the op it was
    /// stamped from (the allocator's spill code): its metric name, so that
    /// cost goes to that op as `<own>.cycles` instead of plain `cycles`.
    own: Option<&'static str>,
}

impl BackwardPass for MachineBackward {
    fn name(&self) -> &str {
        self.name
    }

    fn run(
        &self,
        ctx: &mut Context,
        input: Ptr<Operation>,
        output: Ptr<Operation>,
    ) -> Result<(), String> {
        let ctx = &*ctx;
        let alignment = align::align(ctx, input, output, &machine_key);
        align::transfer(ctx, &alignment);
        let deleted: std::collections::HashSet<_> = alignment.deleted.iter().copied().collect();
        let mut kept_by_derived: HashMap<i64, Vec<Ptr<Operation>>> = HashMap::new();
        for op in ops_under(ctx, input).into_iter().skip(1) {
            if let Some(d) = opmap::derived_from(ctx, op)
                && !deleted.contains(&op)
            {
                kept_by_derived.entry(d).or_default().push(op);
            }
        }
        for &op in &alignment.created {
            let Some(p) = profile::get(ctx, op) else {
                continue;
            };
            let derived = opmap::derived_from(ctx, op);
            if let Some(root) = derived.filter(|&d| d < 0) {
                function_cost(ctx, input, root_metric(root), &p);
                continue;
            }
            // In place of what the pass took out at the same spot.
            let replaced: Vec<_> = alignment
                .replaced_by(op)
                .iter()
                .copied()
                .filter(|&old| {
                    let old_derived = opmap::derived_from(ctx, old);
                    derived.is_none() || old_derived.is_none() || old_derived == derived
                })
                .collect();
            if !replaced.is_empty() {
                profile::spread(ctx, &replaced, &p);
                continue;
            }
            // Stamped from an op that is still there.
            if let Some(owners) = derived.and_then(|d| kept_by_derived.get(&d)) {
                match self.own {
                    Some(own) => {
                        if let Some(cycles) = p.0.get(CYCLES) {
                            let share = cycles / owners.len() as f64;
                            for &owner in owners {
                                profile::add(ctx, owner, &format!("{own}.cycles"), share);
                            }
                        }
                    }
                    None => profile::spread(ctx, owners, &p),
                }
                continue;
            }
            function_cost(ctx, input, "unattributed", &p);
        }
        Ok(())
    }
}

/// Instruction selection: each machine op's `derived_from` is the id of
/// the LLVM op it was selected from (unique at that boundary); roots are
/// the selector's own code (`abi.cycles`).
pub struct IselBackward;

impl BackwardPass for IselBackward {
    fn name(&self) -> &str {
        "llvm-to-aarch64-isel"
    }

    fn run(
        &self,
        ctx: &mut Context,
        input: Ptr<Operation>,
        output: Ptr<Operation>,
    ) -> Result<(), String> {
        let ctx = &*ctx;
        let by_id: HashMap<u32, Ptr<Operation>> = ops_under(ctx, input)
            .into_iter()
            .skip(1)
            .filter_map(|op| opmap::op_id(ctx, op).map(|id| (id, op)))
            .collect();
        for op in ops_under(ctx, output).into_iter().skip(1) {
            let Some(p) = profile::get(ctx, op) else {
                continue;
            };
            match opmap::derived_from(ctx, op) {
                Some(d) if d >= 0 => match u32::try_from(d).ok().and_then(|id| by_id.get(&id)) {
                    Some(&source) => profile::add_all(ctx, source, &p, 1.0),
                    None => function_cost(ctx, input, "unattributed", &p),
                },
                Some(root) => function_cost(ctx, input, root_metric(root), &p),
                None => function_cost(ctx, input, "unattributed", &p),
            }
        }
        Ok(())
    }
}

/// The snapshot the register allocator receives (blocks in the order the
/// frequency models index them): where profile consumers read a profiled
/// trace.
pub const ALLOCATOR_INPUT: &str = "aarch64-blockmap-ids";

/// A snapshot of a profiled trace (one `crabbit-bpgo` wrote), parsed, with
/// its profile back on its ops.
pub struct ProfiledSnapshot {
    pub ctx: Context,
    pub root: Ptr<Operation>,
    /// Events per sample of the measurement.
    pub period: u64,
}

/// The snapshot after `pass` in the profiled trace at `path`.
pub fn profiled_snapshot(path: &std::path::Path, pass: &str) -> Result<ProfiledSnapshot, String> {
    let trace = pliron_trace::Trace::read(path).map_err(|e| format!("{e:#}"))?;
    let bpgo = trace
        .meta
        .extra
        .get("bpgo")
        .ok_or("the trace has no profile (it was not written by crabbit-bpgo)")?;
    let index = trace
        .snapshots
        .iter()
        .rposition(|snapshot| snapshot.label == pass)
        .ok_or_else(|| format!("the trace has no snapshot after {pass}"))?;
    let entries = bpgo["snapshots"][index.to_string()]
        .as_array()
        .ok_or_else(|| format!("the profile does not reach the IR after {pass}"))?;
    let mut by_position: HashMap<(u32, u32), ProfileAttr> = HashMap::new();
    for entry in entries {
        let (Some(line), Some(column), Some(metrics)) =
            (entry[0].as_u64(), entry[1].as_u64(), entry[2].as_object())
        else {
            continue;
        };
        let metrics = metrics
            .iter()
            .filter_map(|(name, value)| Some((name.clone(), value.as_f64()?)))
            .collect();
        by_position.insert((line as u32, column as u32), ProfileAttr(metrics));
    }
    let mut ctx = Context::new();
    crate::aarch64::register(&mut ctx);
    let root = pliron_bpgo::walk::parse(&mut ctx, &trace.snapshots[index].ir)?;
    for op in ops_under(&ctx, root) {
        if let Some(position) = pliron_bpgo::walk::position(&ctx, op)
            && let Some(profile) = by_position.get(&position)
        {
            profile::set(&ctx, op, profile.clone());
        }
    }
    let period = bpgo["measurement"]["period"].as_u64().unwrap_or(1).max(1);
    Ok(ProfiledSnapshot { ctx, root, period })
}

/// The machine functions of `snapshot` by symbol.
fn machine_functions(snapshot: &ProfiledSnapshot) -> Vec<(String, Ptr<Operation>)> {
    let ctx = &snapshot.ctx;
    ops_under(ctx, snapshot.root)
        .into_iter()
        .filter_map(|op| {
            let obj = Operation::get_op_dyn(op, ctx);
            let func = obj.downcast_ref::<FuncOp>()?;
            Some((func.get_symbol_name(ctx).to_string(), op))
        })
        .collect()
}

/// Per function: how often each block ran (the largest [EXECUTIONS] of its
/// ops; 0 for a block nothing was measured in), in region order.
pub fn block_executions(snapshot: &ProfiledSnapshot) -> HashMap<String, Vec<f64>> {
    let ctx = &snapshot.ctx;
    machine_functions(snapshot)
        .into_iter()
        .filter_map(|(symbol, func)| {
            let region = func.deref(ctx).regions().next()?;
            let runs = region
                .deref(ctx)
                .iter(ctx)
                .map(|block| {
                    block
                        .deref(ctx)
                        .iter(ctx)
                        .filter_map(|op| profile::get(ctx, op)?.0.get(EXECUTIONS).copied())
                        .fold(0.0, f64::max)
                })
                .collect();
            Some((symbol, runs))
        })
        .collect()
}

/// Per function: the samples (cycles / period) of the code derived from
/// each LLVM op, by its `derived_from` id (roots included) — the op's own
/// code and the allocator's code for it (`regalloc.cycles`).
pub fn samples_by_derived(snapshot: &ProfiledSnapshot) -> HashMap<String, HashMap<i64, u64>> {
    let ctx = &snapshot.ctx;
    let period = snapshot.period as f64;
    machine_functions(snapshot)
        .into_iter()
        .map(|(symbol, func)| {
            let mut by_id: HashMap<i64, f64> = HashMap::new();
            for op in ops_under(ctx, func).into_iter().skip(1) {
                let (Some(id), Some(p)) = (opmap::derived_from(ctx, op), profile::get(ctx, op))
                else {
                    continue;
                };
                let cycles = p.0.get(CYCLES).copied().unwrap_or(0.0)
                    + p.0.get("regalloc.cycles").copied().unwrap_or(0.0);
                *by_id.entry(id).or_default() += cycles;
            }
            let samples = by_id
                .into_iter()
                .map(|(id, cycles)| (id, (cycles / period).round() as u64))
                .filter(|&(_, samples)| samples > 0)
                .collect();
            (symbol, samples)
        })
        .collect()
}

impl MachineBackward {
    /// The backward pass of the machine pass `name`; `own` names the
    /// decision metric its created code goes to (see the field).
    pub fn new(name: &'static str, own: Option<&'static str>) -> Self {
        MachineBackward { name, own }
    }
}

/// The backward passes of the machine pipeline, instruction selection
/// included.
pub fn register(passes: &mut BackwardPasses) {
    passes.add(IselBackward);
    for name in [
        "aarch64-legalize",
        "aarch64-machine-cfg-cleanup",
        "aarch64-target-opts-pre-ra",
        "aarch64-blockmap-ids",
        "aarch64-frame-lower",
        "aarch64-post-ra-opts",
        "aarch64-block-placement",
        "aarch64-branch-relax",
        "aarch64-asm-lower",
        "aarch64-encode",
    ] {
        passes.add(MachineBackward { name, own: None });
    }
    passes.add(MachineBackward {
        name: "aarch64-register-allocate",
        own: Some("regalloc"),
    });
}
