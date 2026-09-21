use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::{
    codegen_opts::{BlockFreqModel, CodegenOpts, RestoreEstimate, SpillVictimPolicy},
    context::{Context, Ptr},
    dialects::{
        aarch64::{
            op_interfaces::RegisterOperandKind,
            ops::{self as aarch64_ops, FuncOp},
            registers::{Register, RegisterClass, VirtualRegister},
        },
        builtin::op_interfaces::{OneRegionInterface, SymbolOpInterface},
    },
    ir::{basic_block::BasicBlock, operation::Operation},
    linked_list::ContainsLinkedList,
    conversion::pass::{AnalysisManager, Pass, PassResult, changed},
    passes::spectral_freq::spectral_frequencies,
    result::CrabbitResult,
};

use super::{error::Aarch64Err, frontend::module_op, util::cast_operation};

/// Caller-saved GPRs, tried first: they cost nothing in the prologue but
/// calls clobber them, so only values that do not cross a call may use them.
const ALLOCATABLE_GPRS: [Register; 4] = [
    Register::gpr(9),
    Register::gpr(10),
    Register::gpr(11),
    Register::gpr(12),
];
/// Callee-saved GPRs (AAPCS x19-x28): preserved across calls. A value live
/// across a call allocates here instead of force-spilling; each register the
/// function actually uses is saved/restored by the frame-lowering pass (the
/// allocator records them in the func's `saved_regs` attribute).
const CALLEE_SAVED_GPRS: [Register; 10] = [
    Register::gpr(19),
    Register::gpr(20),
    Register::gpr(21),
    Register::gpr(22),
    Register::gpr(23),
    Register::gpr(24),
    Register::gpr(25),
    Register::gpr(26),
    Register::gpr(27),
    Register::gpr(28),
];
const SPILL_SCRATCH_GPRS: [Register; 3] = [Register::gpr(13), Register::gpr(14), Register::gpr(15)];
/// Caller-saved FP pool: d16-d19 and d23-d31 (d20-d22 are spill scratch,
/// d0-d7 are argument/result registers).
const ALLOCATABLE_FPR_NUMBERS: [u8; 13] = [16, 17, 18, 19, 23, 24, 25, 26, 27, 28, 29, 30, 31];
/// Callee-saved FP registers: AAPCS preserves the low 64 bits of v8-v15,
/// which is exactly the d (and s) view this backend uses.
const CALLEE_SAVED_FPR_NUMBERS: [u8; 8] = [8, 9, 10, 11, 12, 13, 14, 15];
const SPILL_SCRATCH_FPR_NUMBERS: [u8; 3] = [20, 21, 22];
const SPILL_SLOT_BYTES: u64 = 8;

/// The register bank an interval allocates from. `Fpr32`, `Fpr64` and
/// `Simd128` virtual registers all share the `Fpr` bank: an `s`/`d`/`q`
/// register of one number is ONE physical allocation unit (`q<n>` aliases
/// `d<n>`), so keeping them in the same pool makes d/q overlap impossible
/// by construction. The vreg's own class picks the spelling at rewrite and
/// the spill-slot size; `Simd128` additionally may never take a
/// callee-saved index (AAPCS preserves only the low 64 bits of v8–v15).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum Bank {
    Gpr,
    Fpr,
}

fn bank_of(class: RegisterClass) -> Option<Bank> {
    match class {
        RegisterClass::Gpr64 => Some(Bank::Gpr),
        RegisterClass::Fpr64 | RegisterClass::Fpr32 | RegisterClass::Simd128 => Some(Bank::Fpr),
        _ => None,
    }
}

/// Pool layout per bank: indices `0..caller_pool_size` are the caller-saved
/// registers, the rest are callee-saved. The free list is kept sorted, so
/// picking the smallest free index prefers caller-saved registers (no
/// prologue cost) and, past those, reuses the same few callee-saved
/// registers (each distinct one costs a save/restore pair).
fn caller_pool_size(bank: Bank) -> usize {
    match bank {
        Bank::Gpr => ALLOCATABLE_GPRS.len(),
        Bank::Fpr => ALLOCATABLE_FPR_NUMBERS.len(),
    }
}

fn pool_size(bank: Bank) -> usize {
    match bank {
        Bank::Gpr => ALLOCATABLE_GPRS.len() + CALLEE_SAVED_GPRS.len(),
        Bank::Fpr => ALLOCATABLE_FPR_NUMBERS.len() + CALLEE_SAVED_FPR_NUMBERS.len(),
    }
}

fn is_callee_saved_index(bank: Bank, phys_index: usize) -> bool {
    phys_index >= caller_pool_size(bank)
}

/// The physical register for `phys_index` in `bank`, spelled in the vreg's
/// own class.
fn pool_register(bank: Bank, phys_index: usize, class: RegisterClass) -> Register {
    let number = match bank {
        Bank::Gpr => {
            return if phys_index < ALLOCATABLE_GPRS.len() {
                ALLOCATABLE_GPRS[phys_index]
            } else {
                CALLEE_SAVED_GPRS[phys_index - ALLOCATABLE_GPRS.len()]
            };
        }
        Bank::Fpr => {
            if phys_index < ALLOCATABLE_FPR_NUMBERS.len() {
                ALLOCATABLE_FPR_NUMBERS[phys_index]
            } else {
                CALLEE_SAVED_FPR_NUMBERS[phys_index - ALLOCATABLE_FPR_NUMBERS.len()]
            }
        }
    };
    match class {
        RegisterClass::Fpr32 => Register::fpr32(number),
        RegisterClass::Simd128 => Register::simd128(number),
        _ => Register::fpr64(number),
    }
}

/// The `saved_regs` attribute bit for a callee-saved pool register:
/// bits 0-31 are x0-x31, bits 32-63 are d0-d31.
fn saved_reg_mask_bit(bank: Bank, phys_index: usize) -> u64 {
    match bank {
        Bank::Gpr => {
            let Register::Physical(
                crate::dialects::aarch64::registers::PhysicalRegister::Gpr64(number),
            ) = CALLEE_SAVED_GPRS[phys_index - ALLOCATABLE_GPRS.len()]
            else {
                unreachable!("callee-saved GPR pool holds physical x registers");
            };
            1u64 << number
        }
        Bank::Fpr => {
            1u64 << (32 + CALLEE_SAVED_FPR_NUMBERS[phys_index - ALLOCATABLE_FPR_NUMBERS.len()])
        }
    }
}

pub struct Aarch64RegisterAllocatePass;

impl Pass for Aarch64RegisterAllocatePass {
    fn name(&self) -> &str {
        "aarch64-register-allocate"
    }

    fn run(&mut self, root: Ptr<Operation>, ctx: &mut Context, _analyses: &mut AnalysisManager) -> pliron::result::Result<PassResult> {
        let opts = CodegenOpts::from_env()?;
        if opts.freq != BlockFreqModel::Uniform && opts.victim != SpillVictimPolicy::WeightedCost {
            // The linear allocator only reads frequencies through the
            // weighted victim policy; say so once instead of silently
            // ignoring a configured profile/spectral source.
            static NOTE: std::sync::Once = std::sync::Once::new();
            NOTE.call_once(|| {
                let source = match opts.freq {
                    BlockFreqModel::Uniform => "uniform",
                    BlockFreqModel::Spectral => "spectral",
                    BlockFreqModel::SpectralLoop => "spectral-loop",
                    BlockFreqModel::Profile => "profile",
                };
                // eprintln! rather than log::warn!: this runs inside a rustc codegen
                // dylib where no logger is installed; the warning must reach the user.
                eprintln!(
                    "crabbit: note: CRABBIT_BLOCK_FREQ={source} has no effect under the linear \
                     allocator's default furthest-end policy; set CRABBIT_SPILL_POLICY=weighted \
                     to consume block frequencies"
                );
            });
        }
        let module = module_op(ctx, root)?;
        let body = module.get_region(ctx).deref(ctx).get_head().unwrap();
        let funcs: Vec<_> = body.deref(ctx).iter(ctx).collect();
        for op in funcs {
            if let Some(func) = cast_operation::<FuncOp>(ctx, op) {
                allocate_function(ctx, func, &opts)?;
            }
        }
        Ok(changed())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct LiveInterval {
    vreg: VirtualRegister,
    start: usize,
    end: usize,
}

#[derive(Clone, Debug)]
struct ActiveInterval {
    interval: LiveInterval,
    phys_index: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum Allocation {
    Phys(usize),
    Spill(u64),
}

/// Per-function liveness facts. `intervals` drives the scan; the position
/// maps and CFG shape feed the optional cost-model policies.
struct FunctionLiveness {
    insts: Vec<Ptr<Operation>>,
    intervals: Vec<LiveInterval>,
    use_positions: HashMap<VirtualRegister, Vec<usize>>,
    def_positions: HashMap<VirtualRegister, Vec<usize>>,
    /// The register class of each virtual register (from its operands; a
    /// vreg has one class by construction).
    classes: HashMap<VirtualRegister, RegisterClass>,
    /// Block index of each instruction position.
    inst_block: Vec<usize>,
    /// Successor block indices, entry block first.
    successors: Vec<Vec<usize>>,
}

fn allocate_function(ctx: &mut Context, func: FuncOp, opts: &CodegenOpts) -> CrabbitResult<()> {
    let live = collect_live_intervals(ctx, func);
    let call_crossing = values_live_across_calls(ctx, &live.insts, &live.intervals);
    let remat_imms = rematerializable_imms(ctx, &live, opts);
    let symbol = func.get_symbol_name(ctx).to_string();
    dump_spectral_freqs(&live, &symbol);
    let weights = spill_weights(&live, &remat_imms, opts, &symbol);
    let allocation = linear_scan(&live.intervals, &live.classes, &call_crossing, weights.as_ref());
    let base_stack_size = func.stack_size(ctx);
    rewrite_allocated_registers(
        ctx,
        &live.insts,
        &allocation.assignments,
        base_stack_size,
        &remat_imms,
    )?;
    func.set_stack_size(
        ctx,
        align_to_16(base_stack_size + allocation.spill_slots * SPILL_SLOT_BYTES),
    );
    if allocation.saved_regs_mask != 0 {
        func.set_saved_regs(ctx, allocation.saved_regs_mask);
    }
    Ok(())
}

/// Experiment E4 support (docs/PROFILE-FEEDBACK-BACKWARD.md): when
/// `CRABBIT_DUMP_FREQS=<path>` is set, append one JSON line per function
/// with the analytic spectral frequency vector over the allocator's own
/// fallthrough-aware machine CFG, in RA block order — the same order the
/// blockmap ids and measured profiles use, so model and measurement join
/// by index. Off by default; never affects allocation.
fn dump_spectral_freqs(live: &FunctionLiveness, symbol: &str) {
    let Ok(path) = std::env::var("CRABBIT_DUMP_FREQS") else {
        return;
    };
    if path.is_empty() {
        return;
    }
    let freqs = spectral_frequencies(&live.successors, 0);
    let entries: Vec<String> = freqs.iter().map(|f| format!("{f}")).collect();
    let line = format!(
        "{{\"symbol\":\"{}\",\"spectral\":[{}]}}\n",
        symbol,
        entries.join(",")
    );
    use std::io::Write as _;
    if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        let _ = file.write_all(line.as_bytes());
    }
}

/// Restore-cost units: a reload is priced like a memory access, a trivial
/// rematerialization like a single ALU op (the eregalloc cost shape).
const RELOAD_COST: f64 = 4.0;
const REMAT_COST: f64 = 1.0;

/// Immediates of single-def `mov_imm` vregs: trivially rematerializable
/// (Briggs / LLVM `isTriviallyReMaterializable` scope). Empty unless the
/// remat estimate is selected.
fn rematerializable_imms(
    ctx: &Context,
    live: &FunctionLiveness,
    opts: &CodegenOpts,
) -> HashMap<VirtualRegister, u64> {
    if opts.restore != RestoreEstimate::TrivialRemat {
        return HashMap::new();
    }
    live.def_positions
        .iter()
        .filter_map(|(vreg, defs)| {
            let [def] = defs.as_slice() else { return None };
            let op = live.insts[*def];
            (aarch64_ops::opcode(ctx, op) == Some(aarch64_ops::MovImmOp::OPCODE))
                .then(|| aarch64_ops::imm(ctx, op).map(|imm| (*vreg, imm)))
                .flatten()
        })
        .collect()
}

/// One frequency per block for the weighted-cost policy, per the selected
/// [BlockFreqModel]. `profile` looks the function's `symbol` up in the
/// `CRABBIT_PROFILE` JSON (RA-order indices — the very order
/// `live.successors` is in) and falls back to uniform when the function is
/// missing, the vector length disagrees with this build's CFG, or the
/// values are unusable: a stale profile must never error.
fn block_frequencies(live: &FunctionLiveness, opts: &CodegenOpts, symbol: &str) -> Vec<f64> {
    let blocks = live.successors.len();
    match opts.freq {
        BlockFreqModel::Uniform => vec![1.0; blocks],
        BlockFreqModel::Spectral => spectral_frequencies(&live.successors, 0),
        BlockFreqModel::SpectralLoop => {
            crate::passes::spectral_freq::spectral_frequencies_loop_aware(&live.successors, 0)
        }
        BlockFreqModel::Profile => crate::passes::profile_freq::frequencies_for(symbol, blocks)
            .unwrap_or_else(|| vec![1.0; blocks]),
    }
}

/// Spill weights for the weighted-cost victim policy:
/// `restore_estimate × Σ freq(use block)`. Lower weight → cheaper to
/// spill. `None` selects the baseline furthest-end policy.
fn spill_weights(
    live: &FunctionLiveness,
    remat_imms: &HashMap<VirtualRegister, u64>,
    opts: &CodegenOpts,
    symbol: &str,
) -> Option<HashMap<VirtualRegister, f64>> {
    if opts.victim != SpillVictimPolicy::WeightedCost {
        return None;
    }
    let freqs = block_frequencies(live, opts, symbol);
    let weights = live
        .intervals
        .iter()
        .map(|interval| {
            let vreg = interval.vreg;
            let restore = if remat_imms.contains_key(&vreg) {
                REMAT_COST
            } else {
                RELOAD_COST
            };
            let executed_uses: f64 = live
                .use_positions
                .get(&vreg)
                .map(|uses| {
                    uses.iter()
                        .map(|&position| freqs[live.inst_block[position]])
                        .sum()
                })
                .unwrap_or(0.0);
            (vreg, restore * executed_uses)
        })
        .collect();
    Some(weights)
}

fn collect_live_intervals(ctx: &Context, func: FuncOp) -> FunctionLiveness {
    let blocks: Vec<_> = func.get_region(ctx).deref(ctx).iter(ctx).collect();
    let block_index: HashMap<_, _> = blocks
        .iter()
        .enumerate()
        .map(|(index, block)| (*block, index))
        .collect();
    let block_insts: Vec<Vec<_>> = blocks
        .iter()
        .map(|block| {
            block
                .deref(ctx)
                .iter(ctx)
                .filter(|op| aarch64_ops::is_instruction(ctx, *op))
                .collect()
        })
        .collect();
    let insts = block_insts.iter().flatten().copied().collect::<Vec<_>>();

    let mut block_uses = vec![BTreeSet::new(); blocks.len()];
    let mut block_defs = vec![BTreeSet::new(); blocks.len()];
    for (index, insts) in block_insts.iter().enumerate() {
        for op in insts {
            if aarch64_ops::is_instruction(ctx, *op) {
                for reg in virtual_uses(ctx, *op) {
                    if !block_defs[index].contains(&reg) {
                        block_uses[index].insert(reg);
                    }
                }
                block_defs[index].extend(virtual_defs(ctx, *op));
            }
        }
    }

    let successors = block_successors(ctx, &block_insts, &block_index);
    let mut live_in = vec![BTreeSet::new(); blocks.len()];
    let mut live_out = vec![BTreeSet::new(); blocks.len()];
    loop {
        let mut changed = false;
        for index in (0..blocks.len()).rev() {
            let new_out = successors[index]
                .iter()
                .flat_map(|successor| live_in[*successor].iter().copied())
                .collect::<BTreeSet<_>>();
            let new_in = block_uses[index]
                .union(
                    &new_out
                        .difference(&block_defs[index])
                        .copied()
                        .collect::<BTreeSet<_>>(),
                )
                .copied()
                .collect::<BTreeSet<_>>();
            if new_out != live_out[index] || new_in != live_in[index] {
                live_out[index] = new_out;
                live_in[index] = new_in;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    // The current allocator consumes conventional intervals. Project the CFG
    // liveness sets to conservative contiguous intervals, so values live on a
    // backedge or through a join cannot be co-allocated accidentally.
    let mut points = BTreeMap::<VirtualRegister, (usize, usize)>::new();
    let mut next_index = 0usize;
    for (block_index, insts) in block_insts.iter().enumerate() {
        let mut live = live_out[block_index].clone();
        for op in insts.iter().rev() {
            if !aarch64_ops::is_instruction(ctx, *op) {
                continue;
            }
            let index = next_index + insts.iter().position(|candidate| candidate == op).unwrap();
            for reg in live
                .iter()
                .copied()
                .chain(virtual_uses(ctx, *op))
                .chain(virtual_defs(ctx, *op))
            {
                points
                    .entry(reg)
                    .and_modify(|(start, end)| {
                        *start = (*start).min(index);
                        *end = (*end).max(index);
                    })
                    .or_insert((index, index));
            }
            for reg in virtual_defs(ctx, *op) {
                live.remove(&reg);
            }
            live.extend(virtual_uses(ctx, *op));
        }
        next_index += insts.len();
    }

    let mut intervals = points
        .into_iter()
        .map(|(vreg, (start, end))| LiveInterval { vreg, start, end })
        .collect::<Vec<_>>();
    intervals.sort_by(|lhs, rhs| {
        lhs.start
            .cmp(&rhs.start)
            .then(lhs.end.cmp(&rhs.end))
            .then(lhs.vreg.cmp(&rhs.vreg))
    });

    let mut use_positions = HashMap::<VirtualRegister, Vec<usize>>::new();
    let mut def_positions = HashMap::<VirtualRegister, Vec<usize>>::new();
    let mut inst_block = Vec::with_capacity(insts.len());
    for (index, insts) in block_insts.iter().enumerate() {
        inst_block.extend(std::iter::repeat_n(index, insts.len()));
    }
    let mut classes = HashMap::<VirtualRegister, RegisterClass>::new();
    for (position, op) in insts.iter().enumerate() {
        for (_, reg, class) in virtual_use_operands(ctx, *op) {
            use_positions.entry(reg).or_default().push(position);
            classes.insert(reg, class);
        }
        for (_, reg, class) in virtual_def_operands(ctx, *op) {
            def_positions.entry(reg).or_default().push(position);
            classes.insert(reg, class);
        }
    }

    FunctionLiveness {
        insts,
        intervals,
        use_positions,
        def_positions,
        classes,
        inst_block,
        successors,
    }
}

fn block_successors(
    ctx: &Context,
    block_insts: &[Vec<Ptr<Operation>>],
    block_index: &HashMap<Ptr<BasicBlock>, usize>,
) -> Vec<Vec<usize>> {
    block_insts
        .iter()
        .enumerate()
        .map(|(index, insts)| {
            let mut successors = BTreeSet::new();
            let mut has_unconditional_branch = false;
            let mut terminates = false;
            for op in insts {
                let Some(opcode) = aarch64_ops::opcode(ctx, *op) else {
                    continue;
                };
                match opcode {
                    aarch64_ops::BOp::OPCODE => {
                        has_unconditional_branch = true;
                        if let Some(target) = aarch64_ops::target(ctx, *op)
                            .and_then(|target| block_index.get(&target))
                        {
                            successors.insert(*target);
                        }
                    }
                    aarch64_ops::BCondOp::OPCODE | aarch64_ops::CbnzOp::OPCODE => {
                        if let Some(target) = aarch64_ops::target(ctx, *op)
                            .and_then(|target| block_index.get(&target))
                        {
                            successors.insert(*target);
                        }
                    }
                    aarch64_ops::RetOp::OPCODE | aarch64_ops::BrkOp::OPCODE => terminates = true,
                    _ => {}
                }
            }
            if !has_unconditional_branch && !terminates && index + 1 < block_insts.len() {
                successors.insert(index + 1);
            }
            successors.into_iter().collect()
        })
        .collect()
}

fn values_live_across_calls(
    ctx: &Context,
    insts: &[Ptr<Operation>],
    intervals: &[LiveInterval],
) -> BTreeSet<VirtualRegister> {
    let call_indexes: BTreeSet<_> = insts
        .iter()
        .enumerate()
        .filter_map(|(index, op)| {
            matches!(
                aarch64_ops::opcode(ctx, *op),
                Some(aarch64_ops::CallOp::OPCODE) | Some(aarch64_ops::BlrOp::OPCODE)
            )
            .then_some(index)
        })
        .collect();

    intervals
        .iter()
        .filter(|interval| {
            call_indexes
                .iter()
                .any(|call_index| interval.start < *call_index && *call_index < interval.end)
        })
        .map(|interval| interval.vreg)
        .collect()
}

#[derive(Clone, Debug)]
struct AllocationResult {
    assignments: HashMap<VirtualRegister, Allocation>,
    spill_slots: u64,
    /// Callee-saved registers the allocation uses, as the func-level
    /// `saved_regs` bitmask (bits 0-31 x0-x31, bits 32-63 d0-d31).
    saved_regs_mask: u64,
}

/// One linear scan over all intervals, with an independent register pool
/// (free list + active set) per [Bank]. The spill-victim policies compare
/// only intervals competing for the same bank; spill slots are shared.
///
/// Values live across a call (`call_crossing`) may only occupy callee-saved
/// registers — calls clobber the caller-saved pool. When no callee-saved
/// register is free and no eligible victim is worth evicting, the value
/// spills (the pre-callee-saved behavior for every call-crossing value).
fn linear_scan(
    intervals: &[LiveInterval],
    classes: &HashMap<VirtualRegister, RegisterClass>,
    call_crossing: &BTreeSet<VirtualRegister>,
    weights: Option<&HashMap<VirtualRegister, f64>>,
) -> AllocationResult {
    let mut active: HashMap<Bank, Vec<ActiveInterval>> = HashMap::new();
    let mut free: HashMap<Bank, Vec<usize>> = HashMap::new();
    for bank in [Bank::Gpr, Bank::Fpr] {
        active.insert(bank, Vec::new());
        free.insert(bank, (0..pool_size(bank)).collect());
    }
    let mut assignments = HashMap::<VirtualRegister, Allocation>::new();
    let mut spill_slots = 0u64;
    let mut saved_regs_mask = 0u64;

    for interval in intervals {
        let class = *classes
            .get(&interval.vreg)
            .expect("interval for a vreg with no class");
        let bank = bank_of(class).expect("interval for a vreg with no allocatable class");
        let crossing = call_crossing.contains(&interval.vreg);
        // No register can hold a 128-bit value across a call: the whole
        // caller-saved file is clobbered and AAPCS preserves only the low
        // 64 bits of v8–v15. Force-spill (the pre-callee-saved behavior).
        let is_q = class == RegisterClass::Simd128;
        if is_q && crossing {
            let slot = take_spill_slots(&mut spill_slots, class);
            assignments.insert(interval.vreg, Allocation::Spill(slot));
            continue;
        }

        let active = active.get_mut(&bank).unwrap();
        let free = free.get_mut(&bank).unwrap();
        expire_old_intervals(interval.start, active, free);
        free.sort_unstable();
        let free_position = if crossing {
            free.iter()
                .position(|index| is_callee_saved_index(bank, *index))
        } else if is_q {
            // Simd128 values may never sit in a callee-saved register:
            // only its low 64 bits would be preserved/restored.
            free.iter()
                .position(|index| !is_callee_saved_index(bank, *index))
        } else {
            (!free.is_empty()).then_some(0)
        };
        // A call-crossing value can only evict a victim holding a
        // callee-saved register; a Simd128 value only one holding a
        // caller-saved register; anything else must be eligible.
        let eligible = |candidate: &ActiveInterval| {
            if crossing {
                return is_callee_saved_index(bank, candidate.phys_index);
            }
            if is_q {
                return !is_callee_saved_index(bank, candidate.phys_index);
            }
            true
        };
        let phys_index = if let Some(position) = free_position {
            free.remove(position)
        } else if let Some(spilled) = match weights {
            None => spill_furthest_end(interval, active, eligible),
            Some(weights) => spill_cheapest_weight(interval, active, weights, eligible),
        } {
            let victim_class = *classes
                .get(&spilled.interval.vreg)
                .expect("active interval for a vreg with no class");
            let slot = take_spill_slots(&mut spill_slots, victim_class);
            assignments.insert(spilled.interval.vreg, Allocation::Spill(slot));
            spilled.phys_index
        } else {
            let slot = take_spill_slots(&mut spill_slots, class);
            assignments.insert(interval.vreg, Allocation::Spill(slot));
            continue;
        };

        if is_callee_saved_index(bank, phys_index) {
            saved_regs_mask |= saved_reg_mask_bit(bank, phys_index);
        }
        assignments.insert(interval.vreg, Allocation::Phys(phys_index));
        active.push(ActiveInterval {
            interval: interval.clone(),
            phys_index,
        });
        active.sort_by(|lhs, rhs| {
            lhs.interval
                .end
                .cmp(&rhs.interval.end)
                .then(lhs.phys_index.cmp(&rhs.phys_index))
        });
    }

    AllocationResult {
        assignments,
        spill_slots,
        saved_regs_mask,
    }
}

/// Baseline (Poletto–Sarkar): evict the `eligible` active interval that ends
/// furthest away, but only if it outlives `current`; otherwise `current`
/// itself spills.
fn spill_furthest_end(
    current: &LiveInterval,
    active: &mut Vec<ActiveInterval>,
    eligible: impl Fn(&ActiveInterval) -> bool,
) -> Option<ActiveInterval> {
    let spill_index = active
        .iter()
        .enumerate()
        .filter(|(_, candidate)| eligible(candidate))
        .max_by(|(_, lhs), (_, rhs)| {
            lhs.interval
                .end
                .cmp(&rhs.interval.end)
                .then(lhs.phys_index.cmp(&rhs.phys_index))
        })
        .map(|(index, _)| index)?;

    (active[spill_index].interval.end > current.end).then(|| active.remove(spill_index))
}

/// Weighted-cost policy: evict the `eligible` candidate (active or
/// `current`) whose total restore cost — `restore_estimate × Σ freq(use)` —
/// is lowest. Returns `None` when `current` itself is the cheapest to spill.
fn spill_cheapest_weight(
    current: &LiveInterval,
    active: &mut Vec<ActiveInterval>,
    weights: &HashMap<VirtualRegister, f64>,
    eligible: impl Fn(&ActiveInterval) -> bool,
) -> Option<ActiveInterval> {
    let weight_of = |vreg: VirtualRegister| weights.get(&vreg).copied().unwrap_or(0.0);
    let cheapest_active = active
        .iter()
        .enumerate()
        .filter(|(_, candidate)| eligible(candidate))
        .min_by(|(_, lhs), (_, rhs)| {
            weight_of(lhs.interval.vreg)
                .total_cmp(&weight_of(rhs.interval.vreg))
                .then(lhs.interval.vreg.cmp(&rhs.interval.vreg))
        })
        .map(|(index, _)| index)?;

    (weight_of(active[cheapest_active].interval.vreg) < weight_of(current.vreg))
        .then(|| active.remove(cheapest_active))
}

/// Take the spill slot(s) for one value of `class` and return the starting
/// slot index. Scalar classes take one 8-byte slot; `Simd128` takes two
/// consecutive slots starting at an even index, so its 16-byte access is
/// 16-byte aligned (the spill base — the function's pre-RA stack size — is
/// itself 16-aligned).
fn take_spill_slots(spill_slots: &mut u64, class: RegisterClass) -> u64 {
    if class == RegisterClass::Simd128 {
        *spill_slots = (*spill_slots + 1) & !1;
        let slot = *spill_slots;
        *spill_slots += 2;
        slot
    } else {
        let slot = *spill_slots;
        *spill_slots += 1;
        slot
    }
}

fn expire_old_intervals(
    current_start: usize,
    active: &mut Vec<ActiveInterval>,
    free: &mut Vec<usize>,
) {
    let mut retained = Vec::with_capacity(active.len());
    for active_interval in active.drain(..) {
        if active_interval.interval.end < current_start {
            free.push(active_interval.phys_index);
        } else {
            retained.push(active_interval);
        }
    }
    *active = retained;
}

fn rewrite_allocated_registers(
    ctx: &mut Context,
    insts: &[Ptr<Operation>],
    assignments: &HashMap<VirtualRegister, Allocation>,
    spill_base_offset: u64,
    remat_imms: &HashMap<VirtualRegister, u64>,
) -> CrabbitResult<()> {
    for op in insts {
        if !aarch64_ops::is_instruction(ctx, *op) {
            continue;
        }
        let mut scratch_index = [0usize; 2];
        let use_operands = virtual_use_operands(ctx, *op);
        let def_operands = virtual_def_operands(ctx, *op);
        let mut spilled_use_scratch =
            HashMap::<(&'static str, VirtualRegister), Register>::new();

        for (key, vreg, class) in use_operands {
            let bank = bank_of(class).expect("operand collected without an allocatable class");
            match assignments.get(&vreg) {
                Some(Allocation::Phys(phys_index)) => {
                    aarch64_ops::rewrite_register_operand(
                        ctx,
                        *op,
                        key,
                        pool_register(bank, *phys_index, class),
                    );
                }
                Some(Allocation::Spill(slot)) => {
                    let scratch = next_spill_scratch(&mut scratch_index, class)?;
                    // The eregalloc "safe form": the slot store stays (see
                    // the def rewrite below), but a use of a trivially
                    // rematerializable value re-executes its constant def
                    // instead of reloading.
                    // Backward attribution: the restore executes at (and
                    // for) this use — it inherits the use op's source id
                    // (docs/PROFILE-FEEDBACK-BACKWARD.md).
                    let restore = if let Some(imm) = remat_imms.get(&vreg) {
                        aarch64_ops::mov_imm(ctx, scratch, *imm)
                    } else {
                        let (_, reload) = spill_opcodes(class);
                        aarch64_ops::ldr_sp_offset_sized(
                            ctx,
                            reload,
                            scratch,
                            spill_base_offset + slot * SPILL_SLOT_BYTES,
                        )
                    };
                    restore.insert_before(ctx, *op);
                    super::opmap::inherit_derived_from(
                        ctx,
                        *op,
                        restore,
                        super::opmap::roots::REGALLOC,
                    );
                    aarch64_ops::rewrite_register_operand(ctx, *op, key, scratch);
                    spilled_use_scratch.insert((key, vreg), scratch);
                }
                None => {}
            }
        }

        for (key, vreg, class) in def_operands {
            let bank = bank_of(class).expect("operand collected without an allocatable class");
            match assignments.get(&vreg) {
                Some(Allocation::Phys(phys_index)) => {
                    aarch64_ops::rewrite_register_operand(
                        ctx,
                        *op,
                        key,
                        pool_register(bank, *phys_index, class),
                    );
                }
                Some(Allocation::Spill(slot)) => {
                    let scratch = match spilled_use_scratch.get(&(key, vreg)) {
                        Some(scratch) => *scratch,
                        None => next_spill_scratch(&mut scratch_index, class)?,
                    };
                    aarch64_ops::rewrite_register_operand(ctx, *op, key, scratch);
                    let (store, _) = spill_opcodes(class);
                    let spill_store = aarch64_ops::str_sp_offset_sized(
                        ctx,
                        store,
                        scratch,
                        spill_base_offset + slot * SPILL_SLOT_BYTES,
                    );
                    spill_store.insert_after(ctx, *op);
                    // The slot store belongs to the spilled value's def.
                    super::opmap::inherit_derived_from(
                        ctx,
                        *op,
                        spill_store,
                        super::opmap::roots::REGALLOC,
                    );
                }
                None => {}
            }
        }
    }
    Ok(())
}

fn virtual_uses(ctx: &Context, inst: Ptr<Operation>) -> Vec<VirtualRegister> {
    virtual_use_operands(ctx, inst)
        .into_iter()
        .map(|(_, reg, _)| reg)
        .collect()
}

fn virtual_defs(ctx: &Context, inst: Ptr<Operation>) -> Vec<VirtualRegister> {
    virtual_def_operands(ctx, inst)
        .into_iter()
        .map(|(_, reg, _)| reg)
        .collect()
}

fn virtual_use_operands(
    ctx: &Context,
    inst: Ptr<Operation>,
) -> Vec<(&'static str, VirtualRegister, RegisterClass)> {
    virtual_operands_with_kind(ctx, inst, RegisterOperandKind::Use)
}

fn virtual_def_operands(
    ctx: &Context,
    inst: Ptr<Operation>,
) -> Vec<(&'static str, VirtualRegister, RegisterClass)> {
    virtual_operands_with_kind(ctx, inst, RegisterOperandKind::Def)
}

fn virtual_operands_with_kind(
    ctx: &Context,
    inst: Ptr<Operation>,
    kind: RegisterOperandKind,
) -> Vec<(&'static str, VirtualRegister, RegisterClass)> {
    aarch64_ops::register_operands(ctx, inst)
        .into_iter()
        .filter_map(|operand| {
            (operand.kind == kind)
                .then_some(operand)
                .and_then(|operand| match operand.reg {
                    Register::Virtual { id, class } if bank_of(class).is_some() => {
                        Some((operand.key, id, class))
                    }
                    _ => None,
                })
        })
        .collect()
}

/// The next spill scratch register in the bank matching `class`, spelled in
/// that class. GPR and FP operands draw from independent scratch sets, so an
/// instruction mixing the files can spill both sides.
fn next_spill_scratch(
    scratch_index: &mut [usize; 2],
    class: RegisterClass,
) -> CrabbitResult<Register> {
    let bank = bank_of(class).expect("spill scratch requested for unallocatable class");
    let index = match bank {
        Bank::Gpr => &mut scratch_index[0],
        Bank::Fpr => &mut scratch_index[1],
    };
    let scratch = match bank {
        Bank::Gpr => SPILL_SCRATCH_GPRS.get(*index).copied(),
        Bank::Fpr => SPILL_SCRATCH_FPR_NUMBERS.get(*index).map(|number| match class {
            RegisterClass::Fpr32 => Register::fpr32(*number),
            // q20–q22 are the full-vector views of the d20–d22 scratch
            // registers: caller-saved and outside every pool, so the q view
            // is just as free (the shared index keeps d/q scratch of one
            // instruction from colliding on the same number).
            RegisterClass::Simd128 => Register::simd128(*number),
            _ => Register::fpr64(*number),
        }),
    };
    let Some(scratch) = scratch else {
        return Err(crate::input_error_noloc!(Aarch64Err::UnsupportedOp(
            "aarch64 instruction needs more spill scratch registers than are reserved".to_string()
        )));
    };
    *index += 1;
    Ok(scratch)
}

/// The sp-offset spill store/reload opcodes for a register class. FP values
/// spill through FP loads/stores (an f32 still occupies a full 8-byte slot;
/// a q value takes two consecutive 16-aligned slots — see
/// [take_spill_slots]).
fn spill_opcodes(
    class: RegisterClass,
) -> (aarch64_ops::Aarch64Opcode, aarch64_ops::Aarch64Opcode) {
    match class {
        RegisterClass::Simd128 => (
            aarch64_ops::StrqSpOffsetOp::OPCODE,
            aarch64_ops::LdrqSpOffsetOp::OPCODE,
        ),
        RegisterClass::Fpr64 => (
            aarch64_ops::StrdSpOffsetOp::OPCODE,
            aarch64_ops::LdrdSpOffsetOp::OPCODE,
        ),
        RegisterClass::Fpr32 => (
            aarch64_ops::StrsSpOffsetOp::OPCODE,
            aarch64_ops::LdrsSpOffsetOp::OPCODE,
        ),
        _ => (
            aarch64_ops::StrSpOffsetOp::OPCODE,
            aarch64_ops::LdrSpOffsetOp::OPCODE,
        ),
    }
}

fn align_to_16(bytes: u64) -> u64 {
    (bytes + 15) & !15
}

#[cfg(test)]
mod tests {
    use crate::ll::LinkageAttr;
    use crate::{
        dialects::aarch64::{
                self,
                ops::{self as aarch64_ops, ATTR_KEY_AARCH64_RD},
            },
        linked_list::ContainsLinkedList,
    };

    use super::*;

    fn context() -> Context {
        let mut ctx = Context::new();
        aarch64::register(&mut ctx);
        ctx
    }

    fn func(ctx: &mut Context) -> FuncOp {
        FuncOp::new(ctx, "test".try_into().unwrap(), LinkageAttr::External)
    }

    #[test]
    fn linear_scan_reuses_expired_registers() {
        let mut ctx = context();
        let func = func(&mut ctx);
        let entry = func.entry_block(&ctx);
        aarch64_ops::mov_imm(&mut ctx, Register::virtual_gpr(0), 1).insert_at_back(entry, &ctx);
        aarch64_ops::mov(&mut ctx, Register::gpr(0), Register::virtual_gpr(0)).insert_at_back(entry, &ctx);
        aarch64_ops::mov_imm(&mut ctx, Register::virtual_gpr(1), 2).insert_at_back(entry, &ctx);
        aarch64_ops::mov(&mut ctx, Register::gpr(1), Register::virtual_gpr(1)).insert_at_back(entry, &ctx);

        allocate_function(&mut ctx, func, &CodegenOpts::default()).unwrap();
        let insts: Vec<_> = entry.deref(&ctx).iter(&ctx).collect();
        assert_eq!(
            aarch64_ops::reg(&ctx, insts[0], ATTR_KEY_AARCH64_RD.as_ref()).unwrap(),
            Register::gpr(9)
        );
        assert_eq!(
            aarch64_ops::reg(&ctx, insts[2], ATTR_KEY_AARCH64_RD.as_ref()).unwrap(),
            Register::gpr(9)
        );
    }

    /// A value live across a call sits in a callee-saved register instead
    /// of spilling; the allocator records the register for frame lowering.
    #[test]
    fn call_crossing_value_uses_callee_saved_register() {
        let mut ctx = context();
        let func = func(&mut ctx);
        let entry = func.entry_block(&ctx);
        aarch64_ops::mov_imm(&mut ctx, Register::virtual_gpr(0), 1).insert_at_back(entry, &ctx);
        aarch64_ops::call(&mut ctx, "callee".try_into().unwrap()).insert_at_back(entry, &ctx);
        aarch64_ops::mov(&mut ctx, Register::gpr(0), Register::virtual_gpr(0)).insert_at_back(entry, &ctx);

        allocate_function(&mut ctx, func, &CodegenOpts::default()).unwrap();
        assert_eq!(func.stack_size(&ctx), 0, "no spill slot is needed");
        assert_eq!(func.saved_regs(&ctx), 1u64 << 19);
        let insts: Vec<_> = entry.deref(&ctx).iter(&ctx).collect();
        assert_eq!(
            aarch64_ops::reg(&ctx, insts[0], ATTR_KEY_AARCH64_RD.as_ref()).unwrap(),
            Register::gpr(19)
        );
    }

    /// When every callee-saved register is taken by other call-crossing
    /// values, the furthest-ending one spills — the pre-callee-saved
    /// fallback.
    #[test]
    fn call_crossing_overflow_spills() {
        let mut ctx = context();
        let func = func(&mut ctx);
        let entry = func.entry_block(&ctx);
        for index in 0..11u32 {
            aarch64_ops::mov_imm(&mut ctx, Register::virtual_gpr(index), index as u64)
                .insert_at_back(entry, &ctx);
        }
        aarch64_ops::call(&mut ctx, "callee".try_into().unwrap()).insert_at_back(entry, &ctx);
        // v0 is used last, so it ends furthest and is the eviction victim.
        for index in (1..11u32).rev() {
            aarch64_ops::mov(&mut ctx, Register::gpr(0), Register::virtual_gpr(index))
                .insert_at_back(entry, &ctx);
        }
        aarch64_ops::mov(&mut ctx, Register::gpr(0), Register::virtual_gpr(0)).insert_at_back(entry, &ctx);

        allocate_function(&mut ctx, func, &CodegenOpts::default()).unwrap();
        assert_eq!(func.stack_size(&ctx), 16, "exactly one value spills");
        let opcodes: Vec<_> = entry
            .deref(&ctx)
            .iter(&ctx)
            .filter_map(|op| aarch64_ops::opcode(&ctx, op))
            .collect();
        assert!(
            opcodes
                .iter()
                .any(|opcode| *opcode == aarch64_ops::StrSpOffsetOp::OPCODE)
        );
        assert!(
            opcodes
                .iter()
                .any(|opcode| *opcode == aarch64_ops::LdrSpOffsetOp::OPCODE)
        );
    }

    #[test]
    fn stores_spilled_tied_movk_definition() {
        let mut ctx = context();
        let func = func(&mut ctx);
        let entry = func.entry_block(&ctx);
        aarch64_ops::mov_imm(&mut ctx, Register::virtual_gpr(0), 1).insert_at_back(entry, &ctx);
        // Fill the callee-saved pool with other call-crossing values so v0
        // (which ends furthest) spills.
        for index in 1..11u32 {
            aarch64_ops::mov_imm(&mut ctx, Register::virtual_gpr(index), index as u64)
                .insert_at_back(entry, &ctx);
        }
        aarch64_ops::call(&mut ctx, "callee".try_into().unwrap()).insert_at_back(entry, &ctx);
        for index in 1..11u32 {
            aarch64_ops::mov(&mut ctx, Register::gpr(0), Register::virtual_gpr(index))
                .insert_at_back(entry, &ctx);
        }
        aarch64_ops::movk(&mut ctx, Register::virtual_gpr(0), 2, 16).insert_at_back(entry, &ctx);
        aarch64_ops::mov(&mut ctx, Register::gpr(0), Register::virtual_gpr(0)).insert_at_back(entry, &ctx);

        allocate_function(&mut ctx, func, &CodegenOpts::default()).unwrap();

        let insts: Vec<_> = entry
            .deref(&ctx)
            .iter(&ctx)
            .filter(|op| aarch64_ops::is_instruction(&ctx, *op))
            .collect();
        let movk_index = insts
            .iter()
            .position(|inst| aarch64_ops::opcode(&ctx, *inst) == Some(aarch64_ops::MovkOp::OPCODE))
            .unwrap();

        assert_eq!(
            aarch64_ops::opcode(&ctx, insts[movk_index - 1]),
            Some(aarch64_ops::LdrSpOffsetOp::OPCODE)
        );
        assert_eq!(
            aarch64_ops::opcode(&ctx, insts[movk_index + 1]),
            Some(aarch64_ops::StrSpOffsetOp::OPCODE)
        );
        assert_eq!(
            aarch64_ops::reg(&ctx, insts[movk_index], ATTR_KEY_AARCH64_RD.as_ref()).unwrap(),
            Register::gpr(13)
        );
    }

    #[test]
    fn spills_when_register_pressure_exceeds_available_registers() {
        let mut ctx = context();
        let func = func(&mut ctx);
        let entry = func.entry_block(&ctx);
        let values = pool_size(Bank::Gpr) as u32 + 2;
        for index in 0..values {
            aarch64_ops::mov_imm(&mut ctx, Register::virtual_gpr(index), index as u64)
                .insert_at_back(entry, &ctx);
        }
        for index in 0..values {
            aarch64_ops::mov(&mut ctx, Register::gpr((index % 8) as u8), Register::virtual_gpr(index))
                .insert_at_back(entry, &ctx);
        }

        allocate_function(&mut ctx, func, &CodegenOpts::default()).unwrap();
        assert!(func.stack_size(&ctx) > 0);
        let opcodes: Vec<_> = entry
            .deref(&ctx)
            .iter(&ctx)
            .filter_map(|op| aarch64_ops::opcode(&ctx, op))
            .collect();
        assert!(
            opcodes
                .iter()
                .any(|opcode| *opcode == aarch64_ops::StrSpOffsetOp::OPCODE)
        );
        assert!(
            opcodes
                .iter()
                .any(|opcode| *opcode == aarch64_ops::LdrSpOffsetOp::OPCODE)
        );
    }

    fn count_opcode(ctx: &Context, entry: Ptr<BasicBlock>, opcode: crate::dialects::aarch64::op_interfaces::Aarch64Opcode) -> usize {
        entry
            .deref(ctx)
            .iter(ctx)
            .filter(|op| aarch64_ops::opcode(ctx, *op) == Some(opcode))
            .count()
    }

    /// A spilled constant: under the trivial-remat estimate its uses
    /// re-execute the `mov_imm` instead of reloading, and the slot store
    /// stays (the safe form). The callee-saved pool is exhausted by other
    /// call-crossing values so the constant actually spills.
    #[test]
    fn remat_restores_constant_without_reload() {
        let mut ctx = context();
        let func = func(&mut ctx);
        let entry = func.entry_block(&ctx);
        aarch64_ops::mov_imm(&mut ctx, Register::virtual_gpr(0), 7).insert_at_back(entry, &ctx);
        for index in 1..11u32 {
            aarch64_ops::mov_imm(&mut ctx, Register::virtual_gpr(index), index as u64)
                .insert_at_back(entry, &ctx);
        }
        aarch64_ops::call(&mut ctx, "callee".try_into().unwrap()).insert_at_back(entry, &ctx);
        for index in 1..11u32 {
            aarch64_ops::mov(&mut ctx, Register::gpr(0), Register::virtual_gpr(index))
                .insert_at_back(entry, &ctx);
        }
        aarch64_ops::mov(&mut ctx, Register::gpr(0), Register::virtual_gpr(0)).insert_at_back(entry, &ctx);

        let opts = CodegenOpts {
            restore: RestoreEstimate::TrivialRemat,
            ..CodegenOpts::default()
        };
        allocate_function(&mut ctx, func, &opts).unwrap();

        assert_eq!(count_opcode(&ctx, entry, aarch64_ops::LdrSpOffsetOp::OPCODE), 0);
        // The 11 original defs plus one rematerialization at the use.
        assert_eq!(count_opcode(&ctx, entry, aarch64_ops::MovImmOp::OPCODE), 12);
        assert_eq!(count_opcode(&ctx, entry, aarch64_ops::StrSpOffsetOp::OPCODE), 1);
    }

    /// Simd128 vregs allocate from the shared Fpr pool: the q vreg takes
    /// q16 (the first caller-saved pool number) and a simultaneously live d
    /// vreg takes d17 — one allocation unit per register number, so d/q
    /// overlap is impossible by construction.
    #[test]
    fn simd128_allocates_q_registers_from_the_shared_fpr_pool() {
        use crate::dialects::aarch64::op_interfaces::Aarch64Opcode;
        let mut ctx = context();
        let func = func(&mut ctx);
        let entry = func.entry_block(&ctx);
        let vq = Register::virtual_simd128(0);
        let vd = Register::virtual_fpr64(1);
        aarch64_ops::unary(&mut ctx, Aarch64Opcode::DupV4sGpr, vq, Register::gpr(0))
            .insert_at_back(entry, &ctx);
        aarch64_ops::unary(&mut ctx, Aarch64Opcode::FmovDX, vd, Register::gpr(0))
            .insert_at_back(entry, &ctx);
        aarch64_ops::fmov_rr(&mut ctx, Aarch64Opcode::MovV16b, Register::simd128(0), vq)
            .insert_at_back(entry, &ctx);
        aarch64_ops::fmov_rr(&mut ctx, Aarch64Opcode::FmovD, Register::fpr64(0), vd)
            .insert_at_back(entry, &ctx);

        allocate_function(&mut ctx, func, &CodegenOpts::default()).unwrap();
        assert_eq!(func.stack_size(&ctx), 0);
        let insts: Vec<_> = entry.deref(&ctx).iter(&ctx).collect();
        assert_eq!(
            aarch64_ops::reg(&ctx, insts[0], ATTR_KEY_AARCH64_RD.as_ref()).unwrap(),
            Register::simd128(16),
            "q vreg takes the first caller-saved pool number, spelled q"
        );
        assert_eq!(
            aarch64_ops::reg(&ctx, insts[1], ATTR_KEY_AARCH64_RD.as_ref()).unwrap(),
            Register::fpr64(17),
            "the overlapping d vreg takes the NEXT pool number, never d16"
        );
    }

    /// With every caller-saved Fpr pool register taken by live q values,
    /// the next q vreg spills to a 16-byte slot instead of touching the
    /// callee-saved d8–d15 range (which preserves only its low 64 bits).
    #[test]
    fn simd128_never_takes_a_callee_saved_register() {
        use crate::dialects::aarch64::op_interfaces::Aarch64Opcode;
        let mut ctx = context();
        let func = func(&mut ctx);
        let entry = func.entry_block(&ctx);
        let live = ALLOCATABLE_FPR_NUMBERS.len() as u32 + 1;
        for index in 0..live {
            aarch64_ops::unary(
                &mut ctx,
                Aarch64Opcode::DupV4sGpr,
                Register::virtual_simd128(index),
                Register::gpr(0),
            )
            .insert_at_back(entry, &ctx);
        }
        for index in 0..live {
            aarch64_ops::fmov_rr(
                &mut ctx,
                Aarch64Opcode::MovV16b,
                Register::simd128(0),
                Register::virtual_simd128(index),
            )
            .insert_at_back(entry, &ctx);
        }

        allocate_function(&mut ctx, func, &CodegenOpts::default()).unwrap();
        assert_eq!(func.saved_regs(&ctx), 0, "no callee-saved register is used");
        assert_eq!(func.stack_size(&ctx), 16, "one q value spills 16 bytes");
        let opcodes: Vec<_> = entry
            .deref(&ctx)
            .iter(&ctx)
            .filter_map(|op| aarch64_ops::opcode(&ctx, op))
            .collect();
        assert!(opcodes.contains(&aarch64_ops::StrqSpOffsetOp::OPCODE));
        assert!(opcodes.contains(&aarch64_ops::LdrqSpOffsetOp::OPCODE));
        // The spill scratch is the q view of the reserved d20.
        let reload = entry
            .deref(&ctx)
            .iter(&ctx)
            .find(|op| {
                aarch64_ops::opcode(&ctx, *op) == Some(aarch64_ops::LdrqSpOffsetOp::OPCODE)
            })
            .unwrap();
        assert_eq!(
            aarch64_ops::reg(&ctx, reload, ATTR_KEY_AARCH64_RD.as_ref()).unwrap(),
            Register::simd128(20)
        );
    }

    /// A q value live across a call force-spills: no register (caller- or
    /// callee-saved) preserves all 128 bits across a call.
    #[test]
    fn simd128_crossing_a_call_spills() {
        use crate::dialects::aarch64::op_interfaces::Aarch64Opcode;
        let mut ctx = context();
        let func = func(&mut ctx);
        let entry = func.entry_block(&ctx);
        let vq = Register::virtual_simd128(0);
        aarch64_ops::unary(&mut ctx, Aarch64Opcode::DupV4sGpr, vq, Register::gpr(0))
            .insert_at_back(entry, &ctx);
        aarch64_ops::call(&mut ctx, "callee".try_into().unwrap()).insert_at_back(entry, &ctx);
        aarch64_ops::fmov_rr(&mut ctx, Aarch64Opcode::MovV16b, Register::simd128(0), vq)
            .insert_at_back(entry, &ctx);

        allocate_function(&mut ctx, func, &CodegenOpts::default()).unwrap();
        assert_eq!(func.saved_regs(&ctx), 0);
        assert_eq!(func.stack_size(&ctx), 16);
        let opcodes: Vec<_> = entry
            .deref(&ctx)
            .iter(&ctx)
            .filter_map(|op| aarch64_ops::opcode(&ctx, op))
            .collect();
        assert!(opcodes.contains(&aarch64_ops::StrqSpOffsetOp::OPCODE));
        assert!(opcodes.contains(&aarch64_ops::LdrqSpOffsetOp::OPCODE));
    }

    /// Simd128 spill slots are two consecutive 8-byte slots starting at an
    /// even index, so their sp offsets stay 16-byte aligned.
    #[test]
    fn simd128_spill_slots_are_16_byte_aligned() {
        let mut slots = 0u64;
        assert_eq!(take_spill_slots(&mut slots, RegisterClass::Fpr64), 0);
        assert_eq!(take_spill_slots(&mut slots, RegisterClass::Simd128), 2);
        assert_eq!(take_spill_slots(&mut slots, RegisterClass::Gpr64), 4);
        assert_eq!(take_spill_slots(&mut slots, RegisterClass::Simd128), 6);
        assert_eq!(slots, 8);
    }

    /// Builds the discrimination case: v0 ends furthest and is used six
    /// times, so the furthest-end baseline spills it (6 reloads) while the
    /// weighted policy spills the single-use candidate instead (1 reload).
    fn build_pressure_with_hot_value(ctx: &mut Context) -> (FuncOp, Ptr<BasicBlock>) {
        let func = func(ctx);
        let entry = func.entry_block(ctx);
        let values = pool_size(Bank::Gpr) as u32 + 1;
        for index in 0..values {
            aarch64_ops::mov_imm(ctx, Register::virtual_gpr(index), index as u64)
                .insert_at_back(entry, ctx);
        }
        aarch64_ops::mov(ctx, Register::gpr(0), Register::virtual_gpr(values - 1))
            .insert_at_back(entry, ctx);
        for index in 1..values - 1 {
            aarch64_ops::mov(ctx, Register::gpr(1), Register::virtual_gpr(index))
                .insert_at_back(entry, ctx);
        }
        for _ in 0..6 {
            aarch64_ops::mov(ctx, Register::gpr(2), Register::virtual_gpr(0))
                .insert_at_back(entry, ctx);
        }
        (func, entry)
    }

    /// `CRABBIT_BLOCK_FREQ=profile` without a usable profile (or with a
    /// profile that lacks this function) must behave exactly like uniform
    /// — never error (docs/PROFILE-FEEDBACK-PLAN.md). The test function's
    /// symbol (`test`) appears in no profile fixture, so this holds even
    /// if another test set `CRABBIT_PROFILE` concurrently.
    #[test]
    fn profile_freq_without_profile_falls_back_to_uniform() {
        let mut ctx = context();
        let (func, entry) = build_pressure_with_hot_value(&mut ctx);
        let opts = CodegenOpts {
            victim: SpillVictimPolicy::WeightedCost,
            freq: BlockFreqModel::Profile,
            ..CodegenOpts::default()
        };
        allocate_function(&mut ctx, func, &opts).unwrap();
        let profile_reloads = count_opcode(&ctx, entry, aarch64_ops::LdrSpOffsetOp::OPCODE);

        let mut ctx = context();
        let (func, entry) = build_pressure_with_hot_value(&mut ctx);
        let opts = CodegenOpts {
            victim: SpillVictimPolicy::WeightedCost,
            ..CodegenOpts::default()
        };
        allocate_function(&mut ctx, func, &opts).unwrap();
        let uniform_reloads = count_opcode(&ctx, entry, aarch64_ops::LdrSpOffsetOp::OPCODE);

        assert_eq!(profile_reloads, uniform_reloads);
    }

    #[test]
    fn weighted_policy_avoids_spilling_hot_value() {
        let mut ctx = context();
        let (func, entry) = build_pressure_with_hot_value(&mut ctx);
        allocate_function(&mut ctx, func, &CodegenOpts::default()).unwrap();
        let baseline_reloads = count_opcode(&ctx, entry, aarch64_ops::LdrSpOffsetOp::OPCODE);

        let mut ctx = context();
        let (func, entry) = build_pressure_with_hot_value(&mut ctx);
        let opts = CodegenOpts {
            victim: SpillVictimPolicy::WeightedCost,
            ..CodegenOpts::default()
        };
        allocate_function(&mut ctx, func, &opts).unwrap();
        let weighted_reloads = count_opcode(&ctx, entry, aarch64_ops::LdrSpOffsetOp::OPCODE);

        assert_eq!(baseline_reloads, 6, "furthest-end spills the hot value");
        assert_eq!(weighted_reloads, 1, "weighted spills the single-use value");
    }
}
