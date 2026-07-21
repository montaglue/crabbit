use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::{
    context::{Context, Ptr},
    dialects::{
        aarch64::{
            op_interfaces::RegisterOperandKind,
            ops::{self as aarch64_ops, FuncOp},
            registers::{Register, RegisterClass, VirtualRegister},
        },
        builtin::op_interfaces::OneRegionInterface,
    },
    ir::{basic_block::BasicBlock, operation::Operation},
    linked_list::ContainsLinkedList,
    conversion::pass::{Pass, PassOptions},
    result::STAIRResult,
};

use super::{error::Aarch64DarwinErr, frontend::module_op, util::cast_operation};

const ALLOCATABLE_GPRS: [Register; 4] = [
    Register::gpr(9),
    Register::gpr(10),
    Register::gpr(11),
    Register::gpr(12),
];
const SPILL_SCRATCH_GPRS: [Register; 3] = [Register::gpr(13), Register::gpr(14), Register::gpr(15)];
const SPILL_SLOT_BYTES: u64 = 8;

pub struct Aarch64RegisterAllocatePass;

impl Pass for Aarch64RegisterAllocatePass {
    fn name(&self) -> &str {
        "aarch64-register-allocate"
    }

    fn run(
        &self,
        root: Ptr<Operation>,
        ctx: &mut Context,
        _options: PassOptions,
    ) -> STAIRResult<Ptr<Operation>> {
        let module = module_op(ctx, root)?;
        let body = module.get_region(ctx).deref(ctx).get_head().unwrap();
        let funcs: Vec<_> = body.deref(ctx).iter(ctx).collect();
        for op in funcs {
            if let Some(func) = cast_operation::<FuncOp>(ctx, op) {
                allocate_function(ctx, func)?;
            }
        }
        Ok(root)
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

fn allocate_function(ctx: &mut Context, func: FuncOp) -> STAIRResult<()> {
    let (insts, intervals) = collect_live_intervals(ctx, func);
    let call_crossing = values_live_across_calls(ctx, &insts, &intervals);
    let allocation = linear_scan(&intervals, &call_crossing);
    let base_stack_size = func.stack_size(ctx);
    rewrite_allocated_registers(ctx, &insts, &allocation.assignments, base_stack_size)?;
    func.set_stack_size(
        ctx,
        align_to_16(base_stack_size + allocation.spill_slots * SPILL_SLOT_BYTES),
    );
    Ok(())
}

fn collect_live_intervals(ctx: &Context, func: FuncOp) -> (Vec<Ptr<Operation>>, Vec<LiveInterval>) {
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
    (insts, intervals)
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
}

fn linear_scan(
    intervals: &[LiveInterval],
    forced_spills: &BTreeSet<VirtualRegister>,
) -> AllocationResult {
    let mut active = Vec::<ActiveInterval>::new();
    let mut free: Vec<_> = (0..ALLOCATABLE_GPRS.len()).collect();
    let mut assignments = HashMap::<VirtualRegister, Allocation>::new();
    let mut spill_slots = 0u64;

    for interval in intervals {
        if forced_spills.contains(&interval.vreg) {
            assignments.insert(interval.vreg, Allocation::Spill(spill_slots));
            spill_slots += 1;
            continue;
        }

        expire_old_intervals(interval.start, &mut active, &mut free);
        free.sort_unstable();
        let phys_index = if let Some(phys_index) = free.first().copied() {
            free.remove(0);
            phys_index
        } else if let Some(spilled) = spill_at_interval(interval, &mut active) {
            assignments.insert(spilled.interval.vreg, Allocation::Spill(spill_slots));
            spill_slots += 1;
            spilled.phys_index
        } else {
            assignments.insert(interval.vreg, Allocation::Spill(spill_slots));
            spill_slots += 1;
            continue;
        };

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
    }
}

fn spill_at_interval(
    current: &LiveInterval,
    active: &mut Vec<ActiveInterval>,
) -> Option<ActiveInterval> {
    let spill_index = active
        .iter()
        .enumerate()
        .max_by(|(_, lhs), (_, rhs)| {
            lhs.interval
                .end
                .cmp(&rhs.interval.end)
                .then(lhs.phys_index.cmp(&rhs.phys_index))
        })
        .map(|(index, _)| index)?;

    (active[spill_index].interval.end > current.end).then(|| active.remove(spill_index))
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
) -> STAIRResult<()> {
    for op in insts {
        if !aarch64_ops::is_instruction(ctx, *op) {
            continue;
        }
        let mut scratch_index = 0usize;
        let use_operands = virtual_use_operands(ctx, *op);
        let def_operands = virtual_def_operands(ctx, *op);
        let mut spilled_use_scratch =
            HashMap::<(&'static str, VirtualRegister), Register>::new();

        for (key, vreg) in use_operands {
            match assignments.get(&vreg) {
                Some(Allocation::Phys(phys_index)) => {
                    aarch64_ops::rewrite_register_operand(
                        ctx,
                        *op,
                        key,
                        ALLOCATABLE_GPRS[*phys_index],
                    );
                }
                Some(Allocation::Spill(slot)) => {
                    let scratch = next_spill_scratch(&mut scratch_index)?;
                    aarch64_ops::ldr_sp_offset(
                        ctx,
                        scratch,
                        spill_base_offset + slot * SPILL_SLOT_BYTES,
                    )
                    .insert_before(ctx, *op);
                    aarch64_ops::rewrite_register_operand(ctx, *op, key, scratch);
                    spilled_use_scratch.insert((key, vreg), scratch);
                }
                None => {}
            }
        }

        for (key, vreg) in def_operands {
            match assignments.get(&vreg) {
                Some(Allocation::Phys(phys_index)) => {
                    aarch64_ops::rewrite_register_operand(
                        ctx,
                        *op,
                        key,
                        ALLOCATABLE_GPRS[*phys_index],
                    );
                }
                Some(Allocation::Spill(slot)) => {
                    let scratch = match spilled_use_scratch.get(&(key, vreg)) {
                        Some(scratch) => *scratch,
                        None => next_spill_scratch(&mut scratch_index)?,
                    };
                    aarch64_ops::rewrite_register_operand(ctx, *op, key, scratch);
                    aarch64_ops::str_sp_offset(
                        ctx,
                        scratch,
                        spill_base_offset + slot * SPILL_SLOT_BYTES,
                    )
                    .insert_after(ctx, *op);
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
        .map(|(_, reg)| reg)
        .collect()
}

fn virtual_defs(ctx: &Context, inst: Ptr<Operation>) -> Vec<VirtualRegister> {
    virtual_def_operands(ctx, inst)
        .into_iter()
        .map(|(_, reg)| reg)
        .collect()
}

fn virtual_use_operands(
    ctx: &Context,
    inst: Ptr<Operation>,
) -> Vec<(&'static str, VirtualRegister)> {
    virtual_operands_with_kind(ctx, inst, RegisterOperandKind::Use)
}

fn virtual_def_operands(
    ctx: &Context,
    inst: Ptr<Operation>,
) -> Vec<(&'static str, VirtualRegister)> {
    virtual_operands_with_kind(ctx, inst, RegisterOperandKind::Def)
}

fn virtual_operands_with_kind(
    ctx: &Context,
    inst: Ptr<Operation>,
    kind: RegisterOperandKind,
) -> Vec<(&'static str, VirtualRegister)> {
    aarch64_ops::register_operands(ctx, inst)
        .into_iter()
        .filter_map(|operand| {
            (operand.kind == kind)
                .then_some(operand)
                .and_then(|operand| match operand.reg {
                    Register::Virtual {
                        id,
                        class: RegisterClass::Gpr64,
                    } => Some((operand.key, id)),
                    _ => None,
                })
        })
        .collect()
}

fn next_spill_scratch(scratch_index: &mut usize) -> STAIRResult<Register> {
    let Some(scratch) = SPILL_SCRATCH_GPRS.get(*scratch_index) else {
        return Err(crate::input_error_noloc!(Aarch64DarwinErr::UnsupportedOp(
            "aarch64 instruction needs more spill scratch registers than are reserved".to_string()
        )));
    };
    *scratch_index += 1;
    Ok(*scratch)
}

fn align_to_16(bytes: u64) -> u64 {
    (bytes + 15) & !15
}

#[cfg(test)]
mod tests {
    use llvm_compat::ll::LinkageAttr;
    use crate::{
        dialects::{
            aarch64::{
                self,
                ops::{self as aarch64_ops, ATTR_KEY_AARCH64_RD},
            },
            builtin,
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

        allocate_function(&mut ctx, func).unwrap();
        let insts: Vec<_> = entry.deref(&ctx).iter(&ctx).collect();
        assert_eq!(
            aarch64_ops::reg(&ctx, insts[0], ATTR_KEY_AARCH64_RD.as_str()).unwrap(),
            Register::gpr(9)
        );
        assert_eq!(
            aarch64_ops::reg(&ctx, insts[2], ATTR_KEY_AARCH64_RD.as_str()).unwrap(),
            Register::gpr(9)
        );
    }

    #[test]
    fn spills_virtual_register_live_across_call() {
        let mut ctx = context();
        let func = func(&mut ctx);
        let entry = func.entry_block(&ctx);
        aarch64_ops::mov_imm(&mut ctx, Register::virtual_gpr(0), 1).insert_at_back(entry, &ctx);
        aarch64_ops::call(&mut ctx, "callee".try_into().unwrap()).insert_at_back(entry, &ctx);
        aarch64_ops::mov(&mut ctx, Register::gpr(0), Register::virtual_gpr(0)).insert_at_back(entry, &ctx);

        allocate_function(&mut ctx, func).unwrap();
        assert_eq!(func.stack_size(&ctx), 16);
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
        aarch64_ops::call(&mut ctx, "callee".try_into().unwrap()).insert_at_back(entry, &ctx);
        aarch64_ops::movk(&mut ctx, Register::virtual_gpr(0), 2, 16).insert_at_back(entry, &ctx);
        aarch64_ops::mov(&mut ctx, Register::gpr(0), Register::virtual_gpr(0)).insert_at_back(entry, &ctx);

        allocate_function(&mut ctx, func).unwrap();

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
            aarch64_ops::reg(&ctx, insts[movk_index], ATTR_KEY_AARCH64_RD.as_str()).unwrap(),
            Register::gpr(13)
        );
    }

    #[test]
    fn spills_when_register_pressure_exceeds_available_registers() {
        let mut ctx = context();
        let func = func(&mut ctx);
        let entry = func.entry_block(&ctx);
        for index in 0..8u32 {
            aarch64_ops::mov_imm(&mut ctx, Register::virtual_gpr(index), index as u64)
                .insert_at_back(entry, &ctx);
        }
        for index in 0..8u32 {
            aarch64_ops::mov(&mut ctx, Register::gpr(index as u8), Register::virtual_gpr(index))
                .insert_at_back(entry, &ctx);
        }

        allocate_function(&mut ctx, func).unwrap();
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
}

use llvm_compat::ll::{LinkageAttr};
