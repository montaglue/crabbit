use crate::{
    context::{Context, Ptr},
    dialects::{
        aarch64::{
            ops::{self as aarch64_ops, ATTR_KEY_AARCH64_RD, FuncOp},
            registers::{LR, X16, X17},
        },
        builtin::op_interfaces::OneRegionInterface,
    },
    ir::{basic_block::BasicBlock, operation::Operation},
    linked_list::{ContainsLinkedList, LinkedList},
    conversion::pass::{AnalysisManager, Pass, PassResult, changed},
};

use super::{frontend::module_op, util::cast_operation};

pub struct Aarch64FrameLowerPass;

impl Pass for Aarch64FrameLowerPass {
    fn name(&self) -> &str {
        "aarch64-frame-lower"
    }

    fn run(&mut self, root: Ptr<Operation>, ctx: &mut Context, _analyses: &mut AnalysisManager) -> pliron::result::Result<PassResult> {
        let module = module_op(ctx, root)?;
        let body = module.get_region(ctx).deref(ctx).get_head().unwrap();
        let funcs: Vec<_> = body.deref(ctx).iter(ctx).collect();
        for op in funcs {
            if let Some(func) = cast_operation::<FuncOp>(ctx, op) {
                lower_function_frame(ctx, func);
            }
        }
        Ok(changed())
    }
}

fn lower_function_frame(ctx: &mut Context, func: FuncOp) {
    let stack_size = func.stack_size(ctx);
    if stack_size == 0 {
        return;
    }

    let entry = func.entry_block(ctx);
    insert_prologue(ctx, entry, stack_size);
    insert_epilogues(ctx, func, stack_size);
    legalize_large_sp_address_offsets(ctx, func);
}

/// `add xd, sp, #imm` only encodes offsets up to 4095. Rewrite larger slot
/// addresses into `add xd, sp, #0` followed by materializing the offset in
/// x16 (a scratch register outside the allocatable set) and adding it.
///
/// SP-relative loads and stores have their own (scale-dependent) immediate
/// limits — e.g. a byte store only reaches 4095 — so out-of-range ones are
/// rewritten to compute the slot address in x17 (via the same x16 offset
/// materialization) and access through a register-base addressing form.
fn legalize_large_sp_address_offsets(ctx: &mut Context, func: FuncOp) {
    let blocks: Vec<_> = func.get_region(ctx).deref(ctx).iter(ctx).collect();
    for block in blocks {
        let insts: Vec<_> = block.deref(ctx).iter(ctx).collect();
        for op in insts {
            let Some(opcode) = aarch64_ops::opcode(ctx, op) else {
                continue;
            };
            if opcode == aarch64_ops::AddSpOffsetOp::OPCODE {
                let Some(imm) = aarch64_ops::imm(ctx, op) else {
                    continue;
                };
                if imm <= 4095 {
                    continue;
                }
                let rd = aarch64_ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_ref())
                    .expect("add_sp_offset must define a destination register");
                aarch64_ops::set_imm(ctx, op, 0);
                let mark = materialize_offset_in_x16(ctx, op, imm);
                aarch64_ops::binary(ctx, aarch64_ops::AddOp::OPCODE, rd, rd, X16)
                    .insert_after(ctx, mark);
                continue;
            }
            let Some((reg_offset_opcode, scale)) = sp_mem_reg_offset_form(opcode) else {
                continue;
            };
            let Some(imm) = aarch64_ops::imm(ctx, op) else {
                continue;
            };
            let scaled_ok = imm <= 4095 * scale && imm % scale == 0;
            if scaled_ok || imm <= 255 {
                continue;
            }
            // x17 = sp + imm (through x16), then the same access at offset 0
            // off the register base.
            let rt = aarch64_ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_ref())
                .expect("sp-offset memory op must carry a data register");
            let base = aarch64_ops::add_sp_offset(ctx, X17, 0);
            base.insert_before(ctx, op);
            let mark = materialize_offset_in_x16(ctx, base, imm);
            aarch64_ops::binary(ctx, aarch64_ops::AddOp::OPCODE, X17, X17, X16)
                .insert_after(ctx, mark);
            let access =
                aarch64_ops::ldr_reg_offset_sized(ctx, reg_offset_opcode, rt, X17, 0);
            access.insert_before(ctx, op);
            Operation::erase(op, ctx);
        }
    }
}

/// Build `imm` in x16 with `movz`/`movk`, inserting after `mark`; returns
/// the last inserted instruction.
fn materialize_offset_in_x16(
    ctx: &mut Context,
    mark: Ptr<Operation>,
    imm: u64,
) -> Ptr<Operation> {
    let mut mark = mark;
    let movz = aarch64_ops::movz(ctx, X16, imm & 0xffff, 0);
    movz.insert_after(ctx, mark);
    mark = movz;
    for half in 1..4u64 {
        let bits = (imm >> (16 * half)) & 0xffff;
        if bits != 0 {
            let movk = aarch64_ops::movk(ctx, X16, bits, 16 * half);
            movk.insert_after(ctx, mark);
            mark = movk;
        }
    }
    mark
}

/// The register-base form and byte scale of an SP-relative memory opcode.
fn sp_mem_reg_offset_form(
    opcode: aarch64_ops::Aarch64Opcode,
) -> Option<(aarch64_ops::Aarch64Opcode, u64)> {
    use aarch64_ops::Aarch64Opcode;
    match opcode {
        Aarch64Opcode::StrSpOffset => Some((Aarch64Opcode::StrRegOffset, 8)),
        Aarch64Opcode::LdrSpOffset => Some((Aarch64Opcode::LdrRegOffset, 8)),
        Aarch64Opcode::StrwSpOffset => Some((Aarch64Opcode::StrwRegOffset, 4)),
        Aarch64Opcode::LdrwSpOffset => Some((Aarch64Opcode::LdrwRegOffset, 4)),
        Aarch64Opcode::StrhSpOffset => Some((Aarch64Opcode::StrhRegOffset, 2)),
        Aarch64Opcode::LdrhSpOffset => Some((Aarch64Opcode::LdrhRegOffset, 2)),
        Aarch64Opcode::StrbSpOffset => Some((Aarch64Opcode::StrbRegOffset, 1)),
        Aarch64Opcode::LdrbSpOffset => Some((Aarch64Opcode::LdrbRegOffset, 1)),
        Aarch64Opcode::StrdSpOffset => Some((Aarch64Opcode::StrdRegOffset, 8)),
        Aarch64Opcode::LdrdSpOffset => Some((Aarch64Opcode::LdrdRegOffset, 8)),
        Aarch64Opcode::StrsSpOffset => Some((Aarch64Opcode::StrsRegOffset, 4)),
        Aarch64Opcode::LdrsSpOffset => Some((Aarch64Opcode::LdrsRegOffset, 4)),
        _ => None,
    }
}

fn insert_prologue(ctx: &mut Context, entry: Ptr<BasicBlock>, stack_size: u64) {
    let mut after = None;
    let mut cursor = entry.deref(ctx).get_head();
    while let Some(op) = cursor {
        let Some(opcode) = aarch64_ops::opcode(ctx, op) else {
            break;
        };
        let is_stack_arg_load = opcode == aarch64_ops::LdrStackArgOp::OPCODE;
        let is_lr_save = opcode == aarch64_ops::StrPreSpOp::OPCODE
            && aarch64_ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_ref()) == Some(LR);
        if !is_stack_arg_load && !is_lr_save {
            break;
        }
        after = Some(op);
        cursor = op.deref(ctx).get_next();
    }

    for bytes in stack_chunks(stack_size) {
        let op = aarch64_ops::sub_sp_imm(ctx, bytes);
        if let Some(mark) = after {
            op.insert_after(ctx, mark);
        } else {
            op.insert_at_front(entry, ctx);
        }
        after = Some(op);
    }
}

fn insert_epilogues(ctx: &mut Context, func: FuncOp, stack_size: u64) {
    let blocks: Vec<_> = func.get_region(ctx).deref(ctx).iter(ctx).collect();
    for block in blocks {
        insert_block_epilogues(ctx, block, stack_size);
    }
}

fn insert_block_epilogues(ctx: &mut Context, block: Ptr<BasicBlock>, stack_size: u64) {
    let insts: Vec<_> = block.deref(ctx).iter(ctx).collect();
    let mut marks: Vec<_> = insts
        .iter()
        .copied()
        .filter(|op| {
            let Some(opcode) = aarch64_ops::opcode(ctx, *op) else {
                return false;
            };
            opcode == aarch64_ops::LdrPostSpOp::OPCODE
                && aarch64_ops::reg(ctx, *op, ATTR_KEY_AARCH64_RD.as_ref()) == Some(LR)
        })
        .collect();

    if marks.is_empty() {
        marks = insts
            .iter()
            .copied()
            .filter(|op| {
                let Some(opcode) = aarch64_ops::opcode(ctx, *op) else {
                    return false;
                };
                opcode == aarch64_ops::RetOp::OPCODE
            })
            .collect();
    }

    for mark in marks {
        for bytes in stack_chunks(stack_size).into_iter().rev() {
            aarch64_ops::add_sp_imm(ctx, bytes).insert_before(ctx, mark);
        }
    }
}

fn stack_chunks(mut bytes: u64) -> Vec<u64> {
    let mut chunks = Vec::new();
    while bytes > 0 {
        let chunk = bytes.min(4095);
        chunks.push(chunk);
        bytes -= chunk;
    }
    chunks
}

#[cfg(test)]
mod tests {
    use crate::ll::LinkageAttr;
    use crate::{
        dialects::{aarch64, aarch64::registers::{FP, Register}, builtin},
        ir::op::Op,
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

    fn instruction_ops(
        ctx: &Context,
        block: Ptr<crate::ir::basic_block::BasicBlock>,
    ) -> Vec<Ptr<Operation>> {
        block
            .deref(ctx)
            .iter(ctx)
            .filter(|op| aarch64_ops::is_instruction(ctx, *op))
            .collect()
    }

    fn opcodes_and_imms(
        ctx: &Context,
        block: Ptr<crate::ir::basic_block::BasicBlock>,
    ) -> Vec<(String, Option<u64>)> {
        instruction_ops(ctx, block)
            .into_iter()
            .map(|inst| {
                (
                    aarch64_ops::mnemonic(ctx, inst).unwrap().to_string(),
                    aarch64_ops::imm(ctx, inst),
                )
            })
            .collect()
    }

    #[test]
    fn run_lowers_aarch64_functions_with_stack_frames() {
        let mut ctx = context();
        let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
        let func = func(&mut ctx);
        func.set_stack_size(&mut ctx, 32);
        func.get_operation().insert_at_back(body, &ctx);
        let entry = func.entry_block(&ctx);
        aarch64_ops::ret(&mut ctx).insert_at_back(entry, &ctx);

        Aarch64FrameLowerPass
            .run(module.get_operation(), &mut ctx, &mut AnalysisManager::default())
            .unwrap();

        assert_eq!(
            opcodes_and_imms(&ctx, entry),
            [
                ("sub_sp_imm".to_string(), Some(32)),
                ("add_sp_imm".to_string(), Some(32)),
                ("ret".to_string(), None),
            ]
        );
    }

    #[test]
    fn prologue_is_inserted_after_stack_arg_loads_and_lr_save_only() {
        let mut ctx = context();
        let func = func(&mut ctx);
        let entry = func.entry_block(&ctx);
        aarch64_ops::ldr_stack_arg(&mut ctx, Register::gpr(0), 0).insert_at_back(entry, &ctx);
        aarch64_ops::str_pre_sp(&mut ctx, LR, 16).insert_at_back(entry, &ctx);
        aarch64_ops::str_pre_sp(&mut ctx, FP, 16).insert_at_back(entry, &ctx);
        aarch64_ops::mov(&mut ctx, Register::gpr(1), Register::gpr(0)).insert_at_back(entry, &ctx);

        insert_prologue(&mut ctx, entry, 5000);

        assert_eq!(
            opcodes_and_imms(&ctx, entry),
            [
                ("ldr_stack_arg".to_string(), Some(0)),
                ("str_pre_sp".to_string(), Some(16)),
                ("sub_sp_imm".to_string(), Some(4095)),
                ("sub_sp_imm".to_string(), Some(905)),
                ("str_pre_sp".to_string(), Some(16)),
                ("mov".to_string(), None),
            ]
        );
    }

    #[test]
    fn epilogue_prefers_lr_restore_over_plain_return() {
        let mut ctx = context();
        let func = func(&mut ctx);
        let entry = func.entry_block(&ctx);
        aarch64_ops::ldr_post_sp(&mut ctx, LR, 16).insert_at_back(entry, &ctx);
        aarch64_ops::ret(&mut ctx).insert_at_back(entry, &ctx);

        insert_block_epilogues(&mut ctx, entry, 32);

        assert_eq!(
            opcodes_and_imms(&ctx, entry),
            [
                ("add_sp_imm".to_string(), Some(32)),
                ("ldr_post_sp".to_string(), Some(16)),
                ("ret".to_string(), None),
            ]
        );
    }

    #[test]
    fn epilogue_falls_back_to_ret_when_lr_restore_does_not_match() {
        let mut ctx = context();
        let func = func(&mut ctx);
        let entry = func.entry_block(&ctx);
        aarch64_ops::mov(&mut ctx, LR, Register::gpr(0)).insert_at_back(entry, &ctx);
        aarch64_ops::ldr_post_sp(&mut ctx, FP, 16).insert_at_back(entry, &ctx);
        aarch64_ops::ret(&mut ctx).insert_at_back(entry, &ctx);

        insert_block_epilogues(&mut ctx, entry, 32);

        assert_eq!(
            opcodes_and_imms(&ctx, entry),
            [
                ("mov".to_string(), None),
                ("ldr_post_sp".to_string(), Some(16)),
                ("add_sp_imm".to_string(), Some(32)),
                ("ret".to_string(), None),
            ]
        );
    }

    #[test]
    fn stack_chunks_split_large_frames_at_encodable_immediates() {
        assert_eq!(stack_chunks(0), Vec::<u64>::new());
        assert_eq!(stack_chunks(1), vec![1]);
        assert_eq!(stack_chunks(4095), vec![4095]);
        assert_eq!(stack_chunks(4096), vec![4095, 1]);
        assert_eq!(stack_chunks(8191), vec![4095, 4095, 1]);
    }
}
