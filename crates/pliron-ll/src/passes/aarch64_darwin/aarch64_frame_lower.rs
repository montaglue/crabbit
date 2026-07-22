use crate::{
    context::{Context, Ptr},
    dialects::{
        aarch64::{
            ops::{self as aarch64_ops, ATTR_KEY_AARCH64_RD, FuncOp},
            registers::{LR, X16},
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
fn legalize_large_sp_address_offsets(ctx: &mut Context, func: FuncOp) {
    let blocks: Vec<_> = func.get_region(ctx).deref(ctx).iter(ctx).collect();
    for block in blocks {
        let insts: Vec<_> = block.deref(ctx).iter(ctx).collect();
        for op in insts {
            if aarch64_ops::opcode(ctx, op) != Some(aarch64_ops::AddSpOffsetOp::OPCODE) {
                continue;
            }
            let Some(imm) = aarch64_ops::imm(ctx, op) else {
                continue;
            };
            if imm <= 4095 {
                continue;
            }
            let rd = aarch64_ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_str())
                .expect("add_sp_offset must define a destination register");
            aarch64_ops::set_imm(ctx, op, 0);
            let mut mark = op;
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
            aarch64_ops::binary(ctx, aarch64_ops::AddOp::OPCODE, rd, rd, X16)
                .insert_after(ctx, mark);
        }
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
            && aarch64_ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_str()) == Some(LR);
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
                && aarch64_ops::reg(ctx, *op, ATTR_KEY_AARCH64_RD.as_str()) == Some(LR)
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
    use llvm_compat::ll::LinkageAttr;
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
