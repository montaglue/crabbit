use crate::{
    context::{Context, Ptr},
    dialects::{
        builtin::op_interfaces::OneRegionInterface,
        x86_64::{
            ops::{self as x86_64_ops, FuncOp},
            registers::{self, Register},
        },
    },
    ir::{basic_block::BasicBlock, operation::Operation},
    linked_list::{ContainsLinkedList, LinkedList},
    conversion::pass::{AnalysisManager, Pass, PassResult, changed},
};

use super::{frontend::module_op, util::cast_operation};

/// Callee-saved registers the register allocator hands out ([`rbx`, `r12`,
/// `r13`, `r14`] allocatable plus the `r15` spill scratch). Every function
/// saves and restores all of them; the odd count also realigns the stack:
/// at entry rsp is 8 (mod 16) after the pushed return address, and five
/// pushes make it 0 (mod 16), so 16-byte-aligned frames keep every call
/// site aligned.
const CALLEE_SAVED_GPRS: [Register; 5] = [
    registers::RBX,
    registers::R12,
    registers::R13,
    registers::R14,
    registers::R15,
];

pub struct X86_64FrameLowerPass;

impl Pass for X86_64FrameLowerPass {
    fn name(&self) -> &str {
        "x86-64-frame-lower"
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
    let entry = func.entry_block(ctx);
    insert_prologue(ctx, entry, stack_size);
    insert_epilogues(ctx, func, stack_size);
}

fn insert_prologue(ctx: &mut Context, entry: Ptr<BasicBlock>, stack_size: u64) {
    // Incoming stack-argument loads address memory relative to the entry rsp
    // (just above the return address); the prologue goes after them.
    let mut after = None;
    let mut cursor = entry.deref(ctx).get_head();
    while let Some(op) = cursor {
        let Some(opcode) = x86_64_ops::opcode(ctx, op) else {
            break;
        };
        if opcode != x86_64_ops::LdrStackArgOp::OPCODE {
            break;
        }
        after = Some(op);
        cursor = op.deref(ctx).get_next();
    }

    let insert = |ctx: &mut Context, op: Ptr<Operation>, after: &mut Option<Ptr<Operation>>| {
        if let Some(mark) = *after {
            op.insert_after(ctx, mark);
        } else {
            op.insert_at_front(entry, ctx);
        }
        *after = Some(op);
    };

    for reg in CALLEE_SAVED_GPRS {
        let op = x86_64_ops::push(ctx, reg);
        insert(ctx, op, &mut after);
    }
    if stack_size > 0 {
        let op = x86_64_ops::sub_sp_imm(ctx, stack_size);
        insert(ctx, op, &mut after);
    }
}

fn insert_epilogues(ctx: &mut Context, func: FuncOp, stack_size: u64) {
    let blocks: Vec<_> = func.get_region(ctx).deref(ctx).iter(ctx).collect();
    for block in blocks {
        insert_block_epilogues(ctx, block, stack_size);
    }
}

fn insert_block_epilogues(ctx: &mut Context, block: Ptr<BasicBlock>, stack_size: u64) {
    let marks: Vec<_> = block
        .deref(ctx)
        .iter(ctx)
        .filter(|op| x86_64_ops::opcode(ctx, *op) == Some(x86_64_ops::RetOp::OPCODE))
        .collect();

    for mark in marks {
        if stack_size > 0 {
            x86_64_ops::add_sp_imm(ctx, stack_size).insert_before(ctx, mark);
        }
        for reg in CALLEE_SAVED_GPRS.iter().rev() {
            x86_64_ops::pop(ctx, *reg).insert_before(ctx, mark);
        }
    }
}

#[cfg(test)]
mod tests {
    use llvm_compat::ll::LinkageAttr;
    use crate::{
        dialects::{builtin, x86_64},
        ir::op::Op,
    };

    use super::*;

    fn context() -> Context {
        let mut ctx = Context::new();
        x86_64::register(&mut ctx);
        ctx
    }

    fn func(ctx: &mut Context) -> FuncOp {
        FuncOp::new(ctx, "test".try_into().unwrap(), LinkageAttr::External)
    }

    fn opcodes_and_regs(
        ctx: &Context,
        block: Ptr<crate::ir::basic_block::BasicBlock>,
    ) -> Vec<(String, Option<String>)> {
        block
            .deref(ctx)
            .iter(ctx)
            .filter(|op| x86_64_ops::is_instruction(ctx, *op))
            .map(|inst| {
                (
                    x86_64_ops::mnemonic(ctx, inst).unwrap().to_string(),
                    x86_64_ops::reg(ctx, inst, x86_64_ops::ATTR_KEY_X86_64_RD.as_str())
                        .map(|reg| reg.to_string()),
                )
            })
            .collect()
    }

    #[test]
    fn run_lowers_x86_64_functions_with_stack_frames() {
        let mut ctx = context();
        let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
        let func = func(&mut ctx);
        func.set_stack_size(&mut ctx, 32);
        func.get_operation().insert_at_back(body, &ctx);
        let entry = func.entry_block(&ctx);
        x86_64_ops::ret(&mut ctx).insert_at_back(entry, &ctx);

        X86_64FrameLowerPass
            .run(module.get_operation(), &mut ctx, &mut AnalysisManager::default())
            .unwrap();

        let expected: Vec<(String, Option<String>)> = CALLEE_SAVED_GPRS
            .iter()
            .map(|reg| ("push".to_string(), Some(reg.to_string())))
            .chain([
                ("sub_sp_imm".to_string(), None),
                ("add_sp_imm".to_string(), None),
            ])
            .chain(
                CALLEE_SAVED_GPRS
                    .iter()
                    .rev()
                    .map(|reg| ("pop".to_string(), Some(reg.to_string()))),
            )
            .chain([("ret".to_string(), None)])
            .collect();
        assert_eq!(opcodes_and_regs(&ctx, entry), expected);
    }

    #[test]
    fn prologue_is_inserted_after_stack_arg_loads_only() {
        let mut ctx = context();
        let func = func(&mut ctx);
        let entry = func.entry_block(&ctx);
        x86_64_ops::ldr_stack_arg(&mut ctx, Register::virtual_gpr(0), 0).insert_at_back(entry, &ctx);
        x86_64_ops::mov(&mut ctx, Register::virtual_gpr(1), Register::virtual_gpr(0)).insert_at_back(entry, &ctx);

        insert_prologue(&mut ctx, entry, 48);

        let ops = opcodes_and_regs(&ctx, entry);
        assert_eq!(ops[0].0, "ldr_stack_arg");
        assert_eq!(ops[1], ("push".to_string(), Some("rbx".to_string())));
        assert_eq!(ops[5], ("push".to_string(), Some("r15".to_string())));
        assert_eq!(ops[6].0, "sub_sp_imm");
        assert_eq!(ops[7].0, "mov");
    }

    #[test]
    fn zero_sized_frames_still_save_callee_saved_registers() {
        let mut ctx = context();
        let func = func(&mut ctx);
        let entry = func.entry_block(&ctx);
        x86_64_ops::ret(&mut ctx).insert_at_back(entry, &ctx);

        insert_prologue(&mut ctx, entry, 0);
        insert_block_epilogues(&mut ctx, entry, 0);

        let ops = opcodes_and_regs(&ctx, entry);
        let mnemonics: Vec<_> = ops.iter().map(|(mnemonic, _)| mnemonic.as_str()).collect();
        assert_eq!(
            mnemonics,
            ["push", "push", "push", "push", "push", "pop", "pop", "pop", "pop", "pop", "ret"]
        );
    }
}
