//! Lower LLVM-dialect block arguments into explicit `llvm.phi` operations.

use crate::{
    arg_err,
    common_traits::Named,
    context::{Context, Ptr},
    debug_info::{get_block_arg_name, set_operation_result_name},
    dialects::{builtin::op_interfaces::OneRegionInterface, llvm::ops as llvm_ops},
    ir::{
        basic_block::BasicBlock, location::Location, op::Op, operation::Operation, r#type::Typed,
        value::Value,
    },
    linked_list::ContainsLinkedList,
    conversion::pass::{Pass, PassOptions},
    result::STAIRResult,
};

/// Pass that converts LLVM block arguments/successor operands into `llvm.phi`.
pub struct LowerLLVMBlockArgsToPhiPass;

impl Pass for LowerLLVMBlockArgsToPhiPass {
    fn name(&self) -> &str {
        "lower-llvm-block-args-to-phi"
    }

    fn run(
        &self,
        root: Ptr<Operation>,
        ctx: &mut Context,
        _options: PassOptions,
    ) -> STAIRResult<Ptr<Operation>> {
        let mut funcs = Vec::new();
        collect_ops(ctx, root, llvm_ops::FuncOp::get_opid_static(), &mut funcs);

        for func in funcs {
            lower_function(ctx, llvm_ops::FuncOp::from_operation(func))?;
        }

        Ok(root)
    }
}

fn collect_ops(
    ctx: &Context,
    op: Ptr<Operation>,
    target: crate::ir::op::OpId,
    out: &mut Vec<Ptr<Operation>>,
) {
    if Operation::get_opid(op, ctx) == target {
        out.push(op);
    }

    let regions: Vec<_> = op.deref(ctx).regions().collect();
    for region in regions {
        let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
        for block in blocks {
            let ops: Vec<_> = block.deref(ctx).iter(ctx).collect();
            for nested in ops {
                collect_ops(ctx, nested, target.clone(), out);
            }
        }
    }
}

fn lower_function(ctx: &mut Context, func: llvm_ops::FuncOp) -> STAIRResult<()> {
    if func.get_region(ctx).deref(ctx).get_head().is_none() {
        return Ok(());
    }

    let region = func.get_region(ctx);
    let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
    let Some(entry) = blocks.first().copied() else {
        return Ok(());
    };

    let mut replacements = Vec::new();

    for &block in blocks.iter().filter(|&&block| block != entry) {
        let arg_count = block.deref(ctx).get_num_arguments();
        if arg_count == 0 {
            continue;
        }

        for arg_idx in 0..arg_count {
            let arg = block.deref(ctx).get_argument(arg_idx);
            let arg_ty = arg.get_type(ctx);
            let (incoming_values, incoming_blocks) =
                collect_incoming_for_block_arg(ctx, &blocks, block, arg_idx, arg_count)?;

            let phi = llvm_ops::PhiOp::new(ctx, incoming_values, incoming_blocks, arg_ty);
            if let Some(name) = get_block_arg_name(ctx, block, arg_idx) {
                set_operation_result_name(ctx, phi.get_operation(), 0, Some(name));
            }
            phi.get_operation().insert_at_front(block, ctx);
            replacements.push((arg, phi.get_result(ctx)));
        }
    }

    for (from, to) in replacements {
        from.replace_some_uses_with(ctx, |_, _| true, &to);
    }

    for &block in blocks.iter().filter(|&&block| block != entry) {
        if block.deref(ctx).get_num_arguments() != 0 {
            while block.deref(ctx).get_num_arguments() > 0 {
                BasicBlock::pop_argument(block, ctx);
            }
        }
    }

    remove_successor_operands(ctx, &blocks)?;

    Ok(())
}

fn collect_incoming_for_block_arg(
    ctx: &Context,
    blocks: &[Ptr<BasicBlock>],
    target_block: Ptr<BasicBlock>,
    arg_idx: usize,
    target_arg_count: usize,
) -> STAIRResult<(Vec<Value>, Vec<crate::identifier::Identifier>)> {
    let mut incoming_values = Vec::new();
    let mut incoming_blocks = Vec::new();

    for &pred_block in blocks {
        let Some(term) = pred_block.deref(ctx).get_terminator(ctx) else {
            continue;
        };
        let successors: Vec<_> = term.deref(ctx).successors().collect();
        for (succ_idx, succ) in successors.into_iter().enumerate() {
            if succ != target_block {
                continue;
            }

            let operands = successor_operands(ctx, term, succ_idx);
            if operands.len() != target_arg_count {
                return arg_err!(
                    Location::Unknown,
                    "successor operand count {} does not match target block argument count {} for ^{}",
                    operands.len(),
                    target_arg_count,
                    target_block.deref(ctx).unique_name(ctx)
                );
            }
            incoming_values.push(operands[arg_idx]);
            incoming_blocks.push(pred_block.deref(ctx).unique_name(ctx));
        }
    }

    Ok((incoming_values, incoming_blocks))
}

fn successor_operands(ctx: &Context, term: Ptr<Operation>, succ_idx: usize) -> Vec<Value> {
    let opid = Operation::get_opid(term, ctx);
    if opid == llvm_ops::BrOp::get_opid_static() {
        llvm_ops::BrOp::from_operation(term).get_dest_operands(ctx)
    } else if opid == llvm_ops::CondBrOp::get_opid_static() {
        let cond_br = llvm_ops::CondBrOp::from_operation(term);
        if succ_idx == 0 {
            cond_br.get_true_operands(ctx)
        } else {
            cond_br.get_false_operands(ctx)
        }
    } else {
        vec![]
    }
}

fn remove_successor_operands(ctx: &mut Context, blocks: &[Ptr<BasicBlock>]) -> STAIRResult<()> {
    for &block in blocks {
        let Some(term) = block.deref(ctx).get_terminator(ctx) else {
            continue;
        };
        let opid = Operation::get_opid(term, ctx);

        if opid == llvm_ops::BrOp::get_opid_static() {
            let br = llvm_ops::BrOp::from_operation(term);
            if br.get_dest_operands(ctx).is_empty() {
                continue;
            }

            let new_br = llvm_ops::BrOp::new(ctx, br.get_dest(ctx), vec![]);
            new_br.get_operation().insert_before(ctx, term);
            Operation::erase(term, ctx);
        } else if opid == llvm_ops::CondBrOp::get_opid_static() {
            let cbr = llvm_ops::CondBrOp::from_operation(term);
            if cbr.get_true_operands(ctx).is_empty() && cbr.get_false_operands(ctx).is_empty() {
                continue;
            }

            let new_cbr = llvm_ops::CondBrOp::new(
                ctx,
                cbr.get_condition(ctx),
                cbr.get_true_dest(ctx),
                vec![],
                cbr.get_false_dest(ctx),
                vec![],
            );
            new_cbr.get_operation().insert_before(ctx, term);
            Operation::erase(term, ctx);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dialects::{builtin, llvm},
        ir::r#type::{TypeHandle, TypedHandle},
        printable::Printable,
    };

    fn test_context() -> Context {
        let mut ctx = Context::new();
        llvm::register(&mut ctx);
        ctx
    }

    #[test]
    fn lowers_non_entry_block_args_to_phi() {
        let mut ctx = test_context();
        let i64_ty: TypeHandle =
            builtin::types::IntegerType::get(&mut ctx, 64, builtin::types::Signedness::Signless)
                .into();
        let fn_ty: TypedHandle<llvm::types::FuncType> =
            llvm::types::FuncType::get(&mut ctx, i64_ty, vec![i64_ty], false);
        let func =
            llvm_ops::FuncOp::new(&mut ctx, "f".try_into().unwrap(), fn_ty, Default::default());
        let entry = func.get_entry_block(&ctx);
        let arg = entry.deref(&ctx).get_argument(0);
        let then_block = BasicBlock::new(&mut ctx, Some("then".try_into().unwrap()), vec![i64_ty]);
        then_block.insert_at_back(func.get_region(&ctx), &ctx);

        let br = llvm_ops::BrOp::new(&mut ctx, then_block, vec![arg]);
        br.get_operation().insert_at_back(entry, &ctx);
        let then_arg = then_block.deref(&ctx).get_argument(0);
        let ret = llvm_ops::ReturnOp::new(&mut ctx, Some(then_arg));
        ret.get_operation().insert_at_back(then_block, &ctx);

        LowerLLVMBlockArgsToPhiPass
            .run(func.get_operation(), &mut ctx, PassOptions::default())
            .unwrap();

        let text = format!("{}", func.get_operation().disp(&ctx));
        assert!(text.contains("llvm.phi"));
        assert!(text.contains("llvm.br ^then"));
        assert!(!text.contains("^then("));
        assert!(!text.contains("llvm.br ^then("));
    }
}
