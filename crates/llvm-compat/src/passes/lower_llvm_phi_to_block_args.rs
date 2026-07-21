//! Raise explicit `llvm.phi` operations back into block arguments and
//! branch successor operands. The inverse of
//! [lower_llvm_block_args_to_phi](super::lower_llvm_block_args_to_phi):
//! passes that produce phis (e.g. mem2reg) can run in front of backends that
//! only understand block arguments, such as the AArch64 Darwin lowering.

use rustc_hash::FxHashMap;

use crate::{
    arg_err,
    common_traits::Named,
    context::{Context, Ptr},
    dialects::builtin::op_interfaces::OneRegionInterface,
    ll::BranchWeightsAttr,
    op_interfaces::ATTR_KEY_BRANCH_WEIGHTS,
    dialects::llvm::ops as llvm_ops,
    ir::{
        basic_block::BasicBlock, location::Location, op::Op, operation::Operation, r#type::Typed,
        value::Value,
    },
    linked_list::ContainsLinkedList,
    conversion::pass::{Pass, PassOptions},
    result::STAIRResult,
};

/// Pass that converts `llvm.phi` operations into block arguments/successor
/// operands.
pub struct LowerLLVMPhiToBlockArgsPass;

impl Pass for LowerLLVMPhiToBlockArgsPass {
    fn name(&self) -> &str {
        "lower-llvm-phi-to-block-args"
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
            raise_function(ctx, llvm_ops::FuncOp::from_operation(func))?;
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

fn raise_function(ctx: &mut Context, func: llvm_ops::FuncOp) -> STAIRResult<()> {
    let region = func.get_region(ctx);
    let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
    let Some(entry) = blocks.first().copied() else {
        return Ok(());
    };

    let block_by_name: FxHashMap<_, _> = blocks
        .iter()
        .map(|&block| (block.deref(ctx).unique_name(ctx), block))
        .collect();

    // Per (predecessor, phi block) edge: one incoming value slot per phi,
    // in phi order — the successor operand list the edge's branch must carry.
    let mut edge_operands: FxHashMap<(Ptr<BasicBlock>, Ptr<BasicBlock>), Vec<Option<Value>>> =
        FxHashMap::default();
    let mut phi_counts: FxHashMap<Ptr<BasicBlock>, usize> = FxHashMap::default();
    let mut replacements: Vec<(Value, Value)> = Vec::new();
    let mut phis_to_erase: Vec<Ptr<Operation>> = Vec::new();

    for &block in &blocks {
        let phis: Vec<_> = block
            .deref(ctx)
            .iter(ctx)
            .take_while(|op| Operation::get_opid(*op, ctx) == llvm_ops::PhiOp::get_opid_static())
            .map(|op| llvm_ops::PhiOp::from_operation(op))
            .collect();
        if phis.is_empty() {
            continue;
        }
        if block == entry {
            return arg_err!(
                Location::Unknown,
                "llvm.phi in entry block ^{} has no predecessors to lower into",
                block.deref(ctx).unique_name(ctx)
            );
        }

        let phi_count = phis.len();
        phi_counts.insert(block, phi_count);

        for (phi_idx, phi) in phis.iter().enumerate() {
            let result = phi.get_result(ctx);
            let arg_idx = BasicBlock::push_argument(block, ctx, result.get_type(ctx));
            let arg = block.deref(ctx).get_argument(arg_idx);
            replacements.push((result, arg));

            let values = phi.get_incoming_values(ctx);
            let names = phi.get_incoming_blocks(ctx);
            for (value, name) in values.into_iter().zip(names) {
                let Some(&pred) = block_by_name.get(&name) else {
                    return arg_err!(
                        Location::Unknown,
                        "llvm.phi in ^{} names unknown predecessor ^{}",
                        block.deref(ctx).unique_name(ctx),
                        name
                    );
                };
                edge_operands.entry((pred, block)).or_insert_with(|| vec![None; phi_count])
                    [phi_idx] = Some(value);
            }
            phis_to_erase.push(phi.get_operation());
        }
    }

    if replacements.is_empty() {
        return Ok(());
    }

    // Rewrite all uses (including other phis' operands) before erasing, so
    // phi-to-phi references — cycles included — become block-argument uses.
    for &(from, to) in &replacements {
        from.replace_some_uses_with(ctx, |_, _| true, &to);
    }
    let replaced: FxHashMap<Value, Value> = replacements.into_iter().collect();
    for slots in edge_operands.values_mut() {
        for slot in slots.iter_mut().flatten() {
            if let Some(&arg) = replaced.get(slot) {
                *slot = arg;
            }
        }
    }
    for phi in phis_to_erase {
        Operation::erase(phi, ctx);
    }

    for &pred in &blocks {
        let Some(term) = pred.deref(ctx).get_terminator(ctx) else {
            continue;
        };
        let successors: Vec<_> = term.deref(ctx).successors().collect();

        let mut per_succ: Vec<Vec<Value>> = Vec::with_capacity(successors.len());
        let mut needs_rebuild = false;
        for &succ in &successors {
            let phi_count = phi_counts.get(&succ).copied().unwrap_or(0);
            if phi_count == 0 {
                per_succ.push(vec![]);
                continue;
            }
            let Some(slots) = edge_operands.get(&(pred, succ)) else {
                return arg_err!(
                    Location::Unknown,
                    "phis in ^{} carry no incoming value for predecessor ^{}",
                    succ.deref(ctx).unique_name(ctx),
                    pred.deref(ctx).unique_name(ctx)
                );
            };
            let mut operands = Vec::with_capacity(phi_count);
            for (phi_idx, slot) in slots.iter().enumerate() {
                let Some(value) = slot else {
                    return arg_err!(
                        Location::Unknown,
                        "phi {} in ^{} carries no incoming value for predecessor ^{}",
                        phi_idx,
                        succ.deref(ctx).unique_name(ctx),
                        pred.deref(ctx).unique_name(ctx)
                    );
                };
                operands.push(*value);
            }
            per_succ.push(operands);
            needs_rebuild = true;
        }
        if !needs_rebuild {
            continue;
        }

        let weights = term
            .deref(ctx)
            .attributes
            .get::<BranchWeightsAttr>(&ATTR_KEY_BRANCH_WEIGHTS)
            .cloned();
        let opid = Operation::get_opid(term, ctx);
        let new_term = if opid == llvm_ops::BrOp::get_opid_static() {
            let br = llvm_ops::BrOp::from_operation(term);
            let dest = br.get_dest(ctx);
            llvm_ops::BrOp::new(ctx, dest, per_succ.remove(0)).get_operation()
        } else if opid == llvm_ops::CondBrOp::get_opid_static() {
            let cbr = llvm_ops::CondBrOp::from_operation(term);
            let condition = cbr.get_condition(ctx);
            let true_dest = cbr.get_true_dest(ctx);
            let false_dest = cbr.get_false_dest(ctx);
            let false_operands = per_succ.remove(1);
            let true_operands = per_succ.remove(0);
            llvm_ops::CondBrOp::new(
                ctx,
                condition,
                true_dest,
                true_operands,
                false_dest,
                false_operands,
            )
            .get_operation()
        } else {
            return arg_err!(
                Location::Unknown,
                "cannot raise phis through terminator {} in ^{}",
                opid,
                pred.deref(ctx).unique_name(ctx)
            );
        };
        if let Some(weights) = weights {
            new_term
                .deref_mut(ctx)
                .attributes
                .set(ATTR_KEY_BRANCH_WEIGHTS.clone(), weights);
        }
        new_term.insert_before(ctx, term);
        Operation::erase(term, ctx);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dialects::{builtin, llvm, llvm::ops as llvm_ops},
        ir::r#type::{TypeHandle, TypedHandle},
        passes::lower_llvm_block_args_to_phi::LowerLLVMBlockArgsToPhiPass,
        printable::Printable,
    };

    fn test_context() -> Context {
        let mut ctx = Context::new();
        llvm::register(&mut ctx);
        ctx
    }

    #[test]
    fn raises_phi_back_to_block_args() {
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

        llvm_ops::BrOp::new(&mut ctx, then_block, vec![arg])
            .get_operation()
            .insert_at_back(entry, &ctx);
        let then_arg = then_block.deref(&ctx).get_argument(0);
        llvm_ops::ReturnOp::new(&mut ctx, Some(then_arg))
            .get_operation()
            .insert_at_back(then_block, &ctx);

        LowerLLVMBlockArgsToPhiPass
            .run(func.get_operation(), &mut ctx, PassOptions::default())
            .unwrap();
        let text = format!("{}", func.get_operation().disp(&ctx));
        assert!(text.contains("llvm.phi"));

        LowerLLVMPhiToBlockArgsPass
            .run(func.get_operation(), &mut ctx, PassOptions::default())
            .unwrap();

        let text = format!("{}", func.get_operation().disp(&ctx));
        assert!(!text.contains("llvm.phi"), "phis remain: {text}");
        // The branch carries the value again and the block declares an
        // argument for it.
        assert!(text.contains("llvm.br ^then"), "no branch: {text}");
        assert!(text.contains("^then"), "no labeled block: {text}");
        let then_args = then_block.deref(&ctx).get_num_arguments();
        assert_eq!(then_args, 1);
        let br_operands = llvm_ops::BrOp::from_operation(entry.deref(&ctx).get_terminator(&ctx).unwrap())
        .get_dest_operands(&ctx);
        assert_eq!(br_operands.len(), 1);
    }
}
