//! CFG simplification for the block-argument form of the LLVM dialect:
//! fold constant conditional branches, remove unreachable blocks, merge
//! straight-line block pairs and bypass empty forwarding blocks.
//!
//! The rewrites change CFG edges without updating `llvm.phi` incoming
//! lists, so functions containing phi ops are skipped.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    context::{Context, Ptr},
    dialects::{
        builtin::{op_interfaces::OneRegionInterface, types::IntegerType},
        llvm::{
            attributes::ICmpPredicateAttr,
            op_interfaces::IsDeclaration,
            ops::{BrOp, CondBrOp, FuncOp, ICmpOp, PhiOp},
        },
    },
    ir::{
        basic_block::BasicBlock, op::Op, operation::Operation, region::Region, r#type::Typed,
        value::Value,
    },
    linked_list::ContainsLinkedList,
    conversion::pass::{AnalysisManager, Pass, PassResult, changed},
};

use super::{inline::collect_functions, simplify::as_const_operand};

const MAX_ITERATIONS: usize = 32;

pub struct LLVMSimplifyCfgPass;

impl Pass for LLVMSimplifyCfgPass {
    fn name(&self) -> &str {
        "llvm-simplify-cfg"
    }

    fn run(&mut self, root: Ptr<Operation>, ctx: &mut Context, _analyses: &mut AnalysisManager) -> pliron::result::Result<PassResult> {
        for func in collect_functions(ctx, root) {
            if func.is_declaration(ctx) || contains_phi(ctx, func) {
                continue;
            }
            let region = func.get_region(ctx);
            for _ in 0..MAX_ITERATIONS {
                let mut changed = fold_constant_branches(ctx, region);
                changed |= remove_unreachable_blocks(ctx, region);
                changed |= merge_straight_line_blocks(ctx, region);
                changed |= bypass_forwarding_blocks(ctx, region);
                if !changed {
                    break;
                }
            }
        }
        Ok(changed())
    }
}

fn contains_phi(ctx: &Context, func: FuncOp) -> bool {
    for block in func.get_region(ctx).deref(ctx).iter(ctx) {
        for op in block.deref(ctx).iter(ctx) {
            if Operation::get_opid(op, ctx) == PhiOp::get_opid_static() {
                return true;
            }
        }
    }
    false
}

/// `llvm.cond_br` on a constant condition, or with identical destinations
/// and operands on both edges, becomes `llvm.br`.
fn fold_constant_branches(ctx: &mut Context, region: Ptr<Region>) -> bool {
    let mut changed = false;
    let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
    for block in blocks {
        let Some(term) = block.deref(ctx).get_terminator(ctx) else {
            continue;
        };
        if Operation::get_opid(term, ctx) != CondBrOp::get_opid_static() {
            continue;
        }
        let cond_br = CondBrOp::from_operation(term);

        // cond_br (icmp eq/ne %c, const-i1) — branch on %c directly,
        // swapping the destinations when the comparison negates it.
        if let Some((direct_cond, negated)) = negated_i1_condition(ctx, cond_br.get_condition(ctx))
        {
            let (true_dest, true_ops, false_dest, false_ops) = if negated {
                (
                    cond_br.get_false_dest(ctx),
                    cond_br.get_false_operands(ctx),
                    cond_br.get_true_dest(ctx),
                    cond_br.get_true_operands(ctx),
                )
            } else {
                (
                    cond_br.get_true_dest(ctx),
                    cond_br.get_true_operands(ctx),
                    cond_br.get_false_dest(ctx),
                    cond_br.get_false_operands(ctx),
                )
            };
            CondBrOp::new(ctx, direct_cond, true_dest, true_ops, false_dest, false_ops)
                .get_operation()
                .insert_before(ctx, term);
            Operation::erase(term, ctx);
            changed = true;
            continue;
        }

        let taken = if let Some(cond) = as_const_operand(ctx, cond_br.get_condition(ctx)) {
            Some(cond.is_true())
        } else if cond_br.get_true_dest(ctx) == cond_br.get_false_dest(ctx)
            && cond_br.get_true_operands(ctx) == cond_br.get_false_operands(ctx)
        {
            Some(true)
        } else {
            None
        };
        let Some(taken) = taken else {
            continue;
        };

        let (dest, operands) = if taken {
            (cond_br.get_true_dest(ctx), cond_br.get_true_operands(ctx))
        } else {
            (cond_br.get_false_dest(ctx), cond_br.get_false_operands(ctx))
        };
        BrOp::new(ctx, dest, operands)
            .get_operation()
            .insert_before(ctx, term);
        Operation::erase(term, ctx);
        changed = true;
    }
    changed
}

/// When `cond` is `icmp eq/ne %c, <const i1>` with `%c` itself i1-typed,
/// return `%c` and whether the comparison negates it.
fn negated_i1_condition(ctx: &Context, cond: Value) -> Option<(Value, bool)> {
    let Some(op) = cond.defining_op().filter(|_| cond.find_index(ctx) == 0) else {
        return None;
    };
    if Operation::get_opid(op, ctx) != ICmpOp::get_opid_static() {
        return None;
    }
    let icmp = ICmpOp::from_operation(op);
    let negate_on_zero = match icmp.get_predicate(ctx) {
        ICmpPredicateAttr::EQ => true,
        ICmpPredicateAttr::NE => false,
        _ => return None,
    };
    // Accept the constant on either side.
    let (value, constant) = if let Some(c) = as_const_operand(ctx, icmp.get_rhs(ctx)) {
        (icmp.get_lhs(ctx), c)
    } else if let Some(c) = as_const_operand(ctx, icmp.get_lhs(ctx)) {
        (icmp.get_rhs(ctx), c)
    } else {
        return None;
    };
    if constant.width != 1 {
        return None;
    }
    let value_is_i1 = value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|int_ty| int_ty.width() == 1);
    if !value_is_i1 {
        return None;
    }
    let negated = negate_on_zero ^ constant.is_true();
    Some((value, negated))
}

fn remove_unreachable_blocks(ctx: &mut Context, region: Ptr<Region>) -> bool {
    let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
    let Some(&entry) = blocks.first() else {
        return false;
    };

    let mut reachable = FxHashSet::default();
    let mut stack = vec![entry];
    while let Some(block) = stack.pop() {
        if !reachable.insert(block) {
            continue;
        }
        for succ in block.deref(ctx).succs(ctx) {
            stack.push(succ);
        }
    }

    let dead: Vec<_> = blocks
        .iter()
        .copied()
        .filter(|block| !reachable.contains(block))
        .collect();
    if dead.is_empty() {
        return false;
    }

    // Sever all def-use edges out of dead ops first: dead blocks may
    // reference each other (branches, operands) in arbitrary order.
    for &block in &dead {
        let ops: Vec<_> = block.deref(ctx).iter(ctx).collect();
        for op in ops {
            Operation::drop_all_uses(op, ctx);
        }
    }
    for &block in &dead {
        let ops: Vec<_> = block.deref(ctx).iter(ctx).collect();
        for op in ops {
            Operation::erase(op, ctx);
        }
        BasicBlock::erase(block, ctx);
    }
    true
}

/// Merge `B -> S` when `B` unconditionally branches to `S` and `S` has no
/// other predecessor: `S`'s arguments become the branch operands, `S`'s
/// body is appended to `B`.
fn merge_straight_line_blocks(ctx: &mut Context, region: Ptr<Region>) -> bool {
    let mut changed = false;
    loop {
        let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
        let mut merged = false;
        for block in blocks {
            let Some(term) = block.deref(ctx).get_terminator(ctx) else {
                continue;
            };
            if Operation::get_opid(term, ctx) != BrOp::get_opid_static() {
                continue;
            }
            let br = BrOp::from_operation(term);
            let succ = br.get_dest(ctx);
            if succ == block || succ.num_preds(ctx) != 1 {
                continue;
            }
            // Never merge away the entry block: its arguments are the
            // function parameters.
            if Some(succ) == region.deref(ctx).get_head() {
                continue;
            }

            let operands = br.get_dest_operands(ctx);
            if operands.len() != succ.deref(ctx).get_num_arguments() {
                continue;
            }
            for (arg_idx, operand) in operands.iter().enumerate() {
                let arg = succ.deref(ctx).get_argument(arg_idx);
                arg.replace_some_uses_with(ctx, |_, _| true, operand);
            }
            Operation::erase(term, ctx);

            let ops: Vec<_> = succ.deref(ctx).iter(ctx).collect();
            for op in ops {
                op.unlink(ctx);
                op.insert_at_back(block, ctx);
            }
            while succ.deref(ctx).get_num_arguments() > 0 {
                BasicBlock::pop_argument(succ, ctx);
            }
            BasicBlock::erase(succ, ctx);
            merged = true;
            changed = true;
            break;
        }
        if !merged {
            break;
        }
    }
    changed
}

/// Retarget predecessors of an empty block that only forwards to another
/// block, substituting the forwarded operands into each predecessor edge.
fn bypass_forwarding_blocks(ctx: &mut Context, region: Ptr<Region>) -> bool {
    let mut changed = false;
    let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
    let Some(&entry) = blocks.first() else {
        return false;
    };

    for &block in blocks.iter().skip(1) {
        if block == entry {
            continue;
        }
        // The block must contain exactly one op: an unconditional branch.
        let Some(term) = block.deref(ctx).get_head() else {
            continue;
        };
        if block.deref(ctx).get_tail() != Some(term)
            || Operation::get_opid(term, ctx) != BrOp::get_opid_static()
        {
            continue;
        }
        let br = BrOp::from_operation(term);
        let dest = br.get_dest(ctx);
        if dest == block {
            continue;
        }
        let forwarded = br.get_dest_operands(ctx);
        let block_args: Vec<Value> = block.deref(ctx).arguments().collect();

        // The block dominates its destination, so downstream blocks may use
        // its arguments directly. Bypassing is only sound when every
        // argument is consumed solely by this block's own branch.
        if block_args.iter().any(|arg| {
            arg.uses(ctx)
                .iter()
                .any(|arg_use| arg_use.user_op() != term)
        }) {
            continue;
        }

        // Map each forwarded operand to either a predecessor edge operand
        // (if it is one of this block's arguments) or itself.
        let arg_index: FxHashMap<Value, usize> = block_args
            .iter()
            .enumerate()
            .map(|(idx, &arg)| (arg, idx))
            .collect();

        let preds: Vec<_> = block.preds(ctx);
        for pred in preds {
            let Some(pred_term) = pred.deref(ctx).get_terminator(ctx) else {
                continue;
            };
            let substituted = |edge_operands: &[Value]| -> Option<Vec<Value>> {
                if edge_operands.len() != block_args.len() {
                    return None;
                }
                Some(
                    forwarded
                        .iter()
                        .map(|value| match arg_index.get(value) {
                            Some(&idx) => edge_operands[idx],
                            None => *value,
                        })
                        .collect(),
                )
            };

            let pred_opid = Operation::get_opid(pred_term, ctx);
            if pred_opid == BrOp::get_opid_static() {
                let pred_br = BrOp::from_operation(pred_term);
                if pred_br.get_dest(ctx) != block {
                    continue;
                }
                let Some(new_operands) = substituted(&pred_br.get_dest_operands(ctx)) else {
                    continue;
                };
                BrOp::new(ctx, dest, new_operands)
                    .get_operation()
                    .insert_before(ctx, pred_term);
                Operation::erase(pred_term, ctx);
                changed = true;
            } else if pred_opid == CondBrOp::get_opid_static() {
                let pred_cbr = CondBrOp::from_operation(pred_term);
                let (mut true_dest, mut true_ops) =
                    (pred_cbr.get_true_dest(ctx), pred_cbr.get_true_operands(ctx));
                let (mut false_dest, mut false_ops) = (
                    pred_cbr.get_false_dest(ctx),
                    pred_cbr.get_false_operands(ctx),
                );
                let mut any = false;
                if true_dest == block {
                    let Some(new_ops) = substituted(&true_ops) else {
                        continue;
                    };
                    true_ops = new_ops;
                    true_dest = dest;
                    any = true;
                }
                if false_dest == block {
                    let Some(new_ops) = substituted(&false_ops) else {
                        continue;
                    };
                    false_ops = new_ops;
                    false_dest = dest;
                    any = true;
                }
                if !any {
                    continue;
                }
                let condition = pred_cbr.get_condition(ctx);
                CondBrOp::new(ctx, condition, true_dest, true_ops, false_dest, false_ops)
                    .get_operation()
                    .insert_before(ctx, pred_term);
                Operation::erase(pred_term, ctx);
                changed = true;
            }
        }
        // Once all predecessors are retargeted the block is unreachable
        // and the next remove_unreachable_blocks sweep deletes it.
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dialects::{
            builtin::{
                self, attributes::IntegerAttr, op_interfaces::OneResultInterface,
                types::Signedness,
            },
            llvm::{
                self,
                attributes::LinkageAttr,
                ops::{ConstantOp, ReturnOp},
                types::FuncType,
            },
        },
        ir::r#type::{TypeHandle, TypedHandle},
        printable::Printable,
        utils::apint::APInt,
    };
    use std::num::NonZero;

    fn test_context() -> Context {
        let mut ctx = Context::new();
        llvm::register(&mut ctx);
        ctx
    }

    #[test]
    fn folds_constant_cond_br_and_prunes_dead_block() {
        let mut ctx = test_context();
        let i64_ty: TypeHandle =
            builtin::types::IntegerType::get(&mut ctx, 64, Signedness::Signless).into();
        let fn_ty: TypedHandle<FuncType> = FuncType::get(&mut ctx, i64_ty, vec![], false);
        let func = FuncOp::new(
            &mut ctx,
            "f".try_into().unwrap(),
            fn_ty,
            LinkageAttr::External,
        );
        let entry = func.get_entry_block(&ctx);
        let then_block = BasicBlock::new(&mut ctx, Some("then".try_into().unwrap()), vec![]);
        let else_block = BasicBlock::new(&mut ctx, Some("else".try_into().unwrap()), vec![]);
        then_block.insert_at_back(func.get_region(&ctx), &ctx);
        else_block.insert_at_back(func.get_region(&ctx), &ctx);

        let i1_ty = builtin::types::IntegerType::get(&mut ctx, 1, Signedness::Signless);
        let cond = ConstantOp::new_integer(
            &mut ctx,
            IntegerAttr::new(i1_ty, APInt::from_u64(1, NonZero::new(1).unwrap())),
        );
        cond.get_operation().insert_at_back(entry, &ctx);
        let cond_v = cond.get_result(&ctx);
        CondBrOp::new(&mut ctx, cond_v, then_block, vec![], else_block, vec![])
            .get_operation()
            .insert_at_back(entry, &ctx);

        let i64_int = builtin::types::IntegerType::get(&mut ctx, 64, Signedness::Signless);
        let ten = ConstantOp::new_integer(
            &mut ctx,
            IntegerAttr::new(i64_int, APInt::from_u64(10, NonZero::new(64).unwrap())),
        );
        ten.get_operation().insert_at_back(then_block, &ctx);
        let ten_v = ten.get_result(&ctx);
        ReturnOp::new(&mut ctx, Some(ten_v))
            .get_operation()
            .insert_at_back(then_block, &ctx);

        let twenty = ConstantOp::new_integer(
            &mut ctx,
            IntegerAttr::new(i64_int, APInt::from_u64(20, NonZero::new(64).unwrap())),
        );
        twenty.get_operation().insert_at_back(else_block, &ctx);
        let twenty_v = twenty.get_result(&ctx);
        ReturnOp::new(&mut ctx, Some(twenty_v))
            .get_operation()
            .insert_at_back(else_block, &ctx);

        LLVMSimplifyCfgPass
            .run(func.get_operation(), &mut ctx, &mut AnalysisManager::default())
            .unwrap();

        let text = format!("{}", func.get_operation().disp(&ctx));
        assert!(!text.contains("llvm.cond_br"), "{text}");
        assert!(text.contains("<10: i64>"), "{text}");
        assert!(!text.contains("<20: i64>"), "{text}");
        // then-block merged into entry: one block remains.
        assert!(!text.contains("^then"), "{text}");
    }
}
