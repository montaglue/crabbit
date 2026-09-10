//! Aggressive dead-code elimination (docs/MIDEND-PLAN.md item 8):
//! mark-and-sweep over SSA. Roots are every block terminator and every op
//! with side effects (anything outside simplify's pure set — stores,
//! calls, ops with unknown effects); marking propagates backward through
//! operands; unmarked ops are erased.
//!
//! v1 limit, deliberate: no branch rewriting or unreachable-block folding
//! (llvm-simplify-cfg owns CFG cleanup), and since terminators are roots,
//! block arguments always keep their feeding branch operands — dead
//! block-arg cycles (an induction variable used only by itself through a
//! back edge) survive; removing them needs branch surgery.
//!
//! ADJOINT (backward attribution): deletion-only: erased ops need no adjoint (their cost ceases to exist).

use rustc_hash::FxHashSet;

use crate::{
    context::{Context, Ptr},
    dialects::llvm::op_interfaces::IsDeclaration,
    ir::{operation::Operation, region::Region},
    linked_list::ContainsLinkedList,
    conversion::pass::{AnalysisManager, Pass, PassResult, changed, unchanged},
};

use super::{
    inline::collect_functions,
    midend_gate::midend_disabled,
    simplify::pure_op_ids,
};
use pliron::builtin::op_interfaces::AtMostOneRegionInterface as _;

pub struct LLVMAdcePass;

impl Pass for LLVMAdcePass {
    fn name(&self) -> &str {
        "llvm-adce"
    }

    fn run(
        &mut self,
        root: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> pliron::result::Result<PassResult> {
        if midend_disabled("adce") {
            return Ok(unchanged());
        }
        let pure = pure_op_ids();
        let mut any = false;
        for func in collect_functions(ctx, root) {
            if func.is_declaration(ctx) {
                continue;
            }
            let region = func
                .get_region(ctx)
                .expect("llvm.func definition must have a body");
            any |= adce_region(ctx, region, &pure);
        }
        Ok(if any { changed() } else { unchanged() })
    }
}

fn adce_region(
    ctx: &mut Context,
    region: Ptr<Region>,
    pure: &FxHashSet<crate::ir::op::OpId>,
) -> bool {
    // Mark phase.
    let mut marked: FxHashSet<Ptr<Operation>> = FxHashSet::default();
    let mut worklist: Vec<Ptr<Operation>> = Vec::new();
    for block in region.deref(ctx).iter(ctx) {
        let terminator = block.deref(ctx).get_terminator(ctx);
        for op in block.deref(ctx).iter(ctx) {
            let opid = Operation::get_opid(op, ctx);
            let is_root = Some(op) == terminator || !pure.contains(&opid);
            if is_root && marked.insert(op) {
                worklist.push(op);
            }
        }
    }
    while let Some(op) = worklist.pop() {
        let operands: Vec<_> = {
            let operation = op.deref(ctx);
            (0..operation.get_num_operands())
                .map(|i| operation.get_operand(i))
                .collect()
        };
        for operand in operands {
            if let Some(def) = operand.defining_op()
                && marked.insert(def) {
                    worklist.push(def);
                }
            // Block-argument operands need no action: the branches that
            // feed them are terminators, i.e. already roots.
        }
    }

    // Sweep phase: erase unmarked ops once nothing uses them; iterate so
    // chains fall in dependency order.
    let mut any = false;
    loop {
        let mut erased = false;
        let ops: Vec<_> = region
            .deref(ctx)
            .iter(ctx)
            .flat_map(|block| block.deref(ctx).iter(ctx).collect::<Vec<_>>())
            .collect();
        for op in ops.into_iter().rev() {
            if marked.contains(&op) {
                continue;
            }
            let unused = {
                let operation = op.deref(ctx);
                (0..operation.get_num_results())
                    .all(|i| !operation.get_result(i).is_used(ctx))
            };
            if unused {
                Operation::erase(op, ctx);
                erased = true;
                any = true;
            }
        }
        if !erased {
            break;
        }
    }
    any
}

#[cfg(test)]
mod tests {
    use crate::ir::op::Op;
    use super::*;
    use pliron::builtin::op_interfaces::OneResultInterface as _;
    use pliron_llvm::op_interfaces::IntBinArithOpWithOverflowFlag as _;
    use crate::{
        dialects::{
            builtin::types::{IntegerType, Signedness},
            llvm::{
                attributes::LinkageAttr,
                ops::{AddOp, BrOp, FuncOp, GepIndex, GetElementPtrOp, MulOp, ReturnOp, StoreOp},
                types::{FuncType, PointerType},
            },
        },
        ir::{basic_block::BasicBlock, value::Value, r#type::TypeHandle},
        printable::Printable,
    };

    fn run_adce(ctx: &mut Context, func: FuncOp) -> String {
        LLVMAdcePass
            .run(func.get_operation(), ctx, &mut AnalysisManager::default())
            .unwrap();
        format!("{}", func.get_operation().disp(ctx))
    }

    fn scaffold(ctx: &mut Context) -> (FuncOp, Ptr<crate::ir::basic_block::BasicBlock>, Value, Value) {
        let i64_ty: TypeHandle =
            IntegerType::get(ctx, 64, Signedness::Signless).into();
        let ptr_ty: TypeHandle = PointerType::get(ctx, 0).into();
        let fn_ty = FuncType::get(ctx, i64_ty, vec![i64_ty, ptr_ty], false);
        let func = FuncOp::new(ctx, "f".try_into().unwrap(), fn_ty);
        func.set_attr_llvm_function_linkage(ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(ctx);
        let entry = func.get_entry_block(ctx).unwrap();
        let a = entry.deref(ctx).get_argument(0);
        let p = entry.deref(ctx).get_argument(1);
        (func, entry, a, p)
    }

    #[test]
    fn removes_cross_block_dead_chain() {
        let mut ctx = Context::new();
        let (func, entry, a, _p) = scaffold(&mut ctx);
        let region = func.get_region(&ctx).unwrap();
        let next = BasicBlock::new(&mut ctx, None, vec![]);
        next.insert_at_back(region, &ctx);
        // add in entry, its only use a mul in the next block, mul unused:
        // the whole chain must go (simplify's local DCE also would, but
        // adce does it in one marking pass).
        let add = AddOp::new_with_overflow_flag(&mut ctx, a, a, Default::default());
        add.get_operation().insert_at_back(entry, &ctx);
        let add_v = add.get_result(&ctx);
        BrOp::new(&mut ctx, next, vec![])
            .get_operation()
            .insert_at_back(entry, &ctx);
        let mul = MulOp::new_with_overflow_flag(&mut ctx, add_v, add_v, Default::default());
        mul.get_operation().insert_at_back(next, &ctx);
        ReturnOp::new(&mut ctx, Some(a))
            .get_operation()
            .insert_at_back(next, &ctx);

        let text = run_adce(&mut ctx, func);
        assert!(!text.contains("llvm.add"), "{text}");
        assert!(!text.contains("llvm.mul"), "{text}");
    }

    #[test]
    fn keeps_chain_feeding_a_store() {
        let mut ctx = Context::new();
        let (func, entry, a, p) = scaffold(&mut ctx);
        let i64_ty: TypeHandle =
            IntegerType::get(&mut ctx, 64, Signedness::Signless).into();
        // gep + add feed a store: everything stays (store is a root).
        let add = AddOp::new_with_overflow_flag(&mut ctx, a, a, Default::default());
        add.get_operation().insert_at_back(entry, &ctx);
        let add_v = add.get_result(&ctx);
        let gep = GetElementPtrOp::new(&mut ctx, p, vec![GepIndex::Value(a)], i64_ty);
        gep.get_operation().insert_at_back(entry, &ctx);
        let gep_v = gep.get_result(&ctx);
        StoreOp::new(&mut ctx, add_v, gep_v)
            .get_operation()
            .insert_at_back(entry, &ctx);
        ReturnOp::new(&mut ctx, Some(a))
            .get_operation()
            .insert_at_back(entry, &ctx);

        let text = run_adce(&mut ctx, func);
        assert!(text.contains("llvm.add"), "{text}");
        assert!(text.contains("llvm.gep"), "{text}");
        assert!(text.contains("llvm.store"), "{text}");
    }
}
