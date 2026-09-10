//! Code sinking (docs/MIDEND-PLAN.md item 7): move a pure single-result
//! op down to the deepest block that still dominates all its uses,
//! shortening live ranges.
//!
//! Placement rule (a deliberate deviation from the plan sketch's
//! "post-dominates all uses", which cannot hold for a def): the target is
//! the nearest common *dominator* of the use blocks — the deepest block
//! through which every path to every use passes — provided it is
//! dominated by (and different from) the defining block, so the move is
//! strictly downward and def-dominates-use is preserved. The op is
//! inserted at the target's front; ops are visited in reverse program
//! order so chains sinking to one block keep their relative order.
//!
//! Never sinks into a loop the defining block is not already in (the
//! containing-loop set of the target must be a subset of the def block's),
//! since that would multiply executions. Because the target is dominated
//! by the def block, execution counts otherwise only shrink — which also
//! makes sinking div/rem safe (no new traps on any path).
//!
//! NOTE for the experiments: this pass changes register pressure — the RA
//! experiments' independent variable. It is individually toggleable
//! (`CRABBIT_MIDEND_DISABLE=sink`) and every sweep records the axis.
//!
//! ADJOINT (backward attribution): sink = pure move: the op keeps its identity attributes; no adjoint needed.

use rustc_hash::FxHashSet;

use crate::{
    context::{Context, Ptr},
    dialects::llvm::{
        op_interfaces::IsDeclaration,
        ops::{SDivOp, SRemOp, UDivOp, URemOp},
    },
    ir::{
        basic_block::BasicBlock,
        op::{Op, OpId},
        operation::Operation,
        region::Region,
    },
    linked_list::{ContainsLinkedList, LinkedList as _},
    conversion::pass::{AnalysisManager, Pass, PassResult, changed, unchanged},
};

use super::{
    analysis::{PostDomTree, RegionCfg, dominator_tree, natural_loops, nearest_common_dominator},
    inline::collect_functions,
    licm::hoistable_op_ids,
    midend_gate::midend_disabled,
};
use pliron::builtin::op_interfaces::AtMostOneRegionInterface as _;
use crate::dialects::builtin::ops::ConstantOp;
use crate::dialects::llvm::ops::{PoisonOp, UndefOp};

const MAX_ITERATIONS: usize = 4;

pub struct LLVMSinkPass {
    /// [TargetProfile::has_branch_divergence]: when set, a sink may not
    /// add control dependence — the target block must POST-dominate the
    /// defining block, so the op still executes exactly when it did
    /// before. Without divergence the plain nearest-common-dominator
    /// placement applies (moving under a guard genuinely skips work on a
    /// CPU).
    ///
    /// [TargetProfile::has_branch_divergence]: crate::target_profile::TargetProfile
    divergent_target: bool,
}

impl LLVMSinkPass {
    pub fn new(profile: &crate::target_profile::TargetProfile) -> Self {
        LLVMSinkPass {
            divergent_target: profile.has_branch_divergence,
        }
    }
}

impl Default for LLVMSinkPass {
    /// Host-CPU behavior (the original pass).
    fn default() -> Self {
        LLVMSinkPass {
            divergent_target: false,
        }
    }
}

impl Pass for LLVMSinkPass {
    fn name(&self) -> &str {
        "llvm-sink"
    }

    fn run(
        &mut self,
        root: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> pliron::result::Result<PassResult> {
        if midend_disabled("sink") {
            return Ok(unchanged());
        }
        let sinkable = sinkable_op_ids();
        let mut any = false;
        for func in collect_functions(ctx, root) {
            if func.is_declaration(ctx) {
                continue;
            }
            let region = func
                .get_region(ctx)
                .expect("llvm.func definition must have a body");
            for _ in 0..MAX_ITERATIONS {
                if !sink_in_region(ctx, region, &sinkable, self.divergent_target) {
                    break;
                }
                any = true;
            }
        }
        Ok(if any { changed() } else { unchanged() })
    }
}

/// licm's hoistable set (pure, single result) plus div/rem — safe to sink,
/// see module docs — plus the operand-less constants, whose live ranges
/// only shrink by moving toward uses. Allocas, loads and address-carrying
/// ops stay put.
fn sinkable_op_ids() -> FxHashSet<OpId> {
    let mut ids = hoistable_op_ids();
    ids.insert(UDivOp::get_opid_static());
    ids.insert(SDivOp::get_opid_static());
    ids.insert(URemOp::get_opid_static());
    ids.insert(SRemOp::get_opid_static());
    ids.insert(ConstantOp::get_opid_static());
    ids.insert(UndefOp::get_opid_static());
    ids.insert(PoisonOp::get_opid_static());
    ids
}

fn sink_in_region(
    ctx: &mut Context,
    region: Ptr<Region>,
    sinkable: &FxHashSet<OpId>,
    divergent_target: bool,
) -> bool {
    let dom = dominator_tree(ctx, region);
    // On divergent targets a sink must not add control dependence: the
    // target has to post-dominate the def block (see LLVMSinkPass docs).
    let no_new_control_dep = if divergent_target {
        let cfg = RegionCfg::new(ctx, region);
        let pdt = PostDomTree::compute(&cfg);
        Some((cfg, pdt))
    } else {
        None
    };
    let loops = natural_loops(ctx, &dom);
    let loops_of = |block: Ptr<BasicBlock>| -> Vec<usize> {
        loops
            .iter()
            .enumerate()
            .filter(|(_, l)| l.body.contains(&block))
            .map(|(i, _)| i)
            .collect()
    };
    let mut ops: Vec<Ptr<Operation>> = region
        .deref(ctx)
        .iter(ctx)
        .flat_map(|block| block.deref(ctx).iter(ctx).collect::<Vec<_>>())
        .collect();
    ops.reverse();

    let mut moved_any = false;
    for op in ops {
        let opid = Operation::get_opid(op, ctx);
        if !sinkable.contains(&opid) {
            continue;
        }
        if op.deref(ctx).get_num_results() != 1 {
            continue;
        }
        let Some(def_block) = op.deref(ctx).get_container() else {
            continue;
        };
        let use_blocks: Vec<Ptr<BasicBlock>> = {
            let result = op.deref(ctx).get_result(0);
            let mut blocks: Vec<Ptr<BasicBlock>> = Vec::new();
            let mut ok = true;
            for r#use in result.uses(ctx) {
                match r#use.user_op().deref(ctx).get_container() {
                    Some(block) => {
                        if !blocks.contains(&block) {
                            blocks.push(block);
                        }
                    }
                    None => ok = false,
                }
            }
            if !ok || blocks.is_empty() {
                continue;
            }
            blocks
        };
        let Some(target) = nearest_common_dominator(ctx, &dom, &use_blocks) else {
            continue;
        };
        if target == def_block || !dom.contains(&def_block) || !dom.dominates(&def_block, &target)
        {
            continue;
        }
        if let Some((cfg, pdt)) = &no_new_control_dep
            && !pdt.postdominates(cfg.idx(target), cfg.idx(def_block)) {
                continue;
            }
        // Never into a loop the def is not already in.
        let target_loops = loops_of(target);
        let def_loops = loops_of(def_block);
        if !target_loops.iter().all(|l| def_loops.contains(l)) {
            continue;
        }
        op.unlink(ctx);
        op.insert_at_front(target, ctx);
        moved_any = true;
    }
    moved_any
}

#[cfg(test)]
mod tests {
    use super::*;
    use pliron::builtin::op_interfaces::OneResultInterface as _;
    use pliron_llvm::op_interfaces::IntBinArithOpWithOverflowFlag as _;
    use crate::{
        dialects::{
            builtin::types::{IntegerType, Signedness},
            llvm::{
                attributes::LinkageAttr,
                ops::{AddOp, BrOp, CondBrOp, FuncOp, ReturnOp, UndefOp},
                types::FuncType,
            },
        },
        ir::{value::Value, r#type::TypeHandle},
        printable::Printable,
    };

    fn run_sink(ctx: &mut Context, func: FuncOp) {
        LLVMSinkPass::default()
            .run(func.get_operation(), ctx, &mut AnalysisManager::default())
            .unwrap();
    }

    fn scaffold(ctx: &mut Context) -> (FuncOp, Ptr<BasicBlock>, Value, Value) {
        let i64_ty: TypeHandle =
            IntegerType::get(ctx, 64, Signedness::Signless).into();
        let i1: TypeHandle = IntegerType::get(ctx, 1, Signedness::Signless).into();
        let fn_ty = FuncType::get(ctx, i64_ty, vec![i64_ty], false);
        let func = FuncOp::new(ctx, "f".try_into().unwrap(), fn_ty);
        func.set_attr_llvm_function_linkage(ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(ctx);
        let entry = func.get_entry_block(ctx).unwrap();
        let a = entry.deref(ctx).get_argument(0);
        let cond = UndefOp::new(ctx, i1);
        cond.get_operation().insert_at_back(entry, ctx);
        let cond_v = cond.get_result(ctx);
        (func, entry, a, cond_v)
    }

    #[test]
    fn divergent_profile_refuses_sink_under_branch() {
        // Same CFG as sinks_op_to_its_single_use_branch, but on a
        // divergent target: `then_b` does not post-dominate `entry`, so
        // the add must stay in the entry block.
        let mut ctx = Context::new();
        let (func, entry, a, cond) = scaffold(&mut ctx);
        let region = func.get_region(&ctx).unwrap();
        let then_b = BasicBlock::new(&mut ctx, None, vec![]);
        then_b.insert_at_back(region, &ctx);
        let else_b = BasicBlock::new(&mut ctx, None, vec![]);
        else_b.insert_at_back(region, &ctx);
        let add = AddOp::new_with_overflow_flag(&mut ctx, a, a, Default::default());
        add.get_operation().insert_at_back(entry, &ctx);
        let add_v = add.get_result(&ctx);
        CondBrOp::new(&mut ctx, cond, then_b, vec![], else_b, vec![])
            .get_operation()
            .insert_at_back(entry, &ctx);
        ReturnOp::new(&mut ctx, Some(add_v))
            .get_operation()
            .insert_at_back(then_b, &ctx);
        ReturnOp::new(&mut ctx, Some(a))
            .get_operation()
            .insert_at_back(else_b, &ctx);

        let sinkable = sinkable_op_ids();
        let moved = sink_in_region(&mut ctx, region, &sinkable, true);
        assert!(!moved, "divergent target must not sink under a branch");
        assert_eq!(
            add.get_operation().deref(&ctx).get_container(),
            Some(entry),
            "add must remain unconditional:\n{}",
            func.get_operation().disp(&ctx)
        );
    }

    #[test]
    fn sinks_op_to_its_single_use_branch() {
        let mut ctx = Context::new();
        let (func, entry, a, cond) = scaffold(&mut ctx);
        let region = func.get_region(&ctx).unwrap();
        let then_b = BasicBlock::new(&mut ctx, None, vec![]);
        then_b.insert_at_back(region, &ctx);
        let else_b = BasicBlock::new(&mut ctx, None, vec![]);
        else_b.insert_at_back(region, &ctx);

        let add = AddOp::new_with_overflow_flag(&mut ctx, a, a, Default::default());
        add.get_operation().insert_at_back(entry, &ctx);
        let add_v = add.get_result(&ctx);
        CondBrOp::new(&mut ctx, cond, then_b, vec![], else_b, vec![])
            .get_operation()
            .insert_at_back(entry, &ctx);
        ReturnOp::new(&mut ctx, Some(add_v))
            .get_operation()
            .insert_at_back(then_b, &ctx);
        ReturnOp::new(&mut ctx, Some(a))
            .get_operation()
            .insert_at_back(else_b, &ctx);

        run_sink(&mut ctx, func);
        assert_eq!(
            add.get_operation().deref(&ctx).get_container(),
            Some(then_b),
            "add used only on the then edge must sink there:\n{}",
            func.get_operation().disp(&ctx)
        );
    }

    #[test]
    fn refuses_to_sink_into_a_loop() {
        let mut ctx = Context::new();
        let (func, entry, a, cond) = scaffold(&mut ctx);
        let region = func.get_region(&ctx).unwrap();
        let header = BasicBlock::new(&mut ctx, None, vec![]);
        header.insert_at_back(region, &ctx);
        let body = BasicBlock::new(&mut ctx, None, vec![]);
        body.insert_at_back(region, &ctx);
        let exit = BasicBlock::new(&mut ctx, None, vec![]);
        exit.insert_at_back(region, &ctx);

        let add = AddOp::new_with_overflow_flag(&mut ctx, a, a, Default::default());
        add.get_operation().insert_at_back(entry, &ctx);
        let add_v = add.get_result(&ctx);
        BrOp::new(&mut ctx, header, vec![])
            .get_operation()
            .insert_at_back(entry, &ctx);
        CondBrOp::new(&mut ctx, cond, body, vec![], exit, vec![])
            .get_operation()
            .insert_at_back(header, &ctx);
        // The only use is inside the loop body: sinking would multiply
        // executions, so the op must stay in the entry block.
        let mul = crate::dialects::llvm::ops::MulOp::new_with_overflow_flag(
            &mut ctx,
            add_v,
            add_v,
            Default::default(),
        );
        mul.get_operation().insert_at_back(body, &ctx);
        BrOp::new(&mut ctx, header, vec![])
            .get_operation()
            .insert_at_back(body, &ctx);
        ReturnOp::new(&mut ctx, Some(a))
            .get_operation()
            .insert_at_back(exit, &ctx);

        run_sink(&mut ctx, func);
        assert_eq!(
            add.get_operation().deref(&ctx).get_container(),
            Some(entry),
            "must not sink into the loop:\n{}",
            func.get_operation().disp(&ctx)
        );
    }

    #[test]
    fn sinks_to_nearest_common_dominator_of_two_uses() {
        let mut ctx = Context::new();
        let (func, entry, a, cond) = scaffold(&mut ctx);
        let region = func.get_region(&ctx).unwrap();
        let mid = BasicBlock::new(&mut ctx, None, vec![]);
        mid.insert_at_back(region, &ctx);
        let then_b = BasicBlock::new(&mut ctx, None, vec![]);
        then_b.insert_at_back(region, &ctx);
        let else_b = BasicBlock::new(&mut ctx, None, vec![]);
        else_b.insert_at_back(region, &ctx);

        let add = AddOp::new_with_overflow_flag(&mut ctx, a, a, Default::default());
        add.get_operation().insert_at_back(entry, &ctx);
        let add_v = add.get_result(&ctx);
        BrOp::new(&mut ctx, mid, vec![])
            .get_operation()
            .insert_at_back(entry, &ctx);
        CondBrOp::new(&mut ctx, cond, then_b, vec![], else_b, vec![])
            .get_operation()
            .insert_at_back(mid, &ctx);
        ReturnOp::new(&mut ctx, Some(add_v))
            .get_operation()
            .insert_at_back(then_b, &ctx);
        ReturnOp::new(&mut ctx, Some(add_v))
            .get_operation()
            .insert_at_back(else_b, &ctx);

        run_sink(&mut ctx, func);
        assert_eq!(
            add.get_operation().deref(&ctx).get_container(),
            Some(mid),
            "add used on both branch arms must sink to their common dominator:\n{}",
            func.get_operation().disp(&ctx)
        );
    }
}
