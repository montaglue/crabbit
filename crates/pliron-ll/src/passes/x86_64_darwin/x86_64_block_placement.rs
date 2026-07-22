//! Hot-path-driven basic block placement for x86-64 machine functions.
//!
//! This is the example consumer of the generic [hot path
//! analysis](crate::passes::hot_path), modeled on LLVM's
//! `MachineBlockPlacement`, which runs after register allocation and before
//! branch relaxation. It lays the hottest control-flow chains out
//! contiguously so that likely edges become fall-throughs, then rewrites
//! branches against the new layout:
//!
//! - an unconditional `b` to the next block in layout is deleted
//!   (fall-through);
//! - a `b_cond` whose taken target became the next block is inverted so the
//!   cold edge is the branch and the hot edge falls through.
//!
//! Edge probabilities come from the `branch_weights` transferred onto
//! conditional branches by instruction selection (the machine-level analogue
//! of `MachineBasicBlock` successor probabilities); branches without weights
//! fall back to the analysis' static loop heuristics.

use std::collections::HashMap;

use crate::{
    context::{Context, Ptr},
    dialects::x86_64::ops::{self as x86_64_ops, JccOp, JmpOp, Ud2Op, TestJnzOp, FuncOp, RetOp},
    dialects::builtin::op_interfaces::OneRegionInterface,
    ir::{basic_block::BasicBlock, operation::Operation},
    linked_list::ContainsLinkedList,
    conversion::pass::{AnalysisManager, Pass, PassResult, changed},
    passes::hot_path::{CfgEdge, HotPathInfo},
};

use super::{frontend::module_op, util::cast_operation};

pub struct X86_64BlockPlacementPass;

impl Pass for X86_64BlockPlacementPass {
    fn name(&self) -> &str {
        "x86-64-block-placement"
    }

    fn run(&mut self, root: Ptr<Operation>, ctx: &mut Context, _analyses: &mut AnalysisManager) -> pliron::result::Result<PassResult> {
        let module = module_op(ctx, root)?;
        let body = module.get_region(ctx).deref(ctx).get_head().unwrap();
        let ops: Vec<_> = body.deref(ctx).iter(ctx).collect();
        for op in ops {
            if let Some(func) = cast_operation::<FuncOp>(ctx, op) {
                place_blocks(ctx, func);
            }
        }
        Ok(changed())
    }
}

fn place_blocks(ctx: &mut Context, func: FuncOp) {
    let region = func.get_region(ctx);
    let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
    if blocks.len() < 2 {
        return;
    }

    // Phase 0: make every fall-through explicit so reordering cannot change
    // behavior (and the pass stays correct on its own output).
    materialize_fallthrough_branches(ctx, &blocks);

    // Phase 1: recover the CFG from the branches' successor blocks and read
    // the branch weights left by instruction selection.
    let block_index: HashMap<_, _> = blocks
        .iter()
        .enumerate()
        .map(|(index, block)| (*block, index))
        .collect();
    let successors = machine_cfg_edges(ctx, &blocks, &block_index);

    // Phase 2: frequencies and the new layout.
    let info = HotPathInfo::from_edges(blocks.clone(), successors.clone());
    let order = chain_order(&blocks, &successors, &info);

    // Phase 3: apply the layout. Re-appending every non-entry block in chain
    // order leaves the region as [entry, order[1], order[2], ...].
    for &index in &order[1..] {
        blocks[index].unlink(ctx);
        blocks[index].insert_at_back(region, ctx);
    }

    // Phase 4: turn branches along the layout into fall-throughs.
    let layout: Vec<_> = region.deref(ctx).iter(ctx).collect();
    simplify_branches_for_layout(ctx, &layout);
}

/// Kinds of control transfer recognized in machine blocks.
enum ControlFlowOp {
    /// `jcc`/`test_jnz`: branches to its successor block, otherwise falls
    /// through.
    Conditional {
        op: Ptr<Operation>,
        target: Ptr<BasicBlock>,
        weights: Option<(u32, u32)>,
    },
    /// `jmp`: always branches to its successor block.
    Unconditional {
        op: Ptr<Operation>,
        target: Ptr<BasicBlock>,
    },
    /// `ret`/`brk`: leaves the function.
    Exit,
}

fn classify_control_flow(ctx: &Context, op: Ptr<Operation>) -> Option<ControlFlowOp> {
    let op_obj = Operation::get_op_dyn(op, ctx);
    if op_obj.downcast_ref::<RetOp>().is_some() || op_obj.downcast_ref::<Ud2Op>().is_some() {
        return Some(ControlFlowOp::Exit);
    }
    let target = x86_64_ops::target(ctx, op)?;
    if op_obj.downcast_ref::<JmpOp>().is_some() {
        return Some(ControlFlowOp::Unconditional { op, target });
    }
    if op_obj.downcast_ref::<JccOp>().is_some() || op_obj.downcast_ref::<TestJnzOp>().is_some() {
        return Some(ControlFlowOp::Conditional {
            op,
            target,
            weights: x86_64_ops::branch_weights(ctx, op),
        });
    }
    None
}

/// The control-flow structure of a block: its conditional branches in order,
/// then how the block ends (`None` means it falls through).
fn block_control_flow(
    ctx: &Context,
    block: Ptr<BasicBlock>,
) -> (Vec<ControlFlowOp>, Option<ControlFlowOp>) {
    let mut conditionals = vec![];
    for op in block.deref(ctx).iter(ctx) {
        match classify_control_flow(ctx, op) {
            Some(cond @ ControlFlowOp::Conditional { .. }) => conditionals.push(cond),
            Some(end @ (ControlFlowOp::Unconditional { .. } | ControlFlowOp::Exit)) => {
                return (conditionals, Some(end));
            }
            None => {}
        }
    }
    (conditionals, None)
}

fn materialize_fallthrough_branches(ctx: &mut Context, blocks: &[Ptr<BasicBlock>]) {
    for pair in blocks.windows(2) {
        let (block, next) = (pair[0], pair[1]);
        let (_, end) = block_control_flow(ctx, block);
        if end.is_none() {
            x86_64_ops::jmp(ctx, next).insert_at_back(block, ctx);
        }
    }
}

/// Outgoing edges of each block. Weighted conditional branches yield weighted
/// edges; everything else is left to the analysis' heuristics. Blocks with
/// more than one conditional branch keep unweighted edges, since a single
/// `[taken, not-taken]` pair no longer describes them.
fn machine_cfg_edges(
    ctx: &Context,
    blocks: &[Ptr<BasicBlock>],
    block_index: &HashMap<Ptr<BasicBlock>, usize>,
) -> Vec<Vec<CfgEdge>> {
    blocks
        .iter()
        .map(|&block| {
            let (conditionals, end) = block_control_flow(ctx, block);
            let simple = conditionals.len() == 1;
            let mut edges = vec![];
            let mut not_taken_weight = None;
            for conditional in &conditionals {
                let ControlFlowOp::Conditional {
                    target, weights, ..
                } = conditional
                else {
                    unreachable!("block_control_flow only collects conditionals");
                };
                let weights = weights.filter(|_| simple);
                if let Some(&to) = block_index.get(target) {
                    edges.push(CfgEdge {
                        to,
                        weight: weights.map(|(taken, _)| taken),
                    });
                }
                not_taken_weight = weights.map(|(_, not_taken)| not_taken);
            }
            if let Some(ControlFlowOp::Unconditional { target, .. }) = end
                && let Some(&to) = block_index.get(&target)
            {
                edges.push(CfgEdge {
                    to,
                    weight: not_taken_weight,
                });
            }
            edges
        })
        .collect()
}

/// Greedy chain formation: starting at the entry, keep extending with the
/// unplaced successor of highest edge frequency; when the chain dies out,
/// restart from the hottest unplaced block. A simplified form of the chain
/// algorithm in LLVM's MachineBlockPlacement.
fn chain_order(
    blocks: &[Ptr<BasicBlock>],
    successors: &[Vec<CfgEdge>],
    info: &HotPathInfo,
) -> Vec<usize> {
    let mut placed = vec![false; blocks.len()];
    let mut order = Vec::with_capacity(blocks.len());
    let mut current = 0;
    placed[0] = true;
    order.push(0);

    while order.len() < blocks.len() {
        // Ties prefer the lower block index, i.e. the original layout.
        let best_successor = successors[current]
            .iter()
            .map(|edge| edge.to)
            .filter(|&to| !placed[to])
            .max_by_key(|&to| {
                (
                    info.edge_frequency(blocks[current], blocks[to]),
                    usize::MAX - to,
                )
            });
        let next = best_successor.unwrap_or_else(|| {
            (0..blocks.len())
                .filter(|&block| !placed[block])
                .max_by_key(|&block| (info.block_frequency(blocks[block]), usize::MAX - block))
                .expect("an unplaced block exists while order is incomplete")
        });
        placed[next] = true;
        order.push(next);
        current = next;
    }
    order
}

fn simplify_branches_for_layout(ctx: &mut Context, layout: &[Ptr<BasicBlock>]) {
    for (index, &block) in layout.iter().enumerate() {
        let Some(&next) = layout.get(index + 1) else {
            continue;
        };
        let (conditionals, end) = block_control_flow(ctx, block);
        let Some(ControlFlowOp::Unconditional {
            op: branch_op,
            target: branch_target,
        }) = end
        else {
            continue;
        };

        if branch_target == next {
            // The unconditional branch just falls through now.
            Operation::erase(branch_op, ctx);
            continue;
        }

        // `b_cond taken; b other` where `taken` is the next block: invert the
        // condition so the hot edge falls through and the cold edge branches.
        if let [
            ControlFlowOp::Conditional {
                op: cond_op,
                target: cond_target,
                weights,
            },
        ] = conditionals.as_slice()
            && *cond_target == next
        {
            let op_obj = Operation::get_op_dyn(*cond_op, ctx);
            if op_obj.downcast_ref::<JccOp>().is_none() {
                continue;
            }
            let Some(cond) = x86_64_ops::cond(ctx, *cond_op) else {
                continue;
            };
            x86_64_ops::set_cond(ctx, *cond_op, cond.invert());
            x86_64_ops::set_target(ctx, *cond_op, branch_target);
            if let Some((taken, not_taken)) = weights {
                x86_64_ops::set_branch_weights(ctx, *cond_op, *not_taken, *taken);
            }
            Operation::erase(branch_op, ctx);
        }
    }
}

#[cfg(test)]
mod tests {
    use llvm_compat::ll::LinkageAttr;
    use super::*;
    use crate::{
        common_traits::Named,
        dialects::x86_64::{self, attributes::ConditionCode},
        ir::op::Op,
    };

    fn context() -> Context {
        let mut ctx = Context::new();
        x86_64::register(&mut ctx);
        ctx
    }

    fn new_block(ctx: &mut Context, func: FuncOp, name: &str) -> (Ptr<BasicBlock>, String) {
        let block = BasicBlock::new(ctx, Some(name.try_into().unwrap()), vec![]);
        block.insert_at_back(func.get_region(ctx), ctx);
        let label = block.deref(ctx).unique_name(ctx).to_string();
        (block, label)
    }

    fn layout_labels(ctx: &Context, func: FuncOp) -> Vec<String> {
        func.get_region(ctx)
            .deref(ctx)
            .iter(ctx)
            .map(|block| block.deref(ctx).unique_name(ctx).to_string())
            .collect()
    }

    #[test]
    fn hot_successor_is_laid_out_as_fallthrough() {
        let mut ctx = context();
        let func = FuncOp::new(&mut ctx, "f".try_into().unwrap(), LinkageAttr::External);
        let entry = func.entry_block(&ctx);
        let (cold, cold_label) = new_block(&mut ctx, func, "cold");
        let (hot, hot_label) = new_block(&mut ctx, func, "hot");

        let cond = x86_64_ops::jcc(&mut ctx, ConditionCode::E, cold);
        x86_64_ops::set_branch_weights(&mut ctx, cond, 1, 99);
        cond.insert_at_back(entry, &ctx);
        x86_64_ops::jmp(&mut ctx, hot).insert_at_back(entry, &ctx);
        x86_64_ops::ret(&mut ctx).insert_at_back(cold, &ctx);
        x86_64_ops::ret(&mut ctx).insert_at_back(hot, &ctx);

        place_blocks(&mut ctx, func);

        let entry_label = entry.deref(&ctx).unique_name(&ctx).to_string();
        assert_eq!(
            layout_labels(&ctx, func),
            [entry_label, hot_label, cold_label]
        );
        // The branch to the hot block became a fall-through; the cold branch
        // stayed, untouched.
        let tail = entry.deref(&ctx).get_tail().unwrap();
        assert!(Operation::get_opid(tail, &ctx) == JccOp::get_opid_static());
        assert!(x86_64_ops::target(&ctx, tail) == Some(cold));
        assert_eq!(x86_64_ops::cond(&ctx, tail), Some(ConditionCode::E));
    }

    #[test]
    fn branch_to_hot_fallthrough_is_inverted() {
        let mut ctx = context();
        let func = FuncOp::new(&mut ctx, "f".try_into().unwrap(), LinkageAttr::External);
        let entry = func.entry_block(&ctx);
        let (hot, hot_label) = new_block(&mut ctx, func, "hot");
        let (cold, cold_label) = new_block(&mut ctx, func, "cold");

        let cond = x86_64_ops::jcc(&mut ctx, ConditionCode::E, hot);
        x86_64_ops::set_branch_weights(&mut ctx, cond, 99, 1);
        cond.insert_at_back(entry, &ctx);
        x86_64_ops::jmp(&mut ctx, cold).insert_at_back(entry, &ctx);
        x86_64_ops::ret(&mut ctx).insert_at_back(hot, &ctx);
        x86_64_ops::ret(&mut ctx).insert_at_back(cold, &ctx);

        place_blocks(&mut ctx, func);

        let entry_label = entry.deref(&ctx).unique_name(&ctx).to_string();
        assert_eq!(
            layout_labels(&ctx, func),
            [entry_label, hot_label, cold_label]
        );
        // `jcc e hot; jmp cold` became `jcc ne cold` falling through to
        // the hot block, with the weights swapped to match.
        let tail = entry.deref(&ctx).get_tail().unwrap();
        assert!(Operation::get_opid(tail, &ctx) == JccOp::get_opid_static());
        assert_eq!(x86_64_ops::cond(&ctx, tail), Some(ConditionCode::Ne));
        assert!(x86_64_ops::target(&ctx, tail) == Some(cold));
        assert_eq!(x86_64_ops::branch_weights(&ctx, tail), Some((1, 99)));
    }

    #[test]
    fn unweighted_loop_falls_through_via_the_heuristic() {
        let mut ctx = context();
        let func = FuncOp::new(&mut ctx, "f".try_into().unwrap(), LinkageAttr::External);
        let entry = func.entry_block(&ctx);
        let (header, header_label) = new_block(&mut ctx, func, "header");
        let (body, body_label) = new_block(&mut ctx, func, "body");
        let (exit, exit_label) = new_block(&mut ctx, func, "exit");

        x86_64_ops::jmp(&mut ctx, header).insert_at_back(entry, &ctx);
        // No branch weights anywhere: the loop heuristic must keep the body
        // on the fall-through path and leave only the loop-exit branch.
        x86_64_ops::jcc(&mut ctx, ConditionCode::E, exit).insert_at_back(header, &ctx);
        x86_64_ops::jmp(&mut ctx, body).insert_at_back(header, &ctx);
        x86_64_ops::jmp(&mut ctx, header).insert_at_back(body, &ctx);
        x86_64_ops::ret(&mut ctx).insert_at_back(exit, &ctx);

        place_blocks(&mut ctx, func);

        let entry_label = entry.deref(&ctx).unique_name(&ctx).to_string();
        assert_eq!(
            layout_labels(&ctx, func),
            [entry_label, header_label, body_label, exit_label]
        );
        // entry falls through into the loop header.
        assert!(entry.deref(&ctx).get_tail().is_none());
        // The header keeps only the conditional loop exit and falls through
        // into the body.
        let header_tail = header.deref(&ctx).get_tail().unwrap();
        assert!(Operation::get_opid(header_tail, &ctx) == JccOp::get_opid_static());
        // The back edge cannot fall through and stays an explicit branch.
        let body_tail = body.deref(&ctx).get_tail().unwrap();
        assert!(Operation::get_opid(body_tail, &ctx) == JmpOp::get_opid_static());
    }

    #[test]
    fn placement_is_stable_when_rerun_on_its_own_output() {
        let mut ctx = context();
        let func = FuncOp::new(&mut ctx, "f".try_into().unwrap(), LinkageAttr::External);
        let entry = func.entry_block(&ctx);
        let (cold, _cold_label) = new_block(&mut ctx, func, "cold");
        let (hot, _hot_label) = new_block(&mut ctx, func, "hot");

        let cond = x86_64_ops::jcc(&mut ctx, ConditionCode::E, cold);
        x86_64_ops::set_branch_weights(&mut ctx, cond, 1, 99);
        cond.insert_at_back(entry, &ctx);
        x86_64_ops::jmp(&mut ctx, hot).insert_at_back(entry, &ctx);
        x86_64_ops::ret(&mut ctx).insert_at_back(cold, &ctx);
        x86_64_ops::ret(&mut ctx).insert_at_back(hot, &ctx);

        place_blocks(&mut ctx, func);
        let first = layout_labels(&ctx, func);
        place_blocks(&mut ctx, func);
        assert_eq!(layout_labels(&ctx, func), first);
        let tail = entry.deref(&ctx).get_tail().unwrap();
        assert!(Operation::get_opid(tail, &ctx) == JccOp::get_opid_static());
    }
}
