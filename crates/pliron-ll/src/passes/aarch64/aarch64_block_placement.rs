//! Hot-path-driven basic block placement for AArch64 machine functions.
//!
//! This is the example consumer of the generic [hot path
//! analysis](crate::passes::hot_path), modeled on LLVM's
//! `MachineBlockPlacement`, which runs after register allocation and before
//! branch relaxation. It lays the hottest control-flow chains out
//! contiguously so that likely edges become fall-throughs. It only orders
//! blocks: every block ends in a terminator naming all its successors, and
//! the encoder turns the edge to the next block into a fall-through (a `b`
//! to it is not emitted, a `b_cond` whose taken block is next is inverted).
//!
//! Edge probabilities come from the `branch_weights` transferred onto
//! conditional branches by instruction selection (the machine-level analogue
//! of `MachineBasicBlock` successor probabilities); branches without weights
//! fall back to the analysis' static loop heuristics.

use std::collections::HashMap;

use crate::{
    context::{Context, Ptr},
    conversion::pass::{AnalysisManager, Pass, PassResult, changed},
    dialects::aarch64::ops::{self as aarch64_ops, FuncOp},
    dialects::builtin::op_interfaces::OneRegionInterface,
    ir::basic_block::BasicBlock,
    linked_list::ContainsLinkedList,
    passes::hot_path::{CfgEdge, HotPathInfo},
};

use super::{frontend::module_op, util::cast_operation};

pub struct Aarch64BlockPlacementPass;

impl Pass for Aarch64BlockPlacementPass {
    fn name(&self) -> &str {
        "aarch64-block-placement"
    }

    fn run(
        &mut self,
        root: Ptr<crate::ir::operation::Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> pliron::result::Result<PassResult> {
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

    // Phase 1: the CFG, from each block's terminator, with the branch
    // weights left by instruction selection.
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
}

/// Outgoing edges of each block, from its terminator: a conditional
/// branch's taken and not-taken edges carry its `[taken, not-taken]`
/// weights; everything else is left to the analysis' heuristics.
fn machine_cfg_edges(
    ctx: &Context,
    blocks: &[Ptr<BasicBlock>],
    block_index: &HashMap<Ptr<BasicBlock>, usize>,
) -> Vec<Vec<CfgEdge>> {
    blocks
        .iter()
        .map(|&block| {
            let Some(terminator) = block.deref(ctx).get_tail() else {
                return vec![];
            };
            let weights = aarch64_ops::branch_weights(ctx, terminator);
            let edge_weights = [
                weights.map(|(taken, _)| taken),
                weights.map(|(_, not_taken)| not_taken),
            ];
            let successors: Vec<_> = terminator.deref(ctx).successors().collect();
            successors
                .into_iter()
                .enumerate()
                .filter_map(|(index, succ)| {
                    Some(CfgEdge {
                        to: *block_index.get(&succ)?,
                        weight: edge_weights.get(index).copied().flatten(),
                    })
                })
                .collect()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ll::LinkageAttr;
    use crate::{
        common_traits::Named,
        dialects::aarch64::{
            self, attributes::ConditionCode, op_interfaces::BinarySerializableOpInterface,
        },
        ir::{op::op_cast, operation::Operation},
    };

    fn context() -> Context {
        let mut ctx = Context::new();
        aarch64::register(&mut ctx);
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

    /// The bytes `block`'s terminator encodes to in the current layout.
    fn terminator_bytes(ctx: &Context, block: Ptr<BasicBlock>) -> u64 {
        let tail = block.deref(ctx).get_tail().unwrap();
        let op = Operation::get_op_dyn(tail, ctx);
        op_cast::<dyn BinarySerializableOpInterface>(&*op)
            .unwrap()
            .byte_len(ctx)
    }

    #[test]
    fn hot_successor_is_laid_out_as_fallthrough() {
        let mut ctx = context();
        let func = FuncOp::new(&mut ctx, "f".try_into().unwrap(), LinkageAttr::External);
        let entry = func.entry_block(&ctx);
        let (cold, cold_label) = new_block(&mut ctx, func, "cold");
        let (hot, hot_label) = new_block(&mut ctx, func, "hot");

        let cond = aarch64_ops::b_cond(&mut ctx, ConditionCode::Eq, cold, hot);
        aarch64_ops::set_branch_weights(&mut ctx, cond, 1, 99);
        cond.insert_at_back(entry, &ctx);
        aarch64_ops::ret(&mut ctx).insert_at_back(cold, &ctx);
        aarch64_ops::ret(&mut ctx).insert_at_back(hot, &ctx);

        place_blocks(&mut ctx, func);

        let entry_label = entry.deref(&ctx).unique_name(&ctx).to_string();
        assert_eq!(
            layout_labels(&ctx, func),
            [entry_label, hot_label, cold_label]
        );
        // The terminator is untouched; with the hot block next, it encodes
        // as the one word `b.eq cold`.
        let tail = entry.deref(&ctx).get_tail().unwrap();
        assert!(aarch64_ops::target(&ctx, tail) == Some(cold));
        assert!(aarch64_ops::not_taken(&ctx, tail) == Some(hot));
        assert_eq!(terminator_bytes(&ctx, entry), 4);
    }

    #[test]
    fn branch_to_hot_fallthrough_encodes_inverted() {
        let mut ctx = context();
        let func = FuncOp::new(&mut ctx, "f".try_into().unwrap(), LinkageAttr::External);
        let entry = func.entry_block(&ctx);
        let (hot, hot_label) = new_block(&mut ctx, func, "hot");
        let (cold, cold_label) = new_block(&mut ctx, func, "cold");

        let cond = aarch64_ops::b_cond(&mut ctx, ConditionCode::Eq, hot, cold);
        aarch64_ops::set_branch_weights(&mut ctx, cond, 99, 1);
        cond.insert_at_back(entry, &ctx);
        aarch64_ops::ret(&mut ctx).insert_at_back(hot, &ctx);
        aarch64_ops::ret(&mut ctx).insert_at_back(cold, &ctx);

        place_blocks(&mut ctx, func);

        let entry_label = entry.deref(&ctx).unique_name(&ctx).to_string();
        assert_eq!(
            layout_labels(&ctx, func),
            [entry_label, hot_label, cold_label]
        );
        // The taken block is next: one word, `b.ne cold`.
        assert_eq!(terminator_bytes(&ctx, entry), 4);
    }

    #[test]
    fn unweighted_loop_falls_through_via_the_heuristic() {
        let mut ctx = context();
        let func = FuncOp::new(&mut ctx, "f".try_into().unwrap(), LinkageAttr::External);
        let entry = func.entry_block(&ctx);
        let (header, header_label) = new_block(&mut ctx, func, "header");
        let (body, body_label) = new_block(&mut ctx, func, "body");
        let (exit, exit_label) = new_block(&mut ctx, func, "exit");

        aarch64_ops::b(&mut ctx, header).insert_at_back(entry, &ctx);
        // No branch weights anywhere: the loop heuristic must keep the body
        // on the fall-through path and leave only the loop-exit branch.
        aarch64_ops::b_cond(&mut ctx, ConditionCode::Eq, exit, body).insert_at_back(header, &ctx);
        aarch64_ops::b(&mut ctx, header).insert_at_back(body, &ctx);
        aarch64_ops::ret(&mut ctx).insert_at_back(exit, &ctx);

        place_blocks(&mut ctx, func);

        let entry_label = entry.deref(&ctx).unique_name(&ctx).to_string();
        assert_eq!(
            layout_labels(&ctx, func),
            [entry_label, header_label, body_label, exit_label]
        );
        // entry falls through into the loop header; the header encodes only
        // the conditional loop exit; the back edge stays an explicit branch.
        assert_eq!(terminator_bytes(&ctx, entry), 0);
        assert_eq!(terminator_bytes(&ctx, header), 4);
        assert_eq!(terminator_bytes(&ctx, body), 4);
    }

    #[test]
    fn placement_is_stable_when_rerun_on_its_own_output() {
        let mut ctx = context();
        let func = FuncOp::new(&mut ctx, "f".try_into().unwrap(), LinkageAttr::External);
        let entry = func.entry_block(&ctx);
        let (cold, _cold_label) = new_block(&mut ctx, func, "cold");
        let (hot, _hot_label) = new_block(&mut ctx, func, "hot");

        let cond = aarch64_ops::b_cond(&mut ctx, ConditionCode::Eq, cold, hot);
        aarch64_ops::set_branch_weights(&mut ctx, cond, 1, 99);
        cond.insert_at_back(entry, &ctx);
        aarch64_ops::ret(&mut ctx).insert_at_back(cold, &ctx);
        aarch64_ops::ret(&mut ctx).insert_at_back(hot, &ctx);

        place_blocks(&mut ctx, func);
        let first = layout_labels(&ctx, func);
        place_blocks(&mut ctx, func);
        assert_eq!(layout_labels(&ctx, func), first);
    }
}
