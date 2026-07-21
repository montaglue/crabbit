//! Dominator-tree and dominance-frontier utilities for one-region operations.

use rustc_hash::FxHashMap;

use crate::{
    context::{Context, Ptr},
    dialects::builtin::op_interfaces::OneRegionInterface,
    ir::{basic_block::BasicBlock, op::Op},
    linked_list::ContainsLinkedList,
};

/// Dominator information for the reachable blocks of a single-region operation.
pub struct DominatorTree {
    entry: Option<Ptr<BasicBlock>>,
    blocks: Vec<Ptr<BasicBlock>>,
    idoms: Vec<Option<usize>>,
    children: Vec<Vec<Ptr<BasicBlock>>>,
    block_indexes: FxHashMap<Ptr<BasicBlock>, usize>,
}

impl DominatorTree {
    pub fn entry(&self) -> Option<Ptr<BasicBlock>> {
        self.entry
    }

    pub fn blocks(&self) -> &[Ptr<BasicBlock>] {
        &self.blocks
    }

    pub fn immediate_dominator(&self, block: Ptr<BasicBlock>) -> Option<Ptr<BasicBlock>> {
        let index = *self.block_indexes.get(&block)?;
        let idom = self.idoms[index]?;
        (idom != index).then_some(self.blocks[idom])
    }

    pub fn children(&self, block: Ptr<BasicBlock>) -> &[Ptr<BasicBlock>] {
        let Some(index) = self.block_indexes.get(&block).copied() else {
            return &[];
        };
        self.children[index].as_slice()
    }
}

/// Dominance-frontier information for the reachable blocks of a one-region op.
pub struct DominanceFrontiers {
    frontiers: Vec<Vec<Ptr<BasicBlock>>>,
    block_indexes: FxHashMap<Ptr<BasicBlock>, usize>,
}

impl DominanceFrontiers {
    pub fn frontier(&self, block: Ptr<BasicBlock>) -> &[Ptr<BasicBlock>] {
        let Some(index) = self.block_indexes.get(&block).copied() else {
            return &[];
        };
        self.frontiers[index].as_slice()
    }
}

/// Compute the dominator tree and dominance frontiers for an operation's region.
pub fn compute_dominance_frontiers_for_op<T: Op + OneRegionInterface>(
    op: &T,
    ctx: &Context,
) -> (DominatorTree, DominanceFrontiers) {
    let blocks = reachable_blocks(op, ctx);
    if blocks.is_empty() {
        let block_indexes = FxHashMap::default();
        return (
            DominatorTree {
                entry: None,
                blocks,
                idoms: vec![],
                children: vec![],
                block_indexes: block_indexes.clone(),
            },
            DominanceFrontiers {
                frontiers: vec![],
                block_indexes,
            },
        );
    }

    let predecessors = predecessor_indexes(ctx, &blocks);
    let idoms = immediate_dominators(&predecessors);
    let children = dominator_children(&blocks, &idoms);
    let frontiers = dominance_frontiers(&blocks, &predecessors, &idoms);
    let block_indexes = block_indexes(&blocks);

    (
        DominatorTree {
            entry: Some(blocks[0]),
            blocks,
            idoms,
            children,
            block_indexes: block_indexes.clone(),
        },
        DominanceFrontiers {
            frontiers,
            block_indexes,
        },
    )
}

fn reachable_blocks<T: Op + OneRegionInterface>(op: &T, ctx: &Context) -> Vec<Ptr<BasicBlock>> {
    let region = op.get_region(ctx);
    let Some(entry) = region.deref(ctx).get_head() else {
        return vec![];
    };

    let region_blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
    let region_indexes = block_indexes(&region_blocks);
    let mut seen = FxHashMap::<Ptr<BasicBlock>, bool>::default();
    let mut postorder = Vec::new();

    visit_reachable_block(ctx, entry, &region_indexes, &mut seen, &mut postorder);
    postorder.reverse();
    postorder
}

fn visit_reachable_block(
    ctx: &Context,
    block: Ptr<BasicBlock>,
    region_indexes: &FxHashMap<Ptr<BasicBlock>, usize>,
    seen: &mut FxHashMap<Ptr<BasicBlock>, bool>,
    postorder: &mut Vec<Ptr<BasicBlock>>,
) {
    if !region_indexes.contains_key(&block) || seen.contains_key(&block) {
        return;
    }

    seen.insert(block, true);
    for succ in block.deref(ctx).succs(ctx) {
        visit_reachable_block(ctx, succ, region_indexes, seen, postorder);
    }
    postorder.push(block);
}

fn predecessor_indexes(ctx: &Context, blocks: &[Ptr<BasicBlock>]) -> Vec<Vec<usize>> {
    let indexes = block_indexes(blocks);
    let mut predecessors = vec![vec![]; blocks.len()];

    for (block_index, block) in blocks.iter().copied().enumerate() {
        for succ in block.deref(ctx).succs(ctx) {
            if let Some(succ_index) = indexes.get(&succ).copied() {
                predecessors[succ_index].push(block_index);
            }
        }
    }

    predecessors
}

/// Iterative immediate-dominator computation (Cooper–Harvey–Kennedy) over
/// predecessor lists. Blocks must be indexed in reverse post order with the
/// entry at index 0.
pub(crate) fn immediate_dominators(predecessors: &[Vec<usize>]) -> Vec<Option<usize>> {
    if predecessors.is_empty() {
        return vec![];
    }

    let mut idoms = vec![None; predecessors.len()];
    idoms[0] = Some(0);

    loop {
        let mut changed = false;
        for block in 1..predecessors.len() {
            let mut processed_preds = predecessors[block]
                .iter()
                .copied()
                .filter(|pred| idoms[*pred].is_some());
            let Some(mut new_idom) = processed_preds.next() else {
                continue;
            };

            for pred in processed_preds {
                new_idom = intersect_idoms(&idoms, pred, new_idom);
            }

            if idoms[block] != Some(new_idom) {
                idoms[block] = Some(new_idom);
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    idoms
}

fn intersect_idoms(idoms: &[Option<usize>], mut lhs: usize, mut rhs: usize) -> usize {
    while lhs != rhs {
        while lhs > rhs {
            lhs = idoms[lhs].expect("dominator chain must be initialized");
        }
        while rhs > lhs {
            rhs = idoms[rhs].expect("dominator chain must be initialized");
        }
    }
    lhs
}

fn dominator_children(
    blocks: &[Ptr<BasicBlock>],
    idoms: &[Option<usize>],
) -> Vec<Vec<Ptr<BasicBlock>>> {
    let mut children = vec![vec![]; blocks.len()];
    for (block, idom) in idoms.iter().copied().enumerate().skip(1) {
        if let Some(idom) = idom {
            children[idom].push(blocks[block]);
        }
    }
    children
}

fn dominance_frontiers(
    blocks: &[Ptr<BasicBlock>],
    predecessors: &[Vec<usize>],
    idoms: &[Option<usize>],
) -> Vec<Vec<Ptr<BasicBlock>>> {
    let mut frontiers = vec![vec![]; blocks.len()];

    for (block, preds) in predecessors.iter().enumerate() {
        if preds.len() < 2 {
            continue;
        }
        let Some(stop) = idoms[block] else {
            continue;
        };

        for &pred in preds {
            let mut runner = pred;
            while runner != stop {
                push_unique(&mut frontiers[runner], blocks[block]);
                runner = idoms[runner].expect("reachable predecessor must have an idom");
            }
        }
    }

    frontiers
}

fn push_unique<T: Eq + Copy>(values: &mut Vec<T>, value: T) {
    if !values.contains(&value) {
        values.push(value);
    }
}

fn block_indexes(blocks: &[Ptr<BasicBlock>]) -> FxHashMap<Ptr<BasicBlock>, usize> {
    blocks
        .iter()
        .copied()
        .enumerate()
        .map(|(index, block)| (block, index))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dialects::{
            builtin::{self, types::Signedness},
            llvm::{
                self,
                attributes::LinkageAttr,
                ops::{BrOp, CondBrOp, FuncOp, ReturnOp},
                types::FuncType,
            },
        },
        ir::r#type::{TypeHandle, TypedHandle},
    };

    fn test_context() -> Context {
        let mut ctx = Context::new();
        llvm::register(&mut ctx);
        ctx
    }

    #[test]
    fn computes_frontier_for_diamond_cfg() {
        let mut ctx = test_context();
        let i1_ty: TypeHandle =
            builtin::types::IntegerType::get(&mut ctx, 1, Signedness::Signless).into();
        let fn_ty: TypedHandle<FuncType> = FuncType::get(&mut ctx, i1_ty, vec![i1_ty], false);
        let func = FuncOp::new(
            &mut ctx,
            "diamond".try_into().unwrap(),
            fn_ty,
            LinkageAttr::External,
        );

        let entry = func.get_entry_block(&ctx);
        let cond = entry.deref(&ctx).get_argument(0);
        let then_block = BasicBlock::new(&mut ctx, Some("then".try_into().unwrap()), vec![]);
        let else_block = BasicBlock::new(&mut ctx, Some("else".try_into().unwrap()), vec![]);
        let join_block = BasicBlock::new(&mut ctx, Some("join".try_into().unwrap()), vec![]);
        then_block.insert_at_back(func.get_region(&ctx), &ctx);
        else_block.insert_at_back(func.get_region(&ctx), &ctx);
        join_block.insert_at_back(func.get_region(&ctx), &ctx);

        CondBrOp::new(&mut ctx, cond, then_block, vec![], else_block, vec![])
            .get_operation()
            .insert_at_back(entry, &ctx);
        BrOp::new(&mut ctx, join_block, vec![])
            .get_operation()
            .insert_at_back(then_block, &ctx);
        BrOp::new(&mut ctx, join_block, vec![])
            .get_operation()
            .insert_at_back(else_block, &ctx);
        ReturnOp::new(&mut ctx, Some(cond))
            .get_operation()
            .insert_at_back(join_block, &ctx);

        let (dom_tree, frontiers) = compute_dominance_frontiers_for_op(&func, &ctx);

        assert!(dom_tree.entry() == Some(entry));
        assert!(dom_tree.immediate_dominator(then_block) == Some(entry));
        assert!(dom_tree.immediate_dominator(else_block) == Some(entry));
        assert!(dom_tree.immediate_dominator(join_block) == Some(entry));
        assert!(frontiers.frontier(entry).is_empty());
        assert!(frontiers.frontier(then_block) == [join_block]);
        assert!(frontiers.frontier(else_block) == [join_block]);
    }

    #[test]
    fn handles_empty_region() {
        let mut ctx = test_context();
        let i1_ty: TypeHandle =
            builtin::types::IntegerType::get(&mut ctx, 1, Signedness::Signless).into();
        let fn_ty: TypedHandle<FuncType> = FuncType::get(&mut ctx, i1_ty, vec![], false);
        let func = FuncOp::new_declaration(
            &mut ctx,
            "decl".try_into().unwrap(),
            fn_ty,
            LinkageAttr::External,
        );

        let (dom_tree, _) = compute_dominance_frontiers_for_op(&func, &ctx);

        assert!(dom_tree.entry().is_none());
        assert!(dom_tree.blocks().is_empty());
    }
}
