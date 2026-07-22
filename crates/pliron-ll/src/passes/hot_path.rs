//! Static hot-path analysis: branch probabilities, block frequencies, and the
//! hottest control-flow path through a CFG.
//!
//! The model follows LLVM's profile-guided code layout stack:
//!
//! - **Branch probabilities** (LLVM `BranchProbabilityInfo`): the likelihood
//!   of each CFG edge, taken from explicit branch weights when a terminator
//!   carries them ([`WeightedBranchOpInterface`], the analogue of MLIR's
//!   `WeightedBranchOpInterface` / LLVM's `!prof branch_weights`), otherwise
//!   estimated with Ball–Larus/Wu–Larus style loop heuristics: edges that stay
//!   inside a natural loop are hot, edges that leave it are cold.
//! - **Block frequencies** (LLVM `BlockFrequencyInfo`): relative execution
//!   counts propagated from probabilities in reverse post order, with loop
//!   bodies scaled by the loop's expected trip count (capped, as in LLVM).
//! - **Hot path**: the chain of blocks reached from the entry by always
//!   following the most probable successor edge.
//!
//! LLVM computes probabilities on IR, transfers them onto machine basic
//! blocks during instruction selection, and consumes the machine-level
//! frequencies after register allocation (`MachineBlockPlacement`). STAIR
//! mirrors that flow: [`HotPathInfo::for_op`] serves successor-based dialects
//! (`llvm`, `cf`), and [`HotPathInfo::from_edges`] serves label-based machine
//! CFGs (e.g. the `aarch64` dialect, see the `aarch64-block-placement` pass).

use pliron::region::Region;
use rustc_hash::FxHashMap;

use crate::{
    context::{Context, Ptr},
    ll::op_interfaces::WeightedBranchOpInterface,
    ir::{basic_block::BasicBlock, op::op_cast, operation::Operation},
    linked_list::ContainsLinkedList,
    passes::dominance_frontier::immediate_dominators,
};

/// Weight LLVM's loop-branch heuristic assigns to an edge that stays in the
/// loop (`LBH_TAKEN_WEIGHT`), against [`LOOP_EXIT_WEIGHT`] for leaving it.
const LOOP_STAY_WEIGHT: u32 = 124;
const LOOP_EXIT_WEIGHT: u32 = 4;

/// Frequency assigned to the function entry block; all block frequencies are
/// relative to it.
pub const ENTRY_FREQUENCY: u64 = 1 << 14;

/// Cap on the frequency amplification a single loop can contribute, matching
/// LLVM `BlockFrequencyInfo`'s bound on loop scales.
const MAX_LOOP_SCALE: f64 = 4096.0;

/// The probability of one CFG edge, in fixed point.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct BranchProbability {
    numerator: u32,
}

impl BranchProbability {
    pub const DENOMINATOR: u32 = 1 << 20;

    pub fn from_ratio(numerator: u64, denominator: u64) -> Self {
        debug_assert!(denominator > 0 && numerator <= denominator);
        BranchProbability {
            numerator: ((numerator * Self::DENOMINATOR as u64) / denominator) as u32,
        }
    }

    pub fn zero() -> Self {
        BranchProbability { numerator: 0 }
    }

    pub fn numerator(self) -> u32 {
        self.numerator
    }

    pub fn complement(self) -> Self {
        BranchProbability {
            numerator: Self::DENOMINATOR - self.numerator,
        }
    }

    pub fn as_f64(self) -> f64 {
        self.numerator as f64 / Self::DENOMINATOR as f64
    }
}

/// One outgoing CFG edge: target block index plus the branch weight carried by
/// the terminator, if any. Weight semantics match
/// [`WeightedBranchOpInterface`]: the edge probability is `weight` over the
/// sum of the block's edge weights. Edges without weights fall back to the
/// static loop heuristics.
#[derive(Clone, Copy, Debug)]
pub struct CfgEdge {
    pub to: usize,
    pub weight: Option<u32>,
}

/// Branch probabilities, block frequencies, and the hot path of one CFG.
pub struct HotPathInfo {
    blocks: Vec<Ptr<BasicBlock>>,
    block_indexes: FxHashMap<Ptr<BasicBlock>, usize>,
    /// Per block: outgoing `(target index, probability)` pairs. Unreachable
    /// blocks have no entries.
    probabilities: Vec<Vec<(usize, BranchProbability)>>,
    /// Per block: estimated execution frequency relative to
    /// [`ENTRY_FREQUENCY`]. Unreachable blocks have frequency 0.
    frequencies: Vec<u64>,
    hot_path: Vec<usize>,
}

impl HotPathInfo {
    /// Analyze a region whose terminators reference successors directly
    /// (`llvm`, `cf`, ...). Explicit weights are read through
    /// [`WeightedBranchOpInterface`].
    pub fn for_region(region: Ptr<Region>, ctx: &Context) -> Self {
        let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
        let block_indexes = block_indexes(&blocks);

        let successors = blocks
            .iter()
            .map(|block| {
                let weights = block
                    .deref(ctx)
                    .get_terminator(ctx)
                    .and_then(|term| terminator_weights(ctx, term));
                block
                    .deref(ctx)
                    .succs(ctx)
                    .into_iter()
                    .enumerate()
                    .filter_map(|(succ_idx, succ)| {
                        let to = *block_indexes.get(&succ)?;
                        let weight = weights
                            .as_ref()
                            .and_then(|weights| weights.get(succ_idx).copied());
                        Some(CfgEdge { to, weight })
                    })
                    .collect()
            })
            .collect();

        Self::from_edges(blocks, successors)
    }

    /// Analyze an explicitly described CFG. `blocks[0]` is the entry;
    /// `successors[i]` lists the outgoing edges of `blocks[i]`. Used for
    /// machine CFGs whose branches reference targets by label rather than by
    /// successor lists.
    pub fn from_edges(blocks: Vec<Ptr<BasicBlock>>, successors: Vec<Vec<CfgEdge>>) -> Self {
        assert_eq!(blocks.len(), successors.len());
        let block_indexes = block_indexes(&blocks);
        if blocks.is_empty() {
            return HotPathInfo {
                blocks,
                block_indexes,
                probabilities: vec![],
                frequencies: vec![],
                hot_path: vec![],
            };
        }

        // Phase 1: probabilities, from explicit weights or loop heuristics.
        let rpo = reverse_post_order(&successors);
        let loop_headers = natural_loop_membership(&successors, &rpo);
        let probabilities = edge_probabilities(&successors, &loop_headers);

        // Phase 2: frequencies, propagated in reverse post order with loop
        // scaling.
        let frequencies = block_frequencies(&probabilities, &rpo);

        // Phase 3: the hot path itself.
        let hot_path = extract_hot_path(&probabilities);

        HotPathInfo {
            blocks,
            block_indexes,
            probabilities,
            frequencies,
            hot_path,
        }
    }

    /// The analyzed blocks; index positions match the other accessors.
    pub fn blocks(&self) -> &[Ptr<BasicBlock>] {
        &self.blocks
    }

    /// Estimated execution frequency of `block` relative to
    /// [`ENTRY_FREQUENCY`]. Unknown and unreachable blocks report 0.
    pub fn block_frequency(&self, block: Ptr<BasicBlock>) -> u64 {
        self.block_indexes
            .get(&block)
            .map(|index| self.frequencies[*index])
            .unwrap_or(0)
    }

    /// Probability of the edge `from -> to`; zero if there is no such edge.
    /// Parallel edges (e.g. a conditional branch with one target on both
    /// sides) are summed.
    pub fn edge_probability(
        &self,
        from: Ptr<BasicBlock>,
        to: Ptr<BasicBlock>,
    ) -> BranchProbability {
        let (Some(&from), Some(&to)) = (self.block_indexes.get(&from), self.block_indexes.get(&to))
        else {
            return BranchProbability::zero();
        };
        let numerator = self.probabilities[from]
            .iter()
            .filter(|(succ, _)| *succ == to)
            .map(|(_, prob)| prob.numerator)
            .sum();
        BranchProbability { numerator }
    }

    /// Estimated frequency of the edge `from -> to`:
    /// `frequency(from) * probability(from -> to)`.
    pub fn edge_frequency(&self, from: Ptr<BasicBlock>, to: Ptr<BasicBlock>) -> u64 {
        let freq = self.block_frequency(from) as f64 * self.edge_probability(from, to).as_f64();
        freq.round() as u64
    }

    /// Probabilities of `block`'s outgoing edges, in terminator successor
    /// order. Empty for blocks without successors or outside the analysis.
    pub fn successor_probabilities(&self, block: Ptr<BasicBlock>) -> Vec<BranchProbability> {
        self.block_indexes
            .get(&block)
            .map(|index| {
                self.probabilities[*index]
                    .iter()
                    .map(|(_, prob)| *prob)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The chain of blocks reached from the entry by always following the
    /// most probable successor edge, ending before the first repeated block.
    pub fn hot_path(&self) -> Vec<Ptr<BasicBlock>> {
        self.hot_path
            .iter()
            .map(|index| self.blocks[*index])
            .collect()
    }

    /// Whether `block` is estimated to run at least as often as the entry.
    pub fn is_hot(&self, block: Ptr<BasicBlock>) -> bool {
        self.block_frequency(block) >= ENTRY_FREQUENCY
    }
}

/// Read a terminator's successor weights if it implements
/// [`WeightedBranchOpInterface`] and carries them.
fn terminator_weights(ctx: &Context, terminator: Ptr<Operation>) -> Option<Vec<u32>> {
    let op = Operation::get_op_dyn(terminator, ctx);
    op_cast::<dyn WeightedBranchOpInterface>(&*op)
        .and_then(|weighted| weighted.successor_weights(ctx))
}

fn block_indexes(blocks: &[Ptr<BasicBlock>]) -> FxHashMap<Ptr<BasicBlock>, usize> {
    blocks
        .iter()
        .copied()
        .enumerate()
        .map(|(index, block)| (block, index))
        .collect()
}

/// Reverse post order of the blocks reachable from block 0.
fn reverse_post_order(successors: &[Vec<CfgEdge>]) -> Vec<usize> {
    let mut seen = vec![false; successors.len()];
    let mut postorder = Vec::new();
    visit_post_order(successors, 0, &mut seen, &mut postorder);
    postorder.reverse();
    postorder
}

fn visit_post_order(
    successors: &[Vec<CfgEdge>],
    block: usize,
    seen: &mut [bool],
    postorder: &mut Vec<usize>,
) {
    if seen[block] {
        return;
    }
    seen[block] = true;
    for edge in &successors[block] {
        visit_post_order(successors, edge.to, seen, postorder);
    }
    postorder.push(block);
}

/// For every block, the set of loop headers whose natural loop contains it.
/// A back edge is an edge whose target dominates its source; its natural loop
/// is the target plus all blocks that reach the source without passing the
/// target.
fn natural_loop_membership(successors: &[Vec<CfgEdge>], rpo: &[usize]) -> Vec<Vec<usize>> {
    let rpo_index: FxHashMap<usize, usize> = rpo
        .iter()
        .copied()
        .enumerate()
        .map(|(rpo_idx, block)| (block, rpo_idx))
        .collect();

    // Predecessor lists in RPO indexing, as required by immediate_dominators.
    let mut predecessors = vec![vec![]; rpo.len()];
    for (block, edges) in successors.iter().enumerate() {
        let Some(&from) = rpo_index.get(&block) else {
            continue;
        };
        for edge in edges {
            if let Some(&to) = rpo_index.get(&edge.to) {
                predecessors[to].push(from);
            }
        }
    }
    let idoms = immediate_dominators(&predecessors);
    let dominates = |dom: usize, mut block: usize| -> bool {
        loop {
            if block == dom {
                return true;
            }
            match idoms[block] {
                Some(idom) if idom != block => block = idom,
                _ => return false,
            }
        }
    };

    let mut headers_of = vec![vec![]; successors.len()];
    for (block, edges) in successors.iter().enumerate() {
        let Some(&source) = rpo_index.get(&block) else {
            continue;
        };
        for edge in edges {
            let Some(&target) = rpo_index.get(&edge.to) else {
                continue;
            };
            if !dominates(target, source) {
                continue;
            }
            // Back edge `block -> edge.to`: flood the natural loop backwards
            // from the source, stopping at the header.
            let header = edge.to;
            let mut in_loop = vec![false; rpo.len()];
            in_loop[target] = true;
            let mut worklist = vec![source];
            while let Some(node) = worklist.pop() {
                if in_loop[node] {
                    continue;
                }
                in_loop[node] = true;
                worklist.extend(predecessors[node].iter().copied());
            }
            for (rpo_idx, in_loop) in in_loop.into_iter().enumerate() {
                let member = rpo[rpo_idx];
                if in_loop && !headers_of[member].contains(&header) {
                    headers_of[member].push(header);
                }
            }
        }
    }
    headers_of
}

/// Per-block successor probabilities. Explicit weights win; otherwise edges
/// that stay in one of the block's loops get [`LOOP_STAY_WEIGHT`] against
/// [`LOOP_EXIT_WEIGHT`] for edges that leave, and blocks outside any loop
/// split uniformly.
fn edge_probabilities(
    successors: &[Vec<CfgEdge>],
    loop_headers: &[Vec<usize>],
) -> Vec<Vec<(usize, BranchProbability)>> {
    successors
        .iter()
        .enumerate()
        .map(|(block, edges)| {
            if edges.is_empty() {
                return vec![];
            }
            let explicit: Option<Vec<u64>> = edges
                .iter()
                .map(|edge| edge.weight.map(u64::from))
                .collect();
            let weights = match explicit {
                Some(weights) if weights.iter().sum::<u64>() > 0 => weights,
                _ => heuristic_weights(block, edges, loop_headers),
            };
            let total: u64 = weights.iter().sum();
            edges
                .iter()
                .zip(weights)
                .map(|(edge, weight)| (edge.to, BranchProbability::from_ratio(weight, total)))
                .collect()
        })
        .collect()
}

fn heuristic_weights(block: usize, edges: &[CfgEdge], loop_headers: &[Vec<usize>]) -> Vec<u64> {
    let stays_in_loop = |target: usize| {
        loop_headers[block]
            .iter()
            .any(|header| loop_headers[target].contains(header))
    };
    if loop_headers[block].is_empty() || edges.iter().all(|edge| stays_in_loop(edge.to)) {
        return vec![1; edges.len()];
    }
    edges
        .iter()
        .map(|edge| {
            if stays_in_loop(edge.to) {
                LOOP_STAY_WEIGHT as u64
            } else {
                LOOP_EXIT_WEIGHT as u64
            }
        })
        .collect()
}

/// Two-pass Wu–Larus style frequency propagation. The first reverse-post-order
/// pass ignores back edges and yields per-iteration mass; from it every loop
/// header's cyclic probability and scale (expected trip count) is derived; the
/// second pass re-propagates with headers amplified by their scale.
fn block_frequencies(probabilities: &[Vec<(usize, BranchProbability)>], rpo: &[usize]) -> Vec<u64> {
    let rpo_position: FxHashMap<usize, usize> = rpo
        .iter()
        .copied()
        .enumerate()
        .map(|(position, block)| (block, position))
        .collect();

    // An edge is a back edge iff it does not advance in RPO; for reducible
    // CFGs this coincides with dominance-based back edges. Successors of
    // reachable blocks are always reachable, so the position lookups succeed.
    let mut forward_preds = vec![vec![]; probabilities.len()];
    let mut back_preds = vec![vec![]; probabilities.len()];
    for &block in rpo {
        for (succ, prob) in &probabilities[block] {
            if rpo_position[&block] < rpo_position[succ] {
                forward_preds[*succ].push((block, *prob));
            } else {
                back_preds[*succ].push((block, *prob));
            }
        }
    }

    let propagate = |scales: &[f64]| -> Vec<f64> {
        let mut freq = vec![0.0; probabilities.len()];
        for (position, &block) in rpo.iter().enumerate() {
            let mut inflow = if position == 0 { 1.0 } else { 0.0 };
            for &(pred, prob) in &forward_preds[block] {
                inflow += freq[pred] * prob.as_f64();
            }
            freq[block] = inflow * scales[block];
        }
        freq
    };

    let base = propagate(&vec![1.0; probabilities.len()]);

    let mut scales = vec![1.0; probabilities.len()];
    for &block in rpo {
        // Cyclic probability of a header: the share of one iteration's mass
        // that its back edges return to it.
        let cyclic: f64 = back_preds[block]
            .iter()
            .map(|&(pred, prob)| base[pred] * prob.as_f64())
            .sum();
        if cyclic > 0.0 {
            let header_mass = base[block].max(f64::MIN_POSITIVE);
            let cyclic_probability = (cyclic / header_mass).min(1.0 - 1.0 / MAX_LOOP_SCALE);
            scales[block] = 1.0 / (1.0 - cyclic_probability);
        }
    }

    propagate(&scales)
        .into_iter()
        .map(|freq| (freq * ENTRY_FREQUENCY as f64).round() as u64)
        .collect()
}

/// Follow the most probable successor edge from the entry until a block
/// repeats or execution leaves the function. Ties go to the earlier successor.
fn extract_hot_path(probabilities: &[Vec<(usize, BranchProbability)>]) -> Vec<usize> {
    let mut path = vec![];
    let mut on_path = vec![false; probabilities.len()];
    let mut block = 0;
    loop {
        path.push(block);
        on_path[block] = true;
        let mut next = None;
        for &(succ, prob) in &probabilities[block] {
            if next.is_none_or(|(_, best)| prob > best) {
                next = Some((succ, prob));
            }
        }
        match next {
            Some((succ, _)) if !on_path[succ] => block = succ,
            _ => break,
        }
    }
    path
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use pliron::builtin::op_interfaces::{
        BranchOpInterface as _, CallOpInterface as _,
    };
    #[allow(unused_imports)]
    use pliron_llvm::op_interfaces::{BinArithOp as _};
    use super::*;
    #[allow(unused_imports)]
    use pliron::op::Op as _;
    #[allow(unused_imports)]
    use pliron::builtin::op_interfaces::AtMostOneRegionInterface as _;
    use crate::{
        dialects::{
            builtin::{self, types::Signedness},
            llvm::{
                attributes::LinkageAttr,
                ops::{BrOp, CondBrOp, FuncOp, ReturnOp},
                types::FuncType,
            },
        },
        ir::r#type::{TypeHandle, TypedHandle},
    };

    fn test_context() -> Context {
        let ctx = Context::new();
        ctx
    }

    fn dummy_blocks(ctx: &mut Context, count: usize) -> Vec<Ptr<BasicBlock>> {
        (0..count)
            .map(|_| BasicBlock::new(ctx, None, vec![]))
            .collect()
    }

    #[test]
    fn explicit_weights_pick_the_heavy_diamond_arm() {
        let mut ctx = test_context();
        let blocks = dummy_blocks(&mut ctx, 4);
        let edge = |to, weight| CfgEdge { to, weight };
        let successors = vec![
            vec![edge(1, Some(1)), edge(2, Some(3))],
            vec![edge(3, None)],
            vec![edge(3, None)],
            vec![],
        ];

        let info = HotPathInfo::from_edges(blocks.clone(), successors);

        assert!(info.edge_probability(blocks[0], blocks[2]) == BranchProbability::from_ratio(3, 4));
        assert_eq!(info.block_frequency(blocks[0]), ENTRY_FREQUENCY);
        assert_eq!(info.block_frequency(blocks[1]), ENTRY_FREQUENCY / 4);
        assert_eq!(info.block_frequency(blocks[2]), 3 * ENTRY_FREQUENCY / 4);
        assert_eq!(info.block_frequency(blocks[3]), ENTRY_FREQUENCY);
        assert!(info.hot_path() == [blocks[0], blocks[2], blocks[3]]);
        assert!(info.is_hot(blocks[3]) && !info.is_hot(blocks[1]));
    }

    #[test]
    fn loop_heuristic_marks_the_loop_body_hot() {
        let mut ctx = test_context();
        // entry -> header; header -> {body, exit}; body -> header (back edge).
        let blocks = dummy_blocks(&mut ctx, 4);
        let edge = |to| CfgEdge { to, weight: None };
        let successors = vec![vec![edge(1)], vec![edge(2), edge(3)], vec![edge(1)], vec![]];

        let info = HotPathInfo::from_edges(blocks.clone(), successors);

        // Staying in the loop gets the 124/128 loop-branch heuristic.
        assert!(
            info.edge_probability(blocks[1], blocks[2]) == BranchProbability::from_ratio(124, 128)
        );
        // The header is amplified by the expected trip count (1/(1-124/128)).
        assert_eq!(info.block_frequency(blocks[1]), 32 * ENTRY_FREQUENCY);
        assert_eq!(info.block_frequency(blocks[3]), ENTRY_FREQUENCY);
        assert!(info.hot_path() == [blocks[0], blocks[1], blocks[2]]);
        assert!(info.is_hot(blocks[1]) && info.is_hot(blocks[2]));
    }

    #[test]
    fn self_loop_is_scaled_and_terminates() {
        let mut ctx = test_context();
        let blocks = dummy_blocks(&mut ctx, 3);
        let edge = |to| CfgEdge { to, weight: None };
        let successors = vec![vec![edge(1)], vec![edge(1), edge(2)], vec![]];

        let info = HotPathInfo::from_edges(blocks.clone(), successors);

        assert_eq!(info.block_frequency(blocks[1]), 32 * ENTRY_FREQUENCY);
        assert_eq!(info.block_frequency(blocks[2]), ENTRY_FREQUENCY);
    }

    #[test]
    fn unreachable_blocks_are_cold() {
        let mut ctx = test_context();
        let blocks = dummy_blocks(&mut ctx, 3);
        let edge = |to| CfgEdge { to, weight: None };
        let successors = vec![vec![edge(1)], vec![], vec![edge(1)]];

        let info = HotPathInfo::from_edges(blocks.clone(), successors);

        assert_eq!(info.block_frequency(blocks[2]), 0);
        assert!(info.hot_path() == [blocks[0], blocks[1]]);
    }

    #[test]
    fn for_op_reads_weights_through_the_interface() {
        use crate::ll::op_interfaces::WeightedBranchOpInterface;

        let mut ctx = test_context();
        let i1_ty: TypeHandle =
            builtin::types::IntegerType::get(&mut ctx, 1, Signedness::Signless).into();
        let fn_ty: TypedHandle<FuncType> = FuncType::get(&mut ctx, i1_ty, vec![i1_ty], false);
        let func = FuncOp::new(&mut ctx, "weighted".try_into().unwrap(), fn_ty);
        func.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(&mut ctx);

        let entry = func.get_entry_block(&ctx).unwrap();
        let cond = entry.deref(&ctx).get_argument(0);
        let then_block = BasicBlock::new(&mut ctx, Some("then".try_into().unwrap()), vec![]);
        let else_block = BasicBlock::new(&mut ctx, Some("else".try_into().unwrap()), vec![]);
        let join_block = BasicBlock::new(&mut ctx, Some("join".try_into().unwrap()), vec![]);
        then_block.insert_at_back(func.get_region(&ctx).expect("llvm.func definition must have a body"), &ctx);
        else_block.insert_at_back(func.get_region(&ctx).expect("llvm.func definition must have a body"), &ctx);
        join_block.insert_at_back(func.get_region(&ctx).expect("llvm.func definition must have a body"), &ctx);

        let cond_br = CondBrOp::new(&mut ctx, cond, then_block, vec![], else_block, vec![]);
        cond_br.set_successor_weights(&ctx, vec![1, 7]);
        cond_br.get_operation().insert_at_back(entry, &ctx);
        BrOp::new(&mut ctx, join_block, vec![])
            .get_operation()
            .insert_at_back(then_block, &ctx);
        BrOp::new(&mut ctx, join_block, vec![])
            .get_operation()
            .insert_at_back(else_block, &ctx);
        ReturnOp::new(&mut ctx, Some(cond))
            .get_operation()
            .insert_at_back(join_block, &ctx);

        let info = HotPathInfo::for_region(func.get_region(&ctx).expect("llvm.func definition must have a body"), &ctx);

        assert!(info.edge_probability(entry, else_block) == BranchProbability::from_ratio(7, 8));
        assert!(info.hot_path() == [entry, else_block, join_block]);
        assert_eq!(info.block_frequency(join_block), ENTRY_FREQUENCY);
    }
}
