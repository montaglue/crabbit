//! CFG analyses shared by the research mid-end passes (docs/MIDEND-PLAN.md):
//! dominators (pliron's Cooper–Harvey–Kennedy implementation, re-exported
//! with region-CFG types fixed), natural-loop detection, post-dominators
//! (the same iterative algorithm on the reversed CFG with a synthetic
//! exit), and a generic backward bitset dataflow fixpoint.

use rustc_hash::FxHashSet;

use crate::{
    context::{Context, Ptr},
    ir::{basic_block::BasicBlock, region::Region},
};

pub use pliron::graph::dominance::{DomTree, compute_dominator_tree};

/// The dominator tree of a function body's region CFG.
pub type RegionDomTree = DomTree<Ptr<Region>, Context>;

/// Compute the dominator tree of `region` (blocks unreachable from the
/// entry are absent from the tree; check with [DomTree::contains]).
pub fn dominator_tree(ctx: &Context, region: Ptr<Region>) -> RegionDomTree {
    compute_dominator_tree(ctx, &region)
}

/// A natural loop: all back edges `latch → header` where `header`
/// dominates `latch`, merged per header. `body` includes the header.
pub struct NaturalLoop {
    pub header: Ptr<BasicBlock>,
    pub latches: Vec<Ptr<BasicBlock>>,
    pub body: FxHashSet<Ptr<BasicBlock>>,
}

impl NaturalLoop {
    /// The unique predecessor of the header from outside the loop, if it
    /// exists and branches only to the header — i.e. an existing
    /// preheader. Passes that need one skip loops without it (creating
    /// preheaders is future CFG surgery; see docs/MIDEND-PLAN.md).
    pub fn preheader(&self, ctx: &Context) -> Option<Ptr<BasicBlock>> {
        let outside: Vec<Ptr<BasicBlock>> = self
            .header
            .preds(ctx)
            .into_iter()
            .filter(|pred| !self.body.contains(pred))
            .collect();
        match outside.as_slice() {
            [single] if single.deref(ctx).num_succ(ctx) == 1 => Some(*single),
            _ => None,
        }
    }
}

/// Natural loops of `region`, innermost first (by body size, ascending).
pub fn natural_loops(ctx: &Context, dom: &RegionDomTree) -> Vec<NaturalLoop> {
    let mut by_header: Vec<(Ptr<BasicBlock>, Vec<Ptr<BasicBlock>>)> = Vec::new();
    for block in dom.nodes() {
        for succ in block.deref(ctx).succs(ctx) {
            if dom.contains(&succ) && dom.dominates(&succ, &block) {
                match by_header.iter_mut().find(|(header, _)| *header == succ) {
                    Some((_, latches)) => latches.push(block),
                    None => by_header.push((succ, vec![block])),
                }
            }
        }
    }
    let mut loops: Vec<NaturalLoop> = by_header
        .into_iter()
        .map(|(header, latches)| {
            // Body: header plus everything that reaches a latch without
            // passing through the header (reverse flood from the latches).
            let mut body: FxHashSet<Ptr<BasicBlock>> = FxHashSet::default();
            body.insert(header);
            let mut work: Vec<Ptr<BasicBlock>> = latches.clone();
            while let Some(block) = work.pop() {
                if body.insert(block) {
                    work.extend(block.preds(ctx));
                }
            }
            NaturalLoop {
                header,
                latches,
                body,
            }
        })
        .collect();
    loops.sort_by_key(|l| l.body.len());
    loops
}

// ============================================================================
// Bitsets and an indexed view of the region CFG
// ============================================================================

/// A plain word-backed bitset sized at construction; the state type of
/// [backward_bitset_fixpoint].
#[derive(Clone, PartialEq, Eq)]
pub struct BitSet {
    words: Vec<u64>,
}

impl BitSet {
    pub fn new(nbits: usize) -> Self {
        BitSet {
            words: vec![0; nbits.div_ceil(64)],
        }
    }
    pub fn insert(&mut self, bit: usize) {
        self.words[bit / 64] |= 1 << (bit % 64);
    }
    pub fn remove(&mut self, bit: usize) {
        self.words[bit / 64] &= !(1 << (bit % 64));
    }
    pub fn contains(&self, bit: usize) -> bool {
        self.words[bit / 64] & (1 << (bit % 64)) != 0
    }
    /// `self |= other`; true if `self` changed.
    pub fn union_with(&mut self, other: &BitSet) -> bool {
        let mut changed = false;
        for (dst, src) in self.words.iter_mut().zip(&other.words) {
            let new = *dst | *src;
            changed |= new != *dst;
            *dst = new;
        }
        changed
    }
}

/// The region CFG as index-addressed adjacency lists (blocks in region
/// order, entry first) — the shape the fixpoint algorithms below and the
/// boolean-matrix differential tests both consume.
pub struct RegionCfg {
    pub blocks: Vec<Ptr<BasicBlock>>,
    index: rustc_hash::FxHashMap<Ptr<BasicBlock>, usize>,
    pub succs: Vec<Vec<usize>>,
    pub preds: Vec<Vec<usize>>,
}

impl RegionCfg {
    pub fn new(ctx: &Context, region: Ptr<Region>) -> Self {
        use crate::linked_list::ContainsLinkedList as _;
        let blocks: Vec<Ptr<BasicBlock>> = region.deref(ctx).iter(ctx).collect();
        let index: rustc_hash::FxHashMap<Ptr<BasicBlock>, usize> = blocks
            .iter()
            .enumerate()
            .map(|(i, b)| (*b, i))
            .collect();
        let mut succs = vec![Vec::new(); blocks.len()];
        let mut preds = vec![Vec::new(); blocks.len()];
        for (i, block) in blocks.iter().enumerate() {
            for succ in block.deref(ctx).succs(ctx) {
                let j = index[&succ];
                if !succs[i].contains(&j) {
                    succs[i].push(j);
                    preds[j].push(i);
                }
            }
        }
        RegionCfg {
            blocks,
            index,
            succs,
            preds,
        }
    }

    pub fn idx(&self, block: Ptr<BasicBlock>) -> usize {
        self.index[&block]
    }

    /// Blocks whose terminator has no successors (returns/unreachable):
    /// the sources of the synthetic exit edge in the reversed CFG.
    pub fn exit_indices(&self) -> Vec<usize> {
        (0..self.blocks.len())
            .filter(|&i| self.succs[i].is_empty())
            .collect()
    }
}

/// Iterative dominators on an arbitrary indexed graph (Cooper–Harvey–
/// Kennedy): `idoms[v]` for every node reachable from `entry`, `None`
/// for the entry itself and for unreachable nodes (also reported in
/// `reachable`). Shared by the post-dominator computation, which runs it
/// on the reversed CFG.
fn iterative_idoms(n: usize, entry: usize, succs: &[Vec<usize>]) -> (Vec<Option<usize>>, Vec<bool>) {
    // Reverse postorder from entry.
    let mut order = Vec::with_capacity(n);
    let mut seen = vec![false; n];
    let mut stack = vec![(entry, 0usize)];
    seen[entry] = true;
    while let Some(&mut (v, ref mut next)) = stack.last_mut() {
        if *next < succs[v].len() {
            let s = succs[v][*next];
            *next += 1;
            if !seen[s] {
                seen[s] = true;
                stack.push((s, 0));
            }
        } else {
            order.push(v);
            stack.pop();
        }
    }
    order.reverse();
    let mut rpo_number = vec![usize::MAX; n];
    for (i, &v) in order.iter().enumerate() {
        rpo_number[v] = i;
    }
    let mut preds = vec![Vec::new(); n];
    for v in 0..n {
        for &s in &succs[v] {
            if seen[v] && seen[s] {
                preds[s].push(v);
            }
        }
    }
    let mut idom: Vec<Option<usize>> = vec![None; n];
    idom[entry] = Some(entry);
    let intersect = |idom: &[Option<usize>], rpo: &[usize], mut a: usize, mut b: usize| {
        while a != b {
            while rpo[a] > rpo[b] {
                a = idom[a].unwrap();
            }
            while rpo[b] > rpo[a] {
                b = idom[b].unwrap();
            }
        }
        a
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &v in order.iter().skip(1) {
            let mut new_idom: Option<usize> = None;
            for &p in &preds[v] {
                if idom[p].is_some() {
                    new_idom = Some(match new_idom {
                        None => p,
                        Some(cur) => intersect(&idom, &rpo_number, cur, p),
                    });
                }
            }
            if idom[v] != new_idom {
                idom[v] = new_idom;
                changed = true;
            }
        }
    }
    idom[entry] = None;
    (idom, seen)
}

/// Post-dominators of a region CFG: dominators of the reversed graph with
/// a synthetic exit fed by every block whose terminator has no successors.
/// Blocks that cannot reach any exit (infinite loops) are unreachable in
/// the reversed graph and report `false`/`None` everywhere — callers must
/// treat them conservatively.
pub struct PostDomTree {
    cfg_len: usize,
    /// Immediate post-dominator by block index; `None` means "the
    /// synthetic exit" (for exit blocks) or "unknown" (unreachable).
    ipdom: Vec<Option<usize>>,
    reaches_exit: Vec<bool>,
}

impl PostDomTree {
    pub fn compute(cfg: &RegionCfg) -> PostDomTree {
        let n = cfg.blocks.len();
        let exit = n; // synthetic
        let mut rev_succs = vec![Vec::new(); n + 1];
        for v in 0..n {
            for &s in &cfg.succs[v] {
                rev_succs[s].push(v);
            }
        }
        for e in cfg.exit_indices() {
            rev_succs[exit].push(e);
        }
        let (idom, seen) = iterative_idoms(n + 1, exit, &rev_succs);
        PostDomTree {
            cfg_len: n,
            ipdom: idom[..n]
                .iter()
                .map(|d| d.filter(|&x| x != exit))
                .collect(),
            reaches_exit: seen[..n].to_vec(),
        }
    }

    pub fn reaches_exit(&self, b: usize) -> bool {
        self.reaches_exit[b]
    }

    /// Does block `a` post-dominate block `b`? False whenever either
    /// cannot reach an exit.
    pub fn postdominates(&self, a: usize, b: usize) -> bool {
        if !self.reaches_exit[a] || !self.reaches_exit[b] {
            return false;
        }
        let mut cur = b;
        loop {
            if cur == a {
                return true;
            }
            match self.ipdom[cur] {
                Some(next) => cur = next,
                None => return false,
            }
        }
    }

    pub fn ipdom(&self, b: usize) -> Option<usize> {
        debug_assert!(b < self.cfg_len);
        self.ipdom[b]
    }
}

/// The nearest common dominator of `blocks` (deepest block dominating all
/// of them); `None` when the list is empty or a block is outside the tree.
pub fn nearest_common_dominator(
    ctx: &Context,
    dom: &RegionDomTree,
    blocks: &[Ptr<BasicBlock>],
) -> Option<Ptr<BasicBlock>> {
    let _ = ctx;
    let mut iter = blocks.iter();
    let mut cur = *iter.next()?;
    if !dom.contains(&cur) {
        return None;
    }
    for block in iter {
        if !dom.contains(block) {
            return None;
        }
        while !dom.dominates(&cur, block) {
            cur = dom.idom(&cur)?;
        }
    }
    Some(cur)
}

/// Generic backward bitset dataflow to fixpoint over an indexed CFG:
/// `out(b) = ∪ in(succ)` (`boundary(b)` seeds `out` for exit-less
/// successors — it is always unioned in for blocks with no successors),
/// `in(b) = transfer(b, out(b))`. Returns `(in, out)` per block index.
pub fn backward_bitset_fixpoint(
    cfg: &RegionCfg,
    nbits: usize,
    boundary: impl Fn(usize) -> BitSet,
    transfer: impl Fn(usize, &BitSet) -> BitSet,
) -> (Vec<BitSet>, Vec<BitSet>) {
    let n = cfg.blocks.len();
    let mut live_in = vec![BitSet::new(nbits); n];
    let mut live_out = vec![BitSet::new(nbits); n];
    for b in 0..n {
        if cfg.succs[b].is_empty() {
            live_out[b] = boundary(b);
        }
        live_in[b] = transfer(b, &live_out[b]);
    }
    let mut changed = true;
    while changed {
        changed = false;
        for b in (0..n).rev() {
            let mut out = if cfg.succs[b].is_empty() {
                boundary(b)
            } else {
                BitSet::new(nbits)
            };
            for &s in &cfg.succs[b] {
                out.union_with(&live_in[s]);
            }
            if out != live_out[b] {
                live_out[b] = out;
                let new_in = transfer(b, &live_out[b]);
                if new_in != live_in[b] {
                    live_in[b] = new_in;
                    changed = true;
                }
            }
        }
    }
    (live_in, live_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pliron::builtin::op_interfaces::{AtMostOneRegionInterface as _, OneResultInterface as _};
    use crate::{
        dialects::{
            builtin::types::{IntegerType, Signedness},
            llvm::{
                attributes::LinkageAttr,
                ops::{BrOp, CondBrOp, FuncOp, ReturnOp, UndefOp},
                types::FuncType,
            },
        },
        ir::{op::Op, r#type::TypeHandle},
    };

    /// Build a CFG from an adjacency list (node 0 = entry appended after
    /// the function's entry block? no — node 0 IS the entry block) where
    /// each node's successors are encoded as: 0 succs → ret, 1 succ → br,
    /// 2 succs → cond_br on an undef i1.
    fn build_cfg(edges: &[&[usize]]) -> (Context, Ptr<Region>) {
        let mut ctx = Context::new();
        let i1: TypeHandle = IntegerType::get(&mut ctx, 1, Signedness::Signless).into();
        let fn_ty = FuncType::get(&mut ctx, i1, vec![], false);
        let func = FuncOp::new(&mut ctx, "cfg".try_into().unwrap(), fn_ty);
        func.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(&mut ctx);
        let region = func.get_region(&ctx).unwrap();
        let mut blocks = vec![func.get_entry_block(&ctx).unwrap()];
        for _ in 1..edges.len() {
            let block = BasicBlock::new(&mut ctx, None, vec![]);
            block.insert_at_back(region, &ctx);
            blocks.push(block);
        }
        for (i, succs) in edges.iter().enumerate() {
            let cond = UndefOp::new(&mut ctx, i1);
            cond.get_operation().insert_at_back(blocks[i], &ctx);
            let cond_v = cond.get_result(&ctx);
            match succs {
                [] => ReturnOp::new(&mut ctx, Some(cond_v))
                    .get_operation()
                    .insert_at_back(blocks[i], &ctx),
                [a] => BrOp::new(&mut ctx, blocks[*a], vec![])
                    .get_operation()
                    .insert_at_back(blocks[i], &ctx),
                [a, b] => CondBrOp::new(&mut ctx, cond_v, blocks[*a], vec![], blocks[*b], vec![])
                    .get_operation()
                    .insert_at_back(blocks[i], &ctx),
                _ => unreachable!(),
            }
        }
        (ctx, region)
    }

    // The differential tests against combinatorial-matrix-theory's
    // boolean-dataflow oracle (dominators + post-dominators over these
    // same CFG shapes) live in crates/crabbit-research/tests/
    // cmt_differential.rs — the oracle is a private research checkout the
    // public workspace must not resolve.

    /// Backward fixpoint sanity on a diamond: a bit gen'd in one arm is
    /// live-out of the entry but not of the other arm.
    #[test]
    fn backward_fixpoint_propagates_against_edges() {
        let shapes: &[&[usize]] = &[&[1, 2], &[3], &[3], &[]];
        let (ctx, region) = build_cfg(&shapes);
        let cfg = RegionCfg::new(&ctx, region);
        // bit 0 is "generated" (read) in block 1 only; nothing kills.
        let (live_in, live_out) = backward_bitset_fixpoint(
            &cfg,
            1,
            |_| BitSet::new(1),
            |b, out| {
                let mut set = out.clone();
                if b == 1 {
                    set.insert(0);
                }
                set
            },
        );
        assert!(live_in[1].contains(0));
        assert!(live_out[0].contains(0), "entry sees the arm's read");
        assert!(!live_in[2].contains(0), "the other arm does not");
        assert!(!live_out[3].contains(0), "nothing after the join");
    }

    /// diamond-plus-loop CFG: entry → header; header → (body | exit);
    /// body → header. Checks dominators and the loop's shape.
    #[test]
    fn finds_natural_loop_and_preheader() {
        let mut ctx = Context::new();
        let i1: TypeHandle = IntegerType::get(&mut ctx, 1, Signedness::Signless).into();
        let fn_ty = FuncType::get(&mut ctx, i1, vec![], false);
        let func = FuncOp::new(&mut ctx, "loops".try_into().unwrap(), fn_ty);
        func.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(&mut ctx);
        let region = func.get_region(&ctx).unwrap();
        let entry = func.get_entry_block(&ctx).unwrap();
        let header = BasicBlock::new(&mut ctx, None, vec![]);
        header.insert_at_back(region, &ctx);
        let body = BasicBlock::new(&mut ctx, None, vec![]);
        body.insert_at_back(region, &ctx);
        let exit = BasicBlock::new(&mut ctx, None, vec![]);
        exit.insert_at_back(region, &ctx);

        let cond = UndefOp::new(&mut ctx, i1);
        cond.get_operation().insert_at_back(entry, &ctx);
        let cond_v = cond.get_result(&ctx);
        BrOp::new(&mut ctx, header, vec![])
            .get_operation()
            .insert_at_back(entry, &ctx);
        CondBrOp::new(&mut ctx, cond_v, body, vec![], exit, vec![])
            .get_operation()
            .insert_at_back(header, &ctx);
        BrOp::new(&mut ctx, header, vec![])
            .get_operation()
            .insert_at_back(body, &ctx);
        ReturnOp::new(&mut ctx, Some(cond_v))
            .get_operation()
            .insert_at_back(exit, &ctx);

        let dom = dominator_tree(&ctx, region);
        assert!(dom.dominates(&entry, &exit));
        assert!(dom.dominates(&header, &body));
        assert!(!dom.dominates(&body, &exit));

        let loops = natural_loops(&ctx, &dom);
        assert_eq!(loops.len(), 1);
        let l = &loops[0];
        assert_eq!(l.header, header);
        assert_eq!(l.latches, vec![body]);
        assert_eq!(l.body.len(), 2);
        assert!(l.body.contains(&header) && l.body.contains(&body));
        assert_eq!(l.preheader(&ctx), Some(entry));
    }
}
