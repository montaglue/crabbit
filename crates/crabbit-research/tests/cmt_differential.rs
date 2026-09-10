//! Differential tests of pliron-ll's dominance analyses against
//! combinatorial-matrix-theory's boolean-dataflow oracle. Relocated from
//! pliron-ll's own test module: the oracle is a private research
//! checkout, so the public workspace must not resolve it, even as a
//! dev-dependency (docs/MIDEND-PLAN.md keeps the cross-check
//! obligations; this crate is where the private world plugs in).

use pliron::builtin::op_interfaces::{AtMostOneRegionInterface as _, OneResultInterface as _};
use pliron::context::{Context, Ptr};
use pliron::linked_list::ContainsLinkedList as _;
use pliron::region::Region;
use pliron_ll::dialects::builtin::types::{IntegerType, Signedness};
use pliron_ll::dialects::llvm::{
    attributes::LinkageAttr,
    ops::{BrOp, CondBrOp, FuncOp, ReturnOp, UndefOp},
    types::FuncType,
};
use pliron_ll::ir::basic_block::BasicBlock;
use pliron_ll::ir::{op::Op, r#type::TypeHandle};
use pliron_ll::passes::llvm::analysis::{PostDomTree, RegionCfg, dominator_tree};

/// Build a CFG from an adjacency list (node 0 = the entry block) where
/// each node's successors are encoded as: 0 succs → ret, 1 succ → br,
/// 2 succs → cond_br on an undef i1. Mirrors the helper the analyses'
/// own unit tests use.
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

/// Differential test: the dominator tree vs combinatorial-matrix-theory's
/// boolean-dataflow DomMatrix, on several tangled CFGs (nested loops,
/// shared exits, unreachable nodes).
#[test]
fn dominators_agree_with_cmt_boolean_dataflow() {
    let shapes: &[&[&[usize]]] = &[
        &[&[1], &[2, 3], &[1], &[]],
        // nested loop: 0→1; 1→2|5; 2→3|4; 3→2 (inner latch); 4→1 (outer latch); 5 exit
        &[&[1], &[2, 5], &[3, 4], &[2], &[1], &[]],
        // diamond into loop with two exits and a shared join
        &[&[1, 2], &[3], &[3], &[4, 5], &[3], &[6, 2], &[]],
        // unreachable node 4
        &[&[1], &[2, 3], &[], &[1], &[2]],
    ];
    for edges in shapes {
        let (ctx, region) = build_cfg(edges);
        let dom = dominator_tree(&ctx, region);
        let (index, matrix) = cmt_boolean_dataflow::DomMatrix::compute(&ctx, region);
        for a in dom.nodes() {
            for b in dom.nodes() {
                assert_eq!(
                    dom.dominates(&a, &b),
                    matrix.dominates(index.idx(a), index.idx(b)),
                    "dominates disagreement in shape {edges:?}"
                );
            }
            let ours = dom.idom(&a).map(|d| index.idx(d));
            let theirs = matrix.idom(index.idx(a));
            assert_eq!(ours, theirs, "idom disagreement in shape {edges:?}");
        }
    }
}

/// Differential: PostDomTree vs the oracle run on the REVERSED adjacency
/// with a synthetic exit as entry — post-dominance is dominance of the
/// reversed graph, so the two must agree exactly (including all-false
/// rows for blocks that never reach an exit).
#[test]
fn postdominators_agree_with_cmt_on_reversed_graph() {
    let shapes: &[&[&[usize]]] = &[
        &[&[1], &[2, 3], &[1], &[]],
        &[&[1], &[2, 5], &[3, 4], &[2], &[1], &[]],
        &[&[1, 2], &[3], &[3], &[4, 5], &[3], &[6, 2], &[]],
        // two exits sharing nothing
        &[&[1, 2], &[3], &[4], &[], &[]],
        // an infinite loop (3↔4) that never reaches the exit
        &[&[1, 3], &[2], &[], &[4], &[3]],
    ];
    for edges in shapes {
        let (ctx, region) = build_cfg(edges);
        let cfg = RegionCfg::new(&ctx, region);
        let pdt = PostDomTree::compute(&cfg);
        let n = cfg.blocks.len();
        // Reversed adjacency, node 0 = synthetic exit, node i+1 = block i.
        let mut adj = cmt_support::matrix::BoolMat::zero(n + 1);
        for e in cfg.exit_indices() {
            adj.set(0, e + 1, true);
        }
        for u in 0..n {
            for &v in &cfg.succs[u] {
                adj.set(v + 1, u + 1, true);
            }
        }
        let matrix = cmt_boolean_dataflow::DomMatrix::from_adjacency(&adj);
        for a in 0..n {
            for b in 0..n {
                assert_eq!(
                    pdt.postdominates(a, b),
                    matrix.dominates(a + 1, b + 1),
                    "postdominates({a},{b}) disagreement in shape {edges:?}"
                );
            }
            let ours = pdt.ipdom(a).map(|i| i + 1);
            let theirs = matrix.idom(a + 1).filter(|&i| i != 0);
            assert_eq!(ours, theirs, "ipdom({a}) disagreement in shape {edges:?}");
        }
    }
}
