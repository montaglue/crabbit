//! Conservative FULL loop unrolling for the llvm-dialect mid-end.
//!
//! Motivation (docs/MIDEND-PLAN.md "unroll/vectorize deferred", revisited
//! after the kernel-corpus GPU measurements): nvcc fully unrolls small
//! constant-trip inner loops (gemm_tiled's 16-iteration tile loop), and
//! the per-iteration index math + loop control is exactly what stalls the
//! SASS crabbit emits. Full unroll of a statically-known short loop turns
//! that into straight-line code the existing simplify/gvn/adce round then
//! constant-folds and CSEs.
//!
//! v2 restrictions, all deliberate (keep it provable):
//! - FULL unroll only, no partial unroll (no remainder loops, no epilogue
//!   CFG surgery).
//! - Natural loops with ONE latch and ONE exit edge, in two shapes:
//!   `header ⇄ header` (test-after-body, do-while) or the generalized
//!   while-shape `header → BODY → header` with the conditional exit in
//!   the header, where BODY is a single-entry single-exit sub-CFG:
//!   entered only through the header edge, ACYCLIC internally (an inner
//!   backedge means an inner loop — it unrolls first, see the rounds
//!   loop), every block's terminator a br/cond_br staying inside the
//!   loop, all paths converging on the single latch. Arbitrary internal
//!   diamonds are fine (rustc's sign-select bodies). The loop must
//!   already have a preheader (like [licm](super::licm)).
//! - Trip count must be statically provable from the canonical rustc
//!   `Range` shape: an induction block-arg starting at a constant,
//!   stepped on the backedge either ADDITIVELY (`add iv, constant`) or
//!   GEOMETRICALLY (`lshr`/`ashr`/`shl` by an in-range constant amount,
//!   or `mul`/`udiv` by a power-of-2 constant — the warp-reduction
//!   `off >>= 1` family), compared against a constant by the exiting
//!   `icmp`. The trip count is found by direct simulation of that
//!   (wrapping) arithmetic, so every predicate and any step sign/width
//!   ≤ 64 handled by the table below is exact — the same simulator runs
//!   both step kinds, only the per-iteration [IvStep::apply] differs.
//! - Bounds: 2 ≤ trip ≤ [MAX_TRIP], loop body ≤ [MAX_LOOP_OPS]
//!   non-terminator ops, trip × body ≤ [MAX_UNROLL_GROWTH] cloned ops
//!   per unroll, and ≤ [MAX_FUNC_GROWTH] cloned ops per function per
//!   pass run (the rounds loop is a fixpoint — nested constant-trip
//!   loops unroll inside-out one per round, so growth compounds).
//! - Every non-terminator op is clonable (no nested regions). Cloning is
//!   execution-count preserving — iteration k's clone runs exactly when
//!   iteration k ran (multi-block bodies clone the whole sub-CFG, so
//!   diamond arms still run under their original conditions) — so
//!   loads/stores/calls need no speculation argument and are cloned
//!   as-is.
//!
//! The rewrite for 1-block bodies places the straight-line clones in the
//! PREHEADER (before its terminator), substitutes the induction value
//! with its per-iteration constant, threads loop-carried block args
//! through the clones, rewires uses of loop-defined values from outside
//! the loop to the final iteration's clone, retargets the preheader
//! branch to the exit block (with the exit edge's operands mapped
//! through the last iteration), and deletes the loop blocks. Multi-block
//! bodies clone the body sub-CFG once per iteration (fresh blocks,
//! branch targets remapped within the copy), chain the copies through
//! continuation blocks carrying the loop-carried header args, and put
//! the final (failing) header evaluation in the tail block that branches
//! to the exit. Dead leftovers (the cloned
//! `icmp`s, the final iv updates) are pure and unused — the downstream
//! adce/simplify round erases them.
//!
//! ADJOINT (backward attribution): unroll is 1→N cloning. Every clone is
//! stamped `derived_from` the original op's effective sources (the
//! copied identity attributes are stripped first — a clone must not
//! duplicate the original's dense `ll.op_id`). Materialized iv constants
//! derive from the iv update op; the new preheader→exit branch derives
//! from the cond_br it replaces.

use rustc_hash::{FxHashMap, FxHashSet};
use std::num::NonZero;

use pliron::builtin::op_interfaces::{AtMostOneRegionInterface as _, BranchOpInterface as _};

use crate::{
    common_traits::Named as _,
    context::{Context, Ptr},
    conversion::pass::{AnalysisManager, Pass, PassResult, changed, unchanged},
    dialects::{
        builtin::{attributes::IntegerAttr, ops::ConstantOp, types::IntegerType},
        llvm::{
            attributes::ICmpPredicateAttr,
            op_interfaces::IsDeclaration,
            ops::{AShrOp, AddOp, BrOp, CondBrOp, ICmpOp, LShrOp, MulOp, ShlOp, UDivOp},
        },
    },
    identifier::Identifier,
    ir::{
        basic_block::BasicBlock,
        op::Op,
        operation::Operation,
        region::Region,
        r#type::{TypeHandle, Typed as _, TypedHandle},
        value::{DefiningEntity, Value},
    },
    irbuild::{cloning, listener::DummyListener, rewriter::IRRewriter},
    linked_list::{ContainsLinkedList, LinkedList as _},
    utils::apint::APInt,
};

use super::{
    analysis::{NaturalLoop, dominator_tree, natural_loops},
    inline::collect_functions,
    midend_gate::midend_disabled,
    simplify::{as_const_operand, mask_to_width, sign_extend},
};
use crate::passes::aarch64::opmap;
use crate::target_profile::TargetProfile;

/// Upper bound on the statically-computed trip count.
const MAX_TRIP: u64 = 32;
/// Lower bound: a 0/1-trip "loop" is simplify-cfg's job, not ours.
const MIN_TRIP: u64 = 2;
/// Upper bound on non-terminator ops across the loop's blocks. Raised
/// 40 → 768 with the multi-block-body support: the shape this extension
/// exists for (iq3_xxs_dequant's outer chunk loop) carries ~650 ops
/// once its inner 4-trip diamond-body loop has been unrolled into it.
/// A bare per-body cap no longer bounds growth by itself — that is
/// [MAX_UNROLL_GROWTH]'s job — this only rejects degenerate giants.
const MAX_LOOP_OPS: usize = 768;
/// Upper bound on trip × body ops — the number of ops ONE unroll clones.
/// The v1 bounds allowed 32 × 40 = 1280; iq3's outer loop needs
/// 4 × ~650 ≈ 2600; 4096 covers that with headroom while keeping any
/// single unroll under ~4k cloned ops.
const MAX_UNROLL_GROWTH: u64 = 4096;
/// Cumulative cloned-op budget per function per pass run. The rounds
/// loop is a fixpoint (nested constant-trip loops unroll inside-out,
/// one per round), so growth compounds across rounds: iq3's full nest
/// (inner 4×~140, outer 4×~650, 32×~10 copy loop) sums to ~3.5k;
/// 8192 bounds the worst case at roughly twice that.
const MAX_FUNC_GROWTH: usize = 8192;
/// Unrolls per function: each successful unroll invalidates the loop
/// forest (blocks are deleted), so loops are re-discovered; nested
/// constant loops unroll inside-out one per round.
const MAX_ROUNDS: usize = 16;

pub struct LLVMUnrollPass {
    /// [TargetProfile::has_branch_divergence]. The thresholds are
    /// currently target-independent (the CPU corpus showed no regression
    /// — see docs/MIDEND-PLAN.md); kept so a target-aware cost split has
    /// its seam ready.
    #[allow(dead_code)]
    divergent_target: bool,
}

impl LLVMUnrollPass {
    pub fn new(profile: &TargetProfile) -> Self {
        LLVMUnrollPass {
            divergent_target: profile.has_branch_divergence,
        }
    }
}

impl Pass for LLVMUnrollPass {
    fn name(&self) -> &str {
        "llvm-unroll"
    }

    fn run(
        &mut self,
        root: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> pliron::result::Result<PassResult> {
        if midend_disabled("unroll") {
            return Ok(unchanged());
        }
        let mut any = false;
        for func in collect_functions(ctx, root) {
            if func.is_declaration(ctx) {
                continue;
            }
            let region = func
                .get_region(ctx)
                .expect("llvm.func definition must have a body");
            let mut growth = 0usize;
            for _ in 0..MAX_ROUNDS {
                let Some(cloned) = unroll_one_loop(ctx, region, growth) else {
                    break;
                };
                growth = growth.saturating_add(cloned);
                any = true;
            }
        }
        Ok(if any { changed() } else { unchanged() })
    }
}

/// Find one fully-unrollable loop (innermost first) within the remaining
/// per-function growth budget and unroll it; returns the number of ops
/// cloned. One loop per call: the rewrite deletes blocks, so the loop
/// forest and dominator tree are recomputed by the caller before the
/// next attempt.
fn unroll_one_loop(ctx: &mut Context, region: Ptr<Region>, growth_so_far: usize) -> Option<usize> {
    let dom = dominator_tree(ctx, region);
    let loops = natural_loops(ctx, &dom);
    for natural_loop in &loops {
        if let Some(plan) = analyze(ctx, natural_loop) {
            let cloned = plan.cloned_ops();
            if growth_so_far.saturating_add(cloned) > MAX_FUNC_GROWTH {
                continue;
            }
            apply(ctx, &plan);
            return Some(cloned);
        }
    }
    None
}

// ============================================================================
// Analysis: loop shape, induction variable, static trip count
// ============================================================================

/// The backedge's per-iteration induction update, applied by direct
/// simulation (see [IvStep::apply]). Every payload constant is already
/// masked to the iv width by the matcher.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IvStep {
    /// `iv + C` (either operand order), wrapping.
    Add(u128),
    /// `iv << C`, C < width (an out-of-range shift is poison — rejected).
    Shl(u128),
    /// `iv >> C` logical, C < width.
    LShr(u128),
    /// `iv >> C` arithmetic, C < width.
    AShr(u128),
    /// `iv * C` (either operand order), C a power of two, wrapping.
    MulPow2(u128),
    /// `iv / C` unsigned, C a power of two (so never 0).
    UDivPow2(u128),
}

impl IvStep {
    /// One simulated iteration: `iv` is masked to `width` on entry and
    /// the result is masked back to `width`, mirroring the wrapping /
    /// bit-exact semantics of the matched op.
    fn apply(self, iv: u128, width: u32) -> u128 {
        let iv = mask_to_width(iv, width);
        let next = match self {
            IvStep::Add(c) => iv.wrapping_add(c),
            IvStep::Shl(c) => iv << c,
            IvStep::LShr(c) => iv >> c,
            IvStep::AShr(c) => (sign_extend(iv, width) >> c) as u128,
            IvStep::MulPow2(c) => iv.wrapping_mul(c),
            IvStep::UDivPow2(c) => iv / c,
        };
        mask_to_width(next, width)
    }
}

fn is_pow2(bits: u128) -> bool {
    bits != 0 && bits & (bits - 1) == 0
}

struct UnrollPlan {
    preheader: Ptr<BasicBlock>,
    header: Ptr<BasicBlock>,
    /// The body sub-CFG in reverse post-order from its entry (the
    /// cond_br's in-loop successor): empty for the 1-block do-while
    /// shape (header==latch), one block for the v1 2-block while-shape,
    /// several for a single-entry acyclic diamond body. The latch is
    /// always the sub-CFG's single block branching back to the header.
    body_rpo: Vec<Ptr<BasicBlock>>,
    /// The single latch (== `header` for the do-while shape).
    latch: Ptr<BasicBlock>,
    /// Non-terminator op count across the loop, for growth accounting.
    loop_ops: usize,
    /// The loop's single exiting terminator (a `cond_br` in the header
    /// for the while-shape / in the single block for the do-while shape).
    cond_br: Ptr<Operation>,
    exit_dest: Ptr<BasicBlock>,
    /// Operands the exiting edge passes to `exit_dest`.
    exit_operands: Vec<Value>,
    /// Operands the cond_br passes on its in-loop edge (the body block's
    /// arguments for the while-shape; the backedge args for do-while).
    inloop_operands: Vec<Value>,
    /// Operands the backedge passes to the header's block args.
    latch_operands: Vec<Value>,
    /// Initial header args: the preheader branch's operands.
    init_operands: Vec<Value>,
    iv_index: usize,
    iv_ty: TypedHandle<IntegerType>,
    iv_width: u32,
    iv_start: u128,
    iv_step: IvStep,
    /// The op producing the next iv (`add`/`shl`/`lshr`/`ashr`/`mul`/
    /// `udiv`; attribution anchor for the materialized per-iteration
    /// constants).
    iv_update_op: Ptr<Operation>,
    trip: u64,
}

impl UnrollPlan {
    /// Ops the rewrite will clone: every loop op once per iteration,
    /// plus one extra header evaluation for the while-shape.
    fn cloned_ops(&self) -> usize {
        let iters = if self.body_rpo.is_empty() {
            self.trip
        } else {
            self.trip + 1
        };
        (self.loop_ops as u64).saturating_mul(iters) as usize
    }
}

fn defined_in_loop(ctx: &Context, value: Value, body: &FxHashSet<Ptr<BasicBlock>>) -> bool {
    match value.defining_entity() {
        DefiningEntity::Op(op) => op
            .deref(ctx)
            .get_container()
            .map(|block| body.contains(&block))
            .unwrap_or(false),
        DefiningEntity::Block(block) => body.contains(&block),
    }
}

fn analyze(ctx: &Context, natural_loop: &NaturalLoop) -> Option<UnrollPlan> {
    if natural_loop.latches.len() != 1 {
        return None;
    }
    let header = natural_loop.header;
    let latch = natural_loop.latches[0];
    let preheader = natural_loop.preheader(ctx)?;
    let pre_term = preheader.deref(ctx).get_terminator(ctx)?;
    if Operation::get_opid(pre_term, ctx) != BrOp::get_opid_static() {
        return None;
    }

    // Shape: 1 block (header==latch, exit test after the body) or the
    // generalized while-shape (exit test in the header, the body a
    // single-entry acyclic sub-CFG whose paths converge on the latch).
    let (body_rpo, cond_br) = if natural_loop.body.len() == 1 {
        if latch != header {
            return None;
        }
        let term = header.deref(ctx).get_terminator(ctx)?;
        if Operation::get_opid(term, ctx) != CondBrOp::get_opid_static() {
            return None;
        }
        (Vec::new(), term)
    } else {
        if latch == header {
            return None;
        }
        let header_term = header.deref(ctx).get_terminator(ctx)?;
        if Operation::get_opid(header_term, ctx) != CondBrOp::get_opid_static() {
            return None;
        }
        let latch_term = latch.deref(ctx).get_terminator(ctx)?;
        if Operation::get_opid(latch_term, ctx) != BrOp::get_opid_static() {
            return None;
        }
        if latch_term.deref(ctx).get_successor(0) != header {
            return None;
        }
        // The body sub-CFG entry: the header cond_br's unique in-loop
        // successor (re-checked with the exit edge below).
        let s0 = header_term.deref(ctx).get_successor(0);
        let s1 = header_term.deref(ctx).get_successor(1);
        let entry = match (
            natural_loop.body.contains(&s0),
            natural_loop.body.contains(&s1),
        ) {
            (true, false) => s0,
            (false, true) => s1,
            _ => return None,
        };
        if entry == header {
            return None;
        }
        let rpo = body_subgraph_rpo(ctx, natural_loop, header, latch, entry)?;
        (rpo, header_term)
    };

    // The cond_br must have exactly one in-loop and one out-of-loop
    // successor; the in-loop one is the body sub-CFG's entry
    // (while-shape) or the header itself (do-while shape).
    let cbr = CondBrOp::from_operation(cond_br);
    let succ0 = cbr.get_operation().deref(ctx).get_successor(0);
    let succ1 = cbr.get_operation().deref(ctx).get_successor(1);
    let expected_inloop = body_rpo.first().copied().unwrap_or(header);
    let (inloop_idx, exit_idx) = if succ0 == expected_inloop && !natural_loop.body.contains(&succ1)
    {
        (0usize, 1usize)
    } else if succ1 == expected_inloop && !natural_loop.body.contains(&succ0) {
        (1usize, 0usize)
    } else {
        return None;
    };
    let exit_dest = cbr.get_operation().deref(ctx).get_successor(exit_idx);
    let exit_operands = cbr.successor_operands(ctx, exit_idx);
    let inloop_operands = cbr.successor_operands(ctx, inloop_idx);

    // Size and clonability: every non-terminator op is region-free.
    let mut loop_ops = 0usize;
    for block in &natural_loop.body {
        let term = block.deref(ctx).get_terminator(ctx)?;
        for op in block.deref(ctx).iter(ctx) {
            if op == term {
                continue;
            }
            if op.deref(ctx).num_regions() != 0 {
                return None;
            }
            loop_ops += 1;
        }
    }
    if loop_ops > MAX_LOOP_OPS {
        return None;
    }

    // Header arg plumbing: init from the preheader branch, backedge from
    // the latch terminator (the body's br, or the cond_br's in-loop edge).
    let num_args = header.deref(ctx).get_num_arguments();
    let init_operands = BrOp::from_operation(pre_term).successor_operands(ctx, 0);
    let latch_operands = if body_rpo.is_empty() {
        inloop_operands.clone()
    } else {
        let latch_term = latch.deref(ctx).get_terminator(ctx)?;
        BrOp::from_operation(latch_term).successor_operands(ctx, 0)
    };
    if init_operands.len() != num_args || latch_operands.len() != num_args {
        return None;
    }
    // While-shape: the in-loop edge's operand count must match the body
    // entry's arguments (it always does in verified IR; cheap to check).
    if let Some(body_entry) = body_rpo.first()
        && inloop_operands.len() != body_entry.deref(ctx).get_num_arguments()
    {
        return None;
    }

    // The exiting condition: `icmp <pred> V, C2` (either operand order),
    // defined inside the loop, where V is the induction block-arg or its
    // update op's result (`add`/`shl`/`lshr`/`ashr`/`mul`/`udiv` by C1).
    let cond = cbr.get_operand_condition(ctx);
    let icmp_op = cond.defining_op()?;
    if Operation::get_opid(icmp_op, ctx) != ICmpOp::get_opid_static() {
        return None;
    }
    if !defined_in_loop(ctx, cond, &natural_loop.body) {
        return None;
    }
    let predicate = ICmpOp::from_operation(icmp_op).predicate(ctx);
    let (lhs, rhs) = {
        let op_ref = icmp_op.deref(ctx);
        (op_ref.get_operand(0), op_ref.get_operand(1))
    };
    let (v, bound, v_is_lhs) = if let Some(c) = as_const_operand(ctx, rhs) {
        (lhs, c, true)
    } else if let Some(c) = as_const_operand(ctx, lhs) {
        (rhs, c, false)
    } else {
        return None;
    };

    let header_args: Vec<Value> = (0..num_args)
        .map(|j| header.deref(ctx).get_argument(j))
        .collect();

    // Induction-update matcher, any block of the loop: `add iv, C1` /
    // `mul iv, C1(pow2)` (either operand order), or the ordered shapes
    // `shl|lshr|ashr iv, C1(<width)` / `udiv iv, C1(pow2)` with the
    // constant on the rhs.
    let match_iv_update = |upd_val: Value| -> Option<(usize, IvStep, u32, Ptr<Operation>)> {
        let upd_op = upd_val.defining_op()?;
        let opid = Operation::get_opid(upd_op, ctx);
        let commutative = opid == AddOp::get_opid_static() || opid == MulOp::get_opid_static();
        let ordered = opid == ShlOp::get_opid_static()
            || opid == LShrOp::get_opid_static()
            || opid == AShrOp::get_opid_static()
            || opid == UDivOp::get_opid_static();
        if !commutative && !ordered {
            return None;
        }
        if !defined_in_loop(ctx, upd_val, &natural_loop.body) {
            return None;
        }
        let (a, b) = {
            let op_ref = upd_op.deref(ctx);
            (op_ref.get_operand(0), op_ref.get_operand(1))
        };
        let (arg, c) = if let Some(c) = as_const_operand(ctx, b) {
            (a, c)
        } else if commutative && let Some(c) = as_const_operand(ctx, a) {
            (b, c)
        } else {
            return None;
        };
        let bits = c.masked();
        let step = if opid == AddOp::get_opid_static() {
            IvStep::Add(bits)
        } else if opid == MulOp::get_opid_static() {
            if !is_pow2(bits) {
                return None;
            }
            IvStep::MulPow2(bits)
        } else if opid == UDivOp::get_opid_static() {
            if !is_pow2(bits) {
                return None;
            }
            IvStep::UDivPow2(bits)
        } else {
            // Shifts: an amount ≥ width is poison in LLVM — never
            // simulate it. (c.width == the iv width is checked below.)
            if bits >= c.width as u128 {
                return None;
            }
            if opid == ShlOp::get_opid_static() {
                IvStep::Shl(bits)
            } else if opid == LShrOp::get_opid_static() {
                IvStep::LShr(bits)
            } else {
                IvStep::AShr(bits)
            }
        };
        let j = header_args.iter().position(|h| *h == arg)?;
        Some((j, step, c.width, upd_op))
    };

    // Identify the induction variable and whether the compare tests the
    // NEXT value (`iv + step`) or the current one.
    let (iv_index, iv_step, step_width, iv_update_op, cmp_on_next) =
        if let Some(j) = header_args.iter().position(|h| *h == v) {
            let (upd_j, step, sw, upd_op) = match_iv_update(latch_operands[j])?;
            if upd_j != j {
                return None;
            }
            (j, step, sw, upd_op, false)
        } else {
            let (j, step, sw, upd_op) = match_iv_update(v)?;
            if latch_operands[j] != v {
                return None;
            }
            (j, step, sw, upd_op, true)
        };

    let start = as_const_operand(ctx, init_operands[iv_index])?;
    let width = start.width;
    if width == 0 || width > 64 || step_width != width || bound.width != width {
        return None;
    }

    // Simulate the loop's wrapping arithmetic to the exact trip count.
    // While-shape tests before the body runs; do-while runs the body once
    // before its first test.
    let continue_on = inloop_idx == 0; // cond true takes successor 0
    let bound_bits = bound.masked();
    let eval_continue = |iv: u128| -> bool {
        let value = if cmp_on_next {
            iv_step.apply(iv, width)
        } else {
            iv
        };
        let (l, r) = if v_is_lhs {
            (value, bound_bits)
        } else {
            (bound_bits, value)
        };
        let cond_true = eval_icmp(&predicate, l, r, width);
        cond_true == continue_on
    };
    let mut iv = mask_to_width(start.masked(), width);
    let mut trip: u64;
    if !body_rpo.is_empty() {
        // Test before body.
        trip = 0;
        loop {
            if !eval_continue(iv) {
                break;
            }
            trip += 1;
            if trip > MAX_TRIP {
                return None;
            }
            iv = iv_step.apply(iv, width);
        }
    } else {
        // Body runs once, then tests.
        trip = 1;
        loop {
            if !eval_continue(iv) {
                break;
            }
            trip += 1;
            if trip > MAX_TRIP {
                return None;
            }
            iv = iv_step.apply(iv, width);
        }
    }
    if !(MIN_TRIP..=MAX_TRIP).contains(&trip) {
        return None;
    }
    // Growth of THIS unroll: trip × body ops (the per-body MAX_LOOP_OPS
    // cap alone no longer bounds it, see the constants).
    if trip.saturating_mul(loop_ops as u64) > MAX_UNROLL_GROWTH {
        return None;
    }

    Some(UnrollPlan {
        preheader,
        header,
        body_rpo,
        latch,
        loop_ops,
        cond_br,
        exit_dest,
        exit_operands,
        inloop_operands,
        latch_operands,
        init_operands,
        iv_index,
        iv_ty: start.ty,
        iv_width: width,
        iv_start: mask_to_width(start.masked(), width),
        iv_step,
        iv_update_op,
        trip,
    })
}

/// Validate the while-shape body sub-CFG `B = loop \ {header}` and
/// return it in reverse post-order from `entry`:
/// - single entry: every predecessor of a B block is inside the loop
///   (no side entries), and the header only enters at `entry` (its
///   cond_br has one in-loop edge — checked by the caller);
/// - single exit: every terminator is a `br`/`cond_br` whose successors
///   all stay inside the loop (no side exits, no returns), and the only
///   edge to the header is the latch's (a second one would be a second
///   latch — the caller checked there is exactly one);
/// - acyclic: a cycle inside B (not through the header) is an inner
///   loop — it must unroll first (the rounds loop re-discovers this
///   loop afterwards).
fn body_subgraph_rpo(
    ctx: &Context,
    natural_loop: &NaturalLoop,
    header: Ptr<BasicBlock>,
    latch: Ptr<BasicBlock>,
    entry: Ptr<BasicBlock>,
) -> Option<Vec<Ptr<BasicBlock>>> {
    let subgraph_size = natural_loop.body.len() - 1;
    for &block in &natural_loop.body {
        if block == header {
            continue;
        }
        let term = block.deref(ctx).get_terminator(ctx)?;
        let opid = Operation::get_opid(term, ctx);
        if opid != BrOp::get_opid_static() && opid != CondBrOp::get_opid_static() {
            return None;
        }
        for succ in block.deref(ctx).succs(ctx) {
            if !natural_loop.body.contains(&succ) {
                return None; // side exit
            }
            if succ == header && block != latch {
                return None; // second backedge (defensive; latches==1)
            }
        }
        for pred in block.preds(ctx) {
            if !natural_loop.body.contains(&pred) {
                return None; // side entry
            }
        }
    }

    // Iterative DFS from `entry` over B-internal edges: cycle check and
    // post-order in one pass (tri-state: 0 unseen, 1 on stack, 2 done).
    let mut state: FxHashMap<Ptr<BasicBlock>, u8> = FxHashMap::default();
    let mut post_order = Vec::new();
    let mut stack = vec![(entry, 0usize)];
    state.insert(entry, 1);
    while let Some((block, succ_idx)) = stack.pop() {
        let succs: Vec<_> = block
            .deref(ctx)
            .succs(ctx)
            .into_iter()
            .filter(|s| *s != header)
            .collect();
        if succ_idx < succs.len() {
            stack.push((block, succ_idx + 1));
            let succ = succs[succ_idx];
            match state.get(&succ).copied().unwrap_or(0) {
                0 => {
                    state.insert(succ, 1);
                    stack.push((succ, 0));
                }
                1 => return None, // back edge inside B: inner loop
                _ => {}
            }
        } else {
            state.insert(block, 2);
            post_order.push(block);
        }
    }
    if post_order.len() != subgraph_size {
        return None; // blocks unreachable from the entry
    }
    post_order.reverse();
    Some(post_order)
}

fn eval_icmp(predicate: &ICmpPredicateAttr, lhs: u128, rhs: u128, width: u32) -> bool {
    let (lm, rm) = (mask_to_width(lhs, width), mask_to_width(rhs, width));
    let (ls, rs) = (sign_extend(lhs, width), sign_extend(rhs, width));
    match predicate {
        ICmpPredicateAttr::EQ => lm == rm,
        ICmpPredicateAttr::NE => lm != rm,
        ICmpPredicateAttr::ULT => lm < rm,
        ICmpPredicateAttr::ULE => lm <= rm,
        ICmpPredicateAttr::UGT => lm > rm,
        ICmpPredicateAttr::UGE => lm >= rm,
        ICmpPredicateAttr::SLT => ls < rs,
        ICmpPredicateAttr::SLE => ls <= rs,
        ICmpPredicateAttr::SGT => ls > rs,
        ICmpPredicateAttr::SGE => ls >= rs,
    }
}

// ============================================================================
// Rewrite: clone trip iterations straight-line into the preheader
// ============================================================================

/// Strip the identity attributes the clone copied from the original —
/// a clone must not carry the original's dense `ll.op_id` — and stamp it
/// as DERIVED from the original's effective sources (1→N, like isel
/// expansion brackets; multi-parent not needed).
fn stamp_clone(ctx: &mut Context, clone: Ptr<Operation>, original: Ptr<Operation>) {
    let sources = opmap::effective_sources(ctx, original);
    {
        let mut clone_ref = clone.deref_mut(ctx);
        clone_ref
            .attributes
            .0
            .remove(&*opmap::ATTR_KEY_AARCH64_OP_ID);
        clone_ref
            .attributes
            .0
            .remove(&*opmap::ATTR_KEY_AARCH64_DERIVED_FROM);
        clone_ref
            .attributes
            .0
            .remove(&*opmap::ATTR_KEY_DERIVED_FROM_MANY);
        clone_ref
            .attributes
            .0
            .remove(&*opmap::ATTR_KEY_INLINED_FROM);
    }
    if !sources.is_empty() {
        opmap::set_derived_from_many(ctx, clone, sources);
    }
}

/// Non-terminator ops of `block`, in program order.
fn collect_nonterm_ops(ctx: &Context, block: Ptr<BasicBlock>) -> Vec<Ptr<Operation>> {
    let term = block
        .deref(ctx)
        .get_terminator(ctx)
        .expect("loop blocks have terminators");
    block
        .deref(ctx)
        .iter(ctx)
        .filter(|op| *op != term)
        .collect()
}

/// Delete the original loop: sever all def-use edges first (loop ops
/// reference each other and their own blocks), then erase ops and blocks.
fn delete_loop_blocks(ctx: &mut Context, loop_blocks: &FxHashSet<Ptr<BasicBlock>>) {
    for &block in loop_blocks {
        let ops: Vec<_> = block.deref(ctx).iter(ctx).collect();
        for op in ops {
            Operation::drop_all_uses(op, ctx);
        }
    }
    for &block in loop_blocks {
        let ops: Vec<_> = block.deref(ctx).iter(ctx).collect();
        for op in ops {
            Operation::erase(op, ctx);
        }
        while block.deref(ctx).get_num_arguments() > 0 {
            BasicBlock::pop_argument(block, ctx);
        }
        BasicBlock::erase(block, ctx);
    }
}

fn apply(ctx: &mut Context, plan: &UnrollPlan) {
    if plan.body_rpo.len() > 1 {
        apply_multi_block(ctx, plan);
    } else {
        apply_straight_line(ctx, plan);
    }
}

/// The v1 rewrite for the 0/1-body-block shapes: all clones straight-
/// lined into the preheader (no new blocks).
fn apply_straight_line(ctx: &mut Context, plan: &UnrollPlan) {
    let pre_term = plan
        .preheader
        .deref(ctx)
        .get_terminator(ctx)
        .expect("analyze verified the preheader terminator");

    let header_ops = collect_nonterm_ops(ctx, plan.header);
    let body_ops: Vec<Ptr<Operation>> = plan
        .body_rpo
        .first()
        .map(|b| collect_nonterm_ops(ctx, *b))
        .unwrap_or_default();

    let num_args = plan.header.deref(ctx).get_num_arguments();
    let header_args: Vec<Value> = (0..num_args)
        .map(|j| plan.header.deref(ctx).get_argument(j))
        .collect();
    let body_args: Vec<Value> = plan
        .body_rpo
        .first()
        .map(|b| {
            (0..b.deref(ctx).get_num_arguments())
                .map(|j| b.deref(ctx).get_argument(j))
                .collect()
        })
        .unwrap_or_default();

    let mut mapper = cloning::IrMapping::new();
    let mut rewriter = IRRewriter::<DummyListener>::default();
    let mut const_cache: FxHashMap<u128, Value> = FxHashMap::default();

    // Materialize the iv's per-iteration constant (cached per value).
    // ADJOINT: constants derive from the iv update op they obsolete.
    let iv_update_op = plan.iv_update_op;
    let mut const_for = |ctx: &mut Context, bits: u128| -> Value {
        if let Some(v) = const_cache.get(&bits) {
            return *v;
        }
        let apint = APInt::from_u128(
            bits,
            NonZero::new(plan.iv_width as usize).expect("analyze rejected width 0"),
        );
        let constant = ConstantOp::new(ctx, Box::new(IntegerAttr::new(plan.iv_ty, apint)));
        let const_op = constant.get_operation();
        const_op.insert_before(ctx, pre_term);
        stamp_clone(ctx, const_op, iv_update_op);
        let value = const_op.deref(ctx).get_result(0);
        const_cache.insert(bits, value);
        value
    };

    // Straight-line clone: while-shape emits header(0), body(0), ...,
    // header(trip-1), body(trip-1), header(trip) — the final header
    // execution is the one whose compare fails, and its ops may feed the
    // exit. Do-while emits the block trip times.
    let test_before_body = !plan.body_rpo.is_empty();
    let total_header_clones = if test_before_body {
        plan.trip + 1
    } else {
        plan.trip
    };
    let mut cur = plan.init_operands.clone();
    let mut iv_bits = plan.iv_start;

    for k in 0..total_header_clones {
        for j in 0..num_args {
            let incoming = if j == plan.iv_index && k > 0 {
                const_for(ctx, iv_bits)
            } else {
                cur[j]
            };
            mapper.map_value(header_args[j], incoming);
        }
        for &op in &header_ops {
            let clone = cloning::clone_operation(op, ctx, &mut rewriter, &mut mapper);
            clone.insert_before(ctx, pre_term);
            stamp_clone(ctx, clone, op);
        }
        if test_before_body {
            if k < plan.trip {
                for (arg, operand) in body_args.iter().zip(&plan.inloop_operands) {
                    let mapped = mapper.lookup_value_or_default(*operand);
                    mapper.map_value(*arg, mapped);
                }
                for &op in &body_ops {
                    let clone = cloning::clone_operation(op, ctx, &mut rewriter, &mut mapper);
                    clone.insert_before(ctx, pre_term);
                    stamp_clone(ctx, clone, op);
                }
                cur = plan
                    .latch_operands
                    .iter()
                    .map(|v| mapper.lookup_value_or_default(*v))
                    .collect();
                iv_bits = plan.iv_step.apply(iv_bits, plan.iv_width);
            }
        } else if k + 1 < plan.trip {
            cur = plan
                .latch_operands
                .iter()
                .map(|v| mapper.lookup_value_or_default(*v))
                .collect();
            iv_bits = plan.iv_step.apply(iv_bits, plan.iv_width);
        }
    }

    // Uses of loop-defined values from OUTSIDE the loop see the value of
    // the defining block's last execution — exactly the mapper's final
    // binding (clones live in the preheader, so they dominate every such
    // use).
    let loop_blocks: FxHashSet<Ptr<BasicBlock>> = plan
        .body_rpo
        .iter()
        .copied()
        .chain(std::iter::once(plan.header))
        .collect();
    let mut loop_defined: Vec<Value> = Vec::new();
    loop_defined.extend(header_args.iter().copied());
    loop_defined.extend(body_args.iter().copied());
    for &block in &loop_blocks {
        for op in block.deref(ctx).iter(ctx) {
            let results: Vec<Value> = op.deref(ctx).results().collect();
            loop_defined.extend(results);
        }
    }
    for value in loop_defined {
        let Some(final_value) = mapper.lookup_value(value) else {
            continue; // terminator "results" don't exist; args always mapped
        };
        value.replace_some_uses_with(
            ctx,
            |c, r#use| {
                r#use
                    .user_op()
                    .deref(c)
                    .get_container()
                    .map(|b| !loop_blocks.contains(&b))
                    .unwrap_or(true)
            },
            &final_value,
        );
    }

    // Retarget the preheader into the exit block with the exit edge's
    // operands mapped through the final iteration.
    // ADJOINT: the new branch derives from the cond_br it replaces.
    let exit_args: Vec<Value> = plan
        .exit_operands
        .iter()
        .map(|v| mapper.lookup_value_or_default(*v))
        .collect();
    let new_br = BrOp::new(ctx, plan.exit_dest, exit_args).get_operation();
    new_br.insert_before(ctx, pre_term);
    opmap::derive_new_from(ctx, new_br, plan.cond_br);
    Operation::erase(pre_term, ctx);

    delete_loop_blocks(ctx, &loop_blocks);
}

/// The multi-block-body rewrite: the body sub-CFG is cloned once per
/// iteration (fresh blocks, branch targets remapped within the copy),
/// and the copies are chained through continuation blocks `un<k>_..._head`
/// that carry the loop-carried header args (the iv arg is bypassed with
/// its per-iteration constant). Layout after the rewrite:
///
/// ```text
/// preheader:      ... ; iv consts ; br un0_head(init)
/// un0_head(args): header ops (iter 0) ; br un0_<entry>(inloop)
/// un0_<body...>:  body sub-CFG clone 0; latch clone: br un1_head(latch)
/// ...
/// un<trip>_head:  header ops (the failing test) ; br exit(exit operands)
/// ```
///
/// The original loop blocks are deleted. simplify-cfg later merges the
/// single-predecessor chain.
fn apply_multi_block(ctx: &mut Context, plan: &UnrollPlan) {
    let pre_term = plan
        .preheader
        .deref(ctx)
        .get_terminator(ctx)
        .expect("analyze verified the preheader terminator");

    let header_ops = collect_nonterm_ops(ctx, plan.header);
    let num_args = plan.header.deref(ctx).get_num_arguments();
    let header_args: Vec<Value> = (0..num_args)
        .map(|j| plan.header.deref(ctx).get_argument(j))
        .collect();
    let header_arg_types: Vec<TypeHandle> =
        header_args.iter().map(|arg| arg.get_type(ctx)).collect();

    // Per-body-block non-terminator ops and terminators, in RPO.
    let body_ops: Vec<Vec<Ptr<Operation>>> = plan
        .body_rpo
        .iter()
        .map(|b| collect_nonterm_ops(ctx, *b))
        .collect();
    let body_terms: Vec<Ptr<Operation>> = plan
        .body_rpo
        .iter()
        .map(|b| {
            b.deref(ctx)
                .get_terminator(ctx)
                .expect("loop blocks have terminators")
        })
        .collect();

    let mut mapper = cloning::IrMapping::new();
    let mut rewriter = IRRewriter::<DummyListener>::default();
    let mut const_cache: FxHashMap<u128, Value> = FxHashMap::default();

    // Materialize the iv's per-iteration constant in the PREHEADER
    // (before its still-live terminator — it dominates every new block).
    // ADJOINT: constants derive from the iv update op they obsolete.
    let iv_update_op = plan.iv_update_op;
    let mut const_for = |ctx: &mut Context, bits: u128| -> Value {
        if let Some(v) = const_cache.get(&bits) {
            return *v;
        }
        let apint = APInt::from_u128(
            bits,
            NonZero::new(plan.iv_width as usize).expect("analyze rejected width 0"),
        );
        let constant = ConstantOp::new(ctx, Box::new(IntegerAttr::new(plan.iv_ty, apint)));
        let const_op = constant.get_operation();
        const_op.insert_before(ctx, pre_term);
        stamp_clone(ctx, const_op, iv_update_op);
        let value = const_op.deref(ctx).get_result(0);
        const_cache.insert(bits, value);
        value
    };

    let head_base = plan.header;
    let mut insert_after = plan.preheader;
    let mut new_named_block =
        |ctx: &mut Context, k: u64, base: Ptr<BasicBlock>, suffix: &str, types: Vec<TypeHandle>| {
            let name = base
                .deref(ctx)
                .given_name(ctx)
                .map(|n| n.to_string())
                .unwrap_or_else(|| "bb".to_string());
            let ident = Identifier::try_from(format!("un{k}_{name}{suffix}")).ok();
            let block = BasicBlock::new(ctx, ident, types);
            block.insert_after(ctx, insert_after);
            insert_after = block;
            block
        };

    let first_cont = new_named_block(ctx, 0, head_base, "_head", header_arg_types.clone());
    let mut cont = first_cont;
    let mut iv_bits = plan.iv_start;

    for k in 0..=plan.trip {
        // Header args for iteration k: the continuation block's args,
        // with the iv bypassed by its simulated constant (iteration 0's
        // init operand IS that constant — analyze proved it).
        for (j, header_arg) in header_args.iter().enumerate() {
            let incoming = if j == plan.iv_index {
                const_for(ctx, iv_bits)
            } else {
                cont.deref(ctx).get_argument(j)
            };
            mapper.map_value(*header_arg, incoming);
        }
        for &op in &header_ops {
            let clone = cloning::clone_operation(op, ctx, &mut rewriter, &mut mapper);
            clone.insert_at_back(cont, ctx);
            stamp_clone(ctx, clone, op);
        }
        if k == plan.trip {
            // The final (failing) header evaluation: branch to the exit
            // with the exit edge's operands mapped through it.
            // ADJOINT: the branch derives from the cond_br it replaces.
            let exit_args: Vec<Value> = plan
                .exit_operands
                .iter()
                .map(|v| mapper.lookup_value_or_default(*v))
                .collect();
            let exit_br = BrOp::new(ctx, plan.exit_dest, exit_args).get_operation();
            exit_br.insert_at_back(cont, ctx);
            opmap::derive_new_from(ctx, exit_br, plan.cond_br);
            break;
        }

        // Fresh blocks for this iteration's body sub-CFG (created before
        // any op is cloned, so branch targets can be remapped), plus the
        // next continuation block.
        let block_clones: Vec<Ptr<BasicBlock>> = plan
            .body_rpo
            .iter()
            .map(|&old_block| {
                let types: Vec<TypeHandle> = (0..old_block.deref(ctx).get_num_arguments())
                    .map(|j| old_block.deref(ctx).get_argument(j).get_type(ctx))
                    .collect();
                let new_block = new_named_block(ctx, k, old_block, "", types);
                mapper.map_block(old_block, new_block);
                for j in 0..old_block.deref(ctx).get_num_arguments() {
                    mapper.map_value(
                        old_block.deref(ctx).get_argument(j),
                        new_block.deref(ctx).get_argument(j),
                    );
                }
                new_block
            })
            .collect();
        let next_cont = new_named_block(ctx, k + 1, head_base, "_head", header_arg_types.clone());

        // The taken in-loop edge: cont → this iteration's body entry.
        // ADJOINT: derives from the cond_br whose decision it encodes.
        let inloop_args: Vec<Value> = plan
            .inloop_operands
            .iter()
            .map(|v| mapper.lookup_value_or_default(*v))
            .collect();
        let entry_br = BrOp::new(ctx, block_clones[0], inloop_args).get_operation();
        entry_br.insert_at_back(cont, ctx);
        opmap::derive_new_from(ctx, entry_br, plan.cond_br);

        // Clone the body in RPO (a topological order of the acyclic
        // sub-CFG, so every SSA def is cloned before its uses).
        for (idx, (&old_block, &new_block)) in
            plan.body_rpo.iter().zip(&block_clones).enumerate()
        {
            for &op in &body_ops[idx] {
                let clone = cloning::clone_operation(op, ctx, &mut rewriter, &mut mapper);
                clone.insert_at_back(new_block, ctx);
                stamp_clone(ctx, clone, op);
            }
            let term = body_terms[idx];
            if old_block == plan.latch {
                // The backedge becomes the edge into the next iteration's
                // continuation, carrying the loop-carried values.
                let latch_args: Vec<Value> = plan
                    .latch_operands
                    .iter()
                    .map(|v| mapper.lookup_value_or_default(*v))
                    .collect();
                let latch_br = BrOp::new(ctx, next_cont, latch_args).get_operation();
                latch_br.insert_at_back(new_block, ctx);
                opmap::derive_new_from(ctx, latch_br, term);
            } else {
                // Internal branch: successors are body blocks of this
                // iteration, remapped through the block map.
                let clone = cloning::clone_operation(term, ctx, &mut rewriter, &mut mapper);
                clone.insert_at_back(new_block, ctx);
                stamp_clone(ctx, clone, term);
            }
        }

        iv_bits = plan.iv_step.apply(iv_bits, plan.iv_width);
        cont = next_cont;
    }

    // Uses of HEADER-defined values from outside the loop see the final
    // header evaluation's clone (it lives in the tail continuation,
    // which dominates every path that previously left through the
    // header's exit edge). Body-defined values cannot be used outside
    // the loop in verified while-shape IR (the body does not dominate
    // the header's exit edge), so they need no rewiring.
    let loop_blocks: FxHashSet<Ptr<BasicBlock>> = plan
        .body_rpo
        .iter()
        .copied()
        .chain(std::iter::once(plan.header))
        .collect();
    let mut header_defined: Vec<Value> = header_args.clone();
    for &op in &header_ops {
        header_defined.extend(op.deref(ctx).results());
    }
    for value in header_defined {
        let Some(final_value) = mapper.lookup_value(value) else {
            continue;
        };
        value.replace_some_uses_with(
            ctx,
            |c, r#use| {
                r#use
                    .user_op()
                    .deref(c)
                    .get_container()
                    .map(|b| !loop_blocks.contains(&b))
                    .unwrap_or(true)
            },
            &final_value,
        );
    }

    // Retarget the preheader into the first continuation block and
    // delete the original loop.
    let pre_br = BrOp::new(ctx, first_cont, plan.init_operands.clone()).get_operation();
    pre_br.insert_before(ctx, pre_term);
    opmap::derive_new_from(ctx, pre_br, pre_term);
    Operation::erase(pre_term, ctx);

    delete_loop_blocks(ctx, &loop_blocks);
}

#[cfg(test)]
mod tests {
    use super::*;
    use pliron::builtin::op_interfaces::OneResultInterface as _;
    use pliron_llvm::op_interfaces::IntBinArithOpWithOverflowFlag as _;

    use crate::{
        dialects::builtin::op_interfaces::SingleBlockRegionInterface as _,
        dialects::{
            builtin::{
                attributes::IntegerAttr,
                ops::{ConstantOp, ModuleOp},
                types::Signedness,
            },
            llvm::{
                attributes::LinkageAttr,
                ops::{FuncOp, ReturnOp},
                types::FuncType,
            },
        },
        ir::r#type::TypeHandle,
        printable::Printable,
    };

    fn int_ty(ctx: &mut Context, width: u32) -> TypedHandle<IntegerType> {
        IntegerType::get(ctx, width, Signedness::Signless)
    }

    fn const_i64(ctx: &mut Context, block: Ptr<BasicBlock>, value: u64) -> Value {
        let ty = int_ty(ctx, 64);
        let c = ConstantOp::new(
            ctx,
            Box::new(IntegerAttr::new(
                ty,
                APInt::from_u64(value, NonZero::new(64).unwrap()),
            )),
        );
        c.get_operation().insert_at_back(block, ctx);
        c.get_result(ctx)
    }

    /// Bound for the loop: `Const(n)` or the function's i64 argument.
    enum Bound {
        Const(u64),
        Arg,
    }

    /// Canonical rustc `for iv in 0..bound { acc += iv }` while-shape:
    /// entry(preheader): consts; br header(0, 0)
    /// header(iv, acc):  cond = icmp slt iv, bound; cond_br cond, body, exit(acc)
    /// body:             acc2 = acc + iv; iv2 = iv + 1; br header(iv2, acc2)
    /// exit(res):        return res
    fn sum_loop(ctx: &mut Context, bound: Bound) -> (FuncOp, Ptr<BasicBlock>) {
        let i64_ty: TypeHandle = int_ty(ctx, 64).into();
        let fn_ty = FuncType::get(ctx, i64_ty, vec![i64_ty], false);
        let func = FuncOp::new(ctx, "sum".try_into().unwrap(), fn_ty);
        func.set_attr_llvm_function_linkage(ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(ctx);
        let region = func.get_region(ctx).unwrap();
        let entry = func.get_entry_block(ctx).unwrap();
        let arg = entry.deref(ctx).get_argument(0);
        let header = BasicBlock::new(ctx, None, vec![i64_ty, i64_ty]);
        header.insert_at_back(region, ctx);
        let body = BasicBlock::new(ctx, None, vec![]);
        body.insert_at_back(region, ctx);
        let exit = BasicBlock::new(ctx, None, vec![i64_ty]);
        exit.insert_at_back(region, ctx);

        let zero = const_i64(ctx, entry, 0);
        let one = const_i64(ctx, entry, 1);
        let bound_v = match bound {
            Bound::Const(n) => const_i64(ctx, entry, n),
            Bound::Arg => arg,
        };
        BrOp::new(ctx, header, vec![zero, zero])
            .get_operation()
            .insert_at_back(entry, ctx);

        let iv = header.deref(ctx).get_argument(0);
        let acc = header.deref(ctx).get_argument(1);
        let cond = ICmpOp::new(ctx, ICmpPredicateAttr::SLT, iv, bound_v);
        cond.get_operation().insert_at_back(header, ctx);
        let cond_v = cond.get_result(ctx);
        CondBrOp::new(ctx, cond_v, body, vec![], exit, vec![acc])
            .get_operation()
            .insert_at_back(header, ctx);

        let acc2 = AddOp::new_with_overflow_flag(ctx, acc, iv, Default::default());
        acc2.get_operation().insert_at_back(body, ctx);
        let acc2_v = acc2.get_result(ctx);
        let iv2 = AddOp::new_with_overflow_flag(ctx, iv, one, Default::default());
        iv2.get_operation().insert_at_back(body, ctx);
        let iv2_v = iv2.get_result(ctx);
        BrOp::new(ctx, header, vec![iv2_v, acc2_v])
            .get_operation()
            .insert_at_back(body, ctx);

        let res = exit.deref(ctx).get_argument(0);
        ReturnOp::new(ctx, Some(res))
            .get_operation()
            .insert_at_back(exit, ctx);
        (func, exit)
    }

    fn run_unroll(ctx: &mut Context, func: FuncOp) -> String {
        LLVMUnrollPass::new(&TargetProfile::host_cpu())
            .run(func.get_operation(), ctx, &mut AnalysisManager::default())
            .unwrap();
        format!("{}", func.get_operation().disp(ctx))
    }

    #[test]
    fn unrolls_exact_trip_and_chains_accumulator() {
        let mut ctx = Context::new();
        let (func, _exit) = sum_loop(&mut ctx, Bound::Const(4));
        let text = run_unroll(&mut ctx, func);
        // The loop control is gone and the body was cloned trip=4 times:
        // 4 accumulator adds chained + 4 (now-dead) iv adds.
        assert!(!text.contains("llvm.cond_br"), "{text}");
        assert_eq!(text.matches("llvm.add").count(), 8, "{text}");
        assert_eq!(text.matches("llvm.icmp").count(), 5, "{text}");

        // Constant-fold check: simplify + simplify-cfg reduce the chained
        // accumulator to the constant 0+1+2+3 = 6 at the return.
        use super::super::{simplify::LLVMSimplifyPass, simplify_cfg::LLVMSimplifyCfgPass};
        LLVMSimplifyPass
            .run(
                func.get_operation(),
                &mut ctx,
                &mut AnalysisManager::default(),
            )
            .unwrap();
        LLVMSimplifyCfgPass
            .run(
                func.get_operation(),
                &mut ctx,
                &mut AnalysisManager::default(),
            )
            .unwrap();
        LLVMSimplifyPass
            .run(
                func.get_operation(),
                &mut ctx,
                &mut AnalysisManager::default(),
            )
            .unwrap();
        let region = func.get_region(&ctx).unwrap();
        let mut ret_const = None;
        for block in region.deref(&ctx).iter(&ctx) {
            let Some(term) = block.deref(&ctx).get_terminator(&ctx) else {
                continue;
            };
            if Operation::get_opid(term, &ctx) == ReturnOp::get_opid_static() {
                let operand = term.deref(&ctx).get_operand(0);
                ret_const = as_const_operand(&ctx, operand).map(|c| c.masked());
            }
        }
        assert_eq!(ret_const, Some(6), "return must fold to 0+1+2+3");
    }

    /// Do-while shape (header == latch): body runs once before the first
    /// test; the compare is on the NEXT iv value.
    #[test]
    fn unrolls_single_block_do_while_loop() {
        let mut ctx = Context::new();
        let i64_ty: TypeHandle = int_ty(&mut ctx, 64).into();
        let fn_ty = FuncType::get(&mut ctx, i64_ty, vec![i64_ty], false);
        let func = FuncOp::new(&mut ctx, "dw".try_into().unwrap(), fn_ty);
        func.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(&mut ctx);
        let region = func.get_region(&ctx).unwrap();
        let entry = func.get_entry_block(&ctx).unwrap();
        let header = BasicBlock::new(&mut ctx, None, vec![i64_ty, i64_ty]);
        header.insert_at_back(region, &ctx);
        let exit = BasicBlock::new(&mut ctx, None, vec![i64_ty]);
        exit.insert_at_back(region, &ctx);

        let zero = const_i64(&mut ctx, entry, 0);
        let one = const_i64(&mut ctx, entry, 1);
        let four = const_i64(&mut ctx, entry, 4);
        BrOp::new(&mut ctx, header, vec![zero, zero])
            .get_operation()
            .insert_at_back(entry, &ctx);

        let iv = header.deref(&ctx).get_argument(0);
        let acc = header.deref(&ctx).get_argument(1);
        let acc2 = AddOp::new_with_overflow_flag(&mut ctx, acc, iv, Default::default());
        acc2.get_operation().insert_at_back(header, &ctx);
        let acc2_v = acc2.get_result(&ctx);
        let iv2 = AddOp::new_with_overflow_flag(&mut ctx, iv, one, Default::default());
        iv2.get_operation().insert_at_back(header, &ctx);
        let iv2_v = iv2.get_result(&ctx);
        let cond = ICmpOp::new(&mut ctx, ICmpPredicateAttr::SLT, iv2_v, four);
        cond.get_operation().insert_at_back(header, &ctx);
        let cond_v = cond.get_result(&ctx);
        CondBrOp::new(
            &mut ctx,
            cond_v,
            header,
            vec![iv2_v, acc2_v],
            exit,
            vec![acc2_v],
        )
        .get_operation()
        .insert_at_back(header, &ctx);

        let res = exit.deref(&ctx).get_argument(0);
        ReturnOp::new(&mut ctx, Some(res))
            .get_operation()
            .insert_at_back(exit, &ctx);

        let text = run_unroll(&mut ctx, func);
        // trip = 4 (iv 0,1,2,3): the loop is gone, 4 accumulator adds +
        // 4 iv adds cloned.
        assert!(!text.contains("llvm.cond_br"), "{text}");
        assert_eq!(text.matches("llvm.add").count(), 8, "{text}");

        use super::super::simplify::LLVMSimplifyPass;
        use super::super::simplify_cfg::LLVMSimplifyCfgPass;
        LLVMSimplifyPass
            .run(
                func.get_operation(),
                &mut ctx,
                &mut AnalysisManager::default(),
            )
            .unwrap();
        LLVMSimplifyCfgPass
            .run(
                func.get_operation(),
                &mut ctx,
                &mut AnalysisManager::default(),
            )
            .unwrap();
        LLVMSimplifyPass
            .run(
                func.get_operation(),
                &mut ctx,
                &mut AnalysisManager::default(),
            )
            .unwrap();
        let region = func.get_region(&ctx).unwrap();
        let mut ret_const = None;
        for block in region.deref(&ctx).iter(&ctx) {
            let Some(term) = block.deref(&ctx).get_terminator(&ctx) else {
                continue;
            };
            if Operation::get_opid(term, &ctx) == ReturnOp::get_opid_static() {
                let operand = term.deref(&ctx).get_operand(0);
                ret_const = as_const_operand(&ctx, operand).map(|c| c.masked());
            }
        }
        assert_eq!(ret_const, Some(6), "return must fold to 0+1+2+3");
    }

    /// Start value for the geometric loops: `Const(n)` or the function's
    /// i64 argument.
    enum GeoStart {
        Const(u64),
        Arg,
    }

    /// Which backedge update op the geometric loop uses.
    enum GeoKind {
        LShr,
        Shl,
        UDiv,
    }

    /// Geometric-induction while-shape, mirroring [sum_loop]:
    /// entry(preheader): consts; br header(start, 0)
    /// header(iv, acc):  cond = icmp <pred> iv, bound; cond_br cond, body, exit(acc)
    /// body:             acc2 = acc + iv; iv2 = <kind>(iv, amount); br header(iv2, acc2)
    /// exit(res):        return res
    fn geo_loop(
        ctx: &mut Context,
        start: GeoStart,
        kind: GeoKind,
        amount: u64,
        pred: ICmpPredicateAttr,
        bound: u64,
    ) -> FuncOp {
        use pliron_llvm::op_interfaces::BinArithOp as _;

        let i64_ty: TypeHandle = int_ty(ctx, 64).into();
        let fn_ty = FuncType::get(ctx, i64_ty, vec![i64_ty], false);
        let func = FuncOp::new(ctx, "geo".try_into().unwrap(), fn_ty);
        func.set_attr_llvm_function_linkage(ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(ctx);
        let region = func.get_region(ctx).unwrap();
        let entry = func.get_entry_block(ctx).unwrap();
        let arg = entry.deref(ctx).get_argument(0);
        let header = BasicBlock::new(ctx, None, vec![i64_ty, i64_ty]);
        header.insert_at_back(region, ctx);
        let body = BasicBlock::new(ctx, None, vec![]);
        body.insert_at_back(region, ctx);
        let exit = BasicBlock::new(ctx, None, vec![i64_ty]);
        exit.insert_at_back(region, ctx);

        let zero = const_i64(ctx, entry, 0);
        let amount_v = const_i64(ctx, entry, amount);
        let bound_v = const_i64(ctx, entry, bound);
        let start_v = match start {
            GeoStart::Const(n) => const_i64(ctx, entry, n),
            GeoStart::Arg => arg,
        };
        BrOp::new(ctx, header, vec![start_v, zero])
            .get_operation()
            .insert_at_back(entry, ctx);

        let iv = header.deref(ctx).get_argument(0);
        let acc = header.deref(ctx).get_argument(1);
        let cond = ICmpOp::new(ctx, pred, iv, bound_v);
        cond.get_operation().insert_at_back(header, ctx);
        let cond_v = cond.get_result(ctx);
        CondBrOp::new(ctx, cond_v, body, vec![], exit, vec![acc])
            .get_operation()
            .insert_at_back(header, ctx);

        let acc2 = AddOp::new_with_overflow_flag(ctx, acc, iv, Default::default());
        acc2.get_operation().insert_at_back(body, ctx);
        let acc2_v = acc2.get_result(ctx);
        let iv2_v = match kind {
            GeoKind::LShr => {
                let op = LShrOp::new(ctx, iv, amount_v);
                op.get_operation().insert_at_back(body, ctx);
                op.get_result(ctx)
            }
            GeoKind::Shl => {
                let op = ShlOp::new_with_overflow_flag(ctx, iv, amount_v, Default::default());
                op.get_operation().insert_at_back(body, ctx);
                op.get_result(ctx)
            }
            GeoKind::UDiv => {
                let op = UDivOp::new(ctx, iv, amount_v);
                op.get_operation().insert_at_back(body, ctx);
                op.get_result(ctx)
            }
        };
        BrOp::new(ctx, header, vec![iv2_v, acc2_v])
            .get_operation()
            .insert_at_back(body, ctx);

        let res = exit.deref(ctx).get_argument(0);
        ReturnOp::new(ctx, Some(res))
            .get_operation()
            .insert_at_back(exit, ctx);
        func
    }

    /// Fold the accumulator through simplify → simplify-cfg → simplify
    /// and return the constant at the (single) return, if any.
    fn folded_return(ctx: &mut Context, func: FuncOp) -> Option<u128> {
        use super::super::{simplify::LLVMSimplifyPass, simplify_cfg::LLVMSimplifyCfgPass};
        LLVMSimplifyPass
            .run(func.get_operation(), ctx, &mut AnalysisManager::default())
            .unwrap();
        LLVMSimplifyCfgPass
            .run(func.get_operation(), ctx, &mut AnalysisManager::default())
            .unwrap();
        LLVMSimplifyPass
            .run(func.get_operation(), ctx, &mut AnalysisManager::default())
            .unwrap();
        let region = func.get_region(ctx).unwrap();
        let mut ret_const = None;
        for block in region.deref(ctx).iter(ctx) {
            let Some(term) = block.deref(ctx).get_terminator(ctx) else {
                continue;
            };
            if Operation::get_opid(term, ctx) == ReturnOp::get_opid_static() {
                let operand = term.deref(ctx).get_operand(0);
                ret_const = as_const_operand(ctx, operand).map(|c| c.masked());
            }
        }
        ret_const
    }

    /// The warp-reduction shape the GPU corpus loses on:
    /// `off = 16; while off > 0 { acc += off; off >>= 1 }` — 5 iterations
    /// with iv constants 16, 8, 4, 2, 1 (geometric, NOT an additive
    /// sequence), fully straight-lined.
    #[test]
    fn unrolls_shfl_reduction_lshr_shape() {
        let mut ctx = Context::new();
        let func = geo_loop(
            &mut ctx,
            GeoStart::Const(16),
            GeoKind::LShr,
            1,
            ICmpPredicateAttr::SGT,
            0,
        );
        let text = run_unroll(&mut ctx, func);
        // trip = 5: the loop control is gone; 5 acc adds + 5 (now-dead)
        // iv lshrs cloned; trip+1 = 6 header icmp clones.
        assert!(!text.contains("llvm.cond_br"), "{text}");
        assert_eq!(text.matches("llvm.add").count(), 5, "{text}");
        assert_eq!(text.matches("llvm.lshr").count(), 5, "{text}");
        assert_eq!(text.matches("llvm.icmp").count(), 6, "{text}");

        // Per-clone iv constants must be the geometric sequence: the
        // accumulator folds to 16+8+4+2+1 = 31 (any additive
        // misinterpretation folds to a different constant).
        assert_eq!(folded_return(&mut ctx, func), Some(31));
    }

    /// Growth direction: `iv = 1; while iv < 16 { acc += iv; iv <<= 1 }`
    /// — iv constants 1, 2, 4, 8; trip 4.
    #[test]
    fn unrolls_shl_growth_shape() {
        let mut ctx = Context::new();
        let func = geo_loop(
            &mut ctx,
            GeoStart::Const(1),
            GeoKind::Shl,
            1,
            ICmpPredicateAttr::SLT,
            16,
        );
        let text = run_unroll(&mut ctx, func);
        assert!(!text.contains("llvm.cond_br"), "{text}");
        assert_eq!(text.matches("llvm.shl").count(), 4, "{text}");
        assert_eq!(folded_return(&mut ctx, func), Some(1 + 2 + 4 + 8));
    }

    /// `udiv` by a power of two is exact division — simulated like the
    /// shifts: 16, 4, 1 (16/4=4, 4/4=1, 1/4=0); trip 3.
    #[test]
    fn unrolls_udiv_pow2_shape() {
        let mut ctx = Context::new();
        let func = geo_loop(
            &mut ctx,
            GeoStart::Const(16),
            GeoKind::UDiv,
            4,
            ICmpPredicateAttr::SGT,
            0,
        );
        let text = run_unroll(&mut ctx, func);
        assert!(!text.contains("llvm.cond_br"), "{text}");
        assert_eq!(text.matches("llvm.udiv").count(), 3, "{text}");
        assert_eq!(folded_return(&mut ctx, func), Some(16 + 4 + 1));
    }

    /// A non-power-of-2 udiv divisor must bail (rounding makes the
    /// "geometric" sequence irregular; out of the v1 contract).
    #[test]
    fn skips_non_power_of_2_udiv_step() {
        let mut ctx = Context::new();
        let func = geo_loop(
            &mut ctx,
            GeoStart::Const(27),
            GeoKind::UDiv,
            3,
            ICmpPredicateAttr::SGT,
            0,
        );
        let text = run_unroll(&mut ctx, func);
        assert!(text.contains("llvm.cond_br"), "{text}");
        assert_eq!(text.matches("llvm.udiv").count(), 1, "{text}");
    }

    /// Non-constant start bails exactly like the additive path.
    #[test]
    fn skips_non_constant_start_geometric() {
        let mut ctx = Context::new();
        let func = geo_loop(
            &mut ctx,
            GeoStart::Arg,
            GeoKind::LShr,
            1,
            ICmpPredicateAttr::SGT,
            0,
        );
        let text = run_unroll(&mut ctx, func);
        assert!(text.contains("llvm.cond_br"), "{text}");
        assert_eq!(text.matches("llvm.lshr").count(), 1, "{text}");
    }

    /// The MAX_TRIP bound holds for geometric induction too: 1<<40
    /// halves 41 times before hitting 0 — over the bound, no unroll.
    #[test]
    fn skips_geometric_trip_too_large() {
        let mut ctx = Context::new();
        let func = geo_loop(
            &mut ctx,
            GeoStart::Const(1u64 << 40),
            GeoKind::LShr,
            1,
            ICmpPredicateAttr::SGT,
            0,
        );
        let text = run_unroll(&mut ctx, func);
        assert!(text.contains("llvm.cond_br"), "{text}");
        assert_eq!(text.matches("llvm.lshr").count(), 1, "{text}");
    }

    /// Attribution on geometric clones: same rules as the additive path
    /// (unique ll.op_id, clones carry derived_from).
    #[test]
    fn stamps_attribution_on_geometric_clones() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "m".try_into().unwrap());
        let module_block = module.get_body(&ctx, 0);
        let func = geo_loop(
            &mut ctx,
            GeoStart::Const(16),
            GeoKind::LShr,
            1,
            ICmpPredicateAttr::SGT,
            0,
        );
        func.get_operation().insert_at_back(module_block, &ctx);
        opmap::assign_op_ids(&mut ctx, module.get_operation()).unwrap();

        LLVMUnrollPass::new(&TargetProfile::host_cpu())
            .run(
                module.get_operation(),
                &mut ctx,
                &mut AnalysisManager::default(),
            )
            .unwrap();

        let region = func.get_region(&ctx).unwrap();
        let mut seen_ids = FxHashSet::default();
        let mut derived_clones = 0usize;
        for block in region.deref(&ctx).iter(&ctx) {
            for op in block.deref(&ctx).iter(&ctx) {
                assert!(
                    opmap::has_attribution(&ctx, op),
                    "unattributed op: {}",
                    op.disp(&ctx)
                );
                if let Some(id) = opmap::op_id(&ctx, op) {
                    assert!(seen_ids.insert(id), "duplicate ll.op_id {id}");
                } else {
                    assert!(
                        !opmap::effective_sources(&ctx, op).is_empty(),
                        "clone without derived_from: {}",
                        op.disp(&ctx)
                    );
                    derived_clones += 1;
                }
            }
        }
        assert!(
            derived_clones > 0,
            "unroll must have produced derived clones"
        );
    }

    // ========================================================================
    // Multi-block bodies (v2): diamonds, nesting, bails
    // ========================================================================

    /// Build the canonical diamond-body while-loop:
    /// entry(preheader): consts; br header(0, 0)
    /// header(iv, acc):  c = icmp slt iv, bound; cond_br c, bentry, exit(acc)
    /// bentry:           p = and iv, 1; pc = icmp eq p, 0; cond_br pc, even, odd
    /// even:             e = acc + (iv + iv); br join(e)
    /// odd:              o = acc + iv;        br join(o)
    /// join(acc2):       iv2 = iv + 1; br header(iv2, acc2)   [latch]
    /// exit(res):        return res
    ///
    /// `odd_breaks` replaces odd's `br join` with `br exit(acc)` — a side
    /// exit out of the body. `filler` appends that many extra (dead) adds
    /// to the even arm.
    fn diamond_loop(ctx: &mut Context, bound: u64, odd_breaks: bool, filler: usize) -> FuncOp {
        use crate::dialects::llvm::ops::AndOp;
        use pliron_llvm::op_interfaces::BinArithOp as _;

        let i64_ty: TypeHandle = int_ty(ctx, 64).into();
        let fn_ty = FuncType::get(ctx, i64_ty, vec![i64_ty], false);
        let func = FuncOp::new(ctx, "diamond".try_into().unwrap(), fn_ty);
        func.set_attr_llvm_function_linkage(ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(ctx);
        let region = func.get_region(ctx).unwrap();
        let entry = func.get_entry_block(ctx).unwrap();
        let header = BasicBlock::new(ctx, None, vec![i64_ty, i64_ty]);
        header.insert_at_back(region, ctx);
        let bentry = BasicBlock::new(ctx, None, vec![]);
        bentry.insert_at_back(region, ctx);
        let even = BasicBlock::new(ctx, None, vec![]);
        even.insert_at_back(region, ctx);
        let odd = BasicBlock::new(ctx, None, vec![]);
        odd.insert_at_back(region, ctx);
        let join = BasicBlock::new(ctx, None, vec![i64_ty]);
        join.insert_at_back(region, ctx);
        let exit = BasicBlock::new(ctx, None, vec![i64_ty]);
        exit.insert_at_back(region, ctx);

        let zero = const_i64(ctx, entry, 0);
        let one = const_i64(ctx, entry, 1);
        let bound_v = const_i64(ctx, entry, bound);
        BrOp::new(ctx, header, vec![zero, zero])
            .get_operation()
            .insert_at_back(entry, ctx);

        let iv = header.deref(ctx).get_argument(0);
        let acc = header.deref(ctx).get_argument(1);
        let cond = ICmpOp::new(ctx, ICmpPredicateAttr::SLT, iv, bound_v);
        cond.get_operation().insert_at_back(header, ctx);
        let cond_v = cond.get_result(ctx);
        CondBrOp::new(ctx, cond_v, bentry, vec![], exit, vec![acc])
            .get_operation()
            .insert_at_back(header, ctx);

        let parity = AndOp::new(ctx, iv, one);
        parity.get_operation().insert_at_back(bentry, ctx);
        let pc = ICmpOp::new(ctx, ICmpPredicateAttr::EQ, parity.get_result(ctx), zero);
        pc.get_operation().insert_at_back(bentry, ctx);
        let pc_v = pc.get_result(ctx);
        CondBrOp::new(ctx, pc_v, even, vec![], odd, vec![])
            .get_operation()
            .insert_at_back(bentry, ctx);

        let dbl = AddOp::new_with_overflow_flag(ctx, iv, iv, Default::default());
        dbl.get_operation().insert_at_back(even, ctx);
        let e = AddOp::new_with_overflow_flag(ctx, acc, dbl.get_result(ctx), Default::default());
        e.get_operation().insert_at_back(even, ctx);
        let mut fill = e.get_result(ctx);
        for _ in 0..filler {
            let f = AddOp::new_with_overflow_flag(ctx, fill, one, Default::default());
            f.get_operation().insert_at_back(even, ctx);
            fill = f.get_result(ctx);
        }
        BrOp::new(ctx, join, vec![e.get_result(ctx)])
            .get_operation()
            .insert_at_back(even, ctx);

        let o = AddOp::new_with_overflow_flag(ctx, acc, iv, Default::default());
        o.get_operation().insert_at_back(odd, ctx);
        if odd_breaks {
            BrOp::new(ctx, exit, vec![acc])
                .get_operation()
                .insert_at_back(odd, ctx);
        } else {
            BrOp::new(ctx, join, vec![o.get_result(ctx)])
                .get_operation()
                .insert_at_back(odd, ctx);
        }

        let acc2 = join.deref(ctx).get_argument(0);
        let iv2 = AddOp::new_with_overflow_flag(ctx, iv, one, Default::default());
        iv2.get_operation().insert_at_back(join, ctx);
        BrOp::new(ctx, header, vec![iv2.get_result(ctx), acc2])
            .get_operation()
            .insert_at_back(join, ctx);

        let res = exit.deref(ctx).get_argument(0);
        ReturnOp::new(ctx, Some(res))
            .get_operation()
            .insert_at_back(exit, ctx);
        func
    }

    /// The iq3 shape in miniature: a constant-trip loop whose BODY holds
    /// a sign-select-style diamond. Fully unrolls (one diamond clone per
    /// iteration, no backedge) and constant-folds to
    /// (0+0)+(1)+(2+2)+(3) = 8.
    #[test]
    fn unrolls_diamond_body_and_folds() {
        let mut ctx = Context::new();
        let func = diamond_loop(&mut ctx, 4, false, 0);
        let text = run_unroll(&mut ctx, func);
        // The loop control cond_br is gone; the 4 diamond clones'
        // cond_brs remain until simplify folds their constant parities.
        // icmps: trip+1 = 5 header-compare clones + 4 parity compares.
        assert_eq!(text.matches("llvm.cond_br").count(), 4, "{text}");
        assert_eq!(text.matches("llvm.icmp").count(), 9, "{text}");
        assert_eq!(folded_return(&mut ctx, func), Some(8));
    }

    /// A side exit out of the body (odd arm breaks straight to the exit)
    /// must bail: the loop keeps both cond_brs.
    #[test]
    fn side_exit_in_body_bails() {
        let mut ctx = Context::new();
        let func = diamond_loop(&mut ctx, 4, true, 0);
        let text = run_unroll(&mut ctx, func);
        assert_eq!(text.matches("llvm.cond_br").count(), 2, "{text}");
    }

    /// Trip bound applies to multi-block bodies too.
    #[test]
    fn diamond_trip_too_large_bails() {
        let mut ctx = Context::new();
        let func = diamond_loop(&mut ctx, MAX_TRIP + 1, false, 0);
        let text = run_unroll(&mut ctx, func);
        assert_eq!(text.matches("llvm.cond_br").count(), 2, "{text}");
    }

    /// trip × body ops over [MAX_UNROLL_GROWTH] bails even though both
    /// individual bounds pass (32 trips × ~135 ops > 4096).
    #[test]
    fn unroll_growth_product_bails() {
        let mut ctx = Context::new();
        let func = diamond_loop(&mut ctx, 32, false, 128);
        let text = run_unroll(&mut ctx, func);
        assert_eq!(text.matches("llvm.cond_br").count(), 2, "{text}");
        // Control: the same body at trip 4 is inside the product bound
        // and unrolls (proving the bail above was the product, not the
        // body size).
        let mut ctx2 = Context::new();
        let func2 = diamond_loop(&mut ctx2, 4, false, 128);
        let text2 = run_unroll(&mut ctx2, func2);
        assert_eq!(text2.matches("llvm.cond_br").count(), 4, "{text2}");
    }

    /// An edge into the middle of the body (side entry) leaves the loop
    /// alone: the header no longer dominates the latch on that path, so
    /// it is not even a natural loop this pass would touch.
    #[test]
    fn side_entry_into_body_bails() {
        let mut ctx = Context::new();
        let func = diamond_loop(&mut ctx, 4, false, 0);
        // Rewire: entry cond_br's into {header-path, sneak}; sneak jumps
        // straight into the even arm.
        let region = func.get_region(&ctx).unwrap();
        let blocks: Vec<_> = region.deref(&ctx).iter(&ctx).collect();
        let (entry, header, even) = (blocks[0], blocks[1], blocks[3]);
        let sneak = BasicBlock::new(&mut ctx, None, vec![]);
        sneak.insert_after(&ctx, entry);
        BrOp::new(&mut ctx, even, vec![])
            .get_operation()
            .insert_at_back(sneak, &ctx);
        let old_term = entry.deref(&ctx).get_terminator(&ctx).unwrap();
        // Replace the entry br with a cond_br → header / sneak, on a
        // compare of two entry consts (init operands are the old br's).
        let ty = int_ty(&mut ctx, 64);
        let mk_const = |ctx: &mut Context, value: u64| {
            let c = ConstantOp::new(
                ctx,
                Box::new(IntegerAttr::new(
                    ty,
                    APInt::from_u64(value, NonZero::new(64).unwrap()),
                )),
            );
            c.get_operation().insert_before(ctx, old_term);
            c.get_result(ctx)
        };
        let zero = mk_const(&mut ctx, 0);
        let one = mk_const(&mut ctx, 1);
        let cmp = ICmpOp::new(&mut ctx, ICmpPredicateAttr::EQ, zero, one);
        cmp.get_operation().insert_before(&ctx, old_term);
        let cmp_v = cmp.get_result(&ctx);
        let init: Vec<Value> = BrOp::from_operation(old_term).successor_operands(&ctx, 0);
        let new_term = CondBrOp::new(&mut ctx, cmp_v, header, init, sneak, vec![]);
        new_term.get_operation().insert_before(&ctx, old_term);
        Operation::erase(old_term, &mut ctx);

        let text = run_unroll(&mut ctx, func);
        // 3 cond_brs: the entry's new one, the loop control, the diamond.
        assert_eq!(text.matches("llvm.cond_br").count(), 3, "{text}");
    }

    /// Nested constant-trip loops:
    /// for l in 0..3 { for j in 0..2 { acc += 2*l + j } } — the rounds
    /// fixpoint unrolls the inner loop first (its clones land in the
    /// outer body), then the outer loop via the multi-block path; folds
    /// to (0+1)+(2+3)+(4+5) = 15. `Bound::Arg` for the inner bound makes
    /// the inner loop un-unrollable, so the outer (its body now cyclic)
    /// must bail too.
    fn nested_loop(ctx: &mut Context, inner_bound: Bound) -> FuncOp {
        let i64_ty: TypeHandle = int_ty(ctx, 64).into();
        let fn_ty = FuncType::get(ctx, i64_ty, vec![i64_ty], false);
        let func = FuncOp::new(ctx, "nest".try_into().unwrap(), fn_ty);
        func.set_attr_llvm_function_linkage(ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(ctx);
        let region = func.get_region(ctx).unwrap();
        let entry = func.get_entry_block(ctx).unwrap();
        let arg = entry.deref(ctx).get_argument(0);
        let oh = BasicBlock::new(ctx, None, vec![i64_ty, i64_ty]);
        oh.insert_at_back(region, ctx);
        let opre = BasicBlock::new(ctx, None, vec![]);
        opre.insert_at_back(region, ctx);
        let ih = BasicBlock::new(ctx, None, vec![i64_ty, i64_ty]);
        ih.insert_at_back(region, ctx);
        let ibody = BasicBlock::new(ctx, None, vec![]);
        ibody.insert_at_back(region, ctx);
        let olatch = BasicBlock::new(ctx, None, vec![i64_ty]);
        olatch.insert_at_back(region, ctx);
        let exit = BasicBlock::new(ctx, None, vec![i64_ty]);
        exit.insert_at_back(region, ctx);

        let zero = const_i64(ctx, entry, 0);
        let one = const_i64(ctx, entry, 1);
        let three = const_i64(ctx, entry, 3);
        let inner_bound_v = match inner_bound {
            Bound::Const(n) => const_i64(ctx, entry, n),
            Bound::Arg => arg,
        };
        BrOp::new(ctx, oh, vec![zero, zero])
            .get_operation()
            .insert_at_back(entry, ctx);

        let l = oh.deref(ctx).get_argument(0);
        let acc = oh.deref(ctx).get_argument(1);
        let c1 = ICmpOp::new(ctx, ICmpPredicateAttr::SLT, l, three);
        c1.get_operation().insert_at_back(oh, ctx);
        let c1_v = c1.get_result(ctx);
        CondBrOp::new(ctx, c1_v, opre, vec![], exit, vec![acc])
            .get_operation()
            .insert_at_back(oh, ctx);

        BrOp::new(ctx, ih, vec![zero, acc])
            .get_operation()
            .insert_at_back(opre, ctx);

        let j = ih.deref(ctx).get_argument(0);
        let acc2 = ih.deref(ctx).get_argument(1);
        let c2 = ICmpOp::new(ctx, ICmpPredicateAttr::SLT, j, inner_bound_v);
        c2.get_operation().insert_at_back(ih, ctx);
        let c2_v = c2.get_result(ctx);
        CondBrOp::new(ctx, c2_v, ibody, vec![], olatch, vec![acc2])
            .get_operation()
            .insert_at_back(ih, ctx);

        let ll = AddOp::new_with_overflow_flag(ctx, l, l, Default::default());
        ll.get_operation().insert_at_back(ibody, ctx);
        let t = AddOp::new_with_overflow_flag(ctx, ll.get_result(ctx), j, Default::default());
        t.get_operation().insert_at_back(ibody, ctx);
        let acc3 = AddOp::new_with_overflow_flag(ctx, acc2, t.get_result(ctx), Default::default());
        acc3.get_operation().insert_at_back(ibody, ctx);
        let j2 = AddOp::new_with_overflow_flag(ctx, j, one, Default::default());
        j2.get_operation().insert_at_back(ibody, ctx);
        BrOp::new(ctx, ih, vec![j2.get_result(ctx), acc3.get_result(ctx)])
            .get_operation()
            .insert_at_back(ibody, ctx);

        let acc4 = olatch.deref(ctx).get_argument(0);
        let l2 = AddOp::new_with_overflow_flag(ctx, l, one, Default::default());
        l2.get_operation().insert_at_back(olatch, ctx);
        BrOp::new(ctx, oh, vec![l2.get_result(ctx), acc4])
            .get_operation()
            .insert_at_back(olatch, ctx);

        let res = exit.deref(ctx).get_argument(0);
        ReturnOp::new(ctx, Some(res))
            .get_operation()
            .insert_at_back(exit, ctx);
        func
    }

    #[test]
    fn nested_constant_trip_loops_unroll_inner_then_outer() {
        let mut ctx = Context::new();
        let func = nested_loop(&mut ctx, Bound::Const(2));
        let text = run_unroll(&mut ctx, func);
        assert!(!text.contains("llvm.cond_br"), "{text}");
        assert_eq!(folded_return(&mut ctx, func), Some(15));
    }

    /// An inner backedge that cannot be removed (non-constant inner
    /// bound) makes the outer body cyclic — the outer loop must bail.
    #[test]
    fn inner_backedge_in_body_bails() {
        let mut ctx = Context::new();
        let func = nested_loop(&mut ctx, Bound::Arg);
        let text = run_unroll(&mut ctx, func);
        assert_eq!(text.matches("llvm.cond_br").count(), 2, "{text}");
    }

    /// Attribution on multi-block clones: same rules as the straight-
    /// line paths (unique ll.op_id, clones carry derived_from).
    #[test]
    fn stamps_attribution_on_multi_block_clones() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "m".try_into().unwrap());
        let module_block = module.get_body(&ctx, 0);
        let func = diamond_loop(&mut ctx, 4, false, 0);
        func.get_operation().insert_at_back(module_block, &ctx);
        opmap::assign_op_ids(&mut ctx, module.get_operation()).unwrap();

        LLVMUnrollPass::new(&TargetProfile::host_cpu())
            .run(
                module.get_operation(),
                &mut ctx,
                &mut AnalysisManager::default(),
            )
            .unwrap();

        let region = func.get_region(&ctx).unwrap();
        let mut seen_ids = FxHashSet::default();
        let mut derived_clones = 0usize;
        for block in region.deref(&ctx).iter(&ctx) {
            for op in block.deref(&ctx).iter(&ctx) {
                assert!(
                    opmap::has_attribution(&ctx, op),
                    "unattributed op: {}",
                    op.disp(&ctx)
                );
                if let Some(id) = opmap::op_id(&ctx, op) {
                    assert!(seen_ids.insert(id), "duplicate ll.op_id {id}");
                } else {
                    assert!(
                        !opmap::effective_sources(&ctx, op).is_empty(),
                        "clone without derived_from: {}",
                        op.disp(&ctx)
                    );
                    derived_clones += 1;
                }
            }
        }
        assert!(
            derived_clones > 0,
            "unroll must have produced derived clones"
        );
    }

    #[test]
    fn skips_trip_too_large() {
        let mut ctx = Context::new();
        let (func, _exit) = sum_loop(&mut ctx, Bound::Const(MAX_TRIP + 1));
        let text = run_unroll(&mut ctx, func);
        assert!(text.contains("llvm.cond_br"), "{text}");
        assert_eq!(text.matches("llvm.add").count(), 2, "{text}");
    }

    #[test]
    fn skips_non_constant_bounds() {
        let mut ctx = Context::new();
        let (func, _exit) = sum_loop(&mut ctx, Bound::Arg);
        let text = run_unroll(&mut ctx, func);
        assert!(text.contains("llvm.cond_br"), "{text}");
        assert_eq!(text.matches("llvm.add").count(), 2, "{text}");
    }

    #[test]
    fn stamps_attribution_on_clones() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "m".try_into().unwrap());
        let module_block = module.get_body(&ctx, 0);
        let (func, _exit) = sum_loop(&mut ctx, Bound::Const(3));
        func.get_operation().insert_at_back(module_block, &ctx);
        opmap::assign_op_ids(&mut ctx, module.get_operation()).unwrap();

        LLVMUnrollPass::new(&TargetProfile::host_cpu())
            .run(
                module.get_operation(),
                &mut ctx,
                &mut AnalysisManager::default(),
            )
            .unwrap();

        // Every surviving op is attributed, ll.op_id stays unique (clones
        // carry derived_from instead of duplicating the original's id),
        // and at least one clone derives from a stamped original.
        let region = func.get_region(&ctx).unwrap();
        let mut seen_ids = FxHashSet::default();
        let mut derived_clones = 0usize;
        for block in region.deref(&ctx).iter(&ctx) {
            for op in block.deref(&ctx).iter(&ctx) {
                assert!(
                    opmap::has_attribution(&ctx, op),
                    "unattributed op: {}",
                    op.disp(&ctx)
                );
                if let Some(id) = opmap::op_id(&ctx, op) {
                    assert!(seen_ids.insert(id), "duplicate ll.op_id {id}");
                } else {
                    assert!(
                        !opmap::effective_sources(&ctx, op).is_empty(),
                        "clone without derived_from: {}",
                        op.disp(&ctx)
                    );
                    derived_clones += 1;
                }
            }
        }
        assert!(
            derived_clones > 0,
            "unroll must have produced derived clones"
        );
    }
}
