//! Loop-carried store→load forwarding (the seidel_2d recurrence): when an
//! in-place sweep stores `base[f(iv)]` and the next iteration loads
//! `base[f(iv - 1)]`, the loaded value is exactly the value stored one
//! iteration earlier — but it round-trips through memory every iteration
//! (ncu: 72% long-scoreboard stalls on that load in seidel_2d). This pass
//! threads the stored value through a loop-carried block argument instead,
//! the way LLVM/nvcc keep the recurrence in a register. The store itself
//! stays (memory must still be updated); only the in-loop load goes away.
//!
//! Recognition (all conditions must hold — correctness over coverage):
//! - A natural loop with ONE latch and an existing preheader, shaped like
//!   [unroll](super::unroll)'s two canonical forms: single-block do-while
//!   (`header ⇄ header`, exit test after the body) or two-block while
//!   (`header → body → header`, exit `cond_br` in the header — rustc's
//!   `Range`/`while` shape). The load(s) and the store live in the SAME
//!   block: the single block, or the body of the while shape.
//! - A canonical induction variable: a header block argument whose
//!   backedge value is `add iv, C` for a constant step.
//! - Exactly ONE store in the whole loop, and every other op in the loop
//!   is in the shared memory-benign set ([analysis](super::analysis)'s
//!   OpEffect table) or a load — any call or unknown-effect op bails.
//! - The store and load addresses are structurally identical GEP chains
//!   ([gvn](super::gvn)'s `AddrKey` model) off the same base, differing in
//!   exactly ONE index, both SSA values. Writing each as an affine form
//!   `scale·root + offset` (peeling add/sub/mul/shl-by-constant), they
//!   must share `root` and `scale`, and `root` must be affine in the iv
//!   (structural derivative `d`); with per-iteration index advance
//!   `adv = scale·d·step`, the pass requires `off_load − off_store =
//!   −adv` — i.e. `k = 1`: the load reads exactly last iteration's store
//!   address. Every OTHER value the addresses share (GEP base, identical
//!   indices) must be defined outside the loop.
//! - Loaded and stored types are equal, and every forwarded load precedes
//!   the store in its block (so the same-iteration store — to a different
//!   address by `adv ≠ 0` — can never partially overlap a "later" load of
//!   the carried slot; loads after the store are left alone).
//!
//! First-iteration value. The carried argument must be initialized with
//! `mem[f(iv0 − 1)]` — the address the FIRST in-loop load reads:
//! - do-while shape: the body runs at least once whenever the preheader
//!   runs, so the preheader load is exactly as safe as iteration 1's load
//!   (same address, same reachability): emit it unconditionally.
//! - while shape with constant start/bound: evaluate the first exit test
//!   (the same simulation as unroll's trip matcher); if it provably
//!   enters, emit the preheader load unconditionally; if it provably does
//!   NOT enter, bail (a dead loop is simplify-cfg's job).
//! - while shape with a runtime bound (the real seidel_2d: `cols` is a
//!   kernel argument): the pass cannot prove the loop runs, and an
//!   unconditional preheader load would speculate a possibly out-of-bounds
//!   address. Instead it emits a GUARDED initial load: the preheader
//!   re-evaluates the loop's own exit condition at `iv = iv0` (a clone of
//!   the condition's pure in-loop chain — bail if it is not clonable) and
//!   branches either to a tiny new block that performs the initial load
//!   and enters the header, or straight to the header with an undef
//!   carried value. The undef is only ever consumed by the body, which is
//!   unreachable when the guard failed (guard ≡ first header test, on the
//!   same pure operands), so no poison escapes and no load speculates.
//!   The guard adds one compare on the loop entry path only.
//!
//! Rewrite: `%carried` is a new header block argument; the preheader (or
//! guard block) passes the initial load; the backedge passes the store's
//! value operand; every forwarded in-loop load is replaced by `%carried`
//! and erased. Address algebra is exact modulo 2^width (all index math
//! wraps), and address distinctness (`adv ≠ 0` ⇒ `f(n) ≠ f(n−1)`) assumes
//! accessed objects do not wrap the address space — the same syntactic
//! model gvn's `AddrKey` uses.
//!
//! ADJOINT (backward attribution): the initial-load clone (and its cloned
//! address chain / guard compare) derives from the originals like unroll's
//! clones; rebuilt terminators derive from the terminators they replace;
//! the erased load's identity flows into the initial load via its clone
//! stamp. `ll.op_id`s stay unique (clones are stripped and re-derived).

use rustc_hash::FxHashSet;

use pliron::builtin::op_interfaces::{AtMostOneRegionInterface as _, BranchOpInterface as _};

use crate::{
    context::{Context, Ptr},
    conversion::pass::{AnalysisManager, Pass, PassResult, changed, unchanged},
    dialects::{
        builtin::types::IntegerType,
        llvm::{
            attributes::ICmpPredicateAttr,
            op_interfaces::IsDeclaration,
            ops::{
                AddOp, BrOp, CondBrOp, ICmpOp, LoadOp, MulOp, ShlOp, StoreOp, SubOp, UndefOp,
            },
        },
    },
    ir::{
        basic_block::BasicBlock,
        op::Op,
        operation::Operation,
        region::Region,
        r#type::{TypeHandle, Typed},
        value::{DefiningEntity, Value},
    },
    irbuild::{cloning, listener::DummyListener, rewriter::IRRewriter},
    linked_list::{ContainsLinkedList, LinkedList as _},
};

use super::{
    analysis::{NaturalLoop, OpEffect, dominator_tree, memory_benign_op_ids, natural_loops, op_effects},
    gvn::{AddrKey, IdxKey, addr_key},
    inline::collect_functions,
    midend_gate::midend_disabled,
    simplify::{as_const_operand, mask_to_width, sign_extend},
};
use crate::passes::aarch64::opmap;

/// Successful forwards per function per run: each one adds blocks/args, so
/// the loop forest is re-discovered from scratch (like unroll's rounds).
const MAX_ROUNDS: usize = 8;
/// Bound on the cloned pure chain (guard condition + initial address).
const MAX_CHAIN_OPS: usize = 64;
/// Bound on the affine peel / derivative walks.
const MAX_AFFINE_DEPTH: usize = 32;

pub struct LLVMLoopCarriedFwdPass;

impl Pass for LLVMLoopCarriedFwdPass {
    fn name(&self) -> &str {
        "llvm-loop-carried-fwd"
    }

    fn run(
        &mut self,
        root: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> pliron::result::Result<PassResult> {
        if midend_disabled("lcfwd") {
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
            for _ in 0..MAX_ROUNDS {
                if !forward_one_loop(ctx, region) {
                    break;
                }
                any = true;
            }
        }
        Ok(if any { changed() } else { unchanged() })
    }
}

/// Find one forwardable loop (innermost first) and rewrite it. One loop
/// per call: the rewrite adds blocks and block arguments, so the caller
/// recomputes the dominator tree and loop forest before the next attempt.
fn forward_one_loop(ctx: &mut Context, region: Ptr<Region>) -> bool {
    let dom = dominator_tree(ctx, region);
    let loops = natural_loops(ctx, &dom);
    for natural_loop in &loops {
        if let Some(plan) = analyze(ctx, natural_loop) {
            apply(ctx, region, &plan);
            return true;
        }
    }
    false
}

// ============================================================================
// Analysis
// ============================================================================

struct FwdPlan {
    preheader: Ptr<BasicBlock>,
    header: Ptr<BasicBlock>,
    /// True for the two-block while shape (latch is the memory block, plain
    /// `br` backedge); false for do-while (backedge on the `cond_br`).
    shape_is_while: bool,
    /// The loop's exiting `cond_br` and which successor stays in-loop.
    cond_br: Ptr<Operation>,
    inloop_idx: usize,
    /// Preheader branch operands (header args on entry).
    init_operands: Vec<Value>,
    /// Backedge operands (header args from the latch).
    latch_operands: Vec<Value>,
    latch_term: Ptr<Operation>,
    /// The single store's value operand — the carried value.
    stored_value: Value,
    /// Loads replaced by the carried argument (all before the store).
    loads: Vec<Ptr<Operation>>,
    carried_ty: TypeHandle,
    /// `Some(cond)`: the entry must be guarded by a preheader clone of
    /// `cond` (the loop's exit condition) at `iv = iv0`. `None`: the
    /// initial load is unconditionally safe (do-while, or proven entry).
    guard_cond: Option<Value>,
    /// Pure in-loop ops to clone for the guard condition, topo order.
    guard_chain: Vec<Ptr<Operation>>,
    /// Pure in-loop ops to clone for the initial load's address, topo
    /// order, excluding ops already in `guard_chain`.
    addr_chain: Vec<Ptr<Operation>>,
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

fn int_width(ctx: &Context, value: Value) -> Option<u32> {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .map(|t| t.width())
}

fn analyze(ctx: &Context, natural_loop: &NaturalLoop) -> Option<FwdPlan> {
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

    // Loop shape, exactly unroll's two canonical forms.
    let (mem_block, cond_br, shape_is_while) = match natural_loop.body.len() {
        1 => {
            if latch != header {
                return None;
            }
            let term = header.deref(ctx).get_terminator(ctx)?;
            if Operation::get_opid(term, ctx) != CondBrOp::get_opid_static() {
                return None;
            }
            (header, term, false)
        }
        2 => {
            let body_blk = *natural_loop.body.iter().find(|b| **b != header)?;
            if latch != body_blk {
                return None;
            }
            let header_term = header.deref(ctx).get_terminator(ctx)?;
            if Operation::get_opid(header_term, ctx) != CondBrOp::get_opid_static() {
                return None;
            }
            let latch_term = body_blk.deref(ctx).get_terminator(ctx)?;
            if Operation::get_opid(latch_term, ctx) != BrOp::get_opid_static() {
                return None;
            }
            if BrOp::from_operation(latch_term)
                .get_operation()
                .deref(ctx)
                .get_successor(0)
                != header
            {
                return None;
            }
            (body_blk, header_term, true)
        }
        _ => return None,
    };
    let latch_term = latch.deref(ctx).get_terminator(ctx)?;

    // One in-loop and one out-of-loop successor on the exiting cond_br.
    let cbr = CondBrOp::from_operation(cond_br);
    let succ0 = cbr.get_operation().deref(ctx).get_successor(0);
    let succ1 = cbr.get_operation().deref(ctx).get_successor(1);
    let expected_inloop = if shape_is_while { mem_block } else { header };
    let inloop_idx = if succ0 == expected_inloop && !natural_loop.body.contains(&succ1) {
        0usize
    } else if succ1 == expected_inloop && !natural_loop.body.contains(&succ0) {
        1usize
    } else {
        return None;
    };

    // Header arg plumbing.
    let num_args = header.deref(ctx).get_num_arguments();
    let init_operands = BrOp::from_operation(pre_term).successor_operands(ctx, 0);
    let latch_operands = if shape_is_while {
        BrOp::from_operation(latch_term).successor_operands(ctx, 0)
    } else {
        cbr.successor_operands(ctx, inloop_idx)
    };
    if init_operands.len() != num_args || latch_operands.len() != num_args {
        return None;
    }
    let header_args: Vec<Value> = (0..num_args)
        .map(|j| header.deref(ctx).get_argument(j))
        .collect();

    // Memory safety: exactly one store in the whole loop, everything else
    // memory-benign (calls and unknown-effect ops are absent from the
    // table, so they bail here).
    let benign = memory_benign_op_ids();
    let mut store: Option<Ptr<Operation>> = None;
    for block in &natural_loop.body {
        for op in block.deref(ctx).iter(ctx) {
            let opid = Operation::get_opid(op, ctx);
            if opid == StoreOp::get_opid_static() {
                if store.is_some() {
                    return None;
                }
                store = Some(op);
                continue;
            }
            if !benign.contains(&opid) {
                return None;
            }
        }
    }
    let store = store?;
    if store.deref(ctx).get_container() != Some(mem_block) {
        return None;
    }
    let store_op = StoreOp::from_operation(store);
    let stored_value = store_op.get_operand_value(ctx);
    let carried_ty = stored_value.get_type(ctx);
    let store_key = addr_key(ctx, store_op.get_operand_address(ctx));
    if !matches!(store_key, AddrKey::Gep(..)) {
        return None;
    }

    // Induction candidates: header args whose backedge value is
    // `add self, C` defined in the loop (either operand order).
    let mut iv_candidates: Vec<(usize, u128, u32)> = Vec::new();
    for (j, latch_v) in latch_operands.iter().enumerate() {
        let Some(add_op) = latch_v.defining_op() else {
            continue;
        };
        if Operation::get_opid(add_op, ctx) != AddOp::get_opid_static()
            || !defined_in_loop(ctx, *latch_v, &natural_loop.body)
        {
            continue;
        }
        let (a, b) = {
            let op_ref = add_op.deref(ctx);
            (op_ref.get_operand(0), op_ref.get_operand(1))
        };
        let (arg, step) = if let Some(c) = as_const_operand(ctx, b) {
            (a, c)
        } else if let Some(c) = as_const_operand(ctx, a) {
            (b, c)
        } else {
            continue;
        };
        if arg == header_args[j] && step.width > 0 && step.width <= 64 {
            iv_candidates.push((j, step.masked(), step.width));
        }
    }

    // Ops of the memory block in program order, and the store's position.
    let mem_ops: Vec<Ptr<Operation>> = mem_block.deref(ctx).iter(ctx).collect();
    let store_pos = mem_ops.iter().position(|op| *op == store)?;

    for (iv_index, step, step_width) in iv_candidates {
        let iv = header_args[iv_index];
        if int_width(ctx, iv) != Some(step_width) {
            continue;
        }
        let Some(loads) = match_loads(
            ctx,
            natural_loop,
            &mem_ops[..store_pos],
            &store_key,
            carried_ty,
            iv,
            step,
            step_width,
        ) else {
            continue;
        };

        // First-iteration guarantee.
        let guard_cond = if !shape_is_while {
            None // the do-while body runs whenever the preheader runs
        } else {
            match provably_enters(
                ctx,
                cond_br,
                inloop_idx,
                iv,
                latch_operands[iv_index],
                init_operands[iv_index],
            ) {
                Some(true) => None,
                Some(false) => return None, // provably dead loop: not ours
                None => Some(cbr.get_operand_condition(ctx)),
            }
        };

        // Clonable pure chains: guard condition first (it lands in the
        // preheader), then the initial load's address (it lands after the
        // guard, so it may reuse the guard's clones).
        let effects = op_effects();
        let mut seen: FxHashSet<Ptr<Operation>> = FxHashSet::default();
        let mut guard_chain = Vec::new();
        if let Some(cond) = guard_cond
            && !collect_chain(
                ctx,
                cond,
                natural_loop,
                header,
                &effects,
                &mut seen,
                &mut guard_chain,
            )
        {
            continue;
        }
        let mut addr_chain = Vec::new();
        let template_addr = LoadOp::from_operation(loads[0]).get_operand_address(ctx);
        if !collect_chain(
            ctx,
            template_addr,
            natural_loop,
            header,
            &effects,
            &mut seen,
            &mut addr_chain,
        ) {
            continue;
        }

        return Some(FwdPlan {
            preheader,
            header,
            shape_is_while,
            cond_br,
            inloop_idx,
            init_operands,
            latch_operands,
            latch_term,
            stored_value,
            loads,
            carried_ty,
            guard_cond,
            guard_chain,
            addr_chain,
        });
    }
    None
}

/// The loads (before the store, in its block) whose address is the store's
/// address shifted by exactly one iteration back (`k = 1`). `None` when no
/// load qualifies.
#[allow(clippy::too_many_arguments)]
fn match_loads(
    ctx: &Context,
    natural_loop: &NaturalLoop,
    ops_before_store: &[Ptr<Operation>],
    store_key: &AddrKey,
    carried_ty: TypeHandle,
    iv: Value,
    step: u128,
    width: u32,
) -> Option<Vec<Ptr<Operation>>> {
    let mut loads = Vec::new();
    for op in ops_before_store {
        if Operation::get_opid(*op, ctx) != LoadOp::get_opid_static() {
            continue;
        }
        let load = LoadOp::from_operation(*op);
        if op.deref(ctx).get_result(0).get_type(ctx) != carried_ty {
            continue;
        }
        let load_key = addr_key(ctx, load.get_operand_address(ctx));
        let mut shared = Vec::new();
        let KeyCmp::Shifted(v_load, v_store) = cmp_keys(&load_key, store_key, &mut shared)
        else {
            continue;
        };
        // Everything the two addresses share must be loop-invariant, or
        // "the store's address one iteration ago" is not the load's.
        if shared
            .iter()
            .any(|v| defined_in_loop(ctx, *v, &natural_loop.body))
        {
            continue;
        }
        // Affine forms scale·root + offset off the same root and scale.
        if int_width(ctx, v_load) != Some(width) || int_width(ctx, v_store) != Some(width) {
            continue;
        }
        let (root_l, scale_l, off_l) = affine_shift(ctx, v_load, width);
        let (root_s, scale_s, off_s) = affine_shift(ctx, v_store, width);
        if root_l != root_s || scale_l != scale_s {
            continue;
        }
        // Structural derivative of the shared root w.r.t. the iv.
        let Some(d_root) = deriv_wrt_iv(ctx, root_l, iv, &natural_loop.body, width, 0) else {
            continue;
        };
        let adv = mask_to_width(
            scale_l.wrapping_mul(d_root).wrapping_mul(step),
            width,
        );
        if adv == 0 {
            continue;
        }
        // k = 1 exactly: off_load − off_store == −adv (mod 2^width).
        if mask_to_width(off_l.wrapping_sub(off_s), width)
            != mask_to_width(0u128.wrapping_sub(adv), width)
        {
            continue;
        }
        loads.push(*op);
    }
    if loads.is_empty() { None } else { Some(loads) }
}

/// Result of structurally comparing two address keys.
enum KeyCmp {
    /// Identical (same-iteration forwarding — gvn's job, not ours).
    Equal,
    /// Identical except exactly one `Val` index pair `(load, store)`.
    Shifted(Value, Value),
    Mismatch,
}

/// Compare two [AddrKey]s; `shared` collects every SSA value the keys have
/// in common (base roots and identical value indices) for the invariance
/// check. Bases must be identical — an index shift is only recognized at
/// one GEP level.
fn cmp_keys(a: &AddrKey, b: &AddrKey, shared: &mut Vec<Value>) -> KeyCmp {
    match (a, b) {
        (AddrKey::Root(x), AddrKey::Root(y)) => {
            if x == y {
                shared.push(*x);
                KeyCmp::Equal
            } else {
                KeyCmp::Mismatch
            }
        }
        (AddrKey::Gep(base_a, ty_a, idx_a), AddrKey::Gep(base_b, ty_b, idx_b)) => {
            if ty_a != ty_b || idx_a.len() != idx_b.len() {
                return KeyCmp::Mismatch;
            }
            match cmp_keys(base_a, base_b, shared) {
                KeyCmp::Equal => {}
                _ => return KeyCmp::Mismatch,
            }
            let mut diff: Option<(Value, Value)> = None;
            for (ka, kb) in idx_a.iter().zip(idx_b) {
                match (ka, kb) {
                    (IdxKey::Const(x), IdxKey::Const(y)) if x == y => {}
                    (IdxKey::Val(x), IdxKey::Val(y)) if x == y => shared.push(*x),
                    (IdxKey::Val(x), IdxKey::Val(y)) => {
                        if diff.is_some() {
                            return KeyCmp::Mismatch;
                        }
                        diff = Some((*x, *y));
                    }
                    _ => return KeyCmp::Mismatch,
                }
            }
            match diff {
                Some((l, s)) => KeyCmp::Shifted(l, s),
                None => KeyCmp::Equal,
            }
        }
        _ => KeyCmp::Mismatch,
    }
}

/// Peel constant add/sub/mul/shl layers from the OUTSIDE in, keeping the
/// invariant `value = scale·root + offset` at every step (all arithmetic
/// wrapping at `width`): peeling `root = x ± C` folds `±scale·C` into the
/// offset; peeling `root = x·C` / `root = x << C` multiplies the scale
/// only. Stops at the first non-peelable op — the remainder is the root.
fn affine_shift(ctx: &Context, value: Value, width: u32) -> (Value, u128, u128) {
    let mut root = value;
    let mut scale: u128 = 1;
    let mut offset: u128 = 0;
    for _ in 0..MAX_AFFINE_DEPTH {
        let Some(op) = root.defining_op() else { break };
        let opid = Operation::get_opid(op, ctx);
        let (a, b) = {
            let op_ref = op.deref(ctx);
            if op_ref.get_num_operands() != 2 {
                break;
            }
            (op_ref.get_operand(0), op_ref.get_operand(1))
        };
        if opid == AddOp::get_opid_static() {
            let (x, c) = if let Some(c) = as_const_operand(ctx, b) {
                (a, c)
            } else if let Some(c) = as_const_operand(ctx, a) {
                (b, c)
            } else {
                break;
            };
            if c.width != width {
                break;
            }
            offset = offset.wrapping_add(scale.wrapping_mul(c.masked()));
            root = x;
        } else if opid == SubOp::get_opid_static() {
            // Only `x - C`; `C - x` flips the scale sign — bail on it.
            let Some(c) = as_const_operand(ctx, b) else { break };
            if c.width != width {
                break;
            }
            offset = offset.wrapping_sub(scale.wrapping_mul(c.masked()));
            root = a;
        } else if opid == MulOp::get_opid_static() {
            let (x, c) = if let Some(c) = as_const_operand(ctx, b) {
                (a, c)
            } else if let Some(c) = as_const_operand(ctx, a) {
                (b, c)
            } else {
                break;
            };
            if c.width != width {
                break;
            }
            scale = scale.wrapping_mul(c.masked());
            root = x;
        } else if opid == ShlOp::get_opid_static() {
            let Some(c) = as_const_operand(ctx, b) else { break };
            let shift = c.masked();
            if c.width != width || shift >= u128::from(width) {
                break;
            }
            scale = scale.wrapping_shl(shift as u32);
            root = a;
        } else {
            break;
        }
    }
    (root, mask_to_width(scale, width), mask_to_width(offset, width))
}

/// Structural derivative of `value` w.r.t. one loop iteration's change of
/// `iv` (per unit of iv): `None` means "not affine in the iv" — any other
/// in-loop block argument, cast, load, or unclassified op bails.
fn deriv_wrt_iv(
    ctx: &Context,
    value: Value,
    iv: Value,
    body: &FxHashSet<Ptr<BasicBlock>>,
    width: u32,
    depth: usize,
) -> Option<u128> {
    if depth > MAX_AFFINE_DEPTH {
        return None;
    }
    if value == iv {
        return Some(1);
    }
    if !defined_in_loop(ctx, value, body) {
        return Some(0);
    }
    if as_const_operand(ctx, value).is_some() {
        return Some(0);
    }
    let op = value.defining_op()?; // an in-loop block arg that isn't the iv: bail
    let opid = Operation::get_opid(op, ctx);
    let (a, b) = {
        let op_ref = op.deref(ctx);
        if op_ref.get_num_operands() != 2 {
            return None;
        }
        (op_ref.get_operand(0), op_ref.get_operand(1))
    };
    if opid == AddOp::get_opid_static() {
        let da = deriv_wrt_iv(ctx, a, iv, body, width, depth + 1)?;
        let db = deriv_wrt_iv(ctx, b, iv, body, width, depth + 1)?;
        return Some(mask_to_width(da.wrapping_add(db), width));
    }
    if opid == SubOp::get_opid_static() {
        let da = deriv_wrt_iv(ctx, a, iv, body, width, depth + 1)?;
        let db = deriv_wrt_iv(ctx, b, iv, body, width, depth + 1)?;
        return Some(mask_to_width(da.wrapping_sub(db), width));
    }
    if opid == MulOp::get_opid_static() {
        let (x, c) = if let Some(c) = as_const_operand(ctx, b) {
            (a, c)
        } else if let Some(c) = as_const_operand(ctx, a) {
            (b, c)
        } else {
            return None; // non-constant product: not affine
        };
        if c.width != width {
            return None;
        }
        let dx = deriv_wrt_iv(ctx, x, iv, body, width, depth + 1)?;
        return Some(mask_to_width(dx.wrapping_mul(c.masked()), width));
    }
    if opid == ShlOp::get_opid_static() {
        let c = as_const_operand(ctx, b)?;
        let shift = c.masked();
        if c.width != width || shift >= u128::from(width) {
            return None;
        }
        let da = deriv_wrt_iv(ctx, a, iv, body, width, depth + 1)?;
        return Some(mask_to_width(da.wrapping_shl(shift as u32), width));
    }
    None
}

/// Does the while-shape loop provably run at least once? `Some(true)` /
/// `Some(false)` when start and bound are constants and the exit compare
/// is the canonical `icmp` on the iv or its update (the same recognition
/// as unroll's trip simulation, evaluated for the first test only);
/// `None` when it cannot be decided statically.
fn provably_enters(
    ctx: &Context,
    cond_br: Ptr<Operation>,
    inloop_idx: usize,
    iv: Value,
    iv_update: Value,
    iv_init: Value,
) -> Option<bool> {
    let cbr = CondBrOp::from_operation(cond_br);
    let cond = cbr.get_operand_condition(ctx);
    let icmp_op = cond.defining_op()?;
    if Operation::get_opid(icmp_op, ctx) != ICmpOp::get_opid_static() {
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
    let cmp_on_next = if v == iv {
        false
    } else if v == iv_update {
        true
    } else {
        return None;
    };
    let start = as_const_operand(ctx, iv_init)?;
    let width = start.width;
    if width == 0 || width > 64 || bound.width != width {
        return None;
    }
    let tested = if cmp_on_next {
        // The update is `add iv, C`; find C again to advance the value.
        let add_op = iv_update.defining_op()?;
        let (a, b) = {
            let op_ref = add_op.deref(ctx);
            (op_ref.get_operand(0), op_ref.get_operand(1))
        };
        let step = as_const_operand(ctx, b)
            .or_else(|| as_const_operand(ctx, a))?
            .masked();
        mask_to_width(start.masked().wrapping_add(step), width)
    } else {
        mask_to_width(start.masked(), width)
    };
    let (l, r) = if v_is_lhs {
        (tested, bound.masked())
    } else {
        (bound.masked(), tested)
    };
    let cond_true = eval_icmp(&predicate, l, r, width);
    let continue_on = inloop_idx == 0;
    Some(cond_true == continue_on)
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

/// Collect the in-loop ops computing `value`, post-order (operands first),
/// into `out`. Values defined outside the loop are leaves; header block
/// arguments are leaves too (the apply step maps every header argument to
/// its initial value — evaluating the chain "at iteration 1"). Any other
/// in-loop block argument, any op with unknown/memory effects, and any op
/// with regions bail.
fn collect_chain(
    ctx: &Context,
    value: Value,
    natural_loop: &NaturalLoop,
    header: Ptr<BasicBlock>,
    effects: &rustc_hash::FxHashMap<crate::ir::op::OpId, OpEffect>,
    seen: &mut FxHashSet<Ptr<Operation>>,
    out: &mut Vec<Ptr<Operation>>,
) -> bool {
    match value.defining_entity() {
        DefiningEntity::Block(block) => {
            // Header args are mapped to their initial values; any other
            // in-loop block argument has no iteration-1 value we know.
            !natural_loop.body.contains(&block) || block == header
        }
        DefiningEntity::Op(op) => {
            let Some(container) = op.deref(ctx).get_container() else {
                return false;
            };
            if !natural_loop.body.contains(&container) {
                return true;
            }
            if seen.contains(&op) {
                return true;
            }
            if out.len() + seen.len() >= MAX_CHAIN_OPS {
                return false;
            }
            let opid = Operation::get_opid(op, ctx);
            if !matches!(
                effects.get(&opid),
                Some(OpEffect::Pure | OpEffect::Materialize)
            ) || op.deref(ctx).num_regions() != 0
            {
                return false;
            }
            let operands: Vec<Value> = {
                let op_ref = op.deref(ctx);
                (0..op_ref.get_num_operands())
                    .map(|i| op_ref.get_operand(i))
                    .collect()
            };
            for operand in operands {
                if !collect_chain(ctx, operand, natural_loop, header, effects, seen, out) {
                    return false;
                }
            }
            seen.insert(op);
            out.push(op);
            true
        }
    }
}

// ============================================================================
// Rewrite
// ============================================================================

/// Strip the identity attributes a clone copied from its original and
/// stamp it as derived from the original's effective sources (same
/// convention as unroll's clones).
fn stamp_clone(ctx: &mut Context, clone: Ptr<Operation>, original: Ptr<Operation>) {
    let sources = opmap::effective_sources(ctx, original);
    {
        let mut clone_ref = clone.deref_mut(ctx);
        clone_ref.attributes.0.remove(&*opmap::ATTR_KEY_AARCH64_OP_ID);
        clone_ref
            .attributes
            .0
            .remove(&*opmap::ATTR_KEY_AARCH64_DERIVED_FROM);
        clone_ref
            .attributes
            .0
            .remove(&*opmap::ATTR_KEY_DERIVED_FROM_MANY);
        clone_ref.attributes.0.remove(&*opmap::ATTR_KEY_INLINED_FROM);
    }
    if !sources.is_empty() {
        opmap::set_derived_from_many(ctx, clone, sources);
    }
}

fn apply(ctx: &mut Context, region: Ptr<Region>, plan: &FwdPlan) {
    let pre_term = plan
        .preheader
        .deref(ctx)
        .get_terminator(ctx)
        .expect("analyze verified the preheader terminator");

    // The carried slot: a fresh header block argument.
    let carried_idx = BasicBlock::push_argument(plan.header, ctx, plan.carried_ty);
    let carried = plan.header.deref(ctx).get_argument(carried_idx);

    // Evaluate the cloned chains "at iteration 1": every header argument
    // maps to its initial value.
    let mut mapper = cloning::IrMapping::new();
    let mut rewriter = IRRewriter::<DummyListener>::default();
    let num_args = plan.init_operands.len();
    for j in 0..num_args {
        let arg = plan.header.deref(ctx).get_argument(j);
        mapper.map_value(arg, plan.init_operands[j]);
    }

    // Guard condition clones live in the preheader (they must run whether
    // or not the loop is entered — they are pure).
    for &op in &plan.guard_chain {
        let clone = cloning::clone_operation(op, ctx, &mut rewriter, &mut mapper);
        clone.insert_before(ctx, pre_term);
        stamp_clone(ctx, clone, op);
    }

    // Where the initial load goes: the preheader itself, or a new guarded
    // block on the entry edge.
    let template_load = plan.loads[0];
    let guard_block = plan.guard_cond.map(|_| {
        let block = BasicBlock::new(ctx, None, vec![]);
        block.insert_at_back(region, ctx);
        block
    });

    // Clone the address chain and the load itself into the init block.
    for &op in &plan.addr_chain {
        let clone = cloning::clone_operation(op, ctx, &mut rewriter, &mut mapper);
        match guard_block {
            Some(block) => clone.insert_at_back(block, ctx),
            None => clone.insert_before(ctx, pre_term),
        }
        stamp_clone(ctx, clone, op);
    }
    let init_load = cloning::clone_operation(template_load, ctx, &mut rewriter, &mut mapper);
    match guard_block {
        Some(block) => init_load.insert_at_back(block, ctx),
        None => init_load.insert_before(ctx, pre_term),
    }
    stamp_clone(ctx, init_load, template_load);
    let init_value = init_load.deref(ctx).get_result(0);

    // Rebuild the entry edge.
    let mut entry_operands = plan.init_operands.clone();
    match (guard_block, plan.guard_cond) {
        (Some(block), Some(cond)) => {
            // guard true (== the loop's first exit test says "enter"):
            // load block → header(init..., loaded); guard false: header
            // directly with an undef carried value the exit path never
            // reads (the body is the only consumer and never runs).
            entry_operands.push(init_value);
            let br = BrOp::new(ctx, plan.header, entry_operands).get_operation();
            br.insert_at_back(block, ctx);
            opmap::derive_new_from(ctx, br, pre_term);

            let undef = UndefOp::new(ctx, plan.carried_ty).get_operation();
            undef.insert_before(ctx, pre_term);
            stamp_clone(ctx, undef, template_load);
            let undef_v = undef.deref(ctx).get_result(0);
            let mut bypass_operands = plan.init_operands.clone();
            bypass_operands.push(undef_v);

            let guard_cond = mapper
                .lookup_value(cond)
                .expect("the guard chain cloned the condition's def");
            let continue_on = plan.inloop_idx == 0;
            let cond_br = if continue_on {
                CondBrOp::new(ctx, guard_cond, block, vec![], plan.header, bypass_operands)
            } else {
                CondBrOp::new(ctx, guard_cond, plan.header, bypass_operands, block, vec![])
            }
            .get_operation();
            cond_br.insert_before(ctx, pre_term);
            opmap::derive_new_from(ctx, cond_br, pre_term);
        }
        _ => {
            entry_operands.push(init_value);
            let br = BrOp::new(ctx, plan.header, entry_operands).get_operation();
            br.insert_before(ctx, pre_term);
            opmap::derive_new_from(ctx, br, pre_term);
        }
    }
    Operation::erase(pre_term, ctx);

    // Rebuild the backedge: it now also passes the stored value.
    let mut back_operands = plan.latch_operands.clone();
    back_operands.push(plan.stored_value);
    let old_latch_term = plan.latch_term;
    let new_latch_term = if plan.shape_is_while {
        BrOp::new(ctx, plan.header, back_operands).get_operation()
    } else {
        let cbr = CondBrOp::from_operation(plan.cond_br);
        let cond = cbr.get_operand_condition(ctx);
        let succ0 = cbr.get_operation().deref(ctx).get_successor(0);
        let succ1 = cbr.get_operation().deref(ctx).get_successor(1);
        let ops0 = if plan.inloop_idx == 0 {
            back_operands.clone()
        } else {
            cbr.successor_operands(ctx, 0)
        };
        let ops1 = if plan.inloop_idx == 1 {
            back_operands.clone()
        } else {
            cbr.successor_operands(ctx, 1)
        };
        CondBrOp::new(ctx, cond, succ0, ops0, succ1, ops1).get_operation()
    };
    new_latch_term.insert_before(ctx, old_latch_term);
    opmap::derive_new_from(ctx, new_latch_term, old_latch_term);
    Operation::erase(old_latch_term, ctx);

    // Replace the forwarded loads with the carried argument and erase
    // them. Their identity lives on in the initial load's clone stamp.
    for &load in &plan.loads {
        let result = load.deref(ctx).get_result(0);
        result.replace_some_uses_with(ctx, |_, _| true, &carried);
        Operation::erase(load, ctx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pliron::builtin::op_interfaces::OneResultInterface as _;
    use pliron_llvm::op_interfaces::IntBinArithOpWithOverflowFlag as _;

    use crate::{
        dialects::{
            builtin::{
                attributes::IntegerAttr,
                ops::ConstantOp,
                types::{IntegerType, Signedness},
            },
            llvm::{
                attributes::LinkageAttr,
                ops::{CallOp, FuncOp, GepIndex, GetElementPtrOp, ReturnOp},
                types::{FuncType, PointerType},
            },
        },
        ir::r#type::TypedHandle,
        printable::Printable,
        utils::apint::APInt,
    };
    use std::num::NonZero;

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

    fn block_ops_text(ctx: &Context, block: Ptr<BasicBlock>) -> String {
        block
            .deref(ctx)
            .iter(ctx)
            .map(|op| format!("{}\n", op.disp(ctx)))
            .collect()
    }

    fn run_pass(ctx: &mut Context, func: FuncOp) -> String {
        LLVMLoopCarriedFwdPass
            .run(func.get_operation(), ctx, &mut AnalysisManager::default())
            .unwrap();
        format!("{}", func.get_operation().disp(ctx))
    }

    enum Bound {
        Const(u64),
        Arg,
    }

    /// The seidel row-sweep shape, canonical rustc while-form:
    ///   entry:      j0 = 1; br header(j0)
    ///   header(j):  cond = icmp ult j, bound; cond_br cond, body, exit
    ///   body:       t = base + j; g0 = &p[t]
    ///               l1 = load &p[t-1]; l0 = load g0
    ///               s = l1 + l0; store s -> g0; j2 = j + 1; br header(j2)
    ///   exit:       return base
    /// Returns (func, entry, header, body, values...) for assertions.
    #[allow(clippy::type_complexity)]
    fn seidel_loop(
        ctx: &mut Context,
        bound: Bound,
    ) -> (
        FuncOp,
        Ptr<BasicBlock>,
        Ptr<BasicBlock>,
        Ptr<BasicBlock>,
        Value, // stored value s
        Value, // p
        Value, // t (shared index root)
    ) {
        let i64_ty: TypeHandle = int_ty(ctx, 64).into();
        let ptr_ty: TypeHandle = PointerType::get(ctx, 0).into();
        let fn_ty = FuncType::get(ctx, i64_ty, vec![ptr_ty, i64_ty, i64_ty], false);
        let func = FuncOp::new(ctx, "seidel_row".try_into().unwrap(), fn_ty);
        func.set_attr_llvm_function_linkage(ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(ctx);
        let region = func.get_region(ctx).unwrap();
        let entry = func.get_entry_block(ctx).unwrap();
        let p = entry.deref(ctx).get_argument(0);
        let base = entry.deref(ctx).get_argument(1);
        let end_arg = entry.deref(ctx).get_argument(2);
        let header = BasicBlock::new(ctx, None, vec![i64_ty]);
        header.insert_at_back(region, ctx);
        let body = BasicBlock::new(ctx, None, vec![]);
        body.insert_at_back(region, ctx);
        let exit = BasicBlock::new(ctx, None, vec![]);
        exit.insert_at_back(region, ctx);

        let j0 = const_i64(ctx, entry, 1);
        let bound_v = match bound {
            Bound::Const(n) => const_i64(ctx, entry, n),
            Bound::Arg => end_arg,
        };
        BrOp::new(ctx, header, vec![j0])
            .get_operation()
            .insert_at_back(entry, ctx);

        let j = header.deref(ctx).get_argument(0);
        let cond = ICmpOp::new(ctx, ICmpPredicateAttr::ULT, j, bound_v);
        cond.get_operation().insert_at_back(header, ctx);
        let cond_v = cond.get_result(ctx);
        CondBrOp::new(ctx, cond_v, body, vec![], exit, vec![])
            .get_operation()
            .insert_at_back(header, ctx);

        let one = const_i64(ctx, body, 1);
        let t = AddOp::new_with_overflow_flag(ctx, base, j, Default::default());
        t.get_operation().insert_at_back(body, ctx);
        let t_v = t.get_result(ctx);
        let tm1 = SubOp::new_with_overflow_flag(ctx, t_v, one, Default::default());
        tm1.get_operation().insert_at_back(body, ctx);
        let tm1_v = tm1.get_result(ctx);
        let g1 = GetElementPtrOp::new(ctx, p, vec![GepIndex::Value(tm1_v)], i64_ty);
        g1.get_operation().insert_at_back(body, ctx);
        let g1_v = g1.get_result(ctx);
        let l1 = LoadOp::new(ctx, g1_v, i64_ty);
        l1.get_operation().insert_at_back(body, ctx);
        let l1_v = l1.get_result(ctx);
        let g0 = GetElementPtrOp::new(ctx, p, vec![GepIndex::Value(t_v)], i64_ty);
        g0.get_operation().insert_at_back(body, ctx);
        let g0_v = g0.get_result(ctx);
        let l0 = LoadOp::new(ctx, g0_v, i64_ty);
        l0.get_operation().insert_at_back(body, ctx);
        let l0_v = l0.get_result(ctx);
        let s = AddOp::new_with_overflow_flag(ctx, l1_v, l0_v, Default::default());
        s.get_operation().insert_at_back(body, ctx);
        let s_v = s.get_result(ctx);
        StoreOp::new(ctx, s_v, g0_v)
            .get_operation()
            .insert_at_back(body, ctx);
        let j2 = AddOp::new_with_overflow_flag(ctx, j, one, Default::default());
        j2.get_operation().insert_at_back(body, ctx);
        let j2_v = j2.get_result(ctx);
        BrOp::new(ctx, header, vec![j2_v])
            .get_operation()
            .insert_at_back(body, ctx);

        ReturnOp::new(ctx, Some(base))
            .get_operation()
            .insert_at_back(exit, ctx);
        (func, entry, header, body, s_v, p, t_v)
    }

    #[test]
    fn forwards_seidel_shape_with_provable_entry() {
        let mut ctx = Context::new();
        let (func, entry, header, body, s_v, _p, _t) = seidel_loop(&mut ctx, Bound::Const(8));
        let text = run_pass(&mut ctx, func);

        // The carried load is gone from the body; the center load stays.
        let body_text = block_ops_text(&ctx, body);
        assert_eq!(body_text.matches("llvm.load").count(), 1, "{text}");
        assert!(body_text.contains("llvm.store"), "{text}");
        // The preheader gained the initial load (unguarded: entry proven).
        let entry_text = block_ops_text(&ctx, entry);
        assert_eq!(entry_text.matches("llvm.load").count(), 1, "{text}");
        assert!(!entry_text.contains("llvm.cond_br"), "{text}");
        // Header carries the forwarded value; the backedge passes the
        // stored value.
        assert_eq!(header.deref(&ctx).get_num_arguments(), 2, "{text}");
        let latch_term = body.deref(&ctx).get_terminator(&ctx).unwrap();
        assert_eq!(latch_term.deref(&ctx).get_num_operands(), 2, "{text}");
        assert_eq!(latch_term.deref(&ctx).get_operand(1), s_v, "{text}");
        // Idempotent: a second run finds nothing.
        let text2 = run_pass(&mut ctx, FuncOp::from_operation(func.get_operation()));
        assert_eq!(text, text2);
    }

    #[test]
    fn forwards_runtime_bound_behind_a_guard() {
        let mut ctx = Context::new();
        let (func, entry, header, body, _s, _p, _t) = seidel_loop(&mut ctx, Bound::Arg);
        let region = func.get_region(&ctx).unwrap();
        let blocks_before: Vec<_> = region.deref(&ctx).iter(&ctx).collect();
        let text = run_pass(&mut ctx, func);

        // Body forwarded as in the provable case.
        let body_text = block_ops_text(&ctx, body);
        assert_eq!(body_text.matches("llvm.load").count(), 1, "{text}");
        assert_eq!(header.deref(&ctx).get_num_arguments(), 2, "{text}");
        // The preheader now ends in the guard (a clone of the loop's own
        // first exit test) and materializes the undef bypass value; the
        // initial load lives in a NEW block.
        let entry_text = block_ops_text(&ctx, entry);
        assert!(entry_text.contains("llvm.cond_br"), "{text}");
        assert!(entry_text.contains("llvm.icmp"), "{text}");
        assert!(entry_text.contains("llvm.undef"), "{text}");
        assert!(!entry_text.contains("llvm.load"), "{text}");
        let blocks_after: Vec<_> = region.deref(&ctx).iter(&ctx).collect();
        assert_eq!(blocks_after.len(), blocks_before.len() + 1, "{text}");
        let guard_block = *blocks_after
            .iter()
            .find(|b| !blocks_before.contains(b))
            .unwrap();
        let guard_text = block_ops_text(&ctx, guard_block);
        assert_eq!(guard_text.matches("llvm.load").count(), 1, "{text}");
        assert!(guard_text.contains("llvm.gep"), "{text}");
        // The guard block enters the header with the loaded value.
        let guard_term = guard_block.deref(&ctx).get_terminator(&ctx).unwrap();
        assert!(Operation::get_opid(guard_term, &ctx) == BrOp::get_opid_static());
        assert_eq!(guard_term.deref(&ctx).get_num_operands(), 2, "{text}");
    }

    #[test]
    fn bails_on_provably_zero_trip() {
        let mut ctx = Context::new();
        // j0 = 1, bound 1, ULT: the body never runs.
        let (func, _entry, header, body, _s, _p, _t) = seidel_loop(&mut ctx, Bound::Const(1));
        let text = run_pass(&mut ctx, func);
        assert_eq!(header.deref(&ctx).get_num_arguments(), 1, "{text}");
        assert_eq!(
            block_ops_text(&ctx, body).matches("llvm.load").count(),
            2,
            "{text}"
        );
    }

    #[test]
    fn bails_on_second_store_in_loop() {
        let mut ctx = Context::new();
        let (func, _entry, header, body, s_v, p, t_v) = seidel_loop(&mut ctx, Bound::Const(8));
        let i64_ty: TypeHandle = int_ty(&mut ctx, 64).into();
        // A second store through another GEP off the same base: may alias
        // the carried slot across iterations.
        let terminator = body.deref(&ctx).get_terminator(&ctx).unwrap();
        let g2 = GetElementPtrOp::new(&mut ctx, p, vec![GepIndex::Value(t_v)], i64_ty);
        g2.get_operation().insert_before(&ctx, terminator);
        let g2_v = g2.get_result(&ctx);
        StoreOp::new(&mut ctx, s_v, g2_v)
            .get_operation()
            .insert_before(&ctx, terminator);
        let text = run_pass(&mut ctx, func);
        assert_eq!(header.deref(&ctx).get_num_arguments(), 1, "{text}");
        assert_eq!(
            block_ops_text(&ctx, body).matches("llvm.load").count(),
            2,
            "{text}"
        );
    }

    #[test]
    fn bails_on_call_in_loop() {
        let mut ctx = Context::new();
        let (func, _entry, header, body, _s, _p, _t) = seidel_loop(&mut ctx, Bound::Const(8));
        let i64_ty: TypeHandle = int_ty(&mut ctx, 64).into();
        let terminator = body.deref(&ctx).get_terminator(&ctx).unwrap();
        let void_fn = FuncType::get(&mut ctx, i64_ty, vec![], false);
        let call = CallOp::new(
            &mut ctx,
            pliron::builtin::op_interfaces::CallOpCallable::Direct("opaque".try_into().unwrap()),
            void_fn,
            vec![],
        );
        call.get_operation().insert_before(&ctx, terminator);
        let text = run_pass(&mut ctx, func);
        assert_eq!(header.deref(&ctx).get_num_arguments(), 1, "{text}");
        assert_eq!(
            block_ops_text(&ctx, body).matches("llvm.load").count(),
            2,
            "{text}"
        );
    }

    #[test]
    fn bails_on_non_affine_index() {
        // Same shape but t = j * j: the root's derivative w.r.t. the iv is
        // not a constant, so "one iteration back" is not a fixed shift.
        let mut ctx = Context::new();
        let i64_ty: TypeHandle = int_ty(&mut ctx, 64).into();
        let ptr_ty: TypeHandle = PointerType::get(&mut ctx, 0).into();
        let fn_ty = FuncType::get(&mut ctx, i64_ty, vec![ptr_ty, i64_ty], false);
        let func = FuncOp::new(&mut ctx, "sq".try_into().unwrap(), fn_ty);
        func.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(&mut ctx);
        let region = func.get_region(&ctx).unwrap();
        let entry = func.get_entry_block(&ctx).unwrap();
        let p = entry.deref(&ctx).get_argument(0);
        let header = BasicBlock::new(&mut ctx, None, vec![i64_ty]);
        header.insert_at_back(region, &ctx);
        let body = BasicBlock::new(&mut ctx, None, vec![]);
        body.insert_at_back(region, &ctx);
        let exit = BasicBlock::new(&mut ctx, None, vec![]);
        exit.insert_at_back(region, &ctx);
        let j0 = const_i64(&mut ctx, entry, 1);
        let bound = const_i64(&mut ctx, entry, 8);
        BrOp::new(&mut ctx, header, vec![j0])
            .get_operation()
            .insert_at_back(entry, &ctx);
        let j = header.deref(&ctx).get_argument(0);
        let cond = ICmpOp::new(&mut ctx, ICmpPredicateAttr::ULT, j, bound);
        cond.get_operation().insert_at_back(header, &ctx);
        let cond_v = cond.get_result(&ctx);
        CondBrOp::new(&mut ctx, cond_v, body, vec![], exit, vec![])
            .get_operation()
            .insert_at_back(header, &ctx);
        let one = const_i64(&mut ctx, body, 1);
        let t = MulOp::new_with_overflow_flag(&mut ctx, j, j, Default::default());
        t.get_operation().insert_at_back(body, &ctx);
        let t_v = t.get_result(&ctx);
        let tm1 = SubOp::new_with_overflow_flag(&mut ctx, t_v, one, Default::default());
        tm1.get_operation().insert_at_back(body, &ctx);
        let tm1_v = tm1.get_result(&ctx);
        let g1 = GetElementPtrOp::new(&mut ctx, p, vec![GepIndex::Value(tm1_v)], i64_ty);
        g1.get_operation().insert_at_back(body, &ctx);
        let g1_v = g1.get_result(&ctx);
        let l1 = LoadOp::new(&mut ctx, g1_v, i64_ty);
        l1.get_operation().insert_at_back(body, &ctx);
        let l1_v = l1.get_result(&ctx);
        let g0 = GetElementPtrOp::new(&mut ctx, p, vec![GepIndex::Value(t_v)], i64_ty);
        g0.get_operation().insert_at_back(body, &ctx);
        let g0_v = g0.get_result(&ctx);
        StoreOp::new(&mut ctx, l1_v, g0_v)
            .get_operation()
            .insert_at_back(body, &ctx);
        let j2 = AddOp::new_with_overflow_flag(&mut ctx, j, one, Default::default());
        j2.get_operation().insert_at_back(body, &ctx);
        let j2_v = j2.get_result(&ctx);
        BrOp::new(&mut ctx, header, vec![j2_v])
            .get_operation()
            .insert_at_back(body, &ctx);
        ReturnOp::new(&mut ctx, Some(j0))
            .get_operation()
            .insert_at_back(exit, &ctx);

        let text = run_pass(&mut ctx, func);
        assert_eq!(header.deref(&ctx).get_num_arguments(), 1, "{text}");
        assert!(block_ops_text(&ctx, body).contains("llvm.load"), "{text}");
    }

    #[test]
    fn bails_on_unclonable_runtime_condition() {
        // Runtime-bound while loop whose exit condition is loaded from
        // memory: neither provable entry nor a clonable pure guard.
        let mut ctx = Context::new();
        let (func, _entry, header, body, _s, _p, _t) = seidel_loop(&mut ctx, Bound::Arg);
        // Replace the header's icmp condition with a load-derived i1.
        let i1: TypeHandle = int_ty(&mut ctx, 1).into();
        let q = func
            .get_entry_block(&ctx)
            .unwrap()
            .deref(&ctx)
            .get_argument(0);
        let header_term = header.deref(&ctx).get_terminator(&ctx).unwrap();
        let flag = LoadOp::new(&mut ctx, q, i1);
        flag.get_operation().insert_before(&ctx, header_term);
        let flag_v = flag.get_result(&ctx);
        let cbr = CondBrOp::from_operation(header_term);
        let succ0 = cbr.get_operation().deref(&ctx).get_successor(0);
        let succ1 = cbr.get_operation().deref(&ctx).get_successor(1);
        let new_term = CondBrOp::new(&mut ctx, flag_v, succ0, vec![], succ1, vec![])
            .get_operation();
        new_term.insert_before(&ctx, header_term);
        Operation::erase(header_term, &mut ctx);

        let text = run_pass(&mut ctx, func);
        assert_eq!(header.deref(&ctx).get_num_arguments(), 1, "{text}");
        assert_eq!(
            block_ops_text(&ctx, body).matches("llvm.load").count(),
            2,
            "{text}"
        );
    }

    #[test]
    fn bails_on_load_from_different_base() {
        // load q[t-1], store p[t]: different bases never match.
        let mut ctx = Context::new();
        let i64_ty: TypeHandle = int_ty(&mut ctx, 64).into();
        let ptr_ty: TypeHandle = PointerType::get(&mut ctx, 0).into();
        let fn_ty = FuncType::get(&mut ctx, i64_ty, vec![ptr_ty, ptr_ty, i64_ty], false);
        let func = FuncOp::new(&mut ctx, "db".try_into().unwrap(), fn_ty);
        func.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(&mut ctx);
        let region = func.get_region(&ctx).unwrap();
        let entry = func.get_entry_block(&ctx).unwrap();
        let p = entry.deref(&ctx).get_argument(0);
        let q = entry.deref(&ctx).get_argument(1);
        let base = entry.deref(&ctx).get_argument(2);
        let header = BasicBlock::new(&mut ctx, None, vec![i64_ty]);
        header.insert_at_back(region, &ctx);
        let body = BasicBlock::new(&mut ctx, None, vec![]);
        body.insert_at_back(region, &ctx);
        let exit = BasicBlock::new(&mut ctx, None, vec![]);
        exit.insert_at_back(region, &ctx);
        let j0 = const_i64(&mut ctx, entry, 1);
        let bound = const_i64(&mut ctx, entry, 8);
        BrOp::new(&mut ctx, header, vec![j0])
            .get_operation()
            .insert_at_back(entry, &ctx);
        let j = header.deref(&ctx).get_argument(0);
        let cond = ICmpOp::new(&mut ctx, ICmpPredicateAttr::ULT, j, bound);
        cond.get_operation().insert_at_back(header, &ctx);
        let cond_v = cond.get_result(&ctx);
        CondBrOp::new(&mut ctx, cond_v, body, vec![], exit, vec![])
            .get_operation()
            .insert_at_back(header, &ctx);
        let one = const_i64(&mut ctx, body, 1);
        let t = AddOp::new_with_overflow_flag(&mut ctx, base, j, Default::default());
        t.get_operation().insert_at_back(body, &ctx);
        let t_v = t.get_result(&ctx);
        let tm1 = SubOp::new_with_overflow_flag(&mut ctx, t_v, one, Default::default());
        tm1.get_operation().insert_at_back(body, &ctx);
        let tm1_v = tm1.get_result(&ctx);
        let g1 = GetElementPtrOp::new(&mut ctx, q, vec![GepIndex::Value(tm1_v)], i64_ty);
        g1.get_operation().insert_at_back(body, &ctx);
        let g1_v = g1.get_result(&ctx);
        let l1 = LoadOp::new(&mut ctx, g1_v, i64_ty);
        l1.get_operation().insert_at_back(body, &ctx);
        let l1_v = l1.get_result(&ctx);
        let g0 = GetElementPtrOp::new(&mut ctx, p, vec![GepIndex::Value(t_v)], i64_ty);
        g0.get_operation().insert_at_back(body, &ctx);
        let g0_v = g0.get_result(&ctx);
        StoreOp::new(&mut ctx, l1_v, g0_v)
            .get_operation()
            .insert_at_back(body, &ctx);
        let j2 = AddOp::new_with_overflow_flag(&mut ctx, j, one, Default::default());
        j2.get_operation().insert_at_back(body, &ctx);
        let j2_v = j2.get_result(&ctx);
        BrOp::new(&mut ctx, header, vec![j2_v])
            .get_operation()
            .insert_at_back(body, &ctx);
        ReturnOp::new(&mut ctx, Some(j0))
            .get_operation()
            .insert_at_back(exit, &ctx);

        let text = run_pass(&mut ctx, func);
        assert_eq!(header.deref(&ctx).get_num_arguments(), 1, "{text}");
        assert!(block_ops_text(&ctx, body).contains("llvm.load"), "{text}");
    }

    #[test]
    fn forwards_do_while_without_guard_even_with_runtime_bound() {
        // Single-block loop: the body runs whenever the preheader runs,
        // so the initial load needs no guard even for a runtime bound.
        let mut ctx = Context::new();
        let i64_ty: TypeHandle = int_ty(&mut ctx, 64).into();
        let ptr_ty: TypeHandle = PointerType::get(&mut ctx, 0).into();
        let fn_ty = FuncType::get(&mut ctx, i64_ty, vec![ptr_ty, i64_ty, i64_ty], false);
        let func = FuncOp::new(&mut ctx, "dw".try_into().unwrap(), fn_ty);
        func.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(&mut ctx);
        let region = func.get_region(&ctx).unwrap();
        let entry = func.get_entry_block(&ctx).unwrap();
        let p = entry.deref(&ctx).get_argument(0);
        let base = entry.deref(&ctx).get_argument(1);
        let end = entry.deref(&ctx).get_argument(2);
        let header = BasicBlock::new(&mut ctx, None, vec![i64_ty]);
        header.insert_at_back(region, &ctx);
        let exit = BasicBlock::new(&mut ctx, None, vec![]);
        exit.insert_at_back(region, &ctx);
        let j0 = const_i64(&mut ctx, entry, 1);
        BrOp::new(&mut ctx, header, vec![j0])
            .get_operation()
            .insert_at_back(entry, &ctx);

        let j = header.deref(&ctx).get_argument(0);
        let one = const_i64(&mut ctx, header, 1);
        let t = AddOp::new_with_overflow_flag(&mut ctx, base, j, Default::default());
        t.get_operation().insert_at_back(header, &ctx);
        let t_v = t.get_result(&ctx);
        let tm1 = SubOp::new_with_overflow_flag(&mut ctx, t_v, one, Default::default());
        tm1.get_operation().insert_at_back(header, &ctx);
        let tm1_v = tm1.get_result(&ctx);
        let g1 = GetElementPtrOp::new(&mut ctx, p, vec![GepIndex::Value(tm1_v)], i64_ty);
        g1.get_operation().insert_at_back(header, &ctx);
        let g1_v = g1.get_result(&ctx);
        let l1 = LoadOp::new(&mut ctx, g1_v, i64_ty);
        l1.get_operation().insert_at_back(header, &ctx);
        let l1_v = l1.get_result(&ctx);
        let g0 = GetElementPtrOp::new(&mut ctx, p, vec![GepIndex::Value(t_v)], i64_ty);
        g0.get_operation().insert_at_back(header, &ctx);
        let g0_v = g0.get_result(&ctx);
        let l0 = LoadOp::new(&mut ctx, g0_v, i64_ty);
        l0.get_operation().insert_at_back(header, &ctx);
        let l0_v = l0.get_result(&ctx);
        let s = AddOp::new_with_overflow_flag(&mut ctx, l1_v, l0_v, Default::default());
        s.get_operation().insert_at_back(header, &ctx);
        let s_v = s.get_result(&ctx);
        StoreOp::new(&mut ctx, s_v, g0_v)
            .get_operation()
            .insert_at_back(header, &ctx);
        let j2 = AddOp::new_with_overflow_flag(&mut ctx, j, one, Default::default());
        j2.get_operation().insert_at_back(header, &ctx);
        let j2_v = j2.get_result(&ctx);
        let cond = ICmpOp::new(&mut ctx, ICmpPredicateAttr::ULT, j2_v, end);
        cond.get_operation().insert_at_back(header, &ctx);
        let cond_v = cond.get_result(&ctx);
        CondBrOp::new(&mut ctx, cond_v, header, vec![j2_v], exit, vec![])
            .get_operation()
            .insert_at_back(header, &ctx);
        ReturnOp::new(&mut ctx, Some(base))
            .get_operation()
            .insert_at_back(exit, &ctx);

        let text = run_pass(&mut ctx, func);
        let header_text = block_ops_text(&ctx, header);
        let entry_text = block_ops_text(&ctx, entry);
        assert_eq!(header.deref(&ctx).get_num_arguments(), 2, "{text}");
        assert_eq!(header_text.matches("llvm.load").count(), 1, "{text}");
        assert_eq!(entry_text.matches("llvm.load").count(), 1, "{text}");
        assert!(!entry_text.contains("llvm.cond_br"), "{text}");
        // Backedge (the cond_br's in-loop edge) passes the stored value:
        // operands are [cond, j2, s | (none for exit)].
        let term = header.deref(&ctx).get_terminator(&ctx).unwrap();
        assert_eq!(term.deref(&ctx).get_num_operands(), 3, "{text}");
        assert_eq!(term.deref(&ctx).get_operand(2), s_v, "{text}");
    }

    #[test]
    fn stamps_attribution_and_keeps_op_ids_unique() {
        use crate::dialects::builtin::ops::ModuleOp;
        use crate::dialects::builtin::op_interfaces::SingleBlockRegionInterface as _;
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "m".try_into().unwrap());
        let module_block = module.get_body(&ctx, 0);
        let (func, _entry, header, _body, _s, _p, _t) = seidel_loop(&mut ctx, Bound::Arg);
        func.get_operation().insert_at_back(module_block, &ctx);
        opmap::assign_op_ids(&mut ctx, module.get_operation()).unwrap();

        LLVMLoopCarriedFwdPass
            .run(module.get_operation(), &mut ctx, &mut AnalysisManager::default())
            .unwrap();
        assert_eq!(header.deref(&ctx).get_num_arguments(), 2);

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
        assert!(derived_clones > 0, "the rewrite must stamp derived clones");
    }

    // ------------------------------------------------------------------
    // Gate 3: the REAL seidel_2d kernel. The IR below is the verbatim
    // CRABBIT_EMIT_IR output (post lower-dialect-mir, pre mid-end) for the
    // seidel_2d device kernel's row sweep (thread-id intrinsics replaced
    // by a plain `i` argument; the loop, indexing and recurrence are the
    // kernel's own, with `cols` a RUNTIME argument). The test replays the
    // exact mid-end prefix that precedes this pass in the pipeline
    // (mod.rs order up to the second gvn), then shows the pass fires: the
    // loop-carried load is gone from the loop, replaced by the carried
    // block argument, with the guarded initial load outside the loop.
    // ------------------------------------------------------------------

    fn parse_module(ctx: &mut Context, text: &str) -> Ptr<Operation> {
        use pliron::combine::Parser as _;
        use pliron::location::Source;
        use pliron::operation::OperationParserConfig;
        use pliron::parsable::{Parsable, State, state_stream_from_iterator};
        let state = State::new(ctx, Source::InMemory);
        let stream = state_stream_from_iterator(text.chars(), state);
        let config = OperationParserConfig {
            look_for_outlined_attrs: false,
        };
        <Operation as Parsable>::parser(config)
            .parse(stream)
            .map(|(op, _)| op)
            .expect("embedded emitted IR must parse")
    }

    /// The mid-end prefix that runs before this pass in
    /// [add_llvm_midend_passes](super::super::add_llvm_midend_passes)
    /// (everything up to and including the second gvn).
    fn run_midend_prefix(ctx: &mut Context, root: Ptr<Operation>) {
        use super::super::{
            div_strength_reduce, inline, op_ids, pin_type_punned_slots, simplify,
            simplify_cfg, sroa, gvn as gvn_mod, licm as licm_mod,
        };
        use crate::conversion::pass::{Mem2RegPass, Passes};
        let mut passes = Passes::default();
        passes.add_pass(op_ids::LlvmOpIdPass);
        passes.add_pass(inline::LLVMInlinePass::default());
        passes.add_pass(simplify::LLVMSimplifyPass);
        passes.add_pass(simplify_cfg::LLVMSimplifyCfgPass);
        passes.add_pass(sroa::LLVMSroaPass);
        passes.add_pass(simplify::LLVMSimplifyPass);
        passes.add_pass(pin_type_punned_slots::LLVMPinTypePunnedSlotsPass);
        passes.add_pass(Mem2RegPass);
        passes.add_pass(simplify::LLVMSimplifyPass);
        passes.add_pass(simplify_cfg::LLVMSimplifyCfgPass);
        passes.add_pass(simplify::LLVMSimplifyPass);
        passes.add_pass(sroa::LLVMSroaPass);
        passes.add_pass(simplify::LLVMSimplifyPass);
        passes.add_pass(pin_type_punned_slots::LLVMPinTypePunnedSlotsPass);
        passes.add_pass(Mem2RegPass);
        passes.add_pass(simplify::LLVMSimplifyPass);
        passes.add_pass(simplify_cfg::LLVMSimplifyCfgPass);
        passes.add_pass(simplify::LLVMSimplifyPass);
        passes.add_pass(gvn_mod::LLVMGvnPass);
        passes.add_pass(div_strength_reduce::LLVMDivStrengthReducePass);
        passes.add_pass(licm_mod::LLVMLicmPass);
        passes.add_pass(gvn_mod::LLVMGvnPass);
        passes
            .run(root, ctx, &mut AnalysisManager::default())
            .unwrap();
    }

    #[test]
    fn fires_on_the_real_seidel_2d_kernel() {
        let mut ctx = Context::new();
        let module = parse_module(&mut ctx, SEIDEL_2D_EMITTED_IR);
        run_midend_prefix(&mut ctx, module);
        let before = format!("{}", module.disp(&ctx));
        if std::env::var("LCFWD_DUMP").is_ok() {
            eprintln!("== after mid-end prefix ==\n{before}");
        }
        assert!(before.matches("llvm.store").count() >= 1, "{before}");

        let result = LLVMLoopCarriedFwdPass
            .run(module, &mut ctx, &mut AnalysisManager::default())
            .unwrap();
        assert!(
            matches!(result.ir_changed, pliron::irbuild::IRStatus::Changed),
            "pass must fire on the real kernel:\n{before}"
        );
        let after = format!("{}", module.disp(&ctx));
        if std::env::var("LCFWD_DUMP").is_ok() {
            eprintln!("== after loop-carried-fwd ==\n{after}");
        }

        // Block-level proof: the store's block lost exactly its
        // loop-carried load; the initial load appears outside the loop.
        let loads_by_store_block = |ctx: &Context, root: Ptr<Operation>| -> (usize, usize) {
            let mut in_store_block = 0;
            let mut elsewhere = 0;
            for func in collect_functions(ctx, root) {
                let Some(region) = func.get_region(ctx) else { continue };
                for block in region.deref(ctx).iter(ctx) {
                    let text = block_ops_text(ctx, block);
                    let loads = text.matches("llvm.load").count();
                    if text.contains("llvm.store") {
                        in_store_block += loads;
                    } else {
                        elsewhere += loads;
                    }
                }
            }
            (in_store_block, elsewhere)
        };
        // Re-parse `before` for the pre-pass census (the module was
        // mutated in place).
        let mut ctx_before = Context::new();
        let module_before = parse_module(&mut ctx_before, &before);
        let (loop_loads_before, outside_before) =
            loads_by_store_block(&ctx_before, module_before);
        let (loop_loads_after, outside_after) = loads_by_store_block(&ctx, module);
        assert_eq!(
            loop_loads_after,
            loop_loads_before - 1,
            "the carried load must leave the loop:\n{after}"
        );
        assert_eq!(
            outside_after,
            outside_before + 1,
            "the initial load must appear outside the loop:\n{after}"
        );
        // `cols` is a runtime argument, so the entry must be guarded: the
        // guard's undef bypass value is the tell.
        assert!(after.contains("llvm.undef"), "{after}");
        // The rewritten module must still round-trip through the typed
        // printer/parser (coherent block args and successor operands).
        let mut ctx_after = Context::new();
        let _ = parse_module(&mut ctx_after, &after);
        // Idempotence on the real kernel.
        let again = LLVMLoopCarriedFwdPass
            .run(module, &mut ctx, &mut AnalysisManager::default())
            .unwrap();
        assert!(matches!(
            again.ir_changed,
            pliron::irbuild::IRStatus::Unchanged
        ));
    }

    const SEIDEL_2D_EMITTED_IR: &str = r"
builtin.module @rust_kernels 
{
  ^block2v1():
    llvm.func @seidel_2d: llvm.func <llvm.void (llvm.ptr (0), builtin.integer i32, builtin.integer i32, builtin.integer i32) variadic = false>
      [llvm_function_linkage: llvm.linkage ExternalLinkage] 
    {
      ^entry_block20v3(v165: llvm.ptr (0), v166: builtin.integer i32, v167: builtin.integer i32, v168: builtin.integer i32):
        llvm.br ^entry_block3v1(v165, v166, v167, v168)

      ^entry_block3v1(v0: llvm.ptr (0), v1: builtin.integer i32, v2: builtin.integer i32, v3: builtin.integer i32):
        v169 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v170 = llvm.alloca [llvm.ptr (0) x v169]  : llvm.ptr (0);
        v171 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v172 = llvm.alloca [builtin.integer i32 x v171]  : llvm.ptr (0);
        v173 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v174 = llvm.alloca [builtin.integer i32 x v173]  : llvm.ptr (0);
        v175 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v176 = llvm.alloca [builtin.integer i32 x v175]  : llvm.ptr (0);
        v177 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v178 = llvm.alloca [builtin.integer i8 x v177]  : llvm.ptr (0);
        v179 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v180 = llvm.alloca [builtin.integer i8 x v179]  : llvm.ptr (0);
        v181 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v182 = llvm.alloca [builtin.integer i32 x v181]  : llvm.ptr (0);
        v183 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v184 = llvm.alloca [builtin.integer i64 x v183]  : llvm.ptr (0);
        v185 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v186 = llvm.alloca [builtin.integer i64 x v185]  : llvm.ptr (0);
        v187 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v188 = llvm.alloca [builtin.integer i64 x v187]  : llvm.ptr (0);
        v189 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v190 = llvm.alloca [builtin.integer i64 x v189]  : llvm.ptr (0);
        v191 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v192 = llvm.alloca [builtin.integer i64 x v191]  : llvm.ptr (0);
        v193 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v194 = llvm.alloca [builtin.integer i8 x v193]  : llvm.ptr (0);
        v195 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v196 = llvm.alloca [builtin.integer i64 x v195]  : llvm.ptr (0);
        v197 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v198 = llvm.alloca [builtin.integer i32 x v197]  : llvm.ptr (0);
        v199 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v200 = llvm.alloca [builtin.integer i32 x v199]  : llvm.ptr (0);
        v201 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v202 = llvm.alloca [builtin.integer i32 x v201]  : llvm.ptr (0);
        v203 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v204 = llvm.alloca [builtin.integer i32 x v203]  : llvm.ptr (0);
        v205 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v206 = llvm.alloca [llvm.ptr (0) x v205]  : llvm.ptr (0);
        v207 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v208 = llvm.alloca [builtin.integer i64 x v207]  : llvm.ptr (0);
        v209 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v210 = llvm.alloca [builtin.integer i64 x v209]  : llvm.ptr (0);
        v211 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v212 = llvm.alloca [builtin.integer i64 x v211]  : llvm.ptr (0);
        v213 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v214 = llvm.alloca [builtin.integer i32 x v213]  : llvm.ptr (0);
        v215 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v216 = llvm.alloca [llvm.ptr (0) x v215]  : llvm.ptr (0);
        v217 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v218 = llvm.alloca [builtin.integer i64 x v217]  : llvm.ptr (0);
        v219 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v220 = llvm.alloca [builtin.integer i64 x v219]  : llvm.ptr (0);
        v221 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v222 = llvm.alloca [builtin.integer i32 x v221]  : llvm.ptr (0);
        v223 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v224 = llvm.alloca [llvm.ptr (0) x v223]  : llvm.ptr (0);
        v225 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v226 = llvm.alloca [builtin.integer i64 x v225]  : llvm.ptr (0);
        v227 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v228 = llvm.alloca [builtin.integer i64 x v227]  : llvm.ptr (0);
        v229 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v230 = llvm.alloca [builtin.integer i64 x v229]  : llvm.ptr (0);
        v231 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v232 = llvm.alloca [llvm.ptr (0) x v231]  : llvm.ptr (0);
        v233 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v234 = llvm.alloca [builtin.integer i64 x v233]  : llvm.ptr (0);
        v235 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v236 = llvm.alloca [builtin.integer i64 x v235]  : llvm.ptr (0);
        llvm.store *v170 <- v0 ;
        llvm.store *v172 <- v1 ;
        llvm.store *v174 <- v2 ;
        llvm.store *v176 <- v3 ;
        v237 = llvm.load v172  : builtin.integer i32;
        v238 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v239 = llvm.icmp v237 <UGE> v238 : builtin.integer i1;
        v240 = llvm.zext <nneg=false> v239 to builtin.integer i8;
        llvm.store *v178 <- v240 ;
        v241 = llvm.load v178  : builtin.integer i8;
        v242 = llvm.trunc v241 to builtin.integer i1;
        v243 = builtin.constant <builtin.integer <0: i1>> : builtin.integer i1;
        v244 = llvm.icmp v242 <EQ> v243 : builtin.integer i1;
        llvm.cond_br if v244 ^bb11_block14v1() else ^bb1_block4v1()

      ^bb1_block4v1():
        v245 = llvm.load v172  : builtin.integer i32;
        v246 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v247 = llvm.add v245, v246 <{nsw=false,nuw=false}>: builtin.integer i32;
        llvm.store *v182 <- v247 ;
        v248 = llvm.load v182  : builtin.integer i32;
        v249 = llvm.load v174  : builtin.integer i32;
        v250 = llvm.icmp v248 <ULT> v249 : builtin.integer i1;
        v251 = llvm.zext <nneg=false> v250 to builtin.integer i8;
        llvm.store *v180 <- v251 ;
        v252 = llvm.load v180  : builtin.integer i8;
        v253 = llvm.trunc v252 to builtin.integer i1;
        v254 = builtin.constant <builtin.integer <0: i1>> : builtin.integer i1;
        v255 = llvm.icmp v253 <EQ> v254 : builtin.integer i1;
        llvm.cond_br if v255 ^bb10_block13v1() else ^bb2_block5v1()

      ^bb2_block5v1():
        v256 = llvm.load v172  : builtin.integer i32;
        v257 = llvm.zext <nneg=false> v256 to builtin.integer i64;
        llvm.store *v186 <- v257 ;
        v258 = llvm.load v176  : builtin.integer i32;
        v259 = llvm.zext <nneg=false> v258 to builtin.integer i64;
        llvm.store *v188 <- v259 ;
        v260 = llvm.load v186  : builtin.integer i64;
        v261 = llvm.load v188  : builtin.integer i64;
        v262 = llvm.mul v260, v261 <{nsw=false,nuw=false}>: builtin.integer i64;
        llvm.store *v184 <- v262 ;
        v263 = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64;
        llvm.store *v190 <- v263 ;
        v264 = llvm.load v188  : builtin.integer i64;
        v265 = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64;
        v266 = llvm.sub v264, v265 <{nsw=false,nuw=false}>: builtin.integer i64;
        llvm.store *v192 <- v266 ;
        llvm.br ^bb3_block6v1()

      ^bb3_block6v1():
        v267 = llvm.load v190  : builtin.integer i64;
        llvm.store *v196 <- v267 ;
        v268 = llvm.load v196  : builtin.integer i64;
        v269 = llvm.load v192  : builtin.integer i64;
        v270 = llvm.icmp v268 <ULT> v269 : builtin.integer i1;
        v271 = llvm.zext <nneg=false> v270 to builtin.integer i8;
        llvm.store *v194 <- v271 ;
        v272 = llvm.load v194  : builtin.integer i8;
        v273 = llvm.trunc v272 to builtin.integer i1;
        v274 = builtin.constant <builtin.integer <0: i1>> : builtin.integer i1;
        v275 = llvm.icmp v273 <EQ> v274 : builtin.integer i1;
        llvm.cond_br if v275 ^bb9_block12v1() else ^bb4_block7v1()

      ^bb4_block7v1():
        v276 = llvm.load v190  : builtin.integer i64;
        llvm.store *v212 <- v276 ;
        v277 = llvm.load v184  : builtin.integer i64;
        v278 = llvm.load v212  : builtin.integer i64;
        v279 = llvm.add v277, v278 <{nsw=false,nuw=false}>: builtin.integer i64;
        llvm.store *v210 <- v279 ;
        v280 = llvm.load v210  : builtin.integer i64;
        v281 = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64;
        v282 = llvm.sub v280, v281 <{nsw=false,nuw=false}>: builtin.integer i64;
        llvm.store *v208 <- v282 ;
        v283 = llvm.load v170  : llvm.ptr (0);
        v284 = llvm.load v208  : builtin.integer i64;
        v285 = llvm.call @_RNvMNtNtCsbBDxv2Oq2Kj_4core3ptr7mut_ptrOl3addCsbCTi45toqUG_11seidel_mini (v283, v284) : llvm.func <llvm.ptr (0)(llvm.ptr (0), builtin.integer i64) variadic = false>;
        llvm.store *v206 <- v285 ;
        llvm.br ^bb5_block8v1()

      ^bb5_block8v1():
        v286 = llvm.load v206  : llvm.ptr (0);
        v287 = llvm.load v286  : builtin.integer i32;
        llvm.store *v204 <- v287 ;
        v288 = llvm.load v190  : builtin.integer i64;
        llvm.store *v220 <- v288 ;
        v289 = llvm.load v184  : builtin.integer i64;
        v290 = llvm.load v220  : builtin.integer i64;
        v291 = llvm.add v289, v290 <{nsw=false,nuw=false}>: builtin.integer i64;
        llvm.store *v218 <- v291 ;
        v292 = llvm.load v170  : llvm.ptr (0);
        v293 = llvm.load v218  : builtin.integer i64;
        v294 = llvm.call @_RNvMNtNtCsbBDxv2Oq2Kj_4core3ptr7mut_ptrOl3addCsbCTi45toqUG_11seidel_mini (v292, v293) : llvm.func <llvm.ptr (0)(llvm.ptr (0), builtin.integer i64) variadic = false>;
        llvm.store *v216 <- v294 ;
        llvm.br ^bb6_block9v1()

      ^bb6_block9v1():
        v295 = llvm.load v216  : llvm.ptr (0);
        v296 = llvm.load v295  : builtin.integer i32;
        llvm.store *v214 <- v296 ;
        v297 = llvm.load v204  : builtin.integer i32;
        v298 = llvm.load v214  : builtin.integer i32;
        v299 = llvm.add v297, v298 <{nsw=false,nuw=false}>: builtin.integer i32;
        llvm.store *v202 <- v299 ;
        v300 = llvm.load v190  : builtin.integer i64;
        llvm.store *v230 <- v300 ;
        v301 = llvm.load v184  : builtin.integer i64;
        v302 = llvm.load v230  : builtin.integer i64;
        v303 = llvm.add v301, v302 <{nsw=false,nuw=false}>: builtin.integer i64;
        llvm.store *v228 <- v303 ;
        v304 = llvm.load v228  : builtin.integer i64;
        v305 = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64;
        v306 = llvm.add v304, v305 <{nsw=false,nuw=false}>: builtin.integer i64;
        llvm.store *v226 <- v306 ;
        v307 = llvm.load v170  : llvm.ptr (0);
        v308 = llvm.load v226  : builtin.integer i64;
        v309 = llvm.call @_RNvMNtNtCsbBDxv2Oq2Kj_4core3ptr7mut_ptrOl3addCsbCTi45toqUG_11seidel_mini (v307, v308) : llvm.func <llvm.ptr (0)(llvm.ptr (0), builtin.integer i64) variadic = false>;
        llvm.store *v224 <- v309 ;
        llvm.br ^bb7_block10v1()

      ^bb7_block10v1():
        v310 = llvm.load v224  : llvm.ptr (0);
        v311 = llvm.load v310  : builtin.integer i32;
        llvm.store *v222 <- v311 ;
        v312 = llvm.load v202  : builtin.integer i32;
        v313 = llvm.load v222  : builtin.integer i32;
        v314 = llvm.add v312, v313 <{nsw=false,nuw=false}>: builtin.integer i32;
        llvm.store *v200 <- v314 ;
        v315 = llvm.load v200  : builtin.integer i32;
        v316 = builtin.constant <builtin.integer <3: i32>> : builtin.integer i32;
        v317 = llvm.sdiv v315, v316 : builtin.integer i32;
        llvm.store *v198 <- v317 ;
        v318 = llvm.load v190  : builtin.integer i64;
        llvm.store *v236 <- v318 ;
        v319 = llvm.load v184  : builtin.integer i64;
        v320 = llvm.load v236  : builtin.integer i64;
        v321 = llvm.add v319, v320 <{nsw=false,nuw=false}>: builtin.integer i64;
        llvm.store *v234 <- v321 ;
        v322 = llvm.load v170  : llvm.ptr (0);
        v323 = llvm.load v234  : builtin.integer i64;
        v324 = llvm.call @_RNvMNtNtCsbBDxv2Oq2Kj_4core3ptr7mut_ptrOl3addCsbCTi45toqUG_11seidel_mini (v322, v323) : llvm.func <llvm.ptr (0)(llvm.ptr (0), builtin.integer i64) variadic = false>;
        llvm.store *v232 <- v324 ;
        llvm.br ^bb8_block11v1()

      ^bb8_block11v1():
        v325 = llvm.load v198  : builtin.integer i32;
        v326 = llvm.load v232  : llvm.ptr (0);
        llvm.store *v326 <- v325 ;
        v327 = llvm.load v190  : builtin.integer i64;
        v328 = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64;
        v329 = llvm.add v327, v328 <{nsw=false,nuw=false}>: builtin.integer i64;
        llvm.store *v190 <- v329 ;
        llvm.br ^bb3_block6v1()

      ^bb9_block12v1():
        llvm.br ^bb11_block14v1()

      ^bb10_block13v1():
        llvm.br ^bb11_block14v1()

      ^bb11_block14v1():
        llvm.return 
    };
    llvm.func @_RNvMNtNtCsbBDxv2Oq2Kj_4core3ptr7mut_ptrOl3addCsbCTi45toqUG_11seidel_mini: llvm.func <llvm.ptr (0)(llvm.ptr (0), builtin.integer i64) variadic = false>
      [llvm_function_linkage: llvm.linkage InternalLinkage] 
    {
      ^entry_block19v3(v330: llvm.ptr (0), v331: builtin.integer i64):
        llvm.br ^entry_block15v1(v330, v331)

      ^entry_block15v1(v86: llvm.ptr (0), v87: builtin.integer i64):
        v332 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v333 = llvm.alloca [llvm.ptr (0) x v332]  : llvm.ptr (0);
        v334 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v335 = llvm.alloca [llvm.ptr (0) x v334]  : llvm.ptr (0);
        v336 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        v337 = llvm.alloca [builtin.integer i64 x v336]  : llvm.ptr (0);
        llvm.store *v335 <- v86 ;
        llvm.store *v337 <- v87 ;
        v338 = llvm.load v335  : llvm.ptr (0);
        v339 = llvm.load v337  : builtin.integer i64;
        v340 = builtin.constant <builtin.integer <4: i64>> : builtin.integer i64;
        v341 = llvm.mul v339, v340 <{nsw=false,nuw=false}>: builtin.integer i64;
        v95 = llvm.gep <builtin.integer ui8> (v338, v341)[OperandIdx(1)] : llvm.ptr (0);
        llvm.store *v333 <- v95 ;
        v342 = llvm.load v333  : llvm.ptr (0);
        llvm.return v342
    }
}";
}
