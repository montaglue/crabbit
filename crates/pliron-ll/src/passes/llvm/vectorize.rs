//! Innermost-loop SIMD vectorization for the llvm-dialect mid-end
//! (docs/MIDEND-PLAN.md, CPU SIMD gap): turn a canonical rustc `Range`
//! streaming loop into a 128-bit NEON vector loop plus the original loop as
//! a scalar epilogue.
//!
//! MVP scope, all deliberate:
//! - **Shape**: 2-block while-loops only — `header` holds exactly the
//!   exiting `icmp` (`iv < bound`, `ult`/`slt`, either operand order) and
//!   the `cond_br` whose TRUE edge enters the argument-less `body` block;
//!   the body's `br` is the only latch; a preheader exists (like
//!   [licm](super::licm)/[unroll](super::unroll)). The induction variable
//!   is a 64-bit header argument stepped by `+1`; `bound` is any
//!   loop-invariant value (runtime trips vectorize; the epilogue handles
//!   the remainder).
//! - **Lanes**: one element type per loop — f32/i32 (VF=4) or f64/i64
//!   (VF=2), always the full 128-bit vector.
//! - **Memory**: unit-stride only. Every load/store address must be
//!   `gep(base, [iv])` or `gep(base, [iv + inv])` with `base` and `inv`
//!   loop-invariant. A base that is stored through must not be accessed
//!   through any other address expression in the loop; DISTINCT bases are
//!   assumed non-aliasing when one is written — sound for rustc-generated
//!   code, where a mutable region is only reachable through one `&mut`
//!   (the same exclusivity rustc encodes as LLVM `noalias`).
//! - **Ops**: lane-wise add/sub/mul (no i64 mul — NEON has none),
//!   and/or/xor, shifts by in-range constants, and fadd/fsub/fmul/fdiv.
//! - **Reductions**: loop-carried accumulators updated by a single
//!   `add`/`fadd` become VF partial sums combined by `ll.vreduce` at the
//!   vector loop's exit. Integer reductions are exact; FADD reassociates
//!   the sum order, so it is emitted only under
//!   `CRABBIT_VECTORIZE_FP_REDUCE=1` (default OFF — exact-checksum
//!   consumers never see it; the corpus FP kernels are tolerance-checked).
//!
//! The transform NEVER touches the original loop: the preheader is
//! retargeted at a fresh vector preheader/loop, whose exit block enters the
//! original header with the remaining iteration count — so partial trips,
//! trip 0, and every live-out value keep their scalar semantics. The
//! vector loop's guard is `wrapping(bound - iv) >= VF`, which for the
//! accepted `lt` predicates is exactly "at least VF more scalar
//! iterations" and cannot overflow (see [analyze]'s trip-safety note).
//!
//! ADJOINT (backward attribution): every emitted op derives from the
//! scalar op it widens or replaces (vector body ops from their scalar
//! originals; the guard/branches from the loop's `cond_br`; iv stepping
//! and constants from the iv update). The epilogue keeps the original ops
//! and their ids untouched.

use rustc_hash::{FxHashMap, FxHashSet};
use std::num::NonZero;

use pliron::builtin::op_interfaces::{AtMostOneRegionInterface as _, BranchOpInterface as _};
use pliron_llvm::op_interfaces::{
    IntBinArithOpWithOverflowFlag as _, IsDeclaration as _,
};

use crate::{
    context::{Context, Ptr},
    conversion::pass::{AnalysisManager, Pass, PassResult, changed, unchanged},
    dialects::{
        builtin::{
            attributes::{FPDoubleAttr, FPSingleAttr, IntegerAttr},
            ops::ConstantOp,
            types::{IntegerType, Signedness},
        },
        llvm::{
            attributes::ICmpPredicateAttr,
            ops::{
                AShrOp, AddOp, AndOp, BrOp, CondBrOp, FAddOp, FDivOp, FMulOp, FSubOp,
                GepIndex, GetElementPtrOp, ICmpOp, LShrOp, LoadOp, MulOp, OrOp, ShlOp,
                StoreOp, SubOp, XorOp,
            },
            types::{VectorType, VectorTypeKind},
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
    linked_list::{ContainsLinkedList, LinkedList as _},
    ll::{
        VBinOpKindAttr, VReduceKindAttr, VectorizeEpilogueAttr,
        ops::{VBinOp, VLoadOp, VReduceOp, VSplatOp, VStoreOp},
    },
    target_profile::TargetProfile,
    utils::apint::APInt,
};

use super::{
    analysis::{dominator_tree, natural_loops, NaturalLoop},
    inline::collect_functions,
    midend_gate::midend_disabled,
    simplify::as_const_operand,
};
use crate::passes::aarch64::opmap;

pliron::dict_key!(ATTR_KEY_VECTORIZE_EPILOGUE, "ll_vectorize_epilogue");

/// Vectorized loops per function per run (each success rebuilds the loop
/// forest; epilogues are marked and never re-vectorized).
const MAX_LOOPS: usize = 64;

/// `CRABBIT_VECTORIZE_FP_REDUCE=1` opts into FP reductions (reassociation).
fn fp_reduce_enabled() -> bool {
    std::env::var("CRABBIT_VECTORIZE_FP_REDUCE").is_ok_and(|v| v == "1")
}

pub struct LLVMVectorizePass {
    /// SIMT targets get nothing from 128-bit CPU lanes, and the NVPTX
    /// translator does not know the vector ops: self-disable there.
    divergent_target: bool,
    /// The backend can lower the vector ops ([TargetProfile::simd128] —
    /// aarch64 only today; a vectorized module would fail x86_64 isel).
    simd128: bool,
}

impl LLVMVectorizePass {
    pub fn new(profile: &TargetProfile) -> Self {
        LLVMVectorizePass {
            divergent_target: profile.has_branch_divergence,
            simd128: profile.simd128,
        }
    }
}

impl Pass for LLVMVectorizePass {
    fn name(&self) -> &str {
        "llvm-vectorize"
    }

    fn run(
        &mut self,
        root: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> pliron::result::Result<PassResult> {
        if midend_disabled("vectorize") || self.divergent_target || !self.simd128 {
            return Ok(unchanged());
        }
        let fp_reduce = fp_reduce_enabled();
        let mut any = false;
        for func in collect_functions(ctx, root) {
            if func.is_declaration(ctx) {
                continue;
            }
            let region = func
                .get_region(ctx)
                .expect("llvm.func definition must have a body");
            any |= vectorize_region(ctx, region, fp_reduce);
        }
        Ok(if any { changed() } else { unchanged() })
    }
}

/// Vectorize every eligible innermost loop of `region`. Public within the
/// crate for direct-driving tests (the env knob is passed, not read).
pub(crate) fn vectorize_region(ctx: &mut Context, region: Ptr<Region>, fp_reduce: bool) -> bool {
    let mut any = false;
    for _ in 0..MAX_LOOPS {
        let dom = dominator_tree(ctx, region);
        let loops = natural_loops(ctx, &dom);
        let mut applied = false;
        for natural_loop in &loops {
            if let Some(plan) = analyze(ctx, natural_loop, fp_reduce) {
                apply(ctx, region, &plan);
                applied = true;
                any = true;
                break;
            }
        }
        if !applied {
            break;
        }
    }
    any
}

// ============================================================================
// Analysis
// ============================================================================

/// The 128-bit lane shape of one element type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ElemKind {
    I32,
    I64,
    F32,
    F64,
}

impl ElemKind {
    fn lanes(self) -> u32 {
        match self {
            ElemKind::I32 | ElemKind::F32 => 4,
            ElemKind::I64 | ElemKind::F64 => 2,
        }
    }
    fn lane_bits(self) -> u64 {
        match self {
            ElemKind::I32 | ElemKind::F32 => 32,
            ElemKind::I64 | ElemKind::F64 => 64,
        }
    }
    fn is_fp(self) -> bool {
        matches!(self, ElemKind::F32 | ElemKind::F64)
    }
}

fn elem_kind_of(ctx: &Context, ty: TypeHandle) -> Option<ElemKind> {
    let ty_ref = ty.deref(ctx);
    if let Some(int_ty) = ty_ref.downcast_ref::<crate::dialects::builtin::types::IntegerType>() {
        return match int_ty.width() {
            32 => Some(ElemKind::I32),
            64 => Some(ElemKind::I64),
            _ => None,
        };
    }
    if ty_ref
        .downcast_ref::<crate::dialects::builtin::types::FP32Type>()
        .is_some()
    {
        return Some(ElemKind::F32);
    }
    if ty_ref
        .downcast_ref::<crate::dialects::builtin::types::FP64Type>()
        .is_some()
    {
        return Some(ElemKind::F64);
    }
    None
}

/// Where an access's base pointer comes from.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum BaseKey {
    /// A loop-invariant SSA value.
    Inv(Value),
    /// The value re-loaded each iteration from a non-escaping alloca slot
    /// that no in-loop store writes (rustc's un-promoted fat-pointer
    /// locals). Invariant by construction; the vector loop re-loads it
    /// once in the preheader. Keyed by the slot LOAD op inside the loop.
    Slot(Ptr<Operation>),
}

/// The unit-stride index shape of one access.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum AddrForm {
    /// `gep<T>(base, [iv (+ inv)])` — element-typed gep.
    Elem { offset: Option<Value> },
    /// `gep<i8>(base, [iv << k])` with `2^k == size_of(T)` — the
    /// byte-addressed form rustc's slice indexing lowers to.
    ByteShl { k: u32, byte_ty: TypeHandle },
}

/// One unit-stride address: `base + (iv [+ offset]) * elem_size`. Two
/// accesses with equal keys touch the same lanes.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct AddrKey {
    base: BaseKey,
    form: AddrForm,
}

/// A vectorizable operand of a body op.
#[derive(Clone, Copy)]
enum Operand {
    /// Result of an earlier vectorized body op.
    Vec(Ptr<Operation>),
    /// Loop-invariant scalar: splat in the preheader.
    Splat(Value),
    /// Body-local constant: clone to the preheader, then splat.
    SplatConst(Ptr<Operation>),
}

/// One vector-body op to emit, in scalar program order.
enum Action {
    Load {
        op: Ptr<Operation>,
        addr: AddrKey,
    },
    Store {
        op: Ptr<Operation>,
        addr: AddrKey,
        value: Operand,
    },
    Bin {
        op: Ptr<Operation>,
        kind: VBinOpKindAttr,
        lhs: Operand,
        rhs: Operand,
    },
    /// `vacc[arg] = vbinop add/fadd (vacc[arg], operand)`.
    ReductionUpdate {
        arg_index: usize,
        op: Ptr<Operation>,
        operand: Operand,
    },
}

/// What each header argument is.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ArgRole {
    Iv,
    /// Updated once per iteration by `add`/`fadd`; vectorized as VF
    /// partial sums.
    Reduction,
    /// Latch passes the argument through unchanged: constant along the
    /// loop, equal to its init operand.
    PassThrough,
}

struct VecPlan {
    preheader: Ptr<BasicBlock>,
    header: Ptr<BasicBlock>,
    cond_br: Ptr<Operation>,
    iv_index: usize,
    bound: Value,
    /// The exit compare's signedness (`slt` vs `ult`), for the vector
    /// guard's own `iv < bound` test.
    signed_cmp: bool,
    init_operands: Vec<Value>,
    arg_roles: Vec<ArgRole>,
    /// Reductions in header-arg order: `(arg index, reduce kind, add op)`.
    reductions: Vec<(usize, VReduceKindAttr, Ptr<Operation>)>,
    elem: ElemKind,
    elem_ty: TypeHandle,
    iv_ty: TypeHandle,
    actions: Vec<Action>,
    iv_update_op: Ptr<Operation>,
    /// One representative in-loop load per slot-based alloca, for the
    /// preheader re-load.
    slot_reps: FxHashMap<Ptr<Operation>, Ptr<Operation>>,
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

/// Users of `value` inside the loop blocks.
fn loop_users(
    ctx: &Context,
    value: Value,
    loop_blocks: &FxHashSet<Ptr<BasicBlock>>,
) -> Vec<Ptr<Operation>> {
    value
        .uses(ctx)
        .into_iter()
        .map(|u| u.user_op())
        .filter(|op| {
            op.deref(ctx)
                .get_container()
                .map(|b| loop_blocks.contains(&b))
                .unwrap_or(false)
        })
        .collect()
}

fn analyze(ctx: &Context, natural_loop: &NaturalLoop, fp_reduce: bool) -> Option<VecPlan> {
    // Shape: 2 blocks, one latch, argument-less body, existing preheader.
    if natural_loop.latches.len() != 1 || natural_loop.body.len() != 2 {
        return None;
    }
    let header = natural_loop.header;
    // Never re-vectorize a scalar epilogue we created.
    if header
        .deref(ctx)
        .attributes
        .get::<VectorizeEpilogueAttr>(&ATTR_KEY_VECTORIZE_EPILOGUE)
        .is_some()
    {
        return None;
    }
    let body_blk = *natural_loop.body.iter().find(|b| **b != header)?;
    if natural_loop.latches[0] != body_blk || body_blk.deref(ctx).get_num_arguments() != 0 {
        return None;
    }
    let preheader = natural_loop.preheader(ctx)?;
    let pre_term = preheader.deref(ctx).get_terminator(ctx)?;
    if Operation::get_opid(pre_term, ctx) != BrOp::get_opid_static() {
        return None;
    }

    // Header: exactly the exiting icmp plus the cond_br whose TRUE edge is
    // the body and whose FALSE edge leaves the loop.
    let cond_br = header.deref(ctx).get_terminator(ctx)?;
    if Operation::get_opid(cond_br, ctx) != CondBrOp::get_opid_static() {
        return None;
    }
    let header_ops: Vec<Ptr<Operation>> = header
        .deref(ctx)
        .iter(ctx)
        .filter(|op| *op != cond_br)
        .collect();
    let [icmp_op] = header_ops.as_slice() else {
        return None;
    };
    let icmp_op = *icmp_op;
    if Operation::get_opid(icmp_op, ctx) != ICmpOp::get_opid_static() {
        return None;
    }
    let cbr = CondBrOp::from_operation(cond_br);
    if cbr.get_operand_condition(ctx) != icmp_op.deref(ctx).get_result(0) {
        return None;
    }
    if cond_br.deref(ctx).get_successor(0) != body_blk
        || natural_loop.body.contains(&cond_br.deref(ctx).get_successor(1))
    {
        return None;
    }

    // The latch and the preheader must pass exactly the header's arguments.
    let latch_term = body_blk.deref(ctx).get_terminator(ctx)?;
    if Operation::get_opid(latch_term, ctx) != BrOp::get_opid_static()
        || latch_term.deref(ctx).get_successor(0) != header
    {
        return None;
    }
    let num_args = header.deref(ctx).get_num_arguments();
    let init_operands = BrOp::from_operation(pre_term).successor_operands(ctx, 0);
    let latch_operands = BrOp::from_operation(latch_term).successor_operands(ctx, 0);
    if init_operands.len() != num_args || latch_operands.len() != num_args {
        return None;
    }
    let header_args: Vec<Value> = (0..num_args)
        .map(|j| header.deref(ctx).get_argument(j))
        .collect();

    // The exiting compare: `iv < bound` (ult/slt), either operand order.
    let (lhs, rhs) = {
        let op_ref = icmp_op.deref(ctx);
        (op_ref.get_operand(0), op_ref.get_operand(1))
    };
    let predicate = ICmpOp::from_operation(icmp_op).predicate(ctx);
    let (iv_index, bound, signed_cmp) =
        if let Some(j) = header_args.iter().position(|a| *a == lhs) {
            match predicate {
                ICmpPredicateAttr::ULT => (j, rhs, false),
                ICmpPredicateAttr::SLT => (j, rhs, true),
                _ => return None,
            }
        } else if let Some(j) = header_args.iter().position(|a| *a == rhs) {
            match predicate {
                ICmpPredicateAttr::UGT => (j, lhs, false),
                ICmpPredicateAttr::SGT => (j, lhs, true),
                _ => return None,
            }
        } else {
            return None;
        };
    if defined_in_loop(ctx, bound, &natural_loop.body) {
        return None;
    }
    // 64-bit induction only: with step +1 and an lt-exit, `bound - iv`
    // computed by a wrapping 64-bit sub is the exact remaining trip count
    // (ult: iv < bound directly; slt: both fit i64, so the true difference
    // is in (0, 2^64)), and `rem >= VF` implies iv+VF-1 does not wrap —
    // the whole trip-safety argument of the vector-loop guard.
    let iv_ty = header_args[iv_index].get_type(ctx);
    match iv_ty.deref(ctx).downcast_ref::<IntegerType>() {
        Some(int_ty) if int_ty.width() == 64 => {}
        _ => return None,
    }

    // iv update: `add iv, 1` in the body, feeding only the latch.
    let iv_update = latch_operands[iv_index];
    let iv_update_op = iv_update.defining_op()?;
    if Operation::get_opid(iv_update_op, ctx) != AddOp::get_opid_static()
        || !defined_in_loop(ctx, iv_update, &natural_loop.body)
    {
        return None;
    }
    {
        let (a, b) = {
            let op_ref = iv_update_op.deref(ctx);
            (op_ref.get_operand(0), op_ref.get_operand(1))
        };
        let step = if a == header_args[iv_index] {
            as_const_operand(ctx, b)?
        } else if b == header_args[iv_index] {
            as_const_operand(ctx, a)?
        } else {
            return None;
        };
        if step.width != 64 || step.masked() != 1 {
            return None;
        }
    }
    for user in loop_users(ctx, iv_update, &natural_loop.body) {
        if user != latch_term {
            return None;
        }
    }

    // Classify the remaining header arguments.
    let mut arg_roles = vec![ArgRole::PassThrough; num_args];
    arg_roles[iv_index] = ArgRole::Iv;
    let mut reduction_adds: Vec<(usize, Ptr<Operation>)> = Vec::new();
    for j in 0..num_args {
        if j == iv_index {
            continue;
        }
        if latch_operands[j] == header_args[j] {
            arg_roles[j] = ArgRole::PassThrough;
            continue;
        }
        // Reduction candidate: latch passes `op(acc, X)` with a single
        // in-loop use of acc (that op) besides the exiting cond_br.
        let update = latch_operands[j];
        let update_op = update.defining_op()?;
        if !defined_in_loop(ctx, update, &natural_loop.body) {
            return None;
        }
        let opid = Operation::get_opid(update_op, ctx);
        let is_int = opid == AddOp::get_opid_static();
        let is_fp = opid == FAddOp::get_opid_static();
        if !is_int && !(is_fp && fp_reduce) {
            return None;
        }
        let (a, b) = {
            let op_ref = update_op.deref(ctx);
            (op_ref.get_operand(0), op_ref.get_operand(1))
        };
        if a != header_args[j] && b != header_args[j] {
            return None;
        }
        for user in loop_users(ctx, header_args[j], &natural_loop.body) {
            if user != update_op && user != cond_br {
                return None;
            }
        }
        // The partial-sum trick changes the accumulator's intermediate
        // values: nothing in the loop may observe them besides the latch.
        for user in loop_users(ctx, update, &natural_loop.body) {
            if user != latch_term && user != cond_br {
                return None;
            }
        }
        // One update op may serve one accumulator only: the rewrite keys
        // its partial-sum updates by the op.
        if reduction_adds.iter().any(|(_, op)| *op == update_op) {
            return None;
        }
        arg_roles[j] = ArgRole::Reduction;
        reduction_adds.push((j, update_op));
    }

    // iv uses: the icmp, the update add, gep indices and index adds, plus
    // the cond_br's exit-edge operands. Collected below during the body
    // walk; validated at the end.
    let loop_blocks = &natural_loop.body;

    let body_ops: Vec<Ptr<Operation>> = body_blk
        .deref(ctx)
        .iter(ctx)
        .filter(|op| *op != latch_term)
        .collect();

    let resolve_invariant = |v: Value| -> Option<Value> {
        if !defined_in_loop(ctx, v, loop_blocks) {
            return Some(v);
        }
        if let Some(j) = header_args.iter().position(|a| *a == v)
            && arg_roles[j] == ArgRole::PassThrough
        {
            return Some(init_operands[j]);
        }
        None
    };

    // Pass 1a: recognize invariant fat-pointer slot loads — a load whose
    // address is a NON-ESCAPING alloca defined outside the loop, with no
    // in-loop store to that slot. rustc's un-promoted slice locals reload
    // their (ptr, len) pair every iteration; the value is invariant by
    // construction, so the vector loop may re-load it once.
    let iv_arg = header_args[iv_index];
    let mut slot_loads: FxHashMap<Ptr<Operation>, Ptr<Operation>> = FxHashMap::default();
    for op in &body_ops {
        if Operation::get_opid(*op, ctx) != LoadOp::get_opid_static() {
            continue;
        }
        let addr = op.deref(ctx).get_operand(0);
        if defined_in_loop(ctx, addr, loop_blocks) {
            continue;
        }
        let Some(alloca_op) = addr.defining_op() else {
            continue;
        };
        if Operation::get_opid(alloca_op, ctx)
            != crate::dialects::llvm::ops::AllocaOp::get_opid_static()
        {
            continue;
        }
        if alloca_slot_is_loop_invariant(ctx, addr, loop_blocks) {
            slot_loads.insert(*op, alloca_op);
        }
    }

    // Pass 1b: recognize index expressions (`iv + inv` element indices and
    // `iv << k` byte indices) and unit-stride geps.
    let mut index_adds: FxHashMap<Ptr<Operation>, Value> = FxHashMap::default();
    let mut index_shls: FxHashMap<Ptr<Operation>, u32> = FxHashMap::default();
    for op in &body_ops {
        if *op == iv_update_op {
            continue;
        }
        let opid = Operation::get_opid(*op, ctx);
        if opid == AddOp::get_opid_static() {
            let (a, b) = {
                let op_ref = op.deref(ctx);
                (op_ref.get_operand(0), op_ref.get_operand(1))
            };
            let other = if a == iv_arg {
                b
            } else if b == iv_arg {
                a
            } else {
                continue;
            };
            if let Some(inv) = resolve_invariant(other) {
                index_adds.insert(*op, inv);
            }
        } else if opid == ShlOp::get_opid_static() {
            let (a, b) = {
                let op_ref = op.deref(ctx);
                (op_ref.get_operand(0), op_ref.get_operand(1))
            };
            if a != iv_arg {
                continue;
            }
            let Some(k) = as_const_operand(ctx, b) else {
                continue;
            };
            let k = k.masked();
            if k < 4 {
                index_shls.insert(*op, k as u32);
            }
        }
    }
    let mut geps: FxHashMap<Ptr<Operation>, AddrKey> = FxHashMap::default();
    for op in &body_ops {
        if Operation::get_opid(*op, ctx) != GetElementPtrOp::get_opid_static() {
            continue;
        }
        let gep = GetElementPtrOp::from_operation(*op);
        let base_val = gep.get_operand_src_ptr(ctx);
        let base = if let Some(inv) = resolve_invariant(base_val) {
            Some(BaseKey::Inv(inv))
        } else {
            base_val
                .defining_op()
                .and_then(|def| slot_loads.get(&def).copied())
                .map(BaseKey::Slot)
        };
        let indices = gep.indices(ctx);
        let (Some(base), [GepIndex::Value(idx)]) = (base, indices.as_slice()) else {
            continue;
        };
        let src_elem = gep.src_elem_type(ctx);
        let is_byte_gep = matches!(
            src_elem.deref(ctx).downcast_ref::<IntegerType>(),
            Some(int_ty) if int_ty.width() == 8
        );
        let form = if *idx == iv_arg {
            AddrForm::Elem { offset: None }
        } else if let Some(add_op) = idx.defining_op().filter(|o| index_adds.contains_key(o)) {
            AddrForm::Elem {
                offset: Some(index_adds[&add_op]),
            }
        } else if let Some(shl_op) = idx.defining_op().filter(|o| index_shls.contains_key(o)) {
            if !is_byte_gep {
                continue;
            }
            AddrForm::ByteShl {
                k: index_shls[&shl_op],
                byte_ty: src_elem,
            }
        } else {
            continue;
        };
        // Element-typed geps must be typed in the loop's element type; the
        // (per-access) size check against the loaded/stored type happens in
        // the main walk.
        if let AddrForm::Elem { .. } = form {
            // Recorded now; validated against elem_ty at the access.
        }
        geps.insert(*op, AddrKey { base, form });
    }
    // Index expressions must feed recognized geps only (any other use
    // would need a per-lane iota vector).
    let mut used_index_ops: FxHashSet<Ptr<Operation>> = FxHashSet::default();
    for gep_op in geps.keys() {
        let gep = GetElementPtrOp::from_operation(*gep_op);
        if let [GepIndex::Value(idx)] = gep.indices(ctx).as_slice()
            && let Some(def) = idx.defining_op()
            && (index_adds.contains_key(&def) || index_shls.contains_key(&def))
        {
            used_index_ops.insert(def);
        }
    }
    for index_op in &used_index_ops {
        let result = index_op.deref(ctx).get_result(0);
        for user in loop_users(ctx, result, loop_blocks) {
            if !geps.contains_key(&user) {
                return None;
            }
        }
    }
    // Slot loads must feed recognized geps only (their value is a fat
    // pointer, not a lane value). Every load of a USED slot alloca is
    // checked, so two loads of one slot can never smuggle its value out.
    let used_slot_allocas: FxHashSet<Ptr<Operation>> = geps
        .values()
        .filter_map(|key| match key.base {
            BaseKey::Slot(alloca) => Some(alloca),
            BaseKey::Inv(_) => None,
        })
        .collect();
    let mut used_slot_loads: FxHashSet<Ptr<Operation>> = FxHashSet::default();
    for (load_op, alloca) in &slot_loads {
        if !used_slot_allocas.contains(alloca) {
            continue;
        }
        used_slot_loads.insert(*load_op);
        let result = load_op.deref(ctx).get_result(0);
        for user in loop_users(ctx, result, loop_blocks) {
            if !geps.contains_key(&user) {
                return None;
            }
        }
    }
    // Recognized geps must feed loads/stores only.
    for gep_op in geps.keys() {
        let result = gep_op.deref(ctx).get_result(0);
        for user in loop_users(ctx, result, loop_blocks) {
            let opid = Operation::get_opid(user, ctx);
            if opid != LoadOp::get_opid_static() && opid != StoreOp::get_opid_static() {
                return None;
            }
        }
    }

    // Pass 2: classify every body op into actions.
    fn require_elem(
        ctx: &Context,
        slot: &mut Option<TypeHandle>,
        ty: TypeHandle,
    ) -> Option<TypeHandle> {
        match *slot {
            None => {
                elem_kind_of(ctx, ty)?;
                *slot = Some(ty);
                Some(ty)
            }
            Some(existing) if existing == ty => Some(ty),
            Some(_) => None,
        }
    }
    let mut elem_slot: Option<TypeHandle> = None;
    let mut vec_defs: FxHashSet<Ptr<Operation>> = FxHashSet::default();
    let mut constants: FxHashSet<Ptr<Operation>> = FxHashSet::default();
    let reduction_add_of: FxHashMap<Ptr<Operation>, usize> = reduction_adds
        .iter()
        .map(|(j, op)| (*op, *j))
        .collect();
    let mut actions: Vec<Action> = Vec::new();

    let resolve_operand = |v: Value,
                           vec_defs: &FxHashSet<Ptr<Operation>>,
                           constants: &FxHashSet<Ptr<Operation>>|
     -> Option<Operand> {
        if let Some(def) = v.defining_op() {
            if vec_defs.contains(&def) {
                return Some(Operand::Vec(def));
            }
            if constants.contains(&def) {
                return Some(Operand::SplatConst(def));
            }
        }
        resolve_invariant(v).map(Operand::Splat)
    };

    for op in &body_ops {
        if *op == iv_update_op
            || geps.contains_key(op)
            || used_index_ops.contains(op)
            || used_slot_loads.contains(op)
        {
            continue;
        }
        let opid = Operation::get_opid(*op, ctx);
        if opid == ConstantOp::get_opid_static() {
            constants.insert(*op);
            continue;
        }
        if let Some(arg_index) = reduction_add_of.get(op) {
            let acc = header_args[*arg_index];
            let (a, b) = {
                let op_ref = op.deref(ctx);
                (op_ref.get_operand(0), op_ref.get_operand(1))
            };
            let x = if a == acc { b } else { a };
            let operand = resolve_operand(x, &vec_defs, &constants)?;
            // The accumulator's element type is the loop's element type.
            let ty = require_elem(ctx, &mut elem_slot, acc.get_type(ctx))?;
            check_operand_type(ctx, operand, ty)?;
            actions.push(Action::ReductionUpdate {
                arg_index: *arg_index,
                op: *op,
                operand,
            });
            continue;
        }
        if opid == LoadOp::get_opid_static() {
            let addr = op.deref(ctx).get_operand(0);
            let (gep_op, key) = addr
                .defining_op()
                .and_then(|g| geps.get(&g).copied().map(|k| (g, k)))?;
            let result_ty = op.deref(ctx).get_result(0).get_type(ctx);
            check_access_shape(ctx, gep_op, key, result_ty)?;
            require_elem(ctx, &mut elem_slot, result_ty)?;
            vec_defs.insert(*op);
            actions.push(Action::Load { op: *op, addr: key });
            continue;
        }
        if opid == StoreOp::get_opid_static() {
            let store = StoreOp::from_operation(*op);
            let value = store.get_operand_value(ctx);
            let addr = store.get_operand_address(ctx);
            let (gep_op, key) = addr
                .defining_op()
                .and_then(|g| geps.get(&g).copied().map(|k| (g, k)))?;
            let value_ty = value.get_type(ctx);
            check_access_shape(ctx, gep_op, key, value_ty)?;
            let ty = require_elem(ctx, &mut elem_slot, value_ty)?;
            let operand = resolve_operand(value, &vec_defs, &constants)?;
            check_operand_type(ctx, operand, ty)?;
            actions.push(Action::Store {
                op: *op,
                addr: key,
                value: operand,
            });
            continue;
        }
        // Lane-wise binops.
        let kind = if opid == AddOp::get_opid_static() {
            VBinOpKindAttr::Add
        } else if opid == SubOp::get_opid_static() {
            VBinOpKindAttr::Sub
        } else if opid == MulOp::get_opid_static() {
            VBinOpKindAttr::Mul
        } else if opid == AndOp::get_opid_static() {
            VBinOpKindAttr::And
        } else if opid == OrOp::get_opid_static() {
            VBinOpKindAttr::Or
        } else if opid == XorOp::get_opid_static() {
            VBinOpKindAttr::Xor
        } else if opid == ShlOp::get_opid_static() {
            VBinOpKindAttr::Shl
        } else if opid == LShrOp::get_opid_static() {
            VBinOpKindAttr::LShr
        } else if opid == AShrOp::get_opid_static() {
            VBinOpKindAttr::AShr
        } else if opid == FAddOp::get_opid_static() {
            VBinOpKindAttr::FAdd
        } else if opid == FSubOp::get_opid_static() {
            VBinOpKindAttr::FSub
        } else if opid == FMulOp::get_opid_static() {
            VBinOpKindAttr::FMul
        } else if opid == FDivOp::get_opid_static() {
            VBinOpKindAttr::FDiv
        } else {
            return None; // unknown op: calls, selects, anything effectful
        };
        let result_ty = op.deref(ctx).get_result(0).get_type(ctx);
        let ty = require_elem(ctx, &mut elem_slot, result_ty)?;
        let kind_elem = elem_kind_of(ctx, ty)?;
        if kind.is_fp() != kind_elem.is_fp() {
            return None;
        }
        if kind == VBinOpKindAttr::Mul && kind_elem == ElemKind::I64 {
            return None; // no 64-bit lane multiply in NEON
        }
        let (a, b) = {
            let op_ref = op.deref(ctx);
            (op_ref.get_operand(0), op_ref.get_operand(1))
        };
        if kind.is_shift() {
            // Shifts must be by an in-range constant (lowered to the
            // immediate NEON forms; a lane-variable shift has no MVP form).
            let sh = as_const_operand(ctx, b)?;
            let lane_bits = kind_elem.lane_bits();
            let sh_bits = sh.masked() as u64;
            let ok = match kind {
                VBinOpKindAttr::Shl => sh_bits < lane_bits,
                _ => sh_bits > 0 && sh_bits < lane_bits,
            };
            if !ok {
                return None;
            }
        }
        let lhs = resolve_operand(a, &vec_defs, &constants)?;
        let rhs = resolve_operand(b, &vec_defs, &constants)?;
        if !kind.is_shift() {
            check_operand_type(ctx, lhs, ty)?;
            check_operand_type(ctx, rhs, ty)?;
        } else {
            check_operand_type(ctx, lhs, ty)?;
            // The shift rhs stays scalar-typed; the splat is emitted with
            // the loop's vector shape and consumed as an immediate.
        }
        vec_defs.insert(*op);
        actions.push(Action::Bin {
            op: *op,
            kind,
            lhs,
            rhs,
        });
    }
    let elem_ty = elem_slot?;
    let elem = elem_kind_of(ctx, elem_ty)?;

    // Profitability floor: at least one unit-stride memory access.
    if !actions
        .iter()
        .any(|a| matches!(a, Action::Load { .. } | Action::Store { .. }))
    {
        return None;
    }

    // iv uses: everything must be one of the ops we rebuild.
    for user in loop_users(ctx, iv_arg, loop_blocks) {
        let allowed = user == icmp_op
            || user == iv_update_op
            || user == cond_br
            || geps.contains_key(&user)
            || used_index_ops.contains(&user);
        if !allowed {
            return None;
        }
    }

    // Every non-terminator body op must have been consumed by the walk:
    // dead constants and dead index adds are fine, everything else became
    // an action or was validated above. (Ops that fell through returned
    // None already.)

    // Alias discipline: a stored-to base may only be accessed through the
    // exact same address expression. Distinct bases are assumed disjoint
    // when one is written (rustc's `&mut` exclusivity — see module docs).
    let mut store_keys: Vec<AddrKey> = Vec::new();
    let mut all_keys: Vec<AddrKey> = Vec::new();
    for action in &actions {
        match action {
            Action::Load { addr, .. } => all_keys.push(*addr),
            Action::Store { addr, .. } => {
                store_keys.push(*addr);
                all_keys.push(*addr);
            }
            _ => {}
        }
    }
    for store in &store_keys {
        for other in &all_keys {
            if other.base == store.base && *other != *store {
                return None;
            }
        }
    }

    let reductions = reduction_adds
        .iter()
        .map(|(j, op)| {
            let kind = if Operation::get_opid(*op, ctx) == AddOp::get_opid_static() {
                VReduceKindAttr::Add
            } else {
                VReduceKindAttr::FAdd
            };
            (*j, kind, *op)
        })
        .collect();

    let mut slot_reps: FxHashMap<Ptr<Operation>, Ptr<Operation>> = FxHashMap::default();
    for (load_op, alloca) in &slot_loads {
        if used_slot_allocas.contains(alloca) {
            slot_reps.entry(*alloca).or_insert(*load_op);
        }
    }

    Some(VecPlan {
        preheader,
        header,
        cond_br,
        iv_index,
        bound,
        signed_cmp,
        init_operands,
        arg_roles,
        reductions,
        elem,
        elem_ty,
        iv_ty,
        actions,
        iv_update_op,
        slot_reps,
    })
}

/// Validate one access's address shape against the loaded/stored element
/// type: element-typed geps must be typed exactly in it; byte geps must
/// scale by its size (`2^k == size_of(elem)`).
fn check_access_shape(
    ctx: &Context,
    gep_op: Ptr<Operation>,
    key: AddrKey,
    elem_ty: TypeHandle,
) -> Option<()> {
    match key.form {
        AddrForm::Elem { .. } => {
            let gep = GetElementPtrOp::from_operation(gep_op);
            (gep.src_elem_type(ctx) == elem_ty).then_some(())
        }
        AddrForm::ByteShl { k, .. } => {
            let bytes = elem_kind_of(ctx, elem_ty)?.lane_bits() / 8;
            (1u64 << k == bytes).then_some(())
        }
    }
}

/// Whether `slot` (an alloca's result) is a non-escaping stack slot that no
/// store inside the loop writes: its only uses, transitively through
/// pointer bitcasts, are loads of the slot and stores TO the slot, and all
/// the stores sit outside the loop. A non-escaping alloca cannot alias any
/// pointer that predates the frame (function arguments, loaded fat
/// pointers), which is what makes the loop's slot re-loads invariant.
fn alloca_slot_is_loop_invariant(
    ctx: &Context,
    slot: Value,
    loop_blocks: &FxHashSet<Ptr<BasicBlock>>,
) -> bool {
    let mut views = vec![slot];
    while let Some(view) = views.pop() {
        for use_ in view.uses(ctx) {
            let user = use_.user_op();
            let opid = Operation::get_opid(user, ctx);
            if opid == LoadOp::get_opid_static() && user.deref(ctx).get_operand(0) == view {
                continue;
            }
            if opid == StoreOp::get_opid_static() {
                let (value, addr) = {
                    let op_ref = user.deref(ctx);
                    (op_ref.get_operand(0), op_ref.get_operand(1))
                };
                if addr == view && value != view {
                    // A store TO the slot: allowed outside the loop only.
                    let in_loop = user
                        .deref(ctx)
                        .get_container()
                        .map(|b| loop_blocks.contains(&b))
                        .unwrap_or(false);
                    if in_loop {
                        return false;
                    }
                    continue;
                }
                return false; // the slot's ADDRESS is stored: it escapes
            }
            if opid == crate::dialects::llvm::ops::BitcastOp::get_opid_static() {
                views.push(user.deref(ctx).get_result(0));
                continue;
            }
            return false; // any other use (gep, call, ...) escapes
        }
    }
    true
}

/// Type of an operand for the element-type check (splats must match the
/// lane type; vector operands match by construction).
fn check_operand_type(ctx: &Context, operand: Operand, elem_ty: TypeHandle) -> Option<()> {
    match operand {
        Operand::Vec(_) => Some(()),
        Operand::Splat(v) => (v.get_type(ctx) == elem_ty).then_some(()),
        Operand::SplatConst(c) => {
            (c.deref(ctx).get_result(0).get_type(ctx) == elem_ty).then_some(())
        }
    }
}

// ============================================================================
// Rewrite
// ============================================================================

/// Stamp a freshly created op as derived from `anchor`'s effective sources
/// (backward attribution; a no-op when the module carries no op ids).
fn stamp_from(ctx: &mut Context, new_op: Ptr<Operation>, anchor: Ptr<Operation>) {
    let sources = opmap::effective_sources(ctx, anchor);
    if !sources.is_empty() {
        opmap::set_derived_from_many(ctx, new_op, sources);
    }
}

fn apply(ctx: &mut Context, region: Ptr<Region>, plan: &VecPlan) {
    let lanes = plan.elem.lanes();
    let vec_ty: TypeHandle =
        VectorType::get(ctx, plan.elem_ty, lanes, VectorTypeKind::Fixed).into();
    let pre_term = plan
        .preheader
        .deref(ctx)
        .get_terminator(ctx)
        .expect("analyze verified the preheader terminator");

    // --- Preheader materializations -------------------------------------
    let iv_int_ty = {
        let (width, signedness) = {
            let ty_ref = plan.iv_ty.deref(ctx);
            let int_ty = ty_ref
                .downcast_ref::<IntegerType>()
                .expect("analyze verified the iv type");
            (
                int_ty.width(),
                if int_ty.is_signed() {
                    Signedness::Signed
                } else if int_ty.is_signless() {
                    Signedness::Signless
                } else {
                    Signedness::Unsigned
                },
            )
        };
        IntegerType::get(ctx, width, signedness)
    };
    let vf_const = ConstantOp::new(
        ctx,
        Box::new(IntegerAttr::new(
            iv_int_ty,
            APInt::from_u64(lanes as u64, NonZero::new(64usize).unwrap()),
        )),
    );
    let vf_op = vf_const.get_operation();
    vf_op.insert_before(ctx, pre_term);
    stamp_from(ctx, vf_op, plan.iv_update_op);
    let vf_val = vf_op.deref(ctx).get_result(0);

    // Splat cache: invariants and cloned body constants.
    let mut splats: FxHashMap<Value, Value> = FxHashMap::default();
    let mut const_splats: FxHashMap<Ptr<Operation>, Value> = FxHashMap::default();
    let splat_of = |ctx: &mut Context,
                        splats: &mut FxHashMap<Value, Value>,
                        scalar: Value,
                        anchor: Ptr<Operation>|
     -> Value {
        if let Some(v) = splats.get(&scalar) {
            return *v;
        }
        let splat = VSplatOp::new(ctx, scalar, vec_ty);
        let splat_op = splat.get_operation();
        splat_op.insert_before(ctx, pre_term);
        stamp_from(ctx, splat_op, anchor);
        let v = splat_op.deref(ctx).get_result(0);
        splats.insert(scalar, v);
        v
    };

    // Reduction identity: a zero vector per accumulator; the scalar init
    // value is added back after the horizontal reduce.
    let mut vacc_inits: Vec<Value> = Vec::new();
    for (_, _, add_op) in &plan.reductions {
        let zero: Box<dyn pliron::attribute::Attribute> = match plan.elem {
            ElemKind::I32 | ElemKind::I64 => {
                let elem_int = {
                    let ty_ref = plan.elem_ty.deref(ctx);
                    let int_ty = ty_ref.downcast_ref::<IntegerType>().unwrap();
                    (int_ty.width(), int_ty.is_signed())
                };
                let width = elem_int.0;
                Box::new(IntegerAttr::new(
                    IntegerType::get(ctx, width, Signedness::Signless),
                    APInt::from_u64(0, NonZero::new(width as usize).unwrap()),
                ))
            }
            ElemKind::F32 => Box::new(FPSingleAttr::from(0.0f32)),
            ElemKind::F64 => Box::new(FPDoubleAttr::from(0.0f64)),
        };
        let zero_const = ConstantOp::new(ctx, zero);
        let zero_op = zero_const.get_operation();
        zero_op.insert_before(ctx, pre_term);
        stamp_from(ctx, zero_op, *add_op);
        let zero_val = zero_op.deref(ctx).get_result(0);
        let vsplat = VSplatOp::new(ctx, zero_val, vec_ty);
        let vsplat_op = vsplat.get_operation();
        vsplat_op.insert_before(ctx, pre_term);
        stamp_from(ctx, vsplat_op, *add_op);
        vacc_inits.push(vsplat_op.deref(ctx).get_result(0));
    }

    // Materialize each access's invariant base in the preheader: slot
    // bases re-load their (non-escaping, not-written-in-loop) alloca once;
    // element geps with an invariant index addend hoist it.
    let mut slot_values: FxHashMap<Ptr<Operation>, Value> = FxHashMap::default();
    let mut byte_shift_consts: FxHashMap<u32, Value> = FxHashMap::default();
    let mut base_of: FxHashMap<AddrKey, Value> = FxHashMap::default();
    for action in &plan.actions {
        let (key, anchor) = match action {
            Action::Load { addr, op } | Action::Store { addr, op, .. } => (*addr, *op),
            _ => continue,
        };
        if base_of.contains_key(&key) {
            continue;
        }
        let raw_base = match key.base {
            BaseKey::Inv(v) => v,
            BaseKey::Slot(alloca) => {
                if let Some(v) = slot_values.get(&alloca) {
                    *v
                } else {
                    let rep = plan.slot_reps[&alloca];
                    let (rep_addr, rep_ty) = {
                        let rep_ref = rep.deref(ctx);
                        (rep_ref.get_operand(0), rep_ref.get_result(0))
                    };
                    let rep_ty = rep_ty.get_type(ctx);
                    let reload = LoadOp::new(ctx, rep_addr, rep_ty).get_operation();
                    reload.insert_before(ctx, pre_term);
                    stamp_from(ctx, reload, rep);
                    let v = reload.deref(ctx).get_result(0);
                    slot_values.insert(alloca, v);
                    v
                }
            }
        };
        let base = match key.form {
            AddrForm::Elem { offset: None } | AddrForm::ByteShl { .. } => raw_base,
            AddrForm::Elem { offset: Some(inv) } => {
                let gep = GetElementPtrOp::new(
                    ctx,
                    raw_base,
                    vec![GepIndex::Value(inv)],
                    plan.elem_ty,
                );
                let gep_op = gep.get_operation();
                gep_op.insert_before(ctx, pre_term);
                stamp_from(ctx, gep_op, anchor);
                gep_op.deref(ctx).get_result(0)
            }
        };
        // Byte-form shift counts, materialized once in the preheader.
        if let AddrForm::ByteShl { k, .. } = key.form
            && !byte_shift_consts.contains_key(&k)
        {
            let c = ConstantOp::new(
                ctx,
                Box::new(IntegerAttr::new(
                    iv_int_ty,
                    APInt::from_u64(k as u64, NonZero::new(64usize).unwrap()),
                )),
            );
            let c_op = c.get_operation();
            c_op.insert_before(ctx, pre_term);
            stamp_from(ctx, c_op, anchor);
            byte_shift_consts.insert(k, c_op.deref(ctx).get_result(0));
        }
        base_of.insert(key, base);
    }

    // --- New blocks ------------------------------------------------------
    let n_red = plan.reductions.len();
    let mut vheader_args: Vec<TypeHandle> = vec![plan.iv_ty];
    vheader_args.extend(std::iter::repeat_n(vec_ty, n_red));
    let vheader = BasicBlock::new(ctx, None, vheader_args);
    vheader.insert_at_back(region, ctx);
    let vbody = BasicBlock::new(ctx, None, vec![]);
    vbody.insert_at_back(region, ctx);
    let mid = BasicBlock::new(ctx, None, vec![]);
    mid.insert_at_back(region, ctx);

    let viv = vheader.deref(ctx).get_argument(0);
    let vacc_args: Vec<Value> = (0..n_red)
        .map(|r| vheader.deref(ctx).get_argument(1 + r))
        .collect();

    // --- Vector header guard ----------------------------------------------
    // `(iv < bound) && (wrapping(bound - iv) >= VF)`: when `iv < bound`
    // holds, the wrapping 64-bit difference is the EXACT remaining trip
    // count for the accepted lt-exits (ult directly; slt because both fit
    // in i64, so the true difference is in (0, 2^64)), and `>= VF` then
    // also proves `iv + VF-1` cannot wrap. The `iv < bound` term is what
    // keeps a zero-trip loop entered with `iv > bound` out of the vector
    // body — the wrapped difference alone would look huge there.
    let in_range_pred = if plan.signed_cmp {
        ICmpPredicateAttr::SLT
    } else {
        ICmpPredicateAttr::ULT
    };
    let in_range = ICmpOp::new(ctx, in_range_pred, viv, plan.bound);
    let in_range_op = in_range.get_operation();
    in_range_op.insert_at_back(vheader, ctx);
    stamp_from(ctx, in_range_op, plan.cond_br);
    let in_range_val = in_range_op.deref(ctx).get_result(0);
    let rem = SubOp::new_with_overflow_flag(ctx, plan.bound, viv, Default::default());
    let rem_op = rem.get_operation();
    rem_op.insert_at_back(vheader, ctx);
    stamp_from(ctx, rem_op, plan.cond_br);
    let rem_val = rem_op.deref(ctx).get_result(0);
    let enough = ICmpOp::new(ctx, ICmpPredicateAttr::UGE, rem_val, vf_val);
    let enough_op = enough.get_operation();
    enough_op.insert_at_back(vheader, ctx);
    stamp_from(ctx, enough_op, plan.cond_br);
    let enough_val = enough_op.deref(ctx).get_result(0);
    let guard = pliron_llvm::op_interfaces::BinArithOp::new(ctx, in_range_val, enough_val);
    let guard: AndOp = guard;
    let guard_op = guard.get_operation();
    guard_op.insert_at_back(vheader, ctx);
    stamp_from(ctx, guard_op, plan.cond_br);
    let guard_val = guard_op.deref(ctx).get_result(0);
    let vcond_br = CondBrOp::new(ctx, guard_val, vbody, vec![], mid, vec![]).get_operation();
    vcond_br.insert_at_back(vheader, ctx);
    stamp_from(ctx, vcond_br, plan.cond_br);

    // --- Vector body -------------------------------------------------------
    let mut vec_values: FxHashMap<Ptr<Operation>, Value> = FxHashMap::default();
    let mut cur_vaccs = vacc_args.clone();
    let red_slot: FxHashMap<usize, usize> = plan
        .reductions
        .iter()
        .enumerate()
        .map(|(slot, (j, _, _))| (*j, slot))
        .collect();

    let operand_value = |ctx: &mut Context,
                             splats: &mut FxHashMap<Value, Value>,
                             const_splats: &mut FxHashMap<Ptr<Operation>, Value>,
                             vec_values: &FxHashMap<Ptr<Operation>, Value>,
                             operand: &Operand,
                             anchor: Ptr<Operation>|
     -> Value {
        match operand {
            Operand::Vec(def) => vec_values[def],
            Operand::Splat(s) => splat_of(ctx, splats, *s, anchor),
            Operand::SplatConst(c) => {
                if let Some(v) = const_splats.get(c) {
                    return *v;
                }
                // Clone the body-local constant into the preheader; the
                // original stays for the scalar epilogue.
                let attr = ConstantOp::from_operation(*c).get_value(ctx);
                let cloned = ConstantOp::new(ctx, attr);
                let cloned_op = cloned.get_operation();
                cloned_op.insert_before(ctx, pre_term);
                stamp_from(ctx, cloned_op, *c);
                let cloned_val = cloned_op.deref(ctx).get_result(0);
                let v = splat_of(ctx, splats, cloned_val, anchor);
                const_splats.insert(*c, v);
                v
            }
        }
    };

    let addr_for = |ctx: &mut Context, key: &AddrKey, anchor: Ptr<Operation>| -> Value {
        let (index, gep_elem) = match key.form {
            AddrForm::Elem { .. } => (viv, plan.elem_ty),
            AddrForm::ByteShl { k, byte_ty } => {
                // Rebuild the scalar's byte index at the vector iv:
                // `viv << k` (identical wrapping semantics per lane 0).
                let shl = ShlOp::new_with_overflow_flag(
                    ctx,
                    viv,
                    byte_shift_consts[&k],
                    Default::default(),
                )
                .get_operation();
                shl.insert_at_back(vbody, ctx);
                stamp_from(ctx, shl, anchor);
                (shl.deref(ctx).get_result(0), byte_ty)
            }
        };
        let gep = GetElementPtrOp::new(ctx, base_of[key], vec![GepIndex::Value(index)], gep_elem);
        let gep_op = gep.get_operation();
        gep_op.insert_at_back(vbody, ctx);
        stamp_from(ctx, gep_op, anchor);
        gep_op.deref(ctx).get_result(0)
    };

    for action in &plan.actions {
        match action {
            Action::Load { op, addr } => {
                let addr_val = addr_for(ctx, addr, *op);
                let vload = VLoadOp::new(ctx, addr_val, vec_ty);
                let vload_op = vload.get_operation();
                vload_op.insert_at_back(vbody, ctx);
                stamp_from(ctx, vload_op, *op);
                vec_values.insert(*op, vload_op.deref(ctx).get_result(0));
            }
            Action::Store { op, addr, value } => {
                let v = operand_value(
                    ctx,
                    &mut splats,
                    &mut const_splats,
                    &vec_values,
                    value,
                    *op,
                );
                let addr_val = addr_for(ctx, addr, *op);
                let vstore = VStoreOp::new(ctx, v, addr_val).get_operation();
                vstore.insert_at_back(vbody, ctx);
                stamp_from(ctx, vstore, *op);
            }
            Action::Bin { op, kind, lhs, rhs } => {
                let l = operand_value(ctx, &mut splats, &mut const_splats, &vec_values, lhs, *op);
                let r = operand_value(ctx, &mut splats, &mut const_splats, &vec_values, rhs, *op);
                let vbin = VBinOp::new(ctx, *kind, l, r);
                let vbin_op = vbin.get_operation();
                vbin_op.insert_at_back(vbody, ctx);
                stamp_from(ctx, vbin_op, *op);
                vec_values.insert(*op, vbin_op.deref(ctx).get_result(0));
            }
            Action::ReductionUpdate {
                arg_index,
                op,
                operand,
            } => {
                let x = operand_value(
                    ctx,
                    &mut splats,
                    &mut const_splats,
                    &vec_values,
                    operand,
                    *op,
                );
                let slot = red_slot[arg_index];
                let kind = if plan.elem.is_fp() {
                    VBinOpKindAttr::FAdd
                } else {
                    VBinOpKindAttr::Add
                };
                let vbin = VBinOp::new(ctx, kind, cur_vaccs[slot], x);
                let vbin_op = vbin.get_operation();
                vbin_op.insert_at_back(vbody, ctx);
                stamp_from(ctx, vbin_op, *op);
                cur_vaccs[slot] = vbin_op.deref(ctx).get_result(0);
            }
        }
    }
    let viv_next = AddOp::new_with_overflow_flag(ctx, viv, vf_val, Default::default());
    let viv_next_op = viv_next.get_operation();
    viv_next_op.insert_at_back(vbody, ctx);
    stamp_from(ctx, viv_next_op, plan.iv_update_op);
    let mut latch_args = vec![viv_next_op.deref(ctx).get_result(0)];
    latch_args.extend(cur_vaccs.iter().copied());
    let vlatch = BrOp::new(ctx, vheader, latch_args).get_operation();
    vlatch.insert_at_back(vbody, ctx);
    stamp_from(ctx, vlatch, plan.cond_br);

    // --- Mid block: reduce accumulators, enter the scalar epilogue --------
    let mut epilogue_inits: Vec<Value> = Vec::with_capacity(plan.init_operands.len());
    let mut reduced: FxHashMap<usize, Value> = FxHashMap::default();
    for (slot, (j, kind, add_op)) in plan.reductions.iter().enumerate() {
        let vreduce = VReduceOp::new(ctx, *kind, vacc_args[slot]);
        let vreduce_op = vreduce.get_operation();
        vreduce_op.insert_at_back(mid, ctx);
        stamp_from(ctx, vreduce_op, *add_op);
        let red_val = vreduce_op.deref(ctx).get_result(0);
        // Fold the scalar init back in: the vector accumulators started at
        // zero, so `init ⊕ reduce(vacc)` is the value the scalar loop
        // would carry into its remaining iterations.
        let combined_op = match kind {
            VReduceKindAttr::Add => {
                AddOp::new_with_overflow_flag(
                    ctx,
                    plan.init_operands[*j],
                    red_val,
                    Default::default(),
                )
                .get_operation()
            }
            VReduceKindAttr::FAdd => {
                use pliron_llvm::op_interfaces::FloatBinArithOpWithFastMathFlags as _;
                FAddOp::new_with_fast_math_flags(
                    ctx,
                    plan.init_operands[*j],
                    red_val,
                    Default::default(),
                )
                .get_operation()
            }
        };
        combined_op.insert_at_back(mid, ctx);
        stamp_from(ctx, combined_op, *add_op);
        reduced.insert(*j, combined_op.deref(ctx).get_result(0));
    }
    for (j, role) in plan.arg_roles.iter().enumerate() {
        epilogue_inits.push(match role {
            ArgRole::Iv => viv,
            ArgRole::Reduction => reduced[&j],
            ArgRole::PassThrough => plan.init_operands[j],
        });
    }
    let mid_br = BrOp::new(ctx, plan.header, epilogue_inits).get_operation();
    mid_br.insert_at_back(mid, ctx);
    stamp_from(ctx, mid_br, plan.cond_br);

    // --- Retarget the preheader into the vector loop -----------------------
    // Last, so every preheader materialization above could insert before
    // the ORIGINAL terminator (it stays valid until here).
    let mut entry_args = vec![plan.init_operands[plan.iv_index]];
    entry_args.extend(vacc_inits.iter().copied());
    let new_pre_br = BrOp::new(ctx, vheader, entry_args).get_operation();
    new_pre_br.insert_before(ctx, pre_term);
    opmap::derive_new_from(ctx, new_pre_br, pre_term);
    Operation::erase(pre_term, ctx);

    // The original loop is now the epilogue: mark it so it is never
    // vectorized again (its shape still matches).
    plan.header
        .deref_mut(ctx)
        .attributes
        .set(ATTR_KEY_VECTORIZE_EPILOGUE.clone(), VectorizeEpilogueAttr);
}

#[cfg(test)]
mod tests {
    use super::*;
    use pliron::builtin::op_interfaces::OneResultInterface as _;
    use pliron::builtin::types::FP32Type;
    use pliron::printable::Printable;
    use pliron_llvm::op_interfaces::{BinArithOp as _, FloatBinArithOpWithFastMathFlags as _};
    use pliron_llvm::ops::{FuncOp, ReturnOp};
    use pliron_llvm::types::{FuncType, PointerType, VoidType};

    fn i64_ty(ctx: &mut Context) -> TypeHandle {
        IntegerType::get(ctx, 64, Signedness::Signless).into()
    }
    fn i32_ty(ctx: &mut Context) -> TypeHandle {
        IntegerType::get(ctx, 32, Signedness::Signless).into()
    }

    fn const_i64(ctx: &mut Context, block: Ptr<BasicBlock>, value: u64) -> Value {
        let ty = IntegerType::get(ctx, 64, Signedness::Signless);
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

    /// Skeleton loop `for iv in 0..n` with `extra_args` after (ptr, ptr, n):
    /// entry -> header(iv, carried...) -> body -> header | exit.
    /// Returns (func, entry, header, body, exit, args).
    struct Loop {
        func: FuncOp,
        entry: Ptr<BasicBlock>,
        #[allow(dead_code)]
        header: Ptr<BasicBlock>,
        body: Ptr<BasicBlock>,
        #[allow(dead_code)]
        exit: Ptr<BasicBlock>,
    }

    /// Builds the CFG shell with header args `[i64 iv] ++ carried`, the
    /// `icmp ult iv, n` + cond_br in the header, and `iv+1` -> latch. The
    /// caller fills the body (before the latch terminator) and must pass
    /// carried latch operands via `set_latch_operands` semantics: we build
    /// the latch last, so the body closure returns the carried values.
    fn build_loop(
        ctx: &mut Context,
        arg_tys: Vec<TypeHandle>,
        carried: Vec<TypeHandle>,
        make_inits: impl FnOnce(&mut Context, Ptr<BasicBlock>) -> Vec<Value>,
        fill_body: impl FnOnce(&mut Context, &Loop, Value, &[Value]) -> Vec<Value>,
        n_arg_index: usize,
    ) -> Loop {
        let void_ty: TypeHandle = VoidType::get(ctx).into();
        let fn_ty = FuncType::get(ctx, void_ty, arg_tys, false);
        let func = FuncOp::new(ctx, "loop".try_into().unwrap(), fn_ty);
        func.set_attr_llvm_function_linkage(
            ctx,
            pliron_llvm::attributes::LinkageAttr::ExternalLinkage,
        );
        func.get_or_create_entry_block(ctx);
        let region = func.get_region(ctx).unwrap();
        let entry = func.get_entry_block(ctx).unwrap();
        let i64t = i64_ty(ctx);
        let mut header_tys = vec![i64t];
        header_tys.extend(carried.iter().copied());
        let header = BasicBlock::new(ctx, None, header_tys);
        header.insert_at_back(region, ctx);
        let body = BasicBlock::new(ctx, None, vec![]);
        body.insert_at_back(region, ctx);
        let exit = BasicBlock::new(ctx, None, vec![]);
        exit.insert_at_back(region, ctx);

        let carried_inits = make_inits(ctx, entry);
        assert_eq!(carried_inits.len(), carried.len());
        let zero = const_i64(ctx, entry, 0);
        // The iv step constant lives in the entry block, BEFORE the
        // branch: the preheader's last op must stay its terminator.
        let one = const_i64(ctx, entry, 1);
        let mut init = vec![zero];
        init.extend(carried_inits);
        BrOp::new(ctx, header, init)
            .get_operation()
            .insert_at_back(entry, ctx);

        let iv = header.deref(ctx).get_argument(0);
        let n = entry.deref(ctx).get_argument(n_arg_index);
        let cond = ICmpOp::new(ctx, ICmpPredicateAttr::ULT, iv, n);
        cond.get_operation().insert_at_back(header, ctx);
        let cond_v = cond.get_result(ctx);
        CondBrOp::new(ctx, cond_v, body, vec![], exit, vec![])
            .get_operation()
            .insert_at_back(header, ctx);

        let shell = Loop {
            func,
            entry,
            header,
            body,
            exit,
        };
        let carried_args: Vec<Value> = (0..carried.len())
            .map(|k| header.deref(ctx).get_argument(1 + k))
            .collect();
        let carried_next = fill_body(ctx, &shell, iv, &carried_args);
        assert_eq!(carried_next.len(), carried.len());

        let iv2 = AddOp::new_with_overflow_flag(ctx, iv, one, Default::default());
        iv2.get_operation().insert_at_back(body, ctx);
        let iv2_v = iv2.get_result(ctx);
        let mut latch = vec![iv2_v];
        latch.extend(carried_next);
        BrOp::new(ctx, header, latch)
            .get_operation()
            .insert_at_back(body, ctx);

        ReturnOp::new(ctx, None)
            .get_operation()
            .insert_at_back(exit, ctx);
        shell
    }

    fn run(ctx: &mut Context, func: FuncOp, fp_reduce: bool) -> (bool, String) {
        let region = func.get_region(ctx).unwrap();
        let changed = vectorize_region(ctx, region, fp_reduce);
        let text = format!("{}", func.get_operation().disp(ctx));
        (changed, text)
    }

    /// The canonical saxpy shape: y[i] = alpha * x[i] + y[i] over f32,
    /// runtime trip. Vectorizes to VF=4 with the original loop kept as the
    /// epilogue; the same-base same-index load+store of y is legal.
    #[test]
    fn vectorizes_f32_saxpy_shape_with_epilogue() {
        let mut ctx = Context::new();
        let f32t: TypeHandle = FP32Type::get(&mut ctx).into();
        let ptr: TypeHandle = PointerType::get(&mut ctx, 0).into();
        let i64t = i64_ty(&mut ctx);
        let shell = build_loop(
            &mut ctx,
            vec![ptr, ptr, f32t, i64t],
            vec![],
            |_, _| vec![],
            |ctx, shell, iv, _| {
                let x = shell.entry.deref(ctx).get_argument(0);
                let y = shell.entry.deref(ctx).get_argument(1);
                let alpha = shell.entry.deref(ctx).get_argument(2);
                let f32t: TypeHandle = FP32Type::get(ctx).into();
                let gx = GetElementPtrOp::new(ctx, x, vec![GepIndex::Value(iv)], f32t);
                gx.get_operation().insert_at_back(shell.body, ctx);
                let gx_v = gx.get_result(ctx);
                let lx = LoadOp::new(ctx, gx_v, f32t);
                lx.get_operation().insert_at_back(shell.body, ctx);
                let lx_v = lx.get_result(ctx);
                let m = FMulOp::new_with_fast_math_flags(ctx, alpha, lx_v, Default::default());
                m.get_operation().insert_at_back(shell.body, ctx);
                let m_v = m.get_result(ctx);
                let gy = GetElementPtrOp::new(ctx, y, vec![GepIndex::Value(iv)], f32t);
                gy.get_operation().insert_at_back(shell.body, ctx);
                let gy_v = gy.get_result(ctx);
                let ly = LoadOp::new(ctx, gy_v, f32t);
                ly.get_operation().insert_at_back(shell.body, ctx);
                let ly_v = ly.get_result(ctx);
                let s = FAddOp::new_with_fast_math_flags(ctx, m_v, ly_v, Default::default());
                s.get_operation().insert_at_back(shell.body, ctx);
                let s_v = s.get_result(ctx);
                StoreOp::new(ctx, s_v, gy_v)
                    .get_operation()
                    .insert_at_back(shell.body, ctx);
                vec![]
            },
            3,
        );
        let (changed, text) = run(&mut ctx, shell.func, false);
        assert!(changed, "{text}");
        for needle in [
            "ll.vload",
            "ll.vsplat",
            "ll.vbinop fmul",
            "ll.vbinop fadd",
            "ll.vstore",
        ] {
            assert!(text.contains(needle), "missing {needle}:\n{text}");
        }
        // Vector guard + the original (epilogue) exit test.
        assert_eq!(text.matches("llvm.cond_br").count(), 2, "{text}");
        // The scalar body survives as the epilogue.
        assert!(text.contains("llvm.fmul"), "{text}");
        assert!(text.contains("llvm.fadd"), "{text}");
        // Idempotent: the epilogue is marked, nothing more to vectorize.
        let region = shell.func.get_region(&ctx).unwrap();
        assert!(!vectorize_region(&mut ctx, region, false));
    }

    /// A store through base `p` while `p` is also read at `iv + k`:
    /// same base, different address expression — must bail.
    #[test]
    fn bails_on_aliasing_store() {
        let mut ctx = Context::new();
        let ptr: TypeHandle = PointerType::get(&mut ctx, 0).into();
        let i64t = i64_ty(&mut ctx);
        let shell = build_loop(
            &mut ctx,
            vec![ptr, i64t, i64t],
            vec![],
            |_, _| vec![],
            |ctx, shell, iv, _| {
                let p = shell.entry.deref(ctx).get_argument(0);
                let k = shell.entry.deref(ctx).get_argument(1);
                let i32t = i32_ty(ctx);
                let idx = AddOp::new_with_overflow_flag(ctx, iv, k, Default::default());
                idx.get_operation().insert_at_back(shell.body, ctx);
                let idx_v = idx.get_result(ctx);
                let gload = GetElementPtrOp::new(ctx, p, vec![GepIndex::Value(idx_v)], i32t);
                gload.get_operation().insert_at_back(shell.body, ctx);
                let gload_v = gload.get_result(ctx);
                let l = LoadOp::new(ctx, gload_v, i32t);
                l.get_operation().insert_at_back(shell.body, ctx);
                let l_v = l.get_result(ctx);
                let gstore = GetElementPtrOp::new(ctx, p, vec![GepIndex::Value(iv)], i32t);
                gstore.get_operation().insert_at_back(shell.body, ctx);
                let gstore_v = gstore.get_result(ctx);
                StoreOp::new(ctx, l_v, gstore_v)
                    .get_operation()
                    .insert_at_back(shell.body, ctx);
                vec![]
            },
            2,
        );
        let (changed, text) = run(&mut ctx, shell.func, false);
        assert!(!changed, "{text}");
        assert!(!text.contains("ll.v"), "{text}");
    }

    /// A call in the body has unknown effects: bail.
    #[test]
    fn bails_on_call_in_body() {
        let mut ctx = Context::new();
        let ptr: TypeHandle = PointerType::get(&mut ctx, 0).into();
        let i64t = i64_ty(&mut ctx);
        let shell = build_loop(
            &mut ctx,
            vec![ptr, ptr, i64t],
            vec![],
            |_, _| vec![],
            |ctx, shell, iv, _| {
                let a = shell.entry.deref(ctx).get_argument(0);
                let out = shell.entry.deref(ctx).get_argument(1);
                let i32t = i32_ty(ctx);
                let void_ty: TypeHandle = VoidType::get(ctx).into();
                let callee_ty = FuncType::get(ctx, void_ty, vec![], false);
                pliron_llvm::ops::CallOp::new(
                    ctx,
                    pliron::builtin::op_interfaces::CallOpCallable::Direct(
                        "opaque".try_into().unwrap(),
                    ),
                    callee_ty,
                    vec![],
                )
                .get_operation()
                .insert_at_back(shell.body, ctx);
                let ga = GetElementPtrOp::new(ctx, a, vec![GepIndex::Value(iv)], i32t);
                ga.get_operation().insert_at_back(shell.body, ctx);
                let ga_v = ga.get_result(ctx);
                let l = LoadOp::new(ctx, ga_v, i32t);
                l.get_operation().insert_at_back(shell.body, ctx);
                let l_v = l.get_result(ctx);
                let go = GetElementPtrOp::new(ctx, out, vec![GepIndex::Value(iv)], i32t);
                go.get_operation().insert_at_back(shell.body, ctx);
                let go_v = go.get_result(ctx);
                StoreOp::new(ctx, l_v, go_v)
                    .get_operation()
                    .insert_at_back(shell.body, ctx);
                vec![]
            },
            2,
        );
        let (changed, text) = run(&mut ctx, shell.func, false);
        assert!(!changed, "{text}");
    }

    /// A non-unit-stride access (`gep p [2*iv]`) must bail.
    #[test]
    fn bails_on_non_unit_stride() {
        let mut ctx = Context::new();
        let ptr: TypeHandle = PointerType::get(&mut ctx, 0).into();
        let i64t = i64_ty(&mut ctx);
        let shell = build_loop(
            &mut ctx,
            vec![ptr, ptr, i64t],
            vec![],
            |_, _| vec![],
            |ctx, shell, iv, _| {
                let a = shell.entry.deref(ctx).get_argument(0);
                let out = shell.entry.deref(ctx).get_argument(1);
                let i32t = i32_ty(ctx);
                let two = const_i64(ctx, shell.body, 2);
                let idx = MulOp::new_with_overflow_flag(ctx, iv, two, Default::default());
                idx.get_operation().insert_at_back(shell.body, ctx);
                let idx_v = idx.get_result(ctx);
                let ga = GetElementPtrOp::new(ctx, a, vec![GepIndex::Value(idx_v)], i32t);
                ga.get_operation().insert_at_back(shell.body, ctx);
                let ga_v = ga.get_result(ctx);
                let l = LoadOp::new(ctx, ga_v, i32t);
                l.get_operation().insert_at_back(shell.body, ctx);
                let l_v = l.get_result(ctx);
                let go = GetElementPtrOp::new(ctx, out, vec![GepIndex::Value(iv)], i32t);
                go.get_operation().insert_at_back(shell.body, ctx);
                let go_v = go.get_result(ctx);
                StoreOp::new(ctx, l_v, go_v)
                    .get_operation()
                    .insert_at_back(shell.body, ctx);
                vec![]
            },
            2,
        );
        let (changed, text) = run(&mut ctx, shell.func, false);
        assert!(!changed, "{text}");
    }

    /// An op with no lane form (sdiv) must bail.
    #[test]
    fn bails_on_unsupported_op() {
        let mut ctx = Context::new();
        let ptr: TypeHandle = PointerType::get(&mut ctx, 0).into();
        let i64t = i64_ty(&mut ctx);
        let shell = build_loop(
            &mut ctx,
            vec![ptr, ptr, i64t],
            vec![],
            |_, _| vec![],
            |ctx, shell, iv, _| {
                let a = shell.entry.deref(ctx).get_argument(0);
                let out = shell.entry.deref(ctx).get_argument(1);
                let i32t = i32_ty(ctx);
                let ga = GetElementPtrOp::new(ctx, a, vec![GepIndex::Value(iv)], i32t);
                ga.get_operation().insert_at_back(shell.body, ctx);
                let ga_v = ga.get_result(ctx);
                let l = LoadOp::new(ctx, ga_v, i32t);
                l.get_operation().insert_at_back(shell.body, ctx);
                let l_v = l.get_result(ctx);
                let d = pliron_llvm::ops::SDivOp::new(ctx, l_v, l_v);
                d.get_operation().insert_at_back(shell.body, ctx);
                let d_v = d.get_result(ctx);
                let go = GetElementPtrOp::new(ctx, out, vec![GepIndex::Value(iv)], i32t);
                go.get_operation().insert_at_back(shell.body, ctx);
                let go_v = go.get_result(ctx);
                StoreOp::new(ctx, d_v, go_v)
                    .get_operation()
                    .insert_at_back(shell.body, ctx);
                vec![]
            },
            2,
        );
        let (changed, text) = run(&mut ctx, shell.func, false);
        assert!(!changed, "{text}");
    }

    fn reduction_loop(ctx: &mut Context, fp: bool) -> Loop {
        let ptr: TypeHandle = PointerType::get(ctx, 0).into();
        let i64t = i64_ty(ctx);
        let acc_ty: TypeHandle = if fp {
            FP32Type::get(ctx).into()
        } else {
            i32_ty(ctx)
        };
        let shell = build_loop(
            ctx,
            vec![ptr, i64t],
            vec![acc_ty],
            move |ctx, entry| {
                // The accumulator's init: an entry-block 0 of elem type.
                let c = if fp {
                    ConstantOp::new(ctx, Box::new(FPSingleAttr::from(0.0f32)))
                } else {
                    let ty = IntegerType::get(ctx, 32, Signedness::Signless);
                    ConstantOp::new(
                        ctx,
                        Box::new(IntegerAttr::new(
                            ty,
                            APInt::from_u64(0, NonZero::new(32).unwrap()),
                        )),
                    )
                };
                c.get_operation().insert_at_back(entry, ctx);
                vec![c.get_result(ctx)]
            },
            |ctx, shell, iv, carried| {
                let a = shell.entry.deref(ctx).get_argument(0);
                let elem: TypeHandle = if fp {
                    FP32Type::get(ctx).into()
                } else {
                    i32_ty(ctx)
                };
                let ga = GetElementPtrOp::new(ctx, a, vec![GepIndex::Value(iv)], elem);
                ga.get_operation().insert_at_back(shell.body, ctx);
                let ga_v = ga.get_result(ctx);
                let l = LoadOp::new(ctx, ga_v, elem);
                l.get_operation().insert_at_back(shell.body, ctx);
                let l_v = l.get_result(ctx);
                let acc2: Value = if fp {
                    let s = FAddOp::new_with_fast_math_flags(
                        ctx,
                        carried[0],
                        l_v,
                        Default::default(),
                    );
                    s.get_operation().insert_at_back(shell.body, ctx);
                    s.get_result(ctx)
                } else {
                    let s = AddOp::new_with_overflow_flag(
                        ctx,
                        carried[0],
                        l_v,
                        Default::default(),
                    );
                    s.get_operation().insert_at_back(shell.body, ctx);
                    s.get_result(ctx)
                };
                vec![acc2]
            },
            1,
        );
        shell
    }

    /// Integer add-reductions are always exact: vectorized into 4 partial
    /// sums plus `ll.vreduce add` at the vector exit.
    #[test]
    fn vectorizes_integer_reduction() {
        let mut ctx = Context::new();
        let shell = reduction_loop(&mut ctx, false);
        let (changed, text) = run(&mut ctx, shell.func, false);
        assert!(changed, "{text}");
        assert!(text.contains("ll.vreduce add"), "{text}");
        assert!(text.contains("ll.vbinop add"), "{text}");
    }

    /// FP reductions reassociate: OFF by default, ON under the knob.
    #[test]
    fn fp_reduction_only_under_knob() {
        let mut ctx = Context::new();
        let shell = reduction_loop(&mut ctx, true);
        let (changed, text) = run(&mut ctx, shell.func, false);
        assert!(!changed, "{text}");
        assert!(!text.contains("ll.vreduce"), "{text}");

        let mut ctx = Context::new();
        let shell = reduction_loop(&mut ctx, true);
        let (changed, text) = run(&mut ctx, shell.func, true);
        assert!(changed, "{text}");
        assert!(text.contains("ll.vreduce fadd"), "{text}");
    }
}
