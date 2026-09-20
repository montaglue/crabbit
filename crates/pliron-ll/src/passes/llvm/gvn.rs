//! Global value numbering (docs/MIDEND-PLAN.md item 1): dominator-scoped
//! CSE of pure operations, plus redundant-load elimination and
//! store-to-load forwarding across blocks.
//!
//! Memory model (deliberately conservative, syntactic): two accesses are
//! to the *same* address only when their addresses are the same SSA value
//! or structurally identical GEP chains off the same base with identical
//! indices (`AddrKey`). Availability of a memory value is killed by any
//! store to a non-identical address key, any call, and any op outside the
//! known-benign set. A load is resolved by scanning backwards from its
//! position within its block and then through the *unique-predecessor
//! chain* (every path into the load passes through those blocks, so a
//! match dominates the load and no kill can be bypassed). Loop-carried
//! forwarding across a back edge (the value stored last iteration) needs
//! block-argument threading and is NOT implemented — see the plan.
//!
//! The dialect does not model volatile or atomic memory accesses; like
//! the existing simplify forwarding, loads/stores are treated as plain.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    context::{Context, Ptr},
    dialects::{
        llvm::{
            attributes::{FCmpPredicateAttr, FastmathFlags, FastmathFlagsAttr, ICmpPredicateAttr},
            op_interfaces::IsDeclaration,
            ops::{
                AShrOp, AddOp, AndOp, BitcastOp, ExtractValueOp, FAddOp, FCmpOp, FDivOp, FMulOp,
                FNegOp, FPExtOp, FPToSIOp, FPToUIOp, FPTruncOp, FRemOp, FSubOp, GetElementPtrOp,
                GepIndex, ICmpOp, InsertValueOp, IntToPtrOp, LShrOp, LoadOp, MulOp, OrOp,
                PtrToIntOp, SDivOp, SExtOp, SIToFPOp, SRemOp, SelectOp, ShlOp, StoreOp, SubOp,
                TruncOp, UDivOp, UIToFPOp, URemOp, XorOp, ZExtOp,
            },
        },
    },
    ir::{
        basic_block::BasicBlock,
        op::{Op, OpId},
        operation::Operation,
        region::Region,
        r#type::{TypeHandle, Typed},
        value::Value,
    },
    linked_list::ContainsLinkedList,
    conversion::pass::{AnalysisManager, Pass, PassResult, changed, unchanged},
};

use crate::passes::aarch64::opmap;
use super::{
    analysis::{dominator_tree, memory_benign_op_ids},
    inline::collect_functions,
    midend_gate::midend_disabled,
};
use pliron::builtin::op_interfaces::AtMostOneRegionInterface as _;

const MAX_ITERATIONS: usize = 8;
/// Bound on the unique-predecessor chain walked per load.
const MAX_CHAIN_BLOCKS: usize = 64;

pub struct LLVMGvnPass;

impl Pass for LLVMGvnPass {
    fn name(&self) -> &str {
        "llvm-gvn"
    }

    fn run(
        &mut self,
        root: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> pliron::result::Result<PassResult> {
        if midend_disabled("gvn") {
            return Ok(unchanged());
        }
        let benign = memory_benign_op_ids();
        let mut any = false;
        for func in collect_functions(ctx, root) {
            if func.is_declaration(ctx) {
                continue;
            }
            let region = func
                .get_region(ctx)
                .expect("llvm.func definition must have a body");
            for _ in 0..MAX_ITERATIONS {
                let mut changed_now = cse(ctx, region);
                changed_now |= eliminate_loads(ctx, region, &benign);
                any |= changed_now;
                if !changed_now {
                    break;
                }
            }
        }
        Ok(if any { changed() } else { unchanged() })
    }
}

// ============================================================================
// Dominator-scoped CSE of pure operations
// ============================================================================

/// A hashable identity for a pure operation: same key ⇒ same value.
/// Only op kinds listed here participate; attributes that change meaning
/// (icmp predicate, GEP element type and indices, extractvalue indices)
/// are part of the key. Constants are left to the folder.
#[derive(PartialEq, Eq, Hash, Clone)]
enum ExprKey {
    /// The `(nsw, nuw)` pair is part of a binary op's identity: merging a
    /// wrapping op into a dominating nsw/nuw op (or vice versa) would give
    /// overflowing inputs poison semantics they never had. Conservative:
    /// ops CSE only when their flags are identical (LLVM's GVN intersects
    /// flags on merge instead; equality is the safe subset).
    Bin(OpId, Value, Value, TypeHandle, (bool, bool)),
    /// FP identity carries the fast-math flags for the same reason the
    /// integer key carries `(nsw, nuw)`: merging into an op with more
    /// flags would grant its inputs poison semantics (nnan/ninf) they
    /// never had. Only bit-identical ops merge — no reassociation, so
    /// strict-FP-safe.
    FBin(OpId, Value, Value, TypeHandle, FastmathFlags),
    FNeg(Value, FastmathFlags),
    Cast(OpId, Value, TypeHandle),
    ICmp(ICmpPredicateAttr, Value, Value),
    FCmp(FCmpPredicateAttr, Value, Value, FastmathFlags),
    Select(Value, Value, Value, FastmathFlags),
    Gep(TypeHandle, Value, Vec<IdxKey>),
    Extract(Value, Vec<u32>),
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub(crate) enum IdxKey {
    Const(u32),
    Val(Value),
}

/// The key-shape sets below say how to build an op kind's identity, not
/// whether it is safe — safety is analysis.rs's table (everything here is
/// `Pure` there). An op missing here merely never CSEs.
fn bin_op_ids() -> FxHashSet<OpId> {
    let mut ids = FxHashSet::default();
    ids.insert(AddOp::get_opid_static());
    ids.insert(SubOp::get_opid_static());
    ids.insert(MulOp::get_opid_static());
    ids.insert(AndOp::get_opid_static());
    ids.insert(OrOp::get_opid_static());
    ids.insert(XorOp::get_opid_static());
    ids.insert(ShlOp::get_opid_static());
    ids.insert(LShrOp::get_opid_static());
    ids.insert(AShrOp::get_opid_static());
    ids.insert(UDivOp::get_opid_static());
    ids.insert(SDivOp::get_opid_static());
    ids.insert(URemOp::get_opid_static());
    ids.insert(SRemOp::get_opid_static());
    ids
}

fn fbin_op_ids() -> FxHashSet<OpId> {
    let mut ids = FxHashSet::default();
    ids.insert(FAddOp::get_opid_static());
    ids.insert(FSubOp::get_opid_static());
    ids.insert(FMulOp::get_opid_static());
    ids.insert(FDivOp::get_opid_static());
    ids.insert(FRemOp::get_opid_static());
    ids
}

fn cast_op_ids() -> FxHashSet<OpId> {
    let mut ids = FxHashSet::default();
    ids.insert(ZExtOp::get_opid_static());
    ids.insert(SExtOp::get_opid_static());
    ids.insert(TruncOp::get_opid_static());
    ids.insert(BitcastOp::get_opid_static());
    ids.insert(IntToPtrOp::get_opid_static());
    ids.insert(PtrToIntOp::get_opid_static());
    ids.insert(FPExtOp::get_opid_static());
    ids.insert(FPTruncOp::get_opid_static());
    ids.insert(SIToFPOp::get_opid_static());
    ids.insert(UIToFPOp::get_opid_static());
    ids.insert(FPToSIOp::get_opid_static());
    ids.insert(FPToUIOp::get_opid_static());
    ids
}

/// The op's fast-math flags; empty when the attribute is absent (absent
/// and empty both mean strict FP).
fn fast_math_flags(ctx: &Context, op: Ptr<Operation>) -> FastmathFlags {
    op.deref(ctx)
        .attributes
        .get::<FastmathFlagsAttr>(&pliron_llvm::op_interfaces::ATTR_KEY_FAST_MATH_FLAGS)
        .map(|attr| attr.0)
        .unwrap_or(FastmathFlags::empty())
}

fn expr_key(
    ctx: &Context,
    op: Ptr<Operation>,
    bins: &FxHashSet<OpId>,
    fbins: &FxHashSet<OpId>,
    casts: &FxHashSet<OpId>,
) -> Option<ExprKey> {
    let opid = Operation::get_opid(op, ctx);
    let operation = op.deref(ctx);
    if bins.contains(&opid) {
        let flags = operation
            .attributes
            .get::<pliron_llvm::attributes::IntegerOverflowFlagsAttr>(
                &pliron_llvm::op_interfaces::ATTR_KEY_INTEGER_OVERFLOW_FLAGS,
            )
            .map(|flag| (flag.nsw, flag.nuw))
            .unwrap_or((false, false));
        return Some(ExprKey::Bin(
            opid,
            operation.get_operand(0),
            operation.get_operand(1),
            operation.get_result(0).get_type(ctx),
            flags,
        ));
    }
    if fbins.contains(&opid) {
        let lhs = operation.get_operand(0);
        let rhs = operation.get_operand(1);
        let ty = operation.get_result(0).get_type(ctx);
        return Some(ExprKey::FBin(opid, lhs, rhs, ty, fast_math_flags(ctx, op)));
    }
    if casts.contains(&opid) {
        return Some(ExprKey::Cast(
            opid,
            operation.get_operand(0),
            operation.get_result(0).get_type(ctx),
        ));
    }
    if opid == FNegOp::get_opid_static() {
        let arg = operation.get_operand(0);
        return Some(ExprKey::FNeg(arg, fast_math_flags(ctx, op)));
    }
    if opid == FCmpOp::get_opid_static() {
        let lhs = operation.get_operand(0);
        let rhs = operation.get_operand(1);
        let fcmp = FCmpOp::from_operation(op);
        return Some(ExprKey::FCmp(
            fcmp.predicate(ctx),
            lhs,
            rhs,
            fast_math_flags(ctx, op),
        ));
    }
    if opid == SelectOp::get_opid_static() {
        let cond = operation.get_operand(0);
        let true_val = operation.get_operand(1);
        let false_val = operation.get_operand(2);
        // Select's fast-math flags live under their own attribute key.
        let flags = SelectOp::from_operation(op)
            .get_attr_llvm_select_fast_math_flags(ctx)
            .map(|attr| attr.0)
            .unwrap_or(FastmathFlags::empty());
        return Some(ExprKey::Select(cond, true_val, false_val, flags));
    }
    if opid == ICmpOp::get_opid_static() {
        let icmp = ICmpOp::from_operation(op);
        return Some(ExprKey::ICmp(
            icmp.predicate(ctx),
            operation.get_operand(0),
            operation.get_operand(1),
        ));
    }
    if opid == GetElementPtrOp::get_opid_static() {
        let gep = GetElementPtrOp::from_operation(op);
        let indices = gep
            .indices(ctx)
            .into_iter()
            .map(|index| match index {
                GepIndex::Constant(c) => IdxKey::Const(c),
                GepIndex::Value(v) => IdxKey::Val(v),
            })
            .collect();
        return Some(ExprKey::Gep(
            gep.src_elem_type(ctx),
            operation.get_operand(0),
            indices,
        ));
    }
    if opid == ExtractValueOp::get_opid_static() {
        let extract = ExtractValueOp::from_operation(op);
        return Some(ExprKey::Extract(
            operation.get_operand(0),
            extract.indices(ctx),
        ));
    }
    None
}

/// ADJOINT (backward attribution): GVN merges are N→1 — the surviving
/// `value`'s defining op absorbs the erased op's identity as a
/// multi-parent `derived_from` (equal weights; docs/PROFILE-FEEDBACK-
/// BACKWARD.md "merge" rule). Applies to CSE dedup and to redundant-load
/// elimination / store-to-load forwarding alike.
fn replace_op_with_value(ctx: &mut Context, op: Ptr<Operation>, value: Value) {
    if let Some(surviving) = value.defining_op() {
        let mut sources = opmap::effective_sources(ctx, surviving);
        let erased_sources = opmap::effective_sources(ctx, op);
        if !erased_sources.is_empty() && !sources.is_empty() {
            sources.extend(erased_sources);
            opmap::set_derived_from_many(ctx, surviving, sources);
        }
    }
    let result = op.deref(ctx).get_result(0);
    result.replace_some_uses_with(ctx, |_, _| true, &value);
    Operation::erase(op, ctx);
}

/// Pre-order walk of the dominator tree with a scoped expression table:
/// an op whose key is bound in any enclosing scope is replaced by the
/// bound value (which dominates it by construction).
fn cse(ctx: &mut Context, region: Ptr<Region>) -> bool {
    let dom = dominator_tree(ctx, region);
    let Some(root) = dom.root() else {
        return false;
    };
    let bins = bin_op_ids();
    let fbins = fbin_op_ids();
    let casts = cast_op_ids();
    let mut scopes: Vec<FxHashMap<ExprKey, Value>> = Vec::new();
    let mut changed = false;

    enum Step {
        Enter(Ptr<BasicBlock>),
        Exit,
    }
    let mut stack = vec![Step::Enter(root)];
    while let Some(step) = stack.pop() {
        let block = match step {
            Step::Exit => {
                scopes.pop();
                continue;
            }
            Step::Enter(block) => block,
        };
        scopes.push(FxHashMap::default());
        stack.push(Step::Exit);
        for child in dom.children(&block) {
            stack.push(Step::Enter(child));
        }
        let ops: Vec<_> = block.deref(ctx).iter(ctx).collect();
        for op in ops {
            let Some(key) = expr_key(ctx, op, &bins, &fbins, &casts) else {
                continue;
            };
            match scopes.iter().rev().find_map(|scope| scope.get(&key)) {
                Some(&existing) => {
                    replace_op_with_value(ctx, op, existing);
                    changed = true;
                }
                None => {
                    let result = op.deref(ctx).get_result(0);
                    scopes
                        .last_mut()
                        .expect("one scope per dominator-preorder frame")
                        .insert(key, result);
                }
            }
        }
    }
    changed
}

// ============================================================================
// Redundant-load elimination / store-to-load forwarding
// ============================================================================

/// Structural address identity: the SSA value itself, seen through
/// pointer-preserving bitcasts, with GEPs compared by (base, element
/// type, indices).
#[derive(PartialEq, Eq, Hash, Clone)]
pub(crate) enum AddrKey {
    Root(Value),
    Gep(Box<AddrKey>, TypeHandle, Vec<IdxKey>),
}

pub(crate) fn addr_key(ctx: &Context, addr: Value) -> AddrKey {
    if let Some(def) = addr.defining_op() {
        let opid = Operation::get_opid(def, ctx);
        if opid == GetElementPtrOp::get_opid_static() {
            let gep = GetElementPtrOp::from_operation(def);
            let base = def.deref(ctx).get_operand(0);
            let indices = gep
                .indices(ctx)
                .into_iter()
                .map(|index| match index {
                    GepIndex::Constant(c) => IdxKey::Const(c),
                    GepIndex::Value(v) => IdxKey::Val(v),
                })
                .collect();
            return AddrKey::Gep(
                Box::new(addr_key(ctx, base)),
                gep.src_elem_type(ctx),
                indices,
            );
        }
        if opid == BitcastOp::get_opid_static() {
            return addr_key(ctx, def.deref(ctx).get_operand(0));
        }
    }
    AddrKey::Root(addr)
}

/// What the backward scan found for one load.
enum Resolution {
    Value(Value),
    /// A dominating store to the same slot whose value has a *layout-
    /// compatible but not identical* aggregate type (the importer emits
    /// signedness-mismatched twins, e.g. `{ptr, ui64}` stored into a
    /// `{ptr, i64}` slot — the exact shape pin_type_punned_slots pins away
    /// from mem2reg). The whole value cannot replace the load (its type
    /// differs), but each `extract_value` use can be forwarded per field.
    Aggregate(Value),
    Killed,
    Unknown,
}

fn eliminate_loads(ctx: &mut Context, region: Ptr<Region>, benign: &FxHashSet<OpId>) -> bool {
    let mut changed = false;
    let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
    for block in blocks {
        let loads: Vec<_> = block
            .deref(ctx)
            .iter(ctx)
            .filter(|op| Operation::get_opid(*op, ctx) == LoadOp::get_opid_static())
            .collect();
        for load in loads {
            let load_op = LoadOp::from_operation(load);
            let key = addr_key(ctx, load_op.get_operand_address(ctx));
            let want_ty = load.deref(ctx).get_result(0).get_type(ctx);
            match resolve_load(ctx, block, load, &key, want_ty, benign) {
                Resolution::Value(value) => {
                    replace_op_with_value(ctx, load, value);
                    changed = true;
                }
                Resolution::Aggregate(stored) => {
                    changed |= forward_extracted_fields(ctx, load, stored);
                }
                Resolution::Killed | Resolution::Unknown => {}
            }
        }
    }
    changed
}

/// Scan backwards from `load` through its block and then the unique-
/// predecessor chain. Every block on that chain is on every path to the
/// load, so a matching store/load found there dominates it and no kill
/// on any path is missed.
fn resolve_load(
    ctx: &Context,
    start_block: Ptr<BasicBlock>,
    load: Ptr<Operation>,
    key: &AddrKey,
    want_ty: TypeHandle,
    benign: &FxHashSet<OpId>,
) -> Resolution {
    let mut block = start_block;
    let mut before = Some(load);
    let mut visited = FxHashSet::default();
    for _ in 0..MAX_CHAIN_BLOCKS {
        if !visited.insert(block) {
            // A single-block self-loop: crossing the back edge would need
            // loop-carried threading; stop.
            return Resolution::Unknown;
        }
        let ops: Vec<_> = block.deref(ctx).iter(ctx).collect();
        let upto = match before {
            Some(mark) => ops.iter().position(|op| *op == mark).unwrap_or(ops.len()),
            None => ops.len(),
        };
        for op in ops[..upto].iter().rev() {
            let opid = Operation::get_opid(*op, ctx);
            if opid == StoreOp::get_opid_static() {
                let store = StoreOp::from_operation(*op);
                let store_key = addr_key(ctx, store.get_operand_address(ctx));
                let value = store.get_operand_value(ctx);
                if store_key == *key {
                    if value.get_type(ctx) == want_ty {
                        return Resolution::Value(value);
                    }
                    if layout_compatible(ctx, value.get_type(ctx), want_ty) {
                        return Resolution::Aggregate(value);
                    }
                    return Resolution::Killed; // same slot, other width
                }
                return Resolution::Killed; // may alias
            }
            if opid == LoadOp::get_opid_static() {
                let other = LoadOp::from_operation(*op);
                if addr_key(ctx, other.get_operand_address(ctx)) == *key {
                    let result = op.deref(ctx).get_result(0);
                    if result.get_type(ctx) == want_ty {
                        return Resolution::Value(result);
                    }
                }
                continue; // loads never kill
            }
            if !benign.contains(&opid) {
                return Resolution::Killed; // call / unknown memory effects
            }
        }
        let preds = block.preds(ctx);
        match preds.as_slice() {
            [single] => {
                block = *single;
                before = None;
            }
            _ => return Resolution::Unknown,
        }
    }
    Resolution::Unknown
}

/// Same in-memory layout: identical types, integers of equal width
/// (signedness is a frontend annotation, not a layout property), or
/// aggregates thereof with identical shape. Field offsets of two
/// layout-compatible aggregates coincide, so a load at the mismatched
/// type reads back exactly the stored fields.
fn layout_compatible(ctx: &Context, a: TypeHandle, b: TypeHandle) -> bool {
    use crate::dialects::builtin::types::IntegerType;
    use crate::dialects::llvm::types::{ArrayType, StructType};
    if a == b {
        return true;
    }
    let (a_ref, b_ref) = (a.deref(ctx), b.deref(ctx));
    if let (Some(a_int), Some(b_int)) = (
        a_ref.downcast_ref::<IntegerType>(),
        b_ref.downcast_ref::<IntegerType>(),
    ) {
        return a_int.width() == b_int.width();
    }
    if let (Some(a_struct), Some(b_struct)) = (
        a_ref.downcast_ref::<StructType>(),
        b_ref.downcast_ref::<StructType>(),
    ) {
        if a_struct.is_opaque()
            || b_struct.is_opaque()
            || a_struct.num_fields() != b_struct.num_fields()
        {
            return false;
        }
        let a_fields: Vec<_> = a_struct.fields().collect();
        let b_fields: Vec<_> = b_struct.fields().collect();
        drop(a_ref);
        drop(b_ref);
        return a_fields
            .into_iter()
            .zip(b_fields)
            .all(|(a_field, b_field)| layout_compatible(ctx, a_field, b_field));
    }
    if let (Some(a_array), Some(b_array)) = (
        a_ref.downcast_ref::<ArrayType>(),
        b_ref.downcast_ref::<ArrayType>(),
    ) {
        if a_array.size() != b_array.size() {
            return false;
        }
        let (a_elem, b_elem) = (a_array.elem_type(), b_array.elem_type());
        drop(a_ref);
        drop(b_ref);
        return layout_compatible(ctx, a_elem, b_elem);
    }
    false
}

/// The value stored at field path `indices` of `aggregate`, recovered by
/// walking its `insert_value` chain. Exact-path matches only; a partial
/// overlap (one path prefixes the other) means the field's bytes were
/// assembled from more than one insert, so give up.
fn stored_field(ctx: &Context, mut aggregate: Value, indices: &[u32]) -> Option<Value> {
    loop {
        let def = aggregate.defining_op()?;
        if Operation::get_opid(def, ctx) != InsertValueOp::get_opid_static() {
            return None;
        }
        let insert = InsertValueOp::from_operation(def);
        let insert_indices = insert.indices(ctx);
        if insert_indices == indices {
            return Some(def.deref(ctx).get_operand(1));
        }
        let min_len = indices.len().min(insert_indices.len());
        if indices[..min_len] == insert_indices[..min_len] {
            return None; // partial overlap
        }
        aggregate = def.deref(ctx).get_operand(0);
    }
}

/// Forward a layout-compatible aggregate store through the load's
/// `extract_value` uses: each extracted field is replaced by the exact SSA
/// value inserted at that path (types must match exactly, so no signedness
/// drift ever reaches the field's users). The load itself is erased once
/// nothing uses it; the now-dead store/insert chain is left to dse/adce.
fn forward_extracted_fields(ctx: &mut Context, load: Ptr<Operation>, stored: Value) -> bool {
    let result = load.deref(ctx).get_result(0);
    let users: Vec<Ptr<Operation>> = result
        .uses(ctx)
        .iter()
        .map(|load_use| load_use.user_op())
        .collect();
    let mut changed = false;
    for user in users {
        if Operation::get_opid(user, ctx) != ExtractValueOp::get_opid_static() {
            continue;
        }
        let indices = ExtractValueOp::from_operation(user).indices(ctx);
        let Some(field) = stored_field(ctx, stored, &indices) else {
            continue;
        };
        if field.get_type(ctx) != user.deref(ctx).get_result(0).get_type(ctx) {
            continue;
        }
        replace_op_with_value(ctx, user, field);
        changed = true;
    }
    if changed && load.deref(ctx).get_result(0).uses(ctx).is_empty() {
        Operation::erase(load, ctx);
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use pliron::builtin::op_interfaces::{OneResultInterface as _};
    use pliron_llvm::op_interfaces::{CastOpInterface as _, IntBinArithOpWithOverflowFlag as _};
    use crate::{
        dialects::{
            builtin::{
                attributes::IntegerAttr,
                ops::ConstantOp,
                types::{IntegerType, Signedness},
            },
            llvm::{
                attributes::LinkageAttr,
                ops::{BrOp, CallOp, CondBrOp, FuncOp, ReturnOp},
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

    fn run_gvn(ctx: &mut Context, func: FuncOp) -> String {
        LLVMGvnPass
            .run(func.get_operation(), ctx, &mut AnalysisManager::default())
            .unwrap();
        format!("{}", func.get_operation().disp(ctx))
    }

    /// One function: i64 f(i64 a, ptr p). Returns (func, entry, a, p).
    fn scaffold(ctx: &mut Context) -> (FuncOp, Ptr<BasicBlock>, Value, Value) {
        let i64_ty: TypeHandle = int_ty(ctx, 64).into();
        let ptr_ty: TypeHandle = PointerType::get(ctx, 0).into();
        let fn_ty = FuncType::get(ctx, i64_ty, vec![i64_ty, ptr_ty], false);
        let func = FuncOp::new(ctx, "g".try_into().unwrap(), fn_ty);
        func.set_attr_llvm_function_linkage(ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(ctx);
        let entry = func.get_entry_block(ctx).unwrap();
        let a = entry.deref(ctx).get_argument(0);
        let p = entry.deref(ctx).get_argument(1);
        (func, entry, a, p)
    }

    #[test]
    fn cse_unifies_identical_adds_across_blocks() {
        let mut ctx = Context::new();
        let (func, entry, a, _p) = scaffold(&mut ctx);
        let region = func.get_region(&ctx).unwrap();
        let next = BasicBlock::new(&mut ctx, None, vec![]);
        next.insert_at_back(region, &ctx);

        let add1 = AddOp::new_with_overflow_flag(&mut ctx, a, a, Default::default());
        add1.get_operation().insert_at_back(entry, &ctx);
        BrOp::new(&mut ctx, next, vec![])
            .get_operation()
            .insert_at_back(entry, &ctx);
        // Identical add in a dominated block: must be CSE'd to add1.
        let add2 = AddOp::new_with_overflow_flag(&mut ctx, a, a, Default::default());
        add2.get_operation().insert_at_back(next, &ctx);
        let add2_v = add2.get_result(&ctx);
        ReturnOp::new(&mut ctx, Some(add2_v))
            .get_operation()
            .insert_at_back(next, &ctx);

        let text = run_gvn(&mut ctx, func);
        assert_eq!(text.matches("llvm.add").count(), 1, "{text}");
    }

    #[test]
    fn cse_respects_overflow_flags() {
        use pliron_llvm::attributes::IntegerOverflowFlagsAttr;
        let mut ctx = Context::new();
        let (func, entry, a, _p) = scaffold(&mut ctx);

        // nsw add dominating a plain wrapping add: MUST NOT merge (the
        // wrapping op would inherit poison-on-overflow semantics).
        let nsw = IntegerOverflowFlagsAttr { nsw: true, nuw: false };
        let add_nsw = AddOp::new_with_overflow_flag(&mut ctx, a, a, nsw.clone());
        add_nsw.get_operation().insert_at_back(entry, &ctx);
        let add_plain = AddOp::new_with_overflow_flag(&mut ctx, a, a, Default::default());
        add_plain.get_operation().insert_at_back(entry, &ctx);
        // Two identical nsw adds: MUST merge.
        let add_nsw2 = AddOp::new_with_overflow_flag(&mut ctx, a, a, nsw);
        add_nsw2.get_operation().insert_at_back(entry, &ctx);
        let ret_v = add_nsw2.get_result(&ctx);
        ReturnOp::new(&mut ctx, Some(ret_v))
            .get_operation()
            .insert_at_back(entry, &ctx);

        let text = run_gvn(&mut ctx, func);
        assert_eq!(
            text.matches("llvm.add").count(),
            2,
            "nsw+plain must stay distinct, nsw+nsw must merge:\n{text}"
        );
    }

    /// f32 g(f32 x, ptr p). Returns (func, entry, x, p).
    fn fp_scaffold(ctx: &mut Context) -> (FuncOp, Ptr<BasicBlock>, Value, Value) {
        let f32_ty: TypeHandle = crate::dialects::builtin::types::FP32Type::get(ctx).into();
        let ptr_ty: TypeHandle = PointerType::get(ctx, 0).into();
        let fn_ty = FuncType::get(ctx, f32_ty, vec![f32_ty, ptr_ty], false);
        let func = FuncOp::new(ctx, "gf".try_into().unwrap(), fn_ty);
        func.set_attr_llvm_function_linkage(ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(ctx);
        let entry = func.get_entry_block(ctx).unwrap();
        let x = entry.deref(ctx).get_argument(0);
        let p = entry.deref(ctx).get_argument(1);
        (func, entry, x, p)
    }

    #[test]
    fn cse_unifies_identical_fadds_and_selects() {
        use pliron_llvm::op_interfaces::FloatBinArithOpWithFastMathFlags as _;
        let mut ctx = Context::new();
        let (func, entry, x, _p) = fp_scaffold(&mut ctx);
        let i1: TypeHandle = int_ty(&mut ctx, 1).into();

        // Two bit-identical fadds must merge; one with different
        // fast-math flags must stay (same conservatism as nsw/nuw).
        let fadd1 = FAddOp::new_with_fast_math_flags(&mut ctx, x, x, FastmathFlagsAttr::default());
        fadd1.get_operation().insert_at_back(entry, &ctx);
        let fadd1_v = fadd1.get_result(&ctx);
        let fadd2 = FAddOp::new_with_fast_math_flags(&mut ctx, x, x, FastmathFlagsAttr::default());
        fadd2.get_operation().insert_at_back(entry, &ctx);
        let fadd2_v = fadd2.get_result(&ctx);
        let nnan = FAddOp::new_with_fast_math_flags(
            &mut ctx,
            x,
            x,
            FastmathFlagsAttr(FastmathFlags::NNAN),
        );
        nnan.get_operation().insert_at_back(entry, &ctx);
        let nnan_v = nnan.get_result(&ctx);

        // Identical selects over the merged fadds must merge too (the
        // second's operand is rewritten to fadd1 before it is keyed).
        let cond = crate::dialects::llvm::ops::UndefOp::new(&mut ctx, i1);
        cond.get_operation().insert_at_back(entry, &ctx);
        let cond_v = cond.get_result(&ctx);
        let sel1 = SelectOp::new(&mut ctx, cond_v, fadd1_v, nnan_v);
        sel1.get_operation().insert_at_back(entry, &ctx);
        let sel2 = SelectOp::new(&mut ctx, cond_v, fadd2_v, nnan_v);
        sel2.get_operation().insert_at_back(entry, &ctx);
        let sel2_v = sel2.get_operation().deref(&ctx).get_result(0);
        ReturnOp::new(&mut ctx, Some(sel2_v))
            .get_operation()
            .insert_at_back(entry, &ctx);

        let text = run_gvn(&mut ctx, func);
        assert_eq!(
            text.matches("llvm.fadd").count(),
            2,
            "identical fadds must merge, the nnan one must stay:\n{text}"
        );
        assert_eq!(text.matches("llvm.select").count(), 1, "{text}");
    }

    #[test]
    fn fadd_does_not_kill_store_to_load_forwarding() {
        use pliron_llvm::op_interfaces::FloatBinArithOpWithFastMathFlags as _;
        let mut ctx = Context::new();
        let (func, entry, x, p) = fp_scaffold(&mut ctx);
        let f32_ty: TypeHandle = crate::dialects::builtin::types::FP32Type::get(&mut ctx).into();

        // store x -> p; fadd; load p — the pure fadd must not kill the
        // forwarding window (it used to, as an unknown-effect op).
        StoreOp::new(&mut ctx, x, p)
            .get_operation()
            .insert_at_back(entry, &ctx);
        let fadd = FAddOp::new_with_fast_math_flags(&mut ctx, x, x, FastmathFlagsAttr::default());
        fadd.get_operation().insert_at_back(entry, &ctx);
        let load = LoadOp::new(&mut ctx, p, f32_ty);
        load.get_operation().insert_at_back(entry, &ctx);
        let load_v = load.get_result(&ctx);
        ReturnOp::new(&mut ctx, Some(load_v))
            .get_operation()
            .insert_at_back(entry, &ctx);

        let text = run_gvn(&mut ctx, func);
        assert!(!text.contains("llvm.load"), "{text}");
    }

    #[test]
    fn forwards_store_to_load_through_gep_across_blocks() {
        let mut ctx = Context::new();
        let (func, entry, a, p) = scaffold(&mut ctx);
        let region = func.get_region(&ctx).unwrap();
        let next = BasicBlock::new(&mut ctx, None, vec![]);
        next.insert_at_back(region, &ctx);
        let i64_ty: TypeHandle = int_ty(&mut ctx, 64).into();

        let gep1 = GetElementPtrOp::new(&mut ctx, p, vec![GepIndex::Value(a)], i64_ty);
        gep1.get_operation().insert_at_back(entry, &ctx);
        let gep1_v = gep1.get_result(&ctx);
        StoreOp::new(&mut ctx, a, gep1_v)
            .get_operation()
            .insert_at_back(entry, &ctx);
        BrOp::new(&mut ctx, next, vec![])
            .get_operation()
            .insert_at_back(entry, &ctx);
        // Structurally identical GEP in the next block; load must forward
        // the stored value even before CSE unifies the GEPs.
        let gep2 = GetElementPtrOp::new(&mut ctx, p, vec![GepIndex::Value(a)], i64_ty);
        gep2.get_operation().insert_at_back(next, &ctx);
        let gep2_v = gep2.get_result(&ctx);
        let load = LoadOp::new(&mut ctx, gep2_v, i64_ty);
        load.get_operation().insert_at_back(next, &ctx);
        let load_v = load.get_result(&ctx);
        ReturnOp::new(&mut ctx, Some(load_v))
            .get_operation()
            .insert_at_back(next, &ctx);

        let text = run_gvn(&mut ctx, func);
        assert!(!text.contains("llvm.load"), "{text}");
    }

    #[test]
    fn may_alias_store_and_call_kill_forwarding() {
        let mut ctx = Context::new();
        let (func, entry, a, p) = scaffold(&mut ctx);
        let i64_ty: TypeHandle = int_ty(&mut ctx, 64).into();

        // store a -> p; store 7 -> q (unknown other pointer: a GEP with a
        // different index); load p must NOT be forwarded.
        let i64_typed = int_ty(&mut ctx, 64);
        let one = ConstantOp::new(
            &mut ctx,
            Box::new(IntegerAttr::new(
                i64_typed,
                APInt::from_u64(1, NonZero::new(64).unwrap()),
            )),
        );
        one.get_operation().insert_at_back(entry, &ctx);
        StoreOp::new(&mut ctx, a, p)
            .get_operation()
            .insert_at_back(entry, &ctx);
        let one_v = one.get_result(&ctx);
        let gep = GetElementPtrOp::new(&mut ctx, p, vec![GepIndex::Value(one_v)], i64_ty);
        gep.get_operation().insert_at_back(entry, &ctx);
        let gep_v = gep.get_result(&ctx);
        StoreOp::new(&mut ctx, one_v, gep_v)
            .get_operation()
            .insert_at_back(entry, &ctx);
        let load = LoadOp::new(&mut ctx, p, i64_ty);
        load.get_operation().insert_at_back(entry, &ctx);
        let load_v = load.get_result(&ctx);
        ReturnOp::new(&mut ctx, Some(load_v))
            .get_operation()
            .insert_at_back(entry, &ctx);

        let text = run_gvn(&mut ctx, func);
        assert!(text.contains("llvm.load"), "{text}");

        // And a call kills a would-be load-load pair.
        let mut ctx = Context::new();
        let (func, entry, _a, p) = scaffold(&mut ctx);
        let i64_ty: TypeHandle = int_ty(&mut ctx, 64).into();
        let load1 = LoadOp::new(&mut ctx, p, i64_ty);
        load1.get_operation().insert_at_back(entry, &ctx);
        let void_fn = FuncType::get(&mut ctx, i64_ty, vec![], false);
        let call = CallOp::new(
            &mut ctx,
            pliron::builtin::op_interfaces::CallOpCallable::Direct("opaque".try_into().unwrap()),
            void_fn,
            vec![],
        );
        call.get_operation().insert_at_back(entry, &ctx);
        let load2 = LoadOp::new(&mut ctx, p, i64_ty);
        load2.get_operation().insert_at_back(entry, &ctx);
        let load2_v = load2.get_result(&ctx);
        ReturnOp::new(&mut ctx, Some(load2_v))
            .get_operation()
            .insert_at_back(entry, &ctx);
        let text = run_gvn(&mut ctx, func);
        assert_eq!(text.matches("llvm.load").count(), 2, "{text}");
    }

    /// Scaffold for the aggregate-forwarding tests: a `{ptr, i64}` slot
    /// whose store goes through a pointer bitcast (the pin_type_punned_slots
    /// shape) and carries a `{ptr, ui64}` value built by an insert chain
    /// (field 0 = `p`, field 1 = an unsigned constant). Returns the ops the
    /// callers place the load/extract after.
    struct AggScaffold {
        func: FuncOp,
        entry: Ptr<BasicBlock>,
        a: Value,
        p: Value,
        slot: Value,
        stored_struct_ty: TypeHandle,
        slot_struct_ty: TypeHandle,
    }

    fn agg_scaffold(ctx: &mut Context, store_field1_width: u32) -> AggScaffold {
        use crate::dialects::llvm::ops::{AllocaOp, UndefOp};
        use crate::dialects::llvm::types::StructType;
        let (func, entry, a, p) = scaffold(ctx);
        let ptr_ty: TypeHandle = PointerType::get(ctx, 0).into();
        let i64_signed: TypeHandle = IntegerType::get(ctx, 64, Signedness::Signed).into();
        let field1_unsigned: TypeHandle =
            IntegerType::get(ctx, store_field1_width, Signedness::Unsigned).into();
        let slot_struct_ty: TypeHandle =
            StructType::get_unnamed(ctx, vec![ptr_ty, i64_signed]).into();
        let stored_struct_ty: TypeHandle =
            StructType::get_unnamed(ctx, vec![ptr_ty, field1_unsigned]).into();

        let i64_typed = int_ty(ctx, 64);
        let one = ConstantOp::new(
            ctx,
            Box::new(IntegerAttr::new(
                i64_typed,
                APInt::from_u64(1, NonZero::new(64).unwrap()),
            )),
        );
        one.get_operation().insert_at_back(entry, ctx);
        let one_v = one.get_result(ctx);
        let alloca = AllocaOp::new(ctx, slot_struct_ty, one_v);
        alloca.get_operation().insert_at_back(entry, ctx);
        let slot = alloca.get_result(ctx);
        // The pinned shape: the store's address is a bitcast of the slot.
        let punned = BitcastOp::new(ctx, slot, ptr_ty);
        punned.get_operation().insert_at_back(entry, ctx);
        let punned_v = punned.get_result(ctx);
        let undef = UndefOp::new(ctx, stored_struct_ty);
        undef.get_operation().insert_at_back(entry, ctx);
        let undef_v = undef.get_result(ctx);
        let ins0 = InsertValueOp::new(ctx, undef_v, p, vec![0]);
        ins0.get_operation().insert_at_back(entry, ctx);
        let ins0_v = ins0.get_operation().deref(ctx).get_result(0);
        let field1_typed = IntegerType::get(ctx, store_field1_width, Signedness::Unsigned);
        let c32 = ConstantOp::new(
            ctx,
            Box::new(IntegerAttr::new(
                field1_typed,
                APInt::from_u64(32, NonZero::new(store_field1_width as usize).unwrap()),
            )),
        );
        c32.get_operation().insert_at_back(entry, ctx);
        let c32_v = c32.get_result(ctx);
        let ins1 = InsertValueOp::new(ctx, ins0_v, c32_v, vec![1]);
        ins1.get_operation().insert_at_back(entry, ctx);
        let ins1_v = ins1.get_operation().deref(ctx).get_result(0);
        StoreOp::new(ctx, ins1_v, punned_v)
            .get_operation()
            .insert_at_back(entry, ctx);
        AggScaffold {
            func,
            entry,
            a,
            p,
            slot,
            stored_struct_ty,
            slot_struct_ty,
        }
    }

    /// Appends `load slot; extract [index]; store a -> extracted-ptr; ret a`
    /// (the extract's result is used as a pointer so it stays live).
    fn append_load_extract(ctx: &mut Context, s: &AggScaffold, index: u32) {
        let load = LoadOp::new(ctx, s.slot, s.slot_struct_ty);
        load.get_operation().insert_at_back(s.entry, ctx);
        let load_v = load.get_result(ctx);
        let extract = ExtractValueOp::new(ctx, load_v, vec![index])
            .expect("field index in range");
        extract.get_operation().insert_at_back(s.entry, ctx);
        let extract_v = extract.get_operation().deref(ctx).get_result(0);
        if index == 0 {
            StoreOp::new(ctx, s.a, extract_v)
                .get_operation()
                .insert_at_back(s.entry, ctx);
        }
        ReturnOp::new(ctx, Some(s.a))
            .get_operation()
            .insert_at_back(s.entry, ctx);
    }

    #[test]
    fn forwards_aggregate_store_through_extract_across_signedness_twins() {
        // store {ptr, ui64} via bitcast; load {ptr, i64}; extract [0]:
        // layout-compatible twins, so the extracted pointer must forward
        // and the load must die.
        let mut ctx = Context::new();
        let s = agg_scaffold(&mut ctx, 64);
        assert!(layout_compatible(&ctx, s.stored_struct_ty, s.slot_struct_ty));
        append_load_extract(&mut ctx, &s, 0);
        let text = run_gvn(&mut ctx, s.func);
        assert!(!text.contains("llvm.extract_value"), "{text}");
        assert!(!text.contains("llvm.load"), "{text}");
    }

    #[test]
    fn aggregate_forwarding_bails_on_layout_mismatch() {
        // Field 1 stored as ui32 into an i64 slot: widths differ, so the
        // twins are NOT layout-compatible and nothing may forward.
        let mut ctx = Context::new();
        let s = agg_scaffold(&mut ctx, 32);
        assert!(!layout_compatible(&ctx, s.stored_struct_ty, s.slot_struct_ty));
        append_load_extract(&mut ctx, &s, 0);
        let text = run_gvn(&mut ctx, s.func);
        assert!(text.contains("llvm.extract_value"), "{text}");
        assert!(text.contains("llvm.load"), "{text}");
    }

    #[test]
    fn aggregate_forwarding_never_drifts_field_signedness() {
        // Extract [1] wants i64 but the inserted field value is ui64: the
        // exact-type guard must refuse (forwarding would retype every
        // downstream user), keeping load + extract.
        let mut ctx = Context::new();
        let s = agg_scaffold(&mut ctx, 64);
        append_load_extract(&mut ctx, &s, 1);
        let text = run_gvn(&mut ctx, s.func);
        assert!(text.contains("llvm.extract_value"), "{text}");
        assert!(text.contains("llvm.load"), "{text}");
    }

    #[test]
    fn aggregate_forwarding_killed_by_may_alias_store_and_call() {
        // An intervening store through an unrelated pointer may clobber
        // the slot under the syntactic alias model: no forwarding.
        let mut ctx = Context::new();
        let s = agg_scaffold(&mut ctx, 64);
        StoreOp::new(&mut ctx, s.a, s.p)
            .get_operation()
            .insert_at_back(s.entry, &mut ctx);
        append_load_extract(&mut ctx, &s, 0);
        let text = run_gvn(&mut ctx, s.func);
        assert!(text.contains("llvm.extract_value"), "{text}");

        // An intervening call may write anything: no forwarding.
        let mut ctx = Context::new();
        let s = agg_scaffold(&mut ctx, 64);
        let i64_ty: TypeHandle = int_ty(&mut ctx, 64).into();
        let void_fn = FuncType::get(&mut ctx, i64_ty, vec![], false);
        let call = CallOp::new(
            &mut ctx,
            pliron::builtin::op_interfaces::CallOpCallable::Direct("opaque".try_into().unwrap()),
            void_fn,
            vec![],
        );
        call.get_operation().insert_at_back(s.entry, &mut ctx);
        append_load_extract(&mut ctx, &s, 0);
        let text = run_gvn(&mut ctx, s.func);
        assert!(text.contains("llvm.extract_value"), "{text}");
    }

    #[test]
    fn merge_point_blocks_forwarding() {
        let mut ctx = Context::new();
        let (func, entry, a, p) = scaffold(&mut ctx);
        let region = func.get_region(&ctx).unwrap();
        let i1: TypeHandle = int_ty(&mut ctx, 1).into();
        let i64_ty: TypeHandle = int_ty(&mut ctx, 64).into();
        let left = BasicBlock::new(&mut ctx, None, vec![]);
        left.insert_at_back(region, &ctx);
        let right = BasicBlock::new(&mut ctx, None, vec![]);
        right.insert_at_back(region, &ctx);
        let join = BasicBlock::new(&mut ctx, None, vec![]);
        join.insert_at_back(region, &ctx);

        StoreOp::new(&mut ctx, a, p)
            .get_operation()
            .insert_at_back(entry, &ctx);
        let cond = crate::dialects::llvm::ops::UndefOp::new(&mut ctx, i1);
        cond.get_operation().insert_at_back(entry, &ctx);
        let cond_v = cond.get_result(&ctx);
        CondBrOp::new(&mut ctx, cond_v, left, vec![], right, vec![])
            .get_operation()
            .insert_at_back(entry, &ctx);
        // Right path clobbers p through an opaque call.
        BrOp::new(&mut ctx, join, vec![])
            .get_operation()
            .insert_at_back(left, &ctx);
        let void_fn = FuncType::get(&mut ctx, i64_ty, vec![], false);
        let call = CallOp::new(
            &mut ctx,
            pliron::builtin::op_interfaces::CallOpCallable::Direct("opaque".try_into().unwrap()),
            void_fn,
            vec![],
        );
        call.get_operation().insert_at_back(right, &ctx);
        BrOp::new(&mut ctx, join, vec![])
            .get_operation()
            .insert_at_back(right, &ctx);
        let load = LoadOp::new(&mut ctx, p, i64_ty);
        load.get_operation().insert_at_back(join, &ctx);
        let load_v = load.get_result(&ctx);
        ReturnOp::new(&mut ctx, Some(load_v))
            .get_operation()
            .insert_at_back(join, &ctx);

        let text = run_gvn(&mut ctx, func);
        assert!(text.contains("llvm.load"), "{text}");
    }
}
