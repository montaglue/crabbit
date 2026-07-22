//! Local LLVM-dialect simplifications: constant folding, insert/extract
//! value forwarding, block-local store-to-load forwarding, dead store
//! elimination, trivial phi elimination and dead code elimination.
//!
//! All rewrites are CFG-neutral, so the pass is safe to run both on the
//! block-argument form and on the phi form of the LLVM dialect.

use std::num::NonZero;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    context::{Context, Ptr},
    dialects::{
        builtin::{
            attributes::IntegerAttr,
            op_interfaces::{OneRegionInterface, OneResultInterface},
            types::{IntegerType, Signedness},
        },
        llvm::{
            attributes::ICmpPredicateAttr,
            op_interfaces::IsDeclaration,
            ops::{
                AddOp, AddressOfOp, AllocaOp, AndOp, BitcastOp, CStrOp, ConstantOp,
                ExtractValueOp, FCmpOp, GetElementPtrOp, ICmpOp, InsertValueOp, IntToPtrOp,
                LShrOp, LoadOp, MulOp, OrOp, PhiOp, PtrToIntOp, SDivOp, SExtOp, SRemOp, ShlOp,
                StoreOp, SubOp, TruncOp, UDivOp, URemOp, UndefOp, XorOp, ZExtOp,
            },
            types::PointerType,
        },
    },
    ir::{
        basic_block::BasicBlock,
        op::{Op, OpId},
        operation::Operation,
        region::Region,
        r#type::{TypedHandle, Typed},
        value::Value,
    },
    linked_list::ContainsLinkedList,
    conversion::pass::{AnalysisManager, Pass, PassResult, changed},
    passes::dominance_frontier::{DominatorTree, compute_dominance_frontiers_for_op},
    result::STAIRResult,
    utils::apint::APInt,
};

use super::inline::collect_functions;
use crate::r#type::TypeHandle;

const MAX_ITERATIONS: usize = 32;

pub struct LLVMSimplifyPass;

impl Pass for LLVMSimplifyPass {
    fn name(&self) -> &str {
        "llvm-simplify"
    }

    fn run(&mut self, root: Ptr<Operation>, ctx: &mut Context, _analyses: &mut AnalysisManager) -> pliron::result::Result<PassResult> {
        let pure_ops = pure_op_ids();
        for func in collect_functions(ctx, root) {
            if func.is_declaration(ctx) {
                continue;
            }
            let region = func.get_region(ctx);
            // This pass never changes CFG edges, so the dominator tree
            // stays valid across every fixpoint iteration.
            let (dom_tree, _) = compute_dominance_frontiers_for_op(&func, ctx);
            for _ in 0..MAX_ITERATIONS {
                let mut changed = fold_ops(ctx, region, &dom_tree)?;
                changed |= forward_stores_to_loads(ctx, region);
                changed |= eliminate_dead_phi_cycles(ctx, region);
                changed |= eliminate_dead_code(ctx, region, &pure_ops);
                if !changed {
                    break;
                }
            }
        }
        Ok(changed())
    }
}

fn function_ops(ctx: &Context, region: Ptr<Region>) -> Vec<Ptr<Operation>> {
    let mut ops = Vec::new();
    for block in region.deref(ctx).iter(ctx) {
        ops.extend(block.deref(ctx).iter(ctx));
    }
    ops
}

// ============================================================================
// Constant folding and value forwarding
// ============================================================================

/// An integer constant operand: its bits, width and full type.
pub(crate) struct ConstOperand {
    pub(crate) bits: u128,
    pub(crate) width: u32,
    pub(crate) ty: TypedHandle<IntegerType>,
}

impl ConstOperand {
    pub(crate) fn masked(&self) -> u128 {
        mask_to_width(self.bits, self.width)
    }

    pub(crate) fn signed(&self) -> i128 {
        sign_extend(self.bits, self.width)
    }

    pub(crate) fn is_true(&self) -> bool {
        self.masked() != 0
    }
}

fn mask_to_width(bits: u128, width: u32) -> u128 {
    if width >= 128 {
        bits
    } else {
        bits & ((1u128 << width) - 1)
    }
}

fn sign_extend(bits: u128, width: u32) -> i128 {
    let masked = mask_to_width(bits, width);
    if width == 0 || width >= 128 {
        return masked as i128;
    }
    let sign_bit = 1u128 << (width - 1);
    if masked & sign_bit != 0 {
        (masked | !((1u128 << width) - 1)) as i128
    } else {
        masked as i128
    }
}

pub(crate) fn as_const_operand(ctx: &Context, value: Value) -> Option<ConstOperand> {
    let Some(op) = value.defining_op().filter(|_| value.find_index(ctx) == 0) else {
        return None;
    };
    if Operation::get_opid(op, ctx) != ConstantOp::get_opid_static() {
        return None;
    }
    let attr = (ConstantOp::from_operation(op)).get_value(ctx)?;
    let apint = attr.value();
    Some(ConstOperand {
        bits: apint.to_u128(),
        width: apint.bw() as u32,
        ty: attr.get_type(),
    })
}

fn is_undef(ctx: &Context, value: Value) -> bool {
    value
        .defining_op()
        .filter(|_| value.find_index(ctx) == 0)
        .is_some_and(|op| Operation::get_opid(op, ctx) == UndefOp::get_opid_static())
}

/// Materialize an integer constant right before `before`, replacing all
/// uses of `before`'s single result, and erase `before`.
fn replace_with_constant(
    ctx: &mut Context,
    before: Ptr<Operation>,
    ty: TypedHandle<IntegerType>,
    bits: u128,
) {
    let width = ty.deref(ctx).width();
    let apint = APInt::from_u128(
        mask_to_width(bits, width),
        NonZero::new(width as usize).unwrap(),
    );
    let constant = ConstantOp::new_integer(ctx, IntegerAttr::new(ty, apint));
    constant.get_operation().insert_before(ctx, before);
    let new_value = constant.get_result(ctx);
    replace_op_with_value(ctx, before, new_value);
}

/// Replace all uses of `op`'s single result with `value` and erase `op`.
fn replace_op_with_value(ctx: &mut Context, op: Ptr<Operation>, value: Value) {
    let result = op.deref(ctx).get_result(0);
    result.replace_some_uses_with(ctx, |_, _| true, &value);
    Operation::erase(op, ctx);
}

fn fold_ops(
    ctx: &mut Context,
    region: Ptr<Region>,
    dom_tree: &DominatorTree,
) -> STAIRResult<bool> {
    let mut changed = false;
    for op in function_ops(ctx, region) {
        changed |= fold_op(ctx, op, dom_tree)?;
    }
    Ok(changed)
}

fn fold_op(ctx: &mut Context, op: Ptr<Operation>, dom_tree: &DominatorTree) -> STAIRResult<bool> {
    let opid = Operation::get_opid(op, ctx);

    if opid == ICmpOp::get_opid_static() {
        return Ok(fold_icmp(ctx, op));
    }
    if is_int_binary_arith(&opid) {
        return Ok(fold_int_binary(ctx, op, &opid));
    }
    if opid == ZExtOp::get_opid_static()
        || opid == SExtOp::get_opid_static()
        || opid == TruncOp::get_opid_static()
    {
        return Ok(fold_int_cast(ctx, op, &opid));
    }
    if opid == ExtractValueOp::get_opid_static() {
        return Ok(fold_extractvalue(ctx, op));
    }
    if opid == GetElementPtrOp::get_opid_static() {
        return Ok(fold_gep_aggregate_base(ctx, op));
    }
    if opid == PhiOp::get_opid_static() {
        return Ok(fold_trivial_phi(ctx, op, dom_tree));
    }
    Ok(false)
}

/// A gep whose base is an aggregate value addresses through the
/// aggregate's first field (the backend materializes exactly that field
/// as the data pointer). When the base is an insertvalue chain that pins
/// field 0 to a pointer, use that pointer directly.
fn fold_gep_aggregate_base(ctx: &mut Context, op: Ptr<Operation>) -> bool {
    let gep = GetElementPtrOp::from_operation(op);
    let mut aggregate = gep.get_base(ctx);
    if aggregate
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<PointerType>()
        .is_some()
    {
        return false;
    }
    loop {
        let Some(agg_op) = aggregate.defining_op().filter(|_| aggregate.find_index(ctx) == 0) else {
            return false;
        };
        if Operation::get_opid(agg_op, ctx) != InsertValueOp::get_opid_static() {
            return false;
        }
        let insert = InsertValueOp::from_operation(agg_op);
        let indices = insert.get_indices(ctx);
        if indices == [0] {
            let pointer = insert.get_value(ctx);
            if pointer
                .get_type(ctx)
                .deref(ctx)
                .downcast_ref::<PointerType>()
                .is_none()
            {
                return false;
            }
            Operation::replace_operand(op, ctx, 0, pointer);
            return true;
        }
        if indices.first() == Some(&0) {
            // A partial write into field 0's substructure; give up.
            return false;
        }
        aggregate = insert.get_aggregate(ctx);
    }
}

fn is_int_binary_arith(opid: &OpId) -> bool {
    *opid == AddOp::get_opid_static()
        || *opid == SubOp::get_opid_static()
        || *opid == MulOp::get_opid_static()
        || *opid == AndOp::get_opid_static()
        || *opid == OrOp::get_opid_static()
        || *opid == XorOp::get_opid_static()
        || *opid == ShlOp::get_opid_static()
        || *opid == LShrOp::get_opid_static()
        || *opid == UDivOp::get_opid_static()
        || *opid == SDivOp::get_opid_static()
        || *opid == URemOp::get_opid_static()
        || *opid == SRemOp::get_opid_static()
}

fn fold_icmp(ctx: &mut Context, op: Ptr<Operation>) -> bool {
    let icmp = ICmpOp::from_operation(op);
    let (Some(lhs), Some(rhs)) = (
        as_const_operand(ctx, icmp.get_lhs(ctx)),
        as_const_operand(ctx, icmp.get_rhs(ctx)),
    ) else {
        return false;
    };
    let result = match icmp.get_predicate(ctx) {
        ICmpPredicateAttr::EQ => lhs.masked() == rhs.masked(),
        ICmpPredicateAttr::NE => lhs.masked() != rhs.masked(),
        ICmpPredicateAttr::ULT => lhs.masked() < rhs.masked(),
        ICmpPredicateAttr::ULE => lhs.masked() <= rhs.masked(),
        ICmpPredicateAttr::UGT => lhs.masked() > rhs.masked(),
        ICmpPredicateAttr::UGE => lhs.masked() >= rhs.masked(),
        ICmpPredicateAttr::SLT => lhs.signed() < rhs.signed(),
        ICmpPredicateAttr::SLE => lhs.signed() <= rhs.signed(),
        ICmpPredicateAttr::SGT => lhs.signed() > rhs.signed(),
        ICmpPredicateAttr::SGE => lhs.signed() >= rhs.signed(),
    };
    let i1 = IntegerType::get(ctx, 1, Signedness::Signless);
    replace_with_constant(ctx, op, i1, result as u128);
    true
}

fn fold_int_binary(ctx: &mut Context, op: Ptr<Operation>, opid: &OpId) -> bool {
    let (lhs_v, rhs_v) = {
        let op_ref = op.deref(ctx);
        if op_ref.get_num_operands() != 2 {
            return false;
        }
        (op_ref.get_operand(0), op_ref.get_operand(1))
    };
    let (Some(lhs), Some(rhs)) = (
        as_const_operand(ctx, lhs_v),
        as_const_operand(ctx, rhs_v),
    ) else {
        return false;
    };
    let width = lhs.width;
    let l = lhs.masked();
    let r = rhs.masked();
    let bits = if *opid == AddOp::get_opid_static() {
        l.wrapping_add(r)
    } else if *opid == SubOp::get_opid_static() {
        l.wrapping_sub(r)
    } else if *opid == MulOp::get_opid_static() {
        l.wrapping_mul(r)
    } else if *opid == AndOp::get_opid_static() {
        l & r
    } else if *opid == OrOp::get_opid_static() {
        l | r
    } else if *opid == XorOp::get_opid_static() {
        l ^ r
    } else if *opid == ShlOp::get_opid_static() {
        if r >= width as u128 {
            return false;
        }
        l << r
    } else if *opid == LShrOp::get_opid_static() {
        if r >= width as u128 {
            return false;
        }
        l >> r
    } else if *opid == UDivOp::get_opid_static() {
        if r == 0 {
            return false;
        }
        l / r
    } else if *opid == URemOp::get_opid_static() {
        if r == 0 {
            return false;
        }
        l % r
    } else if *opid == SDivOp::get_opid_static() {
        if r == 0 {
            return false;
        }
        lhs.signed().wrapping_div(rhs.signed()) as u128
    } else if *opid == SRemOp::get_opid_static() {
        if r == 0 {
            return false;
        }
        lhs.signed().wrapping_rem(rhs.signed()) as u128
    } else {
        return false;
    };
    // The result type of these ops equals the operand type; reuse the lhs
    // constant's integer type so signedness is preserved.
    replace_with_constant(ctx, op, lhs.ty, bits);
    true
}

fn fold_int_cast(ctx: &mut Context, op: Ptr<Operation>, opid: &OpId) -> bool {
    let (input, result_ty) = {
        let op_ref = op.deref(ctx);
        (op_ref.get_operand(0), op_ref.get_type(0))
    };

    // trunc(zext(x)) -> x when the types round-trip exactly.
    if *opid == TruncOp::get_opid_static() {
        if let Some(input_op) = input.defining_op().filter(|_| input.find_index(ctx) == 0) {
            if Operation::get_opid(input_op, ctx) == ZExtOp::get_opid_static() {
                let source = input_op.deref(ctx).get_operand(0);
                if source.get_type(ctx) == result_ty {
                    replace_op_with_value(ctx, op, source);
                    return true;
                }
            }
        }
    }

    let Some(input) = as_const_operand(ctx, input) else {
        return false;
    };
    let Ok(result_ty) = TypedHandle::<IntegerType>::from_handle(result_ty, ctx) else {
        return false;
    };
    let bits = if *opid == SExtOp::get_opid_static() {
        input.signed() as u128
    } else {
        // zext and trunc both take the (zero-extended) masked input bits;
        // replace_with_constant masks to the result width.
        input.masked()
    };
    replace_with_constant(ctx, op, result_ty, bits);
    true
}

/// Forward `extractvalue` through a chain of `insertvalue` ops.
fn fold_extractvalue(ctx: &mut Context, op: Ptr<Operation>) -> bool {
    let extract = ExtractValueOp::from_operation(op);
    let indices = extract.get_indices(ctx);
    let mut aggregate = extract.get_aggregate(ctx);

    loop {
        let Some(agg_op) = aggregate.defining_op().filter(|_| aggregate.find_index(ctx) == 0) else {
            return false;
        };
        if Operation::get_opid(agg_op, ctx) == UndefOp::get_opid_static() {
            // Extracting from undef is undef of the field type.
            let result_ty = op.deref(ctx).get_type(0);
            let undef = UndefOp::new(ctx, result_ty);
            undef.get_operation().insert_before(ctx, op);
            let undef_value = undef.get_result(ctx);
            replace_op_with_value(ctx, op, undef_value);
            return true;
        }
        if Operation::get_opid(agg_op, ctx) != InsertValueOp::get_opid_static() {
            return false;
        }
        let insert = InsertValueOp::from_operation(agg_op);
        let insert_indices = insert.get_indices(ctx);
        if insert_indices == indices {
            let value = insert.get_value(ctx);
            replace_op_with_value(ctx, op, value);
            return true;
        }
        let min_len = indices.len().min(insert_indices.len());
        if indices[..min_len] == insert_indices[..min_len] {
            // One access path prefixes the other: the insert may partially
            // overlap the extracted value; give up.
            return false;
        }
        aggregate = insert.get_aggregate(ctx);
    }
}

/// Replace a phi whose incoming values (ignoring self-references and
/// undefs) are all the same single value.
///
/// Skipping an incoming is semantically sound (undef may be chosen to
/// equal the surviving value), but structurally the survivor then no
/// longer flows in through every predecessor, so it must dominate the
/// phi's block for the replacement to be valid SSA — the backend lowers
/// blocks in RPO and requires defs before uses.
fn fold_trivial_phi(ctx: &mut Context, op: Ptr<Operation>, dom_tree: &DominatorTree) -> bool {
    let phi = PhiOp::from_operation(op);
    let result = phi.get_result(ctx);
    let mut unique: Option<Value> = None;
    let mut skipped_any = false;
    for value in phi.get_incoming_values(ctx) {
        if value == result || is_undef(ctx, value) {
            skipped_any = true;
            continue;
        }
        match unique {
            None => unique = Some(value),
            Some(existing) if existing == value => {}
            Some(_) => return false,
        }
    }
    let Some(replacement) = unique else {
        return false;
    };
    if skipped_any {
        let Some(phi_block) = op.deref(ctx).get_parent_block() else {
            return false;
        };
        let Some(def_block) = defining_block(ctx, replacement) else {
            return false;
        };
        if !dominates(dom_tree, def_block, phi_block) {
            return false;
        }
    }
    replace_op_with_value(ctx, op, replacement);
    true
}

fn defining_block(ctx: &Context, value: Value) -> Option<Ptr<BasicBlock>> {
    match value.defining_entity() {
        crate::value::DefiningEntity::Op(op) => op.deref(ctx).get_parent_block(),
        crate::value::DefiningEntity::Block(block) => Some(block),
    }
}

/// Walk `block`'s immediate-dominator chain looking for `dominator`.
fn dominates(dom_tree: &DominatorTree, dominator: Ptr<BasicBlock>, block: Ptr<BasicBlock>) -> bool {
    let mut current = Some(block);
    while let Some(candidate) = current {
        if candidate == dominator {
            return true;
        }
        current = dom_tree.immediate_dominator(candidate);
    }
    false
}

// ============================================================================
// Store-to-load forwarding and dead store elimination
// ============================================================================

/// An alloca is "simple" when its address is only ever used directly as
/// the address of loads and stores: the address does not escape, so calls
/// and stores through other pointers cannot touch its contents.
fn simple_allocas(ctx: &Context, region: Ptr<Region>) -> Vec<AllocaOp> {
    let mut out = Vec::new();
    for op in function_ops(ctx, region) {
        if Operation::get_opid(op, ctx) != AllocaOp::get_opid_static() {
            continue;
        }
        let alloca = AllocaOp::from_operation(op);
        let slot = alloca.get_result(ctx);
        let simple = slot.uses(ctx).iter().all(|slot_use| {
            let user = slot_use.user_op().deref(ctx);
            let user_opid = Operation::get_opid(slot_use.user_op(), ctx);
            (user_opid == LoadOp::get_opid_static() && slot_use.find_index(ctx) == 0)
                || (user_opid == StoreOp::get_opid_static()
                    && slot_use.find_index(ctx) == 1
                    && user.get_operand(0) != slot)
        });
        if simple {
            out.push(alloca);
        }
    }
    out
}

/// `value` if it already has `want_ty`, or a fresh `llvm.bitcast` inserted
/// before `before` when both types are integers of the same width.
/// `None` when the types cannot be adapted.
fn adapt_to_type(
    ctx: &mut Context,
    value: Value,
    want_ty: TypeHandle,
    before: Ptr<Operation>,
) -> Option<Value> {
    let have_ty = value.get_type(ctx);
    if have_ty == want_ty {
        return Some(value);
    }
    let same_width_ints = match (
        have_ty.deref(ctx).downcast_ref::<IntegerType>(),
        want_ty.deref(ctx).downcast_ref::<IntegerType>(),
    ) {
        (Some(have), Some(want)) => have.width() == want.width(),
        _ => false,
    };
    if !same_width_ints {
        return None;
    }
    let cast = BitcastOp::new(ctx, value, want_ty);
    cast.get_operation().insert_before(ctx, before);
    Some(cast.get_result(ctx))
}

fn forward_stores_to_loads(ctx: &mut Context, region: Ptr<Region>) -> bool {
    let allocas = simple_allocas(ctx, region);
    if allocas.is_empty() {
        return false;
    }
    let simple: FxHashSet<Value> = allocas
        .iter()
        .map(|alloca| alloca.get_result(ctx))
        .collect();
    let mut changed = false;

    // Cross-block forwarding for slots written exactly once, in the entry
    // block: the store dominates every load in the function.
    for alloca in &allocas {
        changed |= forward_single_entry_store(ctx, region, *alloca);
    }

    // Block-local forwarding: track the known contents of each simple slot.
    let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
    for block in blocks {
        let mut known: FxHashMap<Value, Value> = FxHashMap::default();
        // Last store per slot with no load observing it yet: a following
        // store to the same slot makes it dead.
        let mut unread_store: FxHashMap<Value, Ptr<Operation>> = FxHashMap::default();
        let ops: Vec<_> = block.deref(ctx).iter(ctx).collect();
        for op in ops {
            let opid = Operation::get_opid(op, ctx);
            if opid == StoreOp::get_opid_static() {
                let store = StoreOp::from_operation(op);
                let addr = store.get_addr(ctx);
                if !simple.contains(&addr) {
                    continue;
                }
                if let Some(dead) = unread_store.remove(&addr) {
                    Operation::erase(dead, ctx);
                    changed = true;
                }
                known.insert(addr, store.get_value(ctx));
                unread_store.insert(addr, op);
            } else if opid == LoadOp::get_opid_static() {
                let load = LoadOp::from_operation(op);
                let addr = load.get_addr(ctx);
                if !simple.contains(&addr) {
                    continue;
                }
                unread_store.remove(&addr);
                let result = load.get_result(ctx);
                match known.get(&addr).copied() {
                    Some(value) => {
                        let result_ty = result.get_type(ctx);
                        if let Some(adapted) = adapt_to_type(ctx, value, result_ty, op) {
                            replace_op_with_value(ctx, op, adapted);
                            changed = true;
                        }
                    }
                    None => {
                        // Remember the loaded value: a second load from the
                        // same slot in this block yields the same value.
                        known.insert(addr, result);
                    }
                }
            }
        }
    }

    // Whole-slot dead store elimination: a simple slot that is never
    // loaded is unobservable.
    for alloca in simple_allocas(ctx, region) {
        let slot = alloca.get_result(ctx);
        let uses = slot.uses(ctx);
        if uses.is_empty() {
            continue;
        }
        if uses
            .iter()
            .any(|slot_use| Operation::get_opid(slot_use.user_op(), ctx) == LoadOp::get_opid_static())
        {
            continue;
        }
        for slot_use in uses {
            Operation::erase(slot_use.user_op(), ctx);
        }
        Operation::erase(alloca.get_operation(), ctx);
        changed = true;
    }

    changed
}

/// Forward a simple slot's single store to every load when that store sits
/// in the entry block (which dominates all blocks) and no load precedes it
/// within the entry block itself.
fn forward_single_entry_store(ctx: &mut Context, region: Ptr<Region>, alloca: AllocaOp) -> bool {
    let Some(entry) = region.deref(ctx).get_head() else {
        return false;
    };
    let slot = alloca.get_result(ctx);

    let mut stores = Vec::new();
    let mut loads = Vec::new();
    for slot_use in slot.uses(ctx) {
        let user = slot_use.user_op();
        let opid = Operation::get_opid(user, ctx);
        if opid == StoreOp::get_opid_static() {
            stores.push(user);
        } else if opid == LoadOp::get_opid_static() {
            loads.push(user);
        }
    }
    let [store] = stores[..] else {
        return false;
    };
    if store.deref(ctx).get_parent_block() != Some(entry) || loads.is_empty() {
        return false;
    }
    // No load of this slot may precede the store in the entry block.
    for op in entry.deref(ctx).iter(ctx) {
        if op == store {
            break;
        }
        if loads.contains(&op) {
            return false;
        }
    }

    let value = (StoreOp::from_operation(store)).get_value(ctx);
    let mut changed = false;
    for load in loads {
        let result = load.deref(ctx).get_result(0);
        let result_ty = result.get_type(ctx);
        if let Some(adapted) = adapt_to_type(ctx, value, result_ty, load) {
            replace_op_with_value(ctx, load, adapted);
            changed = true;
        }
    }
    changed
}

/// Erase phis that are only used by other dead phis. Plain DCE misses
/// them: loop-carried phis feed each other in cycles, so every phi in the
/// cycle has uses even though no real computation consumes any of them.
fn eliminate_dead_phi_cycles(ctx: &mut Context, region: Ptr<Region>) -> bool {
    let mut phis = FxHashSet::default();
    for op in function_ops(ctx, region) {
        if Operation::get_opid(op, ctx) == PhiOp::get_opid_static() {
            phis.insert(op);
        }
    }
    if phis.is_empty() {
        return false;
    }

    // A phi is live when a non-phi consumes it; liveness propagates to
    // every phi feeding a live phi.
    let mut live: FxHashSet<Ptr<Operation>> = FxHashSet::default();
    let mut worklist: Vec<Ptr<Operation>> = Vec::new();
    for &phi in &phis {
        let result = phi.deref(ctx).get_result(0);
        if result
            .uses(ctx)
            .iter()
            .any(|result_use| !phis.contains(&result_use.user_op()))
        {
            live.insert(phi);
            worklist.push(phi);
        }
    }
    while let Some(phi) = worklist.pop() {
        for value in phi.deref(ctx).operands().collect::<Vec<_>>() {
            if let Some(op) = value.defining_op() {
                if phis.contains(&op) && live.insert(op) {
                    worklist.push(op);
                }
            }
        }
    }

    let dead: Vec<_> = phis
        .iter()
        .copied()
        .filter(|phi| !live.contains(phi))
        .collect();
    if dead.is_empty() {
        return false;
    }
    for &phi in &dead {
        Operation::drop_all_uses(phi, ctx);
    }
    for phi in dead {
        Operation::erase(phi, ctx);
    }
    true
}

// ============================================================================
// Dead code elimination
// ============================================================================

fn pure_op_ids() -> FxHashSet<OpId> {
    let mut ids = FxHashSet::default();
    ids.insert(ConstantOp::get_opid_static());
    ids.insert(UndefOp::get_opid_static());
    ids.insert(ICmpOp::get_opid_static());
    ids.insert(FCmpOp::get_opid_static());
    ids.insert(AddOp::get_opid_static());
    ids.insert(SubOp::get_opid_static());
    ids.insert(MulOp::get_opid_static());
    ids.insert(AndOp::get_opid_static());
    ids.insert(OrOp::get_opid_static());
    ids.insert(XorOp::get_opid_static());
    ids.insert(ShlOp::get_opid_static());
    ids.insert(LShrOp::get_opid_static());
    ids.insert(UDivOp::get_opid_static());
    ids.insert(SDivOp::get_opid_static());
    ids.insert(URemOp::get_opid_static());
    ids.insert(SRemOp::get_opid_static());
    ids.insert(ZExtOp::get_opid_static());
    ids.insert(SExtOp::get_opid_static());
    ids.insert(TruncOp::get_opid_static());
    ids.insert(BitcastOp::get_opid_static());
    ids.insert(IntToPtrOp::get_opid_static());
    ids.insert(PtrToIntOp::get_opid_static());
    ids.insert(GetElementPtrOp::get_opid_static());
    ids.insert(InsertValueOp::get_opid_static());
    ids.insert(ExtractValueOp::get_opid_static());
    ids.insert(PhiOp::get_opid_static());
    ids.insert(AllocaOp::get_opid_static());
    ids.insert(LoadOp::get_opid_static());
    ids.insert(AddressOfOp::get_opid_static());
    ids.insert(CStrOp::get_opid_static());
    ids
}

fn eliminate_dead_code(
    ctx: &mut Context,
    region: Ptr<Region>,
    pure_ops: &FxHashSet<OpId>,
) -> bool {
    let mut changed = false;
    loop {
        let mut erased_any = false;
        let mut ops = function_ops(ctx, region);
        ops.reverse();
        for op in ops {
            if !pure_ops.contains(&Operation::get_opid(op, ctx)) {
                continue;
            }
            if !is_trivially_dead(ctx, op) {
                continue;
            }
            // A self-referential phi holds a use of its own result; drop
            // operand uses first so the erase assertion holds.
            Operation::drop_all_uses(op, ctx);
            Operation::erase(op, ctx);
            erased_any = true;
        }
        changed |= erased_any;
        if !erased_any {
            break;
        }
    }
    changed
}

/// All uses of all results (if any) come from the op itself.
fn is_trivially_dead(ctx: &Context, op: Ptr<Operation>) -> bool {
    let num_results = op.deref(ctx).get_num_results();
    for res_idx in 0..num_results {
        let result = op.deref(ctx).get_result(res_idx);
        if result.uses(ctx).iter().any(|result_use| result_use.user_op() != op) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dialects::{
            builtin::types::Signedness,
            llvm::{
                self,
                attributes::LinkageAttr,
                ops::{FuncOp, ReturnOp},
                types::FuncType,
            },
        },
        ir::r#type::{TypeHandle, TypedHandle},
        printable::Printable,
    };

    fn test_context() -> Context {
        let mut ctx = Context::new();
        llvm::register(&mut ctx);
        ctx
    }

    fn int_ty(ctx: &mut Context, width: u32) -> TypedHandle<IntegerType> {
        IntegerType::get(ctx, width, Signedness::Signless)
    }

    fn int_const(ctx: &mut Context, width: u32, value: u64) -> ConstantOp {
        let ty = int_ty(ctx, width);
        ConstantOp::new_integer(
            ctx,
            IntegerAttr::new(ty, APInt::from_u64(value, NonZero::new(width as usize).unwrap())),
        )
    }

    fn run_simplify(ctx: &mut Context, func: FuncOp) -> String {
        LLVMSimplifyPass
            .run(func.get_operation(), ctx, &mut AnalysisManager::default())
            .unwrap();
        format!("{}", func.get_operation().disp(ctx))
    }

    #[test]
    fn folds_constants_and_removes_dead_code() {
        let mut ctx = test_context();
        let i64_ty: TypeHandle = int_ty(&mut ctx, 64).into();
        let fn_ty: TypedHandle<FuncType> = FuncType::get(&mut ctx, i64_ty, vec![], false);
        let func = FuncOp::new(
            &mut ctx,
            "fold".try_into().unwrap(),
            fn_ty,
            LinkageAttr::External,
        );
        let entry = func.get_entry_block(&ctx);

        let two = int_const(&mut ctx, 64, 2);
        two.get_operation().insert_at_back(entry, &ctx);
        let three = int_const(&mut ctx, 64, 3);
        three.get_operation().insert_at_back(entry, &ctx);
        let two_v = two.get_result(&ctx);
        let three_v = three.get_result(&ctx);
        let add = AddOp::new(&mut ctx, two_v, three_v);
        add.get_operation().insert_at_back(entry, &ctx);
        let sum = add.get_result(&ctx);
        // Dead icmp of the constants: must be folded away entirely.
        let cmp = ICmpOp::new(&mut ctx, ICmpPredicateAttr::ULT, two_v, three_v);
        cmp.get_operation().insert_at_back(entry, &ctx);
        ReturnOp::new(&mut ctx, Some(sum))
            .get_operation()
            .insert_at_back(entry, &ctx);

        let text = run_simplify(&mut ctx, func);
        assert!(!text.contains("llvm.add"), "{text}");
        assert!(!text.contains("llvm.icmp"), "{text}");
        assert!(text.contains("<5: i64>"), "{text}");
    }

    #[test]
    fn forwards_local_store_to_load() {
        let mut ctx = test_context();
        let i64_ty: TypeHandle = int_ty(&mut ctx, 64).into();
        let fn_ty: TypedHandle<FuncType> = FuncType::get(&mut ctx, i64_ty, vec![i64_ty], false);
        let func = FuncOp::new(
            &mut ctx,
            "fwd".try_into().unwrap(),
            fn_ty,
            LinkageAttr::External,
        );
        let entry = func.get_entry_block(&ctx);
        let arg = entry.deref(&ctx).get_argument(0);

        let one = int_const(&mut ctx, 64, 1);
        one.get_operation().insert_at_back(entry, &ctx);
        let one_v = one.get_result(&ctx);
        let alloca = llvm::ops::AllocaOp::new(&mut ctx, one_v, i64_ty);
        alloca.get_operation().insert_at_back(entry, &ctx);
        let slot = alloca.get_result(&ctx);
        StoreOp::new(&mut ctx, arg, slot)
            .get_operation()
            .insert_at_back(entry, &ctx);
        let load = LoadOp::new(&mut ctx, slot, i64_ty);
        load.get_operation().insert_at_back(entry, &ctx);
        let loaded = load.get_result(&ctx);
        ReturnOp::new(&mut ctx, Some(loaded))
            .get_operation()
            .insert_at_back(entry, &ctx);

        let text = run_simplify(&mut ctx, func);
        assert!(!text.contains("llvm.load"), "{text}");
        assert!(!text.contains("llvm.store"), "{text}");
        assert!(!text.contains("llvm.alloca"), "{text}");
    }
}
