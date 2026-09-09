//! Global value numbering (docs/MIDEND-PLAN.md item 1): dominator-scoped
//! CSE of pure operations, plus redundant-load elimination and
//! store-to-load forwarding across blocks.
//!
//! Memory model (deliberately conservative, syntactic): two accesses are
//! to the *same* address only when their addresses are the same SSA value
//! or structurally identical GEP chains off the same base with identical
//! indices ([AddrKey]). Availability of a memory value is killed by any
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
            attributes::ICmpPredicateAttr,
            op_interfaces::IsDeclaration,
            ops::{
                AddOp, AndOp, BitcastOp, BrOp, CondBrOp, ExtractValueOp, GetElementPtrOp,
                GepIndex, ICmpOp, IntToPtrOp, LShrOp, LoadOp, MulOp, OrOp, PtrToIntOp, SDivOp,
                SExtOp, SRemOp, ShlOp, StoreOp, SubOp, TruncOp, UDivOp, URemOp, XorOp, ZExtOp,
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

use super::{
    analysis::dominator_tree,
    inline::collect_functions,
    midend_gate::midend_disabled,
    simplify::pure_op_ids,
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
        let benign = benign_op_ids();
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
    Bin(OpId, Value, Value, TypeHandle),
    Cast(OpId, Value, TypeHandle),
    ICmp(ICmpPredicateAttr, Value, Value),
    Gep(TypeHandle, Value, Vec<IdxKey>),
    Extract(Value, Vec<u32>),
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub(crate) enum IdxKey {
    Const(u32),
    Val(Value),
}

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
    ids.insert(UDivOp::get_opid_static());
    ids.insert(SDivOp::get_opid_static());
    ids.insert(URemOp::get_opid_static());
    ids.insert(SRemOp::get_opid_static());
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
    ids
}

fn expr_key(
    ctx: &Context,
    op: Ptr<Operation>,
    bins: &FxHashSet<OpId>,
    casts: &FxHashSet<OpId>,
) -> Option<ExprKey> {
    let opid = Operation::get_opid(op, ctx);
    let operation = op.deref(ctx);
    if bins.contains(&opid) {
        return Some(ExprKey::Bin(
            opid,
            operation.get_operand(0),
            operation.get_operand(1),
            operation.get_result(0).get_type(ctx),
        ));
    }
    if casts.contains(&opid) {
        return Some(ExprKey::Cast(
            opid,
            operation.get_operand(0),
            operation.get_result(0).get_type(ctx),
        ));
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

fn replace_op_with_value(ctx: &mut Context, op: Ptr<Operation>, value: Value) {
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
            let Some(key) = expr_key(ctx, op, &bins, &casts) else {
                continue;
            };
            match scopes.iter().rev().find_map(|scope| scope.get(&key)) {
                Some(&existing) => {
                    replace_op_with_value(ctx, op, existing);
                    changed = true;
                }
                None => {
                    let result = op.deref(ctx).get_result(0);
                    scopes.last_mut().unwrap().insert(key, result);
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

/// Ops that neither read nor write memory in a way that invalidates a
/// tracked load: the pure set plus branches (crossed when scanning a
/// predecessor) — loads are re-examined explicitly, stores handled by
/// the caller.
pub(crate) fn benign_op_ids() -> FxHashSet<OpId> {
    let mut ids = pure_op_ids();
    ids.insert(BrOp::get_opid_static());
    ids.insert(CondBrOp::get_opid_static());
    ids
}

/// What the backward scan found for one load.
enum Resolution {
    Value(Value),
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

#[cfg(test)]
mod tests {
    use super::*;
    use pliron::builtin::op_interfaces::{OneResultInterface as _};
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
                ops::{BrOp, CallOp, FuncOp, ReturnOp},
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
