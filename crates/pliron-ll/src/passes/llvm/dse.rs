//! Global dead-store elimination (docs/MIDEND-PLAN.md item 6): backward
//! memory liveness over the region CFG, with gvn's syntactic address model.
//!
//! A store is deleted when its address key is not live after it: every
//! path to the function exit overwrites the identical key before anything
//! could read it, or the key is rooted at a non-escaping alloca and never
//! read again. Conservatism, all deliberate:
//! - Reads gen the loaded key plus every key not *provably distinct* from
//!   it. Two keys are provably distinct only when they are rooted at two
//!   different allocas, or one is rooted at a non-escaping alloca and the
//!   other at anything else (a never-escaping address cannot be reached
//!   through an unrelated pointer).
//! - Calls, returns and every op outside gvn's benign set read all keys
//!   except those rooted at non-escaping allocas.
//! - Kills are exact-key only; the function-exit boundary keeps every key
//!   live except the non-escaping-alloca-rooted ones.
//! - An alloca escapes if any value derived from it through GEP/bitcast is
//!   used as anything other than a load address, a store *address*, or a
//!   further GEP/bitcast base.
//!
//! ADJOINT (backward attribution): deletion-only: erased stores need no adjoint (their cost ceases to exist).

use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    context::{Context, Ptr},
    dialects::llvm::{
        op_interfaces::IsDeclaration,
        ops::{AllocaOp, BitcastOp, GetElementPtrOp, LoadOp, StoreOp},
    },
    ir::{
        op::{Op, OpId},
        operation::Operation,
        region::Region,
        value::Value,
    },
    linked_list::ContainsLinkedList,
    conversion::pass::{AnalysisManager, Pass, PassResult, changed, unchanged},
};

use super::{
    analysis::{BitSet, RegionCfg, backward_bitset_fixpoint},
    gvn::{AddrKey, addr_key, benign_op_ids},
    inline::collect_functions,
    midend_gate::midend_disabled,
};
use pliron::builtin::op_interfaces::AtMostOneRegionInterface as _;

pub struct LLVMGlobalDsePass;

impl Pass for LLVMGlobalDsePass {
    fn name(&self) -> &str {
        "llvm-global-dse"
    }

    fn run(
        &mut self,
        root: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> pliron::result::Result<PassResult> {
        if midend_disabled("dse") {
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
            any |= dse_region(ctx, region, &benign);
        }
        Ok(if any { changed() } else { unchanged() })
    }
}

/// The base SSA value a key is rooted at.
fn key_root(key: &AddrKey) -> Value {
    match key {
        AddrKey::Root(value) => *value,
        AddrKey::Gep(base, _, _) => key_root(base),
    }
}

fn is_alloca(ctx: &Context, value: Value) -> bool {
    value
        .defining_op()
        .is_some_and(|op| Operation::get_opid(op, ctx) == AllocaOp::get_opid_static())
}

/// The alloca results in `region` whose address provably never escapes:
/// every value transitively derived through GEP (as base) / bitcast is
/// used only as a load address, a store address, or a further GEP/bitcast
/// base.
fn nonescaping_allocas(ctx: &Context, region: Ptr<Region>) -> FxHashSet<Value> {
    let mut result = FxHashSet::default();
    let allocas: Vec<Value> = region
        .deref(ctx)
        .iter(ctx)
        .flat_map(|block| block.deref(ctx).iter(ctx).collect::<Vec<_>>())
        .filter(|op| Operation::get_opid(*op, ctx) == AllocaOp::get_opid_static())
        .map(|op| op.deref(ctx).get_result(0))
        .collect();
    'next_alloca: for alloca in allocas {
        let mut derived: Vec<Value> = vec![alloca];
        let mut seen: FxHashSet<Value> = FxHashSet::default();
        while let Some(value) = derived.pop() {
            if !seen.insert(value) {
                continue;
            }
            for r#use in value.uses(ctx) {
                let user = r#use.user_op();
                let opid = Operation::get_opid(user, ctx);
                if opid == LoadOp::get_opid_static() {
                    continue; // load address: fine
                }
                if opid == StoreOp::get_opid_static() {
                    let store = StoreOp::from_operation(user);
                    // Escapes when the pointer is the *stored value*.
                    if store.get_operand_value(ctx) == value {
                        continue 'next_alloca;
                    }
                    continue; // store address: fine
                }
                if opid == GetElementPtrOp::get_opid_static() {
                    // Fine only as the base; an address used as an index
                    // is out of model.
                    if user.deref(ctx).get_operand(0) != value {
                        continue 'next_alloca;
                    }
                    derived.push(user.deref(ctx).get_result(0));
                    continue;
                }
                if opid == BitcastOp::get_opid_static() {
                    derived.push(user.deref(ctx).get_result(0));
                    continue;
                }
                continue 'next_alloca; // call, return, ptrtoint, branch, …
            }
        }
        result.insert(alloca);
    }
    result
}

struct KeyTable {
    keys: Vec<AddrKey>,
    index: FxHashMap<AddrKey, usize>,
    /// For each key, the set a read of it makes live (itself plus every
    /// not-provably-distinct key). Keys rooted at a non-escaping alloca
    /// (protected) are invisible to calls/unknown ops and dead at function
    /// exit; `unknown_read` is exactly the unprotected set.
    read_sets: Vec<BitSet>,
    /// What a call/return/unknown op makes live: every unprotected key.
    unknown_read: BitSet,
}

impl KeyTable {
    fn build(ctx: &Context, region: Ptr<Region>, nonescaping: &FxHashSet<Value>) -> KeyTable {
        let mut keys: Vec<AddrKey> = Vec::new();
        let mut index: FxHashMap<AddrKey, usize> = FxHashMap::default();
        for block in region.deref(ctx).iter(ctx) {
            for op in block.deref(ctx).iter(ctx) {
                let opid = Operation::get_opid(op, ctx);
                let address = if opid == StoreOp::get_opid_static() {
                    StoreOp::from_operation(op).get_operand_address(ctx)
                } else if opid == LoadOp::get_opid_static() {
                    LoadOp::from_operation(op).get_operand_address(ctx)
                } else {
                    continue;
                };
                let key = addr_key(ctx, address);
                index.entry(key.clone()).or_insert_with(|| {
                    keys.push(key);
                    keys.len() - 1
                });
            }
        }
        let n = keys.len();
        let roots: Vec<Value> = keys.iter().map(key_root).collect();
        let alloca_root: Vec<bool> = roots.iter().map(|r| is_alloca(ctx, *r)).collect();
        let protected: Vec<bool> = roots.iter().map(|r| nonescaping.contains(r)).collect();
        let mut unknown_read = BitSet::new(n);
        for k in 0..n {
            if !protected[k] {
                unknown_read.insert(k);
            }
        }
        let mut read_sets = Vec::with_capacity(n);
        for a in 0..n {
            let mut set = BitSet::new(n);
            for b in 0..n {
                let distinct = a != b
                    && ((alloca_root[a] && alloca_root[b] && roots[a] != roots[b])
                        || (protected[a] && roots[a] != roots[b])
                        || (protected[b] && roots[a] != roots[b]));
                if !distinct {
                    set.insert(b);
                }
            }
            read_sets.push(set);
        }
        KeyTable {
            keys,
            index,
            read_sets,
            unknown_read,
        }
    }

    fn idx(&self, key: &AddrKey) -> usize {
        self.index[key]
    }

    fn len(&self) -> usize {
        self.keys.len()
    }
}

fn dse_region(ctx: &mut Context, region: Ptr<Region>, benign: &FxHashSet<OpId>) -> bool {
    let nonescaping = nonescaping_allocas(ctx, region);
    let table = KeyTable::build(ctx, region, &nonescaping);
    if table.len() == 0 {
        return false;
    }
    let cfg = RegionCfg::new(ctx, region);
    let n = table.len();

    // Per-block GEN (read before overwritten) / KILL (overwritten before
    // read), by forward scan.
    let store_id = StoreOp::get_opid_static();
    let load_id = LoadOp::get_opid_static();
    let mut gens: Vec<BitSet> = Vec::with_capacity(cfg.blocks.len());
    let mut kills: Vec<BitSet> = Vec::with_capacity(cfg.blocks.len());
    for block in &cfg.blocks {
        let mut upward = BitSet::new(n);
        let mut kill = BitSet::new(n);
        for op in block.deref(ctx).iter(ctx) {
            let opid = Operation::get_opid(op, ctx);
            if opid == store_id {
                let store = StoreOp::from_operation(op);
                let k = table.idx(&addr_key(ctx, store.get_operand_address(ctx)));
                if !upward.contains(k) {
                    kill.insert(k);
                }
            } else if opid == load_id {
                let load = LoadOp::from_operation(op);
                let k = table.idx(&addr_key(ctx, load.get_operand_address(ctx)));
                gen_reads(&mut upward, &kill, &table.read_sets[k], n);
            } else if !benign.contains(&opid) {
                gen_reads(&mut upward, &kill, &table.unknown_read, n);
            }
        }
        gens.push(upward);
        kills.push(kill);
    }

    // Function-exit boundary: everything except protected keys is
    // observable after return.
    let boundary_set = table.unknown_read.clone();
    let (_, live_out) = backward_bitset_fixpoint(
        &cfg,
        n,
        |_| boundary_set.clone(),
        |b, out| {
            let mut live = gens[b].clone();
            let mut escaped = out.clone();
            for k in 0..n {
                if kills[b].contains(k) {
                    escaped.remove(k);
                }
            }
            live.union_with(&escaped);
            live
        },
    );

    // Deletion: walk each block backwards with the live-out state.
    let mut changed = false;
    for (b, block) in cfg.blocks.iter().enumerate() {
        let mut live = live_out[b].clone();
        let ops: Vec<_> = block.deref(ctx).iter(ctx).collect();
        for op in ops.into_iter().rev() {
            let opid = Operation::get_opid(op, ctx);
            if opid == store_id {
                let store = StoreOp::from_operation(op);
                let k = table.idx(&addr_key(ctx, store.get_operand_address(ctx)));
                if !live.contains(k) {
                    Operation::erase(op, ctx);
                    changed = true;
                } else {
                    live.remove(k);
                }
            } else if opid == load_id {
                let load = LoadOp::from_operation(op);
                let k = table.idx(&addr_key(ctx, load.get_operand_address(ctx)));
                live.union_with(&table.read_sets[k].clone());
            } else if !benign.contains(&opid) {
                live.union_with(&table.unknown_read.clone());
            }
        }
    }
    changed
}

/// `gen |= reads − kill` (forward-scan gen update: only keys not already
/// overwritten in this block become upward-exposed reads).
fn gen_reads(upward: &mut BitSet, kill: &BitSet, reads: &BitSet, n: usize) {
    for k in 0..n {
        if reads.contains(k) && !kill.contains(k) {
            upward.insert(k);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pliron::builtin::op_interfaces::OneResultInterface as _;
    use crate::{
        dialects::{
            builtin::{
                attributes::IntegerAttr,
                ops::ConstantOp,
                types::{IntegerType, Signedness},
            },
            llvm::{
                attributes::LinkageAttr,
                ops::{CallOp, FuncOp, GepIndex, ReturnOp},
                types::{FuncType, PointerType},
            },
        },
        ir::{basic_block::BasicBlock, r#type::{TypeHandle, TypedHandle}},
        printable::Printable,
        utils::apint::APInt,
    };
    use std::num::NonZero;

    fn int_ty(ctx: &mut Context, width: u32) -> TypedHandle<IntegerType> {
        IntegerType::get(ctx, width, Signedness::Signless)
    }

    fn run_dse(ctx: &mut Context, func: FuncOp) -> String {
        LLVMGlobalDsePass
            .run(func.get_operation(), ctx, &mut AnalysisManager::default())
            .unwrap();
        format!("{}", func.get_operation().disp(ctx))
    }

    /// i64 f(i64 a, ptr p) with an entry block.
    fn scaffold(ctx: &mut Context) -> (FuncOp, Ptr<BasicBlock>, Value, Value) {
        let i64_ty: TypeHandle = int_ty(ctx, 64).into();
        let ptr_ty: TypeHandle = PointerType::get(ctx, 0).into();
        let fn_ty = FuncType::get(ctx, i64_ty, vec![i64_ty, ptr_ty], false);
        let func = FuncOp::new(ctx, "f".try_into().unwrap(), fn_ty);
        func.set_attr_llvm_function_linkage(ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(ctx);
        let entry = func.get_entry_block(ctx).unwrap();
        let a = entry.deref(ctx).get_argument(0);
        let p = entry.deref(ctx).get_argument(1);
        (func, entry, a, p)
    }

    fn const_i64(ctx: &mut Context, entry: Ptr<BasicBlock>, v: u64) -> Value {
        let i64_typed = int_ty(ctx, 64);
        let c = ConstantOp::new(
            ctx,
            Box::new(IntegerAttr::new(
                i64_typed,
                APInt::from_u64(v, NonZero::new(64).unwrap()),
            )),
        );
        c.get_operation().insert_at_back(entry, ctx);
        c.get_result(ctx)
    }

    fn alloca_i64(ctx: &mut Context, entry: Ptr<BasicBlock>) -> Value {
        let one = const_i64(ctx, entry, 1);
        let i64_ty: TypeHandle = int_ty(ctx, 64).into();
        let slot = crate::dialects::llvm::ops::AllocaOp::new(ctx, i64_ty, one);
        slot.get_operation().insert_at_back(entry, ctx);
        slot.get_result(ctx)
    }

    #[test]
    fn deletes_store_overwritten_before_any_read() {
        let mut ctx = Context::new();
        let (func, entry, a, p) = scaffold(&mut ctx);
        let i64_ty: TypeHandle = int_ty(&mut ctx, 64).into();
        // store a -> p; store 7 -> p; load p; ret — the first store is dead
        // even though p escapes (exact overwrite before any read).
        StoreOp::new(&mut ctx, a, p)
            .get_operation()
            .insert_at_back(entry, &ctx);
        let seven = const_i64(&mut ctx, entry, 7);
        StoreOp::new(&mut ctx, seven, p)
            .get_operation()
            .insert_at_back(entry, &ctx);
        let load = LoadOp::new(&mut ctx, p, i64_ty);
        load.get_operation().insert_at_back(entry, &ctx);
        let load_v = load.get_result(&ctx);
        ReturnOp::new(&mut ctx, Some(load_v))
            .get_operation()
            .insert_at_back(entry, &ctx);
        let text = run_dse(&mut ctx, func);
        assert_eq!(text.matches("llvm.store").count(), 1, "{text}");
    }

    #[test]
    fn keeps_store_when_may_alias_load_follows() {
        let mut ctx = Context::new();
        let (func, entry, a, p) = scaffold(&mut ctx);
        let i64_ty: TypeHandle = int_ty(&mut ctx, 64).into();
        // store a -> p; store 7 -> gep(p, a); load gep(p, a); both stores
        // must survive: the load's key may alias p (same root).
        StoreOp::new(&mut ctx, a, p)
            .get_operation()
            .insert_at_back(entry, &ctx);
        let seven = const_i64(&mut ctx, entry, 7);
        let gep = GetElementPtrOp::new(&mut ctx, p, vec![GepIndex::Value(a)], i64_ty);
        gep.get_operation().insert_at_back(entry, &ctx);
        let gep_v = gep.get_result(&ctx);
        StoreOp::new(&mut ctx, seven, gep_v)
            .get_operation()
            .insert_at_back(entry, &ctx);
        let load = LoadOp::new(&mut ctx, gep_v, i64_ty);
        load.get_operation().insert_at_back(entry, &ctx);
        let load_v = load.get_result(&ctx);
        ReturnOp::new(&mut ctx, Some(load_v))
            .get_operation()
            .insert_at_back(entry, &ctx);
        let text = run_dse(&mut ctx, func);
        assert_eq!(text.matches("llvm.store").count(), 2, "{text}");
    }

    #[test]
    fn deletes_unread_store_to_nonescaping_alloca() {
        let mut ctx = Context::new();
        let (func, entry, a, _p) = scaffold(&mut ctx);
        let slot = alloca_i64(&mut ctx, entry);
        StoreOp::new(&mut ctx, a, slot)
            .get_operation()
            .insert_at_back(entry, &ctx);
        ReturnOp::new(&mut ctx, Some(a))
            .get_operation()
            .insert_at_back(entry, &ctx);
        let text = run_dse(&mut ctx, func);
        assert_eq!(text.matches("llvm.store").count(), 0, "{text}");
    }

    #[test]
    fn escaped_alloca_store_survives_exit_and_calls() {
        let mut ctx = Context::new();
        let (func, entry, a, _p) = scaffold(&mut ctx);
        let i64_ty: TypeHandle = int_ty(&mut ctx, 64).into();
        let slot = alloca_i64(&mut ctx, entry);
        // The address escapes through a call operand, so the store after
        // the call must survive to function exit.
        let sink_fn = FuncType::get(&mut ctx, i64_ty, vec![], false);
        let call = CallOp::new(
            &mut ctx,
            pliron::builtin::op_interfaces::CallOpCallable::Direct(
                "opaque".try_into().unwrap(),
            ),
            sink_fn,
            vec![slot],
        );
        call.get_operation().insert_at_back(entry, &ctx);
        StoreOp::new(&mut ctx, a, slot)
            .get_operation()
            .insert_at_back(entry, &ctx);
        ReturnOp::new(&mut ctx, Some(a))
            .get_operation()
            .insert_at_back(entry, &ctx);
        let text = run_dse(&mut ctx, func);
        assert_eq!(text.matches("llvm.store").count(), 1, "{text}");
    }

    #[test]
    fn store_before_call_survives_even_unloaded() {
        let mut ctx = Context::new();
        let (func, entry, a, p) = scaffold(&mut ctx);
        let i64_ty: TypeHandle = int_ty(&mut ctx, 64).into();
        // store a -> p; call; ret — the callee may read p.
        StoreOp::new(&mut ctx, a, p)
            .get_operation()
            .insert_at_back(entry, &ctx);
        let sink_fn = FuncType::get(&mut ctx, i64_ty, vec![], false);
        let call = CallOp::new(
            &mut ctx,
            pliron::builtin::op_interfaces::CallOpCallable::Direct(
                "opaque".try_into().unwrap(),
            ),
            sink_fn,
            vec![],
        );
        call.get_operation().insert_at_back(entry, &ctx);
        ReturnOp::new(&mut ctx, Some(a))
            .get_operation()
            .insert_at_back(entry, &ctx);
        let text = run_dse(&mut ctx, func);
        assert_eq!(text.matches("llvm.store").count(), 1, "{text}");
    }
}
