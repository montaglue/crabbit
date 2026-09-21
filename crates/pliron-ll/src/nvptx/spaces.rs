//! Pointer state-space provenance for kernel bodies.
//!
//! PTX qualifies memory instructions by state space (`ld.global`,
//! `st.shared`, `atom.global.add…`), and `ptxas` schedules and allocates
//! qualified accesses far better than generic ones (the generic forms are
//! what cost gemm_tiled its occupancy). This module proves, per SSA value,
//! which space an address belongs to; emission then qualifies the accesses
//! it can and keeps the generic forms for everything else.
//!
//! The lattice is `Global < Generic`, `Shared < Generic`, with map absence
//! as bottom (no provenance seen yet). Sources: kernel pointer parameters
//! are GLOBAL-space addresses (grid-visible allocations handed to the
//! launch), and `llvm.addressof` of a module global is that global's space.
//! Provenance propagates through GEPs, the register-aliasing casts
//! (bitcast / inttoptr / ptrtoint), selects, and block arguments (joining
//! over incoming edges); any other producer is generic.
//!
//! Representation invariant (see the module doc of [super]): a value proven
//! `Shared` holds the RAW shared-window address, so a `Shared` value may
//! only reach consumers that understand that representation — qualified
//! loads/stores/atomics, and the propagation ops above (whose emission
//! inserts `cvta.shared.u64` wherever the value flows into a non-shared
//! slot). A `Shared` value used by anything else escapes: it is demoted to
//! `Generic` here, which makes its definition materialize the generic
//! (post-`cvta`) address exactly as before this analysis existed. Global
//! addresses never need demotion: the global window is identity-mapped in
//! generic space, so one register serves both the qualified and the generic
//! forms.

use std::collections::{HashMap, HashSet};

use pliron::builtin::op_interfaces::{
    BranchOpInterface as _, CallOpCallable, CallOpInterface as _, OneOpdInterface as _,
    OneResultInterface,
};

use crate::{
    context::{Context, Ptr},
    dialects::llvm::{
        ops::{
            AddressOfOp, AllocaOp, BitcastOp, BrOp, CallOp, CondBrOp, GepIndex,
            GetElementPtrOp, IntToPtrOp, LoadOp, PtrToIntOp, SelectOp, StoreOp,
        },
        types::PointerType,
    },
    ir::{basic_block::BasicBlock, op::Op, operation::Operation, r#type::Typed, value::Value},
    linked_list::ContainsLinkedList,
};

use super::{CallMemEffect, GlobalSpace, atomic_intrinsic, call_mem_effect, nvvm_intrinsic_name};

/// The state space a pointer value is proven to address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PtrSpace {
    /// A `.global` address. Numerically identical to its generic form, so
    /// the same register works with both `ld.global` and generic `ld`.
    Global,
    /// A RAW `.shared` window address (not run through `cvta.shared`).
    /// Valid only with `.shared`-qualified accesses.
    Shared,
    /// Unknown or mixed provenance: a generic address, today's behavior.
    Generic,
}

/// Provenance for every value of the kernel body: the fixpoint of forward
/// propagation from the sources plus demotion of escaping `Shared` values.
/// Values absent from the map never carried provenance; callers treat them
/// as [PtrSpace::Generic].
///
/// `params_global` is true for `.entry` kernels, whose pointer parameters
/// are grid-visible global allocations. Device `.func`s receive arbitrary
/// (already generic) addresses, so their parameters carry no provenance.
pub(super) fn infer_spaces(
    ctx: &Context,
    entry: Ptr<BasicBlock>,
    blocks: &[Ptr<BasicBlock>],
    globals: &HashMap<String, GlobalSpace>,
    params_global: bool,
) -> HashMap<Value, PtrSpace> {
    let mut spaces: HashMap<Value, PtrSpace> = HashMap::new();
    // Kernel pointer parameters are global-space addresses; every other
    // parameter (and any int a cast may later reinterpret) is generic.
    for param in entry.deref(ctx).arguments() {
        let is_ptr = params_global
            && param
                .get_type(ctx)
                .deref(ctx)
                .downcast_ref::<PointerType>()
                .is_some();
        spaces.insert(
            param,
            if is_ptr { PtrSpace::Global } else { PtrSpace::Generic },
        );
    }
    // Escaped values stay demoted across iterations.
    let mut forced: HashSet<Value> = HashSet::new();
    loop {
        let mut changed = false;
        for block in blocks {
            for op_ptr in block.deref(ctx).iter(ctx) {
                changed |= transfer(ctx, op_ptr, globals, &mut spaces, &mut forced);
            }
        }
        if !changed {
            break;
        }
    }
    spaces
}

/// Join `space` into `value`'s entry: same space is absorbed, differing
/// spaces meet at `Generic`. Returns whether the entry changed. Monotone
/// (entries only rise), so the caller's fixpoint loop terminates.
fn raise(spaces: &mut HashMap<Value, PtrSpace>, value: Value, space: PtrSpace) -> bool {
    match spaces.get(&value).copied() {
        None => {
            spaces.insert(value, space);
            true
        }
        Some(old) if old == space || old == PtrSpace::Generic => false,
        Some(_) => {
            spaces.insert(value, PtrSpace::Generic);
            true
        }
    }
}

/// [raise], except escaped results are pinned at `Generic`.
fn set_result(
    spaces: &mut HashMap<Value, PtrSpace>,
    forced: &HashSet<Value>,
    result: Value,
    space: PtrSpace,
) -> bool {
    let space = if forced.contains(&result) {
        PtrSpace::Generic
    } else {
        space
    };
    raise(spaces, result, space)
}

/// A use that cannot handle the raw-shared representation: demote a
/// `Shared` value to `Generic` (its definition then materializes the
/// generic address). Non-shared values are unaffected — global addresses
/// are already generic-compatible.
fn escape(
    spaces: &mut HashMap<Value, PtrSpace>,
    forced: &mut HashSet<Value>,
    value: Value,
) -> bool {
    if spaces.get(&value).copied() == Some(PtrSpace::Shared) && forced.insert(value) {
        spaces.insert(value, PtrSpace::Generic);
        true
    } else {
        false
    }
}

/// One op's provenance transfer: result spaces, escapes of unsafe operand
/// uses, and joins into branch targets' block arguments.
fn transfer(
    ctx: &Context,
    op_ptr: Ptr<Operation>,
    globals: &HashMap<String, GlobalSpace>,
    spaces: &mut HashMap<Value, PtrSpace>,
    forced: &mut HashSet<Value>,
) -> bool {
    let mut changed = false;
    let op_obj = Operation::get_op_dyn(op_ptr, ctx);

    if let Some(addr) = op_obj.downcast_ref::<AddressOfOp>() {
        let space = match globals.get(&addr.get_global_name(ctx).to_string()) {
            Some(GlobalSpace::Global) => PtrSpace::Global,
            Some(GlobalSpace::Shared) => PtrSpace::Shared,
            // Not a module global: emission reports it; call it generic.
            None => PtrSpace::Generic,
        };
        changed |= set_result(spaces, forced, addr.get_result(ctx), space);
    } else if let Some(gep) = op_obj.downcast_ref::<GetElementPtrOp>() {
        if let Some(base) = spaces.get(&gep.get_operand_src_ptr(ctx)).copied() {
            changed |= set_result(spaces, forced, gep.get_result(ctx), base);
        }
        for index in gep.indices(ctx) {
            if let GepIndex::Value(value) = index {
                changed |= escape(spaces, forced, value);
            }
        }
    } else if let Some(cast) = op_obj.downcast_ref::<BitcastOp>() {
        changed |= alias(spaces, forced, cast.get_operand(ctx), cast.get_result(ctx));
    } else if let Some(cast) = op_obj.downcast_ref::<IntToPtrOp>() {
        changed |= alias(spaces, forced, cast.get_operand(ctx), cast.get_result(ctx));
    } else if let Some(cast) = op_obj.downcast_ref::<PtrToIntOp>() {
        changed |= alias(spaces, forced, cast.get_operand(ctx), cast.get_result(ctx));
    } else if let Some(select) = op_obj.downcast_ref::<SelectOp>() {
        let (on_true, on_false, result) = {
            let op = select.get_operation().deref(ctx);
            (op.get_operand(1), op.get_operand(2), op.get_result(0))
        };
        let a = spaces.get(&on_true).copied();
        let b = spaces.get(&on_false).copied();
        let joined = match (a, b) {
            (Some(a), Some(b)) if a == b => Some(a),
            (Some(_), Some(_)) => Some(PtrSpace::Generic),
            (Some(one), None) | (None, Some(one)) => Some(one),
            (None, None) => None,
        };
        if let Some(space) = joined {
            changed |= set_result(spaces, forced, result, space);
        }
    } else if let Some(load) = op_obj.downcast_ref::<LoadOp>() {
        // The address use is what qualification exists for: safe.
        changed |= set_result(spaces, forced, load.get_result(ctx), PtrSpace::Generic);
    } else if let Some(store) = op_obj.downcast_ref::<StoreOp>() {
        // The address use is safe; the stored VALUE leaves the function's
        // view (memory readers expect a generic address).
        changed |= escape(spaces, forced, store.get_operand_value(ctx));
    } else if let Some(br) = op_obj.downcast_ref::<BrOp>() {
        changed |= branch_edge(ctx, spaces, br.get_operation(), 0, &br.successor_operands(ctx, 0));
    } else if let Some(cond_br) = op_obj.downcast_ref::<CondBrOp>() {
        for succ in 0..2 {
            changed |= branch_edge(
                ctx,
                spaces,
                cond_br.get_operation(),
                succ,
                &cond_br.successor_operands(ctx, succ),
            );
        }
    } else if let Some(call) = op_obj.downcast_ref::<CallOp>() {
        let canonical = match call.callee(ctx) {
            CallOpCallable::Direct(name) => Some(nvvm_intrinsic_name(name.as_ref())),
            CallOpCallable::Indirect(_) => None,
        };
        let atomic_ptr_arg = canonical
            .as_deref()
            .is_some_and(|name| atomic_intrinsic(name).is_some());
        // The dynamic shared-memory window base is a `.shared` source,
        // exactly like `llvm.addressof` of a `.shared` global.
        let result_space = if canonical.as_deref() == Some(super::DYN_SHARED_INTRINSIC) {
            PtrSpace::Shared
        } else {
            PtrSpace::Generic
        };
        let op = call.get_operation().deref(ctx);
        for i in 0..op.get_num_operands() {
            // A proven atomic's pointer operand gets space-qualified, like
            // a load's; every other argument escapes.
            if !(atomic_ptr_arg && i == 0) {
                changed |= escape(spaces, forced, op.get_operand(i));
            }
        }
        for result in op.results() {
            changed |= set_result(spaces, forced, result, result_space);
        }
    } else {
        // Anything else neither produces nor preserves provenance: results
        // are generic and shared operands escape.
        let op = op_ptr.deref(ctx);
        for i in 0..op.get_num_operands() {
            changed |= escape(spaces, forced, op.get_operand(i));
        }
        for result in op.results() {
            changed |= set_result(spaces, forced, result, PtrSpace::Generic);
        }
    }
    changed
}

/// A register-aliasing cast: the result shares the operand's provenance
/// (emission inserts the `cvta.shared` when the result was demoted but the
/// operand stayed shared).
fn alias(
    spaces: &mut HashMap<Value, PtrSpace>,
    forced: &HashSet<Value>,
    operand: Value,
    result: Value,
) -> bool {
    if let Some(space) = spaces.get(&operand).copied() {
        set_result(spaces, forced, result, space)
    } else {
        false
    }
}

// Per-root write sets: which memory a load provably reads unwritten ------

/// Pointer provenance for the write-set analysis: the set of ROOTS a value
/// may be derived from, as a bitmask over interned root ids, or `Unknown`
/// when the derivation escapes the tracked ops. Roots are the values that
/// introduce fresh provenance: kernel pointer parameters, `llvm.addressof`
/// results, allocas, and the dynamic shared-memory window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Prov {
    Roots(u64),
    Unknown,
}

impl Prov {
    fn join(self, other: Prov) -> Prov {
        match (self, other) {
            (Prov::Roots(a), Prov::Roots(b)) => Prov::Roots(a | b),
            _ => Prov::Unknown,
        }
    }
}

/// The values through which this function provably reads ONLY memory that
/// no store, atomic, or opaque call in the function ever writes — for the
/// whole launch, since every thread runs this same body. Sound because:
///
/// - provenance is per ROOT (each pointer parameter, `addressof` global,
///   alloca, and the dyn-shared window), propagated through GEPs, the
///   register-aliasing casts, selects, and block arguments; any other
///   producer (loaded pointers included) is `Unknown`;
/// - every store/atomic dirties the roots its address may derive from, and
///   an `Unknown`-address store — or any call that could write memory
///   (device `.func`s, unrecognized intrinsics) — dirties EVERYTHING;
/// - a value qualifies only when its own derivation is fully known and
///   none of its roots is dirty.
///
/// Distinct roots may alias (two params covering one buffer), and a store
/// dirties only its own roots. That is deliberate, not a hole: a launch
/// whose WRITTEN region overlaps another parameter's READ region is
/// already undefined under the kernel ABI (docs/KERNEL-ABI.md) — the
/// pointers derive from `&[T]`/`&mut [T]` borrows the caller split, and an
/// overlapping `&mut` invalidates the shared borrow. nvcc applies the same
/// per-parameter criterion to plain `const T*` params (verified on this
/// corpus's ref.cu files), where C gives it strictly weaker cover. Device
/// `.func`s get `Unknown` pointer params (their callers may pass
/// anything), so their qualifying values can only derive from module
/// globals or their own allocas.
pub(super) fn readonly_values(
    ctx: &Context,
    entry: Ptr<BasicBlock>,
    blocks: &[Ptr<BasicBlock>],
    params_global: bool,
) -> HashSet<Value> {
    let mut prov: HashMap<Value, Prov> = HashMap::new();
    // Stable root ids: assigned once per introducing value. Beyond 64
    // roots new sources degrade to Unknown (never unsound, just weaker).
    let mut root_ids: HashMap<Value, Prov> = HashMap::new();
    let mut next_root = 0u32;
    let mut root_of = |value: Value, next_root: &mut u32| -> Prov {
        *root_ids.entry(value).or_insert_with(|| {
            if *next_root >= 64 {
                Prov::Unknown
            } else {
                let bit = 1u64 << *next_root;
                *next_root += 1;
                Prov::Roots(bit)
            }
        })
    };
    for param in entry.deref(ctx).arguments() {
        let is_ptr = param
            .get_type(ctx)
            .deref(ctx)
            .downcast_ref::<PointerType>()
            .is_some();
        let p = if is_ptr && params_global {
            root_of(param, &mut next_root)
        } else {
            Prov::Unknown
        };
        prov.insert(param, p);
    }

    fn raise_prov(prov: &mut HashMap<Value, Prov>, value: Value, p: Prov) -> bool {
        match prov.get(&value).copied() {
            None => {
                prov.insert(value, p);
                true
            }
            Some(old) => {
                let joined = old.join(p);
                if joined == old {
                    false
                } else {
                    prov.insert(value, joined);
                    true
                }
            }
        }
    }

    loop {
        let mut changed = false;
        for block in blocks {
            for op_ptr in block.deref(ctx).iter(ctx) {
                let op_obj = Operation::get_op_dyn(op_ptr, ctx);
                if let Some(addr) = op_obj.downcast_ref::<AddressOfOp>() {
                    let result = addr.get_result(ctx);
                    let root = root_of(result, &mut next_root);
                    changed |= raise_prov(&mut prov, result, root);
                } else if let Some(alloca) = op_obj.downcast_ref::<AllocaOp>() {
                    let result = alloca.get_result(ctx);
                    let root = root_of(result, &mut next_root);
                    changed |= raise_prov(&mut prov, result, root);
                } else if let Some(gep) = op_obj.downcast_ref::<GetElementPtrOp>() {
                    if let Some(p) = prov.get(&gep.get_operand_src_ptr(ctx)).copied() {
                        changed |= raise_prov(&mut prov, gep.get_result(ctx), p);
                    }
                } else if let Some(cast) = op_obj.downcast_ref::<BitcastOp>() {
                    if let Some(p) = prov.get(&cast.get_operand(ctx)).copied() {
                        changed |= raise_prov(&mut prov, cast.get_result(ctx), p);
                    }
                } else if let Some(cast) = op_obj.downcast_ref::<IntToPtrOp>() {
                    if let Some(p) = prov.get(&cast.get_operand(ctx)).copied() {
                        changed |= raise_prov(&mut prov, cast.get_result(ctx), p);
                    }
                } else if let Some(cast) = op_obj.downcast_ref::<PtrToIntOp>() {
                    if let Some(p) = prov.get(&cast.get_operand(ctx)).copied() {
                        changed |= raise_prov(&mut prov, cast.get_result(ctx), p);
                    }
                } else if let Some(select) = op_obj.downcast_ref::<SelectOp>() {
                    let (on_true, on_false, result) = {
                        let op = select.get_operation().deref(ctx);
                        (op.get_operand(1), op.get_operand(2), op.get_result(0))
                    };
                    let joined = match (prov.get(&on_true), prov.get(&on_false)) {
                        (Some(&a), Some(&b)) => Some(a.join(b)),
                        (Some(&one), None) | (None, Some(&one)) => Some(one),
                        (None, None) => None,
                    };
                    if let Some(p) = joined {
                        changed |= raise_prov(&mut prov, result, p);
                    }
                } else if op_obj.downcast_ref::<StoreOp>().is_some() {
                    // Writes are collected after the fixpoint.
                } else if let Some(br) = op_obj.downcast_ref::<BrOp>() {
                    changed |= prov_branch_edge(
                        ctx,
                        &mut prov,
                        br.get_operation(),
                        0,
                        &br.successor_operands(ctx, 0),
                        raise_prov,
                    );
                } else if let Some(cond_br) = op_obj.downcast_ref::<CondBrOp>() {
                    for succ in 0..2 {
                        changed |= prov_branch_edge(
                            ctx,
                            &mut prov,
                            cond_br.get_operation(),
                            succ,
                            &cond_br.successor_operands(ctx, succ),
                            raise_prov,
                        );
                    }
                } else if let Some(call) = op_obj.downcast_ref::<CallOp>() {
                    let canonical = match call.callee(ctx) {
                        CallOpCallable::Direct(name) => nvvm_intrinsic_name(name.as_ref()),
                        CallOpCallable::Indirect(_) => String::new(),
                    };
                    let dyn_shared = canonical == super::DYN_SHARED_INTRINSIC;
                    let results: Vec<Value> =
                        call.get_operation().deref(ctx).results().collect();
                    for result in results {
                        // The dyn-shared window base is a fresh provenance
                        // source, like `addressof` of a `.shared` global;
                        // every other call result is untracked.
                        let p = if dyn_shared {
                            root_of(result, &mut next_root)
                        } else {
                            Prov::Unknown
                        };
                        changed |= raise_prov(&mut prov, result, p);
                    }
                } else {
                    // Untracked producers (loads included): the result may
                    // be a pointer derived from anything.
                    let op = op_ptr.deref(ctx);
                    for result in op.results() {
                        changed |= raise_prov(&mut prov, result, Prov::Unknown);
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    // The write set: every store/atomic dirties the roots its address may
    // derive from; unknown addresses and opaque calls dirty everything.
    let mut dirty = 0u64;
    let mut dirty_all = false;
    for block in blocks {
        for op_ptr in block.deref(ctx).iter(ctx) {
            let op_obj = Operation::get_op_dyn(op_ptr, ctx);
            if let Some(store) = op_obj.downcast_ref::<StoreOp>() {
                match prov.get(&store.get_operand_address(ctx)) {
                    Some(Prov::Roots(mask)) => dirty |= mask,
                    _ => dirty_all = true,
                }
            } else if let Some(call) = op_obj.downcast_ref::<CallOp>() {
                let canonical = match call.callee(ctx) {
                    CallOpCallable::Direct(name) => nvvm_intrinsic_name(name.as_ref()),
                    CallOpCallable::Indirect(_) => String::new(),
                };
                match call_mem_effect(&canonical) {
                    CallMemEffect::None => {}
                    CallMemEffect::AtomicPtrArg => {
                        let ptr_arg = call.get_operation().deref(ctx).get_operand(0);
                        match prov.get(&ptr_arg) {
                            Some(Prov::Roots(mask)) => dirty |= mask,
                            _ => dirty_all = true,
                        }
                    }
                    CallMemEffect::Unknown => dirty_all = true,
                }
            }
        }
    }
    if dirty_all {
        return HashSet::new();
    }
    prov.iter()
        .filter_map(|(value, p)| match p {
            Prov::Roots(mask) if mask & dirty == 0 => Some(*value),
            _ => None,
        })
        .collect()
}

/// Join one branch edge's operand provenance into the target's block
/// arguments (the [readonly_values] analogue of [branch_edge]).
fn prov_branch_edge(
    ctx: &Context,
    prov: &mut HashMap<Value, Prov>,
    op: Ptr<Operation>,
    succ: usize,
    args: &[Value],
    raise_prov: fn(&mut HashMap<Value, Prov>, Value, Prov) -> bool,
) -> bool {
    let dest = op.deref(ctx).get_successor(succ);
    let dest_args: Vec<Value> = dest.deref(ctx).arguments().collect();
    let mut changed = false;
    for (arg, dest_arg) in args.iter().copied().zip(dest_args) {
        if let Some(p) = prov.get(&arg).copied() {
            changed |= raise_prov(prov, dest_arg, p);
        }
    }
    changed
}

/// Join one branch edge's operand spaces into the target's block arguments.
fn branch_edge(
    ctx: &Context,
    spaces: &mut HashMap<Value, PtrSpace>,
    op: Ptr<Operation>,
    succ: usize,
    args: &[Value],
) -> bool {
    let dest = op.deref(ctx).get_successor(succ);
    let dest_args: Vec<Value> = dest.deref(ctx).arguments().collect();
    let mut changed = false;
    for (arg, dest_arg) in args.iter().copied().zip(dest_args) {
        if let Some(space) = spaces.get(&arg).copied() {
            changed |= raise(spaces, dest_arg, space);
        }
    }
    changed
}
