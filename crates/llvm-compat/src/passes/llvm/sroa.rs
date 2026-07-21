//! Scalar replacement of aggregates for `llvm.alloca` slots.
//!
//! Splits an alloca of a small struct whose layout flattens to 8-byte
//! leaves (64-bit integers and pointers) into one scalar alloca per leaf,
//! so that scalar mem2reg can promote the pieces. Handled accesses:
//!
//! - whole-aggregate `llvm.load` / `llvm.store`, decomposed into per-leaf
//!   ops plus `insertvalue`/`extractvalue` glue;
//! - direct scalar loads/stores of the first leaf (byte offset 0);
//! - `llvm.gep` with a constant byte offset naming a leaf boundary, whose
//!   result is only used as a scalar load/store address.
//!
//! Integer accesses whose signedness differs from the leaf type are
//! adapted with `llvm.bitcast` so every rewritten slot sees one type.

use crate::{
    context::{Context, Ptr},
    debug_info::{get_operation_result_name, set_operation_result_name},
    dialects::{
        builtin::{
            attributes::IntegerAttr,
            op_interfaces::{OneRegionInterface, OneResultInterface},
            types::{IntegerType, Signedness},
        },
        llvm::{
            attributes::GepIndexAttr,
            op_interfaces::IsDeclaration,
            ops::{
                AllocaOp, BitcastOp, ConstantOp, ExtractValueOp, GetElementPtrOp, InsertValueOp,
                LoadOp, StoreOp, UndefOp,
            },
            types::{ArrayType, PointerType, StructType},
        },
    },
    identifier::Identifier,
    ir::{
        op::Op,
        operation::Operation,
        r#type::{TypeHandle, Typed},
        value::Value,
    },
    linked_list::ContainsLinkedList,
    conversion::pass::{Pass, PassOptions},
    result::STAIRResult,
    utils::apint::APInt,
};

use super::inline::collect_functions;

/// Split struct allocas into at most this many leaf slots.
const MAX_LEAVES: usize = 4;

pub struct LLVMSroaPass;

impl Pass for LLVMSroaPass {
    fn name(&self) -> &str {
        "llvm-sroa"
    }

    fn run(
        &self,
        root: Ptr<Operation>,
        ctx: &mut Context,
        _options: PassOptions,
    ) -> STAIRResult<Ptr<Operation>> {
        for func in collect_functions(ctx, root) {
            if func.is_declaration(ctx) {
                continue;
            }
            let mut allocas = Vec::new();
            for block in func.get_region(ctx).deref(ctx).iter(ctx) {
                for op in block.deref(ctx).iter(ctx) {
                    if Operation::get_opid(op, ctx) == AllocaOp::get_opid_static() {
                        allocas.push(AllocaOp::from_operation(op));
                    }
                }
            }
            for alloca in allocas {
                try_split_alloca(ctx, alloca)?;
            }
        }
        Ok(root)
    }
}

/// One 8-byte leaf of a flattened aggregate: the `insertvalue` /
/// `extractvalue` access path from the aggregate root, and the leaf type.
struct Leaf {
    path: Vec<u32>,
    ty: TypeHandle,
}

/// Flatten `ty` into 8-byte leaves. Returns `None` for types with leaves
/// that are not single-register scalars (the layout would need padding
/// bookkeeping this pass does not have).
fn flatten_type(ctx: &Context, ty: TypeHandle, path: &mut Vec<u32>, out: &mut Vec<Leaf>) -> Option<()> {
    let ty_ref = ty.deref(ctx);
    if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
        if int_ty.width() != 64 {
            return None;
        }
        out.push(Leaf {
            path: path.clone(),
            ty,
        });
        return Some(());
    }
    if ty_ref.downcast_ref::<PointerType>().is_some() {
        out.push(Leaf {
            path: path.clone(),
            ty,
        });
        return Some(());
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<StructType>() {
        let fields: Vec<_> = struct_ty.fields()?.to_vec();
        drop(ty_ref);
        for (idx, field) in fields.into_iter().enumerate() {
            path.push(idx as u32);
            flatten_type(ctx, field, path, out)?;
            path.pop();
        }
        return Some(());
    }
    if let Some(array_ty) = ty_ref.downcast_ref::<ArrayType>() {
        let elem = array_ty.elem_type();
        let size = array_ty.size();
        drop(ty_ref);
        for idx in 0..size {
            path.push(idx as u32);
            flatten_type(ctx, elem, path, out)?;
            path.pop();
        }
        return Some(());
    }
    None
}

/// A classified access to the aggregate slot.
enum SlotAccess {
    WholeLoad(LoadOp),
    WholeStore(StoreOp),
    /// Scalar load/store of leaf `slot`, possibly through the gep in
    /// `via_gep` (which is erased after its accesses are rewritten).
    ScalarLoad {
        load: LoadOp,
        slot: usize,
    },
    ScalarStore {
        store: StoreOp,
        slot: usize,
    },
}

fn try_split_alloca(ctx: &mut Context, alloca: AllocaOp) -> STAIRResult<()> {
    let elem_ty = alloca.get_elem_type(ctx);
    if elem_ty.deref(ctx).downcast_ref::<StructType>().is_none() {
        return Ok(());
    }
    let mut leaves = Vec::new();
    if flatten_type(ctx, elem_ty, &mut Vec::new(), &mut leaves).is_none() {
        return Ok(());
    }
    // A single-leaf struct is left to store-to-load forwarding; splitting
    // pays off when independent leaves untangle.
    if leaves.len() < 2 || leaves.len() > MAX_LEAVES {
        return Ok(());
    }

    let Some((accesses, geps)) = classify_accesses(ctx, alloca, elem_ty, &leaves) else {
        return Ok(());
    };

    // Materialize one scalar alloca per leaf right after the original.
    let base_name = get_operation_result_name(ctx, alloca.get_operation(), 0);
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let mut slot_allocas = Vec::with_capacity(leaves.len());
    let mut insert_after = alloca.get_operation();
    for (idx, leaf) in leaves.iter().enumerate() {
        let one = ConstantOp::new_integer(
            ctx,
            IntegerAttr::new(i64_ty, APInt::from_u64(1, std::num::NonZero::new(64).unwrap())),
        );
        one.get_operation().insert_after(ctx, insert_after);
        insert_after = one.get_operation();
        let one_val = one.get_result(ctx);
        let slot_alloca = AllocaOp::new(ctx, one_val, leaf.ty);
        slot_alloca.get_operation().insert_after(ctx, insert_after);
        insert_after = slot_alloca.get_operation();
        if let Some(name) = &base_name {
            if let Ok(slot_name) = Identifier::try_from(format!("{name}_f{idx}")) {
                set_operation_result_name(ctx, slot_alloca.get_operation(), 0, Some(slot_name));
            }
        }
        slot_allocas.push(slot_alloca.get_result(ctx));
    }

    for access in accesses {
        match access {
            SlotAccess::WholeLoad(load) => {
                rewrite_whole_load(ctx, load, elem_ty, &leaves, &slot_allocas);
            }
            SlotAccess::WholeStore(store) => {
                rewrite_whole_store(ctx, store, &leaves, &slot_allocas);
            }
            SlotAccess::ScalarLoad { load, slot } => {
                rewrite_scalar_load(ctx, load, leaves[slot].ty, slot_allocas[slot]);
            }
            SlotAccess::ScalarStore { store, slot } => {
                rewrite_scalar_store(ctx, store, leaves[slot].ty, slot_allocas[slot]);
            }
        }
    }
    for gep in geps {
        Operation::erase(gep, ctx);
    }
    Operation::erase(alloca.get_operation(), ctx);
    Ok(())
}

/// Gather every access to the slot, or `None` if any use does not fit the
/// supported patterns. Also returns the geps to erase after rewriting.
fn classify_accesses(
    ctx: &Context,
    alloca: AllocaOp,
    elem_ty: TypeHandle,
    leaves: &[Leaf],
) -> Option<(Vec<SlotAccess>, Vec<Ptr<Operation>>)> {
    let slot = alloca.get_result(ctx);
    let mut accesses = Vec::new();
    let mut geps = Vec::new();

    for slot_use in slot.uses(ctx) {
        let user = slot_use.user_op();
        let opid = Operation::get_opid(user, ctx);
        if opid == LoadOp::get_opid_static() && slot_use.find_index(ctx) == 0 {
            let load = LoadOp::from_operation(user);
            let result_ty = load.get_result(ctx).get_type(ctx);
            if result_ty == elem_ty {
                accesses.push(SlotAccess::WholeLoad(load));
            } else if scalar_compatible(ctx, result_ty, leaves[0].ty) {
                accesses.push(SlotAccess::ScalarLoad { load, slot: 0 });
            } else {
                return None;
            }
        } else if opid == StoreOp::get_opid_static() && slot_use.find_index(ctx) == 1 {
            let store = StoreOp::from_operation(user);
            let value = store.get_value(ctx);
            if value == slot {
                return None;
            }
            let value_ty = value.get_type(ctx);
            if value_ty == elem_ty {
                accesses.push(SlotAccess::WholeStore(store));
            } else if scalar_compatible(ctx, value_ty, leaves[0].ty) {
                accesses.push(SlotAccess::ScalarStore { store, slot: 0 });
            } else {
                return None;
            }
        } else if opid == GetElementPtrOp::get_opid_static() && slot_use.find_index(ctx) == 0 {
            let gep = GetElementPtrOp::from_operation(user);
            let offset = constant_byte_offset(ctx, gep)?;
            if offset % 8 != 0 {
                return None;
            }
            let slot_idx = (offset / 8) as usize;
            if slot_idx >= leaves.len() {
                return None;
            }
            let leaf_ty = leaves[slot_idx].ty;
            let gep_result = gep.get_result(ctx);
            for gep_use in gep_result.uses(ctx) {
                let access_op = gep_use.user_op();
                let access_opid = Operation::get_opid(access_op, ctx);
                if access_opid == LoadOp::get_opid_static() && gep_use.find_index(ctx) == 0 {
                    let load = LoadOp::from_operation(access_op);
                    if !scalar_compatible(ctx, load.get_result(ctx).get_type(ctx), leaf_ty) {
                        return None;
                    }
                    accesses.push(SlotAccess::ScalarLoad {
                        load,
                        slot: slot_idx,
                    });
                } else if access_opid == StoreOp::get_opid_static() && gep_use.find_index(ctx) == 1 {
                    let store = StoreOp::from_operation(access_op);
                    if !scalar_compatible(ctx, store.get_value(ctx).get_type(ctx), leaf_ty) {
                        return None;
                    }
                    accesses.push(SlotAccess::ScalarStore {
                        store,
                        slot: slot_idx,
                    });
                } else {
                    return None;
                }
            }
            geps.push(gep.get_operation());
        } else {
            return None;
        }
    }
    Some((accesses, geps))
}

/// The gep's byte offset from its base when the source element type is a
/// byte and every index is a compile-time constant.
fn constant_byte_offset(ctx: &Context, gep: GetElementPtrOp) -> Option<u64> {
    let source = gep.get_source_elem_type(ctx);
    let source_ref = source.deref(ctx);
    let int_ty = source_ref.downcast_ref::<IntegerType>()?;
    if int_ty.width() != 8 {
        return None;
    }
    drop(source_ref);
    let indices = gep.get_indices(ctx).0;
    if indices.len() != 1 {
        return None;
    }
    match indices[0] {
        GepIndexAttr::Constant(value) => Some(value as u64),
        GepIndexAttr::OperandIdx(dyn_idx) => {
            // Dynamic indices start at operand 1 (operand 0 is the base).
            let operand = gep.get_operation().deref(ctx).get_operand(dyn_idx + 1);
            let Some(op) = operand.defining_op().filter(|_| operand.find_index(ctx) == 0) else {
                return None;
            };
            if Operation::get_opid(op, ctx) != ConstantOp::get_opid_static() {
                return None;
            }
            Some((ConstantOp::from_operation(op)).get_value(ctx)?.value().to_u64())
        }
    }
}

/// Same type, or both 64-bit integers that only differ in signedness.
fn scalar_compatible(ctx: &Context, access_ty: TypeHandle, leaf_ty: TypeHandle) -> bool {
    if access_ty == leaf_ty {
        return true;
    }
    match (
        access_ty.deref(ctx).downcast_ref::<IntegerType>(),
        leaf_ty.deref(ctx).downcast_ref::<IntegerType>(),
    ) {
        (Some(a), Some(b)) => a.width() == b.width(),
        _ => false,
    }
}

/// `value` adapted to `want_ty` with a bitcast inserted before `before`
/// when the types differ.
fn adapt_value(
    ctx: &mut Context,
    value: Value,
    want_ty: TypeHandle,
    before: Ptr<Operation>,
) -> Value {
    if value.get_type(ctx) == want_ty {
        return value;
    }
    let cast = BitcastOp::new(ctx, value, want_ty);
    cast.get_operation().insert_before(ctx, before);
    cast.get_result(ctx)
}

fn rewrite_whole_load(
    ctx: &mut Context,
    load: LoadOp,
    elem_ty: TypeHandle,
    leaves: &[Leaf],
    slot_allocas: &[Value],
) {
    let load_op = load.get_operation();
    let mut aggregate = {
        let undef = UndefOp::new(ctx, elem_ty);
        undef.get_operation().insert_before(ctx, load_op);
        undef.get_result(ctx)
    };
    for (leaf, &slot_alloca) in leaves.iter().zip(slot_allocas) {
        let leaf_load = LoadOp::new(ctx, slot_alloca, leaf.ty);
        leaf_load.get_operation().insert_before(ctx, load_op);
        let leaf_value = leaf_load.get_result(ctx);
        let insert = InsertValueOp::new(ctx, leaf_value, aggregate, leaf.path.clone());
        insert.get_operation().insert_before(ctx, load_op);
        aggregate = insert.get_result(ctx);
    }
    let result = load.get_result(ctx);
    result.replace_some_uses_with(ctx, |_, _| true, &aggregate);
    Operation::erase(load_op, ctx);
}

fn rewrite_whole_store(
    ctx: &mut Context,
    store: StoreOp,
    leaves: &[Leaf],
    slot_allocas: &[Value],
) {
    let store_op = store.get_operation();
    let value = store.get_value(ctx);
    for (leaf, &slot_alloca) in leaves.iter().zip(slot_allocas) {
        let leaf_value = resolve_leaf(ctx, value, &leaf.path, leaf.ty, store_op);
        StoreOp::new(ctx, leaf_value, slot_alloca)
            .get_operation()
            .insert_before(ctx, store_op);
    }
    Operation::erase(store_op, ctx);
}

/// The value of the leaf at `path` inside `aggregate`. Walks insertvalue
/// chains; a leaf the chain never set is an explicit `undef` (an
/// `extractvalue` of an unset field cannot be lowered). Falls back to an
/// `extractvalue` op for opaque aggregate producers.
fn resolve_leaf(
    ctx: &mut Context,
    aggregate: Value,
    path: &[u32],
    leaf_ty: TypeHandle,
    before: Ptr<Operation>,
) -> Value {
    let mut current = aggregate;
    let mut path = path.to_vec();
    loop {
        let Some(op) = current.defining_op().filter(|_| current.find_index(ctx) == 0) else {
            break;
        };
        let opid = Operation::get_opid(op, ctx);
        if opid == UndefOp::get_opid_static() {
            let undef = UndefOp::new(ctx, leaf_ty);
            undef.get_operation().insert_before(ctx, before);
            return undef.get_result(ctx);
        }
        if opid != InsertValueOp::get_opid_static() {
            break;
        }
        let insert = InsertValueOp::from_operation(op);
        let indices = insert.get_indices(ctx);
        if indices == path {
            return insert.get_value(ctx);
        }
        if indices.len() < path.len() && path[..indices.len()] == indices[..] {
            // The insert wrote a sub-aggregate containing our leaf.
            path.drain(..indices.len());
            current = insert.get_value(ctx);
            continue;
        }
        if path.len() < indices.len() && indices[..path.len()] == path[..] {
            // The insert wrote inside our leaf: cannot happen for 8-byte
            // scalar leaves; treat the producer as opaque.
            break;
        }
        current = insert.get_aggregate(ctx);
    }
    let extract = ExtractValueOp::new(ctx, current, path, leaf_ty);
    extract.get_operation().insert_before(ctx, before);
    extract.get_result(ctx)
}

fn rewrite_scalar_load(ctx: &mut Context, load: LoadOp, leaf_ty: TypeHandle, slot_alloca: Value) {
    let load_op = load.get_operation();
    let result = load.get_result(ctx);
    let result_ty = result.get_type(ctx);
    let leaf_load = LoadOp::new(ctx, slot_alloca, leaf_ty);
    leaf_load.get_operation().insert_before(ctx, load_op);
    let mut value = leaf_load.get_result(ctx);
    if result_ty != leaf_ty {
        let cast = BitcastOp::new(ctx, value, result_ty);
        cast.get_operation().insert_before(ctx, load_op);
        value = cast.get_result(ctx);
    }
    result.replace_some_uses_with(ctx, |_, _| true, &value);
    Operation::erase(load_op, ctx);
}

fn rewrite_scalar_store(
    ctx: &mut Context,
    store: StoreOp,
    leaf_ty: TypeHandle,
    slot_alloca: Value,
) {
    let store_op = store.get_operation();
    let value = adapt_value(ctx, store.get_value(ctx), leaf_ty, store_op);
    StoreOp::new(ctx, value, slot_alloca)
        .get_operation()
        .insert_before(ctx, store_op);
    Operation::erase(store_op, ctx);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dialects::{
            builtin::{self},
            llvm::{
                self,
                attributes::LinkageAttr,
                ops::{FuncOp, ReturnOp},
                types::FuncType,
            },
        },
        ir::r#type::TypedHandle,
        conversion::pass::Pass,
        printable::Printable,
    };
    use std::num::NonZero;

    fn test_context() -> Context {
        let mut ctx = Context::new();
        llvm::register(&mut ctx);
        ctx
    }

    #[test]
    fn splits_two_field_struct_alloca() {
        let mut ctx = test_context();
        let u64_ty: TypeHandle = IntegerType::get(&mut ctx, 64, Signedness::Signless).into();
        let pair_ty: TypeHandle =
            StructType::get_unnamed(&mut ctx, vec![u64_ty, u64_ty]).into();
        let fn_ty: TypedHandle<FuncType> =
            FuncType::get(&mut ctx, pair_ty, vec![pair_ty], false);
        let func = FuncOp::new(
            &mut ctx,
            "split".try_into().unwrap(),
            fn_ty,
            LinkageAttr::External,
        );
        let entry = func.get_entry_block(&ctx);
        let arg = entry.deref(&ctx).get_argument(0);

        let i64_ty = IntegerType::get(&mut ctx, 64, Signedness::Signless);
        let one = ConstantOp::new_integer(
            &mut ctx,
            IntegerAttr::new(i64_ty, APInt::from_u64(1, NonZero::new(64).unwrap())),
        );
        one.get_operation().insert_at_back(entry, &ctx);
        let one_val = one.get_result(&ctx);
        let alloca = AllocaOp::new(&mut ctx, one_val, pair_ty);
        alloca.get_operation().insert_at_back(entry, &ctx);
        let slot = alloca.get_result(&ctx);
        StoreOp::new(&mut ctx, arg, slot)
            .get_operation()
            .insert_at_back(entry, &ctx);
        let load = LoadOp::new(&mut ctx, slot, pair_ty);
        load.get_operation().insert_at_back(entry, &ctx);
        let loaded = load.get_result(&ctx);
        ReturnOp::new(&mut ctx, Some(loaded))
            .get_operation()
            .insert_at_back(entry, &ctx);

        LLVMSroaPass
            .run(func.get_operation(), &mut ctx, PassOptions::default())
            .unwrap();

        let text = format!("{}", func.get_operation().disp(&ctx));
        // The struct alloca is gone; two scalar allocas remain.
        assert!(!text.contains("x llvm.struct"), "{text}");
        assert_eq!(text.matches("llvm.alloca").count(), 2, "{text}");
        assert!(text.contains("llvm.extractvalue"), "{text}");
        assert!(text.contains("llvm.insertvalue"), "{text}");
    }
}
