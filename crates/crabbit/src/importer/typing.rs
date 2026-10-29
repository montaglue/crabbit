//! Pointer typing of the imported module, after the walk.
//!
//! The walk (`place`, `rvalue`, ...) builds byte-addressed, opaque thin
//! pointers — every one has [ptr_ty] (`mir.ptr<u8>`, Erased): field and
//! element addresses are byte offsets, and the pointee is known at the use,
//! not at the address computation. dialect-mir's verifier wants the pointee
//! named at those uses: a load's operand points at its result type, a
//! store's at its value type, a callee's parameter has its declared type, a
//! `ptr_offset` steps in units of its pointee. This pass makes every such use
//! well-typed with a representation-only `mir.cast PtrToPtr` (same
//! mutability, address space and provenance kind — `mir-lower` lowers it to
//! nothing), which is the typing a pointee-aware walk would produce at the
//! same sites, since the aggregates are LLVM structs addressed by byte
//! offsets.
//!
//! Two kinds of values live outside the dialect's pointer model and are
//! bridged where they meet it:
//! - `llvm.addressof` results — the address of a host global, a thread-local
//!   or a function. dialect-mir has producers for device address spaces
//!   only (`mir.global_alloc`: global/constant memory), none for a
//!   host-generic global, and no thread-locals. An `llvm.ptrtoint` followed
//!   by a `mir.cast PointerWithExposedProvenance` makes such an address the
//!   raw `mir.ptr` the dialect defines for an address turned pointer; the
//!   reverse crossing (a `mir.ptr` into an `llvm.*` op: indirect calls,
//!   pointer compares) is `PointerExposeAddress` plus `llvm.inttoptr`. The
//!   mid-end folds the `inttoptr(ptrtoint x)` pairs back to `x`.
//! - calls to symbols that are not `mir.func`s of the module (`llvm.func`
//!   declarations of C and runtime symbols): the dialect requires such a
//!   call to carry the foreign signature itself (`external_callee_type`
//!   under the `AbiBoundary` authority). It is stamped from the call's own
//!   operand and result types, which the walk derived from the declaration.
//! - the address of a `mir.func` of the module (a reified `fn` item, a
//!   function-pointer constant): an `llvm.addressof` may only name an
//!   `llvm.func` or `llvm.global`, so it becomes `ll.func_addr` until the
//!   lowering (`materialize-function-addresses`).
//!
//! Also here: `llvm.lshr` on signed operands (the walk's synthesized logical
//! shifts; the llvm dialect's arithmetic wants signless integers) becomes
//! `mir.shr` over the unsigned twin type.

use std::collections::BTreeMap;

use super::*;
use crate::dialects::builtin::{
    attr_interfaces::TypedAttrInterface, attributes::TypeAttr,
    type_interfaces::FunctionTypeInterface,
};
use crate::dialects::dialect_mir::{
    attributes::{MirCastKindAttr, MirPointerKindAuthorityAttr},
    ops as dm_ops,
    types::{MirPointerKind, MirPtrType},
};
use crate::dialects::llvm::op_interfaces::CastOpInterface as _;
use crate::ir::r#type::type_cast;

type Signature = (Vec<TypeHandle>, Vec<TypeHandle>);

/// Type every pointer use of the `mir.func`s in `module_body` and stamp the
/// foreign-call signatures (module docs).
pub(super) fn normalize_module(ctx: &mut Context, module_body: Ptr<BasicBlock>) {
    let top_level: Vec<Ptr<Operation>> = module_body.deref(ctx).iter(ctx).collect();
    let mut signatures: BTreeMap<String, Signature> = BTreeMap::new();
    let mut functions = Vec::new();
    for op in top_level {
        let Some(func) = dm_ops::MirFuncOp::wrap(ctx, op) else {
            continue;
        };
        let symbol = func.get_symbol_name(ctx).to_string();
        let func_ty = func
            .get_attr_mir_func_type(ctx)
            .map(|attr| TypedAttrInterface::get_type(&*attr, ctx))
            .expect("mir.func carries its function type");
        signatures.insert(symbol.clone(), function_signature(ctx, func_ty));
        functions.push((op, symbol));
    }
    for (func, symbol) in functions {
        let results = signatures[&symbol].1.clone();
        let blocks: Vec<Ptr<BasicBlock>> =
            func.deref(ctx).get_region(0).deref(ctx).iter(ctx).collect();
        for block in blocks {
            let ops: Vec<Ptr<Operation>> = block.deref(ctx).iter(ctx).collect();
            for op in ops {
                normalize_op(ctx, &signatures, &results, op);
            }
        }
    }
}

fn function_signature(ctx: &Context, ty: TypeHandle) -> Signature {
    let ty_ref = ty.deref(ctx);
    let func = type_cast::<dyn FunctionTypeInterface>(&*ty_ref)
        .expect("mir.func type implements FunctionTypeInterface");
    (func.arg_types().to_vec(), func.res_types().to_vec())
}

fn normalize_op(
    ctx: &mut Context,
    signatures: &BTreeMap<String, Signature>,
    func_results: &[TypeHandle],
    op: Ptr<Operation>,
) {
    let opid = Operation::get_opid(op, ctx).to_string();
    match opid.as_str() {
        "mir.load" => {
            let pointee = result_ty(ctx, op, 0);
            let want = ox::ptr_like(ctx, operand_ty(ctx, op, 0), pointee);
            coerce(ctx, op, 0, want);
        }
        "mir.store" => {
            let pointee = operand_ty(ctx, op, 1);
            let want = ox::ptr_like(ctx, operand_ty(ctx, op, 0), pointee);
            coerce(ctx, op, 0, want);
        }
        "mir.ptr_offset" => {
            if let Some(pointee) = mir_pointee(ctx, result_ty(ctx, op, 0)) {
                let want = ox::ptr_like(ctx, operand_ty(ctx, op, 0), pointee);
                coerce(ctx, op, 0, want);
            }
        }
        "mir.cast" => normalize_cast(ctx, op),
        "mir.call" => normalize_call(ctx, signatures, op),
        "mir.return" => {
            let n = op.deref(ctx).get_num_operands().min(func_results.len());
            for (index, want) in func_results.iter().take(n).enumerate() {
                coerce(ctx, op, index, *want);
            }
        }
        // (aggregate, value): the value takes the addressed field's type.
        "llvm.insert_value" => {
            let indices = aggregate_indices(ctx, op);
            if let Some(field_ty) = aggregate_field_type(ctx, operand_ty(ctx, op, 0), &indices) {
                coerce(ctx, op, 1, field_ty);
            }
        }
        "llvm.extract_value" => {
            let indices = aggregate_indices(ctx, op);
            if let Some(field_ty) = aggregate_field_type(ctx, operand_ty(ctx, op, 0), &indices) {
                let result = op.deref(ctx).get_result(0);
                if result.get_type(ctx) != field_ty {
                    result.set_type(ctx, field_ty);
                }
            }
        }
        // Pointers compare as addresses.
        "llvm.icmp" => {
            let address = address_ty(ctx);
            for index in 0..2 {
                if is_pointer_ty(ctx, operand_ty(ctx, op, index)) {
                    coerce(ctx, op, index, address);
                }
            }
        }
        // An indirect call stays in the llvm dialect: its callee and pointer
        // arguments cross over as `llvm.ptr`.
        "llvm.call" => {
            let n = op.deref(ctx).get_num_operands();
            let llvm_ptr: TypeHandle = llvm::types::PointerType::get(ctx, 0).into();
            for index in 0..n {
                if is_mir_ptr_ty(ctx, operand_ty(ctx, op, index)) {
                    coerce(ctx, op, index, llvm_ptr);
                }
            }
        }
        "llvm.lshr" => normalize_signed_lshr(ctx, op),
        // The address of a `mir.func` of the module: an `llvm.addressof` may
        // only name an `llvm.func` or `llvm.global`, so until the lowering
        // it is `ll.func_addr`.
        "llvm.addressof" => normalize_function_address(ctx, signatures, op),
        _ => {}
    }
}

fn normalize_function_address(
    ctx: &mut Context,
    signatures: &BTreeMap<String, Signature>,
    op: Ptr<Operation>,
) {
    let Some(address_of) = Operation::get_op::<llvm::ops::AddressOfOp>(op, ctx) else {
        return;
    };
    let symbol = address_of.get_global_name(ctx);
    if !signatures.contains_key(&symbol.to_string()) {
        return;
    }
    let result = op.deref(ctx).get_result(0);
    let ty = result.get_type(ctx);
    let func_addr = crate::dialects::ll::ops::FuncAddrOp::new(ctx, symbol, ty).get_operation();
    func_addr.insert_before(ctx, op);
    let replacement = func_addr.deref(ctx).get_result(0);
    result.replace_all_uses_with(ctx, &replacement);
    Operation::erase(op, ctx);
}

fn normalize_cast(ctx: &mut Context, op: Ptr<Operation>) {
    let Some(cast) = Operation::get_op::<dm_ops::MirCastOp>(op, ctx) else {
        return;
    };
    let Some(kind) = cast.get_attr_cast_kind(ctx).map(|kind| kind.clone()) else {
        return;
    };
    let have = operand_ty(ctx, op, 0);
    if !is_llvm_ptr_ty(ctx, have) {
        return;
    }
    match kind {
        // The operand crosses into the dialect; the cast is then an identity.
        MirCastKindAttr::PtrToPtr => {
            let want = result_ty(ctx, op, 0);
            coerce(ctx, op, 0, want);
        }
        // An llvm pointer's address is `llvm.ptrtoint`.
        MirCastKindAttr::PointerExposeAddress => {
            let input = op.deref(ctx).get_operand(0);
            let result = op.deref(ctx).get_result(0);
            let int_ty = result.get_type(ctx);
            let ptrtoint = llvm::ops::PtrToIntOp::new(ctx, input, int_ty).get_operation();
            ptrtoint.insert_before(ctx, op);
            let replacement = ptrtoint.deref(ctx).get_result(0);
            result.replace_all_uses_with(ctx, &replacement);
            Operation::erase(op, ctx);
        }
        _ => {}
    }
}

fn normalize_call(ctx: &mut Context, signatures: &BTreeMap<String, Signature>, op: Ptr<Operation>) {
    let Some(call) = Operation::get_op::<dm_ops::MirCallOp>(op, ctx) else {
        return;
    };
    let Some(callee) = call
        .get_attr_callee(ctx)
        .map(|c| String::from((*c).clone()))
    else {
        return;
    };
    if let Some((inputs, outputs)) = signatures.get(&callee) {
        let n = op.deref(ctx).get_num_operands().min(inputs.len());
        for (index, want) in inputs.iter().take(n).enumerate() {
            coerce(ctx, op, index, *want);
        }
        if op.deref(ctx).get_num_results() == 1
            && let Some(declared) = outputs.first()
        {
            let result = op.deref(ctx).get_result(0);
            let have = result.get_type(ctx);
            if have != *declared && is_pointer_ty(ctx, have) && is_pointer_ty(ctx, *declared) {
                result.set_type(ctx, *declared);
            }
        }
        return;
    }
    // dialect-mir's own intrinsic placeholders have a closed schema and must
    // not claim a foreign signature.
    if callee.starts_with(crate::dialects::dialect_mir::rust_intrinsics::PLACEHOLDER_PREFIX) {
        return;
    }
    // A foreign symbol (an `llvm.func` declaration, or defined elsewhere):
    // the call carries its signature.
    let arg_tys: Vec<TypeHandle> = (0..op.deref(ctx).get_num_operands())
        .map(|index| operand_ty(ctx, op, index))
        .collect();
    let res_tys: Vec<TypeHandle> = (0..op.deref(ctx).get_num_results())
        .map(|index| result_ty(ctx, op, index))
        .collect();
    let signature = FunctionType::get(ctx, arg_tys, res_tys);
    call.set_attr_external_callee_type(ctx, TypeAttr::new(signature.into()));
    call.set_attr_call_pointer_kind_authority(ctx, MirPointerKindAuthorityAttr::AbiBoundary);
}

/// `llvm.lshr` on signed integers → `mir.shr` over the unsigned twin type
/// (a logical shift there), with `IntToInt` retypes around it.
fn normalize_signed_lshr(ctx: &mut Context, op: Ptr<Operation>) {
    let ty = operand_ty(ctx, op, 0);
    let Some((width, signed)) = ty
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .map(|int| (int.width(), int.signedness() == Signedness::Signed))
    else {
        return;
    };
    if !signed {
        return;
    }
    let unsigned: TypeHandle = IntegerType::get(ctx, width, Signedness::Unsigned).into();
    let lhs = op.deref(ctx).get_operand(0);
    let rhs = op.deref(ctx).get_operand(1);
    let lhs_cast = ox::mir_cast(ctx, lhs, unsigned, MirCastKindAttr::IntToInt);
    let lhs = emit_before(ctx, op, lhs_cast);
    let rhs_ty = rhs.get_type(ctx);
    let rhs = if rhs_ty == unsigned {
        rhs
    } else {
        let rhs_cast = ox::mir_cast(ctx, rhs, unsigned, MirCastKindAttr::IntToInt);
        emit_before(ctx, op, rhs_cast)
    };
    let shr = Operation::new(
        ctx,
        dm_ops::MirShrOp::get_concrete_op_info(),
        vec![unsigned],
        vec![lhs, rhs],
        vec![],
        0,
    );
    let shifted = emit_before(ctx, op, shr);
    let back_cast = ox::mir_cast(ctx, shifted, ty, MirCastKindAttr::IntToInt);
    let back = emit_before(ctx, op, back_cast);
    let old = op.deref(ctx).get_result(0);
    old.replace_all_uses_with(ctx, &back);
    Operation::erase(op, ctx);
}

// ----------------------------------------------------------------------------
// Coercions
// ----------------------------------------------------------------------------

/// Make operand `index` of `op` have type `want`, converting before `op`.
fn coerce(ctx: &mut Context, op: Ptr<Operation>, index: usize, want: TypeHandle) {
    let value = op.deref(ctx).get_operand(index);
    if value.get_type(ctx) == want {
        return;
    }
    if let Some(converted) = convert_before(ctx, op, value, want) {
        Operation::replace_operand(op, ctx, index, converted);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Shape {
    MirPtr {
        pointee: TypeHandle,
        is_mutable: bool,
        address_space: u32,
    },
    LlvmPtr,
    Int,
    Other,
}

fn shape(ctx: &Context, ty: TypeHandle) -> Shape {
    let ty_ref = ty.deref(ctx);
    if let Some(ptr) = ty_ref.downcast_ref::<MirPtrType>() {
        Shape::MirPtr {
            pointee: ptr.pointee,
            is_mutable: ptr.is_mutable,
            address_space: ptr.address_space,
        }
    } else if ty_ref.is::<llvm::types::PointerType>() {
        Shape::LlvmPtr
    } else if ty_ref.is::<IntegerType>() {
        Shape::Int
    } else {
        Shape::Other
    }
}

/// `value` converted to `want`, the conversion ops inserted before `before`.
/// `None` when the pair is none of this pass's business.
fn convert_before(
    ctx: &mut Context,
    before: Ptr<Operation>,
    value: Value,
    want: TypeHandle,
) -> Option<Value> {
    let have = value.get_type(ctx);
    match (shape(ctx, have), shape(ctx, want)) {
        (Shape::MirPtr { .. }, Shape::MirPtr { .. }) => {
            let cast = ox::mir_cast(ctx, value, want, MirCastKindAttr::PtrToPtr);
            Some(emit_before(ctx, before, cast))
        }
        (
            Shape::LlvmPtr,
            Shape::MirPtr {
                pointee,
                is_mutable,
                address_space,
            },
        ) => {
            let address_ty = address_ty(ctx);
            let ptrtoint = llvm::ops::PtrToIntOp::new(ctx, value, address_ty).get_operation();
            let address = emit_before(ctx, before, ptrtoint);
            let raw_ty: TypeHandle = MirPtrType::get_with_kind(
                ctx,
                pointee,
                is_mutable,
                address_space,
                MirPointerKind::from_raw_mutability(is_mutable),
            )
            .into();
            let to_raw = ox::mir_cast(
                ctx,
                address,
                raw_ty,
                MirCastKindAttr::PointerWithExposedProvenance,
            );
            // The address of static storage (a global, a thread-local, a
            // function) turned into a raw pointer: the dialect's
            // `StaticAddress` authority covers the integer-to-raw case.
            dm_ops::MirCastOp::new(to_raw)
                .set_attr_pointer_kind_authority(ctx, MirPointerKindAuthorityAttr::StaticAddress);
            let raw = emit_before(ctx, before, to_raw);
            if raw_ty == want {
                Some(raw)
            } else {
                let retype = ox::mir_cast(ctx, raw, want, MirCastKindAttr::PtrToPtr);
                Some(emit_before(ctx, before, retype))
            }
        }
        (Shape::MirPtr { .. }, Shape::LlvmPtr) => {
            let address_ty = address_ty(ctx);
            let expose = ox::mir_cast(
                ctx,
                value,
                address_ty,
                MirCastKindAttr::PointerExposeAddress,
            );
            let address = emit_before(ctx, before, expose);
            let inttoptr = llvm::ops::IntToPtrOp::new(ctx, address, want).get_operation();
            Some(emit_before(ctx, before, inttoptr))
        }
        (Shape::MirPtr { .. }, Shape::Int) => {
            let expose = ox::mir_cast(ctx, value, want, MirCastKindAttr::PointerExposeAddress);
            Some(emit_before(ctx, before, expose))
        }
        (Shape::LlvmPtr, Shape::Int) => {
            let ptrtoint = llvm::ops::PtrToIntOp::new(ctx, value, want).get_operation();
            Some(emit_before(ctx, before, ptrtoint))
        }
        (Shape::Int, Shape::Int) => {
            let retype = ox::mir_cast(ctx, value, want, MirCastKindAttr::IntToInt);
            Some(emit_before(ctx, before, retype))
        }
        _ => None,
    }
}

fn emit_before(ctx: &Context, before: Ptr<Operation>, op: Ptr<Operation>) -> Value {
    op.insert_before(ctx, before);
    op.deref(ctx).get_result(0)
}

// ----------------------------------------------------------------------------
// Type helpers
// ----------------------------------------------------------------------------

/// The integer an address is exposed as: a signless `i64` (what the llvm
/// dialect's casts and compares take).
fn address_ty(ctx: &mut Context) -> TypeHandle {
    IntegerType::get(ctx, 64, Signedness::Signless).into()
}

fn operand_ty(ctx: &Context, op: Ptr<Operation>, index: usize) -> TypeHandle {
    op.deref(ctx).get_operand(index).get_type(ctx)
}

fn result_ty(ctx: &Context, op: Ptr<Operation>, index: usize) -> TypeHandle {
    op.deref(ctx).get_result(index).get_type(ctx)
}

fn mir_pointee(ctx: &Context, ty: TypeHandle) -> Option<TypeHandle> {
    ty.deref(ctx)
        .downcast_ref::<MirPtrType>()
        .map(|p| p.pointee)
}

fn is_mir_ptr_ty(ctx: &Context, ty: TypeHandle) -> bool {
    ty.deref(ctx).is::<MirPtrType>()
}

fn is_llvm_ptr_ty(ctx: &Context, ty: TypeHandle) -> bool {
    ty.deref(ctx).is::<llvm::types::PointerType>()
}

fn is_pointer_ty(ctx: &Context, ty: TypeHandle) -> bool {
    is_mir_ptr_ty(ctx, ty) || is_llvm_ptr_ty(ctx, ty)
}

/// The `indices` of an `llvm.insert_value` / `llvm.extract_value`.
fn aggregate_indices(ctx: &Context, op: Ptr<Operation>) -> Vec<u32> {
    if let Some(insert) = Operation::get_op::<llvm::ops::InsertValueOp>(op, ctx) {
        return insert.indices(ctx);
    }
    if let Some(extract) = Operation::get_op::<llvm::ops::ExtractValueOp>(op, ctx) {
        return extract.indices(ctx);
    }
    Vec::new()
}

/// The type at `indices` inside the llvm aggregate type `ty`.
fn aggregate_field_type(ctx: &Context, ty: TypeHandle, indices: &[u32]) -> Option<TypeHandle> {
    let mut current = ty;
    for index in indices {
        let next = {
            let ty_ref = current.deref(ctx);
            if let Some(struct_ty) = ty_ref.downcast_ref::<llvm::types::StructType>() {
                if struct_ty.is_opaque() {
                    return None;
                }
                struct_ty.fields().nth(*index as usize)?
            } else if let Some(array_ty) = ty_ref.downcast_ref::<llvm::types::ArrayType>() {
                array_ty.elem_type()
            } else {
                return None;
            }
        };
        current = next;
    }
    Some(current)
}
