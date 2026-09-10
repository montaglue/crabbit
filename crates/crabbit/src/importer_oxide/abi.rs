use super::*;

/// How one MIR parameter maps to ABI-level arguments.
pub(super) enum ParamAbi {
    Single(ArgAbi),
    /// "rust-call" spread argument: each tuple element lowered separately.
    Spread(Vec<(TypeHandle, ArgAbi)>),
}

/// Rebuild one Rust-level value from its incoming ABI arguments.
pub(super) fn reassemble_abi_arg(
    ctx: &mut Context,
    entry: Ptr<BasicBlock>,
    insert_block: Ptr<BasicBlock>,
    value_ty: TypeHandle,
    abi: ArgAbi,
    block_arg_idx: &mut usize,
) -> Value {
    match abi {
        ArgAbi::Indirect => {
            let ptr = entry.deref(ctx).get_argument(*block_arg_idx);
            *block_arg_idx += 1;
            let load = mir_dialect::ops::LoadOp::new(ctx, ptr, value_ty);
            load.get_operation().insert_at_back(insert_block, ctx);
            load.get_result(ctx)
        }
        ArgAbi::Leaves(group) if group.len() == 1 && group[0].0.is_empty() => {
            let arg = entry.deref(ctx).get_argument(*block_arg_idx);
            *block_arg_idx += 1;
            arg
        }
        ArgAbi::Leaves(group) => {
            let undef = mir_dialect::ops::UndefOp::new(ctx, value_ty);
            undef.get_operation().insert_at_back(insert_block, ctx);
            let mut current = undef.get_result(ctx);
            for (indices, _) in group {
                let field = entry.deref(ctx).get_argument(*block_arg_idx);
                *block_arg_idx += 1;
                let insert = mir_dialect::ops::InsertValueOp::new(ctx, field, current, indices);
                insert.get_operation().insert_at_back(insert_block, ctx);
                current = insert.get_result(ctx);
            }
            current
        }
    }
}

/// How one Rust-level argument maps to ABI-level arguments.
pub(super) enum ArgAbi {
    /// Flattened into scalar leaves passed directly.
    Leaves(Vec<(Vec<u32>, TypeHandle)>),
    /// Copied to a stack temporary and passed by pointer (aggregates larger
    /// than two registers or containing arrays), matching the AArch64 rule
    /// for composites bigger than 16 bytes.
    Indirect,
}

pub(super) fn arg_abi_for_ty(ctx: &Context, ty: TypeHandle) -> Result<ArgAbi, String> {
    // Zero-sized types (`()`, `!`, captureless closures) have no ABI
    // presence: no signature input, no call argument, no entry-block
    // argument. This must stay symmetric across import_function signatures,
    // entry reassembly, and lower_abi_call_arg — an asymmetry shifts every
    // later parameter one register over.
    if let Ok(0) = crabbit_ty_size(ctx, ty) {
        return Ok(ArgAbi::Leaves(Vec::new()));
    }
    let ty_ref = ty.deref(ctx);
    let is_aggregate =
        ty_ref.is::<llvm::types::StructType>() || ty_ref.is::<llvm::types::ArrayType>();
    drop(ty_ref);
    if is_aggregate {
        if crabbit_ty_size(ctx, ty)? > 16 {
            return Ok(ArgAbi::Indirect);
        }
        if let Ok(leaves) = simple_abi_leaves_for_ty(ctx, ty) {
            return Ok(ArgAbi::Leaves(leaves));
        }
        return Ok(ArgAbi::Indirect);
    }
    Ok(ArgAbi::Leaves(simple_abi_leaves_for_ty(ctx, ty)?))
}

/// Byte size of a converted crabbit type, mirroring the AArch64 lowering's
/// `stack_size_of` alignment rules.
pub(super) fn crabbit_ty_size(ctx: &Context, ty: TypeHandle) -> Result<u64, String> {
    let ty_ref = ty.deref(ctx);
    if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
        return Ok((int_ty.width() as u64).div_ceil(8).max(1));
    }
    if ty_ref.is::<llvm::types::PointerType>() || ty_ref.is::<crate::dialects::dialect_mir::types::MirPtrType>() {
        return Ok(8);
    }
    if ty_ref.is::<FP32Type>() {
        return Ok(4);
    }
    if ty_ref.is::<FP64Type>() {
        return Ok(8);
    }
    if ty_ref.is::<UnitType>() {
        return Ok(0);
    }
    if let Some(array_ty) = ty_ref.downcast_ref::<llvm::types::ArrayType>() {
        let elem = array_ty.elem_type();
        let len = array_ty.size();
        drop(ty_ref);
        let stride = align_to(crabbit_ty_size(ctx, elem)?, crabbit_ty_align(ctx, elem)?);
        return Ok(stride * len);
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<llvm::types::StructType>() {
        if struct_ty.is_opaque() {
            return Err("opaque struct in ABI sizing".to_string());
        }
        let fields: Vec<_> = struct_ty.fields().collect();
        drop(ty_ref);
        let mut size = 0u64;
        let mut align = 1u64;
        for field in fields {
            let field_size = crabbit_ty_size(ctx, field)?;
            if field_size == 0 {
                continue;
            }
            let field_align = crabbit_ty_align(ctx, field)?;
            size = align_to(size, field_align) + field_size;
            align = align.max(field_align);
        }
        return Ok(align_to(size, align));
    }
    Err(format!("unsupported ABI type: {:?}", &*ty_ref))
}

pub(super) fn crabbit_ty_align(ctx: &Context, ty: TypeHandle) -> Result<u64, String> {
    let ty_ref = ty.deref(ctx);
    if ty_ref.is::<UnitType>() {
        return Ok(1);
    }
    if let Some(array_ty) = ty_ref.downcast_ref::<llvm::types::ArrayType>() {
        let elem = array_ty.elem_type();
        drop(ty_ref);
        return crabbit_ty_align(ctx, elem);
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<llvm::types::StructType>() {
        if struct_ty.is_opaque() {
            return Err("opaque struct in ABI sizing".to_string());
        }
        let fields: Vec<_> = struct_ty.fields().collect();
        drop(ty_ref);
        let mut align = 1u64;
        for field in fields {
            if crabbit_ty_size(ctx, field)? == 0 {
                continue;
            }
            align = align.max(crabbit_ty_align(ctx, field)?);
        }
        return Ok(align);
    }
    drop(ty_ref);
    Ok(crabbit_ty_size(ctx, ty)?.clamp(1, 8))
}

pub(super) fn lower_abi_call_arg(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    value: Value,
    out: &mut Vec<Value>,
) -> Result<(), String> {
    let value_ty = value.get_type(ctx);
    match arg_abi_for_ty(ctx, value_ty)? {
        ArgAbi::Indirect => {
            let slot = mir_dialect::ops::AllocaOp::new(ctx, value_ty);
            slot.get_operation().insert_at_back(insert_block, ctx);
            let store = mir_dialect::ops::StoreOp::new(ctx, value, slot.get_result(ctx));
            store.get_operation().insert_at_back(insert_block, ctx);
            out.push(slot.get_result(ctx));
            Ok(())
        }
        ArgAbi::Leaves(leaves) => {
            if leaves.len() == 1 && leaves[0].0.is_empty() {
                out.push(value);
                return Ok(());
            }
            for (indices, result_ty) in leaves {
                let field = mir_dialect::ops::ExtractValueOp::new(ctx, value, indices, result_ty);
                field.get_operation().insert_at_back(insert_block, ctx);
                out.push(field.get_result(ctx));
            }
            Ok(())
        }
    }
}
