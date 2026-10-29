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

/// Kernel (`.entry`) parameters are never passed indirectly: the CUDA
/// launch ABI hands each parameter — scalar or aggregate — as ONE `.param`
/// whose bytes are the host value's C layout, and the NVPTX emitter loads
/// an aggregate's leaves with `ld.param` at their field offsets. The
/// exception is a slice-like parameter (`flattened`, see
/// `is_flattened_kernel_param`): its scalar fields are separate `.param`s,
/// because the host launch pushes them as separate arguments
/// (docs/KERNEL-ABI.md). Zero-sized types keep their no-presence rule so
/// parameter indices stay aligned with the host launch.
pub(super) fn kernel_param_abi(
    ctx: &Context,
    ty: TypeHandle,
    flattened: bool,
) -> Result<ArgAbi, String> {
    if let Ok(0) = crabbit_ty_size(ctx, ty) {
        return Ok(ArgAbi::Leaves(Vec::new()));
    }
    if flattened {
        let mut leaves = Vec::new();
        collect_kernel_param_leaves(ctx, ty, Vec::new(), &mut leaves)?;
        return Ok(ArgAbi::Leaves(leaves));
    }
    Ok(ArgAbi::Leaves(vec![(Vec::new(), ty)]))
}

/// The scalar fields of a flattened kernel parameter, in field order;
/// zero-sized fields (`PhantomData`, an index space without runtime
/// layout) contribute no `.param`, as NVPTX has no empty parameter.
fn collect_kernel_param_leaves(
    ctx: &Context,
    ty: TypeHandle,
    prefix: Vec<u32>,
    out: &mut Vec<(Vec<u32>, TypeHandle)>,
) -> Result<(), String> {
    if let Ok(0) = crabbit_ty_size(ctx, ty) {
        return Ok(());
    }
    if is_simple_abi_scalar_ty(ctx, ty) {
        out.push((prefix, ty));
        return Ok(());
    }
    let fields = match ty.deref(ctx).downcast_ref::<llvm::types::StructType>() {
        Some(struct_ty) if !struct_ty.is_opaque() => struct_ty.fields().collect::<Vec<_>>(),
        _ => {
            return Err(format!(
                "kernel parameter field of type {:?} cannot be flattened",
                ty.deref(ctx)
            ));
        }
    };
    for (index, field_ty) in fields.into_iter().enumerate() {
        let mut field_prefix = prefix.clone();
        field_prefix.push(index as u32);
        collect_kernel_param_leaves(ctx, field_ty, field_prefix, out)?;
    }
    Ok(())
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
    if ty_ref.is::<llvm::types::PointerType>()
        || ty_ref.is::<crate::dialects::dialect_mir::types::MirPtrType>()
    {
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
        ArgAbi::Indirect => lower_abi_call_arg_indirect(ctx, insert_block, value, out),
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

/// Where rustc's Rust ABI puts the result of a call to code rustc compiled
/// (std, dependencies), from the callee's `FnAbi`. crabbit's backend
/// returns an aggregate by its size (in x0/x1 up to 16 bytes, through x8
/// beyond); rustc returns a memory-layout aggregate wider than a pointer
/// through x8 and a scalar pair as one scalar per register, so a call into
/// rustc-compiled code adapts to whichever it is.
pub(super) enum RustReturn {
    /// In registers the way crabbit's backend reads them.
    Registers,
    /// Through the indirect-result pointer (x8).
    Indirect,
    /// A scalar pair whose second scalar sits at `second_offset` (below 8):
    /// it comes back in x1, not in the upper half of x0.
    Pair {
        second_offset: u64,
        second_size: u64,
    },
}

pub(super) fn rust_abi_return<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
) -> Option<RustReturn> {
    use rustc_target::callconv::PassMode;
    let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
    let fn_abi = tcx
        .fn_abi_of_instance(typing_env.as_query_input((instance, rustc_middle::ty::List::empty())))
        .ok()?;
    Some(match &fn_abi.ret.mode {
        PassMode::Indirect { .. } => RustReturn::Indirect,
        PassMode::Pair(..) => match fn_abi.ret.layout.backend_repr {
            rustc_abi::BackendRepr::ScalarPair {
                b: second,
                b_offset,
                ..
            } => {
                let second_offset = b_offset.bytes();
                if second_offset < 8 {
                    RustReturn::Pair {
                        second_offset,
                        second_size: second.size(&tcx).bytes(),
                    }
                } else {
                    RustReturn::Registers
                }
            }
            _ => RustReturn::Registers,
        },
        _ => RustReturn::Registers,
    })
}

/// How rustc's Rust ABI passes one argument of a call into code rustc
/// compiled, from the callee's `FnAbi`. crabbit classifies an argument by
/// its converted type (scalar leaves of aggregates up to 16 bytes in
/// registers, larger ones by reference); rustc instead passes a scalar
/// pair as two scalars, a memory-layout aggregate up to a pointer wide as
/// one integer, a wider one by reference, and a scalar-layout aggregate
/// (a newtype, a niche-encoded enum) as its scalar. A call into
/// rustc-compiled code passes each argument rustc's way.
pub(super) enum RustArg {
    /// No presence (a zero-sized argument).
    Ignore,
    /// By reference to a copy of the value.
    Indirect,
    /// As these scalars, read from the value's bytes at their offsets.
    Scalars(Vec<(u64, TypeHandle)>),
    /// As one 64-bit integer register holding the value's bytes.
    Cast,
}

/// The [RustArg]s of `instance`'s parameters, or `None` when its `FnAbi`
/// is unavailable or has a shape crabbit's call lowering does not mirror
/// (then crabbit's own classification is used, as for its own functions).
pub(super) fn rust_abi_args<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    instance: Instance<'tcx>,
) -> Option<Vec<RustArg>> {
    use rustc_target::callconv::PassMode;
    let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
    let fn_abi = tcx
        .fn_abi_of_instance(typing_env.as_query_input((instance, rustc_middle::ty::List::empty())))
        .ok()?;
    if fn_abi.c_variadic {
        return None;
    }
    fn_abi
        .args
        .iter()
        .map(|arg| {
            Some(match &arg.mode {
                PassMode::Ignore => RustArg::Ignore,
                PassMode::Indirect {
                    meta_attrs: None,
                    on_stack: false,
                    ..
                } => RustArg::Indirect,
                PassMode::Indirect { .. } => return None,
                PassMode::Cast { .. } if arg.layout.size.bytes() <= 8 => RustArg::Cast,
                PassMode::Cast { .. } => return None,
                PassMode::Direct(_) => match arg.layout.backend_repr {
                    rustc_abi::BackendRepr::Scalar(scalar) => {
                        RustArg::Scalars(vec![(0, scalar_abi_ty(tcx, ctx, scalar)?)])
                    }
                    _ => return None,
                },
                PassMode::Pair(..) => match arg.layout.backend_repr {
                    rustc_abi::BackendRepr::ScalarPair { a, b, b_offset } => {
                        RustArg::Scalars(vec![
                            (0, scalar_abi_ty(tcx, ctx, a)?),
                            (b_offset.bytes(), scalar_abi_ty(tcx, ctx, b)?),
                        ])
                    }
                    _ => return None,
                },
            })
        })
        .collect()
}

/// The register type of one ABI scalar.
fn scalar_abi_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    scalar: rustc_abi::Scalar,
) -> Option<TypeHandle> {
    use rustc_abi::{Float, Primitive};
    Some(match scalar.primitive() {
        Primitive::Int(int, _) => {
            IntegerType::get(ctx, int.size().bits() as u32, Signedness::Signless).into()
        }
        Primitive::Float(Float::F32) => FP32Type::get(ctx).into(),
        Primitive::Float(Float::F64) => FP64Type::get(ctx).into(),
        Primitive::Float(_) => return None,
        Primitive::Pointer(_) => {
            let _ = tcx;
            ptr_ty(ctx)
        }
    })
}

/// Lower one argument of a call into rustc-compiled code the way rustc
/// passes it ([RustArg]). The value is crabbit's representation, which
/// keeps rustc's in-memory layout, so its bytes in a stack slot are the
/// bytes rustc's caller would have.
pub(super) fn lower_rust_abi_call_arg(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    value: Value,
    abi: &RustArg,
    out: &mut Vec<Value>,
) -> Result<(), String> {
    let value_ty = value.get_type(ctx);
    let is_aggregate = {
        let ty_ref = value_ty.deref(ctx);
        ty_ref.is::<llvm::types::StructType>() || ty_ref.is::<llvm::types::ArrayType>()
    };
    match abi {
        RustArg::Ignore => Ok(()),
        RustArg::Scalars(scalars) if !is_aggregate && scalars.len() == 1 => {
            // Already the scalar (an integer, float, pointer or bool).
            out.push(value);
            Ok(())
        }
        RustArg::Indirect => lower_abi_call_arg_indirect(ctx, insert_block, value, out),
        RustArg::Cast | RustArg::Scalars(_) => {
            // Through memory: store the value into a slot of at least one
            // word and read the register values back from its bytes.
            let word_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
            let words = crabbit_ty_size(ctx, value_ty)?.div_ceil(8).max(1);
            let slot_ty: TypeHandle = llvm::types::ArrayType::get(ctx, word_ty, words).into();
            let slot = mir_dialect::ops::AllocaOp::new(ctx, slot_ty);
            slot.get_operation().insert_at_back(insert_block, ctx);
            let slot = slot.get_result(ctx);
            let store = mir_dialect::ops::StoreOp::new(ctx, value, slot);
            store.get_operation().insert_at_back(insert_block, ctx);
            let scalars = match abi {
                RustArg::Scalars(scalars) => scalars.clone(),
                _ => vec![(0, IntegerType::get(ctx, 64, Signedness::Signless).into())],
            };
            for (offset, ty) in scalars {
                let addr = if offset == 0 {
                    slot
                } else {
                    ptr_offset_const(ctx, insert_block, slot, offset)?
                };
                let load = mir_dialect::ops::LoadOp::new(ctx, addr, ty);
                load.get_operation().insert_at_back(insert_block, ctx);
                out.push(load.get_result(ctx));
            }
            Ok(())
        }
    }
}

fn lower_abi_call_arg_indirect(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    value: Value,
    out: &mut Vec<Value>,
) -> Result<(), String> {
    let value_ty = value.get_type(ctx);
    let slot = mir_dialect::ops::AllocaOp::new(ctx, value_ty);
    slot.get_operation().insert_at_back(insert_block, ctx);
    let store = mir_dialect::ops::StoreOp::new(ctx, value, slot.get_result(ctx));
    store.get_operation().insert_at_back(insert_block, ctx);
    out.push(slot.get_result(ctx));
    Ok(())
}
