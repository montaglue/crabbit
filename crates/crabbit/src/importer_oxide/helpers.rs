use super::*;

pub(super) fn local_slot<'tcx>(state: &FunctionImportState<'tcx>, local: Local) -> Result<Value, String> {
    local_slot_opt(state, local)?.ok_or_else(|| format!("missing slot for MIR local {local:?}"))
}

pub(super) fn local_slot_opt<'tcx>(
    state: &FunctionImportState<'tcx>,
    local: Local,
) -> Result<Option<Value>, String> {
    state
        .local_slots
        .get(local.index())
        .map(|slot| slot.as_ref().map(|(slot, _)| *slot))
        .ok_or_else(|| format!("unknown MIR local {local:?}"))
}

pub(super) fn place_slot_opt<'tcx>(
    state: &FunctionImportState<'tcx>,
    place: &Place<'tcx>,
) -> Result<Option<Value>, String> {
    if !place.projection.is_empty() {
        return Err(format!(
            "unsupported MIR place projection: {:?}",
            place.projection
        ));
    }
    local_slot_opt(state, place.local)
}

pub(super) fn block_for<'tcx>(
    state: &FunctionImportState<'tcx>,
    block: RustBasicBlock,
) -> Result<Ptr<BasicBlock>, String> {
    state
        .blocks
        .get(block.index())
        .copied()
        .ok_or_else(|| format!("missing crabbit block for MIR block {block:?}"))
}

pub(super) fn usize_ty(ctx: &mut Context) -> TypedHandle<IntegerType> {
    IntegerType::get(ctx, 64, Signedness::Unsigned)
}

pub(super) fn llvm_ptr_ty(ctx: &mut Context) -> TypeHandle {
    llvm::types::PointerType::get(ctx, 0).into()
}

pub(super) fn str_ref_ty(ctx: &mut Context) -> TypeHandle {
    let ptr = llvm_ptr_ty(ctx);
    let usize: TypeHandle = usize_ty(ctx).into();
    llvm::types::StructType::get_unnamed(ctx, vec![ptr, usize]).into()
}

pub(super) fn slice_ref_ty(ctx: &mut Context, _elem: TypeHandle) -> TypeHandle {
    let ptr = llvm_ptr_ty(ctx);
    let usize: TypeHandle = usize_ty(ctx).into();
    llvm::types::StructType::get_unnamed(ctx, vec![ptr, usize]).into()
}

pub(super) fn trait_object_ref_ty(ctx: &mut Context) -> TypeHandle {
    let ptr = llvm_ptr_ty(ctx);
    llvm::types::StructType::get_unnamed(ctx, vec![ptr, ptr]).into()
}

pub(super) fn fmt_arguments_ty(ctx: &mut Context) -> TypeHandle {
    let ptr = llvm_ptr_ty(ctx);
    llvm::types::StructType::get_unnamed(ctx, vec![ptr, ptr]).into()
}

pub(super) fn device_slice_ty(ctx: &mut Context, elem: TypeHandle, mutable: bool) -> TypeHandle {
    let _ = (elem, mutable);
    let ptr: TypeHandle = llvm_ptr_ty(ctx);
    let usize: TypeHandle = usize_ty(ctx).into();
    llvm::types::StructType::get_unnamed(ctx, vec![ptr, usize]).into()
}

/// Memory-order permutation of a struct-like type's fields per the real
/// rustc layout: element `i` is the source index of the field at memory
/// position `i`. Identity when no layout is available.
pub(super) fn struct_memory_order<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>, field_count: usize) -> Vec<usize> {
    let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
    let Ok(layout) = tcx.layout_of(typing_env.as_query_input(ty)) else {
        return (0..field_count).collect();
    };
    match &layout.fields {
        rustc_abi::FieldsShape::Arbitrary {
            in_memory_order, ..
        } if in_memory_order.len() == field_count => in_memory_order
            .iter()
            .map(|source| source.as_usize())
            .collect(),
        _ => (0..field_count).collect(),
    }
}

/// Position of a source-order field within the converted (memory-ordered)
/// struct type.
pub(super) fn field_memory_position<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    field_count: usize,
    source_idx: usize,
) -> u32 {
    let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
    let Ok(layout) = tcx.layout_of(typing_env.as_query_input(ty)) else {
        return source_idx as u32;
    };
    match &layout.fields {
        rustc_abi::FieldsShape::Arbitrary {
            in_memory_order, ..
        } if in_memory_order.len() == field_count => in_memory_order
            .iter()
            .position(|source| source.as_usize() == source_idx)
            .map(|position| position as u32)
            .unwrap_or(source_idx as u32),
        _ => source_idx as u32,
    }
}

/// The struct-like source fields of a type, if it is field-reordered by the
/// converted representation (structs, tuples, closures — not enums).
pub(super) fn struct_like_source_fields<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<Vec<Ty<'tcx>>> {
    match runtime_ty(ty).kind() {
        rustc_middle::ty::TyKind::Tuple(fields) => Some(fields.iter().collect()),
        rustc_middle::ty::TyKind::Closure(_, args) => {
            Some(args.as_closure().upvar_tys().iter().collect())
        }
        rustc_middle::ty::TyKind::Adt(adt_def, args)
            if adt_def.is_struct() && !is_fmt_rt_argument_type(tcx, adt_def.did()) =>
        {
            Some(
                adt_def
                    .non_enum_variant()
                    .fields
                    .iter()
                    .map(|field| field.ty(tcx, args))
                    .collect(),
            )
        }
        _ => None,
    }
}

pub(super) fn runtime_ty<'tcx>(mut ty: Ty<'tcx>) -> Ty<'tcx> {
    while let rustc_middle::ty::TyKind::Pat(base, _) = ty.kind() {
        ty = *base;
    }
    ty
}

/// Resolve associated-type projections (e.g. `AtomicPrimitive::Storage` inside
/// std internals) that reach type conversion unnormalized via field types.
pub(super) fn normalize_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
    if !ty.has_aliases() {
        return ty;
    }
    let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
    tcx.try_normalize_erasing_regions(typing_env, ty)
        .unwrap_or(ty)
}

pub(super) fn is_simple_abi_scalar_ty(ctx: &Context, ty: TypeHandle) -> bool {
    let ty_ref = ty.deref(ctx);
    ty_ref.is::<IntegerType>()
        || ty_ref.is::<FP32Type>()
        || ty_ref.is::<FP64Type>()
        || ty_ref.is::<llvm::types::PointerType>()
        || ty_ref.is::<crate::dialects::dialect_mir::types::MirPtrType>()
}

pub(super) fn is_str_ref_ty<'tcx>(ty: Ty<'tcx>) -> bool {
    let ty = runtime_ty(ty);
    matches!(
        ty.kind(),
        rustc_middle::ty::TyKind::Ref(_, inner, _)
            if matches!(inner.kind(), rustc_middle::ty::TyKind::Str)
    )
}

pub(super) fn byte_array_ref_len<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<u64> {
    let ty = runtime_ty(ty);
    let rustc_middle::ty::TyKind::Ref(_, inner, _) = ty.kind() else {
        return None;
    };
    let rustc_middle::ty::TyKind::Array(elem, len) = runtime_ty(*inner).kind() else {
        return None;
    };
    if matches!(
        runtime_ty(*elem).kind(),
        rustc_middle::ty::TyKind::Uint(rustc_middle::ty::UintTy::U8)
    ) {
        array_len(tcx, *len).ok()
    } else {
        None
    }
}

/// For a pointer/reference pointee, the unsized tail type (`[T]`, `str`, or
/// `dyn Trait`) that determines the fat-pointer metadata, or `None` when the
/// pointee is sized (thin pointer). Peels ADT struct tails, so e.g.
/// `&PolymorphicIter<[T]>` (core's array iterator internals) is fat with a
/// slice-length metadata.
pub(super) fn pointee_unsized_tail<'tcx>(tcx: TyCtxt<'tcx>, pointee: Ty<'tcx>) -> Option<Ty<'tcx>> {
    let pointee = runtime_ty(pointee);
    match pointee.kind() {
        rustc_middle::ty::TyKind::Slice(_)
        | rustc_middle::ty::TyKind::Str
        | rustc_middle::ty::TyKind::Dynamic(_, _) => Some(pointee),
        _ => {
            let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
            if pointee.is_sized(tcx, typing_env) {
                return None;
            }
            let tail = tcx.struct_tail_for_codegen(pointee, typing_env);
            match runtime_ty(tail).kind() {
                rustc_middle::ty::TyKind::Slice(_)
                | rustc_middle::ty::TyKind::Str
                | rustc_middle::ty::TyKind::Dynamic(_, _) => Some(runtime_ty(tail)),
                _ => None,
            }
        }
    }
}

pub(super) fn contains_maybe_uninit_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
    let ty = runtime_ty(ty);
    if format!("{ty:?}").contains("MaybeUninit") {
        return true;
    }
    match ty.kind() {
        rustc_middle::ty::TyKind::Adt(adt_def, args) => {
            tcx.def_path_str(adt_def.did())
                .contains("mem::maybe_uninit::MaybeUninit")
                || args.iter().any(|arg| {
                    arg.as_type()
                        .is_some_and(|ty| contains_maybe_uninit_ty(tcx, ty))
                })
        }
        rustc_middle::ty::TyKind::Array(elem, _)
        | rustc_middle::ty::TyKind::Slice(elem)
        | rustc_middle::ty::TyKind::Ref(_, elem, _)
        | rustc_middle::ty::TyKind::RawPtr(elem, _) => contains_maybe_uninit_ty(tcx, *elem),
        rustc_middle::ty::TyKind::Tuple(fields) => fields
            .iter()
            .any(|field| contains_maybe_uninit_ty(tcx, field)),
        _ => false,
    }
}

pub(super) fn is_maybe_uninit_u8_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
    let ty = runtime_ty(ty);
    let rustc_middle::ty::TyKind::Adt(adt_def, args) = ty.kind() else {
        return false;
    };
    if !tcx.def_path_str(adt_def.did()).contains("MaybeUninit")
        && !format!("{ty:?}").contains("MaybeUninit")
    {
        return false;
    }
    args.iter().any(|arg| {
        arg.as_type().is_some_and(|ty| {
            matches!(
                runtime_ty(ty).kind(),
                rustc_middle::ty::TyKind::Uint(rustc_middle::ty::UintTy::U8)
            )
        })
    })
}

pub(super) fn is_enum_ty<'tcx>(ty: Ty<'tcx>) -> bool {
    matches!(
        runtime_ty(ty).kind(),
        rustc_middle::ty::TyKind::Adt(adt_def, _) if adt_def.is_enum()
    )
}
