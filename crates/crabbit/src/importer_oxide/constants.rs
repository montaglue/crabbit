use super::*;

pub(super) fn import_constant<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    constant: &ConstOperand<'tcx>,
) -> Result<Value, String> {
    let const_ = mono_const(tcx, state, constant.const_);
    let typing_env = state.instance.map_or_else(
        || body.typing_env(tcx),
        |_| rustc_middle::ty::TypingEnv::fully_monomorphized(),
    );
    if is_str_ref_ty(const_.ty()) {
        return import_str_constant(
            tcx,
            ctx,
            state.module_body,
            insert_block,
            typing_env,
            body,
            constant,
            const_,
        );
    }
    if let Some(len) = byte_array_ref_len(tcx, const_.ty())
        && let Some(bytes) = literal_byte_string_constant(tcx, constant)
            .or_else(|| evaluated_byte_string_constant(tcx, typing_env, constant.span, const_, len))
    {
        let ty = convert_ty(tcx, ctx, const_.ty())?;
        let symbol = declare_anonymous_byte_global(ctx, state.module_body, &bytes);
        let op = mir_dialect::ops::AddressOfOp::new(ctx, symbol, ty);
        op.get_operation().insert_at_back(insert_block, ctx);
        return Ok(op.get_result(ctx));
    }

    let ty = convert_immediate_ty(tcx, ctx, const_.ty())?;
    if let Some(def_id) = constant.check_static_ptr(tcx) {
        let mut legaliser = Legaliser::default();
        let symbol = legaliser.legalise(tcx.symbol_name(Instance::mono(tcx, def_id)).name);
        declare_static_global(tcx, ctx, state.module_body, symbol.clone(), def_id)?;
        let op = mir_dialect::ops::AddressOfOp::new(ctx, symbol, ty);
        op.get_operation().insert_at_back(insert_block, ctx);
        return Ok(op.get_result(ctx));
    }
    if layout_size_of_ty(tcx, const_.ty())? == 0 {
        let op = mir_dialect::ops::UndefOp::new(ctx, ty);
        op.get_operation().insert_at_back(insert_block, ctx);
        return Ok(op.get_result(ctx));
    }
    if let Some(value) = import_initialized_maybe_uninit_u8_constant(
        tcx,
        ctx,
        insert_block,
        typing_env,
        constant.span,
        const_,
        ty,
    )? {
        return Ok(value);
    }
    if contains_maybe_uninit_ty(tcx, const_.ty()) {
        let op = mir_dialect::ops::UndefOp::new(ctx, ty);
        op.get_operation().insert_at_back(insert_block, ctx);
        return Ok(op.get_result(ctx));
    }
    if is_enum_ty(const_.ty()) {
        return import_enum_constant(
            tcx,
            ctx,
            state,
            insert_block,
            typing_env,
            constant.span,
            const_,
            ty,
        );
    }
    let Some(bits) = const_.try_eval_bits(tcx, typing_env) else {
        if let Some(value) = import_memory_constant(
            tcx,
            ctx,
            state,
            insert_block,
            typing_env,
            constant.span,
            const_,
            ty,
        )? {
            return Ok(value);
        }
        if let Some(value) = import_pointer_to_plain_data_constant(
            tcx,
            ctx,
            state,
            insert_block,
            typing_env,
            constant.span,
            const_,
            ty,
        )? {
            return Ok(value);
        }
        if let Some(value) = import_pointer_constant(
            tcx,
            ctx,
            state,
            insert_block,
            typing_env,
            constant.span,
            const_,
            ty,
        )? {
            return Ok(value);
        }
        return Err(unsupported_constant_reason(
            tcx,
            typing_env,
            constant.span,
            const_,
        ));
    };
    match constant_from_bits(ctx, ty, bits) {
        Ok(op) => {
            op.get_operation().insert_at_back(insert_block, ctx);
            Ok(op.get_result(ctx))
        }
        Err(_) => {
            // Newtype wrappers around a scalar (Cap, NonZero inner types,
            // ...) evaluate to scalar bits but convert to a struct type.
            if let Some(value) = scalar_constant_in_aggregate(ctx, insert_block, ty, bits)? {
                return Ok(value);
            }
            let op = mir_dialect::ops::UndefOp::new(ctx, ty);
            op.get_operation().insert_at_back(insert_block, ctx);
            Ok(op.get_result(ctx))
        }
    }
}

/// Import a constant whose evaluated value lives in a memory allocation
/// without pointer provenance, by emitting an anonymous byte global and
/// loading the converted value type from it.
#[allow(clippy::too_many_arguments)]
pub(super) fn import_memory_constant<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    typing_env: rustc_middle::ty::TypingEnv<'tcx>,
    span: rustc_span::Span,
    constant: rustc_mir::Const<'tcx>,
    ty: TypeHandle,
) -> Result<Option<Value>, String> {
    let Ok(value) = constant.eval(tcx, typing_env, span) else {
        return Ok(None);
    };
    let rustc_mir::ConstValue::Indirect { alloc_id, offset } = value else {
        return Ok(None);
    };
    let rustc_mir::interpret::GlobalAlloc::Memory(alloc) = tcx.global_alloc(alloc_id) else {
        return Ok(None);
    };
    let alloc = alloc.inner();
    if !alloc.provenance().ptrs().is_empty() {
        return Ok(None);
    }
    // Uninitialized ranges (e.g. enum padding) read as whatever the raw
    // buffer holds, matching what rustc's own codegen emits for globals.
    let bytes = alloc
        .inspect_with_uninit_and_ptr_outside_interpreter(offset.bytes() as usize..alloc.len())
        .to_vec();
    let symbol = declare_anonymous_byte_global(ctx, state.module_body, &bytes);
    let ptr_ty = llvm_ptr_ty(ctx);
    let addr = mir_dialect::ops::AddressOfOp::new(ctx, symbol, ptr_ty);
    addr.get_operation().insert_at_back(insert_block, ctx);
    let load = mir_dialect::ops::LoadOp::new(ctx, addr.get_result(ctx), ty);
    load.get_operation().insert_at_back(insert_block, ctx);
    Ok(Some(load.get_result(ctx)))
}

/// Explain why a constant could not be imported. Pointer-carrying constants
/// are lowered through `ll.data` globals with data relocations, so a constant
/// that still fails here could not be evaluated or has an unhandled value
/// shape; report the MIR dump.
pub(super) fn unsupported_constant_reason<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: rustc_middle::ty::TypingEnv<'tcx>,
    span: rustc_span::Span,
    constant: rustc_mir::Const<'tcx>,
) -> String {
    match constant.eval(tcx, typing_env, span) {
        Ok(value) => format!("unsupported MIR constant value {value:?}: {constant:?}"),
        Err(_) => format!("unsupported MIR constant (evaluation failed): {constant:?}"),
    }
}

/// Import an evaluated constant that is a thin pointer/reference into a
/// read-only allocation without pointer provenance (e.g. the promoted
/// `&ControlFlow<(), ()>` constants inside `Iterator::all`/`any`, or promoted
/// `&[T; 0]` empty-array refs), or a fat `&[T]` slice ref whose data carries
/// no provenance. The pointee bytes become an anonymous byte global and the
/// constant becomes its address (plus the length for slice refs).
#[allow(clippy::too_many_arguments)]
pub(super) fn import_pointer_to_plain_data_constant<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    typing_env: rustc_middle::ty::TypingEnv<'tcx>,
    span: rustc_span::Span,
    constant: rustc_mir::Const<'tcx>,
    ty: TypeHandle,
) -> Result<Option<Value>, String> {
    let pointee = match runtime_ty(constant.ty()).kind() {
        rustc_middle::ty::TyKind::Ref(_, inner, _)
        | rustc_middle::ty::TyKind::RawPtr(inner, _) => runtime_ty(*inner),
        _ => return Ok(None),
    };
    let Ok(value) = constant.eval(tcx, typing_env, span) else {
        return Ok(None);
    };
    match value {
        rustc_mir::ConstValue::Scalar(rustc_mir::interpret::Scalar::Ptr(ptr, _)) => {
            let (provenance, offset) = ptr.prov_and_relative_offset();
            let rustc_mir::interpret::GlobalAlloc::Memory(alloc) =
                tcx.global_alloc(provenance.alloc_id())
            else {
                return Ok(None);
            };
            let alloc = alloc.inner();
            if !alloc.provenance().ptrs().is_empty() {
                return Ok(None);
            }
            let bytes = alloc
                .inspect_with_uninit_and_ptr_outside_interpreter(
                    offset.bytes() as usize..alloc.len(),
                )
                .to_vec();
            let symbol = declare_anonymous_byte_global(ctx, state.module_body, &bytes);
            let op = mir_dialect::ops::AddressOfOp::new(ctx, symbol, ty);
            op.get_operation().insert_at_back(insert_block, ctx);
            Ok(Some(op.get_result(ctx)))
        }
        rustc_mir::ConstValue::Slice { alloc_id, meta } => {
            let rustc_middle::ty::TyKind::Slice(elem) = pointee.kind() else {
                return Ok(None);
            };
            let elem_size = layout_size_of_ty(tcx, *elem)?;
            let Some(bytes) =
                allocation_bytes(tcx, alloc_id, Size::ZERO, meta.saturating_mul(elem_size))
            else {
                return Ok(None);
            };
            let symbol = declare_anonymous_byte_global(ctx, state.module_body, &bytes);
            let ptr_ty = llvm_ptr_ty(ctx);
            let data = mir_dialect::ops::AddressOfOp::new(ctx, symbol, ptr_ty);
            data.get_operation().insert_at_back(insert_block, ctx);
            let usize_ty: TypeHandle = usize_ty(ctx).into();
            let len = integer_constant(ctx, usize_ty, meta as u128)?;
            len.get_operation().insert_at_back(insert_block, ctx);
            let undef = mir_dialect::ops::UndefOp::new(ctx, ty);
            undef.get_operation().insert_at_back(insert_block, ctx);
            let with_ptr = mir_dialect::ops::InsertValueOp::new(
                ctx,
                data.get_result(ctx),
                undef.get_result(ctx),
                vec![0],
            );
            with_ptr.get_operation().insert_at_back(insert_block, ctx);
            let with_len = mir_dialect::ops::InsertValueOp::new(
                ctx,
                len.get_result(ctx),
                with_ptr.get_result(ctx),
                vec![1],
            );
            with_len.get_operation().insert_at_back(insert_block, ctx);
            Ok(Some(with_len.get_result(ctx)))
        }
        _ => Ok(None),
    }
}

/// Import an evaluated constant whose value carries pointer provenance, by
/// materializing the backing allocations as [ll.data](pliron_ll::ll::DataAttr)
/// globals (with data relocations for the pointer slots) and taking their
/// addresses:
///
/// - a `Scalar::Ptr` becomes `llvm.addressof` of the target's global (plus a
///   byte offset for interior pointers),
/// - a `Slice` becomes the `{ptr, len}` fat-pointer pair over the data global,
/// - an `Indirect` (by-ref) value is loaded back out of its relocated global,
///   so pointer slots inside it hold real addresses at runtime.
#[allow(clippy::too_many_arguments)]
pub(super) fn import_pointer_constant<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    typing_env: rustc_middle::ty::TypingEnv<'tcx>,
    span: rustc_span::Span,
    constant: rustc_mir::Const<'tcx>,
    ty: TypeHandle,
) -> Result<Option<Value>, String> {
    let Ok(value) = constant.eval(tcx, typing_env, span) else {
        return Ok(None);
    };
    match value {
        rustc_mir::ConstValue::Scalar(rustc_mir::interpret::Scalar::Ptr(ptr, _)) => {
            let (provenance, offset) = ptr.prov_and_relative_offset();
            // A type-id "pointer" is not an address: its offset bytes are a
            // hash segment, materialized as a plain integer.
            if matches!(
                tcx.global_alloc(provenance.alloc_id()),
                rustc_mir::interpret::GlobalAlloc::TypeId { .. }
            ) {
                let usize_ty: TypeHandle = usize_ty(ctx).into();
                let op = integer_constant(ctx, usize_ty, offset.bytes() as u128)?;
                op.get_operation().insert_at_back(insert_block, ctx);
                return Ok(Some(cast_value_to_type(
                    ctx,
                    insert_block,
                    op.get_result(ctx),
                    ty,
                )));
            }
            let symbol =
                data_global_for_alloc(tcx, ctx, state.module_body, provenance.alloc_id())?;
            let ptr_ty = llvm_ptr_ty(ctx);
            let addr = mir_dialect::ops::AddressOfOp::new(ctx, symbol, ptr_ty);
            addr.get_operation().insert_at_back(insert_block, ctx);
            let mut value = addr.get_result(ctx);
            if offset.bytes() != 0 {
                value = ptr_offset_const(ctx, insert_block, value, offset.bytes())?;
            }
            Ok(Some(cast_value_to_type(ctx, insert_block, value, ty)))
        }
        rustc_mir::ConstValue::Slice { alloc_id, meta } => {
            let symbol = data_global_for_alloc(tcx, ctx, state.module_body, alloc_id)?;
            let ptr_ty = llvm_ptr_ty(ctx);
            let data = mir_dialect::ops::AddressOfOp::new(ctx, symbol, ptr_ty);
            data.get_operation().insert_at_back(insert_block, ctx);
            let usize_ty: TypeHandle = usize_ty(ctx).into();
            let len = integer_constant(ctx, usize_ty, meta as u128)?;
            len.get_operation().insert_at_back(insert_block, ctx);
            let undef = mir_dialect::ops::UndefOp::new(ctx, ty);
            undef.get_operation().insert_at_back(insert_block, ctx);
            let with_ptr = mir_dialect::ops::InsertValueOp::new(
                ctx,
                data.get_result(ctx),
                undef.get_result(ctx),
                vec![0],
            );
            with_ptr.get_operation().insert_at_back(insert_block, ctx);
            let with_len = mir_dialect::ops::InsertValueOp::new(
                ctx,
                len.get_result(ctx),
                with_ptr.get_result(ctx),
                vec![1],
            );
            with_len.get_operation().insert_at_back(insert_block, ctx);
            Ok(Some(with_len.get_result(ctx)))
        }
        rustc_mir::ConstValue::Indirect { alloc_id, offset } => {
            let symbol = data_global_for_alloc(tcx, ctx, state.module_body, alloc_id)?;
            let ptr_ty = llvm_ptr_ty(ctx);
            let addr = mir_dialect::ops::AddressOfOp::new(ctx, symbol, ptr_ty);
            addr.get_operation().insert_at_back(insert_block, ctx);
            let mut addr_value = addr.get_result(ctx);
            if offset.bytes() != 0 {
                addr_value = ptr_offset_const(ctx, insert_block, addr_value, offset.bytes())?;
            }
            let load = mir_dialect::ops::LoadOp::new(ctx, addr_value, ty);
            load.get_operation().insert_at_back(insert_block, ctx);
            Ok(Some(load.get_result(ctx)))
        }
        _ => Ok(None),
    }
}

/// Load a value stored with the real rustc layout (e.g. written by prebuilt
/// std code) and rebuild it in this importer's representation, following the
/// real field offsets recursively.
pub(super) fn load_value_from_real_layout<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    rust_ty: Ty<'tcx>,
    addr: Value,
) -> Result<Value, String> {
    use rustc_middle::ty::TyKind;
    let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
    let rust_ty = normalize_ty(tcx, runtime_ty(rust_ty));

    let load_direct = |ctx: &mut Context, ty: TypeHandle| -> Value {
        let load = mir_dialect::ops::LoadOp::new(ctx, addr, ty);
        load.get_operation().insert_at_back(insert_block, ctx);
        load.get_result(ctx)
    };

    let field_based = |ctx: &mut Context, fields: Vec<Ty<'tcx>>| -> Result<Value, String> {
        let our_ty = convert_ty(tcx, ctx, rust_ty)?;
        let layout = tcx
            .layout_of(typing_env.as_query_input(rust_ty))
            .map_err(|error| format!("no layout for {rust_ty:?}: {error:?}"))?;
        let undef = mir_dialect::ops::UndefOp::new(ctx, our_ty);
        undef.get_operation().insert_at_back(insert_block, ctx);
        let mut current = undef.get_result(ctx);
        for (idx, field_ty) in fields.into_iter().enumerate() {
            if layout_size_of_ty(tcx, field_ty)? == 0 {
                continue;
            }
            let offset = layout.fields.offset(idx).bytes();
            let field_addr = if offset != 0 {
                ptr_offset_const(ctx, insert_block, addr, offset)?
            } else {
                addr
            };
            let value = load_value_from_real_layout(tcx, ctx, insert_block, field_ty, field_addr)?;
            let index = converted_field_index(tcx, rust_ty, idx)?;
            let insert = mir_dialect::ops::InsertValueOp::new(ctx, value, current, vec![index]);
            insert.get_operation().insert_at_back(insert_block, ctx);
            current = insert.get_result(ctx);
        }
        Ok(current)
    };

    match rust_ty.kind() {
        TyKind::Bool
        | TyKind::Char
        | TyKind::Int(_)
        | TyKind::Uint(_)
        | TyKind::Float(_)
        | TyKind::FnPtr(_, _)
        | TyKind::Ref(_, _, _)
        | TyKind::RawPtr(_, _) => {
            // Scalars and references; fat references share the real
            // {ptr, meta} two-word layout.
            let our_ty = convert_ty(tcx, ctx, rust_ty)?;
            Ok(load_direct(ctx, our_ty))
        }
        TyKind::Tuple(fields) => field_based(ctx, fields.iter().collect()),
        TyKind::Closure(_, args) => {
            field_based(ctx, args.as_closure().upvar_tys().iter().collect())
        }
        TyKind::Adt(adt_def, _) if is_fmt_rt_argument_type(tcx, adt_def.did()) => {
            let our_ty = convert_ty(tcx, ctx, rust_ty)?;
            Ok(load_direct(ctx, our_ty))
        }
        TyKind::Adt(adt_def, args) if adt_def.is_struct() => {
            let fields = adt_def
                .non_enum_variant()
                .fields
                .iter()
                .map(|field| field.ty(tcx, args))
                .collect::<Vec<_>>();
            field_based(ctx, fields)
        }
        TyKind::Adt(adt_def, _) if adt_def.is_union() => {
            let our_ty = convert_ty(tcx, ctx, rust_ty)?;
            Ok(load_direct(ctx, our_ty))
        }
        TyKind::Adt(adt_def, _) if adt_def.is_enum() => {
            // An enum value is simply its real in-memory bytes, so its converted
            // type already matches the source layout: load it directly.
            let our_ty = convert_ty(tcx, ctx, rust_ty)?;
            Ok(load_direct(ctx, our_ty))
        }
        TyKind::Array(elem, len) => {
            let our_ty = convert_ty(tcx, ctx, rust_ty)?;
            let len = array_len(tcx, *len)?;
            let elem_stride = rustc_layout_size_of_ty(tcx, typing_env, *elem)?;
            let undef = mir_dialect::ops::UndefOp::new(ctx, our_ty);
            undef.get_operation().insert_at_back(insert_block, ctx);
            let mut current = undef.get_result(ctx);
            for idx in 0..len {
                let offset = idx * elem_stride;
                let elem_addr = if offset != 0 {
                    ptr_offset_const(ctx, insert_block, addr, offset)?
                } else {
                    addr
                };
                let value = load_value_from_real_layout(tcx, ctx, insert_block, *elem, elem_addr)?;
                let insert =
                    mir_dialect::ops::InsertValueOp::new(ctx, value, current, vec![idx as u32]);
                insert.get_operation().insert_at_back(insert_block, ctx);
                current = insert.get_result(ctx);
            }
            Ok(current)
        }
        other => Err(format!("unsupported real-layout load: {other:?}")),
    }
}

// Threads the importer's per-function lowering state; a parameter struct
// would be packed and unpacked at every call site for no clarity gain.
#[allow(clippy::too_many_arguments)]
pub(super) fn import_enum_constant<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    typing_env: rustc_middle::ty::TypingEnv<'tcx>,
    span: rustc_span::Span,
    constant: rustc_mir::Const<'tcx>,
    ty: TypeHandle,
) -> Result<Value, String> {
    // An enum value is its real in-memory bytes, so materialize the constant's
    // real layout into an anonymous global and load the enum blob from it. This
    // handles C-like, niche-encoded, and data-carrying enum constants uniformly.
    let rust_ty = normalize_ty(tcx, runtime_ty(constant.ty()));
    let size = rustc_layout_size_of_ty(tcx, typing_env, rust_ty)?;
    if size == 0 {
        let op = mir_dialect::ops::UndefOp::new(ctx, ty);
        op.get_operation().insert_at_back(insert_block, ctx);
        return Ok(op.get_result(ctx));
    }
    let Some(bytes) = enum_constant_bytes(tcx, typing_env, span, constant, size) else {
        // The variant payload carries pointers (e.g. `Some(&STATIC)`): fall
        // back to the relocated-global path, which loads the enum blob out of
        // an `ll.data` global whose pointer slots the linker fills in.
        return import_pointer_constant(
            tcx, ctx, state, insert_block, typing_env, span, constant, ty,
        )?
        .ok_or_else(|| format!("unsupported MIR enum constant: {:?}", constant));
    };
    let symbol = declare_anonymous_byte_global(ctx, state.module_body, &bytes);
    let ptr_ty = llvm_ptr_ty(ctx);
    let addr = mir_dialect::ops::AddressOfOp::new(ctx, symbol, ptr_ty);
    addr.get_operation().insert_at_back(insert_block, ctx);
    let load = mir_dialect::ops::LoadOp::new(ctx, addr.get_result(ctx), ty);
    load.get_operation().insert_at_back(insert_block, ctx);
    Ok(load.get_result(ctx))
}

/// The `size` real-layout bytes of an enum constant, or `None` when it cannot be
/// evaluated to plain bytes (e.g. it carries pointer relocations).
pub(super) fn enum_constant_bytes<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: rustc_middle::ty::TypingEnv<'tcx>,
    span: rustc_span::Span,
    constant: rustc_mir::Const<'tcx>,
    size: u64,
) -> Option<Vec<u8>> {
    match constant.eval(tcx, typing_env, span).ok()? {
        rustc_mir::ConstValue::Scalar(scalar) => {
            let scalar = scalar.try_to_scalar_int().ok()?;
            let mut bytes = scalar.to_uint(scalar.size()).to_le_bytes().to_vec();
            bytes.resize(size as usize, 0);
            Some(bytes)
        }
        rustc_mir::ConstValue::Indirect { alloc_id, offset } => {
            let rustc_mir::interpret::GlobalAlloc::Memory(alloc) = tcx.global_alloc(alloc_id)
            else {
                return None;
            };
            let alloc = alloc.inner();
            if !alloc.provenance().ptrs().is_empty() {
                return None;
            }
            let start = offset.bytes() as usize;
            Some(
                alloc
                    .inspect_with_uninit_and_ptr_outside_interpreter(start..start + size as usize)
                    .to_vec(),
            )
        }
        _ => None,
    }
}

pub(super) fn import_initialized_maybe_uninit_u8_constant<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    typing_env: rustc_middle::ty::TypingEnv<'tcx>,
    span: rustc_span::Span,
    constant: rustc_mir::Const<'tcx>,
    ty: TypeHandle,
) -> Result<Option<Value>, String> {
    if !is_maybe_uninit_u8_ty(tcx, constant.ty()) {
        return Ok(None);
    }
    let Some(bytes) = evaluated_constant_bytes(tcx, typing_env, span, constant, 1) else {
        return Ok(None);
    };
    let Some(byte) = bytes.first().copied() else {
        return Ok(None);
    };

    let byte_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
    let byte = integer_constant(ctx, byte_ty, byte as u128)?;
    byte.get_operation().insert_at_back(insert_block, ctx);

    let array_ty = llvm::types::ArrayType::get(ctx, byte_ty, 1).into();
    let array_undef = mir_dialect::ops::UndefOp::new(ctx, array_ty);
    array_undef
        .get_operation()
        .insert_at_back(insert_block, ctx);
    let array = mir_dialect::ops::InsertValueOp::new(
        ctx,
        byte.get_result(ctx),
        array_undef.get_result(ctx),
        vec![0],
    );
    array.get_operation().insert_at_back(insert_block, ctx);

    let outer_undef = mir_dialect::ops::UndefOp::new(ctx, ty);
    outer_undef
        .get_operation()
        .insert_at_back(insert_block, ctx);
    let outer = mir_dialect::ops::InsertValueOp::new(
        ctx,
        array.get_result(ctx),
        outer_undef.get_result(ctx),
        vec![0],
    );
    outer.get_operation().insert_at_back(insert_block, ctx);
    Ok(Some(outer.get_result(ctx)))
}

pub(super) fn evaluated_constant_bytes<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: rustc_middle::ty::TypingEnv<'tcx>,
    span: rustc_span::Span,
    constant: rustc_mir::Const<'tcx>,
    len: u64,
) -> Option<Vec<u8>> {
    match constant.eval(tcx, typing_env, span).ok()? {
        rustc_mir::ConstValue::Indirect { alloc_id, offset } => {
            allocation_bytes(tcx, alloc_id, offset, len)
        }
        rustc_mir::ConstValue::Scalar(scalar) => scalar
            .try_to_scalar_int()
            .ok()
            .map(|scalar| scalar.to_uint(scalar.size()).to_le_bytes()[..len as usize].to_vec()),
        _ => None,
    }
}

pub(super) fn rustc_layout_size_of_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: rustc_middle::ty::TypingEnv<'tcx>,
    ty: Ty<'tcx>,
) -> Result<u64, String> {
    tcx.layout_of(typing_env.as_query_input(ty))
        .map(|layout| layout.size.bytes())
        .map_err(|error| format!("unsupported Rust type layout in MIR importer: {ty:?}: {error:?}"))
}

pub(super) fn rustc_layout_align_of_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: rustc_middle::ty::TypingEnv<'tcx>,
    ty: Ty<'tcx>,
) -> Result<u64, String> {
    tcx.layout_of(typing_env.as_query_input(ty))
        .map(|layout| layout.align.abi.bytes())
        .map_err(|error| format!("unsupported Rust type layout in MIR importer: {ty:?}: {error:?}"))
}

pub(super) fn mono_const<'tcx>(
    tcx: TyCtxt<'tcx>,
    state: &FunctionImportState<'tcx>,
    constant: rustc_mir::Const<'tcx>,
) -> rustc_mir::Const<'tcx> {
    state.instance.map_or(constant, |instance| {
        instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            rustc_middle::ty::TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(constant),
        )
    })
}

/// Monomorphize a type-level constant (e.g. an array length that mentions a
/// generic const parameter) the same way `mono_ty`/`mono_const` do.
pub(super) fn mono_ty_const<'tcx>(
    tcx: TyCtxt<'tcx>,
    state: &FunctionImportState<'tcx>,
    constant: rustc_middle::ty::Const<'tcx>,
) -> rustc_middle::ty::Const<'tcx> {
    state.instance.map_or(constant, |instance| {
        instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            rustc_middle::ty::TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(constant),
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn import_str_constant<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    module_body: Ptr<BasicBlock>,
    insert_block: Ptr<BasicBlock>,
    typing_env: rustc_middle::ty::TypingEnv<'tcx>,
    body: &Body<'tcx>,
    constant: &ConstOperand<'tcx>,
    const_: rustc_mir::Const<'tcx>,
) -> Result<Value, String> {
    let value = literal_string_constant(tcx, body, constant)
        .or_else(|| evaluated_str_constant(tcx, typing_env, constant.span, const_))
        .ok_or_else(|| format!("unsupported string constant: {:?}", const_))?;
    // The old `cmir.cstr` becomes a private NUL-terminated byte global plus
    // an address-of; the NUL matches the old ll.cstr literal emission and is
    // excluded from the `{ptr, len}` str-ref length below.
    let mut bytes = value.clone().into_bytes();
    bytes.push(0);
    let symbol = declare_anonymous_byte_global(ctx, module_body, &bytes);
    let ptr_ty = llvm_ptr_ty(ctx);
    let ptr = mir_dialect::ops::AddressOfOp::new(ctx, symbol, ptr_ty);
    ptr.get_operation().insert_at_back(insert_block, ctx);

    let usize_ty = usize_ty(ctx);
    let len = mir_dialect::ops::ConstantOp::new_integer(
        ctx,
        IntegerAttr::new(
            usize_ty,
            APInt::from_u64(value.len() as u64, NonZero::new(64).expect("64 is nonzero")),
        ),
    );
    len.get_operation().insert_at_back(insert_block, ctx);

    let str_ref_ty = str_ref_ty(ctx);
    let undef = mir_dialect::ops::UndefOp::new(ctx, str_ref_ty);
    undef.get_operation().insert_at_back(insert_block, ctx);
    let with_ptr = mir_dialect::ops::InsertValueOp::new(
        ctx,
        ptr.get_result(ctx),
        undef.get_result(ctx),
        vec![0],
    );
    with_ptr.get_operation().insert_at_back(insert_block, ctx);
    let with_len = mir_dialect::ops::InsertValueOp::new(
        ctx,
        len.get_result(ctx),
        with_ptr.get_result(ctx),
        vec![1],
    );
    with_len.get_operation().insert_at_back(insert_block, ctx);
    Ok(with_len.get_result(ctx))
}

/// Materialize scalar constant bits into a struct type that (recursively)
/// wraps exactly one sized scalar field.
pub(super) fn scalar_constant_in_aggregate(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    ty: TypeHandle,
    bits: u128,
) -> Result<Option<Value>, String> {
    if let Ok(op) = constant_from_bits(ctx, ty, bits) {
        op.get_operation().insert_at_back(insert_block, ctx);
        return Ok(Some(op.get_result(ctx)));
    }
    let fields = {
        let ty_ref = ty.deref(ctx);
        let Some(struct_ty) = ty_ref.downcast_ref::<llvm::types::StructType>() else {
            return Ok(None);
        };
        if struct_ty.is_opaque() {
            return Ok(None);
        }
        struct_ty.fields().collect::<Vec<_>>()
    };
    let mut sized = fields.iter().enumerate().filter(|(_, field)| {
        crabbit_ty_size(ctx, **field)
            .map(|size| size > 0)
            .unwrap_or(true)
    });
    let Some((index, field_ty)) = sized.next() else {
        return Ok(None);
    };
    if sized.next().is_some() {
        return Ok(None);
    }
    let field_ty = *field_ty;
    let Some(inner) = scalar_constant_in_aggregate(ctx, insert_block, field_ty, bits)? else {
        return Ok(None);
    };
    let undef = mir_dialect::ops::UndefOp::new(ctx, ty);
    undef.get_operation().insert_at_back(insert_block, ctx);
    let wrap =
        mir_dialect::ops::InsertValueOp::new(ctx, inner, undef.get_result(ctx), vec![index as u32]);
    wrap.get_operation().insert_at_back(insert_block, ctx);
    Ok(Some(wrap.get_result(ctx)))
}

pub(super) fn constant_from_bits(
    ctx: &mut Context,
    ty: TypeHandle,
    bits: u128,
) -> Result<mir_dialect::ops::ConstantOp, String> {
    if ty.deref(ctx).downcast_ref::<FP32Type>().is_some() {
        return Ok(mir_dialect::ops::ConstantOp::new(
            ctx,
            FPSingleAttr::from(f32::from_bits(bits as u32)).into(),
        ));
    }
    if ty.deref(ctx).downcast_ref::<FP64Type>().is_some() {
        return Ok(mir_dialect::ops::ConstantOp::new(
            ctx,
            FPDoubleAttr::from(f64::from_bits(bits as u64)).into(),
        ));
    }
    integer_constant(ctx, ty, bits)
}

pub(super) fn integer_constant(
    ctx: &mut Context,
    ty: TypeHandle,
    bits: u128,
) -> Result<mir_dialect::ops::ConstantOp, String> {
    let ty_ref = ty.deref(ctx);
    let int_ty = ty_ref
        .downcast_ref::<IntegerType>()
        .ok_or_else(|| "MIR integer constant has non-integer type".to_string())?;
    let width = int_ty.width();
    let int_ty: TypedHandle<IntegerType> = TypedHandle::from_handle(ty, ctx).expect("integer_constant is only called with integer types");
    drop(ty_ref);
    Ok(mir_dialect::ops::ConstantOp::new_integer(
        ctx,
        IntegerAttr::new(
            int_ty,
            APInt::from_u128(bits, NonZero::new(width as usize).expect("integer types have nonzero width")),
        ),
    ))
}
