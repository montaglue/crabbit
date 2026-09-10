use super::*;

pub(super) fn load_place<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    place: Place<'tcx>,
) -> Result<Value, String> {
    if let Some(value) = try_load_field_projection(tcx, ctx, state, insert_block, body, place)? {
        return Ok(value);
    }

    let rust_ty = mono_ty(tcx, state, place.ty(body, tcx).ty);
    if convert_storage_ty(tcx, ctx, rust_ty)?.is_none() {
        let unit_ty = convert_ty(tcx, ctx, rust_ty)?;
        let undef = mir_dialect::ops::UndefOp::new(ctx, unit_ty);
        undef.get_operation().insert_at_back(insert_block, ctx);
        return Ok(undef.get_result(ctx));
    }

    let slot = place_addr(tcx, ctx, state, insert_block, body, &place)?;
    let ty = convert_ty(tcx, ctx, rust_ty)?;
    let load = mir_dialect::ops::LoadOp::new(ctx, slot, ty);
    load.get_operation().insert_at_back(insert_block, ctx);
    normalize_bool_for_immediate(tcx, ctx, state, insert_block, rust_ty, load.get_result(ctx))
}

pub(super) fn try_load_field_projection<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    place: Place<'tcx>,
) -> Result<Option<Value>, String> {
    if place.projection.is_empty()
        || !place
            .projection
            .iter()
            .all(|elem| matches!(elem, rustc_mir::ProjectionElem::Field(_, _)))
        || projection_touches_union(tcx, state, body, place)?
    {
        return Ok(None);
    }

    let slot = local_slot(state, place.local)?;
    let aggregate_ty = convert_ty(
        tcx,
        ctx,
        mono_ty(tcx, state, body.local_decls[place.local].ty),
    )?;
    let load = mir_dialect::ops::LoadOp::new(ctx, slot, aggregate_ty);
    load.get_operation().insert_at_back(insert_block, ctx);
    let mut current = load.get_result(ctx);
    let mut current_rust_ty = mono_ty(tcx, state, body.local_decls[place.local].ty);

    for elem in place.projection {
        let rustc_mir::ProjectionElem::Field(field, field_ty) = elem else {
            unreachable!("field-only projection checked above");
        };
        let index = converted_field_index(tcx, current_rust_ty, field.index())?;
        let result_ty = convert_ty(tcx, ctx, mono_ty(tcx, state, field_ty))?;
        let extract = mir_dialect::ops::ExtractValueOp::new(ctx, current, vec![index], result_ty);
        extract.get_operation().insert_at_back(insert_block, ctx);
        current = extract.get_result(ctx);
        current_rust_ty = mono_ty(tcx, state, field_ty);
    }

    let rust_ty = mono_ty(tcx, state, place.ty(body, tcx).ty);
    normalize_bool_for_immediate(tcx, ctx, state, insert_block, rust_ty, current).map(Some)
}

/// The index of a source-order field within the converted struct type.
pub(super) fn converted_field_index<'tcx>(
    tcx: TyCtxt<'tcx>,
    aggregate_ty: Ty<'tcx>,
    source_idx: usize,
) -> Result<u32, String> {
    let aggregate_ty = normalize_ty(tcx, runtime_ty(aggregate_ty));
    match struct_like_source_fields(tcx, aggregate_ty) {
        Some(fields) => Ok(field_memory_position(
            tcx,
            aggregate_ty,
            fields.len(),
            source_idx,
        )),
        None => Ok(source_idx as u32),
    }
}

pub(super) fn store_place<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    place: &Place<'tcx>,
    value: Value,
) -> Result<(), String> {
    let rust_ty = mono_ty(tcx, state, place.ty(body, tcx).ty);
    let value = normalize_bool_for_storage(tcx, ctx, state, insert_block, rust_ty, value)?;
    if !place.projection.is_empty()
        && place
            .projection
            .iter()
            .all(|elem| matches!(elem, rustc_mir::ProjectionElem::Field(_, _)))
        && !projection_touches_union(tcx, state, body, *place)?
    {
        return store_field_projection(tcx, ctx, state, insert_block, body, place, value);
    }

    let addr = place_addr(tcx, ctx, state, insert_block, body, place)?;
    let store = mir_dialect::ops::StoreOp::new(ctx, value, addr);
    store.get_operation().insert_at_back(insert_block, ctx);
    Ok(())
}

pub(super) fn store_field_projection<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    place: &Place<'tcx>,
    value: Value,
) -> Result<(), String> {
    let slot = local_slot(state, place.local)?;
    let aggregate_ty = convert_ty(
        tcx,
        ctx,
        mono_ty(tcx, state, body.local_decls[place.local].ty),
    )?;
    let load = mir_dialect::ops::LoadOp::new(ctx, slot, aggregate_ty);
    load.get_operation().insert_at_back(insert_block, ctx);

    let mut current = load.get_result(ctx);
    let mut current_rust_ty = mono_ty(tcx, state, body.local_decls[place.local].ty);
    let mut parents = Vec::<(u32, Value)>::new();
    for elem in place
        .projection
        .iter()
        .take(place.projection.len().saturating_sub(1))
    {
        let rustc_mir::ProjectionElem::Field(field, field_ty) = elem else {
            unreachable!("field-only projection checked above");
        };
        let index = converted_field_index(tcx, current_rust_ty, field.index())?;
        let result_ty = convert_ty(tcx, ctx, mono_ty(tcx, state, field_ty))?;
        parents.push((index, current));
        let extract = mir_dialect::ops::ExtractValueOp::new(ctx, current, vec![index], result_ty);
        extract.get_operation().insert_at_back(insert_block, ctx);
        current = extract.get_result(ctx);
        current_rust_ty = mono_ty(tcx, state, field_ty);
    }

    let Some(rustc_mir::ProjectionElem::Field(field, _)) = place.projection.last() else {
        unreachable!("non-empty field-only projection checked above");
    };
    let last_index = converted_field_index(tcx, current_rust_ty, field.index())?;
    let insert = mir_dialect::ops::InsertValueOp::new(ctx, value, current, vec![last_index]);
    insert.get_operation().insert_at_back(insert_block, ctx);
    let mut updated = insert.get_result(ctx);

    for (index, parent) in parents.into_iter().rev() {
        let insert = mir_dialect::ops::InsertValueOp::new(ctx, updated, parent, vec![index]);
        insert.get_operation().insert_at_back(insert_block, ctx);
        updated = insert.get_result(ctx);
    }

    let store = mir_dialect::ops::StoreOp::new(ctx, updated, slot);
    store.get_operation().insert_at_back(insert_block, ctx);
    Ok(())
}

/// The "address" of a slot-less local (type `()` or `!`): a dangling pointer
/// at the type's alignment, the same convention rustc's codegen uses for ZST
/// places. The pointee is never read or written through it.
pub(super) fn dangling_zst_addr<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    local: Local,
) -> Result<Value, String> {
    let ty = mono_ty(tcx, state, body.local_decls[local].ty);
    let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
    let align = rustc_layout_align_of_ty(tcx, typing_env, runtime_ty(ty))
        .unwrap_or(1)
        .max(1);
    let usize_ty: TypeHandle = usize_ty(ctx).into();
    let addr = integer_constant(ctx, usize_ty, align as u128)?;
    addr.get_operation().insert_at_back(insert_block, ctx);
    let ptr_ty = llvm_ptr_ty(ctx);
    Ok(cast_value_to_type(
        ctx,
        insert_block,
        addr.get_result(ctx),
        ptr_ty,
    ))
}

pub(super) fn place_addr<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    place: &Place<'tcx>,
) -> Result<Value, String> {
    let base = match local_slot_opt(state, place.local)? {
        Some(slot) => slot,
        // A local of type `()` or `!` gets no storage slot, but MIR may
        // still take its address (e.g. `&core::ptr::metadata(p)` where the
        // metadata is `()`). Mirror rustc's ZST codegen: a dangling,
        // suitably aligned pointer.
        None => dangling_zst_addr(tcx, ctx, state, insert_block, body, place.local)?,
    };
    if place.projection.is_empty() {
        return Ok(base);
    }

    let mut addr = base;
    let mut current_ty = mono_ty(tcx, state, body.local_decls[place.local].ty);
    let mut current_variant = None;
    for elem in place.projection {
        match elem {
            rustc_mir::ProjectionElem::Deref => {
                let ptr_ty = convert_ty(tcx, ctx, mono_ty(tcx, state, current_ty))?;
                let load = mir_dialect::ops::LoadOp::new(ctx, addr, ptr_ty);
                load.get_operation().insert_at_back(insert_block, ctx);
                addr = load.get_result(ctx);
                current_ty = mono_ty(tcx, state, pointee_ty(current_ty)?);
                current_variant = None;
            }
            rustc_mir::ProjectionElem::Field(field, field_ty) => {
                let offset = field_offset_of(tcx, current_ty, current_variant, field.index())?;
                let field_ty = mono_ty(tcx, state, field_ty);
                let addr_is_fat = addr
                    .get_type(ctx)
                    .deref(ctx)
                    .is::<llvm::types::StructType>();
                if addr_is_fat {
                    // The base is a fat `{ptr, meta}` value (a Deref of a
                    // reference to an unsized ADT). Sized fields live at byte
                    // offsets off the data pointer; the unsized tail field
                    // keeps the metadata.
                    let fat_ty = addr.get_type(ctx);
                    let ptr_ty = llvm_ptr_ty(ctx);
                    let data = mir_dialect::ops::ExtractValueOp::new(ctx, addr, vec![0], ptr_ty);
                    data.get_operation().insert_at_back(insert_block, ctx);
                    let mut data = data.get_result(ctx);
                    if offset != 0 {
                        data = ptr_offset_const(ctx, insert_block, data, offset)?;
                    }
                    let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
                    if runtime_ty(field_ty).is_sized(tcx, typing_env) {
                        addr = data;
                    } else {
                        let usize_ty: TypeHandle = usize_ty(ctx).into();
                        let meta =
                            mir_dialect::ops::ExtractValueOp::new(ctx, addr, vec![1], usize_ty);
                        meta.get_operation().insert_at_back(insert_block, ctx);
                        let undef = mir_dialect::ops::UndefOp::new(ctx, fat_ty);
                        undef.get_operation().insert_at_back(insert_block, ctx);
                        let with_ptr = mir_dialect::ops::InsertValueOp::new(
                            ctx,
                            data,
                            undef.get_result(ctx),
                            vec![0],
                        );
                        with_ptr.get_operation().insert_at_back(insert_block, ctx);
                        let with_meta = mir_dialect::ops::InsertValueOp::new(
                            ctx,
                            meta.get_result(ctx),
                            with_ptr.get_result(ctx),
                            vec![1],
                        );
                        with_meta.get_operation().insert_at_back(insert_block, ctx);
                        addr = with_meta.get_result(ctx);
                    }
                } else if offset != 0 {
                    addr = ptr_offset_const(ctx, insert_block, addr, offset)?;
                }
                current_ty = field_ty;
                current_variant = None;
            }
            rustc_mir::ProjectionElem::Downcast(_, variant) => {
                current_variant = Some(variant);
            }
            rustc_mir::ProjectionElem::Index(index_local) => {
                let index = load_place(tcx, ctx, state, insert_block, body, index_local.into())?;
                let elem_size = indexed_elem_size(tcx, current_ty)?;
                let byte_offset = scale_index(ctx, insert_block, index, elem_size)?;
                let offset = mir_dialect::ops::PtrOffsetOp::new(ctx, addr, byte_offset);
                offset.get_operation().insert_at_back(insert_block, ctx);
                addr = offset.get_result(ctx);
                current_ty = mono_ty(tcx, state, indexed_elem_ty(current_ty)?);
                current_variant = None;
            }
            rustc_mir::ProjectionElem::ConstantIndex {
                offset,
                min_length,
                from_end,
            } => {
                let index = if from_end {
                    min_length - offset
                } else {
                    offset
                };
                let elem_size = indexed_elem_size(tcx, current_ty)?;
                let byte_offset = index * elem_size;
                if byte_offset != 0 {
                    addr = ptr_offset_const(ctx, insert_block, addr, byte_offset)?;
                }
                current_ty = mono_ty(tcx, state, indexed_elem_ty(current_ty)?);
                current_variant = None;
            }
            rustc_mir::ProjectionElem::Subslice { from, to, from_end } => {
                match runtime_ty(current_ty).kind() {
                    rustc_middle::ty::TyKind::Array(elem, len) => {
                        // `array[from..]` starts `from` elements in; the result
                        // is a shorter array at a constant byte offset.
                        let elem_ty = mono_ty(tcx, state, *elem);
                        let elem_size = indexed_elem_size(tcx, current_ty)?;
                        let byte_offset = from * elem_size;
                        if byte_offset != 0 {
                            addr = ptr_offset_const(ctx, insert_block, addr, byte_offset)?;
                        }
                        let new_len = if from_end {
                            array_len(tcx, mono_ty_const(tcx, state, *len))?
                                .checked_sub(from + to)
                                .ok_or_else(|| {
                                    format!("MIR Subslice out of bounds: {elem:?} {from}..-{to}")
                                })?
                        } else {
                            to - from
                        };
                        current_ty = Ty::new_array(tcx, elem_ty, new_len);
                    }
                    rustc_middle::ty::TyKind::Slice(_) => {
                        // A slice place's "address" is the fat `{ptr, len}`
                        // value the preceding Deref loaded: rebase the data
                        // pointer by `from` elements and shrink the length.
                        let elem_size = indexed_elem_size(tcx, current_ty)?;
                        let fat_ty = addr.get_type(ctx);
                        let ptr_ty = llvm_ptr_ty(ctx);
                        let usize_ty: TypeHandle = usize_ty(ctx).into();
                        let data = mir_dialect::ops::ExtractValueOp::new(ctx, addr, vec![0], ptr_ty);
                        data.get_operation().insert_at_back(insert_block, ctx);
                        let mut data = data.get_result(ctx);
                        if from != 0 {
                            data = ptr_offset_const(ctx, insert_block, data, from * elem_size)?;
                        }
                        let new_len = if from_end {
                            let len =
                                mir_dialect::ops::ExtractValueOp::new(ctx, addr, vec![1], usize_ty);
                            len.get_operation().insert_at_back(insert_block, ctx);
                            let dropped = integer_constant(ctx, usize_ty, (from + to) as u128)?;
                            dropped.get_operation().insert_at_back(insert_block, ctx);
                            let sub = mir_dialect::ops::SubOp::new(
                                ctx,
                                len.get_result(ctx),
                                dropped.get_result(ctx),
                            );
                            sub.get_operation().insert_at_back(insert_block, ctx);
                            sub.get_result(ctx)
                        } else {
                            let len = integer_constant(ctx, usize_ty, (to - from) as u128)?;
                            len.get_operation().insert_at_back(insert_block, ctx);
                            len.get_result(ctx)
                        };
                        let undef = mir_dialect::ops::UndefOp::new(ctx, fat_ty);
                        undef.get_operation().insert_at_back(insert_block, ctx);
                        let with_ptr = mir_dialect::ops::InsertValueOp::new(
                            ctx,
                            data,
                            undef.get_result(ctx),
                            vec![0],
                        );
                        with_ptr.get_operation().insert_at_back(insert_block, ctx);
                        let with_len = mir_dialect::ops::InsertValueOp::new(
                            ctx,
                            new_len,
                            with_ptr.get_result(ctx),
                            vec![1],
                        );
                        with_len.get_operation().insert_at_back(insert_block, ctx);
                        addr = with_len.get_result(ctx);
                        // The place stays the same slice type.
                    }
                    other => {
                        return Err(format!(
                            "unsupported MIR Subslice base type: {other:?}"
                        ));
                    }
                }
                current_variant = None;
            }
            rustc_mir::ProjectionElem::OpaqueCast(ty)
            | rustc_mir::ProjectionElem::UnwrapUnsafeBinder(ty) => {
                current_ty = mono_ty(tcx, state, ty);
                current_variant = None;
            }
        }
    }

    Ok(addr)
}

pub(super) fn ptr_offset_const(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    base: Value,
    offset: u64,
) -> Result<Value, String> {
    let usize_ty: TypeHandle = usize_ty(ctx).into();
    let offset_op = integer_constant(ctx, usize_ty, offset as u128)?;
    offset_op.get_operation().insert_at_back(insert_block, ctx);
    let ptr_offset = mir_dialect::ops::PtrOffsetOp::new(ctx, base, offset_op.get_result(ctx));
    ptr_offset.get_operation().insert_at_back(insert_block, ctx);
    Ok(ptr_offset.get_result(ctx))
}

pub(super) fn scale_index(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    index: Value,
    element_size: u64,
) -> Result<Value, String> {
    if element_size == 1 {
        return Ok(index);
    }
    let scale = integer_constant(ctx, index.get_type(ctx), element_size as u128)?;
    scale.get_operation().insert_at_back(insert_block, ctx);
    let mul = mir_dialect::ops::MulOp::new(ctx, index, scale.get_result(ctx));
    mul.get_operation().insert_at_back(insert_block, ctx);
    Ok(mul.get_result(ctx))
}

pub(super) fn pointee_ty<'tcx>(ty: Ty<'tcx>) -> Result<Ty<'tcx>, String> {
    match runtime_ty(ty).kind() {
        rustc_middle::ty::TyKind::Ref(_, inner, _) | rustc_middle::ty::TyKind::RawPtr(inner, _) => {
            Ok(*inner)
        }
        other => Err(format!("MIR deref of non-pointer type: {other:?}")),
    }
}

pub(super) fn indexed_elem_ty<'tcx>(ty: Ty<'tcx>) -> Result<Ty<'tcx>, String> {
    match runtime_ty(ty).kind() {
        rustc_middle::ty::TyKind::Array(elem, _) | rustc_middle::ty::TyKind::Slice(elem) => {
            Ok(*elem)
        }
        other => Err(format!("MIR index of non-array type: {other:?}")),
    }
}

pub(super) fn aggregate_is_union<'tcx>(tcx: TyCtxt<'tcx>, kind: &rustc_mir::AggregateKind<'tcx>) -> bool {
    match kind {
        rustc_mir::AggregateKind::Adt(def_id, _, _, _, _) => tcx.adt_def(*def_id).is_union(),
        _ => false,
    }
}

pub(super) fn projection_touches_union<'tcx>(
    tcx: TyCtxt<'tcx>,
    state: &FunctionImportState<'tcx>,
    body: &Body<'tcx>,
    place: Place<'tcx>,
) -> Result<bool, String> {
    let mut current_ty = mono_ty(tcx, state, body.local_decls[place.local].ty);
    let mut current_variant = None;
    for elem in place.projection {
        match elem {
            rustc_mir::ProjectionElem::Field(_, field_ty) => {
                if matches!(
                    runtime_ty(current_ty).kind(),
                    rustc_middle::ty::TyKind::Adt(adt_def, _) if adt_def.is_union()
                ) {
                    return Ok(true);
                }
                current_ty = mono_ty(tcx, state, field_ty);
                current_variant = None;
            }
            rustc_mir::ProjectionElem::Downcast(_, variant) => {
                current_variant = Some(variant);
            }
            rustc_mir::ProjectionElem::Deref => {
                current_ty = mono_ty(tcx, state, pointee_ty(current_ty)?);
                current_variant = None;
            }
            rustc_mir::ProjectionElem::Index(_) => {
                current_ty = mono_ty(tcx, state, indexed_elem_ty(current_ty)?);
                current_variant = None;
            }
            rustc_mir::ProjectionElem::ConstantIndex { .. } => {
                current_ty = mono_ty(tcx, state, indexed_elem_ty(current_ty)?);
                current_variant = None;
            }
            rustc_mir::ProjectionElem::OpaqueCast(ty)
            | rustc_mir::ProjectionElem::UnwrapUnsafeBinder(ty) => {
                current_ty = mono_ty(tcx, state, ty);
                current_variant = None;
            }
            _ => {}
        }
        let _ = current_variant;
    }
    Ok(false)
}

pub(super) fn indexed_elem_size<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Result<u64, String> {
    layout_size_of_ty(tcx, indexed_elem_ty(ty)?)
}
