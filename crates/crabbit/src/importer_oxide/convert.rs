use super::*;

pub(super) fn convert_return_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    ty: Ty<'tcx>,
) -> Result<Vec<TypeHandle>, String> {
    let ty = runtime_ty(ty);
    match ty.kind() {
        rustc_middle::ty::TyKind::Tuple(fields) if fields.is_empty() => Ok(Vec::new()),
        rustc_middle::ty::TyKind::Never => Ok(Vec::new()),
        rustc_middle::ty::TyKind::Bool => Ok(vec![bool_immediate_ty(ctx)]),
        _ => Ok(vec![convert_ty(tcx, ctx, ty)?]),
    }
}

pub(super) fn convert_storage_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    ty: Ty<'tcx>,
) -> Result<Option<TypeHandle>, String> {
    let ty = runtime_ty(ty);
    match ty.kind() {
        rustc_middle::ty::TyKind::Tuple(fields) if fields.is_empty() => Ok(None),
        rustc_middle::ty::TyKind::Never => Ok(None),
        _ => Ok(Some(convert_ty(tcx, ctx, ty)?)),
    }
}

pub(super) fn convert_immediate_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    ty: Ty<'tcx>,
) -> Result<TypeHandle, String> {
    if is_bool_ty(ty) {
        return Ok(bool_immediate_ty(ctx));
    }
    convert_ty(tcx, ctx, ty)
}

pub(super) fn convert_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    ty: Ty<'tcx>,
) -> Result<TypeHandle, String> {
    use rustc_middle::ty::TyKind;

    let ty = normalize_ty(tcx, runtime_ty(ty));
    let ty: TypedHandle<IntegerType> = match ty.kind() {
        TyKind::Bool => return Ok(bool_storage_ty(ctx)),
        TyKind::Str => {
            let byte: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
            return Ok(llvm::types::ArrayType::get(ctx, byte, 0).into());
        }
        TyKind::Dynamic(_, _) => {
            let byte: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
            return Ok(llvm::types::ArrayType::get(ctx, byte, 0).into());
        }
        TyKind::Int(kind) => IntegerType::get(ctx, int_width(*kind), Signedness::Signed),
        TyKind::Uint(kind) => IntegerType::get(ctx, uint_width(*kind), Signedness::Unsigned),
        TyKind::Char => IntegerType::get(ctx, 32, Signedness::Unsigned),
        TyKind::Float(rustc_middle::ty::FloatTy::F32) => return Ok(FP32Type::get(ctx).into()),
        TyKind::Float(rustc_middle::ty::FloatTy::F64) => return Ok(FP64Type::get(ctx).into()),
        TyKind::FnPtr(_, _) => return Ok(llvm_ptr_ty(ctx)),
        TyKind::FnDef(_, _) => return Ok(unit_ty(ctx)),
        TyKind::Never => return Ok(unit_ty(ctx)),
        TyKind::Tuple(fields) if fields.is_empty() => return Ok(unit_ty(ctx)),
        TyKind::Tuple(fields) => {
            let source: Vec<_> = fields.iter().collect();
            let order = struct_memory_order(tcx, ty, source.len());
            let fields = order
                .into_iter()
                .map(|idx| convert_ty(tcx, ctx, source[idx]))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(llvm::types::StructType::get_unnamed(ctx, fields).into());
        }
        TyKind::Array(elem, len) => {
            let elem = convert_ty(tcx, ctx, *elem)?;
            let len = array_len(tcx, *len)?;
            return Ok(llvm::types::ArrayType::get(ctx, elem, len).into());
        }
        TyKind::Slice(elem) => {
            let elem = convert_ty(tcx, ctx, *elem)?;
            return Ok(llvm::types::ArrayType::get(ctx, elem, 0).into());
        }
        TyKind::Ref(_, inner, _) | TyKind::RawPtr(inner, _) => {
            return match pointee_unsized_tail(tcx, *inner) {
                Some(tail) => match runtime_ty(tail).kind() {
                    TyKind::Str => Ok(str_ref_ty(ctx)),
                    TyKind::Slice(elem) => {
                        let elem = convert_ty(tcx, ctx, *elem)?;
                        Ok(slice_ref_ty(ctx, elem))
                    }
                    TyKind::Dynamic(_, _) => Ok(trait_object_ref_ty(ctx)),
                    other => Err(format!("unsupported unsized pointee tail: {other:?}")),
                },
                None => Ok(llvm_ptr_ty(ctx)),
            };
        }
        TyKind::Adt(_, _)
            if format!("{ty:?}").starts_with("std::fmt::Arguments")
                || format!("{ty:?}").starts_with("core::fmt::Arguments") =>
        {
            return Ok(fmt_arguments_ty(ctx));
        }
        TyKind::Adt(adt_def, args) if adt_def.is_struct() => {
            let name = type_symbol(tcx, ty);
            // `get_named` with no fields returns the type if the name is
            // already registered, and reserves it (as an opaque struct)
            // otherwise; only proceed to build the body for opaque results.
            let reserved = llvm::types::StructType::get_named(ctx, name.clone(), None)
                .map_err(|error| error.to_string())?;
            if !reserved.deref(ctx).is_opaque() {
                return Ok(reserved.into());
            }
            let source: Vec<_> = adt_def
                .non_enum_variant()
                .fields
                .iter()
                .map(|field| field.ty(tcx, args))
                .collect();
            let order = struct_memory_order(tcx, ty, source.len());
            let fields = order
                .into_iter()
                .map(|idx| convert_ty(tcx, ctx, source[idx]))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(llvm::types::StructType::get_named(ctx, name, Some(fields))
                .map_err(|error| error.to_string())?
                .into());
        }
        TyKind::Adt(adt_def, args) if adt_def.is_union() => {
            let name = type_symbol(tcx, ty);
            // `get_named` with no fields returns the type if the name is
            // already registered, and reserves it (as an opaque struct)
            // otherwise; only proceed to build the body for opaque results.
            let reserved = llvm::types::StructType::get_named(ctx, name.clone(), None)
                .map_err(|error| error.to_string())?;
            if !reserved.deref(ctx).is_opaque() {
                return Ok(reserved.into());
            }
            let size = layout_size_of_ty(tcx, ty)?;
            let byte_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
            let storage: TypeHandle = if size == 0 {
                unit_ty(ctx)
            } else {
                llvm::types::ArrayType::get(ctx, byte_ty, size).into()
            };
            let _ = args;
            return Ok(
                llvm::types::StructType::get_named(ctx, name, Some(vec![storage]))
                    .map_err(|error| error.to_string())?
                    .into(),
            );
        }
        TyKind::Adt(adt_def, _) if is_fmt_rt_argument_type(tcx, adt_def.did()) => {
            // `core::fmt::rt::ArgumentType` is read by prebuilt std code, so
            // it must use the real niche layout: {value ptr, formatter fn ptr}.
            return Ok(fmt_arguments_ty(ctx));
        }
        TyKind::Adt(adt_def, _) if adt_def.is_enum() => {
            return enum_blob_ty(tcx, ctx, ty);
        }
        TyKind::Adt(_, args)
            if is_crabbit_device_wrapper_ty(ty, "DeviceSliceMut")
                || is_crabbit_device_wrapper_ty(ty, "DisjointSlice") =>
        {
            let elem = args[0].expect_ty();
            let elem = convert_ty(tcx, ctx, elem)?;
            return Ok(device_slice_ty(ctx, elem, true));
        }
        TyKind::Adt(_, args) if is_crabbit_device_wrapper_ty(ty, "DeviceSlice") => {
            let elem = args[0].expect_ty();
            let elem = convert_ty(tcx, ctx, elem)?;
            return Ok(device_slice_ty(ctx, elem, false));
        }
        TyKind::Closure(_, args) => {
            let name = type_symbol(tcx, ty);
            // `get_named` with no fields returns the type if the name is
            // already registered, and reserves it (as an opaque struct)
            // otherwise; only proceed to build the body for opaque results.
            let reserved = llvm::types::StructType::get_named(ctx, name.clone(), None)
                .map_err(|error| error.to_string())?;
            if !reserved.deref(ctx).is_opaque() {
                return Ok(reserved.into());
            }
            let source: Vec<_> = args.as_closure().upvar_tys().iter().collect();
            let order = struct_memory_order(tcx, ty, source.len());
            let fields = order
                .into_iter()
                .map(|idx| convert_ty(tcx, ctx, source[idx]))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(llvm::types::StructType::get_named(ctx, name, Some(fields))
                .map_err(|error| error.to_string())?
                .into());
        }
        other => return Err(format!("unsupported Rust type in MIR importer: {other:?}")),
    };
    Ok(ty.into())
}

/// The opaque, real-layout blob type for an enum: a named struct wrapping an
/// `[N x iM]` array sized and aligned to the enum's real rustc layout. Enum
/// values are simply their real bytes, so every discriminant read, variant
/// field access, and construction goes through real byte offsets (see
/// [`read_enum_discriminant`], [`field_offset_of`], and the `Rvalue::Aggregate`
/// enum path). This is what lets enums interoperate with prebuilt std code that
/// reads and writes them by reference.
pub(super) fn enum_blob_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    ty: Ty<'tcx>,
) -> Result<TypeHandle, String> {
    let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
    let size = rustc_layout_size_of_ty(tcx, typing_env, ty)?;
    if size == 0 {
        return Ok(unit_ty(ctx));
    }
    let name = type_symbol(tcx, ty);
    let reserved = llvm::types::StructType::get_named(ctx, name.clone(), None)
        .map_err(|error| error.to_string())?;
    if !reserved.deref(ctx).is_opaque() {
        return Ok(reserved.into());
    }
    let align = enum_stack_align(tcx, ty)?;
    let elem: TypeHandle = IntegerType::get(ctx, (align * 8) as u32, Signedness::Unsigned).into();
    let blob: TypeHandle = llvm::types::ArrayType::get(ctx, elem, size / align).into();
    Ok(
        llvm::types::StructType::get_named(ctx, name, Some(vec![blob]))
            .map_err(|error| error.to_string())?
            .into(),
    )
}

/// The backend stack alignment (capped at the 8-byte maximum the AArch64
/// lowering supports) of an enum's real rustc layout.
pub(super) fn enum_stack_align<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Result<u64, String> {
    let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
    Ok(rustc_layout_align_of_ty(tcx, typing_env, ty)?.clamp(1, 8))
}

/// Read an enum's discriminant from its real in-memory layout at `addr`,
/// producing a value of the converted discriminant type. Mirrors what rustc's
/// own codegen emits for `Rvalue::Discriminant`: a direct tag load for `Direct`
/// encoding, and the niche-decode arithmetic for `Niche` encoding.
pub(super) fn read_enum_discriminant<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    enum_ty: Ty<'tcx>,
    addr: Value,
) -> Result<Value, String> {
    let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
    let enum_ty = normalize_ty(tcx, runtime_ty(enum_ty));
    let rustc_middle::ty::TyKind::Adt(adt_def, _) = enum_ty.kind() else {
        return Err(format!("MIR discriminant of non-ADT type: {enum_ty:?}"));
    };
    let layout = tcx
        .layout_of(typing_env.as_query_input(enum_ty))
        .map_err(|error| format!("no layout for {enum_ty:?}: {error:?}"))?;
    let discr_ty = convert_ty(tcx, ctx, enum_ty.discriminant_ty(tcx))?;
    let discr_width = discr_ty
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .map(|ty| ty.width())
        .ok_or_else(|| "enum discriminant is not an integer".to_string())?;
    let discr_mask = if discr_width >= 128 {
        u128::MAX
    } else {
        (1u128 << discr_width) - 1
    };

    match &layout.variants {
        Variants::Empty => {
            // An uninhabited enum (e.g. `Result<Infallible, !>`) has no
            // values, so this read can never execute. Mirror rustc's cg_ssa,
            // which emits an arbitrary (poison) value: a zero constant of the
            // discriminant type, matching the non-enum branch of
            // `Rvalue::Discriminant`.
            let op = integer_constant(ctx, discr_ty, 0)?;
            op.get_operation().insert_at_back(insert_block, ctx);
            Ok(op.get_result(ctx))
        }
        Variants::Single { index } => {
            let value = adt_def.discriminant_for_variant(tcx, *index).val & discr_mask;
            let op = integer_constant(ctx, discr_ty, value)?;
            op.get_operation().insert_at_back(insert_block, ctx);
            Ok(op.get_result(ctx))
        }
        Variants::Multiple {
            tag,
            tag_encoding,
            tag_field,
            ..
        } => {
            let tag_offset = layout.fields.offset(tag_field.as_usize()).bytes();
            let tag_bits = tag.size(&tcx).bits() as u32;
            let tag_ty: TypeHandle = IntegerType::get(ctx, tag_bits, Signedness::Unsigned).into();
            let tag_addr = if tag_offset != 0 {
                ptr_offset_const(ctx, insert_block, addr, tag_offset)?
            } else {
                addr
            };
            let load = mir_dialect::ops::LoadOp::new(ctx, tag_addr, tag_ty);
            load.get_operation().insert_at_back(insert_block, ctx);
            let tag_value = load.get_result(ctx);
            match tag_encoding {
                TagEncoding::Direct => {
                    Ok(cast_value_to_type(ctx, insert_block, tag_value, discr_ty))
                }
                TagEncoding::Niche {
                    untagged_variant,
                    niche_variants,
                    niche_start,
                } => {
                    let tag_mask = if tag_bits >= 128 {
                        u128::MAX
                    } else {
                        (1u128 << tag_bits) - 1
                    };
                    let relative_max =
                        niche_variants.end().as_u32() - niche_variants.start().as_u32();

                    // relative_tag = tag - niche_start, wrapping in the tag width
                    // so the comparison below matches rustc's niche decoding.
                    let niche_start_const = integer_constant(ctx, tag_ty, niche_start & tag_mask)?;
                    niche_start_const
                        .get_operation()
                        .insert_at_back(insert_block, ctx);
                    let relative_tag = mir_dialect::ops::SubOp::new(
                        ctx,
                        tag_value,
                        niche_start_const.get_result(ctx),
                    )
                    .get_operation();
                    relative_tag.insert_at_back(insert_block, ctx);
                    let relative_tag = relative_tag.deref(ctx).get_result(0);

                    // is_niche = relative_tag <= relative_max (unsigned).
                    let relative_max_const =
                        integer_constant(ctx, tag_ty, relative_max as u128 & tag_mask)?;
                    relative_max_const
                        .get_operation()
                        .insert_at_back(insert_block, ctx);
                    let is_niche = mir_dialect::ops::LeOp::new(
                        ctx,
                        relative_tag,
                        relative_max_const.get_result(ctx),
                    );
                    is_niche.get_operation().insert_at_back(insert_block, ctx);
                    let is_niche =
                        cast_value_to_type(ctx, insert_block, is_niche.get_result(ctx), discr_ty);

                    // For niche encoding the discriminant equals the variant
                    // index, so the tagged discriminant is
                    // `niche_variants.start() + relative_tag`.
                    let relative_discr =
                        cast_value_to_type(ctx, insert_block, relative_tag, discr_ty);
                    let niche_base = integer_constant(
                        ctx,
                        discr_ty,
                        niche_variants.start().as_u32() as u128 & discr_mask,
                    )?;
                    niche_base.get_operation().insert_at_back(insert_block, ctx);
                    let tagged =
                        mir_dialect::ops::AddOp::new(ctx, niche_base.get_result(ctx), relative_discr)
                            .get_operation();
                    tagged.insert_at_back(insert_block, ctx);
                    let tagged = tagged.deref(ctx).get_result(0);

                    // discr = untagged + (tagged - untagged) * is_niche.
                    let untagged = adt_def.discriminant_for_variant(tcx, *untagged_variant).val;
                    let base = integer_constant(ctx, discr_ty, untagged & discr_mask)?;
                    base.get_operation().insert_at_back(insert_block, ctx);
                    let base = base.get_result(ctx);
                    let diff = mir_dialect::ops::SubOp::new(ctx, tagged, base).get_operation();
                    diff.insert_at_back(insert_block, ctx);
                    let diff = diff.deref(ctx).get_result(0);
                    let scaled = mir_dialect::ops::MulOp::new(ctx, diff, is_niche).get_operation();
                    scaled.insert_at_back(insert_block, ctx);
                    let scaled = scaled.deref(ctx).get_result(0);
                    let discr = mir_dialect::ops::AddOp::new(ctx, base, scaled).get_operation();
                    discr.insert_at_back(insert_block, ctx);
                    Ok(discr.deref(ctx).get_result(0))
                }
            }
        }
    }
}

/// Write the tag for `variant` into an enum stored with its real layout at
/// `addr`. A no-op when the variant needs no tag (a single-variant enum or the
/// untagged niche variant), mirroring rustc's `tag_for_variant`.
pub(super) fn write_enum_tag<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    enum_ty: Ty<'tcx>,
    variant: VariantIdx,
    addr: Value,
) -> Result<(), String> {
    let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
    let enum_ty = normalize_ty(tcx, runtime_ty(enum_ty));
    let layout = tcx
        .layout_of(typing_env.as_query_input(enum_ty))
        .map_err(|error| format!("no layout for {enum_ty:?}: {error:?}"))?;
    let Variants::Multiple { tag, tag_field, .. } = &layout.variants else {
        return Ok(());
    };
    let Some(tag_value) = tcx.tag_for_variant(typing_env.as_query_input((enum_ty, variant))) else {
        return Ok(());
    };
    let tag_bits = tag.size(&tcx).bits() as u32;
    let tag_ty: TypeHandle = IntegerType::get(ctx, tag_bits, Signedness::Unsigned).into();
    let bits = tag_value.to_uint(tag_value.size());
    let tag_const = integer_constant(ctx, tag_ty, bits)?;
    tag_const.get_operation().insert_at_back(insert_block, ctx);
    let tag_offset = layout.fields.offset(tag_field.as_usize()).bytes();
    let tag_addr = if tag_offset != 0 {
        ptr_offset_const(ctx, insert_block, addr, tag_offset)?
    } else {
        addr
    };
    let store = mir_dialect::ops::StoreOp::new(ctx, tag_const.get_result(ctx), tag_addr);
    store.get_operation().insert_at_back(insert_block, ctx);
    Ok(())
}

pub(super) fn type_symbol<'tcx>(_tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> crate::identifier::Identifier {
    // `{ty:?}` prints paths of local items without the crate name, so e.g.
    // arrow-buffer's own `bytes::Bytes` and the external `bytes` crate's
    // `Bytes` would print identically and alias one converted struct (with
    // whichever layout was converted first). `with_resolve_crate_name`
    // prefixes every path (local ones included) with the real crate name,
    // in generic-argument position too, keeping the symbols distinct.
    let printed = rustc_middle::ty::print::with_resolve_crate_name!(format!("{ty:?}"));
    let mut legaliser = Legaliser::default();
    legaliser.legalise(&printed)
}

pub(super) fn is_fmt_rt_argument_type<'tcx>(tcx: TyCtxt<'tcx>, def_id: rustc_hir::def_id::DefId) -> bool {
    tcx.def_path_str(def_id).contains("fmt::rt::ArgumentType")
}

pub(super) fn is_crabbit_device_wrapper_ty<'tcx>(ty: Ty<'tcx>, name: &str) -> bool {
    let ty = format!("{ty:?}");
    ty.starts_with(&format!("crabbit_device::{name}"))
        || ty.starts_with(&format!("crabbit_device::slice::{name}"))
}

pub(super) fn int_width(kind: rustc_middle::ty::IntTy) -> u32 {
    use rustc_middle::ty::IntTy;
    match kind {
        IntTy::I8 => 8,
        IntTy::I16 => 16,
        IntTy::I32 => 32,
        IntTy::I64 => 64,
        IntTy::I128 => 128,
        IntTy::Isize => 64,
    }
}

pub(super) fn uint_width(kind: rustc_middle::ty::UintTy) -> u32 {
    use rustc_middle::ty::UintTy;
    match kind {
        UintTy::U8 => 8,
        UintTy::U16 => 16,
        UintTy::U32 => 32,
        UintTy::U64 => 64,
        UintTy::U128 => 128,
        UintTy::Usize => 64,
    }
}
