use super::*;

pub(super) fn field_offset_of<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    variant: Option<VariantIdx>,
    index: usize,
) -> Result<u64, String> {
    let ty = runtime_ty(ty);
    let fields = match ty.kind() {
        rustc_middle::ty::TyKind::Adt(adt_def, _) if adt_def.is_enum() => {
            let variant = variant.ok_or_else(|| {
                format!("MIR enum field projection without variant downcast: {ty:?}.{index}")
            })?;
            if index >= adt_def.variant(variant).fields.len() {
                return Err(format!(
                    "MIR enum field index {index} out of bounds for variant {:?}",
                    adt_def.variant(variant).name
                ));
            }
            // Enums use the real rustc layout, so the field offset comes
            // straight from the variant's layout.
            let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
            let layout = tcx
                .layout_of(typing_env.as_query_input(ty))
                .map_err(|error| format!("no layout for {ty:?}: {error:?}"))?;
            let cx = rustc_middle::ty::layout::LayoutCx::new(tcx, typing_env);
            let variant_layout = layout.for_variant(&cx, variant);
            return Ok(variant_layout.fields.offset(index).bytes());
        }
        rustc_middle::ty::TyKind::Adt(adt_def, args) if adt_def.is_struct() => adt_def
            .non_enum_variant()
            .fields
            .iter()
            .map(|field| field.ty(tcx, args))
            .collect::<Vec<_>>(),
        rustc_middle::ty::TyKind::Adt(adt_def, args) if adt_def.is_union() => {
            let fields = adt_def
                .non_enum_variant()
                .fields
                .iter()
                .map(|field| field.ty(tcx, args))
                .collect::<Vec<_>>();
            if index >= fields.len() {
                return Err(format!(
                    "MIR union field index {index} out of bounds for {} fields",
                    fields.len()
                ));
            }
            return Ok(0);
        }
        rustc_middle::ty::TyKind::Tuple(fields) => fields.iter().collect::<Vec<_>>(),
        rustc_middle::ty::TyKind::Closure(_, args) => {
            args.as_closure().upvar_tys().iter().collect::<Vec<_>>()
        }
        other => {
            return Err(format!(
                "MIR field projection of non-aggregate type: {other:?}"
            ));
        }
    };
    if index >= fields.len() {
        return Err(format!(
            "MIR field index {index} out of bounds for {} fields",
            fields.len()
        ));
    }
    let order = struct_memory_order(tcx, ty, fields.len());
    let ordered: Vec<_> = order.iter().map(|&idx| fields[idx]).collect();
    let offsets = struct_like_field_offsets(tcx, &ordered)?;
    let position = field_memory_position(tcx, ty, fields.len(), index);
    Ok(offsets[position as usize])
}

/// Field offsets of a struct-like sequence of fields, aligned exactly the
/// way the AArch64 lowering lays out converted `llvm.struct` types.
pub(super) fn struct_like_field_offsets<'tcx>(
    tcx: TyCtxt<'tcx>,
    fields: &[Ty<'tcx>],
) -> Result<Vec<u64>, String> {
    let mut offset = 0u64;
    let mut offsets = Vec::with_capacity(fields.len());
    for field in fields {
        let size = layout_size_of_ty(tcx, *field)?;
        if size == 0 {
            offsets.push(offset);
            continue;
        }
        offset = align_to(offset, layout_align_of_ty(tcx, *field)?);
        offsets.push(offset);
        offset += size;
    }
    Ok(offsets)
}

pub(super) fn struct_like_size<'tcx>(tcx: TyCtxt<'tcx>, fields: &[Ty<'tcx>]) -> Result<u64, String> {
    let offsets = struct_like_field_offsets(tcx, fields)?;
    let mut align = 1u64;
    let mut unpadded = 0u64;
    for (field, offset) in fields.iter().zip(offsets) {
        let size = layout_size_of_ty(tcx, *field)?;
        if size == 0 {
            continue;
        }
        align = align.max(layout_align_of_ty(tcx, *field)?);
        unpadded = unpadded.max(offset + size);
    }
    Ok(align_to(unpadded, align))
}

pub(super) fn align_to(value: u64, align: u64) -> u64 {
    if align <= 1 {
        value
    } else {
        (value + align - 1) & !(align - 1)
    }
}

/// Alignment of the converted crabbit type for a Rust type, mirroring the
/// AArch64 lowering's `stack_align_of` rules.
pub(super) fn layout_align_of_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Result<u64, String> {
    let ty = normalize_ty(tcx, runtime_ty(ty));
    match ty.kind() {
        rustc_middle::ty::TyKind::Bool => Ok(1),
        rustc_middle::ty::TyKind::Char => Ok(4),
        rustc_middle::ty::TyKind::Str | rustc_middle::ty::TyKind::Dynamic(_, _) => Ok(1),
        rustc_middle::ty::TyKind::Int(kind) => {
            Ok((int_width(*kind) as u64).div_ceil(8).clamp(1, 8))
        }
        rustc_middle::ty::TyKind::Uint(kind) => {
            Ok((uint_width(*kind) as u64).div_ceil(8).clamp(1, 8))
        }
        rustc_middle::ty::TyKind::Float(rustc_middle::ty::FloatTy::F32) => Ok(4),
        rustc_middle::ty::TyKind::Float(_) => Ok(8),
        rustc_middle::ty::TyKind::FnPtr(_, _)
        | rustc_middle::ty::TyKind::Ref(_, _, _)
        | rustc_middle::ty::TyKind::RawPtr(_, _) => Ok(8),
        rustc_middle::ty::TyKind::FnDef(_, _) | rustc_middle::ty::TyKind::Never => Ok(1),
        rustc_middle::ty::TyKind::Tuple(fields) => {
            let fields: Vec<_> = fields.iter().collect();
            struct_like_align(tcx, &fields)
        }
        rustc_middle::ty::TyKind::Array(elem, _) | rustc_middle::ty::TyKind::Slice(elem) => {
            layout_align_of_ty(tcx, *elem)
        }
        rustc_middle::ty::TyKind::Closure(_, args) => {
            let fields: Vec<_> = args.as_closure().upvar_tys().iter().collect();
            struct_like_align(tcx, &fields)
        }
        rustc_middle::ty::TyKind::Adt(adt_def, _)
            if is_fmt_rt_argument_type(tcx, adt_def.did()) =>
        {
            Ok(8)
        }
        rustc_middle::ty::TyKind::Adt(adt_def, args) if adt_def.is_struct() => {
            let fields: Vec<_> = adt_def
                .non_enum_variant()
                .fields
                .iter()
                .map(|field| field.ty(tcx, args))
                .collect();
            struct_like_align(tcx, &fields)
        }
        // Unions convert to a byte-array wrapper, which the lowering treats
        // as alignment 1.
        rustc_middle::ty::TyKind::Adt(adt_def, _) if adt_def.is_union() => Ok(1),
        rustc_middle::ty::TyKind::Adt(adt_def, _) if adt_def.is_enum() => enum_stack_align(tcx, ty),
        other => Err(format!(
            "unsupported Rust type alignment in MIR importer: {other:?}"
        )),
    }
}

pub(super) fn struct_like_align<'tcx>(tcx: TyCtxt<'tcx>, fields: &[Ty<'tcx>]) -> Result<u64, String> {
    let mut align = 1u64;
    for field in fields {
        if layout_size_of_ty(tcx, *field)? == 0 {
            continue;
        }
        align = align.max(layout_align_of_ty(tcx, *field)?);
    }
    Ok(align)
}

pub(super) fn layout_size_of_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Result<u64, String> {
    let ty = normalize_ty(tcx, runtime_ty(ty));
    match ty.kind() {
        rustc_middle::ty::TyKind::Bool => Ok(1),
        rustc_middle::ty::TyKind::Char => Ok(4),
        rustc_middle::ty::TyKind::Str => Ok(0),
        rustc_middle::ty::TyKind::Dynamic(_, _) => Ok(0),
        rustc_middle::ty::TyKind::Int(kind) => Ok((int_width(*kind) as u64).div_ceil(8)),
        rustc_middle::ty::TyKind::Uint(kind) => Ok((uint_width(*kind) as u64).div_ceil(8)),
        rustc_middle::ty::TyKind::Float(rustc_middle::ty::FloatTy::F32) => Ok(4),
        rustc_middle::ty::TyKind::Float(rustc_middle::ty::FloatTy::F64) => Ok(8),
        rustc_middle::ty::TyKind::FnPtr(_, _) => Ok(8),
        rustc_middle::ty::TyKind::FnDef(_, _) | rustc_middle::ty::TyKind::Never => Ok(0),
        rustc_middle::ty::TyKind::Ref(_, inner, _) | rustc_middle::ty::TyKind::RawPtr(inner, _) => {
            // Fat references convert to a two-word {ptr, meta} struct.
            Ok(if pointee_unsized_tail(tcx, *inner).is_some() {
                16
            } else {
                8
            })
        }
        rustc_middle::ty::TyKind::Tuple(_) => {
            let fields = memory_ordered_fields(tcx, ty)?;
            struct_like_size(tcx, &fields)
        }
        rustc_middle::ty::TyKind::Array(elem, len) => {
            let len = array_len(tcx, *len)?;
            let elem_size = layout_size_of_ty(tcx, *elem)?;
            let stride = align_to(elem_size, layout_align_of_ty(tcx, *elem)?);
            Ok(stride * len)
        }
        rustc_middle::ty::TyKind::Slice(elem) => layout_size_of_ty(tcx, *elem),
        rustc_middle::ty::TyKind::Adt(adt_def, _)
            if is_fmt_rt_argument_type(tcx, adt_def.did()) =>
        {
            Ok(16)
        }
        rustc_middle::ty::TyKind::Adt(adt_def, _) if adt_def.is_struct() => {
            let fields = memory_ordered_fields(tcx, ty)?;
            struct_like_size(tcx, &fields)
        }
        rustc_middle::ty::TyKind::Adt(adt_def, args) if adt_def.is_union() => adt_def
            .non_enum_variant()
            .fields
            .iter()
            .try_fold(0u64, |size, field| {
                Ok(size.max(layout_size_of_ty(tcx, field.ty(tcx, args))?))
            }),
        rustc_middle::ty::TyKind::Adt(adt_def, _) if adt_def.is_enum() => {
            let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
            rustc_layout_size_of_ty(tcx, typing_env, ty)
        }
        rustc_middle::ty::TyKind::Closure(_, _) => {
            let fields = memory_ordered_fields(tcx, ty)?;
            struct_like_size(tcx, &fields)
        }
        other => Err(format!(
            "unsupported Rust type layout in MIR importer: {other:?}"
        )),
    }
}

/// The struct-like fields of `ty` in converted (memory) order.
pub(super) fn memory_ordered_fields<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Result<Vec<Ty<'tcx>>, String> {
    let source = struct_like_source_fields(tcx, ty)
        .ok_or_else(|| format!("expected struct-like type: {ty:?}"))?;
    let order = struct_memory_order(tcx, ty, source.len());
    Ok(order.into_iter().map(|idx| source[idx]).collect())
}

pub(super) fn array_len<'tcx>(tcx: TyCtxt<'tcx>, len: rustc_middle::ty::Const<'tcx>) -> Result<u64, String> {
    if let Some(value) = len.try_to_target_usize(tcx) {
        return Ok(value);
    }
    let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
    let normalized = tcx
        .try_normalize_erasing_regions(typing_env, len)
        .map_err(|error| format!("unsupported non-constant array length: {len:?}: {error:?}"))?;
    normalized
        .try_to_target_usize(tcx)
        .ok_or_else(|| format!("unsupported non-constant array length: {len:?}"))
}
