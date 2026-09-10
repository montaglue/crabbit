use super::*;

pub(super) fn import_rvalue<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    rvalue: &Rvalue<'tcx>,
) -> Result<Value, String> {
    match rvalue {
        Rvalue::Use(operand) => import_operand(tcx, ctx, state, insert_block, body, operand),
        Rvalue::Ref(_, _, place) | Rvalue::RawPtr(_, place) => {
            place_addr(tcx, ctx, state, insert_block, body, place)
        }
        Rvalue::ThreadLocalRef(def_id) => {
            // The address of the current thread's copy of a `#[thread_local]`
            // static: emitted like a static's address, but the global carries
            // the `ll.tls` marker so the backend goes through the thread
            // pointer instead of a data-section relocation.
            let mut legaliser = Legaliser::default();
            let symbol = legaliser.legalise(tcx.symbol_name(Instance::mono(tcx, *def_id)).name);
            declare_thread_local_global(tcx, ctx, state.module_body, symbol.clone(), *def_id)?;
            let ty = convert_immediate_ty(tcx, ctx, mono_ty(tcx, state, rvalue.ty(body, tcx)))?;
            let op = mir_dialect::ops::AddressOfOp::new(ctx, symbol, ty);
            op.get_operation().insert_at_back(insert_block, ctx);
            Ok(op.get_result(ctx))
        }
        Rvalue::BinaryOp(op, operands) => {
            let lhs = import_operand(tcx, ctx, state, insert_block, body, &operands.0)?;
            let rhs = import_operand(tcx, ctx, state, insert_block, body, &operands.1)?;
            if matches!(op, BinOp::Offset) {
                let lhs_ty = mono_ty(tcx, state, operands.0.ty(body, tcx));
                let elem_size = layout_size_of_ty(tcx, pointee_ty(lhs_ty)?)?;
                let byte_offset = scale_index(ctx, insert_block, rhs, elem_size)?;
                let offset = mir_dialect::ops::PtrOffsetOp::new(ctx, lhs, byte_offset);
                offset.get_operation().insert_at_back(insert_block, ctx);
                return Ok(offset.get_result(ctx));
            }
            if matches!(
                op,
                BinOp::AddWithOverflow | BinOp::SubWithOverflow | BinOp::MulWithOverflow
            ) {
                return lower_overflow_binary(ctx, insert_block, *op, lhs, rhs);
            }
            if matches!(op, BinOp::Cmp) {
                let ordering_ty = mono_ty(tcx, state, rvalue.ty(body, tcx));
                return lower_three_way_cmp(tcx, ctx, insert_block, ordering_ty, lhs, rhs);
            }
            if matches!(op, BinOp::Div | BinOp::Rem) && is_128_bit_integer_value(ctx, lhs) {
                return lower_i128_divrem(ctx, state.module_body, insert_block, *op, lhs, rhs);
            }
            let op = match op {
                BinOp::Add | BinOp::AddUnchecked => {
                    mir_dialect::ops::AddOp::new(ctx, lhs, rhs).get_operation()
                }
                BinOp::Sub | BinOp::SubUnchecked => {
                    mir_dialect::ops::SubOp::new(ctx, lhs, rhs).get_operation()
                }
                BinOp::Mul | BinOp::MulUnchecked => {
                    mir_dialect::ops::MulOp::new(ctx, lhs, rhs).get_operation()
                }
                BinOp::Shr | BinOp::ShrUnchecked => {
                    mir_dialect::ops::SignAwareShrOp::new(ctx, lhs, rhs).get_operation()
                }
                BinOp::Shl | BinOp::ShlUnchecked => {
                    mir_dialect::ops::ShlOp::new(ctx, lhs, rhs).get_operation()
                }
                BinOp::Div => mir_dialect::ops::DivOp::new(ctx, lhs, rhs).get_operation(),
                BinOp::Rem => mir_dialect::ops::RemOp::new(ctx, lhs, rhs).get_operation(),
                BinOp::BitAnd => mir_dialect::ops::BitAndOp::new(ctx, lhs, rhs).get_operation(),
                BinOp::BitOr => mir_dialect::ops::BitOrOp::new(ctx, lhs, rhs).get_operation(),
                BinOp::BitXor => mir_dialect::ops::BitXorOp::new(ctx, lhs, rhs).get_operation(),
                BinOp::Eq => mir_dialect::ops::EqOp::new(ctx, lhs, rhs).get_operation(),
                BinOp::Ne => mir_dialect::ops::NeOp::new(ctx, lhs, rhs).get_operation(),
                BinOp::Lt => mir_dialect::ops::LtOp::new(ctx, lhs, rhs).get_operation(),
                BinOp::Le => mir_dialect::ops::LeOp::new(ctx, lhs, rhs).get_operation(),
                BinOp::Gt => mir_dialect::ops::GtOp::new(ctx, lhs, rhs).get_operation(),
                BinOp::Ge => mir_dialect::ops::GeOp::new(ctx, lhs, rhs).get_operation(),
                other => return Err(format!("unsupported MIR binary op: {other:?}")),
            };
            op.insert_at_back(insert_block, ctx);
            Ok(op.deref(ctx).get_result(0))
        }
        Rvalue::Cast(kind, operand, ty) => {
            if matches!(
                kind,
                rustc_mir::CastKind::PointerCoercion(
                    rustc_middle::ty::adjustment::PointerCoercion::ReifyFnPointer(_),
                    _
                )
            ) {
                return reify_fn_pointer(tcx, ctx, state, insert_block, body, operand);
            }
            if matches!(kind, rustc_mir::CastKind::Transmute) {
                return import_transmute(tcx, ctx, state, insert_block, body, operand, *ty);
            }
            if let Some(value) =
                lower_pointer_unsize_cast(tcx, ctx, state, insert_block, body, kind, operand, *ty)?
            {
                return Ok(value);
            }
            let src_ty = mono_ty(tcx, state, operand.ty(body, tcx));
            let dst_ty = mono_ty(tcx, state, *ty);
            if let Some(value) = lower_128_bit_float_cast(
                tcx,
                ctx,
                state,
                insert_block,
                body,
                operand,
                src_ty,
                dst_ty,
            )? {
                return Ok(value);
            }
            let input = import_operand(tcx, ctx, state, insert_block, body, operand)?;
            let result_type = convert_immediate_ty(tcx, ctx, mono_ty(tcx, state, *ty))?;
            let cast = mir_dialect::ops::CastOp::new(ctx, input, result_type);
            cast.get_operation().insert_at_back(insert_block, ctx);
            Ok(cast.get_result(ctx))
        }
        Rvalue::UnaryOp(rustc_mir::UnOp::Neg, operand) => {
            let input = import_operand(tcx, ctx, state, insert_block, body, operand)?;
            let neg = mir_dialect::ops::NegOp::new(ctx, input);
            neg.get_operation().insert_at_back(insert_block, ctx);
            Ok(neg.get_result(ctx))
        }
        Rvalue::UnaryOp(rustc_mir::UnOp::Not, operand) => {
            let input = import_operand(tcx, ctx, state, insert_block, body, operand)?;
            let input_ty = input.get_type(ctx);
            let width = input_ty
                .deref(ctx)
                .downcast_ref::<IntegerType>()
                .map(|ty| ty.width())
                .ok_or_else(|| format!("unsupported non-integer MIR not: {rvalue:?}"))?;
            if width == 1 {
                let false_value = integer_constant(ctx, input_ty, 0)?;
                false_value
                    .get_operation()
                    .insert_at_back(insert_block, ctx);
                let eq = mir_dialect::ops::EqOp::new(ctx, input, false_value.get_result(ctx));
                eq.get_operation().insert_at_back(insert_block, ctx);
                return Ok(eq.get_result(ctx));
            }
            let ones = integer_constant(ctx, input_ty, u128::MAX >> (128 - width as usize))?;
            ones.get_operation().insert_at_back(insert_block, ctx);
            let xor = mir_dialect::ops::BitXorOp::new(ctx, input, ones.get_result(ctx));
            xor.get_operation().insert_at_back(insert_block, ctx);
            Ok(xor.get_result(ctx))
        }
        Rvalue::UnaryOp(rustc_mir::UnOp::PtrMetadata, operand) => {
            let value = import_operand(tcx, ctx, state, insert_block, body, operand)?;
            let value_ty = value.get_type(ctx);
            let result_ty = {
                let value_ty_ref = value_ty.deref(ctx);
                let Some(struct_ty) = value_ty_ref.downcast_ref::<llvm::types::StructType>() else {
                    return Err(format!(
                        "unsupported MIR pointer metadata operand type: {}",
                        value_ty_ref.disp(ctx)
                    ));
                };
                if struct_ty.num_fields() < 2 {
                    return Err(format!(
                        "unsupported MIR pointer metadata operand shape: {}",
                        value_ty_ref.disp(ctx)
                    ));
                }
                
                struct_ty.field_type(1)
            };
            let extract = mir_dialect::ops::ExtractValueOp::new(ctx, value, vec![1], result_ty);
            extract.get_operation().insert_at_back(insert_block, ctx);
            Ok(extract.get_result(ctx))
        }
        Rvalue::Discriminant(place) => {
            let enum_ty = mono_ty(tcx, state, place.ty(body, tcx).ty);
            if !is_enum_ty(enum_ty) {
                // A non-enum place (e.g. a struct or single-variant ADT) has a
                // discriminant that is always zero.
                let discr_ty = convert_ty(tcx, ctx, enum_ty.discriminant_ty(tcx))?;
                let zero = integer_constant(ctx, discr_ty, 0)?;
                zero.get_operation().insert_at_back(insert_block, ctx);
                return Ok(zero.get_result(ctx));
            }
            let slot = place_addr(tcx, ctx, state, insert_block, body, place)?;
            read_enum_discriminant(tcx, ctx, insert_block, enum_ty, slot)
        }
        Rvalue::Aggregate(kind, operands) => {
            let operand_tys = operands
                .iter()
                .map(|operand| mono_ty(tcx, state, operand.ty(body, tcx)))
                .collect::<Vec<_>>();
            let aggregate_ty = aggregate_result_ty(tcx, ctx, state, kind, &operand_tys)?;
            if let rustc_mir::AggregateKind::RawPtr(pointee_ty, mutability) = **kind {
                if operands.len() != 2 {
                    return Err(format!(
                        "unsupported MIR raw pointer aggregate operand count: {}",
                        operands.len()
                    ));
                }
                let mut operands = operands.iter();
                let data_operand = operands.next().expect("operand count checked above");
                let metadata_operand = operands.next().expect("operand count checked above");
                let data = import_operand(tcx, ctx, state, insert_block, body, data_operand)?;
                let metadata =
                    import_operand(tcx, ctx, state, insert_block, body, metadata_operand)?;
                let pointee_ty = mono_ty(tcx, state, pointee_ty);
                if matches!(
                    runtime_ty(pointee_ty).kind(),
                    rustc_middle::ty::TyKind::Slice(_)
                ) {
                    let ptr_ty = llvm_ptr_ty(ctx);
                    let data = if data.get_type(ctx) == ptr_ty {
                        data
                    } else {
                        let cast = mir_dialect::ops::CastOp::new(ctx, data, ptr_ty);
                        cast.get_operation().insert_at_back(insert_block, ctx);
                        cast.get_result(ctx)
                    };
                    let undef = mir_dialect::ops::UndefOp::new(ctx, aggregate_ty);
                    undef.get_operation().insert_at_back(insert_block, ctx);
                    let with_data = mir_dialect::ops::InsertValueOp::new(
                        ctx,
                        data,
                        undef.get_result(ctx),
                        vec![0],
                    );
                    with_data.get_operation().insert_at_back(insert_block, ctx);
                    let with_metadata = mir_dialect::ops::InsertValueOp::new(
                        ctx,
                        metadata,
                        with_data.get_result(ctx),
                        vec![1],
                    );
                    with_metadata
                        .get_operation()
                        .insert_at_back(insert_block, ctx);
                    return Ok(with_metadata.get_result(ctx));
                }
                let _ = mutability;
                return Ok(data);
            }
            let undef = mir_dialect::ops::UndefOp::new(ctx, aggregate_ty);
            undef.get_operation().insert_at_back(insert_block, ctx);
            let mut current = undef.get_result(ctx);

            if aggregate_is_union(tcx, kind) {
                if let Some(operand) = operands.iter().next()
                    && layout_size_of_ty(tcx, mono_ty(tcx, state, operand.ty(body, tcx)))? != 0
                {
                    return import_operand(tcx, ctx, state, insert_block, body, operand);
                }
                return Ok(current);
            }

            if let rustc_mir::AggregateKind::Adt(def_id, variant_idx, _, _, _) = **kind
                && is_fmt_rt_argument_type(tcx, def_id)
            {
                let variant = tcx.adt_def(def_id).variant(variant_idx).name.to_string();
                let ptr_ty = llvm_ptr_ty(ctx);
                let (value, formatter) = match variant.as_str() {
                    "Placeholder" => {
                        let mut operands = operands.iter();
                        let value_operand = operands.next().ok_or_else(|| {
                            "fmt Placeholder aggregate is missing its value".to_string()
                        })?;
                        let formatter_operand = operands.next().ok_or_else(|| {
                            "fmt Placeholder aggregate is missing its formatter".to_string()
                        })?;
                        let value =
                            import_operand(tcx, ctx, state, insert_block, body, value_operand)?;
                        let formatter =
                            import_operand(tcx, ctx, state, insert_block, body, formatter_operand)?;
                        (
                            cast_value_to_type(ctx, insert_block, value, ptr_ty),
                            cast_value_to_type(ctx, insert_block, formatter, ptr_ty),
                        )
                    }
                    "Count" => {
                        let count_operand = operands.iter().next().ok_or_else(|| {
                            "fmt Count aggregate is missing its count".to_string()
                        })?;
                        let count =
                            import_operand(tcx, ctx, state, insert_block, body, count_operand)?;
                        let usize_ty: TypeHandle = usize_ty(ctx).into();
                        let count = cast_value_to_type(ctx, insert_block, count, usize_ty);
                        let null = integer_constant(ctx, usize_ty, 0)?;
                        null.get_operation().insert_at_back(insert_block, ctx);
                        let null =
                            cast_value_to_type(ctx, insert_block, null.get_result(ctx), ptr_ty);
                        (cast_value_to_type(ctx, insert_block, count, ptr_ty), null)
                    }
                    other => {
                        return Err(format!("unsupported fmt ArgumentType variant: {other}"));
                    }
                };
                let undef = mir_dialect::ops::UndefOp::new(ctx, aggregate_ty);
                undef.get_operation().insert_at_back(insert_block, ctx);
                let with_value =
                    mir_dialect::ops::InsertValueOp::new(ctx, value, undef.get_result(ctx), vec![0]);
                with_value.get_operation().insert_at_back(insert_block, ctx);
                let with_formatter = mir_dialect::ops::InsertValueOp::new(
                    ctx,
                    formatter,
                    with_value.get_result(ctx),
                    vec![1],
                );
                with_formatter
                    .get_operation()
                    .insert_at_back(insert_block, ctx);
                return Ok(with_formatter.get_result(ctx));
            }

            if let rustc_mir::AggregateKind::Adt(def_id, variant_idx, args, _, _) = **kind
                && tcx.adt_def(def_id).is_enum()
            {
                let enum_ty = mono_ty(tcx, state, Ty::new_adt(tcx, tcx.adt_def(def_id), args));
                let blob_ty = convert_ty(tcx, ctx, enum_ty)?;
                if is_unit_converted_ty(ctx, blob_ty) {
                    // Zero-sized enum (uninhabited or a single ZST variant).
                    return Ok(current);
                }

                // Materialize the enum in a stack temporary using its real
                // layout, then load it back: an enum value simply *is* its real
                // in-memory bytes. This keeps it interoperable with prebuilt std
                // code that reads and writes enums by reference.
                let slot = mir_dialect::ops::AllocaOp::new(ctx, blob_ty);
                slot.get_operation().insert_at_back(insert_block, ctx);
                let slot = slot.get_result(ctx);
                write_enum_tag(tcx, ctx, insert_block, enum_ty, variant_idx, slot)?;

                let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
                let cx = rustc_middle::ty::layout::LayoutCx::new(tcx, typing_env);
                let layout = tcx
                    .layout_of(typing_env.as_query_input(enum_ty))
                    .map_err(|error| format!("no layout for {enum_ty:?}: {error:?}"))?;
                let variant_layout = layout.for_variant(&cx, variant_idx);
                for (idx, operand) in operands.iter().enumerate() {
                    let field_ty = mono_ty(tcx, state, operand.ty(body, tcx));
                    if layout_size_of_ty(tcx, field_ty)? == 0 {
                        continue;
                    }
                    let offset = variant_layout.fields.offset(idx).bytes();
                    let field_addr = if offset != 0 {
                        ptr_offset_const(ctx, insert_block, slot, offset)?
                    } else {
                        slot
                    };
                    let value = import_operand(tcx, ctx, state, insert_block, body, operand)?;
                    let value =
                        normalize_bool_for_storage(tcx, ctx, state, insert_block, field_ty, value)?;
                    let conv_field_ty = convert_ty(tcx, ctx, field_ty)?;
                    let value = cast_value_to_type(ctx, insert_block, value, conv_field_ty);
                    let store = mir_dialect::ops::StoreOp::new(ctx, value, field_addr);
                    store.get_operation().insert_at_back(insert_block, ctx);
                }

                let load = mir_dialect::ops::LoadOp::new(ctx, slot, blob_ty);
                load.get_operation().insert_at_back(insert_block, ctx);
                return Ok(load.get_result(ctx));
            }

            let rust_aggregate_ty = match **kind {
                rustc_mir::AggregateKind::Adt(def_id, _, adt_args, _, _) => Some(mono_ty(
                    tcx,
                    state,
                    Ty::new_adt(tcx, tcx.adt_def(def_id), adt_args),
                )),
                rustc_mir::AggregateKind::Closure(def_id, closure_args) => Some(mono_ty(
                    tcx,
                    state,
                    Ty::new_closure(tcx, def_id, closure_args),
                )),
                rustc_mir::AggregateKind::Tuple => Some(Ty::new_tup(tcx, &operand_tys)),
                _ => None,
            };
            for (idx, operand) in operands.iter().enumerate() {
                let index = match rust_aggregate_ty {
                    Some(aggregate) => converted_field_index(tcx, aggregate, idx)?,
                    None => idx as u32,
                };
                let value = import_operand(tcx, ctx, state, insert_block, body, operand)?;
                let field_ty = aggregate_field_type(ctx, current.get_type(ctx), index as usize)?;
                let value = cast_value_to_type(ctx, insert_block, value, field_ty);
                let insert = mir_dialect::ops::InsertValueOp::new(ctx, value, current, vec![index]);
                insert.get_operation().insert_at_back(insert_block, ctx);
                current = insert.get_result(ctx);
            }

            Ok(current)
        }
        Rvalue::Repeat(operand, len) => {
            let value = import_operand(tcx, ctx, state, insert_block, body, operand)?;
            let elem_ty = mono_ty(tcx, state, operand.ty(body, tcx));
            let elem_ty = convert_ty(tcx, ctx, elem_ty)?;
            let len = array_len(tcx, mono_ty_const(tcx, state, *len))?;
            let aggregate_ty = llvm::types::ArrayType::get(ctx, elem_ty, len).into();
            let undef = mir_dialect::ops::UndefOp::new(ctx, aggregate_ty);
            undef.get_operation().insert_at_back(insert_block, ctx);
            let mut current = undef.get_result(ctx);
            let value = cast_value_to_type(ctx, insert_block, value, elem_ty);
            for idx in 0..len {
                let insert =
                    mir_dialect::ops::InsertValueOp::new(ctx, value, current, vec![idx as u32]);
                insert.get_operation().insert_at_back(insert_block, ctx);
                current = insert.get_result(ctx);
            }
            Ok(current)
        }
        other => Err(format!("unsupported MIR rvalue: {other:?}")),
    }
}

/// Lower `ReifyFnPointer`: materialize the address of the target function.
/// When the target's MIR is available it is imported into this module and
/// addressed directly; otherwise a local thunk that forwards to the external
/// symbol is synthesized, so no data relocations are required.
pub(super) fn reify_fn_pointer<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    operand: &Operand<'tcx>,
) -> Result<Value, String> {
    let fn_ty = mono_ty(tcx, state, operand.ty(body, tcx));
    let rustc_middle::ty::TyKind::FnDef(def_id, args) = fn_ty.kind() else {
        return Err(format!("unsupported reified fn pointer source: {fn_ty:?}"));
    };
    let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
    let instance = Instance::resolve_for_fn_ptr(tcx, typing_env, *def_id, args)
        .ok_or_else(|| format!("cannot resolve reified fn pointer target: {fn_ty:?}"))?;
    let mut legaliser = Legaliser::default();
    let symbol = legaliser.legalise(tcx.symbol_name(instance).name);

    let target_symbol = if should_import_instance(tcx, instance) {
        import_upstream_instance(tcx, ctx, state.module_body, instance)
            .map_err(|error| format!("while importing reified fn {symbol}: {error}"))?;
        symbol
    } else {
        emit_fn_ptr_thunk(tcx, ctx, state.module_body, symbol, instance)?
    };

    let ptr_ty = llvm_ptr_ty(ctx);
    let op = mir_dialect::ops::AddressOfOp::new(ctx, target_symbol, ptr_ty);
    op.get_operation().insert_at_back(insert_block, ctx);
    Ok(op.get_result(ctx))
}

/// Create a local function that forwards all arguments to an external symbol.
/// Taking its address only needs pc-relative addressing within the module.
pub(super) fn emit_fn_ptr_thunk<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    module_body: Ptr<BasicBlock>,
    symbol: crate::identifier::Identifier,
    instance: Instance<'tcx>,
) -> Result<crate::identifier::Identifier, String> {
    let mut legaliser = Legaliser::default();
    let thunk_symbol = legaliser.legalise(&format!("{symbol}__crabbit_fnptr_thunk"));
    if symbol_exists(ctx, module_body, thunk_symbol.as_ref()) {
        return Ok(thunk_symbol);
    }

    let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
    let fn_sig = tcx
        .fn_sig(instance.def.def_id())
        .instantiate(tcx, instance.args);
    let fn_sig = tcx.normalize_erasing_regions(typing_env, fn_sig);
    let fn_sig = tcx.instantiate_bound_regions_with_erased(fn_sig);

    let mut inputs = Vec::new();
    for input_ty in fn_sig.inputs() {
        let abi_ty = convert_immediate_ty(tcx, ctx, *input_ty)?;
        match arg_abi_for_ty(ctx, abi_ty)? {
            ArgAbi::Leaves(leaves) => inputs.extend(leaves.iter().map(|(_, ty)| *ty)),
            ArgAbi::Indirect => inputs.push(llvm_ptr_ty(ctx)),
        }
    }
    let results = convert_return_ty(tcx, ctx, fn_sig.output())?;
    let result_ty = match results.as_slice() {
        [] => None,
        [ty] => Some(*ty),
        _ => return Err("unsupported multi-value reified fn result".to_string()),
    };

    declare_external_function(ctx, module_body, symbol.clone(), inputs.clone(), result_ty);

    let fn_ty = FunctionType::get(ctx, inputs, results);
    let func = mir_dialect::ops::FuncOp::new(ctx, thunk_symbol.clone(), fn_ty);
    func.get_operation().insert_at_back(module_body, ctx);
    let entry = func.get_entry_block(ctx);
    let args: Vec<Value> = entry.deref(ctx).arguments().collect();
    let call = mir_dialect::ops::CallOp::new_direct(ctx, symbol, args, result_ty);
    call.get_operation().insert_at_back(entry, ctx);
    let ret_val = result_ty.map(|_| call.get_operation().deref(ctx).get_result(0));
    let ret = mir_dialect::ops::ReturnOp::new(ctx, ret_val);
    ret.get_operation().insert_at_back(entry, ctx);
    set_internal_linkage(ctx, module_body, &thunk_symbol);
    Ok(thunk_symbol)
}

/// Lower `CastKind::Transmute`. Transmute semantics are defined on the real
/// rustc layout, which matches this importer's representation for scalars
/// and (memory-ordered) structs but not for enums.
pub(super) fn import_transmute<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    operand: &Operand<'tcx>,
    target: Ty<'tcx>,
) -> Result<Value, String> {
    let src_ty = mono_ty(tcx, state, operand.ty(body, tcx));
    let dst_ty = mono_ty(tcx, state, target);
    let value = import_operand(tcx, ctx, state, insert_block, body, operand)?;

    if !is_enum_ty(src_ty) && !is_enum_ty(dst_ty) {
        let result_ty = convert_immediate_ty(tcx, ctx, dst_ty)?;
        if value.get_type(ctx) == result_ty {
            return Ok(value);
        }
        // A direct cast is only meaningful between scalar (integer, pointer,
        // float) representations. Aggregate reinterpretations (e.g.
        // `[T; N] -> [MaybeUninit<T>; N]`) must go through memory below: a
        // struct-to-struct "bitcast" does not remap the value's pieces.
        let is_scalar = |ctx: &Context, ty: TypeHandle| {
            let ty_ref = ty.deref(ctx);
            ty_ref.is::<IntegerType>()
                || ty_ref.is::<llvm::types::PointerType>()
                || ty_ref.is::<FP32Type>()
                || ty_ref.is::<FP64Type>()
        };
        if is_scalar(ctx, value.get_type(ctx)) && is_scalar(ctx, result_ty) {
            let cast = mir_dialect::ops::CastOp::new(ctx, value, result_ty);
            cast.get_operation().insert_at_back(insert_block, ctx);
            return Ok(cast.get_result(ctx));
        }
    }

    // Any other transmute is a pure reinterpretation of bytes. Since every
    // converted type shares rustc's real memory layout, spill the source
    // value and reload it as the destination type.
    let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
    let src_size = rustc_layout_size_of_ty(tcx, typing_env, src_ty)?;
    let dst_size = rustc_layout_size_of_ty(tcx, typing_env, dst_ty)?;
    let word_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let words = src_size.max(dst_size).div_ceil(8).max(1);
    let blob_ty: TypeHandle = llvm::types::ArrayType::get(ctx, word_ty, words).into();
    let slot = mir_dialect::ops::AllocaOp::new(ctx, blob_ty);
    slot.get_operation().insert_at_back(insert_block, ctx);
    let slot = slot.get_result(ctx);
    let store = mir_dialect::ops::StoreOp::new(ctx, value, slot);
    store.get_operation().insert_at_back(insert_block, ctx);
    load_value_from_real_layout(tcx, ctx, insert_block, dst_ty, slot)
}

// Threads the importer's per-function lowering state; a parameter struct
// would be packed and unpacked at every call site for no clarity gain.
#[allow(clippy::too_many_arguments)]
pub(super) fn lower_pointer_unsize_cast<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    kind: &rustc_mir::CastKind,
    operand: &Operand<'tcx>,
    target_ty: Ty<'tcx>,
) -> Result<Option<Value>, String> {
    if !matches!(
        kind,
        rustc_mir::CastKind::PointerCoercion(
            rustc_middle::ty::adjustment::PointerCoercion::Unsize,
            _
        )
    ) {
        return Ok(None);
    }

    let source_ty = mono_ty(tcx, state, operand.ty(body, tcx));
    let target_ty = mono_ty(tcx, state, target_ty);
    let Some(metadata) = unsize_metadata(tcx, source_ty, target_ty)? else {
        return Ok(None);
    };

    let input = import_operand(tcx, ctx, state, insert_block, body, operand)?;
    let ptr_ty = llvm_ptr_ty(ctx);

    // The metadata half of the resulting fat pointer.
    let metadata_value: Value = match metadata {
        UnsizeMetadata::SliceLen(len) => {
            let usize_ty: TypeHandle = usize_ty(ctx).into();
            let len_op = integer_constant(ctx, usize_ty, len as u128)?;
            len_op.get_operation().insert_at_back(insert_block, ctx);
            len_op.get_result(ctx)
        }
        UnsizeMetadata::Vtable {
            concrete,
            principal,
        } => {
            // The same vtable rustc's own codegen would reference: emit its
            // allocation as an internal data global and take its address.
            let vtable_alloc_id = tcx.vtable_allocation((concrete, principal));
            let rustc_mir::interpret::GlobalAlloc::Memory(alloc) =
                tcx.global_alloc(vtable_alloc_id)
            else {
                return Err(format!(
                    "vtable allocation for {concrete:?} is not a memory allocation"
                ));
            };
            let mut legaliser = Legaliser::default();
            let symbol = legaliser.legalise(&format!("__crabbit_{vtable_alloc_id:?}"));
            emit_allocation_global(
                tcx,
                ctx,
                state.module_body,
                symbol.clone(),
                alloc.inner(),
                LinkageAttr::InternalLinkage,
                false,
            )?;
            let op = mir_dialect::ops::AddressOfOp::new(ctx, symbol, ptr_ty);
            op.get_operation().insert_at_back(insert_block, ctx);
            op.get_result(ctx)
        }
        UnsizeMetadata::ReuseSource => {
            // dyn -> dyn with the same principal: a NOP coercion — the fat
            // pointer (data and vtable) is unchanged. Only the converted
            // type may differ nominally, so reinterpret through a spill
            // when it does.
            let result_ty = convert_ty(tcx, ctx, target_ty)?;
            if input.get_type(ctx) == result_ty {
                return Ok(Some(input));
            }
            let input_ty = input.get_type(ctx);
            let spill = mir_dialect::ops::AllocaOp::new(ctx, input_ty);
            spill.get_operation().insert_at_back(insert_block, ctx);
            let spill = spill.get_result(ctx);
            let store = mir_dialect::ops::StoreOp::new(ctx, input, spill);
            store.get_operation().insert_at_back(insert_block, ctx);
            let load = mir_dialect::ops::LoadOp::new(ctx, spill, result_ty);
            load.get_operation().insert_at_back(insert_block, ctx);
            return Ok(Some(load.get_result(ctx)));
        }
    };

    if runtime_ty(source_ty).is_box() {
        // `Box<[T; N]> -> Box<[T]>`: the source value is a thin pointer at
        // byte offset 0 of its (nested) struct layout, and the target's real
        // layout is `{ ptr, len }`. The converted struct types nest the
        // pointer several levels deep (Unique -> NonNull -> ptr), so
        // assemble the fat value through memory at real byte offsets rather
        // than by insertvalue paths.
        let input_ty = input.get_type(ctx);
        let spill = mir_dialect::ops::AllocaOp::new(ctx, input_ty);
        spill.get_operation().insert_at_back(insert_block, ctx);
        let spill = spill.get_result(ctx);
        let store = mir_dialect::ops::StoreOp::new(ctx, input, spill);
        store.get_operation().insert_at_back(insert_block, ctx);
        let load_ptr = mir_dialect::ops::LoadOp::new(ctx, spill, ptr_ty);
        load_ptr.get_operation().insert_at_back(insert_block, ctx);
        let data_ptr = load_ptr.get_result(ctx);

        let result_ty = convert_ty(tcx, ctx, target_ty)?;
        let out = mir_dialect::ops::AllocaOp::new(ctx, result_ty);
        out.get_operation().insert_at_back(insert_block, ctx);
        let out = out.get_result(ctx);
        let store_ptr = mir_dialect::ops::StoreOp::new(ctx, data_ptr, out);
        store_ptr.get_operation().insert_at_back(insert_block, ctx);
        let meta_addr = ptr_offset_const(ctx, insert_block, out, 8)?;
        let store_meta = mir_dialect::ops::StoreOp::new(ctx, metadata_value, meta_addr);
        store_meta.get_operation().insert_at_back(insert_block, ctx);
        let load = mir_dialect::ops::LoadOp::new(ctx, out, result_ty);
        load.get_operation().insert_at_back(insert_block, ctx);
        return Ok(Some(load.get_result(ctx)));
    }

    let data_ptr = if input.get_type(ctx) == ptr_ty {
        input
    } else {
        let cast = mir_dialect::ops::CastOp::new(ctx, input, ptr_ty);
        cast.get_operation().insert_at_back(insert_block, ctx);
        cast.get_result(ctx)
    };

    let result_ty = convert_ty(tcx, ctx, target_ty)?;
    let undef = mir_dialect::ops::UndefOp::new(ctx, result_ty);
    undef.get_operation().insert_at_back(insert_block, ctx);
    let with_ptr =
        mir_dialect::ops::InsertValueOp::new(ctx, data_ptr, undef.get_result(ctx), vec![0]);
    with_ptr.get_operation().insert_at_back(insert_block, ctx);
    let with_meta = mir_dialect::ops::InsertValueOp::new(
        ctx,
        metadata_value,
        with_ptr.get_result(ctx),
        vec![1],
    );
    with_meta.get_operation().insert_at_back(insert_block, ctx);
    Ok(Some(with_meta.get_result(ctx)))
}

/// What the fat pointer's metadata half is for an unsize coercion; the same
/// classification as rustc_codegen_ssa's `unsized_info`.
pub(super) enum UnsizeMetadata<'tcx> {
    /// `[T; N] -> [T]` (possibly through a struct tail): the slice length.
    SliceLen(u64),
    /// `T -> dyn Trait`: the vtable of `concrete` for `principal`.
    Vtable {
        concrete: Ty<'tcx>,
        principal: Option<rustc_middle::ty::ExistentialTraitRef<'tcx>>,
    },
    /// `dyn Trait -> dyn Trait` with the same principal (adding/removing
    /// auto traits): the source's metadata is reused unchanged.
    ReuseSource,
}

pub(super) fn unsize_metadata<'tcx>(
    tcx: TyCtxt<'tcx>,
    source_ty: Ty<'tcx>,
    target_ty: Ty<'tcx>,
) -> Result<Option<UnsizeMetadata<'tcx>>, String> {
    let source_ty = runtime_ty(source_ty);
    let target_ty = runtime_ty(target_ty);
    let (source_inner, target_inner) = match (source_ty.kind(), target_ty.kind()) {
        (
            rustc_middle::ty::TyKind::Ref(_, source_inner, _),
            rustc_middle::ty::TyKind::Ref(_, target_inner, _),
        )
        | (
            rustc_middle::ty::TyKind::RawPtr(source_inner, _),
            rustc_middle::ty::TyKind::RawPtr(target_inner, _),
        ) => (*source_inner, *target_inner),
        _ if source_ty.is_box() && target_ty.is_box() => {
            (source_ty.expect_boxed_ty(), target_ty.expect_boxed_ty())
        }
        _ => return Ok(None),
    };

    // Unsizing may act through a struct's (recursive) tail field, e.g.
    // `&PolymorphicIter<[T; N]> -> &PolymorphicIter<[T]>` in core's array
    // iterator: the fat pointer's metadata is still the slice length. The
    // *lockstep* tails stop peeling as soon as the sides diverge, so a
    // sized `Sq(u64) -> dyn Shape` coercion keeps `Sq` as the concrete
    // type instead of drilling to its last field (rustc_codegen_ssa's
    // `unsized_info` does the same).
    let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
    let (source_tail, target_tail) = tcx.struct_lockstep_tails_for_codegen(
        runtime_ty(source_inner),
        runtime_ty(target_inner),
        typing_env,
    );
    match (
        runtime_ty(source_tail).kind(),
        runtime_ty(target_tail).kind(),
    ) {
        (rustc_middle::ty::TyKind::Array(_, len), rustc_middle::ty::TyKind::Slice(_)) => {
            Ok(Some(UnsizeMetadata::SliceLen(array_len(tcx, *len)?)))
        }
        (
            rustc_middle::ty::TyKind::Dynamic(data_a, _),
            rustc_middle::ty::TyKind::Dynamic(data_b, _),
        ) => {
            let b_principal = data_b.principal_def_id();
            if data_a.principal_def_id() == b_principal || b_principal.is_none() {
                Ok(Some(UnsizeMetadata::ReuseSource))
            } else {
                Err(format!(
                    "unsupported trait upcasting coercion: {source_tail:?} -> {target_tail:?}"
                ))
            }
        }
        (_, rustc_middle::ty::TyKind::Dynamic(data, _)) => Ok(Some(UnsizeMetadata::Vtable {
            concrete: runtime_ty(source_tail),
            principal: data
                .principal()
                .map(|principal| tcx.instantiate_bound_regions_with_erased(principal)),
        })),
        _ => Ok(None),
    }
}

pub(super) fn aggregate_result_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    kind: &rustc_mir::AggregateKind<'tcx>,
    operand_tys: &[Ty<'tcx>],
) -> Result<TypeHandle, String> {
    match kind {
        rustc_mir::AggregateKind::Array(elem_ty) => {
            let elem_ty = convert_ty(tcx, ctx, mono_ty(tcx, state, *elem_ty))?;
            Ok(llvm::types::ArrayType::get(ctx, elem_ty, operand_tys.len() as u64).into())
        }
        rustc_mir::AggregateKind::Adt(def_id, variant_idx, args, _, None) => {
            let ty = Ty::new_adt(tcx, tcx.adt_def(*def_id), args);
            let rustc_middle::ty::TyKind::Adt(adt_def, _) = ty.kind() else {
                unreachable!("Ty::new_adt must produce an ADT type");
            };
            if !adt_def.is_struct() && !adt_def.is_union() && !adt_def.is_enum() {
                return Err(format!("unsupported aggregate ADT kind: {kind:?}"));
            }
            let _ = variant_idx;
            convert_ty(tcx, ctx, mono_ty(tcx, state, ty))
        }
        rustc_mir::AggregateKind::Adt(def_id, variant_idx, args, _, Some(_)) => {
            let _ = variant_idx;
            convert_ty(
                tcx,
                ctx,
                mono_ty(tcx, state, Ty::new_adt(tcx, tcx.adt_def(*def_id), args)),
            )
        }
        rustc_mir::AggregateKind::Tuple => {
            let fields = operand_tys
                .iter()
                .map(|operand_ty| convert_ty(tcx, ctx, mono_ty(tcx, state, *operand_ty)))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(llvm::types::StructType::get_unnamed(ctx, fields).into())
        }
        rustc_mir::AggregateKind::RawPtr(pointee_ty, mutability) => {
            let pointee_ty = mono_ty(tcx, state, *pointee_ty);
            let ptr_ty = Ty::new_ptr(tcx, pointee_ty, *mutability);
            convert_ty(tcx, ctx, ptr_ty)
        }
        rustc_mir::AggregateKind::Closure(def_id, args) => {
            let ty = Ty::new_closure(tcx, *def_id, args);
            convert_ty(tcx, ctx, mono_ty(tcx, state, ty))
        }
        _ => Err(format!("unsupported MIR aggregate kind: {kind:?}")),
    }
}

pub(super) fn import_operand<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    operand: &Operand<'tcx>,
) -> Result<Value, String> {
    match operand {
        Operand::Copy(place) | Operand::Move(place) => {
            load_place(tcx, ctx, state, insert_block, body, *place)
        }
        Operand::Constant(constant) => {
            import_constant(tcx, ctx, state, insert_block, body, constant)
        }
        Operand::RuntimeChecks(check) => {
            let ty = bool_immediate_ty(ctx);
            let value = if check.value(tcx.sess) { 1 } else { 0 };
            let op = integer_constant(ctx, ty, value)?;
            op.get_operation().insert_at_back(insert_block, ctx);
            Ok(op.get_result(ctx))
        }
    }
}
