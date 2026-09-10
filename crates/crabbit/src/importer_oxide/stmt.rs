use super::*;

pub(super) fn is_arguments_from_str_call<'tcx>(tcx: TyCtxt<'tcx>, func: &Operand<'tcx>) -> bool {
    let Operand::Constant(constant) = func else {
        return false;
    };
    let rustc_middle::ty::TyKind::FnDef(def_id, _) = constant.const_.ty().kind() else {
        return false;
    };
    let path = tcx.def_path_str(*def_id);
    path.contains("fmt::Arguments") && path.ends_with("from_str")
}

pub(super) fn contains_kernel_call<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) -> bool {
    body.basic_blocks.iter().any(|data| {
        matches!(
            &data.terminator().kind,
            TerminatorKind::Call { func, .. } if is_call_to_kernel(tcx, func)
        )
    })
}

pub(super) fn is_call_to_kernel<'tcx>(tcx: TyCtxt<'tcx>, func: &Operand<'tcx>) -> bool {
    let Operand::Constant(constant) = func else {
        return false;
    };
    let rustc_middle::ty::TyKind::FnDef(def_id, _) = constant.const_.ty().kind() else {
        return false;
    };
    is_kernel_def_id(tcx, *def_id)
}

pub(super) fn import_statement<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    statement: &StatementKind<'tcx>,
) -> Result<(), String> {
    match statement {
        StatementKind::Assign(boxed) => {
            let (place, rvalue) = &**boxed;
            if place.projection.is_empty() && place_slot_opt(state, place)?.is_none() {
                // The destination is zero-sized (no slot). MIR rvalues are
                // pure, so the whole assignment can be dropped.
                return Ok(());
            }
            let value = import_rvalue(tcx, ctx, state, insert_block, body, rvalue)?;
            store_place(tcx, ctx, state, insert_block, body, place, value)?;
            Ok(())
        }
        StatementKind::StorageLive(_) | StatementKind::StorageDead(_) | StatementKind::Nop => {
            Ok(())
        }
        StatementKind::SetDiscriminant {
            place,
            variant_index,
        } => {
            let enum_ty = mono_ty(tcx, state, place.ty(body, tcx).ty);
            if !is_enum_ty(enum_ty) {
                return Ok(());
            }
            let addr = place_addr(tcx, ctx, state, insert_block, body, place)?;
            write_enum_tag(tcx, ctx, insert_block, enum_ty, *variant_index, addr)
        }
        StatementKind::Intrinsic(intrinsic)
            if matches!(&**intrinsic, rustc_mir::NonDivergingIntrinsic::Assume(_)) =>
        {
            Ok(())
        }
        StatementKind::Intrinsic(intrinsic)
            if matches!(
                &**intrinsic,
                rustc_mir::NonDivergingIntrinsic::CopyNonOverlapping(_)
            ) =>
        {
            let rustc_mir::NonDivergingIntrinsic::CopyNonOverlapping(copy) = &**intrinsic else {
                unreachable!("copy_nonoverlapping checked above");
            };
            let dst = import_operand(tcx, ctx, state, insert_block, body, &copy.dst)?;
            let src = import_operand(tcx, ctx, state, insert_block, body, &copy.src)?;
            let count = import_operand(tcx, ctx, state, insert_block, body, &copy.count)?;
            let src_ty = mono_ty(tcx, state, copy.src.ty(body, tcx));
            let elem_size = layout_size_of_ty(tcx, pointee_ty(src_ty)?)?;
            let byte_count = scale_index(ctx, insert_block, count, elem_size)?;

            let memcpy: crate::identifier::Identifier = "memcpy".try_into().unwrap();
            let ptr_ty = llvm_ptr_ty(ctx);
            let usize_ty: TypeHandle = usize_ty(ctx).into();
            declare_external_function(
                ctx,
                state.module_body,
                memcpy.clone(),
                vec![ptr_ty, ptr_ty, usize_ty],
                Some(ptr_ty),
            );
            let call = mir_dialect::ops::CallOp::new_direct(
                ctx,
                memcpy,
                vec![dst, src, byte_count],
                Some(ptr_ty),
            );
            call.get_operation().insert_at_back(insert_block, ctx);
            Ok(())
        }
        other => Err(format!("unsupported MIR statement: {other:?}")),
    }
}

pub(super) fn import_terminator<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    terminator: &TerminatorKind<'tcx>,
) -> Result<(), String> {
    match terminator {
        TerminatorKind::Return => {
            let retval = if convert_return_ty(
                tcx,
                ctx,
                mono_ty(tcx, state, body.local_decls[rustc_mir::RETURN_PLACE].ty),
            )?
            .is_empty()
            {
                None
            } else {
                Some(load_place(
                    tcx,
                    ctx,
                    state,
                    insert_block,
                    body,
                    rustc_mir::RETURN_PLACE.into(),
                )?)
            };
            let ret = mir_dialect::ops::ReturnOp::new(ctx, retval);
            ret.get_operation().insert_at_back(insert_block, ctx);
            Ok(())
        }
        TerminatorKind::Goto { target } => {
            let dest = block_for(state, *target)?;
            let goto = mir_dialect::ops::GotoOp::new(ctx, dest, vec![]);
            goto.get_operation().insert_at_back(insert_block, ctx);
            Ok(())
        }
        TerminatorKind::Unreachable => {
            let unreachable = mir_dialect::ops::UnreachableOp::new(ctx);
            unreachable
                .get_operation()
                .insert_at_back(insert_block, ctx);
            Ok(())
        }
        TerminatorKind::UnwindResume => {
            let unreachable = mir_dialect::ops::UnreachableOp::new(ctx);
            unreachable
                .get_operation()
                .insert_at_back(insert_block, ctx);
            Ok(())
        }
        TerminatorKind::SwitchInt { discr, targets } => {
            let discr = import_operand(tcx, ctx, state, insert_block, body, discr)?;
            let otherwise = targets.otherwise();
            let cases = targets.iter().collect::<Vec<_>>();
            if cases.is_empty() {
                let goto = mir_dialect::ops::GotoOp::new(ctx, block_for(state, otherwise)?, vec![]);
                goto.get_operation().insert_at_back(insert_block, ctx);
                return Ok(());
            }

            let mut compare_block = insert_block;
            for (idx, (expected, target)) in cases.iter().copied().enumerate() {
                let next_block = if idx + 1 == cases.len() {
                    block_for(state, otherwise)?
                } else {
                    let block = BasicBlock::new(
                        ctx,
                        Some(format!("switch{}", idx + 1).try_into().unwrap()),
                        vec![],
                    );
                    let region = insert_block
                        .deref(ctx)
                        .get_parent_region()
                        .ok_or_else(|| "MIR switch block is not in a region".to_string())?;
                    block.insert_at_back(region, ctx);
                    block
                };
                let expected = integer_constant(ctx, discr.get_type(ctx), expected)?;
                expected.get_operation().insert_at_back(compare_block, ctx);
                let cmp = mir_dialect::ops::EqOp::new(ctx, discr, expected.get_result(ctx));
                cmp.get_operation().insert_at_back(compare_block, ctx);
                let branch = mir_dialect::ops::CondBrOp::new(
                    ctx,
                    cmp.get_result(ctx),
                    block_for(state, target)?,
                    vec![],
                    next_block,
                    vec![],
                );
                branch.get_operation().insert_at_back(compare_block, ctx);
                compare_block = next_block;
            }
            Ok(())
        }
        TerminatorKind::Call {
            func,
            args,
            destination,
            target,
            ..
        } => {
            if is_arguments_from_str_call(tcx, func) {
                let Some(target) = target else {
                    return Err("unsupported MIR call without return target".to_string());
                };
                lower_arguments_from_str_call(
                    tcx,
                    ctx,
                    state,
                    insert_block,
                    body,
                    args,
                    destination,
                )?;
                let goto = mir_dialect::ops::GotoOp::new(ctx, block_for(state, *target)?, vec![]);
                goto.get_operation().insert_at_back(insert_block, ctx);
                return Ok(());
            }

            if is_unreachable_unchecked_call(tcx, state, body, func) {
                mir_dialect::ops::UnreachableOp::new(ctx)
                    .get_operation()
                    .insert_at_back(insert_block, ctx);
                return Ok(());
            }

            if is_noop_intrinsic_call(tcx, state, body, func) {
                let Some(target) = target else {
                    return Err(
                        "unsupported MIR no-op intrinsic call without return target".to_string()
                    );
                };
                let goto = mir_dialect::ops::GotoOp::new(ctx, block_for(state, *target)?, vec![]);
                goto.get_operation().insert_at_back(insert_block, ctx);
                return Ok(());
            }

            if lower_known_intrinsic_call(
                tcx,
                ctx,
                state,
                insert_block,
                body,
                func,
                args,
                destination,
            )? {
                let Some(target) = target else {
                    mir_dialect::ops::UnreachableOp::new(ctx)
                        .get_operation()
                        .insert_at_back(insert_block, ctx);
                    return Ok(());
                };
                let goto = mir_dialect::ops::GotoOp::new(ctx, block_for(state, *target)?, vec![]);
                goto.get_operation().insert_at_back(insert_block, ctx);
                return Ok(());
            }

            // A virtual call: the callee resolves to a vtable slot instead of
            // a symbol, so the function pointer is loaded from the receiver's
            // vtable at `index * ptr_size` (the index already counts the
            // drop/size/align header, as in rustc_codegen_ssa's
            // `VirtualIndex`), and the receiver's data half becomes the first
            // argument.
            let virtual_slot = call_instance(tcx, state, body, func).and_then(|instance| {
                match instance.def {
                    InstanceKind::Virtual(_, index) => Some(index as u64),
                    _ => None,
                }
            });

            // An indirect call: the callee is a function-pointer value rather
            // than a (constant or zero-sized function-item) `FnDef`.
            let callee_value = if call_fn_def(tcx, state, body, func).is_some() {
                None
            } else {
                Some(import_operand(tcx, ctx, state, insert_block, body, func)?)
            };

            // Callees with the "rust-call" ABI take their trailing tuple
            // argument untupled (matching `spread_arg` on the body).
            let untuple_last = callee_value.is_none()
                && call_fn_def(tcx, state, body, func).is_some_and(|(def_id, _)| {
                    tcx.fn_sig(def_id).skip_binder().skip_binder().abi
                        == rustc_abi::ExternAbi::RustCall
                });

            let mut call_args = Vec::with_capacity(args.len());
            let mut vtable_ptr = None;
            for (idx, arg) in args.iter().enumerate() {
                if idx == 0 && virtual_slot.is_some() {
                    let receiver =
                        import_operand(tcx, ctx, state, insert_block, body, &arg.node)?;
                    let (data, vtable) = split_dyn_receiver(ctx, insert_block, receiver)?;
                    vtable_ptr = Some(vtable);
                    lower_abi_call_arg(ctx, insert_block, data, &mut call_args)?;
                    continue;
                }
                let value = import_operand(tcx, ctx, state, insert_block, body, &arg.node)?;
                if untuple_last && idx + 1 == args.len() {
                    let tuple_ty = mono_ty(tcx, state, arg.node.ty(body, tcx));
                    let rustc_middle::ty::TyKind::Tuple(elem_tys) = runtime_ty(tuple_ty).kind()
                    else {
                        return Err(format!(
                            "rust-call trailing argument is not a tuple: {tuple_ty:?}"
                        ));
                    };
                    for (elem_idx, elem_ty) in elem_tys.iter().enumerate() {
                        let index = converted_field_index(tcx, tuple_ty, elem_idx)?;
                        let elem_conv = convert_ty(tcx, ctx, elem_ty)?;
                        let element =
                            mir_dialect::ops::ExtractValueOp::new(ctx, value, vec![index], elem_conv);
                        element.get_operation().insert_at_back(insert_block, ctx);
                        lower_abi_call_arg(
                            ctx,
                            insert_block,
                            element.get_result(ctx),
                            &mut call_args,
                        )?;
                    }
                    continue;
                }
                lower_abi_call_arg(ctx, insert_block, value, &mut call_args)?;
            }

            let result_type = place_slot_opt(state, destination)?
                .map(|_| {
                    let result_tys = convert_return_ty(
                        tcx,
                        ctx,
                        mono_ty(tcx, state, body.local_decls[destination.local].ty),
                    )?;
                    match result_tys.as_slice() {
                        [] => Ok(None),
                        [ty] => Ok(Some(*ty)),
                        _ => Err("unsupported multi-value MIR call result".to_string()),
                    }
                })
                .transpose()?
                .flatten();

            // Results of calls into prebuilt std code use the real rustc
            // enum layout, which differs from this importer's discriminant-
            // first representation; such results are returned as raw bytes
            // and decoded afterwards.
            let mut external_enum_result = None;
            let mut result_type = result_type;

            let call = if let Some(slot) = virtual_slot {
                let vtable = vtable_ptr
                    .ok_or_else(|| "virtual call without a receiver argument".to_string())?;
                let usize_ty: TypeHandle = usize_ty(ctx).into();
                let ptr_size = tcx.data_layout.pointer_size().bytes();
                let offset = integer_constant(ctx, usize_ty, (slot * ptr_size) as u128)?;
                offset.get_operation().insert_at_back(insert_block, ctx);
                let fn_slot =
                    mir_dialect::ops::PtrOffsetOp::new(ctx, vtable, offset.get_result(ctx));
                fn_slot.get_operation().insert_at_back(insert_block, ctx);
                let ptr_ty = llvm_ptr_ty(ctx);
                let fn_ptr = mir_dialect::ops::LoadOp::new(ctx, fn_slot.get_result(ctx), ptr_ty);
                fn_ptr.get_operation().insert_at_back(insert_block, ctx);
                mir_dialect::ops::CallOp::new_indirect(
                    ctx,
                    fn_ptr.get_result(ctx),
                    call_args,
                    result_type,
                )
            } else if let Some(callee_value) = callee_value {
                mir_dialect::ops::CallOp::new_indirect(ctx, callee_value, call_args, result_type)
            } else {
                let callee = call_callee(tcx, state, body, func)?;
                let external_call = is_upstream_call(tcx, state, body, func);
                let imported = if let Some(instance) = call_instance(tcx, state, body, func)
                    && should_import_instance(tcx, instance)
                {
                    import_upstream_instance(tcx, ctx, state.module_body, instance).map_err(
                        |error| {
                            let mut legaliser = Legaliser::default();
                            let symbol = legaliser.legalise(tcx.symbol_name(instance).name);
                            format!("while importing instance {symbol}: {error}")
                        },
                    )?;
                    true
                } else {
                    false
                };
                if !imported && external_call {
                    let rust_ret_ty = mono_ty(tcx, state, body.local_decls[destination.local].ty);
                    let layout_sensitive = matches!(
                        runtime_ty(rust_ret_ty).kind(),
                        rustc_middle::ty::TyKind::Adt(_, _)
                            | rustc_middle::ty::TyKind::Closure(_, _)
                            | rustc_middle::ty::TyKind::Array(_, _)
                    ) || matches!(
                        runtime_ty(rust_ret_ty).kind(),
                        rustc_middle::ty::TyKind::Tuple(fields) if !fields.is_empty()
                    );
                    if result_type.is_some() && layout_sensitive {
                        let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
                        let real_size = rustc_layout_size_of_ty(tcx, typing_env, rust_ret_ty)?;
                        // Use a word-array blob so the decoded loads stay
                        // 8-byte aligned.
                        let word_ty: TypeHandle =
                            IntegerType::get(ctx, 64, Signedness::Unsigned).into();
                        let blob_ty: TypeHandle =
                            llvm::types::ArrayType::get(ctx, word_ty, real_size.div_ceil(8)).into();
                        result_type = Some(blob_ty);
                        external_enum_result = Some((rust_ret_ty, blob_ty));
                    }
                    let arg_types = call_args
                        .iter()
                        .map(|arg| arg.get_type(ctx))
                        .collect::<Vec<_>>();
                    declare_external_function(
                        ctx,
                        state.module_body,
                        callee.clone(),
                        arg_types,
                        result_type,
                    );
                }
                mir_dialect::ops::CallOp::new_direct(ctx, callee, call_args, result_type)
            };
            call.get_operation().insert_at_back(insert_block, ctx);

            if let Some((rust_ret_ty, blob_ty)) = external_enum_result {
                let blob = call.get_operation().deref(ctx).get_result(0);
                let blob_slot = mir_dialect::ops::AllocaOp::new(ctx, blob_ty);
                blob_slot.get_operation().insert_at_back(insert_block, ctx);
                let store = mir_dialect::ops::StoreOp::new(ctx, blob, blob_slot.get_result(ctx));
                store.get_operation().insert_at_back(insert_block, ctx);
                let value = load_value_from_real_layout(
                    tcx,
                    ctx,
                    insert_block,
                    rust_ret_ty,
                    blob_slot.get_result(ctx),
                )?;
                store_place(tcx, ctx, state, insert_block, body, destination, value)?;
            } else if let Some(slot) = place_slot_opt(state, destination)?
                && call.get_operation().deref(ctx).get_num_results() > 0
            {
                let result = call.get_operation().deref(ctx).get_result(0);
                let result = normalize_bool_for_storage(
                    tcx,
                    ctx,
                    state,
                    insert_block,
                    body.local_decls[destination.local].ty,
                    result,
                )?;
                mir_dialect::ops::StoreOp::new(ctx, result, slot)
                    .get_operation()
                    .insert_at_back(insert_block, ctx);
            }

            let Some(target) = target else {
                mir_dialect::ops::UnreachableOp::new(ctx)
                    .get_operation()
                    .insert_at_back(insert_block, ctx);
                return Ok(());
            };
            let goto = mir_dialect::ops::GotoOp::new(ctx, block_for(state, *target)?, vec![]);
            goto.get_operation().insert_at_back(insert_block, ctx);
            Ok(())
        }
        TerminatorKind::Assert { target, .. } => {
            let goto = mir_dialect::ops::GotoOp::new(ctx, block_for(state, *target)?, vec![]);
            goto.get_operation().insert_at_back(insert_block, ctx);
            Ok(())
        }
        TerminatorKind::Drop { place, target, .. } => {
            let ty = mono_ty(tcx, state, place.ty(body, tcx).ty);
            let typing_env = state.instance.map_or_else(
                || body.typing_env(tcx),
                |_| rustc_middle::ty::TypingEnv::fully_monomorphized(),
            );
            if ty.needs_drop(tcx, typing_env) {
                let drop_instance = Instance::resolve_drop_in_place(tcx, ty);
                let mut legaliser = Legaliser::default();
                let symbol = legaliser.legalise(tcx.symbol_name(drop_instance).name);
                import_upstream_instance(tcx, ctx, state.module_body, drop_instance)
                    .map_err(|error| format!("while importing drop glue for {ty:?}: {error}"))?;
                let addr = place_addr(tcx, ctx, state, insert_block, body, place)?;
                let mut call_args = Vec::new();
                lower_abi_call_arg(ctx, insert_block, addr, &mut call_args)?;
                let call = mir_dialect::ops::CallOp::new_direct(ctx, symbol, call_args, None);
                call.get_operation().insert_at_back(insert_block, ctx);
            }
            let goto = mir_dialect::ops::GotoOp::new(ctx, block_for(state, *target)?, vec![]);
            goto.get_operation().insert_at_back(insert_block, ctx);
            Ok(())
        }
        other => Err(format!("unsupported MIR terminator: {other:?}")),
    }
}

pub(super) fn call_callee<'tcx>(
    tcx: TyCtxt<'tcx>,
    state: &FunctionImportState<'tcx>,
    body: &Body<'tcx>,
    func: &Operand<'tcx>,
) -> Result<crate::identifier::Identifier, String> {
    call_symbol(tcx, state, body, func)
}

/// The `FnDef` (def id, generic args) of a call, whether the callee operand
/// is a constant or a zero-sized function-item value read from a place.
pub(super) fn call_fn_def<'tcx>(
    tcx: TyCtxt<'tcx>,
    state: &FunctionImportState<'tcx>,
    body: &Body<'tcx>,
    func: &Operand<'tcx>,
) -> Option<(
    rustc_hir::def_id::DefId,
    rustc_middle::ty::GenericArgsRef<'tcx>,
)> {
    let func_ty = mono_ty(tcx, state, func.ty(body, tcx));
    let rustc_middle::ty::TyKind::FnDef(def_id, args) = runtime_ty(func_ty).kind() else {
        return None;
    };
    Some((*def_id, args))
}

pub(super) fn call_instance<'tcx>(
    tcx: TyCtxt<'tcx>,
    state: &FunctionImportState<'tcx>,
    body: &Body<'tcx>,
    func: &Operand<'tcx>,
) -> Option<Instance<'tcx>> {
    let (def_id, args) = call_fn_def(tcx, state, body, func)?;
    let args = mono_generic_args(tcx, state, args);
    Instance::try_resolve(tcx, body.typing_env(tcx), def_id, args)
        .ok()
        .flatten()
}

pub(super) fn mono_generic_args<'tcx>(
    tcx: TyCtxt<'tcx>,
    state: &FunctionImportState<'tcx>,
    args: rustc_middle::ty::GenericArgsRef<'tcx>,
) -> rustc_middle::ty::GenericArgsRef<'tcx> {
    state.instance.map_or(args, |instance| {
        instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            rustc_middle::ty::TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(args),
        )
    })
}

/// Require a SIMD vector type to be exactly 8 bytes (a NEON D register),
/// the only shape the scalar SWAR lowering handles.
pub(super) fn require_8_byte_vector<'tcx>(
    tcx: TyCtxt<'tcx>,
    name: &str,
    vector_ty: Ty<'tcx>,
) -> Result<(), String> {
    let size = layout_size_of_ty(tcx, vector_ty)?;
    if size != 8 {
        return Err(format!(
            "unsupported {name} intrinsic vector size: {size} bytes (only 64-bit vectors are \
             lowered)"
        ));
    }
    Ok(())
}

/// The u64 bit pattern of an 8-byte SIMD vector value. Scalars are cast
/// directly; aggregate representations round-trip through a stack slot.
pub(super) fn simd_value_bits(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    value: Value,
) -> Result<Value, String> {
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    if value.get_type(ctx).deref(ctx).is::<IntegerType>() {
        return Ok(cast_value_to_type(ctx, insert_block, value, u64_ty));
    }
    let slot = mir_dialect::ops::AllocaOp::new(ctx, u64_ty);
    slot.get_operation().insert_at_back(insert_block, ctx);
    let slot = slot.get_result(ctx);
    let store = mir_dialect::ops::StoreOp::new(ctx, value, slot);
    store.get_operation().insert_at_back(insert_block, ctx);
    let load = mir_dialect::ops::LoadOp::new(ctx, slot, u64_ty);
    load.get_operation().insert_at_back(insert_block, ctx);
    Ok(load.get_result(ctx))
}

/// Store a u64 bit pattern into `destination`, whose type is an 8-byte SIMD
/// vector: spill the bits and reload them with the destination's real
/// layout, then store the place.
pub(super) fn store_simd_bits<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    destination: &Place<'tcx>,
    bits: Value,
) -> Result<(), String> {
    let dest_rust_ty = mono_ty(tcx, state, destination.ty(body, tcx).ty);
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let slot = mir_dialect::ops::AllocaOp::new(ctx, u64_ty);
    slot.get_operation().insert_at_back(insert_block, ctx);
    let slot = slot.get_result(ctx);
    let store = mir_dialect::ops::StoreOp::new(ctx, bits, slot);
    store.get_operation().insert_at_back(insert_block, ctx);
    let value = load_value_from_real_layout(tcx, ctx, insert_block, dest_rust_ty, slot)?;
    store_place(tcx, ctx, state, insert_block, body, destination, value)
}

/// Split a `dyn` method receiver into its (data pointer, vtable pointer)
/// halves. Mirrors rustc_codegen_ssa's virtual-call handling: peel
/// `DispatchFromDyn` newtype wrappers (`Pin<&mut dyn ..>`, `Box<dyn ..>`
/// converted as single-field structs) until the fat pointer itself, then
/// take its two fields.
pub(super) fn split_dyn_receiver(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    mut receiver: Value,
) -> Result<(Value, Value), String> {
    loop {
        let ty = receiver.get_type(ctx);
        let (num_fields, field_tys) = {
            let ty_ref = ty.deref(ctx);
            let Some(struct_ty) = ty_ref.downcast_ref::<llvm::types::StructType>() else {
                return Err(format!(
                    "unsupported dyn receiver type: {}",
                    ty_ref.disp(ctx)
                ));
            };
            let fields: Vec<TypeHandle> = (0..struct_ty.num_fields())
                .map(|index| struct_ty.field_type(index))
                .collect();
            (fields.len(), fields)
        };
        match num_fields {
            2 => {
                let data = emit_op(
                    mir_dialect::ops::ExtractValueOp::new(ctx, receiver, vec![0], field_tys[0])
                        .get_operation(),
                    ctx,
                    insert_block,
                );
                let vtable = emit_op(
                    mir_dialect::ops::ExtractValueOp::new(ctx, receiver, vec![1], field_tys[1])
                        .get_operation(),
                    ctx,
                    insert_block,
                );
                return Ok((data, vtable));
            }
            1 => {
                receiver = emit_op(
                    mir_dialect::ops::ExtractValueOp::new(ctx, receiver, vec![0], field_tys[0])
                        .get_operation(),
                    ctx,
                    insert_block,
                );
            }
            other => {
                return Err(format!(
                    "unsupported dyn receiver shape: {other}-field struct"
                ));
            }
        }
    }
}

pub(super) fn should_import_instance<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> bool {
    match instance.def {
        // Tuple-variant constructor bodies are synthesized by `instance_mir`.
        InstanceKind::Item(def_id) => {
            tcx.is_mir_available(def_id)
                || matches!(tcx.def_kind(def_id), rustc_hir::def::DefKind::Ctor(_, _))
        }
        // Shim MIR is synthesized on demand by `tcx.instance_mir`.
        InstanceKind::DropGlue(_, Some(_))
        | InstanceKind::ClosureOnceShim { .. }
        | InstanceKind::CloneShim(_, _)
        | InstanceKind::FnPtrShim(_, _)
        // Vtable and reify shims (receiver/ABI adapters referenced from
        // vtables and fn-pointer casts) exist in no upstream object; their
        // MIR is synthesized like the other shims.
        | InstanceKind::VTableShim(_)
        | InstanceKind::ReifyShim(_, _) => true,
        _ => false,
    }
}

pub(super) fn is_unreachable_unchecked_call<'tcx>(
    tcx: TyCtxt<'tcx>,
    state: &FunctionImportState<'tcx>,
    body: &Body<'tcx>,
    func: &Operand<'tcx>,
) -> bool {
    call_instance(tcx, state, body, func).is_some_and(|instance| {
        tcx.def_path_str(instance.def.def_id())
            .contains("unreachable_unchecked")
    })
}

pub(super) fn is_noop_intrinsic_call<'tcx>(
    tcx: TyCtxt<'tcx>,
    state: &FunctionImportState<'tcx>,
    body: &Body<'tcx>,
    func: &Operand<'tcx>,
) -> bool {
    call_instance(tcx, state, body, func).is_some_and(|instance| {
        let path = tcx.def_path_str(instance.def.def_id());
        path.contains("core::intrinsics::assert_inhabited")
    }) || call_fn_def(tcx, state, body, func).is_some_and(|(def_id, _)| {
        tcx.def_path_str(def_id)
            .contains("core::intrinsics::assert_inhabited")
    }) || call_symbol(tcx, state, body, func)
        .is_ok_and(|symbol| symbol.as_ref().contains("assert_inhabited"))
}
