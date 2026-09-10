use super::*;

/// Lower well-known codegen intrinsic calls that have no MIR body. Returns
/// `true` when the call was handled and only the branch to the return target
/// still needs to be emitted.
#[allow(clippy::too_many_arguments)]
pub(super) fn lower_known_intrinsic_call<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    func: &Operand<'tcx>,
    args: &[rustc_span::Spanned<Operand<'tcx>>],
    destination: &Place<'tcx>,
) -> Result<bool, String> {
    let Some(instance) = call_instance(tcx, state, body, func) else {
        return Ok(false);
    };
    let name = match instance.def {
        InstanceKind::Intrinsic(def_id) => tcx.item_name(def_id).to_string(),
        // Both explicit `ptr::drop_in_place` calls and drop-glue shims
        // resolve to this instance kind (the shim was named `ptr::drop_glue`
        // on newer nightlies; matching the kind covers both spellings).
        InstanceKind::DropGlue(..) => "drop_glue".to_string(),
        _ => return Ok(false),
    };

    match name.as_str() {
        // Pure optimization hints with no runtime semantics.
        "cold_path" => Ok(true),
        "black_box" => {
            if args.len() != 1 {
                return Err(format!(
                    "unsupported black_box intrinsic arity: {}",
                    args.len()
                ));
            }
            // The optimization barrier means nothing to this backend; the
            // value passes through unchanged (and a ZST has nothing to copy).
            if layout_size_of_ty(tcx, mono_ty(tcx, state, instance.args.type_at(0)))? == 0 {
                return Ok(true);
            }
            let value = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            store_place(tcx, ctx, state, insert_block, body, destination, value)?;
            Ok(true)
        }
        // `atomic_load`/`atomic_store` are `(*const T) -> T` and
        // `(*mut T, T) -> ()`, with the ordering as a const generic rather than
        // a runtime argument.
        //
        // These lower to plain loads and stores. On AArch64 a naturally aligned
        // access up to 8 bytes is single-copy atomic, so the *atomicity* holds;
        // what is dropped is the ordering, because no barrier is emitted. That
        // is sound only for the single-threaded programs this backend targets
        // (std reaches these through the uncontended `Mutex` behind
        // `io::stdin()`). Honouring the ordering needs real barrier/acquire
        // instructions, and read-modify-write atomics (`atomic_cxchg`,
        // `atomic_rmw*`) are deliberately not lowered here — they still fail
        // loudly as unresolved symbols rather than being silently mis-compiled.
        "atomic_load" => {
            if args.len() != 1 {
                return Err(format!(
                    "unsupported atomic_load intrinsic arity: {}",
                    args.len()
                ));
            }
            let ptr = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let value_ty = mono_ty(tcx, state, instance.args.type_at(0));
            let Some(load_ty) = convert_storage_ty(tcx, ctx, value_ty)? else {
                return Ok(true);
            };
            let load = mir_dialect::ops::LoadOp::new(ctx, ptr, load_ty);
            load.get_operation().insert_at_back(insert_block, ctx);
            store_place(
                tcx,
                ctx,
                state,
                insert_block,
                body,
                destination,
                load.get_result(ctx),
            )?;
            Ok(true)
        }
        "atomic_store" => {
            if args.len() != 2 {
                return Err(format!(
                    "unsupported atomic_store intrinsic arity: {}",
                    args.len()
                ));
            }
            let ptr = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let value_ty = mono_ty(tcx, state, instance.args.type_at(0));
            let Some(store_ty) = convert_storage_ty(tcx, ctx, value_ty)? else {
                return Ok(true);
            };
            let value = import_operand(tcx, ctx, state, insert_block, body, &args[1].node)?;
            let value = normalize_bool_for_storage(tcx, ctx, state, insert_block, value_ty, value)?;
            let value = cast_value_to_type(ctx, insert_block, value, store_ty);
            let store = mir_dialect::ops::StoreOp::new(ctx, value, ptr);
            store.get_operation().insert_at_back(insert_block, ctx);
            Ok(true)
        }
        // A genuinely atomic exchange, lowered to LLVM's aarch64 outline-
        // atomic helper (`__aarch64_swpN_acq_rel`) from compiler-builtins,
        // which dispatches to LSE `SWPAL` or an LL/SC loop at runtime. The
        // ordering is strengthened to acquire+release regardless of the
        // requested one, which is always sound. The helpers only exist in
        // aarch64-linux's compiler-builtins; on other targets the intrinsic
        // stays unhandled and fails loudly at link time (macOS std never
        // reaches it — its `Mutex` is `os_unfair_lock`-based).
        "atomic_xchg" => {
            if !(tcx.sess.target.arch.to_string() == "aarch64"
                && tcx.sess.target.os.to_string() == "linux")
            {
                return Ok(false);
            }
            if args.len() != 2 {
                return Err(format!(
                    "unsupported atomic_xchg intrinsic arity: {}",
                    args.len()
                ));
            }
            let value_ty = mono_ty(tcx, state, instance.args.type_at(0));
            let Some(storage_ty) = convert_storage_ty(tcx, ctx, value_ty)? else {
                return Ok(true);
            };
            let size = layout_size_of_ty(tcx, value_ty)?;
            if !matches!(size, 1 | 2 | 4 | 8) {
                return Err(format!(
                    "unsupported atomic_xchg operand size: {size} bytes"
                ));
            }
            let helper: crate::identifier::Identifier =
                format!("__aarch64_swp{size}_acq_rel").try_into().unwrap();
            let ptr_ty = llvm_ptr_ty(ctx);
            declare_external_function(
                ctx,
                state.module_body,
                helper.clone(),
                vec![storage_ty, ptr_ty],
                Some(storage_ty),
            );
            let ptr = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let value = import_operand(tcx, ctx, state, insert_block, body, &args[1].node)?;
            let value = normalize_bool_for_storage(tcx, ctx, state, insert_block, value_ty, value)?;
            let value = cast_value_to_type(ctx, insert_block, value, storage_ty);
            let mut call_args = Vec::new();
            lower_abi_call_arg(ctx, insert_block, value, &mut call_args)?;
            lower_abi_call_arg(ctx, insert_block, ptr, &mut call_args)?;
            let call =
                mir_dialect::ops::CallOp::new_direct(ctx, helper, call_args, Some(storage_ty));
            call.get_operation().insert_at_back(insert_block, ctx);
            let old = call.get_result(ctx);
            store_place(tcx, ctx, state, insert_block, body, destination, old)?;
            Ok(true)
        }
        "write_bytes" => {
            if args.len() != 3 {
                return Err(format!(
                    "unsupported write_bytes intrinsic arity: {}",
                    args.len()
                ));
            }
            let dst = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let val = import_operand(tcx, ctx, state, insert_block, body, &args[1].node)?;
            let count = import_operand(tcx, ctx, state, insert_block, body, &args[2].node)?;
            let elem = instance.args.type_at(0);
            let elem_size = layout_size_of_ty(tcx, mono_ty(tcx, state, elem))?;
            let byte_count = scale_index(ctx, insert_block, count, elem_size)?;
            let val_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signed).into();
            let val = cast_value_to_type(ctx, insert_block, val, val_ty);

            let memset: crate::identifier::Identifier = "memset".try_into().unwrap();
            let ptr_ty = llvm_ptr_ty(ctx);
            let usize_ty: TypeHandle = usize_ty(ctx).into();
            declare_external_function(
                ctx,
                state.module_body,
                memset.clone(),
                vec![ptr_ty, val_ty, usize_ty],
                Some(ptr_ty),
            );
            let call = mir_dialect::ops::CallOp::new_direct(
                ctx,
                memset,
                vec![dst, val, byte_count],
                Some(ptr_ty),
            );
            call.get_operation().insert_at_back(insert_block, ctx);
            Ok(true)
        }
        "ptr_offset_from_unsigned" | "ptr_offset_from" => {
            if args.len() != 2 {
                return Err(format!(
                    "unsupported ptr_offset_from intrinsic arity: {}",
                    args.len()
                ));
            }
            let lhs = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let rhs = import_operand(tcx, ctx, state, insert_block, body, &args[1].node)?;
            let usize_ty: TypeHandle = usize_ty(ctx).into();
            let lhs = cast_value_to_type(ctx, insert_block, lhs, usize_ty);
            let rhs = cast_value_to_type(ctx, insert_block, rhs, usize_ty);
            let diff = mir_dialect::ops::SubOp::new(ctx, lhs, rhs);
            diff.get_operation().insert_at_back(insert_block, ctx);
            let mut result = diff.get_operation().deref(ctx).get_result(0);
            let elem = instance.args.type_at(0);
            let elem_size = layout_size_of_ty(tcx, mono_ty(tcx, state, elem))?;
            if elem_size > 1 {
                let size = integer_constant(ctx, usize_ty, elem_size as u128)?;
                size.get_operation().insert_at_back(insert_block, ctx);
                let div = mir_dialect::ops::DivOp::new(ctx, result, size.get_result(ctx));
                div.get_operation().insert_at_back(insert_block, ctx);
                result = div.get_operation().deref(ctx).get_result(0);
            }
            let dest_ty =
                convert_immediate_ty(tcx, ctx, mono_ty(tcx, state, destination.ty(body, tcx).ty))?;
            let result = cast_value_to_type(ctx, insert_block, result, dest_ty);
            store_place(tcx, ctx, state, insert_block, body, destination, result)?;
            Ok(true)
        }
        "ctpop" => {
            if args.len() != 1 {
                return Err(format!("unsupported ctpop intrinsic arity: {}", args.len()));
            }
            let input = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let width = input
                .get_type(ctx)
                .deref(ctx)
                .downcast_ref::<IntegerType>()
                .map(|ty| ty.width())
                .ok_or_else(|| "ctpop on non-integer type".to_string())?;
            let total = if width == 128 {
                let (lo, hi) = split_u128_halves(ctx, insert_block, input)?;
                let lo_count = emit_popcount64(ctx, insert_block, lo)?;
                let hi_count = emit_popcount64(ctx, insert_block, hi)?;
                emit_op(
                    mir_dialect::ops::AddOp::new(ctx, lo_count, hi_count).get_operation(),
                    ctx,
                    insert_block,
                )
            } else {
                let x = widen_to_u64(ctx, insert_block, input)?;
                emit_popcount64(ctx, insert_block, x)?
            };
            let dest_ty =
                convert_immediate_ty(tcx, ctx, mono_ty(tcx, state, destination.ty(body, tcx).ty))?;
            let result = cast_value_to_type(ctx, insert_block, total, dest_ty);
            store_place(tcx, ctx, state, insert_block, body, destination, result)?;
            Ok(true)
        }
        "cttz" | "cttz_nonzero" => {
            if args.len() != 1 {
                return Err(format!("unsupported cttz intrinsic arity: {}", args.len()));
            }
            let input = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let width = input
                .get_type(ctx)
                .deref(ctx)
                .downcast_ref::<IntegerType>()
                .map(|ty| ty.width())
                .ok_or_else(|| "cttz on non-integer type".to_string())?;
            let total = if width == 128 {
                emit_cttz128(ctx, insert_block, input)?
            } else if width > 64 {
                return Err(format!("unsupported {width}-bit cttz intrinsic"));
            } else {
                let x = widen_to_u64(ctx, insert_block, input)?;
                emit_cttz64(ctx, insert_block, x, width)?
            };
            let dest_ty =
                convert_immediate_ty(tcx, ctx, mono_ty(tcx, state, destination.ty(body, tcx).ty))?;
            let result = cast_value_to_type(ctx, insert_block, total, dest_ty);
            store_place(tcx, ctx, state, insert_block, body, destination, result)?;
            Ok(true)
        }
        "ctlz" | "ctlz_nonzero" => {
            if args.len() != 1 {
                return Err(format!("unsupported ctlz intrinsic arity: {}", args.len()));
            }
            let input = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let width = input
                .get_type(ctx)
                .deref(ctx)
                .downcast_ref::<IntegerType>()
                .map(|ty| ty.width())
                .ok_or_else(|| "ctlz on non-integer type".to_string())?;
            let total = if width == 128 {
                emit_ctlz128(ctx, insert_block, input)?
            } else if width > 64 {
                return Err(format!("unsupported {width}-bit ctlz intrinsic"));
            } else {
                let x = widen_to_u64(ctx, insert_block, input)?;
                emit_ctlz64(ctx, insert_block, x, width)?
            };
            let dest_ty =
                convert_immediate_ty(tcx, ctx, mono_ty(tcx, state, destination.ty(body, tcx).ty))?;
            let result = cast_value_to_type(ctx, insert_block, total, dest_ty);
            store_place(tcx, ctx, state, insert_block, body, destination, result)?;
            Ok(true)
        }
        "bswap" => {
            if args.len() != 1 {
                return Err(format!("unsupported bswap intrinsic arity: {}", args.len()));
            }
            let input = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let width = input
                .get_type(ctx)
                .deref(ctx)
                .downcast_ref::<IntegerType>()
                .map(|ty| ty.width())
                .ok_or_else(|| "bswap on non-integer type".to_string())?;
            let swapped = if width == 128 {
                // Swap each 64-bit half's bytes, then swap the halves.
                let (lo, hi) = split_u128_halves(ctx, insert_block, input)?;
                let lo_swapped = emit_bswap64(ctx, insert_block, lo, 64)?;
                let hi_swapped = emit_bswap64(ctx, insert_block, hi, 64)?;
                join_u128_halves(ctx, insert_block, hi_swapped, lo_swapped)?
            } else if width > 64 || width % 8 != 0 {
                return Err(format!("unsupported {width}-bit bswap intrinsic"));
            } else if width == 8 {
                widen_to_u64(ctx, insert_block, input)?
            } else {
                let x = widen_to_u64(ctx, insert_block, input)?;
                emit_bswap64(ctx, insert_block, x, width)?
            };
            let dest_ty =
                convert_immediate_ty(tcx, ctx, mono_ty(tcx, state, destination.ty(body, tcx).ty))?;
            let result = cast_value_to_type(ctx, insert_block, swapped, dest_ty);
            store_place(tcx, ctx, state, insert_block, body, destination, result)?;
            Ok(true)
        }
        "bitreverse" => {
            if args.len() != 1 {
                return Err(format!(
                    "unsupported bitreverse intrinsic arity: {}",
                    args.len()
                ));
            }
            let input = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let width = input
                .get_type(ctx)
                .deref(ctx)
                .downcast_ref::<IntegerType>()
                .map(|ty| ty.width())
                .ok_or_else(|| "bitreverse on non-integer type".to_string())?;
            let reversed = if width == 128 {
                // Reverse each 64-bit half, then swap the halves.
                let (lo, hi) = split_u128_halves(ctx, insert_block, input)?;
                let lo_reversed = emit_bitreverse64(ctx, insert_block, lo, 64)?;
                let hi_reversed = emit_bitreverse64(ctx, insert_block, hi, 64)?;
                join_u128_halves(ctx, insert_block, hi_reversed, lo_reversed)?
            } else if width > 64 {
                return Err(format!("unsupported {width}-bit bitreverse intrinsic"));
            } else {
                let x = widen_to_u64(ctx, insert_block, input)?;
                emit_bitreverse64(ctx, insert_block, x, width)?
            };
            let dest_ty =
                convert_immediate_ty(tcx, ctx, mono_ty(tcx, state, destination.ty(body, tcx).ty))?;
            let result = cast_value_to_type(ctx, insert_block, reversed, dest_ty);
            store_place(tcx, ctx, state, insert_block, body, destination, result)?;
            Ok(true)
        }
        "rotate_left" | "rotate_right" => {
            if args.len() != 2 {
                return Err(format!(
                    "unsupported rotate intrinsic arity: {}",
                    args.len()
                ));
            }
            let input = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let width = input
                .get_type(ctx)
                .deref(ctx)
                .downcast_ref::<IntegerType>()
                .map(|ty| ty.width())
                .ok_or_else(|| "rotate on non-integer type".to_string())?;
            if width == 128 {
                // Same modular-shift construction as the u64 path below,
                // carried out at u128 (dynamic i128 shifts exist in isel).
                let u128_ty: TypeHandle =
                    IntegerType::get(ctx, 128, Signedness::Unsigned).into();
                let x = cast_value_to_type(ctx, insert_block, input, u128_ty);
                let shift = import_operand(tcx, ctx, state, insert_block, body, &args[1].node)?;
                let shift = cast_value_to_type(ctx, insert_block, shift, u128_ty);
                let modulus_mask = integer_constant(ctx, u128_ty, 127)?;
                modulus_mask.get_operation().insert_at_back(insert_block, ctx);
                let modulus_mask = modulus_mask.get_result(ctx);
                let k = emit_op(
                    mir_dialect::ops::BitAndOp::new(ctx, shift, modulus_mask).get_operation(),
                    ctx,
                    insert_block,
                );
                let width_value = integer_constant(ctx, u128_ty, 128)?;
                width_value.get_operation().insert_at_back(insert_block, ctx);
                let complement = emit_op(
                    mir_dialect::ops::SubOp::new(ctx, width_value.get_result(ctx), k)
                        .get_operation(),
                    ctx,
                    insert_block,
                );
                let inv = emit_op(
                    mir_dialect::ops::BitAndOp::new(ctx, complement, modulus_mask).get_operation(),
                    ctx,
                    insert_block,
                );
                let (left_amount, right_amount) = if name == "rotate_left" {
                    (k, inv)
                } else {
                    (inv, k)
                };
                let left = emit_op(
                    mir_dialect::ops::ShlOp::new(ctx, x, left_amount).get_operation(),
                    ctx,
                    insert_block,
                );
                let right = emit_op(
                    mir_dialect::ops::ShrOp::new(ctx, x, right_amount).get_operation(),
                    ctx,
                    insert_block,
                );
                let rotated = emit_op(
                    mir_dialect::ops::BitOrOp::new(ctx, left, right).get_operation(),
                    ctx,
                    insert_block,
                );
                let dest_ty = convert_immediate_ty(
                    tcx,
                    ctx,
                    mono_ty(tcx, state, destination.ty(body, tcx).ty),
                )?;
                let result = cast_value_to_type(ctx, insert_block, rotated, dest_ty);
                store_place(tcx, ctx, state, insert_block, body, destination, result)?;
                return Ok(true);
            }
            if width > 64 || !width.is_power_of_two() {
                return Err(format!("unsupported {width}-bit rotate intrinsic"));
            }
            let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
            let constant = |ctx: &mut Context, bits: u128| -> Result<Value, String> {
                let op = integer_constant(ctx, u64_ty, bits)?;
                op.get_operation().insert_at_back(insert_block, ctx);
                Ok(op.get_result(ctx))
            };
            let width_mask = (u64::MAX >> (64 - width as usize)) as u128;
            let mut x = widen_to_u64(ctx, insert_block, input)?;
            if width < 64 {
                // Widening may sign-extend; the rotate must only see the low
                // `width` bits.
                let mask = constant(ctx, width_mask)?;
                x = emit_op(
                    mir_dialect::ops::BitAndOp::new(ctx, x, mask).get_operation(),
                    ctx,
                    insert_block,
                );
            }
            let shift = import_operand(tcx, ctx, state, insert_block, body, &args[1].node)?;
            let shift = widen_to_u64(ctx, insert_block, shift)?;
            // Rotation is modular: k = shift & (width - 1); the opposite
            // shift is (width - k) & (width - 1), which is 0 when k is 0 so
            // both halves degenerate to the untouched value.
            let modulus_mask = constant(ctx, (width - 1) as u128)?;
            let k = emit_op(
                mir_dialect::ops::BitAndOp::new(ctx, shift, modulus_mask).get_operation(),
                ctx,
                insert_block,
            );
            let width_value = constant(ctx, width as u128)?;
            let complement = emit_op(
                mir_dialect::ops::SubOp::new(ctx, width_value, k).get_operation(),
                ctx,
                insert_block,
            );
            let inv = emit_op(
                mir_dialect::ops::BitAndOp::new(ctx, complement, modulus_mask).get_operation(),
                ctx,
                insert_block,
            );
            let (left_amount, right_amount) = if name == "rotate_left" {
                (k, inv)
            } else {
                (inv, k)
            };
            let left = emit_op(
                mir_dialect::ops::ShlOp::new(ctx, x, left_amount).get_operation(),
                ctx,
                insert_block,
            );
            let right = emit_op(
                mir_dialect::ops::ShrOp::new(ctx, x, right_amount).get_operation(),
                ctx,
                insert_block,
            );
            let mut rotated = emit_op(
                mir_dialect::ops::BitOrOp::new(ctx, left, right).get_operation(),
                ctx,
                insert_block,
            );
            if width < 64 {
                let mask = constant(ctx, width_mask)?;
                rotated = emit_op(
                    mir_dialect::ops::BitAndOp::new(ctx, rotated, mask).get_operation(),
                    ctx,
                    insert_block,
                );
            }
            let dest_ty =
                convert_immediate_ty(tcx, ctx, mono_ty(tcx, state, destination.ty(body, tcx).ty))?;
            let result = cast_value_to_type(ctx, insert_block, rotated, dest_ty);
            store_place(tcx, ctx, state, insert_block, body, destination, result)?;
            Ok(true)
        }
        // `copy(src, dst, count)` allows overlap (memmove);
        // `copy_nonoverlapping` does not (memcpy).
        "copy" | "copy_nonoverlapping" => {
            if args.len() != 3 {
                return Err(format!(
                    "unsupported copy intrinsic arity: {}",
                    args.len()
                ));
            }
            let src = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let dst = import_operand(tcx, ctx, state, insert_block, body, &args[1].node)?;
            let count = import_operand(tcx, ctx, state, insert_block, body, &args[2].node)?;
            let elem = instance.args.type_at(0);
            let elem_size = layout_size_of_ty(tcx, mono_ty(tcx, state, elem))?;
            let byte_count = scale_index(ctx, insert_block, count, elem_size)?;
            let helper: crate::identifier::Identifier = if name == "copy" {
                "memmove".try_into().unwrap()
            } else {
                "memcpy".try_into().unwrap()
            };
            let ptr_ty = llvm_ptr_ty(ctx);
            let usize_ty: TypeHandle = usize_ty(ctx).into();
            declare_external_function(
                ctx,
                state.module_body,
                helper.clone(),
                vec![ptr_ty, ptr_ty, usize_ty],
                Some(ptr_ty),
            );
            let call = mir_dialect::ops::CallOp::new_direct(
                ctx,
                helper,
                vec![dst, src, byte_count],
                Some(ptr_ty),
            );
            call.get_operation().insert_at_back(insert_block, ctx);
            Ok(true)
        }
        "typed_swap_nonoverlapping" => {
            if args.len() != 2 {
                return Err(format!(
                    "unsupported typed_swap_nonoverlapping intrinsic arity: {}",
                    args.len()
                ));
            }
            let size = layout_size_of_ty(tcx, mono_ty(tcx, state, instance.args.type_at(0)))?;
            if size == 0 {
                return Ok(true);
            }
            let x = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let y = import_operand(tcx, ctx, state, insert_block, body, &args[1].node)?;
            // Swap through a stack temporary with three memcpys; the operands
            // are guaranteed non-overlapping.
            let byte_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
            let temp_ty: TypeHandle = llvm::types::ArrayType::get(ctx, byte_ty, size).into();
            let temp = mir_dialect::ops::AllocaOp::new(ctx, temp_ty);
            temp.get_operation().insert_at_back(insert_block, ctx);
            let temp_ptr = temp.get_result(ctx);
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
            let size_value = integer_constant(ctx, usize_ty, size as u128)?;
            size_value.get_operation().insert_at_back(insert_block, ctx);
            let size_value = size_value.get_result(ctx);
            for (to, from) in [(temp_ptr, x), (x, y), (y, temp_ptr)] {
                let call = mir_dialect::ops::CallOp::new_direct(
                    ctx,
                    memcpy.clone(),
                    vec![to, from, size_value],
                    Some(ptr_ty),
                );
                call.get_operation().insert_at_back(insert_block, ctx);
            }
            Ok(true)
        }
        // NEON 64-bit vector intrinsics reached through std's hashbrown group
        // scan (`uint8x8_t`/`int8x8_t`: 8 byte lanes in 64 bits). The backend
        // has no vector registers, so the lanes are computed as scalar SWAR /
        // per-lane code on the vector's u64 bit pattern; the D-register
        // vectors are 8 bytes, so a u64 round-trips their layout exactly.
        "simd_splat" => {
            if args.len() != 1 {
                return Err(format!(
                    "unsupported simd_splat intrinsic arity: {}",
                    args.len()
                ));
            }
            let vector_ty = mono_ty(tcx, state, instance.args.type_at(0));
            require_8_byte_vector(tcx, "simd_splat", vector_ty)?;
            let lane = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            if lane
                .get_type(ctx)
                .deref(ctx)
                .downcast_ref::<IntegerType>()
                .map(|ty| ty.width())
                != Some(8)
            {
                return Err("unsupported simd_splat lane type (expected u8)".to_string());
            }
            let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
            let lane = widen_to_u64(ctx, insert_block, lane)?;
            let byte_mask = integer_constant(ctx, u64_ty, 0xff)?;
            byte_mask.get_operation().insert_at_back(insert_block, ctx);
            let lane = emit_op(
                mir_dialect::ops::BitAndOp::new(ctx, lane, byte_mask.get_result(ctx))
                    .get_operation(),
                ctx,
                insert_block,
            );
            let spread = integer_constant(ctx, u64_ty, 0x0101_0101_0101_0101)?;
            spread.get_operation().insert_at_back(insert_block, ctx);
            let bits = emit_op(
                mir_dialect::ops::MulOp::new(ctx, lane, spread.get_result(ctx)).get_operation(),
                ctx,
                insert_block,
            );
            store_simd_bits(tcx, ctx, state, insert_block, body, destination, bits)?;
            Ok(true)
        }
        "simd_or" => {
            if args.len() != 2 {
                return Err(format!(
                    "unsupported simd_or intrinsic arity: {}",
                    args.len()
                ));
            }
            let vector_ty = mono_ty(tcx, state, instance.args.type_at(0));
            require_8_byte_vector(tcx, "simd_or", vector_ty)?;
            let lhs = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let rhs = import_operand(tcx, ctx, state, insert_block, body, &args[1].node)?;
            let lhs = simd_value_bits(ctx, insert_block, lhs)?;
            let rhs = simd_value_bits(ctx, insert_block, rhs)?;
            let bits = emit_op(
                mir_dialect::ops::BitOrOp::new(ctx, lhs, rhs).get_operation(),
                ctx,
                insert_block,
            );
            store_simd_bits(tcx, ctx, state, insert_block, body, destination, bits)?;
            Ok(true)
        }
        "simd_extract" | "simd_extract_dyn" => {
            if args.len() != 2 {
                return Err(format!(
                    "unsupported simd_extract intrinsic arity: {}",
                    args.len()
                ));
            }
            let vector_ty = mono_ty(tcx, state, instance.args.type_at(0));
            require_8_byte_vector(tcx, "simd_extract", vector_ty)?;
            // Only the single-lane `uint64x1_t -> u64` form: the lane index
            // can only be zero, so the extract is the bit pattern itself.
            let lane_size = layout_size_of_ty(
                tcx,
                mono_ty(tcx, state, instance.args.type_at(1)),
            )?;
            if lane_size != 8 {
                return Err(format!(
                    "unsupported simd_extract lane size: {lane_size} bytes (expected a \
                     single-lane 64-bit vector)"
                ));
            }
            let vector = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let bits = simd_value_bits(ctx, insert_block, vector)?;
            let dest_ty =
                convert_immediate_ty(tcx, ctx, mono_ty(tcx, state, destination.ty(body, tcx).ty))?;
            let result = cast_value_to_type(ctx, insert_block, bits, dest_ty);
            store_place(tcx, ctx, state, insert_block, body, destination, result)?;
            Ok(true)
        }
        "simd_eq" => {
            if args.len() != 2 {
                return Err(format!(
                    "unsupported simd_eq intrinsic arity: {}",
                    args.len()
                ));
            }
            let vector_ty = mono_ty(tcx, state, instance.args.type_at(0));
            require_8_byte_vector(tcx, "simd_eq", vector_ty)?;
            let lhs = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let rhs = import_operand(tcx, ctx, state, insert_block, body, &args[1].node)?;
            let lhs = simd_value_bits(ctx, insert_block, lhs)?;
            let rhs = simd_value_bits(ctx, insert_block, rhs)?;
            // Byte-wise equality mask via the SWAR zero-byte trick on the
            // xor: `(t - 0x01..) & !t & 0x80..` marks equal lanes with 0x80,
            // then the mark is smeared down to fill the lane with 0xff.
            let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
            let constant = |ctx: &mut Context, bits: u128| -> Result<Value, String> {
                let op = integer_constant(ctx, u64_ty, bits)?;
                op.get_operation().insert_at_back(insert_block, ctx);
                Ok(op.get_result(ctx))
            };
            let t = emit_op(
                mir_dialect::ops::BitXorOp::new(ctx, lhs, rhs).get_operation(),
                ctx,
                insert_block,
            );
            let low_ones = constant(ctx, 0x0101_0101_0101_0101)?;
            let minus = emit_op(
                mir_dialect::ops::SubOp::new(ctx, t, low_ones).get_operation(),
                ctx,
                insert_block,
            );
            let all_ones = constant(ctx, u64::MAX as u128)?;
            let not_t = emit_op(
                mir_dialect::ops::BitXorOp::new(ctx, t, all_ones).get_operation(),
                ctx,
                insert_block,
            );
            let and1 = emit_op(
                mir_dialect::ops::BitAndOp::new(ctx, minus, not_t).get_operation(),
                ctx,
                insert_block,
            );
            let high_bits = constant(ctx, 0x8080_8080_8080_8080)?;
            let mut mask = emit_op(
                mir_dialect::ops::BitAndOp::new(ctx, and1, high_bits).get_operation(),
                ctx,
                insert_block,
            );
            for shift in [1u128, 2, 4] {
                let amount = constant(ctx, shift)?;
                let shifted = emit_op(
                    mir_dialect::ops::ShrOp::new(ctx, mask, amount).get_operation(),
                    ctx,
                    insert_block,
                );
                mask = emit_op(
                    mir_dialect::ops::BitOrOp::new(ctx, mask, shifted).get_operation(),
                    ctx,
                    insert_block,
                );
            }
            store_simd_bits(tcx, ctx, state, insert_block, body, destination, mask)?;
            Ok(true)
        }
        "simd_lt" | "simd_ge" => {
            if args.len() != 2 {
                return Err(format!(
                    "unsupported {name} intrinsic arity: {}",
                    args.len()
                ));
            }
            let vector_ty = mono_ty(tcx, state, instance.args.type_at(0));
            require_8_byte_vector(tcx, name.as_str(), vector_ty)?;
            let lhs = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let rhs = import_operand(tcx, ctx, state, insert_block, body, &args[1].node)?;
            let lhs = simd_value_bits(ctx, insert_block, lhs)?;
            let rhs = simd_value_bits(ctx, insert_block, rhs)?;
            // Signed per-lane compare, one i8 lane at a time (correctness
            // over speed; the group scan only runs on 8 lanes).
            let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
            let i8_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Signed).into();
            let constant = |ctx: &mut Context, bits: u128| -> Result<Value, String> {
                let op = integer_constant(ctx, u64_ty, bits)?;
                op.get_operation().insert_at_back(insert_block, ctx);
                Ok(op.get_result(ctx))
            };
            let zero = constant(ctx, 0)?;
            let mut result = zero;
            for lane in 0..8u32 {
                let amount = constant(ctx, (lane * 8) as u128)?;
                let lhs_lane = emit_op(
                    mir_dialect::ops::ShrOp::new(ctx, lhs, amount).get_operation(),
                    ctx,
                    insert_block,
                );
                let lhs_lane = cast_value_to_type(ctx, insert_block, lhs_lane, i8_ty);
                let rhs_lane = emit_op(
                    mir_dialect::ops::ShrOp::new(ctx, rhs, amount).get_operation(),
                    ctx,
                    insert_block,
                );
                let rhs_lane = cast_value_to_type(ctx, insert_block, rhs_lane, i8_ty);
                let cond = if name == "simd_lt" {
                    mir_dialect::ops::LtOp::new(ctx, lhs_lane, rhs_lane).get_operation()
                } else {
                    mir_dialect::ops::GeOp::new(ctx, lhs_lane, rhs_lane).get_operation()
                };
                let cond = emit_op(cond, ctx, insert_block);
                let cond = cast_value_to_type(ctx, insert_block, cond, u64_ty);
                // true -> 0xff in this lane: (0 - cond) & 0xff, shifted home.
                let neg = emit_op(
                    mir_dialect::ops::SubOp::new(ctx, zero, cond).get_operation(),
                    ctx,
                    insert_block,
                );
                let byte_mask = constant(ctx, 0xff)?;
                let lane_mask = emit_op(
                    mir_dialect::ops::BitAndOp::new(ctx, neg, byte_mask).get_operation(),
                    ctx,
                    insert_block,
                );
                let placed = emit_op(
                    mir_dialect::ops::ShlOp::new(ctx, lane_mask, amount).get_operation(),
                    ctx,
                    insert_block,
                );
                result = emit_op(
                    mir_dialect::ops::BitOrOp::new(ctx, result, placed).get_operation(),
                    ctx,
                    insert_block,
                );
            }
            store_simd_bits(tcx, ctx, state, insert_block, body, destination, result)?;
            Ok(true)
        }
        "saturating_add" | "saturating_sub" => {
            if args.len() != 2 {
                return Err(format!(
                    "unsupported saturating intrinsic arity: {}",
                    args.len()
                ));
            }
            let lhs = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let rhs = import_operand(tcx, ctx, state, insert_block, body, &args[1].node)?;
            if !is_unsigned_integer_value(ctx, lhs) {
                let result = lower_signed_saturating(
                    ctx,
                    insert_block,
                    name == "saturating_add",
                    lhs,
                    rhs,
                )?;
                store_place(tcx, ctx, state, insert_block, body, destination, result)?;
                return Ok(true);
            }
            let int_ty = lhs.get_type(ctx);
            let result = if name == "saturating_add" {
                // r = a + b; if r < a saturate to all-ones: r | (0 - (r < a)).
                let sum = mir_dialect::ops::AddOp::new(ctx, lhs, rhs).get_operation();
                sum.insert_at_back(insert_block, ctx);
                let sum = sum.deref(ctx).get_result(0);
                let overflow = mir_dialect::ops::LtOp::new(ctx, sum, lhs).get_operation();
                overflow.insert_at_back(insert_block, ctx);
                let overflow = overflow.deref(ctx).get_result(0);
                let overflow = cast_value_to_type(ctx, insert_block, overflow, int_ty);
                let zero = integer_constant(ctx, int_ty, 0)?;
                zero.get_operation().insert_at_back(insert_block, ctx);
                let mask =
                    mir_dialect::ops::SubOp::new(ctx, zero.get_result(ctx), overflow).get_operation();
                mask.insert_at_back(insert_block, ctx);
                let mask = mask.deref(ctx).get_result(0);
                let saturated = mir_dialect::ops::BitOrOp::new(ctx, sum, mask).get_operation();
                saturated.insert_at_back(insert_block, ctx);
                saturated.deref(ctx).get_result(0)
            } else {
                // r = (a - b) & (0 - (a >= b)): zero when the subtraction
                // would underflow.
                let diff = mir_dialect::ops::SubOp::new(ctx, lhs, rhs).get_operation();
                diff.insert_at_back(insert_block, ctx);
                let diff = diff.deref(ctx).get_result(0);
                let no_borrow = mir_dialect::ops::GeOp::new(ctx, lhs, rhs).get_operation();
                no_borrow.insert_at_back(insert_block, ctx);
                let no_borrow = no_borrow.deref(ctx).get_result(0);
                let no_borrow = cast_value_to_type(ctx, insert_block, no_borrow, int_ty);
                let zero = integer_constant(ctx, int_ty, 0)?;
                zero.get_operation().insert_at_back(insert_block, ctx);
                let mask = mir_dialect::ops::SubOp::new(ctx, zero.get_result(ctx), no_borrow)
                    .get_operation();
                mask.insert_at_back(insert_block, ctx);
                let mask = mask.deref(ctx).get_result(0);
                let saturated = mir_dialect::ops::BitAndOp::new(ctx, diff, mask).get_operation();
                saturated.insert_at_back(insert_block, ctx);
                saturated.deref(ctx).get_result(0)
            };
            store_place(tcx, ctx, state, insert_block, body, destination, result)?;
            Ok(true)
        }
        "exact_div" => {
            if args.len() != 2 {
                return Err(format!(
                    "unsupported exact_div intrinsic arity: {}",
                    args.len()
                ));
            }
            let lhs = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let rhs = import_operand(tcx, ctx, state, insert_block, body, &args[1].node)?;
            // `exact_div` is division with UB on a non-zero remainder (or
            // overflow), so plain division is a valid lowering.
            let result = if is_128_bit_integer_value(ctx, lhs) {
                lower_i128_divrem(
                    ctx,
                    state.module_body,
                    insert_block,
                    BinOp::Div,
                    lhs,
                    rhs,
                )?
            } else {
                emit_op(
                    mir_dialect::ops::DivOp::new(ctx, lhs, rhs).get_operation(),
                    ctx,
                    insert_block,
                )
            };
            store_place(tcx, ctx, state, insert_block, body, destination, result)?;
            Ok(true)
        }
        "float_to_int_unchecked" => {
            if args.len() != 1 {
                return Err(format!(
                    "unsupported float_to_int_unchecked intrinsic arity: {}",
                    args.len()
                ));
            }
            let input = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let dest_ty =
                convert_immediate_ty(tcx, ctx, mono_ty(tcx, state, destination.ty(body, tcx).ty))?;
            // In-range inputs are the caller's obligation, so the saturating
            // `as`-cast lowering is a valid refinement at every width.
            let result = if dest_ty
                .deref(ctx)
                .downcast_ref::<IntegerType>()
                .is_some_and(|ty| ty.width() == 128)
            {
                lower_float_to_i128_sat(ctx, state.module_body, insert_block, input, dest_ty)?
            } else {
                let cast = mir_dialect::ops::CastOp::new(ctx, input, dest_ty);
                cast.get_operation().insert_at_back(insert_block, ctx);
                cast.get_result(ctx)
            };
            store_place(tcx, ctx, state, insert_block, body, destination, result)?;
            Ok(true)
        }
        "arith_offset" => {
            if args.len() != 2 {
                return Err(format!(
                    "unsupported arith_offset intrinsic arity: {}",
                    args.len()
                ));
            }
            let ptr = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let count = import_operand(tcx, ctx, state, insert_block, body, &args[1].node)?;
            let elem = normalize_ty(tcx, mono_ty(tcx, state, instance.args.type_at(0)));
            let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
            let elem_size = rustc_layout_size_of_ty(tcx, typing_env, elem)?;
            let byte_offset = scale_index(ctx, insert_block, count, elem_size)?;
            let offset = mir_dialect::ops::PtrOffsetOp::new(ctx, ptr, byte_offset);
            offset.get_operation().insert_at_back(insert_block, ctx);
            store_place(
                tcx,
                ctx,
                state,
                insert_block,
                body,
                destination,
                offset.get_result(ctx),
            )?;
            Ok(true)
        }
        "size_of_val" | "align_of_val" => {
            if args.len() != 1 {
                return Err(format!(
                    "unsupported {name} intrinsic arity: {}",
                    args.len()
                ));
            }
            let elem = normalize_ty(tcx, mono_ty(tcx, state, instance.args.type_at(0)));
            let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
            let usize_ty: TypeHandle = usize_ty(ctx).into();
            let result = match runtime_ty(elem).kind() {
                rustc_middle::ty::TyKind::Dynamic(_, _) => {
                    // Fat pointer {data, vtable}; the real vtable stores size
                    // at +8 and align at +16.
                    let fat = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
                    let ptr_ty = llvm_ptr_ty(ctx);
                    let vtable = mir_dialect::ops::ExtractValueOp::new(ctx, fat, vec![1], ptr_ty);
                    vtable.get_operation().insert_at_back(insert_block, ctx);
                    let offset = if name == "size_of_val" { 8 } else { 16 };
                    let addr = ptr_offset_const(ctx, insert_block, vtable.get_result(ctx), offset)?;
                    let load = mir_dialect::ops::LoadOp::new(ctx, addr, usize_ty);
                    load.get_operation().insert_at_back(insert_block, ctx);
                    load.get_result(ctx)
                }
                rustc_middle::ty::TyKind::Slice(inner) => {
                    let fat = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
                    let len = mir_dialect::ops::ExtractValueOp::new(ctx, fat, vec![1], usize_ty);
                    len.get_operation().insert_at_back(insert_block, ctx);
                    if name == "size_of_val" {
                        let elem_size = rustc_layout_size_of_ty(tcx, typing_env, *inner)?;
                        scale_index(ctx, insert_block, len.get_result(ctx), elem_size)?
                    } else {
                        let align = rustc_layout_align_of_ty(tcx, typing_env, *inner)?;
                        let constant = integer_constant(ctx, usize_ty, align as u128)?;
                        constant.get_operation().insert_at_back(insert_block, ctx);
                        constant.get_result(ctx)
                    }
                }
                _ => {
                    let bytes = if name == "size_of_val" {
                        rustc_layout_size_of_ty(tcx, typing_env, elem)?
                    } else {
                        rustc_layout_align_of_ty(tcx, typing_env, elem)?
                    };
                    let constant = integer_constant(ctx, usize_ty, bytes as u128)?;
                    constant.get_operation().insert_at_back(insert_block, ctx);
                    constant.get_result(ctx)
                }
            };
            store_place(tcx, ctx, state, insert_block, body, destination, result)?;
            Ok(true)
        }
        "drop_glue" => {
            let ty = mono_ty(tcx, state, instance.args.type_at(0));
            let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
            if !ty.needs_drop(tcx, typing_env) {
                return Ok(true);
            }
            if args.len() != 1 {
                return Err(format!(
                    "unsupported drop_glue intrinsic arity: {}",
                    args.len()
                ));
            }
            let drop_instance = Instance::resolve_drop_in_place(tcx, ty);
            let mut legaliser = Legaliser::default();
            let symbol = legaliser.legalise(tcx.symbol_name(drop_instance).name);
            import_upstream_instance(tcx, ctx, state.module_body, drop_instance)
                .map_err(|error| format!("while importing drop glue for {ty:?}: {error}"))?;
            let addr = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let mut call_args = Vec::new();
            lower_abi_call_arg(ctx, insert_block, addr, &mut call_args)?;
            let call = mir_dialect::ops::CallOp::new_direct(ctx, symbol, call_args, None);
            call.get_operation().insert_at_back(insert_block, ctx);
            Ok(true)
        }
        _ if float_math_intrinsic(name.as_str()).is_some() => lower_float_math_intrinsic(
            tcx,
            ctx,
            state,
            insert_block,
            body,
            name.as_str(),
            args,
            destination,
        ),
        _ => Ok(false),
    }
}

/// How a float math intrinsic reaches the backend: as a call to an
/// `llvm_<op>_f{32,64}` declaration that instruction selection turns into
/// one FP instruction (mirroring LLVM's `llvm.<op>.f32` intrinsics), or as
/// a call into libm / compiler-builtins, the way LLVM's own lowering
/// expands the transcendental intrinsics.
pub(super) enum FloatMathLowering {
    Inline(&'static str),
    Libcall {
        f32_symbol: &'static str,
        f64_symbol: &'static str,
    },
}

pub(super) fn float_math_intrinsic(name: &str) -> Option<FloatMathLowering> {
    use FloatMathLowering::{Inline, Libcall};
    // Strip the width suffix (`sqrtf32`, `round_ties_even_f64`); the
    // generic `fabs` carries none, and the operand type decides the width.
    let base = name
        .strip_suffix("f32")
        .or_else(|| name.strip_suffix("f64"))
        .unwrap_or(name);
    let base = base.strip_suffix('_').unwrap_or(base);
    Some(match base {
        "sqrt" => Inline("sqrt"),
        "fabs" => Inline("fabs"),
        "floor" => Inline("floor"),
        "ceil" => Inline("ceil"),
        "trunc" => Inline("trunc"),
        "round" => Inline("round"),
        "round_ties_even" => Inline("rint"),
        "minnum" | "minimum_number_nsz" => Inline("minnum"),
        "maxnum" | "maximum_number_nsz" => Inline("maxnum"),
        "minimum" => Inline("minimum"),
        "maximum" => Inline("maximum"),
        "exp" => Libcall { f32_symbol: "expf", f64_symbol: "exp" },
        "exp2" => Libcall { f32_symbol: "exp2f", f64_symbol: "exp2" },
        "log" => Libcall { f32_symbol: "logf", f64_symbol: "log" },
        "log2" => Libcall { f32_symbol: "log2f", f64_symbol: "log2" },
        "log10" => Libcall { f32_symbol: "log10f", f64_symbol: "log10" },
        "sin" => Libcall { f32_symbol: "sinf", f64_symbol: "sin" },
        "cos" => Libcall { f32_symbol: "cosf", f64_symbol: "cos" },
        "pow" => Libcall { f32_symbol: "powf", f64_symbol: "pow" },
        "fma" | "fmuladd" => Libcall { f32_symbol: "fmaf", f64_symbol: "fma" },
        "copysign" => Libcall { f32_symbol: "copysignf", f64_symbol: "copysign" },
        // Integer power: compiler-builtins' `__powi{s,d}f2`, as LLVM
        // expands `llvm.powi`.
        "powi" => Libcall { f32_symbol: "__powisf2", f64_symbol: "__powidf2" },
        _ => return None,
    })
}

/// Lower a float math intrinsic (see [float_math_intrinsic]) to a call the
/// backend resolves. Returns `Ok(false)` for float widths this backend
/// does not lower (f16/f128), leaving the call unresolved as before.
#[allow(clippy::too_many_arguments)]
pub(super) fn lower_float_math_intrinsic<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    name: &str,
    args: &[rustc_span::Spanned<Operand<'tcx>>],
    destination: &Place<'tcx>,
) -> Result<bool, String> {
    let Some(lowering) = float_math_intrinsic(name) else {
        return Ok(false);
    };
    if args.is_empty() {
        return Err(format!("unsupported {name} intrinsic arity: 0"));
    }
    let mut values = Vec::with_capacity(args.len());
    for arg in args {
        values.push(import_operand(tcx, ctx, state, insert_block, body, &arg.node)?);
    }
    let float_ty = values[0].get_type(ctx);
    let is_f32 = float_ty.deref(ctx).downcast_ref::<FP32Type>().is_some();
    if !is_f32 && float_ty.deref(ctx).downcast_ref::<FP64Type>().is_none() {
        return Ok(false);
    }
    let callee: crate::identifier::Identifier = match lowering {
        FloatMathLowering::Inline(op) => {
            format!("llvm_{op}_f{}", if is_f32 { 32 } else { 64 })
        }
        FloatMathLowering::Libcall {
            f32_symbol,
            f64_symbol,
        } => (if is_f32 { f32_symbol } else { f64_symbol }).to_string(),
    }
    .try_into()
    .unwrap();
    let arg_tys: Vec<TypeHandle> = values.iter().map(|value| value.get_type(ctx)).collect();
    declare_external_function(ctx, state.module_body, callee.clone(), arg_tys, Some(float_ty));
    let call = mir_dialect::ops::CallOp::new_direct(ctx, callee, values, Some(float_ty));
    call.get_operation().insert_at_back(insert_block, ctx);
    let result = call.get_result(ctx);
    store_place(tcx, ctx, state, insert_block, body, destination, result)?;
    Ok(true)
}

/// Zero-extend an integer value of width <= 64 to u64.
pub(super) fn widen_to_u64(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    value: Value,
) -> Result<Value, String> {
    let width = value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .map(|ty| ty.width())
        .ok_or_else(|| "bit intrinsic on non-integer type".to_string())?;
    if width > 64 {
        // Every 128-bit consumer splits halves via `split_u128_halves`
        // instead of widening; this guards against a new caller forgetting.
        return Err("128-bit operand reached widen_to_u64 (split into halves instead)".to_string());
    }
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    Ok(cast_value_to_type(ctx, insert_block, value, u64_ty))
}

/// Branch-free trailing-zero count of the low `width` bits of a u64 value:
/// `popcount(((x & -x) - 1) & width_mask)`. For `x == 0` the mask makes this
/// exactly `width`, as the `cttz` intrinsic requires.
pub(super) fn emit_cttz64(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    x: Value,
    width: u32,
) -> Result<Value, String> {
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let zero = integer_constant(ctx, u64_ty, 0)?;
    zero.get_operation().insert_at_back(insert_block, ctx);
    let neg = emit_op(
        mir_dialect::ops::SubOp::new(ctx, zero.get_result(ctx), x).get_operation(),
        ctx,
        insert_block,
    );
    let low_bit = emit_op(
        mir_dialect::ops::BitAndOp::new(ctx, x, neg).get_operation(),
        ctx,
        insert_block,
    );
    let one = integer_constant(ctx, u64_ty, 1)?;
    one.get_operation().insert_at_back(insert_block, ctx);
    let below = emit_op(
        mir_dialect::ops::SubOp::new(ctx, low_bit, one.get_result(ctx)).get_operation(),
        ctx,
        insert_block,
    );
    let width_mask = integer_constant(ctx, u64_ty, u64::MAX as u128 >> (64 - width as usize))?;
    width_mask.get_operation().insert_at_back(insert_block, ctx);
    let masked = emit_op(
        mir_dialect::ops::BitAndOp::new(ctx, below, width_mask.get_result(ctx)).get_operation(),
        ctx,
        insert_block,
    );
    emit_popcount64(ctx, insert_block, masked)
}

/// The low and high 64-bit halves of a 128-bit value, as u64s.
pub(super) fn split_u128_halves(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    input: Value,
) -> Result<(Value, Value), String> {
    let u128_ty: TypeHandle = IntegerType::get(ctx, 128, Signedness::Unsigned).into();
    let bits = cast_value_to_type(ctx, insert_block, input, u128_ty);
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let lo = cast_value_to_type(ctx, insert_block, bits, u64_ty);
    let sixty_four = integer_constant(ctx, u128_ty, 64)?;
    sixty_four.get_operation().insert_at_back(insert_block, ctx);
    let hi_wide = emit_op(
        mir_dialect::ops::ShrOp::new(ctx, bits, sixty_four.get_result(ctx)).get_operation(),
        ctx,
        insert_block,
    );
    let hi = cast_value_to_type(ctx, insert_block, hi_wide, u64_ty);
    Ok((lo, hi))
}

/// 128-bit trailing-zero count on 64-bit halves:
/// `cttz(x) = cttz64(lo) + (lo == 0 ? cttz64(hi) : 0)`. `cttz64` already
/// yields 64 for a zero half, so the total is 128 for `x == 0`.
pub(super) fn emit_cttz128(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    input: Value,
) -> Result<Value, String> {
    let (lo, hi) = split_u128_halves(ctx, insert_block, input)?;
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();

    let lo_count = emit_cttz64(ctx, insert_block, lo, 64)?;
    let hi_count = emit_cttz64(ctx, insert_block, hi, 64)?;

    // (lo == 0 ? hi_count : 0) as `(0 - (lo == 0)) & hi_count`, branch-free.
    let zero = integer_constant(ctx, u64_ty, 0)?;
    zero.get_operation().insert_at_back(insert_block, ctx);
    let lo_is_zero = emit_op(
        mir_dialect::ops::EqOp::new(ctx, lo, zero.get_result(ctx)).get_operation(),
        ctx,
        insert_block,
    );
    let lo_is_zero = cast_value_to_type(ctx, insert_block, lo_is_zero, u64_ty);
    let mask = emit_op(
        mir_dialect::ops::SubOp::new(ctx, zero.get_result(ctx), lo_is_zero).get_operation(),
        ctx,
        insert_block,
    );
    let extra = emit_op(
        mir_dialect::ops::BitAndOp::new(ctx, mask, hi_count).get_operation(),
        ctx,
        insert_block,
    );
    Ok(emit_op(
        mir_dialect::ops::AddOp::new(ctx, lo_count, extra).get_operation(),
        ctx,
        insert_block,
    ))
}

/// Branch-free leading-zero count of the low `width` bits of `x` (a u64
/// value with any widening garbage cleared as part of the MSB alignment).
pub(super) fn emit_ctlz64(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    mut x: Value,
    width: u32,
) -> Result<Value, String> {
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    // Align the operand's MSB with bit 63 so a 64-bit count is exact
    // (and clears any widening garbage above `width` on the way).
    if width < 64 {
        let up = integer_constant(ctx, u64_ty, (64 - width) as u128)?;
        up.get_operation().insert_at_back(insert_block, ctx);
        x = emit_op(
            mir_dialect::ops::ShlOp::new(ctx, x, up.get_result(ctx)).get_operation(),
            ctx,
            insert_block,
        );
    }
    // Smear the highest set bit rightward; the complement then has
    // ones exactly in the leading-zero positions.
    for shift in [1u128, 2, 4, 8, 16, 32] {
        let amount = integer_constant(ctx, u64_ty, shift)?;
        amount.get_operation().insert_at_back(insert_block, ctx);
        let shifted = emit_op(
            mir_dialect::ops::ShrOp::new(ctx, x, amount.get_result(ctx)).get_operation(),
            ctx,
            insert_block,
        );
        x = emit_op(
            mir_dialect::ops::BitOrOp::new(ctx, x, shifted).get_operation(),
            ctx,
            insert_block,
        );
    }
    let ones = integer_constant(ctx, u64_ty, u64::MAX as u128)?;
    ones.get_operation().insert_at_back(insert_block, ctx);
    let inverted = emit_op(
        mir_dialect::ops::BitXorOp::new(ctx, x, ones.get_result(ctx)).get_operation(),
        ctx,
        insert_block,
    );
    emit_popcount64(ctx, insert_block, inverted)
}

/// Branch-free 128-bit leading-zero count, mirroring [emit_cttz128] with the
/// halves' roles swapped: `ctlz(x) = ctlz64(hi) + (hi == 0 ? ctlz64(lo) : 0)`
/// (`ctlz64` of a zero half yields 64, so the total reaches 128 for zero).
pub(super) fn emit_ctlz128(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    input: Value,
) -> Result<Value, String> {
    let (lo, hi) = split_u128_halves(ctx, insert_block, input)?;
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();

    let hi_count = emit_ctlz64(ctx, insert_block, hi, 64)?;
    let lo_count = emit_ctlz64(ctx, insert_block, lo, 64)?;

    // (hi == 0 ? lo_count : 0) as `(0 - (hi == 0)) & lo_count`, branch-free.
    let zero = integer_constant(ctx, u64_ty, 0)?;
    zero.get_operation().insert_at_back(insert_block, ctx);
    let hi_is_zero = emit_op(
        mir_dialect::ops::EqOp::new(ctx, hi, zero.get_result(ctx)).get_operation(),
        ctx,
        insert_block,
    );
    let hi_is_zero = cast_value_to_type(ctx, insert_block, hi_is_zero, u64_ty);
    let mask = emit_op(
        mir_dialect::ops::SubOp::new(ctx, zero.get_result(ctx), hi_is_zero).get_operation(),
        ctx,
        insert_block,
    );
    let extra = emit_op(
        mir_dialect::ops::BitAndOp::new(ctx, mask, lo_count).get_operation(),
        ctx,
        insert_block,
    );
    Ok(emit_op(
        mir_dialect::ops::AddOp::new(ctx, hi_count, extra).get_operation(),
        ctx,
        insert_block,
    ))
}

/// Branch-free 64-bit population count.
pub(super) fn emit_popcount64(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    mut x: Value,
) -> Result<Value, String> {
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let constant = |ctx: &mut Context, bits: u128| -> Result<Value, String> {
        let op = integer_constant(ctx, u64_ty, bits)?;
        op.get_operation().insert_at_back(insert_block, ctx);
        Ok(op.get_result(ctx))
    };
    fn emit(ctx: &Context, insert_block: Ptr<BasicBlock>, op: Ptr<Operation>) -> Value {
        op.insert_at_back(insert_block, ctx);
        op.deref(ctx).get_result(0)
    }

    let one = constant(ctx, 1)?;
    let shifted = mir_dialect::ops::ShrOp::new(ctx, x, one).get_operation();
    let shifted = emit(ctx, insert_block, shifted);
    let mask55 = constant(ctx, 0x5555_5555_5555_5555)?;
    let and55 = mir_dialect::ops::BitAndOp::new(ctx, shifted, mask55).get_operation();
    let and55 = emit(ctx, insert_block, and55);
    let sub = mir_dialect::ops::SubOp::new(ctx, x, and55).get_operation();
    x = emit(ctx, insert_block, sub);

    let mask33 = constant(ctx, 0x3333_3333_3333_3333)?;
    let low_pairs = mir_dialect::ops::BitAndOp::new(ctx, x, mask33).get_operation();
    let low_pairs = emit(ctx, insert_block, low_pairs);
    let two = constant(ctx, 2)?;
    let shr2 = mir_dialect::ops::ShrOp::new(ctx, x, two).get_operation();
    let shr2 = emit(ctx, insert_block, shr2);
    let high_pairs = mir_dialect::ops::BitAndOp::new(ctx, shr2, mask33).get_operation();
    let high_pairs = emit(ctx, insert_block, high_pairs);
    let pair_sum = mir_dialect::ops::AddOp::new(ctx, low_pairs, high_pairs).get_operation();
    x = emit(ctx, insert_block, pair_sum);

    let four = constant(ctx, 4)?;
    let shr4 = mir_dialect::ops::ShrOp::new(ctx, x, four).get_operation();
    let shr4 = emit(ctx, insert_block, shr4);
    let nibble_sum = mir_dialect::ops::AddOp::new(ctx, x, shr4).get_operation();
    let nibble_sum = emit(ctx, insert_block, nibble_sum);
    let mask0f = constant(ctx, 0x0f0f_0f0f_0f0f_0f0f)?;
    let nibbles = mir_dialect::ops::BitAndOp::new(ctx, nibble_sum, mask0f).get_operation();
    let nibbles = emit(ctx, insert_block, nibbles);

    let ones = constant(ctx, 0x0101_0101_0101_0101)?;
    let spread = mir_dialect::ops::MulOp::new(ctx, nibbles, ones).get_operation();
    let spread = emit(ctx, insert_block, spread);
    let fifty_six = constant(ctx, 56)?;
    let total = mir_dialect::ops::ShrOp::new(ctx, spread, fifty_six).get_operation();
    Ok(emit(ctx, insert_block, total))
}

/// A u128 value assembled as `(hi << 64) | lo` from two u64 halves.
pub(super) fn join_u128_halves(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    lo: Value,
    hi: Value,
) -> Result<Value, String> {
    let u128_ty: TypeHandle = IntegerType::get(ctx, 128, Signedness::Unsigned).into();
    let lo_wide = cast_value_to_type(ctx, insert_block, lo, u128_ty);
    let hi_wide = cast_value_to_type(ctx, insert_block, hi, u128_ty);
    let sixty_four = integer_constant(ctx, u128_ty, 64)?;
    sixty_four.get_operation().insert_at_back(insert_block, ctx);
    let shifted = emit_op(
        mir_dialect::ops::ShlOp::new(ctx, hi_wide, sixty_four.get_result(ctx)).get_operation(),
        ctx,
        insert_block,
    );
    Ok(emit_op(
        mir_dialect::ops::BitOrOp::new(ctx, shifted, lo_wide).get_operation(),
        ctx,
        insert_block,
    ))
}

/// Byte swap of the low `width` bits of a u64 value (`width` a multiple of
/// 8): byte lane `i` moves to lane `width/8 - 1 - i`, one lane at a time
/// (correctness over speed, matching the other bit-intrinsic expansions).
pub(super) fn emit_bswap64(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    x: Value,
    width: u32,
) -> Result<Value, String> {
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let constant = |ctx: &mut Context, bits: u128| -> Result<Value, String> {
        let op = integer_constant(ctx, u64_ty, bits)?;
        op.get_operation().insert_at_back(insert_block, ctx);
        Ok(op.get_result(ctx))
    };
    let lanes = width / 8;
    let byte_mask = constant(ctx, 0xff)?;
    let mut result = constant(ctx, 0)?;
    for lane in 0..lanes {
        let down = constant(ctx, (lane * 8) as u128)?;
        let shifted = emit_op(
            mir_dialect::ops::ShrOp::new(ctx, x, down).get_operation(),
            ctx,
            insert_block,
        );
        let byte = emit_op(
            mir_dialect::ops::BitAndOp::new(ctx, shifted, byte_mask).get_operation(),
            ctx,
            insert_block,
        );
        let up = constant(ctx, ((lanes - 1 - lane) * 8) as u128)?;
        let placed = emit_op(
            mir_dialect::ops::ShlOp::new(ctx, byte, up).get_operation(),
            ctx,
            insert_block,
        );
        result = emit_op(
            mir_dialect::ops::BitOrOp::new(ctx, result, placed).get_operation(),
            ctx,
            insert_block,
        );
    }
    Ok(result)
}

/// Bit reversal of the low `width` bits of a u64 value: the classic SWAR
/// swaps of adjacent 1-, 2-, and 4-bit groups reverse each byte, a byte swap
/// reverses the full 64 bits, and (for `width < 64`) a final right shift
/// re-aligns the reversed field, pushing out the reversed zero-extension
/// garbage that landed below it.
pub(super) fn emit_bitreverse64(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    mut x: Value,
    width: u32,
) -> Result<Value, String> {
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let constant = |ctx: &mut Context, bits: u128| -> Result<Value, String> {
        let op = integer_constant(ctx, u64_ty, bits)?;
        op.get_operation().insert_at_back(insert_block, ctx);
        Ok(op.get_result(ctx))
    };
    for (mask_bits, shift) in [
        (0x5555_5555_5555_5555u128, 1u128),
        (0x3333_3333_3333_3333, 2),
        (0x0f0f_0f0f_0f0f_0f0f, 4),
    ] {
        // x = ((x >> s) & m) | ((x & m) << s)
        let mask = constant(ctx, mask_bits)?;
        let amount = constant(ctx, shift)?;
        let down = emit_op(
            mir_dialect::ops::ShrOp::new(ctx, x, amount).get_operation(),
            ctx,
            insert_block,
        );
        let down = emit_op(
            mir_dialect::ops::BitAndOp::new(ctx, down, mask).get_operation(),
            ctx,
            insert_block,
        );
        let up = emit_op(
            mir_dialect::ops::BitAndOp::new(ctx, x, mask).get_operation(),
            ctx,
            insert_block,
        );
        let up = emit_op(
            mir_dialect::ops::ShlOp::new(ctx, up, amount).get_operation(),
            ctx,
            insert_block,
        );
        x = emit_op(
            mir_dialect::ops::BitOrOp::new(ctx, down, up).get_operation(),
            ctx,
            insert_block,
        );
    }
    x = emit_bswap64(ctx, insert_block, x, 64)?;
    if width < 64 {
        let down = constant(ctx, (64 - width) as u128)?;
        x = emit_op(
            mir_dialect::ops::ShrOp::new(ctx, x, down).get_operation(),
            ctx,
            insert_block,
        );
    }
    Ok(x)
}

pub(super) fn import_upstream_instance<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    module_body: Ptr<BasicBlock>,
    instance: Instance<'tcx>,
) -> Result<(), String> {
    let mut legaliser = Legaliser::default();
    let symbol = legaliser.legalise(tcx.symbol_name(instance).name);
    if symbol_exists(ctx, module_body, symbol.as_ref()) {
        return Ok(());
    }
    let body = tcx.instance_mir(instance.def);
    import_function(
        tcx,
        ctx,
        module_body,
        symbol.clone(),
        body,
        false,
        Some(instance),
    )?;
    // Shared-generic instances also exist in prebuilt std rlibs; keep our
    // copy internal so the object does not export colliding symbols.
    set_internal_linkage(ctx, module_body, &symbol);
    Ok(())
}

pub(super) fn set_internal_linkage(
    ctx: &mut Context,
    module_body: Ptr<BasicBlock>,
    symbol: &crate::identifier::Identifier,
) {
    if std::env::var_os("CRABBIT_DEBUG_EXPORT_ALL").is_some() {
        return;
    }
    set_function_linkage(ctx, module_body, symbol, LinkageAttr::InternalLinkage);
}

pub(super) fn set_function_linkage(
    ctx: &mut Context,
    module_body: Ptr<BasicBlock>,
    symbol: &crate::identifier::Identifier,
    linkage: LinkageAttr,
) {
    let func = module_body.deref(ctx).iter(ctx).find(|op| {
        let op_obj = Operation::get_op_dyn(*op, ctx);
        op_cast::<dyn SymbolOpInterface>(&*op_obj)
            .is_some_and(|symbol_op| symbol_op.get_symbol_name(ctx) == *symbol)
    });
    if let Some(func) = func {
        if let Some(llvm_func) = Operation::get_op::<llvm::ops::FuncOp>(func, ctx) {
            llvm_func.set_attr_llvm_function_linkage(ctx, linkage);
        } else {
            // A `mir.func`: mir-lower's propagate_linkage_attr carries this
            // key onto the lowered `llvm.func`.
            func.deref_mut(ctx)
                .attributes
                .set(ox::func_linkage_key(), linkage);
        }
    }
}

pub(super) fn symbol_exists(ctx: &Context, module_body: Ptr<BasicBlock>, symbol: &str) -> bool {
    module_body.deref(ctx).iter(ctx).any(|op| {
        let op_obj = Operation::get_op_dyn(op, ctx);
        op_cast::<dyn SymbolOpInterface>(&*op_obj)
            .is_some_and(|symbol_op| symbol_op.get_symbol_name(ctx).as_ref() == symbol)
    })
}

pub(super) fn is_upstream_call<'tcx>(
    tcx: TyCtxt<'tcx>,
    state: &FunctionImportState<'tcx>,
    body: &Body<'tcx>,
    func: &Operand<'tcx>,
) -> bool {
    let Some((def_id, _)) = call_fn_def(tcx, state, body, func) else {
        return false;
    };
    !def_id.is_local()
}
