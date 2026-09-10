use super::*;

/// Whether a value's static type is a 128-bit integer.
pub(super) fn is_128_bit_integer_value(ctx: &Context, value: Value) -> bool {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|ty| ty.width() == 128)
}

/// 128-bit division and remainder lower to the compiler-builtins libcalls
/// (`__udivti3` family), the way every backend handles them: there is no
/// wider type to widen into, and the Rust sysroot's compiler-builtins
/// already provides the symbols.
pub(super) fn lower_i128_divrem(
    ctx: &mut Context,
    module_body: Ptr<BasicBlock>,
    insert_block: Ptr<BasicBlock>,
    op: BinOp,
    lhs: Value,
    rhs: Value,
) -> Result<Value, String> {
    let signed = !is_unsigned_integer_value(ctx, lhs);
    let callee: crate::identifier::Identifier = match (op, signed) {
        (BinOp::Div, true) => "__divti3",
        (BinOp::Div, false) => "__udivti3",
        (BinOp::Rem, true) => "__modti3",
        (BinOp::Rem, false) => "__umodti3",
        _ => return Err(format!("lower_i128_divrem on non-div/rem binop {op:?}")),
    }
    .try_into()
    .unwrap();
    let int_ty = lhs.get_type(ctx);
    declare_external_function(
        ctx,
        module_body,
        callee.clone(),
        vec![int_ty, int_ty],
        Some(int_ty),
    );
    let call = mir_dialect::ops::CallOp::new_direct(ctx, callee, vec![lhs, rhs], Some(int_ty));
    call.get_operation().insert_at_back(insert_block, ctx);
    Ok(call.get_result(ctx))
}

/// 128-bit int -> float casts lower to the compiler-builtins libcalls
/// (`__floattidf` family): they round correctly and there is no wider
/// integer to decompose through.
pub(super) fn lower_i128_to_float(
    ctx: &mut Context,
    module_body: Ptr<BasicBlock>,
    insert_block: Ptr<BasicBlock>,
    input: Value,
    float_ty: TypeHandle,
) -> Result<Value, String> {
    let signed = !is_unsigned_integer_value(ctx, input);
    let to_f32 = float_ty.deref(ctx).downcast_ref::<FP32Type>().is_some();
    if !to_f32 && float_ty.deref(ctx).downcast_ref::<FP64Type>().is_none() {
        return Err("unsupported float type for 128-bit int-to-float cast".to_string());
    }
    let callee: crate::identifier::Identifier = match (signed, to_f32) {
        (true, true) => "__floattisf",
        (true, false) => "__floattidf",
        (false, true) => "__floatuntisf",
        (false, false) => "__floatuntidf",
    }
    .try_into()
    .unwrap();
    let int_ty = input.get_type(ctx);
    declare_external_function(ctx, module_body, callee.clone(), vec![int_ty], Some(float_ty));
    let call = mir_dialect::ops::CallOp::new_direct(ctx, callee, vec![input], Some(float_ty));
    call.get_operation().insert_at_back(insert_block, ctx);
    Ok(call.get_result(ctx))
}

/// Saturating float -> 128-bit int casts (Rust `as` semantics): the
/// compiler-builtins `__fixdfti` family does the in-range conversion, and an
/// explicit branch-free clamp enforces the saturation contract — MAX above
/// the range, MIN (0 for unsigned) below it, 0 for NaN — instead of relying
/// on the libcall's own out-of-range behavior.
pub(super) fn lower_float_to_i128_sat(
    ctx: &mut Context,
    module_body: Ptr<BasicBlock>,
    insert_block: Ptr<BasicBlock>,
    input: Value,
    dest_ty: TypeHandle,
) -> Result<Value, String> {
    let float_ty = input.get_type(ctx);
    let from_f32 = float_ty.deref(ctx).downcast_ref::<FP32Type>().is_some();
    if !from_f32 && float_ty.deref(ctx).downcast_ref::<FP64Type>().is_none() {
        return Err("unsupported float type for 128-bit float-to-int cast".to_string());
    }
    let signed = dest_ty
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|ty| ty.signedness() == Signedness::Signed);
    let callee: crate::identifier::Identifier = match (signed, from_f32) {
        (true, true) => "__fixsfti",
        (true, false) => "__fixdfti",
        (false, true) => "__fixunssfti",
        (false, false) => "__fixunsdfti",
    }
    .try_into()
    .unwrap();
    declare_external_function(
        ctx,
        module_body,
        callee.clone(),
        vec![float_ty],
        Some(dest_ty),
    );
    let call = mir_dialect::ops::CallOp::new_direct(ctx, callee, vec![input], Some(dest_ty));
    call.get_operation().insert_at_back(insert_block, ctx);
    let raw = call.get_result(ctx);

    // A float constant in the input's own width. `2^127` and `-1.0` are
    // exact in both f32 and f64; `2^128` is exact in f64 and rounds to +inf
    // in f32, which compares exactly as needed (only +inf saturates high).
    let float_const = |ctx: &mut Context, value: f64| -> Result<Value, String> {
        let bits = if from_f32 {
            (value as f32).to_bits() as u128
        } else {
            value.to_bits() as u128
        };
        let op = constant_from_bits(ctx, float_ty, bits)?;
        op.get_operation().insert_at_back(insert_block, ctx);
        Ok(op.get_result(ctx))
    };
    let int_const = |ctx: &mut Context, bits: u128| -> Result<Value, String> {
        let op = integer_constant(ctx, dest_ty, bits)?;
        op.get_operation().insert_at_back(insert_block, ctx);
        Ok(op.get_result(ctx))
    };
    let zero = int_const(ctx, 0)?;
    let ones = int_const(ctx, u128::MAX)?;
    // All-ones when the (ordered) float compare holds, zero otherwise —
    // every ordered predicate is false for NaN, which is what routes NaN to
    // the final zero mask below.
    let mask_of = |ctx: &mut Context,
                   insert_block: Ptr<BasicBlock>,
                   cond: Value|
     -> Result<Value, String> {
        let wide = cast_value_to_type(ctx, insert_block, cond, dest_ty);
        Ok(emit_op(
            mir_dialect::ops::SubOp::new(ctx, zero, wide).get_operation(),
            ctx,
            insert_block,
        ))
    };

    let two_pow_127 = (1u128 << 127) as f64;
    let (upper_bound, high_sat_bits, mask_low_sat, mask_keep) = if signed {
        // Overflow high at x >= 2^127; low at x < -2^127 (exactly -2^127 is
        // i128::MIN, which the libcall already produces).
        let low_bound = float_const(ctx, -two_pow_127)?;
        let below = emit_op(
            mir_dialect::ops::LtOp::new(ctx, input, low_bound).get_operation(),
            ctx,
            insert_block,
        );
        let mask_low = mask_of(ctx, insert_block, below)?;
        // Ordered self-equality: false exactly for NaN.
        let ordered = emit_op(
            mir_dialect::ops::EqOp::new(ctx, input, input).get_operation(),
            ctx,
            insert_block,
        );
        let mask_ordered = mask_of(ctx, insert_block, ordered)?;
        (two_pow_127, u128::MAX >> 1, mask_low, mask_ordered)
    } else {
        // Overflow high at x >= 2^128; everything in (-1, 2^128) converts,
        // and NaN or x <= -1 goes to zero (`x > -1.0` is false for both).
        let minus_one = float_const(ctx, -1.0)?;
        let in_low_range = emit_op(
            mir_dialect::ops::GtOp::new(ctx, input, minus_one).get_operation(),
            ctx,
            insert_block,
        );
        let mask_keep = mask_of(ctx, insert_block, in_low_range)?;
        (two_pow_127 * 2.0, u128::MAX, zero, mask_keep)
    };

    let upper = float_const(ctx, upper_bound)?;
    let above = emit_op(
        mir_dialect::ops::GeOp::new(ctx, input, upper).get_operation(),
        ctx,
        insert_block,
    );
    let mask_high = mask_of(ctx, insert_block, above)?;
    let not_high = emit_op(
        mir_dialect::ops::BitXorOp::new(ctx, mask_high, ones).get_operation(),
        ctx,
        insert_block,
    );
    let high_sat = int_const(ctx, high_sat_bits)?;
    let high_sel = emit_op(
        mir_dialect::ops::BitAndOp::new(ctx, high_sat, mask_high).get_operation(),
        ctx,
        insert_block,
    );

    let mut result = emit_op(
        mir_dialect::ops::BitAndOp::new(ctx, raw, not_high).get_operation(),
        ctx,
        insert_block,
    );
    if signed {
        let not_low = emit_op(
            mir_dialect::ops::BitXorOp::new(ctx, mask_low_sat, ones).get_operation(),
            ctx,
            insert_block,
        );
        result = emit_op(
            mir_dialect::ops::BitAndOp::new(ctx, result, not_low).get_operation(),
            ctx,
            insert_block,
        );
        let min_sat = int_const(ctx, 1u128 << 127)?;
        let low_sel = emit_op(
            mir_dialect::ops::BitAndOp::new(ctx, min_sat, mask_low_sat).get_operation(),
            ctx,
            insert_block,
        );
        result = emit_op(
            mir_dialect::ops::BitOrOp::new(ctx, result, low_sel).get_operation(),
            ctx,
            insert_block,
        );
        result = emit_op(
            mir_dialect::ops::BitOrOp::new(ctx, result, high_sel).get_operation(),
            ctx,
            insert_block,
        );
        // NaN: every compare above was false, so force the result to zero.
        result = emit_op(
            mir_dialect::ops::BitAndOp::new(ctx, result, mask_keep).get_operation(),
            ctx,
            insert_block,
        );
    } else {
        result = emit_op(
            mir_dialect::ops::BitOrOp::new(ctx, result, high_sel).get_operation(),
            ctx,
            insert_block,
        );
        result = emit_op(
            mir_dialect::ops::BitAndOp::new(ctx, result, mask_keep).get_operation(),
            ctx,
            insert_block,
        );
    }
    Ok(result)
}

/// Casts between 128-bit integers and floats, which bypass the generic
/// `CastOp` path (no 128-bit fcvt exists; see [lower_i128_to_float] and
/// [lower_float_to_i128_sat]). Returns `None` for every other cast.
// Threads the importer's per-function lowering state; a parameter struct
// would be packed and unpacked at every call site for no clarity gain.
#[allow(clippy::too_many_arguments)]
pub(super) fn lower_128_bit_float_cast<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    operand: &Operand<'tcx>,
    src_ty: Ty<'tcx>,
    dst_ty: Ty<'tcx>,
) -> Result<Option<Value>, String> {
    use rustc_middle::ty::TyKind;
    let is_int128 = |ty: Ty<'tcx>| {
        matches!(
            runtime_ty(ty).kind(),
            TyKind::Int(rustc_middle::ty::IntTy::I128)
                | TyKind::Uint(rustc_middle::ty::UintTy::U128)
        )
    };
    let is_float = |ty: Ty<'tcx>| matches!(runtime_ty(ty).kind(), TyKind::Float(_));
    if is_int128(src_ty) && is_float(dst_ty) {
        let input = import_operand(tcx, ctx, state, insert_block, body, operand)?;
        let float_ty = convert_immediate_ty(tcx, ctx, dst_ty)?;
        return Ok(Some(lower_i128_to_float(
            ctx,
            state.module_body,
            insert_block,
            input,
            float_ty,
        )?));
    }
    if is_float(src_ty) && is_int128(dst_ty) {
        let input = import_operand(tcx, ctx, state, insert_block, body, operand)?;
        let dest_ty = convert_immediate_ty(tcx, ctx, dst_ty)?;
        return Ok(Some(lower_float_to_i128_sat(
            ctx,
            state.module_body,
            insert_block,
            input,
            dest_ty,
        )?));
    }
    Ok(None)
}

/// Signed saturating add/sub, branch-free at any width (128 included): the
/// wrapping result, with overflow detected by the sign-bit rule from
/// [lower_signed_overflow_binary] turned into an all-ones mask via an
/// arithmetic shift, selecting `MAX ^ (lhs >>s (w-1))` — MAX for a
/// non-negative lhs, MIN for a negative one — exactly when overflow occurs.
pub(super) fn lower_signed_saturating(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    is_add: bool,
    lhs: Value,
    rhs: Value,
) -> Result<Value, String> {
    let int_ty = lhs.get_type(ctx);
    let width = int_ty
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .map(|ty| ty.width())
        .ok_or_else(|| "saturating intrinsic on non-integer type".to_string())?;
    let wrapped = emit_op(
        if is_add {
            mir_dialect::ops::AddOp::new(ctx, lhs, rhs).get_operation()
        } else {
            mir_dialect::ops::SubOp::new(ctx, lhs, rhs).get_operation()
        },
        ctx,
        insert_block,
    );
    // Same overflow rule as lower_signed_overflow_binary: the sign bit of
    // `(res ^ lhs) & (res ^ rhs)` (add) / `(lhs ^ rhs) & (lhs ^ res)` (sub).
    let (xor_a, xor_b) = if is_add {
        ((wrapped, lhs), (wrapped, rhs))
    } else {
        ((lhs, rhs), (lhs, wrapped))
    };
    let a = emit_op(
        mir_dialect::ops::BitXorOp::new(ctx, xor_a.0, xor_a.1).get_operation(),
        ctx,
        insert_block,
    );
    let b = emit_op(
        mir_dialect::ops::BitXorOp::new(ctx, xor_b.0, xor_b.1).get_operation(),
        ctx,
        insert_block,
    );
    let sign = emit_op(
        mir_dialect::ops::BitAndOp::new(ctx, a, b).get_operation(),
        ctx,
        insert_block,
    );
    let top_bit = integer_constant(ctx, int_ty, (width - 1) as u128)?;
    top_bit.get_operation().insert_at_back(insert_block, ctx);
    let top_bit = top_bit.get_result(ctx);
    // `int_ty` is Signed, so the sign-aware shift is arithmetic: all-ones
    // when the sign bit is set, zero otherwise.
    let overflow_mask = emit_op(
        mir_dialect::ops::SignAwareShrOp::new(ctx, sign, top_bit).get_operation(),
        ctx,
        insert_block,
    );
    let lhs_sign = emit_op(
        mir_dialect::ops::SignAwareShrOp::new(ctx, lhs, top_bit).get_operation(),
        ctx,
        insert_block,
    );
    let max = integer_constant(ctx, int_ty, u128::MAX >> (129 - width as usize))?;
    max.get_operation().insert_at_back(insert_block, ctx);
    let saturated = emit_op(
        mir_dialect::ops::BitXorOp::new(ctx, lhs_sign, max.get_result(ctx)).get_operation(),
        ctx,
        insert_block,
    );
    let ones = integer_constant(ctx, int_ty, u128::MAX >> (128 - width as usize))?;
    ones.get_operation().insert_at_back(insert_block, ctx);
    let keep_mask = emit_op(
        mir_dialect::ops::BitXorOp::new(ctx, overflow_mask, ones.get_result(ctx)).get_operation(),
        ctx,
        insert_block,
    );
    let kept = emit_op(
        mir_dialect::ops::BitAndOp::new(ctx, wrapped, keep_mask).get_operation(),
        ctx,
        insert_block,
    );
    let sat_sel = emit_op(
        mir_dialect::ops::BitAndOp::new(ctx, saturated, overflow_mask).get_operation(),
        ctx,
        insert_block,
    );
    Ok(emit_op(
        mir_dialect::ops::BitOrOp::new(ctx, kept, sat_sel).get_operation(),
        ctx,
        insert_block,
    ))
}

pub(super) fn lower_overflow_binary(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    op: BinOp,
    lhs: Value,
    rhs: Value,
) -> Result<Value, String> {
    if is_unsigned_integer_value(ctx, lhs) {
        lower_unsigned_overflow_binary(ctx, insert_block, op, lhs, rhs)
    } else {
        lower_signed_overflow_binary(ctx, insert_block, op, lhs, rhs)
    }
}

pub(super) fn lower_unsigned_overflow_binary(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    op: BinOp,
    lhs: Value,
    rhs: Value,
) -> Result<Value, String> {
    if matches!(op, BinOp::MulWithOverflow) {
        return lower_unsigned_mul_overflow(ctx, insert_block, lhs, rhs);
    }

    let wrapped = match op {
        BinOp::AddWithOverflow => mir_dialect::ops::AddOp::new(ctx, lhs, rhs).get_operation(),
        BinOp::SubWithOverflow => mir_dialect::ops::SubOp::new(ctx, lhs, rhs).get_operation(),
        other => return Err(format!("unsupported MIR overflow binary op: {other:?}")),
    };
    wrapped.insert_at_back(insert_block, ctx);
    let wrapped_value = wrapped.deref(ctx).get_result(0);

    let overflow = match op {
        BinOp::AddWithOverflow => {
            mir_dialect::ops::LtOp::new(ctx, wrapped_value, lhs).get_operation()
        }
        BinOp::SubWithOverflow => mir_dialect::ops::LtOp::new(ctx, lhs, rhs).get_operation(),
        _ => unreachable!(),
    };
    overflow.insert_at_back(insert_block, ctx);
    let overflow_value = overflow.deref(ctx).get_result(0);
    Ok(pack_overflow_result(
        ctx,
        insert_block,
        wrapped_value,
        overflow_value,
    ))
}

pub(super) fn lower_signed_overflow_binary(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    op: BinOp,
    lhs: Value,
    rhs: Value,
) -> Result<Value, String> {
    let ty = lhs.get_type(ctx);
    if !ty
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|int_ty| int_ty.signedness() == Signedness::Signed)
    {
        return Err(format!(
            "unsupported non-integer MIR overflow binary op: {op:?}"
        ));
    }

    if matches!(op, BinOp::MulWithOverflow) {
        return lower_signed_mul_overflow(ctx, insert_block, lhs, rhs);
    }

    let wrapped = match op {
        BinOp::AddWithOverflow => mir_dialect::ops::AddOp::new(ctx, lhs, rhs).get_operation(),
        BinOp::SubWithOverflow => mir_dialect::ops::SubOp::new(ctx, lhs, rhs).get_operation(),
        other => return Err(format!("unsupported MIR overflow binary op: {other:?}")),
    };
    wrapped.insert_at_back(insert_block, ctx);
    let wrapped_value = wrapped.deref(ctx).get_result(0);

    // Add overflows iff the operands share a sign and the result's differs:
    // the sign bit of `(res ^ lhs) & (res ^ rhs)`. Sub overflows iff the
    // operands' signs differ and the result's sign differs from lhs's: the
    // sign bit of `(lhs ^ rhs) & (lhs ^ res)`.
    let (xor_a, xor_b) = match op {
        BinOp::AddWithOverflow => ((wrapped_value, lhs), (wrapped_value, rhs)),
        BinOp::SubWithOverflow => ((lhs, rhs), (lhs, wrapped_value)),
        _ => unreachable!(),
    };
    let a = mir_dialect::ops::BitXorOp::new(ctx, xor_a.0, xor_a.1).get_operation();
    a.insert_at_back(insert_block, ctx);
    let a = a.deref(ctx).get_result(0);
    let b = mir_dialect::ops::BitXorOp::new(ctx, xor_b.0, xor_b.1).get_operation();
    b.insert_at_back(insert_block, ctx);
    let b = b.deref(ctx).get_result(0);
    let sign = mir_dialect::ops::BitAndOp::new(ctx, a, b).get_operation();
    sign.insert_at_back(insert_block, ctx);
    let sign = sign.deref(ctx).get_result(0);

    let zero = integer_constant(ctx, ty, 0)?;
    zero.get_operation().insert_at_back(insert_block, ctx);
    // A signed lt: `ty` is a Signed integer type, so dialect-mir picks slt.
    let overflow = mir_dialect::ops::LtOp::new(ctx, sign, zero.get_result(ctx)).get_operation();
    overflow.insert_at_back(insert_block, ctx);
    let overflow_value = overflow.deref(ctx).get_result(0);
    Ok(pack_overflow_result(
        ctx,
        insert_block,
        wrapped_value,
        overflow_value,
    ))
}

pub(super) fn lower_signed_mul_overflow(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    lhs: Value,
    rhs: Value,
) -> Result<Value, String> {
    let ty = lhs.get_type(ctx);
    let width = ty
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .map(|int_ty| int_ty.width())
        .ok_or_else(|| "MIR mul-with-overflow on non-integer type".to_string())?;
    if width == 128 {
        return lower_signed_mul_overflow_128(ctx, insert_block, lhs, rhs);
    }
    if width > 64 {
        return Err(format!(
            "unsupported {width}-bit MIR mul-with-overflow"
        ));
    }

    // Multiply at double width (sext: the operands are Signed), then check
    // that the product survives a round-trip through the narrow type.
    let wide_ty: TypeHandle = IntegerType::get(ctx, width * 2, Signedness::Signed).into();
    let wide_lhs = cast_value_to_type(ctx, insert_block, lhs, wide_ty);
    let wide_rhs = cast_value_to_type(ctx, insert_block, rhs, wide_ty);
    let wide_mul = mir_dialect::ops::MulOp::new(ctx, wide_lhs, wide_rhs);
    wide_mul.get_operation().insert_at_back(insert_block, ctx);
    let wide_value = wide_mul.get_operation().deref(ctx).get_result(0);
    let wrapped_value = cast_value_to_type(ctx, insert_block, wide_value, ty);
    let widened_back = cast_value_to_type(ctx, insert_block, wrapped_value, wide_ty);
    let overflow = mir_dialect::ops::NeOp::new(ctx, wide_value, widened_back);
    overflow.get_operation().insert_at_back(insert_block, ctx);
    let overflow_value = overflow.get_result(ctx);
    Ok(pack_overflow_result(
        ctx,
        insert_block,
        wrapped_value,
        overflow_value,
    ))
}

/// 128-bit signed mul-with-overflow. The wrapped product's bits are the same
/// as the unsigned wrapping product; the exact 256-bit product's high half is
/// recovered from the unsigned high half with the standard sign fixup
/// `smulh(a, b) = umulh(a, b) - (a < 0 ? b : 0) - (b < 0 ? a : 0)` (all mod
/// 2^128), and overflow holds iff that high half differs from the sign
/// extension of the wrapped low half.
pub(super) fn lower_signed_mul_overflow_128(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    lhs: Value,
    rhs: Value,
) -> Result<Value, String> {
    let u128_ty: TypeHandle = IntegerType::get(ctx, 128, Signedness::Unsigned).into();
    let lhs_bits = cast_value_to_type(ctx, insert_block, lhs, u128_ty);
    let rhs_bits = cast_value_to_type(ctx, insert_block, rhs, u128_ty);

    let wrapped = emit_op(
        mir_dialect::ops::MulOp::new(ctx, lhs, rhs).get_operation(),
        ctx,
        insert_block,
    );

    let unsigned_high = emit_umulh128(ctx, insert_block, lhs_bits, rhs_bits)?;
    // (a < 0 ? b : 0) as `sign_mask(a) & b`, branch-free.
    let lhs_mask = emit_sign_mask128(ctx, insert_block, lhs_bits)?;
    let rhs_mask = emit_sign_mask128(ctx, insert_block, rhs_bits)?;
    let lhs_fix = emit_op(
        mir_dialect::ops::BitAndOp::new(ctx, lhs_mask, rhs_bits).get_operation(),
        ctx,
        insert_block,
    );
    let rhs_fix = emit_op(
        mir_dialect::ops::BitAndOp::new(ctx, rhs_mask, lhs_bits).get_operation(),
        ctx,
        insert_block,
    );
    let signed_high = emit_op(
        mir_dialect::ops::SubOp::new(ctx, unsigned_high, lhs_fix).get_operation(),
        ctx,
        insert_block,
    );
    let signed_high = emit_op(
        mir_dialect::ops::SubOp::new(ctx, signed_high, rhs_fix).get_operation(),
        ctx,
        insert_block,
    );

    // Sign extension of the wrapped low half: all-ones iff it is negative.
    let wrapped_bits = cast_value_to_type(ctx, insert_block, wrapped, u128_ty);
    let expected_high = emit_sign_mask128(ctx, insert_block, wrapped_bits)?;
    let overflow = emit_op(
        mir_dialect::ops::NeOp::new(ctx, signed_high, expected_high).get_operation(),
        ctx,
        insert_block,
    );
    Ok(pack_overflow_result(ctx, insert_block, wrapped, overflow))
}

/// Package a wrapped result and its overflow flag into the `(value, i8)`
/// struct the importer uses for the MIR `*WithOverflow` result tuple.
pub(super) fn pack_overflow_result(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    wrapped_value: Value,
    overflow_value: Value,
) -> Value {
    let overflow_ty = bool_storage_ty(ctx);
    let overflow_value = cast_value_to_type(ctx, insert_block, overflow_value, overflow_ty);

    let result_ty =
        llvm::types::StructType::get_unnamed(ctx, vec![wrapped_value.get_type(ctx), overflow_ty])
            .into();
    let undef = mir_dialect::ops::UndefOp::new(ctx, result_ty);
    undef.get_operation().insert_at_back(insert_block, ctx);
    let with_value =
        mir_dialect::ops::InsertValueOp::new(ctx, wrapped_value, undef.get_result(ctx), vec![0]);
    with_value.get_operation().insert_at_back(insert_block, ctx);
    let with_overflow = mir_dialect::ops::InsertValueOp::new(
        ctx,
        overflow_value,
        with_value.get_result(ctx),
        vec![1],
    );
    with_overflow
        .get_operation()
        .insert_at_back(insert_block, ctx);
    with_overflow.get_result(ctx)
}

/// Insert a freshly built operation and return its first result value.
pub(super) fn emit_op(op: Ptr<Operation>, ctx: &Context, insert_block: Ptr<BasicBlock>) -> Value {
    op.insert_at_back(insert_block, ctx);
    op.deref(ctx).get_result(0)
}

/// The high 128 bits of the exact 256-bit unsigned product `lhs * rhs` (both
/// u128), via 64-bit half decomposition. With `a = a_hi·2^64 + a_lo` (halves
/// < 2^64) the partial products `a_lo·b_lo`, `a_hi·b_lo`, `a_lo·b_hi`, and
/// `a_hi·b_hi` are all exact in u128, as is the carry accumulation below, so
/// no wider type is needed. This is the standard compiler-builtins-style
/// decomposition for 128-bit mul-with-overflow, where no wider type exists
/// to widen into.
pub(super) fn emit_umulh128(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    lhs: Value,
    rhs: Value,
) -> Result<Value, String> {
    let u128_ty: TypeHandle = IntegerType::get(ctx, 128, Signedness::Unsigned).into();
    let constant = |ctx: &mut Context, bits: u128| -> Result<Value, String> {
        let op = integer_constant(ctx, u128_ty, bits)?;
        op.get_operation().insert_at_back(insert_block, ctx);
        Ok(op.get_result(ctx))
    };
    let mask = constant(ctx, u64::MAX as u128)?;
    let sixty_four = constant(ctx, 64)?;

    let a_lo = emit_op(
        mir_dialect::ops::BitAndOp::new(ctx, lhs, mask).get_operation(),
        ctx,
        insert_block,
    );
    let a_hi = emit_op(
        mir_dialect::ops::ShrOp::new(ctx, lhs, sixty_four).get_operation(),
        ctx,
        insert_block,
    );
    let b_lo = emit_op(
        mir_dialect::ops::BitAndOp::new(ctx, rhs, mask).get_operation(),
        ctx,
        insert_block,
    );
    let b_hi = emit_op(
        mir_dialect::ops::ShrOp::new(ctx, rhs, sixty_four).get_operation(),
        ctx,
        insert_block,
    );

    let lo_lo = emit_op(
        mir_dialect::ops::MulOp::new(ctx, a_lo, b_lo).get_operation(),
        ctx,
        insert_block,
    );
    let cross_a = emit_op(
        mir_dialect::ops::MulOp::new(ctx, a_hi, b_lo).get_operation(),
        ctx,
        insert_block,
    );
    let cross_b = emit_op(
        mir_dialect::ops::MulOp::new(ctx, a_lo, b_hi).get_operation(),
        ctx,
        insert_block,
    );
    let hi_hi = emit_op(
        mir_dialect::ops::MulOp::new(ctx, a_hi, b_hi).get_operation(),
        ctx,
        insert_block,
    );

    // carry = ((cross_a & M) + (cross_b & M) + (lo_lo >> 64)) >> 64: the
    // carry out of bit 127 of the full product (sum of three values < 2^64,
    // exact in u128).
    let cross_a_lo = emit_op(
        mir_dialect::ops::BitAndOp::new(ctx, cross_a, mask).get_operation(),
        ctx,
        insert_block,
    );
    let cross_b_lo = emit_op(
        mir_dialect::ops::BitAndOp::new(ctx, cross_b, mask).get_operation(),
        ctx,
        insert_block,
    );
    let lo_lo_hi = emit_op(
        mir_dialect::ops::ShrOp::new(ctx, lo_lo, sixty_four).get_operation(),
        ctx,
        insert_block,
    );
    let mid_sum = emit_op(
        mir_dialect::ops::AddOp::new(ctx, cross_a_lo, cross_b_lo).get_operation(),
        ctx,
        insert_block,
    );
    let mid_sum = emit_op(
        mir_dialect::ops::AddOp::new(ctx, mid_sum, lo_lo_hi).get_operation(),
        ctx,
        insert_block,
    );
    let carry = emit_op(
        mir_dialect::ops::ShrOp::new(ctx, mid_sum, sixty_four).get_operation(),
        ctx,
        insert_block,
    );

    // high = hi_hi + (cross_a >> 64) + (cross_b >> 64) + carry; the true
    // high half is < 2^128, and no intermediate sum wraps.
    let cross_a_hi = emit_op(
        mir_dialect::ops::ShrOp::new(ctx, cross_a, sixty_four).get_operation(),
        ctx,
        insert_block,
    );
    let cross_b_hi = emit_op(
        mir_dialect::ops::ShrOp::new(ctx, cross_b, sixty_four).get_operation(),
        ctx,
        insert_block,
    );
    let high = emit_op(
        mir_dialect::ops::AddOp::new(ctx, hi_hi, cross_a_hi).get_operation(),
        ctx,
        insert_block,
    );
    let high = emit_op(
        mir_dialect::ops::AddOp::new(ctx, high, cross_b_hi).get_operation(),
        ctx,
        insert_block,
    );
    let high = emit_op(
        mir_dialect::ops::AddOp::new(ctx, high, carry).get_operation(),
        ctx,
        insert_block,
    );
    Ok(high)
}

/// `0 - (value >> 127)` on u128: all-ones when the top bit of `value` is
/// set, zero otherwise.
pub(super) fn emit_sign_mask128(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    value: Value,
) -> Result<Value, String> {
    let u128_ty: TypeHandle = IntegerType::get(ctx, 128, Signedness::Unsigned).into();
    let shift = integer_constant(ctx, u128_ty, 127)?;
    shift.get_operation().insert_at_back(insert_block, ctx);
    let sign_bit = emit_op(
        mir_dialect::ops::ShrOp::new(ctx, value, shift.get_result(ctx)).get_operation(),
        ctx,
        insert_block,
    );
    let zero = integer_constant(ctx, u128_ty, 0)?;
    zero.get_operation().insert_at_back(insert_block, ctx);
    Ok(emit_op(
        mir_dialect::ops::SubOp::new(ctx, zero.get_result(ctx), sign_bit).get_operation(),
        ctx,
        insert_block,
    ))
}

pub(super) fn lower_unsigned_mul_overflow(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    lhs: Value,
    rhs: Value,
) -> Result<Value, String> {
    let ty = lhs.get_type(ctx);
    let width = ty
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .map(|int_ty| int_ty.width())
        .ok_or_else(|| "MIR mul-with-overflow on non-integer type".to_string())?;
    if width == 128 {
        // No wider type exists: the product wraps in-register and overflow
        // is exactly "the true high 128 bits are non-zero".
        let wrapped = emit_op(
            mir_dialect::ops::MulOp::new(ctx, lhs, rhs).get_operation(),
            ctx,
            insert_block,
        );
        let high = emit_umulh128(ctx, insert_block, lhs, rhs)?;
        let u128_ty: TypeHandle = IntegerType::get(ctx, 128, Signedness::Unsigned).into();
        let zero = integer_constant(ctx, u128_ty, 0)?;
        zero.get_operation().insert_at_back(insert_block, ctx);
        let overflow = emit_op(
            mir_dialect::ops::NeOp::new(ctx, high, zero.get_result(ctx)).get_operation(),
            ctx,
            insert_block,
        );
        return Ok(pack_overflow_result(ctx, insert_block, wrapped, overflow));
    }
    if width > 64 {
        return Err(format!(
            "unsupported {width}-bit MIR mul-with-overflow"
        ));
    }

    let wide_ty: TypeHandle = IntegerType::get(ctx, width * 2, Signedness::Unsigned).into();
    let wide_lhs = cast_value_to_type(ctx, insert_block, lhs, wide_ty);
    let wide_rhs = cast_value_to_type(ctx, insert_block, rhs, wide_ty);
    let wide_mul = mir_dialect::ops::MulOp::new(ctx, wide_lhs, wide_rhs);
    wide_mul.get_operation().insert_at_back(insert_block, ctx);
    let wide_value = wide_mul.get_operation().deref(ctx).get_result(0);
    let wrapped_value = cast_value_to_type(ctx, insert_block, wide_value, ty);

    let narrow_max = integer_constant(ctx, wide_ty, u128::MAX >> (128 - width as usize))?;
    narrow_max.get_operation().insert_at_back(insert_block, ctx);
    let overflow = mir_dialect::ops::GtOp::new(ctx, wide_value, narrow_max.get_result(ctx));
    overflow.get_operation().insert_at_back(insert_block, ctx);
    let overflow_value = overflow.get_operation().deref(ctx).get_result(0);
    Ok(pack_overflow_result(
        ctx,
        insert_block,
        wrapped_value,
        overflow_value,
    ))
}

/// Lower MIR's three-way compare to `(lhs > rhs) as i8 - (lhs < rhs) as i8`.
/// That byte is exactly `core::cmp::Ordering`'s direct tag (-1/0/1), so it is
/// materialized as the enum blob the usual way: written into a stack
/// temporary and loaded back as the real in-memory bytes.
pub(super) fn lower_three_way_cmp<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    ordering_ty: Ty<'tcx>,
    lhs: Value,
    rhs: Value,
) -> Result<Value, String> {
    // dialect-mir compares resolve signedness from the operand types.
    let gt = mir_dialect::ops::GtOp::new(ctx, lhs, rhs);
    gt.get_operation().insert_at_back(insert_block, ctx);
    let lt = mir_dialect::ops::LtOp::new(ctx, lhs, rhs);
    lt.get_operation().insert_at_back(insert_block, ctx);
    let byte_ty = bool_storage_ty(ctx);
    let gt = cast_value_to_type(ctx, insert_block, gt.get_result(ctx), byte_ty);
    let lt = cast_value_to_type(ctx, insert_block, lt.get_result(ctx), byte_ty);
    let tag = mir_dialect::ops::SubOp::new(ctx, gt, lt).get_operation();
    tag.insert_at_back(insert_block, ctx);
    let tag = tag.deref(ctx).get_result(0);

    let blob_ty = convert_ty(tcx, ctx, ordering_ty)?;
    let slot = mir_dialect::ops::AllocaOp::new(ctx, blob_ty);
    slot.get_operation().insert_at_back(insert_block, ctx);
    let slot = slot.get_result(ctx);
    let store = mir_dialect::ops::StoreOp::new(ctx, tag, slot);
    store.get_operation().insert_at_back(insert_block, ctx);
    let load = mir_dialect::ops::LoadOp::new(ctx, slot, blob_ty);
    load.get_operation().insert_at_back(insert_block, ctx);
    Ok(load.get_result(ctx))
}

pub(super) fn is_unsigned_integer_value(ctx: &Context, value: Value) -> bool {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|ty| ty.signedness() != Signedness::Signed)
}

pub(super) fn bool_immediate_ty(ctx: &mut Context) -> TypeHandle {
    IntegerType::get(ctx, 1, Signedness::Signless).into()
}

pub(super) fn bool_storage_ty(ctx: &mut Context) -> TypeHandle {
    IntegerType::get(ctx, 8, Signedness::Signless).into()
}

pub(super) fn is_bool_ty<'tcx>(ty: Ty<'tcx>) -> bool {
    matches!(runtime_ty(ty).kind(), rustc_middle::ty::TyKind::Bool)
}

pub(super) fn cast_value_to_type(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    value: Value,
    target_ty: TypeHandle,
) -> Value {
    if value.get_type(ctx) == target_ty {
        return value;
    }
    let cast = mir_dialect::ops::CastOp::new(ctx, value, target_ty);
    cast.get_operation().insert_at_back(insert_block, ctx);
    cast.get_result(ctx)
}

pub(super) fn aggregate_field_type(
    ctx: &Context,
    aggregate_ty: TypeHandle,
    index: usize,
) -> Result<TypeHandle, String> {
    let ty_ref = aggregate_ty.deref(ctx);
    if let Some(array_ty) = ty_ref.downcast_ref::<llvm::types::ArrayType>() {
        return Ok(array_ty.elem_type());
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<llvm::types::StructType>() {
        if index >= struct_ty.num_fields() {
            return Err(format!(
                "aggregate field index {index} is out of bounds for {}",
                ty_ref.disp(ctx)
            ));
        }
        return Ok(struct_ty.field_type(index));
    }
    Err(format!("unsupported aggregate type: {}", ty_ref.disp(ctx)))
}

pub(super) fn normalize_bool_for_storage<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    rust_ty: Ty<'tcx>,
    value: Value,
) -> Result<Value, String> {
    let rust_ty = mono_ty(tcx, state, rust_ty);
    if !is_bool_ty(rust_ty) {
        return Ok(value);
    }
    let storage_ty = bool_storage_ty(ctx);
    Ok(cast_value_to_type(ctx, insert_block, value, storage_ty))
}

pub(super) fn normalize_bool_for_immediate<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    rust_ty: Ty<'tcx>,
    value: Value,
) -> Result<Value, String> {
    let rust_ty = mono_ty(tcx, state, rust_ty);
    if !is_bool_ty(rust_ty) {
        return Ok(value);
    }
    let immediate_ty = bool_immediate_ty(ctx);
    Ok(cast_value_to_type(ctx, insert_block, value, immediate_ty))
}

pub(super) fn simple_abi_leaves_for_ty(
    ctx: &Context,
    ty: TypeHandle,
) -> Result<Vec<(Vec<u32>, TypeHandle)>, String> {
    let mut leaves = Vec::new();
    collect_simple_abi_fields(ctx, ty, Vec::new(), &mut leaves)?;
    Ok(leaves)
}

pub(super) fn collect_simple_abi_fields(
    ctx: &Context,
    ty: TypeHandle,
    prefix: Vec<u32>,
    out: &mut Vec<(Vec<u32>, TypeHandle)>,
) -> Result<(), String> {
    if is_simple_abi_scalar_ty(ctx, ty) {
        out.push((prefix, ty));
        return Ok(());
    }
    if ty.deref(ctx).is::<UnitType>() {
        return Ok(());
    }

    let fields = {
        let ty_ref = ty.deref(ctx);
        let Some(struct_ty) = ty_ref.downcast_ref::<llvm::types::StructType>() else {
            return Err(format!(
                "unsupported aggregate argument type for external call ABI: {:?}",
                ty_ref
            ));
        };
        if struct_ty.is_opaque() {
            return Err("opaque struct argument cannot be ABI-lowered".to_string());
        }
        struct_ty.fields().collect::<Vec<_>>()
    };

    for (index, field_ty) in fields.into_iter().enumerate() {
        let mut field_prefix = prefix.clone();
        field_prefix.push(index as u32);
        collect_simple_abi_fields(ctx, field_ty, field_prefix, out)?;
    }
    Ok(())
}

pub(super) fn lower_arguments_from_str_call<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    args: &[rustc_span::Spanned<Operand<'tcx>>],
    destination: &Place<'tcx>,
) -> Result<(), String> {
    if args.len() != 1 {
        return Err(format!(
            "unsupported Arguments::from_str argument count: {}",
            args.len()
        ));
    }

    let str_value = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
    let ptr_ty = llvm_ptr_ty(ctx);
    let fmt_ty = fmt_arguments_ty(ctx);

    let template = mir_dialect::ops::ExtractValueOp::new(ctx, str_value, vec![0], ptr_ty);
    template.get_operation().insert_at_back(insert_block, ctx);

    let usize_ty: TypeHandle = usize_ty(ctx).into();
    let len = mir_dialect::ops::ExtractValueOp::new(ctx, str_value, vec![1], usize_ty);
    len.get_operation().insert_at_back(insert_block, ctx);

    let one = mir_dialect::ops::ConstantOp::new_integer(
        ctx,
        IntegerAttr::new(
            TypedHandle::from_handle(usize_ty, ctx).unwrap(),
            APInt::from_u64(1, NonZero::new(64).unwrap()),
        ),
    );
    one.get_operation().insert_at_back(insert_block, ctx);

    let shifted = mir_dialect::ops::ShlOp::new(ctx, len.get_result(ctx), one.get_result(ctx));
    shifted.get_operation().insert_at_back(insert_block, ctx);
    let encoded = mir_dialect::ops::BitOrOp::new(ctx, shifted.get_result(ctx), one.get_result(ctx));
    encoded.get_operation().insert_at_back(insert_block, ctx);

    let args_ptr = mir_dialect::ops::CastOp::new(ctx, encoded.get_result(ctx), ptr_ty);
    args_ptr.get_operation().insert_at_back(insert_block, ctx);

    let undef = mir_dialect::ops::UndefOp::new(ctx, fmt_ty);
    undef.get_operation().insert_at_back(insert_block, ctx);
    let with_template = mir_dialect::ops::InsertValueOp::new(
        ctx,
        template.get_result(ctx),
        undef.get_result(ctx),
        vec![0],
    );
    with_template
        .get_operation()
        .insert_at_back(insert_block, ctx);
    let args = mir_dialect::ops::InsertValueOp::new(
        ctx,
        args_ptr.get_result(ctx),
        with_template.get_result(ctx),
        vec![1],
    );
    args.get_operation().insert_at_back(insert_block, ctx);

    store_place(
        tcx,
        ctx,
        state,
        insert_block,
        body,
        destination,
        args.get_result(ctx),
    )
}

pub(super) fn call_symbol<'tcx>(
    tcx: TyCtxt<'tcx>,
    state: &FunctionImportState<'tcx>,
    body: &Body<'tcx>,
    func: &Operand<'tcx>,
) -> Result<crate::identifier::Identifier, String> {
    let Some((def_id, args)) = call_fn_def(tcx, state, body, func) else {
        return Err(format!("unsupported MIR call callee: {func:?}"));
    };
    let mut legaliser = Legaliser::default();
    let args = mono_generic_args(tcx, state, args);
    if let Ok(Some(instance)) = Instance::try_resolve(tcx, body.typing_env(tcx), def_id, args) {
        if let Some(symbol) = known_codegen_symbol(tcx, instance) {
            return Ok(legaliser.legalise(symbol));
        }
        return Ok(legaliser.legalise(tcx.symbol_name(instance).name));
    }
    Ok(legaliser.legalise(&tcx.def_path_str(def_id)))
}

pub(super) fn known_codegen_symbol<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> Option<&'static str> {
    let path = tcx.def_path_str(instance.def.def_id());
    if path.contains("compare_bytes") {
        return Some("memcmp");
    }
    None
}
