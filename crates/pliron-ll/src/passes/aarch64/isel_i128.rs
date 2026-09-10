use crate::{
    context::{Context, Ptr},
    dialects::{
        aarch64::{op_interfaces::Aarch64Opcode, ops as aarch64_ops, registers::Register},
        llvm::attributes::ICmpPredicateAttr,
    },
    input_error_noloc,
    result::CrabbitResult,
};

use super::{
    error::Aarch64Err,
    frontend::BinaryKind,
    llvm_to_aarch64_isel::{
        CompareValue, LoweredValue, condition_code, fold_binary, fresh_vreg, is_128_bit_integer,
        materialize, materialize_pair, materialize_typed, materialize_u64_immediate, opcode,
    },
};
use crate::r#type::TypeHandle;

/// Lowers the operations which need a two-register i128 representation.
pub(super) fn lower_binary_128(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    kind: BinaryKind,
    lhs: LoweredValue,
    rhs: LoweredValue,
    result_ty: TypeHandle,
    next_vreg: &mut usize,
) -> CrabbitResult<LoweredValue> {
    if let (LoweredValue::Imm(lhs), LoweredValue::Imm(rhs)) = (&lhs, &rhs)
        && let Some(imm) = fold_binary(ctx, kind, *lhs, *rhs, result_ty)
    {
        return Ok(LoweredValue::Imm(imm));
    }

    let shift_amount = match &rhs {
        LoweredValue::Imm(imm) => Some((*imm).min(127) as u32),
        _ => None,
    };
    let (lhs_lo, lhs_hi) = materialize_pair(ctx, entry, lhs, result_ty, next_vreg, "i128 lhs")?;

    match kind {
        BinaryKind::Add => {
            let (rhs_lo, rhs_hi) =
                materialize_pair(ctx, entry, rhs, result_ty, next_vreg, "i128 rhs")?;
            let lo = fresh_vreg(next_vreg);
            aarch64_ops::binary(
                ctx,
                aarch64_ops::AddOp::OPCODE,
                lo,
                lhs_lo,
                rhs_lo,
            )
            .insert_at_back(entry, ctx);
            aarch64_ops::cmp(ctx, lo, lhs_lo).insert_at_back(entry, ctx);
            let carry = fresh_vreg(next_vreg);
            aarch64_ops::cset(
                ctx,
                carry,
                condition_code(ICmpPredicateAttr::ULT),
            )
            .insert_at_back(entry, ctx);
            let hi_sum = fresh_vreg(next_vreg);
            aarch64_ops::binary(
                ctx,
                aarch64_ops::AddOp::OPCODE,
                hi_sum,
                lhs_hi,
                rhs_hi,
            )
            .insert_at_back(entry, ctx);
            let hi = fresh_vreg(next_vreg);
            aarch64_ops::binary(ctx, aarch64_ops::AddOp::OPCODE, hi, hi_sum, carry)
                .insert_at_back(entry, ctx);
            Ok(LoweredValue::RegPair(lo, hi))
        }
        BinaryKind::Sub => {
            let (rhs_lo, rhs_hi) =
                materialize_pair(ctx, entry, rhs, result_ty, next_vreg, "i128 rhs")?;
            aarch64_ops::cmp(ctx, lhs_lo, rhs_lo).insert_at_back(entry, ctx);
            let borrow = fresh_vreg(next_vreg);
            aarch64_ops::cset(
                ctx,
                borrow,
                condition_code(ICmpPredicateAttr::ULT),
            )
            .insert_at_back(entry, ctx);
            let lo = fresh_vreg(next_vreg);
            aarch64_ops::binary(ctx, aarch64_ops::SubOp::OPCODE, lo, lhs_lo, rhs_lo)
                .insert_at_back(entry, ctx);
            let hi_sub = fresh_vreg(next_vreg);
            aarch64_ops::binary(
                ctx,
                aarch64_ops::SubOp::OPCODE,
                hi_sub,
                lhs_hi,
                rhs_hi,
            )
            .insert_at_back(entry, ctx);
            let hi = fresh_vreg(next_vreg);
            aarch64_ops::binary(ctx, aarch64_ops::SubOp::OPCODE, hi, hi_sub, borrow)
                .insert_at_back(entry, ctx);
            Ok(LoweredValue::RegPair(lo, hi))
        }
        BinaryKind::Mul => {
            let (rhs_lo, rhs_hi) =
                materialize_pair(ctx, entry, rhs, result_ty, next_vreg, "i128 rhs")?;
            let lo = fresh_vreg(next_vreg);
            aarch64_ops::binary(
                ctx,
                aarch64_ops::MulOp::OPCODE,
                lo,
                lhs_lo,
                rhs_lo,
            )
            .insert_at_back(entry, ctx);
            let high_low = fresh_vreg(next_vreg);
            aarch64_ops::binary(
                ctx,
                aarch64_ops::UmulhOp::OPCODE,
                high_low,
                lhs_lo,
                rhs_lo,
            )
            .insert_at_back(entry, ctx);
            let cross_a = fresh_vreg(next_vreg);
            aarch64_ops::binary(
                ctx,
                aarch64_ops::MulOp::OPCODE,
                cross_a,
                lhs_hi,
                rhs_lo,
            )
            .insert_at_back(entry, ctx);
            let cross_b = fresh_vreg(next_vreg);
            aarch64_ops::binary(
                ctx,
                aarch64_ops::MulOp::OPCODE,
                cross_b,
                lhs_lo,
                rhs_hi,
            )
            .insert_at_back(entry, ctx);
            let hi_partial = fresh_vreg(next_vreg);
            aarch64_ops::binary(
                ctx,
                aarch64_ops::AddOp::OPCODE,
                hi_partial,
                high_low,
                cross_a,
            )
            .insert_at_back(entry, ctx);
            let hi = fresh_vreg(next_vreg);
            aarch64_ops::binary(
                ctx,
                aarch64_ops::AddOp::OPCODE,
                hi,
                hi_partial,
                cross_b,
            )
            .insert_at_back(entry, ctx);
            Ok(LoweredValue::RegPair(lo, hi))
        }
        BinaryKind::Shr => match shift_amount {
            Some(shift) => {
                lower_shift_right_128(ctx, entry, lhs_lo, lhs_hi, shift, false, next_vreg)
            }
            None => {
                let amount = materialize(ctx, entry, rhs, next_vreg, "i128 shift amount")?;
                lower_dynamic_shift_128(
                    ctx, entry, DynamicShift::Lshr, lhs_lo, lhs_hi, amount, next_vreg,
                )
            }
        },
        BinaryKind::AShr => match shift_amount {
            Some(shift) => {
                lower_shift_right_128(ctx, entry, lhs_lo, lhs_hi, shift, true, next_vreg)
            }
            None => {
                let amount = materialize(ctx, entry, rhs, next_vreg, "i128 shift amount")?;
                lower_dynamic_shift_128(
                    ctx, entry, DynamicShift::Ashr, lhs_lo, lhs_hi, amount, next_vreg,
                )
            }
        },
        BinaryKind::Shl => match shift_amount {
            Some(shift) => lower_shift_left_128(ctx, entry, lhs_lo, lhs_hi, shift, next_vreg),
            None => {
                let amount = materialize(ctx, entry, rhs, next_vreg, "i128 shift amount")?;
                lower_dynamic_shift_128(
                    ctx, entry, DynamicShift::Shl, lhs_lo, lhs_hi, amount, next_vreg,
                )
            }
        },
        BinaryKind::And | BinaryKind::Or | BinaryKind::Xor => {
            let (rhs_lo, rhs_hi) =
                materialize_pair(ctx, entry, rhs, result_ty, next_vreg, "i128 rhs")?;
            let opcode = opcode(kind);
            let lo = fresh_vreg(next_vreg);
            aarch64_ops::binary(ctx, opcode, lo, lhs_lo, rhs_lo).insert_at_back(entry, ctx);
            let hi = fresh_vreg(next_vreg);
            aarch64_ops::binary(ctx, opcode, hi, lhs_hi, rhs_hi).insert_at_back(entry, ctx);
            Ok(LoweredValue::RegPair(lo, hi))
        }
        // Unreachable from the crabbit importer, which lowers 128-bit
        // div/rem to the `__divti3`-family libcalls before isel; kept as a
        // guard for hand-built LLVM-dialect input.
        BinaryKind::SDiv | BinaryKind::UDiv | BinaryKind::SRem | BinaryKind::URem => {
            Err(input_error_noloc!(Aarch64Err::UnsupportedOp(format!(
                "128-bit {kind:?} (lower to the __divti3-family libcalls before isel)"
            ))))
        }
    }
}

pub(super) fn lower_compare_value(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    compare: CompareValue,
    next_vreg: &mut usize,
) -> CrabbitResult<Register> {
    if is_128_bit_integer(ctx, compare.lhs_ty) {
        return lower_compare_128(ctx, entry, compare, next_vreg);
    }
    let lhs_ty = super::llvm_to_aarch64_isel::compare_operand_ty(ctx, compare.predicate.clone(), compare.lhs_ty);
    let rhs_ty = super::llvm_to_aarch64_isel::compare_operand_ty(ctx, compare.predicate.clone(), compare.rhs_ty);
    let lhs = materialize_typed(
        ctx,
        entry,
        *compare.lhs,
        lhs_ty,
        next_vreg,
        "icmp lhs",
    )?;
    let rhs = materialize_typed(
        ctx,
        entry,
        *compare.rhs,
        rhs_ty,
        next_vreg,
        "icmp rhs",
    )?;
    aarch64_ops::cmp(ctx, lhs, rhs).insert_at_back(entry, ctx);
    let dst = fresh_vreg(next_vreg);
    aarch64_ops::cset(ctx, dst, condition_code(compare.predicate))
        .insert_at_back(entry, ctx);
    Ok(dst)
}

fn lower_compare_128(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    compare: CompareValue,
    next_vreg: &mut usize,
) -> CrabbitResult<Register> {
    let (lhs_lo, lhs_hi) = materialize_pair(
        ctx,
        entry,
        *compare.lhs,
        compare.lhs_ty,
        next_vreg,
        "icmp lhs",
    )?;
    let (rhs_lo, rhs_hi) = materialize_pair(
        ctx,
        entry,
        *compare.rhs,
        compare.rhs_ty,
        next_vreg,
        "icmp rhs",
    )?;
    match compare.predicate {
        ICmpPredicateAttr::EQ => {
            let hi_eq =
                emit_compare_bit(ctx, entry, lhs_hi, rhs_hi, ICmpPredicateAttr::EQ, next_vreg);
            let lo_eq =
                emit_compare_bit(ctx, entry, lhs_lo, rhs_lo, ICmpPredicateAttr::EQ, next_vreg);
            emit_logic_bit(
                ctx,
                entry,
                aarch64_ops::AndOp::OPCODE,
                hi_eq,
                lo_eq,
                next_vreg,
            )
        }
        ICmpPredicateAttr::NE => {
            let hi_ne =
                emit_compare_bit(ctx, entry, lhs_hi, rhs_hi, ICmpPredicateAttr::NE, next_vreg);
            let lo_ne =
                emit_compare_bit(ctx, entry, lhs_lo, rhs_lo, ICmpPredicateAttr::NE, next_vreg);
            emit_logic_bit(
                ctx,
                entry,
                aarch64_ops::OrOp::OPCODE,
                hi_ne,
                lo_ne,
                next_vreg,
            )
        }
        ICmpPredicateAttr::ULT
        | ICmpPredicateAttr::ULE
        | ICmpPredicateAttr::UGT
        | ICmpPredicateAttr::UGE
        | ICmpPredicateAttr::SLT
        | ICmpPredicateAttr::SLE
        | ICmpPredicateAttr::SGT
        | ICmpPredicateAttr::SGE => {
            let (hi_pred, lo_pred) = match compare.predicate {
                ICmpPredicateAttr::ULT => (ICmpPredicateAttr::ULT, ICmpPredicateAttr::ULT),
                ICmpPredicateAttr::ULE => (ICmpPredicateAttr::ULT, ICmpPredicateAttr::ULE),
                ICmpPredicateAttr::UGT => (ICmpPredicateAttr::UGT, ICmpPredicateAttr::UGT),
                ICmpPredicateAttr::UGE => (ICmpPredicateAttr::UGT, ICmpPredicateAttr::UGE),
                ICmpPredicateAttr::SLT => (ICmpPredicateAttr::SLT, ICmpPredicateAttr::ULT),
                ICmpPredicateAttr::SLE => (ICmpPredicateAttr::SLT, ICmpPredicateAttr::ULE),
                ICmpPredicateAttr::SGT => (ICmpPredicateAttr::SGT, ICmpPredicateAttr::UGT),
                ICmpPredicateAttr::SGE => (ICmpPredicateAttr::SGT, ICmpPredicateAttr::UGE),
                ICmpPredicateAttr::EQ | ICmpPredicateAttr::NE => unreachable!(),
            };
            let hi_cmp = emit_compare_bit(
                ctx,
                entry,
                lhs_hi,
                rhs_hi,
                hi_pred,
                next_vreg,
            );
            let hi_eq =
                emit_compare_bit(ctx, entry, lhs_hi, rhs_hi, ICmpPredicateAttr::EQ, next_vreg);
            let lo_cmp = emit_compare_bit(ctx, entry, lhs_lo, rhs_lo, lo_pred, next_vreg);
            let eq_and_lo = emit_logic_bit(
                ctx,
                entry,
                aarch64_ops::AndOp::OPCODE,
                hi_eq,
                lo_cmp,
                next_vreg,
            )?;
            emit_logic_bit(
                ctx,
                entry,
                aarch64_ops::OrOp::OPCODE,
                hi_cmp,
                eq_and_lo,
                next_vreg,
            )
        }
    }
}

fn emit_compare_bit(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    lhs: Register,
    rhs: Register,
    predicate: ICmpPredicateAttr,
    next_vreg: &mut usize,
) -> Register {
    aarch64_ops::cmp(ctx, lhs, rhs).insert_at_back(entry, ctx);
    let dst = fresh_vreg(next_vreg);
    aarch64_ops::cset(ctx, dst, condition_code(predicate))
        .insert_at_back(entry, ctx);
    dst
}

fn emit_logic_bit(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    opcode: Aarch64Opcode,
    lhs: Register,
    rhs: Register,
    next_vreg: &mut usize,
) -> CrabbitResult<Register> {
    let dst = fresh_vreg(next_vreg);
    aarch64_ops::binary(ctx, opcode, dst, lhs, rhs).insert_at_back(entry, ctx);
    Ok(dst)
}

fn lower_shift_right_128(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    lo: Register,
    hi: Register,
    shift: u32,
    signed: bool,
    next_vreg: &mut usize,
) -> CrabbitResult<LoweredValue> {
    let hi_shift_opcode = if signed {
        aarch64_ops::AsrOp::OPCODE
    } else {
        aarch64_ops::LsrOp::OPCODE
    };
    if shift == 0 {
        return Ok(LoweredValue::RegPair(lo, hi));
    }
    if shift < 64 {
        let shift_reg = fresh_vreg(next_vreg);
        materialize_u64_immediate(ctx, entry, shift_reg, shift as u64);
        let lo_part = fresh_vreg(next_vreg);
        aarch64_ops::binary(
            ctx,
            aarch64_ops::LsrOp::OPCODE,
            lo_part,
            lo,
            shift_reg,
        )
        .insert_at_back(entry, ctx);
        let inv_shift = fresh_vreg(next_vreg);
        materialize_u64_immediate(ctx, entry, inv_shift, (64 - shift) as u64);
        let hi_part = fresh_vreg(next_vreg);
        aarch64_ops::binary(
            ctx,
            aarch64_ops::ShlOp::OPCODE,
            hi_part,
            hi,
            inv_shift,
        )
        .insert_at_back(entry, ctx);
        let new_lo = fresh_vreg(next_vreg);
        aarch64_ops::binary(
            ctx,
            aarch64_ops::OrOp::OPCODE,
            new_lo,
            lo_part,
            hi_part,
        )
        .insert_at_back(entry, ctx);
        let new_hi = fresh_vreg(next_vreg);
        let shift_reg = fresh_shift(ctx, entry, shift, next_vreg)?;
        aarch64_ops::binary(ctx, hi_shift_opcode, new_hi, hi, shift_reg)
            .insert_at_back(entry, ctx);
        Ok(LoweredValue::RegPair(new_lo, new_hi))
    } else {
        let new_lo = if shift == 64 {
            hi
        } else {
            let shift_reg = fresh_vreg(next_vreg);
            materialize_u64_immediate(ctx, entry, shift_reg, (shift - 64) as u64);
            let shifted = fresh_vreg(next_vreg);
            aarch64_ops::binary(ctx, hi_shift_opcode, shifted, hi, shift_reg)
                .insert_at_back(entry, ctx);
            shifted
        };
        let new_hi = if signed {
            // The high half becomes the sign extension of the old high half.
            let sixty_three = fresh_shift(ctx, entry, 63, next_vreg)?;
            let sign = fresh_vreg(next_vreg);
            aarch64_ops::binary(ctx, aarch64_ops::AsrOp::OPCODE, sign, hi, sixty_three)
                .insert_at_back(entry, ctx);
            sign
        } else {
            let zero = fresh_vreg(next_vreg);
            materialize_u64_immediate(ctx, entry, zero, 0);
            zero
        };
        Ok(LoweredValue::RegPair(new_lo, new_hi))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DynamicShift {
    Shl,
    Lshr,
    Ashr,
}

/// Branch-free variable-amount 128-bit shift for amounts in `0..=127`.
///
/// The construction relies on AArch64's variable shifts using the amount
/// modulo 64 (so one `lslv`/`lsrv`/`asrv` covers both the `n < 64` and the
/// `n - 64` cases), on `(x >> 1) >> (63 - (n & 63))` being `x >> (64 - n)`
/// for `n > 0` and `0` for `n == 0` (the cross-half carry), and on
/// `cmp n, #64` + `cset` producing all-ones/all-zero masks that select
/// between the two cases without a branch.
fn lower_dynamic_shift_128(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    kind: DynamicShift,
    lo: Register,
    hi: Register,
    amount: Register,
    next_vreg: &mut usize,
) -> CrabbitResult<LoweredValue> {
    let emit_binary = |ctx: &mut Context,
                       opcode: crate::dialects::aarch64::op_interfaces::Aarch64Opcode,
                       lhs: Register,
                       rhs: Register,
                       next_vreg: &mut usize| {
        let dst = fresh_vreg(next_vreg);
        aarch64_ops::binary(ctx, opcode, dst, lhs, rhs).insert_at_back(entry, ctx);
        dst
    };
    let constant = |ctx: &mut Context, value: u64, next_vreg: &mut usize| {
        let dst = fresh_vreg(next_vreg);
        materialize_u64_immediate(ctx, entry, dst, value);
        dst
    };
    use aarch64_ops::{AndOp, AsrOp, LsrOp, OrOp, ShlOp, SubOp};

    // inv = 63 - (n & 63), the carry shift with the n == 0 case folded out.
    let c63 = constant(ctx, 63, next_vreg);
    let n63 = emit_binary(ctx, AndOp::OPCODE, amount, c63, next_vreg);
    let inv = emit_binary(ctx, SubOp::OPCODE, c63, n63, next_vreg);

    // mask_lo = all-ones when n < 64, else zero; mask_hi is its complement.
    let c64 = constant(ctx, 64, next_vreg);
    aarch64_ops::cmp(ctx, amount, c64).insert_at_back(entry, ctx);
    let lt64 = fresh_vreg(next_vreg);
    aarch64_ops::cset(ctx, lt64, condition_code(ICmpPredicateAttr::ULT))
        .insert_at_back(entry, ctx);
    let ge64 = fresh_vreg(next_vreg);
    aarch64_ops::cset(ctx, ge64, condition_code(ICmpPredicateAttr::UGE))
        .insert_at_back(entry, ctx);
    let zero = constant(ctx, 0, next_vreg);
    let mask_lo = emit_binary(ctx, SubOp::OPCODE, zero, lt64, next_vreg);
    let mask_hi = emit_binary(ctx, SubOp::OPCODE, zero, ge64, next_vreg);

    let select = |ctx: &mut Context,
                  low_case: Register,
                  high_case: Register,
                  next_vreg: &mut usize| {
        let low = emit_binary(ctx, AndOp::OPCODE, low_case, mask_lo, next_vreg);
        let high = emit_binary(ctx, AndOp::OPCODE, high_case, mask_hi, next_vreg);
        emit_binary(ctx, OrOp::OPCODE, low, high, next_vreg)
    };

    match kind {
        DynamicShift::Shl => {
            // shl_lo = lo << (n mod 64): the n < 64 low half, and (for
            // n >= 64) exactly lo << (n - 64), the high half.
            let shl_lo = emit_binary(ctx, ShlOp::OPCODE, lo, amount, next_vreg);
            let shl_hi = emit_binary(ctx, ShlOp::OPCODE, hi, amount, next_vreg);
            let one = constant(ctx, 1, next_vreg);
            let lo_half = emit_binary(ctx, LsrOp::OPCODE, lo, one, next_vreg);
            let carry = emit_binary(ctx, LsrOp::OPCODE, lo_half, inv, next_vreg);
            let hi_low_case = emit_binary(ctx, OrOp::OPCODE, shl_hi, carry, next_vreg);
            let new_lo = emit_binary(ctx, AndOp::OPCODE, shl_lo, mask_lo, next_vreg);
            let new_hi = select(ctx, hi_low_case, shl_lo, next_vreg);
            Ok(LoweredValue::RegPair(new_lo, new_hi))
        }
        DynamicShift::Lshr | DynamicShift::Ashr => {
            let hi_shift_opcode = if kind == DynamicShift::Ashr {
                AsrOp::OPCODE
            } else {
                LsrOp::OPCODE
            };
            // hi_shifted = hi >>(s) (n mod 64): the n < 64 high half, and
            // (for n >= 64) exactly hi >>(s) (n - 64), the low half.
            let hi_shifted =
                emit_binary(ctx, hi_shift_opcode, hi, amount, next_vreg);
            let lsr_lo = emit_binary(ctx, LsrOp::OPCODE, lo, amount, next_vreg);
            let one = constant(ctx, 1, next_vreg);
            let hi_double = emit_binary(ctx, ShlOp::OPCODE, hi, one, next_vreg);
            let carry = emit_binary(ctx, ShlOp::OPCODE, hi_double, inv, next_vreg);
            let lo_low_case = emit_binary(ctx, OrOp::OPCODE, lsr_lo, carry, next_vreg);
            let new_lo = select(ctx, lo_low_case, hi_shifted, next_vreg);
            let new_hi = if kind == DynamicShift::Ashr {
                let sixty_three = constant(ctx, 63, next_vreg);
                let sign = emit_binary(ctx, AsrOp::OPCODE, hi, sixty_three, next_vreg);
                select(ctx, hi_shifted, sign, next_vreg)
            } else {
                emit_binary(ctx, AndOp::OPCODE, hi_shifted, mask_lo, next_vreg)
            };
            Ok(LoweredValue::RegPair(new_lo, new_hi))
        }
    }
}

fn lower_shift_left_128(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    lo: Register,
    hi: Register,
    shift: u32,
    next_vreg: &mut usize,
) -> CrabbitResult<LoweredValue> {
    if shift == 0 {
        return Ok(LoweredValue::RegPair(lo, hi));
    }
    if shift < 64 {
        let shift_reg = fresh_shift(ctx, entry, shift, next_vreg)?;
        let new_lo = fresh_vreg(next_vreg);
        aarch64_ops::binary(
            ctx,
            aarch64_ops::ShlOp::OPCODE,
            new_lo,
            lo,
            shift_reg,
        )
        .insert_at_back(entry, ctx);
        let inv_shift = fresh_shift(ctx, entry, 64 - shift, next_vreg)?;
        let carry = fresh_vreg(next_vreg);
        aarch64_ops::binary(
            ctx,
            aarch64_ops::LsrOp::OPCODE,
            carry,
            lo,
            inv_shift,
        )
        .insert_at_back(entry, ctx);
        let hi_part = fresh_vreg(next_vreg);
        let shift_reg = fresh_shift(ctx, entry, shift, next_vreg)?;
        aarch64_ops::binary(
            ctx,
            aarch64_ops::ShlOp::OPCODE,
            hi_part,
            hi,
            shift_reg,
        )
        .insert_at_back(entry, ctx);
        let new_hi = fresh_vreg(next_vreg);
        aarch64_ops::binary(
            ctx,
            aarch64_ops::OrOp::OPCODE,
            new_hi,
            hi_part,
            carry,
        )
        .insert_at_back(entry, ctx);
        Ok(LoweredValue::RegPair(new_lo, new_hi))
    } else {
        let new_lo = fresh_vreg(next_vreg);
        materialize_u64_immediate(ctx, entry, new_lo, 0);
        let new_hi = if shift == 64 {
            lo
        } else {
            let shifted = fresh_vreg(next_vreg);
            let shift_reg = fresh_shift(ctx, entry, shift - 64, next_vreg)?;
            aarch64_ops::binary(
                ctx,
                aarch64_ops::ShlOp::OPCODE,
                shifted,
                lo,
                shift_reg,
            )
            .insert_at_back(entry, ctx);
            shifted
        };
        Ok(LoweredValue::RegPair(new_lo, new_hi))
    }
}

fn fresh_shift(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    shift: u32,
    next_vreg: &mut usize,
) -> CrabbitResult<Register> {
    let reg = fresh_vreg(next_vreg);
    materialize_u64_immediate(ctx, entry, reg, shift as u64);
    Ok(reg)
}

// Addressing and memory -----------------------------------------------------
