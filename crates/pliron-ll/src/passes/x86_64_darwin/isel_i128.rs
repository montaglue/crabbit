use crate::{
    context::{Context, Ptr},
    dialects::{
        x86_64::{op_interfaces::X86_64Opcode, ops as x86_64_ops, registers::Register},
        llvm::attributes::ICmpPredicateAttr,
    },
    input_error_noloc,
    result::STAIRResult,
};

use super::{
    error::X86_64DarwinErr,
    frontend::BinaryKind,
    llvm_to_x86_64_isel::{
        CompareValue, LoweredValue, condition_code, fold_binary, fresh_vreg, is_128_bit_integer,
        materialize_pair, materialize_typed, materialize_u64_immediate, opcode,
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
) -> STAIRResult<LoweredValue> {
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
            x86_64_ops::binary(
                ctx,
                x86_64_ops::AddOp::OPCODE,
                lo.clone(),
                lhs_lo.clone(),
                rhs_lo.clone(),
            )
            .insert_at_back(entry, ctx);
            x86_64_ops::cmp(ctx, lo.clone(), lhs_lo).insert_at_back(entry, ctx);
            let carry = fresh_vreg(next_vreg);
            x86_64_ops::cset(
                ctx,
                carry.clone(),
                condition_code(ICmpPredicateAttr::ULT),
            )
            .insert_at_back(entry, ctx);
            let hi_sum = fresh_vreg(next_vreg);
            x86_64_ops::binary(
                ctx,
                x86_64_ops::AddOp::OPCODE,
                hi_sum.clone(),
                lhs_hi,
                rhs_hi,
            )
            .insert_at_back(entry, ctx);
            let hi = fresh_vreg(next_vreg);
            x86_64_ops::binary(ctx, x86_64_ops::AddOp::OPCODE, hi.clone(), hi_sum, carry)
                .insert_at_back(entry, ctx);
            Ok(LoweredValue::RegPair(lo, hi))
        }
        BinaryKind::Sub => {
            let (rhs_lo, rhs_hi) =
                materialize_pair(ctx, entry, rhs, result_ty, next_vreg, "i128 rhs")?;
            x86_64_ops::cmp(ctx, lhs_lo.clone(), rhs_lo.clone()).insert_at_back(entry, ctx);
            let borrow = fresh_vreg(next_vreg);
            x86_64_ops::cset(
                ctx,
                borrow.clone(),
                condition_code(ICmpPredicateAttr::ULT),
            )
            .insert_at_back(entry, ctx);
            let lo = fresh_vreg(next_vreg);
            x86_64_ops::binary(ctx, x86_64_ops::SubOp::OPCODE, lo.clone(), lhs_lo, rhs_lo)
                .insert_at_back(entry, ctx);
            let hi_sub = fresh_vreg(next_vreg);
            x86_64_ops::binary(
                ctx,
                x86_64_ops::SubOp::OPCODE,
                hi_sub.clone(),
                lhs_hi,
                rhs_hi,
            )
            .insert_at_back(entry, ctx);
            let hi = fresh_vreg(next_vreg);
            x86_64_ops::binary(ctx, x86_64_ops::SubOp::OPCODE, hi.clone(), hi_sub, borrow)
                .insert_at_back(entry, ctx);
            Ok(LoweredValue::RegPair(lo, hi))
        }
        BinaryKind::Mul => {
            let (rhs_lo, rhs_hi) =
                materialize_pair(ctx, entry, rhs, result_ty, next_vreg, "i128 rhs")?;
            let lo = fresh_vreg(next_vreg);
            x86_64_ops::binary(
                ctx,
                x86_64_ops::MulOp::OPCODE,
                lo.clone(),
                lhs_lo.clone(),
                rhs_lo.clone(),
            )
            .insert_at_back(entry, ctx);
            let high_low = fresh_vreg(next_vreg);
            x86_64_ops::binary(
                ctx,
                x86_64_ops::UmulhOp::OPCODE,
                high_low.clone(),
                lhs_lo.clone(),
                rhs_lo.clone(),
            )
            .insert_at_back(entry, ctx);
            let cross_a = fresh_vreg(next_vreg);
            x86_64_ops::binary(
                ctx,
                x86_64_ops::MulOp::OPCODE,
                cross_a.clone(),
                lhs_hi,
                rhs_lo,
            )
            .insert_at_back(entry, ctx);
            let cross_b = fresh_vreg(next_vreg);
            x86_64_ops::binary(
                ctx,
                x86_64_ops::MulOp::OPCODE,
                cross_b.clone(),
                lhs_lo,
                rhs_hi,
            )
            .insert_at_back(entry, ctx);
            let hi_partial = fresh_vreg(next_vreg);
            x86_64_ops::binary(
                ctx,
                x86_64_ops::AddOp::OPCODE,
                hi_partial.clone(),
                high_low,
                cross_a,
            )
            .insert_at_back(entry, ctx);
            let hi = fresh_vreg(next_vreg);
            x86_64_ops::binary(
                ctx,
                x86_64_ops::AddOp::OPCODE,
                hi.clone(),
                hi_partial,
                cross_b,
            )
            .insert_at_back(entry, ctx);
            Ok(LoweredValue::RegPair(lo, hi))
        }
        BinaryKind::Shr => {
            let Some(shift) = shift_amount else {
                return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedOp(
                    "dynamic i128 logical shift right".to_string(),
                )));
            };
            lower_shift_right_128(ctx, entry, lhs_lo, lhs_hi, shift, next_vreg)
        }
        BinaryKind::Shl => {
            let Some(shift) = shift_amount else {
                return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedOp(
                    "dynamic i128 shift left".to_string(),
                )));
            };
            lower_shift_left_128(ctx, entry, lhs_lo, lhs_hi, shift, next_vreg)
        }
        BinaryKind::And | BinaryKind::Or | BinaryKind::Xor => {
            let (rhs_lo, rhs_hi) =
                materialize_pair(ctx, entry, rhs, result_ty, next_vreg, "i128 rhs")?;
            let opcode = opcode(kind);
            let lo = fresh_vreg(next_vreg);
            x86_64_ops::binary(ctx, opcode, lo.clone(), lhs_lo, rhs_lo).insert_at_back(entry, ctx);
            let hi = fresh_vreg(next_vreg);
            x86_64_ops::binary(ctx, opcode, hi.clone(), lhs_hi, rhs_hi).insert_at_back(entry, ctx);
            Ok(LoweredValue::RegPair(lo, hi))
        }
        BinaryKind::SDiv | BinaryKind::UDiv | BinaryKind::SRem | BinaryKind::URem => Err(
            input_error_noloc!(X86_64DarwinErr::UnsupportedOp(format!("128-bit {kind:?}"))),
        ),
    }
}

pub(super) fn lower_compare_value(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    compare: CompareValue,
    next_vreg: &mut usize,
) -> STAIRResult<Register> {
    if is_128_bit_integer(ctx, compare.lhs_ty) {
        return lower_compare_128(ctx, entry, compare, next_vreg);
    }
    let lhs = materialize_typed(
        ctx,
        entry,
        *compare.lhs,
        compare.lhs_ty,
        next_vreg,
        "icmp lhs",
    )?;
    let rhs = materialize_typed(
        ctx,
        entry,
        *compare.rhs,
        compare.rhs_ty,
        next_vreg,
        "icmp rhs",
    )?;
    x86_64_ops::cmp(ctx, lhs, rhs).insert_at_back(entry, ctx);
    let dst = fresh_vreg(next_vreg);
    x86_64_ops::cset(ctx, dst.clone(), condition_code(compare.predicate))
        .insert_at_back(entry, ctx);
    Ok(dst)
}

fn lower_compare_128(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    compare: CompareValue,
    next_vreg: &mut usize,
) -> STAIRResult<Register> {
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
                x86_64_ops::AndOp::OPCODE,
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
                x86_64_ops::OrOp::OPCODE,
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
                lhs_hi.clone(),
                rhs_hi.clone(),
                hi_pred,
                next_vreg,
            );
            let hi_eq =
                emit_compare_bit(ctx, entry, lhs_hi, rhs_hi, ICmpPredicateAttr::EQ, next_vreg);
            let lo_cmp = emit_compare_bit(ctx, entry, lhs_lo, rhs_lo, lo_pred, next_vreg);
            let eq_and_lo = emit_logic_bit(
                ctx,
                entry,
                x86_64_ops::AndOp::OPCODE,
                hi_eq,
                lo_cmp,
                next_vreg,
            )?;
            emit_logic_bit(
                ctx,
                entry,
                x86_64_ops::OrOp::OPCODE,
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
    x86_64_ops::cmp(ctx, lhs, rhs).insert_at_back(entry, ctx);
    let dst = fresh_vreg(next_vreg);
    x86_64_ops::cset(ctx, dst.clone(), condition_code(predicate))
        .insert_at_back(entry, ctx);
    dst
}

fn emit_logic_bit(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    opcode: X86_64Opcode,
    lhs: Register,
    rhs: Register,
    next_vreg: &mut usize,
) -> STAIRResult<Register> {
    let dst = fresh_vreg(next_vreg);
    x86_64_ops::binary(ctx, opcode, dst.clone(), lhs, rhs).insert_at_back(entry, ctx);
    Ok(dst)
}

fn lower_shift_right_128(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    lo: Register,
    hi: Register,
    shift: u32,
    next_vreg: &mut usize,
) -> STAIRResult<LoweredValue> {
    if shift == 0 {
        return Ok(LoweredValue::RegPair(lo, hi));
    }
    if shift < 64 {
        let shift_reg = fresh_vreg(next_vreg);
        materialize_u64_immediate(ctx, entry, shift_reg, shift as u64);
        let lo_part = fresh_vreg(next_vreg);
        x86_64_ops::binary(
            ctx,
            x86_64_ops::LsrOp::OPCODE,
            lo_part.clone(),
            lo,
            shift_reg,
        )
        .insert_at_back(entry, ctx);
        let inv_shift = fresh_vreg(next_vreg);
        materialize_u64_immediate(ctx, entry, inv_shift, (64 - shift) as u64);
        let hi_part = fresh_vreg(next_vreg);
        x86_64_ops::binary(
            ctx,
            x86_64_ops::ShlOp::OPCODE,
            hi_part.clone(),
            hi.clone(),
            inv_shift,
        )
        .insert_at_back(entry, ctx);
        let new_lo = fresh_vreg(next_vreg);
        x86_64_ops::binary(
            ctx,
            x86_64_ops::OrOp::OPCODE,
            new_lo.clone(),
            lo_part,
            hi_part,
        )
        .insert_at_back(entry, ctx);
        let new_hi = fresh_vreg(next_vreg);
        let shift_reg = fresh_shift(ctx, entry, shift, next_vreg)?;
        x86_64_ops::binary(
            ctx,
            x86_64_ops::LsrOp::OPCODE,
            new_hi.clone(),
            hi,
            shift_reg,
        )
        .insert_at_back(entry, ctx);
        Ok(LoweredValue::RegPair(new_lo, new_hi))
    } else {
        let new_lo = if shift == 64 {
            hi
        } else {
            let shift_reg = fresh_vreg(next_vreg);
            materialize_u64_immediate(ctx, entry, shift_reg, (shift - 64) as u64);
            let shifted = fresh_vreg(next_vreg);
            x86_64_ops::binary(
                ctx,
                x86_64_ops::LsrOp::OPCODE,
                shifted.clone(),
                hi,
                shift_reg,
            )
            .insert_at_back(entry, ctx);
            shifted
        };
        let new_hi = fresh_vreg(next_vreg);
        materialize_u64_immediate(ctx, entry, new_hi, 0);
        Ok(LoweredValue::RegPair(new_lo, new_hi))
    }
}

fn lower_shift_left_128(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    lo: Register,
    hi: Register,
    shift: u32,
    next_vreg: &mut usize,
) -> STAIRResult<LoweredValue> {
    if shift == 0 {
        return Ok(LoweredValue::RegPair(lo, hi));
    }
    if shift < 64 {
        let shift_reg = fresh_shift(ctx, entry, shift, next_vreg)?;
        let new_lo = fresh_vreg(next_vreg);
        x86_64_ops::binary(
            ctx,
            x86_64_ops::ShlOp::OPCODE,
            new_lo.clone(),
            lo.clone(),
            shift_reg,
        )
        .insert_at_back(entry, ctx);
        let inv_shift = fresh_shift(ctx, entry, 64 - shift, next_vreg)?;
        let carry = fresh_vreg(next_vreg);
        x86_64_ops::binary(
            ctx,
            x86_64_ops::LsrOp::OPCODE,
            carry.clone(),
            lo,
            inv_shift,
        )
        .insert_at_back(entry, ctx);
        let hi_part = fresh_vreg(next_vreg);
        let shift_reg = fresh_shift(ctx, entry, shift, next_vreg)?;
        x86_64_ops::binary(
            ctx,
            x86_64_ops::ShlOp::OPCODE,
            hi_part.clone(),
            hi,
            shift_reg,
        )
        .insert_at_back(entry, ctx);
        let new_hi = fresh_vreg(next_vreg);
        x86_64_ops::binary(
            ctx,
            x86_64_ops::OrOp::OPCODE,
            new_hi.clone(),
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
            x86_64_ops::binary(
                ctx,
                x86_64_ops::ShlOp::OPCODE,
                shifted.clone(),
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
) -> STAIRResult<Register> {
    let reg = fresh_vreg(next_vreg);
    materialize_u64_immediate(ctx, entry, reg, shift as u64);
    Ok(reg)
}

// Addressing and memory -----------------------------------------------------
