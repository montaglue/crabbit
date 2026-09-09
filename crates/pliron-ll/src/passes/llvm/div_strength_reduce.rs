//! Strength reduction of integer division and remainder by constants
//! (docs/MIDEND-PLAN.md item 2): `sdiv`/`udiv`/`srem`/`urem` with a
//! non-zero constant divisor become shift/multiply sequences.
//!
//! - Powers of two: `udiv` → `lshr`, `urem` → `and`, `sdiv` → the LLVM
//!   sign-fix sequence (`add (lshr (ashr a, w-1), w-k)` then `ashr`),
//!   `srem` → `a - (a/c)*c` on that quotient.
//! - General constants: magic-number multiply-high (Hacker's Delight
//!   ch. 10; LLVM `TargetLowering::BuildSDIV`/`BuildUDIV`), including the
//!   add/sub fixups for magics that overflow the word. Remainders are
//!   `a - (a/c)*c`.
//! - The multiply-high itself: the LLVM dialect has no `mulh` op, so i32
//!   widens to i64 (`ext` → `mul` → `lshr 32` → `trunc`). i64 does NOT go
//!   through i128 — the NVPTX emitter (which shares this mid-end) has no
//!   i128 register class — but through the portable 4-multiply half-word
//!   decomposition (Hacker's Delight 8-2 `mulhu`/`mulhs`) in i64 ops.
//! - i8/i16 divisions widen to i32 first (`sext` for the signed forms,
//!   `zext` for the unsigned ones) and truncate the result back.
//! - Skipped: i128 (lowered to libcalls later), divisor 0 (UB, left as a
//!   real divide), and SSA-value divisors. Divisor 1/-1 folds to the
//!   identity/negation, signed divisor `MIN` to `zext (icmp eq a, MIN)`.
//!
//! Disabled by `CRABBIT_MIDEND_DISABLE=divmagic` (see
//! [super::midend_gate]).

use std::num::NonZero;

use pliron::builtin::op_interfaces::AtMostOneRegionInterface as _;
use pliron_llvm::op_interfaces::{
    BinArithOp as _, CastOpInterface as _, CastOpWithNNegInterface as _,
    IntBinArithOpWithOverflowFlag as _,
};

use crate::{
    context::{Context, Ptr},
    conversion::pass::{AnalysisManager, Pass, PassResult, changed, unchanged},
    dialects::{
        builtin::{
            attributes::IntegerAttr,
            op_interfaces::OneResultInterface,
            types::{IntegerType, Signedness},
        },
        builtin::ops::ConstantOp,
        llvm::{
            attributes::ICmpPredicateAttr,
            op_interfaces::IsDeclaration,
            ops::{
                AShrOp, AddOp, AndOp, ICmpOp, LShrOp, MulOp, SDivOp, SExtOp, SRemOp, SubOp,
                TruncOp, UDivOp, URemOp, ZExtOp,
            },
        },
    },
    ir::{
        op::Op,
        operation::Operation,
        r#type::TypedHandle,
        value::Value,
    },
    result::STAIRResult,
    utils::apint::APInt,
};

use super::inline::collect_functions;
use super::midend_gate::midend_disabled;
// ADJOINT (backward attribution): every expansion funnels through
// `simplify::replace_op_with_value`, whose chain walk stamps the whole
// freshly built multiply-high/shift sequence as derived from the divide
// it replaces (1→N, single parent). No per-site stamping needed here.
use super::simplify::{as_const_operand, function_ops, mask_to_width, replace_op_with_value, sign_extend};

pub struct LLVMDivStrengthReducePass;

impl Pass for LLVMDivStrengthReducePass {
    fn name(&self) -> &str {
        "llvm-div-strength-reduce"
    }

    fn run(
        &mut self,
        root: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> pliron::result::Result<PassResult> {
        if midend_disabled("divmagic") {
            return Ok(unchanged());
        }
        let mut any = false;
        for func in collect_functions(ctx, root) {
            if func.is_declaration(ctx) {
                continue;
            }
            let region = func
                .get_region(ctx)
                .expect("llvm.func definition must have a body");
            for op in function_ops(ctx, region) {
                any |= rewrite_div(ctx, op)?;
            }
        }
        Ok(if any { changed() } else { unchanged() })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DivKind {
    SDiv,
    UDiv,
    SRem,
    URem,
}

impl DivKind {
    fn signed(self) -> bool {
        matches!(self, DivKind::SDiv | DivKind::SRem)
    }

    fn rem(self) -> bool {
        matches!(self, DivKind::SRem | DivKind::URem)
    }
}

fn div_kind(ctx: &Context, op: Ptr<Operation>) -> Option<DivKind> {
    let opid = Operation::get_opid(op, ctx);
    if opid == SDivOp::get_opid_static() {
        Some(DivKind::SDiv)
    } else if opid == UDivOp::get_opid_static() {
        Some(DivKind::UDiv)
    } else if opid == SRemOp::get_opid_static() {
        Some(DivKind::SRem)
    } else if opid == URemOp::get_opid_static() {
        Some(DivKind::URem)
    } else {
        None
    }
}

fn rewrite_div(ctx: &mut Context, op: Ptr<Operation>) -> STAIRResult<bool> {
    let Some(kind) = div_kind(ctx, op) else {
        return Ok(false);
    };
    let (dividend, divisor_v) = {
        let op_ref = op.deref(ctx);
        (op_ref.get_operand(0), op_ref.get_operand(1))
    };
    let Some(divisor) = as_const_operand(ctx, divisor_v) else {
        return Ok(false);
    };
    let width = divisor.width;
    // i128 goes to the __divti3-family libcalls; i1 divides are degenerate.
    if !matches!(width, 8 | 16 | 32 | 64) {
        return Ok(false);
    }
    let d_bits = divisor.masked();
    if d_bits == 0 {
        // Division by zero is UB; leave the real divide in place.
        return Ok(false);
    }
    let ty = divisor.ty;

    // Divisor 1/-1: identity, negation, or a zero remainder.
    if d_bits == 1 {
        if kind.rem() {
            let zero = emit_const(ctx, op, ty, 0);
            replace_op_with_value(ctx, op, zero);
        } else {
            replace_op_with_value(ctx, op, dividend);
        }
        return Ok(true);
    }
    if kind.signed() && divisor.signed() == -1 {
        let value = if kind.rem() {
            emit_const(ctx, op, ty, 0)
        } else {
            let zero = emit_const(ctx, op, ty, 0);
            emit_sub(ctx, op, zero, dividend)
        };
        replace_op_with_value(ctx, op, value);
        return Ok(true);
    }

    let result = if width < 32 {
        // Widen to i32: the quotient/remainder of the widened operands
        // fits the narrow width (divisor -1 was folded above, so the
        // MIN/-1 wrap case cannot arise), and two's complement truncation
        // restores the narrow result exactly.
        let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
        let wide = if kind.signed() {
            let cast = SExtOp::new(ctx, dividend, i32_ty.into());
            cast.get_operation().insert_before(ctx, op);
            cast.get_result(ctx)
        } else {
            let cast = ZExtOp::new_with_nneg(ctx, dividend, i32_ty.into(), false);
            cast.get_operation().insert_before(ctx, op);
            cast.get_result(ctx)
        };
        let wide_d = if kind.signed() {
            (divisor.signed() as u128) & mask_to_width(u128::MAX, 32)
        } else {
            d_bits
        };
        let wide_result = expand(ctx, op, kind, 32, i32_ty, wide, wide_d);
        let trunc = TruncOp::new(ctx, wide_result, ty.into());
        trunc.get_operation().insert_before(ctx, op);
        trunc.get_result(ctx)
    } else {
        expand(ctx, op, kind, width, ty, dividend, d_bits)
    };
    replace_op_with_value(ctx, op, result);
    Ok(true)
}

/// Emit the shift/multiply sequence for `a <kind> d` right before `before`
/// and return the resulting value. `d_bits` is the divisor masked to
/// `width`; `width` is 32 or 64.
fn expand(
    ctx: &mut Context,
    before: Ptr<Operation>,
    kind: DivKind,
    width: u32,
    ty: TypedHandle<IntegerType>,
    a: Value,
    d_bits: u128,
) -> Value {
    match kind {
        DivKind::UDiv => expand_udiv(ctx, before, width, ty, a, d_bits),
        DivKind::SDiv => expand_sdiv(ctx, before, width, ty, a, d_bits),
        DivKind::URem => {
            if d_bits.is_power_of_two() {
                let mask = emit_const(ctx, before, ty, d_bits - 1);
                return emit_and(ctx, before, a, mask);
            }
            let q = expand_udiv(ctx, before, width, ty, a, d_bits);
            emit_rem_from_quotient(ctx, before, ty, a, q, d_bits)
        }
        DivKind::SRem => {
            let q = expand_sdiv(ctx, before, width, ty, a, d_bits);
            emit_rem_from_quotient(ctx, before, ty, a, q, d_bits)
        }
    }
}

/// `rem = a - q*d`, exact in wrapping arithmetic for a correct quotient.
fn emit_rem_from_quotient(
    ctx: &mut Context,
    before: Ptr<Operation>,
    ty: TypedHandle<IntegerType>,
    a: Value,
    q: Value,
    d_bits: u128,
) -> Value {
    let d = emit_const(ctx, before, ty, d_bits);
    let prod = emit_mul(ctx, before, q, d);
    emit_sub(ctx, before, a, prod)
}

fn expand_udiv(
    ctx: &mut Context,
    before: Ptr<Operation>,
    width: u32,
    ty: TypedHandle<IntegerType>,
    a: Value,
    d: u128,
) -> Value {
    debug_assert!(d >= 2);
    if d.is_power_of_two() {
        let k = emit_const(ctx, before, ty, d.trailing_zeros() as u128);
        return emit_lshr(ctx, before, a, k);
    }
    let magic = magic_unsigned(d, width);
    let t = emit_mulh(ctx, before, width, ty, a, magic.multiplier, false);
    if !magic.add {
        if magic.shift == 0 {
            return t;
        }
        let s = emit_const(ctx, before, ty, magic.shift as u128);
        emit_lshr(ctx, before, t, s)
    } else {
        // The magic overflowed the word: q = (t + ((a - t) >> 1)) >> (s-1),
        // LLVM's BuildUDIV fixup. `shift >= 1` always holds here.
        debug_assert!(magic.shift >= 1);
        let diff = emit_sub(ctx, before, a, t);
        let one = emit_const(ctx, before, ty, 1);
        let half = emit_lshr(ctx, before, diff, one);
        let sum = emit_add(ctx, before, half, t);
        let s = emit_const(ctx, before, ty, (magic.shift - 1) as u128);
        emit_lshr(ctx, before, sum, s)
    }
}

fn expand_sdiv(
    ctx: &mut Context,
    before: Ptr<Operation>,
    width: u32,
    ty: TypedHandle<IntegerType>,
    a: Value,
    d_bits: u128,
) -> Value {
    let d = sign_extend(d_bits, width);
    let min = -(1i128 << (width - 1));
    if d == min {
        // Only MIN / MIN == 1; everything else quotients to 0.
        let min_c = emit_const(ctx, before, ty, d_bits);
        let cmp = ICmpOp::new(ctx, ICmpPredicateAttr::EQ, a, min_c);
        cmp.get_operation().insert_before(ctx, before);
        let cast = ZExtOp::new_with_nneg(ctx, cmp.get_result(ctx), ty.into(), false);
        cast.get_operation().insert_before(ctx, before);
        return cast.get_result(ctx);
    }
    let ad = d.unsigned_abs();
    if ad.is_power_of_two() {
        // LLVM's sign-fix sequence: bias negative dividends by |d|-1 so
        // the arithmetic shift rounds toward zero.
        let k = ad.trailing_zeros();
        debug_assert!(k >= 1);
        let w_m1 = emit_const(ctx, before, ty, (width - 1) as u128);
        let sgn = emit_ashr(ctx, before, a, w_m1);
        let w_m_k = emit_const(ctx, before, ty, (width - k) as u128);
        let srl = emit_lshr(ctx, before, sgn, w_m_k);
        let biased = emit_add(ctx, before, a, srl);
        let k_c = emit_const(ctx, before, ty, k as u128);
        let q = emit_ashr(ctx, before, biased, k_c);
        return if d < 0 {
            let zero = emit_const(ctx, before, ty, 0);
            emit_sub(ctx, before, zero, q)
        } else {
            q
        };
    }
    let magic = magic_signed(d, width);
    let m_bits = (magic.multiplier as u128) & mask_to_width(u128::MAX, width);
    let mut q = emit_mulh(ctx, before, width, ty, a, m_bits, true);
    // BuildSDIV fixups for magics whose sign disagrees with the divisor.
    if d > 0 && magic.multiplier < 0 {
        q = emit_add(ctx, before, q, a);
    } else if d < 0 && magic.multiplier > 0 {
        q = emit_sub(ctx, before, q, a);
    }
    if magic.shift > 0 {
        let s = emit_const(ctx, before, ty, magic.shift as u128);
        q = emit_ashr(ctx, before, q, s);
    }
    // Add the sign bit to round the shifted quotient toward zero.
    let w_m1 = emit_const(ctx, before, ty, (width - 1) as u128);
    let sign = emit_lshr(ctx, before, q, w_m1);
    emit_add(ctx, before, q, sign)
}

/// The high `width` bits of `a * m` (`m` a constant, both `width` wide).
///
/// i32 widens to i64; i64 uses the Hacker's Delight 8-2 half-word
/// decomposition in i64 arithmetic (no i128 — the NVPTX backend, which
/// shares this mid-end, has no i128 register class).
fn emit_mulh(
    ctx: &mut Context,
    before: Ptr<Operation>,
    width: u32,
    ty: TypedHandle<IntegerType>,
    a: Value,
    m_bits: u128,
    signed: bool,
) -> Value {
    if width == 32 {
        let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
        let wide_a = if signed {
            let cast = SExtOp::new(ctx, a, i64_ty.into());
            cast.get_operation().insert_before(ctx, before);
            cast.get_result(ctx)
        } else {
            let cast = ZExtOp::new_with_nneg(ctx, a, i64_ty.into(), false);
            cast.get_operation().insert_before(ctx, before);
            cast.get_result(ctx)
        };
        let m_wide = if signed {
            (sign_extend(m_bits, 32) as u128) & mask_to_width(u128::MAX, 64)
        } else {
            m_bits
        };
        let m_c = emit_const(ctx, before, i64_ty, m_wide);
        let prod = emit_mul(ctx, before, wide_a, m_c);
        let s32 = emit_const(ctx, before, i64_ty, 32);
        let hi = emit_lshr(ctx, before, prod, s32);
        let trunc = TruncOp::new(ctx, hi, ty.into());
        trunc.get_operation().insert_before(ctx, before);
        return trunc.get_result(ctx);
    }
    debug_assert_eq!(width, 64);
    let shr = |ctx: &mut Context, v: Value, k: Value, signed: bool| {
        if signed {
            emit_ashr(ctx, before, v, k)
        } else {
            emit_lshr(ctx, before, v, k)
        }
    };
    let mask32 = emit_const(ctx, before, ty, 0xFFFF_FFFF);
    let c32 = emit_const(ctx, before, ty, 32);
    let u0 = emit_and(ctx, before, a, mask32);
    let u1 = shr(ctx, a, c32, signed);
    let v0_bits = m_bits & 0xFFFF_FFFF;
    let v1_bits = if signed {
        ((sign_extend(m_bits, 64) >> 32) as u128) & mask_to_width(u128::MAX, 64)
    } else {
        mask_to_width(m_bits, 64) >> 32
    };
    let v0 = emit_const(ctx, before, ty, v0_bits);
    let v1 = emit_const(ctx, before, ty, v1_bits);
    let w0 = emit_mul(ctx, before, u0, v0);
    let u1v0 = emit_mul(ctx, before, u1, v0);
    let w0_hi = emit_lshr(ctx, before, w0, c32);
    let t = emit_add(ctx, before, u1v0, w0_hi);
    let w1_lo = emit_and(ctx, before, t, mask32);
    let w2 = shr(ctx, t, c32, signed);
    let u0v1 = emit_mul(ctx, before, u0, v1);
    let w1 = emit_add(ctx, before, u0v1, w1_lo);
    let w1_hi = shr(ctx, w1, c32, signed);
    let u1v1 = emit_mul(ctx, before, u1, v1);
    let partial = emit_add(ctx, before, u1v1, w2);
    emit_add(ctx, before, partial, w1_hi)
}

// ============================================================================
// Magic numbers (Hacker's Delight 10-4 / 10-9, as in LLVM's APInt magics)
// ============================================================================

struct UnsignedMagic {
    multiplier: u128,
    shift: u32,
    /// The true magic is `multiplier + 2^width`: use the add fixup.
    add: bool,
}

/// Hacker's Delight 10-9 `magicu` for a `width`-bit unsigned divisor
/// `d >= 2` that is not a power of two, computed in u128.
fn magic_unsigned(d: u128, width: u32) -> UnsignedMagic {
    debug_assert!(d >= 2 && !d.is_power_of_two() && d <= mask_to_width(u128::MAX, width));
    let two_w_m1 = 1u128 << (width - 1);
    let msk = mask_to_width(u128::MAX, width);
    let mut add = false;
    let mut p = width - 1;
    let nc = msk - (msk - d + 1) % d;
    let mut q1 = two_w_m1 / nc;
    let mut r1 = two_w_m1 - q1 * nc;
    let mut q2 = (two_w_m1 - 1) / d;
    let mut r2 = (two_w_m1 - 1) - q2 * d;
    loop {
        p += 1;
        if r1 >= nc - r1 {
            q1 = 2 * q1 + 1;
            r1 = 2 * r1 - nc;
        } else {
            q1 *= 2;
            r1 *= 2;
        }
        if r2 + 1 >= d - r2 {
            if q2 >= two_w_m1 - 1 {
                add = true;
            }
            q2 = 2 * q2 + 1;
            r2 = 2 * r2 + 1 - d;
        } else {
            if q2 >= two_w_m1 {
                add = true;
            }
            q2 *= 2;
            r2 = 2 * r2 + 1;
        }
        let delta = d - 1 - r2;
        if !(p < 2 * width && (q1 < delta || (q1 == delta && r1 == 0))) {
            break;
        }
    }
    UnsignedMagic {
        multiplier: (q2 + 1) & msk,
        shift: p - width,
        add,
    }
}

struct SignedMagic {
    /// Sign-extended `width`-bit magic multiplier.
    multiplier: i128,
    shift: u32,
}

/// Hacker's Delight 10-4 `magic` for a `width`-bit signed divisor with
/// `2 <= |d| < 2^(width-1)` (i.e. not 0, ±1, or MIN), computed in u128.
fn magic_signed(d: i128, width: u32) -> SignedMagic {
    let msk = mask_to_width(u128::MAX, width);
    let two_w_m1 = 1u128 << (width - 1);
    let ad = d.unsigned_abs() & msk;
    debug_assert!(ad >= 2 && ad != two_w_m1);
    let t = two_w_m1 + (((d as u128) & msk) >> (width - 1));
    let anc = t - 1 - t % ad;
    let mut p = width - 1;
    let mut q1 = two_w_m1 / anc;
    let mut r1 = two_w_m1 - q1 * anc;
    let mut q2 = two_w_m1 / ad;
    let mut r2 = two_w_m1 - q2 * ad;
    loop {
        p += 1;
        q1 *= 2;
        r1 *= 2;
        if r1 >= anc {
            q1 += 1;
            r1 -= anc;
        }
        q2 *= 2;
        r2 *= 2;
        if r2 >= ad {
            q2 += 1;
            r2 -= ad;
        }
        let delta = ad - r2;
        if !(q1 < delta || (q1 == delta && r1 == 0)) {
            break;
        }
    }
    let mut m = (q2 + 1) & msk;
    if d < 0 {
        m = msk.wrapping_sub(m).wrapping_add(1) & msk;
    }
    SignedMagic {
        multiplier: sign_extend(m, width),
        shift: p - width,
    }
}

// ============================================================================
// Op emission helpers (insert before a given op, return the result value)
// ============================================================================

fn emit_const(
    ctx: &mut Context,
    before: Ptr<Operation>,
    ty: TypedHandle<IntegerType>,
    bits: u128,
) -> Value {
    let width = ty.deref(ctx).width();
    let apint = APInt::from_u128(
        mask_to_width(bits, width),
        NonZero::new(width as usize).unwrap(),
    );
    let constant = ConstantOp::new(ctx, Box::new(IntegerAttr::new(ty, apint)));
    constant.get_operation().insert_before(ctx, before);
    constant.get_result(ctx)
}

macro_rules! emit_overflow_bin {
    ($fn_name:ident, $op:ident) => {
        fn $fn_name(ctx: &mut Context, before: Ptr<Operation>, lhs: Value, rhs: Value) -> Value {
            let op = $op::new_with_overflow_flag(ctx, lhs, rhs, Default::default());
            op.get_operation().insert_before(ctx, before);
            op.get_result(ctx)
        }
    };
}

macro_rules! emit_plain_bin {
    ($fn_name:ident, $op:ident) => {
        fn $fn_name(ctx: &mut Context, before: Ptr<Operation>, lhs: Value, rhs: Value) -> Value {
            let op = $op::new(ctx, lhs, rhs);
            op.get_operation().insert_before(ctx, before);
            op.get_result(ctx)
        }
    };
}

emit_overflow_bin!(emit_add, AddOp);
emit_overflow_bin!(emit_sub, SubOp);
emit_overflow_bin!(emit_mul, MulOp);
emit_plain_bin!(emit_and, AndOp);
emit_plain_bin!(emit_lshr, LShrOp);
emit_plain_bin!(emit_ashr, AShrOp);

#[cfg(test)]
mod tests {
    use crate::r#type::TypeHandle;
    use super::*;
    use crate::{
        dialects::llvm::{
            attributes::LinkageAttr,
            ops::{FuncOp, ReturnOp},
            types::FuncType,
        },
        printable::Printable,
    };
    use super::super::simplify::LLVMSimplifyPass;

    fn int_ty(ctx: &mut Context, width: u32) -> TypedHandle<IntegerType> {
        IntegerType::get(ctx, width, Signedness::Signless)
    }

    fn int_const(ctx: &mut Context, width: u32, bits: u128) -> ConstantOp {
        let ty = int_ty(ctx, width);
        let apint = APInt::from_u128(
            mask_to_width(bits, width),
            NonZero::new(width as usize).unwrap(),
        );
        ConstantOp::new(ctx, Box::new(IntegerAttr::new(ty, apint)))
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Kind {
        SDiv,
        UDiv,
        SRem,
        URem,
    }

    /// Build `fn() -> iW { return <kind>(<a>, <b>) }`, run the
    /// strength-reduction pass, assert no divide survives, constant-fold
    /// the emitted sequence with the simplify pass and return the folded
    /// result bits.
    fn run_case(width: u32, kind: Kind, a_bits: u128, b_bits: u128) -> u128 {
        let mut ctx = Context::new();
        let ty: TypeHandle = int_ty(&mut ctx, width).into();
        let fn_ty: TypedHandle<FuncType> = FuncType::get(&mut ctx, ty, vec![], false);
        let func = FuncOp::new(&mut ctx, "case".try_into().unwrap(), fn_ty);
        func.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(&mut ctx);
        let entry = func.get_entry_block(&ctx).unwrap();

        let a = int_const(&mut ctx, width, a_bits);
        a.get_operation().insert_at_back(entry, &ctx);
        let b = int_const(&mut ctx, width, b_bits);
        b.get_operation().insert_at_back(entry, &ctx);
        let a_v = a.get_result(&ctx);
        let b_v = b.get_result(&ctx);
        let div = match kind {
            Kind::SDiv => SDivOp::new(&mut ctx, a_v, b_v).get_operation(),
            Kind::UDiv => UDivOp::new(&mut ctx, a_v, b_v).get_operation(),
            Kind::SRem => SRemOp::new(&mut ctx, a_v, b_v).get_operation(),
            Kind::URem => URemOp::new(&mut ctx, a_v, b_v).get_operation(),
        };
        div.insert_at_back(entry, &ctx);
        let result = div.deref(&ctx).get_result(0);
        ReturnOp::new(&mut ctx, Some(result))
            .get_operation()
            .insert_at_back(entry, &ctx);

        let root = func.get_operation();
        LLVMDivStrengthReducePass
            .run(root, &mut ctx, &mut AnalysisManager::default())
            .unwrap();
        let text = format!("{}", root.disp(&ctx));
        assert!(
            !text.contains("div") && !text.contains("rem"),
            "divide survived strength reduction (w={width} {kind:?} b={b_bits:#x}):\n{text}"
        );
        LLVMSimplifyPass
            .run(root, &mut ctx, &mut AnalysisManager::default())
            .unwrap();

        // The folded body must be [constant, return]; read the constant.
        let region = func.get_region(&ctx).unwrap();
        for op in function_ops(&ctx, region) {
            if Operation::get_opid(op, &ctx) == ReturnOp::get_opid_static() {
                let ret_val = op.deref(&ctx).get_operand(0);
                let folded = as_const_operand(&ctx, ret_val).unwrap_or_else(|| {
                    panic!(
                        "sequence did not fold to a constant (w={width} {kind:?} \
                         a={a_bits:#x} b={b_bits:#x}):\n{}",
                        root.disp(&ctx)
                    )
                });
                return folded.masked();
            }
        }
        panic!("no return op found");
    }

    fn expected(width: u32, kind: Kind, a_bits: u128, b_bits: u128) -> u128 {
        let msk = mask_to_width(u128::MAX, width);
        match kind {
            Kind::UDiv => ((a_bits & msk) / (b_bits & msk)) & msk,
            Kind::URem => ((a_bits & msk) % (b_bits & msk)) & msk,
            Kind::SDiv => {
                let a = sign_extend(a_bits, width);
                let d = sign_extend(b_bits, width);
                // Wrapping semantics for MIN / -1 (matches the negate fold).
                let q = if d == -1 { a.wrapping_neg() } else { a / d };
                (q as u128) & msk
            }
            Kind::SRem => {
                let a = sign_extend(a_bits, width);
                let d = sign_extend(b_bits, width);
                let r = if d == -1 { 0 } else { a % d };
                (r as u128) & msk
            }
        }
    }

    fn check(width: u32, kind: Kind, a_bits: u128, b_bits: u128) {
        let got = run_case(width, kind, a_bits, b_bits);
        let want = expected(width, kind, a_bits, b_bits);
        assert_eq!(
            got,
            want,
            "w={width} {kind:?} a={:#x} b={:#x}: got {got:#x} want {want:#x}",
            a_bits,
            b_bits
        );
    }

    fn dividends(width: u32) -> Vec<u128> {
        let msk = mask_to_width(u128::MAX, width);
        let min = 1u128 << (width - 1);
        let max = min - 1;
        let mut out = vec![
            0,
            1,
            2,
            6,
            7,
            msk,             // -1
            msk - 1,         // -2
            msk - 5,         // -6
            msk - 6,         // -7
            min,             // MIN
            max,             // MAX
            (min + 1) & msk, // MIN + 1
        ];
        // Fixed-seed xorshift randoms.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        for _ in 0..8 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            out.push(state as u128 & msk);
        }
        out
    }

    /// The deliverable's exhaustive sweep: every divisor in the list (and
    /// its negation, which the unsigned forms see as a huge divisor) x
    /// every dividend x all four ops, for i32 and i64, compared against
    /// the Rust operators.
    #[test]
    fn sweeps_i32_i64_against_rust_semantics() {
        let divisors: [i128; 12] = [2, 3, 5, 6, 7, 9, 10, 255, 256, 257, 641, 1000003];
        for width in [32u32, 64] {
            let msk = mask_to_width(u128::MAX, width);
            for &d0 in &divisors {
                for d in [d0, -d0] {
                    let d_bits = (d as u128) & msk;
                    for a in dividends(width) {
                        for kind in [Kind::SDiv, Kind::UDiv, Kind::SRem, Kind::URem] {
                            check(width, kind, a, d_bits);
                        }
                    }
                }
            }
        }
    }

    /// Signed divisor MIN takes the icmp path; +1/-1 fold to the
    /// identity/negation.
    #[test]
    fn handles_edge_divisors() {
        for width in [32u32, 64] {
            let msk = mask_to_width(u128::MAX, width);
            let min = 1u128 << (width - 1);
            for a in dividends(width) {
                for kind in [Kind::SDiv, Kind::SRem] {
                    check(width, kind, a, min); // MIN
                }
                for kind in [Kind::SDiv, Kind::UDiv, Kind::SRem, Kind::URem] {
                    check(width, kind, a, 1);
                }
                for kind in [Kind::SDiv, Kind::SRem] {
                    check(width, kind, a, msk); // -1
                }
            }
        }
    }

    /// i8/i16 divide via widening to i32; every i8 divisor is swept
    /// exhaustively, i16 with the standard list.
    #[test]
    fn sweeps_narrow_widths_via_widening() {
        let dividends8: Vec<u128> = (0..=255u128).step_by(17).chain([1, 127, 128, 129, 255]).collect();
        for d in 2..=255u128 {
            for &a in &dividends8 {
                for kind in [Kind::SDiv, Kind::UDiv, Kind::SRem, Kind::URem] {
                    check(8, kind, a, d);
                }
            }
        }
        let divisors16: [i128; 9] = [2, 3, 5, 7, 10, 255, 256, 257, 641];
        for &d0 in &divisors16 {
            for d in [d0, -d0] {
                let d_bits = (d as u128) & 0xFFFF;
                for a in dividends(16) {
                    for kind in [Kind::SDiv, Kind::UDiv, Kind::SRem, Kind::URem] {
                        check(16, kind, a, d_bits);
                    }
                }
            }
        }
    }

    /// A divisor that is an SSA value (function argument) must stay a real
    /// divide, and so must a constant-zero divisor and i128 divides.
    #[test]
    fn leaves_non_constant_zero_and_i128_divisors_alone() {
        // SSA divisor.
        let mut ctx = Context::new();
        let ty: TypeHandle = int_ty(&mut ctx, 32).into();
        let fn_ty: TypedHandle<FuncType> = FuncType::get(&mut ctx, ty, vec![ty, ty], false);
        let func = FuncOp::new(&mut ctx, "ssa_divisor".try_into().unwrap(), fn_ty);
        func.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(&mut ctx);
        let entry = func.get_entry_block(&ctx).unwrap();
        let a = entry.deref(&ctx).get_argument(0);
        let b = entry.deref(&ctx).get_argument(1);
        let div = SDivOp::new(&mut ctx, a, b);
        div.get_operation().insert_at_back(entry, &ctx);
        let q = div.get_result(&ctx);
        ReturnOp::new(&mut ctx, Some(q))
            .get_operation()
            .insert_at_back(entry, &ctx);
        let root = func.get_operation();
        LLVMDivStrengthReducePass
            .run(root, &mut ctx, &mut AnalysisManager::default())
            .unwrap();
        let text = format!("{}", root.disp(&ctx));
        assert!(text.contains("llvm.sdiv"), "{text}");

        // Constant-zero divisor.
        for (width, expect_kept) in [(32u32, "llvm.udiv"), (128, "llvm.udiv")] {
            let mut ctx = Context::new();
            let ty: TypeHandle = int_ty(&mut ctx, width).into();
            let fn_ty: TypedHandle<FuncType> = FuncType::get(&mut ctx, ty, vec![ty], false);
            let func = FuncOp::new(&mut ctx, "kept".try_into().unwrap(), fn_ty);
            func.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
            func.get_or_create_entry_block(&mut ctx);
            let entry = func.get_entry_block(&ctx).unwrap();
            let a = entry.deref(&ctx).get_argument(0);
            // width==32 gets divisor 0; width==128 a normal constant (i128
            // must be left for the libcall path).
            let d_bits = if width == 32 { 0 } else { 3 };
            let d = int_const(&mut ctx, width, d_bits);
            d.get_operation().insert_at_back(entry, &ctx);
            let d_v = d.get_result(&ctx);
            let div = UDivOp::new(&mut ctx, a, d_v);
            div.get_operation().insert_at_back(entry, &ctx);
            let q = div.get_result(&ctx);
            ReturnOp::new(&mut ctx, Some(q))
                .get_operation()
                .insert_at_back(entry, &ctx);
            let root = func.get_operation();
            LLVMDivStrengthReducePass
                .run(root, &mut ctx, &mut AnalysisManager::default())
                .unwrap();
            let text = format!("{}", root.disp(&ctx));
            assert!(text.contains(expect_kept), "w={width}: {text}");
        }
    }

    /// Spot-check the magic constants against LLVM's known values.
    #[test]
    fn magic_constants_match_llvm() {
        // u32 / 7: magic 0x24924925, shift 3, with the add fixup.
        let m = magic_unsigned(7, 32);
        assert_eq!(m.multiplier, 0x2492_4925);
        assert_eq!(m.shift, 3);
        assert!(m.add);
        // u32 / 5: magic 0xCCCCCCCD, shift 2, no fixup.
        let m = magic_unsigned(5, 32);
        assert_eq!(m.multiplier, 0xCCCC_CCCD);
        assert_eq!(m.shift, 2);
        assert!(!m.add);
        // i32 / 3: magic 0x55555556, shift 0.
        let m = magic_signed(3, 32);
        assert_eq!(m.multiplier, 0x5555_5556);
        assert_eq!(m.shift, 0);
        // i32 / 7: magic 0x92492493 (negative), shift 2.
        let m = magic_signed(7, 32);
        assert_eq!((m.multiplier as u128) & 0xFFFF_FFFF, 0x9249_2493);
        assert_eq!(m.shift, 2);
        // i32 / -5: magic 0x99999999, shift 1.
        let m = magic_signed(-5, 32);
        assert_eq!((m.multiplier as u128) & 0xFFFF_FFFF, 0x9999_9999);
        assert_eq!(m.shift, 1);
        // u64 / 3: magic 0xAAAAAAAAAAAAAAAB, shift 1.
        let m = magic_unsigned(3, 64);
        assert_eq!(m.multiplier, 0xAAAA_AAAA_AAAA_AAAB);
        assert_eq!(m.shift, 1);
        assert!(!m.add);
    }

    // NOTE: the CRABBIT_MIDEND_DISABLE=divmagic ablation gate is verified
    // end-to-end (the PTX of a gated build keeps a real `div`); an in-process
    // test would have to mutate the process-global environment while the
    // sweep tests run concurrently.
}
