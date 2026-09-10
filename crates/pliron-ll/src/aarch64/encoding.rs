use std::collections::HashMap;

use crate::{
    common_traits::Named,
    context::{Context, Ptr},
    ir::{basic_block::BasicBlock, operation::Operation},
    result::CrabbitResult,
};

use super::{
    op_interfaces::{
        Aarch64Opcode, BinaryEncoding, BinaryFixup, BinarySerializationContext, FixupKind,
    },
    ops::{self, ATTR_KEY_AARCH64_RD, ATTR_KEY_AARCH64_RM, ATTR_KEY_AARCH64_RN},
    registers::{PhysicalRegister, Register},
};

pub(super) fn literal_for_inst(
    ctx: &Context,
    op: Ptr<Operation>,
    opcode: Aarch64Opcode,
) -> Option<(String, Vec<u8>)> {
    (opcode == ops::AdrLiteralOp::OPCODE)
        .then(|| Some((ops::literal_label(ctx, op)?, ops::literal_bytes(ctx, op)?)))
        .flatten()
}

pub(super) fn encode_inst(
    ctx: &Context,
    op: Ptr<Operation>,
    opcode: Aarch64Opcode,
    mnemonic: &'static str,
    pc: u64,
    refs: &BinarySerializationContext<'_>,
) -> CrabbitResult<BinaryEncoding> {
    let word = match opcode {
        ops::CallOp::OPCODE => match encode_call(ctx, refs.function_offsets, op, pc)? {
            EncodedCall::Local(word) => {
                return Ok(word_encoding(word, None));
            }
            EncodedCall::External(symbol) => {
                return Ok(word_encoding(
                    0x9400_0000,
                    Some(BinaryFixup {
                        offset: pc as u32,
                        symbol,
                        kind: FixupKind::Call26,
                    }),
                ));
            }
        },
        ops::AdrLiteralOp::OPCODE => {
            return encode_adr_literal(ctx, refs.literal_offsets, op, pc);
        }
        ops::AdrFunctionOp::OPCODE => {
            return encode_adr_function(ctx, refs.function_offsets, op, pc);
        }
        // ADRP/ADD address materialization: section addresses are unknown
        // until link time, so the immediates are zero and the deltas travel
        // entirely in the fixups.
        ops::AdrpOp::OPCODE => {
            let rd = xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_ref()).unwrap()) as u32;
            return Ok(word_encoding(
                0x9000_0000 | rd,
                Some(BinaryFixup {
                    offset: pc as u32,
                    symbol: ops::callee(ctx, op).unwrap(),
                    kind: FixupKind::AdrpPage21,
                }),
            ));
        }
        ops::AddLo12Op::OPCODE => {
            let rd = xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_ref()).unwrap()) as u32;
            let rn = xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RN.as_ref()).unwrap()) as u32;
            return Ok(word_encoding(
                0x9100_0000 | (rn << 5) | rd,
                Some(BinaryFixup {
                    offset: pc as u32,
                    symbol: ops::callee(ctx, op).unwrap(),
                    kind: FixupKind::AddLo12,
                }),
            ));
        }
        // Local-exec TLS address halves: the TP-relative offset is a link-time
        // constant, so the immediates are zero and the offsets travel in the
        // fixups (same scheme as ADRP/ADD above).
        ops::AddTprelHi12Op::OPCODE => {
            let rd = xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_ref()).unwrap()) as u32;
            let rn = xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RN.as_ref()).unwrap()) as u32;
            return Ok(word_encoding(
                0x9140_0000 | (rn << 5) | rd,
                Some(BinaryFixup {
                    offset: pc as u32,
                    symbol: ops::callee(ctx, op).unwrap(),
                    kind: FixupKind::TprelHi12,
                }),
            ));
        }
        ops::AddTprelLo12NcOp::OPCODE => {
            let rd = xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_ref()).unwrap()) as u32;
            let rn = xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RN.as_ref()).unwrap()) as u32;
            return Ok(word_encoding(
                0x9100_0000 | (rn << 5) | rd,
                Some(BinaryFixup {
                    offset: pc as u32,
                    symbol: ops::callee(ctx, op).unwrap(),
                    kind: FixupKind::TprelLo12Nc,
                }),
            ));
        }
        ops::BOp::OPCODE => encode_b(ctx, refs.block_offsets, op, pc)?,
        ops::BCondOp::OPCODE => encode_b_cond(ctx, refs.block_offsets, op, pc)?,
        ops::CbnzOp::OPCODE => encode_cbnz(ctx, refs.block_offsets, op, pc)?,
        _ => encode_fixed_inst(ctx, op, opcode).ok_or_else(|| {
            crate::input_error_noloc!(
                "unencodable AArch64 instruction `{}` rd={:?} rn={:?} imm={:?}",
                mnemonic,
                ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_ref()),
                ops::reg(ctx, op, ATTR_KEY_AARCH64_RN.as_ref()),
                ops::imm(ctx, op)
            )
        })?,
    };
    Ok(word_encoding(word, None))
}

fn word_encoding(word: u32, fixup: Option<BinaryFixup>) -> BinaryEncoding {
    BinaryEncoding {
        bytes: word.to_le_bytes().to_vec(),
        fixups: fixup.into_iter().collect(),
    }
}

fn encode_fixed_inst(ctx: &Context, op: Ptr<Operation>, opcode: Aarch64Opcode) -> Option<u32> {
    match opcode {
        ops::MovImmOp::OPCODE => Some(
            0xd280_0000
                | (((ops::imm(ctx, op).unwrap() as u32) & 0xffff) << 5)
                | xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_ref()).unwrap()) as u32,
        ),
        ops::MovzOp::OPCODE => encode_wide_move(ctx, op, 0xd280_0000),
        ops::MovkOp::OPCODE => encode_wide_move(ctx, op, 0xf280_0000),
        ops::MovOp::OPCODE => Some(
            0xaa00_03e0
                | ((xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RM.as_ref()).unwrap()) as u32)
                    << 16)
                | xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_ref()).unwrap()) as u32,
        ),
        ops::AddOp::OPCODE => encode_three_reg(ctx, op, 0x8b00_0000),
        ops::SubOp::OPCODE => encode_three_reg(ctx, op, 0xcb00_0000),
        ops::MulOp::OPCODE => encode_three_reg(ctx, op, 0x9b00_7c00),
        ops::UmulhOp::OPCODE => encode_three_reg(ctx, op, 0x9bc0_7c00),
        ops::SdivOp::OPCODE => encode_three_reg(ctx, op, 0x9ac0_0c00),
        ops::UdivOp::OPCODE => encode_three_reg(ctx, op, 0x9ac0_0800),
        ops::AndOp::OPCODE => encode_three_reg(ctx, op, 0x8a00_0000),
        ops::OrOp::OPCODE => encode_three_reg(ctx, op, 0xaa00_0000),
        ops::XorOp::OPCODE => encode_three_reg(ctx, op, 0xca00_0000),
        ops::ShlOp::OPCODE => encode_three_reg(ctx, op, 0x9ac0_2000),
        ops::LsrOp::OPCODE => encode_three_reg(ctx, op, 0x9ac0_2400),
        ops::AsrOp::OPCODE => encode_three_reg(ctx, op, 0x9ac0_2800),
        // Scalar FP data-processing (FADD/FSUB/FMUL/FDIV, d and s forms).
        ops::FaddDOp::OPCODE => encode_three_freg(ctx, op, 0x1e60_2800),
        ops::FaddSOp::OPCODE => encode_three_freg(ctx, op, 0x1e20_2800),
        ops::FsubDOp::OPCODE => encode_three_freg(ctx, op, 0x1e60_3800),
        ops::FsubSOp::OPCODE => encode_three_freg(ctx, op, 0x1e20_3800),
        ops::FmulDOp::OPCODE => encode_three_freg(ctx, op, 0x1e60_0800),
        ops::FmulSOp::OPCODE => encode_three_freg(ctx, op, 0x1e20_0800),
        ops::FdivDOp::OPCODE => encode_three_freg(ctx, op, 0x1e60_1800),
        ops::FdivSOp::OPCODE => encode_three_freg(ctx, op, 0x1e20_1800),
        ops::FnegDOp::OPCODE => encode_two_reg(ctx, op, 0x1e61_4000, freg, freg),
        ops::FnegSOp::OPCODE => encode_two_reg(ctx, op, 0x1e21_4000, freg, freg),
        // FP 1-source data-processing: fabs, fsqrt and the frint* rounding
        // family (frintm floor, frintp ceil, frintz trunc, frinta round
        // half away from zero, frintn round half to even).
        ops::FabsDOp::OPCODE => encode_two_reg(ctx, op, 0x1e60_c000, freg, freg),
        ops::FabsSOp::OPCODE => encode_two_reg(ctx, op, 0x1e20_c000, freg, freg),
        ops::FsqrtDOp::OPCODE => encode_two_reg(ctx, op, 0x1e61_c000, freg, freg),
        ops::FsqrtSOp::OPCODE => encode_two_reg(ctx, op, 0x1e21_c000, freg, freg),
        ops::FrintmDOp::OPCODE => encode_two_reg(ctx, op, 0x1e65_4000, freg, freg),
        ops::FrintmSOp::OPCODE => encode_two_reg(ctx, op, 0x1e25_4000, freg, freg),
        ops::FrintpDOp::OPCODE => encode_two_reg(ctx, op, 0x1e64_c000, freg, freg),
        ops::FrintpSOp::OPCODE => encode_two_reg(ctx, op, 0x1e24_c000, freg, freg),
        ops::FrintzDOp::OPCODE => encode_two_reg(ctx, op, 0x1e65_c000, freg, freg),
        ops::FrintzSOp::OPCODE => encode_two_reg(ctx, op, 0x1e25_c000, freg, freg),
        ops::FrintaDOp::OPCODE => encode_two_reg(ctx, op, 0x1e66_4000, freg, freg),
        ops::FrintaSOp::OPCODE => encode_two_reg(ctx, op, 0x1e26_4000, freg, freg),
        ops::FrintnDOp::OPCODE => encode_two_reg(ctx, op, 0x1e64_4000, freg, freg),
        ops::FrintnSOp::OPCODE => encode_two_reg(ctx, op, 0x1e24_4000, freg, freg),
        // FP 2-source min/max: the *nm forms return the number when one
        // operand is NaN (IEEE minNum/maxNum), the plain forms propagate it.
        ops::FminnmDOp::OPCODE => encode_three_freg(ctx, op, 0x1e60_7800),
        ops::FminnmSOp::OPCODE => encode_three_freg(ctx, op, 0x1e20_7800),
        ops::FmaxnmDOp::OPCODE => encode_three_freg(ctx, op, 0x1e60_6800),
        ops::FmaxnmSOp::OPCODE => encode_three_freg(ctx, op, 0x1e20_6800),
        ops::FminDOp::OPCODE => encode_three_freg(ctx, op, 0x1e60_5800),
        ops::FminSOp::OPCODE => encode_three_freg(ctx, op, 0x1e20_5800),
        ops::FmaxDOp::OPCODE => encode_three_freg(ctx, op, 0x1e60_4800),
        ops::FmaxSOp::OPCODE => encode_three_freg(ctx, op, 0x1e20_4800),
        // FCMP sets nzcv; rd is hard-wired to zero in the encoding.
        ops::FcmpDOp::OPCODE => encode_fcmp(ctx, op, 0x1e60_2000),
        ops::FcmpSOp::OPCODE => encode_fcmp(ctx, op, 0x1e20_2000),
        // FP precision conversions (FCVT) and int<->FP conversions.
        ops::FcvtDSOp::OPCODE => encode_two_reg(ctx, op, 0x1e22_c000, freg, freg),
        ops::FcvtSDOp::OPCODE => encode_two_reg(ctx, op, 0x1e62_4000, freg, freg),
        ops::ScvtfDXOp::OPCODE => encode_two_reg(ctx, op, 0x9e62_0000, freg, xreg),
        ops::ScvtfSXOp::OPCODE => encode_two_reg(ctx, op, 0x9e22_0000, freg, xreg),
        ops::UcvtfDXOp::OPCODE => encode_two_reg(ctx, op, 0x9e63_0000, freg, xreg),
        ops::UcvtfSXOp::OPCODE => encode_two_reg(ctx, op, 0x9e23_0000, freg, xreg),
        ops::FcvtzsXDOp::OPCODE => encode_two_reg(ctx, op, 0x9e78_0000, xreg, freg),
        ops::FcvtzsWDOp::OPCODE => encode_two_reg(ctx, op, 0x1e78_0000, xreg, freg),
        ops::FcvtzsXSOp::OPCODE => encode_two_reg(ctx, op, 0x9e38_0000, xreg, freg),
        ops::FcvtzsWSOp::OPCODE => encode_two_reg(ctx, op, 0x1e38_0000, xreg, freg),
        ops::FcvtzuXDOp::OPCODE => encode_two_reg(ctx, op, 0x9e79_0000, xreg, freg),
        ops::FcvtzuWDOp::OPCODE => encode_two_reg(ctx, op, 0x1e79_0000, xreg, freg),
        ops::FcvtzuXSOp::OPCODE => encode_two_reg(ctx, op, 0x9e39_0000, xreg, freg),
        ops::FcvtzuWSOp::OPCODE => encode_two_reg(ctx, op, 0x1e39_0000, xreg, freg),
        // FMOV between the register files and within the FP file.
        ops::FmovXDOp::OPCODE => encode_two_reg(ctx, op, 0x9e66_0000, xreg, freg),
        ops::FmovDXOp::OPCODE => encode_two_reg(ctx, op, 0x9e67_0000, freg, xreg),
        ops::FmovWSOp::OPCODE => encode_two_reg(ctx, op, 0x1e26_0000, xreg, freg),
        ops::FmovSWOp::OPCODE => encode_two_reg(ctx, op, 0x1e27_0000, freg, xreg),
        ops::FmovDOp::OPCODE => encode_fmov_rr(ctx, op, 0x1e60_4000),
        ops::FmovSOp::OPCODE => encode_fmov_rr(ctx, op, 0x1e20_4000),
        ops::FmovImmDOp::OPCODE => encode_fmov_imm(ctx, op, 0x1e60_1000),
        ops::FmovImmSOp::OPCODE => encode_fmov_imm(ctx, op, 0x1e20_1000),
        // FP loads/stores mirroring the GPR sp-offset and reg-offset forms.
        ops::StrdSpOffsetOp::OPCODE => encode_fp_mem_offset(ctx, op, 0xfd00_0000, 8, Some(31)),
        ops::LdrdSpOffsetOp::OPCODE => encode_fp_mem_offset(ctx, op, 0xfd40_0000, 8, Some(31)),
        ops::StrsSpOffsetOp::OPCODE => encode_fp_mem_offset(ctx, op, 0xbd00_0000, 4, Some(31)),
        ops::LdrsSpOffsetOp::OPCODE => encode_fp_mem_offset(ctx, op, 0xbd40_0000, 4, Some(31)),
        ops::StrdRegOffsetOp::OPCODE => encode_fp_mem_offset(ctx, op, 0xfd00_0000, 8, None),
        ops::LdrdRegOffsetOp::OPCODE => encode_fp_mem_offset(ctx, op, 0xfd40_0000, 8, None),
        ops::StrsRegOffsetOp::OPCODE => encode_fp_mem_offset(ctx, op, 0xbd00_0000, 4, None),
        ops::LdrsRegOffsetOp::OPCODE => encode_fp_mem_offset(ctx, op, 0xbd40_0000, 4, None),
        ops::CmpOp::OPCODE => encode_cmp(ctx, op),
        ops::StrPreSpOp::OPCODE => encode_str_pre_sp(ctx, op),
        ops::LdrPostSpOp::OPCODE => encode_ldr_post_sp(ctx, op),
        ops::SubSpImmOp::OPCODE => encode_sp_imm(ctx, op, 0xd100_03ff),
        ops::AddSpImmOp::OPCODE => encode_sp_imm(ctx, op, 0x9100_03ff),
        ops::AddSpOffsetOp::OPCODE => encode_add_sp_offset(ctx, op),
        ops::StrSpOffsetOp::OPCODE => encode_mem_offset(ctx, op, 0xf900_0000, 8, Some(31)),
        ops::LdrSpOffsetOp::OPCODE => encode_mem_offset(ctx, op, 0xf940_0000, 8, Some(31)),
        ops::StrwSpOffsetOp::OPCODE => encode_mem_offset(ctx, op, 0xb900_0000, 4, Some(31)),
        ops::LdrwSpOffsetOp::OPCODE => encode_mem_offset(ctx, op, 0xb940_0000, 4, Some(31)),
        ops::StrhSpOffsetOp::OPCODE => encode_mem_offset(ctx, op, 0x7900_0000, 2, Some(31)),
        ops::LdrhSpOffsetOp::OPCODE => encode_mem_offset(ctx, op, 0x7940_0000, 2, Some(31)),
        ops::StrbSpOffsetOp::OPCODE => encode_mem_offset(ctx, op, 0x3900_0000, 1, Some(31)),
        ops::LdrbSpOffsetOp::OPCODE => encode_mem_offset(ctx, op, 0x3940_0000, 1, Some(31)),
        ops::LdrStackArgOp::OPCODE => encode_sp_offset(ctx, op, 0xf940_03e0),
        ops::StrRegOffsetOp::OPCODE => encode_mem_offset(ctx, op, 0xf900_0000, 8, None),
        ops::LdrRegOffsetOp::OPCODE => encode_mem_offset(ctx, op, 0xf940_0000, 8, None),
        ops::StrwRegOffsetOp::OPCODE => encode_mem_offset(ctx, op, 0xb900_0000, 4, None),
        ops::LdrwRegOffsetOp::OPCODE => encode_mem_offset(ctx, op, 0xb940_0000, 4, None),
        ops::StrhRegOffsetOp::OPCODE => encode_mem_offset(ctx, op, 0x7900_0000, 2, None),
        ops::LdrhRegOffsetOp::OPCODE => encode_mem_offset(ctx, op, 0x7940_0000, 2, None),
        ops::StrbRegOffsetOp::OPCODE => encode_mem_offset(ctx, op, 0x3900_0000, 1, None),
        ops::LdrbRegOffsetOp::OPCODE => encode_mem_offset(ctx, op, 0x3940_0000, 1, None),
        ops::SvcOp::OPCODE => encode_svc(ctx, op),
        // mrs rd, tpidr_el0 (system register S3_3_C13_C0_2, the EL0
        // read-write software thread ID register).
        ops::MrsTpidrOp::OPCODE => Some(
            0xd53b_d040 | xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_ref()).unwrap()) as u32,
        ),
        ops::BlrOp::OPCODE => Some(
            0xd63f_0000
                | ((xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RN.as_ref()).unwrap()) as u32)
                    << 5),
        ),
        ops::CsetOp::OPCODE => encode_cset(ctx, op),
        ops::RetOp::OPCODE => Some(0xd65f_03c0),
        ops::BrkOp::OPCODE => Some(0xd420_0000),
        _ => None,
    }
}

fn encode_wide_move(ctx: &Context, op: Ptr<Operation>, base: u32) -> Option<u32> {
    let shift = ops::shift(ctx, op)?;
    if shift > 48 || shift % 16 != 0 {
        return None;
    }
    Some(
        base | (((shift / 16) as u32) << 21)
            | (((ops::imm(ctx, op)? as u32) & 0xffff) << 5)
            | xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_ref()).unwrap()) as u32,
    )
}

/// The hardware number of a physical 64-bit GPR. Encoders only see
/// register-allocated instructions, where every operand is an `x<n>`.
fn xreg(reg: Register) -> u8 {
    match reg {
        Register::Physical(PhysicalRegister::Gpr64(number)) => number,
        other => panic!("expected a physical AArch64 GPR at encoding, got `{other}`"),
    }
}

/// The hardware number of a physical FP register (`d<n>` or `s<n>`; the
/// element size lives in the opcode, not the register number).
fn freg(reg: Register) -> u8 {
    match reg {
        Register::Physical(PhysicalRegister::Fpr64(number))
        | Register::Physical(PhysicalRegister::Fpr32(number)) => number,
        other => panic!("expected a physical AArch64 FP register at encoding, got `{other}`"),
    }
}

fn encode_three_freg(ctx: &Context, op: Ptr<Operation>, base: u32) -> Option<u32> {
    Some(
        base | ((freg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RM.as_ref()).unwrap()) as u32) << 16)
            | ((freg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RN.as_ref()).unwrap()) as u32) << 5)
            | freg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_ref()).unwrap()) as u32,
    )
}

/// An `rd <- rn` instruction whose two operands may live in different
/// register files (conversions, cross-file fmov, fneg, fcvt).
fn encode_two_reg(
    ctx: &Context,
    op: Ptr<Operation>,
    base: u32,
    rd_number: fn(Register) -> u8,
    rn_number: fn(Register) -> u8,
) -> Option<u32> {
    Some(
        base | ((rn_number(ops::reg(ctx, op, ATTR_KEY_AARCH64_RN.as_ref()).unwrap()) as u32) << 5)
            | rd_number(ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_ref()).unwrap()) as u32,
    )
}

fn encode_fcmp(ctx: &Context, op: Ptr<Operation>, base: u32) -> Option<u32> {
    Some(
        base | ((freg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RM.as_ref()).unwrap()) as u32) << 16)
            | ((freg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RN.as_ref()).unwrap()) as u32) << 5),
    )
}

fn encode_fmov_rr(ctx: &Context, op: Ptr<Operation>, base: u32) -> Option<u32> {
    Some(
        base | ((freg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RM.as_ref()).unwrap()) as u32) << 5)
            | freg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_ref()).unwrap()) as u32,
    )
}

fn encode_fmov_imm(ctx: &Context, op: Ptr<Operation>, base: u32) -> Option<u32> {
    let imm8 = ops::imm(ctx, op)?;
    if imm8 > 0xff {
        return None;
    }
    Some(
        base | ((imm8 as u32) << 13)
            | freg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_ref()).unwrap()) as u32,
    )
}

/// The 64-bit value `VFPExpandImm(imm8)` denotes (ARM ARM pseudocode): sign,
/// then `NOT(b6):Replicate(b6, 8):b5:b4` as the 11-bit exponent, then the low
/// four bits as the fraction's top nibble.
pub fn vfp_expand_imm_f64(imm8: u8) -> u64 {
    let imm8 = imm8 as u64;
    let sign = (imm8 >> 7) & 1;
    let b6 = (imm8 >> 6) & 1;
    let exp = ((b6 ^ 1) << 10) | (if b6 == 1 { 0xff << 2 } else { 0 }) | ((imm8 >> 4) & 0x3);
    let frac = imm8 & 0xf;
    (sign << 63) | (exp << 52) | (frac << 48)
}

/// The 32-bit analogue of [vfp_expand_imm_f64]: `NOT(b6):Replicate(b6, 5)`
/// heads an 8-bit exponent.
pub fn vfp_expand_imm_f32(imm8: u8) -> u32 {
    let imm8 = imm8 as u32;
    let sign = (imm8 >> 7) & 1;
    let b6 = (imm8 >> 6) & 1;
    let exp = ((b6 ^ 1) << 7) | (if b6 == 1 { 0x1f << 2 } else { 0 }) | ((imm8 >> 4) & 0x3);
    let frac = imm8 & 0xf;
    (sign << 31) | (exp << 23) | (frac << 19)
}

/// The `imm8` whose VFPExpandImm equals `bits`, if one exists (the FMOV
/// immediate form covers only +/-(16..=31)/16 x 2^(-3..=4)).
pub fn fmov_imm8_for_f64_bits(bits: u64) -> Option<u8> {
    (0..=u8::MAX).find(|imm8| vfp_expand_imm_f64(*imm8) == bits)
}

/// See [fmov_imm8_for_f64_bits]; the f32 form.
pub fn fmov_imm8_for_f32_bits(bits: u32) -> Option<u8> {
    (0..=u8::MAX).find(|imm8| vfp_expand_imm_f32(*imm8) == bits)
}

/// FP loads/stores share the GPR addressing forms but read the data register
/// number from the FP file.
fn encode_fp_mem_offset(
    ctx: &Context,
    op: Ptr<Operation>,
    base: u32,
    scale: u64,
    fixed_rn: Option<u8>,
) -> Option<u32> {
    encode_mem_offset_with(ctx, op, base, scale, fixed_rn, freg)
}

fn encode_svc(ctx: &Context, op: Ptr<Operation>) -> Option<u32> {
    let imm = ops::imm(ctx, op)?;
    if imm > 0xffff {
        return None;
    }
    Some(0xd400_0001 | ((imm as u32) << 5))
}

fn encode_str_pre_sp(ctx: &Context, op: Ptr<Operation>) -> Option<u32> {
    let bytes = ops::imm(ctx, op)?;
    if bytes > 256 || bytes % 8 != 0 {
        return None;
    }
    let imm9 = (512u32 - bytes as u32) & 0x1ff;
    Some(
        0xf800_0c00
            | (imm9 << 12)
            | (31 << 5)
            | xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_ref()).unwrap()) as u32,
    )
}

fn encode_ldr_post_sp(ctx: &Context, op: Ptr<Operation>) -> Option<u32> {
    let bytes = ops::imm(ctx, op)?;
    if bytes > 255 || bytes % 8 != 0 {
        return None;
    }
    Some(
        0xf840_0400
            | ((bytes as u32) << 12)
            | (31 << 5)
            | xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_ref()).unwrap()) as u32,
    )
}

fn encode_sp_imm(ctx: &Context, op: Ptr<Operation>, base: u32) -> Option<u32> {
    let bytes = ops::imm(ctx, op)?;
    if bytes > 4095 {
        return None;
    }
    Some(base | ((bytes as u32) << 10))
}

fn encode_add_sp_offset(ctx: &Context, op: Ptr<Operation>) -> Option<u32> {
    let bytes = ops::imm(ctx, op)?;
    if bytes > 4095 {
        return None;
    }
    Some(
        0x9100_03e0
            | ((bytes as u32) << 10)
            | xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_ref()).unwrap()) as u32,
    )
}

fn encode_sp_offset(ctx: &Context, op: Ptr<Operation>, base: u32) -> Option<u32> {
    let bytes = ops::imm(ctx, op)?;
    if bytes > 32760 || bytes % 8 != 0 {
        return None;
    }
    Some(
        base | (((bytes as u32) / 8) << 10)
            | xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_ref()).unwrap()) as u32,
    )
}

fn encode_mem_offset(
    ctx: &Context,
    op: Ptr<Operation>,
    base: u32,
    scale: u64,
    fixed_rn: Option<u8>,
) -> Option<u32> {
    encode_mem_offset_with(ctx, op, base, scale, fixed_rn, xreg)
}

/// Shared encoder for scaled-immediate (with unscaled `ldur`/`stur`
/// fallback) loads and stores; `data_number` extracts the data register's
/// hardware number from its register file (GPR or FP).
fn encode_mem_offset_with(
    ctx: &Context,
    op: Ptr<Operation>,
    base: u32,
    scale: u64,
    fixed_rn: Option<u8>,
    data_number: fn(Register) -> u8,
) -> Option<u32> {
    let bytes = ops::imm(ctx, op)?;
    let rn = fixed_rn
        .unwrap_or_else(|| xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RN.as_ref()).unwrap()));
    if bytes <= 4095 * scale && bytes % scale == 0 {
        return Some(
            base | (((bytes / scale) as u32) << 10)
                | ((rn as u32) << 5)
                | data_number(ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_ref()).unwrap()) as u32,
        );
    }
    if bytes > 255 {
        return None;
    }
    let unscaled_base = base.checked_sub(0x0100_0000)?;
    Some(
        unscaled_base
            | ((bytes as u32) << 12)
            | ((rn as u32) << 5)
            | data_number(ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_ref()).unwrap()) as u32,
    )
}

fn encode_cmp(ctx: &Context, op: Ptr<Operation>) -> Option<u32> {
    Some(
        0xeb00_001f
            | ((xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RM.as_ref()).unwrap()) as u32)
                << 16)
            | ((xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RN.as_ref()).unwrap()) as u32) << 5),
    )
}

fn encode_cset(ctx: &Context, op: Ptr<Operation>) -> Option<u32> {
    // cset rd, cond is csinc rd, xzr, xzr, invert(cond).
    let cond = ops::cond(ctx, op)?;
    Some(
        0x9a9f_07e0
            | ((cond.invert().encoding() as u32) << 12)
            | xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_ref()).unwrap()) as u32,
    )
}

fn encode_three_reg(ctx: &Context, op: Ptr<Operation>, base: u32) -> Option<u32> {
    Some(
        base | ((xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RM.as_ref()).unwrap()) as u32)
            << 16)
            | ((xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RN.as_ref()).unwrap()) as u32) << 5)
            | xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_ref()).unwrap()) as u32,
    )
}

#[cfg(test)]
mod tests {
    use crate::{
        context::Context,
        dialects::aarch64::{self, ops, registers::Register},
    };

    use std::collections::HashMap;

    use super::{encode_fixed_inst, encode_inst};
    use crate::dialects::aarch64::op_interfaces::{BinarySerializationContext, FixupKind};

    fn context() -> Context {
        let mut ctx = Context::new();
        aarch64::register(&mut ctx);
        ctx
    }

    #[test]
    fn encodes_variable_shift_register_fields() {
        let mut ctx = context();

        let lsr = ops::binary(&mut ctx, ops::LsrOp::OPCODE, Register::gpr(9), Register::gpr(10), Register::gpr(12));
        assert_eq!(
            encode_fixed_inst(&ctx, lsr, ops::LsrOp::OPCODE),
            Some(0x9acc_2549)
        );

        let shl = ops::binary(&mut ctx, ops::ShlOp::OPCODE, Register::gpr(9), Register::gpr(10), Register::gpr(12));
        assert_eq!(
            encode_fixed_inst(&ctx, shl, ops::ShlOp::OPCODE),
            Some(0x9acc_2149)
        );
    }

    /// Every expected word below was produced by the system assembler
    /// (`as` + `objdump -d`) from the textual form in the comment.
    #[test]
    fn encodes_scalar_fp_instructions_to_reference_words() {
        use crate::dialects::aarch64::op_interfaces::Aarch64Opcode;
        let mut ctx = context();
        let d = Register::fpr64;
        let s = Register::fpr32;
        let x = Register::gpr;

        let three: &[(Aarch64Opcode, Register, Register, Register, u32)] = &[
            // fadd d0, d1, d2 ; fadd s0, s1, s2
            (Aarch64Opcode::FaddD, d(0), d(1), d(2), 0x1e62_2820),
            (Aarch64Opcode::FaddS, s(0), s(1), s(2), 0x1e22_2820),
            // fsub d3, d4, d5 ; fsub s3, s4, s5
            (Aarch64Opcode::FsubD, d(3), d(4), d(5), 0x1e65_3883),
            (Aarch64Opcode::FsubS, s(3), s(4), s(5), 0x1e25_3883),
            // fmul d6, d7, d16 ; fmul s6, s7, s16
            (Aarch64Opcode::FmulD, d(6), d(7), d(16), 0x1e70_08e6),
            (Aarch64Opcode::FmulS, s(6), s(7), s(16), 0x1e30_08e6),
            // fdiv d17, d18, d19 ; fdiv s17, s18, s19
            (Aarch64Opcode::FdivD, d(17), d(18), d(19), 0x1e73_1a51),
            (Aarch64Opcode::FdivS, s(17), s(18), s(19), 0x1e33_1a51),
            // fminnm d3, d4, d5 ; fminnm s3, s4, s5
            (Aarch64Opcode::FminnmD, d(3), d(4), d(5), 0x1e65_7883),
            (Aarch64Opcode::FminnmS, s(3), s(4), s(5), 0x1e25_7883),
            // fmaxnm d3, d4, d5 ; fmaxnm s3, s4, s5
            (Aarch64Opcode::FmaxnmD, d(3), d(4), d(5), 0x1e65_6883),
            (Aarch64Opcode::FmaxnmS, s(3), s(4), s(5), 0x1e25_6883),
            // fmin d3, d4, d5 ; fmin s3, s4, s5
            (Aarch64Opcode::FminD, d(3), d(4), d(5), 0x1e65_5883),
            (Aarch64Opcode::FminS, s(3), s(4), s(5), 0x1e25_5883),
            // fmax d3, d4, d5 ; fmax s3, s4, s5
            (Aarch64Opcode::FmaxD, d(3), d(4), d(5), 0x1e65_4883),
            (Aarch64Opcode::FmaxS, s(3), s(4), s(5), 0x1e25_4883),
            // asr x9, x10, x12
            (Aarch64Opcode::Asr, x(9), x(10), x(12), 0x9acc_2949),
        ];
        for (opcode, rd, rn, rm, expected) in three {
            let inst = ops::binary(&mut ctx, *opcode, *rd, *rn, *rm);
            assert_eq!(encode_fixed_inst(&ctx, inst, *opcode), Some(*expected), "{opcode:?}");
        }

        let two: &[(Aarch64Opcode, Register, Register, u32)] = &[
            // fneg d20, d21 ; fneg s20, s21
            (Aarch64Opcode::FnegD, d(20), d(21), 0x1e61_42b4),
            (Aarch64Opcode::FnegS, s(20), s(21), 0x1e21_42b4),
            // fabs d20, d21 ; fabs s20, s21
            (Aarch64Opcode::FabsD, d(20), d(21), 0x1e60_c2b4),
            (Aarch64Opcode::FabsS, s(20), s(21), 0x1e20_c2b4),
            // fsqrt d20, d21 ; fsqrt s20, s21
            (Aarch64Opcode::FsqrtD, d(20), d(21), 0x1e61_c2b4),
            (Aarch64Opcode::FsqrtS, s(20), s(21), 0x1e21_c2b4),
            // frintm d20, d21 ; frintm s20, s21
            (Aarch64Opcode::FrintmD, d(20), d(21), 0x1e65_42b4),
            (Aarch64Opcode::FrintmS, s(20), s(21), 0x1e25_42b4),
            // frintp d20, d21 ; frintp s20, s21
            (Aarch64Opcode::FrintpD, d(20), d(21), 0x1e64_c2b4),
            (Aarch64Opcode::FrintpS, s(20), s(21), 0x1e24_c2b4),
            // frintz d20, d21 ; frintz s20, s21
            (Aarch64Opcode::FrintzD, d(20), d(21), 0x1e65_c2b4),
            (Aarch64Opcode::FrintzS, s(20), s(21), 0x1e25_c2b4),
            // frinta d20, d21 ; frinta s20, s21
            (Aarch64Opcode::FrintaD, d(20), d(21), 0x1e66_42b4),
            (Aarch64Opcode::FrintaS, s(20), s(21), 0x1e26_42b4),
            // frintn d20, d21 ; frintn s20, s21
            (Aarch64Opcode::FrintnD, d(20), d(21), 0x1e64_42b4),
            (Aarch64Opcode::FrintnS, s(20), s(21), 0x1e24_42b4),
            // fcvt d0, s1 ; fcvt s0, d1
            (Aarch64Opcode::FcvtDS, d(0), s(1), 0x1e22_c020),
            (Aarch64Opcode::FcvtSD, s(0), d(1), 0x1e62_4020),
            // scvtf/ucvtf {d0,s0}, x1
            (Aarch64Opcode::ScvtfDX, d(0), x(1), 0x9e62_0020),
            (Aarch64Opcode::ScvtfSX, s(0), x(1), 0x9e22_0020),
            (Aarch64Opcode::UcvtfDX, d(0), x(1), 0x9e63_0020),
            (Aarch64Opcode::UcvtfSX, s(0), x(1), 0x9e23_0020),
            // fcvtzs {x0,w0}, {d1,s1}
            (Aarch64Opcode::FcvtzsXD, x(0), d(1), 0x9e78_0020),
            (Aarch64Opcode::FcvtzsWD, x(0), d(1), 0x1e78_0020),
            (Aarch64Opcode::FcvtzsXS, x(0), s(1), 0x9e38_0020),
            (Aarch64Opcode::FcvtzsWS, x(0), s(1), 0x1e38_0020),
            // fcvtzu {x0,w0}, {d1,s1}
            (Aarch64Opcode::FcvtzuXD, x(0), d(1), 0x9e79_0020),
            (Aarch64Opcode::FcvtzuWD, x(0), d(1), 0x1e79_0020),
            (Aarch64Opcode::FcvtzuXS, x(0), s(1), 0x9e39_0020),
            (Aarch64Opcode::FcvtzuWS, x(0), s(1), 0x1e39_0020),
            // fmov x0, d1 ; fmov d0, x1 ; fmov w0, s1 ; fmov s0, w1
            (Aarch64Opcode::FmovXD, x(0), d(1), 0x9e66_0020),
            (Aarch64Opcode::FmovDX, d(0), x(1), 0x9e67_0020),
            (Aarch64Opcode::FmovWS, x(0), s(1), 0x1e26_0020),
            (Aarch64Opcode::FmovSW, s(0), x(1), 0x1e27_0020),
        ];
        for (opcode, rd, rn, expected) in two {
            let inst = ops::unary(&mut ctx, *opcode, *rd, *rn);
            assert_eq!(encode_fixed_inst(&ctx, inst, *opcode), Some(*expected), "{opcode:?}");
        }

        // fcmp d1, d2 ; fcmp s1, s2
        let fcmp_d = ops::fcmp(&mut ctx, Aarch64Opcode::FcmpD, d(1), d(2));
        assert_eq!(encode_fixed_inst(&ctx, fcmp_d, Aarch64Opcode::FcmpD), Some(0x1e62_2020));
        let fcmp_s = ops::fcmp(&mut ctx, Aarch64Opcode::FcmpS, s(1), s(2));
        assert_eq!(encode_fixed_inst(&ctx, fcmp_s, Aarch64Opcode::FcmpS), Some(0x1e22_2020));

        // fmov d0, d1 ; fmov s0, s1
        let fmov_d = ops::fmov_rr(&mut ctx, Aarch64Opcode::FmovD, d(0), d(1));
        assert_eq!(encode_fixed_inst(&ctx, fmov_d, Aarch64Opcode::FmovD), Some(0x1e60_4020));
        let fmov_s = ops::fmov_rr(&mut ctx, Aarch64Opcode::FmovS, s(0), s(1));
        assert_eq!(encode_fixed_inst(&ctx, fmov_s, Aarch64Opcode::FmovS), Some(0x1e20_4020));

        // fmov d0, #1.5 ; fmov s0, #-2.0
        let imm8_d = super::fmov_imm8_for_f64_bits(1.5f64.to_bits()).unwrap();
        let fmov_imm_d = ops::fmov_imm(&mut ctx, Aarch64Opcode::FmovImmD, d(0), imm8_d as u64);
        assert_eq!(
            encode_fixed_inst(&ctx, fmov_imm_d, Aarch64Opcode::FmovImmD),
            Some(0x1e6f_1000)
        );
        let imm8_s = super::fmov_imm8_for_f32_bits((-2.0f32).to_bits()).unwrap();
        let fmov_imm_s = ops::fmov_imm(&mut ctx, Aarch64Opcode::FmovImmS, s(0), imm8_s as u64);
        assert_eq!(
            encode_fixed_inst(&ctx, fmov_imm_s, Aarch64Opcode::FmovImmS),
            Some(0x1e30_1000)
        );
    }

    #[test]
    fn encodes_fp_memory_forms_to_reference_words() {
        use crate::dialects::aarch64::op_interfaces::Aarch64Opcode;
        let mut ctx = context();
        let d = Register::fpr64;
        let s = Register::fpr32;
        let x = Register::gpr;

        // ldr/str d0, [sp, #16] ; ldr/str s0, [sp, #16]
        let cases_sp: &[(Aarch64Opcode, Register, u64, u32)] = &[
            (Aarch64Opcode::LdrdSpOffset, d(0), 16, 0xfd40_0be0),
            (Aarch64Opcode::StrdSpOffset, d(0), 16, 0xfd00_0be0),
            (Aarch64Opcode::LdrsSpOffset, s(0), 16, 0xbd40_13e0),
            (Aarch64Opcode::StrsSpOffset, s(0), 16, 0xbd00_13e0),
        ];
        for (opcode, rt, offset, expected) in cases_sp {
            let inst = ops::ldr_sp_offset_sized(&mut ctx, *opcode, *rt, *offset);
            assert_eq!(encode_fixed_inst(&ctx, inst, *opcode), Some(*expected), "{opcode:?}");
        }

        // ldr/str d0, [x1, #24] ; ldr/str s0, [x1, #12]
        // ldur/stur {d0,s0}, [x1, #3] (unscaled fallback)
        let cases_reg: &[(Aarch64Opcode, Register, u64, u32)] = &[
            (Aarch64Opcode::LdrdRegOffset, d(0), 24, 0xfd40_0c20),
            (Aarch64Opcode::StrdRegOffset, d(0), 24, 0xfd00_0c20),
            (Aarch64Opcode::LdrsRegOffset, s(0), 12, 0xbd40_0c20),
            (Aarch64Opcode::StrsRegOffset, s(0), 12, 0xbd00_0c20),
            (Aarch64Opcode::LdrdRegOffset, d(0), 3, 0xfc40_3020),
            (Aarch64Opcode::StrdRegOffset, d(0), 3, 0xfc00_3020),
            (Aarch64Opcode::LdrsRegOffset, s(0), 3, 0xbc40_3020),
            (Aarch64Opcode::StrsRegOffset, s(0), 3, 0xbc00_3020),
        ];
        for (opcode, rt, offset, expected) in cases_reg {
            let inst = ops::ldr_reg_offset_sized(&mut ctx, *opcode, *rt, x(1), *offset);
            assert_eq!(encode_fixed_inst(&ctx, inst, *opcode), Some(*expected), "{opcode:?}");
        }
    }

    #[test]
    fn fmov_imm8_expansion_covers_representable_values_only() {
        // 0.0 and NaN are not FMOV-immediate representable.
        assert_eq!(super::fmov_imm8_for_f64_bits(0.0f64.to_bits()), None);
        assert_eq!(super::fmov_imm8_for_f64_bits(f64::NAN.to_bits()), None);
        assert_eq!(super::fmov_imm8_for_f32_bits(0.0f32.to_bits()), None);
        // Every imm8 round-trips through its expansion.
        for imm8 in 0..=u8::MAX {
            assert_eq!(
                super::fmov_imm8_for_f64_bits(super::vfp_expand_imm_f64(imm8)),
                Some(imm8)
            );
            assert_eq!(
                super::fmov_imm8_for_f32_bits(super::vfp_expand_imm_f32(imm8)),
                Some(imm8)
            );
        }
        // Spot values against Rust's own float semantics.
        assert_eq!(
            super::vfp_expand_imm_f64(super::fmov_imm8_for_f64_bits(1.0f64.to_bits()).unwrap()),
            1.0f64.to_bits()
        );
        assert_eq!(
            super::vfp_expand_imm_f32(super::fmov_imm8_for_f32_bits(0.5f32.to_bits()).unwrap()),
            0.5f32.to_bits()
        );
    }

    #[test]
    fn external_call_emits_a_typed_aarch64_fixup() {
        let mut ctx = context();
        let call = ops::call(&mut ctx, "external".try_into().unwrap());
        let refs = BinarySerializationContext {
            function_offsets: &HashMap::new(),
            block_offsets: &HashMap::new(),
            literal_offsets: &HashMap::new(),
        };

        let encoding = encode_inst(
            &ctx,
            call,
            ops::CallOp::OPCODE,
            ops::CallOp::MNEMONIC,
            0,
            &refs,
        )
        .unwrap();
        assert_eq!(encoding.fixups.len(), 1);
        assert_eq!(encoding.fixups[0].kind, FixupKind::Call26);
        assert_eq!(encoding.fixups[0].symbol, "external");
    }
}

enum EncodedCall {
    Local(u32),
    External(String),
}

fn encode_call(
    ctx: &Context,
    offsets: &HashMap<String, u64>,
    op: Ptr<Operation>,
    pc: u64,
) -> CrabbitResult<EncodedCall> {
    let callee = ops::callee(ctx, op).unwrap();
    let Some(target) = offsets.get(&callee) else {
        return Ok(EncodedCall::External(callee));
    };
    let delta_words = ((*target as i64) - (pc as i64)) / 4;
    if !(-(1 << 25)..(1 << 25)).contains(&delta_words) {
        return Err(crate::input_error_noloc!("call target is out of BL range"));
    }
    Ok(EncodedCall::Local(
        0x9400_0000 | ((delta_words as u32) & 0x03ff_ffff),
    ))
}

/// The fixed byte length of an instruction's encoding, used by the layout
/// passes (block offsets, function offsets) before any bytes are emitted.
/// Every instruction is one 4-byte word except the pc-relative address
/// materializations, which are a fixed 12-byte sequence (see
/// [encode_pc_relative_address]).
pub(super) fn byte_len_for(opcode: Aarch64Opcode) -> u64 {
    if opcode == ops::AdrLiteralOp::OPCODE || opcode == ops::AdrFunctionOp::OPCODE {
        12
    } else {
        4
    }
}

/// Materialize `pc + delta` into `rd` position-independently:
/// `adr rd, #0` (the address of this instruction) followed by two 12-bit
/// add-or-subtract immediates. The fixed three-word form reaches +/-16MiB —
/// far beyond ADR's +/-1MiB — and needs no relocations or page alignment,
/// so it works identically in ELF and Mach-O text sections.
fn encode_pc_relative_address(rd: u32, delta: i64, what: &str) -> CrabbitResult<BinaryEncoding> {
    if !(-(1 << 24)..(1 << 24)).contains(&delta) {
        return Err(crate::input_error_noloc!(
            "{what} is out of the 16MiB pc-relative addressing range"
        ));
    }
    let magnitude = delta.unsigned_abs();
    let lo12 = (magnitude & 0xfff) as u32;
    let hi12 = ((magnitude >> 12) & 0xfff) as u32;
    // ADD (immediate) is 0x9100_0000, SUB (immediate) 0xd100_0000; bit 22
    // shifts the immediate left by 12.
    let base = if delta < 0 { 0xd100_0000u32 } else { 0x9100_0000u32 };
    let words = [
        0x1000_0000 | rd,
        base | (lo12 << 10) | (rd << 5) | rd,
        base | (1 << 22) | (hi12 << 10) | (rd << 5) | rd,
    ];
    Ok(BinaryEncoding {
        bytes: words.iter().flat_map(|word| word.to_le_bytes()).collect(),
        fixups: vec![],
    })
}

fn encode_adr_literal(
    ctx: &Context,
    literal_offsets: &HashMap<String, u64>,
    op: Ptr<Operation>,
    pc: u64,
) -> CrabbitResult<BinaryEncoding> {
    let label = ops::literal_label(ctx, op).unwrap();
    let target = literal_offsets
        .get(&label)
        .ok_or_else(|| crate::input_error_noloc!("unknown literal label `{label}`"))?;
    let delta = (*target as i64) - (pc as i64);
    let rd = xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_ref()).unwrap()) as u32;
    encode_pc_relative_address(rd, delta, "literal")
}

fn encode_adr_function(
    ctx: &Context,
    function_offsets: &HashMap<String, u64>,
    op: Ptr<Operation>,
    pc: u64,
) -> CrabbitResult<BinaryEncoding> {
    let symbol = ops::callee(ctx, op).unwrap();
    let target = function_offsets.get(&symbol).ok_or_else(|| {
        crate::input_error_noloc!("adr_function target `{symbol}` is not defined in this module")
    })?;
    let delta = (*target as i64) - (pc as i64);
    let rd = xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_ref()).unwrap()) as u32;
    encode_pc_relative_address(rd, delta, "function address")
}

fn encode_b(
    ctx: &Context,
    block_offsets: &HashMap<Ptr<BasicBlock>, u64>,
    op: Ptr<Operation>,
    pc: u64,
) -> CrabbitResult<u32> {
    let target = branch_target(ctx, block_offsets, op)?;
    let delta_words = ((target as i64) - (pc as i64)) / 4;
    if !(-(1 << 25)..(1 << 25)).contains(&delta_words) {
        return Err(crate::input_error_noloc!("branch target is out of B range"));
    }
    Ok(0x1400_0000 | ((delta_words as u32) & 0x03ff_ffff))
}

fn encode_cbnz(
    ctx: &Context,
    block_offsets: &HashMap<Ptr<BasicBlock>, u64>,
    op: Ptr<Operation>,
    pc: u64,
) -> CrabbitResult<u32> {
    let target = branch_target(ctx, block_offsets, op)?;
    let delta_words = ((target as i64) - (pc as i64)) / 4;
    if !(-(1 << 18)..(1 << 18)).contains(&delta_words) {
        return Err(crate::input_error_noloc!(
            "branch target is out of CBNZ range"
        ));
    }
    let rn = xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RN.as_ref()).unwrap()) as u32;
    Ok(0xb500_0000 | (((delta_words as u32) & 0x7ffff) << 5) | rn)
}

fn encode_b_cond(
    ctx: &Context,
    block_offsets: &HashMap<Ptr<BasicBlock>, u64>,
    op: Ptr<Operation>,
    pc: u64,
) -> CrabbitResult<u32> {
    let target = branch_target(ctx, block_offsets, op)?;
    let delta_words = ((target as i64) - (pc as i64)) / 4;
    if !(-(1 << 18)..(1 << 18)).contains(&delta_words) {
        return Err(crate::input_error_noloc!(
            "branch target is out of B.cond range"
        ));
    }
    let cond = ops::cond(ctx, op).ok_or_else(|| {
        crate::input_error_noloc!("aarch64 `b_cond` is missing its condition code")
    })?;
    Ok(0x5400_0000 | (((delta_words as u32) & 0x7ffff) << 5) | cond.encoding() as u32)
}

fn branch_target(
    ctx: &Context,
    block_offsets: &HashMap<Ptr<BasicBlock>, u64>,
    op: Ptr<Operation>,
) -> CrabbitResult<u64> {
    let target = ops::target(ctx, op)
        .ok_or_else(|| crate::input_error_noloc!("branch has no target block"))?;
    block_offsets.get(&target).copied().ok_or_else(|| {
        crate::input_error_noloc!(
            "branch target `{}` is outside the enclosing function",
            target.deref(ctx).unique_name(ctx)
        )
    })
}
