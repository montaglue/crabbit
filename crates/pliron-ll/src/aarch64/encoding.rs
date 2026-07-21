use std::collections::HashMap;

use crate::{
    common_traits::Named,
    context::{Context, Ptr},
    ir::{basic_block::BasicBlock, operation::Operation},
    result::STAIRResult,
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
) -> STAIRResult<BinaryEncoding> {
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
        ops::AdrLiteralOp::OPCODE => encode_adr_literal(ctx, refs.literal_offsets, op, pc)?,
        ops::AdrFunctionOp::OPCODE => encode_adr_function(ctx, refs.function_offsets, op, pc)?,
        ops::BOp::OPCODE => encode_b(ctx, refs.block_offsets, op, pc)?,
        ops::BCondOp::OPCODE => encode_b_cond(ctx, refs.block_offsets, op, pc)?,
        ops::CbnzOp::OPCODE => encode_cbnz(ctx, refs.block_offsets, op, pc)?,
        _ => encode_fixed_inst(ctx, op, opcode).ok_or_else(|| {
            crate::input_error_noloc!(
                "unencodable AArch64 instruction `{}` rd={:?} rn={:?} imm={:?}",
                mnemonic,
                ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_str()),
                ops::reg(ctx, op, ATTR_KEY_AARCH64_RN.as_str()),
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
                | xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_str()).unwrap()) as u32,
        ),
        ops::MovzOp::OPCODE => encode_wide_move(ctx, op, 0xd280_0000),
        ops::MovkOp::OPCODE => encode_wide_move(ctx, op, 0xf280_0000),
        ops::MovOp::OPCODE => Some(
            0xaa00_03e0
                | ((xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RM.as_str()).unwrap()) as u32)
                    << 16)
                | xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_str()).unwrap()) as u32,
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
        ops::BlrOp::OPCODE => Some(
            0xd63f_0000
                | ((xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RN.as_str()).unwrap()) as u32)
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
            | xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_str()).unwrap()) as u32,
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
            | xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_str()).unwrap()) as u32,
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
            | xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_str()).unwrap()) as u32,
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
            | xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_str()).unwrap()) as u32,
    )
}

fn encode_sp_offset(ctx: &Context, op: Ptr<Operation>, base: u32) -> Option<u32> {
    let bytes = ops::imm(ctx, op)?;
    if bytes > 32760 || bytes % 8 != 0 {
        return None;
    }
    Some(
        base | (((bytes as u32) / 8) << 10)
            | xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_str()).unwrap()) as u32,
    )
}

fn encode_mem_offset(
    ctx: &Context,
    op: Ptr<Operation>,
    base: u32,
    scale: u64,
    fixed_rn: Option<u8>,
) -> Option<u32> {
    let bytes = ops::imm(ctx, op)?;
    let rn = fixed_rn
        .unwrap_or_else(|| xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RN.as_str()).unwrap()));
    if bytes <= 4095 * scale && bytes % scale == 0 {
        return Some(
            base | (((bytes / scale) as u32) << 10)
                | ((rn as u32) << 5)
                | xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_str()).unwrap()) as u32,
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
            | xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_str()).unwrap()) as u32,
    )
}

fn encode_cmp(ctx: &Context, op: Ptr<Operation>) -> Option<u32> {
    Some(
        0xeb00_001f
            | ((xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RM.as_str()).unwrap()) as u32)
                << 16)
            | ((xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RN.as_str()).unwrap()) as u32) << 5),
    )
}

fn encode_cset(ctx: &Context, op: Ptr<Operation>) -> Option<u32> {
    // cset rd, cond is csinc rd, xzr, xzr, invert(cond).
    let cond = ops::cond(ctx, op)?;
    Some(
        0x9a9f_07e0
            | ((cond.invert().encoding() as u32) << 12)
            | xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_str()).unwrap()) as u32,
    )
}

fn encode_three_reg(ctx: &Context, op: Ptr<Operation>, base: u32) -> Option<u32> {
    Some(
        base | ((xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RM.as_str()).unwrap()) as u32)
            << 16)
            | ((xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RN.as_str()).unwrap()) as u32) << 5)
            | xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_str()).unwrap()) as u32,
    )
}

#[cfg(test)]
mod tests {
    use crate::{
        context::Context,
        dialects::{
            aarch64::{self, ops, registers::Register},
            builtin,
        },
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
) -> STAIRResult<EncodedCall> {
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

fn encode_adr_literal(
    ctx: &Context,
    literal_offsets: &HashMap<String, u64>,
    op: Ptr<Operation>,
    pc: u64,
) -> STAIRResult<u32> {
    let label = ops::literal_label(ctx, op).unwrap();
    let target = literal_offsets
        .get(&label)
        .ok_or_else(|| crate::input_error_noloc!("unknown literal label `{label}`"))?;
    let delta = (*target as i64) - (pc as i64);
    if !(-(1 << 20)..(1 << 20)).contains(&delta) {
        return Err(crate::input_error_noloc!("literal is out of ADR range"));
    }
    let imm = delta as u32;
    let immlo = imm & 0x3;
    let immhi = (imm >> 2) & 0x7ffff;
    let rd = xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_str()).unwrap()) as u32;
    Ok(0x1000_0000 | (immlo << 29) | (immhi << 5) | rd)
}

fn encode_adr_function(
    ctx: &Context,
    function_offsets: &HashMap<String, u64>,
    op: Ptr<Operation>,
    pc: u64,
) -> STAIRResult<u32> {
    let symbol = ops::callee(ctx, op).unwrap();
    let target = function_offsets.get(&symbol).ok_or_else(|| {
        crate::input_error_noloc!("adr_function target `{symbol}` is not defined in this module")
    })?;
    let delta = (*target as i64) - (pc as i64);
    if !(-(1 << 20)..(1 << 20)).contains(&delta) {
        return Err(crate::input_error_noloc!("function is out of ADR range"));
    }
    let imm = delta as u32;
    let immlo = imm & 0x3;
    let immhi = (imm >> 2) & 0x7ffff;
    let rd = xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RD.as_str()).unwrap()) as u32;
    Ok(0x1000_0000 | (immlo << 29) | (immhi << 5) | rd)
}

fn encode_b(
    ctx: &Context,
    block_offsets: &HashMap<Ptr<BasicBlock>, u64>,
    op: Ptr<Operation>,
    pc: u64,
) -> STAIRResult<u32> {
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
) -> STAIRResult<u32> {
    let target = branch_target(ctx, block_offsets, op)?;
    let delta_words = ((target as i64) - (pc as i64)) / 4;
    if !(-(1 << 18)..(1 << 18)).contains(&delta_words) {
        return Err(crate::input_error_noloc!(
            "branch target is out of CBNZ range"
        ));
    }
    let rn = xreg(ops::reg(ctx, op, ATTR_KEY_AARCH64_RN.as_str()).unwrap()) as u32;
    Ok(0xb500_0000 | (((delta_words as u32) & 0x7ffff) << 5) | rn)
}

fn encode_b_cond(
    ctx: &Context,
    block_offsets: &HashMap<Ptr<BasicBlock>, u64>,
    op: Ptr<Operation>,
    pc: u64,
) -> STAIRResult<u32> {
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
) -> STAIRResult<u64> {
    let target = ops::target(ctx, op)
        .ok_or_else(|| crate::input_error_noloc!("branch has no target block"))?;
    block_offsets.get(&target).copied().ok_or_else(|| {
        crate::input_error_noloc!(
            "branch target `{}` is outside the enclosing function",
            target.deref(ctx).unique_name(ctx)
        )
    })
}
