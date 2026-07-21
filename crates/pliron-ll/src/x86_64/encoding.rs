//! x86-64 machine-code emission for the attribute-based instruction dialect.
//!
//! The dialect's arithmetic ops are three-address (`rd = rn op rm`); this
//! module expands them into two-address x86 sequences (`mov rd, rn; op rd,
//! rm`), keeps `rax`/`rdx`/`rcx` free for the division, widening-multiply and
//! shift-count expansions, and always emits a REX prefix plus disp32/rel32
//! forms so every instruction's length depends only on its attributes — never
//! on how branch targets resolve. That property is what lets `byte_len`
//! (sizing) and `encode_binary` (emission) agree by construction: both run
//! the same encoder, sizing with unresolved targets treated as offset 0.

use std::collections::HashMap;

use crate::{
    common_traits::Named,
    context::{Context, Ptr},
    ir::operation::Operation,
    result::STAIRResult,
};

use super::{
    op_interfaces::{
        BinaryEncoding, BinaryFixup, BinarySerializationContext, FixupKind, X86_64Opcode,
    },
    ops::{self, ATTR_KEY_X86_64_RD, ATTR_KEY_X86_64_RM, ATTR_KEY_X86_64_RN},
    registers::{PhysicalRegister, Register},
};

// Hardware register numbers used by fixed expansions.
const RAX: u8 = 0;
const RCX: u8 = 1;
const RDX: u8 = 2;
const RSP: u8 = 4;

pub(super) fn literal_for_inst(
    ctx: &Context,
    op: Ptr<Operation>,
    opcode: X86_64Opcode,
) -> Option<(String, Vec<u8>)> {
    (opcode == ops::AdrLiteralOp::OPCODE)
        .then(|| Some((ops::literal_label(ctx, op)?, ops::literal_bytes(ctx, op)?)))
        .flatten()
}

/// Encoded length of `op`. Runs the encoder in sizing mode, where unresolved
/// branch targets read as offset 0; lengths never depend on target values.
pub(super) fn inst_len(
    ctx: &Context,
    op: Ptr<Operation>,
    opcode: X86_64Opcode,
    mnemonic: &'static str,
) -> u64 {
    encode_inst(ctx, op, opcode, mnemonic, 0, None)
        .map(|encoding| encoding.bytes.len() as u64)
        .unwrap_or_else(|error| {
            panic!("cannot size x86-64 instruction `{mnemonic}`: {error}");
        })
}

pub(super) fn encode_inst(
    ctx: &Context,
    op: Ptr<Operation>,
    opcode: X86_64Opcode,
    mnemonic: &'static str,
    pc: u64,
    refs: Option<&BinarySerializationContext<'_>>,
) -> STAIRResult<BinaryEncoding> {
    let mut out = Vec::with_capacity(16);
    let mut fixups = Vec::new();
    encode_into(ctx, op, opcode, mnemonic, pc, refs, &mut out, &mut fixups)?;
    Ok(BinaryEncoding { bytes: out, fixups })
}

#[allow(clippy::too_many_arguments)]
fn encode_into(
    ctx: &Context,
    op: Ptr<Operation>,
    opcode: X86_64Opcode,
    mnemonic: &'static str,
    pc: u64,
    refs: Option<&BinarySerializationContext<'_>>,
    out: &mut Vec<u8>,
    fixups: &mut Vec<BinaryFixup>,
) -> STAIRResult<()> {
    let rd = || parse_gpr(ctx, op, ATTR_KEY_X86_64_RD.as_str(), mnemonic);
    let rn = || parse_gpr(ctx, op, ATTR_KEY_X86_64_RN.as_str(), mnemonic);
    let rm = || parse_gpr(ctx, op, ATTR_KEY_X86_64_RM.as_str(), mnemonic);
    let imm = || {
        ops::imm(ctx, op).ok_or_else(|| {
            crate::input_error_noloc!("x86-64 `{mnemonic}` is missing its immediate")
        })
    };
    let cond = || {
        ops::cond(ctx, op).ok_or_else(|| {
            crate::input_error_noloc!("x86-64 `{mnemonic}` is missing its condition code")
        })
    };

    match opcode {
        ops::MovImmOp::OPCODE => {
            let rd = rd()?;
            out.push(rex(true, false, false, rd >= 8));
            out.push(0xb8 + (rd & 7));
            out.extend_from_slice(&imm()?.to_le_bytes());
        }
        ops::MovOp::OPCODE => emit_mov_rr(out, rd()?, rm()?),
        ops::AddOp::OPCODE => emit_alu_three(out, 0x01, rd()?, rn()?, rm()?, true)?,
        ops::SubOp::OPCODE => emit_alu_three(out, 0x29, rd()?, rn()?, rm()?, false)?,
        ops::AndOp::OPCODE => emit_alu_three(out, 0x21, rd()?, rn()?, rm()?, true)?,
        ops::OrOp::OPCODE => emit_alu_three(out, 0x09, rd()?, rn()?, rm()?, true)?,
        ops::XorOp::OPCODE => emit_alu_three(out, 0x31, rd()?, rn()?, rm()?, true)?,
        ops::MulOp::OPCODE => {
            // Non-widening low multiply via two-operand imul (0f af /r).
            let (rd, rn, rm) = (rd()?, rn()?, rm()?);
            let src = if rd == rn {
                rm
            } else if rd == rm {
                rn
            } else {
                emit_mov_rr(out, rd, rn);
                rm
            };
            out.push(rex(true, rd >= 8, false, src >= 8));
            out.extend_from_slice(&[0x0f, 0xaf, modrm(0b11, rd & 7, src & 7)]);
        }
        ops::UmulhOp::OPCODE => {
            // mov rax, rn; mul rm; mov rd, rdx. RA never hands out rax/rdx.
            let (rd, rn, rm) = (rd()?, rn()?, rm()?);
            check_expansion_operands(mnemonic, &[rd, rn, rm], &[RAX, RDX])?;
            emit_mov_rr(out, RAX, rn);
            out.push(rex(true, false, false, rm >= 8));
            out.extend_from_slice(&[0xf7, modrm(0b11, 4, rm & 7)]);
            emit_mov_rr(out, rd, RDX);
        }
        ops::SdivOp::OPCODE => {
            // mov rax, rn; cqo; idiv rm; mov rd, rax.
            let (rd, rn, rm) = (rd()?, rn()?, rm()?);
            check_expansion_operands(mnemonic, &[rd, rn, rm], &[RAX, RDX])?;
            emit_mov_rr(out, RAX, rn);
            out.extend_from_slice(&[0x48, 0x99]);
            out.push(rex(true, false, false, rm >= 8));
            out.extend_from_slice(&[0xf7, modrm(0b11, 7, rm & 7)]);
            emit_mov_rr(out, rd, RAX);
        }
        ops::UdivOp::OPCODE => {
            // mov rax, rn; xor edx, edx; div rm; mov rd, rax.
            let (rd, rn, rm) = (rd()?, rn()?, rm()?);
            check_expansion_operands(mnemonic, &[rd, rn, rm], &[RAX, RDX])?;
            emit_mov_rr(out, RAX, rn);
            out.extend_from_slice(&[0x31, 0xd2]);
            out.push(rex(true, false, false, rm >= 8));
            out.extend_from_slice(&[0xf7, modrm(0b11, 6, rm & 7)]);
            emit_mov_rr(out, rd, RAX);
        }
        ops::ShlOp::OPCODE | ops::LsrOp::OPCODE => {
            // mov rcx, rm; (mov rd, rn); shift rd, cl. rm is saved into rcx
            // before rd is written, so rd == rm stays correct.
            let (rd, rn, rm) = (rd()?, rn()?, rm()?);
            check_expansion_operands(mnemonic, &[rd, rn, rm], &[RCX])?;
            emit_mov_rr(out, RCX, rm);
            if rd != rn {
                emit_mov_rr(out, rd, rn);
            }
            let ext = if opcode == ops::ShlOp::OPCODE { 4 } else { 5 };
            out.push(rex(true, false, false, rd >= 8));
            out.extend_from_slice(&[0xd3, modrm(0b11, ext, rd & 7)]);
        }
        ops::CmpOp::OPCODE => {
            // cmp rn, rm (39 /r computes rm-field minus reg-field).
            let (rn, rm) = (rn()?, rm()?);
            out.push(rex(true, rm >= 8, false, rn >= 8));
            out.extend_from_slice(&[0x39, modrm(0b11, rm & 7, rn & 7)]);
        }
        ops::CsetOp::OPCODE => {
            // setcc rd8; movzx rd, rd8.
            let rd = rd()?;
            out.push(rex(false, false, false, rd >= 8));
            out.extend_from_slice(&[0x0f, 0x90 + cond()?.encoding(), modrm(0b11, 0, rd & 7)]);
            out.push(rex(true, rd >= 8, false, rd >= 8));
            out.extend_from_slice(&[0x0f, 0xb6, modrm(0b11, rd & 7, rd & 7)]);
        }
        ops::PushOp::OPCODE => {
            let rt = rd()?;
            out.push(rex(false, false, false, rt >= 8));
            out.push(0x50 + (rt & 7));
        }
        ops::PopOp::OPCODE => {
            let rt = rd()?;
            out.push(rex(false, false, false, rt >= 8));
            out.push(0x58 + (rt & 7));
        }
        ops::SubSpImmOp::OPCODE => {
            out.extend_from_slice(&[0x48, 0x81, 0xec]);
            out.extend_from_slice(&(disp32(imm()?, mnemonic)? as u32).to_le_bytes());
        }
        ops::AddSpImmOp::OPCODE => {
            out.extend_from_slice(&[0x48, 0x81, 0xc4]);
            out.extend_from_slice(&(disp32(imm()?, mnemonic)? as u32).to_le_bytes());
        }
        ops::AddSpOffsetOp::OPCODE => {
            // lea rd, [rsp + imm].
            let rd = rd()?;
            out.push(rex(true, rd >= 8, false, false));
            out.push(0x8d);
            emit_mem_operand(out, rd, RSP, disp32(imm()?, mnemonic)?);
        }
        ops::StrSpOffsetOp::OPCODE => emit_store(out, 8, rd()?, RSP, disp32(imm()?, mnemonic)?),
        ops::LdrSpOffsetOp::OPCODE => emit_load(out, 8, rd()?, RSP, disp32(imm()?, mnemonic)?),
        ops::StrwSpOffsetOp::OPCODE => emit_store(out, 4, rd()?, RSP, disp32(imm()?, mnemonic)?),
        ops::LdrwSpOffsetOp::OPCODE => emit_load(out, 4, rd()?, RSP, disp32(imm()?, mnemonic)?),
        ops::StrhSpOffsetOp::OPCODE => emit_store(out, 2, rd()?, RSP, disp32(imm()?, mnemonic)?),
        ops::LdrhSpOffsetOp::OPCODE => emit_load(out, 2, rd()?, RSP, disp32(imm()?, mnemonic)?),
        ops::StrbSpOffsetOp::OPCODE => emit_store(out, 1, rd()?, RSP, disp32(imm()?, mnemonic)?),
        ops::LdrbSpOffsetOp::OPCODE => emit_load(out, 1, rd()?, RSP, disp32(imm()?, mnemonic)?),
        ops::LdrStackArgOp::OPCODE => {
            // Incoming stack arguments sit above the return address pushed by
            // `call`: [rsp + 8 + offset] before the prologue adjusts rsp.
            emit_load(out, 8, rd()?, RSP, disp32(imm()? + 8, mnemonic)?)
        }
        ops::StrRegOffsetOp::OPCODE => emit_store(out, 8, rd()?, rn()?, disp32(imm()?, mnemonic)?),
        ops::LdrRegOffsetOp::OPCODE => emit_load(out, 8, rd()?, rn()?, disp32(imm()?, mnemonic)?),
        ops::StrwRegOffsetOp::OPCODE => emit_store(out, 4, rd()?, rn()?, disp32(imm()?, mnemonic)?),
        ops::LdrwRegOffsetOp::OPCODE => emit_load(out, 4, rd()?, rn()?, disp32(imm()?, mnemonic)?),
        ops::StrhRegOffsetOp::OPCODE => emit_store(out, 2, rd()?, rn()?, disp32(imm()?, mnemonic)?),
        ops::LdrhRegOffsetOp::OPCODE => emit_load(out, 2, rd()?, rn()?, disp32(imm()?, mnemonic)?),
        ops::StrbRegOffsetOp::OPCODE => emit_store(out, 1, rd()?, rn()?, disp32(imm()?, mnemonic)?),
        ops::LdrbRegOffsetOp::OPCODE => emit_load(out, 1, rd()?, rn()?, disp32(imm()?, mnemonic)?),
        ops::AdrLiteralOp::OPCODE => {
            let rd = rd()?;
            let label = ops::literal_label(ctx, op).unwrap();
            let target = resolve(refs.map(|refs| refs.literal_offsets), &label, "literal label")?;
            emit_lea_rip(out, rd, target, pc, mnemonic)?;
        }
        ops::AdrFunctionOp::OPCODE => {
            let rd = rd()?;
            let symbol = ops::callee(ctx, op).unwrap();
            let target = resolve(
                refs.map(|refs| refs.function_offsets),
                &symbol,
                "adr_function target",
            )?;
            emit_lea_rip(out, rd, target, pc, mnemonic)?;
        }
        ops::RetOp::OPCODE => out.push(0xc3),
        ops::Ud2Op::OPCODE => out.extend_from_slice(&[0x0f, 0x0b]),
        ops::CallOp::OPCODE => {
            let callee = ops::callee(ctx, op).unwrap();
            let local = match refs {
                Some(refs) => refs.function_offsets.get(&callee).copied(),
                None => Some(0),
            };
            out.push(0xe8);
            match local {
                Some(target) => {
                    let rel = rel32(target, pc, 5, mnemonic)?;
                    out.extend_from_slice(&rel.to_le_bytes());
                }
                None => {
                    fixups.push(BinaryFixup {
                        offset: pc as u32 + 1,
                        symbol: callee,
                        kind: FixupKind::Branch32,
                    });
                    out.extend_from_slice(&0i32.to_le_bytes());
                }
            }
        }
        ops::CallRegOp::OPCODE => {
            let rn = rn()?;
            out.push(rex(false, false, false, rn >= 8));
            out.extend_from_slice(&[0xff, modrm(0b11, 2, rn & 7)]);
        }
        ops::JmpOp::OPCODE => {
            let target = branch_target(ctx, op, refs)?;
            out.push(0xe9);
            let rel = rel32(target, pc, 5, mnemonic)?;
            out.extend_from_slice(&rel.to_le_bytes());
        }
        ops::JccOp::OPCODE => {
            let target = branch_target(ctx, op, refs)?;
            out.extend_from_slice(&[0x0f, 0x80 + cond()?.encoding()]);
            let rel = rel32(target, pc, 6, mnemonic)?;
            out.extend_from_slice(&rel.to_le_bytes());
        }
        ops::TestJnzOp::OPCODE => {
            // test rn, rn; jnz target.
            let rn = rn()?;
            let target = branch_target(ctx, op, refs)?;
            out.push(rex(true, rn >= 8, false, rn >= 8));
            out.extend_from_slice(&[0x85, modrm(0b11, rn & 7, rn & 7)]);
            out.extend_from_slice(&[0x0f, 0x85]);
            let rel = rel32(target, pc, 9, mnemonic)?;
            out.extend_from_slice(&rel.to_le_bytes());
        }
    }
    Ok(())
}

// Operand and prefix helpers -------------------------------------------------

fn parse_gpr(
    ctx: &Context,
    op: Ptr<Operation>,
    key: &str,
    mnemonic: &'static str,
) -> STAIRResult<u8> {
    let register = ops::reg(ctx, op, key).ok_or_else(|| {
        crate::input_error_noloc!("x86-64 `{mnemonic}` is missing register operand `{key}`")
    })?;
    match register {
        Register::Physical(PhysicalRegister::Gpr64(number)) => Ok(number),
        other => Err(crate::input_error_noloc!(
            "x86-64 `{mnemonic}` operand `{key}` is not a physical 64-bit GPR: `{other}`"
        )),
    }
}

fn rex(w: bool, r: bool, x: bool, b: bool) -> u8 {
    0x40 | ((w as u8) << 3) | ((r as u8) << 2) | ((x as u8) << 1) | (b as u8)
}

fn modrm(mode: u8, reg: u8, rm: u8) -> u8 {
    (mode << 6) | (reg << 3) | rm
}

/// mov rd, rm (register to register, 64-bit).
fn emit_mov_rr(out: &mut Vec<u8>, rd: u8, rm: u8) {
    out.push(rex(true, rm >= 8, false, rd >= 8));
    out.extend_from_slice(&[0x89, modrm(0b11, rm & 7, rd & 7)]);
}

/// `rd = rn op rm` for a two-address ALU opcode (01/09/21/29/31: op r/m64,
/// r64). Non-commutative subtraction with `rd == rm` computes `rm - rn` and
/// negates.
fn emit_alu_three(
    out: &mut Vec<u8>,
    alu_opcode: u8,
    rd: u8,
    rn: u8,
    rm: u8,
    commutative: bool,
) -> STAIRResult<()> {
    let alu = |out: &mut Vec<u8>, dst: u8, src: u8| {
        out.push(rex(true, src >= 8, false, dst >= 8));
        out.extend_from_slice(&[alu_opcode, modrm(0b11, src & 7, dst & 7)]);
    };
    if rd == rn {
        alu(out, rd, rm);
    } else if rd == rm {
        alu(out, rd, rn);
        if !commutative {
            // rd holds rm - rn; negate to get rn - rm.
            out.push(rex(true, false, false, rd >= 8));
            out.extend_from_slice(&[0xf7, modrm(0b11, 3, rd & 7)]);
        }
    } else {
        emit_mov_rr(out, rd, rn);
        alu(out, rd, rm);
    }
    Ok(())
}

fn check_expansion_operands(
    mnemonic: &'static str,
    operands: &[u8],
    reserved: &[u8],
) -> STAIRResult<()> {
    for operand in operands {
        if reserved.contains(operand) {
            return Err(crate::input_error_noloc!(
                "x86-64 `{mnemonic}` expansion clobbers a register its operands live in"
            ));
        }
    }
    Ok(())
}

/// ModRM+SIB+disp32 memory operand `[base + disp]` for register field `reg`.
fn emit_mem_operand(out: &mut Vec<u8>, reg: u8, base: u8, disp: i32) {
    if base & 7 == 4 {
        // rsp/r12 as base require a SIB byte (scale 0, no index).
        out.push(modrm(0b10, reg & 7, 0b100));
        out.push(0x24);
    } else {
        out.push(modrm(0b10, reg & 7, base & 7));
    }
    out.extend_from_slice(&disp.to_le_bytes());
}

/// Store `rt` to `[base + disp]` with the given access size in bytes.
/// 32-bit and narrower stores write only the low bytes.
fn emit_store(out: &mut Vec<u8>, size: u8, rt: u8, base: u8, disp: i32) {
    match size {
        8 => {
            out.push(rex(true, rt >= 8, false, base >= 8));
            out.push(0x89);
        }
        4 => {
            out.push(rex(false, rt >= 8, false, base >= 8));
            out.push(0x89);
        }
        2 => {
            out.push(0x66);
            out.push(rex(false, rt >= 8, false, base >= 8));
            out.push(0x89);
        }
        1 => {
            out.push(rex(false, rt >= 8, false, base >= 8));
            out.push(0x88);
        }
        _ => unreachable!("unsupported store size"),
    }
    emit_mem_operand(out, rt, base, disp);
}

/// Load `[base + disp]` into `rt`, zero-extending to 64 bits.
fn emit_load(out: &mut Vec<u8>, size: u8, rt: u8, base: u8, disp: i32) {
    match size {
        8 => {
            out.push(rex(true, rt >= 8, false, base >= 8));
            out.push(0x8b);
        }
        4 => {
            // mov r32, m32 zero-extends the upper half.
            out.push(rex(false, rt >= 8, false, base >= 8));
            out.push(0x8b);
        }
        2 => {
            out.push(rex(true, rt >= 8, false, base >= 8));
            out.extend_from_slice(&[0x0f, 0xb7]);
        }
        1 => {
            out.push(rex(true, rt >= 8, false, base >= 8));
            out.extend_from_slice(&[0x0f, 0xb6]);
        }
        _ => unreachable!("unsupported load size"),
    }
    emit_mem_operand(out, rt, base, disp);
}

fn emit_lea_rip(
    out: &mut Vec<u8>,
    rd: u8,
    target: u64,
    pc: u64,
    mnemonic: &'static str,
) -> STAIRResult<()> {
    out.push(rex(true, rd >= 8, false, false));
    out.push(0x8d);
    out.push(modrm(0b00, rd & 7, 0b101));
    let rel = rel32(target, pc, 7, mnemonic)?;
    out.extend_from_slice(&rel.to_le_bytes());
    Ok(())
}

fn disp32(value: u64, mnemonic: &'static str) -> STAIRResult<i32> {
    i32::try_from(value)
        .map_err(|_| crate::input_error_noloc!("x86-64 `{mnemonic}` displacement out of range"))
}

fn rel32(target: u64, pc: u64, inst_len: u64, mnemonic: &'static str) -> STAIRResult<i32> {
    let delta = (target as i64) - (pc as i64) - (inst_len as i64);
    i32::try_from(delta)
        .map_err(|_| crate::input_error_noloc!("x86-64 `{mnemonic}` branch target out of range"))
}

fn resolve(
    offsets: Option<&HashMap<String, u64>>,
    key: &str,
    what: &str,
) -> STAIRResult<u64> {
    match offsets {
        None => Ok(0),
        Some(offsets) => offsets
            .get(key)
            .copied()
            .ok_or_else(|| crate::input_error_noloc!("unknown {what} `{key}`")),
    }
}

fn branch_target(
    ctx: &Context,
    op: Ptr<Operation>,
    refs: Option<&BinarySerializationContext<'_>>,
) -> STAIRResult<u64> {
    // Sizing mode (`refs == None`): branch targets read as offset 0; the
    // rel32 forms keep lengths independent of resolution anyway.
    let Some(refs) = refs else {
        return Ok(0);
    };
    let target = ops::target(ctx, op)
        .ok_or_else(|| crate::input_error_noloc!("branch has no target block"))?;
    refs.block_offsets.get(&target).copied().ok_or_else(|| {
        crate::input_error_noloc!(
            "branch target `{}` is outside the enclosing function",
            target.deref(ctx).unique_name(ctx)
        )
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::{
        context::Context,
        dialects::{
            builtin,
            x86_64::{
                self,
                attributes::ConditionCode,
                ops,
                registers::{R11, R12, R13, RAX, RBX},
            },
        },
        ir::basic_block::BasicBlock,
    };

    use super::{encode_inst, inst_len};
    use crate::dialects::x86_64::op_interfaces::{BinarySerializationContext, FixupKind};

    fn context() -> Context {
        let mut ctx = Context::new();
        x86_64::register(&mut ctx);
        ctx
    }

    /// Encode `op` at pc 0, with its branch target (if any) at offset 0.
    fn encode(ctx: &Context, op: crate::context::Ptr<crate::ir::operation::Operation>) -> Vec<u8> {
        let opcode = ops::opcode(ctx, op).unwrap();
        let mnemonic = ops::mnemonic(ctx, op).unwrap();
        let mut block_offsets = HashMap::new();
        if let Some(target) = ops::target(ctx, op) {
            block_offsets.insert(target, 0u64);
        }
        let refs = BinarySerializationContext {
            function_offsets: &HashMap::new(),
            block_offsets: &block_offsets,
            literal_offsets: &HashMap::from([("lit".to_string(), 0u64)]),
        };
        encode_inst(ctx, op, opcode, mnemonic, 0, Some(&refs))
            .unwrap()
            .bytes
    }

    // Expected encodings cross-checked against `clang -arch x86_64` output.
    #[test]
    fn encodes_register_moves_and_alu() {
        let mut ctx = context();

        // mov rax, 42 -> movabs
        let inst = ops::mov_imm(&mut ctx, RAX, 42);
        assert_eq!(
            encode(&ctx, inst),
            [0x48, 0xb8, 42, 0, 0, 0, 0, 0, 0, 0]
        );

        // mov rbx, r12 -> 4c 89 e3
        let inst = ops::mov(&mut ctx, RBX, R12);
        assert_eq!(encode(&ctx, inst), [0x4c, 0x89, 0xe3]);

        // add rbx, rbx, r12 (rd == rn) -> add rbx, r12 = 4c 01 e3
        let inst = ops::binary(&mut ctx, ops::AddOp::OPCODE, RBX, RBX, R12);
        assert_eq!(encode(&ctx, inst), [0x4c, 0x01, 0xe3]);

        // sub r12, rbx, r12 (rd == rm) -> sub r12, rbx; neg r12
        let inst = ops::binary(&mut ctx, ops::SubOp::OPCODE, R12, RBX, R12);
        assert_eq!(
            encode(&ctx, inst),
            [0x49, 0x29, 0xdc, 0x49, 0xf7, 0xdc]
        );

        // add r13, rbx, r12 (all distinct) -> mov r13, rbx; add r13, r12
        let inst = ops::binary(&mut ctx, ops::AddOp::OPCODE, R13, RBX, R12);
        assert_eq!(
            encode(&ctx, inst),
            [0x49, 0x89, 0xdd, 0x4d, 0x01, 0xe5]
        );
    }

    #[test]
    fn encodes_stack_accesses() {
        let mut ctx = context();

        // mov [rsp+16], rbx -> 48 89 5c 24 10 with disp32: 48 89 9c 24 10 00 00 00
        let inst = ops::str_sp_offset(&mut ctx, RBX, 16);
        assert_eq!(
            encode(&ctx, inst),
            [0x48, 0x89, 0x9c, 0x24, 0x10, 0, 0, 0]
        );

        // mov rbx, [rsp+16]
        let inst = ops::ldr_sp_offset(&mut ctx, RBX, 16);
        assert_eq!(
            encode(&ctx, inst),
            [0x48, 0x8b, 0x9c, 0x24, 0x10, 0, 0, 0]
        );

        // movzx rbx, byte [r12+1]: base r12 needs SIB.
        let inst =
            ops::ldr_reg_offset_sized(&mut ctx, ops::LdrbRegOffsetOp::OPCODE, RBX, R12, 1);
        assert_eq!(
            encode(&ctx, inst),
            [0x49, 0x0f, 0xb6, 0x9c, 0x24, 0x01, 0, 0, 0]
        );

        // lea rbx, [rsp+8]
        let inst = ops::add_sp_offset(&mut ctx, RBX, 8);
        assert_eq!(
            encode(&ctx, inst),
            [0x48, 0x8d, 0x9c, 0x24, 0x08, 0, 0, 0]
        );

        // push rbx / pop r12
        let inst = ops::push(&mut ctx, RBX);
        assert_eq!(encode(&ctx, inst), [0x40, 0x53]);
        let inst = ops::pop(&mut ctx, R12);
        assert_eq!(encode(&ctx, inst), [0x41, 0x5c]);

        // sub rsp, 32 / add rsp, 32
        let inst = ops::sub_sp_imm(&mut ctx, 32);
        assert_eq!(encode(&ctx, inst), [0x48, 0x81, 0xec, 32, 0, 0, 0]);
        let inst = ops::add_sp_imm(&mut ctx, 32);
        assert_eq!(encode(&ctx, inst), [0x48, 0x81, 0xc4, 32, 0, 0, 0]);
    }

    #[test]
    fn encodes_control_flow() {
        let mut ctx = context();

        let inst = ops::ret(&mut ctx);
        assert_eq!(encode(&ctx, inst), [0xc3]);

        let target = BasicBlock::new(&mut ctx, Some("target".try_into().unwrap()), vec![]);

        // jmp to offset 0 from pc 0: rel = -5.
        let inst = ops::jmp(&mut ctx, target);
        assert_eq!(encode(&ctx, inst), [0xe9, 0xfb, 0xff, 0xff, 0xff]);

        // je target -> 0f 84 rel32(-6)
        let inst = ops::jcc(&mut ctx, ConditionCode::E, target);
        assert_eq!(
            encode(&ctx, inst),
            [0x0f, 0x84, 0xfa, 0xff, 0xff, 0xff]
        );

        // test rbx, rbx; jnz target
        let inst = ops::test_jnz(&mut ctx, RBX, target);
        assert_eq!(
            encode(&ctx, inst),
            [0x48, 0x85, 0xdb, 0x0f, 0x85, 0xf7, 0xff, 0xff, 0xff]
        );

        // cmp rbx, r12 -> 4c 39 e3
        let inst = ops::cmp(&mut ctx, RBX, R12);
        assert_eq!(encode(&ctx, inst), [0x4c, 0x39, 0xe3]);

        // sete r12b; movzx r12, r12b
        let inst = ops::cset(&mut ctx, R12, ConditionCode::E);
        assert_eq!(
            encode(&ctx, inst),
            [0x41, 0x0f, 0x94, 0xc4, 0x4d, 0x0f, 0xb6, 0xe4]
        );

        // call r11 -> 41 ff d3
        let inst = ops::call_reg(&mut ctx, R11);
        assert_eq!(encode(&ctx, inst), [0x41, 0xff, 0xd3]);
    }

    #[test]
    fn external_call_emits_a_typed_branch32_fixup() {
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
            16,
            Some(&refs),
        )
        .unwrap();
        assert_eq!(encoding.bytes, [0xe8, 0, 0, 0, 0]);
        assert_eq!(encoding.fixups.len(), 1);
        assert_eq!(encoding.fixups[0].kind, FixupKind::Branch32);
        assert_eq!(encoding.fixups[0].symbol, "external");
        // The fixup points at the disp32 field, not the opcode byte.
        assert_eq!(encoding.fixups[0].offset, 17);
    }

    #[test]
    fn sizing_matches_emission() {
        let mut ctx = context();
        let target = BasicBlock::new(&mut ctx, Some("target".try_into().unwrap()), vec![]);
        let insts = [
            ops::mov_imm(&mut ctx, RBX, u64::MAX),
            ops::mov(&mut ctx, RBX, R12),
            ops::binary(&mut ctx, ops::SdivOp::OPCODE, RBX, R12, R13),
            ops::binary(&mut ctx, ops::UdivOp::OPCODE, RBX, R12, R13),
            ops::binary(&mut ctx, ops::UmulhOp::OPCODE, RBX, R12, R13),
            ops::binary(&mut ctx, ops::ShlOp::OPCODE, RBX, R12, R13),
            ops::binary(&mut ctx, ops::MulOp::OPCODE, RBX, R12, R13),
            ops::str_sp_offset(&mut ctx, RBX, 4096),
            ops::ldr_stack_arg(&mut ctx, RBX, 0),
            ops::jmp(&mut ctx, target),
            ops::jcc(&mut ctx, ConditionCode::Ne, target),
            ops::test_jnz(&mut ctx, RBX, target),
            ops::adr_literal(&mut ctx, RBX, "lit", vec![0x00]),
            ops::call(&mut ctx, "somewhere".try_into().unwrap()),
            ops::cset(&mut ctx, RBX, ConditionCode::E),
        ];
        let refs = BinarySerializationContext {
            function_offsets: &HashMap::from([("somewhere".to_string(), 64u64)]),
            block_offsets: &HashMap::from([(target, 128u64)]),
            literal_offsets: &HashMap::from([("lit".to_string(), 256u64)]),
        };
        for inst in insts {
            let opcode = ops::opcode(&ctx, inst).unwrap();
            let mnemonic = ops::mnemonic(&ctx, inst).unwrap();
            let sized = inst_len(&ctx, inst, opcode, mnemonic);
            let emitted = encode_inst(&ctx, inst, opcode, mnemonic, 0, Some(&refs))
                .unwrap()
                .bytes
                .len() as u64;
            assert_eq!(sized, emitted, "length mismatch for {mnemonic}");
        }
    }
}
