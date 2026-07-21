use std::collections::HashMap;

use pliron::derive::op_interface;

use crate::{
    context::{Context, Ptr},
    ir::{basic_block::BasicBlock, op::Op},
    result::STAIRResult,
};

use super::registers::Register;

pub use super::ops::X86_64Opcode;

#[op_interface]
pub trait X86_64InstructionOpInterface {
    fn opcode(&self) -> X86_64Opcode;

    fn mnemonic(&self) -> &'static str;

    fn verify(_op: &dyn Op, _ctx: &Context) -> STAIRResult<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegisterOperandKind {
    Use,
    Def,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegisterOperand {
    pub key: &'static str,
    pub reg: Register,
    pub kind: RegisterOperandKind,
}

#[op_interface]
pub trait RegisterOperandsOpInterface {
    fn register_operands(&self, ctx: &Context) -> Vec<RegisterOperand>;

    fn rewrite_register_operand(&self, ctx: &mut Context, key: &str, reg: Register);

    fn verify(_op: &dyn Op, _ctx: &Context) -> STAIRResult<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}

/// A target-level fixup. Object formats choose the relocation record used to
/// represent it; an x86-64 instruction must not name Mach-O relocation kinds.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum FixupKind {
    /// A `call`/`jmp` rel32 displacement referencing an external symbol. The
    /// offset points at the 4-byte displacement field, not the instruction
    /// start.
    Branch32,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct BinaryFixup {
    pub offset: u32,
    pub symbol: String,
    pub kind: FixupKind,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BinaryEncoding {
    pub bytes: Vec<u8>,
    pub fixups: Vec<BinaryFixup>,
}

pub struct BinarySerializationContext<'a> {
    pub function_offsets: &'a HashMap<String, u64>,
    pub block_offsets: &'a HashMap<Ptr<BasicBlock>, u64>,
    pub literal_offsets: &'a HashMap<String, u64>,
}

#[op_interface]
pub trait BinarySerializableOpInterface {
    /// Encoded size in bytes. x86-64 instructions are variable-length, so
    /// this must agree exactly with what `encode_binary` produces; encoders
    /// keep lengths independent of branch-target resolution to guarantee it.
    fn byte_len(&self, ctx: &Context) -> u64;

    fn literal(&self, _ctx: &Context) -> Option<(String, Vec<u8>)> {
        None
    }

    fn encode_binary(
        &self,
        ctx: &Context,
        pc: u64,
        refs: &BinarySerializationContext<'_>,
    ) -> STAIRResult<BinaryEncoding>;

    fn verify(_op: &dyn Op, _ctx: &Context) -> STAIRResult<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}
