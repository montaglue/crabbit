use awint::bw;
use combine::{Parser, attempt, many, many1, optional, satisfy, token};
use pliron::derive::{def_op, derive_op_interface_impl};
use pliron::derive::op_interface_impl;

use crate::ll::BranchWeightsAttr;
use crate::ll::op_interfaces::ATTR_KEY_BRANCH_WEIGHTS;

use crate::{
    common_traits::Named,
    context::{Context, Ptr},
    dialects::builtin::{
        attributes::{IntegerAttr, StringAttr},
        op_interfaces::{
            IsolatedFromAboveInterface, OneRegionInterface,
            SymbolOpInterface, NOpdsInterface, NResultsInterface,
        },
        types::{IntegerType, Signedness},
    },
    dict_key,
    identifier::Identifier,
    impl_verify_succ, input_err, input_error,
    ir::{
        basic_block::BasicBlock,
        irfmt::{
            parsers::{block_opd_parser, delimited_list_parser, int_parser, spaced},
            printers::op::region,
        },
        location::{Located, Location},
        op::{Op, OpObj, op_cast},
        operation::Operation,
        region::Region,
    },
    linked_list::ContainsLinkedList,
    parsable::{Parsable, ParseResult, StateStream},
    printable::{self, Printable},
    result::STAIRResult,
    utils::apint::APInt,
};

use super::{
    attributes::{ConditionCode, ConditionCodeAttr, RegisterAttr},
    encoding,
    op_interfaces::{
        Aarch64InstructionOpInterface, BinaryEncoding, BinarySerializableOpInterface,
        BinarySerializationContext, RegisterOperand, RegisterOperandKind,
        RegisterOperandsOpInterface,
    },
    registers::Register,
};

dict_key!(ATTR_KEY_AARCH64_LINKAGE, "aarch64_linkage");
dict_key!(ATTR_KEY_AARCH64_RD, "aarch64_rd");
dict_key!(ATTR_KEY_AARCH64_RN, "aarch64_rn");
dict_key!(ATTR_KEY_AARCH64_RM, "aarch64_rm");
dict_key!(ATTR_KEY_AARCH64_IMM, "aarch64_imm");
dict_key!(ATTR_KEY_AARCH64_COND, "aarch64_cond");
dict_key!(ATTR_KEY_AARCH64_SHIFT, "aarch64_shift");
dict_key!(ATTR_KEY_AARCH64_CALLEE, "aarch64_callee");
dict_key!(ATTR_KEY_AARCH64_LITERAL_LABEL, "aarch64_literal_label");
dict_key!(ATTR_KEY_AARCH64_LITERAL_BYTES, "aarch64_literal_bytes");
dict_key!(ATTR_KEY_AARCH64_STACK_SIZE, "aarch64_stack_size");

#[def_op("aarch64.func")]
#[derive_op_interface_impl(
    OneRegionInterface,
    SymbolOpInterface,
    IsolatedFromAboveInterface,
    NOpdsInterface<0>,
    NResultsInterface<0>
)]
pub struct FuncOp;

impl FuncOp {
    pub fn new(ctx: &mut Context, name: Identifier, linkage: LinkageAttr) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 1);
        let region = op.deref_mut(ctx).get_region(0);
        let entry = BasicBlock::new(ctx, Some("entry".try_into().unwrap()), vec![]);
        entry.insert_at_front(region, ctx);
        let func = Self { op };
        func.set_symbol_name(ctx, name);
        func.get_operation()
            .deref_mut(ctx)
            .attributes
            .set(ATTR_KEY_AARCH64_LINKAGE.clone(), linkage);
        func
    }

    pub fn linkage(&self, ctx: &Context) -> LinkageAttr {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<LinkageAttr>(&ATTR_KEY_AARCH64_LINKAGE)
            .cloned()
            .unwrap_or_default()
    }

    pub fn entry_block(&self, ctx: &Context) -> Ptr<BasicBlock> {
        self.get_region(ctx).deref(ctx).get_head().unwrap()
    }

    pub fn stack_size(&self, ctx: &Context) -> u64 {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<IntegerAttr>(&ATTR_KEY_AARCH64_STACK_SIZE)
            .map(|attr| attr.value().to_u64())
            .unwrap_or(0)
    }

    pub fn set_stack_size(&self, ctx: &mut Context, bytes: u64) {
        let ty = IntegerType::get(ctx, 64, Signedness::Signless);
        self.get_operation().deref_mut(ctx).attributes.set(
            ATTR_KEY_AARCH64_STACK_SIZE.clone(),
            IntegerAttr::new(ty, APInt::from_u64(bytes, bw(64))),
        );
    }
}

impl Printable for FuncOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(
            f,
            "{} {} @{} ",
            self.get_opid().disp(ctx),
            self.linkage(ctx),
            self.get_symbol_name(ctx)
        )?;
        let stack_size = self.stack_size(ctx);
        if stack_size != 0 {
            write!(f, "stack_size={stack_size} ")?;
        }
        region(self).fmt(ctx, state, f)
    }
}

impl Parsable for FuncOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        if !results.is_empty() {
            input_err!(state_stream.loc(), "aarch64.func produces no results")?
        }

        let op = Operation::new(
            state_stream.state.ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );

        let stack_size = combine::parser::char::string("stack_size")
            .skip(token('='))
            .with(int_parser::<u64>());
        let mut parser = (
            spaced(LinkageAttr::parser(())),
            token('@').with(Identifier::parser(())),
            optional(attempt(spaced(stack_size))),
            spaced(Region::parser(op)),
        );

        parser
            .parse_stream(state_stream)
            .map(|(linkage, name, stack_size, _region)| -> OpObj {
                let ctx = &mut *state_stream.state.ctx;
                let func = FuncOp { op };
                func.set_symbol_name(ctx, name);
                func.get_operation()
                    .deref_mut(ctx)
                    .attributes
                    .set(ATTR_KEY_AARCH64_LINKAGE.clone(), linkage);
                if let Some(bytes) = stack_size {
                    func.set_stack_size(ctx, bytes);
                }
                OpObj::new(func)
            })
            .into()
    }
}

impl_verify_succ!(FuncOp);

pub fn mov_imm(ctx: &mut Context, rd: Register, imm: u64) -> Ptr<Operation> {
    let op = MovImmOp::new(ctx).op;
    set_reg(ctx, op, ATTR_KEY_AARCH64_RD.as_str(), rd);
    set_imm(ctx, op, imm);
    op
}

pub fn movz(ctx: &mut Context, rd: Register, imm: u64, shift: u64) -> Ptr<Operation> {
    let op = MovzOp::new(ctx).op;
    set_reg(ctx, op, ATTR_KEY_AARCH64_RD.as_str(), rd);
    set_imm(ctx, op, imm);
    set_shift(ctx, op, shift);
    op
}

pub fn movk(ctx: &mut Context, rd: Register, imm: u64, shift: u64) -> Ptr<Operation> {
    let op = MovkOp::new(ctx).op;
    set_reg(ctx, op, ATTR_KEY_AARCH64_RD.as_str(), rd);
    set_imm(ctx, op, imm);
    set_shift(ctx, op, shift);
    op
}

pub fn mov(ctx: &mut Context, rd: Register, rm: Register) -> Ptr<Operation> {
    let op = MovOp::new(ctx).op;
    set_reg(ctx, op, ATTR_KEY_AARCH64_RD.as_str(), rd);
    set_reg(ctx, op, ATTR_KEY_AARCH64_RM.as_str(), rm);
    op
}

pub fn binary(
    ctx: &mut Context,
    opcode: Aarch64Opcode,
    rd: Register,
    rn: Register,
    rm: Register,
) -> Ptr<Operation> {
    let op = create_instruction(ctx, opcode);
    set_reg(ctx, op, ATTR_KEY_AARCH64_RD.as_str(), rd);
    set_reg(ctx, op, ATTR_KEY_AARCH64_RN.as_str(), rn);
    set_reg(ctx, op, ATTR_KEY_AARCH64_RM.as_str(), rm);
    op
}

pub fn str_pre_sp(ctx: &mut Context, rt: Register, bytes: u64) -> Ptr<Operation> {
    let op = StrPreSpOp::new(ctx).op;
    set_reg(ctx, op, ATTR_KEY_AARCH64_RD.as_str(), rt);
    set_imm(ctx, op, bytes);
    op
}

pub fn ldr_post_sp(ctx: &mut Context, rt: Register, bytes: u64) -> Ptr<Operation> {
    let op = LdrPostSpOp::new(ctx).op;
    set_reg(ctx, op, ATTR_KEY_AARCH64_RD.as_str(), rt);
    set_imm(ctx, op, bytes);
    op
}

pub fn sub_sp_imm(ctx: &mut Context, bytes: u64) -> Ptr<Operation> {
    let op = SubSpImmOp::new(ctx).op;
    set_imm(ctx, op, bytes);
    op
}

pub fn add_sp_imm(ctx: &mut Context, bytes: u64) -> Ptr<Operation> {
    let op = AddSpImmOp::new(ctx).op;
    set_imm(ctx, op, bytes);
    op
}

pub fn add_sp_offset(ctx: &mut Context, rd: Register, offset: u64) -> Ptr<Operation> {
    let op = AddSpOffsetOp::new(ctx).op;
    set_reg(ctx, op, ATTR_KEY_AARCH64_RD.as_str(), rd);
    set_imm(ctx, op, offset);
    op
}

pub fn str_sp_offset(ctx: &mut Context, rt: Register, offset: u64) -> Ptr<Operation> {
    str_sp_offset_sized(ctx, StrSpOffsetOp::OPCODE, rt, offset)
}

pub fn str_sp_offset_sized(
    ctx: &mut Context,
    opcode: Aarch64Opcode,
    rt: Register,
    offset: u64,
) -> Ptr<Operation> {
    let op = create_instruction(ctx, opcode);
    set_reg(ctx, op, ATTR_KEY_AARCH64_RD.as_str(), rt);
    set_imm(ctx, op, offset);
    op
}

pub fn ldr_sp_offset(ctx: &mut Context, rt: Register, offset: u64) -> Ptr<Operation> {
    ldr_sp_offset_sized(ctx, LdrSpOffsetOp::OPCODE, rt, offset)
}

pub fn ldr_sp_offset_sized(
    ctx: &mut Context,
    opcode: Aarch64Opcode,
    rt: Register,
    offset: u64,
) -> Ptr<Operation> {
    let op = create_instruction(ctx, opcode);
    set_reg(ctx, op, ATTR_KEY_AARCH64_RD.as_str(), rt);
    set_imm(ctx, op, offset);
    op
}

pub fn ldr_stack_arg(ctx: &mut Context, rt: Register, offset: u64) -> Ptr<Operation> {
    let op = LdrStackArgOp::new(ctx).op;
    set_reg(ctx, op, ATTR_KEY_AARCH64_RD.as_str(), rt);
    set_imm(ctx, op, offset);
    op
}

pub fn str_reg_offset(
    ctx: &mut Context,
    rt: Register,
    rn: Register,
    offset: u64,
) -> Ptr<Operation> {
    str_reg_offset_sized(ctx, StrRegOffsetOp::OPCODE, rt, rn, offset)
}

pub fn str_reg_offset_sized(
    ctx: &mut Context,
    opcode: Aarch64Opcode,
    rt: Register,
    rn: Register,
    offset: u64,
) -> Ptr<Operation> {
    let op = create_instruction(ctx, opcode);
    set_reg(ctx, op, ATTR_KEY_AARCH64_RD.as_str(), rt);
    set_reg(ctx, op, ATTR_KEY_AARCH64_RN.as_str(), rn);
    set_imm(ctx, op, offset);
    op
}

pub fn ldr_reg_offset(
    ctx: &mut Context,
    rt: Register,
    rn: Register,
    offset: u64,
) -> Ptr<Operation> {
    ldr_reg_offset_sized(ctx, LdrRegOffsetOp::OPCODE, rt, rn, offset)
}

pub fn ldr_reg_offset_sized(
    ctx: &mut Context,
    opcode: Aarch64Opcode,
    rt: Register,
    rn: Register,
    offset: u64,
) -> Ptr<Operation> {
    let op = create_instruction(ctx, opcode);
    set_reg(ctx, op, ATTR_KEY_AARCH64_RD.as_str(), rt);
    set_reg(ctx, op, ATTR_KEY_AARCH64_RN.as_str(), rn);
    set_imm(ctx, op, offset);
    op
}

pub fn adr_literal(
    ctx: &mut Context,
    rd: Register,
    label: impl Into<String>,
    bytes: Vec<u8>,
) -> Ptr<Operation> {
    let op = AdrLiteralOp::new(ctx).op;
    set_reg(ctx, op, ATTR_KEY_AARCH64_RD.as_str(), rd);
    set_literal_label(ctx, op, label);
    set_literal_bytes(ctx, op, bytes);
    op
}

/// Materialize the address of a function defined in this module (pc-relative
/// `adr` resolved against the function's offset in the text section).
pub fn adr_function(
    ctx: &mut Context,
    rd: Register,
    symbol: Identifier,
) -> Ptr<Operation> {
    let op = AdrFunctionOp::new(ctx).op;
    set_reg(ctx, op, ATTR_KEY_AARCH64_RD.as_str(), rd);
    set_callee(ctx, op, symbol.to_string());
    op
}

pub fn svc(ctx: &mut Context, imm: u64) -> Ptr<Operation> {
    let op = SvcOp::new(ctx).op;
    set_imm(ctx, op, imm);
    op
}

pub fn ret(ctx: &mut Context) -> Ptr<Operation> {
    RetOp::new(ctx).op
}

pub fn brk(ctx: &mut Context) -> Ptr<Operation> {
    BrkOp::new(ctx).op
}

pub fn call(ctx: &mut Context, callee: Identifier) -> Ptr<Operation> {
    let op = CallOp::new(ctx).op;
    set_callee(ctx, op, callee.to_string());
    op
}

/// Indirect call through the function pointer in `rn`.
pub fn blr(ctx: &mut Context, rn: Register) -> Ptr<Operation> {
    let op = BlrOp::new(ctx).op;
    set_reg(ctx, op, ATTR_KEY_AARCH64_RN.as_str(), rn);
    op
}

pub fn b(ctx: &mut Context, target: Ptr<BasicBlock>) -> Ptr<Operation> {
    let op = BOp::new(ctx).op;
    set_target(ctx, op, target);
    op
}

pub fn cbnz(ctx: &mut Context, rn: Register, target: Ptr<BasicBlock>) -> Ptr<Operation> {
    let op = CbnzOp::new(ctx).op;
    set_reg(ctx, op, ATTR_KEY_AARCH64_RN.as_str(), rn);
    set_target(ctx, op, target);
    op
}

pub fn cmp(ctx: &mut Context, rn: Register, rm: Register) -> Ptr<Operation> {
    let op = CmpOp::new(ctx).op;
    set_reg(ctx, op, ATTR_KEY_AARCH64_RN.as_str(), rn);
    set_reg(ctx, op, ATTR_KEY_AARCH64_RM.as_str(), rm);
    op
}

pub fn b_cond(ctx: &mut Context, cond: ConditionCode, target: Ptr<BasicBlock>) -> Ptr<Operation> {
    let op = BCondOp::new(ctx).op;
    set_cond(ctx, op, cond);
    set_target(ctx, op, target);
    op
}

pub fn cset(ctx: &mut Context, rd: Register, cond: ConditionCode) -> Ptr<Operation> {
    let op = CsetOp::new(ctx).op;
    set_reg(ctx, op, ATTR_KEY_AARCH64_RD.as_str(), rd);
    set_cond(ctx, op, cond);
    op
}

pub fn opcode(ctx: &Context, op: Ptr<Operation>) -> Option<Aarch64Opcode> {
    opcode_for_operation_opt(ctx, op)
}

pub fn mnemonic(ctx: &Context, op: Ptr<Operation>) -> Option<&'static str> {
    let operation = Operation::get_op_dyn(op, ctx);
    op_cast::<dyn Aarch64InstructionOpInterface>(&*operation)
        .map(|instruction| instruction.mnemonic())
}

pub fn is_instruction(ctx: &Context, op: Ptr<Operation>) -> bool {
    opcode_for_operation_opt(ctx, op).is_some()
}

pub fn reg(ctx: &Context, op: Ptr<Operation>, key: &str) -> Option<Register> {
    let key: Identifier = key.try_into().unwrap();
    op.deref(ctx)
        .attributes
        .get::<RegisterAttr>(&key)
        .map(|attr| attr.0)
}

pub fn imm(ctx: &Context, op: Ptr<Operation>) -> Option<u64> {
    op.deref(ctx)
        .attributes
        .get::<IntegerAttr>(&ATTR_KEY_AARCH64_IMM)
        .map(|attr| attr.value().to_u64())
}

pub fn cond(ctx: &Context, op: Ptr<Operation>) -> Option<ConditionCode> {
    op.deref(ctx)
        .attributes
        .get::<ConditionCodeAttr>(&ATTR_KEY_AARCH64_COND)
        .map(|attr| attr.0)
}

pub fn shift(ctx: &Context, op: Ptr<Operation>) -> Option<u64> {
    op.deref(ctx)
        .attributes
        .get::<IntegerAttr>(&ATTR_KEY_AARCH64_SHIFT)
        .map(|attr| attr.value().to_u64())
}

pub fn callee(ctx: &Context, op: Ptr<Operation>) -> Option<String> {
    op.deref(ctx)
        .attributes
        .get::<StringAttr>(&ATTR_KEY_AARCH64_CALLEE)
        .map(|attr| -> String { attr.clone().into() })
}

/// A branch's target block: its first (and only) CFG successor.
pub fn target(ctx: &Context, op: Ptr<Operation>) -> Option<Ptr<BasicBlock>> {
    let op_ref = op.deref(ctx);
    (op_ref.get_num_successors() > 0).then(|| op_ref.get_successor(0))
}

pub fn literal_label(ctx: &Context, op: Ptr<Operation>) -> Option<String> {
    op.deref(ctx)
        .attributes
        .get::<StringAttr>(&ATTR_KEY_AARCH64_LITERAL_LABEL)
        .map(|attr| -> String { attr.clone().into() })
}

pub fn literal_bytes(ctx: &Context, op: Ptr<Operation>) -> Option<Vec<u8>> {
    op.deref(ctx)
        .attributes
        .get::<BytesAttr>(&ATTR_KEY_AARCH64_LITERAL_BYTES)
        .map(|attr| attr.0.clone())
}

pub fn set_reg(ctx: &mut Context, op: Ptr<Operation>, key: &str, value: Register) {
    op.deref_mut(ctx)
        .attributes
        .set(key.try_into().unwrap(), RegisterAttr(value));
}

/// Point a branch at `target`, as its first (and only) CFG successor.
pub fn set_target(ctx: &mut Context, op: Ptr<Operation>, target: Ptr<BasicBlock>) {
    if op.deref(ctx).get_num_successors() == 0 {
        Operation::push_successor(op, ctx, target);
    } else {
        Operation::replace_successor(op, ctx, 0, target);
    }
}

pub fn set_callee(ctx: &mut Context, op: Ptr<Operation>, value: impl Into<String>) {
    op.deref_mut(ctx).attributes.set(
        ATTR_KEY_AARCH64_CALLEE.clone(),
        StringAttr::new(value.into()),
    );
}

pub fn set_literal_label(ctx: &mut Context, op: Ptr<Operation>, value: impl Into<String>) {
    op.deref_mut(ctx).attributes.set(
        ATTR_KEY_AARCH64_LITERAL_LABEL.clone(),
        StringAttr::new(value.into()),
    );
}

pub fn set_literal_bytes(ctx: &mut Context, op: Ptr<Operation>, bytes: Vec<u8>) {
    op.deref_mut(ctx)
        .attributes
        .set(ATTR_KEY_AARCH64_LITERAL_BYTES.clone(), BytesAttr(bytes));
}

/// Attach `[taken, not-taken]` branch weights to a conditional branch. This is
/// the machine-level analogue of `MachineBasicBlock` successor probabilities
/// in LLVM: instruction selection transfers IR-level branch probabilities
/// here, and late layout passes (block placement) consume them.
pub fn set_branch_weights(ctx: &mut Context, op: Ptr<Operation>, taken: u32, not_taken: u32) {
    op.deref_mut(ctx).attributes.set(
        ATTR_KEY_BRANCH_WEIGHTS.clone(),
        BranchWeightsAttr(vec![taken, not_taken]),
    );
}

/// Read a conditional branch's `[taken, not-taken]` weights, if attached.
pub fn branch_weights(ctx: &Context, op: Ptr<Operation>) -> Option<(u32, u32)> {
    let op_ref = op.deref(ctx);
    let attr = op_ref
        .attributes
        .get::<BranchWeightsAttr>(&ATTR_KEY_BRANCH_WEIGHTS)?;
    match attr.0.as_slice() {
        [taken, not_taken] => Some((*taken, *not_taken)),
        _ => None,
    }
}

pub fn set_imm(ctx: &mut Context, op: Ptr<Operation>, value: u64) {
    let ty = IntegerType::get(ctx, 64, Signedness::Signless);
    op.deref_mut(ctx).attributes.set(
        ATTR_KEY_AARCH64_IMM.clone(),
        IntegerAttr::new(ty, APInt::from_u64(value, bw(64))),
    );
}

pub fn set_cond(ctx: &mut Context, op: Ptr<Operation>, cond: ConditionCode) {
    op.deref_mut(ctx)
        .attributes
        .set(ATTR_KEY_AARCH64_COND.clone(), ConditionCodeAttr(cond));
}

pub fn set_shift(ctx: &mut Context, op: Ptr<Operation>, value: u64) {
    let ty = IntegerType::get(ctx, 64, Signedness::Signless);
    op.deref_mut(ctx).attributes.set(
        ATTR_KEY_AARCH64_SHIFT.clone(),
        IntegerAttr::new(ty, APInt::from_u64(value, bw(64))),
    );
}

pub fn register_operands(ctx: &Context, op: Ptr<Operation>) -> Vec<RegisterOperand> {
    let operation = Operation::get_op_dyn(op, ctx);
    op_cast::<dyn RegisterOperandsOpInterface>(&*operation)
        .map(|registers| registers.register_operands(ctx))
        .unwrap_or_default()
}

pub fn rewrite_register_operand(ctx: &mut Context, op: Ptr<Operation>, key: &str, reg: Register) {
    let operation = Operation::get_op_dyn(op, ctx);
    if let Some(registers) = op_cast::<dyn RegisterOperandsOpInterface>(&*operation) {
        registers.rewrite_register_operand(ctx, key, reg);
    }
}

fn collect_register_operands(
    ctx: &Context,
    op: Ptr<Operation>,
    specs: &[(&'static str, RegisterOperandKind)],
) -> Vec<RegisterOperand> {
    let mut out = Vec::new();
    for (key, kind) in specs {
        if let Some(reg) = reg(ctx, op, key) {
            out.push(RegisterOperand {
                key,
                reg,
                kind: *kind,
            });
        }
    }
    out
}

fn set_register_for(ctx: &mut Context, op: Ptr<Operation>, key: &str, reg: Register) {
    set_reg(ctx, op, key, reg);
}

/// Print an instruction's full textual payload: its register operands (via
/// [RegisterOperandsOpInterface]) followed by the scalar attributes it may
/// carry, as comma-separated `key=value` fields. This is the exact form
/// [parse_instruction_op] reads back, so the textual IR is a faithful
/// serialization, not just a display aid.
fn print_instruction_fields(
    ctx: &Context,
    op: Ptr<Operation>,
    f: &mut core::fmt::Formatter<'_>,
) -> core::fmt::Result {
    let mut sep = " ";
    let mut printed_registers: Vec<&str> = Vec::new();
    for operand in register_operands(ctx, op) {
        let key = operand.key.strip_prefix("aarch64_").unwrap_or(operand.key);
        // A register both read and written (e.g. movk's rd) is one field.
        if printed_registers.contains(&key) {
            continue;
        }
        printed_registers.push(key);
        write!(f, "{sep}{key}={}", operand.reg)?;
        sep = ", ";
    }

    let mut scalars: Vec<(&str, String)> = Vec::new();
    if let Some(value) = cond(ctx, op) {
        scalars.push(("cond", value.to_string()));
    }
    if let Some(value) = imm(ctx, op) {
        scalars.push(("imm", value.to_string()));
    }
    if let Some(value) = shift(ctx, op) {
        scalars.push(("shift", value.to_string()));
    }
    if let Some(value) = target(ctx, op) {
        scalars.push(("target", format!("^{}", value.deref(ctx).unique_name(ctx))));
    }
    if let Some(value) = callee(ctx, op) {
        scalars.push(("callee", value));
    }
    if let Some(value) = literal_label(ctx, op) {
        scalars.push(("label", value));
    }
    if let Some(value) = literal_bytes(ctx, op) {
        scalars.push(("hex", BytesAttr(value).to_string()));
    }
    if let Some(weights) = op
        .deref(ctx)
        .attributes
        .get::<BranchWeightsAttr>(&ATTR_KEY_BRANCH_WEIGHTS)
    {
        let weights = weights
            .0
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        scalars.push(("weights", format!("[{weights}]")));
    }
    for (key, value) in scalars {
        write!(f, "{sep}{key}={value}")?;
        sep = ", ";
    }
    Ok(())
}

/// A parsed `key=value` field value: a bare token, a `[..]` integer list, or
/// a `^`-prefixed block reference.
enum ParsedInstructionField {
    Text(String),
    List(Vec<u32>),
    Block(Ptr<BasicBlock>),
}

fn instruction_field_value_parser<'a>()
-> impl Parser<StateStream<'a>, Output = ParsedInstructionField> {
    let list = delimited_list_parser('[', ']', ',', int_parser::<u32>())
        .map(ParsedInstructionField::List);
    let block = attempt(block_opd_parser()).map(ParsedInstructionField::Block);
    let text = many1::<String, _, _>(satisfy(|c: char| {
        c.is_alphanumeric() || matches!(c, '_' | '.' | '$' | '+' | '-' | '^')
    }))
    .map(ParsedInstructionField::Text);
    list.or(block).or(text)
}

fn instruction_fields_parser<'a>()
-> impl Parser<StateStream<'a>, Output = Vec<(Identifier, ParsedInstructionField)>> {
    let field = || {
        (
            Identifier::parser(()).skip(token('=')),
            instruction_field_value_parser(),
        )
    };
    let spaces = combine::parser::char::spaces;
    optional(attempt(spaces().with(field())))
        .and(many::<Vec<_>, _, _>(attempt(
            (spaces(), token(','), spaces()).with(field()),
        )))
        .map(|(first, rest)| first.into_iter().chain(rest).collect())
}

fn apply_instruction_field(
    ctx: &mut Context,
    op: Ptr<Operation>,
    loc: &Location,
    key: &Identifier,
    value: ParsedInstructionField,
) -> STAIRResult<()> {
    use ParsedInstructionField::{Block, List, Text};
    let int_value = |text: &str| -> STAIRResult<u64> {
        text.parse().map_err(|_| {
            input_error!(
                loc.clone(),
                "expected an integer for aarch64 field `{key}`, got `{text}`"
            )
        })
    };
    let register_value = |text: &str| -> STAIRResult<Register> {
        Register::parse(text).ok_or_else(|| {
            input_error!(loc.clone(), "invalid AArch64 register `{text}` for `{key}`")
        })
    };
    match (key.as_str(), value) {
        ("rd", Text(text)) => set_reg(ctx, op, ATTR_KEY_AARCH64_RD.as_str(), register_value(&text)?),
        ("rn", Text(text)) => set_reg(ctx, op, ATTR_KEY_AARCH64_RN.as_str(), register_value(&text)?),
        ("rm", Text(text)) => set_reg(ctx, op, ATTR_KEY_AARCH64_RM.as_str(), register_value(&text)?),
        ("imm", Text(text)) => set_imm(ctx, op, int_value(&text)?),
        ("cond", Text(text)) => {
            let cond = ConditionCode::parse(&text).ok_or_else(|| {
                input_error!(
                    loc.clone(),
                    "invalid AArch64 condition code `{text}` for `{key}`"
                )
            })?;
            set_cond(ctx, op, cond);
        }
        ("shift", Text(text)) => set_shift(ctx, op, int_value(&text)?),
        ("target", Block(block)) => set_target(ctx, op, block),
        ("callee", Text(text)) => set_callee(ctx, op, text),
        ("label", Text(text)) => set_literal_label(ctx, op, text),
        ("hex", Text(text)) => {
            let bytes = BytesAttr::parse_str(&text).ok_or_else(|| {
                input_error!(
                    loc.clone(),
                    "invalid bytes literal `{text}` for `{key}`: expected `0x` followed by an even number of hex digits"
                )
            })?;
            set_literal_bytes(ctx, op, bytes.0);
        }
        ("weights", List(weights)) => {
            op.deref_mut(ctx)
                .attributes
                .set(ATTR_KEY_BRANCH_WEIGHTS.clone(), BranchWeightsAttr(weights));
        }
        _ => input_err!(
            loc.clone(),
            "unknown or malformed aarch64 instruction field `{key}`"
        )?,
    }
    Ok(())
}

/// Parse the `key=value` fields following an instruction's opid and apply
/// them to a freshly created operation. Shared by every instruction's
/// [Parsable] impl; the inverse of [print_instruction_fields].
fn parse_instruction_op<'a>(
    state_stream: &mut StateStream<'a>,
    results: &[(Identifier, Location)],
    opid: &str,
    create: fn(&mut Context) -> Ptr<Operation>,
) -> ParseResult<'a, Ptr<Operation>> {
    let loc = state_stream.loc();
    if !results.is_empty() {
        input_err!(loc.clone(), "{opid} produces no results")?
    }

    let parsed: ParseResult<'a, Vec<_>> = instruction_fields_parser()
        .parse_stream(state_stream)
        .into();
    let (fields, commit) = parsed?;

    let ctx = &mut *state_stream.state.ctx;
    let op = create(ctx);
    for (key, value) in fields {
        apply_instruction_field(ctx, op, &loc, &key, value)?;
    }
    Ok((op, commit))
}

// Concrete machine operations. They deliberately share the low-level
// attribute storage helpers below, but their IR identity is the operation ID,
// never a string opcode attribute.
macro_rules! define_aarch64_instruction {
    ($name:ident, $id:literal, $variant:ident, $mnemonic:literal, [$($reg_key:ident => $reg_kind:ident),* $(,)?]) => {
        #[def_op($id)]
        pub struct $name;

        impl $name {
            pub const OPCODE: Aarch64Opcode = Aarch64Opcode::$variant;
            pub const MNEMONIC: &'static str = $mnemonic;

            fn new(ctx: &mut Context) -> Self {
                Self {
                    op: Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0),
                }
            }
        }

        #[op_interface_impl]
        impl Aarch64InstructionOpInterface for $name {
            fn opcode(&self) -> Aarch64Opcode {
                Self::OPCODE
            }

            fn mnemonic(&self) -> &'static str {
                Self::MNEMONIC
            }
        }

        #[op_interface_impl]
        impl RegisterOperandsOpInterface for $name {
            fn register_operands(&self, ctx: &Context) -> Vec<RegisterOperand> {
                collect_register_operands(
                    ctx,
                    self.op,
                    &[$(($reg_key.as_str(), RegisterOperandKind::$reg_kind)),*],
                )
            }
            fn rewrite_register_operand(&self, ctx: &mut Context, key: &str, reg: Register) {
                set_register_for(ctx, self.op, key, reg);
            }
        }

        #[op_interface_impl]
        impl BinarySerializableOpInterface for $name {
            fn literal(&self, ctx: &Context) -> Option<(String, Vec<u8>)> {
                encoding::literal_for_inst(ctx, self.op, Self::OPCODE)
            }
            fn encode_binary(
                &self,
                ctx: &Context,
                pc: u64,
                refs: &BinarySerializationContext<'_>,
            ) -> STAIRResult<BinaryEncoding> {
                encoding::encode_inst(ctx, self.op, Self::OPCODE, Self::MNEMONIC, pc, refs)
            }
        }

        impl Printable for $name {
            fn fmt(
                &self,
                ctx: &Context,
                _state: &printable::State,
                f: &mut core::fmt::Formatter<'_>,
            ) -> core::fmt::Result {
                write!(f, "{}", self.get_opid().disp(ctx))?;
                print_instruction_fields(ctx, self.op, f)
            }
        }

        impl Parsable for $name {
            type Arg = Vec<(Identifier, Location)>;
            type Parsed = OpObj;
            fn parse<'a>(
                state_stream: &mut StateStream<'a>,
                results: Self::Arg,
            ) -> ParseResult<'a, Self::Parsed> {
                let (op, commit) =
                    parse_instruction_op(state_stream, &results, $id, |ctx| $name::new(ctx).op)?;
                Ok((OpObj::new($name { op }), commit))
            }
        }

        impl_verify_succ!($name);
    };
}

// One row per instruction: `Variant => OpType, opid, mnemonic, [register
// operands]`. Generates the `Aarch64Opcode` enum (one variant per row, so
// `create_instruction` is an exhaustive match), every op definition, and
// their registration.
macro_rules! define_aarch64_instructions {
    ($($variant:ident => $name:ident, $id:literal, $mnemonic:literal, [$($reg_key:ident => $reg_kind:ident),* $(,)?]);+ $(;)?) => {
        #[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
        pub enum Aarch64Opcode {
            $($variant),+
        }

        impl Aarch64Opcode {
            pub const ALL: &'static [Aarch64Opcode] = &[$(Aarch64Opcode::$variant),+];
        }

        fn create_instruction(ctx: &mut Context, opcode: Aarch64Opcode) -> Ptr<Operation> {
            match opcode {
                $(Aarch64Opcode::$variant => $name::new(ctx).op),+
            }
        }

        fn register_instructions(ctx: &mut Context) {
            
        }

        $(define_aarch64_instruction!($name, $id, $variant, $mnemonic, [$($reg_key => $reg_kind),*]);)+
    };
}

define_aarch64_instructions! {
    MovImm => MovImmOp, "aarch64.mov_imm", "mov_imm", [ATTR_KEY_AARCH64_RD => Def];
    Movz => MovzOp, "aarch64.movz", "movz", [ATTR_KEY_AARCH64_RD => Def];
    Movk => MovkOp, "aarch64.movk", "movk", [ATTR_KEY_AARCH64_RD => Use, ATTR_KEY_AARCH64_RD => Def];
    Mov => MovOp, "aarch64.mov", "mov", [ATTR_KEY_AARCH64_RD => Def, ATTR_KEY_AARCH64_RM => Use];
    Add => AddOp, "aarch64.add", "add", [ATTR_KEY_AARCH64_RD => Def, ATTR_KEY_AARCH64_RN => Use, ATTR_KEY_AARCH64_RM => Use];
    Sub => SubOp, "aarch64.sub", "sub", [ATTR_KEY_AARCH64_RD => Def, ATTR_KEY_AARCH64_RN => Use, ATTR_KEY_AARCH64_RM => Use];
    Mul => MulOp, "aarch64.mul", "mul", [ATTR_KEY_AARCH64_RD => Def, ATTR_KEY_AARCH64_RN => Use, ATTR_KEY_AARCH64_RM => Use];
    Umulh => UmulhOp, "aarch64.umulh", "umulh", [ATTR_KEY_AARCH64_RD => Def, ATTR_KEY_AARCH64_RN => Use, ATTR_KEY_AARCH64_RM => Use];
    Sdiv => SdivOp, "aarch64.sdiv", "sdiv", [ATTR_KEY_AARCH64_RD => Def, ATTR_KEY_AARCH64_RN => Use, ATTR_KEY_AARCH64_RM => Use];
    Udiv => UdivOp, "aarch64.udiv", "udiv", [ATTR_KEY_AARCH64_RD => Def, ATTR_KEY_AARCH64_RN => Use, ATTR_KEY_AARCH64_RM => Use];
    And => AndOp, "aarch64.and", "and", [ATTR_KEY_AARCH64_RD => Def, ATTR_KEY_AARCH64_RN => Use, ATTR_KEY_AARCH64_RM => Use];
    Or => OrOp, "aarch64.or", "or", [ATTR_KEY_AARCH64_RD => Def, ATTR_KEY_AARCH64_RN => Use, ATTR_KEY_AARCH64_RM => Use];
    Xor => XorOp, "aarch64.xor", "xor", [ATTR_KEY_AARCH64_RD => Def, ATTR_KEY_AARCH64_RN => Use, ATTR_KEY_AARCH64_RM => Use];
    Shl => ShlOp, "aarch64.shl", "shl", [ATTR_KEY_AARCH64_RD => Def, ATTR_KEY_AARCH64_RN => Use, ATTR_KEY_AARCH64_RM => Use];
    Lsr => LsrOp, "aarch64.lsr", "lsr", [ATTR_KEY_AARCH64_RD => Def, ATTR_KEY_AARCH64_RN => Use, ATTR_KEY_AARCH64_RM => Use];
    Cmp => CmpOp, "aarch64.cmp", "cmp", [ATTR_KEY_AARCH64_RN => Use, ATTR_KEY_AARCH64_RM => Use];
    StrPreSp => StrPreSpOp, "aarch64.str_pre_sp", "str_pre_sp", [ATTR_KEY_AARCH64_RD => Use];
    LdrPostSp => LdrPostSpOp, "aarch64.ldr_post_sp", "ldr_post_sp", [ATTR_KEY_AARCH64_RD => Def];
    SubSpImm => SubSpImmOp, "aarch64.sub_sp_imm", "sub_sp_imm", [];
    AddSpImm => AddSpImmOp, "aarch64.add_sp_imm", "add_sp_imm", [];
    AddSpOffset => AddSpOffsetOp, "aarch64.add_sp_offset", "add_sp_offset", [ATTR_KEY_AARCH64_RD => Def];
    StrSpOffset => StrSpOffsetOp, "aarch64.str_sp_offset", "str_sp_offset", [ATTR_KEY_AARCH64_RD => Use];
    LdrSpOffset => LdrSpOffsetOp, "aarch64.ldr_sp_offset", "ldr_sp_offset", [ATTR_KEY_AARCH64_RD => Def];
    StrwSpOffset => StrwSpOffsetOp, "aarch64.strw_sp_offset", "strw_sp_offset", [ATTR_KEY_AARCH64_RD => Use];
    LdrwSpOffset => LdrwSpOffsetOp, "aarch64.ldrw_sp_offset", "ldrw_sp_offset", [ATTR_KEY_AARCH64_RD => Def];
    StrhSpOffset => StrhSpOffsetOp, "aarch64.strh_sp_offset", "strh_sp_offset", [ATTR_KEY_AARCH64_RD => Use];
    LdrhSpOffset => LdrhSpOffsetOp, "aarch64.ldrh_sp_offset", "ldrh_sp_offset", [ATTR_KEY_AARCH64_RD => Def];
    StrbSpOffset => StrbSpOffsetOp, "aarch64.strb_sp_offset", "strb_sp_offset", [ATTR_KEY_AARCH64_RD => Use];
    LdrbSpOffset => LdrbSpOffsetOp, "aarch64.ldrb_sp_offset", "ldrb_sp_offset", [ATTR_KEY_AARCH64_RD => Def];
    LdrStackArg => LdrStackArgOp, "aarch64.ldr_stack_arg", "ldr_stack_arg", [ATTR_KEY_AARCH64_RD => Def];
    StrRegOffset => StrRegOffsetOp, "aarch64.str_reg_offset", "str_reg_offset", [ATTR_KEY_AARCH64_RD => Use, ATTR_KEY_AARCH64_RN => Use];
    LdrRegOffset => LdrRegOffsetOp, "aarch64.ldr_reg_offset", "ldr_reg_offset", [ATTR_KEY_AARCH64_RD => Def, ATTR_KEY_AARCH64_RN => Use];
    StrwRegOffset => StrwRegOffsetOp, "aarch64.strw_reg_offset", "strw_reg_offset", [ATTR_KEY_AARCH64_RD => Use, ATTR_KEY_AARCH64_RN => Use];
    LdrwRegOffset => LdrwRegOffsetOp, "aarch64.ldrw_reg_offset", "ldrw_reg_offset", [ATTR_KEY_AARCH64_RD => Def, ATTR_KEY_AARCH64_RN => Use];
    StrhRegOffset => StrhRegOffsetOp, "aarch64.strh_reg_offset", "strh_reg_offset", [ATTR_KEY_AARCH64_RD => Use, ATTR_KEY_AARCH64_RN => Use];
    LdrhRegOffset => LdrhRegOffsetOp, "aarch64.ldrh_reg_offset", "ldrh_reg_offset", [ATTR_KEY_AARCH64_RD => Def, ATTR_KEY_AARCH64_RN => Use];
    StrbRegOffset => StrbRegOffsetOp, "aarch64.strb_reg_offset", "strb_reg_offset", [ATTR_KEY_AARCH64_RD => Use, ATTR_KEY_AARCH64_RN => Use];
    LdrbRegOffset => LdrbRegOffsetOp, "aarch64.ldrb_reg_offset", "ldrb_reg_offset", [ATTR_KEY_AARCH64_RD => Def, ATTR_KEY_AARCH64_RN => Use];
    AdrLiteral => AdrLiteralOp, "aarch64.adr_literal", "adr_literal", [ATTR_KEY_AARCH64_RD => Def];
    Svc => SvcOp, "aarch64.svc", "svc", [];
    Cset => CsetOp, "aarch64.cset", "cset", [ATTR_KEY_AARCH64_RD => Def];
    Ret => RetOp, "aarch64.ret", "ret", [];
    Brk => BrkOp, "aarch64.brk", "brk", [];
    Call => CallOp, "aarch64.call", "call", [];
    B => BOp, "aarch64.b", "b", [];
    BCond => BCondOp, "aarch64.b_cond", "b_cond", [];
    Cbnz => CbnzOp, "aarch64.cbnz", "cbnz", [ATTR_KEY_AARCH64_RN => Use];
    AdrFunction => AdrFunctionOp, "aarch64.adr_function", "adr_function", [ATTR_KEY_AARCH64_RD => Def];
    Blr => BlrOp, "aarch64.blr", "blr", [ATTR_KEY_AARCH64_RN => Use];
}

fn opcode_for_operation_opt(ctx: &Context, op: Ptr<Operation>) -> Option<Aarch64Opcode> {
    let operation = Operation::get_op_dyn(op, ctx);
    crate::ir::op::op_cast::<dyn Aarch64InstructionOpInterface>(&*operation)
        .map(|instruction| instruction.opcode())
}

pub fn register(ctx: &mut Context) {
    register_instructions(ctx);
}

#[cfg(test)]
mod tests {
    fn plain(ctx: &Context, op: Ptr<Operation>) -> String {
        let state = crate::printable::State::default();
        op.print(ctx, &state).to_string()
    }

    use combine::Parser;

    use crate::{
        context::{Context, Ptr},
        dialects::aarch64,
        ir::{
            location,
            op::Op,
            operation::{Operation, OperationParserConfig},
        },
        linked_list::ContainsLinkedList,
        parsable::{Parsable, State, state_stream_from_iterator},
        printable::Printable,
    };

    use super::*;

    fn context() -> Context {
        let mut ctx = Context::new();
        aarch64::register(&mut ctx);
        ctx
    }

    fn parse_op_text(ctx: &mut Context, text: &str) -> Ptr<Operation> {
        let state_stream = state_stream_from_iterator(
            text.chars(),
            State::new(ctx, location::Source::InMemory),
        );
        <Operation as Parsable>::parser(OperationParserConfig {
            look_for_outlined_attrs: false,
        })
        .parse(state_stream)
        .unwrap_or_else(|err| panic!("failed to parse `{text}`: {err}"))
        .0
    }

    /// The printed form must parse back into an operation that prints
    /// byte-identically: the textual IR is a serialization format.
    fn assert_print_parse_one_to_one(ctx: &mut Context, op: Ptr<Operation>) {
        let printed = plain(&ctx, op);
        let parsed = parse_op_text(ctx, &printed);
        assert!(Operation::get_opid(parsed, ctx) == Operation::get_opid(op, ctx));
        // The freshly built `op` has no source location; the reparsed `parsed`
        // does (it was just parsed from text). Normalize so the comparison
        // covers instruction fields, not this parse-vs-build asymmetry.
        parsed.deref_mut(ctx).set_loc(location::Location::Unknown);
        let reprinted = plain(&ctx, parsed);
        assert_eq!(printed, reprinted);
    }

    #[test]
    fn factory_emits_a_concrete_instruction_operation() {
        let mut ctx = context();

        let inst = binary(&mut ctx, AddOp::OPCODE, Register::virtual_gpr(0), Register::virtual_gpr(1), Register::virtual_gpr(2));
        assert!(Operation::get_opid(inst, &ctx) == AddOp::get_opid_static());
        assert_ne!(Operation::get_opid(inst, &ctx).to_string(), "aarch64.inst");
    }

    #[test]
    fn prints_register_operands_and_scalar_attributes() {
        let mut ctx = context();

        let inst = binary(&mut ctx, AddOp::OPCODE, Register::gpr(0), Register::gpr(1), Register::virtual_gpr(7));
        assert_eq!(
            plain(&ctx, inst),
            "aarch64.add rd=x0, rn=x1, rm=vr7"
        );

        let inst = movz(&mut ctx, Register::gpr(0), 42, 16);
        assert_eq!(
            plain(&ctx, inst),
            "aarch64.movz rd=x0, imm=42, shift=16"
        );

        // movk reads and writes rd; the field is printed once.
        let inst = movk(&mut ctx, Register::gpr(3), 7, 32);
        assert_eq!(
            plain(&ctx, inst),
            "aarch64.movk rd=x3, imm=7, shift=32"
        );

        let then_block = BasicBlock::new(&mut ctx, Some("then".try_into().unwrap()), vec![]);
        let then_label = then_block.deref(&ctx).unique_name(&ctx);
        let inst = b_cond(&mut ctx, ConditionCode::Ne, then_block);
        set_branch_weights(&mut ctx, inst, 2000, 1);
        assert_eq!(
            plain(&ctx, inst),
            format!("aarch64.b_cond cond=ne, target=^{then_label}, weights=[2000, 1]")
        );

        let inst = adr_literal(
            &mut ctx,
            Register::gpr(8),
            "lCPI0_0",
            vec![0x3f, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        );
        assert_eq!(
            plain(&ctx, inst),
            "aarch64.adr_literal rd=x8, label=lCPI0_0, hex=0x3ff0000000000000"
        );

        let inst = call(&mut ctx, "_main".try_into().unwrap());
        assert_eq!(plain(&ctx, inst), "aarch64.call callee=_main");

        let inst = ret(&mut ctx);
        assert_eq!(plain(&ctx, inst), "aarch64.ret");
    }

    #[test]
    fn every_instruction_round_trips_fully_populated() {
        let mut ctx = context();

        for &opcode in Aarch64Opcode::ALL {
            let op = create_instruction(&mut ctx, opcode);
            assert_eq!(opcode_for_operation_opt(&ctx, op), Some(opcode));

            // Populate every attribute an instruction can carry. Register
            // fields print only when the instruction declares them; the
            // scalar fields print whenever present. Branch targets are CFG
            // successors, not attributes; they only resolve inside a region
            // and round-trip in `branch_targets_round_trip_inside_a_func`.
            set_reg(&mut ctx, op, ATTR_KEY_AARCH64_RD.as_str(), Register::gpr(0));
            set_reg(&mut ctx, op, ATTR_KEY_AARCH64_RN.as_str(), super::super::registers::SP);
            set_reg(&mut ctx, op, ATTR_KEY_AARCH64_RM.as_str(), Register::virtual_gpr(7));
            set_imm(&mut ctx, op, 42);
            set_cond(&mut ctx, op, ConditionCode::Ls);
            set_shift(&mut ctx, op, 16);
            set_callee(&mut ctx, op, "_callee");
            set_literal_label(&mut ctx, op, "lCPI0_0");
            set_literal_bytes(
                &mut ctx,
                op,
                vec![0x3f, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            );
            set_branch_weights(&mut ctx, op, 1, 2000);

            assert_print_parse_one_to_one(&mut ctx, op);
        }
    }

    #[test]
    fn every_instruction_round_trips_bare() {
        let mut ctx = context();

        for &opcode in Aarch64Opcode::ALL {
            let op = create_instruction(&mut ctx, opcode);
            assert_print_parse_one_to_one(&mut ctx, op);
        }
    }

    #[test]
    fn func_round_trips_through_text() {
        let mut ctx = context();

        let func = FuncOp::new(&mut ctx, "main".try_into().unwrap(), LinkageAttr::External);
        func.set_stack_size(&mut ctx, 32);
        let entry = func.entry_block(&ctx);
        mov_imm(&mut ctx, Register::gpr(0), 7).insert_at_back(entry, &ctx);
        ret(&mut ctx).insert_at_back(entry, &ctx);

        let printed = plain(&ctx, func.get_operation());
        assert!(
            printed.contains("aarch64.func external @main stack_size=32 "),
            "printed form was: {printed}"
        );

        let parsed = FuncOp {
            op: parse_op_text(&mut ctx, &printed),
        };
        assert_eq!(parsed.get_symbol_name(&ctx).to_string(), "main");
        assert_eq!(parsed.linkage(&ctx), LinkageAttr::External);
        assert_eq!(parsed.stack_size(&ctx), 32);

        // Block labels embed fresh arena ids on reparse, so compare the
        // instruction sequence instead of raw text.
        let ops: Vec<String> = parsed
            .entry_block(&ctx)
            .deref(&ctx)
            .iter(&ctx)
            .map(|op| Operation::get_opid(op, &ctx).to_string())
            .collect();
        assert_eq!(ops, ["aarch64.mov_imm", "aarch64.ret"]);
        let reprinted = plain(&ctx, parsed.get_operation());
        assert!(reprinted.contains("aarch64.mov_imm rd=x0, imm=7"));
    }

    #[test]
    fn branch_targets_round_trip_inside_a_func() {
        let mut ctx = context();

        let func = FuncOp::new(&mut ctx, "f".try_into().unwrap(), LinkageAttr::External);
        let entry = func.entry_block(&ctx);
        let exit = BasicBlock::new(&mut ctx, Some("exit".try_into().unwrap()), vec![]);
        exit.insert_at_back(func.get_region(&ctx), &ctx);

        // A forward branch and a backedge: the parser must resolve a target
        // before its block is defined and one that is already defined.
        b_cond(&mut ctx, ConditionCode::Eq, exit).insert_at_back(entry, &ctx);
        b(&mut ctx, entry).insert_at_back(entry, &ctx);
        ret(&mut ctx).insert_at_back(exit, &ctx);

        let printed = plain(&ctx, func.get_operation());
        let parsed = FuncOp {
            op: parse_op_text(&mut ctx, &printed),
        };

        // The reparsed branches reference the reparsed blocks by identity.
        let blocks: Vec<_> = parsed.get_region(&ctx).deref(&ctx).iter(&ctx).collect();
        assert_eq!(blocks.len(), 2);
        let entry_ops: Vec<_> = blocks[0].deref(&ctx).iter(&ctx).collect();
        assert!(target(&ctx, entry_ops[0]) == Some(blocks[1]));
        assert!(target(&ctx, entry_ops[1]) == Some(blocks[0]));
    }
}

use crate::ll::{BytesAttr, LinkageAttr};
