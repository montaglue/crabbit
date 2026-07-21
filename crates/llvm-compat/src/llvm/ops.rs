//! Operations defined in the LLVM dialect.

use combine::{Parser, attempt, many, optional, sep_by, token};
use pliron::derive::{def_op, derive_op_interface_impl};
use pliron::derive::op_interface_impl;
use thiserror::Error;

use crate::ll::BytesAttr;
use crate::op_interfaces::{FunctionLikeInterface, WeightedBranchOpInterface};
use crate::{
    common_traits::{Named, Verify},
    context::{Context, Ptr},
    dialects::builtin::{
        attributes::{IdentifierAttr, IntegerAttr, StringAttr, TypeAttr},
        op_interfaces::{
            self, ATTR_KEY_SYM_NAME, BranchOpInterface,
            IsTerminatorInterface, IsolatedFromAboveInterface, OneRegionInterface,
            OneResultInterface, OperandSegmentInterface, SameOperandsAndResultType,
            SameOperandsType, SameResultsType, SymbolOpInterface,
            NOpdsInterface, NResultsInterface,
        },
        type_interfaces::FunctionTypeInterface,
        types::{IntegerType, Signedness},
    },
    dict_key,
    identifier::Identifier,
    impl_verify_succ, input_err,
    ir::basic_block::BasicBlock,
    ir::irfmt::{
        parsers::{block_opd_parser, process_parsed_ssa_defs, spaced, ssa_opd_parser, type_parser},
        printers::op::region,
    },
    ir::location::{Located, Location},
    ir::op::{Op, OpObj},
    ir::operation::Operation,
    ir::region::Region,
    ir::r#type::{TypeHandle, TypedHandle, Typed},
    ir::value::Value,
    linked_list::ContainsLinkedList,
    parsable::{Parsable, ParseResult, StateStream},
    printable::{self, Printable},
    result::STAIRResult,
    verify_err,
};

use super::{
    attributes::{
        FCmpPredicateAttr, GepIndicesAttr, ICmpPredicateAttr, InsertExtractValueIndicesAttr,
        IntegerOverflowFlagsAttr, LinkageAttr, PhiIncomingBlocksAttr,
    },
    op_interfaces::{
        AlignableOpInterface, BinArithOp, CastOpInterface, FloatBinArithOp, IntBinArithOp,
        IntBinArithOpWithOverflowFlag, IsDeclaration, PointerTypeResult,
    },
    types::{FuncType, PointerType},
};

// ============================================================================
// Macros for reducing boilerplate
// ============================================================================

/// Define an LLVM integer binary arithmetic op WITHOUT overflow flags.
macro_rules! def_llvm_int_bin_op {
    ($name:ident, $opid:literal, $doc:literal) => {
        #[doc = $doc]
        #[def_op($opid)]
        #[derive_op_interface_impl(
            OneResultInterface,
            SameOperandsType,
            SameResultsType,
            SameOperandsAndResultType
        )]
        pub struct $name;

        #[op_interface_impl]
        impl BinArithOp for $name {}

        #[op_interface_impl]
        impl IntBinArithOp for $name {}

        impl $name {
            pub fn new(ctx: &mut Context, lhs: Value, rhs: Value) -> Self {
                let result_type = lhs.get_type(ctx);
                let op = Operation::new(
                    ctx,
                    Self::get_concrete_op_info(),
                    vec![result_type],
                    vec![lhs, rhs],
                    vec![],
                    0,
                );
                $name { op }
            }
        }

        impl Printable for $name {
            fn fmt(
                &self,
                ctx: &Context,
                state: &printable::State,
                f: &mut core::fmt::Formatter<'_>,
            ) -> core::fmt::Result {
                self.get_result(ctx).fmt(ctx, state, f)?;
                write!(f, " = {} ", self.get_opid().disp(ctx))?;
                self.get_lhs(ctx).fmt(ctx, state, f)?;
                write!(f, ", ")?;
                self.get_rhs(ctx).fmt(ctx, state, f)?;
                write!(f, " : ")?;
                self.get_lhs(ctx).get_type(ctx).fmt(ctx, state, f)?;
                Ok(())
            }
        }

        impl Parsable for $name {
            type Arg = Vec<(Identifier, Location)>;
            type Parsed = OpObj;

            fn parse<'a>(
                state_stream: &mut StateStream<'a>,
                results: Self::Arg,
            ) -> ParseResult<'a, Self::Parsed> {
                let mut parser = spaced(ssa_opd_parser())
                    .skip(spaced(token(',')))
                    .and(spaced(ssa_opd_parser()))
                    .skip(spaced(token(':')))
                    .and(spaced(type_parser()));

                parser
                    .parse_stream(state_stream)
                    .map(|((lhs, rhs), _ty)| -> OpObj {
                        let ctx = &mut *state_stream.state.ctx;
                        let op = $name::new(ctx, lhs, rhs);
                        if !results.is_empty() {
                            process_parsed_ssa_defs(state_stream, &results, op.get_operation()).ok();
                        }
                        OpObj::new(op)
                    })
                    .into()
            }
        }

        impl_verify_succ!($name);
    };
}

/// Define an LLVM integer binary arithmetic op WITH overflow flags (nsw/nuw).
macro_rules! def_llvm_int_bin_op_with_overflow {
    ($name:ident, $opid:literal, $doc:literal) => {
        #[doc = $doc]
        #[def_op($opid)]
        #[derive_op_interface_impl(
            OneResultInterface,
            SameOperandsType,
            SameResultsType,
            SameOperandsAndResultType
        )]
        pub struct $name;

        #[op_interface_impl]
        impl BinArithOp for $name {}

        #[op_interface_impl]
        impl IntBinArithOp for $name {}

        #[op_interface_impl]
        impl IntBinArithOpWithOverflowFlag for $name {}

        impl $name {
            pub fn new(ctx: &mut Context, lhs: Value, rhs: Value) -> Self {
                let result_type = lhs.get_type(ctx);
                let op = Operation::new(
                    ctx,
                    Self::get_concrete_op_info(),
                    vec![result_type],
                    vec![lhs, rhs],
                    vec![],
                    0,
                );
                let inst = $name { op };
                inst.set_integer_overflow_flag(ctx, IntegerOverflowFlagsAttr::default());
                inst
            }

            pub fn new_with_overflow(
                ctx: &mut Context,
                lhs: Value,
                rhs: Value,
                flags: IntegerOverflowFlagsAttr,
            ) -> Self {
                let result_type = lhs.get_type(ctx);
                let op = Operation::new(
                    ctx,
                    Self::get_concrete_op_info(),
                    vec![result_type],
                    vec![lhs, rhs],
                    vec![],
                    0,
                );
                let inst = $name { op };
                inst.set_integer_overflow_flag(ctx, flags);
                inst
            }
        }

        impl Printable for $name {
            fn fmt(
                &self,
                ctx: &Context,
                state: &printable::State,
                f: &mut core::fmt::Formatter<'_>,
            ) -> core::fmt::Result {
                self.get_result(ctx).fmt(ctx, state, f)?;
                write!(f, " = {} ", self.get_opid().disp(ctx))?;
                self.get_lhs(ctx).fmt(ctx, state, f)?;
                write!(f, ", ")?;
                self.get_rhs(ctx).fmt(ctx, state, f)?;
                // Print overflow flags if any are set
                let flags = self.integer_overflow_flag(ctx);
                if flags.nsw || flags.nuw {
                    write!(f, " ")?;
                    flags.fmt(ctx, state, f)?;
                }
                write!(f, " : ")?;
                self.get_lhs(ctx).get_type(ctx).fmt(ctx, state, f)?;
                Ok(())
            }
        }

        impl Parsable for $name {
            type Arg = Vec<(Identifier, Location)>;
            type Parsed = OpObj;

            fn parse<'a>(
                state_stream: &mut StateStream<'a>,
                results: Self::Arg,
            ) -> ParseResult<'a, Self::Parsed> {
                let mut parser = spaced(ssa_opd_parser())
                    .skip(spaced(token(',')))
                    .and(spaced(ssa_opd_parser()))
                    .and(optional(spaced(IntegerOverflowFlagsAttr::parser(()))))
                    .skip(spaced(token(':')))
                    .and(spaced(type_parser()));

                parser
                    .parse_stream(state_stream)
                    .map(|(((lhs, rhs), flags), _ty)| -> OpObj {
                        let ctx = &mut *state_stream.state.ctx;
                        let flags = flags.unwrap_or_default();
                        let op = $name::new_with_overflow(ctx, lhs, rhs, flags);
                        if !results.is_empty() {
                            process_parsed_ssa_defs(state_stream, &results, op.get_operation()).ok();
                        }
                        OpObj::new(op)
                    })
                    .into()
            }
        }

        impl_verify_succ!($name);
    };
}

/// Define an LLVM float binary arithmetic op.
macro_rules! def_llvm_float_bin_op {
    ($name:ident, $opid:literal, $doc:literal) => {
        #[doc = $doc]
        #[def_op($opid)]
        #[derive_op_interface_impl(
            OneResultInterface,
            SameOperandsType,
            SameResultsType,
            SameOperandsAndResultType
        )]
        pub struct $name;

        #[op_interface_impl]
        impl BinArithOp for $name {}

        #[op_interface_impl]
        impl FloatBinArithOp for $name {}

        impl $name {
            // TODO: FastmathFlags — add new_with_fast_math_flags constructor when available
            pub fn new(ctx: &mut Context, lhs: Value, rhs: Value) -> Self {
                let result_type = lhs.get_type(ctx);
                let op = Operation::new(
                    ctx,
                    Self::get_concrete_op_info(),
                    vec![result_type],
                    vec![lhs, rhs],
                    vec![],
                    0,
                );
                $name { op }
            }
        }

        impl Printable for $name {
            fn fmt(
                &self,
                ctx: &Context,
                state: &printable::State,
                f: &mut core::fmt::Formatter<'_>,
            ) -> core::fmt::Result {
                self.get_result(ctx).fmt(ctx, state, f)?;
                write!(f, " = {} ", self.get_opid().disp(ctx))?;
                // TODO: FastmathFlags — print fast math flags here
                self.get_lhs(ctx).fmt(ctx, state, f)?;
                write!(f, ", ")?;
                self.get_rhs(ctx).fmt(ctx, state, f)?;
                write!(f, " : ")?;
                self.get_lhs(ctx).get_type(ctx).fmt(ctx, state, f)?;
                Ok(())
            }
        }

        impl Parsable for $name {
            type Arg = Vec<(Identifier, Location)>;
            type Parsed = OpObj;

            fn parse<'a>(
                state_stream: &mut StateStream<'a>,
                results: Self::Arg,
            ) -> ParseResult<'a, Self::Parsed> {
                // TODO: FastmathFlags — parse fast math flags here
                let mut parser = spaced(ssa_opd_parser())
                    .skip(spaced(token(',')))
                    .and(spaced(ssa_opd_parser()))
                    .skip(spaced(token(':')))
                    .and(spaced(type_parser()));

                parser
                    .parse_stream(state_stream)
                    .map(|((lhs, rhs), _ty)| -> OpObj {
                        let ctx = &mut *state_stream.state.ctx;
                        let op = $name::new(ctx, lhs, rhs);
                        if !results.is_empty() {
                            process_parsed_ssa_defs(state_stream, &results, op.get_operation()).ok();
                        }
                        OpObj::new(op)
                    })
                    .into()
            }
        }

        impl_verify_succ!($name);
    };
}

/// Define an LLVM cast op (one operand, one result, possibly different types).
macro_rules! def_llvm_cast_op {
    ($name:ident, $opid:literal, $doc:literal) => {
        #[doc = $doc]
        #[def_op($opid)]
        #[derive_op_interface_impl(OneResultInterface)]
        pub struct $name;

        #[op_interface_impl]
        impl CastOpInterface for $name {}

        impl $name {
            pub fn new(ctx: &mut Context, operand: Value, res_type: TypeHandle) -> Self {
                let op = Operation::new(
                    ctx,
                    Self::get_concrete_op_info(),
                    vec![res_type],
                    vec![operand],
                    vec![],
                    0,
                );
                $name { op }
            }

            /// Get the operand being cast.
            pub fn get_input(&self, ctx: &Context) -> Value {
                self.get_operation().deref(ctx).get_operand(0)
            }
        }

        impl Printable for $name {
            fn fmt(
                &self,
                ctx: &Context,
                state: &printable::State,
                f: &mut core::fmt::Formatter<'_>,
            ) -> core::fmt::Result {
                self.get_result(ctx).fmt(ctx, state, f)?;
                write!(f, " = {} ", self.get_opid().disp(ctx))?;
                self.get_input(ctx).fmt(ctx, state, f)?;
                write!(f, " : ")?;
                self.get_input(ctx).get_type(ctx).fmt(ctx, state, f)?;
                write!(f, " to ")?;
                self.get_result(ctx).get_type(ctx).fmt(ctx, state, f)?;
                Ok(())
            }
        }

        impl Parsable for $name {
            type Arg = Vec<(Identifier, Location)>;
            type Parsed = OpObj;

            fn parse<'a>(
                state_stream: &mut StateStream<'a>,
                results: Self::Arg,
            ) -> ParseResult<'a, Self::Parsed> {
                use combine::parser::char::string;
                let mut parser = spaced(ssa_opd_parser())
                    .skip(spaced(token(':')))
                    .skip(spaced(type_parser()))
                    .skip(spaced(string("to")))
                    .and(spaced(type_parser()));

                parser
                    .parse_stream(state_stream)
                    .map(|(operand, res_type)| -> OpObj {
                        let ctx = &mut *state_stream.state.ctx;
                        let op = $name::new(ctx, operand, res_type);
                        if !results.is_empty() {
                            process_parsed_ssa_defs(state_stream, &results, op.get_operation()).ok();
                        }
                        OpObj::new(op)
                    })
                    .into()
            }
        }

        impl_verify_succ!($name);
    };
}

// ============================================================================
// Terminators
// ============================================================================

/// LLVM return operation. Terminates a function, optionally returning a value.
#[def_op("llvm.return")]
#[derive_op_interface_impl(IsTerminatorInterface, NResultsInterface<0>)]
pub struct ReturnOp;

impl ReturnOp {
    pub fn new(ctx: &mut Context, value: Option<Value>) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            value.into_iter().collect(),
            vec![],
            0,
        );
        ReturnOp { op }
    }

    /// Get the returned value, if it exists.
    pub fn retval(&self, ctx: &Context) -> Option<Value> {
        let op_ref = self.get_operation().deref(ctx);
        if op_ref.get_num_operands() > 0 {
            Some(op_ref.get_operand(0))
        } else {
            None
        }
    }
}

impl Printable for ReturnOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "{}", self.get_opid().disp(ctx))?;
        if let Some(val) = self.retval(ctx) {
            write!(f, " ")?;
            val.fmt(ctx, state, f)?;
            write!(f, " : ")?;
            val.get_type(ctx).fmt(ctx, state, f)?;
        }
        Ok(())
    }
}

impl Parsable for ReturnOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        if !results.is_empty() {
            input_err!(
                state_stream.loc(),
                op_interfaces::NResultsVerifyErr(0, results.len())
            )?
        }

        let mut parser = optional(
            spaced(ssa_opd_parser())
                .skip(spaced(token(':')))
                .skip(spaced(type_parser())),
        );

        parser
            .parse_stream(state_stream)
            .map(|val| -> OpObj {
                let ctx = &mut *state_stream.state.ctx;
                OpObj::new(ReturnOp::new(ctx, val))
            })
            .into()
    }
}

impl_verify_succ!(ReturnOp);

/// LLVM unreachable operation. Marks unreachable code.
#[def_op("llvm.unreachable")]
#[derive_op_interface_impl(IsTerminatorInterface, NResultsInterface<0>, NOpdsInterface<0>)]
pub struct UnreachableOp;

impl UnreachableOp {
    pub fn new(ctx: &mut Context) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        UnreachableOp { op }
    }
}

impl Printable for UnreachableOp {
    fn fmt(
        &self,
        ctx: &Context,
        _state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "{}", self.get_opid().disp(ctx))
    }
}

impl Parsable for UnreachableOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        if !results.is_empty() {
            input_err!(
                state_stream.loc(),
                op_interfaces::NResultsVerifyErr(0, results.len())
            )?
        }
        Ok(OpObj::new(UnreachableOp::new(state_stream.state.ctx))).into_parse_result()
    }
}

use crate::parsable::IntoParseResult;

impl_verify_succ!(UnreachableOp);

// ============================================================================
// Integer arithmetic with overflow flags
// ============================================================================

def_llvm_int_bin_op_with_overflow!(AddOp, "llvm.add", "LLVM integer addition.");
def_llvm_int_bin_op_with_overflow!(SubOp, "llvm.sub", "LLVM integer subtraction.");
def_llvm_int_bin_op_with_overflow!(MulOp, "llvm.mul", "LLVM integer multiplication.");
def_llvm_int_bin_op_with_overflow!(ShlOp, "llvm.shl", "LLVM integer left shift.");
def_llvm_int_bin_op_with_overflow!(LShrOp, "llvm.lshr", "LLVM logical right shift.");

// ============================================================================
// Integer arithmetic without overflow flags
// ============================================================================

def_llvm_int_bin_op!(UDivOp, "llvm.udiv", "LLVM unsigned integer division.");
def_llvm_int_bin_op!(SDivOp, "llvm.sdiv", "LLVM signed integer division.");
def_llvm_int_bin_op!(URemOp, "llvm.urem", "LLVM unsigned integer remainder.");
def_llvm_int_bin_op!(SRemOp, "llvm.srem", "LLVM signed integer remainder.");
def_llvm_int_bin_op!(AndOp, "llvm.and", "LLVM bitwise AND.");
def_llvm_int_bin_op!(OrOp, "llvm.or", "LLVM bitwise OR.");
def_llvm_int_bin_op!(XorOp, "llvm.xor", "LLVM bitwise XOR.");

// ============================================================================
// Float arithmetic
// ============================================================================

def_llvm_float_bin_op!(FAddOp, "llvm.fadd", "LLVM floating-point addition.");
def_llvm_float_bin_op!(FSubOp, "llvm.fsub", "LLVM floating-point subtraction.");
def_llvm_float_bin_op!(FMulOp, "llvm.fmul", "LLVM floating-point multiplication.");
def_llvm_float_bin_op!(FDivOp, "llvm.fdiv", "LLVM floating-point division.");
def_llvm_float_bin_op!(FRemOp, "llvm.frem", "LLVM floating-point remainder.");

// ============================================================================
// Cast operations
// ============================================================================

def_llvm_cast_op!(BitcastOp, "llvm.bitcast", "LLVM bitwise cast.");
def_llvm_cast_op!(IntToPtrOp, "llvm.inttoptr", "LLVM integer to pointer cast.");
def_llvm_cast_op!(PtrToIntOp, "llvm.ptrtoint", "LLVM pointer to integer cast.");
def_llvm_cast_op!(SExtOp, "llvm.sext", "LLVM sign extend.");
def_llvm_cast_op!(ZExtOp, "llvm.zext", "LLVM zero extend.");
def_llvm_cast_op!(TruncOp, "llvm.trunc", "LLVM truncate integer.");
def_llvm_cast_op!(FPExtOp, "llvm.fpext", "LLVM extend floating-point.");
def_llvm_cast_op!(FPTruncOp, "llvm.fptrunc", "LLVM truncate floating-point.");
def_llvm_cast_op!(
    FPToSIOp,
    "llvm.fptosi",
    "LLVM floating-point to signed integer."
);
def_llvm_cast_op!(
    FPToUIOp,
    "llvm.fptoui",
    "LLVM floating-point to unsigned integer."
);
def_llvm_cast_op!(
    SIToFPOp,
    "llvm.sitofp",
    "LLVM signed integer to floating-point."
);
def_llvm_cast_op!(
    UIToFPOp,
    "llvm.uitofp",
    "LLVM unsigned integer to floating-point."
);

// ============================================================================
// Comparison operations
// ============================================================================

dict_key!(ATTR_KEY_ICMP_PREDICATE, "llvm_icmp_predicate");

/// LLVM integer comparison operation.
#[def_op("llvm.icmp")]
#[derive_op_interface_impl(OneResultInterface)]
pub struct ICmpOp;

impl ICmpOp {
    pub fn new(ctx: &mut Context, predicate: ICmpPredicateAttr, lhs: Value, rhs: Value) -> Self {
        let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![i1_ty.into()],
            vec![lhs, rhs],
            vec![],
            0,
        );
        let icmp = ICmpOp { op };
        icmp.get_operation()
            .deref_mut(ctx)
            .attributes
            .set(ATTR_KEY_ICMP_PREDICATE.clone(), predicate);
        icmp
    }

    pub fn get_predicate(&self, ctx: &Context) -> ICmpPredicateAttr {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<ICmpPredicateAttr>(&ATTR_KEY_ICMP_PREDICATE)
            .cloned()
            .unwrap()
    }

    pub fn get_lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    pub fn get_rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}

impl Printable for ICmpOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        self.get_result(ctx).fmt(ctx, state, f)?;
        write!(f, " = {} ", self.get_opid().disp(ctx))?;
        write!(f, "{} ", self.get_predicate(ctx))?;
        self.get_lhs(ctx).fmt(ctx, state, f)?;
        write!(f, ", ")?;
        self.get_rhs(ctx).fmt(ctx, state, f)?;
        write!(f, " : ")?;
        self.get_lhs(ctx).get_type(ctx).fmt(ctx, state, f)?;
        Ok(())
    }
}

impl Parsable for ICmpOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let mut parser = spaced(ICmpPredicateAttr::parser(()))
            .skip(optional(spaced(token(','))))
            .and(spaced(ssa_opd_parser()))
            .skip(spaced(token(',')))
            .and(spaced(ssa_opd_parser()))
            .skip(spaced(token(':')))
            .and(spaced(type_parser()));

        parser
            .parse_stream(state_stream)
            .map(|(((pred, lhs), rhs), _ty)| -> OpObj {
                let ctx = &mut *state_stream.state.ctx;
                let op = ICmpOp::new(ctx, pred, lhs, rhs);
                if !results.is_empty() {
                    process_parsed_ssa_defs(state_stream, &results, op.get_operation()).ok();
                }
                OpObj::new(op)
            })
            .into()
    }
}

impl_verify_succ!(ICmpOp);

dict_key!(ATTR_KEY_FCMP_PREDICATE, "llvm_fcmp_predicate");

/// LLVM floating-point comparison operation.
#[def_op("llvm.fcmp")]
#[derive_op_interface_impl(OneResultInterface)]
pub struct FCmpOp;

impl FCmpOp {
    // TODO: FastmathFlags — add fast math flags parameter
    pub fn new(ctx: &mut Context, predicate: FCmpPredicateAttr, lhs: Value, rhs: Value) -> Self {
        let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![i1_ty.into()],
            vec![lhs, rhs],
            vec![],
            0,
        );
        let fcmp = FCmpOp { op };
        fcmp.get_operation()
            .deref_mut(ctx)
            .attributes
            .set(ATTR_KEY_FCMP_PREDICATE.clone(), predicate);
        fcmp
    }

    pub fn get_predicate(&self, ctx: &Context) -> FCmpPredicateAttr {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<FCmpPredicateAttr>(&ATTR_KEY_FCMP_PREDICATE)
            .cloned()
            .unwrap()
    }

    pub fn get_lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    pub fn get_rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}

impl Printable for FCmpOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        self.get_result(ctx).fmt(ctx, state, f)?;
        write!(f, " = {} ", self.get_opid().disp(ctx))?;
        // TODO: FastmathFlags — print fast math flags here
        write!(f, "{} ", self.get_predicate(ctx))?;
        self.get_lhs(ctx).fmt(ctx, state, f)?;
        write!(f, ", ")?;
        self.get_rhs(ctx).fmt(ctx, state, f)?;
        write!(f, " : ")?;
        self.get_lhs(ctx).get_type(ctx).fmt(ctx, state, f)?;
        Ok(())
    }
}

impl Parsable for FCmpOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        // TODO: FastmathFlags — parse fast math flags here
        let mut parser = spaced(FCmpPredicateAttr::parser(()))
            .skip(optional(spaced(token(','))))
            .and(spaced(ssa_opd_parser()))
            .skip(spaced(token(',')))
            .and(spaced(ssa_opd_parser()))
            .skip(spaced(token(':')))
            .and(spaced(type_parser()));

        parser
            .parse_stream(state_stream)
            .map(|(((pred, lhs), rhs), _ty)| -> OpObj {
                let ctx = &mut *state_stream.state.ctx;
                let op = FCmpOp::new(ctx, pred, lhs, rhs);
                if !results.is_empty() {
                    process_parsed_ssa_defs(state_stream, &results, op.get_operation()).ok();
                }
                OpObj::new(op)
            })
            .into()
    }
}

impl_verify_succ!(FCmpOp);

// ============================================================================
// Memory operations
// ============================================================================

dict_key!(ATTR_KEY_ALLOCA_ELEM_TYPE, "llvm_alloca_elem_type");

/// LLVM alloca operation: allocate memory on the stack.
#[def_op("llvm.alloca")]
#[derive_op_interface_impl(OneResultInterface)]
pub struct AllocaOp;

#[op_interface_impl]
impl PointerTypeResult for AllocaOp {}

#[op_interface_impl]
impl AlignableOpInterface for AllocaOp {}

impl AllocaOp {
    pub fn new(ctx: &mut Context, array_size: Value, elem_type: TypeHandle) -> Self {
        let ptr_ty: TypeHandle = PointerType::get(ctx).into();
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![ptr_ty],
            vec![array_size],
            vec![],
            0,
        );
        let alloca = AllocaOp { op };
        alloca
            .get_operation()
            .deref_mut(ctx)
            .attributes
            .set(ATTR_KEY_ALLOCA_ELEM_TYPE.clone(), TypeAttr::new(elem_type));
        alloca
    }

    pub fn get_array_size(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    pub fn get_elem_type(&self, ctx: &Context) -> TypeHandle {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<TypeAttr>(&ATTR_KEY_ALLOCA_ELEM_TYPE)
            .unwrap()
            .get_type(ctx)
    }
}

impl Printable for AllocaOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        self.get_result(ctx).fmt(ctx, state, f)?;
        write!(f, " = {} ", self.get_opid().disp(ctx))?;
        self.get_array_size(ctx).fmt(ctx, state, f)?;
        write!(f, " x ")?;
        self.get_elem_type(ctx).fmt(ctx, state, f)?;
        if let Some(align) = self.alignment(ctx) {
            write!(f, ", align {}", align)?;
        }
        Ok(())
    }
}

impl Parsable for AllocaOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        use combine::parser::char::string;

        let mut parser = spaced(ssa_opd_parser())
            .skip(spaced(token('x')))
            .and(spaced(type_parser()))
            .and(optional(
                spaced(token(','))
                    .with(spaced(string("align")))
                    .with(spaced(crate::ir::irfmt::parsers::int_parser::<u32>())),
            ));

        parser
            .parse_stream(state_stream)
            .map(|((array_size, elem_type), align)| -> OpObj {
                let ctx = &mut *state_stream.state.ctx;
                let op = AllocaOp::new(ctx, array_size, elem_type);
                if let Some(align) = align {
                    op.set_alignment(ctx, align);
                }
                if !results.is_empty() {
                    process_parsed_ssa_defs(state_stream, &results, op.get_operation()).ok();
                }
                OpObj::new(op)
            })
            .into()
    }
}

impl_verify_succ!(AllocaOp);

/// LLVM load operation: load a value from a pointer.
#[def_op("llvm.load")]
#[derive_op_interface_impl(OneResultInterface)]
pub struct LoadOp;

#[op_interface_impl]
impl AlignableOpInterface for LoadOp {}

impl LoadOp {
    pub fn new(ctx: &mut Context, addr: Value, result_type: TypeHandle) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_type],
            vec![addr],
            vec![],
            0,
        );
        LoadOp { op }
    }

    pub fn get_addr(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}

impl Printable for LoadOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        self.get_result(ctx).fmt(ctx, state, f)?;
        write!(f, " = {} ", self.get_opid().disp(ctx))?;
        self.get_addr(ctx).fmt(ctx, state, f)?;
        write!(f, " : ")?;
        self.get_result(ctx).get_type(ctx).fmt(ctx, state, f)?;
        if let Some(align) = self.alignment(ctx) {
            write!(f, ", align {}", align)?;
        }
        Ok(())
    }
}

impl Parsable for LoadOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        use combine::parser::char::string;

        let mut parser = spaced(ssa_opd_parser())
            .skip(spaced(token(':')))
            .and(spaced(type_parser()))
            .and(optional(
                spaced(token(','))
                    .with(spaced(string("align")))
                    .with(spaced(crate::ir::irfmt::parsers::int_parser::<u32>())),
            ));

        parser
            .parse_stream(state_stream)
            .map(|((addr, result_type), align)| -> OpObj {
                let ctx = &mut *state_stream.state.ctx;
                let op = LoadOp::new(ctx, addr, result_type);
                if let Some(align) = align {
                    op.set_alignment(ctx, align);
                }
                if !results.is_empty() {
                    process_parsed_ssa_defs(state_stream, &results, op.get_operation()).ok();
                }
                OpObj::new(op)
            })
            .into()
    }
}

impl_verify_succ!(LoadOp);

/// LLVM store operation: store a value to a pointer.
#[def_op("llvm.store")]
#[derive_op_interface_impl(NResultsInterface<0>)]
pub struct StoreOp;

#[op_interface_impl]
impl AlignableOpInterface for StoreOp {}

impl StoreOp {
    pub fn new(ctx: &mut Context, value: Value, addr: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![value, addr],
            vec![],
            0,
        );
        StoreOp { op }
    }

    pub fn get_value(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    pub fn get_addr(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}

impl Printable for StoreOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "{} ", self.get_opid().disp(ctx))?;
        self.get_value(ctx).fmt(ctx, state, f)?;
        write!(f, ", ")?;
        self.get_addr(ctx).fmt(ctx, state, f)?;
        write!(f, " : ")?;
        self.get_value(ctx).get_type(ctx).fmt(ctx, state, f)?;
        if let Some(align) = self.alignment(ctx) {
            write!(f, ", align {}", align)?;
        }
        Ok(())
    }
}

impl Parsable for StoreOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        use combine::parser::char::string;

        if !results.is_empty() {
            input_err!(
                state_stream.loc(),
                op_interfaces::NResultsVerifyErr(0, results.len())
            )?
        }

        let mut parser = spaced(ssa_opd_parser())
            .skip(spaced(token(',')))
            .and(spaced(ssa_opd_parser()))
            .skip(spaced(token(':')))
            .skip(spaced(type_parser()))
            .and(optional(
                spaced(token(','))
                    .with(spaced(string("align")))
                    .with(spaced(crate::ir::irfmt::parsers::int_parser::<u32>())),
            ));

        parser
            .parse_stream(state_stream)
            .map(|((value, addr), align)| -> OpObj {
                let ctx = &mut *state_stream.state.ctx;
                let op = StoreOp::new(ctx, value, addr);
                if let Some(align) = align {
                    op.set_alignment(ctx, align);
                }
                OpObj::new(op)
            })
            .into()
    }
}

impl_verify_succ!(StoreOp);

dict_key!(ATTR_KEY_GEP_SOURCE_ELEM_TYPE, "llvm_gep_source_elem_type");
dict_key!(ATTR_KEY_GEP_INDICES, "llvm_gep_indices");

/// LLVM getelementptr operation: compute element pointer with type-based indexing.
#[def_op("llvm.gep")]
#[derive_op_interface_impl(OneResultInterface)]
pub struct GetElementPtrOp;

#[op_interface_impl]
impl PointerTypeResult for GetElementPtrOp {}

impl GetElementPtrOp {
    pub fn new(
        ctx: &mut Context,
        base: Value,
        dynamic_indices: Vec<Value>,
        indices: GepIndicesAttr,
        source_elem_type: TypeHandle,
    ) -> Self {
        let ptr_ty: TypeHandle = PointerType::get(ctx).into();
        let mut operands = vec![base];
        operands.extend(dynamic_indices);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![ptr_ty],
            operands,
            vec![],
            0,
        );
        let gep = GetElementPtrOp { op };
        {
            let mut op_ref = gep.get_operation().deref_mut(ctx);
            op_ref.attributes.set(
                ATTR_KEY_GEP_SOURCE_ELEM_TYPE.clone(),
                TypeAttr::new(source_elem_type),
            );
            op_ref.attributes.set(ATTR_KEY_GEP_INDICES.clone(), indices);
        }
        gep
    }

    pub fn get_base(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    pub fn get_indices(&self, ctx: &Context) -> GepIndicesAttr {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<GepIndicesAttr>(&ATTR_KEY_GEP_INDICES)
            .cloned()
            .unwrap()
    }

    pub fn get_source_elem_type(&self, ctx: &Context) -> TypeHandle {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<TypeAttr>(&ATTR_KEY_GEP_SOURCE_ELEM_TYPE)
            .unwrap()
            .get_type(ctx)
    }
}

impl Printable for GetElementPtrOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        self.get_result(ctx).fmt(ctx, state, f)?;
        write!(f, " = {} ", self.get_opid().disp(ctx))?;
        self.get_source_elem_type(ctx).fmt(ctx, state, f)?;
        write!(f, ", ")?;
        self.get_base(ctx).fmt(ctx, state, f)?;
        write!(f, " ")?;
        self.get_indices(ctx).fmt(ctx, state, f)?;
        Ok(())
    }
}

impl Parsable for GetElementPtrOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let mut parser = spaced(type_parser())
            .skip(spaced(token(',')))
            .and(spaced(ssa_opd_parser()))
            .and(spaced(GepIndicesAttr::parser(())));

        parser
            .parse_stream(state_stream)
            .map(|((source_elem_type, base), indices)| -> OpObj {
                let ctx = &mut *state_stream.state.ctx;
                // Collect dynamic operands from the indices
                let dynamic_indices = Vec::new();
                // For now, dynamic indices need to be resolved from the GepIndicesAttr
                // This is a simplified implementation
                let op =
                    GetElementPtrOp::new(ctx, base, dynamic_indices, indices, source_elem_type);
                if !results.is_empty() {
                    process_parsed_ssa_defs(state_stream, &results, op.get_operation()).ok();
                }
                OpObj::new(op)
            })
            .into()
    }
}

impl_verify_succ!(GetElementPtrOp);

// ============================================================================
// Phi
// ============================================================================

dict_key!(
    ATTR_KEY_LLVM_PHI_INCOMING_BLOCKS,
    "llvm_phi_incoming_blocks"
);

/// LLVM phi node.
#[def_op("llvm.phi")]
#[derive_op_interface_impl(OneResultInterface)]
pub struct PhiOp;

impl PhiOp {
    pub fn new(
        ctx: &mut Context,
        incoming_values: Vec<Value>,
        incoming_blocks: Vec<Identifier>,
        result_type: TypeHandle,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_type],
            incoming_values,
            vec![],
            0,
        );
        let phi = PhiOp { op };
        phi.get_operation().deref_mut(ctx).attributes.set(
            ATTR_KEY_LLVM_PHI_INCOMING_BLOCKS.clone(),
            PhiIncomingBlocksAttr(incoming_blocks),
        );
        phi
    }

    pub fn get_result(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }

    pub fn get_incoming_values(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().collect()
    }

    pub fn get_incoming_blocks(&self, ctx: &Context) -> Vec<Identifier> {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<PhiIncomingBlocksAttr>(&ATTR_KEY_LLVM_PHI_INCOMING_BLOCKS)
            .map(|attr| attr.0.clone())
            .unwrap_or_default()
    }
}

impl Printable for PhiOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        self.get_result(ctx).fmt(ctx, state, f)?;
        write!(f, " = {} ", self.get_opid().disp(ctx))?;

        let values = self.get_incoming_values(ctx);
        let blocks = self.get_incoming_blocks(ctx);
        for (idx, (value, block)) in values.iter().zip(blocks.iter()).enumerate() {
            if idx > 0 {
                write!(f, ", ")?;
            }
            write!(f, "[")?;
            value.fmt(ctx, state, f)?;
            write!(f, ", ^{}]", block)?;
        }

        write!(f, " : ")?;
        self.get_result(ctx).get_type(ctx).fmt(ctx, state, f)
    }
}

impl Parsable for PhiOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        if results.len() != 1 {
            input_err!(
                state_stream.loc(),
                op_interfaces::NResultsVerifyErr(1, results.len())
            )?
        }

        let incoming_parser = token('[')
            .with(spaced(ssa_opd_parser()))
            .skip(spaced(token(',')))
            .and(spaced(token('^')).with(Identifier::parser(())))
            .skip(spaced(token(']')));

        let mut parser = sep_by::<Vec<_>, _, _, _>(spaced(incoming_parser), spaced(token(',')))
            .skip(spaced(token(':')))
            .and(spaced(type_parser()));

        parser
            .parse_stream(state_stream)
            .map(|(incoming, result_type)| -> OpObj {
                let ctx = &mut *state_stream.state.ctx;
                let (incoming_values, incoming_blocks): (Vec<_>, Vec<_>) =
                    incoming.into_iter().unzip();
                let op = PhiOp::new(ctx, incoming_values, incoming_blocks, result_type);
                process_parsed_ssa_defs(state_stream, &results, op.get_operation()).ok();
                OpObj::new(op)
            })
            .into()
    }
}

#[derive(Error, Debug)]
#[error("llvm.phi incoming value count must match incoming block count")]
pub struct PhiIncomingCountMismatch;

#[derive(Error, Debug)]
#[error("llvm.phi incoming value type must match result type")]
pub struct PhiIncomingTypeMismatch;

impl Verify for PhiOp {
    fn verify(&self, ctx: &Context) -> STAIRResult<()> {
        let values = self.get_incoming_values(ctx);
        let blocks = self.get_incoming_blocks(ctx);
        if values.len() != blocks.len() {
            verify_err!(self.loc(ctx), PhiIncomingCountMismatch)?;
        }

        let result_ty = self.get_result(ctx).get_type(ctx);
        for value in values {
            if value.get_type(ctx) != result_ty {
                verify_err!(self.loc(ctx), PhiIncomingTypeMismatch)?;
            }
        }
        Ok(())
    }
}

// ============================================================================
// Control flow
// ============================================================================

/// LLVM unconditional branch.
#[def_op("llvm.br")]
#[derive_op_interface_impl(IsTerminatorInterface, NResultsInterface<0>)]
pub struct BrOp;

impl BranchOpInterface for BrOp {
    fn add_successor_operand(&self, ctx: &mut Context, _succ_idx: usize, operand: Value) -> usize {
        Operation::push_operand(self.get_operation(), ctx, operand)
    }

    fn remove_successor_operand(&self, ctx: &mut Context, _succ_idx: usize, opd_idx: usize) -> Value {
        Operation::remove_operand(self.get_operation(), ctx, opd_idx)
    }

    fn successor_operands(&self, ctx: &Context, _succ_idx: usize) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().collect()
    }
}

impl BrOp {
    pub fn new(ctx: &mut Context, dest: Ptr<BasicBlock>, dest_operands: Vec<Value>) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            dest_operands,
            vec![dest],
            0,
        );
        BrOp { op }
    }

    pub fn get_dest(&self, ctx: &Context) -> Ptr<BasicBlock> {
        self.get_operation().deref(ctx).get_successor(0)
    }

    pub fn get_dest_operands(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().collect()
    }
}

impl Printable for BrOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "{} ", self.get_opid().disp(ctx))?;
        let dest = self.get_dest(ctx);
        write!(f, "^{}", dest.deref(ctx).unique_name(ctx))?;

        let operands: Vec<_> = self.get_dest_operands(ctx);
        if !operands.is_empty() {
            write!(f, "(")?;
            for (i, opd) in operands.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                opd.fmt(ctx, state, f)?;
                write!(f, " : ")?;
                opd.get_type(ctx).fmt(ctx, state, f)?;
            }
            write!(f, ")")?;
        }
        Ok(())
    }
}

impl Parsable for BrOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        if !results.is_empty() {
            input_err!(
                state_stream.loc(),
                op_interfaces::NResultsVerifyErr(0, results.len())
            )?
        }

        let operands_parser = token('(')
            .with(many::<Vec<_>, _, _>(
                spaced(ssa_opd_parser())
                    .skip(spaced(token(':')))
                    .skip(spaced(type_parser()))
                    .skip(optional(spaced(token(',')))),
            ))
            .skip(token(')'));

        let mut parser = spaced(block_opd_parser())
            .and(optional(spaced(operands_parser)));

        parser
            .parse_stream(state_stream)
            .map(|(dest, operands)| -> OpObj {
                let ctx = &mut *state_stream.state.ctx;
                let operands = operands.unwrap_or_default();
                OpObj::new(BrOp::new(ctx, dest, operands))
            })
            .into()
    }
}

impl_verify_succ!(BrOp);

/// LLVM conditional branch.
#[def_op("llvm.cond_br")]
#[derive_op_interface_impl(IsTerminatorInterface, NResultsInterface<0>, OperandSegmentInterface)]
pub struct CondBrOp;

#[op_interface_impl]
impl BranchOpInterface for CondBrOp {
    fn add_successor_operand(&self, ctx: &mut Context, succ_idx: usize, operand: Value) -> usize {
        // The successor operands start at segment 1; segment 0 is the condition.
        self.push_to_segment(ctx, succ_idx + 1, operand)
    }

    fn remove_successor_operand(&self, ctx: &mut Context, succ_idx: usize, opd_idx: usize) -> Value {
        // The successor operands start at segment 1; segment 0 is the condition.
        self.remove_from_segment(ctx, succ_idx + 1, opd_idx)
    }

    fn successor_operands(&self, ctx: &Context, succ_idx: usize) -> Vec<Value> {
        // Segment 0: condition (1 operand)
        // Segment 1: true dest operands
        // Segment 2: false dest operands
        self.get_segment(ctx, succ_idx + 1)
    }
}

/// Weights are `[true dest, false dest]`, matching LLVM's `!prof
/// branch_weights` on `br i1` and MLIR's `branch_weights` on `llvm.cond_br`.
#[op_interface_impl]
impl WeightedBranchOpInterface for CondBrOp {}

impl CondBrOp {
    pub fn new(
        ctx: &mut Context,
        condition: Value,
        true_dest: Ptr<BasicBlock>,
        true_operands: Vec<Value>,
        false_dest: Ptr<BasicBlock>,
        false_operands: Vec<Value>,
    ) -> Self {
        let (operands, sizes) =
            Self::compute_segment_sizes(vec![vec![condition], true_operands, false_operands]);

        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            operands,
            vec![true_dest, false_dest],
            0,
        );
        let cond_br = CondBrOp { op };
        cond_br.set_operand_segment_sizes(ctx, sizes);
        cond_br
    }

    pub fn get_condition(&self, ctx: &Context) -> Value {
        self.get_segment(ctx, 0).into_iter().next().unwrap()
    }

    pub fn get_true_dest(&self, ctx: &Context) -> Ptr<BasicBlock> {
        self.get_operation().deref(ctx).get_successor(0)
    }

    pub fn get_false_dest(&self, ctx: &Context) -> Ptr<BasicBlock> {
        self.get_operation().deref(ctx).get_successor(1)
    }

    pub fn get_true_operands(&self, ctx: &Context) -> Vec<Value> {
        self.get_segment(ctx, 1)
    }

    pub fn get_false_operands(&self, ctx: &Context) -> Vec<Value> {
        self.get_segment(ctx, 2)
    }
}

impl Printable for CondBrOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "{} ", self.get_opid().disp(ctx))?;
        self.get_condition(ctx).fmt(ctx, state, f)?;
        write!(f, ", ")?;

        // True destination
        let true_dest = self.get_true_dest(ctx);
        write!(f, "^{}", true_dest.deref(ctx).unique_name(ctx))?;
        let true_opds = self.get_true_operands(ctx);
        if !true_opds.is_empty() {
            write!(f, "(")?;
            for (i, opd) in true_opds.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                opd.fmt(ctx, state, f)?;
                write!(f, " : ")?;
                opd.get_type(ctx).fmt(ctx, state, f)?;
            }
            write!(f, ")")?;
        }

        write!(f, ", ")?;

        // False destination
        let false_dest = self.get_false_dest(ctx);
        write!(f, "^{}", false_dest.deref(ctx).unique_name(ctx))?;
        let false_opds = self.get_false_operands(ctx);
        if !false_opds.is_empty() {
            write!(f, "(")?;
            for (i, opd) in false_opds.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                opd.fmt(ctx, state, f)?;
                write!(f, " : ")?;
                opd.get_type(ctx).fmt(ctx, state, f)?;
            }
            write!(f, ")")?;
        }
        Ok(())
    }
}

impl Parsable for CondBrOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        if !results.is_empty() {
            input_err!(
                state_stream.loc(),
                op_interfaces::NResultsVerifyErr(0, results.len())
            )?
        }

        let make_block_parser = || {
            let operands_parser = token('(')
                .with(many::<Vec<_>, _, _>(
                    spaced(ssa_opd_parser())
                        .skip(spaced(token(':')))
                        .skip(spaced(type_parser()))
                        .skip(optional(spaced(token(',')))),
                ))
                .skip(token(')'));
            spaced(block_opd_parser())
                .and(optional(attempt(spaced(operands_parser))))
        };

        let mut parser = spaced(ssa_opd_parser())
            .skip(spaced(token(',')))
            .and(make_block_parser())
            .skip(spaced(token(',')))
            .and(make_block_parser());

        parser
            .parse_stream(state_stream)
            .map(
                |((cond, (true_dest, true_opds)), (false_dest, false_opds))| -> OpObj {
                    let ctx = &mut *state_stream.state.ctx;

                    OpObj::new(CondBrOp::new(
                        ctx,
                        cond,
                        true_dest,
                        true_opds.unwrap_or_default(),
                        false_dest,
                        false_opds.unwrap_or_default(),
                    ))
                },
            )
            .into()
    }
}

#[derive(Error, Debug)]
#[error("llvm.cond_br condition must be i1 type")]
pub struct CondBrOpConditionTypeErr;

impl Verify for CondBrOp {
    fn verify(&self, ctx: &Context) -> STAIRResult<()> {
        let cond_ty = self.get_condition(ctx).get_type(ctx);
        let cond_ty_obj = cond_ty.deref(ctx);
        if let Some(int_ty) = cond_ty_obj.downcast_ref::<IntegerType>()
            && int_ty.width() == 1
            && int_ty.is_signless()
        {
            return Ok(());
        }
        verify_err!(self.loc(ctx), CondBrOpConditionTypeErr)
    }
}

// ============================================================================
// Call operation
// ============================================================================

dict_key!(ATTR_KEY_LLVM_CALLEE, "llvm_callee");

/// LLVM call operation: call a function directly by symbol or indirectly via pointer.
#[def_op("llvm.call")]
pub struct CallOp;

impl CallOp {
    /// Create a direct call (by symbol name).
    pub fn new_direct(
        ctx: &mut Context,
        callee: Identifier,
        args: Vec<Value>,
        result_type: Option<TypeHandle>,
    ) -> Self {
        let result_types = result_type.into_iter().collect();
        let op = Operation::new(ctx, Self::get_concrete_op_info(), result_types, args, vec![], 0);
        let call = CallOp { op };
        call.get_operation()
            .deref_mut(ctx)
            .attributes
            .set(ATTR_KEY_LLVM_CALLEE.clone(), IdentifierAttr::new(callee));
        call
    }

    /// Indirect call: the callee function pointer is operand 0 and the call
    /// arguments follow it. No callee symbol attribute is set.
    pub fn new_indirect(
        ctx: &mut Context,
        callee: Value,
        args: Vec<Value>,
        result_type: Option<TypeHandle>,
    ) -> Self {
        let mut operands = Vec::with_capacity(args.len() + 1);
        operands.push(callee);
        operands.extend(args);
        let result_types = result_type.into_iter().collect();
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            result_types,
            operands,
            vec![],
            0,
        );
        CallOp { op }
    }

    /// Get the callee symbol name (for direct calls).
    pub fn get_callee(&self, ctx: &Context) -> Option<Identifier> {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<IdentifierAttr>(&ATTR_KEY_LLVM_CALLEE)
            .map(|a| a.clone().into())
    }

    /// Get the arguments passed to the function.
    pub fn get_args(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().collect()
    }
}

impl Printable for CallOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        let op_ref = self.get_operation().deref(ctx);
        if op_ref.get_num_results() > 0 {
            let result = op_ref.get_result(0);
            result.fmt(ctx, state, f)?;
            write!(f, " = ")?;
        }
        drop(op_ref);

        write!(f, "{} ", self.get_opid().disp(ctx))?;
        if let Some(callee) = self.get_callee(ctx) {
            write!(f, "@{}(", callee)?;
        } else {
            write!(f, "(")?;
        }

        let args = self.get_args(ctx);
        for (i, arg) in args.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            arg.fmt(ctx, state, f)?;
        }
        write!(f, ")")?;

        let op_ref = self.get_operation().deref(ctx);
        if op_ref.get_num_results() > 0 {
            write!(f, " : ")?;
            op_ref.get_result(0).get_type(ctx).fmt(ctx, state, f)?;
        }
        Ok(())
    }
}

impl Parsable for CallOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let args_parser = token('(')
            .with(sep_by(spaced(ssa_opd_parser()), spaced(token(','))))
            .skip(token(')'));
        let mut parser = spaced(token('@').with(Identifier::parser(())))
            .and(spaced(args_parser))
            .and(optional(spaced(token(':')).with(spaced(type_parser()))));

        parser
            .parse_stream(state_stream)
            .map(|((callee, args), result_type)| -> OpObj {
                let ctx = &mut *state_stream.state.ctx;
                let op = CallOp::new_direct(ctx, callee, args, result_type);
                if !results.is_empty() && op.get_operation().deref(ctx).get_num_results() > 0 {
                    let result = op.get_operation().deref(ctx).get_result(0);
                    process_parsed_ssa_defs(state_stream, &results, op.get_operation()).ok();
                }
                OpObj::new(op)
            })
            .into()
    }
}

impl_verify_succ!(CallOp);

// ============================================================================
// Constant operations
// ============================================================================

dict_key!(ATTR_KEY_CONSTANT_VALUE, "llvm_constant_value");

/// LLVM constant operation: produces a constant value.
#[def_op("llvm.constant")]
#[derive_op_interface_impl(OneResultInterface, NOpdsInterface<0>)]
pub struct ConstantOp;

impl ConstantOp {
    pub fn new_integer(ctx: &mut Context, attr: IntegerAttr) -> Self {
        let ty: TypeHandle = attr.get_type().into();
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![ty], vec![], vec![], 0);
        let constant = ConstantOp { op };
        constant
            .get_operation()
            .deref_mut(ctx)
            .attributes
            .set(ATTR_KEY_CONSTANT_VALUE.clone(), attr);
        constant
    }

    pub fn get_value(&self, ctx: &Context) -> Option<IntegerAttr> {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<IntegerAttr>(&ATTR_KEY_CONSTANT_VALUE)
            .cloned()
    }
}

impl Printable for ConstantOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        self.get_result(ctx).fmt(ctx, state, f)?;
        write!(f, " = {} ", self.get_opid().disp(ctx))?;
        if let Some(val) = self.get_value(ctx) {
            val.fmt(ctx, state, f)?;
        }
        Ok(())
    }
}

impl Parsable for ConstantOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let mut parser = spaced(IntegerAttr::parser(()));

        parser
            .parse_stream(state_stream)
            .map(|attr| -> OpObj {
                let ctx = &mut *state_stream.state.ctx;
                let op = ConstantOp::new_integer(ctx, attr);
                if !results.is_empty() {
                    process_parsed_ssa_defs(state_stream, &results, op.get_operation()).ok();
                }
                OpObj::new(op)
            })
            .into()
    }
}

impl_verify_succ!(ConstantOp);

/// LLVM undef operation: produces an undefined value.
#[def_op("llvm.undef")]
#[derive_op_interface_impl(OneResultInterface, NOpdsInterface<0>)]
pub struct UndefOp;

impl UndefOp {
    pub fn new(ctx: &mut Context, result_ty: TypeHandle) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_ty],
            vec![],
            vec![],
            0,
        );
        UndefOp { op }
    }
}

impl Printable for UndefOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        self.get_result(ctx).fmt(ctx, state, f)?;
        write!(f, " = {} : ", self.get_opid().disp(ctx))?;
        self.get_result(ctx).get_type(ctx).fmt(ctx, state, f)?;
        Ok(())
    }
}

impl Parsable for UndefOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let mut parser = spaced(token(':')).with(spaced(type_parser()));

        parser
            .parse_stream(state_stream)
            .map(|ty| -> OpObj {
                let ctx = &mut *state_stream.state.ctx;
                let op = UndefOp::new(ctx, ty);
                if !results.is_empty() {
                    process_parsed_ssa_defs(state_stream, &results, op.get_operation()).ok();
                }
                OpObj::new(op)
            })
            .into()
    }
}

impl_verify_succ!(UndefOp);

// ============================================================================
// Global / Function / AddressOf
// ============================================================================

dict_key!(ATTR_KEY_LLVM_GLOBAL_TYPE, "llvm_global_type");
dict_key!(ATTR_KEY_LLVM_GLOBAL_INIT, "llvm_global_init");
dict_key!(ATTR_KEY_LLVM_LINKAGE, "llvm_linkage");
dict_key!(ATTR_KEY_LLVM_CSTR_VALUE, "llvm_cstr_value");

/// LLVM global variable operation.
#[def_op("llvm.global")]
#[derive_op_interface_impl(
    SymbolOpInterface,
    IsolatedFromAboveInterface,
    NOpdsInterface<0>,
    NResultsInterface<0>
)]
pub struct GlobalOp;

impl GlobalOp {
    pub fn new(
        ctx: &mut Context,
        name: Identifier,
        global_type: TypeHandle,
        linkage: LinkageAttr,
    ) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        let global = GlobalOp { op };
        global.set_symbol_name(ctx, name);
        {
            let mut op_ref = global.get_operation().deref_mut(ctx);
            op_ref.attributes.set(
                ATTR_KEY_LLVM_GLOBAL_TYPE.clone(),
                TypeAttr::new(global_type),
            );
            op_ref
                .attributes
                .set(ATTR_KEY_LLVM_LINKAGE.clone(), linkage);
        }
        global
    }

    pub fn get_global_type(&self, ctx: &Context) -> TypeHandle {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<TypeAttr>(&ATTR_KEY_LLVM_GLOBAL_TYPE)
            .unwrap()
            .get_type(ctx)
    }

    pub fn get_linkage(&self, ctx: &Context) -> LinkageAttr {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<LinkageAttr>(&ATTR_KEY_LLVM_LINKAGE)
            .cloned()
            .unwrap_or_default()
    }

    pub fn set_initializer_bytes(&self, ctx: &mut Context, bytes: Vec<u8>) {
        self.get_operation()
            .deref_mut(ctx)
            .attributes
            .set(ATTR_KEY_LLVM_GLOBAL_INIT.clone(), BytesAttr(bytes));
    }

    pub fn get_initializer_bytes(&self, ctx: &Context) -> Option<Vec<u8>> {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<BytesAttr>(&ATTR_KEY_LLVM_GLOBAL_INIT)
            .map(|attr| attr.0.clone())
    }
}

#[op_interface_impl]
impl IsDeclaration for GlobalOp {
    fn is_declaration(&self, ctx: &Context) -> bool {
        // A global is a declaration if it has no initializer region
        self.get_operation().deref(ctx).regions().next().is_none()
    }
}

impl Printable for GlobalOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "{} ", self.get_opid().disp(ctx))?;
        write!(f, "{} ", self.get_linkage(ctx))?;
        write!(f, "@{} : ", self.get_symbol_name(ctx))?;
        self.get_global_type(ctx).fmt(ctx, state, f)?;
        if let Some(bytes) = self.get_initializer_bytes(ctx) {
            write!(f, " = ")?;
            BytesAttr(bytes).fmt(ctx, state, f)?;
        }
        Ok(())
    }
}

impl Parsable for GlobalOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        if !results.is_empty() {
            input_err!(
                state_stream.loc(),
                op_interfaces::NResultsVerifyErr(0, results.len())
            )?
        }

        let mut parser = spaced(LinkageAttr::parser(()))
            .and(spaced(token('@').with(Identifier::parser(()))))
            .skip(spaced(token(':')))
            .and(spaced(type_parser()))
            .and(optional(spaced(token('=').with(BytesAttr::parser(())))));

        parser
            .parse_stream(state_stream)
            .map(|(((linkage, name), global_type), init)| -> OpObj {
                let ctx = &mut *state_stream.state.ctx;
                let global = GlobalOp::new(ctx, name, global_type, linkage);
                if let Some(init) = init {
                    global.set_initializer_bytes(ctx, init.0);
                }
                OpObj::new(global)
            })
            .into()
    }
}

impl_verify_succ!(GlobalOp);

/// Pointer to a null-terminated string literal.
#[def_op("llvm.cstr")]
#[derive_op_interface_impl(OneResultInterface, NOpdsInterface<0>)]
pub struct CStrOp;

#[op_interface_impl]
impl PointerTypeResult for CStrOp {}

impl CStrOp {
    pub fn new(ctx: &mut Context, value: String) -> Self {
        let ptr_ty: TypeHandle = PointerType::get(ctx).into();
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![ptr_ty],
            vec![],
            vec![],
            0,
        );
        let cstr = CStrOp { op };
        cstr.get_operation()
            .deref_mut(ctx)
            .attributes
            .set(ATTR_KEY_LLVM_CSTR_VALUE.clone(), StringAttr::new(value));
        cstr
    }

    pub fn get_value(&self, ctx: &Context) -> String {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<StringAttr>(&ATTR_KEY_LLVM_CSTR_VALUE)
            .cloned()
            .map(String::from)
            .unwrap()
    }
}

impl Printable for CStrOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        self.get_result(ctx).fmt(ctx, state, f)?;
        write!(f, " = {} ", self.get_opid().disp(ctx))?;
        StringAttr::new(self.get_value(ctx)).fmt(ctx, state, f)
    }
}

impl Parsable for CStrOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let mut parser = spaced(StringAttr::parser(()));
        parser
            .parse_stream(state_stream)
            .map(|attr| -> OpObj {
                let ctx = &mut *state_stream.state.ctx;
                let op = CStrOp::new(ctx, attr.into());
                if !results.is_empty() {
                    process_parsed_ssa_defs(state_stream, &results, op.get_operation()).ok();
                }
                OpObj::new(op)
            })
            .into()
    }
}

impl_verify_succ!(CStrOp);

dict_key!(ATTR_KEY_LLVM_FUNC_TYPE, "llvm_func_type");

/// LLVM function definition/declaration.
#[def_op("llvm.func")]
#[derive_op_interface_impl(
    OneRegionInterface,
    SymbolOpInterface,
    IsolatedFromAboveInterface,
    NOpdsInterface<0>,
    NResultsInterface<0>
)]
pub struct FuncOp;

impl FuncOp {
    pub fn new(
        ctx: &mut Context,
        name: Identifier,
        func_type: TypedHandle<FuncType>,
        linkage: LinkageAttr,
    ) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 1);

        // Create an empty entry block with arguments matching the function type.
        let arg_types = func_type.deref(ctx).arg_types();
        let region = op.deref_mut(ctx).get_region(0);
        let body = BasicBlock::new(ctx, Some("entry".try_into().unwrap()), arg_types);
        body.insert_at_front(region, ctx);

        let func = FuncOp { op };
        func.set_symbol_name(ctx, name);
        {
            let mut op_ref = func.get_operation().deref_mut(ctx);
            op_ref.attributes.set(
                ATTR_KEY_LLVM_FUNC_TYPE.clone(),
                TypeAttr::new(func_type.into()),
            );
            op_ref
                .attributes
                .set(ATTR_KEY_LLVM_LINKAGE.clone(), linkage);
        }
        func
    }

    /// Create a function declaration (external linkage, no body).
    pub fn new_declaration(
        ctx: &mut Context,
        name: Identifier,
        func_type: TypedHandle<FuncType>,
        linkage: LinkageAttr,
    ) -> Self {
        // Create with 1 region but do NOT insert any blocks — this makes is_declaration() true.
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 1);

        let func = FuncOp { op };
        func.set_symbol_name(ctx, name);
        {
            let mut op_ref = func.get_operation().deref_mut(ctx);
            op_ref.attributes.set(
                ATTR_KEY_LLVM_FUNC_TYPE.clone(),
                TypeAttr::new(func_type.into()),
            );
            op_ref
                .attributes
                .set(ATTR_KEY_LLVM_LINKAGE.clone(), linkage);
        }
        func
    }

    pub fn get_func_type(&self, ctx: &Context) -> TypeHandle {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<TypeAttr>(&ATTR_KEY_LLVM_FUNC_TYPE)
            .unwrap()
            .get_type(ctx)
    }

    pub fn get_linkage(&self, ctx: &Context) -> LinkageAttr {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<LinkageAttr>(&ATTR_KEY_LLVM_LINKAGE)
            .cloned()
            .unwrap_or_default()
    }

    pub fn get_entry_block(&self, ctx: &Context) -> Ptr<BasicBlock> {
        self.get_region(ctx).deref(ctx).get_head().unwrap()
    }
}

#[op_interface_impl]
impl IsDeclaration for FuncOp {
    fn is_declaration(&self, ctx: &Context) -> bool {
        // A function is a declaration if its region is empty (no blocks).
        self.get_region(ctx).deref(ctx).get_head().is_none()
    }
}

#[op_interface_impl]
impl FunctionLikeInterface for FuncOp {
    fn body_region(&self, ctx: &Context) -> Option<Ptr<Region>> {
        (!self.is_declaration(ctx)).then(|| self.get_region(ctx))
    }
}

impl Printable for FuncOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "{} ", self.get_opid().disp(ctx))?;
        write!(f, "{} ", self.get_linkage(ctx))?;
        write!(f, "@{} : ", self.get_symbol_name(ctx))?;
        self.get_func_type(ctx).fmt(ctx, state, f)?;
        write!(f, " ")?;
        region(self).fmt(ctx, state, f)?;
        Ok(())
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
            input_err!(
                state_stream.loc(),
                op_interfaces::NResultsVerifyErr(0, results.len())
            )?
        }

        let op = Operation::new(
            state_stream.state.ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );

        let mut parser = (
            spaced(LinkageAttr::parser(())),
            spaced(token('@').with(Identifier::parser(()))),
            spaced(token(':')).with(spaced(type_parser())),
            spaced(Region::parser(op)),
        );

        parser
            .parse_stream(state_stream)
            .map(|(linkage, name, func_type, _region)| -> OpObj {
                let ctx = &mut *state_stream.state.ctx;
                let opop = FuncOp { op };
                opop.set_symbol_name(ctx, name);
                {
                    let mut op_ref = opop.get_operation().deref_mut(ctx);
                    op_ref
                        .attributes
                        .set(ATTR_KEY_LLVM_FUNC_TYPE.clone(), TypeAttr::new(func_type));
                    op_ref
                        .attributes
                        .set(ATTR_KEY_LLVM_LINKAGE.clone(), linkage);
                }
                OpObj::new(opop)
            })
            .into()
    }
}

impl_verify_succ!(FuncOp);

/// LLVM addressof operation: get the address of a global or function symbol.
#[def_op("llvm.addressof")]
#[derive_op_interface_impl(OneResultInterface, NOpdsInterface<0>)]
pub struct AddressOfOp;

#[op_interface_impl]
impl PointerTypeResult for AddressOfOp {}

impl AddressOfOp {
    pub fn new(ctx: &mut Context, symbol: Identifier) -> Self {
        let ptr_ty: TypeHandle = PointerType::get(ctx).into();
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![ptr_ty],
            vec![],
            vec![],
            0,
        );
        let addr = AddressOfOp { op };
        addr.get_operation()
            .deref_mut(ctx)
            .attributes
            .set(ATTR_KEY_SYM_NAME.clone(), IdentifierAttr::new(symbol));
        addr
    }

    pub fn get_symbol(&self, ctx: &Context) -> Identifier {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<IdentifierAttr>(&ATTR_KEY_SYM_NAME)
            .cloned()
            .unwrap()
            .into()
    }
}

impl Printable for AddressOfOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        self.get_result(ctx).fmt(ctx, state, f)?;
        write!(
            f,
            " = {} @{}",
            self.get_opid().disp(ctx),
            self.get_symbol(ctx)
        )
    }
}

impl Parsable for AddressOfOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let mut parser = spaced(token('@').with(Identifier::parser(())));

        parser
            .parse_stream(state_stream)
            .map(|symbol| -> OpObj {
                let ctx = &mut *state_stream.state.ctx;
                let op = AddressOfOp::new(ctx, symbol);
                if !results.is_empty() {
                    process_parsed_ssa_defs(state_stream, &results, op.get_operation()).ok();
                }
                OpObj::new(op)
            })
            .into()
    }
}

impl_verify_succ!(AddressOfOp);

// ============================================================================
// ExtractValueOp
// ============================================================================

dict_key!(ATTR_KEY_EXTRACTVALUE_INDICES, "llvm_extractvalue_indices");

/// LLVM extractvalue: extract a value from an aggregate (struct or array).
///
/// Example:
/// ```mlir
/// %val = llvm.extractvalue %agg [0, 1] : !llvm.struct<(ptr, ptr, i64)>
/// ```
#[def_op("llvm.extractvalue")]
#[derive_op_interface_impl(OneResultInterface)]
pub struct ExtractValueOp;

impl ExtractValueOp {
    pub fn new(
        ctx: &mut Context,
        aggregate: Value,
        indices: Vec<u32>,
        result_type: TypeHandle,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_type],
            vec![aggregate],
            vec![],
            0,
        );
        let ev = ExtractValueOp { op };
        ev.get_operation().deref_mut(ctx).attributes.set(
            ATTR_KEY_EXTRACTVALUE_INDICES.clone(),
            InsertExtractValueIndicesAttr(indices),
        );
        ev
    }

    pub fn get_aggregate(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    pub fn get_indices(&self, ctx: &Context) -> Vec<u32> {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<InsertExtractValueIndicesAttr>(&ATTR_KEY_EXTRACTVALUE_INDICES)
            .map(|a| a.0.clone())
            .unwrap_or_default()
    }
}

impl Printable for ExtractValueOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        self.get_result(ctx).fmt(ctx, state, f)?;
        write!(f, " = {} ", self.get_opid().disp(ctx))?;
        self.get_aggregate(ctx).fmt(ctx, state, f)?;
        write!(f, " ")?;
        let indices = self.get_indices(ctx);
        InsertExtractValueIndicesAttr(indices).fmt(ctx, state, f)?;
        write!(f, " : ")?;
        self.get_aggregate(ctx).get_type(ctx).fmt(ctx, state, f)?;
        Ok(())
    }
}

impl Parsable for ExtractValueOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let mut parser = spaced(ssa_opd_parser())
            .and(spaced(InsertExtractValueIndicesAttr::parser(())))
            .skip(spaced(token(':')))
            .and(spaced(type_parser()));

        parser
            .parse_stream(state_stream)
            .map(|((aggregate, indices_attr), agg_type)| -> OpObj {
                let ctx = &mut *state_stream.state.ctx;
                // For now, use agg_type as result type placeholder.
                // The verifier should resolve the actual indexed type.
                let op = ExtractValueOp::new(ctx, aggregate, indices_attr.0, agg_type);
                if !results.is_empty() {
                    process_parsed_ssa_defs(state_stream, &results, op.get_operation()).ok();
                }
                OpObj::new(op)
            })
            .into()
    }
}

impl_verify_succ!(ExtractValueOp);

// ============================================================================
// InsertValueOp
// ============================================================================

dict_key!(ATTR_KEY_INSERTVALUE_INDICES, "llvm_insertvalue_indices");

/// LLVM insertvalue: insert a value into an aggregate (struct or array).
///
/// Example:
/// ```mlir
/// %new_agg = llvm.insertvalue %val, %agg [0] : !llvm.struct<(ptr, ptr, i64)>
/// ```
#[def_op("llvm.insertvalue")]
#[derive_op_interface_impl(OneResultInterface)]
pub struct InsertValueOp;

impl InsertValueOp {
    pub fn new(ctx: &mut Context, value: Value, aggregate: Value, indices: Vec<u32>) -> Self {
        let result_type = aggregate.get_type(ctx);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_type],
            vec![value, aggregate],
            vec![],
            0,
        );
        let iv = InsertValueOp { op };
        iv.get_operation().deref_mut(ctx).attributes.set(
            ATTR_KEY_INSERTVALUE_INDICES.clone(),
            InsertExtractValueIndicesAttr(indices),
        );
        iv
    }

    pub fn get_value(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    pub fn get_aggregate(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }

    pub fn get_indices(&self, ctx: &Context) -> Vec<u32> {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<InsertExtractValueIndicesAttr>(&ATTR_KEY_INSERTVALUE_INDICES)
            .map(|a| a.0.clone())
            .unwrap_or_default()
    }
}

impl Printable for InsertValueOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        self.get_result(ctx).fmt(ctx, state, f)?;
        write!(f, " = {} ", self.get_opid().disp(ctx))?;
        self.get_value(ctx).fmt(ctx, state, f)?;
        write!(f, ", ")?;
        self.get_aggregate(ctx).fmt(ctx, state, f)?;
        write!(f, " ")?;
        let indices = self.get_indices(ctx);
        InsertExtractValueIndicesAttr(indices).fmt(ctx, state, f)?;
        write!(f, " : ")?;
        self.get_aggregate(ctx).get_type(ctx).fmt(ctx, state, f)?;
        Ok(())
    }
}

impl Parsable for InsertValueOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let mut parser = spaced(ssa_opd_parser())
            .skip(spaced(token(',')))
            .and(spaced(ssa_opd_parser()))
            .and(spaced(InsertExtractValueIndicesAttr::parser(())))
            .skip(spaced(token(':')))
            .skip(spaced(type_parser()));

        parser
            .parse_stream(state_stream)
            .map(|((value, aggregate), indices_attr)| -> OpObj {
                let ctx = &mut *state_stream.state.ctx;
                let op = InsertValueOp::new(ctx, value, aggregate, indices_attr.0);
                if !results.is_empty() {
                    process_parsed_ssa_defs(state_stream, &results, op.get_operation()).ok();
                }
                OpObj::new(op)
            })
            .into()
    }
}

impl_verify_succ!(InsertValueOp);

// ============================================================================
// Registration
// ============================================================================

pub fn register(ctx: &mut Context) {
    // Terminators

    // Integer arithmetic with overflow flags

    // Integer arithmetic without overflow flags

    // Float arithmetic

    // Cast operations

    // Comparison

    // Memory

    // Control flow

    // Call

    // Constants

    // Aggregate

    // Global / Function / AddressOf
}

#[cfg(test)]
mod tests {
    use combine::Parser;

    use crate::{
        context::Context,
        dialects::{builtin, llvm},
        ir::{
            location,
            operation::{Operation, OperationParserConfig},
        },
        parsable::{Parsable, State, state_stream_from_iterator},
        printable::Printable,
    };

    #[test]
    fn parses_printed_direct_call_forms() {
        let mut ctx = Context::new();
        llvm::register(&mut ctx);

        let input = "builtin.module @m {
  ^entry(arg0: llvm.ptr , arg1: builtin.integer ui64):
    op0 = llvm.call @alloc(arg1, arg1) : llvm.ptr ;
    llvm.call @dealloc(op0, arg1);
    llvm.return
}";
        let state_stream = state_stream_from_iterator(
            input.chars(),
            State::new(&mut ctx, location::Source::InMemory),
        );
        let config = OperationParserConfig {
            look_for_outlined_attrs: false,
        };

        let parsed = <Operation as Parsable>::parser(config)
            .parse(state_stream)
            .unwrap()
            .0;
        let printed = parsed.disp(&ctx).to_string();
        assert!(printed.contains(" = llvm.call @alloc"));
        assert!(printed.contains("llvm.call @dealloc"));
    }
}
