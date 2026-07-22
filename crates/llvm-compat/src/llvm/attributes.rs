//! Attributes belonging to the LLVM dialect.

use std::fmt::Display;

use combine::{
    Parser, attempt, between, choice, many,
    parser::char::{char, string},
    sep_by,
};
use pliron::derive::def_attribute;
use thiserror::Error;

use crate::{
    common_traits::Verify,
    context::Context,
    dialects::builtin::attributes::IntegerAttr,
    identifier::Identifier,
    impl_verify_succ,
    ir::attribute::Attribute,
    ir::irfmt::parsers::spaced,
    parsable::{Parsable, ParseResult, StateStream},
    printable::{self, Printable},
    result::STAIRResult,
    verify_err_noloc,
};

// TODO: FastmathFlags — requires pure-Rust bitflags reimplementation (pliron uses llvm_sys)

// ============================================================================
// IntegerOverflowFlagsAttr
// ============================================================================

/// Integer overflow flags for arithmetic operations.
/// "nsw" and "nuw" bits indicate that the operation is guaranteed to not overflow
/// (in the signed or unsigned case, respectively).
#[def_attribute("llvm.integer_overflow_flags")]
#[derive(PartialEq, Eq, Clone, Debug, Default, Hash)]
pub struct IntegerOverflowFlagsAttr {
    pub nsw: bool,
    pub nuw: bool,
}

impl IntegerOverflowFlagsAttr {
    pub fn new(nsw: bool, nuw: bool) -> Self {
        Self { nsw, nuw }
    }
}

impl Printable for IntegerOverflowFlagsAttr {
    fn fmt(
        &self,
        _ctx: &Context,
        _state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        let mut flags = Vec::new();
        if self.nsw {
            flags.push("nsw");
        }
        if self.nuw {
            flags.push("nuw");
        }
        write!(f, "<{}>", flags.join(" "))
    }
}

impl Parsable for IntegerOverflowFlagsAttr {
    type Arg = ();
    type Parsed = Self;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let flag_parser = choice((string("nsw").map(|_| "nsw"), string("nuw").map(|_| "nuw")));

        let mut parser = between(
            spaced(char('<')),
            spaced(char('>')),
            many::<Vec<_>, _, _>(spaced(flag_parser)),
        );

        parser
            .parse_stream(state_stream)
            .map(|flags| {
                let mut attr = IntegerOverflowFlagsAttr::default();
                for flag in flags {
                    match flag {
                        "nsw" => attr.nsw = true,
                        "nuw" => attr.nuw = true,
                        _ => {}
                    }
                }
                attr
            })
            .into()
    }
}

impl_verify_succ!(IntegerOverflowFlagsAttr);

// ============================================================================
// ICmpPredicateAttr
// ============================================================================

/// Integer comparison predicates.
#[def_attribute("llvm.icmp_predicate")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum ICmpPredicateAttr {
    EQ,
    NE,
    SLT,
    SLE,
    SGT,
    SGE,
    ULT,
    ULE,
    UGT,
    UGE,
}

impl Display for ICmpPredicateAttr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EQ => write!(f, "eq"),
            Self::NE => write!(f, "ne"),
            Self::SLT => write!(f, "slt"),
            Self::SLE => write!(f, "sle"),
            Self::SGT => write!(f, "sgt"),
            Self::SGE => write!(f, "sge"),
            Self::ULT => write!(f, "ult"),
            Self::ULE => write!(f, "ule"),
            Self::UGT => write!(f, "ugt"),
            Self::UGE => write!(f, "uge"),
        }
    }
}

impl Printable for ICmpPredicateAttr {
    fn fmt(
        &self,
        _ctx: &Context,
        _state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "{}", self)
    }
}

impl Parsable for ICmpPredicateAttr {
    type Arg = ();
    type Parsed = Self;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        // Parse predicates with shared prefixes in an order that does not consume
        // the common prefix before trying the final character.
        let mut parser = choice((
            attempt(string("sle")).map(|_| Self::SLE),
            attempt(string("slt")).map(|_| Self::SLT),
            attempt(string("sge")).map(|_| Self::SGE),
            attempt(string("sgt")).map(|_| Self::SGT),
            attempt(string("ule")).map(|_| Self::ULE),
            attempt(string("ult")).map(|_| Self::ULT),
            attempt(string("uge")).map(|_| Self::UGE),
            attempt(string("ugt")).map(|_| Self::UGT),
            attempt(string("eq")).map(|_| Self::EQ),
            attempt(string("ne")).map(|_| Self::NE),
        ));

        parser.parse_stream(state_stream).into()
    }
}

impl_verify_succ!(ICmpPredicateAttr);

// ============================================================================
// FCmpPredicateAttr
// ============================================================================

/// Floating-point comparison predicates.
#[def_attribute("llvm.fcmp_predicate")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum FCmpPredicateAttr {
    False,
    OEQ,
    OGT,
    OGE,
    OLT,
    OLE,
    ONE,
    ORD,
    UEQ,
    UGT,
    UGE,
    ULT,
    ULE,
    UNE,
    UNO,
    True,
}

impl Display for FCmpPredicateAttr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::False => write!(f, "false"),
            Self::OEQ => write!(f, "oeq"),
            Self::OGT => write!(f, "ogt"),
            Self::OGE => write!(f, "oge"),
            Self::OLT => write!(f, "olt"),
            Self::OLE => write!(f, "ole"),
            Self::ONE => write!(f, "one"),
            Self::ORD => write!(f, "ord"),
            Self::UEQ => write!(f, "ueq"),
            Self::UGT => write!(f, "ugt"),
            Self::UGE => write!(f, "uge"),
            Self::ULT => write!(f, "ult"),
            Self::ULE => write!(f, "ule"),
            Self::UNE => write!(f, "une"),
            Self::UNO => write!(f, "uno"),
            Self::True => write!(f, "true"),
        }
    }
}

impl Printable for FCmpPredicateAttr {
    fn fmt(
        &self,
        _ctx: &Context,
        _state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "{}", self)
    }
}

impl Parsable for FCmpPredicateAttr {
    type Arg = ();
    type Parsed = Self;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let mut parser = choice((
            attempt(string("false")).map(|_| Self::False),
            attempt(string("true")).map(|_| Self::True),
            attempt(string("oeq")).map(|_| Self::OEQ),
            attempt(string("oge")).map(|_| Self::OGE),
            attempt(string("ogt")).map(|_| Self::OGT),
            attempt(string("ole")).map(|_| Self::OLE),
            attempt(string("olt")).map(|_| Self::OLT),
            attempt(string("one")).map(|_| Self::ONE),
            attempt(string("ord")).map(|_| Self::ORD),
            attempt(string("ueq")).map(|_| Self::UEQ),
            attempt(string("uge")).map(|_| Self::UGE),
            attempt(string("ugt")).map(|_| Self::UGT),
            attempt(string("ule")).map(|_| Self::ULE),
            attempt(string("ult")).map(|_| Self::ULT),
            attempt(string("une")).map(|_| Self::UNE),
            attempt(string("uno")).map(|_| Self::UNO),
        ));

        parser.parse_stream(state_stream).into()
    }
}

impl_verify_succ!(FCmpPredicateAttr);

// ============================================================================
// GepIndexAttr / GepIndicesAttr
// ============================================================================

/// A single GEP index: either a compile-time constant or a reference to a runtime operand.
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum GepIndexAttr {
    /// A compile-time constant index.
    Constant(u32),
    /// An index that refers to an operand by its index in the operation.
    OperandIdx(usize),
}

/// GEP indices attribute: a list of GEP index entries.
#[def_attribute("llvm.gep_indices")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct GepIndicesAttr(pub Vec<GepIndexAttr>);

impl Printable for GepIndicesAttr {
    fn fmt(
        &self,
        _ctx: &Context,
        _state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "[")?;
        for (i, idx) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            match idx {
                GepIndexAttr::Constant(c) => write!(f, "{}", c)?,
                GepIndexAttr::OperandIdx(o) => write!(f, "${}", o)?,
            }
        }
        write!(f, "]")
    }
}

impl Parsable for GepIndicesAttr {
    type Arg = ();
    type Parsed = Self;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        use crate::ir::irfmt::parsers::int_parser;

        let index_parser = choice((
            char('$')
                .with(int_parser::<usize>())
                .map(GepIndexAttr::OperandIdx),
            int_parser::<u32>().map(GepIndexAttr::Constant),
        ));

        let mut parser = between(
            spaced(char('[')),
            spaced(char(']')),
            sep_by::<Vec<_>, _, _, _>(spaced(index_parser), spaced(char(','))),
        );

        parser.parse_stream(state_stream).map(GepIndicesAttr).into()
    }
}

impl_verify_succ!(GepIndicesAttr);

// ============================================================================
// LinkageAttr
// ============================================================================

/// LLVM linkage types for global variables and functions.
#[def_attribute("llvm.linkage")]
#[derive(PartialEq, Eq, Clone, Debug, Hash, Default)]
pub enum LinkageAttr {
    #[default]
    External,
    AvailableExternally,
    LinkOnceAny,
    LinkOnceODR,
    WeakAny,
    WeakODR,
    Appending,
    Internal,
    Private,
    DLLImport,
    DLLExport,
    ExternalWeak,
    Common,
}

impl Display for LinkageAttr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::External => write!(f, "external"),
            Self::AvailableExternally => write!(f, "available_externally"),
            Self::LinkOnceAny => write!(f, "linkonce"),
            Self::LinkOnceODR => write!(f, "linkonce_odr"),
            Self::WeakAny => write!(f, "weak"),
            Self::WeakODR => write!(f, "weak_odr"),
            Self::Appending => write!(f, "appending"),
            Self::Internal => write!(f, "internal"),
            Self::Private => write!(f, "private"),
            Self::DLLImport => write!(f, "dllimport"),
            Self::DLLExport => write!(f, "dllexport"),
            Self::ExternalWeak => write!(f, "extern_weak"),
            Self::Common => write!(f, "common"),
        }
    }
}

impl Printable for LinkageAttr {
    fn fmt(
        &self,
        _ctx: &Context,
        _state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "{}", self)
    }
}

impl Parsable for LinkageAttr {
    type Arg = ();
    type Parsed = Self;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let mut parser = choice((
            attempt(string("available_externally")).map(|_| Self::AvailableExternally),
            attempt(string("linkonce_odr")).map(|_| Self::LinkOnceODR),
            attempt(string("linkonce")).map(|_| Self::LinkOnceAny),
            attempt(string("weak_odr")).map(|_| Self::WeakODR),
            attempt(string("weak")).map(|_| Self::WeakAny),
            attempt(string("appending")).map(|_| Self::Appending),
            attempt(string("internal")).map(|_| Self::Internal),
            attempt(string("private")).map(|_| Self::Private),
            attempt(string("dllimport")).map(|_| Self::DLLImport),
            attempt(string("dllexport")).map(|_| Self::DLLExport),
            attempt(string("extern_weak")).map(|_| Self::ExternalWeak),
            attempt(string("external")).map(|_| Self::External),
            attempt(string("common")).map(|_| Self::Common),
        ));

        parser.parse_stream(state_stream).into()
    }
}

impl_verify_succ!(LinkageAttr);

// ============================================================================
// AlignmentAttr
// ============================================================================

/// Alignment attribute in bytes.
#[def_attribute("llvm.alignment")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct AlignmentAttr(pub u32);

impl Printable for AlignmentAttr {
    fn fmt(
        &self,
        _ctx: &Context,
        _state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Parsable for AlignmentAttr {
    type Arg = ();
    type Parsed = Self;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        use crate::ir::irfmt::parsers::int_parser;
        int_parser::<u32>()
            .map(AlignmentAttr)
            .parse_stream(state_stream)
            .into()
    }
}

impl_verify_succ!(AlignmentAttr);

// ============================================================================
// InsertExtractValueIndicesAttr
// ============================================================================

/// Indices for insertvalue/extractvalue operations.
#[def_attribute("llvm.insert_extract_value_indices")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct InsertExtractValueIndicesAttr(pub Vec<u32>);

impl Printable for InsertExtractValueIndicesAttr {
    fn fmt(
        &self,
        _ctx: &Context,
        _state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "[")?;
        for (i, idx) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", idx)?;
        }
        write!(f, "]")
    }
}

impl Parsable for InsertExtractValueIndicesAttr {
    type Arg = ();
    type Parsed = Self;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        use crate::ir::irfmt::parsers::int_parser;

        let mut parser = between(
            spaced(char('[')),
            spaced(char(']')),
            sep_by::<Vec<_>, _, _, _>(spaced(int_parser::<u32>()), spaced(char(','))),
        );

        parser
            .parse_stream(state_stream)
            .map(InsertExtractValueIndicesAttr)
            .into()
    }
}

impl_verify_succ!(InsertExtractValueIndicesAttr);

// ============================================================================
// PhiIncomingBlocksAttr
// ============================================================================

/// Predecessor block labels for `llvm.phi` incoming values.
#[def_attribute("llvm.phi_incoming_blocks")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct PhiIncomingBlocksAttr(pub Vec<Identifier>);

impl Printable for PhiIncomingBlocksAttr {
    fn fmt(
        &self,
        _ctx: &Context,
        _state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "[")?;
        for (idx, block) in self.0.iter().enumerate() {
            if idx > 0 {
                write!(f, ", ")?;
            }
            write!(f, "^{}", block)?;
        }
        write!(f, "]")
    }
}

impl Parsable for PhiIncomingBlocksAttr {
    type Arg = ();
    type Parsed = Self;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let mut parser = between(
            spaced(char('[')),
            spaced(char(']')),
            sep_by::<Vec<_>, _, _, _>(
                spaced(char('^')).with(Identifier::parser(())),
                spaced(char(',')),
            ),
        );

        parser
            .parse_stream(state_stream)
            .map(PhiIncomingBlocksAttr)
            .into()
    }
}

impl_verify_succ!(PhiIncomingBlocksAttr);

// ============================================================================
// CaseValuesAttr
// ============================================================================

/// Case values for switch operations: a list of integer attributes.
#[def_attribute("llvm.case_values")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct CaseValuesAttr(pub Vec<IntegerAttr>);

impl Printable for CaseValuesAttr {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "[")?;
        for (i, val) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            val.fmt(ctx, state, f)?;
        }
        write!(f, "]")
    }
}

impl Parsable for CaseValuesAttr {
    type Arg = ();
    type Parsed = Self;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let mut parser = between(
            spaced(char('[')),
            spaced(char(']')),
            sep_by::<Vec<_>, _, _, _>(spaced(IntegerAttr::parser(())), spaced(char(','))),
        );

        parser.parse_stream(state_stream).map(CaseValuesAttr).into()
    }
}

#[derive(Error, Debug)]
#[error("All case values in CaseValuesAttr must have the same type")]
pub struct CaseValuesAttrTypeMismatch;

impl Verify for CaseValuesAttr {
    fn verify(&self, _ctx: &Context) -> STAIRResult<()> {
        if self.0.len() > 1 {
            let first_ty = self.0[0].get_type();
            for val in &self.0[1..] {
                if val.get_type() != first_ty {
                    return verify_err_noloc!(CaseValuesAttrTypeMismatch);
                }
            }
        }
        Ok(())
    }
}

// ============================================================================
// ShuffleVectorMaskAttr
// ============================================================================

/// Mask for shufflevector operations.
#[def_attribute("llvm.shuffle_vector_mask")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct ShuffleVectorMaskAttr(pub Vec<i32>);

impl Printable for ShuffleVectorMaskAttr {
    fn fmt(
        &self,
        _ctx: &Context,
        _state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "[")?;
        for (i, val) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", val)?;
        }
        write!(f, "]")
    }
}

impl Parsable for ShuffleVectorMaskAttr {
    type Arg = ();
    type Parsed = Self;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        use crate::ir::irfmt::parsers::int_parser;

        let mut parser = between(
            spaced(char('[')),
            spaced(char(']')),
            sep_by::<Vec<_>, _, _, _>(spaced(int_parser::<i32>()), spaced(char(','))),
        );

        parser
            .parse_stream(state_stream)
            .map(ShuffleVectorMaskAttr)
            .into()
    }
}

impl_verify_succ!(ShuffleVectorMaskAttr);

// ============================================================================
// Registration
// ============================================================================

pub fn register(_ctx: &mut Context) {
}

#[cfg(test)]
mod tests {
    use combine::Parser;

    use crate::{
        context::Context,
        ir::location,
        parsable::{Parsable, State, state_stream_from_iterator},
    };

    use super::LinkageAttr;

    #[test]
    fn parses_external_linkage_without_extern_weak_prefix_failure() {
        let mut ctx = Context::new();
        let state_stream = state_stream_from_iterator(
            "external".chars(),
            State::new(&mut ctx, location::Source::InMemory),
        );

        let parsed = LinkageAttr::parser(()).parse(state_stream).unwrap().0;
        assert_eq!(parsed, LinkageAttr::External);
    }
}
