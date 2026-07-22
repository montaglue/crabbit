//! Attributes defined by the Rust MIR dialect.

use combine::{Parser, between, parser::char::char, sep_by};
use pliron::derive::def_attribute;

use crate::{
    context::Context,
    impl_verify_succ,
    ir::irfmt::parsers::{int_parser, spaced},
    parsable::{Parsable, ParseResult, StateStream},
    printable::{self, Printable},
};

/// Indices for MIR aggregate insert/extract operations.
#[def_attribute("mir.insert_extract_value_indices")]
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
        for (idx, value) in self.0.iter().enumerate() {
            if idx > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{value}")?;
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

pub fn register(_ctx: &mut Context) {
}
