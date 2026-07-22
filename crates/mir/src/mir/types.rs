//! Types defined by the Rust MIR dialect.

use combine::{
    Parser, between, optional,
    parser::char::{char, string},
};
use pliron::derive::def_type;

use crate::{
    context::Context,
    impl_verify_succ,
    ir::{
        irfmt::parsers::{spaced, type_parser},
        r#type::{Type, TypeHandle, TypedHandle},
    },
    parsable::{Parsable, ParseResult, StateStream},
    printable::{self, Printable},
};

/// Pointer/reference-like MIR place type.
#[def_type("mir.ptr")]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct PtrType {
    elem: TypeHandle,
    mutable: bool,
}

impl PtrType {
    pub fn get(ctx: &mut Context, elem: TypeHandle, mutable: bool) -> TypedHandle<Self> {
        Type::register_instance(Self { elem, mutable }, ctx)
    }

    pub fn elem_type(&self) -> TypeHandle {
        self.elem
    }

    pub fn is_mutable(&self) -> bool {
        self.mutable
    }
}

impl Printable for PtrType {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "<")?;
        if self.mutable {
            write!(f, "mut ")?;
        }
        self.elem.fmt(ctx, state, f)?;
        write!(f, ">")
    }
}

impl Parsable for PtrType {
    type Arg = ();
    type Parsed = TypedHandle<Self>;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let mut parser = between(
            spaced(char('<')),
            spaced(char('>')),
            optional(spaced(string("mut"))).and(spaced(type_parser())),
        );

        parser
            .parse_stream(state_stream)
            .map(|(mutable, elem)| PtrType::get(state_stream.state.ctx, elem, mutable.is_some()))
            .into()
    }
}

impl_verify_succ!(PtrType);

pub fn register(_ctx: &mut Context) {
}
