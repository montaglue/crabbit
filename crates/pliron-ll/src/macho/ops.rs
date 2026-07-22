use pliron::derive::{def_op, derive_op_interface_impl};

use crate::{
    context::Context,
    dialects::builtin::{
        attributes::{StringAttr},
        op_interfaces::{SymbolOpInterface, NOpdsInterface, NResultsInterface},
    },
    dict_key,
    identifier::Identifier,
    impl_verify_succ, input_err,
    ir::{
        location::{Located, Location},
        op::{Op, OpObj},
        operation::Operation,
    },
    parsable::{Parsable, ParseResult, StateStream},
    printable::{self, Printable},
};

dict_key!(ATTR_KEY_MACHO_TEXT, "macho_text");
dict_key!(ATTR_KEY_MACHO_SYMBOLS, "macho_symbols");
dict_key!(ATTR_KEY_MACHO_RELOCATIONS, "macho_relocations");

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Symbol {
    pub name: String,
    pub offset: u64,
    pub external: bool,
    pub defined: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Relocation {
    pub offset: u32,
    pub symbol: String,
    pub pcrel: bool,
    pub length: u8,
    pub extern_: bool,
    pub kind: u8,
}

#[def_op("macho.object")]
#[derive_op_interface_impl(SymbolOpInterface, NOpdsInterface<0>, NResultsInterface<0>)]
pub struct ObjectOp;

impl ObjectOp {
    pub fn new(ctx: &mut Context, name: Identifier, text: Vec<u8>, symbols: Vec<Symbol>) -> Self {
        Self::new_with_relocations(ctx, name, text, symbols, Vec::new())
    }

    pub fn new_with_relocations(
        ctx: &mut Context,
        name: Identifier,
        text: Vec<u8>,
        symbols: Vec<Symbol>,
        relocations: Vec<Relocation>,
    ) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        let object = Self { op };
        object.set_symbol_name(ctx, name);
        object
            .get_operation()
            .deref_mut(ctx)
            .attributes
            .set(ATTR_KEY_MACHO_TEXT.clone(), BytesAttr(text));
        object.get_operation().deref_mut(ctx).attributes.set(
            ATTR_KEY_MACHO_SYMBOLS.clone(),
            StringAttr::new(encode_symbols(&symbols)),
        );
        object.get_operation().deref_mut(ctx).attributes.set(
            ATTR_KEY_MACHO_RELOCATIONS.clone(),
            StringAttr::new(encode_relocations(&relocations)),
        );
        object
    }

    pub fn text(&self, ctx: &Context) -> Vec<u8> {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<BytesAttr>(&ATTR_KEY_MACHO_TEXT)
            .map(|attr| attr.0.clone())
            .unwrap_or_default()
    }

    pub fn symbols(&self, ctx: &Context) -> Vec<Symbol> {
        let symbols = self
            .get_operation()
            .deref(ctx)
            .attributes
            .get::<StringAttr>(&ATTR_KEY_MACHO_SYMBOLS)
            .map(|attr| -> String { attr.clone().into() })
            .unwrap_or_default();
        decode_symbols(&symbols)
    }

    pub fn relocations(&self, ctx: &Context) -> Vec<Relocation> {
        let relocations = self
            .get_operation()
            .deref(ctx)
            .attributes
            .get::<StringAttr>(&ATTR_KEY_MACHO_RELOCATIONS)
            .map(|attr| -> String { attr.clone().into() })
            .unwrap_or_default();
        decode_relocations(&relocations)
    }
}

impl Printable for ObjectOp {
    fn fmt(
        &self,
        ctx: &Context,
        _state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(
            f,
            "{} @{} text_bytes={} symbols={}",
            self.get_opid().disp(ctx),
            self.get_symbol_name(ctx),
            self.text(ctx).len(),
            self.symbols(ctx).len()
        )
    }
}

impl Parsable for ObjectOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        input_err!(
            state_stream.loc(),
            "macho.object parsing is not implemented"
        )?
    }
}

impl_verify_succ!(ObjectOp);

fn encode_symbols(symbols: &[Symbol]) -> String {
    symbols
        .iter()
        .map(|symbol| {
            format!(
                "{}:{}:{}:{}",
                symbol.name,
                symbol.offset,
                if symbol.external { 1 } else { 0 },
                if symbol.defined { 1 } else { 0 }
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn decode_symbols(symbols: &str) -> Vec<Symbol> {
    symbols
        .lines()
        .filter_map(|line| {
            let mut parts = line.split(':');
            let name = parts.next()?.to_string();
            let offset = parts.next()?.parse().ok()?;
            let external = parts.next()? == "1";
            let defined = parts.next().is_none_or(|value| value == "1");
            Some(Symbol {
                name,
                offset,
                external,
                defined,
            })
        })
        .collect()
}

fn encode_relocations(relocations: &[Relocation]) -> String {
    relocations
        .iter()
        .map(|relocation| {
            format!(
                "{}:{}:{}:{}:{}:{}",
                relocation.offset,
                relocation.symbol,
                if relocation.pcrel { 1 } else { 0 },
                relocation.length,
                if relocation.extern_ { 1 } else { 0 },
                relocation.kind
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn decode_relocations(relocations: &str) -> Vec<Relocation> {
    relocations
        .lines()
        .filter_map(|line| {
            let mut parts = line.split(':');
            Some(Relocation {
                offset: parts.next()?.parse().ok()?,
                symbol: parts.next()?.to_string(),
                pcrel: parts.next()? == "1",
                length: parts.next()?.parse().ok()?,
                extern_: parts.next()? == "1",
                kind: parts.next()?.parse().ok()?,
            })
        })
        .collect()
}

pub fn register(_ctx: &mut Context) {
}

use llvm_compat::ll::{BytesAttr};
