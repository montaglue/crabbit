use combine::{Parser, between, many1, optional, satisfy, sep_by, token};
use pliron::derive::def_attribute;
use thiserror::Error;

use crate::{
    common_traits::Verify,
    context::Context,
    impl_verify_succ, input_err,
    ir::irfmt::parsers::{int_parser, spaced},
    ir::location::Located,
    parsable::{Parsable, ParseResult, StateStream},
    printable::{self, Printable},
    result::STAIRResult,
    verify_err_noloc,
};

use super::{
    op_interfaces::{BinaryFixup, FixupKind},
    registers::Register,
};

/// A typed x86-64 register operand. Stored on instructions instead of a
/// string so that only values [Register] can represent exist in the IR;
/// malformed spellings are rejected at parse time.
#[def_attribute("x86_64.register")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct RegisterAttr(pub Register);

impl_verify_succ!(RegisterAttr);

impl Printable for RegisterAttr {
    fn fmt(
        &self,
        _ctx: &Context,
        _state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Parsable for RegisterAttr {
    type Arg = ();
    type Parsed = Self;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let loc = state_stream.loc();
        let mut parser = many1::<String, _, _>(satisfy(|c: char| c.is_ascii_alphanumeric()));
        let (text, commit) = parser.parse_stream(state_stream).into_result()?;
        let Some(register) = Register::parse(&text) else {
            input_err!(loc, "invalid x86-64 register `{text}`")?
        };
        Ok((RegisterAttr(register), commit))
    }
}

/// An x86-64 condition code: the `cc` nibble tested by `jcc`/`setcc`.
/// Discriminants are the hardware nibble, by suffix name (`e`, `ne`, `l`,
/// ...).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum ConditionCode {
    O = 0x0,
    No = 0x1,
    B = 0x2,
    Ae = 0x3,
    E = 0x4,
    Ne = 0x5,
    Be = 0x6,
    A = 0x7,
    S = 0x8,
    Ns = 0x9,
    P = 0xa,
    Np = 0xb,
    L = 0xc,
    Ge = 0xd,
    Le = 0xe,
    G = 0xf,
}

const CONDITION_NAMES: [&str; 16] = [
    "o", "no", "b", "ae", "e", "ne", "be", "a", "s", "ns", "p", "np", "l", "ge", "le", "g",
];

impl ConditionCode {
    pub const ALL: [ConditionCode; 16] = [
        Self::O,
        Self::No,
        Self::B,
        Self::Ae,
        Self::E,
        Self::Ne,
        Self::Be,
        Self::A,
        Self::S,
        Self::Ns,
        Self::P,
        Self::Np,
        Self::L,
        Self::Ge,
        Self::Le,
        Self::G,
    ];

    /// The `cc` nibble added to the `0f 8x` (jcc) / `0f 9x` (setcc) opcode
    /// bytes.
    pub fn encoding(self) -> u8 {
        self as u8
    }

    /// The condition that holds exactly when `self` does not (e<->ne, b<->ae,
    /// l<->ge, ...). Every x86 condition code has an inverse.
    pub fn invert(self) -> Self {
        match self {
            Self::O => Self::No,
            Self::No => Self::O,
            Self::B => Self::Ae,
            Self::Ae => Self::B,
            Self::E => Self::Ne,
            Self::Ne => Self::E,
            Self::Be => Self::A,
            Self::A => Self::Be,
            Self::S => Self::Ns,
            Self::Ns => Self::S,
            Self::P => Self::Np,
            Self::Np => Self::P,
            Self::L => Self::Ge,
            Self::Ge => Self::L,
            Self::Le => Self::G,
            Self::G => Self::Le,
        }
    }

    pub fn name(self) -> &'static str {
        CONDITION_NAMES[self as usize]
    }

    pub fn parse(text: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|condition| condition.name() == text)
    }
}

impl core::fmt::Display for ConditionCode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// A [ConditionCode] as an attribute. Stored on `jcc`/`cset` instead of a raw
/// immediate so an invalid condition code cannot exist in the IR; malformed
/// spellings are rejected at parse time.
#[def_attribute("x86_64.condition_code")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct ConditionCodeAttr(pub ConditionCode);

impl_verify_succ!(ConditionCodeAttr);

impl Printable for ConditionCodeAttr {
    fn fmt(
        &self,
        _ctx: &Context,
        _state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Parsable for ConditionCodeAttr {
    type Arg = ();
    type Parsed = Self;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let loc = state_stream.loc();
        let mut parser = many1::<String, _, _>(satisfy(|c: char| c.is_ascii_alphanumeric()));
        let (text, commit) = parser.parse_stream(state_stream).into_result()?;
        let Some(condition) = ConditionCode::parse(&text) else {
            input_err!(loc, "invalid x86-64 condition code `{text}`")?
        };
        Ok((ConditionCodeAttr(condition), commit))
    }
}

/// An argument or result location assigned by the Darwin x86-64 ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AbiLocation {
    /// A single general-purpose register.
    Gpr(Register),
    /// Two registers holding the low and high 8-byte halves of a value.
    GprPair(Register, Register),
    /// The caller passes a result pointer in `rdi` (the hidden first
    /// argument); the callee returns that pointer in `rax`.
    IndirectResult,
    /// Caller-owned stack memory at this byte offset into the outgoing
    /// argument area.
    Stack(u64),
    /// A zero-sized value with no location.
    Void,
}

/// Where a function's arguments and result live under the Darwin x86-64 ABI.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FunctionAbi {
    pub args: Vec<AbiLocation>,
    pub result: AbiLocation,
}

/// [FunctionAbi] as an attribute: the ABI pass computes it once per
/// `llvm.func` and instruction selection consumes it directly, so no
/// stringly-typed location can reach lowering.
#[def_attribute("x86_64.function_abi")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct FunctionAbiAttr(pub FunctionAbi);

#[derive(Debug, Error)]
#[error("the indirect-result location is not a valid argument location")]
pub struct FunctionAbiArgErr;

impl Verify for FunctionAbiAttr {
    fn verify(&self, _ctx: &Context) -> STAIRResult<()> {
        if self.0.args.contains(&AbiLocation::IndirectResult) {
            return verify_err_noloc!(FunctionAbiArgErr);
        }
        Ok(())
    }
}

impl Printable for AbiLocation {
    fn fmt(
        &self,
        _ctx: &Context,
        _state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        match self {
            Self::Gpr(reg) => write!(f, "{reg}"),
            Self::GprPair(lo, hi) => write!(f, "{lo}:{hi}"),
            Self::IndirectResult => write!(f, "sret"),
            Self::Stack(offset) => write!(f, "stack:{offset}"),
            Self::Void => write!(f, "void"),
        }
    }
}

impl Printable for FunctionAbiAttr {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "([")?;
        for (i, arg) in self.0.args.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            arg.fmt(ctx, state, f)?;
        }
        write!(f, "], ")?;
        self.0.result.fmt(ctx, state, f)?;
        write!(f, ")")
    }
}

/// Parses one [AbiLocation] using the same spellings [Printable] emits:
/// `void`, `sret`, `stack:<offset>`, `<reg>`, or `<reg>:<reg>`.
fn parse_abi_location<'a>(
    state_stream: &mut StateStream<'a>,
) -> ParseResult<'a, AbiLocation> {
    let loc = state_stream.loc();
    let mut word = many1::<String, _, _>(satisfy(|c: char| c.is_ascii_alphanumeric()));
    let (text, commit) = word.parse_stream(state_stream).into_result()?;
    match text.as_str() {
        "void" => Ok((AbiLocation::Void, commit)),
        "sret" => Ok((AbiLocation::IndirectResult, commit)),
        "stack" => {
            let (offset, commit) = token(':')
                .with(int_parser::<u64>())
                .parse_stream(state_stream)
                .into_result()?;
            Ok((AbiLocation::Stack(offset), commit))
        }
        first => {
            let Some(first) = Register::parse(first) else {
                input_err!(loc, "invalid x86-64 ABI location `{text}`")?
            };
            let (second, commit) = optional(token(':').with(many1::<String, _, _>(satisfy(
                |c: char| c.is_ascii_alphanumeric(),
            ))))
            .parse_stream(state_stream)
            .into_result()?;
            let Some(second) = second else {
                return Ok((AbiLocation::Gpr(first), commit));
            };
            let Some(second) = Register::parse(&second) else {
                input_err!(loc, "invalid x86-64 ABI location register `{second}`")?
            };
            Ok((AbiLocation::GprPair(first, second), commit))
        }
    }
}

impl Parsable for FunctionAbiAttr {
    type Arg = ();
    type Parsed = Self;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let mut parser = between(
            spaced(token('(')),
            spaced(token(')')),
            (
                between(
                    spaced(token('[')),
                    spaced(token(']')),
                    sep_by::<Vec<_>, _, _, _>(
                        spaced(combine::parser(parse_abi_location)),
                        spaced(token(',')),
                    ),
                ),
                spaced(token(',')).with(spaced(combine::parser(parse_abi_location))),
            ),
        );
        parser
            .parse_stream(state_stream)
            .map(|(args, result)| FunctionAbiAttr(FunctionAbi { args, result }))
            .into()
    }
}

/// The module's [BinaryFixup]s as an attribute: the encode pass records them
/// and MachO lowering consumes them directly, so a malformed fixup is a parse
/// error instead of a silently dropped relocation.
#[def_attribute("x86_64.fixups")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct FixupsAttr(pub Vec<BinaryFixup>);

impl_verify_succ!(FixupsAttr);

impl Printable for FixupsAttr {
    fn fmt(
        &self,
        _ctx: &Context,
        _state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "[")?;
        for (i, fixup) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            let kind = match fixup.kind {
                FixupKind::Branch32 => "branch32",
            };
            write!(f, "{kind}:{}:{}", fixup.offset, fixup.symbol)?;
        }
        write!(f, "]")
    }
}

/// Parses one [BinaryFixup] using the same `kind:offset:symbol` spelling
/// [Printable] emits, e.g. `branch32:11:puts`.
fn parse_fixup<'a>(state_stream: &mut StateStream<'a>) -> ParseResult<'a, BinaryFixup> {
    let loc = state_stream.loc();
    let mut kind_word = many1::<String, _, _>(satisfy(|c: char| c.is_ascii_alphanumeric()));
    let (kind, _commit) = kind_word.parse_stream(state_stream).into_result()?;
    let kind = match kind.as_str() {
        "branch32" => FixupKind::Branch32,
        _ => input_err!(loc, "invalid x86-64 fixup kind `{kind}`")?,
    };
    let (offset, _commit) = token(':')
        .with(int_parser::<u32>())
        .parse_stream(state_stream)
        .into_result()?;
    let (symbol, commit) = token(':')
        .with(many1::<String, _, _>(satisfy(|c: char| {
            c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '$'
        })))
        .parse_stream(state_stream)
        .into_result()?;
    Ok((
        BinaryFixup {
            offset,
            symbol,
            kind,
        },
        commit,
    ))
}

impl Parsable for FixupsAttr {
    type Arg = ();
    type Parsed = Self;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let mut parser = between(
            spaced(token('[')),
            spaced(token(']')),
            sep_by::<Vec<_>, _, _, _>(spaced(combine::parser(parse_fixup)), spaced(token(','))),
        );
        parser.parse_stream(state_stream).map(FixupsAttr).into()
    }
}

pub fn register(_ctx: &mut Context) {
}

#[cfg(test)]
mod tests {
    use combine::Parser;

    use crate::{
        context::Context,
        dialects::x86_64::{
            op_interfaces::{BinaryFixup, FixupKind},
            registers::{RDI, RDX, RSI},
        },
        ir::location,
        parsable::{Parsable, State, state_stream_from_iterator},
        printable::Printable,
    };

    use super::{AbiLocation, FixupsAttr, FunctionAbi, FunctionAbiAttr};

    #[test]
    fn function_abi_attr_round_trips_through_text() {
        let mut ctx = Context::new();
        let attr = FunctionAbiAttr(FunctionAbi {
            args: vec![
                AbiLocation::Gpr(RSI),
                AbiLocation::GprPair(RSI, RDX),
                AbiLocation::Stack(8),
                AbiLocation::Void,
            ],
            result: AbiLocation::IndirectResult,
        });
        let text = attr.disp(&ctx).to_string();
        assert_eq!(text, "([rsi, rsi:rdx, stack:8, void], sret)");

        let state_stream = state_stream_from_iterator(
            text.chars(),
            State::new(&mut ctx, location::Source::InMemory),
        );
        let parsed = FunctionAbiAttr::parser(()).parse(state_stream).unwrap().0;
        assert_eq!(parsed, attr);
        // rdi is taken by the hidden sret argument in this layout.
        assert_ne!(parsed.0.args[0], AbiLocation::Gpr(RDI));
    }

    #[test]
    fn fixups_attr_round_trips_through_text() {
        let mut ctx = Context::new();
        let attr = FixupsAttr(vec![
            BinaryFixup {
                offset: 11,
                symbol: "puts".to_string(),
                kind: FixupKind::Branch32,
            },
            BinaryFixup {
                offset: 27,
                symbol: "_my.helper$1".to_string(),
                kind: FixupKind::Branch32,
            },
        ]);
        let text = attr.disp(&ctx).to_string();
        assert_eq!(text, "[branch32:11:puts, branch32:27:_my.helper$1]");

        let state_stream = state_stream_from_iterator(
            text.chars(),
            State::new(&mut ctx, location::Source::InMemory),
        );
        let parsed = FixupsAttr::parser(()).parse(state_stream).unwrap().0;
        assert_eq!(parsed, attr);
    }

    #[test]
    fn fixups_attr_parses_empty_list() {
        let mut ctx = Context::new();
        let attr = FixupsAttr(vec![]);
        let text = attr.disp(&ctx).to_string();
        assert_eq!(text, "[]");

        let state_stream = state_stream_from_iterator(
            text.chars(),
            State::new(&mut ctx, location::Source::InMemory),
        );
        let parsed = FixupsAttr::parser(()).parse(state_stream).unwrap().0;
        assert_eq!(parsed, attr);
    }

    #[test]
    fn fixups_attr_rejects_unknown_kind() {
        let mut ctx = Context::new();
        let state_stream = state_stream_from_iterator(
            "[call26:4:puts]".chars(),
            State::new(&mut ctx, location::Source::InMemory),
        );
        assert!(FixupsAttr::parser(()).parse(state_stream).is_err());
    }

    #[test]
    fn function_abi_attr_parses_empty_args() {
        let mut ctx = Context::new();
        let attr = FunctionAbiAttr(FunctionAbi {
            args: vec![],
            result: AbiLocation::Void,
        });
        let text = attr.disp(&ctx).to_string();
        assert_eq!(text, "([], void)");

        let state_stream = state_stream_from_iterator(
            text.chars(),
            State::new(&mut ctx, location::Source::InMemory),
        );
        let parsed = FunctionAbiAttr::parser(()).parse(state_stream).unwrap().0;
        assert_eq!(parsed, attr);
    }
}
