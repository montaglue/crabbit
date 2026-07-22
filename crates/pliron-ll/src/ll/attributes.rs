//! Attributes of the `ll` dialect: payloads shared by the machine-level
//! backends and the LLVM-level extension ops.

use combine::{Parser, many1, satisfy};

use pliron::derive::pliron_attr;
use pliron::{
    context::Context,
    location::Located,
    input_err,
    parsable::{Parsable, ParseResult, StateStream},
    printable::{self, Printable},
};

/// A raw binary payload — encoded machine code, literal-pool data, or a
/// global's initializer bytes. Stored as bytes so passes hand it around
/// without re-encoding; hex exists only in the textual IR, spelled `0x`
/// followed by two lowercase digits per byte (`0x` alone is the empty
/// payload).
#[pliron_attr(name = "ll.bytes", verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct BytesAttr(pub Vec<u8>);

impl BytesAttr {
    /// Decode the canonical textual spelling (`0x…`). `None` if the prefix is
    /// missing, a digit is not hex, or the digit count is odd.
    pub fn parse_str(text: &str) -> Option<Self> {
        let digits = text.strip_prefix("0x")?;
        if digits.len() % 2 != 0 {
            return None;
        }
        digits
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                let pair = core::str::from_utf8(pair).ok()?;
                u8::from_str_radix(pair, 16).ok()
            })
            .collect::<Option<Vec<u8>>>()
            .map(BytesAttr)
    }
}

impl core::fmt::Display for BytesAttr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "0x")?;
        for byte in &self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl Printable for BytesAttr {
    fn fmt(
        &self,
        _ctx: &Context,
        _state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "{self}")
    }
}

impl Parsable for BytesAttr {
    type Arg = ();
    type Parsed = Self;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let loc = state_stream.loc();
        let mut parser = many1::<String, _, _>(satisfy(|c: char| c.is_ascii_alphanumeric()));
        let (text, commit) = parser.parse_stream(state_stream).into_result()?;
        let Some(bytes) = Self::parse_str(&text) else {
            input_err!(
                loc,
                "invalid bytes literal `{}`: expected `0x` followed by an even number of hex digits",
                text
            )?
        };
        Ok((bytes, commit))
    }
}

/// Relative weights of a terminator's successors, one per successor.
/// The probability of an edge is its weight divided by the sum of all weights.
/// Same as LLVM's `!prof branch_weights` metadata and the `branch_weights`
/// attribute consumed by MLIR's `WeightedBranchOpInterface`.
#[pliron_attr(
    name = "ll.branch_weights",
    format = "`[` vec($0, CharSpace(`,`)) `]`",
    verifier = "succ"
)]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct BranchWeightsAttr(pub Vec<u32>);

/// Linkage of a machine-level function symbol.
#[pliron_attr(name = "ll.linkage", verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash, Default)]
pub enum LinkageAttr {
    #[default]
    External,
    Internal,
    Private,
}

impl LinkageAttr {
    pub fn parse_str(text: &str) -> Option<Self> {
        match text {
            "external" => Some(Self::External),
            "internal" => Some(Self::Internal),
            "private" => Some(Self::Private),
            _ => None,
        }
    }
}

impl core::fmt::Display for LinkageAttr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::External => write!(f, "external"),
            Self::Internal => write!(f, "internal"),
            Self::Private => write!(f, "private"),
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
        write!(f, "{self}")
    }
}

impl Parsable for LinkageAttr {
    type Arg = ();
    type Parsed = Self;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let loc = state_stream.loc();
        let mut parser = many1::<String, _, _>(satisfy(|c: char| c.is_ascii_alphanumeric()));
        let (text, commit) = parser.parse_stream(state_stream).into_result()?;
        let Some(linkage) = Self::parse_str(&text) else {
            input_err!(
                loc,
                "invalid linkage `{}`: expected `external`, `internal` or `private`",
                text
            )?
        };
        Ok((linkage, commit))
    }
}
