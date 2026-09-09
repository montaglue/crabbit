//! Attributes of the `ll` dialect: payloads shared by the machine-level
//! backends and the LLVM-level extension ops.

use combine::{Parser, between, many1, optional, satisfy, sep_by, token};

use pliron::derive::pliron_attr;
use pliron::{
    context::Context,
    irfmt::parsers::{int_parser, spaced},
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

/// A pointer slot inside a [DataAttr] initializer: the 8 bytes at `offset`
/// are the address of `symbol` plus `addend`, filled in by the linker
/// (`R_AARCH64_ABS64`-style). The bytes under the slot are ignored.
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct DataReloc {
    pub offset: u64,
    pub symbol: String,
    pub addend: i64,
}

/// The initializer of a global that must live in a data section: raw bytes
/// plus the pointer slots inside them — the same shape as rustc's
/// `Allocation` (bytes + provenance). Contrast with [BytesAttr], whose
/// pointer-free payloads may be inlined into the text section: a
/// [DataAttr] global always gets its own symbol in `.rodata` (immutable)
/// or `.data` (mutable), so its address is meaningful and its pointer
/// slots can carry relocations.
///
/// Attach it to an `llvm.global` as the initializer value
/// ([set_global_data](crate::ll::set_global_data)); backends read it back
/// with [global_data](crate::ll::global_data).
///
/// Textual form:
/// `align=<u64>, mut=<true|false>, bytes=0x<hex>, relocs=[<offset>:<symbol>:<addend>, ...]`
#[pliron_attr(name = "ll.data", verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct DataAttr {
    pub bytes: Vec<u8>,
    pub align: u64,
    /// `true` places the global in a writable section (`.data`), `false`
    /// in a read-only one (`.rodata`).
    pub mutable: bool,
    pub relocs: Vec<DataReloc>,
}

impl core::fmt::Display for DataAttr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "align={}, mut={}, bytes={}, relocs=[",
            self.align,
            self.mutable,
            BytesAttr(self.bytes.clone())
        )?;
        for (i, reloc) in self.relocs.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}:{}:{}", reloc.offset, reloc.symbol, reloc.addend)?;
        }
        write!(f, "]")
    }
}

impl Printable for DataAttr {
    fn fmt(
        &self,
        _ctx: &Context,
        _state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "{self}")
    }
}

/// Parses one [DataReloc] using the same `offset:symbol:addend` spelling
/// [Printable] emits, e.g. `8:other_global:-16`.
fn parse_data_reloc<'a>(state_stream: &mut StateStream<'a>) -> ParseResult<'a, DataReloc> {
    let (offset, _commit) = int_parser::<u64>()
        .parse_stream(state_stream)
        .into_result()?;
    let (symbol, _commit) = token(':')
        .with(many1::<String, _, _>(satisfy(|c: char| {
            c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '$')
        })))
        .parse_stream(state_stream)
        .into_result()?;
    let ((negative, magnitude), commit) = token(':')
        .with(optional(token('-')).and(int_parser::<i64>()))
        .parse_stream(state_stream)
        .into_result()?;
    let addend = if negative.is_some() {
        -magnitude
    } else {
        magnitude
    };
    Ok((
        DataReloc {
            offset,
            symbol,
            addend,
        },
        commit,
    ))
}

impl Parsable for DataAttr {
    type Arg = ();
    type Parsed = Self;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let loc = state_stream.loc();
        let word = || many1::<String, _, _>(satisfy(|c: char| c.is_ascii_alphanumeric()));
        let mut parser = (
            combine::parser::char::string("align=").with(int_parser::<u64>()),
            spaced(token(',')).skip(combine::parser::char::string("mut=")).with(word()),
            spaced(token(',')).skip(combine::parser::char::string("bytes=")).with(word()),
            spaced(token(',')).skip(combine::parser::char::string("relocs=")).with(between(
                spaced(token('[')),
                spaced(token(']')),
                sep_by::<Vec<_>, _, _, _>(
                    spaced(combine::parser(parse_data_reloc)),
                    spaced(token(',')),
                ),
            )),
        );
        let ((align, mutable, bytes, relocs), commit) =
            parser.parse_stream(state_stream).into_result()?;
        let mutable = match mutable.as_str() {
            "true" => true,
            "false" => false,
            other => input_err!(loc.clone(), "invalid data mutability `{other}`: expected `true` or `false`")?,
        };
        let Some(bytes) = BytesAttr::parse_str(&bytes) else {
            input_err!(
                loc,
                "invalid bytes literal `{}`: expected `0x` followed by an even number of hex digits",
                bytes
            )?
        };
        Ok((
            DataAttr {
                bytes: bytes.0,
                align,
                mutable,
                relocs,
            },
            commit,
        ))
    }
}

/// Unit marker for a thread-local `llvm.global`: its storage lives in the
/// TLS segment (`.tdata`/`.tbss` on ELF) and taking its address yields the
/// current thread's copy, so backends must materialize it through the
/// thread pointer instead of an ordinary data-section address. Attach and
/// query it with [set_global_thread_local](crate::ll::set_global_thread_local)
/// / [global_is_thread_local](crate::ll::global_is_thread_local); it marks
/// both defined globals (with an [ll.data](DataAttr) initializer) and
/// extern declarations resolved by another object's TLS definition.
#[pliron_attr(name = "ll.tls", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Default, Hash)]
pub struct TlsAttr;

/// A machine block's stable identity for the profile-feedback blockmap
/// (docs/PROFILE-FEEDBACK-PLAN.md): its index in RA-time region order,
/// attached to the [BasicBlock](pliron::basic_block::BasicBlock)'s own
/// attribute dictionary at the register-allocation pipeline position.
/// Blocks are reordered but never recreated after RA, so the attribute
/// rides the block through placement, relaxation, and encoding down to
/// final layout.
#[pliron_attr(name = "ll.blockmap_id", format = "$0", verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct BlockmapIdAttr(pub u32);

/// An LLVM-dialect op's stable identity for backward profile attribution
/// (docs/PROFILE-FEEDBACK-BACKWARD.md): its dense per-function index in
/// program order at the LLVM→machine boundary, stamped immediately before
/// instruction selection.
#[pliron_attr(name = "ll.op_id", format = "$0", verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct OpIdAttr(pub u32);

/// Which op a machine op was lowered from, for backward profile
/// attribution: a non-negative value is the [OpIdAttr] of the LLVM-dialect
/// source op; a negative value is a synthetic per-pass root (see
/// `passes::aarch64::opmap::roots`) for code created from nothing
/// (prologue, ABI moves, layout branches), so that overhead is visible as
/// the creating pass's own cost.
#[pliron_attr(name = "ll.derived_from", format = "$0", verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct DerivedFromAttr(pub i64);

/// Multi-parent `derived_from` for backward profile attribution
/// (docs/PROFILE-FEEDBACK-BACKWARD.md): the op is the merged survivor of
/// several source ops (GVN CSE dedup), and measured cost lifts onto the
/// parents as an equal split (weights are implicit: 1/n each — no current
/// producer needs unequal weights; the ingest normalizes). Ids follow the
/// [DerivedFromAttr] convention (non-negative = [OpIdAttr] of a source op,
/// negative = synthetic root).
#[pliron_attr(
    name = "ll.derived_from_many",
    format = "`[` vec($0, CharSpace(`,`)) `]`",
    verifier = "succ"
)]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct DerivedFromManyAttr(pub Vec<i64>);

/// The cross-function hop of backward attribution: this op was inlined
/// into its function, and its cost belongs to the CALL SITE op whose id
/// this attr carries (the callee-local [OpIdAttr] is preserved alongside
/// for future cross-function lifting). Set by `llvm-inline`'s adjoint.
#[pliron_attr(name = "ll.inlined_from", format = "$0", verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct InlinedFromAttr(pub i64);

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
    /// Exported, but overridable: another object's strong definition of the
    /// same symbol wins at link time (ELF `STB_WEAK`). Used for
    /// monomorphizations that LLVM-built rlibs may also export
    /// (share-generics), where both copies implement the same function.
    Weak,
}

impl LinkageAttr {
    pub fn parse_str(text: &str) -> Option<Self> {
        match text {
            "external" => Some(Self::External),
            "internal" => Some(Self::Internal),
            "private" => Some(Self::Private),
            "weak" => Some(Self::Weak),
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
            Self::Weak => write!(f, "weak"),
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
                "invalid linkage `{}`: expected `external`, `internal`, `private` or `weak`",
                text
            )?
        };
        Ok((linkage, commit))
    }
}

#[cfg(test)]
mod tests {
    use combine::Parser;
    use pliron::{
        context::Context,
        location,
        parsable::{Parsable, State, state_stream_from_iterator},
        printable::Printable,
    };

    use super::{DataAttr, DataReloc};

    #[test]
    fn data_attr_round_trips_through_text() {
        let mut ctx = Context::new();
        let attr = DataAttr {
            bytes: vec![0x2a, 0x00, 0xff],
            align: 8,
            mutable: true,
            relocs: vec![
                DataReloc {
                    offset: 0,
                    symbol: "other_global".to_string(),
                    addend: 16,
                },
                DataReloc {
                    offset: 8,
                    symbol: "_my.helper$1".to_string(),
                    addend: -4,
                },
            ],
        };
        let text = attr.disp(&ctx).to_string();
        assert_eq!(
            text,
            "align=8, mut=true, bytes=0x2a00ff, relocs=[0:other_global:16, 8:_my.helper$1:-4]"
        );

        let state_stream = state_stream_from_iterator(
            text.chars(),
            State::new(&mut ctx, location::Source::InMemory),
        );
        let parsed = DataAttr::parser(()).parse(state_stream).unwrap().0;
        assert_eq!(parsed, attr);
    }

    #[test]
    fn data_attr_parses_empty_relocs() {
        let mut ctx = Context::new();
        let attr = DataAttr {
            bytes: vec![],
            align: 1,
            mutable: false,
            relocs: vec![],
        };
        let text = attr.disp(&ctx).to_string();
        assert_eq!(text, "align=1, mut=false, bytes=0x, relocs=[]");

        let state_stream = state_stream_from_iterator(
            text.chars(),
            State::new(&mut ctx, location::Source::InMemory),
        );
        let parsed = DataAttr::parser(()).parse(state_stream).unwrap().0;
        assert_eq!(parsed, attr);
    }
}
