//! The profile of an op: named measurements, carried as a [ProfileAttr].

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

use pliron::builtin::attr_interfaces::OutlinedAttr;
use pliron::combine::{Parser, between, many1, satisfy, sep_by, token};
use pliron::context::{Context, Ptr};
use pliron::derive::{attr_interface_impl, pliron_attr};
use pliron::dict_key;
use pliron::input_err;
use pliron::irfmt::parsers::spaced;
use pliron::operation::Operation;
use pliron::parsable::{Parsable, ParseResult, StateStream, StreamLocation};
use pliron::printable::{self, Printable};

dict_key!(ATTR_KEY_PROFILE, "bpgo_profile");

/// Estimated cycles spent in the op over the whole measured run.
pub const CYCLES: &str = "cycles";
/// Estimated number of times the op ran.
pub const EXECUTIONS: &str = "executions";

/// Named measurements of one op: [CYCLES], [EXECUTIONS], and the costs a
/// backward pass attributes to its own decisions (`regalloc.cycles`,
/// `frame.cycles`, ...). Printed as `bpgo.profile <cycles: 1200, executions: 40>`.
#[pliron_attr(name = "bpgo.profile", verifier = "succ")]
#[derive(Clone, Debug, Default)]
pub struct ProfileAttr(pub BTreeMap<String, f64>);

impl PartialEq for ProfileAttr {
    fn eq(&self, other: &Self) -> bool {
        self.0.len() == other.0.len()
            && self
                .0
                .iter()
                .zip(other.0.iter())
                .all(|((a, x), (b, y))| a == b && x.to_bits() == y.to_bits())
    }
}

impl Eq for ProfileAttr {}

impl Hash for ProfileAttr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for (name, value) in &self.0 {
            name.hash(state);
            value.to_bits().hash(state);
        }
    }
}

// Metadata no op format mentions: printed, if ever, in pliron's outlined
// attributes section.
#[attr_interface_impl]
impl OutlinedAttr for ProfileAttr {}

impl Printable for ProfileAttr {
    fn fmt(
        &self,
        _ctx: &Context,
        _state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "<")?;
        for (i, (name, value)) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{name}: {value}")?;
        }
        write!(f, ">")
    }
}

impl Parsable for ProfileAttr {
    type Arg = ();
    type Parsed = Self;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let loc = state_stream.loc();
        let word = || {
            many1::<String, _, _>(satisfy(|c: char| {
                c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-' | '+')
            }))
        };
        let entry = spaced(word()).skip(spaced(token(':'))).and(spaced(word()));
        let mut parser = between(
            spaced(token('<')),
            spaced(token('>')),
            sep_by::<Vec<(String, String)>, _, _, _>(entry, spaced(token(','))),
        );
        let (entries, commit) = parser.parse_stream(state_stream).into_result()?;
        let mut metrics = BTreeMap::new();
        for (name, value) in entries {
            let Ok(value) = value.parse::<f64>() else {
                input_err!(loc, "`{}` is not a number", value)?
            };
            metrics.insert(name, value);
        }
        Ok((ProfileAttr(metrics), commit))
    }
}

/// How a metric combines when several sources land on one op: costs add
/// up; counts of runs (executions) do not — an op expanded into three
/// still ran as often as each of the three.
pub fn is_count(metric: &str) -> bool {
    metric == EXECUTIONS || metric.ends_with(".executions")
}

/// The profile on `op`, if any.
pub fn get(ctx: &Context, op: Ptr<Operation>) -> Option<ProfileAttr> {
    op.deref(ctx)
        .attributes
        .get::<ProfileAttr>(&ATTR_KEY_PROFILE)
        .cloned()
}

pub fn set(ctx: &Context, op: Ptr<Operation>, profile: ProfileAttr) {
    op.deref_mut(ctx)
        .attributes
        .set(ATTR_KEY_PROFILE.clone(), profile);
}

/// Add `value` to `metric` on `op`: summed for costs, the larger kept for
/// counts.
pub fn add(ctx: &Context, op: Ptr<Operation>, metric: &str, value: f64) {
    let mut profile = get(ctx, op).unwrap_or_default();
    let slot = profile.0.entry(metric.to_string()).or_insert(0.0);
    if is_count(metric) {
        *slot = slot.max(value);
    } else {
        *slot += value;
    }
    set(ctx, op, profile);
}

/// Add every metric of `from` to `op`, costs scaled by `share` (the part of
/// `from` that belongs to `op`), counts as they are.
pub fn add_all(ctx: &Context, op: Ptr<Operation>, from: &ProfileAttr, share: f64) {
    if from.0.is_empty() {
        return;
    }
    let mut profile = get(ctx, op).unwrap_or_default();
    for (metric, value) in &from.0 {
        let slot = profile.0.entry(metric.clone()).or_insert(0.0);
        if is_count(metric) {
            *slot = slot.max(*value);
        } else {
            *slot += value * share;
        }
    }
    set(ctx, op, profile);
}

/// Spread `from` over `targets`: costs split evenly, counts copied.
pub fn spread(ctx: &Context, targets: &[Ptr<Operation>], from: &ProfileAttr) {
    if targets.is_empty() {
        return;
    }
    let share = 1.0 / targets.len() as f64;
    for &target in targets {
        add_all(ctx, target, from, share);
    }
}

/// Add `from` to `op` as another instance of it — a clone that runs
/// alongside it (an unrolled copy): its runs add to `op`'s, costs too.
pub fn add_instance(ctx: &Context, op: Ptr<Operation>, from: &ProfileAttr) {
    if from.0.is_empty() {
        return;
    }
    let mut profile = get(ctx, op).unwrap_or_default();
    for (metric, value) in &from.0 {
        *profile.0.entry(metric.clone()).or_insert(0.0) += value;
    }
    set(ctx, op, profile);
}

/// The ops of a block run equally often: give every op of each block of
/// `func` the block's [EXECUTIONS] (the largest any of its ops got — an op
/// moved out of a loop carries the count of where it went, not of where
/// it was).
pub fn even_out_executions(ctx: &Context, func: Ptr<Operation>) {
    use pliron::linked_list::ContainsLinkedList;
    let regions: Vec<_> = func.deref(ctx).regions().collect();
    for region in regions {
        let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
        for block in blocks {
            let ops: Vec<_> = block.deref(ctx).iter(ctx).collect();
            let runs = ops
                .iter()
                .filter_map(|&op| get(ctx, op)?.0.get(EXECUTIONS).copied())
                .fold(None, |acc: Option<f64>, x| {
                    Some(acc.map_or(x, |a| a.max(x)))
                });
            let Some(runs) = runs else {
                continue;
            };
            for op in ops {
                let mut profile = get(ctx, op).unwrap_or_default();
                profile.0.insert(EXECUTIONS.to_string(), runs);
                set(ctx, op, profile);
            }
        }
    }
}
