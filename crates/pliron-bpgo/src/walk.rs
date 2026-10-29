//! The backward walk over a trace.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use pliron::builtin::op_interfaces::SymbolOpInterface;
use pliron::combine::Parser;
use pliron::context::{Context, Ptr};
use pliron::linked_list::ContainsLinkedList;
use pliron::location::{Location, Source};
use pliron::op::op_cast;
use pliron::operation::{Operation, OperationParserConfig};
use pliron::parsable::{Parsable, State, state_stream_from_iterator};
use pliron_trace::Trace;
use serde::Serialize;

use crate::profile;

/// The backward counterpart of a forward pass: spreads the profile on the
/// pass's output back onto its input.
pub trait BackwardPass {
    /// The name of the forward pass, as the trace records it.
    fn name(&self) -> &str;

    /// `output` is a function after the forward pass, carrying its profile;
    /// `input` is the same function before it. Put the profile on `input`'s
    /// ops (and on `input` itself for what belongs to the function as a
    /// whole), reading the metadata the forward pass recorded.
    fn run(
        &self,
        ctx: &mut Context,
        input: Ptr<Operation>,
        output: Ptr<Operation>,
    ) -> Result<(), String>;
}

/// The backward passes known to a walk, by forward pass name.
#[derive(Default)]
pub struct BackwardPasses {
    passes: HashMap<String, Box<dyn BackwardPass>>,
}

impl BackwardPasses {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, pass: impl BackwardPass + 'static) {
        self.passes.insert(pass.name().to_string(), Box::new(pass));
    }

    pub fn get(&self, name: &str) -> Option<&dyn BackwardPass> {
        self.passes.get(name).map(|pass| &**pass)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.passes.keys().map(String::as_str)
    }
}

/// The profile of one op of a snapshot, keyed by where the op starts in the
/// snapshot's text (1-based line and column). Serialized compactly as
/// `[line, column, {metric: value}]`, values rounded to whole numbers.
#[derive(Clone, Debug)]
pub struct OpProfile {
    pub line: u32,
    pub column: u32,
    pub metrics: BTreeMap<String, f64>,
}

impl Serialize for OpProfile {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeTuple;
        let rounded: BTreeMap<&str, f64> = self
            .metrics
            .iter()
            .map(|(name, value)| (name.as_str(), value.round()))
            .collect();
        let mut tuple = serializer.serialize_tuple(3)?;
        tuple.serialize_element(&self.line)?;
        tuple.serialize_element(&self.column)?;
        tuple.serialize_element(&rounded)?;
        tuple.end()
    }
}

/// Where a function's profile stops.
#[derive(Clone, Debug, Serialize)]
pub struct Stop {
    /// The earliest snapshot (index) the function has a profile at.
    pub reached: usize,
    /// The pass the walk could not go back through, if any.
    pub pass: Option<String>,
    pub reason: String,
}

/// What a walk produced.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Walk {
    /// Per snapshot (index): the profile of its ops.
    pub snapshots: BTreeMap<usize, Vec<OpProfile>>,
    /// Per function (symbol): where its profile stops.
    pub stops: BTreeMap<String, Stop>,
}

/// Puts the measured profile on the ops under the root it is given.
pub type Measure<'a> = dyn FnMut(&mut Context, Ptr<Operation>) -> Result<(), String> + 'a;

/// Walk `trace` backward. `measure` puts the measured profile on the ops
/// of the last snapshot; then each pass, from the last, hands its output's
/// profile to its input: a function the pass did not change keeps its
/// profile as is, a changed one goes through the pass's [BackwardPass].
/// A function stops where a pass changed it and has no backward pass (or
/// its backward pass fails); the others go on.
pub fn walk(trace: &Trace, passes: &BackwardPasses, measure: &mut Measure) -> Result<Walk, String> {
    let snapshots = &trace.snapshots;
    let Some(last) = snapshots.len().checked_sub(1) else {
        return Err("the trace has no snapshots".into());
    };
    let mut ctx = Context::new();
    let mut output = parse(&mut ctx, &snapshots[last].ir).map_err(|e| {
        format!(
            "the last snapshot ({}) does not parse: {e}",
            snapshots[last].label
        )
    })?;
    measure(&mut ctx, output)?;

    let mut walk = Walk::default();
    walk.snapshots.insert(last, collect(&ctx, output));
    let mut out_funcs = functions(&ctx, output, &snapshots[last].ir);
    let mut live: BTreeSet<String> = out_funcs
        .iter()
        .filter(|(_, f)| has_profile(&ctx, f.op))
        .map(|(symbol, _)| symbol.clone())
        .collect();

    let mut index = last;
    while index > 0 && !live.is_empty() {
        let pass = snapshots[index].label.clone();
        let input = match parse(&mut ctx, &snapshots[index - 1].ir) {
            Ok(op) => op,
            Err(e) => {
                for symbol in std::mem::take(&mut live) {
                    let reason = format!("the IR before {pass} does not parse: {e}");
                    walk.stops.insert(symbol, stop(index, &pass, reason));
                }
                break;
            }
        };
        let in_funcs = functions(&ctx, input, &snapshots[index - 1].ir);
        for symbol in live.clone() {
            let out_f = &out_funcs[&symbol];
            let Some(in_f) = in_funcs.get(&symbol) else {
                live.remove(&symbol);
                walk.stops.insert(
                    symbol,
                    stop(index, &pass, format!("{pass} created the function")),
                );
                continue;
            };
            if in_f.text == out_f.text {
                copy(&ctx, in_f.op, out_f.op);
                continue;
            }
            let failure = match passes.get(&pass) {
                None => Some(format!(
                    "{pass} changed the function and has no backward pass"
                )),
                Some(backward) => {
                    // What belongs to the function as a whole stays with it.
                    if let Some(own) = profile::get(&ctx, out_f.op) {
                        profile::add_all(&ctx, in_f.op, &own, 1.0);
                    }
                    let result = backward.run(&mut ctx, in_f.op, out_f.op);
                    if result.is_ok() {
                        profile::even_out_executions(&ctx, in_f.op);
                    }
                    result
                        .err()
                        .map(|e| format!("{pass}'s backward pass failed: {e}"))
                }
            };
            if let Some(reason) = failure {
                live.remove(&symbol);
                walk.stops.insert(symbol, stop(index, &pass, reason));
            }
        }
        walk.snapshots.insert(index - 1, collect(&ctx, input));
        Operation::erase(output, &mut ctx);
        output = input;
        out_funcs = in_funcs;
        index -= 1;
    }
    for symbol in live {
        walk.stops.insert(
            symbol,
            Stop {
                reached: index,
                pass: None,
                reason: "reached the start of the pipeline".into(),
            },
        );
    }
    walk.snapshots.retain(|_, ops| !ops.is_empty());
    Ok(walk)
}

fn stop(reached: usize, pass: &str, reason: String) -> Stop {
    Stop {
        reached,
        pass: Some(pass.to_string()),
        reason,
    }
}

/// Parse printed IR, with its outlined attributes section.
pub fn parse(ctx: &mut Context, text: &str) -> Result<Ptr<Operation>, String> {
    let state = State::new(ctx, Source::InMemory);
    let stream = state_stream_from_iterator(text.chars(), state);
    let config = OperationParserConfig {
        look_for_outlined_attrs: true,
    };
    <Operation as Parsable>::parser(config)
        .parse(stream)
        .map(|(op, _)| op)
        .map_err(|e| format!("{e}"))
}

/// Where `op` starts in the text it was parsed from (1-based).
pub fn position(ctx: &Context, op: Ptr<Operation>) -> Option<(u32, u32)> {
    match op.deref(ctx).loc() {
        Location::SrcPos { pos, .. } => Some((pos.line as u32, pos.column as u32)),
        _ => None,
    }
}

/// Every op under `root` (itself included), in program order.
pub fn ops_under(ctx: &Context, root: Ptr<Operation>) -> Vec<Ptr<Operation>> {
    let mut out = Vec::new();
    let mut stack = vec![root];
    while let Some(op) = stack.pop() {
        out.push(op);
        let regions: Vec<_> = op.deref(ctx).regions().collect();
        let mut children = Vec::new();
        for region in regions {
            for block in region.deref(ctx).iter(ctx) {
                children.extend(block.deref(ctx).iter(ctx));
            }
        }
        stack.extend(children.into_iter().rev());
    }
    out
}

fn collect(ctx: &Context, root: Ptr<Operation>) -> Vec<OpProfile> {
    ops_under(ctx, root)
        .into_iter()
        .filter_map(|op| {
            let profile = profile::get(ctx, op)?;
            let (line, column) = position(ctx, op)?;
            Some(OpProfile {
                line,
                column,
                metrics: profile.0,
            })
        })
        .collect()
}

fn has_profile(ctx: &Context, func: Ptr<Operation>) -> bool {
    ops_under(ctx, func)
        .into_iter()
        .any(|op| profile::get(ctx, op).is_some())
}

/// A top-level symbol op (a function) of a snapshot, with its text.
struct Function {
    op: Ptr<Operation>,
    /// Its lines in the snapshot, without the outlined-attribute markers
    /// (` !N`): equal texts mean the pass left the function alone.
    text: String,
}

fn functions(ctx: &Context, root: Ptr<Operation>, text: &str) -> HashMap<String, Function> {
    let lines: Vec<&str> = text.lines().collect();
    // The module's body ends where the outlined section starts.
    let end = lines
        .iter()
        .position(|line| line.starts_with("outlined_attributes:"))
        .unwrap_or(lines.len());
    let mut top: Vec<(Ptr<Operation>, usize)> = Vec::new();
    let regions: Vec<_> = root.deref(ctx).regions().collect();
    for region in regions {
        for block in region.deref(ctx).iter(ctx) {
            for op in block.deref(ctx).iter(ctx) {
                if let Some((line, _)) = position(ctx, op) {
                    top.push((op, line as usize));
                }
            }
        }
    }
    let mut out = HashMap::new();
    for (i, &(op, line)) in top.iter().enumerate() {
        let op_obj = Operation::get_op_dyn(op, ctx);
        let Some(symbol) = op_cast::<dyn SymbolOpInterface>(&*op_obj) else {
            continue;
        };
        if op.deref(ctx).regions().next().is_none() {
            continue;
        }
        let from = line.saturating_sub(1);
        let to = top
            .get(i + 1)
            .map_or(end, |&(_, next)| next.saturating_sub(1));
        let text = lines[from.min(end)..to.min(end).max(from.min(end))]
            .iter()
            .map(|line| strip_markers(line))
            .collect::<Vec<_>>()
            .join("\n");
        out.insert(
            symbol.get_symbol_name(ctx).to_string(),
            Function { op, text },
        );
    }
    out
}

/// `line` without pliron's outlined-attribute markers (` !N`).
fn strip_markers(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b' '
            && bytes.get(i + 1) == Some(&b'!')
            && bytes.get(i + 2).is_some_and(u8::is_ascii_digit)
        {
            i += 2;
            while bytes.get(i).is_some_and(u8::is_ascii_digit) {
                i += 1;
            }
            continue;
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

/// Copy the profile of `output` onto `input`, which has the same shape.
fn copy(ctx: &Context, input: Ptr<Operation>, output: Ptr<Operation>) {
    let ins = ops_under(ctx, input);
    let outs = ops_under(ctx, output);
    for (i, o) in ins.into_iter().zip(outs) {
        if let Some(p) = profile::get(ctx, o) {
            profile::add_all(ctx, i, &p, 1.0);
        }
    }
}

/// `trace` with the profile of `walk` in its header (`bpgo`): what was
/// measured (`measurement`, the producer's to describe), the metrics, the
/// profile of every snapshot reached, and where each function's stops. The
/// IR of the snapshots stays as it was.
pub fn annotate(trace: &Trace, walk: &Walk, measurement: serde_json::Value) -> Trace {
    let mut annotated = trace.clone();
    annotated.meta.extra.insert(
        "bpgo".into(),
        serde_json::json!({
            "measurement": measurement,
            "metrics": {
                "cycles": "estimated cycles spent in the op over the run",
                "executions": "estimated number of times the op ran",
                "<pass>.cycles": "cycles of code that pass created, as its own decision",
            },
            "snapshots": walk.snapshots,
            "stops": walk.stops,
        }),
    );
    annotated
}

#[cfg(test)]
mod tests {
    use super::strip_markers;

    #[test]
    fn markers_are_stripped() {
        assert_eq!(
            strip_markers("  x = llvm.add a, b !12;"),
            "  x = llvm.add a, b;"
        );
        assert_eq!(strip_markers("a !b"), "a !b");
    }
}
