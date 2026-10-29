//! Backward PGO for the LLVM-level passes (`pliron-bpgo`): from the IR
//! instruction selection received back through the mid-end.
//!
//! The mid-end's passes stamp their adjoints on what they create
//! ([opmap]): an op keeps the `ll.op_id`
//! the mid-end's first pass gave it while it lives (moved or not); an op a
//! pass creates names its source ops (`derived_from`,
//! `derived_from_many`); a GVN survivor adds the ops it absorbed to its
//! `derived_from_many`; an unrolled clone names its original's sources; an
//! inlined op names its call site (`inlined_from`). The backward pass of
//! each reads that back:
//!
//! - ops on both sides (matched by opcode and identity, moves included)
//!   hand their profile over as is, except that a survivor that absorbed
//!   ops in this pass shares it with them;
//! - a created op goes to what it replaced at the same spot when that
//!   shares its sources, else to the ops it names as sources — the ones
//!   the pass took out first, of the same kind first — or, for an
//!   unrolled clone, to its original as another instance of it;
//! - what none of that places is the function's `unattributed.cycles`.

use std::collections::{BTreeSet, HashSet};

use pliron_bpgo::{
    BackwardPass, BackwardPasses, CYCLES, ProfileAttr, align, profile, walk::ops_under,
};

use crate::passes::aarch64::opmap;
use crate::{
    context::{Context, Ptr},
    ir::operation::Operation,
};

/// Identity for matching: an op keeps its `ll.op_id` (with the call site,
/// for inlined copies whose callee-local ids repeat) while it lives; an op
/// without one is known by its sources.
fn key(ctx: &Context, op: Ptr<Operation>) -> String {
    let opid = Operation::get_opid(op, ctx);
    match opmap::op_id(ctx, op) {
        Some(id) => format!(
            "{opid}|#{id}|{}",
            opmap::inlined_from(ctx, op).map_or("-".to_string(), |c| c.to_string())
        ),
        None => format!("{opid}|{:?}", opmap::effective_sources(ctx, op)),
    }
}

fn sources(ctx: &Context, op: Ptr<Operation>) -> BTreeSet<i64> {
    opmap::effective_sources(ctx, op).into_iter().collect()
}

fn merged(ctx: &Context, op: Ptr<Operation>) -> Option<BTreeSet<i64>> {
    opmap::derived_from_many(ctx, op).map(|ids| ids.into_iter().collect())
}

fn function_cost(ctx: &Context, func: Ptr<Operation>, what: &str, from: &ProfileAttr) {
    if let Some(cycles) = from.0.get(CYCLES) {
        profile::add(ctx, func, &format!("{what}.cycles"), *cycles);
    }
}

/// The backward pass of an LLVM-level pass.
pub struct LlvmBackward {
    name: &'static str,
    /// The pass clones ops (unrolling): a created op with an original of
    /// the same kind and sources is another instance of it.
    clones: bool,
}

impl BackwardPass for LlvmBackward {
    fn name(&self) -> &str {
        self.name
    }

    fn run(
        &self,
        ctx: &mut Context,
        input: Ptr<Operation>,
        output: Ptr<Operation>,
    ) -> Result<(), String> {
        let ctx = &*ctx;
        let alignment = align::align(ctx, input, output, &key);
        let deleted: HashSet<_> = alignment.deleted.iter().copied().collect();
        let inputs: Vec<_> = ops_under(ctx, input).into_iter().skip(1).collect();

        for &(i, o) in &alignment.pairs {
            let Some(p) = profile::get(ctx, o) else {
                continue;
            };
            // A survivor that absorbed ops in this pass (GVN) shares its
            // profile with them.
            match merged(ctx, o).filter(|m| merged(ctx, i).as_ref() != Some(m)) {
                Some(set) => {
                    let opid = Operation::get_opid(o, ctx);
                    let mut targets = vec![i];
                    targets.extend(alignment.deleted.iter().copied().filter(|&d| {
                        Operation::get_opid(d, ctx) == opid && !sources(ctx, d).is_disjoint(&set)
                    }));
                    profile::spread(ctx, &targets, &p);
                }
                None => profile::add_all(ctx, i, &p, 1.0),
            }
        }

        for &o in &alignment.created {
            let Some(p) = profile::get(ctx, o) else {
                continue;
            };
            let from = sources(ctx, o);
            let opid = Operation::get_opid(o, ctx);
            let shares =
                |i: Ptr<Operation>| !from.is_empty() && !sources(ctx, i).is_disjoint(&from);
            let same_kind = |i: Ptr<Operation>| Operation::get_opid(i, ctx) == opid;

            if self.clones {
                let originals: Vec<_> = inputs
                    .iter()
                    .copied()
                    .filter(|&i| !deleted.contains(&i) && same_kind(i) && sources(ctx, i) == from)
                    .collect();
                if !originals.is_empty() {
                    let share = ProfileAttr(
                        p.0.iter()
                            .map(|(k, v)| (k.clone(), v / originals.len() as f64))
                            .collect(),
                    );
                    for original in originals {
                        profile::add_instance(ctx, original, &share);
                    }
                    continue;
                }
            }

            let replaced: Vec<_> = alignment
                .replaced_by(o)
                .iter()
                .copied()
                .filter(|&i| from.is_empty() || shares(i))
                .collect();
            if !replaced.is_empty() {
                profile::spread(ctx, &replaced, &p);
                continue;
            }

            let candidates = [
                (true, true),   // taken out, same kind
                (true, false),  // taken out
                (false, true),  // still there, same kind
                (false, false), // still there
            ]
            .into_iter()
            .map(|(gone, kind)| {
                inputs
                    .iter()
                    .copied()
                    .filter(|&i| {
                        deleted.contains(&i) == gone && (!kind || same_kind(i)) && shares(i)
                    })
                    .collect::<Vec<_>>()
            })
            .find(|targets| !targets.is_empty());
            match candidates {
                Some(targets) => profile::spread(ctx, &targets, &p),
                None => function_cost(ctx, input, "unattributed", &p),
            }
        }
        Ok(())
    }
}

/// The backward passes of the LLVM-level passes.
pub fn register(passes: &mut BackwardPasses) {
    for name in [
        "llvm-aarch64-linux-abi",
        "verify-llvm-for-aarch64-linux",
        "llvm-vectorize",
        "llvm-simplify",
        "llvm-simplify-cfg",
        "llvm-sink",
        "llvm-adce",
        "llvm-global-dse",
        "llvm-loop-carried-fwd",
        "llvm-gvn",
        "llvm-licm",
        "llvm-div-strength-reduce",
        "mem2reg",
        "llvm-pin-type-punned-slots",
        "llvm-sroa",
        "llvm-inline",
    ] {
        passes.add(LlvmBackward {
            name,
            clones: false,
        });
    }
    passes.add(LlvmBackward {
        name: "llvm-unroll",
        clones: true,
    });
}
