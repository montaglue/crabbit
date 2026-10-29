//! Pass infrastructure: pliron's own pass manager ([pliron::pass]),
//! re-exported — the [Pass] trait, [Passes], analyses, and the
//! [PassManager] hooks configured through [PMConfig] on the
//! [AnalysisManager] (print before/after into `{n}-after-{name}.plir`
//! files, verify, time, skip — all or by pass name). crabbit adds only what
//! pliron does not have: building a pipeline from an order of pass names
//! ([passes_from_order], [into_passes]).
//!
//! `Pass::run` mutates the operation it is given and keeps its identity.
//! Transformations that used to swap the root operation (instruction
//! selection) now rewrite the module in place; producing a different IR
//! altogether (Mach-O objects) is a translation out of the pass pipeline,
//! not a pass.

use pliron::{
    context::{Context, Ptr},
    operation::Operation,
};

pub use pliron::pass::*;

/// pliron's own mem2reg pass (its name, `mem2reg`, is what the pass dumps
/// are keyed on).
pub use pliron::opts::mem2reg::Mem2RegPass;

/// A pipeline as a list: what an order of pass names builds into.
pub type PassList = Vec<Box<dyn Pass>>;

/// The passes an order of names describes: `construct` turns each name
/// into its pass; a name it does not know is an error naming it.
/// Pipelines are data (orders of names) so they can be edited — see the
/// default orders next to each stage's passes and crabbit-config.
pub fn passes_from_order(
    order: &[&str],
    mut construct: impl FnMut(&str) -> Option<Box<dyn Pass>>,
) -> Result<PassList, String> {
    order
        .iter()
        .map(|name| construct(name).ok_or_else(|| format!("unknown pass `{name}`")))
        .collect()
}

/// pliron's [Passes] holding `list`, in order.
pub fn into_passes(list: PassList) -> Passes {
    let mut passes = Passes::default();
    for pass in list {
        passes.add_pass(Boxed(pass));
    }
    passes
}

/// [Passes::add_pass] takes `impl Pass`, and `Box<dyn Pass>` cannot
/// implement the foreign trait (orphan rule): this adapter is the glue. It
/// forwards [Pass::as_pass_manager] so a boxed pass manager stays a manager
/// for pliron's hooks (which print, verify and count leaf passes only).
struct Boxed(Box<dyn Pass>);

impl Pass for Boxed {
    fn name(&self) -> &str {
        self.0.name()
    }

    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        analyses: &mut AnalysisManager,
    ) -> pliron::result::Result<PassResult> {
        self.0.run(op, ctx, analyses)
    }

    fn as_pass_manager(&mut self) -> Option<&mut dyn PassManager> {
        self.0.as_pass_manager()
    }
}

/// A [PassResult] reporting that the IR changed — the common case for every
/// pass here, none of which currently participate in analysis caching.
pub fn changed() -> PassResult {
    let mut result = PassResult::default();
    result.ir_changed = pliron::irbuild::IRStatus::Changed;
    result
}

/// A [PassResult] reporting that the IR is untouched (all analyses are
/// preserved) — what verification-only passes return.
pub fn unchanged() -> PassResult {
    PassResult::default()
}
