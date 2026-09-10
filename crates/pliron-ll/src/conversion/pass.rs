//! Pass infrastructure: pliron's own [pliron::pass] module, re-exported
//! under the crate's compatibility facade. Upstream 0.17 grew its own
//! `Passes` runner and `PMConfig` printing hooks, but its `print_after_all`
//! dumps every *leaf* pass with a global run counter, while crabbit's trace
//! collection (crabbit/src/lib.rs) expects one `{n}-after-{name}.plir` dump
//! per top-level pipeline entry (a nested target pipeline is a single
//! entry). The local [Passes]/[PMConfig] pair below preserves exactly that
//! dump granularity and naming, so it stays in place and shadows the
//! upstream types in the glob re-export.
//!
//! `Pass::run` mutates the operation it is given and keeps its identity.
//! Transformations that used to swap the root operation (instruction
//! selection) now rewrite the module in place; producing a different IR
//! altogether (Mach-O objects) is a translation out of the pass pipeline,
//! not a pass.

use pliron::{
    context::{Context, Ptr},
    operation::Operation,
    printable::Printable,
};

pub use pliron::pass::*;

/// pliron's own mem2reg pass (its name, `mem2reg`, is what the pass dumps
/// are keyed on). Re-exported here because callers historically got it from
/// this facade, back when the pinned pliron rev only exposed a function.
pub use pliron::opts::mem2reg::Mem2RegPass;

/// A type-erased [Pass], so a pipeline constructor can accept a pass chosen
/// at runtime (e.g. an alternative register allocator selected by
/// environment flag) through a plain `fn` pointer. `Box<dyn Pass>` cannot
/// implement the foreign `Pass` trait directly (orphan rule), hence the
/// newtype.
pub struct DynPass(pub Box<dyn Pass>);

impl DynPass {
    pub fn new(pass: impl Pass + 'static) -> Self {
        DynPass(Box::new(pass))
    }
}

impl Pass for DynPass {
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

/// Pipeline printing configuration, mirroring upstream pliron's `PMConfig`:
/// when `print_after_all` is set, [Passes::run] writes
/// `{count}-after-{name}.plir` dumps into `ir_printing_dir`.
#[derive(Default, Clone)]
pub struct PMConfig {
    pub print_after_all: bool,
    pub ir_printing_dir: Option<std::path::PathBuf>,
}

/// A sequential pass runner, mirroring upstream pliron's `Passes`: runs
/// each added [Pass] on the same root operation in order. Analyses are
/// conservatively discarded after every IR-changing pass.
#[derive(Default)]
pub struct Passes {
    passes: Vec<Box<dyn Pass>>,
    config: PMConfig,
}

/// Callback invoked after each pass in [Passes::run_observed]: receives the
/// pass index, name, and the module it just ran on.
pub type PassObserver<'a> =
    dyn FnMut(usize, &str, &Context, Ptr<Operation>) -> PassControl + 'a;

/// What [Passes::run_observed]'s observer returns: keep going or stop the
/// pipeline cleanly after the current pass (used for cancellation and for
/// "run the first k passes" replay).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassControl {
    Continue,
    Stop,
}

impl Passes {
    pub fn add_pass(&mut self, pass: impl Pass + 'static) {
        self.passes.push(Box::new(pass));
    }

    /// Move every pass of `other` to the end of this pipeline, flattening
    /// it (unlike `add_pass(other)`, which would nest it as a single
    /// opaque entry — invisible to per-pass observers).
    pub fn extend(&mut self, other: Passes) {
        self.passes.extend(other.passes);
    }

    /// The passes' names, in run order.
    pub fn names(&self) -> Vec<String> {
        self.passes.iter().map(|p| p.name().to_string()).collect()
    }

    pub fn len(&self) -> usize {
        self.passes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.passes.is_empty()
    }

    /// [Passes::run], but calling `observer` after every pass with the
    /// pass index, its name, and the context/root (for progress reporting
    /// or IR capture). The observer returning [PassControl::Stop] ends the
    /// run cleanly (Ok) without executing the remaining passes.
    pub fn run_observed(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        analyses: &mut AnalysisManager,
        observer: &mut PassObserver<'_>,
    ) -> pliron::result::Result<PassResult> {
        let mut aggregate = changed();
        for (count, pass) in self.passes.iter_mut().enumerate() {
            let result = pass.run(op, ctx, analyses)?;
            if matches!(result.ir_changed, pliron::irbuild::IRStatus::Changed) {
                *analyses = AnalysisManager::default();
            }
            aggregate.ir_changed = match (aggregate.ir_changed, result.ir_changed) {
                (pliron::irbuild::IRStatus::Changed, _)
                | (_, pliron::irbuild::IRStatus::Changed) => {
                    pliron::irbuild::IRStatus::Changed
                }
                _ => pliron::irbuild::IRStatus::Unchanged,
            };
            if observer(count, pass.name(), ctx, op) == PassControl::Stop {
                break;
            }
        }
        Ok(aggregate)
    }

    pub fn set_config(&mut self, config: PMConfig) {
        self.config = config;
    }

    pub fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        analyses: &mut AnalysisManager,
    ) -> pliron::result::Result<PassResult> {
        let mut aggregate = changed();
        for (count, pass) in self.passes.iter_mut().enumerate() {
            let result = pass.run(op, ctx, analyses)?;
            if matches!(result.ir_changed, pliron::irbuild::IRStatus::Changed) {
                // No fine-grained invalidation: every cached analysis is
                // dropped once the IR moved under it.
                *analyses = AnalysisManager::default();
            }
            if self.config.print_after_all
                && let Some(dir) = &self.config.ir_printing_dir
            {
                let dump = op.disp(ctx).to_string();
                let path = dir.join(format!("{count}-after-{}.plir", pass.name()));
                let _ = std::fs::write(path, dump);
            }
            aggregate.ir_changed = match (aggregate.ir_changed, result.ir_changed) {
                (pliron::irbuild::IRStatus::Changed, _)
                | (_, pliron::irbuild::IRStatus::Changed) => {
                    pliron::irbuild::IRStatus::Changed
                }
                _ => pliron::irbuild::IRStatus::Unchanged,
            };
        }
        Ok(aggregate)
    }
}

impl Pass for Passes {
    fn name(&self) -> &str {
        "passes"
    }

    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        analyses: &mut AnalysisManager,
    ) -> pliron::result::Result<PassResult> {
        Passes::run(self, op, ctx, analyses)
    }
}
