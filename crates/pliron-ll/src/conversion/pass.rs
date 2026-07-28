//! Pass infrastructure: pliron's own [pliron::pass_manager] module,
//! re-exported under the crate's compatibility facade, plus the small
//! conveniences (a sequential [Passes] runner, [PMConfig] IR printing) that
//! upstream pliron only grew after the rev this workspace pins. The pin
//! follows NVlabs/cuda-oxide, which crabbit consumes unpatched; when their
//! pin moves past the upstream additions, these substitutes disappear in
//! favor of the upstream types.
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

pub use pliron::pass_manager::*;

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

/// Pipeline printing configuration, mirroring upstream pliron's newer
/// `PMConfig`: when `print_after_all` is set, [Passes::run] writes
/// `{count}-after-{name}.plir` dumps into `ir_printing_dir`.
#[derive(Default, Clone)]
pub struct PMConfig {
    pub print_after_all: bool,
    pub ir_printing_dir: Option<std::path::PathBuf>,
}

/// A sequential pass runner, mirroring upstream pliron's newer `Passes`:
/// runs each added [Pass] on the same root operation in order. Analyses are
/// conservatively discarded after every IR-changing pass.
#[derive(Default)]
pub struct Passes {
    passes: Vec<Box<dyn Pass>>,
    config: PMConfig,
}

impl Passes {
    pub fn add_pass(&mut self, pass: impl Pass + 'static) {
        self.passes.push(Box::new(pass));
    }

    pub fn set_config(&mut self, config: PMConfig) {
        self.config = config;
    }

    pub fn run(
        &self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        analyses: &mut AnalysisManager,
    ) -> pliron::result::Result<PassResult> {
        let mut aggregate = changed();
        for (count, pass) in self.passes.iter().enumerate() {
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
        &self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        analyses: &mut AnalysisManager,
    ) -> pliron::result::Result<PassResult> {
        Passes::run(self, op, ctx, analyses)
    }
}

/// [Pass] wrapper over [pliron::opts::mem2reg::mem2reg], which this pliron
/// rev only exposes as a function (upstream grew a `Mem2RegPass` later).
pub struct Mem2RegPass;

impl Pass for Mem2RegPass {
    fn name(&self) -> &str {
        "mem2reg"
    }

    fn run(
        &self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        analyses: &mut AnalysisManager,
    ) -> pliron::result::Result<PassResult> {
        let mut result = PassResult::default();
        result.ir_changed = pliron::opts::mem2reg::mem2reg(op, ctx, analyses)?;
        Ok(result)
    }
}
