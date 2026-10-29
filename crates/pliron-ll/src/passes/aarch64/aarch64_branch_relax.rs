use crate::{
    context::{Context, Ptr},
    conversion::pass::{AnalysisManager, Pass, PassResult, changed},
    ir::operation::Operation,
};

pub struct Aarch64BranchRelaxPass;

impl Pass for Aarch64BranchRelaxPass {
    fn name(&self) -> &str {
        "aarch64-branch-relax"
    }

    fn run(
        &mut self,
        _root: Ptr<Operation>,
        _ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> pliron::result::Result<PassResult> {
        Ok(changed())
    }
}
