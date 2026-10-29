use crate::{
    context::{Context, Ptr},
    conversion::pass::{AnalysisManager, Pass, PassResult, changed},
    ir::operation::Operation,
};

pub struct X86_64TargetOptsPreRaPass;

impl Pass for X86_64TargetOptsPreRaPass {
    fn name(&self) -> &str {
        "x86-64-target-opts-pre-ra"
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
