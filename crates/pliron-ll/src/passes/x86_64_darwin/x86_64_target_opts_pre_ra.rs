use crate::{
    context::{Context, Ptr},
    ir::operation::Operation,
    conversion::pass::{AnalysisManager, Pass, PassResult, changed},
};

pub struct X86_64TargetOptsPreRaPass;

impl Pass for X86_64TargetOptsPreRaPass {
    fn name(&self) -> &str {
        "x86-64-target-opts-pre-ra"
    }

    fn run(&mut self, _root: Ptr<Operation>, _ctx: &mut Context, _analyses: &mut AnalysisManager) -> pliron::result::Result<PassResult> {
        Ok(changed())
    }
}
