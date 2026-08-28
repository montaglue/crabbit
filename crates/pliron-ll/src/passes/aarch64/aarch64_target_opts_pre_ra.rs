use crate::{
    context::{Context, Ptr},
    ir::operation::Operation,
    conversion::pass::{AnalysisManager, Pass, PassResult, changed},
};

pub struct Aarch64TargetOptsPreRaPass;

impl Pass for Aarch64TargetOptsPreRaPass {
    fn name(&self) -> &str {
        "aarch64-target-opts-pre-ra"
    }

    fn run(&mut self, _root: Ptr<Operation>, _ctx: &mut Context, _analyses: &mut AnalysisManager) -> pliron::result::Result<PassResult> {
        Ok(changed())
    }
}
