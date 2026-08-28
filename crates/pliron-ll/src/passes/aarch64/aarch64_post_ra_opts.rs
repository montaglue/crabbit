use crate::{
    context::{Context, Ptr},
    ir::operation::Operation,
    conversion::pass::{AnalysisManager, Pass, PassResult, changed},
};

pub struct Aarch64PostRaOptsPass;

impl Pass for Aarch64PostRaOptsPass {
    fn name(&self) -> &str {
        "aarch64-post-ra-opts"
    }

    fn run(&mut self, _root: Ptr<Operation>, _ctx: &mut Context, _analyses: &mut AnalysisManager) -> pliron::result::Result<PassResult> {
        Ok(changed())
    }
}
