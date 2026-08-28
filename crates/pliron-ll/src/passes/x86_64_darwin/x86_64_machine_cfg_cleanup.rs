use crate::{
    context::{Context, Ptr},
    ir::operation::Operation,
    conversion::pass::{AnalysisManager, Pass, PassResult, changed},
};

pub struct X86_64MachineCfgCleanupPass;

impl Pass for X86_64MachineCfgCleanupPass {
    fn name(&self) -> &str {
        "x86-64-machine-cfg-cleanup"
    }

    fn run(&mut self, _root: Ptr<Operation>, _ctx: &mut Context, _analyses: &mut AnalysisManager) -> pliron::result::Result<PassResult> {
        Ok(changed())
    }
}
