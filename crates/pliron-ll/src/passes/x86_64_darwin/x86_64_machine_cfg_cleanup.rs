use crate::{
    context::{Context, Ptr},
    ir::operation::Operation,
    conversion::pass::{Pass, PassOptions},
    result::STAIRResult,
};

pub struct X86_64MachineCfgCleanupPass;

impl Pass for X86_64MachineCfgCleanupPass {
    fn name(&self) -> &str {
        "x86-64-machine-cfg-cleanup"
    }

    fn run(
        &self,
        root: Ptr<Operation>,
        _ctx: &mut Context,
        _options: PassOptions,
    ) -> STAIRResult<Ptr<Operation>> {
        Ok(root)
    }
}
