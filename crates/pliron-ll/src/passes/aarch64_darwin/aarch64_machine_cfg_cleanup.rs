use crate::{
    context::{Context, Ptr},
    ir::operation::Operation,
    conversion::pass::{Pass, PassOptions},
    result::STAIRResult,
};

pub struct Aarch64MachineCfgCleanupPass;

impl Pass for Aarch64MachineCfgCleanupPass {
    fn name(&self) -> &str {
        "aarch64-machine-cfg-cleanup"
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
