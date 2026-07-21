use crate::{
    context::{Context, Ptr},
    ir::operation::Operation,
    conversion::pass::{Pass, PassOptions},
    result::STAIRResult,
};

pub struct Aarch64TargetOptsPreRaPass;

impl Pass for Aarch64TargetOptsPreRaPass {
    fn name(&self) -> &str {
        "aarch64-target-opts-pre-ra"
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
