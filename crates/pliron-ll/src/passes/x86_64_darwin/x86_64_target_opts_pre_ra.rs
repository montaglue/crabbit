use crate::{
    context::{Context, Ptr},
    ir::operation::Operation,
    conversion::pass::{Pass, PassOptions},
    result::STAIRResult,
};

pub struct X86_64TargetOptsPreRaPass;

impl Pass for X86_64TargetOptsPreRaPass {
    fn name(&self) -> &str {
        "x86-64-target-opts-pre-ra"
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
