//! pliron-inspect driver linking the crabbit dialect stack:
//! mir (cuda-oxide's dialect-mir), llvm, ll, aarch64, x86_64, macho — plus
//! the mid-level pass pipeline.

use std::path::PathBuf;

use clap::Parser;
use crabbit_mir::passes::lower_dialect_mir::LowerDialectMirPass;
use pliron::context::{Context, Ptr};
use pliron::operation::Operation;
use pliron_inspect_driver::{DriverHooks, run_stdio_driver};
use pliron_ll::conversion::pass::{AnalysisManager, Mem2RegPass, Pass};
use pliron_ll::passes::llvm::{
    inline::LLVMInlinePass, pin_type_punned_slots::LLVMPinTypePunnedSlotsPass,
    simplify::LLVMSimplifyPass, simplify_cfg::LLVMSimplifyCfgPass,
    sroa::LLVMSroaPass,
};
use pliron_ll::passes::verify::VerifyPass;

#[derive(Parser)]
#[command(name = "crabbit-inspect-driver")]
#[command(about = "crabbit IR driver for pliron-inspect")]
struct Args {
    /// Input IR file
    input: Option<PathBuf>,
}

struct CrabbitHooks {
    passes: Vec<Box<dyn Pass>>,
}

impl CrabbitHooks {
    fn new() -> Self {
        let passes: Vec<Box<dyn Pass>> = vec![
            Box::new(VerifyPass::new()),
            Box::new(LowerDialectMirPass),
            Box::new(LLVMPinTypePunnedSlotsPass),
            Box::new(Mem2RegPass),
            Box::new(LLVMInlinePass::default()),
            Box::new(LLVMSimplifyPass),
            Box::new(LLVMSimplifyCfgPass),
            Box::new(LLVMSroaPass),
        ];
        CrabbitHooks { passes }
    }
}

impl DriverHooks for CrabbitHooks {
    fn pass_names(&self) -> Vec<String> {
        self.passes.iter().map(|p| p.name().to_string()).collect()
    }

    fn run_pass(
        &self,
        name: &str,
        root: Ptr<Operation>,
        ctx: &mut Context,
    ) -> Result<Ptr<Operation>, String> {
        let Some(pass) = self.passes.iter().find(|p| p.name() == name) else {
            return Err(format!("unknown pass: {name}"));
        };
        let mut analyses = AnalysisManager::default();
        pass.run(root, ctx, &mut analyses)
            .map(|_| root)
            .map_err(|e| format!("{e}"))
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    run_stdio_driver(&CrabbitHooks::new(), args.input.as_deref())
}
