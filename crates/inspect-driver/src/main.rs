//! pliron-inspect driver linking the crabbit dialect stack:
//! mir, llvm, ll, aarch64, x86_64, macho — plus the mid-level pass pipeline.

use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use llvm_compat::conversion::pass::{Pass, PassOptions, PassObj};
use llvm_compat::passes::llvm::{
    inline::LLVMInlinePass, mem2reg::Mem2RegPass, simplify::LLVMSimplifyPass,
    simplify_cfg::LLVMSimplifyCfgPass, sroa::LLVMSroaPass,
};
use llvm_compat::passes::lower_llvm_block_args_to_phi::LowerLLVMBlockArgsToPhiPass;
use llvm_compat::passes::lower_llvm_phi_to_block_args::LowerLLVMPhiToBlockArgsPass;
use llvm_compat::passes::verify::VerifyPass;
use crabbit_mir::passes::convert_mir_to_llvm::ConvertMirToLLVMPass;
use pliron::context::{Context, Ptr};
use pliron::operation::Operation;
use pliron_inspect_driver::{DriverHooks, run_stdio_driver};

#[derive(Parser)]
#[command(name = "crabbit-inspect-driver")]
#[command(about = "crabbit IR driver for pliron-inspect")]
struct Args {
    /// Input IR file
    input: Option<PathBuf>,
}

struct CrabbitHooks {
    passes: Vec<PassObj>,
}

impl CrabbitHooks {
    fn new() -> Self {
        let passes: Vec<PassObj> = vec![
            Arc::new(VerifyPass::new()),
            Arc::new(ConvertMirToLLVMPass),
            Arc::new(LowerLLVMBlockArgsToPhiPass),
            Arc::new(Mem2RegPass::default()),
            Arc::new(LLVMInlinePass::default()),
            Arc::new(LLVMSimplifyPass),
            Arc::new(LLVMSimplifyCfgPass),
            Arc::new(LLVMSroaPass),
            Arc::new(LowerLLVMPhiToBlockArgsPass),
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
        pass.run(root, ctx, PassOptions::default())
            .map_err(|e| format!("{e}"))
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    run_stdio_driver(&CrabbitHooks::new(), args.input.as_deref())
}
