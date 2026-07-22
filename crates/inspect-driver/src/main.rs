//! pliron-inspect driver linking the crabbit dialect stack:
//! mir, llvm, ll, aarch64, x86_64, macho — plus the mid-level pass pipeline.

use std::cell::RefCell;
use std::path::PathBuf;

use clap::Parser;
use pliron::pass::{AnalysisManager, Pass, PassManager, Passes};
use pliron::opts::mem2reg::Mem2RegPass;
use pliron_ll::passes::llvm::{
    inline::LLVMInlinePass, pin_type_punned_slots::LLVMPinTypePunnedSlotsPass,
    simplify::LLVMSimplifyPass, simplify_cfg::LLVMSimplifyCfgPass,
    sroa::LLVMSroaPass,
};
use pliron_ll::passes::verify::VerifyPass;
use crabbit_mir::passes::convert_mir_to_llvm::convert_mir_to_llvm_pass;
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

/// `DriverHooks::run_pass` is `&self` (it's `pliron-inspect`'s fixed driver
/// protocol), but running a real `pliron::pass::Pass` needs `&mut`. Each
/// pass is behind a `RefCell` so `run_pass` can borrow one mutably for the
/// one call it needs, without owning `&mut self`.
struct CrabbitHooks {
    passes: Vec<RefCell<Box<dyn Pass>>>,
}

impl CrabbitHooks {
    fn new() -> Self {
        let passes: Vec<Box<dyn Pass>> = vec![
            Box::new(VerifyPass::new()),
            Box::new(convert_mir_to_llvm_pass()),
            Box::new(LLVMPinTypePunnedSlotsPass),
            Box::new(Mem2RegPass),
            Box::new(LLVMInlinePass::default()),
            Box::new(LLVMSimplifyPass),
            Box::new(LLVMSimplifyCfgPass),
            Box::new(LLVMSroaPass),
        ];
        CrabbitHooks {
            passes: passes.into_iter().map(RefCell::new).collect(),
        }
    }
}

impl DriverHooks for CrabbitHooks {
    fn pass_names(&self) -> Vec<String> {
        self.passes
            .iter()
            .map(|p| p.borrow().name().to_string())
            .collect()
    }

    fn run_pass(
        &self,
        name: &str,
        root: Ptr<Operation>,
        ctx: &mut Context,
    ) -> Result<Ptr<Operation>, String> {
        let Some(cell) = self.passes.iter().find(|p| p.borrow().name() == name) else {
            return Err(format!("unknown pass: {name}"));
        };
        let mut pass = cell.borrow_mut();
        let mut analyses = AnalysisManager::default();
        <Passes as PassManager>::run_pass(pass.as_mut(), root, ctx, &mut analyses)
            .map(|_| root)
            .map_err(|e| format!("{e}"))
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    run_stdio_driver(&CrabbitHooks::new(), args.input.as_deref())
}
