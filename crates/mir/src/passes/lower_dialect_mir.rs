//! pliron [Pass] wrapper over cuda-oxide's `mir-lower` (dialect-mir → llvm
//! dialect lowering), consumed unpatched from upstream, completed for
//! crabbit's module by [erase_mir_types](super::erase_mir_types).

use pliron::{
    context::{Context, Ptr},
    operation::Operation,
    result::Result,
};
use pliron_ll::conversion::pass::{AnalysisManager, Pass, PassResult, changed};

/// Lowers a module of [dialect_mir] ops to the [pliron_llvm] dialect by
/// delegating to [mir_lower::lower_mir_to_llvm]. That entry point verifies
/// the whole module against dialect-mir's typing contract before it
/// lowers anything, so the importer's output must meet the contract
/// (docs/MIR-DIALECT.md); there is no lenient mode. The MIR types the
/// lowering leaves on the importer's llvm-dialect ops are erased right
/// after it, so the module this pass hands on is in the LLVM dialect.
/// Linkage is restored afterwards by [super::linkage::ApplyModuleLinkagePass].
pub struct LowerDialectMirPass;

impl Pass for LowerDialectMirPass {
    fn name(&self) -> &str {
        "lower-dialect-mir"
    }

    fn run(
        &mut self,
        root: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        mir_lower::lower_mir_to_llvm(ctx, root)?;
        super::erase_mir_types::erase(ctx, root)?;
        Ok(changed())
    }
}
