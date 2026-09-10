//! pliron [Pass] wrapper over cuda-oxide's `mir-lower` (dialect-mir → llvm
//! dialect lowering).
//!
//! [dialect_mir] and crabbit's own `mir` dialect (`crate::mir`) both register
//! under the dialect name `mir` with overlapping op names, so a [Context] can
//! hold only one of them. This pass belongs to a dialect-mir pipeline; running
//! it in a context where crabbit's dialect is registered is a logic error.

use pliron::{
    context::{Context, Ptr},
    operation::Operation,
    result::Result,
};
use pliron_ll::conversion::pass::{AnalysisManager, Pass, PassResult, changed};

/// Lowers a module of [dialect_mir] ops to the [pliron_llvm] dialect by
/// delegating to [mir_lower::lower_mir_to_llvm].
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
        Ok(changed())
    }
}

// Linkage is carried across the lowering by mir-lower itself (the
// `crabbit-patches` series propagates the `mir_func_linkage` attribute onto
// the lowered `llvm.func`), so no stamp pass is needed here anymore.
