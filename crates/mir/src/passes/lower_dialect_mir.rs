//! pliron [Pass] wrapper over cuda-oxide's `mir-lower` (dialect-mir → llvm
//! dialect lowering).
//!
//! [dialect_mir] and crabbit's own [mir](crate::mir) dialect both register
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
        &self,
        root: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        mir_lower::lower_mir_to_llvm(ctx, root)?;
        Ok(changed())
    }
}

/// Stamps function linkage onto every `llvm.func` after [LowerDialectMirPass]:
/// mir-lower creates functions without a linkage attribute, while crabbit's
/// llvm passes and object backends require one. Functions whose symbol is in
/// `internal` get internal linkage (lazily imported upstream instances);
/// everything else that lacks a linkage becomes external. Declarations the
/// importer inserted directly as `llvm.func` keep whatever they carry.
pub struct StampFunctionLinkagePass {
    internal: std::collections::HashSet<String>,
}

impl StampFunctionLinkagePass {
    pub fn new(internal: impl IntoIterator<Item = String>) -> Self {
        StampFunctionLinkagePass {
            internal: internal.into_iter().collect(),
        }
    }
}

impl Pass for StampFunctionLinkagePass {
    fn name(&self) -> &str {
        "stamp-function-linkage"
    }

    fn run(
        &self,
        root: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        use pliron::builtin::op_interfaces::SymbolOpInterface;
        use pliron::linked_list::ContainsLinkedList;
        use pliron::op::Op;
        use pliron_llvm::attributes::LinkageAttr;

        let funcs: Vec<(Ptr<Operation>, String)> = root
            .deref(ctx)
            .regions()
            .flat_map(|region| region.deref(ctx).iter(ctx))
            .flat_map(|block| block.deref(ctx).iter(ctx))
            .filter_map(|op| {
                let func = pliron::operation::Operation::get_op::<pliron_llvm::ops::FuncOp>(
                    op, ctx,
                )?;
                if func.get_attr_llvm_function_linkage(ctx).is_some() {
                    return None;
                }
                Some((op, func.get_symbol_name(ctx).to_string()))
            })
            .collect();
        for (op, symbol) in funcs {
            let linkage = if self.internal.contains(&symbol) {
                LinkageAttr::InternalLinkage
            } else {
                LinkageAttr::ExternalLinkage
            };
            pliron_llvm::ops::FuncOp::from_operation(op)
                .set_attr_llvm_function_linkage(ctx, linkage);
        }
        Ok(changed())
    }
}
