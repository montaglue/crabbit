//! Verification Pass
//!
//! This pass runs verification on all operations in the IR, similar to MLIR's
//! builtin verification infrastructure. It can be inserted between other passes
//! to ensure IR validity at each stage.
//!
//! # Example
//! ```ignore
//! let mut ctx = Context::new();
//! VerifyPass::register(&mut ctx);
//!
//! // Run verification after SSA conversion
//! let verify_pass = VerifyPass::new();
//! verify_pass.run(module_op, &mut ctx)?;
//! ```


use crate::{
    common_traits::Verify,
    context::{Context, Ptr},
    ir::operation::Operation,
    conversion::pass::{AnalysisManager, Pass, PassResult, changed},
};

/// Pass that verifies the IR is well-formed.
///
/// This pass walks the entire IR and calls verify() on each operation,
/// checking that all structural invariants are satisfied (e.g., blocks
/// end with terminators, operand types match, etc.).
///
/// In MLIR, verification runs automatically after each pass. This pass
/// allows explicit verification to be inserted into the pass pipeline.
pub struct VerifyPass;

impl VerifyPass {
    pub fn new() -> Self {
        VerifyPass
    }
}

impl Default for VerifyPass {
    fn default() -> Self {
        Self::new()
    }
}

impl Pass for VerifyPass {
    fn name(&self) -> &str {
        "verify"
    }

    fn run(&mut self, root: Ptr<Operation>, ctx: &mut Context, _analyses: &mut AnalysisManager) -> pliron::result::Result<PassResult> {
        // Verify the root operation and all nested operations recursively
        root.deref(ctx).verify(ctx)?;
        Ok(changed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pass_name() {
        let pass = VerifyPass::new();
        assert_eq!(pass.name(), "verify");
    }
}
