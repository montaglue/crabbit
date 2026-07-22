//! Op interfaces proposed upstream but not yet merged into pliron core
//! (see MIGRATION-TO-PLIRON.md item #5 in the stair repo). Defined here,
//! downstream, so the workspace builds against vanilla pliron; moves
//! verbatim into the eventual upstream PR.
//!
//! [FunctionLikeInterface] (item #6) lives in `pliron-inspect-driver`
//! instead, since pliron-inspect's generic root-selection is its one
//! shared consumer; re-exported here for convenience.

use thiserror::Error;

use pliron::{
    context::Context,
    derive::op_interface,
    dict_key,
    location::Located,
    op::{Op, op_cast},
    result::Result,
    builtin::op_interfaces::BranchOpInterface,
    verify_err,
};

pub use pliron_inspect_driver::FunctionLikeInterface;

use crate::ll::BranchWeightsAttr;

dict_key!(
    /// Key for the `branch_weights` attribute.
    ATTR_KEY_BRANCH_WEIGHTS, "branch_weights"
);

#[derive(Error, Debug)]
pub enum WeightedBranchOpInterfaceVerifyErr {
    #[error("Op has {weights} branch weights, but {succs} successors")]
    WeightCountMismatch { weights: usize, succs: usize },
    #[error("branch weights must not all be zero")]
    AllZeroWeights,
}

/// A [branch](BranchOpInterface) whose successors can carry relative
/// probabilities, stored as integer weights in the `branch_weights` attribute
/// (one weight per successor; an edge's probability is its weight divided by
/// the sum of all weights on the op). Mirrors MLIR's WeightedBranchOpInterface
/// and LLVM's `!prof branch_weights` metadata.
#[op_interface]
pub trait WeightedBranchOpInterface: BranchOpInterface {
    /// Get this op's successor weights, if any were attached.
    fn successor_weights(&self, ctx: &Context) -> Option<Vec<u32>> {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<BranchWeightsAttr>(&ATTR_KEY_BRANCH_WEIGHTS)
            .map(|attr| attr.0.clone())
    }

    /// Set this op's successor weights; one weight per successor.
    fn set_successor_weights(&self, ctx: &Context, weights: Vec<u32>) {
        self.get_operation().deref_mut(ctx).attributes.set(
            ATTR_KEY_BRANCH_WEIGHTS.clone(),
            BranchWeightsAttr(weights),
        );
    }

    fn verify(op: &dyn Op, ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        let self_op = op_cast::<dyn WeightedBranchOpInterface>(op).unwrap();
        let Some(weights) = self_op.successor_weights(ctx) else {
            return Ok(());
        };
        let succs = op.get_operation().deref(ctx).successors().count();
        if weights.len() != succs {
            return verify_err!(
                op.loc(ctx),
                WeightedBranchOpInterfaceVerifyErr::WeightCountMismatch {
                    weights: weights.len(),
                    succs,
                }
            );
        }
        if weights.iter().all(|weight| *weight == 0) {
            return verify_err!(
                op.loc(ctx),
                WeightedBranchOpInterfaceVerifyErr::AllZeroWeights
            );
        }
        Ok(())
    }
}
