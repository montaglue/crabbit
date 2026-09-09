//! `llvm-op-ids`: the mid-end HEAD stamping for backward profile
//! attribution (docs/PROFILE-FEEDBACK-BACKWARD.md). Runs before any
//! mid-end transformation and gives every op of every `llvm.func` its
//! dense per-function SOURCE id (`ll.op_id`). Every later pass's adjoint
//! expresses new/merged ops in terms of these ids (eagerly collapsed via
//! [opmap::effective_sources]), and the RA-boundary pass
//! (`aarch64-op-ids`) preserves surviving ids, so measured machine cost
//! lifts all the way back to this numbering. Gated like all attribution
//! ([blockmap_enabled]: `CRABBIT_BLOCKMAP` / `CRABBIT_PROFILE_MAP`).

use crate::{
    context::{Context, Ptr},
    ir::operation::Operation,
    conversion::pass::{AnalysisManager, Pass, PassResult, changed, unchanged},
};

use crate::passes::aarch64::blockmap::blockmap_enabled;
use crate::passes::aarch64::opmap;

pub struct LlvmOpIdPass;

impl Pass for LlvmOpIdPass {
    fn name(&self) -> &str {
        "llvm-op-ids"
    }

    fn run(
        &mut self,
        root: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> pliron::result::Result<PassResult> {
        if !blockmap_enabled() {
            return Ok(unchanged());
        }
        opmap::assign_op_ids(ctx, root)?;
        Ok(changed())
    }
}
