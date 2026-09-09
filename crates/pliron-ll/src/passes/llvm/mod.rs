pub mod adce;
pub mod analysis;
pub mod div_strength_reduce;
pub mod dse;
pub mod gvn;
pub mod inline;
pub mod licm;
pub mod midend_gate;
pub mod pin_type_punned_slots;
pub mod simplify;
pub mod simplify_cfg;
pub mod sink;
pub mod sroa;

use crate::conversion::pass::{Mem2RegPass, Passes};
use crate::target_profile::TargetProfile;

/// The target-independent LLVM-dialect mid-end, starting from freshly
/// lowered (post-`lower-dialect-mir`) IR: inlining, simplification, SROA +
/// mem2reg (twice), then the research passes (gvn, div-strength-reduce,
/// licm, gvn, then the backward-dataflow round: global-dse, adce, sink —
/// each self-disables via [midend_gate]) and a final clean-up round.
/// Shared by crabbit's host and kernel pipelines and by the resident
/// driver/server, so the pass list can never drift between them. The
/// `profile` parameterizes profitability decisions per target
/// ([TargetProfile], crabbit's miniature TargetTransformInfo) — the pass
/// LIST is identical everywhere, the answers passes get are not.
pub fn add_llvm_midend_passes(passes: &mut Passes, profile: &TargetProfile) {
    passes.add_pass(inline::LLVMInlinePass::default());
    passes.add_pass(simplify::LLVMSimplifyPass);
    passes.add_pass(simplify_cfg::LLVMSimplifyCfgPass);
    passes.add_pass(sroa::LLVMSroaPass);
    passes.add_pass(simplify::LLVMSimplifyPass);
    passes.add_pass(pin_type_punned_slots::LLVMPinTypePunnedSlotsPass);
    passes.add_pass(Mem2RegPass);
    passes.add_pass(simplify::LLVMSimplifyPass);
    passes.add_pass(simplify_cfg::LLVMSimplifyCfgPass);
    passes.add_pass(simplify::LLVMSimplifyPass);
    passes.add_pass(sroa::LLVMSroaPass);
    passes.add_pass(simplify::LLVMSimplifyPass);
    passes.add_pass(pin_type_punned_slots::LLVMPinTypePunnedSlotsPass);
    passes.add_pass(Mem2RegPass);
    passes.add_pass(simplify::LLVMSimplifyPass);
    passes.add_pass(simplify_cfg::LLVMSimplifyCfgPass);
    passes.add_pass(simplify::LLVMSimplifyPass);
    passes.add_pass(gvn::LLVMGvnPass);
    passes.add_pass(div_strength_reduce::LLVMDivStrengthReducePass);
    passes.add_pass(licm::LLVMLicmPass);
    passes.add_pass(gvn::LLVMGvnPass);
    // Backward-dataflow round (docs/MIDEND-PLAN.md items 6–8): dead
    // stores first (they would anchor adce roots), then dead code, then
    // sinking on the cleaned function.
    passes.add_pass(dse::LLVMGlobalDsePass);
    passes.add_pass(adce::LLVMAdcePass);
    passes.add_pass(sink::LLVMSinkPass::new(profile));
    passes.add_pass(simplify::LLVMSimplifyPass);
    passes.add_pass(simplify_cfg::LLVMSimplifyCfgPass);
    passes.add_pass(simplify::LLVMSimplifyPass);
}
