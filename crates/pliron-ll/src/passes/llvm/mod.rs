pub mod adce;
pub mod analysis;
pub mod div_strength_reduce;
pub mod dse;
pub mod gvn;
pub mod inline;
pub mod licm;
pub mod loop_carried_fwd;
pub mod midend_gate;
pub mod op_ids;
pub mod pin_type_punned_slots;
pub mod simplify;
pub mod simplify_cfg;
pub mod sink;
pub mod sroa;
pub mod unroll;
pub mod vectorize;

use crate::conversion::pass::{Mem2RegPass, Passes};
use crate::target_profile::TargetProfile;

/// The target-independent LLVM-dialect mid-end, starting from freshly
/// lowered (post-`lower-dialect-mir`) IR: inlining, simplification, SROA +
/// mem2reg (twice), then the research passes (gvn, div-strength-reduce,
/// licm, gvn, unroll, then the backward-dataflow round: global-dse, adce, sink —
/// each self-disables via [midend_gate]) and a final clean-up round.
/// Shared by crabbit's host and kernel pipelines and by the resident
/// driver/server, so the pass list can never drift between them. The
/// `profile` parameterizes profitability decisions per target
/// ([TargetProfile], crabbit's miniature TargetTransformInfo) — the pass
/// LIST is identical everywhere, the answers passes get are not.
pub fn add_llvm_midend_passes(passes: &mut Passes, profile: &TargetProfile) {
    // Attribution head: source ids for the backward lift (no-op unless
    // CRABBIT_BLOCKMAP/CRABBIT_PROFILE_MAP is set).
    passes.add_pass(op_ids::LlvmOpIdPass);
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
    // Loop-carried store→load forwarding needs gvn's CSE of the shared
    // index root and licm's hoisting; the dead address ops it strands are
    // erased by the dse/adce round below.
    passes.add_pass(loop_carried_fwd::LLVMLoopCarriedFwdPass);
    // Full unroll AFTER gvn/licm (so invariants are already hoisted and
    // the loop body is minimal) and BEFORE the backward round, whose
    // adce/simplify erase the dead per-iteration iv/compare clones and
    // constant-fold the per-iteration address math.
    passes.add_pass(unroll::LLVMUnrollPass::new(profile));
    // Backward-dataflow round (docs/MIDEND-PLAN.md items 6–8): dead
    // stores first (they would anchor adce roots), then dead code, then
    // sinking on the cleaned function.
    passes.add_pass(dse::LLVMGlobalDsePass);
    passes.add_pass(adce::LLVMAdcePass);
    passes.add_pass(sink::LLVMSinkPass::new(profile));
    passes.add_pass(simplify::LLVMSimplifyPass);
    passes.add_pass(simplify_cfg::LLVMSimplifyCfgPass);
    passes.add_pass(simplify::LLVMSimplifyPass);
    // LAST on purpose: no other mid-end pass ever sees the crabbit-internal
    // vector ops, so the op-safety tables need no entries for them.
    // Self-disables on divergent targets and backends without SIMD lowering
    // ([TargetProfile::simd128]).
    passes.add_pass(vectorize::LLVMVectorizePass::new(profile));
}
