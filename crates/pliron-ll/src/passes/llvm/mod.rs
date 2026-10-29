pub mod adce;
pub mod analysis;
pub mod backward;
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

use crate::conversion::pass::{Mem2RegPass, Pass, Passes, into_passes, passes_from_order};
use crate::target_profile::PipelineTarget;

/// The default order of the target-independent LLVM-dialect mid-end,
/// starting from freshly lowered (post-`lower-dialect-mir`) IR: inlining,
/// simplification, SROA + mem2reg (twice), then gvn, div-strength-reduce,
/// licm, gvn, loop-carried forwarding, unroll, the backward-dataflow round
/// (global-dse, adce, sink — each self-disables via [midend_gate]), a
/// final clean-up round and the vectorizer. Shared by every host and
/// kernel pipeline and by the resident driver/server; what differs per
/// target is the [TargetProfile] the passes are built with ([construct]).
/// crabbit-config builds pipelines from this order and lets a registered
/// order replace it.
///
/// [TargetProfile]: crate::target_profile::TargetProfile
pub const DEFAULT_ORDER: &[&str] = &[
    // Attribution head: source ids for the backward lift (no-op unless
    // CRABBIT_BLOCKMAP/CRABBIT_PROFILE_MAP is set).
    "llvm-op-ids",
    "llvm-inline",
    "llvm-simplify",
    "llvm-simplify-cfg",
    "llvm-sroa",
    "llvm-simplify",
    "llvm-pin-type-punned-slots",
    "mem2reg",
    "llvm-simplify",
    "llvm-simplify-cfg",
    "llvm-simplify",
    "llvm-sroa",
    "llvm-simplify",
    "llvm-pin-type-punned-slots",
    "mem2reg",
    "llvm-simplify",
    "llvm-simplify-cfg",
    "llvm-simplify",
    "llvm-gvn",
    "llvm-div-strength-reduce",
    "llvm-licm",
    "llvm-gvn",
    // Loop-carried store→load forwarding needs gvn's CSE of the shared
    // index root and licm's hoisting; the dead address ops it strands are
    // erased by the dse/adce round below.
    "llvm-loop-carried-fwd",
    // Full unroll AFTER gvn/licm (so invariants are already hoisted and
    // the loop body is minimal) and BEFORE the backward round, whose
    // adce/simplify erase the dead per-iteration iv/compare clones and
    // constant-fold the per-iteration address math.
    "llvm-unroll",
    // Backward-dataflow round (docs/MIDEND-PLAN.md items 6–8): dead
    // stores first (they would anchor adce roots), then dead code, then
    // sinking on the cleaned function.
    "llvm-global-dse",
    "llvm-adce",
    "llvm-sink",
    "llvm-simplify",
    "llvm-simplify-cfg",
    "llvm-simplify",
    // LAST on purpose: no other mid-end pass ever sees the crabbit-internal
    // vector ops, so the op-safety tables need no entries for them.
    // Self-disables on divergent targets and backends without SIMD lowering
    // ([TargetProfile::simd128](crate::target_profile::TargetProfile::simd128)).
    "llvm-vectorize",
];

/// The mid-end pass called `name`, built for `target`; `None` for a name
/// that is not a mid-end pass.
pub fn construct(name: &str, target: &PipelineTarget) -> Option<Box<dyn Pass>> {
    let profile = &target.profile;
    Some(match name {
        "llvm-op-ids" => Box::new(op_ids::LlvmOpIdPass),
        "llvm-inline" => Box::new(inline::LLVMInlinePass::default()),
        "llvm-simplify" => Box::new(simplify::LLVMSimplifyPass),
        "llvm-simplify-cfg" => Box::new(simplify_cfg::LLVMSimplifyCfgPass),
        "llvm-sroa" => Box::new(sroa::LLVMSroaPass),
        "llvm-pin-type-punned-slots" => Box::new(pin_type_punned_slots::LLVMPinTypePunnedSlotsPass),
        "mem2reg" => Box::new(Mem2RegPass),
        "llvm-gvn" => Box::new(gvn::LLVMGvnPass),
        "llvm-div-strength-reduce" => Box::new(div_strength_reduce::LLVMDivStrengthReducePass),
        "llvm-licm" => Box::new(licm::LLVMLicmPass),
        "llvm-loop-carried-fwd" => Box::new(loop_carried_fwd::LLVMLoopCarriedFwdPass),
        "llvm-unroll" => Box::new(unroll::LLVMUnrollPass::new(profile)),
        "llvm-global-dse" => Box::new(dse::LLVMGlobalDsePass),
        "llvm-adce" => Box::new(adce::LLVMAdcePass),
        "llvm-sink" => Box::new(sink::LLVMSinkPass::new(profile)),
        "llvm-vectorize" => Box::new(vectorize::LLVMVectorizePass::new(profile)),
        _ => return None,
    })
}

/// The default mid-end ([DEFAULT_ORDER]) for `target`, for callers that
/// run crabbit's passes without crabbit-config (this crate's tests,
/// offline tools).
pub fn default_midend(target: &PipelineTarget) -> Passes {
    into_passes(
        passes_from_order(DEFAULT_ORDER, |name| construct(name, target))
            .expect("DEFAULT_ORDER names only passes construct knows"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every name in [DEFAULT_ORDER] constructs a pass reporting that name,
    /// for the host and the kernel targets alike.
    #[test]
    fn default_order_constructs_passes_by_their_names() {
        let host = PipelineTarget::host(crate::triple::Triple::parse("aarch64-unknown-linux-gnu"));
        for target in [host, PipelineTarget::nvptx_kernel()] {
            for name in DEFAULT_ORDER {
                let pass = construct(name, &target).unwrap_or_else(|| panic!("`{name}` does not construct"));
                assert_eq!(pass.name(), *name);
            }
        }
    }
}
