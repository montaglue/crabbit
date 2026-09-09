//! Per-target answers to the mid-end's profitability questions — crabbit's
//! miniature of LLVM's TargetTransformInfo. The mid-end pass *list* is
//! shared by every pipeline (host, kernel, resident server) so it cannot
//! drift; what differs per target is the answers passes get when they ask
//! whether a transformation pays. First consumer: [sink] refuses to add
//! control dependence on divergent targets (measured on gemm_tiled:
//! sinking two adds under a divergent tile-bound guard cost +10% on the
//! GB10 while the same move relieved spill pressure on the Cortex-X925 —
//! docs/MIDEND-PLAN.md "Sink cross-target finding").
//!
//! [sink]: crate::passes::llvm::sink

/// What the mid-end may assume about the machine it is compiling for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetProfile {
    /// Branches may diverge within a hardware execution group (SIMT
    /// warps): a "skipped" path under a non-uniform guard is still
    /// executed by the group, so moving work under control flow saves
    /// nothing and may lengthen convergence-critical paths.
    pub has_branch_divergence: bool,
    /// Hardware integer division is cheap enough that expanding a
    /// constant-divisor divide into a multiply-high sequence is not an
    /// obvious win. Not consumed yet (div-strength-reduce currently
    /// mirrors LLVM's always-expand); recorded for the measured,
    /// per-site policy the backward-attribution work enables.
    pub int_div_cheap: bool,
}

impl TargetProfile {
    /// The host CPU (aarch64/x86_64 machine pipelines).
    pub const fn host_cpu() -> Self {
        TargetProfile {
            has_branch_divergence: false,
            int_div_cheap: false,
        }
    }

    /// The GPU kernel pipeline (NVPTX translation; ptxas owns the machine
    /// work downstream).
    pub const fn gpu_kernel() -> Self {
        TargetProfile {
            has_branch_divergence: true,
            int_div_cheap: false,
        }
    }
}
