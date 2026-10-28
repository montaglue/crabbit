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
    /// The backend can select and allocate 128-bit vector operations
    /// (aarch64 NEON today; x86_64's isel has no vector lowering).
    /// Consumed by [vectorize]: a vectorized module would fail isel on a
    /// backend without it.
    ///
    /// [vectorize]: crate::passes::llvm::vectorize
    pub simd128: bool,
}

impl TargetProfile {
    /// The host CPU (aarch64/x86_64 machine pipelines). SIMD is opt-in
    /// per target via [Self::with_simd128] — only the aarch64 backends
    /// lower vector ops.
    pub const fn host_cpu() -> Self {
        TargetProfile {
            has_branch_divergence: false,
            int_div_cheap: false,
            simd128: false,
        }
    }

    /// The GPU kernel pipeline (NVPTX translation; ptxas owns the machine
    /// work downstream). `simd128` stays off: SIMT lanes are the
    /// parallelism model, and the NVPTX translator does not know the
    /// vector ops.
    pub const fn gpu_kernel() -> Self {
        TargetProfile {
            has_branch_divergence: true,
            int_div_cheap: false,
            simd128: false,
        }
    }

    pub const fn with_simd128(mut self, simd128: bool) -> Self {
        self.simd128 = simd128;
        self
    }
}

/// What a pipeline compiles for: the target triple — the LLVM-style
/// identity of the pipeline (the kernel module compiles for
/// `nvptx64-nvidia-cuda`) — and the [TargetProfile] answers the mid-end's
/// profitability questions get.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PipelineTarget {
    pub triple: crate::triple::Triple,
    pub profile: TargetProfile,
}

impl PipelineTarget {
    /// A host pipeline for `triple`: the CPU profile, with SIMD on the
    /// backends that lower the mid-end's vector ops (aarch64 only).
    pub fn host(triple: crate::triple::Triple) -> Self {
        let simd128 = triple.arch == crate::triple::Arch::Aarch64;
        PipelineTarget {
            triple,
            profile: TargetProfile::host_cpu().with_simd128(simd128),
        }
    }

    /// The kernel (`rust_kernels` module) pipeline: `nvptx64-nvidia-cuda`
    /// with the GPU profile.
    pub fn nvptx_kernel() -> Self {
        PipelineTarget {
            triple: crate::triple::Triple::parse("nvptx64-nvidia-cuda"),
            profile: TargetProfile::gpu_kernel(),
        }
    }
}
