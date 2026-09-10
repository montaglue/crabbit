//! Device-side intrinsics for crabbit kernels (see crabbit/docs/KERNEL-ABI.md).
//!
//! Every function here is a *foreign declaration* whose `link_name` is the
//! genuine LLVM/NVVM intrinsic name. crabbit's importer keeps that name (dots
//! become underscores), and its NVPTX translator — or LLVM, on the
//! llvm-export route — lowers the call to the PTX instruction. Kernels call
//! these directly (`raw::tid_x()`); the `#[inline(always)]` safe wrappers are
//! only for the CPU/host side of a crate and must not be used inside kernel
//! bodies (kernels are leaf functions in v1).
#![no_std]
// `link_name = "llvm.*"` is gated (rust-lang/rust#29602); the corpus is
// nightly-only anyway (crabbit needs -Zcodegen-backend).
#![feature(link_llvm_intrinsics)]
#![allow(clippy::missing_safety_doc)]

pub mod raw {
    unsafe extern "C" {
        // Special registers.
        #[link_name = "llvm.nvvm.read.ptx.sreg.tid.x"]
        pub fn tid_x() -> u32;
        #[link_name = "llvm.nvvm.read.ptx.sreg.tid.y"]
        pub fn tid_y() -> u32;
        #[link_name = "llvm.nvvm.read.ptx.sreg.tid.z"]
        pub fn tid_z() -> u32;
        #[link_name = "llvm.nvvm.read.ptx.sreg.ntid.x"]
        pub fn ntid_x() -> u32;
        #[link_name = "llvm.nvvm.read.ptx.sreg.ntid.y"]
        pub fn ntid_y() -> u32;
        #[link_name = "llvm.nvvm.read.ptx.sreg.ntid.z"]
        pub fn ntid_z() -> u32;
        #[link_name = "llvm.nvvm.read.ptx.sreg.ctaid.x"]
        pub fn ctaid_x() -> u32;
        #[link_name = "llvm.nvvm.read.ptx.sreg.ctaid.y"]
        pub fn ctaid_y() -> u32;
        #[link_name = "llvm.nvvm.read.ptx.sreg.ctaid.z"]
        pub fn ctaid_z() -> u32;
        #[link_name = "llvm.nvvm.read.ptx.sreg.nctaid.x"]
        pub fn nctaid_x() -> u32;
        #[link_name = "llvm.nvvm.read.ptx.sreg.nctaid.y"]
        pub fn nctaid_y() -> u32;
        #[link_name = "llvm.nvvm.read.ptx.sreg.nctaid.z"]
        pub fn nctaid_z() -> u32;
        #[link_name = "llvm.nvvm.read.ptx.sreg.laneid"]
        pub fn laneid() -> u32;
        #[link_name = "llvm.nvvm.read.ptx.sreg.warpsize"]
        pub fn warpsize() -> u32;

        // Barriers / fences.
        #[link_name = "llvm.nvvm.barrier0"]
        pub fn barrier0();
        #[link_name = "llvm.nvvm.membar.gl"]
        pub fn membar_gl();
        #[link_name = "llvm.nvvm.membar.cta"]
        pub fn membar_cta();

        // Atomics on global/shared memory (generic addressing).
        #[link_name = "llvm.nvvm.atomic.add.gen.i.cta"]
        pub fn atomic_add_i32_cta(ptr: *mut i32, v: i32) -> i32;
        #[link_name = "llvm.nvvm.atomic.add.gen.i"]
        pub fn atomic_add_i32(ptr: *mut i32, v: i32) -> i32;
        #[link_name = "llvm.nvvm.atomic.add.gen.f"]
        pub fn atomic_add_f32(ptr: *mut f32, v: f32) -> f32;

        // Math (f32).
        #[link_name = "llvm.nvvm.sqrt.rn.f"]
        pub fn sqrt_f32(x: f32) -> f32;
        #[link_name = "llvm.nvvm.rsqrt.approx.f"]
        pub fn rsqrt_f32(x: f32) -> f32;
        #[link_name = "llvm.nvvm.ex2.approx.f"]
        pub fn ex2_f32(x: f32) -> f32;
        #[link_name = "llvm.nvvm.lg2.approx.f"]
        pub fn lg2_f32(x: f32) -> f32;
        #[link_name = "llvm.nvvm.fmax.f"]
        pub fn fmax_f32(a: f32, b: f32) -> f32;
        #[link_name = "llvm.nvvm.fmin.f"]
        pub fn fmin_f32(a: f32, b: f32) -> f32;
        #[link_name = "llvm.nvvm.fabs.f"]
        pub fn fabs_f32(x: f32) -> f32;
        // Math (f64).
        #[link_name = "llvm.nvvm.sqrt.rn.d"]
        pub fn sqrt_f64(x: f64) -> f64;
        #[link_name = "llvm.nvvm.rsqrt.approx.d"]
        pub fn rsqrt_f64(x: f64) -> f64;
        #[link_name = "llvm.nvvm.fmax.d"]
        pub fn fmax_f64(a: f64, b: f64) -> f64;
        #[link_name = "llvm.nvvm.fmin.d"]
        pub fn fmin_f64(a: f64, b: f64) -> f64;
    }
}

/// log2(e), for exp(x) = ex2(x * LOG2E).
pub const LOG2E_F32: f32 = 1.442_695_f32;

/// Global 1-D thread index `ctaid.x * ntid.x + tid.x`. Expands inline so
/// kernels stay leaf functions.
#[macro_export]
macro_rules! global_id_x {
    () => {
        $crate::raw::ctaid_x() * $crate::raw::ntid_x() + $crate::raw::tid_x()
    };
}
