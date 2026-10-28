//! The crabbit backend dylib: nothing but the exported entry over
//! [crabbit_core]. A backend with extensions has the same shape plus its
//! registrations (docs/EXTENDING.md); the core stays an rlib so exactly
//! one Rust dylib exists per rustc process.
#![feature(rustc_private)]

extern crate rustc_codegen_ssa;
extern crate rustc_driver;

use rustc_codegen_ssa::traits::CodegenBackend;

// SAFETY: rustc loads custom codegen backends by looking up this exact exported symbol.
#[unsafe(no_mangle)]
pub fn __rustc_codegen_backend() -> Box<dyn CodegenBackend> {
    crabbit_core::create_backend()
}
