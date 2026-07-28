//! The aarch64-linux backend: the shared [aarch64](super::aarch64) pipeline
//! configured for ELF/Linux (AAPCS64, unprefixed symbols).

use crate::{
    context::{Context, Ptr},
    ir::operation::Operation,
    conversion::pass::Passes,
    result::STAIRResult,
};

use super::aarch64::{self, TargetOs};

pub use super::aarch64::{emit_elf_object_bytes, write_elf_object_from_ir};

/// The aarch64 pipeline configured for Linux.
pub fn pipeline() -> Passes {
    aarch64::pipeline(TargetOs::Linux)
}

/// Runs [pipeline] on `root` (a `builtin.module`) in place.
pub fn lower_module(ctx: &mut Context, root: Ptr<Operation>) -> STAIRResult<()> {
    aarch64::lower_module(ctx, root, TargetOs::Linux)
}
