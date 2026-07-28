//! The aarch64-darwin backend: the shared [aarch64](super::aarch64) pipeline
//! configured for Mach-O/Darwin. Kept as a facade so existing users keep
//! their entry points while the core lives in the OS-neutral module.

use crate::{
    context::{Context, Ptr},
    ir::operation::Operation,
    conversion::pass::Passes,
    result::STAIRResult,
};

use super::aarch64::{self, TargetOs};

pub use super::aarch64::{emit_macho_object_bytes, write_macho_object_from_ir};

/// The aarch64 pipeline configured for Darwin.
pub fn pipeline() -> Passes {
    aarch64::pipeline(TargetOs::Darwin)
}

/// Runs [pipeline] on `root` (a `builtin.module`) in place.
pub fn lower_module(ctx: &mut Context, root: Ptr<Operation>) -> STAIRResult<()> {
    aarch64::lower_module(ctx, root, TargetOs::Darwin)
}
