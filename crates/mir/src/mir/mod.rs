//! Rust MIR dialect for STAIR.
//!
//! This dialect models a small, SSA-friendly subset of rustc MIR that is useful
//! as a frontend boundary before lowering to the existing LLVM dialect.

pub mod attributes;
pub mod ops;
pub mod types;

use crate::{
    context::Context,
    ir::dialect::{Dialect, DialectName},
};

pub fn register(ctx: &mut Context) {
    Dialect::register(ctx, &DialectName::try_new("mir").unwrap());
    attributes::register(ctx);
    types::register(ctx);
    ops::register(ctx);
}
