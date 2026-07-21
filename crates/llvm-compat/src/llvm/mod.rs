//! LLVM dialect for the STAIR IR framework.
//!
//! This dialect provides types, attributes, and operations corresponding to
//! LLVM IR, ported from the pliron LLVM dialect as a reference.

pub mod attributes;
pub mod op_interfaces;
pub mod ops;
pub mod types;

use crate::{
    context::Context,
    ir::dialect::{Dialect, DialectName},
};

pub fn register(ctx: &mut Context) {
    Dialect::register(ctx, &DialectName::try_new("llvm").unwrap());
    types::register(ctx);
    attributes::register(ctx);
    ops::register(ctx);
}
