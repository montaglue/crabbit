pub mod attributes;
pub mod encoding;
pub mod op_interfaces;
pub mod ops;
pub mod registers;

use crate::context::Context;
use crate::ir::dialect::{Dialect, DialectName};

pub fn register(ctx: &mut Context) {
    Dialect::register(
        ctx,
        &DialectName::try_new("aarch64").expect("invalid dialect name"),
    );
    ops::register(ctx);
    attributes::register(ctx);
}
