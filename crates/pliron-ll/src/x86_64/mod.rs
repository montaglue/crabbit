pub mod attributes;
mod encoding;
pub mod op_interfaces;
pub mod ops;
pub mod registers;

use crate::context::Context;
use crate::ir::dialect::{Dialect, DialectName};

pub fn register(ctx: &mut Context) {
    Dialect::register(ctx, &DialectName::try_new("x86_64").unwrap());
    ops::register(ctx);
    attributes::register(ctx);
}
