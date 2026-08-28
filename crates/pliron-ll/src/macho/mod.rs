pub mod ops;

use crate::context::Context;
use crate::ir::dialect::{Dialect, DialectName};

pub fn register(ctx: &mut Context) {
    Dialect::register(
        ctx,
        &DialectName::try_new("macho").expect("invalid dialect name"),
    );
    ops::register(ctx);
}
