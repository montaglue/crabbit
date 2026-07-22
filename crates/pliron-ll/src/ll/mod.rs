//! `ll` dialect: crabbit's extensions around the upstream `pliron-llvm`
//! dialect and the machine-level backends — binary payload attributes,
//! machine-symbol linkage, branch weights, and the [CStrOp](ops::CStrOp)
//! string-literal operation.

pub mod attributes;
pub mod op_interfaces;
pub mod ops;

pub use attributes::{BranchWeightsAttr, BytesAttr, LinkageAttr};

use pliron::{
    context::Context,
    dialect::{Dialect, DialectName},
};

pub fn register(ctx: &mut Context) {
    Dialect::register(ctx, &DialectName::try_new("ll").unwrap());
}

/// The raw-byte initializer of a global, if it has one. Byte initializers are
/// stored as an [ll.bytes](BytesAttr) initializer value on the
/// `llvm.global`.
pub fn global_initializer_bytes(
    ctx: &Context,
    global: &pliron_llvm::ops::GlobalOp,
) -> Option<Vec<u8>> {
    let value = global.get_initializer_value(ctx)?;
    value.downcast_ref::<BytesAttr>().map(|bytes| bytes.0.clone())
}

/// Attach a raw-byte initializer ([ll.bytes](BytesAttr)) to a global.
pub fn set_global_initializer_bytes(
    ctx: &Context,
    global: &pliron_llvm::ops::GlobalOp,
    bytes: Vec<u8>,
) {
    global.set_initializer_value(ctx, Box::new(BytesAttr(bytes)));
}
