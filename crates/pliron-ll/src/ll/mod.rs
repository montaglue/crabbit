//! `ll` dialect: crabbit's extensions around the upstream `pliron-llvm`
//! dialect and the machine-level backends — binary payload attributes,
//! machine-symbol linkage, branch weights, and the [CStrOp](ops::CStrOp)
//! string-literal operation.

pub mod attributes;
pub mod op_interfaces;
pub mod ops;

pub use attributes::{BranchWeightsAttr, BytesAttr, DataAttr, DataReloc, LinkageAttr, TlsAttr};

use pliron::{
    context::Context,
    dialect::{Dialect, DialectName},
    dict_key,
    op::Op,
};

dict_key!(ATTR_KEY_LL_TLS, "ll_tls");

pub fn register(ctx: &mut Context) {
    Dialect::register(
        ctx,
        &DialectName::try_new("ll").expect("invalid dialect name"),
    );
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

/// The data-section initializer of a global, if it has one. Data-section
/// initializers are stored as an [ll.data](DataAttr) initializer value on
/// the `llvm.global`.
pub fn global_data(ctx: &Context, global: &pliron_llvm::ops::GlobalOp) -> Option<DataAttr> {
    let value = global.get_initializer_value(ctx)?;
    value.downcast_ref::<DataAttr>().cloned()
}

/// Attach a data-section initializer ([ll.data](DataAttr)) to a global.
pub fn set_global_data(ctx: &Context, global: &pliron_llvm::ops::GlobalOp, data: DataAttr) {
    global.set_initializer_value(ctx, Box::new(data));
}

/// Mark a global as thread-local ([ll.tls](TlsAttr)): its address is the
/// current thread's copy, and object writers place its initializer in the
/// TLS segment.
pub fn set_global_thread_local(ctx: &mut Context, global: &pliron_llvm::ops::GlobalOp) {
    global
        .get_operation()
        .deref_mut(ctx)
        .attributes
        .set(ATTR_KEY_LL_TLS.clone(), TlsAttr);
}

/// Whether a global carries the [ll.tls](TlsAttr) thread-local marker.
pub fn global_is_thread_local(ctx: &Context, global: &pliron_llvm::ops::GlobalOp) -> bool {
    global
        .get_operation()
        .deref(ctx)
        .attributes
        .get::<TlsAttr>(&ATTR_KEY_LL_TLS)
        .is_some()
}
