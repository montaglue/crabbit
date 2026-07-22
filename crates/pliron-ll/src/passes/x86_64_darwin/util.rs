use crate::ll::BytesAttr;

use crate::{
    context::{Context, Ptr},
    dialects::{
        builtin::{op_interfaces::OneRegionInterface, ops::ModuleOp},
        x86_64::{attributes::FixupsAttr, op_interfaces::BinaryFixup},
    },
    identifier::Identifier,
    ir::{op::Op, operation::Operation},
    linked_list::ContainsLinkedList,
};

pub(super) fn module_body(
    ctx: &Context,
    module: ModuleOp,
) -> Ptr<crate::ir::basic_block::BasicBlock> {
    module
        .get_region(ctx)
        .deref(ctx)
        .get_head()
        .expect("module must have a body block")
}

pub(super) fn cast_operation<T: Op + 'static>(ctx: &Context, op: Ptr<Operation>) -> Option<T> {
    Operation::get_op_dyn(op, ctx).downcast::<T>()
}

pub(super) fn get_fixups_attr(
    op: Ptr<Operation>,
    ctx: &Context,
    key: &str,
) -> Option<Vec<BinaryFixup>> {
    let key: Identifier = key.try_into().unwrap();
    op.deref(ctx)
        .attributes
        .get::<FixupsAttr>(&key)
        .map(|attr| attr.0.clone())
}

pub(super) fn set_fixups_attr(
    op: Ptr<Operation>,
    ctx: &mut Context,
    key: &str,
    fixups: Vec<BinaryFixup>,
) {
    op.deref_mut(ctx)
        .attributes
        .set(key.try_into().unwrap(), FixupsAttr(fixups));
}

pub(super) fn get_bytes_attr(op: Ptr<Operation>, ctx: &Context, key: &str) -> Option<Vec<u8>> {
    let key: Identifier = key.try_into().unwrap();
    op.deref(ctx)
        .attributes
        .get::<BytesAttr>(&key)
        .map(|attr| attr.0.clone())
}

pub(super) fn set_bytes_attr(op: Ptr<Operation>, ctx: &mut Context, key: &str, bytes: Vec<u8>) {
    op.deref_mut(ctx)
        .attributes
        .set(key.try_into().unwrap(), BytesAttr(bytes));
}

pub(super) fn darwin_symbol(name: &str) -> String {
    format!("_{name}")
}

pub(super) fn identifier(value: &str) -> Identifier {
    value.try_into().unwrap()
}
