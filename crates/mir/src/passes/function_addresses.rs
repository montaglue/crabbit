//! `materialize-function-addresses`: after `lower-dialect-mir`, every
//! [ll.func_addr](pliron_ll::ll::ops::FuncAddrOp) becomes the
//! `llvm.addressof` of its symbol. The importer takes function addresses
//! (reified `fn` items) while the functions are still `mir.func`s, which an
//! `llvm.addressof` may not name; once lowered they are `llvm.func`s.

use pliron::{
    context::{Context, Ptr},
    linked_list::ContainsLinkedList as _,
    op::Op as _,
    operation::Operation,
    result::Result,
};
use pliron_ll::{
    conversion::pass::{AnalysisManager, Pass, PassResult, changed, unchanged},
    ll::ops::FuncAddrOp,
};
use pliron_llvm::ops::AddressOfOp;

/// The pass name.
pub const PASS_NAME: &str = "materialize-function-addresses";

pub struct MaterializeFunctionAddressesPass;

impl Pass for MaterializeFunctionAddressesPass {
    fn name(&self) -> &str {
        PASS_NAME
    }

    fn run(
        &mut self,
        root: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut pending = Vec::new();
        collect(ctx, root, &mut pending);
        if pending.is_empty() {
            return Ok(unchanged());
        }
        for op in pending {
            let func_addr = Operation::get_op::<FuncAddrOp>(op, ctx).expect("collected by op id");
            let symbol = func_addr.symbol(ctx);
            let address_of = AddressOfOp::new(ctx, symbol, 0).get_operation();
            address_of.insert_before(ctx, op);
            let replacement = address_of.deref(ctx).get_result(0);
            let old = op.deref(ctx).get_result(0);
            old.replace_all_uses_with(ctx, &replacement);
            Operation::erase(op, ctx);
        }
        Ok(changed())
    }
}

fn collect(ctx: &Context, op: Ptr<Operation>, out: &mut Vec<Ptr<Operation>>) {
    if Operation::get_opid(op, ctx) == FuncAddrOp::get_opid_static() {
        out.push(op);
    }
    for region in op.deref(ctx).regions() {
        for block in region.deref(ctx).iter(ctx) {
            for inner in block.deref(ctx).iter(ctx) {
                collect(ctx, inner, out);
            }
        }
    }
}
