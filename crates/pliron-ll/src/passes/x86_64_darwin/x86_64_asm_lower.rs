use crate::{
    context::{Context, Ptr},
    dialects::{
        x86_64::op_interfaces::BinarySerializableOpInterface,
        builtin::op_interfaces::OneRegionInterface,
    },
    input_error_noloc,
    ir::{op::op_cast, operation::Operation},
    linked_list::ContainsLinkedList,
    conversion::pass::{AnalysisManager, Pass, PassResult, changed},
};

use super::{error::X86_64DarwinErr, frontend::module_op};

pub struct X86_64AsmLowerPass;

impl Pass for X86_64AsmLowerPass {
    fn name(&self) -> &str {
        "x86-64-asm-lower"
    }

    fn run(&mut self, root: Ptr<Operation>, ctx: &mut Context, _analyses: &mut AnalysisManager) -> pliron::result::Result<PassResult> {
        let module = module_op(ctx, root)?;
        let body = module.get_region(ctx).deref(ctx).get_head().unwrap();
        for func in body.deref(ctx).iter(ctx) {
            for region in func.deref(ctx).regions() {
                for block in region.deref(ctx).iter(ctx) {
                    for op in block.deref(ctx).iter(ctx) {
                        let op_obj = Operation::get_op_dyn(op, ctx);
                        if op_cast::<dyn BinarySerializableOpInterface>(&*op_obj).is_none() {
                            return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedOp(
                                format!(
                                    "non-serializable x86-64 op `{}` reached asm lowering",
                                    Operation::get_opid(op, ctx)
                                )
                            )));
                        }
                    }
                }
            }
        }
        Ok(changed())
    }
}
