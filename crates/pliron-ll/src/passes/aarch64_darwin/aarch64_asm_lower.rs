use crate::{
    context::{Context, Ptr},
    dialects::{
        aarch64::op_interfaces::BinarySerializableOpInterface,
        builtin::op_interfaces::OneRegionInterface,
    },
    input_error_noloc,
    ir::{op::op_cast, operation::Operation},
    linked_list::ContainsLinkedList,
    conversion::pass::{Pass, PassOptions},
    result::STAIRResult,
};

use super::{error::Aarch64DarwinErr, frontend::module_op};

pub struct Aarch64AsmLowerPass;

impl Pass for Aarch64AsmLowerPass {
    fn name(&self) -> &str {
        "aarch64-asm-lower"
    }

    fn run(
        &self,
        root: Ptr<Operation>,
        ctx: &mut Context,
        _options: PassOptions,
    ) -> STAIRResult<Ptr<Operation>> {
        let module = module_op(ctx, root)?;
        let body = module.get_region(ctx).deref(ctx).get_head().unwrap();
        for func in body.deref(ctx).iter(ctx) {
            for region in func.deref(ctx).regions() {
                for block in region.deref(ctx).iter(ctx) {
                    for op in block.deref(ctx).iter(ctx) {
                        let op_obj = Operation::get_op_dyn(op, ctx);
                        if op_cast::<dyn BinarySerializableOpInterface>(&*op_obj).is_none() {
                            return Err(input_error_noloc!(Aarch64DarwinErr::UnsupportedOp(
                                format!(
                                    "non-serializable AArch64 op `{}` reached asm lowering",
                                    Operation::get_opid(op, ctx)
                                )
                            )));
                        }
                    }
                }
            }
        }
        Ok(root)
    }
}
