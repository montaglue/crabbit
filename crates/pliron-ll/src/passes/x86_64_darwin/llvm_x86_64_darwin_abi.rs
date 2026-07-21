use crate::{
    context::Context,
    dialects::{
        x86_64::attributes::FunctionAbiAttr,
        builtin::op_interfaces::SymbolOpInterface,
        llvm::ops::FuncOp,
    },
    ir::op::Op,
    conversion::pass::OperationPass,
    result::STAIRResult,
};

use super::{
    attrs::ATTR_KEY_DARWIN_ABI,
    frontend::{assign_darwin_abi, function_abi_classes},
};

pub struct LlvmX86_64DarwinAbiPass;

impl OperationPass for LlvmX86_64DarwinAbiPass {
    type OpType = FuncOp;

    fn name(&self) -> &str {
        "llvm-x86-64-darwin-abi"
    }

    fn run_on_operation(&self, func: FuncOp, ctx: &mut Context) -> STAIRResult<()> {
        let name = func.get_symbol_name(ctx).to_string();
        let (args, result) = function_abi_classes(ctx, func.get_func_type(ctx))?;
        let abi = assign_darwin_abi(&name, &args, result)?;
        func.get_operation()
            .deref_mut(ctx)
            .attributes
            .set(ATTR_KEY_DARWIN_ABI.clone(), FunctionAbiAttr(abi));
        Ok(())
    }
}
