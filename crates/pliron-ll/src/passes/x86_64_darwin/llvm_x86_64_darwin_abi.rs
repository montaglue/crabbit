use crate::{
    context::{Context, Ptr},
    dialects::{
        x86_64::attributes::FunctionAbiAttr,
        builtin::op_interfaces::SymbolOpInterface,
        llvm::ops::FuncOp,
    },
    ir::{op::Op, operation::Operation},
    linked_list::ContainsLinkedList,
    conversion::pass::{AnalysisManager, Pass, PassResult, changed},
    result::STAIRResult,
};

use super::{
    attrs::ATTR_KEY_DARWIN_ABI,
    frontend::{assign_darwin_abi, function_abi_classes, module_op},
    util::{cast_operation, module_body},
};

/// Records each function's Darwin ABI argument/result locations as a
/// [FunctionAbiAttr] on the `llvm.func`, for instruction selection to consume.
pub struct LlvmX86_64DarwinAbiPass;

impl Pass for LlvmX86_64DarwinAbiPass {
    fn name(&self) -> &str {
        "llvm-x86-64-darwin-abi"
    }

    fn run(
        &mut self,
        root: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> pliron::result::Result<PassResult> {
        let module = module_op(ctx, root)?;
        let body = module_body(ctx, module);
        let funcs: Vec<_> = body
            .deref(ctx)
            .iter(ctx)
            .filter_map(|op| cast_operation::<FuncOp>(ctx, op))
            .collect();
        for func in funcs {
            assign_function_abi(ctx, func)?;
        }
        Ok(changed())
    }
}

fn assign_function_abi(ctx: &mut Context, func: FuncOp) -> STAIRResult<()> {
    let name = func.get_symbol_name(ctx).to_string();
    let (args, result) = function_abi_classes(ctx, func.get_type(ctx).into())?;
    let abi = assign_darwin_abi(&name, &args, result)?;
    func.get_operation()
        .deref_mut(ctx)
        .attributes
        .set(ATTR_KEY_DARWIN_ABI.clone(), FunctionAbiAttr(abi));
    Ok(())
}
