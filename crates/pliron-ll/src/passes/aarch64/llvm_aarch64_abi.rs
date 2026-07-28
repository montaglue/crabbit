use crate::{
    context::{Context, Ptr},
    dialects::{
        aarch64::attributes::FunctionAbiAttr,
        builtin::op_interfaces::SymbolOpInterface,
        llvm::ops::FuncOp,
    },
    ir::{op::Op, operation::Operation},
    linked_list::ContainsLinkedList,
    conversion::pass::{AnalysisManager, Pass, PassResult, changed},
    result::STAIRResult,
};

use super::{
    attrs::ATTR_KEY_AARCH64_ABI,
    frontend::{assign_abi, function_abi_classes, module_op},
    target::TargetOs,
    util::{cast_operation, module_body},
};

/// Records each function's ABI argument/result locations for the configured
/// [TargetOs] as a [FunctionAbiAttr] on the `llvm.func`, for instruction
/// selection to consume.
pub struct LlvmAarch64AbiPass {
    os: TargetOs,
}

impl LlvmAarch64AbiPass {
    pub fn new(os: TargetOs) -> Self {
        Self { os }
    }
}

impl Pass for LlvmAarch64AbiPass {
    fn name(&self) -> &str {
        match self.os {
            TargetOs::Darwin => "llvm-aarch64-darwin-abi",
            TargetOs::Linux => "llvm-aarch64-linux-abi",
        }
    }

    fn run(
        &self,
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
            assign_function_abi(ctx, self.os, func)?;
        }
        Ok(changed())
    }
}

fn assign_function_abi(ctx: &mut Context, os: TargetOs, func: FuncOp) -> STAIRResult<()> {
    let name = func.get_symbol_name(ctx).to_string();
    let (args, result) = function_abi_classes(ctx, func.get_type(ctx).into())?;
    let abi = assign_abi(os, &name, &args, result)?;
    func.get_operation()
        .deref_mut(ctx)
        .attributes
        .set(ATTR_KEY_AARCH64_ABI.clone(), FunctionAbiAttr(abi));
    Ok(())
}
