use crate::{
    context::{Context, Ptr},
    dialects::{
        builtin::op_interfaces::SymbolOpInterface,
        llvm::{
            op_interfaces::IsDeclaration,
            ops::{FuncOp, GlobalOp},
        },
    },
    input_error_noloc,
    ir::operation::Operation,
    linked_list::{ContainsLinkedList, LinkedList},
    conversion::pass::{AnalysisManager, Pass, PassResult, unchanged},
    passes::aarch64_darwin::util::module_body,
};

use super::{
    error::Aarch64DarwinErr,
    frontend::{module_op, validate_body, validate_function_type, validate_linkage},
    util::cast_operation,
};

pub struct VerifyLlvmForAarch64DarwinPass;

impl Pass for VerifyLlvmForAarch64DarwinPass {
    fn name(&self) -> &str {
        "verify-llvm-for-aarch64-darwin"
    }

    fn run(
        &mut self,
        root: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> pliron::result::Result<PassResult> {
        let module = module_op(ctx, root)?;
        let body = module_body(ctx, module);
        let mut op = body.deref(ctx).get_head();
        while let Some(op_ptr) = op {
            if let Some(global) = cast_operation::<GlobalOp>(ctx, op_ptr) {
                let name = global.get_symbol_name(ctx).to_string();
                validate_linkage(&name, global.get_attr_llvm_global_linkage(ctx).expect("llvm function without linkage").clone())?;
                op = op_ptr.deref(ctx).get_next();
                continue;
            } else if let Some(func) = cast_operation::<FuncOp>(ctx, op_ptr) {
                let name = func.get_symbol_name(ctx).to_string();
                validate_linkage(&name, func.get_attr_llvm_function_linkage(ctx).expect("llvm function without linkage").clone())?;
                validate_function_type(ctx, &name, func.get_type(ctx).into())?;
                if !func.is_declaration(ctx) {
                    validate_body(ctx, &func)?;
                }
                op = op_ptr.deref(ctx).get_next();
            } else {
                return Err(input_error_noloc!(Aarch64DarwinErr::UnsupportedOp(
                    Operation::get_opid(op_ptr, ctx).to_string()
                )));
            }
        }
        Ok(unchanged())
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use pliron::builtin::op_interfaces::{
        AtMostOneRegionInterface as _, BranchOpInterface as _, CallOpInterface as _,
    };
    #[allow(unused_imports)]
    use pliron_llvm::op_interfaces::{BinArithOp as _, CastOpInterface as _};
    use crate::{
        context::Context,
        dialects::{
            aarch64,
            builtin::{self, op_interfaces::OneRegionInterface},
            llvm::{
                attributes::LinkageAttr,
                ops::{FuncOp, GlobalOp},
                types::FuncType,
            },
        },
        ir::op::Op,
        linked_list::ContainsLinkedList,
        conversion::pass::{AnalysisManager, Pass},
    };

    use super::VerifyLlvmForAarch64DarwinPass;

    fn context() -> Context {
        let mut ctx = Context::new();
        aarch64::register(&mut ctx);
        ctx
    }

    #[test]
    fn accepts_global_before_function() {
        let mut ctx = context();
        let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
        let i64_ty =
            builtin::types::IntegerType::get(&mut ctx, 64, builtin::types::Signedness::Signless);
        let global = GlobalOp::new(&mut ctx, "extern_data".try_into().unwrap(), i64_ty.into());
        global.set_attr_llvm_global_linkage(&ctx, LinkageAttr::ExternalLinkage);
        global.get_operation().insert_at_back(body, &ctx);
        let func_ty = FuncType::get(&mut ctx, i64_ty.into(), vec![], false);
        let func = FuncOp::new(&mut ctx, "extern_func".try_into().unwrap(), func_ty);
        func.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
        func.get_operation().insert_at_back(body, &ctx);

        VerifyLlvmForAarch64DarwinPass
            .run(module.get_operation(), &mut ctx, &mut AnalysisManager::default())
            .unwrap();
    }

    #[test]
    fn rejects_unsupported_operation_in_function_body() {
        let mut ctx = context();
        let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
        let i64_ty =
            builtin::types::IntegerType::get(&mut ctx, 64, builtin::types::Signedness::Signless);
        let func_ty = FuncType::get(&mut ctx, i64_ty.into(), vec![], false);
        let func = FuncOp::new(&mut ctx, "bad_body".try_into().unwrap(), func_ty);
        func.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(&mut ctx);
        func.get_operation().insert_at_back(body, &ctx);
        aarch64::ops::ret(&mut ctx).insert_at_back(func.get_entry_block(&ctx).unwrap(), &ctx);

        let err = match VerifyLlvmForAarch64DarwinPass.run(
            module.get_operation(),
            &mut ctx,
            &mut AnalysisManager::default(),
        ) {
            Ok(_) => panic!("unsupported function body unexpectedly verified"),
            Err(err) => err,
        };
        assert!(!err.to_string().is_empty());
    }
}
