use crate::{
    context::Context,
    dialects::{
        builtin::{op_interfaces::SymbolOpInterface, ops::ModuleOp},
        llvm::{
            op_interfaces::IsDeclaration,
            ops::{FuncOp, GlobalOp},
        },
    },
    input_error_noloc,
    linked_list::{ContainsLinkedList, LinkedList},
    conversion::pass::OperationPass,
    passes::x86_64_darwin::util::module_body,
    result::STAIRResult,
};

use super::{
    error::X86_64DarwinErr,
    frontend::{validate_body, validate_function_type, validate_linkage},
    util::cast_operation,
};

pub struct VerifyLlvmForX86_64DarwinPass;

impl OperationPass for VerifyLlvmForX86_64DarwinPass {
    type OpType = ModuleOp;

    fn name(&self) -> &str {
        "verify-llvm-for-x86-64-darwin"
    }

    fn run_on_operation(&self, module: ModuleOp, ctx: &mut Context) -> STAIRResult<()> {
        let body = module_body(ctx, module);
        let mut op = body.deref(ctx).get_head();
        while let Some(op_ptr) = op {
            if let Some(global) = cast_operation::<GlobalOp>(ctx, op_ptr) {
                let name = global.get_symbol_name(ctx).to_string();
                validate_linkage(&name, global.get_linkage(ctx))?;
                op = op_ptr.deref(ctx).get_next();
                continue;
            } else if let Some(func) = cast_operation::<FuncOp>(ctx, op_ptr) {
                let name = func.get_symbol_name(ctx).to_string();
                validate_linkage(&name, func.get_linkage(ctx))?;
                validate_function_type(ctx, &name, func.get_func_type(ctx))?;
                if !func.is_declaration(ctx) {
                    validate_body(ctx, &func)?;
                }
                op = op_ptr.deref(ctx).get_next();
            } else {
                return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedOp(
                    Operation::get_opid(op_ptr, ctx).to_string()
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        context::Context,
        dialects::{
            x86_64,
            builtin::{self, op_interfaces::OneRegionInterface},
            llvm::{
                self,
                attributes::LinkageAttr,
                ops::{FuncOp, GlobalOp},
                types::FuncType,
            },
        },
        ir::op::Op,
        linked_list::ContainsLinkedList,
        conversion::pass::{Pass, PassOptions},
    };

    use super::VerifyLlvmForX86_64DarwinPass;

    fn context() -> Context {
        let mut ctx = Context::new();
        llvm::register(&mut ctx);
        x86_64::register(&mut ctx);
        ctx
    }

    #[test]
    fn accepts_global_before_function() {
        let mut ctx = context();
        let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
        let i64_ty =
            builtin::types::IntegerType::get(&mut ctx, 64, builtin::types::Signedness::Signless);
        let global = GlobalOp::new(
            &mut ctx,
            "extern_data".try_into().unwrap(),
            i64_ty.into(),
            LinkageAttr::External,
        );
        global.get_operation().insert_at_back(body, &ctx);
        let func_ty = FuncType::get(&mut ctx, i64_ty.into(), vec![], false);
        let func = FuncOp::new_declaration(
            &mut ctx,
            "extern_func".try_into().unwrap(),
            func_ty,
            LinkageAttr::External,
        );
        func.get_operation().insert_at_back(body, &ctx);

        VerifyLlvmForX86_64DarwinPass
            .run(module.get_operation(), &mut ctx, PassOptions::default())
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
        let func = FuncOp::new(
            &mut ctx,
            "bad_body".try_into().unwrap(),
            func_ty,
            LinkageAttr::External,
        );
        func.get_operation().insert_at_back(body, &ctx);
        x86_64::ops::ret(&mut ctx).insert_at_back(func.get_entry_block(&ctx), &ctx);

        let err = match VerifyLlvmForX86_64DarwinPass.run(
            module.get_operation(),
            &mut ctx,
            PassOptions::default(),
        ) {
            Ok(_) => panic!("unsupported function body unexpectedly verified"),
            Err(err) => err,
        };
        assert!(!err.to_string().is_empty());
    }
}

use crate::operation::Operation;

use llvm_compat::ll::{LinkageAttr};
