//! Smoke test for the cuda-oxide integration: build a small `dialect-mir`
//! function, run [LowerDialectMirPass], and check the module comes out in the
//! llvm dialect.

use crabbit_mir::passes::lower_dialect_mir::LowerDialectMirPass;
use dialect_mir::ops as mir;
use pliron::basic_block::BasicBlock;
use pliron::builtin::attributes::TypeAttr;
use pliron::builtin::op_interfaces::SymbolOpInterface;
use pliron::builtin::ops::ModuleOp;
use pliron::builtin::types::FunctionType;
use pliron::context::Context;
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::printable::Printable;
use pliron_ll::conversion::pass::{AnalysisManager, Pass};

#[test]
fn lowers_dialect_mir_func_to_llvm_dialect() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    mir_lower::register(&mut ctx);

    let module = ModuleOp::new(&mut ctx, "smoke".try_into().unwrap());
    let module_ptr = module.get_operation();

    let func_ty = FunctionType::get(&mut ctx, vec![], vec![]);
    let func_op_ptr = Operation::new(
        &mut ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let func = mir::MirFuncOp::new(&mut ctx, func_op_ptr, TypeAttr::new(func_ty.into()));
    func.set_symbol_name(&mut ctx, "smoke_fn".try_into().unwrap());

    let region = func.get_operation().deref(&ctx).get_region(0);
    let block = BasicBlock::new(&mut ctx, None, vec![]);
    block.insert_at_back(region, &ctx);

    let ret_op_ptr = Operation::new(
        &mut ctx,
        mir::MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    mir::MirReturnOp::new(ret_op_ptr)
        .get_operation()
        .insert_at_back(block, &ctx);

    let module_block = module
        .get_operation()
        .deref(&ctx)
        .get_region(0)
        .deref(&ctx)
        .iter(&ctx)
        .next()
        .unwrap();
    func.get_operation().insert_at_back(module_block, &ctx);

    let mut pass = LowerDialectMirPass;
    let mut analyses = AnalysisManager::default();
    pass.run(module_ptr, &mut ctx, &mut analyses)
        .expect("dialect-mir → llvm lowering failed");

    let lowered = module_ptr.disp(&ctx).to_string();
    assert!(
        lowered.contains("llvm.func"),
        "expected an llvm.func in the lowered module:\n{lowered}"
    );
    assert!(
        !lowered.contains("mir."),
        "mir ops survived lowering:\n{lowered}"
    );
}
