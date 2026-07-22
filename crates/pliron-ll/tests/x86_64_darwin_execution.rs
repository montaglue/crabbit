//! End-to-end execution tests for the x86-64 Darwin pipeline: build LLVM
//! dialect IR by hand, emit a Mach-O object, link it with the system clang
//! (cross-arch linking works natively on Apple Silicon), and run the binary —
//! under Rosetta 2 when the host is arm64.

#![cfg(target_os = "macos")]

#[allow(unused_imports)]
use pliron::builtin::op_interfaces::AtMostOneRegionInterface as _;
#[allow(unused_imports)]
use pliron_llvm::op_interfaces::{BinArithOp as _, IntBinArithOpWithOverflowFlag as _};
use std::num::NonZero;
use std::path::PathBuf;
use std::process::Command;

use pliron_ll::{
    context::Context,
    dialects::{
        builtin::{
            self,
            attributes::IntegerAttr,
            op_interfaces::{OneRegionInterface, OneResultInterface},
            types::{IntegerType, Signedness},
        },
        llvm::{
            attributes::{ICmpPredicateAttr, LinkageAttr},
            ops::{
                AddOp, BrOp, CallOp, CondBrOp, ConstantOp, FuncOp, ICmpOp, MulOp, ReturnOp, SubOp,
            },
            types::FuncType,
        },
        macho, x86_64,
    },
    ir::{basic_block::BasicBlock, op::Op, value::Value},
    linked_list::ContainsLinkedList,
    passes::x86_64_darwin,
    utils::apint::APInt,
};

fn context() -> Context {
    let mut ctx = Context::new();
    x86_64::register(&mut ctx);
    macho::register(&mut ctx);
    ctx
}

fn i64_ty(ctx: &mut Context) -> pliron_ll::r#type::TypeHandle {
    IntegerType::get(ctx, 64, Signedness::Signless).into()
}

fn i64_const(ctx: &mut Context, entry: pliron_ll::context::Ptr<BasicBlock>, value: u64) -> Value {
    let ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let constant = ConstantOp::new(ctx, Box::new(IntegerAttr::new(ty, APInt::from_u64(value, NonZero::new(64).unwrap()))));
    constant.get_operation().insert_at_back(entry, ctx);
    constant.get_result(ctx)
}

/// Link `object_bytes` against a `main` shim and run the result, returning
/// the exit status. The object provides `_main` itself.
fn link_and_run(test: &str, object_bytes: &[u8]) -> i32 {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(test);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let object = dir.join("test.o");
    let binary = dir.join("test");
    std::fs::write(&object, object_bytes).expect("write object");

    let link = Command::new("clang")
        .args(["-arch", "x86_64", "-o"])
        .arg(&binary)
        .arg(&object)
        .output()
        .expect("run clang");
    assert!(
        link.status.success(),
        "linking failed:\n{}",
        String::from_utf8_lossy(&link.stderr)
    );

    let run = Command::new(&binary).status().expect("run test binary");
    run.code().expect("test binary exited without a code")
}

#[test]
fn returns_constant_exit_code() {
    let mut ctx = context();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
    let i64_ty = i64_ty(&mut ctx);
    let func_ty = FuncType::get(&mut ctx, i64_ty, vec![], false);
    let func = FuncOp::new(&mut ctx, "main".try_into().unwrap(), func_ty);
        func.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(&mut ctx);
    func.get_operation().insert_at_back(body, &ctx);
    let entry = func.get_entry_block(&ctx).unwrap();
    let value = i64_const(&mut ctx, entry, 42);
    ReturnOp::new(&mut ctx, Some(value))
        .get_operation()
        .insert_at_back(entry, &ctx);

    let bytes = x86_64_darwin::emit_macho_object_bytes(&mut ctx, module.get_operation()).unwrap();
    assert_eq!(link_and_run("returns_constant_exit_code", &bytes), 42);
}

#[test]
fn computes_arithmetic_across_calls() {
    // main() { helper(20, 4) - 11 } where helper(a, b) = a * b + 3 → 72.
    let mut ctx = context();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
    let i64_ty = i64_ty(&mut ctx);

    let helper_ty = FuncType::get(&mut ctx, i64_ty, vec![i64_ty, i64_ty], false);
    let helper = FuncOp::new(&mut ctx, "helper".try_into().unwrap(), helper_ty);
    helper.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
    helper.get_or_create_entry_block(&mut ctx);
    helper.get_operation().insert_at_back(body, &ctx);
    let entry = helper.get_entry_block(&ctx).unwrap();
    let args: Vec<_> = entry.deref(&ctx).arguments().collect();
    let product = MulOp::new_with_overflow_flag(&mut ctx, args[0], args[1], Default::default());
    product.get_operation().insert_at_back(entry, &ctx);
    let three = i64_const(&mut ctx, entry, 3);
    let product_result = product.get_result(&ctx);
    let sum = AddOp::new_with_overflow_flag(&mut ctx, product_result, three, Default::default());
    sum.get_operation().insert_at_back(entry, &ctx);
    let sum_result = sum.get_result(&ctx);
    ReturnOp::new(&mut ctx, Some(sum_result))
        .get_operation()
        .insert_at_back(entry, &ctx);

    let main_ty = FuncType::get(&mut ctx, i64_ty, vec![], false);
    let main = FuncOp::new(&mut ctx, "main".try_into().unwrap(), main_ty);
    main.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
    main.get_or_create_entry_block(&mut ctx);
    main.get_operation().insert_at_back(body, &ctx);
    let entry = main.get_entry_block(&ctx).unwrap();
    let twenty = i64_const(&mut ctx, entry, 20);
    let four = i64_const(&mut ctx, entry, 4);
    let helper_call_ty = FuncType::get(&mut ctx, i64_ty, vec![i64_ty, i64_ty], false);
    let call = CallOp::new(
        &mut ctx,
        pliron::builtin::op_interfaces::CallOpCallable::Direct("helper".try_into().unwrap()),
        helper_call_ty,
        vec![twenty, four],
    );
    call.get_operation().insert_at_back(entry, &ctx);
    let call_result = call.get_operation().deref(&ctx).get_result(0);
    let eleven = i64_const(&mut ctx, entry, 11);
    let difference = SubOp::new_with_overflow_flag(&mut ctx, call_result, eleven, Default::default());
    difference.get_operation().insert_at_back(entry, &ctx);
    let difference_result = difference.get_result(&ctx);
    ReturnOp::new(&mut ctx, Some(difference_result))
        .get_operation()
        .insert_at_back(entry, &ctx);

    let bytes = x86_64_darwin::emit_macho_object_bytes(&mut ctx, module.get_operation()).unwrap();
    assert_eq!(link_and_run("computes_arithmetic_across_calls", &bytes), 72);
}

#[test]
fn branches_on_comparisons() {
    // main() { if 7 < 9 { 5 } else { 6 } } → 5, via cmp/jcc.
    let mut ctx = context();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
    let i64_ty = i64_ty(&mut ctx);
    let func_ty = FuncType::get(&mut ctx, i64_ty, vec![], false);
    let func = FuncOp::new(&mut ctx, "main".try_into().unwrap(), func_ty);
        func.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(&mut ctx);
    func.get_operation().insert_at_back(body, &ctx);
    let entry = func.get_entry_block(&ctx).unwrap();
    let then_block = BasicBlock::new(&mut ctx, Some("then".try_into().unwrap()), vec![]);
    then_block.insert_at_back(func.get_region(&ctx).unwrap(), &ctx);
    let else_block = BasicBlock::new(&mut ctx, Some("else".try_into().unwrap()), vec![]);
    else_block.insert_at_back(func.get_region(&ctx).unwrap(), &ctx);
    let exit_block = BasicBlock::new(
        &mut ctx,
        Some("exit".try_into().unwrap()),
        vec![i64_ty],
    );
    exit_block.insert_at_back(func.get_region(&ctx).unwrap(), &ctx);

    let seven = i64_const(&mut ctx, entry, 7);
    let nine = i64_const(&mut ctx, entry, 9);
    let compare = ICmpOp::new(&mut ctx, ICmpPredicateAttr::SLT, seven, nine);
    compare.get_operation().insert_at_back(entry, &ctx);
    let compare_result = compare.get_result(&ctx);
    CondBrOp::new(
        &mut ctx,
        compare_result,
        then_block,
        vec![],
        else_block,
        vec![],
    )
    .get_operation()
    .insert_at_back(entry, &ctx);

    let five = i64_const(&mut ctx, then_block, 5);
    BrOp::new(&mut ctx, exit_block, vec![five])
        .get_operation()
        .insert_at_back(then_block, &ctx);
    let six = i64_const(&mut ctx, else_block, 6);
    BrOp::new(&mut ctx, exit_block, vec![six])
        .get_operation()
        .insert_at_back(else_block, &ctx);
    let exit_value = exit_block.deref(&ctx).get_argument(0);
    ReturnOp::new(&mut ctx, Some(exit_value))
        .get_operation()
        .insert_at_back(exit_block, &ctx);

    let bytes = x86_64_darwin::emit_macho_object_bytes(&mut ctx, module.get_operation()).unwrap();
    assert_eq!(link_and_run("branches_on_comparisons", &bytes), 5);
}
