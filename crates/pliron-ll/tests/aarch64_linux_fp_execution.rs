//! End-to-end execution tests for scalar floating-point support in the
//! aarch64-linux pipeline: build LLVM dialect IR by hand, emit an ELF
//! relocatable object, link it with a C harness via the system `cc`, and run
//! the binary, asserting on its exit code.

#![cfg(all(target_os = "linux", target_arch = "aarch64"))]

use pliron::builtin::op_interfaces::AtMostOneRegionInterface as _;
use pliron::builtin::ops::ConstantOp;
use std::num::NonZero;
use std::path::PathBuf;
use std::process::Command;

use pliron_ll::{
    context::Context,
    dialects::{
        aarch64, builtin,
        builtin::{
            attributes::{FPDoubleAttr, FPSingleAttr, IntegerAttr},
            op_interfaces::{OneRegionInterface, OneResultInterface},
            types::{FP32Type, FP64Type, IntegerType, Signedness},
        },
        llvm::{
            attributes::{FCmpPredicateAttr, FastmathFlagsAttr, LinkageAttr},
            ops::{
                AShrOp, CondBrOp, FAddOp, FCmpOp, FDivOp, FMulOp, FNegOp, FPExtOp, FPToSIOp,
                FPToUIOp, FPTruncOp, FSubOp, FuncOp, LShrOp, ReturnOp, SIToFPOp, ShlOp, UIToFPOp,
                CallOp,
            },
            types::FuncType,
        },
        macho,
    },
    ir::{basic_block::BasicBlock, op::Op, value::Value},
    linked_list::ContainsLinkedList,
    passes::aarch64_linux,
    utils::apint::APInt,
};
#[allow(unused_imports)]
use pliron_llvm::op_interfaces::{
    BinArithOp as _, CastOpInterface as _, FastMathFlags as _,
    FloatBinArithOpWithFastMathFlags as _, IntBinArithOpWithOverflowFlag as _,
};

type Ty = pliron_ll::r#type::TypeHandle;
type BlockPtr = pliron_ll::context::Ptr<BasicBlock>;

fn context() -> Context {
    let mut ctx = Context::new();
    aarch64::register(&mut ctx);
    macho::register(&mut ctx);
    ctx
}

fn f64_ty(ctx: &Context) -> Ty {
    FP64Type::get(ctx).into()
}

fn f32_ty(ctx: &Context) -> Ty {
    FP32Type::get(ctx).into()
}

fn i64_ty(ctx: &mut Context) -> Ty {
    IntegerType::get(ctx, 64, Signedness::Signless).into()
}

fn i32_ty(ctx: &mut Context) -> Ty {
    IntegerType::get(ctx, 32, Signedness::Signless).into()
}

fn i128_ty(ctx: &mut Context) -> Ty {
    IntegerType::get(ctx, 128, Signedness::Signless).into()
}

fn f64_const(ctx: &mut Context, block: BlockPtr, value: f64) -> Value {
    let constant = ConstantOp::new(ctx, Box::new(FPDoubleAttr::from(value)));
    constant.get_operation().insert_at_back(block, ctx);
    constant.get_result(ctx)
}

fn f32_const(ctx: &mut Context, block: BlockPtr, value: f32) -> Value {
    let constant = ConstantOp::new(ctx, Box::new(FPSingleAttr::from(value)));
    constant.get_operation().insert_at_back(block, ctx);
    constant.get_result(ctx)
}

/// Define an external function and return its entry block and arguments.
fn func(
    ctx: &mut Context,
    body: BlockPtr,
    name: &str,
    result: Ty,
    args: Vec<Ty>,
) -> (BlockPtr, Vec<Value>) {
    let func_ty = FuncType::get(ctx, result, args, false);
    let func = FuncOp::new(ctx, name.try_into().unwrap(), func_ty);
    func.set_attr_llvm_function_linkage(ctx, LinkageAttr::ExternalLinkage);
    func.get_or_create_entry_block(ctx);
    func.get_operation().insert_at_back(body, ctx);
    let entry = func.get_entry_block(ctx).unwrap();
    let args = entry.deref(ctx).arguments().collect();
    (entry, args)
}

fn ret(ctx: &mut Context, block: BlockPtr, value: Value) {
    ReturnOp::new(ctx, Some(value))
        .get_operation()
        .insert_at_back(block, ctx);
}

fn fadd(ctx: &mut Context, block: BlockPtr, lhs: Value, rhs: Value) -> Value {
    let op = FAddOp::new_with_fast_math_flags(ctx, lhs, rhs, FastmathFlagsAttr::default());
    op.get_operation().insert_at_back(block, ctx);
    op.get_result(ctx)
}

fn fsub(ctx: &mut Context, block: BlockPtr, lhs: Value, rhs: Value) -> Value {
    let op = FSubOp::new_with_fast_math_flags(ctx, lhs, rhs, FastmathFlagsAttr::default());
    op.get_operation().insert_at_back(block, ctx);
    op.get_result(ctx)
}

fn fmul(ctx: &mut Context, block: BlockPtr, lhs: Value, rhs: Value) -> Value {
    let op = FMulOp::new_with_fast_math_flags(ctx, lhs, rhs, FastmathFlagsAttr::default());
    op.get_operation().insert_at_back(block, ctx);
    op.get_result(ctx)
}

fn fdiv(ctx: &mut Context, block: BlockPtr, lhs: Value, rhs: Value) -> Value {
    let op = FDivOp::new_with_fast_math_flags(ctx, lhs, rhs, FastmathFlagsAttr::default());
    op.get_operation().insert_at_back(block, ctx);
    op.get_result(ctx)
}

/// Link `object_bytes` with a C `main` from `harness_source` and run the
/// result, returning the exit status.
fn link_with_c_harness_and_run(test: &str, object_bytes: &[u8], harness_source: &str) -> i32 {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(test);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let object = dir.join("test.o");
    let harness = dir.join("main.c");
    let binary = dir.join("test");
    std::fs::write(&object, object_bytes).expect("write object");
    std::fs::write(&harness, harness_source).expect("write harness");

    let link = Command::new("cc")
        .arg("-o")
        .arg(&binary)
        .arg(&harness)
        .arg(&object)
        .arg("-lm")
        .output()
        .expect("run cc");
    assert!(
        link.status.success(),
        "linking failed:\n{}",
        String::from_utf8_lossy(&link.stderr)
    );

    let run = Command::new(&binary).status().expect("run test binary");
    run.code().expect("test binary exited without a code")
}

#[test]
fn fp_polynomial_and_arithmetic_chain() {
    // poly(x: f64) -> f64 { x*x*0.5 + x - 1.0 } and the f32 variant.
    let mut ctx = context();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();

    let f64_ty = f64_ty(&ctx);
    let (entry, args) = func(&mut ctx, body, "poly", f64_ty, vec![f64_ty]);
    let x = args[0];
    let xx = fmul(&mut ctx, entry, x, x);
    let half = f64_const(&mut ctx, entry, 0.5);
    let xx_half = fmul(&mut ctx, entry, xx, half);
    let sum = fadd(&mut ctx, entry, xx_half, x);
    let one = f64_const(&mut ctx, entry, 1.0);
    let result = fsub(&mut ctx, entry, sum, one);
    ret(&mut ctx, entry, result);

    let f32_ty = f32_ty(&ctx);
    let (entry, args) = func(&mut ctx, body, "polyf", f32_ty, vec![f32_ty]);
    let x = args[0];
    let xx = fmul(&mut ctx, entry, x, x);
    let half = f32_const(&mut ctx, entry, 0.5);
    let xx_half = fmul(&mut ctx, entry, xx, half);
    let sum = fadd(&mut ctx, entry, xx_half, x);
    let one = f32_const(&mut ctx, entry, 1.0);
    let result = fsub(&mut ctx, entry, sum, one);
    ret(&mut ctx, entry, result);

    // quarter(x: f64) -> f64 { -(x / -4.0) }: fdiv and fneg.
    let (entry, args) = func(&mut ctx, body, "quarter", f64_ty, vec![f64_ty]);
    let minus_four = f64_const(&mut ctx, entry, -4.0);
    let divided = fdiv(&mut ctx, entry, args[0], minus_four);
    let neg = FNegOp::new_with_fast_math_flags(&mut ctx, divided, FastmathFlagsAttr::default());
    neg.get_operation().insert_at_back(entry, &ctx);
    let neg_result = neg.get_result(&ctx);
    ret(&mut ctx, entry, neg_result);

    let bytes = aarch64_linux::emit_elf_object_bytes(&mut ctx, module.get_operation()).unwrap();
    const HARNESS: &str = r#"
#include <math.h>
extern double poly(double);
extern float polyf(float);
extern double quarter(double);
int main(void) {
    if (poly(2.0) != 3.0) return 1;
    if (poly(-1.5) != -1.375) return 2;
    if (poly(0.0) != -1.0) return 3;
    if (polyf(2.0f) != 3.0f) return 4;
    if (polyf(-1.5f) != -1.375f) return 5;
    if (quarter(8.0) != 2.0) return 6;
    if (quarter(-8.0) != -2.0) return 7;
    return 0;
}
"#;
    assert_eq!(
        link_with_c_harness_and_run("fp_polynomial_and_arithmetic_chain", &bytes, HARNESS),
        0
    );
}

#[test]
fn fp_compare_branch_and_nan_predicates() {
    let mut ctx = context();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
    let f64_ty = f64_ty(&ctx);
    let i64_ty = i64_ty(&mut ctx);

    // pick_max(a, b) -> f64 { if a > b { a } else { b } } (fcmp + branch).
    let func_ty = FuncType::get(&mut ctx, f64_ty, vec![f64_ty, f64_ty], false);
    let pick = FuncOp::new(&mut ctx, "pick_max".try_into().unwrap(), func_ty);
    pick.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
    pick.get_or_create_entry_block(&mut ctx);
    pick.get_operation().insert_at_back(body, &ctx);
    let entry = pick.get_entry_block(&ctx).unwrap();
    let a = entry.deref(&ctx).get_argument(0);
    let b = entry.deref(&ctx).get_argument(1);
    let then_block = BasicBlock::new(&mut ctx, Some("then".try_into().unwrap()), vec![]);
    then_block.insert_at_back(pick.get_region(&ctx).unwrap(), &ctx);
    let else_block = BasicBlock::new(&mut ctx, Some("else".try_into().unwrap()), vec![]);
    else_block.insert_at_back(pick.get_region(&ctx).unwrap(), &ctx);
    let gt = FCmpOp::new(&mut ctx, FCmpPredicateAttr::OGT, a, b);
    gt.set_fast_math_flags(&ctx, FastmathFlagsAttr::default());
    gt.get_operation().insert_at_back(entry, &ctx);
    let gt_result = gt.get_result(&ctx);
    CondBrOp::new(&mut ctx, gt_result, then_block, vec![], else_block, vec![])
        .get_operation()
        .insert_at_back(entry, &ctx);
    ret(&mut ctx, then_block, a);
    ret(&mut ctx, else_block, b);

    // One function per predicate group returning the comparison bit as i64.
    for (name, predicate) in [
        ("cmp_oeq", FCmpPredicateAttr::OEQ),
        ("cmp_olt", FCmpPredicateAttr::OLT),
        ("cmp_ole", FCmpPredicateAttr::OLE),
        ("cmp_oge", FCmpPredicateAttr::OGE),
        ("cmp_one", FCmpPredicateAttr::ONE),
        ("cmp_ord", FCmpPredicateAttr::ORD),
        ("cmp_ueq", FCmpPredicateAttr::UEQ),
        ("cmp_une", FCmpPredicateAttr::UNE),
        ("cmp_ult", FCmpPredicateAttr::ULT),
        ("cmp_uge", FCmpPredicateAttr::UGE),
        ("cmp_uno", FCmpPredicateAttr::UNO),
    ] {
        let (entry, args) = func(&mut ctx, body, name, i64_ty, vec![f64_ty, f64_ty]);
        let cmp = FCmpOp::new(&mut ctx, predicate, args[0], args[1]);
        cmp.set_fast_math_flags(&ctx, FastmathFlagsAttr::default());
        cmp.get_operation().insert_at_back(entry, &ctx);
        let bit = cmp.get_result(&ctx);
        // i1 -> i64 zext is a representation no-op in this backend.
        let zext = pliron_ll::dialects::llvm::ops::ZExtOp::new(&mut ctx, bit, i64_ty);
        zext.get_operation().insert_at_back(entry, &ctx);
        let wide = zext.get_result(&ctx);
        ret(&mut ctx, entry, wide);
    }

    let bytes = aarch64_linux::emit_elf_object_bytes(&mut ctx, module.get_operation()).unwrap();
    const HARNESS: &str = r#"
#include <math.h>
#include <stdint.h>
extern double pick_max(double, double);
extern int64_t cmp_oeq(double, double);
extern int64_t cmp_olt(double, double);
extern int64_t cmp_ole(double, double);
extern int64_t cmp_oge(double, double);
extern int64_t cmp_one(double, double);
extern int64_t cmp_ord(double, double);
extern int64_t cmp_ueq(double, double);
extern int64_t cmp_une(double, double);
extern int64_t cmp_ult(double, double);
extern int64_t cmp_uge(double, double);
extern int64_t cmp_uno(double, double);
int main(void) {
    double nan = NAN;
    if (pick_max(1.5, 2.5) != 2.5) return 1;
    if (pick_max(-1.0, -2.0) != -1.0) return 2;
    /* NaN > b is false, so the else edge returns b */
    if (pick_max(nan, 7.0) != 7.0) return 3;
    if (cmp_oeq(1.0, 1.0) != 1 || cmp_oeq(1.0, 2.0) != 0 || cmp_oeq(nan, nan) != 0) return 4;
    if (cmp_olt(1.0, 2.0) != 1 || cmp_olt(2.0, 1.0) != 0 || cmp_olt(nan, 1.0) != 0) return 5;
    if (cmp_ole(1.0, 1.0) != 1 || cmp_ole(2.0, 1.0) != 0 || cmp_ole(nan, 1.0) != 0) return 6;
    if (cmp_oge(2.0, 1.0) != 1 || cmp_oge(1.0, 2.0) != 0 || cmp_oge(nan, 1.0) != 0) return 7;
    if (cmp_one(1.0, 2.0) != 1 || cmp_one(1.0, 1.0) != 0 || cmp_one(nan, 1.0) != 0) return 8;
    if (cmp_ord(1.0, 2.0) != 1 || cmp_ord(nan, 1.0) != 0) return 9;
    if (cmp_ueq(1.0, 1.0) != 1 || cmp_ueq(1.0, 2.0) != 0 || cmp_ueq(nan, 1.0) != 1) return 10;
    if (cmp_une(1.0, 2.0) != 1 || cmp_une(1.0, 1.0) != 0 || cmp_une(nan, 1.0) != 1) return 11;
    if (cmp_ult(1.0, 2.0) != 1 || cmp_ult(2.0, 1.0) != 0 || cmp_ult(nan, 1.0) != 1) return 12;
    if (cmp_uge(2.0, 1.0) != 1 || cmp_uge(1.0, 2.0) != 0 || cmp_uge(nan, 1.0) != 1) return 13;
    if (cmp_uno(nan, 1.0) != 1 || cmp_uno(1.0, nan) != 1 || cmp_uno(1.0, 2.0) != 0) return 14;
    return 0;
}
"#;
    assert_eq!(
        link_with_c_harness_and_run("fp_compare_branch_and_nan_predicates", &bytes, HARNESS),
        0
    );
}

#[test]
fn int_float_conversions_saturate_like_rust() {
    let mut ctx = context();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
    let f64_ty = f64_ty(&ctx);
    let f32_ty = f32_ty(&ctx);
    let i64_ty = i64_ty(&mut ctx);
    let i32_ty = i32_ty(&mut ctx);

    // i2d(x: i64) -> f64 and u2d(x: u64) -> f64.
    let (entry, args) = func(&mut ctx, body, "i2d", f64_ty, vec![i64_ty]);
    let conv = SIToFPOp::new(&mut ctx, args[0], f64_ty);
    conv.get_operation().insert_at_back(entry, &ctx);
    let conv_result = conv.get_result(&ctx);
    ret(&mut ctx, entry, conv_result);

    let (entry, args) = func(&mut ctx, body, "u2d", f64_ty, vec![i64_ty]);
    let conv = UIToFPOp::new(&mut ctx, args[0], f64_ty);
    conv.get_operation().insert_at_back(entry, &ctx);
    let conv_result = conv.get_result(&ctx);
    ret(&mut ctx, entry, conv_result);

    // d2i(x: f64) -> i64 (fcvtzs) and d2u32(x: f64) -> u32 (fcvtzu, 32-bit).
    let (entry, args) = func(&mut ctx, body, "d2i", i64_ty, vec![f64_ty]);
    let conv = FPToSIOp::new(&mut ctx, args[0], i64_ty);
    conv.get_operation().insert_at_back(entry, &ctx);
    let conv_result = conv.get_result(&ctx);
    ret(&mut ctx, entry, conv_result);

    let (entry, args) = func(&mut ctx, body, "d2u32", i32_ty, vec![f64_ty]);
    let conv = FPToUIOp::new(&mut ctx, args[0], i32_ty);
    conv.get_operation().insert_at_back(entry, &ctx);
    let conv_result = conv.get_result(&ctx);
    ret(&mut ctx, entry, conv_result);

    // f2i32(x: f32) -> i32 (fcvtzs, single source, 32-bit destination).
    let (entry, args) = func(&mut ctx, body, "f2i32", i32_ty, vec![f32_ty]);
    let conv = FPToSIOp::new(&mut ctx, args[0], i32_ty);
    conv.get_operation().insert_at_back(entry, &ctx);
    let conv_result = conv.get_result(&ctx);
    ret(&mut ctx, entry, conv_result);

    // ext(x: f32) -> f64 and trunc(x: f64) -> f32.
    let (entry, args) = func(&mut ctx, body, "ext", f64_ty, vec![f32_ty]);
    let conv = FPExtOp::new(&mut ctx, args[0], f64_ty);
    conv.get_operation().insert_at_back(entry, &ctx);
    let conv_result = conv.get_result(&ctx);
    ret(&mut ctx, entry, conv_result);

    let (entry, args) = func(&mut ctx, body, "truncd", f32_ty, vec![f64_ty]);
    let conv = FPTruncOp::new(&mut ctx, args[0], f32_ty);
    conv.get_operation().insert_at_back(entry, &ctx);
    let conv_result = conv.get_result(&ctx);
    ret(&mut ctx, entry, conv_result);

    let bytes = aarch64_linux::emit_elf_object_bytes(&mut ctx, module.get_operation()).unwrap();
    // Saturation semantics: aarch64 fcvtzs/fcvtzu saturate at the
    // destination bounds and convert NaN to 0, matching Rust `as` casts.
    const HARNESS: &str = r#"
#include <math.h>
#include <stdint.h>
extern double i2d(int64_t);
extern double u2d(uint64_t);
extern int64_t d2i(double);
extern uint32_t d2u32(double);
extern int32_t f2i32(float);
extern double ext(float);
extern float truncd(double);
int main(void) {
    if (i2d(-5) != -5.0) return 1;
    if (i2d(INT64_MIN) != -9223372036854775808.0) return 2;
    if (u2d(5) != 5.0) return 3;
    /* u64::MAX -> f64 rounds to 2^64 */
    if (u2d(UINT64_MAX) != 18446744073709551616.0) return 4;
    if (d2i(-3.9) != -3) return 5;                 /* truncates toward zero */
    if (d2i(3.9) != 3) return 6;
    if (d2i(1e300) != INT64_MAX) return 7;         /* saturates high */
    if (d2i(-1e300) != INT64_MIN) return 8;        /* saturates low */
    if (d2i(NAN) != 0) return 9;                   /* NaN -> 0 like Rust */
    if (d2u32(-1.0) != 0) return 10;               /* saturates at 0 */
    if (d2u32(5000000000.0) != UINT32_MAX) return 11;
    if (d2u32(4294967295.0) != UINT32_MAX) return 12;
    if (d2u32(NAN) != 0) return 13;
    if (d2u32(3.5) != 3) return 14;
    if (f2i32(-2.5f) != -2) return 15;
    if (f2i32(1e30f) != INT32_MAX) return 16;
    if (f2i32(NAN) != 0) return 17;
    if (ext(1.5f) != 1.5) return 18;
    if (truncd(2.5) != 2.5f) return 19;
    if (!isnan(ext(NAN))) return 20;
    return 0;
}
"#;
    assert_eq!(
        link_with_c_harness_and_run("int_float_conversions_saturate_like_rust", &bytes, HARNESS),
        0
    );
}

#[test]
fn fp_args_span_registers_and_stack() {
    // sum10(f64 x 10) -> f64: the ninth and tenth arguments arrive on the
    // stack. mixed(i64, f64, i64, f64) -> f64 exercises the independent
    // integer/FP register lanes.
    let mut ctx = context();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
    let f64_ty = f64_ty(&ctx);
    let i64_ty = i64_ty(&mut ctx);

    let (entry, args) = func(&mut ctx, body, "sum10", f64_ty, vec![f64_ty; 10]);
    let mut acc = args[0];
    for arg in &args[1..] {
        acc = fadd(&mut ctx, entry, acc, *arg);
    }
    ret(&mut ctx, entry, acc);

    let (entry, args) = func(
        &mut ctx,
        body,
        "mixed",
        f64_ty,
        vec![i64_ty, f64_ty, i64_ty, f64_ty],
    );
    let a = SIToFPOp::new(&mut ctx, args[0], f64_ty);
    a.get_operation().insert_at_back(entry, &ctx);
    let a = a.get_result(&ctx);
    let c = SIToFPOp::new(&mut ctx, args[2], f64_ty);
    c.get_operation().insert_at_back(entry, &ctx);
    let c = c.get_result(&ctx);
    let ints = fadd(&mut ctx, entry, a, c);
    let floats = fadd(&mut ctx, entry, args[1], args[3]);
    let hundred = f64_const(&mut ctx, entry, 100.0);
    let scaled = fmul(&mut ctx, entry, ints, hundred);
    let result = fadd(&mut ctx, entry, scaled, floats);
    ret(&mut ctx, entry, result);

    let bytes = aarch64_linux::emit_elf_object_bytes(&mut ctx, module.get_operation()).unwrap();
    const HARNESS: &str = r#"
extern double sum10(double, double, double, double, double, double, double, double, double, double);
extern double mixed(long, double, long, double);
int main(void) {
    if (sum10(1, 2, 3, 4, 5, 6, 7, 8, 9, 10) != 55.0) return 1;
    if (sum10(0.5, 0, 0, 0, 0, 0, 0, 0, 0.25, 0.125) != 0.875) return 2;
    if (mixed(1, 0.5, 2, 0.25) != 300.75) return 3;
    if (mixed(-1, 0.5, -2, -0.25) != -299.75) return 4;
    return 0;
}
"#;
    assert_eq!(
        link_with_c_harness_and_run("fp_args_span_registers_and_stack", &bytes, HARNESS),
        0
    );
}

#[test]
fn fp_spilling_under_pressure_and_across_calls() {
    // pressure(x) computes eight values live at once (the FP pool holds
    // four), calls a helper in the middle (forcing spills of everything
    // live across the call), then combines them.
    let mut ctx = context();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
    let f64_ty = f64_ty(&ctx);

    let (entry, args) = func(&mut ctx, body, "double_it", f64_ty, vec![f64_ty]);
    let two = f64_const(&mut ctx, entry, 2.0);
    let result = fmul(&mut ctx, entry, args[0], two);
    ret(&mut ctx, entry, result);

    let (entry, args) = func(&mut ctx, body, "pressure", f64_ty, vec![f64_ty]);
    let x = args[0];
    let mut terms = Vec::new();
    for index in 1..=8u32 {
        let offset = f64_const(&mut ctx, entry, index as f64);
        terms.push(fadd(&mut ctx, entry, x, offset));
    }
    // A call in the middle: every term is live across it.
    let call_ty = FuncType::get(&mut ctx, f64_ty, vec![f64_ty], false);
    let call = CallOp::new(
        &mut ctx,
        pliron::builtin::op_interfaces::CallOpCallable::Direct("double_it".try_into().unwrap()),
        call_ty,
        vec![x],
    );
    call.get_operation().insert_at_back(entry, &ctx);
    let doubled = call.get_operation().deref(&ctx).get_result(0);
    let mut acc = doubled;
    for (index, term) in terms.iter().enumerate() {
        let scale = f64_const(&mut ctx, entry, (index + 1) as f64);
        let scaled = fmul(&mut ctx, entry, *term, scale);
        acc = fadd(&mut ctx, entry, acc, scaled);
    }
    ret(&mut ctx, entry, acc);

    let bytes = aarch64_linux::emit_elf_object_bytes(&mut ctx, module.get_operation()).unwrap();
    // pressure(x) = 2x + sum_{i=1..8} i*(x + i) = 2x + 36x + 204.
    const HARNESS: &str = r#"
extern double pressure(double);
int main(void) {
    if (pressure(0.0) != 204.0) return 1;
    if (pressure(1.0) != 242.0) return 2;
    if (pressure(-1.5) != 147.0) return 3;
    return 0;
}
"#;
    assert_eq!(
        link_with_c_harness_and_run("fp_spilling_under_pressure_and_across_calls", &bytes, HARNESS),
        0
    );
}

#[test]
fn fp_constants_in_all_materialization_forms() {
    // 1.5 uses the fmov-immediate form, 0.0 the GPR route, pi the literal
    // pool, and NaN must round-trip bit-exactly through the pool.
    let mut ctx = context();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
    let f64_ty = f64_ty(&ctx);
    let f32_ty = f32_ty(&ctx);

    for (name, value) in [
        ("const_zero", 0.0f64),
        ("const_imm", 1.5),
        ("const_pi", std::f64::consts::PI),
        ("const_nan", f64::NAN),
        ("const_denormal", f64::from_bits(1)),
    ] {
        let (entry, _) = func(&mut ctx, body, name, f64_ty, vec![]);
        let constant = f64_const(&mut ctx, entry, value);
        ret(&mut ctx, entry, constant);
    }
    for (name, value) in [("constf_third", 1.0f32 / 3.0), ("constf_neg2", -2.0f32)] {
        let (entry, _) = func(&mut ctx, body, name, f32_ty, vec![]);
        let constant = f32_const(&mut ctx, entry, value);
        ret(&mut ctx, entry, constant);
    }

    let bytes = aarch64_linux::emit_elf_object_bytes(&mut ctx, module.get_operation()).unwrap();
    const HARNESS: &str = r#"
#include <math.h>
#include <string.h>
#include <stdint.h>
extern double const_zero(void);
extern double const_imm(void);
extern double const_pi(void);
extern double const_nan(void);
extern double const_denormal(void);
extern float constf_third(void);
extern float constf_neg2(void);
int main(void) {
    if (const_zero() != 0.0) return 1;
    if (const_imm() != 1.5) return 2;
    if (const_pi() != 3.141592653589793) return 3;
    if (!isnan(const_nan())) return 4;
    double d = const_denormal();
    uint64_t bits;
    memcpy(&bits, &d, 8);
    if (bits != 1) return 5;
    if (constf_third() != 1.0f / 3.0f) return 6;
    if (constf_neg2() != -2.0f) return 7;
    return 0;
}
"#;
    assert_eq!(
        link_with_c_harness_and_run("fp_constants_in_all_materialization_forms", &bytes, HARNESS),
        0
    );
}

#[test]
fn dynamic_i128_shifts_match_native_semantics() {
    // shl128 / lshr128 / ashr128 (x: i128, n: u64) -> i128 with the shift
    // amount only known at run time.
    let mut ctx = context();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
    let i128_ty = i128_ty(&mut ctx);
    let i64_ty = i64_ty(&mut ctx);

    // The shift amount must be an i128 operand to match the value type; the
    // (zero-extending) conversion from the i64 argument is a ZExt.
    let (entry, args) = func(&mut ctx, body, "shl128", i128_ty, vec![i128_ty, i64_ty]);
    let amount = pliron_ll::dialects::llvm::ops::ZExtOp::new(&mut ctx, args[1], i128_ty);
    amount.get_operation().insert_at_back(entry, &ctx);
    let amount_wide = amount.get_result(&ctx);
    let shifted = ShlOp::new_with_overflow_flag(&mut ctx, args[0], amount_wide, Default::default());
    shifted.get_operation().insert_at_back(entry, &ctx);
    let shifted_result = shifted.get_result(&ctx);
    ret(&mut ctx, entry, shifted_result);

    let (entry, args) = func(&mut ctx, body, "lshr128", i128_ty, vec![i128_ty, i64_ty]);
    let amount = pliron_ll::dialects::llvm::ops::ZExtOp::new(&mut ctx, args[1], i128_ty);
    amount.get_operation().insert_at_back(entry, &ctx);
    let amount_wide = amount.get_result(&ctx);
    let shifted = LShrOp::new(&mut ctx, args[0], amount_wide);
    shifted.get_operation().insert_at_back(entry, &ctx);
    let shifted_result = shifted.get_result(&ctx);
    ret(&mut ctx, entry, shifted_result);

    let (entry, args) = func(&mut ctx, body, "ashr128", i128_ty, vec![i128_ty, i64_ty]);
    let amount = pliron_ll::dialects::llvm::ops::ZExtOp::new(&mut ctx, args[1], i128_ty);
    amount.get_operation().insert_at_back(entry, &ctx);
    let amount_wide = amount.get_result(&ctx);
    let shifted = AShrOp::new(&mut ctx, args[0], amount_wide);
    shifted.get_operation().insert_at_back(entry, &ctx);
    let shifted_result = shifted.get_result(&ctx);
    ret(&mut ctx, entry, shifted_result);

    // ashr64(x: i64, n: i64) -> i64: the plain 64-bit asr path.
    let (entry, args) = func(&mut ctx, body, "ashr64", i64_ty, vec![i64_ty, i64_ty]);
    let shifted = AShrOp::new(&mut ctx, args[0], args[1]);
    shifted.get_operation().insert_at_back(entry, &ctx);
    let shifted_result = shifted.get_result(&ctx);
    ret(&mut ctx, entry, shifted_result);

    let bytes = aarch64_linux::emit_elf_object_bytes(&mut ctx, module.get_operation()).unwrap();
    const HARNESS: &str = r#"
#include <stdint.h>
typedef unsigned __int128 u128;
typedef __int128 i128;
extern u128 shl128(u128, uint64_t);
extern u128 lshr128(u128, uint64_t);
extern i128 ashr128(i128, uint64_t);
extern int64_t ashr64(int64_t, int64_t);
int main(void) {
    uint64_t amounts[] = {0, 1, 5, 63, 64, 65, 100, 127};
    u128 patterns[] = {
        ((u128)0x0123456789abcdefULL << 64) | 0xfedcba9876543210ULL,
        (u128)1,
        ~(u128)0,
        ((u128)0x8000000000000000ULL << 64),
        ((u128)0x00000000deadbeefULL << 64) | 0x00c0ffee00c0ffeeULL,
    };
    for (unsigned p = 0; p < sizeof(patterns) / sizeof(patterns[0]); p++) {
        for (unsigned a = 0; a < sizeof(amounts) / sizeof(amounts[0]); a++) {
            u128 x = patterns[p];
            uint64_t n = amounts[a];
            volatile uint64_t vn = n; /* defeat constant folding on the C side */
            if (shl128(x, n) != (x << vn)) return 1 + 10 * p + a;
            if (lshr128(x, n) != (x >> vn)) return 101 + 10 * p + a;
            if (ashr128((i128)x, n) != ((i128)x >> vn)) return 201 + 10 * p + a;
        }
    }
    if (ashr64(-8, 1) != -4) return 250;
    if (ashr64(INT64_MIN, 63) != -1) return 251;
    if (ashr64(8, 2) != 2) return 252;
    return 0;
}
"#;
    assert_eq!(
        link_with_c_harness_and_run("dynamic_i128_shifts_match_native_semantics", &bytes, HARNESS),
        0
    );
}

#[test]
fn fp_values_survive_memory_and_bitcasts() {
    // store/load f64 and f32 through an alloca; bitcast f64 <-> i64.
    let mut ctx = context();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
    let f64_ty = f64_ty(&ctx);
    let f32_ty = f32_ty(&ctx);
    let i64_ty = i64_ty(&mut ctx);

    // round_trip(x: f64) -> f64 { let slot = alloca; *slot = x; *slot }
    let (entry, args) = func(&mut ctx, body, "round_trip", f64_ty, vec![f64_ty]);
    let count_ty = IntegerType::get(&mut ctx, 64, Signedness::Signless);
    let one = ConstantOp::new(
        &mut ctx,
        Box::new(IntegerAttr::new(
            count_ty,
            APInt::from_u64(1, NonZero::new(64).unwrap()),
        )),
    );
    one.get_operation().insert_at_back(entry, &ctx);
    let one_result = one.get_result(&ctx);
    let slot = pliron_ll::dialects::llvm::ops::AllocaOp::new(&mut ctx, f64_ty, one_result);
    slot.get_operation().insert_at_back(entry, &ctx);
    let slot_result = slot.get_result(&ctx);
    pliron_ll::dialects::llvm::ops::StoreOp::new(&mut ctx, args[0], slot_result)
        .get_operation()
        .insert_at_back(entry, &ctx);
    let load = pliron_ll::dialects::llvm::ops::LoadOp::new(&mut ctx, slot_result, f64_ty);
    load.get_operation().insert_at_back(entry, &ctx);
    let loaded = load.get_result(&ctx);
    ret(&mut ctx, entry, loaded);

    // round_trip_f(x: f32) -> f32, same shape.
    let (entry, args) = func(&mut ctx, body, "round_trip_f", f32_ty, vec![f32_ty]);
    let count_ty = IntegerType::get(&mut ctx, 64, Signedness::Signless);
    let one = ConstantOp::new(
        &mut ctx,
        Box::new(IntegerAttr::new(
            count_ty,
            APInt::from_u64(1, NonZero::new(64).unwrap()),
        )),
    );
    one.get_operation().insert_at_back(entry, &ctx);
    let one_result = one.get_result(&ctx);
    let slot = pliron_ll::dialects::llvm::ops::AllocaOp::new(&mut ctx, f32_ty, one_result);
    slot.get_operation().insert_at_back(entry, &ctx);
    let slot_result = slot.get_result(&ctx);
    pliron_ll::dialects::llvm::ops::StoreOp::new(&mut ctx, args[0], slot_result)
        .get_operation()
        .insert_at_back(entry, &ctx);
    let load = pliron_ll::dialects::llvm::ops::LoadOp::new(&mut ctx, slot_result, f32_ty);
    load.get_operation().insert_at_back(entry, &ctx);
    let loaded = load.get_result(&ctx);
    ret(&mut ctx, entry, loaded);

    // bits(x: f64) -> i64 and unbits(x: i64) -> f64 (bitcasts).
    let (entry, args) = func(&mut ctx, body, "bits", i64_ty, vec![f64_ty]);
    let cast = pliron_ll::dialects::llvm::ops::BitcastOp::new(&mut ctx, args[0], i64_ty);
    cast.get_operation().insert_at_back(entry, &ctx);
    let cast_result = cast.get_result(&ctx);
    ret(&mut ctx, entry, cast_result);

    let (entry, args) = func(&mut ctx, body, "unbits", f64_ty, vec![i64_ty]);
    let cast = pliron_ll::dialects::llvm::ops::BitcastOp::new(&mut ctx, args[0], f64_ty);
    cast.get_operation().insert_at_back(entry, &ctx);
    let cast_result = cast.get_result(&ctx);
    ret(&mut ctx, entry, cast_result);

    let bytes = aarch64_linux::emit_elf_object_bytes(&mut ctx, module.get_operation()).unwrap();
    const HARNESS: &str = r#"
#include <math.h>
#include <string.h>
#include <stdint.h>
extern double round_trip(double);
extern float round_trip_f(float);
extern int64_t bits(double);
extern double unbits(int64_t);
int main(void) {
    if (round_trip(2.75) != 2.75) return 1;
    if (!isnan(round_trip(NAN))) return 2;
    if (round_trip_f(-0.5f) != -0.5f) return 3;
    if (bits(1.0) != 0x3ff0000000000000LL) return 4;
    if (unbits(0x4000000000000000LL) != 2.0) return 5;
    double x = -123.456;
    if (unbits(bits(x)) != x) return 6;
    return 0;
}
"#;
    assert_eq!(
        link_with_c_harness_and_run("fp_values_survive_memory_and_bitcasts", &bytes, HARNESS),
        0
    );
}
