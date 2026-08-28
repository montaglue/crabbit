//! NVPTX emission tests: build LLVM dialect IR by hand, translate it to PTX
//! text with `nvptx::write_ptx_from_ir`, and — when a CUDA toolkit is
//! installed — assemble the result with `ptxas` as an external correctness
//! oracle.

use std::num::NonZero;
use std::path::PathBuf;
use std::process::Command;

use pliron::builtin::op_interfaces::AtMostOneRegionInterface as _;
use pliron_ll::{
    context::Context,
    dialects::{
        builtin::{
            self,
            attributes::IntegerAttr,
            op_interfaces::{OneRegionInterface, OneResultInterface},
            ops::ConstantOp,
            types::{IntegerType, Signedness},
        },
        llvm::{
            attributes::{ICmpPredicateAttr, LinkageAttr},
            op_interfaces::IntBinArithOpWithOverflowFlag as _,
            ops::{
                AddOp, BrOp, CallOp, CondBrOp, FuncOp, GepIndex, GetElementPtrOp, ICmpOp, LoadOp,
                ReturnOp, StoreOp,
            },
            types::{FuncType, PointerType, VoidType},
        },
    },
    ir::{basic_block::BasicBlock, op::Op, value::Value},
    linked_list::ContainsLinkedList,
    nvptx::{self, PtxTarget},
    utils::apint::APInt,
};

fn i64_ty(ctx: &mut Context) -> pliron_ll::r#type::TypeHandle {
    IntegerType::get(ctx, 64, Signedness::Signless).into()
}

fn i32_ty(ctx: &mut Context) -> pliron_ll::r#type::TypeHandle {
    IntegerType::get(ctx, 32, Signedness::Signless).into()
}

fn ptr_ty(ctx: &mut Context) -> pliron_ll::r#type::TypeHandle {
    PointerType::get(ctx, 0).into()
}

fn i64_const(
    ctx: &mut Context,
    block: pliron_ll::context::Ptr<BasicBlock>,
    value: u64,
) -> Value {
    let ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let constant = ConstantOp::new(
        ctx,
        Box::new(IntegerAttr::new(
            ty,
            APInt::from_u64(value, NonZero::new(64).unwrap()),
        )),
    );
    constant.get_operation().insert_at_back(block, ctx);
    constant.get_result(ctx)
}

/// Assemble `ptx` with ptxas as an external verifier. Skips (with a note)
/// when no CUDA toolkit is installed so the suite still runs on machines
/// without one.
fn ptxas_verify(test: &str, ptx: &str) {
    let ptxas = ["ptxas", "/usr/local/cuda/bin/ptxas"]
        .iter()
        .find(|p| Command::new(p).arg("--version").output().is_ok())
        .copied();
    let Some(ptxas) = ptxas else {
        eprintln!("ptxas not found; skipping assembly verification for {test}");
        return;
    };
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(test);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let ptx_path = dir.join("kernel.ptx");
    let cubin_path = dir.join("kernel.cubin");
    std::fs::write(&ptx_path, ptx).expect("write ptx");
    let output = Command::new(ptxas)
        .args(["--gpu-name", "sm_121", "-o"])
        .arg(&cubin_path)
        .arg(&ptx_path)
        .output()
        .expect("run ptxas");
    assert!(
        output.status.success(),
        "ptxas rejected the module:\n{}\n--- ptx ---\n{ptx}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn new_kernel(
    ctx: &mut Context,
    module_body: pliron_ll::context::Ptr<BasicBlock>,
    name: &str,
    params: Vec<pliron_ll::r#type::TypeHandle>,
) -> FuncOp {
    let void_ty = VoidType::get(ctx).into();
    let func_ty = FuncType::get(ctx, void_ty, params, false);
    let func = FuncOp::new(ctx, name.try_into().unwrap(), func_ty);
    func.set_attr_llvm_function_linkage(ctx, LinkageAttr::ExternalLinkage);
    func.get_or_create_entry_block(ctx);
    func.get_operation().insert_at_back(module_body, ctx);
    func
}

/// c[tid] = a[tid] + b[tid], with the thread index from `%tid.x`.
#[test]
fn vector_add_kernel() {
    let mut ctx = Context::new();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "kernels".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();

    let i64_ty = i64_ty(&mut ctx);
    let i32_ty = i32_ty(&mut ctx);
    let ptr = ptr_ty(&mut ctx);
    let func = new_kernel(&mut ctx, body, "vadd", vec![ptr, ptr, ptr]);
    let entry = func.get_entry_block(&ctx).unwrap();
    let args: Vec<_> = entry.deref(&ctx).arguments().collect();
    let (a, b, c) = (args[0], args[1], args[2]);

    let tid_ty = FuncType::get(&mut ctx, i32_ty, vec![], false);
    let tid_call = CallOp::new(
        &mut ctx,
        pliron::builtin::op_interfaces::CallOpCallable::Direct(
            "llvm_nvvm_read_ptx_sreg_tid_x".try_into().unwrap(),
        ),
        tid_ty,
        vec![],
    );
    tid_call.get_operation().insert_at_back(entry, &ctx);
    let tid = tid_call.get_operation().deref(&ctx).get_result(0);

    let load_elem = |ctx: &mut Context, base: Value| -> Value {
        let gep = GetElementPtrOp::new(ctx, base, vec![GepIndex::Value(tid)], i64_ty);
        gep.get_operation().insert_at_back(entry, ctx);
        let addr = gep.get_result(ctx);
        let load = LoadOp::new(ctx, addr, i64_ty);
        load.get_operation().insert_at_back(entry, ctx);
        load.get_result(ctx)
    };
    let lhs = load_elem(&mut ctx, a);
    let rhs = load_elem(&mut ctx, b);

    let sum = AddOp::new_with_overflow_flag(&mut ctx, lhs, rhs, Default::default());
    sum.get_operation().insert_at_back(entry, &ctx);
    let sum = sum.get_result(&ctx);

    let gep = GetElementPtrOp::new(&mut ctx, c, vec![GepIndex::Value(tid)], i64_ty);
    gep.get_operation().insert_at_back(entry, &ctx);
    let dest = gep.get_result(&ctx);
    StoreOp::new(&mut ctx, sum, dest)
        .get_operation()
        .insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);

    let ptx = nvptx::write_ptx_from_ir(&ctx, module.get_operation(), &PtxTarget::default())
        .expect("ptx emission");

    assert!(ptx.contains(".visible .entry vadd("), "{ptx}");
    assert!(ptx.contains("mov.u32 %r0, %tid.x;"), "{ptx}");
    assert!(ptx.contains("mad.lo.s64"), "{ptx}");
    assert!(ptx.contains("ld.u64"), "{ptx}");
    assert!(ptx.contains("st.u64"), "{ptx}");
    ptxas_verify("vector_add_kernel", &ptx);
}

/// out[0] = sum(0..n): a loop with block arguments on the backedge, so the
/// emitter's edge blocks and parallel copies are exercised.
#[test]
fn loop_sum_kernel() {
    let mut ctx = Context::new();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "kernels".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();

    let i64_ty = i64_ty(&mut ctx);
    let ptr = ptr_ty(&mut ctx);
    let func = new_kernel(&mut ctx, body, "sum_upto", vec![i64_ty, ptr]);
    let entry = func.get_entry_block(&ctx).unwrap();
    let args: Vec<_> = entry.deref(&ctx).arguments().collect();
    let (n, out) = (args[0], args[1]);
    let region = func.get_region(&ctx).unwrap();

    let header = BasicBlock::new(
        &mut ctx,
        Some("header".try_into().unwrap()),
        vec![i64_ty, i64_ty],
    );
    header.insert_at_back(region, &ctx);
    let body_block = BasicBlock::new(&mut ctx, Some("body".try_into().unwrap()), vec![]);
    body_block.insert_at_back(region, &ctx);
    let exit = BasicBlock::new(&mut ctx, Some("exit".try_into().unwrap()), vec![]);
    exit.insert_at_back(region, &ctx);

    let zero = i64_const(&mut ctx, entry, 0);
    BrOp::new(&mut ctx, header, vec![zero, zero])
        .get_operation()
        .insert_at_back(entry, &ctx);

    let header_args: Vec<_> = header.deref(&ctx).arguments().collect();
    let (i, acc) = (header_args[0], header_args[1]);
    let in_range = ICmpOp::new(&mut ctx, ICmpPredicateAttr::SLT, i, n);
    in_range.get_operation().insert_at_back(header, &ctx);
    let in_range = in_range.get_result(&ctx);
    CondBrOp::new(&mut ctx, in_range, body_block, vec![], exit, vec![])
        .get_operation()
        .insert_at_back(header, &ctx);

    let acc_next = AddOp::new_with_overflow_flag(&mut ctx, acc, i, Default::default());
    acc_next.get_operation().insert_at_back(body_block, &ctx);
    let acc_next = acc_next.get_result(&ctx);
    let one = i64_const(&mut ctx, body_block, 1);
    let i_next = AddOp::new_with_overflow_flag(&mut ctx, i, one, Default::default());
    i_next.get_operation().insert_at_back(body_block, &ctx);
    let i_next = i_next.get_result(&ctx);
    BrOp::new(&mut ctx, header, vec![i_next, acc_next])
        .get_operation()
        .insert_at_back(body_block, &ctx);

    StoreOp::new(&mut ctx, acc, out)
        .get_operation()
        .insert_at_back(exit, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(exit, &ctx);

    let ptx = nvptx::write_ptx_from_ir(&ctx, module.get_operation(), &PtxTarget::default())
        .expect("ptx emission");

    assert!(ptx.contains(".visible .entry sum_upto("), "{ptx}");
    assert!(ptx.contains("setp.lt.s64"), "{ptx}");
    assert!(ptx.contains("@%p0 bra"), "{ptx}");
    // The backedge copies both loop-carried values.
    assert!(ptx.contains("$L_bb1:"), "{ptx}");
    ptxas_verify("loop_sum_kernel", &ptx);
}
