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

// ---------------------------------------------------------------------------
// FP, shared memory, select, math/atomic intrinsics
// ---------------------------------------------------------------------------

use pliron_ll::dialects::builtin::types::{FP32Type, FP64Type};
use pliron_ll::dialects::llvm::op_interfaces::{BinArithOp as _, CastOpInterface as _};
use pliron_ll::dialects::llvm::{
    attributes::FCmpPredicateAttr,
    ops::{AddressOfOp, FAddOp, FCmpOp, FDivOp, FMulOp, FPExtOp, GlobalOp, SIToFPOp, SelectOp, UIToFPOp, FPToSIOp},
    types::ArrayType,
};

fn call_intrinsic(
    ctx: &mut Context,
    block: pliron_ll::context::Ptr<BasicBlock>,
    name: &str,
    result: Option<pliron_ll::r#type::TypeHandle>,
    args: Vec<Value>,
) -> Option<Value> {
    let ret = result.unwrap_or_else(|| VoidType::get(ctx).into());
    let arg_tys = args.iter().map(|a| pliron::r#type::Typed::get_type(a, ctx)).collect();
    let ty = FuncType::get(ctx, ret, arg_tys, false);
    let call = CallOp::new(
        ctx,
        pliron::builtin::op_interfaces::CallOpCallable::Direct(name.try_into().unwrap()),
        ty,
        args,
    );
    call.get_operation().insert_at_back(block, ctx);
    result.map(|_| call.get_operation().deref(ctx).get_result(0))
}

/// out[tid] = sqrt(a[tid] * b[tid] + f64→f32(c[tid])) with a select on an
/// ordered compare, an atomic add into `count`, and a `.shared` scratch
/// slot: every FP path the corpus kernels need, assembled by ptxas.
#[test]
fn float_math_kernel() {
    let mut ctx = Context::new();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "kernels".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();

    // A `.shared` static: [f32; 256] as raw bytes, zero-initialized.
    let byte_ty: pliron_ll::r#type::TypeHandle =
        IntegerType::get(&mut ctx, 8, Signedness::Unsigned).into();
    let tile_ty = ArrayType::get(&mut ctx, byte_ty, 1024).into();
    let tile = GlobalOp::new(&mut ctx, "TILE".try_into().unwrap(), tile_ty);
    tile.set_attr_llvm_global_linkage(&ctx, LinkageAttr::ExternalLinkage);
    pliron_ll::ll::set_global_data(
        &ctx,
        &tile,
        pliron_ll::ll::DataAttr { bytes: vec![0; 1024], align: 4, mutable: true, relocs: vec![] },
    );
    pliron_ll::ll::set_global_section(&mut ctx, &tile, ".shared");
    tile.get_operation().insert_at_back(body, &ctx);

    let f32_ty: pliron_ll::r#type::TypeHandle = FP32Type::get(&mut ctx).into();
    let f64_ty: pliron_ll::r#type::TypeHandle = FP64Type::get(&mut ctx).into();
    let i32_ty = i32_ty(&mut ctx);
    let i64_ty = i64_ty(&mut ctx);
    let ptr = ptr_ty(&mut ctx);
    let func = new_kernel(&mut ctx, body, "fmath", vec![ptr, ptr, ptr, ptr, ptr]);
    let entry = func.get_entry_block(&ctx).unwrap();
    let args: Vec<_> = entry.deref(&ctx).arguments().collect();
    let (a, b, c, out, count) = (args[0], args[1], args[2], args[3], args[4]);

    let tid = call_intrinsic(&mut ctx, entry, "llvm_nvvm_read_ptx_sreg_tid_x", Some(i32_ty), vec![]).unwrap();
    let load = |ctx: &mut Context, base: Value, elem: pliron_ll::r#type::TypeHandle| -> Value {
        let gep = GetElementPtrOp::new(ctx, base, vec![GepIndex::Value(tid)], elem);
        gep.get_operation().insert_at_back(entry, ctx);
        let load = LoadOp::new(ctx, gep.get_result(ctx), elem);
        load.get_operation().insert_at_back(entry, ctx);
        load.get_result(ctx)
    };
    let av = load(&mut ctx, a, f32_ty);
    let bv = load(&mut ctx, b, f32_ty);
    let cv = load(&mut ctx, c, f64_ty);

    let prod = FMulOp::new(&mut ctx, av, bv);
    prod.get_operation().insert_at_back(entry, &ctx);
    let prod = prod.get_result(&ctx);
    let c32 = pliron_ll::dialects::llvm::ops::FPTruncOp::new(&mut ctx, cv, f32_ty);
    c32.get_operation().insert_at_back(entry, &ctx);
    let c32 = c32.get_result(&ctx);
    let sum = FAddOp::new(&mut ctx, prod, c32);
    sum.get_operation().insert_at_back(entry, &ctx);
    let sum = sum.get_result(&ctx);
    let root = call_intrinsic(&mut ctx, entry, "llvm_nvvm_sqrt_rn_f", Some(f32_ty), vec![sum]).unwrap();

    // select(sum >= 0 ? root : sum / 2)
    let zero = ConstantOp::new(
        &mut ctx,
        Box::new(pliron_ll::dialects::builtin::attributes::FPSingleAttr(
            pliron::utils::apfloat::f32_to_single(0.0),
        )),
    );
    zero.get_operation().insert_at_back(entry, &ctx);
    let zero = zero.get_result(&ctx);
    let two = ConstantOp::new(
        &mut ctx,
        Box::new(pliron_ll::dialects::builtin::attributes::FPSingleAttr(
            pliron::utils::apfloat::f32_to_single(2.0),
        )),
    );
    two.get_operation().insert_at_back(entry, &ctx);
    let two = two.get_result(&ctx);
    let half = FDivOp::new(&mut ctx, sum, two);
    half.get_operation().insert_at_back(entry, &ctx);
    let half = half.get_result(&ctx);
    let nonneg = FCmpOp::new(&mut ctx, FCmpPredicateAttr::OGE, sum, zero);
    nonneg.get_operation().insert_at_back(entry, &ctx);
    let nonneg = nonneg.get_result(&ctx);
    let picked = SelectOp::new(&mut ctx, nonneg, root, half);
    picked.get_operation().insert_at_back(entry, &ctx);
    let picked = picked.get_result(&ctx);

    // Round-trip through the shared tile: st then ld.
    let tile_addr = AddressOfOp::new(&mut ctx, "TILE".try_into().unwrap(), 0);
    tile_addr.get_operation().insert_at_back(entry, &ctx);
    let tile_ptr = tile_addr.get_result(&ctx);
    let slot = GetElementPtrOp::new(&mut ctx, tile_ptr, vec![GepIndex::Value(tid)], f32_ty);
    slot.get_operation().insert_at_back(entry, &ctx);
    let slot = slot.get_result(&ctx);
    StoreOp::new(&mut ctx, picked, slot).get_operation().insert_at_back(entry, &ctx);
    call_intrinsic(&mut ctx, entry, "llvm_nvvm_barrier0", None, vec![]);
    let back = LoadOp::new(&mut ctx, slot, f32_ty);
    back.get_operation().insert_at_back(entry, &ctx);
    let back = back.get_result(&ctx);

    // int<->float casts and fpext.
    let as_int = FPToSIOp::new(&mut ctx, back, i32_ty);
    as_int.get_operation().insert_at_back(entry, &ctx);
    let as_int = as_int.get_result(&ctx);
    let as_f = SIToFPOp::new(&mut ctx, as_int, f32_ty);
    as_f.get_operation().insert_at_back(entry, &ctx);
    let as_f = as_f.get_result(&ctx);
    let tid_f = UIToFPOp::new(&mut ctx, tid, f64_ty);
    tid_f.get_operation().insert_at_back(entry, &ctx);
    let wide = FPExtOp::new(&mut ctx, as_f, f64_ty);
    wide.get_operation().insert_at_back(entry, &ctx);
    let (wide_v, tid_fv) = (wide.get_result(&ctx), tid_f.get_result(&ctx));
    let wide_sum = FAddOp::new(&mut ctx, wide_v, tid_fv);
    wide_sum.get_operation().insert_at_back(entry, &ctx);
    let wide_sum = wide_sum.get_result(&ctx);
    let narrow = pliron_ll::dialects::llvm::ops::FPTruncOp::new(&mut ctx, wide_sum, f32_ty);
    narrow.get_operation().insert_at_back(entry, &ctx);
    let narrow = narrow.get_result(&ctx);

    let dst = GetElementPtrOp::new(&mut ctx, out, vec![GepIndex::Value(tid)], f32_ty);
    dst.get_operation().insert_at_back(entry, &ctx);
    let dst = dst.get_result(&ctx);
    StoreOp::new(&mut ctx, narrow, dst)
        .get_operation()
        .insert_at_back(entry, &ctx);

    let one = i64_const(&mut ctx, entry, 1);
    let one32 = pliron_ll::dialects::llvm::ops::TruncOp::new(&mut ctx, one, i32_ty);
    one32.get_operation().insert_at_back(entry, &ctx);
    let one32 = one32.get_result(&ctx);
    call_intrinsic(&mut ctx, entry, "llvm_nvvm_atomic_add_gen_i", Some(i32_ty), vec![count, one32]);
    let _ = i64_ty;

    ReturnOp::new(&mut ctx, None).get_operation().insert_at_back(entry, &ctx);

    let ptx = nvptx::write_ptx_from_ir(&ctx, module.get_operation(), &PtxTarget::default())
        .expect("ptx emission");
    for needle in [
        ".shared .align 4 .b8 TILE[1024];",
        "cvta.shared.u64",
        "ld.f32",
        "ld.f64",
        "mul.rn.f32",
        "add.rn.f32",
        "add.rn.f64",
        "div.rn.f32",
        "cvt.rn.f32.f64",
        "cvt.f64.f32",
        "sqrt.rn.f32",
        "setp.ge.f32",
        "selp.f32",
        "cvt.rzi.s32.f32",
        "cvt.rn.f32.s32",
        "cvt.rn.f64.u32",
        "mov.f32 %f",
        "0f40000000",
        "atom.add.u32",
        "bar.sync 0;",
        "st.f32",
    ] {
        assert!(ptx.contains(needle), "missing `{needle}` in:\n{ptx}");
    }
    ptxas_verify("float_math_kernel", &ptx);
}

/// Anything unsupported is reported by name.
#[test]
fn unsupported_intrinsic_is_named() {
    let mut ctx = Context::new();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "kernels".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
    let func = new_kernel(&mut ctx, body, "bad", vec![]);
    let entry = func.get_entry_block(&ctx).unwrap();
    call_intrinsic(&mut ctx, entry, "llvm_nvvm_wmma_load", None, vec![]);
    ReturnOp::new(&mut ctx, None).get_operation().insert_at_back(entry, &ctx);
    let err = nvptx::write_ptx_from_ir(&ctx, module.get_operation(), &PtxTarget::default())
        .expect_err("must reject");
    assert!(err.to_string().contains("llvm_nvvm_wmma_load"), "{err}");
}
