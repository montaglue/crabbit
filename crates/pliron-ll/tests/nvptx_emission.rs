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
    // The element offset materializes as shift+add (ptxas lowers that to
    // LEA pairs; a mad.lo.s64 would become a real 64-bit multiply).
    assert!(ptx.contains("shl.b64"), "{ptx}");
    // Kernel pointer params are global-space addresses, so provenance
    // qualifies the accesses; a and b are never stored through, so their
    // loads also take the read-only cache.
    assert!(ptx.contains("ld.global.nc.u64"), "{ptx}");
    assert!(ptx.contains("st.global.u64"), "{ptx}");
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
        // The tile is only ever accessed through proven-shared addresses,
        // so its accesses are `.shared`-qualified and no cvta is needed.
        "st.shared.f32",
        "ld.shared.f32",
        // a/b/c are read-only params (only `out` and the atomic's `count`
        // are written), so their loads take `.nc`.
        "ld.global.nc.f32",
        "ld.global.nc.f64",
        // a*b + c with a single-use product contracts (nvcc -fmad=true).
        "fma.rn.f32",
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
        "atom.global.add.u32",
        "bar.sync 0;",
        "st.global.f32",
    ] {
        assert!(ptx.contains(needle), "missing `{needle}` in:\n{ptx}");
    }
    assert!(!ptx.contains("cvta."), "no address escapes generic here:\n{ptx}");
    assert!(!ptx.contains("mul.rn.f32"), "the product must fuse:\n{ptx}");
    ptxas_verify("float_math_kernel", &ptx);
}

/// FP contraction fuses only single-use products: a reused `fmul` stays a
/// `mul.rn` and its adds stay `add.rn`.
#[test]
fn fp_contraction_single_use_only() {
    let mut ctx = Context::new();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "kernels".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();

    let f32_ty: pliron_ll::r#type::TypeHandle = FP32Type::get(&mut ctx).into();
    let i32_ty = i32_ty(&mut ctx);
    let ptr = ptr_ty(&mut ctx);
    let func = new_kernel(&mut ctx, body, "fmad", vec![ptr, ptr]);
    let entry = func.get_entry_block(&ctx).unwrap();
    let args: Vec<_> = entry.deref(&ctx).arguments().collect();
    let (p, q) = (args[0], args[1]);

    let tid = call_intrinsic(&mut ctx, entry, "llvm_nvvm_read_ptx_sreg_tid_x", Some(i32_ty), vec![]).unwrap();
    let load = |ctx: &mut Context, base: Value, off: u32| -> Value {
        let gep = GetElementPtrOp::new(ctx, base, vec![GepIndex::Constant(off)], f32_ty);
        gep.get_operation().insert_at_back(entry, ctx);
        let load = LoadOp::new(ctx, gep.get_result(ctx), f32_ty);
        load.get_operation().insert_at_back(entry, ctx);
        load.get_result(ctx)
    };
    let (a, b, c) = (load(&mut ctx, p, 0), load(&mut ctx, p, 1), load(&mut ctx, p, 2));

    // Single-use product: contracts.
    let prod = FMulOp::new(&mut ctx, a, b);
    prod.get_operation().insert_at_back(entry, &ctx);
    let prod = prod.get_result(&ctx);
    let fused = FAddOp::new(&mut ctx, prod, c);
    fused.get_operation().insert_at_back(entry, &ctx);
    let fused = fused.get_result(&ctx);

    // Reused product: must NOT contract.
    let prod2 = FMulOp::new(&mut ctx, a, c);
    prod2.get_operation().insert_at_back(entry, &ctx);
    let prod2 = prod2.get_result(&ctx);
    let s1 = FAddOp::new(&mut ctx, prod2, b);
    s1.get_operation().insert_at_back(entry, &ctx);
    let s1 = s1.get_result(&ctx);
    let s2 = FAddOp::new(&mut ctx, s1, prod2);
    s2.get_operation().insert_at_back(entry, &ctx);
    let s2 = s2.get_result(&ctx);

    let sum = FAddOp::new(&mut ctx, fused, s2);
    sum.get_operation().insert_at_back(entry, &ctx);
    let sum = sum.get_result(&ctx);
    let dst = GetElementPtrOp::new(&mut ctx, q, vec![GepIndex::Value(tid)], f32_ty);
    dst.get_operation().insert_at_back(entry, &ctx);
    let dst = dst.get_result(&ctx);
    StoreOp::new(&mut ctx, sum, dst).get_operation().insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None).get_operation().insert_at_back(entry, &ctx);

    let ptx = nvptx::write_ptx_from_ir(&ctx, module.get_operation(), &PtxTarget::default())
        .expect("ptx emission");
    assert_eq!(ptx.matches("fma.rn.f32").count(), 1, "{ptx}");
    assert_eq!(ptx.matches("mul.rn.f32").count(), 1, "{ptx}");
    // s1, s2, and the final sum stay plain adds.
    assert_eq!(ptx.matches("add.rn.f32").count(), 3, "{ptx}");
    ptxas_verify("fp_contraction_single_use_only", &ptx);
}

/// State-space provenance: a `.shared` address that stays inside the proven
/// world keeps its raw form (`st.shared`), while the copy of it that mixes
/// with a global param in a `select` is converted (`cvta.shared.u64`) and
/// dereferenced generically.
#[test]
fn shared_provenance_and_escape() {
    let mut ctx = Context::new();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "kernels".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();

    let byte_ty: pliron_ll::r#type::TypeHandle =
        IntegerType::get(&mut ctx, 8, Signedness::Unsigned).into();
    let tile_ty = ArrayType::get(&mut ctx, byte_ty, 256).into();
    let tile = GlobalOp::new(&mut ctx, "SCRATCH".try_into().unwrap(), tile_ty);
    tile.set_attr_llvm_global_linkage(&ctx, LinkageAttr::ExternalLinkage);
    pliron_ll::ll::set_global_data(
        &ctx,
        &tile,
        pliron_ll::ll::DataAttr { bytes: vec![0; 256], align: 4, mutable: true, relocs: vec![] },
    );
    pliron_ll::ll::set_global_section(&mut ctx, &tile, ".shared");
    tile.get_operation().insert_at_back(body, &ctx);

    let i32_ty = i32_ty(&mut ctx);
    let ptr = ptr_ty(&mut ctx);
    let func = new_kernel(&mut ctx, body, "spaces", vec![ptr, i32_ty]);
    let entry = func.get_entry_block(&ctx).unwrap();
    let args: Vec<_> = entry.deref(&ctx).arguments().collect();
    let (p, n) = (args[0], args[1]);

    let tid = call_intrinsic(&mut ctx, entry, "llvm_nvvm_read_ptx_sreg_tid_x", Some(i32_ty), vec![]).unwrap();
    let tile_addr = AddressOfOp::new(&mut ctx, "SCRATCH".try_into().unwrap(), 0);
    tile_addr.get_operation().insert_at_back(entry, &ctx);
    let t = tile_addr.get_result(&ctx);

    // Proven-shared store: st.shared through a gep off the raw address.
    let slot = GetElementPtrOp::new(&mut ctx, t, vec![GepIndex::Value(tid)], i32_ty);
    slot.get_operation().insert_at_back(entry, &ctx);
    let slot = slot.get_result(&ctx);
    StoreOp::new(&mut ctx, tid, slot).get_operation().insert_at_back(entry, &ctx);

    // Mixed provenance: select(shared, global param) joins at generic, so
    // the shared arm converts and the load stays generic.
    let cond = ICmpOp::new(&mut ctx, ICmpPredicateAttr::ULT, tid, n);
    cond.get_operation().insert_at_back(entry, &ctx);
    let cond = cond.get_result(&ctx);
    let picked = SelectOp::new(&mut ctx, cond, t, p);
    picked.get_operation().insert_at_back(entry, &ctx);
    let picked = picked.get_result(&ctx);
    let v = LoadOp::new(&mut ctx, picked, i32_ty);
    v.get_operation().insert_at_back(entry, &ctx);
    let v = v.get_result(&ctx);

    let dst = GetElementPtrOp::new(&mut ctx, p, vec![GepIndex::Value(tid)], i32_ty);
    dst.get_operation().insert_at_back(entry, &ctx);
    let dst = dst.get_result(&ctx);
    StoreOp::new(&mut ctx, v, dst).get_operation().insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None).get_operation().insert_at_back(entry, &ctx);

    let ptx = nvptx::write_ptx_from_ir(&ctx, module.get_operation(), &PtxTarget::default())
        .expect("ptx emission");
    assert!(ptx.contains("st.shared.u32"), "{ptx}");
    assert!(ptx.contains("cvta.shared.u64"), "{ptx}");
    assert!(ptx.contains("ld.u32"), "select result must load generically:\n{ptx}");
    assert!(!ptx.contains("ld.shared.u32"), "{ptx}");
    assert!(ptx.contains("st.global.u32"), "{ptx}");
    ptxas_verify("shared_provenance_and_escape", &ptx);
}

/// Provenance follows block arguments: a shared pointer passed around a
/// branch keeps its raw form and its loads stay `.shared`-qualified.
#[test]
fn shared_provenance_through_block_args() {
    let mut ctx = Context::new();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "kernels".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();

    let byte_ty: pliron_ll::r#type::TypeHandle =
        IntegerType::get(&mut ctx, 8, Signedness::Unsigned).into();
    let tile_ty = ArrayType::get(&mut ctx, byte_ty, 256).into();
    let tile = GlobalOp::new(&mut ctx, "BUF".try_into().unwrap(), tile_ty);
    tile.set_attr_llvm_global_linkage(&ctx, LinkageAttr::ExternalLinkage);
    pliron_ll::ll::set_global_data(
        &ctx,
        &tile,
        pliron_ll::ll::DataAttr { bytes: vec![0; 256], align: 4, mutable: true, relocs: vec![] },
    );
    pliron_ll::ll::set_global_section(&mut ctx, &tile, ".shared");
    tile.get_operation().insert_at_back(body, &ctx);

    let i32_ty = i32_ty(&mut ctx);
    let ptr = ptr_ty(&mut ctx);
    let func = new_kernel(&mut ctx, body, "bbarg", vec![ptr]);
    let entry = func.get_entry_block(&ctx).unwrap();
    let out = entry.deref(&ctx).arguments().next().unwrap();
    let region = func.get_region(&ctx).unwrap();
    let tail = BasicBlock::new(&mut ctx, Some("tail".try_into().unwrap()), vec![ptr]);
    tail.insert_at_back(region, &ctx);

    let tile_addr = AddressOfOp::new(&mut ctx, "BUF".try_into().unwrap(), 0);
    tile_addr.get_operation().insert_at_back(entry, &ctx);
    let t = tile_addr.get_result(&ctx);
    BrOp::new(&mut ctx, tail, vec![t])
        .get_operation()
        .insert_at_back(entry, &ctx);

    let x = tail.deref(&ctx).arguments().next().unwrap();
    let v = LoadOp::new(&mut ctx, x, i32_ty);
    v.get_operation().insert_at_back(tail, &ctx);
    let v = v.get_result(&ctx);
    StoreOp::new(&mut ctx, v, out).get_operation().insert_at_back(tail, &ctx);
    ReturnOp::new(&mut ctx, None).get_operation().insert_at_back(tail, &ctx);

    let ptx = nvptx::write_ptx_from_ir(&ctx, module.get_operation(), &PtxTarget::default())
        .expect("ptx emission");
    assert!(ptx.contains("ld.shared.u32"), "{ptx}");
    assert!(ptx.contains("st.global.u32"), "{ptx}");
    assert!(!ptx.contains("cvta."), "{ptx}");
    ptxas_verify("shared_provenance_through_block_args", &ptx);
}

// ---------------------------------------------------------------------------
// Coverage: struct GEPs, .local allocas, device-function calls, immediates
// ---------------------------------------------------------------------------

use pliron_ll::dialects::llvm::{
    op_interfaces::CastOpWithNNegInterface as _,
    ops::{AllocaOp, MulOp},
    types::StructType,
};

/// Struct GEPs: field offsets follow the natural layout ({i32, f32, i64}
/// puts the i64 at offset 8; the whole struct strides at 16), and mixing a
/// dynamic element index with constant field indices folds the field
/// offset into a single trailing add.
#[test]
fn struct_gep_kernel() {
    let mut ctx = Context::new();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "kernels".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();

    let i32t = i32_ty(&mut ctx);
    let i64t = i64_ty(&mut ctx);
    let f32t: pliron_ll::r#type::TypeHandle = FP32Type::get(&mut ctx).into();
    let ptr = ptr_ty(&mut ctx);
    let node = StructType::get_named(
        &mut ctx,
        "node".try_into().unwrap(),
        Some(vec![i32t, f32t, i64t]),
    )
    .expect("fresh named struct")
    .into();

    let func = new_kernel(&mut ctx, body, "sgep", vec![ptr, ptr]);
    let entry = func.get_entry_block(&ctx).unwrap();
    let args: Vec<_> = entry.deref(&ctx).arguments().collect();
    let (nodes, out) = (args[0], args[1]);

    let tid = call_intrinsic(&mut ctx, entry, "llvm_nvvm_read_ptx_sreg_tid_x", Some(i32t), vec![]).unwrap();
    // &nodes[tid].2 — the i64 field at offset 8, elements striding at 16.
    let gep = GetElementPtrOp::new(
        &mut ctx,
        nodes,
        vec![GepIndex::Value(tid), GepIndex::Constant(2)],
        node,
    );
    gep.get_operation().insert_at_back(entry, &ctx);
    let field = gep.get_result(&ctx);
    let value = LoadOp::new(&mut ctx, field, i64t);
    value.get_operation().insert_at_back(entry, &ctx);
    let value = value.get_result(&ctx);

    let dst = GetElementPtrOp::new(&mut ctx, out, vec![GepIndex::Value(tid)], i64t);
    dst.get_operation().insert_at_back(entry, &ctx);
    let dst = dst.get_result(&ctx);
    StoreOp::new(&mut ctx, value, dst).get_operation().insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None).get_operation().insert_at_back(entry, &ctx);

    let ptx = nvptx::write_ptx_from_ir(&ctx, module.get_operation(), &PtxTarget::default())
        .expect("struct geps must lower");
    // Element stride 16 (shift by 4), field offset 8 folded into the
    // memory operand.
    assert!(ptx.contains("shl.b64 %rd3, %rd2, 4;"), "{ptx}");
    assert!(ptx.contains("ld.global.nc.u64 %rd5, [%rd4+8];"), "{ptx}");
    ptxas_verify("struct_gep_kernel", &ptx);
}

/// Allocas that survive mem2reg lower to `.local` arrays: declared in the
/// function header, addressed via `cvta.local.u64`, and accessed with the
/// generic forms.
#[test]
fn alloca_local_kernel() {
    let mut ctx = Context::new();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "kernels".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();

    let i64t = i64_ty(&mut ctx);
    let i32t = i32_ty(&mut ctx);
    let ptr = ptr_ty(&mut ctx);
    let func = new_kernel(&mut ctx, body, "loc", vec![ptr]);
    let entry = func.get_entry_block(&ctx).unwrap();
    let out = entry.deref(&ctx).arguments().next().unwrap();

    let four = i64_const(&mut ctx, entry, 4);
    let slot = AllocaOp::new(&mut ctx, i64t, four);
    slot.get_operation().insert_at_back(entry, &ctx);
    let slot_ptr = slot.get_result(&ctx);

    let tid = call_intrinsic(&mut ctx, entry, "llvm_nvvm_read_ptx_sreg_tid_x", Some(i32t), vec![]).unwrap();
    let tid64 = pliron_ll::dialects::llvm::ops::ZExtOp::new_with_nneg(&mut ctx, tid, i64t, false);
    tid64.get_operation().insert_at_back(entry, &ctx);
    let tid64 = tid64.get_result(&ctx);

    // slot[3] = tid; out[tid] = slot[3]
    let elt = GetElementPtrOp::new(&mut ctx, slot_ptr, vec![GepIndex::Constant(3)], i64t);
    elt.get_operation().insert_at_back(entry, &ctx);
    let elt = elt.get_result(&ctx);
    StoreOp::new(&mut ctx, tid64, elt).get_operation().insert_at_back(entry, &ctx);
    let back = LoadOp::new(&mut ctx, elt, i64t);
    back.get_operation().insert_at_back(entry, &ctx);
    let back = back.get_result(&ctx);
    let dst = GetElementPtrOp::new(&mut ctx, out, vec![GepIndex::Value(tid)], i64t);
    dst.get_operation().insert_at_back(entry, &ctx);
    let dst = dst.get_result(&ctx);
    StoreOp::new(&mut ctx, back, dst).get_operation().insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None).get_operation().insert_at_back(entry, &ctx);

    let ptx = nvptx::write_ptx_from_ir(&ctx, module.get_operation(), &PtxTarget::default())
        .expect("allocas must lower");
    assert!(ptx.contains(".local .align 8 .b8 __crabbit_local_loc_0[32];"), "{ptx}");
    assert!(ptx.contains("cvta.local.u64"), "{ptx}");
    ptxas_verify("alloca_local_kernel", &ptx);
}

/// A surviving internal function is emitted as a device `.func` with the
/// `.param` ABI, and the kernel's call becomes a `call.uni` sequence.
#[test]
fn device_function_call_kernel() {
    let mut ctx = Context::new();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "kernels".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();

    let i32t = i32_ty(&mut ctx);
    let ptr = ptr_ty(&mut ctx);

    // internal fn scale_add(a: i32, b: i32) -> i32 { a * b + 7 }
    let helper_ty = FuncType::get(&mut ctx, i32t, vec![i32t, i32t], false);
    let helper = FuncOp::new(&mut ctx, "scale_add".try_into().unwrap(), helper_ty);
    helper.set_attr_llvm_function_linkage(&ctx, LinkageAttr::InternalLinkage);
    let helper_entry = helper.get_or_create_entry_block(&mut ctx);
    helper.get_operation().insert_at_back(body, &ctx);
    let hargs: Vec<_> = helper_entry.deref(&ctx).arguments().collect();
    let prod = MulOp::new_with_overflow_flag(&mut ctx, hargs[0], hargs[1], Default::default());
    prod.get_operation().insert_at_back(helper_entry, &ctx);
    let prod = prod.get_result(&ctx);
    let i32_int_ty = IntegerType::get(&mut ctx, 32, Signedness::Signless);
    let seven = ConstantOp::new(
        &mut ctx,
        Box::new(IntegerAttr::new(
            i32_int_ty,
            APInt::from_u64(7, NonZero::new(32).unwrap()),
        )),
    );
    seven.get_operation().insert_at_back(helper_entry, &ctx);
    let seven = seven.get_result(&ctx);
    let sum = AddOp::new_with_overflow_flag(&mut ctx, prod, seven, Default::default());
    sum.get_operation().insert_at_back(helper_entry, &ctx);
    let sum = sum.get_result(&ctx);
    ReturnOp::new(&mut ctx, Some(sum)).get_operation().insert_at_back(helper_entry, &ctx);

    // kernel: out[tid] = scale_add(out[tid], tid)
    let func = new_kernel(&mut ctx, body, "devcall", vec![ptr]);
    let entry = func.get_entry_block(&ctx).unwrap();
    let out = entry.deref(&ctx).arguments().next().unwrap();
    let tid = call_intrinsic(&mut ctx, entry, "llvm_nvvm_read_ptx_sreg_tid_x", Some(i32t), vec![]).unwrap();
    let slot = GetElementPtrOp::new(&mut ctx, out, vec![GepIndex::Value(tid)], i32t);
    slot.get_operation().insert_at_back(entry, &ctx);
    let slot = slot.get_result(&ctx);
    let cur = LoadOp::new(&mut ctx, slot, i32t);
    cur.get_operation().insert_at_back(entry, &ctx);
    let cur = cur.get_result(&ctx);
    let helper_call_ty = FuncType::get(&mut ctx, i32t, vec![i32t, i32t], false);
    let call = CallOp::new(
        &mut ctx,
        pliron::builtin::op_interfaces::CallOpCallable::Direct("scale_add".try_into().unwrap()),
        helper_call_ty,
        vec![cur, tid],
    );
    call.get_operation().insert_at_back(entry, &ctx);
    let scaled = call.get_operation().deref(&ctx).get_result(0);
    StoreOp::new(&mut ctx, scaled, slot).get_operation().insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None).get_operation().insert_at_back(entry, &ctx);

    let ptx = nvptx::write_ptx_from_ir(&ctx, module.get_operation(), &PtxTarget::default())
        .expect("device calls must lower");
    // Forward declaration + definition + the call ABI.
    assert!(ptx.contains(".func (.param .b32 scale_add_ret) scale_add (.param .b32 scale_add_param_0, .param .b32 scale_add_param_1);"), "{ptx}");
    assert!(ptx.contains("call.uni (c0_ret), scale_add, (c0_param_0, c0_param_1);"), "{ptx}");
    assert!(ptx.contains("st.param.b32"), "{ptx}");
    assert!(ptx.contains("ld.param.b32"), "{ptx}");
    // The helper body: immediate 7 folds into the add.
    assert!(ptx.contains("st.param.b32 [scale_add_ret]"), "{ptx}");
    assert!(!ptx.contains(".visible .entry scale_add"), "device fns are not entries:\n{ptx}");
    ptxas_verify("device_function_call_kernel", &ptx);
}

/// Integer constants fold into using instructions as immediates: no
/// `mov` materialization, no `cvt` for constant shift amounts, and the
/// GEP scale/offset arithmetic stays folded.
#[test]
fn integer_immediates_fold() {
    let mut ctx = Context::new();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "kernels".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();

    let i64t = i64_ty(&mut ctx);
    let ptr = ptr_ty(&mut ctx);
    let func = new_kernel(&mut ctx, body, "imm", vec![ptr, i64t]);
    let entry = func.get_entry_block(&ctx).unwrap();
    let args: Vec<_> = entry.deref(&ctx).arguments().collect();
    let (buf, n) = (args[0], args[1]);

    // v = (n + 5) << 2; buf[3] = v
    let five = i64_const(&mut ctx, entry, 5);
    let two = i64_const(&mut ctx, entry, 2);
    let sum = AddOp::new_with_overflow_flag(&mut ctx, n, five, Default::default());
    sum.get_operation().insert_at_back(entry, &ctx);
    let sum = sum.get_result(&ctx);
    let shifted = pliron_ll::dialects::llvm::ops::ShlOp::new_with_overflow_flag(
        &mut ctx,
        sum,
        two,
        Default::default(),
    );
    shifted.get_operation().insert_at_back(entry, &ctx);
    let shifted = shifted.get_result(&ctx);
    let slot = GetElementPtrOp::new(&mut ctx, buf, vec![GepIndex::Constant(3)], i64t);
    slot.get_operation().insert_at_back(entry, &ctx);
    let slot = slot.get_result(&ctx);
    StoreOp::new(&mut ctx, shifted, slot).get_operation().insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None).get_operation().insert_at_back(entry, &ctx);

    let ptx = nvptx::write_ptx_from_ir(&ctx, module.get_operation(), &PtxTarget::default())
        .expect("ptx emission");
    assert!(ptx.contains("add.s64 %rd2, %rd1, 5;"), "{ptx}");
    assert!(ptx.contains("shl.b64 %rd3, %rd2, 2;"), "{ptx}");
    // buf[3]: the byte offset 24 folds into the memory operand itself.
    assert!(ptx.contains("st.global.u64 [%rd0+24]"), "{ptx}");
    assert!(!ptx.contains("mov.u64 %rd"), "constants must not materialize:\n{ptx}");
    assert!(!ptx.contains("cvt.u32.u64"), "constant shift amounts need no cvt:\n{ptx}");
    ptxas_verify("integer_immediates_fold", &ptx);
}

// ---------------------------------------------------------------------------
// Address folding ([reg+imm] memory operands) and vectorized shared loads
// ---------------------------------------------------------------------------

fn i32_const(
    ctx: &mut Context,
    block: pliron_ll::context::Ptr<BasicBlock>,
    value: u32,
) -> Value {
    let ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let constant = ConstantOp::new(
        ctx,
        Box::new(IntegerAttr::new(
            ty,
            APInt::from_u64(value as u64, NonZero::new(32).unwrap()),
        )),
    );
    constant.get_operation().insert_at_back(block, ctx);
    constant.get_result(ctx)
}

/// `a[n+5]` and `a[n-1]` must share ONE base register (`mad n*8 + a`) with
/// the constants folded into the memory operands (`+40`, `+-8`), and the
/// index arithmetic must vanish from the PTX entirely. `out[1]` folds off
/// the raw parameter. ptxas is the oracle that `[reg+imm]` (negative too)
/// assembles in the `.global` space.
#[test]
fn memory_offsets_fold_and_reassociate() {
    let mut ctx = Context::new();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "kernels".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();

    let i64t = i64_ty(&mut ctx);
    let ptr = ptr_ty(&mut ctx);
    let func = new_kernel(&mut ctx, body, "fold", vec![ptr, ptr, i64t]);
    let entry = func.get_entry_block(&ctx).unwrap();
    let args: Vec<_> = entry.deref(&ctx).arguments().collect();
    let (a, out, n) = (args[0], args[1], args[2]);

    let five = i64_const(&mut ctx, entry, 5);
    let one = i64_const(&mut ctx, entry, 1);
    let load_at = |ctx: &mut Context, index: Value| -> Value {
        let gep = GetElementPtrOp::new(ctx, a, vec![GepIndex::Value(index)], i64t);
        gep.get_operation().insert_at_back(entry, ctx);
        let load = LoadOp::new(ctx, gep.get_result(ctx), i64t);
        load.get_operation().insert_at_back(entry, ctx);
        load.get_result(ctx)
    };
    let plus5 = AddOp::new_with_overflow_flag(&mut ctx, n, five, Default::default());
    plus5.get_operation().insert_at_back(entry, &ctx);
    let plus5 = plus5.get_result(&ctx);
    let hi = load_at(&mut ctx, plus5);
    let minus1 = pliron_ll::dialects::llvm::ops::SubOp::new_with_overflow_flag(
        &mut ctx,
        n,
        one,
        Default::default(),
    );
    minus1.get_operation().insert_at_back(entry, &ctx);
    let minus1 = minus1.get_result(&ctx);
    let lo = load_at(&mut ctx, minus1);

    let sum = AddOp::new_with_overflow_flag(&mut ctx, hi, lo, Default::default());
    sum.get_operation().insert_at_back(entry, &ctx);
    let dst = GetElementPtrOp::new(&mut ctx, out, vec![GepIndex::Constant(1)], i64t);
    dst.get_operation().insert_at_back(entry, &ctx);
    let sum = sum.get_result(&ctx);
    let dst = dst.get_result(&ctx);
    StoreOp::new(&mut ctx, sum, dst).get_operation().insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None).get_operation().insert_at_back(entry, &ctx);

    let ptx = nvptx::write_ptx_from_ir(&ctx, module.get_operation(), &PtxTarget::default())
        .expect("ptx emission");
    // One shared base register: n*8 + a, materialized once as shift+add.
    assert!(ptx.contains("shl.b64 %rd3, %rd2, 3;"), "{ptx}");
    assert!(ptx.contains("add.s64 %rd4, %rd0, %rd3;"), "{ptx}");
    assert_eq!(
        ptx.matches("shl.b64").count(),
        1,
        "both loads must share one base register:\n{ptx}"
    );
    assert!(ptx.contains("ld.global.nc.u64 %rd5, [%rd4+40];"), "{ptx}");
    assert!(ptx.contains("ld.global.nc.u64 %rd6, [%rd4+-8];"), "{ptx}");
    assert!(ptx.contains("st.global.u64 [%rd1+8]"), "{ptx}");
    // The peeled index arithmetic feeds nothing else: it must not emit.
    assert!(!ptx.contains("sub.s64"), "{ptx}");
    assert!(!ptx.contains("mad.lo.s64"), "{ptx}");
    ptxas_verify("memory_offsets_fold_and_reassociate", &ptx);
}

/// Four `.shared` loads at consecutive folded offsets group into ONE
/// `ld.shared.v4.u32`, and the shared global's declaration alignment is
/// raised to 16 to make the group provably aligned. The dynamic part
/// (`(tid.x << 2) * 4` bytes = 16-byte stride) keeps the proof honest, and
/// the wrap-free peel through the 32-bit shift needs `%tid.x`'s hardware
/// range. ptxas assembles the vector form.
#[test]
fn shared_loads_vectorize_v4() {
    let mut ctx = Context::new();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "kernels".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();

    let byte_ty: pliron_ll::r#type::TypeHandle =
        IntegerType::get(&mut ctx, 8, Signedness::Unsigned).into();
    let tile_ty = ArrayType::get(&mut ctx, byte_ty, 1024).into();
    let tile = GlobalOp::new(&mut ctx, "TILEV".try_into().unwrap(), tile_ty);
    tile.set_attr_llvm_global_linkage(&ctx, LinkageAttr::ExternalLinkage);
    pliron_ll::ll::set_global_data(
        &ctx,
        &tile,
        pliron_ll::ll::DataAttr { bytes: vec![0; 1024], align: 4, mutable: true, relocs: vec![] },
    );
    pliron_ll::ll::set_global_section(&mut ctx, &tile, ".shared");
    tile.get_operation().insert_at_back(body, &ctx);

    let i32t = i32_ty(&mut ctx);
    let i64t = i64_ty(&mut ctx);
    let ptr = ptr_ty(&mut ctx);
    let func = new_kernel(&mut ctx, body, "vec4", vec![ptr]);
    let entry = func.get_entry_block(&ctx).unwrap();
    let out = entry.deref(&ctx).arguments().next().unwrap();

    let tid =
        call_intrinsic(&mut ctx, entry, "llvm_nvvm_read_ptx_sreg_tid_x", Some(i32t), vec![])
            .unwrap();
    let tile_addr = AddressOfOp::new(&mut ctx, "TILEV".try_into().unwrap(), 0);
    tile_addr.get_operation().insert_at_back(entry, &ctx);
    let t = tile_addr.get_result(&ctx);

    let two = i32_const(&mut ctx, entry, 2);
    let row = pliron_ll::dialects::llvm::ops::ShlOp::new_with_overflow_flag(
        &mut ctx,
        tid,
        two,
        Default::default(),
    );
    row.get_operation().insert_at_back(entry, &ctx);
    let row = row.get_result(&ctx);

    let mut loaded = Vec::new();
    for lane in 0..4u32 {
        let k = i32_const(&mut ctx, entry, lane);
        let idx32 = AddOp::new_with_overflow_flag(&mut ctx, row, k, Default::default());
        idx32.get_operation().insert_at_back(entry, &ctx);
        let idx32 = idx32.get_result(&ctx);
        let idx = pliron_ll::dialects::llvm::ops::ZExtOp::new(&mut ctx, idx32, i64t);
        idx.get_operation().insert_at_back(entry, &ctx);
        let idx = idx.get_result(&ctx);
        let gep = GetElementPtrOp::new(&mut ctx, t, vec![GepIndex::Value(idx)], i32t);
        gep.get_operation().insert_at_back(entry, &ctx);
        let slot = gep.get_result(&ctx);
        let load = LoadOp::new(&mut ctx, slot, i32t);
        load.get_operation().insert_at_back(entry, &ctx);
        loaded.push(load.get_result(&ctx));
    }
    let mut sum = loaded[0];
    for &v in &loaded[1..] {
        let add = AddOp::new_with_overflow_flag(&mut ctx, sum, v, Default::default());
        add.get_operation().insert_at_back(entry, &ctx);
        sum = add.get_result(&ctx);
    }
    let dst = GetElementPtrOp::new(&mut ctx, out, vec![GepIndex::Value(tid)], i32t);
    dst.get_operation().insert_at_back(entry, &ctx);
    let dst = dst.get_result(&ctx);
    StoreOp::new(&mut ctx, sum, dst).get_operation().insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None).get_operation().insert_at_back(entry, &ctx);

    let ptx = nvptx::write_ptx_from_ir(&ctx, module.get_operation(), &PtxTarget::default())
        .expect("ptx emission");
    assert!(ptx.contains(".shared .align 16 .b8 TILEV[1024];"), "{ptx}");
    assert!(ptx.contains("ld.shared.v4.u32 {"), "{ptx}");
    assert!(
        !ptx.contains("ld.shared.u32 "),
        "all four loads must be in the vector group:\n{ptx}"
    );
    ptxas_verify("shared_loads_vectorize_v4", &ptx);
}

/// Global loads never vectorize — a kernel parameter is a raw pointer with
/// only element alignment promised — but their offsets still fold.
#[test]
fn global_loads_fold_but_do_not_vectorize() {
    let mut ctx = Context::new();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "kernels".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();

    let i32t = i32_ty(&mut ctx);
    let ptr = ptr_ty(&mut ctx);
    let func = new_kernel(&mut ctx, body, "novec", vec![ptr, ptr]);
    let entry = func.get_entry_block(&ctx).unwrap();
    let args: Vec<_> = entry.deref(&ctx).arguments().collect();
    let (a, out) = (args[0], args[1]);

    let mut loaded = Vec::new();
    for lane in 0..4i32 {
        let gep =
            GetElementPtrOp::new(&mut ctx, a, vec![GepIndex::Constant(lane as u32)], i32t);
        gep.get_operation().insert_at_back(entry, &ctx);
        let slot = gep.get_result(&ctx);
        let load = LoadOp::new(&mut ctx, slot, i32t);
        load.get_operation().insert_at_back(entry, &ctx);
        loaded.push(load.get_result(&ctx));
    }
    let mut sum = loaded[0];
    for &v in &loaded[1..] {
        let add = AddOp::new_with_overflow_flag(&mut ctx, sum, v, Default::default());
        add.get_operation().insert_at_back(entry, &ctx);
        sum = add.get_result(&ctx);
    }
    StoreOp::new(&mut ctx, sum, out).get_operation().insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None).get_operation().insert_at_back(entry, &ctx);

    let ptx = nvptx::write_ptx_from_ir(&ctx, module.get_operation(), &PtxTarget::default())
        .expect("ptx emission");
    assert!(!ptx.contains(".v4"), "global loads must stay scalar:\n{ptx}");
    assert!(ptx.contains("ld.global.nc.u32 %r0, [%rd0];"), "{ptx}");
    assert!(ptx.contains("ld.global.nc.u32 %r1, [%rd0+4];"), "{ptx}");
    assert!(ptx.contains("ld.global.nc.u32 %r3, [%rd0+12];"), "{ptx}");
    ptxas_verify("global_loads_fold_but_do_not_vectorize", &ptx);
}

/// Reassociating `zext(x +₃₂ c)` demands a proof that the 32-bit add
/// cannot wrap; an unbounded kernel parameter has none, so the constant
/// must NOT migrate into the memory operand.
#[test]
fn zext_peel_requires_range_proof() {
    let mut ctx = Context::new();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "kernels".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();

    let i32t = i32_ty(&mut ctx);
    let i64t = i64_ty(&mut ctx);
    let ptr = ptr_ty(&mut ctx);
    let func = new_kernel(&mut ctx, body, "nopeel", vec![ptr, i32t]);
    let entry = func.get_entry_block(&ctx).unwrap();
    let args: Vec<_> = entry.deref(&ctx).arguments().collect();
    let (a, x) = (args[0], args[1]);

    let one = i32_const(&mut ctx, entry, 1);
    let idx32 = AddOp::new_with_overflow_flag(&mut ctx, x, one, Default::default());
    idx32.get_operation().insert_at_back(entry, &ctx);
    let idx32 = idx32.get_result(&ctx);
    let idx = pliron_ll::dialects::llvm::ops::ZExtOp::new(&mut ctx, idx32, i64t);
    idx.get_operation().insert_at_back(entry, &ctx);
    let idx = idx.get_result(&ctx);
    let gep = GetElementPtrOp::new(&mut ctx, a, vec![GepIndex::Value(idx)], i64t);
    gep.get_operation().insert_at_back(entry, &ctx);
    let slot = gep.get_result(&ctx);
    let load = LoadOp::new(&mut ctx, slot, i64t);
    load.get_operation().insert_at_back(entry, &ctx);
    let v = load.get_result(&ctx);
    StoreOp::new(&mut ctx, v, a).get_operation().insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None).get_operation().insert_at_back(entry, &ctx);

    let ptx = nvptx::write_ptx_from_ir(&ctx, module.get_operation(), &PtxTarget::default())
        .expect("ptx emission");
    assert!(
        !ptx.contains("+8]"),
        "an unprovable 32-bit add must not peel:\n{ptx}"
    );
    // The add stays in 32 bits, zext'd and folded as the whole term.
    assert!(ptx.contains("add.s32"), "{ptx}");
    ptxas_verify("zext_peel_requires_range_proof", &ptx);
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

/// GPU leg of the backward-attribution design (docs/PROFILE-FEEDBACK-
/// BACKWARD.md): the PTX linemap must not perturb the emitted PTX, and it
/// must cover every line of every entry — each line either lowers stamped
/// source ops or carries a named synthetic root.
#[test]
fn linemap_covers_every_line_and_never_changes_ptx() {
    use pliron_ll::conversion::pass::{AnalysisManager, Pass};

    let mut ctx = Context::new();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "kernels".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();

    let i64_ty = i64_ty(&mut ctx);
    let ptr = ptr_ty(&mut ctx);
    let func = new_kernel(&mut ctx, body, "lm", vec![ptr, ptr]);
    let entry = func.get_entry_block(&ctx).unwrap();
    let args: Vec<_> = entry.deref(&ctx).arguments().collect();
    let (a, out) = (args[0], args[1]);

    let load = LoadOp::new(&mut ctx, a, i64_ty);
    load.get_operation().insert_at_back(entry, &ctx);
    let v = load.get_result(&ctx);
    let sum = AddOp::new_with_overflow_flag(&mut ctx, v, v, Default::default());
    sum.get_operation().insert_at_back(entry, &ctx);
    let sum = sum.get_result(&ctx);
    StoreOp::new(&mut ctx, sum, out)
        .get_operation()
        .insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);

    // Stamp source ids the way the kernel pipeline's mid-end head does.
    unsafe { std::env::set_var("CRABBIT_PROFILE_MAP", "1") };
    pliron_ll::passes::llvm::op_ids::LlvmOpIdPass
        .run(module.get_operation(), &mut ctx, &mut AnalysisManager::default())
        .expect("op-id stamping");

    let plain = nvptx::write_ptx_from_ir(&ctx, module.get_operation(), &PtxTarget::default())
        .expect("plain emission");
    let (ptx, linemap) = nvptx::write_ptx_with_forced_linemap(
        &ctx,
        module.get_operation(),
        &PtxTarget::default(),
    )
    .expect("linemap emission");
    unsafe { std::env::remove_var("CRABBIT_PROFILE_MAP") };

    assert_eq!(plain, ptx, "linemap recording must never change the PTX");

    let map: serde_json::Value = serde_json::from_str(&linemap).expect("valid JSON");
    let entry_map = map.get("lm").expect("entry present");
    let start = entry_map["start"].as_u64().unwrap() as usize;
    let end = entry_map["end"].as_u64().unwrap() as usize;
    let lines = entry_map["lines"].as_object().unwrap();
    let total_lines = ptx.matches('\n').count();
    assert!(start >= 1 && end <= total_lines && start < end, "{start}..{end} vs {total_lines}");
    let mut op_lines = 0;
    for lineno in start..=end {
        let value = lines
            .get(&lineno.to_string())
            .unwrap_or_else(|| panic!("line {lineno} unmapped in {linemap}"));
        match value {
            serde_json::Value::Array(ids) => {
                assert!(!ids.is_empty());
                op_lines += 1;
            }
            serde_json::Value::String(root) => {
                assert!(root.starts_with("ptx:"), "unexpected root {root}");
            }
            other => panic!("unexpected linemap value {other}"),
        }
    }
    assert!(op_lines >= 4, "expected ld/add/st/ret lines with op ids, got {op_lines}");
    assert!(entry_map["midend"].is_null());
}

// ---------------------------------------------------------------------------
// Warp primitives (shfl.sync / vote.sync / bar.warp.sync), dynamic shared
// memory, and by-value aggregate kernel parameters
// ---------------------------------------------------------------------------

/// All four shuffle modes (i32 and f32 via the same `.b32` instruction),
/// `bar.warp.sync`, and the vote family, with the NVVM operand order
/// `(membermask, a, b, c)` re-ordered to PTX's `d, a, b, c, membermask`.
/// Constant lane counts, packed-c words and membermasks fold as immediates.
#[test]
fn warp_shuffle_and_vote_kernel() {
    let mut ctx = Context::new();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "kernels".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();

    let f32t: pliron_ll::r#type::TypeHandle = FP32Type::get(&ctx).into();
    let i32t = i32_ty(&mut ctx);
    let ptr = ptr_ty(&mut ctx);
    let func = new_kernel(&mut ctx, body, "warp", vec![ptr, ptr, ptr]);
    let entry = func.get_entry_block(&ctx).unwrap();
    let args: Vec<_> = entry.deref(&ctx).arguments().collect();
    let (x, out, flags) = (args[0], args[1], args[2]);

    let tid = call_intrinsic(&mut ctx, entry, "llvm_nvvm_read_ptx_sreg_tid_x", Some(i32t), vec![]).unwrap();
    let gep = GetElementPtrOp::new(&mut ctx, x, vec![GepIndex::Value(tid)], f32t);
    gep.get_operation().insert_at_back(entry, &ctx);
    let gep = gep.get_result(&ctx);
    let load = LoadOp::new(&mut ctx, gep, f32t);
    load.get_operation().insert_at_back(entry, &ctx);
    let v = load.get_result(&ctx);

    let mask = i32_const(&mut ctx, entry, 0xffff_ffff);
    let clamp_down = i32_const(&mut ctx, entry, 0x1f);
    let zero_c = i32_const(&mut ctx, entry, 0);
    let sixteen = i32_const(&mut ctx, entry, 16);
    let one = i32_const(&mut ctx, entry, 1);
    let five = i32_const(&mut ctx, entry, 5);

    // f32 down + up, i32 bfly + idx: every mode once.
    let down = call_intrinsic(&mut ctx, entry, "llvm_nvvm_shfl_sync_down_f32", Some(f32t),
        vec![mask, v, sixteen, clamp_down]).unwrap();
    let up = call_intrinsic(&mut ctx, entry, "llvm_nvvm_shfl_sync_up_f32", Some(f32t),
        vec![mask, down, one, zero_c]).unwrap();
    let bfly = call_intrinsic(&mut ctx, entry, "llvm_nvvm_shfl_sync_bfly_i32", Some(i32t),
        vec![mask, tid, one, clamp_down]).unwrap();
    let idx = call_intrinsic(&mut ctx, entry, "llvm_nvvm_shfl_sync_idx_i32", Some(i32t),
        vec![mask, bfly, five, clamp_down]).unwrap();

    call_intrinsic(&mut ctx, entry, "llvm_nvvm_bar_warp_sync", None, vec![mask]);

    // vote: ballot into flags[tid], all/any zext'd into flags via arithmetic.
    let pred = ICmpOp::new(&mut ctx, ICmpPredicateAttr::ULT, tid, sixteen);
    pred.get_operation().insert_at_back(entry, &ctx);
    let pred = pred.get_result(&ctx);
    let ballot = call_intrinsic(&mut ctx, entry, "llvm_nvvm_vote_ballot_sync", Some(i32t),
        vec![mask, pred]).unwrap();
    let i1t: pliron_ll::r#type::TypeHandle =
        IntegerType::get(&ctx, 1, Signedness::Signless).into();
    let all = call_intrinsic(&mut ctx, entry, "llvm_nvvm_vote_all_sync", Some(i1t),
        vec![mask, pred]).unwrap();
    let any = call_intrinsic(&mut ctx, entry, "llvm_nvvm_vote_any_sync", Some(i1t),
        vec![mask, pred]).unwrap();
    let all32 = pliron_ll::dialects::llvm::ops::ZExtOp::new(&mut ctx, all, i32t);
    all32.get_operation().insert_at_back(entry, &ctx);
    let any32 = pliron_ll::dialects::llvm::ops::ZExtOp::new(&mut ctx, any, i32t);
    any32.get_operation().insert_at_back(entry, &ctx);
    let (all32, any32) = (all32.get_result(&ctx), any32.get_result(&ctx));

    // Combine and store so nothing is dead.
    let sum = FAddOp::new(&mut ctx, up, down);
    sum.get_operation().insert_at_back(entry, &ctx);
    let dst = GetElementPtrOp::new(&mut ctx, out, vec![GepIndex::Value(tid)], f32t);
    dst.get_operation().insert_at_back(entry, &ctx);
    let (sum, dst) = (sum.get_result(&ctx), dst.get_result(&ctx));
    StoreOp::new(&mut ctx, sum, dst).get_operation().insert_at_back(entry, &ctx);
    let mixed = AddOp::new_with_overflow_flag(&mut ctx, idx, ballot, Default::default());
    mixed.get_operation().insert_at_back(entry, &ctx);
    let mixed = mixed.get_result(&ctx);
    let mixed2 = AddOp::new_with_overflow_flag(&mut ctx, mixed, all32, Default::default());
    mixed2.get_operation().insert_at_back(entry, &ctx);
    let mixed2 = mixed2.get_result(&ctx);
    let mixed3 = AddOp::new_with_overflow_flag(&mut ctx, mixed2, any32, Default::default());
    mixed3.get_operation().insert_at_back(entry, &ctx);
    let mixed3 = mixed3.get_result(&ctx);
    let fdst = GetElementPtrOp::new(&mut ctx, flags, vec![GepIndex::Value(tid)], i32t);
    fdst.get_operation().insert_at_back(entry, &ctx);
    let fdst = fdst.get_result(&ctx);
    StoreOp::new(&mut ctx, mixed3, fdst).get_operation().insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None).get_operation().insert_at_back(entry, &ctx);

    let ptx = nvptx::write_ptx_from_ir(&ctx, module.get_operation(), &PtxTarget::default())
        .expect("ptx emission");
    for needle in [
        // f32 values ride the untyped .b32 shuffle in .f32 registers.
        "shfl.sync.down.b32 %f",
        ", 16, 31, 4294967295;",
        "shfl.sync.up.b32 %f",
        ", 1, 0, 4294967295;",
        "shfl.sync.bfly.b32 %r",
        "shfl.sync.idx.b32 %r",
        ", 5, 31, 4294967295;",
        "bar.warp.sync 4294967295;",
        "vote.sync.ballot.b32 %r",
        "vote.sync.all.pred %p",
        "vote.sync.any.pred %p",
    ] {
        assert!(ptx.contains(needle), "missing `{needle}` in:\n{ptx}");
    }
    ptxas_verify("warp_shuffle_and_vote_kernel", &ptx);
}

/// The dynamic shared-memory window: `crabbit.dyn.shared.base` declares
/// ONE module-level `.extern .shared` array (size = the launch's
/// `sharedMemBytes`), and its address is a proven-`.shared` source, so
/// round-trips through it stay `.shared`-qualified with no `cvta`.
#[test]
fn dynamic_shared_memory_kernel() {
    let mut ctx = Context::new();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "kernels".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();

    let f32t: pliron_ll::r#type::TypeHandle = FP32Type::get(&ctx).into();
    let i32t = i32_ty(&mut ctx);
    let ptr = ptr_ty(&mut ctx);
    let func = new_kernel(&mut ctx, body, "dsm", vec![ptr, ptr]);
    let entry = func.get_entry_block(&ctx).unwrap();
    let args: Vec<_> = entry.deref(&ctx).arguments().collect();
    let (x, out) = (args[0], args[1]);

    let tid = call_intrinsic(&mut ctx, entry, "llvm_nvvm_read_ptx_sreg_tid_x", Some(i32t), vec![]).unwrap();
    let smem = call_intrinsic(&mut ctx, entry, "crabbit_dyn_shared_base", Some(ptr), vec![]).unwrap();

    let src = GetElementPtrOp::new(&mut ctx, x, vec![GepIndex::Value(tid)], f32t);
    src.get_operation().insert_at_back(entry, &ctx);
    let src = src.get_result(&ctx);
    let load = LoadOp::new(&mut ctx, src, f32t);
    load.get_operation().insert_at_back(entry, &ctx);
    let v = load.get_result(&ctx);

    let slot = GetElementPtrOp::new(&mut ctx, smem, vec![GepIndex::Value(tid)], f32t);
    slot.get_operation().insert_at_back(entry, &ctx);
    let slot = slot.get_result(&ctx);
    StoreOp::new(&mut ctx, v, slot).get_operation().insert_at_back(entry, &ctx);
    call_intrinsic(&mut ctx, entry, "llvm_nvvm_barrier0", None, vec![]);
    let back = LoadOp::new(&mut ctx, slot, f32t);
    back.get_operation().insert_at_back(entry, &ctx);
    let back = back.get_result(&ctx);

    let dst = GetElementPtrOp::new(&mut ctx, out, vec![GepIndex::Value(tid)], f32t);
    dst.get_operation().insert_at_back(entry, &ctx);
    let dst = dst.get_result(&ctx);
    StoreOp::new(&mut ctx, back, dst).get_operation().insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None).get_operation().insert_at_back(entry, &ctx);

    let ptx = nvptx::write_ptx_from_ir(&ctx, module.get_operation(), &PtxTarget::default())
        .expect("ptx emission");
    assert!(ptx.contains(".extern .shared .align 16 .b8 __crabbit_dyn_shared[];"), "{ptx}");
    assert!(ptx.contains("mov.u64 %rd2, __crabbit_dyn_shared;"), "{ptx}");
    assert!(ptx.contains("st.shared.f32"), "{ptx}");
    assert!(ptx.contains("ld.shared.f32"), "{ptx}");
    assert!(!ptx.contains("cvta."), "the window is fully shared-proven:\n{ptx}");
    ptxas_verify("dynamic_shared_memory_kernel", &ptx);
}

/// A by-value aggregate kernel parameter becomes one `.param .align A .b8
/// name[bytes]` in the natural layout, its leaves loaded with `ld.param`
/// at the same field offsets struct GEPs use ({i32, f32, i64}: 0, 4, 8;
/// size 16, align 8 — what a host launch passing the C struct provides).
#[test]
fn struct_param_by_value_kernel() {
    let mut ctx = Context::new();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "kernels".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();

    let i32t = i32_ty(&mut ctx);
    let i64t = i64_ty(&mut ctx);
    let f32t: pliron_ll::r#type::TypeHandle = FP32Type::get(&ctx).into();
    let ptr = ptr_ty(&mut ctx);
    let cfg = StructType::get_named(
        &ctx,
        "cfg".try_into().unwrap(),
        Some(vec![i32t, f32t, i64t]),
    )
    .expect("fresh named struct")
    .into();

    let func = new_kernel(&mut ctx, body, "sparam", vec![cfg, ptr, ptr]);
    let entry = func.get_entry_block(&ctx).unwrap();
    let args: Vec<_> = entry.deref(&ctx).arguments().collect();
    let (w, iout, fout) = (args[0], args[1], args[2]);

    let tid = call_intrinsic(&mut ctx, entry, "llvm_nvvm_read_ptx_sreg_tid_x", Some(i32t), vec![]).unwrap();
    let a = pliron_ll::dialects::llvm::ops::ExtractValueOp::new(&mut ctx, w, vec![0]).unwrap();
    a.get_operation().insert_at_back(entry, &ctx);
    let b = pliron_ll::dialects::llvm::ops::ExtractValueOp::new(&mut ctx, w, vec![1]).unwrap();
    b.get_operation().insert_at_back(entry, &ctx);
    let c = pliron_ll::dialects::llvm::ops::ExtractValueOp::new(&mut ctx, w, vec![2]).unwrap();
    c.get_operation().insert_at_back(entry, &ctx);
    let (a, b, c) = (a.get_result(&ctx), b.get_result(&ctx), c.get_result(&ctx));

    // iout[tid] = a; iout[tid+…] via i64 field; fout[tid] = b.
    let dst_a = GetElementPtrOp::new(&mut ctx, iout, vec![GepIndex::Value(tid)], i64t);
    dst_a.get_operation().insert_at_back(entry, &ctx);
    let dst_a = dst_a.get_result(&ctx);
    let a64 = pliron_ll::dialects::llvm::ops::ZExtOp::new(&mut ctx, a, i64t);
    a64.get_operation().insert_at_back(entry, &ctx);
    let a64 = a64.get_result(&ctx);
    let sum = AddOp::new_with_overflow_flag(&mut ctx, a64, c, Default::default());
    sum.get_operation().insert_at_back(entry, &ctx);
    let sum = sum.get_result(&ctx);
    StoreOp::new(&mut ctx, sum, dst_a).get_operation().insert_at_back(entry, &ctx);
    let dst_b = GetElementPtrOp::new(&mut ctx, fout, vec![GepIndex::Value(tid)], f32t);
    dst_b.get_operation().insert_at_back(entry, &ctx);
    let dst_b = dst_b.get_result(&ctx);
    StoreOp::new(&mut ctx, b, dst_b).get_operation().insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None).get_operation().insert_at_back(entry, &ctx);

    let ptx = nvptx::write_ptx_from_ir(&ctx, module.get_operation(), &PtxTarget::default())
        .expect("aggregate kernel params must lower");
    assert!(ptx.contains(".param .align 8 .b8 sparam_param_0[16]"), "{ptx}");
    assert!(ptx.contains("ld.param.u32 %r0, [sparam_param_0];"), "{ptx}");
    assert!(ptx.contains("ld.param.f32 %f0, [sparam_param_0+4];"), "{ptx}");
    assert!(ptx.contains("ld.param.u64 %rd0, [sparam_param_0+8];"), "{ptx}");
    ptxas_verify("struct_param_by_value_kernel", &ptx);
}

// ---------------------------------------------------------------------------
// ld.global.nc (read-only cache) and sub-word load combining
// ---------------------------------------------------------------------------

fn u8_ty(ctx: &mut Context) -> pliron_ll::r#type::TypeHandle {
    IntegerType::get(ctx, 8, Signedness::Signless).into()
}

/// A `.global` byte table module global named `name`, `len` zero bytes,
/// declared alignment 1 (the combiner must raise it itself).
fn byte_table(
    ctx: &mut Context,
    module_body: pliron_ll::context::Ptr<BasicBlock>,
    name: &str,
    len: usize,
) {
    let byte_ty: pliron_ll::r#type::TypeHandle =
        IntegerType::get(ctx, 8, Signedness::Unsigned).into();
    let table_ty = ArrayType::get(ctx, byte_ty, len as u64).into();
    let table = GlobalOp::new(ctx, name.try_into().unwrap(), table_ty);
    table.set_attr_llvm_global_linkage(ctx, LinkageAttr::ExternalLinkage);
    pliron_ll::ll::set_global_data(
        ctx,
        &table,
        pliron_ll::ll::DataAttr { bytes: vec![0; len], align: 1, mutable: false, relocs: vec![] },
    );
    table.get_operation().insert_at_back(module_body, ctx);
}

/// `.nc` follows the per-root write set: loads through a never-written
/// param take the read-only cache, loads through the stored-through param
/// do not — in the same kernel.
#[test]
fn nc_on_readonly_param_not_on_written_param() {
    let mut ctx = Context::new();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "kernels".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();

    let i32t = i32_ty(&mut ctx);
    let ptr = ptr_ty(&mut ctx);
    let func = new_kernel(&mut ctx, body, "ncsplit", vec![ptr, ptr]);
    let entry = func.get_entry_block(&ctx).unwrap();
    let args: Vec<_> = entry.deref(&ctx).arguments().collect();
    let (a, out) = (args[0], args[1]);

    let tid = call_intrinsic(&mut ctx, entry, "llvm_nvvm_read_ptx_sreg_tid_x", Some(i32t), vec![]).unwrap();
    let at = |ctx: &mut Context, base: Value| -> Value {
        let gep = GetElementPtrOp::new(ctx, base, vec![GepIndex::Value(tid)], i32t);
        gep.get_operation().insert_at_back(entry, ctx);
        gep.get_result(ctx)
    };
    // v = a[tid]; out[tid] = v; back = out[tid]; out[tid] = back + v
    let a_slot = at(&mut ctx, a);
    let v = LoadOp::new(&mut ctx, a_slot, i32t);
    v.get_operation().insert_at_back(entry, &ctx);
    let v = v.get_result(&ctx);
    let out_slot = at(&mut ctx, out);
    StoreOp::new(&mut ctx, v, out_slot).get_operation().insert_at_back(entry, &ctx);
    let back = LoadOp::new(&mut ctx, out_slot, i32t);
    back.get_operation().insert_at_back(entry, &ctx);
    let back = back.get_result(&ctx);
    let sum = AddOp::new_with_overflow_flag(&mut ctx, back, v, Default::default());
    sum.get_operation().insert_at_back(entry, &ctx);
    let sum = sum.get_result(&ctx);
    StoreOp::new(&mut ctx, sum, out_slot).get_operation().insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None).get_operation().insert_at_back(entry, &ctx);

    let ptx = nvptx::write_ptx_from_ir(&ctx, module.get_operation(), &PtxTarget::default())
        .expect("ptx emission");
    assert!(ptx.contains("ld.global.nc.u32"), "a is never written:\n{ptx}");
    assert!(
        ptx.contains("ld.global.u32"),
        "out is stored through, its load must NOT take .nc:\n{ptx}"
    );
    assert_eq!(ptx.matches(".nc.").count(), 1, "{ptx}");
    ptxas_verify("nc_on_readonly_param_not_on_written_param", &ptx);
}

/// `CRABBIT_NVPTX_NC=0` kills the qualifier. Environment mutation would
/// race the other tests in this binary, so the gated emission runs in a
/// subprocess (the `#[ignore]`d helper below).
#[test]
fn nc_kill_switch() {
    let exe = std::env::current_exe().expect("test binary path");
    let output = Command::new(exe)
        .args(["--exact", "nc_kill_switch_subprocess_helper", "--include-ignored", "--nocapture"])
        .env("CRABBIT_NVPTX_NC", "0")
        .output()
        .expect("run helper subprocess");
    assert!(
        output.status.success(),
        "helper failed under CRABBIT_NVPTX_NC=0:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Not a test of its own: [nc_kill_switch]'s subprocess body. Emits the
/// simplest read-only load and checks the env var suppressed `.nc`.
#[test]
#[ignore = "runs only as nc_kill_switch's subprocess, under CRABBIT_NVPTX_NC=0"]
fn nc_kill_switch_subprocess_helper() {
    assert_eq!(
        std::env::var("CRABBIT_NVPTX_NC").as_deref(),
        Ok("0"),
        "helper must run under CRABBIT_NVPTX_NC=0"
    );
    let mut ctx = Context::new();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "kernels".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
    let i32t = i32_ty(&mut ctx);
    let ptr = ptr_ty(&mut ctx);
    let func = new_kernel(&mut ctx, body, "nckill", vec![ptr, ptr]);
    let entry = func.get_entry_block(&ctx).unwrap();
    let args: Vec<_> = entry.deref(&ctx).arguments().collect();
    let (a, out) = (args[0], args[1]);
    let v = LoadOp::new(&mut ctx, a, i32t);
    v.get_operation().insert_at_back(entry, &ctx);
    let v = v.get_result(&ctx);
    StoreOp::new(&mut ctx, v, out).get_operation().insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None).get_operation().insert_at_back(entry, &ctx);

    let ptx = nvptx::write_ptx_from_ir(&ctx, module.get_operation(), &PtxTarget::default())
        .expect("ptx emission");
    assert!(!ptx.contains(".nc"), "kill switch must suppress .nc:\n{ptx}");
    assert!(ptx.contains("ld.global.u32"), "{ptx}");
}

/// Four adjacent byte loads from a module-global table at a 4-scaled
/// dynamic base combine into ONE `ld.global.nc.u32` plus four `prmt`
/// extracts, and the table's declared alignment is raised to 4 for the
/// window proof. Two u16 loads covering a window combine through
/// `and`/`shr`. ptxas assembles both.
#[test]
fn byte_loads_combine_into_word() {
    let mut ctx = Context::new();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "kernels".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
    byte_table(&mut ctx, body, "LUT", 1024);

    let i32t = i32_ty(&mut ctx);
    let i64t = i64_ty(&mut ctx);
    let u8t = u8_ty(&mut ctx);
    let i16t: pliron_ll::r#type::TypeHandle =
        IntegerType::get(&mut ctx, 16, Signedness::Signless).into();
    let ptr = ptr_ty(&mut ctx);
    let func = new_kernel(&mut ctx, body, "comb", vec![ptr]);
    let entry = func.get_entry_block(&ctx).unwrap();
    let out = entry.deref(&ctx).arguments().next().unwrap();

    let tid = call_intrinsic(&mut ctx, entry, "llvm_nvvm_read_ptx_sreg_tid_x", Some(i32t), vec![]).unwrap();
    let tid64 = pliron_ll::dialects::llvm::ops::ZExtOp::new(&mut ctx, tid, i64t);
    tid64.get_operation().insert_at_back(entry, &ctx);
    let tid64 = tid64.get_result(&ctx);
    let lut = AddressOfOp::new(&mut ctx, "LUT".try_into().unwrap(), 0);
    lut.get_operation().insert_at_back(entry, &ctx);
    let lut = lut.get_result(&ctx);
    // base = &LUT[tid * 4] (i32-typed stride keeps the window 4-aligned).
    let base = GetElementPtrOp::new(&mut ctx, lut, vec![GepIndex::Value(tid64)], i32t);
    base.get_operation().insert_at_back(entry, &ctx);
    let base = base.get_result(&ctx);

    // Four adjacent bytes...
    let mut sum: Option<Value> = None;
    for j in 0..4u32 {
        let slot = GetElementPtrOp::new(&mut ctx, base, vec![GepIndex::Constant(j)], u8t);
        slot.get_operation().insert_at_back(entry, &ctx);
        let slot = slot.get_result(&ctx);
        let load = LoadOp::new(&mut ctx, slot, u8t);
        load.get_operation().insert_at_back(entry, &ctx);
        let byte = load.get_result(&ctx);
        let wide = pliron_ll::dialects::llvm::ops::ZExtOp::new(&mut ctx, byte, i32t);
        wide.get_operation().insert_at_back(entry, &ctx);
        let wide = wide.get_result(&ctx);
        sum = Some(match sum {
            None => wide,
            Some(acc) => {
                let add = AddOp::new_with_overflow_flag(&mut ctx, acc, wide, Default::default());
                add.get_operation().insert_at_back(entry, &ctx);
                add.get_result(&ctx)
            }
        });
    }
    // ... and the same window as two u16 halves.
    for h in 0..2u32 {
        let slot = GetElementPtrOp::new(&mut ctx, base, vec![GepIndex::Constant(h)], i16t);
        slot.get_operation().insert_at_back(entry, &ctx);
        let slot = slot.get_result(&ctx);
        let load = LoadOp::new(&mut ctx, slot, i16t);
        load.get_operation().insert_at_back(entry, &ctx);
        let half = load.get_result(&ctx);
        let wide = pliron_ll::dialects::llvm::ops::ZExtOp::new(&mut ctx, half, i32t);
        wide.get_operation().insert_at_back(entry, &ctx);
        let wide = wide.get_result(&ctx);
        let add = AddOp::new_with_overflow_flag(
            &mut ctx,
            sum.expect("byte sum built"),
            wide,
            Default::default(),
        );
        add.get_operation().insert_at_back(entry, &ctx);
        sum = Some(add.get_result(&ctx));
    }

    let dst = GetElementPtrOp::new(&mut ctx, out, vec![GepIndex::Value(tid)], i32t);
    dst.get_operation().insert_at_back(entry, &ctx);
    let dst = dst.get_result(&ctx);
    StoreOp::new(&mut ctx, sum.unwrap(), dst).get_operation().insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None).get_operation().insert_at_back(entry, &ctx);

    let ptx = nvptx::write_ptx_from_ir(&ctx, module.get_operation(), &PtxTarget::default())
        .expect("ptx emission");
    // The byte window and the half window each load once; alignment raise.
    assert!(ptx.contains(".global .align 4 .b8 LUT[1024]"), "{ptx}");
    assert_eq!(ptx.matches("ld.global.nc.u32").count(), 2, "{ptx}");
    assert!(!ptx.contains("ld.global.nc.u8"), "bytes must combine:\n{ptx}");
    assert!(!ptx.contains("ld.global.nc.u16"), "halves must combine:\n{ptx}");
    for needle in [
        "prmt.b32 %r", ", 0, 0x4440;", ", 0, 0x4441;", ", 0, 0x4442;", ", 0, 0x4443;",
        ", 65535;", ", 16;",
    ] {
        assert!(ptx.contains(needle), "missing `{needle}` in:\n{ptx}");
    }
    ptxas_verify("byte_loads_combine_into_word", &ptx);
}

/// Combining bails when the window is not fully covered (4-apart stride)
/// and when the table is stored through (no read-only proof — which also
/// strips `.nc`).
#[test]
fn byte_loads_bail_on_stride_and_on_write() {
    let mut ctx = Context::new();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "kernels".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
    byte_table(&mut ctx, body, "TAB", 256);

    let i32t = i32_ty(&mut ctx);
    let u8t = u8_ty(&mut ctx);
    let ptr = ptr_ty(&mut ctx);

    let sum_4_loads = |ctx: &mut Context,
                       entry: pliron_ll::context::Ptr<BasicBlock>,
                       base: Value,
                       stride: u32|
     -> Value {
        let mut sum: Option<Value> = None;
        for j in 0..4u32 {
            let slot =
                GetElementPtrOp::new(ctx, base, vec![GepIndex::Constant(j * stride)], u8t);
            slot.get_operation().insert_at_back(entry, ctx);
            let load = LoadOp::new(ctx, slot.get_result(ctx), u8t);
            load.get_operation().insert_at_back(entry, ctx);
            let byte = load.get_result(ctx);
            let wide = pliron_ll::dialects::llvm::ops::ZExtOp::new(ctx, byte, i32t);
            wide.get_operation().insert_at_back(entry, ctx);
            let wide = wide.get_result(ctx);
            sum = Some(match sum {
                None => wide,
                Some(acc) => {
                    let add =
                        AddOp::new_with_overflow_flag(ctx, acc, wide, Default::default());
                    add.get_operation().insert_at_back(entry, ctx);
                    add.get_result(ctx)
                }
            });
        }
        sum.unwrap()
    };

    // Kernel 1: loads at 0, 4, 8, 12 — every window only partially covered.
    let strided = new_kernel(&mut ctx, body, "strided", vec![ptr]);
    let entry = strided.get_entry_block(&ctx).unwrap();
    let out = entry.deref(&ctx).arguments().next().unwrap();
    let tab = AddressOfOp::new(&mut ctx, "TAB".try_into().unwrap(), 0);
    tab.get_operation().insert_at_back(entry, &ctx);
    let tab = tab.get_result(&ctx);
    let sum = sum_4_loads(&mut ctx, entry, tab, 4);
    StoreOp::new(&mut ctx, sum, out).get_operation().insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None).get_operation().insert_at_back(entry, &ctx);

    // Kernel 2: adjacent loads at 0..3, but the kernel also stores into the
    // table: no read-only proof, so no combining and no `.nc`.
    let written = new_kernel(&mut ctx, body, "written", vec![ptr]);
    let entry2 = written.get_entry_block(&ctx).unwrap();
    let out2 = entry2.deref(&ctx).arguments().next().unwrap();
    let tab2 = AddressOfOp::new(&mut ctx, "TAB".try_into().unwrap(), 0);
    tab2.get_operation().insert_at_back(entry2, &ctx);
    let tab2 = tab2.get_result(&ctx);
    let tid = call_intrinsic(&mut ctx, entry2, "llvm_nvvm_read_ptx_sreg_tid_x", Some(i32t), vec![]).unwrap();
    let tid8 = pliron_ll::dialects::llvm::ops::TruncOp::new(&mut ctx, tid, u8t);
    tid8.get_operation().insert_at_back(entry2, &ctx);
    let tid8 = tid8.get_result(&ctx);
    let wslot = GetElementPtrOp::new(&mut ctx, tab2, vec![GepIndex::Constant(100)], u8t);
    wslot.get_operation().insert_at_back(entry2, &ctx);
    let wslot = wslot.get_result(&ctx);
    StoreOp::new(&mut ctx, tid8, wslot).get_operation().insert_at_back(entry2, &ctx);
    let sum2 = sum_4_loads(&mut ctx, entry2, tab2, 1);
    StoreOp::new(&mut ctx, sum2, out2).get_operation().insert_at_back(entry2, &ctx);
    ReturnOp::new(&mut ctx, None).get_operation().insert_at_back(entry2, &ctx);

    let ptx = nvptx::write_ptx_from_ir(&ctx, module.get_operation(), &PtxTarget::default())
        .expect("ptx emission");
    assert!(!ptx.contains("prmt"), "neither kernel may combine:\n{ptx}");
    assert!(!ptx.contains("ld.global.nc.u32"), "{ptx}");
    // strided: scalar but still read-only. written: scalar AND coherent.
    assert_eq!(ptx.matches("ld.global.nc.u8").count(), 4, "{ptx}");
    assert_eq!(ptx.matches("ld.global.u8").count(), 4, "{ptx}");
    // The bailed windows demand no alignment raise.
    assert!(ptx.contains(".global .align 1 .b8 TAB[256]"), "{ptx}");
    ptxas_verify("byte_loads_bail_on_stride_and_on_write", &ptx);
}

/// The iq3 shape: one window's byte loads spread across blocks, separated
/// by control flow — they still combine because the memory is read-only-
/// proven and the first load's block dominates the rest. The mirrored
/// negative: bytes split across two branch ARMS have no dominating leader
/// and must stay scalar.
#[test]
fn byte_loads_combine_across_dominated_blocks_only() {
    let mut ctx = Context::new();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "kernels".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
    byte_table(&mut ctx, body, "XLUT", 64);

    let i32t = i32_ty(&mut ctx);
    let u8t = u8_ty(&mut ctx);
    let ptr = ptr_ty(&mut ctx);
    let func = new_kernel(&mut ctx, body, "xblock", vec![ptr, i32t]);
    let entry = func.get_entry_block(&ctx).unwrap();
    let args: Vec<_> = entry.deref(&ctx).arguments().collect();
    let (out, n) = (args[0], args[1]);
    let region = func.get_region(&ctx).unwrap();
    let arm_a = BasicBlock::new(&mut ctx, Some("arma".try_into().unwrap()), vec![]);
    arm_a.insert_at_back(region, &ctx);
    let arm_b = BasicBlock::new(&mut ctx, Some("armb".try_into().unwrap()), vec![]);
    arm_b.insert_at_back(region, &ctx);
    let join = BasicBlock::new(&mut ctx, Some("join".try_into().unwrap()), vec![i32t]);
    join.insert_at_back(region, &ctx);

    let byte_at = |ctx: &mut Context,
                   block: pliron_ll::context::Ptr<BasicBlock>,
                   base: Value,
                   off: u32|
     -> Value {
        let slot = GetElementPtrOp::new(ctx, base, vec![GepIndex::Constant(off)], u8t);
        slot.get_operation().insert_at_back(block, ctx);
        let load = LoadOp::new(ctx, slot.get_result(ctx), u8t);
        load.get_operation().insert_at_back(block, ctx);
        let byte = load.get_result(ctx);
        let wide = pliron_ll::dialects::llvm::ops::ZExtOp::new(ctx, byte, i32t);
        wide.get_operation().insert_at_back(block, ctx);
        wide.get_result(ctx)
    };

    let lut = AddressOfOp::new(&mut ctx, "XLUT".try_into().unwrap(), 0);
    lut.get_operation().insert_at_back(entry, &ctx);
    let lut = lut.get_result(&ctx);
    // Entry: bytes 0 and 1 of window A (offsets 0..3), which continues in
    // BOTH arms and the join — every later block is dominated by entry.
    let a0 = byte_at(&mut ctx, entry, lut, 0);
    let a1 = byte_at(&mut ctx, entry, lut, 1);
    let head = AddOp::new_with_overflow_flag(&mut ctx, a0, a1, Default::default());
    head.get_operation().insert_at_back(entry, &ctx);
    let head = head.get_result(&ctx);
    let zero = i32_const(&mut ctx, entry, 0);
    let cond = ICmpOp::new(&mut ctx, ICmpPredicateAttr::SLT, n, zero);
    cond.get_operation().insert_at_back(entry, &ctx);
    let cond = cond.get_result(&ctx);
    CondBrOp::new(&mut ctx, cond, arm_a, vec![], arm_b, vec![])
        .get_operation()
        .insert_at_back(entry, &ctx);

    // Window A's bytes 2 and 3 sit in arm A; window B's (offsets 4..7)
    // bytes are split 2/2 between the arms, so no leader dominates them.
    let a2 = byte_at(&mut ctx, arm_a, lut, 2);
    let a3 = byte_at(&mut ctx, arm_a, lut, 3);
    let b0 = byte_at(&mut ctx, arm_a, lut, 4);
    let b1 = byte_at(&mut ctx, arm_a, lut, 5);
    let s = AddOp::new_with_overflow_flag(&mut ctx, a2, a3, Default::default());
    s.get_operation().insert_at_back(arm_a, &ctx);
    let s = s.get_result(&ctx);
    let s2 = AddOp::new_with_overflow_flag(&mut ctx, s, b0, Default::default());
    s2.get_operation().insert_at_back(arm_a, &ctx);
    let s2 = s2.get_result(&ctx);
    let s3 = AddOp::new_with_overflow_flag(&mut ctx, s2, b1, Default::default());
    s3.get_operation().insert_at_back(arm_a, &ctx);
    let s3 = s3.get_result(&ctx);
    BrOp::new(&mut ctx, join, vec![s3]).get_operation().insert_at_back(arm_a, &ctx);

    let b2 = byte_at(&mut ctx, arm_b, lut, 6);
    let b3 = byte_at(&mut ctx, arm_b, lut, 7);
    let t = AddOp::new_with_overflow_flag(&mut ctx, b2, b3, Default::default());
    t.get_operation().insert_at_back(arm_b, &ctx);
    let t = t.get_result(&ctx);
    BrOp::new(&mut ctx, join, vec![t]).get_operation().insert_at_back(arm_b, &ctx);

    let picked = join.deref(&ctx).arguments().next().unwrap();
    let total = AddOp::new_with_overflow_flag(&mut ctx, head, picked, Default::default());
    total.get_operation().insert_at_back(join, &ctx);
    let total = total.get_result(&ctx);
    StoreOp::new(&mut ctx, total, out).get_operation().insert_at_back(join, &ctx);
    ReturnOp::new(&mut ctx, None).get_operation().insert_at_back(join, &ctx);

    let ptx = nvptx::write_ptx_from_ir(&ctx, module.get_operation(), &PtxTarget::default())
        .expect("ptx emission");
    // Window A combines across entry/arm A; window B stays scalar.
    assert_eq!(ptx.matches("ld.global.nc.u32").count(), 1, "{ptx}");
    assert_eq!(ptx.matches("prmt.b32").count(), 4, "{ptx}");
    assert_eq!(
        ptx.matches("ld.global.nc.u8").count(),
        4,
        "window B has no dominating leader and must stay scalar:\n{ptx}"
    );
    ptxas_verify("byte_loads_combine_across_dominated_blocks_only", &ptx);
}
