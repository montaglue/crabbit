//! End-to-end execution tests for the llvm-vectorize pass on the
//! aarch64-linux pipeline: build canonical streaming loops by hand, run the
//! vectorizer (asserting it fired), lower through the full machine
//! pipeline, link with `cc`, execute, and compare against the scalar
//! semantics computed in Rust. Trip counts are chosen NOT to divide the
//! vector factor, so the scalar epilogue runs too.

#![cfg(all(target_os = "linux", target_arch = "aarch64"))]

use pliron::builtin::op_interfaces::AtMostOneRegionInterface as _;
use pliron::builtin::ops::ConstantOp;
use std::num::NonZero;
use std::path::PathBuf;
use std::process::Command;

use pliron_ll::{
    context::{Context, Ptr},
    conversion::pass::{AnalysisManager, Pass},
    dialects::{
        aarch64, builtin,
        builtin::{
            attributes::{FPSingleAttr, IntegerAttr},
            op_interfaces::{OneRegionInterface, OneResultInterface},
            types::{FP32Type, IntegerType, Signedness},
        },
        llvm::{
            attributes::{ICmpPredicateAttr, LinkageAttr},
            ops::{
                AddOp, AddressOfOp, AndOp, AShrOp, BrOp, CondBrOp, FAddOp, FMulOp,
                FPToSIOp, FuncOp, GepIndex, GetElementPtrOp, GlobalOp, ICmpOp, LoadOp,
                LShrOp, MulOp, OrOp, ReturnOp, ShlOp, StoreOp, SubOp, XorOp,
            },
            types::FuncType,
        },
        macho,
    },
    ir::{basic_block::BasicBlock, op::Op, r#type::TypeHandle, value::Value},
    linked_list::ContainsLinkedList,
    ll::DataAttr,
    passes::aarch64_linux,
    passes::llvm::vectorize::LLVMVectorizePass,
    target_profile::TargetProfile,
    utils::apint::APInt,
};
#[allow(unused_imports)]
use pliron_llvm::op_interfaces::{
    BinArithOp as _, CastOpInterface as _, FloatBinArithOpWithFastMathFlags as _,
    IntBinArithOpWithOverflowFlag as _,
};
use pliron::printable::Printable;

fn context() -> Context {
    let mut ctx = Context::new();
    aarch64::register(&mut ctx);
    macho::register(&mut ctx);
    ctx
}

fn i64_ty(ctx: &mut Context) -> TypeHandle {
    IntegerType::get(ctx, 64, Signedness::Signless).into()
}

fn i32_ty(ctx: &mut Context) -> TypeHandle {
    IntegerType::get(ctx, 32, Signedness::Signless).into()
}

fn const_int(ctx: &mut Context, block: Ptr<BasicBlock>, value: u64, width: u32) -> Value {
    let ty = IntegerType::get(ctx, width, Signedness::Signless);
    let c = ConstantOp::new(
        ctx,
        Box::new(IntegerAttr::new(
            ty,
            APInt::from_u64(value, NonZero::new(width as usize).unwrap()),
        )),
    );
    c.get_operation().insert_at_back(block, ctx);
    c.get_result(ctx)
}

fn const_f32(ctx: &mut Context, block: Ptr<BasicBlock>, value: f32) -> Value {
    let c = ConstantOp::new(ctx, Box::new(FPSingleAttr::from(value)));
    c.get_operation().insert_at_back(block, ctx);
    c.get_result(ctx)
}

fn data_global(ctx: &mut Context, body: Ptr<BasicBlock>, name: &str, bytes: Vec<u8>, mutable: bool) {
    let ty = i64_ty(ctx);
    let global = GlobalOp::new(ctx, name.try_into().unwrap(), ty);
    global.set_attr_llvm_global_linkage(ctx, LinkageAttr::ExternalLinkage);
    pliron_ll::ll::set_global_data(
        ctx,
        &global,
        DataAttr {
            bytes,
            align: 16,
            mutable,
            relocs: vec![],
        },
    );
    global.get_operation().insert_at_back(body, ctx);
}

fn address_of(ctx: &mut Context, block: Ptr<BasicBlock>, name: &str) -> Value {
    let addr = AddressOfOp::new(ctx, name.try_into().unwrap(), 0);
    addr.get_operation().insert_at_back(block, ctx);
    addr.get_result(ctx)
}

fn gep_at(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    base: Value,
    index: Value,
    elem: TypeHandle,
) -> Value {
    let gep = GetElementPtrOp::new(ctx, base, vec![GepIndex::Value(index)], elem);
    gep.get_operation().insert_at_back(block, ctx);
    gep.get_result(ctx)
}

fn load(ctx: &mut Context, block: Ptr<BasicBlock>, addr: Value, ty: TypeHandle) -> Value {
    let l = LoadOp::new(ctx, addr, ty);
    l.get_operation().insert_at_back(block, ctx);
    l.get_result(ctx)
}

fn link_and_run(test: &str, object_bytes: &[u8]) -> i32 {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(test);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let object = dir.join("test.o");
    let binary = dir.join("test");
    std::fs::write(&object, object_bytes).expect("write object");

    let link = Command::new("cc")
        .arg("-o")
        .arg(&binary)
        .arg(&object)
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

/// Run the vectorizer over `module` and assert it produced vector ops
/// (`marker` names an op the transformed module must contain).
fn vectorize_expecting(ctx: &mut Context, module: builtin::ops::ModuleOp, marker: &str) -> String {
    LLVMVectorizePass::new(&TargetProfile::host_cpu().with_simd128(true))
        .run(module.get_operation(), ctx, &mut AnalysisManager::default())
        .unwrap();
    let text = format!("{}", module.get_operation().disp(ctx));
    assert!(text.contains(marker), "vectorizer did not fire ({marker}):\n{text}");
    text
}

/// saxpy over f32 with n = 10 (VF 4 + epilogue 2): y[i] = 2.5*x[i] + y[i]
/// with x[i] = y[i] = i, then a scalar (FP-reduction, knob off) checksum
/// loop sums y and returns fptosi(sum) = 3.5 * 45 = 157.
#[test]
fn vectorized_f32_saxpy_matches_scalar_semantics() {
    const N: usize = 10;
    let mut ctx = context();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
    let xs: Vec<u8> = (0..N).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let ys: Vec<u8> = (0..N).flat_map(|i| (i as f32).to_le_bytes()).collect();
    data_global(&mut ctx, body, "xs", xs, false);
    data_global(&mut ctx, body, "ys", ys, true);

    let i64t = i64_ty(&mut ctx);
    let f32t: TypeHandle = FP32Type::get(&mut ctx).into();
    let func_ty = FuncType::get(&mut ctx, i64t, vec![], false);
    let func = FuncOp::new(&mut ctx, "main".try_into().unwrap(), func_ty);
    func.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
    func.get_or_create_entry_block(&mut ctx);
    func.get_operation().insert_at_back(body, &ctx);
    let region = func.get_region(&ctx).unwrap();
    let entry = func.get_entry_block(&ctx).unwrap();

    let header = BasicBlock::new(&mut ctx, None, vec![i64t]);
    header.insert_at_back(region, &ctx);
    let loop_body = BasicBlock::new(&mut ctx, None, vec![]);
    loop_body.insert_at_back(region, &ctx);
    let pre2 = BasicBlock::new(&mut ctx, None, vec![]);
    pre2.insert_at_back(region, &ctx);
    let sheader = BasicBlock::new(&mut ctx, None, vec![i64t, f32t]);
    sheader.insert_at_back(region, &ctx);
    let sbody = BasicBlock::new(&mut ctx, None, vec![]);
    sbody.insert_at_back(region, &ctx);
    let done = BasicBlock::new(&mut ctx, None, vec![]);
    done.insert_at_back(region, &ctx);

    // entry: constants, addresses, br header(0)
    let x = address_of(&mut ctx, entry, "xs");
    let y = address_of(&mut ctx, entry, "ys");
    let zero = const_int(&mut ctx, entry, 0, 64);
    let one = const_int(&mut ctx, entry, 1, 64);
    let n = const_int(&mut ctx, entry, N as u64, 64);
    let alpha = const_f32(&mut ctx, entry, 2.5);
    let fzero = const_f32(&mut ctx, entry, 0.0);
    BrOp::new(&mut ctx, header, vec![zero])
        .get_operation()
        .insert_at_back(entry, &ctx);

    // header(iv): iv < n ? body : pre2
    let iv = header.deref(&ctx).get_argument(0);
    let cond = ICmpOp::new(&mut ctx, ICmpPredicateAttr::ULT, iv, n);
    cond.get_operation().insert_at_back(header, &ctx);
    let cond_v = cond.get_result(&ctx);
    CondBrOp::new(&mut ctx, cond_v, loop_body, vec![], pre2, vec![])
        .get_operation()
        .insert_at_back(header, &ctx);

    // body: y[iv] = alpha * x[iv] + y[iv]
    let gx = gep_at(&mut ctx, loop_body, x, iv, f32t);
    let lx = load(&mut ctx, loop_body, gx, f32t);
    let m = FMulOp::new_with_fast_math_flags(&mut ctx, alpha, lx, Default::default());
    m.get_operation().insert_at_back(loop_body, &ctx);
    let m_v = m.get_result(&ctx);
    let gy = gep_at(&mut ctx, loop_body, y, iv, f32t);
    let ly = load(&mut ctx, loop_body, gy, f32t);
    let s = FAddOp::new_with_fast_math_flags(&mut ctx, m_v, ly, Default::default());
    s.get_operation().insert_at_back(loop_body, &ctx);
    let s_v = s.get_result(&ctx);
    StoreOp::new(&mut ctx, s_v, gy)
        .get_operation()
        .insert_at_back(loop_body, &ctx);
    let iv2 = AddOp::new_with_overflow_flag(&mut ctx, iv, one, Default::default());
    iv2.get_operation().insert_at_back(loop_body, &ctx);
    let iv2_v = iv2.get_result(&ctx);
    BrOp::new(&mut ctx, header, vec![iv2_v])
        .get_operation()
        .insert_at_back(loop_body, &ctx);

    // pre2: br sheader(0, 0.0)
    BrOp::new(&mut ctx, sheader, vec![zero, fzero])
        .get_operation()
        .insert_at_back(pre2, &ctx);

    // sheader(siv, acc): siv < n ? sbody : done   (FP reduction: the knob
    // is off, so this loop must stay scalar.)
    let siv = sheader.deref(&ctx).get_argument(0);
    let acc = sheader.deref(&ctx).get_argument(1);
    let scond = ICmpOp::new(&mut ctx, ICmpPredicateAttr::ULT, siv, n);
    scond.get_operation().insert_at_back(sheader, &ctx);
    let scond_v = scond.get_result(&ctx);
    CondBrOp::new(&mut ctx, scond_v, sbody, vec![], done, vec![])
        .get_operation()
        .insert_at_back(sheader, &ctx);

    let sgy = gep_at(&mut ctx, sbody, y, siv, f32t);
    let sly = load(&mut ctx, sbody, sgy, f32t);
    let acc2 = FAddOp::new_with_fast_math_flags(&mut ctx, acc, sly, Default::default());
    acc2.get_operation().insert_at_back(sbody, &ctx);
    let acc2_v = acc2.get_result(&ctx);
    let siv2 = AddOp::new_with_overflow_flag(&mut ctx, siv, one, Default::default());
    siv2.get_operation().insert_at_back(sbody, &ctx);
    let siv2_v = siv2.get_result(&ctx);
    BrOp::new(&mut ctx, sheader, vec![siv2_v, acc2_v])
        .get_operation()
        .insert_at_back(sbody, &ctx);

    // done: return fptosi(acc)
    let conv = FPToSIOp::new(&mut ctx, acc, i64t);
    conv.get_operation().insert_at_back(done, &ctx);
    let conv_v = conv.get_result(&ctx);
    ReturnOp::new(&mut ctx, Some(conv_v))
        .get_operation()
        .insert_at_back(done, &ctx);

    let text = vectorize_expecting(&mut ctx, module, "ll.vload");
    assert!(text.contains("ll.vbinop fmul"), "{text}");
    assert!(text.contains("ll.vbinop fadd"), "{text}");
    assert!(
        !text.contains("ll.vreduce"),
        "FP reduction must stay scalar with the knob off:\n{text}"
    );

    let bytes = aarch64_linux::emit_elf_object_bytes(&mut ctx, module.get_operation()).unwrap();
    // sum_i 3.5*i for i in 0..10 = 3.5 * 45 = 157.5 -> fptosi -> 157.
    assert_eq!(link_and_run("vectorized_f32_saxpy", &bytes), 157);
}

const A_VALS: [i32; 11] = [3, -7, 100, -25000, 12345, -1, 0, 77, -88, 4096, -32768];
const B_VALS: [i32; 11] = [5, 9, -13, 217, -999, 42, -7, 1, 250, -4, 8];

/// The scalar semantics of the chain loop, computed with Rust's wrapping
/// i32 ops (the same wrap/shift semantics as the llvm dialect ops built in
/// the test body).
fn expected_chain_checksum() -> i32 {
    let mut sum: i32 = 0;
    for i in 0..A_VALS.len() {
        let av = A_VALS[i];
        let bv = B_VALS[i];
        let t2 = av.wrapping_mul(3).wrapping_add(bv);
        let t3 = av >> 2;
        let t4 = t2 ^ t3;
        let t7 = t4.wrapping_shl(1) | (((t4 as u32) >> 7) as i32);
        let t8 = t7 & 0xFFFF;
        let t9 = t8.wrapping_sub(av);
        sum = sum.wrapping_add(t9);
    }
    sum & 0x7f
}

/// An elementwise_chain-shaped i32 loop (mul/add/xor/shifts/and/or/sub over
/// a[i], b[i] -> out[i], n = 11 = 2*VF + 3) followed by an integer
/// add-reduction loop over out — both vectorize; the checksum must equal
/// the scalar semantics bit for bit.
#[test]
fn vectorized_i32_chain_and_reduction_match_scalar_semantics() {
    const N: usize = 11;
    let mut ctx = context();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
    let a_bytes: Vec<u8> = A_VALS.iter().flat_map(|v| v.to_le_bytes()).collect();
    let b_bytes: Vec<u8> = B_VALS.iter().flat_map(|v| v.to_le_bytes()).collect();
    let out_bytes: Vec<u8> = vec![0; N * 4];
    data_global(&mut ctx, body, "av", a_bytes, false);
    data_global(&mut ctx, body, "bv", b_bytes, false);
    data_global(&mut ctx, body, "outv", out_bytes, true);

    let i64t = i64_ty(&mut ctx);
    let i32t = i32_ty(&mut ctx);
    let func_ty = FuncType::get(&mut ctx, i64t, vec![], false);
    let func = FuncOp::new(&mut ctx, "main".try_into().unwrap(), func_ty);
    func.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
    func.get_or_create_entry_block(&mut ctx);
    func.get_operation().insert_at_back(body, &ctx);
    let region = func.get_region(&ctx).unwrap();
    let entry = func.get_entry_block(&ctx).unwrap();

    let header = BasicBlock::new(&mut ctx, None, vec![i64t]);
    header.insert_at_back(region, &ctx);
    let loop_body = BasicBlock::new(&mut ctx, None, vec![]);
    loop_body.insert_at_back(region, &ctx);
    let pre2 = BasicBlock::new(&mut ctx, None, vec![]);
    pre2.insert_at_back(region, &ctx);
    let sheader = BasicBlock::new(&mut ctx, None, vec![i64t, i32t]);
    sheader.insert_at_back(region, &ctx);
    let sbody = BasicBlock::new(&mut ctx, None, vec![]);
    sbody.insert_at_back(region, &ctx);
    let done = BasicBlock::new(&mut ctx, None, vec![]);
    done.insert_at_back(region, &ctx);

    let a = address_of(&mut ctx, entry, "av");
    let b = address_of(&mut ctx, entry, "bv");
    let out = address_of(&mut ctx, entry, "outv");
    let zero = const_int(&mut ctx, entry, 0, 64);
    let one = const_int(&mut ctx, entry, 1, 64);
    let n = const_int(&mut ctx, entry, N as u64, 64);
    let c3 = const_int(&mut ctx, entry, 3, 32);
    let c1 = const_int(&mut ctx, entry, 1, 32);
    let c7 = const_int(&mut ctx, entry, 7, 32);
    let cffff = const_int(&mut ctx, entry, 0xFFFF, 32);
    let c127 = const_int(&mut ctx, entry, 0x7f, 32);
    let i32zero = const_int(&mut ctx, entry, 0, 32);
    BrOp::new(&mut ctx, header, vec![zero])
        .get_operation()
        .insert_at_back(entry, &ctx);

    let iv = header.deref(&ctx).get_argument(0);
    let cond = ICmpOp::new(&mut ctx, ICmpPredicateAttr::ULT, iv, n);
    cond.get_operation().insert_at_back(header, &ctx);
    let cond_v = cond.get_result(&ctx);
    CondBrOp::new(&mut ctx, cond_v, loop_body, vec![], pre2, vec![])
        .get_operation()
        .insert_at_back(header, &ctx);

    // body: the chain. The `>> 2` shift constant lives in the BODY, to
    // exercise the clone-and-splat path for loop-local constants.
    let ga = gep_at(&mut ctx, loop_body, a, iv, i32t);
    let av = load(&mut ctx, loop_body, ga, i32t);
    let gb = gep_at(&mut ctx, loop_body, b, iv, i32t);
    let bv = load(&mut ctx, loop_body, gb, i32t);
    let t1 = MulOp::new_with_overflow_flag(&mut ctx, av, c3, Default::default());
    t1.get_operation().insert_at_back(loop_body, &ctx);
    let t1_v = t1.get_result(&ctx);
    let t2 = AddOp::new_with_overflow_flag(&mut ctx, t1_v, bv, Default::default());
    t2.get_operation().insert_at_back(loop_body, &ctx);
    let t2_v = t2.get_result(&ctx);
    let c2_body = const_int(&mut ctx, loop_body, 2, 32);
    let t3 = AShrOp::new(&mut ctx, av, c2_body);
    t3.get_operation().insert_at_back(loop_body, &ctx);
    let t3_v = t3.get_result(&ctx);
    let t4 = XorOp::new(&mut ctx, t2_v, t3_v);
    t4.get_operation().insert_at_back(loop_body, &ctx);
    let t4_v = t4.get_result(&ctx);
    let t5 = ShlOp::new_with_overflow_flag(&mut ctx, t4_v, c1, Default::default());
    t5.get_operation().insert_at_back(loop_body, &ctx);
    let t5_v = t5.get_result(&ctx);
    let t6 = LShrOp::new(&mut ctx, t4_v, c7);
    t6.get_operation().insert_at_back(loop_body, &ctx);
    let t6_v = t6.get_result(&ctx);
    let t7 = OrOp::new(&mut ctx, t5_v, t6_v);
    t7.get_operation().insert_at_back(loop_body, &ctx);
    let t7_v = t7.get_result(&ctx);
    let t8 = AndOp::new(&mut ctx, t7_v, cffff);
    t8.get_operation().insert_at_back(loop_body, &ctx);
    let t8_v = t8.get_result(&ctx);
    let t9 = SubOp::new_with_overflow_flag(&mut ctx, t8_v, av, Default::default());
    t9.get_operation().insert_at_back(loop_body, &ctx);
    let t9_v = t9.get_result(&ctx);
    let go = gep_at(&mut ctx, loop_body, out, iv, i32t);
    StoreOp::new(&mut ctx, t9_v, go)
        .get_operation()
        .insert_at_back(loop_body, &ctx);
    let iv2 = AddOp::new_with_overflow_flag(&mut ctx, iv, one, Default::default());
    iv2.get_operation().insert_at_back(loop_body, &ctx);
    let iv2_v = iv2.get_result(&ctx);
    BrOp::new(&mut ctx, header, vec![iv2_v])
        .get_operation()
        .insert_at_back(loop_body, &ctx);

    BrOp::new(&mut ctx, sheader, vec![zero, i32zero])
        .get_operation()
        .insert_at_back(pre2, &ctx);

    // sheader(siv, sum): the integer add-reduction over out.
    let siv = sheader.deref(&ctx).get_argument(0);
    let sum = sheader.deref(&ctx).get_argument(1);
    let scond = ICmpOp::new(&mut ctx, ICmpPredicateAttr::ULT, siv, n);
    scond.get_operation().insert_at_back(sheader, &ctx);
    let scond_v = scond.get_result(&ctx);
    CondBrOp::new(&mut ctx, scond_v, sbody, vec![], done, vec![])
        .get_operation()
        .insert_at_back(sheader, &ctx);

    let sgo = gep_at(&mut ctx, sbody, out, siv, i32t);
    let so = load(&mut ctx, sbody, sgo, i32t);
    let sum2 = AddOp::new_with_overflow_flag(&mut ctx, sum, so, Default::default());
    sum2.get_operation().insert_at_back(sbody, &ctx);
    let sum2_v = sum2.get_result(&ctx);
    let siv2 = AddOp::new_with_overflow_flag(&mut ctx, siv, one, Default::default());
    siv2.get_operation().insert_at_back(sbody, &ctx);
    let siv2_v = siv2.get_result(&ctx);
    BrOp::new(&mut ctx, sheader, vec![siv2_v, sum2_v])
        .get_operation()
        .insert_at_back(sbody, &ctx);

    // done: return sum & 0x7f
    let masked = AndOp::new(&mut ctx, sum, c127);
    masked.get_operation().insert_at_back(done, &ctx);
    let masked_v = masked.get_result(&ctx);
    ReturnOp::new(&mut ctx, Some(masked_v))
        .get_operation()
        .insert_at_back(done, &ctx);

    let text = vectorize_expecting(&mut ctx, module, "ll.vload");
    for needle in [
        "ll.vbinop mul",
        "ll.vbinop add",
        "ll.vbinop xor",
        "ll.vbinop shl",
        "ll.vbinop lshr",
        "ll.vbinop ashr",
        "ll.vbinop and",
        "ll.vbinop or",
        "ll.vreduce add",
        "ll.vstore",
    ] {
        assert!(text.contains(needle), "missing {needle}:\n{text}");
    }

    let bytes = aarch64_linux::emit_elf_object_bytes(&mut ctx, module.get_operation()).unwrap();
    assert_eq!(
        link_and_run("vectorized_i32_chain", &bytes),
        expected_chain_checksum(),
        "vectorized checksum diverged from scalar semantics"
    );
}

/// The shape rustc's slice indexing actually reaches the vectorizer in
/// (verified against the corpus elementwise_chain dump): the (ptr, len)
/// fat pointers sit in un-promoted alloca slots re-loaded EVERY iteration,
/// and addressing is byte-typed geps indexed by `iv << 2`. The vectorizer
/// must re-load the non-escaping slots once in the preheader and rebuild
/// the byte index at the vector iv. n = 9 leaves a 1-lane epilogue.
#[test]
fn vectorized_slot_loaded_byte_gep_matches_scalar_semantics() {
    const N: usize = 9;
    let mut ctx = context();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
    let a_bytes: Vec<u8> = A_VALS[..N].iter().flat_map(|v| v.to_le_bytes()).collect();
    let out_bytes: Vec<u8> = vec![0; N * 4];
    data_global(&mut ctx, body, "sa", a_bytes, false);
    data_global(&mut ctx, body, "sout", out_bytes, true);

    let i64t = i64_ty(&mut ctx);
    let i32t = i32_ty(&mut ctx);
    let u8t: TypeHandle =
        IntegerType::get(&mut ctx, 8, Signedness::Unsigned).into();
    let ptr_ty: TypeHandle =
        pliron_llvm::types::PointerType::get(&mut ctx, 0).into();
    let fat_ty: TypeHandle =
        pliron_llvm::types::StructType::get_unnamed(&mut ctx, vec![ptr_ty, i64t]).into();
    let func_ty = FuncType::get(&mut ctx, i64t, vec![], false);
    let func = FuncOp::new(&mut ctx, "main".try_into().unwrap(), func_ty);
    func.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
    func.get_or_create_entry_block(&mut ctx);
    func.get_operation().insert_at_back(body, &ctx);
    let region = func.get_region(&ctx).unwrap();
    let entry = func.get_entry_block(&ctx).unwrap();

    let header = BasicBlock::new(&mut ctx, None, vec![i64t]);
    header.insert_at_back(region, &ctx);
    let loop_body = BasicBlock::new(&mut ctx, None, vec![]);
    loop_body.insert_at_back(region, &ctx);
    let pre2 = BasicBlock::new(&mut ctx, None, vec![]);
    pre2.insert_at_back(region, &ctx);
    let sheader = BasicBlock::new(&mut ctx, None, vec![i64t, i32t]);
    sheader.insert_at_back(region, &ctx);
    let sbody = BasicBlock::new(&mut ctx, None, vec![]);
    sbody.insert_at_back(region, &ctx);
    let done = BasicBlock::new(&mut ctx, None, vec![]);
    done.insert_at_back(region, &ctx);

    // entry: two fat-pointer slots, stored once (the slots never escape).
    let one_i32 = const_int(&mut ctx, entry, 1, 32);
    let slot_a = pliron_llvm::ops::AllocaOp::new(&mut ctx, fat_ty, one_i32);
    slot_a.get_operation().insert_at_back(entry, &ctx);
    let slot_a_v = slot_a.get_result(&ctx);
    let slot_out = pliron_llvm::ops::AllocaOp::new(&mut ctx, fat_ty, one_i32);
    slot_out.get_operation().insert_at_back(entry, &ctx);
    let slot_out_v = slot_out.get_result(&ctx);
    let a = address_of(&mut ctx, entry, "sa");
    let out = address_of(&mut ctx, entry, "sout");
    let zero = const_int(&mut ctx, entry, 0, 64);
    let one = const_int(&mut ctx, entry, 1, 64);
    let n = const_int(&mut ctx, entry, N as u64, 64);
    let len = const_int(&mut ctx, entry, N as u64, 64);
    let c3 = const_int(&mut ctx, entry, 3, 32);
    let c2_shift = const_int(&mut ctx, entry, 2, 64);
    let c2_ashr = const_int(&mut ctx, entry, 2, 32);
    let c127 = const_int(&mut ctx, entry, 0x7f, 32);
    let i32zero = const_int(&mut ctx, entry, 0, 32);
    for (slot, ptr) in [(slot_a_v, a), (slot_out_v, out)] {
        let undef = pliron_llvm::ops::UndefOp::new(&mut ctx, fat_ty);
        undef.get_operation().insert_at_back(entry, &ctx);
        let undef_v = undef.get_result(&ctx);
        let with_ptr = pliron_llvm::ops::InsertValueOp::new(&mut ctx, undef_v, ptr, vec![0]);
        with_ptr.get_operation().insert_at_back(entry, &ctx);
        let with_ptr_v = with_ptr.get_result(&ctx);
        let with_len = pliron_llvm::ops::InsertValueOp::new(&mut ctx, with_ptr_v, len, vec![1]);
        with_len.get_operation().insert_at_back(entry, &ctx);
        let with_len_v = with_len.get_result(&ctx);
        StoreOp::new(&mut ctx, with_len_v, slot)
            .get_operation()
            .insert_at_back(entry, &ctx);
    }
    BrOp::new(&mut ctx, header, vec![zero])
        .get_operation()
        .insert_at_back(entry, &ctx);

    let iv = header.deref(&ctx).get_argument(0);
    let cond = ICmpOp::new(&mut ctx, ICmpPredicateAttr::ULT, iv, n);
    cond.get_operation().insert_at_back(header, &ctx);
    let cond_v = cond.get_result(&ctx);
    CondBrOp::new(&mut ctx, cond_v, loop_body, vec![], pre2, vec![])
        .get_operation()
        .insert_at_back(header, &ctx);

    // body: fat_a = load slot_a; av = load gep<u8>(fat_a, iv << 2);
    //       t = (av * 3) ^ (av >> 2); store t -> gep<u8>(load slot_out, iv << 2)
    let fat_a = load(&mut ctx, loop_body, slot_a_v, fat_ty);
    let byte_idx = ShlOp::new_with_overflow_flag(&mut ctx, iv, c2_shift, Default::default());
    byte_idx.get_operation().insert_at_back(loop_body, &ctx);
    let byte_idx_v = byte_idx.get_result(&ctx);
    let ga = gep_at(&mut ctx, loop_body, fat_a, byte_idx_v, u8t);
    let av = load(&mut ctx, loop_body, ga, i32t);
    let t1 = MulOp::new_with_overflow_flag(&mut ctx, av, c3, Default::default());
    t1.get_operation().insert_at_back(loop_body, &ctx);
    let t1_v = t1.get_result(&ctx);
    let t2 = AShrOp::new(&mut ctx, av, c2_ashr);
    t2.get_operation().insert_at_back(loop_body, &ctx);
    let t2_v = t2.get_result(&ctx);
    let t3 = XorOp::new(&mut ctx, t1_v, t2_v);
    t3.get_operation().insert_at_back(loop_body, &ctx);
    let t3_v = t3.get_result(&ctx);
    let fat_out = load(&mut ctx, loop_body, slot_out_v, fat_ty);
    let go = gep_at(&mut ctx, loop_body, fat_out, byte_idx_v, u8t);
    StoreOp::new(&mut ctx, t3_v, go)
        .get_operation()
        .insert_at_back(loop_body, &ctx);
    let iv2 = AddOp::new_with_overflow_flag(&mut ctx, iv, one, Default::default());
    iv2.get_operation().insert_at_back(loop_body, &ctx);
    let iv2_v = iv2.get_result(&ctx);
    BrOp::new(&mut ctx, header, vec![iv2_v])
        .get_operation()
        .insert_at_back(loop_body, &ctx);

    BrOp::new(&mut ctx, sheader, vec![zero, i32zero])
        .get_operation()
        .insert_at_back(pre2, &ctx);

    // Scalar checksum over out (element-typed geps; integer reduction).
    let siv = sheader.deref(&ctx).get_argument(0);
    let sum = sheader.deref(&ctx).get_argument(1);
    let scond = ICmpOp::new(&mut ctx, ICmpPredicateAttr::ULT, siv, n);
    scond.get_operation().insert_at_back(sheader, &ctx);
    let scond_v = scond.get_result(&ctx);
    CondBrOp::new(&mut ctx, scond_v, sbody, vec![], done, vec![])
        .get_operation()
        .insert_at_back(sheader, &ctx);
    let sgo = gep_at(&mut ctx, sbody, out, siv, i32t);
    let so = load(&mut ctx, sbody, sgo, i32t);
    let sum2 = AddOp::new_with_overflow_flag(&mut ctx, sum, so, Default::default());
    sum2.get_operation().insert_at_back(sbody, &ctx);
    let sum2_v = sum2.get_result(&ctx);
    let siv2 = AddOp::new_with_overflow_flag(&mut ctx, siv, one, Default::default());
    siv2.get_operation().insert_at_back(sbody, &ctx);
    let siv2_v = siv2.get_result(&ctx);
    BrOp::new(&mut ctx, sheader, vec![siv2_v, sum2_v])
        .get_operation()
        .insert_at_back(sbody, &ctx);

    let masked = AndOp::new(&mut ctx, sum, c127);
    masked.get_operation().insert_at_back(done, &ctx);
    let masked_v = masked.get_result(&ctx);
    ReturnOp::new(&mut ctx, Some(masked_v))
        .get_operation()
        .insert_at_back(done, &ctx);

    let text = vectorize_expecting(&mut ctx, module, "ll.vload");
    assert!(text.contains("ll.vbinop xor"), "{text}");

    let expected = {
        let mut sum: i32 = 0;
        for av in &A_VALS[..N] {
            let t = av.wrapping_mul(3) ^ (av >> 2);
            sum = sum.wrapping_add(t);
        }
        sum & 0x7f
    };
    let bytes = aarch64_linux::emit_elf_object_bytes(&mut ctx, module.get_operation()).unwrap();
    assert_eq!(
        link_and_run("vectorized_slot_byte_gep", &bytes),
        expected,
        "slot-loaded byte-gep vectorization diverged from scalar semantics"
    );
}

/// A zero-trip loop entered with iv0 > bound: the vector guard's
/// `iv < bound` term must keep the vector body from running (the wrapped
/// `bound - iv` difference alone would look like a huge remaining trip).
/// The loop would store a marker; nothing may be written.
#[test]
fn vectorized_zero_trip_loop_stores_nothing() {
    let mut ctx = context();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
    // 16 i32 slots, all zero; the (never-executed) loop would write 1s.
    data_global(&mut ctx, body, "zflags", vec![0; 16 * 4], true);

    let i64t = i64_ty(&mut ctx);
    let i32t = i32_ty(&mut ctx);
    let func_ty = FuncType::get(&mut ctx, i64t, vec![], false);
    let func = FuncOp::new(&mut ctx, "main".try_into().unwrap(), func_ty);
    func.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
    func.get_or_create_entry_block(&mut ctx);
    func.get_operation().insert_at_back(body, &ctx);
    let region = func.get_region(&ctx).unwrap();
    let entry = func.get_entry_block(&ctx).unwrap();

    let header = BasicBlock::new(&mut ctx, None, vec![i64t]);
    header.insert_at_back(region, &ctx);
    let loop_body = BasicBlock::new(&mut ctx, None, vec![]);
    loop_body.insert_at_back(region, &ctx);
    let done = BasicBlock::new(&mut ctx, None, vec![]);
    done.insert_at_back(region, &ctx);

    let flags = address_of(&mut ctx, entry, "zflags");
    let five = const_int(&mut ctx, entry, 5, 64);
    let three = const_int(&mut ctx, entry, 3, 64);
    let one = const_int(&mut ctx, entry, 1, 64);
    let marker = const_int(&mut ctx, entry, 1, 32);
    BrOp::new(&mut ctx, header, vec![five])
        .get_operation()
        .insert_at_back(entry, &ctx);

    // header(iv = 5): iv < 3 is false immediately.
    let iv = header.deref(&ctx).get_argument(0);
    let cond = ICmpOp::new(&mut ctx, ICmpPredicateAttr::ULT, iv, three);
    cond.get_operation().insert_at_back(header, &ctx);
    let cond_v = cond.get_result(&ctx);
    CondBrOp::new(&mut ctx, cond_v, loop_body, vec![], done, vec![])
        .get_operation()
        .insert_at_back(header, &ctx);

    let gf = gep_at(&mut ctx, loop_body, flags, iv, i32t);
    StoreOp::new(&mut ctx, marker, gf)
        .get_operation()
        .insert_at_back(loop_body, &ctx);
    let iv2 = AddOp::new_with_overflow_flag(&mut ctx, iv, one, Default::default());
    iv2.get_operation().insert_at_back(loop_body, &ctx);
    let iv2_v = iv2.get_result(&ctx);
    BrOp::new(&mut ctx, header, vec![iv2_v])
        .get_operation()
        .insert_at_back(loop_body, &ctx);

    // done: return flags[5] (would be 1 if the vector body ever ran).
    let g5 = gep_at(&mut ctx, done, flags, five, i32t);
    let f5 = load(&mut ctx, done, g5, i32t);
    ReturnOp::new(&mut ctx, Some(f5))
        .get_operation()
        .insert_at_back(done, &ctx);

    // The loop itself is vectorizable in shape (splat store), so the pass
    // fires; only the guard keeps the vector body dead.
    let text = vectorize_expecting(&mut ctx, module, "ll.vstore");
    assert!(text.contains("ll.vstore"), "{text}");

    let bytes = aarch64_linux::emit_elf_object_bytes(&mut ctx, module.get_operation()).unwrap();
    assert_eq!(link_and_run("vectorized_zero_trip", &bytes), 0);
}
