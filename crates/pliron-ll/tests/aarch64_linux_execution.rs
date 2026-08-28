//! End-to-end execution tests for the aarch64-linux pipeline: build LLVM
//! dialect IR by hand — including `ll.data` globals with pointer slots —
//! emit an ELF relocatable object, link it with the system `cc`, and run
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
            attributes::IntegerAttr,
            op_interfaces::{OneRegionInterface, OneResultInterface},
            types::{IntegerType, Signedness},
        },
        llvm::{
            attributes::{ICmpPredicateAttr, LinkageAttr},
            ops::{
                AddOp, AddressOfOp, CallOp, CondBrOp, FuncOp, GlobalOp, ICmpOp, LoadOp,
                ReturnOp, StoreOp,
            },
            types::{FuncType, PointerType},
        },
        macho,
    },
    ir::{basic_block::BasicBlock, op::Op, value::Value},
    linked_list::ContainsLinkedList,
    ll::{DataAttr, DataReloc},
    passes::aarch64_linux,
    utils::apint::APInt,
};
#[allow(unused_imports)]
use pliron_llvm::op_interfaces::IntBinArithOpWithOverflowFlag as _;

fn context() -> Context {
    let mut ctx = Context::new();
    aarch64::register(&mut ctx);
    macho::register(&mut ctx);
    ctx
}

fn i64_ty(ctx: &mut Context) -> pliron_ll::r#type::TypeHandle {
    IntegerType::get(ctx, 64, Signedness::Signless).into()
}

fn i64_const(ctx: &mut Context, block: pliron_ll::context::Ptr<BasicBlock>, value: u64) -> Value {
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

/// Define an external `main() -> i64` in `body` and return it with its entry
/// block.
fn main_func(
    ctx: &mut Context,
    body: pliron_ll::context::Ptr<BasicBlock>,
) -> (FuncOp, pliron_ll::context::Ptr<BasicBlock>) {
    let i64_ty = i64_ty(ctx);
    let func_ty = FuncType::get(ctx, i64_ty, vec![], false);
    let func = FuncOp::new(ctx, "main".try_into().unwrap(), func_ty);
    func.set_attr_llvm_function_linkage(ctx, LinkageAttr::ExternalLinkage);
    func.get_or_create_entry_block(ctx);
    func.get_operation().insert_at_back(body, ctx);
    let entry = func.get_entry_block(ctx).unwrap();
    (func, entry)
}

/// Define an external global initialized from `data`.
fn data_global(
    ctx: &mut Context,
    body: pliron_ll::context::Ptr<BasicBlock>,
    name: &str,
    ty: pliron_ll::r#type::TypeHandle,
    data: DataAttr,
) {
    let global = GlobalOp::new(ctx, name.try_into().unwrap(), ty);
    global.set_attr_llvm_global_linkage(ctx, LinkageAttr::ExternalLinkage);
    pliron_ll::ll::set_global_data(ctx, &global, data);
    global.get_operation().insert_at_back(body, ctx);
}

/// Define an external thread-local global initialized from `data` (an
/// `ll.data` initializer plus the `ll.tls` marker).
fn tls_global(
    ctx: &mut Context,
    body: pliron_ll::context::Ptr<BasicBlock>,
    name: &str,
    ty: pliron_ll::r#type::TypeHandle,
    data: DataAttr,
) {
    let global = GlobalOp::new(ctx, name.try_into().unwrap(), ty);
    global.set_attr_llvm_global_linkage(ctx, LinkageAttr::ExternalLinkage);
    pliron_ll::ll::set_global_data(ctx, &global, data);
    pliron_ll::ll::set_global_thread_local(ctx, &global);
    global.get_operation().insert_at_back(body, ctx);
}

/// Take the address of `name` in `block`, yielding a pointer value.
fn address_of(
    ctx: &mut Context,
    block: pliron_ll::context::Ptr<BasicBlock>,
    name: &str,
) -> Value {
    let addr = AddressOfOp::new(ctx, name.try_into().unwrap(), 0);
    addr.get_operation().insert_at_back(block, ctx);
    addr.get_result(ctx)
}

fn load(
    ctx: &mut Context,
    block: pliron_ll::context::Ptr<BasicBlock>,
    ptr: Value,
    ty: pliron_ll::r#type::TypeHandle,
) -> Value {
    let load = LoadOp::new(ctx, ptr, ty);
    load.get_operation().insert_at_back(block, ctx);
    load.get_result(ctx)
}

/// Link `object_bytes` and run the result, returning the exit status. The
/// object provides `main` itself.
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

/// Link `object_bytes` together with a C `main` from `harness_source` (built
/// with `-pthread`) and run the result, returning the exit status.
fn link_with_c_harness_and_run(test: &str, object_bytes: &[u8], harness_source: &str) -> i32 {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(test);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let object = dir.join("test.o");
    let harness = dir.join("main.c");
    let binary = dir.join("test");
    std::fs::write(&object, object_bytes).expect("write object");
    std::fs::write(&harness, harness_source).expect("write harness");

    let link = Command::new("cc")
        .arg("-pthread")
        .arg("-o")
        .arg(&binary)
        .arg(&harness)
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

#[test]
fn loads_a_rodata_global() {
    // static ANSWER: u64 = 42; main() { ANSWER }
    let mut ctx = context();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
    let i64_ty = i64_ty(&mut ctx);
    data_global(
        &mut ctx,
        body,
        "answer",
        i64_ty,
        DataAttr {
            bytes: 42u64.to_le_bytes().to_vec(),
            align: 8,
            mutable: false,
            relocs: vec![],
        },
    );
    let (_main, entry) = main_func(&mut ctx, body);
    let addr = address_of(&mut ctx, entry, "answer");
    let value = load(&mut ctx, entry, addr, i64_ty);
    ReturnOp::new(&mut ctx, Some(value))
        .get_operation()
        .insert_at_back(entry, &ctx);

    let bytes = aarch64_linux::emit_elf_object_bytes(&mut ctx, module.get_operation()).unwrap();
    assert_eq!(link_and_run("loads_a_rodata_global", &bytes), 42);
}

#[test]
fn stores_through_a_data_global() {
    // static mut COUNTER: u64 = 11; main() { COUNTER += 22; COUNTER }
    let mut ctx = context();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
    let i64_ty = i64_ty(&mut ctx);
    data_global(
        &mut ctx,
        body,
        "counter",
        i64_ty,
        DataAttr {
            bytes: 11u64.to_le_bytes().to_vec(),
            align: 8,
            mutable: true,
            relocs: vec![],
        },
    );
    let (_main, entry) = main_func(&mut ctx, body);
    let addr = address_of(&mut ctx, entry, "counter");
    let value = load(&mut ctx, entry, addr, i64_ty);
    let increment = i64_const(&mut ctx, entry, 22);
    let sum = AddOp::new_with_overflow_flag(&mut ctx, value, increment, Default::default());
    sum.get_operation().insert_at_back(entry, &ctx);
    let sum_result = sum.get_result(&ctx);
    StoreOp::new(&mut ctx, sum_result, addr)
        .get_operation()
        .insert_at_back(entry, &ctx);
    let reloaded = load(&mut ctx, entry, addr, i64_ty);
    ReturnOp::new(&mut ctx, Some(reloaded))
        .get_operation()
        .insert_at_back(entry, &ctx);

    let bytes = aarch64_linux::emit_elf_object_bytes(&mut ctx, module.get_operation()).unwrap();
    assert_eq!(link_and_run("stores_through_a_data_global", &bytes), 33);
}

#[test]
fn follows_a_pointer_slot_through_a_relocation() {
    // static TARGET: [u64; 2] = [66, 77];
    // static HOLDER: &u64 = &TARGET[1];   // pointer slot: TARGET + 8
    // main() { **&HOLDER }
    let mut ctx = context();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
    let i64_ty = i64_ty(&mut ctx);
    let ptr_ty: pliron_ll::r#type::TypeHandle = PointerType::get(&mut ctx, 0).into();
    let mut target_bytes = 66u64.to_le_bytes().to_vec();
    target_bytes.extend_from_slice(&77u64.to_le_bytes());
    data_global(
        &mut ctx,
        body,
        "target",
        i64_ty,
        DataAttr {
            bytes: target_bytes,
            align: 8,
            mutable: false,
            relocs: vec![],
        },
    );
    data_global(
        &mut ctx,
        body,
        "holder",
        ptr_ty,
        DataAttr {
            bytes: vec![0; 8],
            align: 8,
            mutable: false,
            relocs: vec![DataReloc {
                offset: 0,
                symbol: "target".to_string(),
                addend: 8,
            }],
        },
    );
    let (_main, entry) = main_func(&mut ctx, body);
    let holder_addr = address_of(&mut ctx, entry, "holder");
    let target_addr = load(&mut ctx, entry, holder_addr, ptr_ty);
    let value = load(&mut ctx, entry, target_addr, i64_ty);
    ReturnOp::new(&mut ctx, Some(value))
        .get_operation()
        .insert_at_back(entry, &ctx);

    let bytes = aarch64_linux::emit_elf_object_bytes(&mut ctx, module.get_operation()).unwrap();
    assert_eq!(
        link_and_run("follows_a_pointer_slot_through_a_relocation", &bytes),
        77
    );
}

#[test]
fn passes_global_addresses_as_values() {
    // static VALUE: u64 = 55;
    // helper(p: &u64) -> u64 { *p }
    // main() { let a = &VALUE; let b = &VALUE; if a == b { helper(a) } else { 1 } }
    let mut ctx = context();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
    let i64_ty = i64_ty(&mut ctx);
    let ptr_ty: pliron_ll::r#type::TypeHandle = PointerType::get(&mut ctx, 0).into();
    data_global(
        &mut ctx,
        body,
        "value",
        i64_ty,
        DataAttr {
            bytes: 55u64.to_le_bytes().to_vec(),
            align: 8,
            mutable: false,
            relocs: vec![],
        },
    );

    let helper_ty = FuncType::get(&mut ctx, i64_ty, vec![ptr_ty], false);
    let helper = FuncOp::new(&mut ctx, "helper".try_into().unwrap(), helper_ty);
    helper.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
    helper.get_or_create_entry_block(&mut ctx);
    helper.get_operation().insert_at_back(body, &ctx);
    let helper_entry = helper.get_entry_block(&ctx).unwrap();
    let arg = helper_entry.deref(&ctx).get_argument(0);
    let loaded = load(&mut ctx, helper_entry, arg, i64_ty);
    ReturnOp::new(&mut ctx, Some(loaded))
        .get_operation()
        .insert_at_back(helper_entry, &ctx);

    let (main, entry) = main_func(&mut ctx, body);
    let then_block = BasicBlock::new(&mut ctx, Some("then".try_into().unwrap()), vec![]);
    then_block.insert_at_back(main.get_region(&ctx).unwrap(), &ctx);
    let else_block = BasicBlock::new(&mut ctx, Some("else".try_into().unwrap()), vec![]);
    else_block.insert_at_back(main.get_region(&ctx).unwrap(), &ctx);

    let a = address_of(&mut ctx, entry, "value");
    let b = address_of(&mut ctx, entry, "value");
    let same = ICmpOp::new(&mut ctx, ICmpPredicateAttr::EQ, a, b);
    same.get_operation().insert_at_back(entry, &ctx);
    let same_result = same.get_result(&ctx);
    CondBrOp::new(&mut ctx, same_result, then_block, vec![], else_block, vec![])
        .get_operation()
        .insert_at_back(entry, &ctx);

    let call_ty = FuncType::get(&mut ctx, i64_ty, vec![ptr_ty], false);
    let call = CallOp::new(
        &mut ctx,
        pliron::builtin::op_interfaces::CallOpCallable::Direct("helper".try_into().unwrap()),
        call_ty,
        vec![a],
    );
    call.get_operation().insert_at_back(then_block, &ctx);
    let call_result = call.get_operation().deref(&ctx).get_result(0);
    ReturnOp::new(&mut ctx, Some(call_result))
        .get_operation()
        .insert_at_back(then_block, &ctx);

    let one = i64_const(&mut ctx, else_block, 1);
    ReturnOp::new(&mut ctx, Some(one))
        .get_operation()
        .insert_at_back(else_block, &ctx);

    let bytes = aarch64_linux::emit_elf_object_bytes(&mut ctx, module.get_operation()).unwrap();
    assert_eq!(link_and_run("passes_global_addresses_as_values", &bytes), 55);
}

#[test]
fn reads_and_writes_a_thread_local_global() {
    // #[thread_local] static COUNTER: u64 = 7;
    // main() { COUNTER += 35; COUNTER }
    let mut ctx = context();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
    let i64_ty = i64_ty(&mut ctx);
    tls_global(
        &mut ctx,
        body,
        "tls_counter",
        i64_ty,
        DataAttr {
            bytes: 7u64.to_le_bytes().to_vec(),
            align: 8,
            mutable: true,
            relocs: vec![],
        },
    );
    let (_main, entry) = main_func(&mut ctx, body);
    let addr = address_of(&mut ctx, entry, "tls_counter");
    let value = load(&mut ctx, entry, addr, i64_ty);
    let increment = i64_const(&mut ctx, entry, 35);
    let sum = AddOp::new_with_overflow_flag(&mut ctx, value, increment, Default::default());
    sum.get_operation().insert_at_back(entry, &ctx);
    let sum_result = sum.get_result(&ctx);
    StoreOp::new(&mut ctx, sum_result, addr)
        .get_operation()
        .insert_at_back(entry, &ctx);
    let reloaded = load(&mut ctx, entry, addr, i64_ty);
    ReturnOp::new(&mut ctx, Some(reloaded))
        .get_operation()
        .insert_at_back(entry, &ctx);

    let bytes = aarch64_linux::emit_elf_object_bytes(&mut ctx, module.get_operation()).unwrap();
    assert_eq!(link_and_run("reads_and_writes_a_thread_local_global", &bytes), 42);
}

/// Define an external `name() -> i64` that loads the thread-local `global`,
/// adds one, stores it back, and returns the new value.
fn tls_bump_func(
    ctx: &mut Context,
    body: pliron_ll::context::Ptr<BasicBlock>,
    name: &str,
    global: &str,
) {
    let i64_ty = i64_ty(ctx);
    let func_ty = FuncType::get(ctx, i64_ty, vec![], false);
    let func = FuncOp::new(ctx, name.try_into().unwrap(), func_ty);
    func.set_attr_llvm_function_linkage(ctx, LinkageAttr::ExternalLinkage);
    func.get_or_create_entry_block(ctx);
    func.get_operation().insert_at_back(body, ctx);
    let entry = func.get_entry_block(ctx).unwrap();
    let addr = address_of(ctx, entry, global);
    let value = load(ctx, entry, addr, i64_ty);
    let one = i64_const(ctx, entry, 1);
    let sum = AddOp::new_with_overflow_flag(ctx, value, one, Default::default());
    sum.get_operation().insert_at_back(entry, ctx);
    let sum_result = sum.get_result(ctx);
    StoreOp::new(ctx, sum_result, addr)
        .get_operation()
        .insert_at_back(entry, ctx);
    let reloaded = load(ctx, entry, addr, i64_ty);
    ReturnOp::new(ctx, Some(reloaded))
        .get_operation()
        .insert_at_back(entry, ctx);
}

#[test]
fn thread_local_copies_are_independent() {
    // #[thread_local] static COUNTER: u64 = 7;   // .tdata
    // #[thread_local] static ZEROED: u64 = 0;    // .tbss
    // bump() { COUNTER += 1; COUNTER }  zbump() { ZEROED += 1; ZEROED }
    // A two-pthread C harness asserts every thread starts from a fresh
    // 7/0 copy and never observes another thread's increments.
    let mut ctx = context();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
    let i64_ty = i64_ty(&mut ctx);
    tls_global(
        &mut ctx,
        body,
        "tls_counter",
        i64_ty,
        DataAttr {
            bytes: 7u64.to_le_bytes().to_vec(),
            align: 8,
            mutable: true,
            relocs: vec![],
        },
    );
    tls_global(
        &mut ctx,
        body,
        "tls_zeroed",
        i64_ty,
        DataAttr {
            bytes: vec![0; 8],
            align: 8,
            mutable: true,
            relocs: vec![],
        },
    );
    tls_bump_func(&mut ctx, body, "bump", "tls_counter");
    tls_bump_func(&mut ctx, body, "zbump", "tls_zeroed");

    let bytes = aarch64_linux::emit_elf_object_bytes(&mut ctx, module.get_operation()).unwrap();
    const HARNESS: &str = r#"
#include <pthread.h>
#include <stdint.h>
extern uint64_t bump(void);
extern uint64_t zbump(void);
static void *worker(void *arg) {
    uint64_t *results = arg;
    for (int i = 0; i < 3; i++) results[i] = bump();
    results[3] = zbump();
    return 0;
}
int main(void) {
    uint64_t a[4], b[4];
    pthread_t t1, t2;
    if (pthread_create(&t1, 0, worker, a) != 0) return 10;
    if (pthread_create(&t2, 0, worker, b) != 0) return 11;
    pthread_join(t1, 0);
    pthread_join(t2, 0);
    /* each thread bumped its own fresh copies: 7 -> 8,9,10 and 0 -> 1 */
    if (a[0] != 8 || a[1] != 9 || a[2] != 10 || a[3] != 1) return 1;
    if (b[0] != 8 || b[1] != 9 || b[2] != 10 || b[3] != 1) return 2;
    /* the main thread's copies were untouched by both workers */
    if (bump() != 8) return 3;
    if (zbump() != 1) return 4;
    return 0;
}
"#;
    assert_eq!(
        link_with_c_harness_and_run("thread_local_copies_are_independent", &bytes, HARNESS),
        0
    );
}

#[test]
fn text_only_modules_keep_working() {
    // No data globals: the ELF path must still produce a working object.
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

    let bytes = aarch64_linux::emit_elf_object_bytes(&mut ctx, module.get_operation()).unwrap();
    assert_eq!(link_and_run("text_only_modules_keep_working", &bytes), 42);
}
