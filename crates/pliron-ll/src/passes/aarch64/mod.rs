pub mod aarch64_asm_lower;
pub mod aarch64_block_placement;
pub mod aarch64_branch_relax;
pub mod aarch64_encode;
pub mod aarch64_frame_lower;
pub mod aarch64_legalize;
pub mod aarch64_machine_cfg_cleanup;
pub mod aarch64_object_lower;
pub mod opmap;
pub mod aarch64_post_ra_opts;
pub mod aarch64_register_allocate;
pub mod aarch64_target_opts_pre_ra;
mod attrs;
pub mod blockmap;
mod elf;
mod error;
mod frontend;
mod isel_control_flow;
mod isel_i128;
mod isel_memory_abi;
pub mod llvm_aarch64_abi;
pub mod llvm_to_aarch64_isel;
mod macho;
pub mod target;
mod util;
pub mod verify_llvm_for_aarch64;

use crate::{
    context::{Context, Ptr},
    ir::operation::Operation,
    conversion::pass::{AnalysisManager, Pass, Passes},
    result::CrabbitResult,
};

use self::{
    aarch64_asm_lower::Aarch64AsmLowerPass, aarch64_block_placement::Aarch64BlockPlacementPass,
    aarch64_branch_relax::Aarch64BranchRelaxPass, aarch64_encode::Aarch64EncodePass,
    aarch64_frame_lower::Aarch64FrameLowerPass, aarch64_legalize::Aarch64LegalizePass,
    aarch64_machine_cfg_cleanup::Aarch64MachineCfgCleanupPass,
    aarch64_object_lower::{aarch64_macho_lower, collect_object_parts},
    aarch64_post_ra_opts::Aarch64PostRaOptsPass,
    aarch64_register_allocate::Aarch64RegisterAllocatePass,
    aarch64_target_opts_pre_ra::Aarch64TargetOptsPreRaPass,
    llvm_aarch64_abi::LlvmAarch64AbiPass,
    llvm_to_aarch64_isel::LlvmToAarch64IselPass,
    verify_llvm_for_aarch64::VerifyLlvmForAarch64Pass,
};
pub use self::target::TargetOs;

/// The aarch64 lowering pipeline for `os`: every step is a [Pass] on the
/// `builtin.module`, from LLVM-dialect verification down to encoded machine
/// code. Only verification and ABI assignment differ per OS; the machine
/// passes are shared. Translation to object-container bytes happens outside
/// the pipeline, in [write_macho_object_from_ir] / [write_elf_object_from_ir].
pub fn pipeline(os: TargetOs) -> Passes {
    pipeline_with_allocator(os, Aarch64RegisterAllocatePass)
}

/// [pipeline] with `allocator` in the register-allocation slot (between
/// `aarch64-target-opts-pre-ra` and `aarch64-frame-lower`). `allocator`
/// must satisfy [Aarch64RegisterAllocatePass]'s post-conditions: every
/// virtual register rewritten to a physical register or a spill-slot access
/// through the reserved scratch registers, and `FuncOp::stack_size` raised
/// by the spill area. This is the seam through which research allocators
/// (e.g. the eregalloc engine, which lives in a crate that depends on this
/// one and so cannot be named here) enter the pipeline.
pub fn pipeline_with_allocator(os: TargetOs, allocator: impl Pass + 'static) -> Passes {
    let mut passes = Passes::default();
    passes.add_pass(VerifyLlvmForAarch64Pass::new(os));
    passes.add_pass(LlvmAarch64AbiPass::new(os));
    // Stamp LLVM-level op ids at the LLVM→machine boundary so isel can
    // record what each machine op was lowered from (backward profile
    // attribution). No-op unless the blockmap/profile-map gate is set.
    passes.add_pass(opmap::Aarch64OpIdPass);
    passes.add_pass(LlvmToAarch64IselPass);
    passes.add_pass(Aarch64LegalizePass);
    passes.add_pass(Aarch64MachineCfgCleanupPass);
    passes.add_pass(Aarch64TargetOptsPreRaPass);
    // Stamp stable block ids at the RA position (the CFG the frequency
    // models describe) so the blockmap sidecar can map final .text ranges
    // back to RA-order blocks. No-op unless CRABBIT_BLOCKMAP is set.
    passes.add_pass(blockmap::Aarch64BlockmapIdPass);
    passes.add_pass(allocator);
    passes.add_pass(Aarch64FrameLowerPass);
    passes.add_pass(Aarch64PostRaOptsPass);
    passes.add_pass(Aarch64BlockPlacementPass);
    passes.add_pass(Aarch64BranchRelaxPass);
    passes.add_pass(Aarch64AsmLowerPass);
    passes.add_pass(Aarch64EncodePass);
    passes
}

/// Runs [pipeline] on `root` (a `builtin.module`) in place.
pub fn lower_module(ctx: &mut Context, root: Ptr<Operation>, os: TargetOs) -> CrabbitResult<()> {
    pipeline(os).run(root, ctx, &mut AnalysisManager::default())?;
    Ok(())
}

pub fn emit_macho_object_bytes(ctx: &mut Context, root: Ptr<Operation>) -> CrabbitResult<Vec<u8>> {
    lower_module(ctx, root, TargetOs::Darwin)?;
    write_macho_object_from_ir(ctx, root)
}

pub fn emit_elf_object_bytes(ctx: &mut Context, root: Ptr<Operation>) -> CrabbitResult<Vec<u8>> {
    lower_module(ctx, root, TargetOs::Linux)?;
    write_elf_object_from_ir(ctx, root)
}

/// Translates a module lowered by [pipeline] into Mach-O object bytes.
pub fn write_macho_object_from_ir(ctx: &mut Context, root: Ptr<Operation>) -> CrabbitResult<Vec<u8>> {
    let object = aarch64_macho_lower(ctx, root)?;
    Ok(macho::write_macho_object(ctx, object))
}

/// Translates a module lowered by [pipeline] into ELF object bytes.
pub fn write_elf_object_from_ir(ctx: &mut Context, root: Ptr<Operation>) -> CrabbitResult<Vec<u8>> {
    let parts = collect_object_parts(ctx, root, TargetOs::Linux)?;
    Ok(elf::write_elf_object(&parts))
}

#[cfg(test)]
mod tests {
    use crate::dialects::builtin::ops::ConstantOp;
    #[allow(unused_imports)]
    use pliron::builtin::op_interfaces::{
        AtMostOneRegionInterface as _, BranchOpInterface as _, CallOpInterface as _,
        SymbolOpInterface as _,
    };
    #[allow(unused_imports)]
    use pliron_llvm::op_interfaces::{
        BinArithOp as _, CastOpInterface as _, IntBinArithOpWithOverflowFlag as _,
    };
    use std::num::NonZero;

    use crate::{
        dialects::{
            aarch64, builtin,
            builtin::{
                attributes::IntegerAttr,
                op_interfaces::{OneRegionInterface, OneResultInterface},
                types::FP32Type,
            },
            llvm::{
                attributes::LinkageAttr,
                ops::GepIndex,
                ops::{
                    AddOp, AllocaOp, BrOp, CondBrOp, FuncOp, GetElementPtrOp, LoadOp,
                    ReturnOp, SDivOp, SRemOp, StoreOp, UDivOp,
                },
                types::{ArrayType, FuncType},
            },
            macho,
        },
        ir::{basic_block::BasicBlock, op::Op, value::Value},
        linked_list::ContainsLinkedList,
        r#type::TypeHandle,
        utils::apint::APInt,
    };

    use super::*;

    fn context() -> Context {
        let mut ctx = Context::new();
        aarch64::register(&mut ctx);
        macho::register(&mut ctx);
        ctx
    }

    #[test]
    fn passes_have_stable_names() {
        assert_eq!(
            VerifyLlvmForAarch64Pass::new(TargetOs::Darwin).name(),
            "verify-llvm-for-aarch64-darwin"
        );
        assert_eq!(
            VerifyLlvmForAarch64Pass::new(TargetOs::Linux).name(),
            "verify-llvm-for-aarch64-linux"
        );
        assert_eq!(
            LlvmAarch64AbiPass::new(TargetOs::Darwin).name(),
            "llvm-aarch64-darwin-abi"
        );
        assert_eq!(
            LlvmAarch64AbiPass::new(TargetOs::Linux).name(),
            "llvm-aarch64-linux-abi"
        );
        assert_eq!(LlvmToAarch64IselPass.name(), "llvm-to-aarch64-isel");
        assert_eq!(Aarch64LegalizePass.name(), "aarch64-legalize");
        assert_eq!(
            Aarch64MachineCfgCleanupPass.name(),
            "aarch64-machine-cfg-cleanup"
        );
        assert_eq!(
            Aarch64TargetOptsPreRaPass.name(),
            "aarch64-target-opts-pre-ra"
        );
        assert_eq!(
            blockmap::Aarch64BlockmapIdPass.name(),
            "aarch64-blockmap-ids"
        );
        assert_eq!(
            Aarch64RegisterAllocatePass.name(),
            "aarch64-register-allocate"
        );
        assert_eq!(Aarch64FrameLowerPass.name(), "aarch64-frame-lower");
        assert_eq!(Aarch64PostRaOptsPass.name(), "aarch64-post-ra-opts");
        assert_eq!(Aarch64BlockPlacementPass.name(), "aarch64-block-placement");
        assert_eq!(Aarch64BranchRelaxPass.name(), "aarch64-branch-relax");
        assert_eq!(Aarch64AsmLowerPass.name(), "aarch64-asm-lower");
        assert_eq!(Aarch64EncodePass.name(), "aarch64-encode");
    }

    #[test]
    fn emits_return_constant_object() {
        let mut ctx = context();
        let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
        let i64_ty =
            builtin::types::IntegerType::get(&mut ctx, 64, builtin::types::Signedness::Signless);
        let func_ty = FuncType::get(&mut ctx, i64_ty.into(), vec![], false);
        let func = FuncOp::new(&mut ctx, "main".try_into().unwrap(), func_ty);
        func.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(&mut ctx);
        func.get_operation().insert_at_back(body, &ctx);
        let entry = func.get_entry_block(&ctx).unwrap();
        let constant = ConstantOp::new(&mut ctx, Box::new(IntegerAttr::new(i64_ty, APInt::from_u64(7, NonZero::new(64).unwrap()))));
        constant.get_operation().insert_at_back(entry, &ctx);
        let ret_value = constant.get_result(&ctx);
        ReturnOp::new(&mut ctx, Some(ret_value))
            .get_operation()
            .insert_at_back(entry, &ctx);

        let bytes = emit_macho_object_bytes(&mut ctx, module.get_operation()).unwrap();
        assert_eq!(&bytes[0..4], &[0xcf, 0xfa, 0xed, 0xfe]);
        assert!(bytes.windows(5).any(|window| window == b"_main"));
    }

    #[test]
    fn lowers_scalar_fp_signature_through_the_pipeline() {
        // fp_identity(x: f32) -> f32 { x }: the argument arrives in s0 and
        // returns in s0.
        let mut ctx = context();
        let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
        let f32_ty = FP32Type::get(&ctx);
        let func_ty = FuncType::get(&mut ctx, f32_ty.into(), vec![f32_ty.into()], false);
        let func = FuncOp::new(&mut ctx, "fp_identity".try_into().unwrap(), func_ty);
        func.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(&mut ctx);
        func.get_operation().insert_at_back(body, &ctx);
        let entry = func.get_entry_block(&ctx).unwrap();
        let arg = entry.deref(&ctx).get_argument(0);
        ReturnOp::new(&mut ctx, Some(arg))
            .get_operation()
            .insert_at_back(entry, &ctx);

        lower_module(&mut ctx, module.get_operation(), TargetOs::Darwin).unwrap();
        let object = aarch64_macho_lower(&mut ctx, module.get_operation()).unwrap();
        assert_eq!(object.symbols(&ctx)[0].name, "_fp_identity");
        assert!(!object.text(&ctx).is_empty());
    }

    #[test]
    fn lowers_two_integer_args_to_aarch64_ir() {
        let mut ctx = context();
        let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
        let i64_ty =
            builtin::types::IntegerType::get(&mut ctx, 64, builtin::types::Signedness::Signless);
        let func_ty = FuncType::get(
            &mut ctx,
            i64_ty.into(),
            vec![i64_ty.into(), i64_ty.into()],
            false,
        );
        let func = FuncOp::new(&mut ctx, "add2".try_into().unwrap(), func_ty);
        func.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(&mut ctx);
        func.get_operation().insert_at_back(body, &ctx);
        let entry = func.get_entry_block(&ctx).unwrap();
        let args: Vec<_> = entry.deref(&ctx).arguments().collect();
        let add = AddOp::new_with_overflow_flag(&mut ctx, args[0], args[1], Default::default());
        add.get_operation().insert_at_back(entry, &ctx);
        let add_result = add.get_result(&ctx);
        ReturnOp::new(&mut ctx, Some(add_result))
            .get_operation()
            .insert_at_back(entry, &ctx);

        lower_module(&mut ctx, module.get_operation(), TargetOs::Darwin).unwrap();
        let object = aarch64_macho_lower(&mut ctx, module.get_operation()).unwrap();
        // Entry copies of the two incoming argument registers, the add, the
        // result move, and ret: five instructions.
        assert_eq!(object.text(&ctx).len(), 20);
        assert_eq!(object.symbols(&ctx)[0].name, "_add2");
    }

    #[test]
    fn lowers_ninth_integer_arg_from_stack() {
        let mut ctx = context();
        let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
        let i64_ty =
            builtin::types::IntegerType::get(&mut ctx, 64, builtin::types::Signedness::Signless);
        let func_ty = FuncType::get(&mut ctx, i64_ty.into(), vec![i64_ty.into(); 9], false);
        let func = FuncOp::new(&mut ctx, "ninth".try_into().unwrap(), func_ty);
        func.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(&mut ctx);
        func.get_operation().insert_at_back(body, &ctx);
        let entry = func.get_entry_block(&ctx).unwrap();
        let ninth_arg = entry.deref(&ctx).get_argument(8);
        ReturnOp::new(&mut ctx, Some(ninth_arg))
            .get_operation()
            .insert_at_back(entry, &ctx);

        lower_module(&mut ctx, module.get_operation(), TargetOs::Darwin).unwrap();
        let object = aarch64_macho_lower(&mut ctx, module.get_operation()).unwrap();
        assert_eq!(object.symbols(&ctx)[0].name, "_ninth");
        assert!(!object.text(&ctx).is_empty());
    }

    /// The `aarch64.func` the pipeline produced for `module`, after running
    /// the passes up to and including `last`.
    fn machine_function_after(
        ctx: &mut Context,
        module: builtin::ops::ModuleOp,
        mut passes: Passes,
    ) -> aarch64::ops::FuncOp {
        let root = module.get_operation();
        passes
            .run(root, ctx, &mut AnalysisManager::default())
            .unwrap();
        let machine_body = root
            .deref(ctx)
            .get_region(0)
            .deref(ctx)
            .get_head()
            .unwrap();
        machine_body
            .deref(ctx)
            .iter(ctx)
            .find_map(|op| util::cast_operation::<aarch64::ops::FuncOp>(ctx, op))
            .unwrap()
    }

    fn isel_passes() -> Passes {
        let mut passes = Passes::default();
        passes.add_pass(VerifyLlvmForAarch64Pass::new(TargetOs::Linux));
        passes.add_pass(LlvmAarch64AbiPass::new(TargetOs::Linux));
        passes.add_pass(LlvmToAarch64IselPass);
        passes
    }

    fn block_mnemonics(ctx: &Context, block: Ptr<BasicBlock>) -> Vec<&'static str> {
        block
            .deref(ctx)
            .iter(ctx)
            .filter_map(|op| aarch64::ops::mnemonic(ctx, op))
            .collect()
    }

    /// The mnemonic of the instruction in `block` defining `reg`.
    fn defining_mnemonic(
        ctx: &Context,
        block: Ptr<BasicBlock>,
        reg: aarch64::registers::Register,
    ) -> Option<&'static str> {
        block.deref(ctx).iter(ctx).find_map(|op| {
            (aarch64::ops::reg(ctx, op, aarch64::ops::ATTR_KEY_AARCH64_RD.as_ref()) == Some(reg))
                .then(|| aarch64::ops::mnemonic(ctx, op))
                .flatten()
        })
    }

    fn int_binary_module(
        ctx: &mut Context,
        width: u32,
        build: fn(&mut Context, Value, Value) -> Ptr<Operation>,
    ) -> builtin::ops::ModuleOp {
        let module = builtin::ops::ModuleOp::new(ctx, "test".try_into().unwrap());
        let body = module.get_region(ctx).deref(ctx).get_head().unwrap();
        let int_ty =
            builtin::types::IntegerType::get(ctx, width, builtin::types::Signedness::Signless);
        let func_ty = FuncType::get(ctx, int_ty.into(), vec![int_ty.into(), int_ty.into()], false);
        let func = FuncOp::new(ctx, "binop".try_into().unwrap(), func_ty);
        func.set_attr_llvm_function_linkage(ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(ctx);
        func.get_operation().insert_at_back(body, ctx);
        let entry = func.get_entry_block(ctx).unwrap();
        let args: Vec<_> = entry.deref(ctx).arguments().collect();
        let op = build(ctx, args[0], args[1]);
        op.insert_at_back(entry, ctx);
        let result = op.deref(ctx).get_result(0);
        ReturnOp::new(ctx, Some(result))
            .get_operation()
            .insert_at_back(entry, ctx);
        module
    }

    /// Signed division and remainder run on the 64-bit `sdiv`, so a
    /// narrower dividend/divisor must be sign-extended first: the
    /// zero-extended `-7i32 / 3` would otherwise give 1431655763. The
    /// sign-extension sequence ends in a `sub` (`(x ^ sign) - sign`); the
    /// unsigned forms only mask (`and`).
    #[test]
    fn narrow_signed_division_sign_extends_both_operands() {
        for width in [8u32, 16, 32] {
            for (build, div_mnemonic) in [
                (
                    (|ctx: &mut Context, l, r| SDivOp::new(ctx, l, r).get_operation())
                        as fn(&mut Context, Value, Value) -> Ptr<Operation>,
                    "sdiv",
                ),
                (
                    |ctx: &mut Context, l, r| SRemOp::new(ctx, l, r).get_operation(),
                    "sdiv",
                ),
            ] {
                let mut ctx = context();
                let module = int_binary_module(&mut ctx, width, build);
                let func = machine_function_after(&mut ctx, module, isel_passes());
                let entry = func.entry_block(&ctx);
                let div = entry
                    .deref(&ctx)
                    .iter(&ctx)
                    .find(|op| aarch64::ops::mnemonic(&ctx, *op) == Some(div_mnemonic))
                    .expect("signed division selects sdiv");
                for key in [
                    aarch64::ops::ATTR_KEY_AARCH64_RN.as_ref(),
                    aarch64::ops::ATTR_KEY_AARCH64_RM.as_ref(),
                ] {
                    let operand = aarch64::ops::reg(&ctx, div, key).unwrap();
                    assert_eq!(
                        defining_mnemonic(&ctx, entry, operand),
                        Some("sub"),
                        "i{width} {div_mnemonic} operand {key} is not sign-extended"
                    );
                }
            }
        }

        let mut ctx = context();
        let module = int_binary_module(&mut ctx, 32, |ctx, l, r| {
            UDivOp::new(ctx, l, r).get_operation()
        });
        let func = machine_function_after(&mut ctx, module, isel_passes());
        let entry = func.entry_block(&ctx);
        let div = entry
            .deref(&ctx)
            .iter(&ctx)
            .find(|op| aarch64::ops::mnemonic(&ctx, *op) == Some("udiv"))
            .unwrap();
        let rn = aarch64::ops::reg(&ctx, div, aarch64::ops::ATTR_KEY_AARCH64_RN.as_ref()).unwrap();
        assert_eq!(defining_mnemonic(&ctx, entry, rn), Some("and"));
    }

    /// A call with stack-passed arguments stores them into the outgoing
    /// area reserved at the bottom of the caller's frame; sp never moves
    /// around the call (a spill reload between the argument stores and the
    /// `bl` would otherwise read through a shifted sp), and the callee
    /// loads them past its own frame and link-register save.
    #[test]
    fn ten_integer_args_use_the_reserved_outgoing_area() {
        let mut ctx = context();
        let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
        let i64_ty =
            builtin::types::IntegerType::get(&mut ctx, 64, builtin::types::Signedness::Signless);
        let func_ty = FuncType::get(&mut ctx, i64_ty.into(), vec![i64_ty.into(); 10], false);

        // callee(a0..a9) -> a9 + a8: forces both stack arguments live past a
        // call so the loads and the frame interact.
        let callee = FuncOp::new(&mut ctx, "callee".try_into().unwrap(), func_ty);
        callee.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
        callee.get_or_create_entry_block(&mut ctx);
        callee.get_operation().insert_at_back(body, &ctx);
        let callee_entry = callee.get_entry_block(&ctx).unwrap();
        let callee_args: Vec<_> = callee_entry.deref(&ctx).arguments().collect();
        let sum = AddOp::new_with_overflow_flag(
            &mut ctx,
            callee_args[8],
            callee_args[9],
            Default::default(),
        );
        sum.get_operation().insert_at_back(callee_entry, &ctx);
        let sum_result = sum.get_result(&ctx);
        ReturnOp::new(&mut ctx, Some(sum_result))
            .get_operation()
            .insert_at_back(callee_entry, &ctx);

        // caller(a0..a9) -> callee(a0..a9)
        let caller = FuncOp::new(&mut ctx, "caller".try_into().unwrap(), func_ty);
        caller.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
        caller.get_or_create_entry_block(&mut ctx);
        caller.get_operation().insert_at_back(body, &ctx);
        let caller_entry = caller.get_entry_block(&ctx).unwrap();
        let caller_args: Vec<_> = caller_entry.deref(&ctx).arguments().collect();
        let call = crate::dialects::llvm::ops::CallOp::new(
            &mut ctx,
            pliron::builtin::op_interfaces::CallOpCallable::Direct("callee".try_into().unwrap()),
            func_ty,
            caller_args,
        );
        call.get_operation().insert_at_back(caller_entry, &ctx);
        let call_result = call.get_operation().deref(&ctx).get_result(0);
        ReturnOp::new(&mut ctx, Some(call_result))
            .get_operation()
            .insert_at_back(caller_entry, &ctx);

        let mut passes = isel_passes();
        passes.add_pass(Aarch64LegalizePass);
        passes.add_pass(Aarch64MachineCfgCleanupPass);
        passes.add_pass(Aarch64TargetOptsPreRaPass);
        passes.add_pass(Aarch64RegisterAllocatePass);
        passes.add_pass(Aarch64FrameLowerPass);
        let root = module.get_operation();
        passes
            .run(root, &mut ctx, &mut AnalysisManager::default())
            .unwrap();
        let machine_body = root
            .deref(&ctx)
            .get_region(0)
            .deref(&ctx)
            .get_head()
            .unwrap();
        let machine_funcs: Vec<_> = machine_body
            .deref(&ctx)
            .iter(&ctx)
            .filter_map(|op| util::cast_operation::<aarch64::ops::FuncOp>(&ctx, op))
            .collect();
        let machine_callee = machine_funcs
            .iter()
            .find(|func| func.get_symbol_name(&ctx).to_string() == "callee")
            .unwrap();
        let machine_caller = machine_funcs
            .iter()
            .find(|func| func.get_symbol_name(&ctx).to_string() == "caller")
            .unwrap();

        // Caller: the frame covers the 16-byte outgoing area, the prologue
        // is the only sp adjustment before the `bl`, and the two stack
        // arguments are stored at sp+0 and sp+8 right before the call.
        assert!(machine_caller.stack_size(&ctx) >= 16);
        let caller_entry = machine_caller.entry_block(&ctx);
        let caller_insts: Vec<_> = caller_entry.deref(&ctx).iter(&ctx).collect();
        let bl_index = caller_insts
            .iter()
            .position(|op| aarch64::ops::mnemonic(&ctx, *op) == Some("call"))
            .expect("caller emits call");
        let sp_adjusts_before_call = caller_insts[..bl_index]
            .iter()
            .filter(|op| {
                matches!(
                    aarch64::ops::mnemonic(&ctx, **op),
                    Some("sub_sp_imm") | Some("add_sp_imm")
                )
            })
            .count();
        assert_eq!(sp_adjusts_before_call, 1, "only the prologue adjusts sp before the call");
        let outgoing_stores: Vec<u64> = caller_insts[..bl_index]
            .iter()
            .filter(|op| aarch64::ops::mnemonic(&ctx, **op) == Some("str_sp_offset"))
            .filter_map(|op| aarch64::ops::imm(&ctx, *op))
            .filter(|imm| *imm < 16)
            .collect();
        assert_eq!(outgoing_stores, vec![0, 8]);

        // Callee: after frame lowering the stack argument loads are plain
        // sp-relative loads rebased past the frame; `ldr_stack_arg` is gone.
        let callee_entry = machine_callee.entry_block(&ctx);
        let callee_mnemonics = block_mnemonics(&ctx, callee_entry);
        assert!(!callee_mnemonics.contains(&"ldr_stack_arg"), "{callee_mnemonics:?}");
        let frame = machine_callee.stack_size(&ctx);
        let arg_loads: Vec<u64> = callee_entry
            .deref(&ctx)
            .iter(&ctx)
            .filter(|op| aarch64::ops::mnemonic(&ctx, *op) == Some("ldr_sp_offset"))
            .filter_map(|op| aarch64::ops::imm(&ctx, op))
            .filter(|imm| *imm >= frame)
            .collect();
        assert_eq!(arg_loads, vec![frame, frame + 8]);
    }

    /// `llvm_<op>_f{32,64}` calls select the FP instruction instead of a
    /// call.
    #[test]
    fn fp_math_intrinsic_calls_select_fp_instructions() {
        for (name, mnemonic, is_f32, binary) in [
            ("llvm_sqrt_f64", "fsqrt_d", false, false),
            ("llvm_sqrt_f32", "fsqrt_s", true, false),
            ("llvm_fabs_f64", "fabs_d", false, false),
            ("llvm_floor_f32", "frintm_s", true, false),
            ("llvm_ceil_f64", "frintp_d", false, false),
            ("llvm_trunc_f64", "frintz_d", false, false),
            ("llvm_round_f64", "frinta_d", false, false),
            ("llvm_rint_f32", "frintn_s", true, false),
            ("llvm_minnum_f64", "fminnm_d", false, true),
            ("llvm_maxnum_f32", "fmaxnm_s", true, true),
            ("llvm_minimum_f64", "fmin_d", false, true),
            ("llvm_maximum_f64", "fmax_d", false, true),
        ] {
            let mut ctx = context();
            let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
            let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
            let fp_ty: TypeHandle = if is_f32 {
                FP32Type::get(&ctx).into()
            } else {
                crate::dialects::builtin::types::FP64Type::get(&ctx).into()
            };
            let arity = if binary { 2 } else { 1 };
            let func_ty = FuncType::get(&mut ctx, fp_ty, vec![fp_ty; arity], false);
            let decl = FuncOp::new(&mut ctx, name.try_into().unwrap(), func_ty);
            decl.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
            decl.get_operation().insert_at_back(body, &ctx);

            let func = FuncOp::new(&mut ctx, "user".try_into().unwrap(), func_ty);
            func.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
            func.get_or_create_entry_block(&mut ctx);
            func.get_operation().insert_at_back(body, &ctx);
            let entry = func.get_entry_block(&ctx).unwrap();
            let args: Vec<_> = entry.deref(&ctx).arguments().collect();
            let call = crate::dialects::llvm::ops::CallOp::new(
                &mut ctx,
                pliron::builtin::op_interfaces::CallOpCallable::Direct(name.try_into().unwrap()),
                func_ty,
                args,
            );
            call.get_operation().insert_at_back(entry, &ctx);
            let result = call.get_operation().deref(&ctx).get_result(0);
            ReturnOp::new(&mut ctx, Some(result))
                .get_operation()
                .insert_at_back(entry, &ctx);

            let machine = machine_function_after(&mut ctx, module, isel_passes());
            let mnemonics = block_mnemonics(&ctx, machine.entry_block(&ctx));
            assert!(mnemonics.contains(&mnemonic), "{name}: {mnemonics:?}");
            assert!(!mnemonics.contains(&"call"), "{name}: {mnemonics:?}");
        }
        assert!(llvm_to_aarch64_isel::fp_math_intrinsic("llvm_exp2_f32").is_none());
        assert!(llvm_to_aarch64_isel::fp_math_intrinsic("sqrtf32").is_none());
    }

    #[test]
    fn lowers_scalar_alloca_load_store_to_stack_memory() {
        let mut ctx = context();
        let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
        let i64_ty =
            builtin::types::IntegerType::get(&mut ctx, 64, builtin::types::Signedness::Signless);
        let func_ty = FuncType::get(&mut ctx, i64_ty.into(), vec![i64_ty.into()], false);
        let func = FuncOp::new(&mut ctx, "slot".try_into().unwrap(), func_ty);
        func.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(&mut ctx);
        func.get_operation().insert_at_back(body, &ctx);
        let entry = func.get_entry_block(&ctx).unwrap();
        let arg = entry.deref(&ctx).get_argument(0);
        let one = ConstantOp::new(&mut ctx, Box::new(IntegerAttr::new(i64_ty, APInt::from_u64(1, NonZero::new(64).unwrap()))));
        one.get_operation().insert_at_back(entry, &ctx);
        let one_result = one.get_result(&ctx);
        let alloca = AllocaOp::new(&mut ctx, i64_ty.into(), one_result);
        alloca.get_operation().insert_at_back(entry, &ctx);
        let slot = alloca.get_result(&ctx);
        StoreOp::new(&mut ctx, arg, slot)
            .get_operation()
            .insert_at_back(entry, &ctx);
        let load = LoadOp::new(&mut ctx, slot, i64_ty.into());
        load.get_operation().insert_at_back(entry, &ctx);
        let loaded = load.get_result(&ctx);
        ReturnOp::new(&mut ctx, Some(loaded))
            .get_operation()
            .insert_at_back(entry, &ctx);

        lower_module(&mut ctx, module.get_operation(), TargetOs::Darwin).unwrap();
        let object = aarch64_macho_lower(&mut ctx, module.get_operation()).unwrap();
        assert_eq!(object.symbols(&ctx)[0].name, "_slot");
        assert!(!object.text(&ctx).is_empty());
    }

    #[test]
    fn lowers_non_fallthrough_branch() {
        let mut ctx = context();
        let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
        let i64_ty =
            builtin::types::IntegerType::get(&mut ctx, 64, builtin::types::Signedness::Signless);
        let func_ty = FuncType::get(&mut ctx, i64_ty.into(), vec![], false);
        let func = FuncOp::new(&mut ctx, "branchy".try_into().unwrap(), func_ty);
        func.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(&mut ctx);
        func.get_operation().insert_at_back(body, &ctx);
        let entry = func.get_entry_block(&ctx).unwrap();
        let second = BasicBlock::new(&mut ctx, Some("bb1".try_into().unwrap()), vec![]);
        second.insert_at_back(func.get_region(&ctx).unwrap(), &ctx);
        BrOp::new(&mut ctx, second, vec![])
            .get_operation()
            .insert_at_back(entry, &ctx);
        let constant = ConstantOp::new(&mut ctx, Box::new(IntegerAttr::new(i64_ty, APInt::from_u64(0, NonZero::new(64).unwrap()))));
        constant.get_operation().insert_at_back(second, &ctx);
        let constant_result = constant.get_result(&ctx);
        ReturnOp::new(&mut ctx, Some(constant_result))
            .get_operation()
            .insert_at_back(second, &ctx);

        lower_module(&mut ctx, module.get_operation(), TargetOs::Darwin).unwrap();
        let object = aarch64_macho_lower(&mut ctx, module.get_operation()).unwrap();
        assert_eq!(object.symbols(&ctx)[0].name, "_branchy");
        assert!(!object.text(&ctx).is_empty());
    }

    #[test]
    fn lowers_conditional_branch() {
        let mut ctx = context();
        let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
        let i1_ty =
            builtin::types::IntegerType::get(&mut ctx, 1, builtin::types::Signedness::Signless);
        let i64_ty =
            builtin::types::IntegerType::get(&mut ctx, 64, builtin::types::Signedness::Signless);
        let func_ty = FuncType::get(&mut ctx, i64_ty.into(), vec![], false);
        let func = FuncOp::new(&mut ctx, "cond".try_into().unwrap(), func_ty);
        func.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(&mut ctx);
        func.get_operation().insert_at_back(body, &ctx);
        let entry = func.get_entry_block(&ctx).unwrap();
        let then_block = BasicBlock::new(&mut ctx, Some("then".try_into().unwrap()), vec![]);
        then_block.insert_at_back(func.get_region(&ctx).unwrap(), &ctx);
        let else_block = BasicBlock::new(&mut ctx, Some("else".try_into().unwrap()), vec![]);
        else_block.insert_at_back(func.get_region(&ctx).unwrap(), &ctx);

        let cond = ConstantOp::new(&mut ctx, Box::new(IntegerAttr::new(i1_ty, APInt::from_u64(1, NonZero::new(1).unwrap()))));
        cond.get_operation().insert_at_back(entry, &ctx);
        let cond_result = cond.get_result(&ctx);
        CondBrOp::new(
            &mut ctx,
            cond_result,
            then_block,
            vec![],
            else_block,
            vec![],
        )
        .get_operation()
        .insert_at_back(entry, &ctx);

        let then_value = ConstantOp::new(&mut ctx, Box::new(IntegerAttr::new(i64_ty, APInt::from_u64(1, NonZero::new(64).unwrap()))));
        then_value.get_operation().insert_at_back(then_block, &ctx);
        let then_result = then_value.get_result(&ctx);
        ReturnOp::new(&mut ctx, Some(then_result))
            .get_operation()
            .insert_at_back(then_block, &ctx);

        let else_value = ConstantOp::new(&mut ctx, Box::new(IntegerAttr::new(i64_ty, APInt::from_u64(0, NonZero::new(64).unwrap()))));
        else_value.get_operation().insert_at_back(else_block, &ctx);
        let else_result = else_value.get_result(&ctx);
        ReturnOp::new(&mut ctx, Some(else_result))
            .get_operation()
            .insert_at_back(else_block, &ctx);

        lower_module(&mut ctx, module.get_operation(), TargetOs::Darwin).unwrap();
        let object = aarch64_macho_lower(&mut ctx, module.get_operation()).unwrap();
        assert_eq!(object.symbols(&ctx)[0].name, "_cond");
        assert!(!object.text(&ctx).is_empty());
    }

    #[test]
    fn lowers_branch_with_block_argument() {
        let mut ctx = context();
        let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
        let i64_ty =
            builtin::types::IntegerType::get(&mut ctx, 64, builtin::types::Signedness::Signless);
        let func_ty = FuncType::get(&mut ctx, i64_ty.into(), vec![], false);
        let func = FuncOp::new(&mut ctx, "block_arg".try_into().unwrap(), func_ty);
        func.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(&mut ctx);
        func.get_operation().insert_at_back(body, &ctx);
        let entry = func.get_entry_block(&ctx).unwrap();
        let target = BasicBlock::new(
            &mut ctx,
            Some("target".try_into().unwrap()),
            vec![i64_ty.into()],
        );
        target.insert_at_back(func.get_region(&ctx).unwrap(), &ctx);

        let value = ConstantOp::new(&mut ctx, Box::new(IntegerAttr::new(i64_ty, APInt::from_u64(42, NonZero::new(64).unwrap()))));
        value.get_operation().insert_at_back(entry, &ctx);
        let value_result = value.get_result(&ctx);
        BrOp::new(&mut ctx, target, vec![value_result])
            .get_operation()
            .insert_at_back(entry, &ctx);
        let target_arg = target.deref(&ctx).get_argument(0);
        ReturnOp::new(&mut ctx, Some(target_arg))
            .get_operation()
            .insert_at_back(target, &ctx);

        lower_module(&mut ctx, module.get_operation(), TargetOs::Darwin).unwrap();
        let object = aarch64_macho_lower(&mut ctx, module.get_operation()).unwrap();
        assert_eq!(object.symbols(&ctx)[0].name, "_block_arg");
        assert!(!object.text(&ctx).is_empty());
    }

    #[test]
    fn lowers_conditional_branch_with_block_arguments() {
        let mut ctx = context();
        let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
        let i1_ty =
            builtin::types::IntegerType::get(&mut ctx, 1, builtin::types::Signedness::Signless);
        let i64_ty =
            builtin::types::IntegerType::get(&mut ctx, 64, builtin::types::Signedness::Signless);
        let func_ty = FuncType::get(&mut ctx, i64_ty.into(), vec![], false);
        let func = FuncOp::new(&mut ctx, "cond_block_args".try_into().unwrap(), func_ty);
        func.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(&mut ctx);
        func.get_operation().insert_at_back(body, &ctx);
        let entry = func.get_entry_block(&ctx).unwrap();
        let then_block = BasicBlock::new(
            &mut ctx,
            Some("then".try_into().unwrap()),
            vec![i64_ty.into()],
        );
        then_block.insert_at_back(func.get_region(&ctx).unwrap(), &ctx);
        let else_block = BasicBlock::new(
            &mut ctx,
            Some("else".try_into().unwrap()),
            vec![i64_ty.into()],
        );
        else_block.insert_at_back(func.get_region(&ctx).unwrap(), &ctx);

        let cond = ConstantOp::new(&mut ctx, Box::new(IntegerAttr::new(i1_ty, APInt::from_u64(1, NonZero::new(1).unwrap()))));
        cond.get_operation().insert_at_back(entry, &ctx);
        let true_value = ConstantOp::new(&mut ctx, Box::new(IntegerAttr::new(i64_ty, APInt::from_u64(7, NonZero::new(64).unwrap()))));
        true_value.get_operation().insert_at_back(entry, &ctx);
        let false_value = ConstantOp::new(&mut ctx, Box::new(IntegerAttr::new(i64_ty, APInt::from_u64(9, NonZero::new(64).unwrap()))));
        false_value.get_operation().insert_at_back(entry, &ctx);
        let cond_result = cond.get_result(&ctx);
        let true_result = true_value.get_result(&ctx);
        let false_result = false_value.get_result(&ctx);
        CondBrOp::new(
            &mut ctx,
            cond_result,
            then_block,
            vec![true_result],
            else_block,
            vec![false_result],
        )
        .get_operation()
        .insert_at_back(entry, &ctx);

        let then_arg = then_block.deref(&ctx).get_argument(0);
        ReturnOp::new(&mut ctx, Some(then_arg))
            .get_operation()
            .insert_at_back(then_block, &ctx);
        let else_arg = else_block.deref(&ctx).get_argument(0);
        ReturnOp::new(&mut ctx, Some(else_arg))
            .get_operation()
            .insert_at_back(else_block, &ctx);

        lower_module(&mut ctx, module.get_operation(), TargetOs::Darwin).unwrap();
        let object = aarch64_macho_lower(&mut ctx, module.get_operation()).unwrap();
        assert_eq!(object.symbols(&ctx)[0].name, "_cond_block_args");
        assert!(!object.text(&ctx).is_empty());
    }

    #[test]
    fn lowers_gep_to_register_address_load_store() {
        let mut ctx = context();
        let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
        let i64_ty =
            builtin::types::IntegerType::get(&mut ctx, 64, builtin::types::Signedness::Signless);
        let array_ty = ArrayType::get(&mut ctx, i64_ty.into(), 4);
        let func_ty = FuncType::get(&mut ctx, i64_ty.into(), vec![], false);
        let func = FuncOp::new(&mut ctx, "gep".try_into().unwrap(), func_ty);
        func.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(&mut ctx);
        func.get_operation().insert_at_back(body, &ctx);
        let entry = func.get_entry_block(&ctx).unwrap();

        let one = ConstantOp::new(&mut ctx, Box::new(IntegerAttr::new(i64_ty, APInt::from_u64(1, NonZero::new(64).unwrap()))));
        one.get_operation().insert_at_back(entry, &ctx);
        let one_result = one.get_result(&ctx);
        let slot = AllocaOp::new(&mut ctx, array_ty.into(), one_result);
        slot.get_operation().insert_at_back(entry, &ctx);
        let value = ConstantOp::new(&mut ctx, Box::new(IntegerAttr::new(i64_ty, APInt::from_u64(99, NonZero::new(64).unwrap()))));
        value.get_operation().insert_at_back(entry, &ctx);
        let slot_result = slot.get_result(&ctx);
        let elem = GetElementPtrOp::new(&mut ctx, slot_result, vec![GepIndex::Constant(2)], i64_ty.into());
        elem.get_operation().insert_at_back(entry, &ctx);
        let value_result = value.get_result(&ctx);
        let elem_result = elem.get_result(&ctx);
        StoreOp::new(&mut ctx, value_result, elem_result)
            .get_operation()
            .insert_at_back(entry, &ctx);
        let load = LoadOp::new(&mut ctx, elem_result, i64_ty.into());
        load.get_operation().insert_at_back(entry, &ctx);
        let load_result = load.get_result(&ctx);
        ReturnOp::new(&mut ctx, Some(load_result))
            .get_operation()
            .insert_at_back(entry, &ctx);

        lower_module(&mut ctx, module.get_operation(), TargetOs::Darwin).unwrap();
        let object = aarch64_macho_lower(&mut ctx, module.get_operation()).unwrap();
        assert_eq!(object.symbols(&ctx)[0].name, "_gep");
        assert!(!object.text(&ctx).is_empty());
    }

    #[test]
    fn block_placement_lays_out_weighted_hot_path_as_fallthrough() {
        use crate::common_traits::Named;
        use crate::ll::op_interfaces::WeightedBranchOpInterface;

        let mut ctx = context();
        let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
        let i1_ty =
            builtin::types::IntegerType::get(&mut ctx, 1, builtin::types::Signedness::Signless);
        let i64_ty =
            builtin::types::IntegerType::get(&mut ctx, 64, builtin::types::Signedness::Signless);
        let func_ty = FuncType::get(&mut ctx, i64_ty.into(), vec![], false);
        let func = FuncOp::new(&mut ctx, "biased".try_into().unwrap(), func_ty);
        func.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(&mut ctx);
        func.get_operation().insert_at_back(body, &ctx);
        let entry = func.get_entry_block(&ctx).unwrap();
        let then_block = BasicBlock::new(&mut ctx, Some("then".try_into().unwrap()), vec![]);
        then_block.insert_at_back(func.get_region(&ctx).unwrap(), &ctx);
        let else_block = BasicBlock::new(&mut ctx, Some("else".try_into().unwrap()), vec![]);
        else_block.insert_at_back(func.get_region(&ctx).unwrap(), &ctx);

        let cond = ConstantOp::new(&mut ctx, Box::new(IntegerAttr::new(i1_ty, APInt::from_u64(1, NonZero::new(1).unwrap()))));
        cond.get_operation().insert_at_back(entry, &ctx);
        let cond_result = cond.get_result(&ctx);
        let cond_br = CondBrOp::new(
            &mut ctx,
            cond_result,
            then_block,
            vec![],
            else_block,
            vec![],
        );
        // Profile-style weights: the true edge is cold, the false edge hot.
        cond_br.set_successor_weights(&ctx, vec![1, 2000]);
        cond_br.get_operation().insert_at_back(entry, &ctx);

        for (block, value) in [(then_block, 1u64), (else_block, 0u64)] {
            let constant = ConstantOp::new(&mut ctx, Box::new(IntegerAttr::new(i64_ty, APInt::from_u64(value, NonZero::new(64).unwrap()))));
            constant.get_operation().insert_at_back(block, &ctx);
            let result = constant.get_result(&ctx);
            ReturnOp::new(&mut ctx, Some(result))
                .get_operation()
                .insert_at_back(block, &ctx);
        }

        // Run the pipeline through block placement (post-RA layout, before
        // branch relaxation, as in LLVM).
        let mut prefix = Passes::default();
        prefix.add_pass(VerifyLlvmForAarch64Pass::new(TargetOs::Darwin));
        prefix.add_pass(LlvmAarch64AbiPass::new(TargetOs::Darwin));
        prefix.add_pass(LlvmToAarch64IselPass);
        prefix.add_pass(Aarch64LegalizePass);
        prefix.add_pass(Aarch64MachineCfgCleanupPass);
        prefix.add_pass(Aarch64TargetOptsPreRaPass);
        prefix.add_pass(Aarch64RegisterAllocatePass);
        prefix.add_pass(Aarch64FrameLowerPass);
        prefix.add_pass(Aarch64PostRaOptsPass);
        prefix.add_pass(Aarch64BlockPlacementPass);
        let root = module.get_operation();
        prefix
            .run(root, &mut ctx, &mut AnalysisManager::default())
            .unwrap();

        // The cold `then` block moved out of line; the hot `else` block now
        // falls through directly after the entry.
        let machine_body = root
            .deref(&ctx)
            .get_region(0)
            .deref(&ctx)
            .get_head()
            .unwrap();
        let machine_func = machine_body
            .deref(&ctx)
            .iter(&ctx)
            .find_map(|op| util::cast_operation::<aarch64::ops::FuncOp>(&ctx, op))
            .unwrap();
        let labels: Vec<String> = machine_func
            .get_region(&ctx)
            .deref(&ctx)
            .iter(&ctx)
            .map(|block| block.deref(&ctx).unique_name(&ctx).to_string())
            .collect();
        assert_eq!(labels.len(), 3);
        assert!(labels[1].starts_with("else"), "layout was {labels:?}");
        assert!(labels[2].starts_with("then"), "layout was {labels:?}");
        // The unconditional branch to the hot block became a fall-through;
        // only the conditional branch to the cold block remains.
        let machine_entry = machine_func.entry_block(&ctx);
        let tail = machine_entry.deref(&ctx).get_tail().unwrap();
        let cold_target = aarch64::ops::target(&ctx, tail).unwrap();
        assert!(
            cold_target
                .deref(&ctx)
                .unique_name(&ctx)
                .to_string()
                .starts_with("then")
        );
        assert!(aarch64::ops::branch_weights(&ctx, tail).is_some());

        // The rest of the pipeline still produces a valid MachO object.
        let mut suffix = Passes::default();
        suffix.add_pass(Aarch64BranchRelaxPass);
        suffix.add_pass(Aarch64AsmLowerPass);
        suffix.add_pass(Aarch64EncodePass);
        suffix
            .run(root, &mut ctx, &mut AnalysisManager::default())
            .unwrap();
        let bytes = write_macho_object_from_ir(&mut ctx, root).unwrap();
        assert_eq!(&bytes[0..4], &[0xcf, 0xfa, 0xed, 0xfe]);
    }

    #[test]
    fn macho_object_includes_build_version_command() {
        let mut ctx = context();
        let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
        let i64_ty =
            builtin::types::IntegerType::get(&mut ctx, 64, builtin::types::Signedness::Signless);
        let func_ty = FuncType::get(&mut ctx, i64_ty.into(), vec![], false);
        let func = FuncOp::new(&mut ctx, "main".try_into().unwrap(), func_ty);
        func.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(&mut ctx);
        func.get_operation().insert_at_back(body, &ctx);
        let entry = func.get_entry_block(&ctx).unwrap();
        let zero = ConstantOp::new(&mut ctx, Box::new(IntegerAttr::new(i64_ty, APInt::from_u64(0, NonZero::new(64).unwrap()))));
        zero.get_operation().insert_at_back(entry, &ctx);
        let zero_result = zero.get_result(&ctx);
        ReturnOp::new(&mut ctx, Some(zero_result))
            .get_operation()
            .insert_at_back(entry, &ctx);

        let bytes = emit_macho_object_bytes(&mut ctx, module.get_operation()).unwrap();
        assert_eq!(u32::from_le_bytes(bytes[16..20].try_into().unwrap()), 3);
        assert!(
            bytes
                .windows(4)
                .any(|window| u32::from_le_bytes(window.try_into().unwrap()) == 0x32)
        );
    }
}
