pub mod x86_64_asm_lower;
pub mod x86_64_block_placement;
pub mod x86_64_branch_relax;
pub mod x86_64_encode;
pub mod x86_64_frame_lower;
pub mod x86_64_legalize;
pub mod x86_64_machine_cfg_cleanup;
pub mod x86_64_macho_lower;
pub mod x86_64_post_ra_opts;
pub mod x86_64_register_allocate;
pub mod x86_64_target_opts_pre_ra;
mod attrs;
mod error;
mod frontend;
mod isel_control_flow;
mod isel_i128;
mod isel_memory_abi;
pub mod llvm_x86_64_darwin_abi;
pub mod llvm_to_x86_64_isel;
mod macho;
mod util;
pub mod verify_llvm_for_x86_64_darwin;

use std::sync::Arc;

use crate::{
    context::{Context, Ptr},
    dialects::macho::ops::ObjectOp,
    input_error_noloc,
    ir::operation::Operation,
    conversion::{
        pass::PassObj,
        pass_manager::{PassManager, Pipeline},
    },
    result::STAIRResult,
};

use self::{
    x86_64_asm_lower::X86_64AsmLowerPass, x86_64_block_placement::X86_64BlockPlacementPass,
    x86_64_branch_relax::X86_64BranchRelaxPass, x86_64_encode::X86_64EncodePass,
    x86_64_frame_lower::X86_64FrameLowerPass, x86_64_legalize::X86_64LegalizePass,
    x86_64_machine_cfg_cleanup::X86_64MachineCfgCleanupPass,
    x86_64_macho_lower::X86_64MachOLowerPass, x86_64_post_ra_opts::X86_64PostRaOptsPass,
    x86_64_register_allocate::X86_64RegisterAllocatePass,
    x86_64_target_opts_pre_ra::X86_64TargetOptsPreRaPass,
    llvm_x86_64_darwin_abi::LlvmX86_64DarwinAbiPass, llvm_to_x86_64_isel::LlvmToX86_64ISelPass,
    verify_llvm_for_x86_64_darwin::VerifyLlvmForX86_64DarwinPass,
};

pub fn pipeline() -> Pipeline {
    vec![
        Arc::new(VerifyLlvmForX86_64DarwinPass) as PassObj,
        Arc::new(LlvmX86_64DarwinAbiPass),
        Arc::new(LlvmToX86_64ISelPass),
        Arc::new(X86_64LegalizePass),
        Arc::new(X86_64MachineCfgCleanupPass),
        Arc::new(X86_64TargetOptsPreRaPass),
        Arc::new(X86_64RegisterAllocatePass),
        Arc::new(X86_64FrameLowerPass),
        Arc::new(X86_64PostRaOptsPass),
        Arc::new(X86_64BlockPlacementPass),
        Arc::new(X86_64BranchRelaxPass),
        Arc::new(X86_64AsmLowerPass),
        Arc::new(X86_64EncodePass),
        Arc::new(X86_64MachOLowerPass),
    ]
    .into()
}

pub fn lower_to_macho_ir(ctx: &mut Context, root: Ptr<Operation>) -> STAIRResult<Ptr<Operation>> {
    PassManager::without_callback().run(pipeline(), ctx, root)
}

pub fn emit_macho_object_bytes(ctx: &mut Context, root: Ptr<Operation>) -> STAIRResult<Vec<u8>> {
    let root = lower_to_macho_ir(ctx, root)?;
    write_macho_object_from_ir(ctx, root)
}

pub fn write_macho_object_from_ir(ctx: &Context, root: Ptr<Operation>) -> STAIRResult<Vec<u8>> {
    let Some(object) = util::cast_operation::<ObjectOp>(ctx, root) else {
        return Err(input_error_noloc!(error::X86_64DarwinErr::UnsupportedOp(
            Operation::get_opid(root, ctx).to_string()
        )));
    };
    Ok(macho::write_macho_object(ctx, object))
}

#[cfg(test)]
mod tests {
    use std::num::NonZero;

    use crate::{
        dialects::{
            x86_64, builtin,
            builtin::{
                attributes::IntegerAttr,
                op_interfaces::{OneRegionInterface, OneResultInterface},
                types::FP32Type,
            },
            llvm::{
                self,
                attributes::{GepIndexAttr, GepIndicesAttr},
                ops::{
                    AddOp, AllocaOp, BrOp, CondBrOp, ConstantOp, FuncOp, GetElementPtrOp, LoadOp,
                    ReturnOp, StoreOp,
                },
                types::{ArrayType, FuncType},
            },
            macho,
        },
        ir::{basic_block::BasicBlock, op::Op},
        linked_list::ContainsLinkedList,
        utils::apint::APInt,
    };

    use super::*;

    fn context() -> Context {
        let mut ctx = Context::new();
        llvm::register(&mut ctx);
        x86_64::register(&mut ctx);
        macho::register(&mut ctx);
        ctx
    }

    #[test]
    fn pipeline_has_stable_pass_order_and_names() {
        let pipeline = pipeline();
        let names: Vec<_> = pipeline
            .passes
            .iter()
            .map(|prepared| prepared.pass.name())
            .collect();
        assert_eq!(
            names,
            [
                "verify-llvm-for-x86-64-darwin",
                "llvm-x86-64-darwin-abi",
                "llvm-to-x86-64-isel",
                "x86-64-legalize",
                "x86-64-machine-cfg-cleanup",
                "x86-64-target-opts-pre-ra",
                "x86-64-register-allocate",
                "x86-64-frame-lower",
                "x86-64-post-ra-opts",
                "x86-64-block-placement",
                "x86-64-branch-relax",
                "x86-64-asm-lower",
                "x86-64-encode",
                "x86-64-macho-lower",
            ]
        );
    }

    #[test]
    fn emits_return_constant_object() {
        let mut ctx = context();
        let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
        let i64_ty =
            builtin::types::IntegerType::get(&mut ctx, 64, builtin::types::Signedness::Signless);
        let func_ty = FuncType::get(&mut ctx, i64_ty.into(), vec![], false);
        let func = FuncOp::new(
            &mut ctx,
            "main".try_into().unwrap(),
            func_ty,
            LinkageAttr::External,
        );
        func.get_operation().insert_at_back(body, &ctx);
        let entry = func.get_entry_block(&ctx);
        let constant = ConstantOp::new_integer(
            &mut ctx,
            IntegerAttr::new(i64_ty, APInt::from_u64(7, NonZero::new(64).unwrap())),
        );
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
    fn rejects_fp_abi_before_instruction_selection() {
        let mut ctx = context();
        let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
        let f32_ty = FP32Type::get(&ctx);
        let func_ty = FuncType::get(&mut ctx, f32_ty.into(), vec![f32_ty.into()], false);
        let func = FuncOp::new(
            &mut ctx,
            "fp_identity".try_into().unwrap(),
            func_ty,
            LinkageAttr::External,
        );
        func.get_operation().insert_at_back(body, &ctx);

        let err = match lower_to_macho_ir(&mut ctx, module.get_operation()) {
            Ok(_) => panic!("floating-point ABI signature unexpectedly lowered"),
            Err(err) => err,
        };
        assert!(
            err.to_string()
                .contains("floating-point ABI lowering is not implemented")
        );
    }

    #[test]
    fn lowers_two_integer_args_to_x86_64_ir() {
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
        let func = FuncOp::new(
            &mut ctx,
            "add2".try_into().unwrap(),
            func_ty,
            LinkageAttr::External,
        );
        func.get_operation().insert_at_back(body, &ctx);
        let entry = func.get_entry_block(&ctx);
        let args: Vec<_> = entry.deref(&ctx).arguments().collect();
        let add = AddOp::new(&mut ctx, args[0], args[1]);
        add.get_operation().insert_at_back(entry, &ctx);
        let add_result = add.get_result(&ctx);
        ReturnOp::new(&mut ctx, Some(add_result))
            .get_operation()
            .insert_at_back(entry, &ctx);

        let root = lower_to_macho_ir(&mut ctx, module.get_operation()).unwrap();
        let object = util::cast_operation::<ObjectOp>(&ctx, root).unwrap();
        // Five callee-saved pushes/pops (2 bytes each), the two entry copies
        // of incoming argument registers (3 each), the two-address add
        // expansion (mov+add, 6), the result move (3), and ret (1).
        assert_eq!(object.text(&ctx).len(), 36);
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
        let func = FuncOp::new(
            &mut ctx,
            "ninth".try_into().unwrap(),
            func_ty,
            LinkageAttr::External,
        );
        func.get_operation().insert_at_back(body, &ctx);
        let entry = func.get_entry_block(&ctx);
        let ninth_arg = entry.deref(&ctx).get_argument(8);
        ReturnOp::new(&mut ctx, Some(ninth_arg))
            .get_operation()
            .insert_at_back(entry, &ctx);

        let root = lower_to_macho_ir(&mut ctx, module.get_operation()).unwrap();
        let object = util::cast_operation::<ObjectOp>(&ctx, root).unwrap();
        assert_eq!(object.symbols(&ctx)[0].name, "_ninth");
        assert!(!object.text(&ctx).is_empty());
    }

    #[test]
    fn lowers_scalar_alloca_load_store_to_stack_memory() {
        let mut ctx = context();
        let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
        let i64_ty =
            builtin::types::IntegerType::get(&mut ctx, 64, builtin::types::Signedness::Signless);
        let func_ty = FuncType::get(&mut ctx, i64_ty.into(), vec![i64_ty.into()], false);
        let func = FuncOp::new(
            &mut ctx,
            "slot".try_into().unwrap(),
            func_ty,
            LinkageAttr::External,
        );
        func.get_operation().insert_at_back(body, &ctx);
        let entry = func.get_entry_block(&ctx);
        let arg = entry.deref(&ctx).get_argument(0);
        let one = ConstantOp::new_integer(
            &mut ctx,
            IntegerAttr::new(i64_ty, APInt::from_u64(1, NonZero::new(64).unwrap())),
        );
        one.get_operation().insert_at_back(entry, &ctx);
        let one_result = one.get_result(&ctx);
        let alloca = AllocaOp::new(&mut ctx, one_result, i64_ty.into());
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

        let root = lower_to_macho_ir(&mut ctx, module.get_operation()).unwrap();
        let object = util::cast_operation::<ObjectOp>(&ctx, root).unwrap();
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
        let func = FuncOp::new(
            &mut ctx,
            "branchy".try_into().unwrap(),
            func_ty,
            LinkageAttr::External,
        );
        func.get_operation().insert_at_back(body, &ctx);
        let entry = func.get_entry_block(&ctx);
        let second = BasicBlock::new(&mut ctx, Some("bb1".try_into().unwrap()), vec![]);
        second.insert_at_back(func.get_region(&ctx), &ctx);
        BrOp::new(&mut ctx, second, vec![])
            .get_operation()
            .insert_at_back(entry, &ctx);
        let constant = ConstantOp::new_integer(
            &mut ctx,
            IntegerAttr::new(i64_ty, APInt::from_u64(0, NonZero::new(64).unwrap())),
        );
        constant.get_operation().insert_at_back(second, &ctx);
        let constant_result = constant.get_result(&ctx);
        ReturnOp::new(&mut ctx, Some(constant_result))
            .get_operation()
            .insert_at_back(second, &ctx);

        let root = lower_to_macho_ir(&mut ctx, module.get_operation()).unwrap();
        let object = util::cast_operation::<ObjectOp>(&ctx, root).unwrap();
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
        let func = FuncOp::new(
            &mut ctx,
            "cond".try_into().unwrap(),
            func_ty,
            LinkageAttr::External,
        );
        func.get_operation().insert_at_back(body, &ctx);
        let entry = func.get_entry_block(&ctx);
        let then_block = BasicBlock::new(&mut ctx, Some("then".try_into().unwrap()), vec![]);
        then_block.insert_at_back(func.get_region(&ctx), &ctx);
        let else_block = BasicBlock::new(&mut ctx, Some("else".try_into().unwrap()), vec![]);
        else_block.insert_at_back(func.get_region(&ctx), &ctx);

        let cond = ConstantOp::new_integer(
            &mut ctx,
            IntegerAttr::new(i1_ty, APInt::from_u64(1, NonZero::new(1).unwrap())),
        );
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

        let then_value = ConstantOp::new_integer(
            &mut ctx,
            IntegerAttr::new(i64_ty, APInt::from_u64(1, NonZero::new(64).unwrap())),
        );
        then_value.get_operation().insert_at_back(then_block, &ctx);
        let then_result = then_value.get_result(&ctx);
        ReturnOp::new(&mut ctx, Some(then_result))
            .get_operation()
            .insert_at_back(then_block, &ctx);

        let else_value = ConstantOp::new_integer(
            &mut ctx,
            IntegerAttr::new(i64_ty, APInt::from_u64(0, NonZero::new(64).unwrap())),
        );
        else_value.get_operation().insert_at_back(else_block, &ctx);
        let else_result = else_value.get_result(&ctx);
        ReturnOp::new(&mut ctx, Some(else_result))
            .get_operation()
            .insert_at_back(else_block, &ctx);

        let root = lower_to_macho_ir(&mut ctx, module.get_operation()).unwrap();
        let object = util::cast_operation::<ObjectOp>(&ctx, root).unwrap();
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
        let func = FuncOp::new(
            &mut ctx,
            "block_arg".try_into().unwrap(),
            func_ty,
            LinkageAttr::External,
        );
        func.get_operation().insert_at_back(body, &ctx);
        let entry = func.get_entry_block(&ctx);
        let target = BasicBlock::new(
            &mut ctx,
            Some("target".try_into().unwrap()),
            vec![i64_ty.into()],
        );
        target.insert_at_back(func.get_region(&ctx), &ctx);

        let value = ConstantOp::new_integer(
            &mut ctx,
            IntegerAttr::new(i64_ty, APInt::from_u64(42, NonZero::new(64).unwrap())),
        );
        value.get_operation().insert_at_back(entry, &ctx);
        let value_result = value.get_result(&ctx);
        BrOp::new(&mut ctx, target, vec![value_result])
            .get_operation()
            .insert_at_back(entry, &ctx);
        let target_arg = target.deref(&ctx).get_argument(0);
        ReturnOp::new(&mut ctx, Some(target_arg))
            .get_operation()
            .insert_at_back(target, &ctx);

        let root = lower_to_macho_ir(&mut ctx, module.get_operation()).unwrap();
        let object = util::cast_operation::<ObjectOp>(&ctx, root).unwrap();
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
        let func = FuncOp::new(
            &mut ctx,
            "cond_block_args".try_into().unwrap(),
            func_ty,
            LinkageAttr::External,
        );
        func.get_operation().insert_at_back(body, &ctx);
        let entry = func.get_entry_block(&ctx);
        let then_block = BasicBlock::new(
            &mut ctx,
            Some("then".try_into().unwrap()),
            vec![i64_ty.into()],
        );
        then_block.insert_at_back(func.get_region(&ctx), &ctx);
        let else_block = BasicBlock::new(
            &mut ctx,
            Some("else".try_into().unwrap()),
            vec![i64_ty.into()],
        );
        else_block.insert_at_back(func.get_region(&ctx), &ctx);

        let cond = ConstantOp::new_integer(
            &mut ctx,
            IntegerAttr::new(i1_ty, APInt::from_u64(1, NonZero::new(1).unwrap())),
        );
        cond.get_operation().insert_at_back(entry, &ctx);
        let true_value = ConstantOp::new_integer(
            &mut ctx,
            IntegerAttr::new(i64_ty, APInt::from_u64(7, NonZero::new(64).unwrap())),
        );
        true_value.get_operation().insert_at_back(entry, &ctx);
        let false_value = ConstantOp::new_integer(
            &mut ctx,
            IntegerAttr::new(i64_ty, APInt::from_u64(9, NonZero::new(64).unwrap())),
        );
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

        let root = lower_to_macho_ir(&mut ctx, module.get_operation()).unwrap();
        let object = util::cast_operation::<ObjectOp>(&ctx, root).unwrap();
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
        let func = FuncOp::new(
            &mut ctx,
            "gep".try_into().unwrap(),
            func_ty,
            LinkageAttr::External,
        );
        func.get_operation().insert_at_back(body, &ctx);
        let entry = func.get_entry_block(&ctx);

        let one = ConstantOp::new_integer(
            &mut ctx,
            IntegerAttr::new(i64_ty, APInt::from_u64(1, NonZero::new(64).unwrap())),
        );
        one.get_operation().insert_at_back(entry, &ctx);
        let one_result = one.get_result(&ctx);
        let slot = AllocaOp::new(&mut ctx, one_result, array_ty.into());
        slot.get_operation().insert_at_back(entry, &ctx);
        let value = ConstantOp::new_integer(
            &mut ctx,
            IntegerAttr::new(i64_ty, APInt::from_u64(99, NonZero::new(64).unwrap())),
        );
        value.get_operation().insert_at_back(entry, &ctx);
        let slot_result = slot.get_result(&ctx);
        let elem = GetElementPtrOp::new(
            &mut ctx,
            slot_result,
            vec![],
            GepIndicesAttr(vec![GepIndexAttr::Constant(2)]),
            i64_ty.into(),
        );
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

        let root = lower_to_macho_ir(&mut ctx, module.get_operation()).unwrap();
        let object = util::cast_operation::<ObjectOp>(&ctx, root).unwrap();
        assert_eq!(object.symbols(&ctx)[0].name, "_gep");
        assert!(!object.text(&ctx).is_empty());
    }

    #[test]
    fn block_placement_lays_out_weighted_hot_path_as_fallthrough() {
        use crate::common_traits::Named;
        use llvm_compat::op_interfaces::WeightedBranchOpInterface;

        let mut ctx = context();
        let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
        let i1_ty =
            builtin::types::IntegerType::get(&mut ctx, 1, builtin::types::Signedness::Signless);
        let i64_ty =
            builtin::types::IntegerType::get(&mut ctx, 64, builtin::types::Signedness::Signless);
        let func_ty = FuncType::get(&mut ctx, i64_ty.into(), vec![], false);
        let func = FuncOp::new(
            &mut ctx,
            "biased".try_into().unwrap(),
            func_ty,
            LinkageAttr::External,
        );
        func.get_operation().insert_at_back(body, &ctx);
        let entry = func.get_entry_block(&ctx);
        let then_block = BasicBlock::new(&mut ctx, Some("then".try_into().unwrap()), vec![]);
        then_block.insert_at_back(func.get_region(&ctx), &ctx);
        let else_block = BasicBlock::new(&mut ctx, Some("else".try_into().unwrap()), vec![]);
        else_block.insert_at_back(func.get_region(&ctx), &ctx);

        let cond = ConstantOp::new_integer(
            &mut ctx,
            IntegerAttr::new(i1_ty, APInt::from_u64(1, NonZero::new(1).unwrap())),
        );
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
            let constant = ConstantOp::new_integer(
                &mut ctx,
                IntegerAttr::new(i64_ty, APInt::from_u64(value, NonZero::new(64).unwrap())),
            );
            constant.get_operation().insert_at_back(block, &ctx);
            let result = constant.get_result(&ctx);
            ReturnOp::new(&mut ctx, Some(result))
                .get_operation()
                .insert_at_back(block, &ctx);
        }

        // Run the pipeline through block placement (post-RA layout, before
        // branch relaxation, as in LLVM).
        let prefix: Pipeline = vec![
            Arc::new(VerifyLlvmForX86_64DarwinPass) as PassObj,
            Arc::new(LlvmX86_64DarwinAbiPass),
            Arc::new(LlvmToX86_64ISelPass),
            Arc::new(X86_64LegalizePass),
            Arc::new(X86_64MachineCfgCleanupPass),
            Arc::new(X86_64TargetOptsPreRaPass),
            Arc::new(X86_64RegisterAllocatePass),
            Arc::new(X86_64FrameLowerPass),
            Arc::new(X86_64PostRaOptsPass),
            Arc::new(x86_64_block_placement::X86_64BlockPlacementPass),
        ]
        .into();
        let root = PassManager::without_callback()
            .run(prefix, &mut ctx, module.get_operation())
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
            .find_map(|op| util::cast_operation::<x86_64::ops::FuncOp>(&ctx, op))
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
        let cold_target = x86_64::ops::target(&ctx, tail).unwrap();
        assert!(
            cold_target
                .deref(&ctx)
                .unique_name(&ctx)
                .to_string()
                .starts_with("then")
        );
        assert!(x86_64::ops::branch_weights(&ctx, tail).is_some());

        // The rest of the pipeline still produces a valid MachO object.
        let suffix: Pipeline = vec![
            Arc::new(X86_64BranchRelaxPass) as PassObj,
            Arc::new(X86_64AsmLowerPass),
            Arc::new(X86_64EncodePass),
            Arc::new(X86_64MachOLowerPass),
        ]
        .into();
        let root = PassManager::without_callback()
            .run(suffix, &mut ctx, root)
            .unwrap();
        let bytes = write_macho_object_from_ir(&ctx, root).unwrap();
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
        let func = FuncOp::new(
            &mut ctx,
            "main".try_into().unwrap(),
            func_ty,
            LinkageAttr::External,
        );
        func.get_operation().insert_at_back(body, &ctx);
        let entry = func.get_entry_block(&ctx);
        let zero = ConstantOp::new_integer(
            &mut ctx,
            IntegerAttr::new(i64_ty, APInt::from_u64(0, NonZero::new(64).unwrap())),
        );
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

use llvm_compat::llvm::attributes::LinkageAttr;
