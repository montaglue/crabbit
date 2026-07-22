//! Keep type-punned stack slots in memory.
//!
//! The MIR importer reads enum discriminants by loading an integer directly
//! from an aggregate-typed slot (a type-punned access). Register promotion
//! must leave such slots in memory: promoting the slot would substitute an
//! aggregate SSA value into the integer use. pliron's mem2reg only inspects
//! direct load/store uses of the allocation pointer, so this pass reroutes
//! every type-punned access through a pointer-to-pointer `llvm.bitcast` —
//! semantically a no-op, but an opaque (non-promotable) use of the slot that
//! keeps mem2reg away from it.

use pliron::builtin::op_interfaces::{AtMostOneRegionInterface as _, OneResultInterface as _};
use pliron_llvm::op_interfaces::{CastOpInterface as _, PointerTypeResult as _};

use crate::{
    context::{Context, Ptr},
    dialects::llvm::{
        ops::{AllocaOp, FuncOp, LoadOp, StoreOp},
        types::PointerType,
    },
    ir::{op::Op, operation::Operation, r#type::Typed},
    linked_list::ContainsLinkedList,
    conversion::pass::{AnalysisManager, Pass, PassResult, changed, unchanged},
    result::STAIRResult,
};

use super::inline::collect_functions;

pub struct LLVMPinTypePunnedSlotsPass;

impl Pass for LLVMPinTypePunnedSlotsPass {
    fn name(&self) -> &str {
        "llvm-pin-type-punned-slots"
    }

    fn run(
        &mut self,
        root: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> pliron::result::Result<PassResult> {
        let mut any = false;
        for func in collect_functions(ctx, root) {
            any |= pin_function_slots(ctx, func)?;
        }
        Ok(if any { changed() } else { unchanged() })
    }
}

fn pin_function_slots(ctx: &mut Context, func: FuncOp) -> STAIRResult<bool> {
    let Some(region) = func.get_region(ctx) else {
        return Ok(false);
    };
    let mut allocas = Vec::new();
    for block in region.deref(ctx).iter(ctx) {
        for op in block.deref(ctx).iter(ctx) {
            if let Some(alloca) = Operation::get_op_dyn(op, ctx).downcast_ref::<AllocaOp>() {
                allocas.push(*alloca);
            }
        }
    }

    let mut any = false;
    for alloca in allocas {
        any |= pin_slot_if_punned(ctx, alloca)?;
    }
    Ok(any)
}

/// Reroute `slot`'s type-punned loads/stores through a fresh pointer bitcast,
/// if it has any.
fn pin_slot_if_punned(ctx: &mut Context, alloca: AllocaOp) -> STAIRResult<bool> {
    let slot = alloca.get_result(ctx);
    let slot_ty = alloca.result_pointee_type(ctx);

    let punned: Vec<Ptr<Operation>> = slot
        .uses(ctx)
        .iter()
        .map(|slot_use| slot_use.user_op())
        .filter(|user| is_type_punned_access(ctx, *user, &alloca))
        .collect();
    if punned.is_empty() {
        return Ok(false);
    }
    // slot_ty read; keep it out of the punned check closure's borrow.
    let _ = slot_ty;

    let ptr_ty = PointerType::get(ctx, 0).into();
    let shield = pliron_llvm::ops::BitcastOp::new(ctx, slot, ptr_ty);
    shield.get_operation().insert_after(ctx, alloca.get_operation());
    let shield_val = shield.get_result(ctx);
    slot.replace_some_uses_with(ctx, |_, slot_use| punned.contains(&slot_use.user_op()), &shield_val);
    Ok(true)
}

/// Is `user` a load or store on `alloca`'s pointer whose accessed type
/// disagrees with the allocated type?
fn is_type_punned_access(ctx: &Context, user: Ptr<Operation>, alloca: &AllocaOp) -> bool {
    let slot = alloca.get_result(ctx);
    let slot_ty = alloca.result_pointee_type(ctx);
    let user_obj = Operation::get_op_dyn(user, ctx);
    if let Some(load) = user_obj.downcast_ref::<LoadOp>() {
        return load.get_operand_address(ctx) == slot
            && load.get_result(ctx).get_type(ctx) != slot_ty;
    }
    if let Some(store) = user_obj.downcast_ref::<StoreOp>() {
        return store.get_operand_address(ctx) == slot
            && store.get_operand_value(ctx).get_type(ctx) != slot_ty;
    }
    false
}

#[cfg(test)]
mod tests {
    use std::num::NonZero;

    use pliron::opts::mem2reg::Mem2RegPass;

    use pliron::builtin::op_interfaces::OneResultInterface as _;

    use crate::{
        context::Context,
        dialects::{
            builtin::{
                self,
                attributes::IntegerAttr,
                op_interfaces::OneRegionInterface,
                types::{IntegerType, Signedness},
            },
            llvm::{
                attributes::LinkageAttr,
                ops::{AllocaOp, ConstantOp, FuncOp, LoadOp, ReturnOp},
                types::{FuncType, StructType},
            },
        },
        ir::op::Op,
        linked_list::ContainsLinkedList,
        conversion::pass::{AnalysisManager, Pass},
        utils::apint::APInt,
    };

    use super::LLVMPinTypePunnedSlotsPass;

    #[test]
    fn punned_discriminant_load_keeps_the_slot_in_memory() {
        let mut ctx = Context::new();

        let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
        let i64_ty = IntegerType::get(&mut ctx, 64, Signedness::Signless);
        let pair_ty =
            StructType::get_unnamed(&ctx, vec![i64_ty.into(), i64_ty.into()]);
        let func_ty = FuncType::get(&mut ctx, i64_ty.into(), vec![], false);
        let func = FuncOp::new(&mut ctx, "f".try_into().unwrap(), func_ty);
        func.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
        let entry = func.get_or_create_entry_block(&mut ctx);
        func.get_operation().insert_at_back(body, &ctx);

        let one = ConstantOp::new(
            &mut ctx,
            Box::new(IntegerAttr::new(
                i64_ty,
                APInt::from_u64(1, NonZero::new(64).unwrap()),
            )),
        );
        one.get_operation().insert_at_back(entry, &ctx);
        let one_val = one.get_result(&ctx);
        let slot = AllocaOp::new(&mut ctx, pair_ty.into(), one_val);
        slot.get_operation().insert_at_back(entry, &ctx);
        let slot_val = slot.get_result(&ctx);
        // Type-punned discriminant-style read: i64 from a {i64, i64} slot.
        let load = LoadOp::new(&mut ctx, slot_val, i64_ty.into());
        load.get_operation().insert_at_back(entry, &ctx);
        let loaded = load.get_result(&ctx);
        ReturnOp::new(&mut ctx, Some(loaded))
            .get_operation()
            .insert_at_back(entry, &ctx);

        LLVMPinTypePunnedSlotsPass
            .run(module.get_operation(), &mut ctx, &mut AnalysisManager::default())
            .unwrap();
        Mem2RegPass
            .run(module.get_operation(), &mut ctx, &mut AnalysisManager::default())
            .unwrap();

        // The punned slot must survive promotion.
        let printed = format!("{}", pliron::printable::Printable::disp(&module.get_operation(), &ctx));
        assert!(printed.contains("llvm.alloca"), "slot was promoted:\n{printed}");
        assert!(printed.contains("llvm.bitcast"), "shield missing:\n{printed}");
    }
}
