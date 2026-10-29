//! Stable RA-position block ids.
//!
//! [Aarch64BlockmapIdPass] runs at the register-allocation pipeline
//! position (immediately before the allocator — the point whose CFG the
//! block-frequency models describe) and, when [blockmap_enabled],
//! stamps every machine block with a [BlockmapIdAttr]: its index in
//! RA-time region order. **Mechanism**: pliron 0.17 [BasicBlock]s carry
//! their own attribute dictionary, and no pass after RA recreates, splits,
//! or merges blocks (placement only unlinks/relinks them, relaxation and
//! post-RA opts only edit ops), so the attribute survives to final layout
//! on the block's own identity — no side table, no first-op anchoring.
//! A block created after RA simply lacks the attribute.
//!
//! The frequency consumers ([profile_freq](crate::passes::profile_freq))
//! index blocks in this RA-time order.

use crate::ll::BlockmapIdAttr;
use crate::{
    context::{Context, Ptr},
    dialects::{aarch64::ops::FuncOp, builtin::op_interfaces::OneRegionInterface},
    ir::{basic_block::BasicBlock, operation::Operation},
    linked_list::ContainsLinkedList,
    conversion::pass::{AnalysisManager, Pass, PassResult, changed},
};

use super::{frontend::module_op, util::cast_operation};

use crate::dict_key;

dict_key!(ATTR_KEY_AARCH64_BLOCKMAP_ID, "aarch64_blockmap_id");

/// Whether block ids and op provenance are stamped (`CRABBIT_PROFILE_MAP`
/// or its older name `CRABBIT_BLOCKMAP` set, non-empty, not `0`).
pub fn blockmap_enabled() -> bool {
    let set = |var: &str| {
        std::env::var(var).is_ok_and(|value| !value.is_empty() && value != "0")
    };
    set("CRABBIT_BLOCKMAP") || set("CRABBIT_PROFILE_MAP")
}

/// Stamps RA-position block ids when [blockmap_enabled]; a no-op
/// otherwise (cheap, off by default).
pub struct Aarch64BlockmapIdPass;

impl Pass for Aarch64BlockmapIdPass {
    fn name(&self) -> &str {
        "aarch64-blockmap-ids"
    }

    fn run(
        &mut self,
        root: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> pliron::result::Result<PassResult> {
        if blockmap_enabled() {
            assign_blockmap_ids(ctx, root)?;
        }
        Ok(changed())
    }
}

/// Stamp every block of every machine function in `root` with its index in
/// the current (RA-time) region order. Public so tests can drive it
/// without touching the environment.
pub fn assign_blockmap_ids(
    ctx: &mut Context,
    root: Ptr<Operation>,
) -> pliron::result::Result<()> {
    let module = module_op(ctx, root)?;
    let body = module.get_region(ctx).deref(ctx).get_head().unwrap();
    let ops: Vec<_> = body.deref(ctx).iter(ctx).collect();
    for op in ops {
        let Some(func) = cast_operation::<FuncOp>(ctx, op) else {
            continue;
        };
        let blocks: Vec<_> = func.get_region(ctx).deref(ctx).iter(ctx).collect();
        for (index, block) in blocks.into_iter().enumerate() {
            block.deref_mut(ctx).attributes.set(
                ATTR_KEY_AARCH64_BLOCKMAP_ID.clone(),
                BlockmapIdAttr(index as u32),
            );
        }
    }
    Ok(())
}

/// The RA-position id stamped on `block`, if any.
pub fn blockmap_id(ctx: &Context, block: Ptr<BasicBlock>) -> Option<u32> {
    block
        .deref(ctx)
        .attributes
        .get::<BlockmapIdAttr>(&ATTR_KEY_AARCH64_BLOCKMAP_ID)
        .map(|attr| attr.0)
}

#[cfg(test)]
mod tests {
    use std::num::NonZero;

    use pliron::builtin::op_interfaces::AtMostOneRegionInterface as _;

    use super::*;
    use crate::ir::op::Op as _;
    use crate::passes::aarch64::{
        aarch64_asm_lower::Aarch64AsmLowerPass, aarch64_block_placement::Aarch64BlockPlacementPass,
        aarch64_branch_relax::Aarch64BranchRelaxPass, aarch64_encode::Aarch64EncodePass,
        aarch64_frame_lower::Aarch64FrameLowerPass, aarch64_legalize::Aarch64LegalizePass,
        aarch64_machine_cfg_cleanup::Aarch64MachineCfgCleanupPass,
        aarch64_post_ra_opts::Aarch64PostRaOptsPass,
        aarch64_register_allocate::Aarch64RegisterAllocatePass,
        aarch64_target_opts_pre_ra::Aarch64TargetOptsPreRaPass,
        llvm_aarch64_abi::LlvmAarch64AbiPass, llvm_to_aarch64_isel::LlvmToAarch64IselPass,
        target::TargetOs, verify_llvm_for_aarch64::VerifyLlvmForAarch64Pass,
    };
    use crate::{
        conversion::pass::Passes,
        dialects::{
            aarch64,
            builtin::{
                self,
                attributes::IntegerAttr,
                op_interfaces::{OneRegionInterface, OneResultInterface, SymbolOpInterface},
            },
            llvm::{
                attributes::LinkageAttr,
                ops::{CondBrOp, FuncOp as LlvmFuncOp, ReturnOp},
                types::FuncType,
            },
            macho,
        },
        dialects::builtin::ops::ConstantOp,
        ll::op_interfaces::WeightedBranchOpInterface,
        utils::apint::APInt,
    };

    fn context() -> Context {
        let mut ctx = Context::new();
        aarch64::register(&mut ctx);
        macho::register(&mut ctx);
        ctx
    }

    /// Two functions; the first has a weighted diamond whose hot side gets
    /// moved next to the entry by block placement, so its blocks must end
    /// up out of RA order with their RA-order ids intact.
    #[test]
    fn blockmap_ids_survive_placement_and_relax() {
        let mut ctx = context();
        let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
        let i1_ty =
            builtin::types::IntegerType::get(&mut ctx, 1, builtin::types::Signedness::Signless);
        let i64_ty =
            builtin::types::IntegerType::get(&mut ctx, 64, builtin::types::Signedness::Signless);
        let func_ty = FuncType::get(&mut ctx, i64_ty.into(), vec![], false);
        let func = LlvmFuncOp::new(&mut ctx, "biased".try_into().unwrap(), func_ty);
        func.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(&mut ctx);
        func.get_operation().insert_at_back(body, &ctx);
        let entry = func.get_entry_block(&ctx).unwrap();
        let then_block = BasicBlock::new(&mut ctx, Some("then".try_into().unwrap()), vec![]);
        then_block.insert_at_back(func.get_region(&ctx).unwrap(), &ctx);
        let else_block = BasicBlock::new(&mut ctx, Some("else".try_into().unwrap()), vec![]);
        else_block.insert_at_back(func.get_region(&ctx).unwrap(), &ctx);

        let cond = ConstantOp::new(
            &mut ctx,
            Box::new(IntegerAttr::new(
                i1_ty,
                APInt::from_u64(1, NonZero::new(1).unwrap()),
            )),
        );
        cond.get_operation().insert_at_back(entry, &ctx);
        let cond_result = cond.get_result(&ctx);
        let cond_br = CondBrOp::new(&mut ctx, cond_result, then_block, vec![], else_block, vec![]);
        // The true edge is cold, the false edge hot: placement moves the
        // else block next to the entry, ahead of the then block.
        cond_br.set_successor_weights(&ctx, vec![1, 2000]);
        cond_br.get_operation().insert_at_back(entry, &ctx);
        for (block, value) in [(then_block, 1u64), (else_block, 0u64)] {
            let constant = ConstantOp::new(
                &mut ctx,
                Box::new(IntegerAttr::new(
                    i64_ty,
                    APInt::from_u64(value, NonZero::new(64).unwrap()),
                )),
            );
            constant.get_operation().insert_at_back(block, &ctx);
            let result = constant.get_result(&ctx);
            ReturnOp::new(&mut ctx, Some(result))
                .get_operation()
                .insert_at_back(block, &ctx);
        }

        // A second, single-block function behind the first.
        let tail = LlvmFuncOp::new(&mut ctx, "tail".try_into().unwrap(), func_ty);
        tail.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
        tail.get_or_create_entry_block(&mut ctx);
        tail.get_operation().insert_at_back(body, &ctx);
        let tail_entry = tail.get_entry_block(&ctx).unwrap();
        let zero = ConstantOp::new(
            &mut ctx,
            Box::new(IntegerAttr::new(
                i64_ty,
                APInt::from_u64(0, NonZero::new(64).unwrap()),
            )),
        );
        zero.get_operation().insert_at_back(tail_entry, &ctx);
        let zero_result = zero.get_result(&ctx);
        ReturnOp::new(&mut ctx, Some(zero_result))
            .get_operation()
            .insert_at_back(tail_entry, &ctx);

        // Pipeline up to the RA position, then stamp ids exactly where the
        // aarch64-blockmap-ids pass sits.
        let root = module.get_operation();
        let mut prefix = Passes::default();
        prefix.add_pass(VerifyLlvmForAarch64Pass::new(TargetOs::Linux));
        prefix.add_pass(LlvmAarch64AbiPass::new(TargetOs::Linux));
        prefix.add_pass(LlvmToAarch64IselPass);
        prefix.add_pass(Aarch64LegalizePass);
        prefix.add_pass(Aarch64MachineCfgCleanupPass);
        prefix.add_pass(Aarch64TargetOptsPreRaPass);
        prefix
            .run(root, &mut ctx, &mut AnalysisManager::default())
            .unwrap();
        assign_blockmap_ids(&mut ctx, root).unwrap();

        // The rest of the pipeline: RA, frame, post-RA opts, placement
        // (reorders blocks), relaxation, lowering, encoding.
        let mut suffix = Passes::default();
        suffix.add_pass(Aarch64RegisterAllocatePass);
        suffix.add_pass(Aarch64FrameLowerPass);
        suffix.add_pass(Aarch64PostRaOptsPass);
        suffix.add_pass(Aarch64BlockPlacementPass);
        suffix.add_pass(Aarch64BranchRelaxPass);
        suffix.add_pass(Aarch64AsmLowerPass);
        suffix.add_pass(Aarch64EncodePass);
        suffix
            .run(root, &mut ctx, &mut AnalysisManager::default())
            .unwrap();

        // Every RA id survived, exactly once, and placement really did
        // reorder: the hot else block (RA id 2) now sits right after the
        // entry, before the cold then block (RA id 1).
        let layout_ids = |symbol: &str| -> Vec<Option<u32>> {
            let func = pliron_bpgo::walk::ops_under(&ctx, root)
                .into_iter()
                .find(|&op| {
                    Operation::get_op_dyn(op, &ctx)
                        .downcast_ref::<aarch64::ops::FuncOp>()
                        .is_some_and(|func| func.get_symbol_name(&ctx).to_string() == symbol)
                })
                .unwrap();
            let region = func.deref(&ctx).regions().next().unwrap();
            region
                .deref(&ctx)
                .iter(&ctx)
                .map(|block| blockmap_id(&ctx, block))
                .collect()
        };
        assert_eq!(
            layout_ids("biased"),
            vec![Some(0), Some(2), Some(1)],
            "hot path moved up, ids intact"
        );
        assert_eq!(layout_ids("tail"), vec![Some(0)]);
    }
}
