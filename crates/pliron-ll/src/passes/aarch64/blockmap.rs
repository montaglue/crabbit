//! Stable RA-position block ids and the blockmap sidecar
//! (docs/PROFILE-FEEDBACK-PLAN.md).
//!
//! [Aarch64BlockmapIdPass] runs at the register-allocation pipeline
//! position (immediately before the allocator — the point whose CFG the
//! block-frequency models describe) and, when `CRABBIT_BLOCKMAP` is set,
//! stamps every machine block with a [BlockmapIdAttr]: its index in
//! RA-time region order. **Mechanism**: pliron 0.17 [BasicBlock]s carry
//! their own attribute dictionary, and no pass after RA recreates, splits,
//! or merges blocks (placement only unlinks/relinks them, relaxation and
//! post-RA opts only edit ops), so the attribute survives to final layout
//! on the block's own identity — no side table, no first-op anchoring.
//! A block created after RA would simply lack the attribute and is
//! reported with id `-1` (its samples are dropped at ingestion).
//!
//! After encoding, [collect_blockmap] recomputes each block's final
//! `.text` byte range (function offsets accumulate exactly like
//! `collect_object_parts`; block offsets are `byte_len` sums in final
//! layout order, exactly like the encoder's own `block_offsets`) and
//! [blockmap_json_from_ir] renders the sidecar payload:
//!
//! ```json
//! { "<function symbol>": [ {"id": 0, "start": 0, "end": 24}, ... ], ... }
//! ```
//!
//! Symbols are the un-mangled `FuncOp` names — the same keys the
//! frequency consumers look up
//! ([profile_freq](crate::passes::profile_freq)); on ELF/Linux they equal
//! the object's symbol names (Darwin prepends `_`, which the ingestion
//! tool strips when matching).

use std::collections::BTreeMap;

use crate::ll::BlockmapIdAttr;
use crate::{
    context::{Context, Ptr},
    dialects::{
        aarch64::op_interfaces::BinarySerializableOpInterface,
        aarch64::ops::FuncOp,
        builtin::op_interfaces::{OneRegionInterface, SymbolOpInterface},
    },
    ir::{basic_block::BasicBlock, op::op_cast, operation::Operation},
    linked_list::ContainsLinkedList,
    conversion::pass::{AnalysisManager, Pass, PassResult, changed},
};

use super::{
    attrs::ATTR_KEY_AARCH64_ENCODED,
    frontend::module_op,
    util::{cast_operation, get_bytes_attr},
};

use crate::dict_key;

dict_key!(ATTR_KEY_AARCH64_BLOCKMAP_ID, "aarch64_blockmap_id");

/// Whether the blockmap sidecar is requested (`CRABBIT_BLOCKMAP` set,
/// non-empty, not `0`).
pub fn blockmap_enabled() -> bool {
    let set = |var: &str| {
        std::env::var(var).is_ok_and(|value| !value.is_empty() && value != "0")
    };
    // CRABBIT_PROFILE_MAP is the rename-friendly alias now that the map
    // carries op-level attribution, not only block ranges.
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

/// One block's byte range in the final `.text`: `[start, end)`, absolute
/// section offsets. `id` is the RA-position block id, or `-1` for a block
/// created after RA (no samples are attributed to it). When op-level
/// attribution ran (docs/PROFILE-FEEDBACK-BACKWARD.md), `ops` carries the
/// per-instruction ranges in layout order.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct BlockRange {
    pub id: i64,
    pub start: u64,
    pub end: u64,
    pub ops: Vec<OpRange>,
}

/// One machine instruction's byte range and backward attribution: `index`
/// is its per-function position in final layout order; `derived_from` is
/// the LLVM-level op id it was lowered from (≥ 0) or a synthetic root
/// (< 0, see [super::opmap::roots]; [super::opmap::roots::UNATTRIBUTED]
/// for ops nothing stamped).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpRange {
    pub index: u32,
    pub derived_from: i64,
    pub start: u64,
    pub end: u64,
}

/// The blockmap of a fully-encoded module: per function symbol, the block
/// ranges in final layout order. Functions without stamped ids (blockmap
/// disabled during RA) are omitted; an empty map means "no sidecar".
pub fn collect_blockmap(
    ctx: &Context,
    root: Ptr<Operation>,
) -> pliron::result::Result<BTreeMap<String, Vec<BlockRange>>> {
    let module = module_op(ctx, root)?;
    let body = module.get_region(ctx).deref(ctx).get_head().unwrap();
    let mut map = BTreeMap::new();
    let mut function_offset = 0u64;
    for op in body.deref(ctx).iter(ctx) {
        let Some(func) = cast_operation::<FuncOp>(ctx, op) else {
            continue;
        };
        // Function offsets accumulate over the encoded payloads exactly the
        // way `collect_object_parts` builds `.text`.
        let encoded_len = get_bytes_attr(op, ctx, ATTR_KEY_AARCH64_ENCODED.as_ref())
            .map(|bytes| bytes.len() as u64)
            .unwrap_or_default();
        let mut ranges = Vec::new();
        let mut stamped = false;
        // Whether op-level stamping ran for this function at all: any op
        // anywhere carrying attribution. Distinguishes blockmap-only mode
        // (no per-op rows wanted) from a stamping run with a block the
        // stampers missed — whose rows must SURVIVE as `unattributed`
        // (roots::UNATTRIBUTED exists so nothing is silently dropped).
        let op_stamping_ran = func.get_region(ctx).deref(ctx).iter(ctx).any(|block| {
            block
                .deref(ctx)
                .iter(ctx)
                .any(|inst| super::opmap::derived_from(ctx, inst).is_some())
        });
        let mut offset = function_offset;
        let mut op_index = 0u32;
        for block in func.get_region(ctx).deref(ctx).iter(ctx) {
            let mut ops = Vec::new();
            let mut size = 0u64;
            for inst in block.deref(ctx).iter(ctx) {
                let op_obj = Operation::get_op_dyn(inst, ctx);
                let Some(bytes) = op_cast::<dyn BinarySerializableOpInterface>(&*op_obj)
                    .map(|serializable| serializable.byte_len(ctx))
                else {
                    continue;
                };
                drop(op_obj);
                if bytes > 0 {
                    ops.push(OpRange {
                        index: op_index,
                        derived_from: super::opmap::derived_from(ctx, inst)
                            .unwrap_or(super::opmap::roots::UNATTRIBUTED),
                        start: offset + size,
                        end: offset + size + bytes,
                    });
                }
                op_index += 1;
                size += bytes;
            }
            // Op-level attribution rows are only kept when stamping ran
            // (blockmap-only mode wants block ranges, not op rows). When
            // stamping DID run, an all-UNATTRIBUTED block is a stamping
            // gap and its rows are kept so ingest reports the cost under
            // the `unattributed` root instead of dropping it.
            if !op_stamping_ran {
                ops.clear();
            }
            let id = match blockmap_id(ctx, block) {
                Some(id) => {
                    stamped = true;
                    id as i64
                }
                None => -1,
            };
            ranges.push(BlockRange {
                id,
                start: offset,
                end: offset + size,
                ops,
            });
            offset += size;
        }
        debug_assert_eq!(
            offset - function_offset,
            encoded_len,
            "byte_len sum disagrees with the encoded payload"
        );
        if stamped {
            map.insert(func.get_symbol_name(ctx).to_string(), ranges);
        }
        function_offset += encoded_len;
    }
    Ok(map)
}

/// The `<object stem>.blockmap.json` payload for a fully-encoded module,
/// or `None` when there is nothing to write (no machine functions carry
/// ids — blockmap disabled, or a non-aarch64 module). Never fails the
/// build: collection errors degrade to `None`.
pub fn blockmap_json_from_ir(ctx: &Context, root: Ptr<Operation>) -> Option<String> {
    let map = match collect_blockmap(ctx, root) {
        Ok(map) => map,
        Err(_) => return None,
    };
    if map.is_empty() {
        return None;
    }
    let mut top = serde_json::Map::new();
    if let Some(midend) = super::opmap::midend_boundary_json(ctx, root) {
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&midend) {
            // Per-function fresh-id → source-parents table from the
            // RA-boundary pass; consumers must skip this non-function key.
            top.insert("__midend__".to_string(), parsed);
        }
    }
    let value = serde_json::Value::Object(
        map.iter()
            .map(|(symbol, ranges)| {
                (
                    symbol.clone(),
                    serde_json::Value::Array(
                        ranges
                            .iter()
                            .map(|range| {
                                let mut object = serde_json::json!({
                                    "id": range.id,
                                    "start": range.start,
                                    "end": range.end,
                                });
                                if !range.ops.is_empty() {
                                    object["ops"] = serde_json::Value::Array(
                                        range
                                            .ops
                                            .iter()
                                            .map(|op| {
                                                serde_json::json!({
                                                    "index": op.index,
                                                    "derived_from": op.derived_from,
                                                    "start": op.start,
                                                    "end": op.end,
                                                })
                                            })
                                            .collect(),
                                    );
                                }
                                object
                            })
                            .collect(),
                    ),
                )
            })
            .collect::<serde_json::Map<_, _>>()
            .into_iter()
            .chain(top)
            .collect(),
    );
    serde_json::to_string_pretty(&value).ok()
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
                op_interfaces::{OneRegionInterface, OneResultInterface},
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
    /// moved next to the entry by block placement, so the blockmap must
    /// report the RA-order ids out of layout order while still covering
    /// the function's final bytes contiguously.
    #[test]
    fn blockmap_ids_survive_placement_and_relax_with_contiguous_coverage() {
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

        // A second, single-block function behind the first: its blockmap
        // ranges must start exactly where the first function's bytes end.
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

        let map = collect_blockmap(&ctx, root).unwrap();
        assert_eq!(map.keys().collect::<Vec<_>>(), vec!["biased", "tail"]);

        let biased = &map["biased"];
        assert_eq!(biased.len(), 3, "all three RA-time blocks survive");
        // Contiguous, monotone coverage of the function's final bytes.
        assert_eq!(biased[0].start, 0, "first function starts .text");
        for pair in biased.windows(2) {
            assert!(pair[0].start < pair[0].end, "empty range {pair:?}");
            assert_eq!(pair[0].end, pair[1].start, "gap or overlap {pair:?}");
        }
        // Every RA id survived, exactly once, and placement really did
        // reorder: the hot else block (RA id 2) now sits right after the
        // entry, before the cold then block (RA id 1).
        let layout_ids: Vec<i64> = biased.iter().map(|range| range.id).collect();
        assert_eq!(layout_ids, vec![0, 2, 1], "hot path moved up, ids intact");

        let tail_ranges = &map["tail"];
        assert_eq!(tail_ranges.len(), 1);
        assert_eq!(tail_ranges[0].id, 0);
        assert_eq!(
            tail_ranges[0].start,
            biased.last().unwrap().end,
            "second function's ranges continue where the first ends"
        );
        assert!(tail_ranges[0].end > tail_ranges[0].start);

        // The sidecar payload renders and round-trips as JSON.
        let json = blockmap_json_from_ir(&ctx, root).expect("stamped module yields a sidecar");
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["biased"][1]["id"], serde_json::json!(2));
        assert_eq!(
            parsed["biased"][1]["start"],
            serde_json::json!(biased[1].start)
        );
    }

    /// Without stamped ids (blockmap disabled) nothing is collected and no
    /// sidecar is produced.
    #[test]
    fn unstamped_module_yields_no_sidecar() {
        let mut ctx = context();
        let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let root = module.get_operation();
        assert!(collect_blockmap(&ctx, root).unwrap().is_empty());
        assert!(blockmap_json_from_ir(&ctx, root).is_none());
    }
}
