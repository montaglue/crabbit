//! Promote single-scalar `llvm.alloca` slots to SSA values (mem2reg).

use std::{collections::VecDeque, num::NonZero};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    common_traits::Named,
    context::{Context, Ptr},
    debug_info::{get_operation_result_name, set_operation_result_name},
    dialects::{
        builtin::{
            op_interfaces::{OneRegionInterface, OneResultInterface},
            types::IntegerType,
        },
        llvm::{
            ops::{AllocaOp, ConstantOp, FuncOp, LoadOp, PhiOp, StoreOp, UndefOp},
            types::PointerType,
        },
    },
    identifier::Identifier,
    ir::{
        basic_block::BasicBlock,
        location::Location,
        op::Op,
        operation::Operation,
        r#type::{TypeHandle, Typed},
        value::Value,
    },
    linked_list::ContainsLinkedList,
    conversion::pass::{Pass, PassOptions},
    passes::dominance_frontier::{
        DominanceFrontiers, DominatorTree, compute_dominance_frontiers_for_op,
    },
    result::STAIRResult,
    utils::apint::APInt,
};

#[derive(Default)]
pub struct Mem2RegPass {
    /// Only promote slots whose element type fits a single general-purpose
    /// register (integers up to 64 bits and pointers). Promotion can turn a
    /// slot into a phi and hence a block argument, so backends whose block
    /// arguments carry exactly one register need this restriction until they
    /// support multi-register block arguments.
    pub single_register_scalars_only: bool,
}

impl Pass for Mem2RegPass {
    fn name(&self) -> &str {
        "mem2reg"
    }

    // Allocas are collected up front and promoted outside of any IR walk:
    // promotion erases the alloca itself, which is not safe while a walker
    // still holds a pointer to it.
    fn run(
        &self,
        root: Ptr<Operation>,
        ctx: &mut Context,
        _options: PassOptions,
    ) -> STAIRResult<Ptr<Operation>> {
        let mut allocas = Vec::new();
        collect_allocas(ctx, root, &mut allocas);

        for alloca in allocas {
            if self.single_register_scalars_only
                && !is_single_register_scalar(ctx, alloca.get_elem_type(ctx))
            {
                continue;
            }
            if let Some(uses) = is_alloca_promotable(alloca, ctx) {
                promote_mem_to_reg(alloca, uses, ctx)?;
            }
        }

        Ok(root)
    }
}

fn is_single_register_scalar(ctx: &Context, ty: TypeHandle) -> bool {
    if let Some(int_ty) = ty.deref(ctx).downcast_ref::<IntegerType>() {
        return int_ty.width() <= 64;
    }
    ty.deref(ctx).downcast_ref::<PointerType>().is_some()
}

fn collect_allocas(ctx: &Context, op: Ptr<Operation>, out: &mut Vec<AllocaOp>) {
    if Operation::get_opid(op, ctx) == AllocaOp::get_opid_static() {
        out.push(AllocaOp::from_operation(op));
    }

    let regions: Vec<_> = op.deref(ctx).regions().collect();
    for region in regions {
        let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
        for block in blocks {
            let ops: Vec<_> = block.deref(ctx).iter(ctx).collect();
            for nested in ops {
                collect_allocas(ctx, nested, out);
            }
        }
    }
}

fn is_alloca_promotable(alloca: AllocaOp, ctx: &Context) -> Option<AllocaUses> {
    is_single_scalar_alloca(alloca, ctx)?;
    let uses = collect_alloca_uses(alloca, ctx)?;
    check_promotable_alloca_accesses(alloca, &uses, ctx)?;
    Some(uses)
}

fn promote_mem_to_reg(alloca: AllocaOp, uses: AllocaUses, ctx: &mut Context) -> STAIRResult<()> {
    let cfg = collect_parent_function_cfg(alloca, ctx)?;
    let def_use_blocks = collect_def_use_blocks(&uses, &cfg);
    let (dom_tree, frontiers) = compute_dominance_frontiers_for_op(&cfg.function, ctx);
    let phi_placement = compute_phi_insertion_blocks(&def_use_blocks, &frontiers, &dom_tree);

    let value_type = get_promoted_value_type(alloca, ctx);
    let phis = insert_alloca_phi_nodes(alloca, value_type, &phi_placement, ctx)?;
    let rename_plan = build_rename_plan(alloca, &uses, &cfg, &dom_tree, &phis, ctx)?;

    apply_rename_plan(rename_plan, ctx)?;
    erase_promoted_alloca_ops(alloca, uses, ctx)
}

struct AllocaUses {
    loads: Vec<AllocaLoad>,
    stores: Vec<AllocaStore>,
}

struct AllocaLoad {
    load: LoadOp,
    result: Value,
    block: Ptr<BasicBlock>,
}

struct AllocaStore {
    store: StoreOp,
    stored_value: Value,
    block: Ptr<BasicBlock>,
}

struct FunctionCfg {
    function: FuncOp,
    entry: Ptr<BasicBlock>,
    blocks: Vec<Ptr<BasicBlock>>,
}

struct DefUseBlocks {
    def_blocks: Vec<Ptr<BasicBlock>>,
    /// Blocks containing loads. Unused until liveness pruning of phi
    /// placement is implemented.
    #[allow(dead_code)]
    use_blocks: Vec<Ptr<BasicBlock>>,
}

struct PhiPlacement {
    phi_blocks: Vec<Ptr<BasicBlock>>,
}

struct InsertedPhi {
    block: Ptr<BasicBlock>,
    phi: PhiOp,
    result: Value,
}

/// The outcome of the dominator-tree rename walk. Reaching values are
/// `None` where no store dominates the access; those become `llvm.undef`
/// when the plan is applied.
struct RenamePlan {
    entry: Ptr<BasicBlock>,
    value_type: TypeHandle,
    loads_to_replace: Vec<(LoadOp, Option<Value>)>,
    phi_incoming_values: Vec<PhiIncomingValues>,
}

struct PhiIncomingValues {
    phi: PhiOp,
    incoming_values: Vec<Option<Value>>,
    incoming_blocks: Vec<Ptr<BasicBlock>>,
}

fn collect_alloca_uses(alloca: AllocaOp, ctx: &Context) -> Option<AllocaUses> {
    let slot = alloca.get_result(ctx);
    let mut loads = Vec::new();
    let mut stores = Vec::new();

    for result_use in slot.uses(ctx) {
        if let Some(load) = Operation::get_op::<LoadOp>(result_use.user_op(), ctx) {
            if result_use.find_index(ctx) != 0 || load.get_addr(ctx) != slot {
                return None;
            }
            loads.push(AllocaLoad {
                result: load.get_result(ctx),
                block: load.get_operation().deref(ctx).get_parent_block()?,
                load,
            });
        } else if let Some(store) = Operation::get_op::<StoreOp>(result_use.user_op(), ctx) {
            if result_use.find_index(ctx) != 1 || store.get_addr(ctx) != slot {
                return None;
            }
            stores.push(AllocaStore {
                stored_value: store.get_value(ctx),
                block: store.get_operation().deref(ctx).get_parent_block()?,
                store,
            });
        } else {
            // Any other user (e.g. the address escaping into a call) makes
            // the alloca non-promotable.
            return None;
        }
    }

    Some(AllocaUses { loads, stores })
}

fn is_single_scalar_alloca(alloca: AllocaOp, ctx: &Context) -> Option<()> {
    let Some(array_size_op) = alloca.get_array_size(ctx).defining_op().filter(|_| alloca.get_array_size(ctx).find_index(ctx) == 0) else {
        return None;
    };
    let constant = Operation::get_op_dyn(array_size_op, ctx);
    let constant = constant.downcast_ref::<ConstantOp>()?;
    let array_size = constant.get_value(ctx)?.value();
    let one = APInt::from_u64(1, NonZero::new(array_size.bw()).unwrap());
    (array_size == one).then_some(())
}

fn check_promotable_alloca_accesses(
    alloca: AllocaOp,
    uses: &AllocaUses,
    ctx: &Context,
) -> Option<()> {
    let slot = alloca.get_result(ctx);
    let elem_type = alloca.get_elem_type(ctx);

    for load in &uses.loads {
        if load.result.get_type(ctx) != elem_type {
            return None;
        }
    }

    for store in &uses.stores {
        if store.stored_value == slot || store.stored_value.get_type(ctx) != elem_type {
            return None;
        }
    }

    Some(())
}

fn get_promoted_value_type(alloca: AllocaOp, ctx: &Context) -> TypeHandle {
    alloca.get_elem_type(ctx)
}

fn collect_parent_function_cfg(alloca: AllocaOp, ctx: &Context) -> STAIRResult<FunctionCfg> {
    let mut parent = alloca.get_operation().deref(ctx).get_parent_op(ctx);

    while let Some(parent_op) = parent {
        let op = Operation::get_op_dyn(parent_op, ctx);
        if let Some(func) = op.downcast::<FuncOp>() {
            let function = func;
            let region = function.get_region(ctx);
            let region_ref = region.deref(ctx);
            let Some(entry) = region_ref.get_head() else {
                return crate::arg_err!(
                    Location::Unknown,
                    "mem2reg alloca parent llvm.func has no entry block"
                );
            };
            let blocks = region_ref.iter(ctx).collect();
            return Ok(FunctionCfg {
                function,
                entry,
                blocks,
            });
        }

        parent = parent_op.deref(ctx).get_parent_op(ctx);
    }

    crate::arg_err!(
        Location::Unknown,
        "mem2reg alloca is not nested in an llvm.func"
    )
}

fn collect_def_use_blocks(uses: &AllocaUses, cfg: &FunctionCfg) -> DefUseBlocks {
    let def_blocks: FxHashSet<_> = uses.stores.iter().map(|store| store.block).collect();
    let use_blocks: FxHashSet<_> = uses.loads.iter().map(|load| load.block).collect();

    // Filter the function's block list instead of collecting from the use
    // lists directly so the result follows block order, not use order.
    DefUseBlocks {
        def_blocks: cfg
            .blocks
            .iter()
            .copied()
            .filter(|block| def_blocks.contains(block))
            .collect(),
        use_blocks: cfg
            .blocks
            .iter()
            .copied()
            .filter(|block| use_blocks.contains(block))
            .collect(),
    }
}

fn compute_phi_insertion_blocks(
    blocks: &DefUseBlocks,
    frontiers: &DominanceFrontiers,
    dom_tree: &DominatorTree,
) -> PhiPlacement {
    let mut worklist: VecDeque<Ptr<BasicBlock>> = blocks.def_blocks.iter().copied().collect();
    let mut phi_blocks = FxHashSet::default();

    // Iterated dominance frontier: a phi is itself a new definition, so
    // newly added phi blocks go back on the worklist.
    while let Some(block) = worklist.pop_front() {
        for &frontier_block in frontiers.frontier(block) {
            if phi_blocks.insert(frontier_block) {
                worklist.push_back(frontier_block);
            }
        }
    }

    let phi_blocks = dom_tree
        .blocks()
        .iter()
        .copied()
        .filter(|block| phi_blocks.contains(block))
        .collect();
    PhiPlacement { phi_blocks }
}

fn insert_alloca_phi_nodes(
    alloca: AllocaOp,
    value_type: TypeHandle,
    placement: &PhiPlacement,
    ctx: &mut Context,
) -> STAIRResult<Vec<InsertedPhi>> {
    let alloca_name = get_operation_result_name(ctx, alloca.get_operation(), 0);
    let mut phis = Vec::with_capacity(placement.phi_blocks.len());

    // Incoming values are not known yet; placeholders are rebuilt with the
    // full incoming lists when the rename plan is applied.
    for &block in &placement.phi_blocks {
        let phi = PhiOp::new(ctx, vec![], vec![], value_type);
        if let Some(name) = alloca_name.clone() {
            set_operation_result_name(ctx, phi.get_operation(), 0, Some(name));
        }
        phi.get_operation().insert_at_front(block, ctx);
        phis.push(InsertedPhi {
            block,
            phi,
            result: phi.get_result(ctx),
        });
    }

    Ok(phis)
}

struct RenameState {
    phi_results: FxHashMap<Ptr<BasicBlock>, Value>,
    load_ops: FxHashSet<Ptr<Operation>>,
    store_ops: FxHashMap<Ptr<Operation>, StoreOp>,
    load_replacements: FxHashMap<Ptr<Operation>, Option<Value>>,
    phi_incoming: FxHashMap<Ptr<BasicBlock>, (Vec<Option<Value>>, Vec<Ptr<BasicBlock>>)>,
}

impl RenameState {
    /// Resolve a stored value through pending load replacements: a store of
    /// a load from the same slot must forward the load's reaching value,
    /// since that load is about to be erased.
    fn resolve(&self, value: Value) -> Option<Value> {
        if let Some(op) = value.defining_op() {
            if let Some(replacement) = self.load_replacements.get(&op) {
                return *replacement;
            }
        }
        Some(value)
    }
}

fn build_rename_plan(
    alloca: AllocaOp,
    uses: &AllocaUses,
    cfg: &FunctionCfg,
    dom_tree: &DominatorTree,
    phis: &[InsertedPhi],
    ctx: &Context,
) -> STAIRResult<RenamePlan> {
    let mut state = RenameState {
        phi_results: phis.iter().map(|phi| (phi.block, phi.result)).collect(),
        load_ops: uses
            .loads
            .iter()
            .map(|load| load.load.get_operation())
            .collect(),
        store_ops: uses
            .stores
            .iter()
            .map(|store| (store.store.get_operation(), store.store))
            .collect(),
        load_replacements: FxHashMap::default(),
        phi_incoming: phis
            .iter()
            .map(|phi| (phi.block, Default::default()))
            .collect(),
    };

    rename_block(&mut state, dom_tree, ctx, cfg.entry, None);

    // Loads in blocks unreachable from the entry are never visited by the
    // rename walk; they read an uninitialized slot and become undef.
    let loads_to_replace = uses
        .loads
        .iter()
        .map(|load| {
            let replacement = state
                .load_replacements
                .get(&load.load.get_operation())
                .copied()
                .flatten();
            (load.load, replacement)
        })
        .collect();
    let phi_incoming_values = phis
        .iter()
        .map(|phi| {
            let (incoming_values, incoming_blocks) =
                state.phi_incoming.remove(&phi.block).unwrap_or_default();
            PhiIncomingValues {
                phi: phi.phi,
                incoming_values,
                incoming_blocks,
            }
        })
        .collect();

    Ok(RenamePlan {
        entry: cfg.entry,
        value_type: get_promoted_value_type(alloca, ctx),
        loads_to_replace,
        phi_incoming_values,
    })
}

fn rename_block(
    state: &mut RenameState,
    dom_tree: &DominatorTree,
    ctx: &Context,
    block: Ptr<BasicBlock>,
    mut reaching: Option<Value>,
) {
    if let Some(&phi_result) = state.phi_results.get(&block) {
        reaching = Some(phi_result);
    }

    let ops: Vec<_> = block.deref(ctx).iter(ctx).collect();
    for op in ops {
        if let Some(&store) = state.store_ops.get(&op) {
            reaching = state.resolve(store.get_value(ctx));
        } else if state.load_ops.contains(&op) {
            state.load_replacements.insert(op, reaching);
        }
    }

    // The value reaching the end of this block flows into the phis of its
    // CFG successors, one entry per predecessor edge.
    for succ in block.deref(ctx).succs(ctx) {
        if let Some((values, blocks)) = state.phi_incoming.get_mut(&succ) {
            values.push(reaching);
            blocks.push(block);
        }
    }

    for &child in dom_tree.children(block) {
        rename_block(state, dom_tree, ctx, child, reaching);
    }
}

fn apply_rename_plan(plan: RenamePlan, ctx: &mut Context) -> STAIRResult<()> {
    let needs_undef = plan
        .loads_to_replace
        .iter()
        .any(|(_, replacement)| replacement.is_none())
        || plan
            .phi_incoming_values
            .iter()
            .any(|incoming| incoming.incoming_values.iter().any(Option::is_none));
    let undef = if needs_undef {
        let undef = UndefOp::new(ctx, plan.value_type);
        undef.get_operation().insert_at_front(plan.entry, ctx);
        Some(undef.get_result(ctx))
    } else {
        None
    };
    let reaching_or_undef = |value: Option<Value>| {
        value
            .or(undef)
            .expect("undef is materialized whenever a reaching value is missing")
    };

    // Rebuild every placeholder phi with its full incoming list. All new
    // phis are created before any placeholder is erased: a new phi's
    // incoming value may be another placeholder's result, which the
    // replacement below redirects to that placeholder's rebuilt phi.
    let mut phi_swaps = Vec::with_capacity(plan.phi_incoming_values.len());
    for incoming in &plan.phi_incoming_values {
        let incoming_values: Vec<Value> = incoming
            .incoming_values
            .iter()
            .map(|value| reaching_or_undef(*value))
            .collect();
        let incoming_blocks: Vec<Identifier> = incoming
            .incoming_blocks
            .iter()
            .map(|block| block.deref(ctx).unique_name(ctx))
            .collect();
        let placeholder = incoming.phi;
        let phi = PhiOp::new(ctx, incoming_values, incoming_blocks, plan.value_type);
        if let Some(name) = get_operation_result_name(ctx, placeholder.get_operation(), 0) {
            set_operation_result_name(ctx, phi.get_operation(), 0, Some(name));
        }
        phi.get_operation()
            .insert_before(ctx, placeholder.get_operation());
        phi_swaps.push((placeholder, phi));
    }

    for (load, replacement) in &plan.loads_to_replace {
        let value = reaching_or_undef(*replacement);
        load.get_result(ctx)
            .replace_some_uses_with(ctx, |_, _| true, &value);
    }

    for (placeholder, phi) in phi_swaps {
        let result = phi.get_result(ctx);
        placeholder
            .get_result(ctx)
            .replace_some_uses_with(ctx, |_, _| true, &result);
        Operation::erase(placeholder.get_operation(), ctx);
    }

    Ok(())
}

fn erase_promoted_alloca_ops(
    alloca: AllocaOp,
    uses: AllocaUses,
    ctx: &mut Context,
) -> STAIRResult<()> {
    for load in &uses.loads {
        if !load.result.uses(ctx).is_empty() {
            return crate::verify_err!(
                load.load.loc(ctx),
                "mem2reg promoted load still has uses after rename"
            );
        }
    }

    for load in &uses.loads {
        Operation::erase(load.load.get_operation(), ctx);
    }
    for store in &uses.stores {
        Operation::erase(store.store.get_operation(), ctx);
    }

    if !alloca.get_result(ctx).uses(ctx).is_empty() {
        return crate::verify_err!(
            alloca.loc(ctx),
            "mem2reg promoted alloca still has uses after erasing its loads and stores"
        );
    }
    Operation::erase(alloca.get_operation(), ctx);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dialects::{
            builtin::{self, attributes::IntegerAttr, types::Signedness},
            llvm::{
                self,
                attributes::LinkageAttr,
                ops::{BrOp, CondBrOp, ReturnOp},
                types::FuncType,
            },
        },
        ir::r#type::TypedHandle,
        printable::Printable,
    };

    fn test_context() -> Context {
        let mut ctx = Context::new();
        llvm::register(&mut ctx);
        ctx
    }

    fn i64_type(ctx: &mut Context) -> TypeHandle {
        builtin::types::IntegerType::get(ctx, 64, Signedness::Signless).into()
    }

    fn i64_constant(ctx: &mut Context, value: u64) -> ConstantOp {
        let ty = builtin::types::IntegerType::get(ctx, 64, Signedness::Signless);
        ConstantOp::new_integer(
            ctx,
            IntegerAttr::new(ty, APInt::from_u64(value, NonZero::new(64).unwrap())),
        )
    }

    fn single_scalar_alloca(ctx: &mut Context, entry: Ptr<BasicBlock>) -> AllocaOp {
        let one = i64_constant(ctx, 1);
        one.get_operation().insert_at_back(entry, ctx);
        let elem_ty = i64_type(ctx);
        let alloca = AllocaOp::new(ctx, one.get_result(ctx), elem_ty);
        alloca.get_operation().insert_at_back(entry, ctx);
        alloca
    }

    fn run_mem2reg(ctx: &mut Context, func: llvm::ops::FuncOp) -> String {
        Mem2RegPass::default()
            .run(func.get_operation(), ctx, PassOptions::default())
            .unwrap();
        format!("{}", func.get_operation().disp(ctx))
    }

    #[test]
    fn promotes_straight_line_alloca() {
        let mut ctx = test_context();
        let i64_ty = i64_type(&mut ctx);
        let fn_ty: TypedHandle<FuncType> = FuncType::get(&mut ctx, i64_ty, vec![i64_ty], false);
        let func = llvm::ops::FuncOp::new(
            &mut ctx,
            "straight".try_into().unwrap(),
            fn_ty,
            LinkageAttr::External,
        );
        let entry = func.get_entry_block(&ctx);
        let arg = entry.deref(&ctx).get_argument(0);

        let alloca = single_scalar_alloca(&mut ctx, entry);
        let slot = alloca.get_result(&ctx);
        StoreOp::new(&mut ctx, arg, slot)
            .get_operation()
            .insert_at_back(entry, &ctx);
        let load = LoadOp::new(&mut ctx, slot, i64_ty);
        load.get_operation().insert_at_back(entry, &ctx);
        let loaded = load.get_result(&ctx);
        ReturnOp::new(&mut ctx, Some(loaded))
            .get_operation()
            .insert_at_back(entry, &ctx);

        let text = run_mem2reg(&mut ctx, func);

        assert!(!text.contains("llvm.alloca"), "{text}");
        assert!(!text.contains("llvm.load"), "{text}");
        assert!(!text.contains("llvm.store"), "{text}");
        assert!(!text.contains("llvm.phi"), "{text}");
        let ret = entry.deref(&ctx).get_terminator(&ctx).unwrap();
        assert!(ret.deref(&ctx).get_operand(0) == arg);
    }

    #[test]
    fn inserts_phi_for_diamond_stores() {
        let mut ctx = test_context();
        let i64_ty = i64_type(&mut ctx);
        let i1_ty: TypeHandle =
            builtin::types::IntegerType::get(&mut ctx, 1, Signedness::Signless).into();
        let fn_ty: TypedHandle<FuncType> = FuncType::get(&mut ctx, i64_ty, vec![i1_ty], false);
        let func = llvm::ops::FuncOp::new(
            &mut ctx,
            "diamond".try_into().unwrap(),
            fn_ty,
            LinkageAttr::External,
        );
        let entry = func.get_entry_block(&ctx);
        let cond = entry.deref(&ctx).get_argument(0);
        let then_block = BasicBlock::new(&mut ctx, Some("then".try_into().unwrap()), vec![]);
        let else_block = BasicBlock::new(&mut ctx, Some("else".try_into().unwrap()), vec![]);
        let join_block = BasicBlock::new(&mut ctx, Some("join".try_into().unwrap()), vec![]);
        then_block.insert_at_back(func.get_region(&ctx), &ctx);
        else_block.insert_at_back(func.get_region(&ctx), &ctx);
        join_block.insert_at_back(func.get_region(&ctx), &ctx);

        let alloca = single_scalar_alloca(&mut ctx, entry);
        let slot = alloca.get_result(&ctx);
        CondBrOp::new(&mut ctx, cond, then_block, vec![], else_block, vec![])
            .get_operation()
            .insert_at_back(entry, &ctx);

        let ten = i64_constant(&mut ctx, 10);
        ten.get_operation().insert_at_back(then_block, &ctx);
        let ten = ten.get_result(&ctx);
        StoreOp::new(&mut ctx, ten, slot)
            .get_operation()
            .insert_at_back(then_block, &ctx);
        BrOp::new(&mut ctx, join_block, vec![])
            .get_operation()
            .insert_at_back(then_block, &ctx);

        let twenty = i64_constant(&mut ctx, 20);
        twenty.get_operation().insert_at_back(else_block, &ctx);
        let twenty = twenty.get_result(&ctx);
        StoreOp::new(&mut ctx, twenty, slot)
            .get_operation()
            .insert_at_back(else_block, &ctx);
        BrOp::new(&mut ctx, join_block, vec![])
            .get_operation()
            .insert_at_back(else_block, &ctx);

        let load = LoadOp::new(&mut ctx, slot, i64_ty);
        load.get_operation().insert_at_back(join_block, &ctx);
        let loaded = load.get_result(&ctx);
        ReturnOp::new(&mut ctx, Some(loaded))
            .get_operation()
            .insert_at_back(join_block, &ctx);

        let text = run_mem2reg(&mut ctx, func);

        assert!(!text.contains("llvm.alloca"), "{text}");
        assert!(!text.contains("llvm.load"), "{text}");
        assert!(!text.contains("llvm.store"), "{text}");
        assert!(text.contains("llvm.phi"), "{text}");

        let phi_op = join_block.deref(&ctx).get_head().unwrap();
        let phi = Operation::get_op_dyn(phi_op, &ctx)
            .downcast::<PhiOp>()
            .expect("join block must start with a phi");
        let incoming = phi.get_incoming_values(&ctx);
        assert!(incoming.len() == 2);
        assert!(incoming.contains(&ten) && incoming.contains(&twenty));
        let ret = join_block.deref(&ctx).get_terminator(&ctx).unwrap();
        assert!(ret.deref(&ctx).get_operand(0) == phi.get_result(&ctx));
    }

    #[test]
    fn load_without_store_becomes_undef() {
        let mut ctx = test_context();
        let i64_ty = i64_type(&mut ctx);
        let fn_ty: TypedHandle<FuncType> = FuncType::get(&mut ctx, i64_ty, vec![], false);
        let func = llvm::ops::FuncOp::new(
            &mut ctx,
            "uninit".try_into().unwrap(),
            fn_ty,
            LinkageAttr::External,
        );
        let entry = func.get_entry_block(&ctx);

        let alloca = single_scalar_alloca(&mut ctx, entry);
        let slot = alloca.get_result(&ctx);
        let load = LoadOp::new(&mut ctx, slot, i64_ty);
        load.get_operation().insert_at_back(entry, &ctx);
        let loaded = load.get_result(&ctx);
        ReturnOp::new(&mut ctx, Some(loaded))
            .get_operation()
            .insert_at_back(entry, &ctx);

        let text = run_mem2reg(&mut ctx, func);

        assert!(!text.contains("llvm.alloca"), "{text}");
        assert!(!text.contains("llvm.load"), "{text}");
        assert!(text.contains("llvm.undef"), "{text}");
    }
}
