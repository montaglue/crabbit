//! Inline calls to module-internal LLVM functions into their callers.
//!
//! Runs on the block-argument form of the LLVM dialect (before
//! `lower-llvm-block-args-to-phi`): callee bodies must not contain
//! `llvm.phi` operations. After inlining, internal functions that are no
//! longer referenced anywhere in the module are removed.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    arg_err,
    common_traits::Named,
    context::{Context, Ptr},
    debug_info::{get_operation_result_name, set_operation_result_name},
    dialects::{
        builtin::op_interfaces::{OneRegionInterface, SymbolOpInterface},
        llvm::{
            attributes::LinkageAttr,
            op_interfaces::IsDeclaration,
            ops::{AddressOfOp, BrOp, CallOp, FuncOp, PhiOp, ReturnOp},
        },
    },
    identifier::Identifier,
    ir::{
        basic_block::BasicBlock,
        location::{Located, Location},
        op::Op,
        operation::Operation,
        r#type::Typed,
        value::Value,
    },
    irbuild::{
        cloning,
        listener::DummyListener,
        rewriter::{IRRewriter, Rewriter},
    },
    linked_list::{ContainsLinkedList, LinkedList},
    conversion::pass::{Pass, PassOptions},
    result::STAIRResult,
};

pub struct LLVMInlinePass {
    /// Callees whose body has more operations than this are not inlined.
    pub max_callee_ops: usize,
    /// Stop inlining into a caller once it has grown by this many operations.
    pub max_caller_growth: usize,
}

impl Default for LLVMInlinePass {
    fn default() -> Self {
        LLVMInlinePass {
            max_callee_ops: 400,
            max_caller_growth: 4000,
        }
    }
}

impl Pass for LLVMInlinePass {
    fn name(&self) -> &str {
        "llvm-inline"
    }

    fn run(
        &self,
        root: Ptr<Operation>,
        ctx: &mut Context,
        _options: PassOptions,
    ) -> STAIRResult<Ptr<Operation>> {
        let funcs = collect_functions(ctx, root);
        let by_symbol: FxHashMap<Identifier, FuncOp> = funcs
            .iter()
            .map(|func| (func.get_symbol_name(ctx), *func))
            .collect();

        for func in &funcs {
            self.inline_into_function(ctx, *func, &by_symbol)?;
        }

        remove_dead_internal_functions(ctx, root);
        Ok(root)
    }
}

pub(crate) fn collect_functions(ctx: &Context, root: Ptr<Operation>) -> Vec<FuncOp> {
    let mut funcs = Vec::new();
    if Operation::get_opid(root, ctx) == FuncOp::get_opid_static() {
        funcs.push(FuncOp::from_operation(root));
    }
    let regions: Vec<_> = root.deref(ctx).regions().collect();
    for region in regions {
        let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
        for block in blocks {
            let ops: Vec<_> = block.deref(ctx).iter(ctx).collect();
            for op in ops {
                if Operation::get_opid(op, ctx) == FuncOp::get_opid_static() {
                    funcs.push(FuncOp::from_operation(op));
                }
            }
        }
    }
    funcs
}

impl LLVMInlinePass {
    fn inline_into_function(
        &self,
        ctx: &mut Context,
        func: FuncOp,
        by_symbol: &FxHashMap<Identifier, FuncOp>,
    ) -> STAIRResult<()> {
        if func.is_declaration(ctx) {
            return Ok(());
        }
        let caller_symbol = func.get_symbol_name(ctx);
        let mut budget = self.max_caller_growth;
        let mut tag = 0usize;

        while let Some((call, callee, callee_size)) =
            self.find_inlinable_call(ctx, func, &caller_symbol, by_symbol, budget)
        {
            inline_call(ctx, func, call, callee, tag)?;
            budget = budget.saturating_sub(callee_size);
            tag += 1;
        }
        Ok(())
    }

    /// Find the first call in `func` whose callee should be inlined: a
    /// direct call to an internal, non-declaration function in this module
    /// that fits the size budget and whose body this inliner can clone.
    fn find_inlinable_call(
        &self,
        ctx: &Context,
        func: FuncOp,
        caller_symbol: &Identifier,
        by_symbol: &FxHashMap<Identifier, FuncOp>,
        budget: usize,
    ) -> Option<(CallOp, FuncOp, usize)> {
        let blocks: Vec<_> = func.get_region(ctx).deref(ctx).iter(ctx).collect();
        for block in blocks {
            let ops: Vec<_> = block.deref(ctx).iter(ctx).collect();
            for op in ops {
                if Operation::get_opid(op, ctx) != CallOp::get_opid_static() {
                    continue;
                }
                let call = CallOp::from_operation(op);
                let Some(callee_symbol) = call.get_callee(ctx) else {
                    continue;
                };
                if callee_symbol == *caller_symbol {
                    continue;
                }
                let Some(callee) = by_symbol.get(&callee_symbol) else {
                    continue;
                };
                if callee.is_declaration(ctx)
                    || !matches!(callee.get_linkage(ctx), LinkageAttr::Internal)
                {
                    continue;
                }
                let Some(size) = clonable_body_size(ctx, *callee) else {
                    continue;
                };
                if size > self.max_callee_ops || size > budget {
                    continue;
                }
                return Some((call, *callee, size));
            }
        }
        None
    }
}

/// Count the callee's body operations, or `None` if the body contains
/// something the inliner cannot clone (nested regions or phi ops).
fn clonable_body_size(ctx: &Context, callee: FuncOp) -> Option<usize> {
    let mut size = 0usize;
    for block in callee.get_region(ctx).deref(ctx).iter(ctx) {
        for op in block.deref(ctx).iter(ctx) {
            let op_ref = op.deref(ctx);
            if op_ref.num_regions() != 0 || Operation::get_opid(op, ctx) == PhiOp::get_opid_static() {
                return None;
            }
            size += 1;
        }
    }
    Some(size)
}

fn inline_call(
    ctx: &mut Context,
    caller: FuncOp,
    call: CallOp,
    callee: FuncOp,
    tag: usize,
) -> STAIRResult<()> {
    let call_op = call.get_operation();
    let Some(call_block) = call_op.deref(ctx).get_parent_block() else {
        return arg_err!(Location::Unknown, "llvm-inline: call has no parent block");
    };
    let caller_symbol = caller.get_symbol_name(ctx);

    let callee_blocks: Vec<_> = callee.get_region(ctx).deref(ctx).iter(ctx).collect();
    let reachable = reachable_blocks(ctx, callee_blocks[0]);
    let rpo = reverse_post_order(ctx, callee_blocks[0]);

    // Create one empty block per reachable callee block, mirroring its
    // block arguments, and record the argument mapping. Blocks are laid out
    // after the call block in the callee's original order.
    let mut block_map: FxHashMap<Ptr<BasicBlock>, Ptr<BasicBlock>> = FxHashMap::default();
    let mut value_map: FxHashMap<Value, Value> = FxHashMap::default();
    let mut insert_after = call_block;
    for &old_block in callee_blocks.iter().filter(|b| reachable.contains(b)) {
        let arg_types: Vec<_> = (0..old_block.deref(ctx).get_num_arguments())
            .map(|idx| old_block.deref(ctx).get_argument(idx).get_type(ctx))
            .collect();
        let name = inlined_block_name(ctx, &caller_symbol, tag, old_block)?;
        let new_block = BasicBlock::new(ctx, Some(name), arg_types);
        new_block.insert_after(ctx, insert_after);
        insert_after = new_block;
        for arg_idx in 0..old_block.deref(ctx).get_num_arguments() {
            value_map.insert(
                old_block.deref(ctx).get_argument(arg_idx),
                new_block.deref(ctx).get_argument(arg_idx),
            );
        }
        block_map.insert(old_block, new_block);
    }

    // The continuation block receives the return value as a block argument.
    let result_types: Vec<_> = call_op.deref(ctx).result_types().collect();
    let tail_name = Identifier::try_from(format!("inl{tag}_ret"))
        .map_err(|_| crate::input_error_noloc!("llvm-inline: invalid block name"))?;
    let tail = BasicBlock::new(ctx, Some(tail_name), result_types);
    tail.insert_after(ctx, insert_after);

    // Move everything after the call (including the caller's terminator)
    // into the continuation block.
    let mut next = call_op.deref(ctx).get_next();
    while let Some(op) = next {
        next = op.deref(ctx).get_next();
        op.unlink(ctx);
        op.insert_at_back(tail, ctx);
    }

    // Clone the callee body in reverse post-order so every SSA definition
    // is cloned before its uses.
    for &old_block in &rpo {
        let new_block = block_map[&old_block];
        let ops: Vec<_> = old_block.deref(ctx).iter(ctx).collect();
        for old_op in ops {
            if Operation::get_opid(old_op, ctx) == ReturnOp::get_opid_static() {
                let operands: Vec<_> = old_op.deref(ctx).operands().collect();
                let mapped = map_values(ctx, &value_map, &operands)?;
                BrOp::new(ctx, tail, mapped)
                    .get_operation()
                    .insert_at_back(new_block, ctx);
                continue;
            }
            let new_op = clone_operation(ctx, old_op, &value_map, &block_map)?;
            new_op.insert_at_back(new_block, ctx);
            for res_idx in 0..old_op.deref(ctx).get_num_results() {
                value_map.insert(
                    old_op.deref(ctx).get_result(res_idx),
                    new_op.deref(ctx).get_result(res_idx),
                );
            }
        }
    }

    // Redirect the caller into the inlined entry and replace the call
    // result with the continuation block argument.
    let args = call.get_args(ctx);
    BrOp::new(ctx, block_map[&callee_blocks[0]], args)
        .get_operation()
        .insert_at_back(call_block, ctx);
    if call_op.deref(ctx).get_num_results() > 0 {
        let result = call_op.deref(ctx).get_result(0);
        let replacement = tail.deref(ctx).get_argument(0);
        result.replace_some_uses_with(ctx, |_, _| true, &replacement);
    }
    Operation::erase(call_op, ctx);

    Ok(())
}

fn inlined_block_name(
    ctx: &Context,
    _caller_symbol: &Identifier,
    tag: usize,
    old_block: Ptr<BasicBlock>,
) -> STAIRResult<Identifier> {
    let base = old_block
        .deref(ctx)
        .given_name(ctx)
        .map(|name| name.to_string())
        .unwrap_or_else(|| "bb".to_string());
    Identifier::try_from(format!("inl{tag}_{base}"))
        .map_err(|_| crate::input_error_noloc!("llvm-inline: invalid block name"))
}

fn reachable_blocks(ctx: &Context, entry: Ptr<BasicBlock>) -> FxHashSet<Ptr<BasicBlock>> {
    let mut seen = FxHashSet::default();
    let mut stack = vec![entry];
    while let Some(block) = stack.pop() {
        if !seen.insert(block) {
            continue;
        }
        for succ in block.deref(ctx).succs(ctx) {
            stack.push(succ);
        }
    }
    seen
}

fn reverse_post_order(ctx: &Context, entry: Ptr<BasicBlock>) -> Vec<Ptr<BasicBlock>> {
    let mut post_order = Vec::new();
    let mut visited = FxHashSet::default();
    // Iterative DFS with an explicit stack of (block, next successor index).
    let mut stack = vec![(entry, 0usize)];
    visited.insert(entry);
    while let Some((block, succ_idx)) = stack.pop() {
        let succs = block.deref(ctx).succs(ctx);
        if succ_idx < succs.len() {
            stack.push((block, succ_idx + 1));
            let succ = succs[succ_idx];
            if visited.insert(succ) {
                stack.push((succ, 0));
            }
        } else {
            post_order.push(block);
        }
    }
    post_order.reverse();
    post_order
}

fn map_values(
    ctx: &Context,
    value_map: &FxHashMap<Value, Value>,
    values: &[Value],
) -> STAIRResult<Vec<Value>> {
    values
        .iter()
        .map(|value| {
            value_map.get(value).copied().ok_or_else(|| {
                crate::input_error_noloc!(
                    "llvm-inline: callee value {} has no mapping",
                    value.unique_name(ctx)
                )
            })
        })
        .collect()
}

/// Clone a region-free operation with operands, successors, attributes,
/// location and result names remapped into the caller.
fn clone_operation(
    ctx: &mut Context,
    old_op: Ptr<Operation>,
    value_map: &FxHashMap<Value, Value>,
    block_map: &FxHashMap<Ptr<BasicBlock>, Ptr<BasicBlock>>,
) -> STAIRResult<Ptr<Operation>> {
    let (operands, successors) = {
        let op_ref = old_op.deref(ctx);
        (
            op_ref.operands().collect::<Vec<_>>(),
            op_ref.successors().collect::<Vec<_>>(),
        )
    };

    let mut mapper = cloning::IrMapping::new();
    for operand in &operands {
        let mapped = value_map.get(operand).copied().ok_or_else(|| {
            crate::input_error_noloc!(
                "llvm-inline: callee value {} has no mapping",
                operand.unique_name(ctx)
            )
        })?;
        mapper.map_value(*operand, mapped);
    }
    for succ in &successors {
        let mapped = block_map.get(succ).copied().ok_or_else(|| {
            crate::input_error_noloc!("llvm-inline: callee successor block has no mapping")
        })?;
        mapper.map_block(*succ, mapped);
    }

    let mut rewriter = IRRewriter::<DummyListener>::default();
    let new_op = cloning::clone_operation(old_op, ctx, &mut rewriter, &mut mapper);
    for res_idx in 0..old_op.deref(ctx).get_num_results() {
        if let Some(name) = get_operation_result_name(ctx, old_op, res_idx) {
            set_operation_result_name(ctx, new_op, res_idx, Some(name));
        }
    }
    Ok(new_op)
}

/// Remove internal functions that are no longer referenced by any call or
/// addressof in the module. Removal can make other internal functions
/// unreferenced, so iterate to a fixpoint.
fn remove_dead_internal_functions(ctx: &mut Context, root: Ptr<Operation>) {
    loop {
        let funcs = collect_functions(ctx, root);
        let referenced = referenced_symbols(ctx, root);
        let mut changed = false;
        for func in funcs {
            if func.is_declaration(ctx)
                || !matches!(func.get_linkage(ctx), LinkageAttr::Internal)
            {
                continue;
            }
            if !referenced.contains(&func.get_symbol_name(ctx)) {
                Operation::erase(func.get_operation(), ctx);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}

fn referenced_symbols(ctx: &Context, root: Ptr<Operation>) -> FxHashSet<Identifier> {
    let mut symbols = FxHashSet::default();
    collect_referenced_symbols(ctx, root, &mut symbols);
    symbols
}

fn collect_referenced_symbols(
    ctx: &Context,
    op: Ptr<Operation>,
    symbols: &mut FxHashSet<Identifier>,
) {
    let opid = Operation::get_opid(op, ctx);
    if opid == CallOp::get_opid_static() {
        if let Some(callee) = (CallOp::from_operation(op)).get_callee(ctx) {
            symbols.insert(callee);
        }
    } else if opid == AddressOfOp::get_opid_static() {
        symbols.insert((AddressOfOp::from_operation(op)).get_symbol(ctx));
    }

    let regions: Vec<_> = op.deref(ctx).regions().collect();
    for region in regions {
        let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
        for block in blocks {
            let ops: Vec<_> = block.deref(ctx).iter(ctx).collect();
            for nested in ops {
                collect_referenced_symbols(ctx, nested, symbols);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dialects::{
            builtin::{
                self,
                attributes::IntegerAttr,
                op_interfaces::{OneResultInterface, SingleBlockRegionInterface},
                ops::ModuleOp,
                types::Signedness,
            },
            llvm::{
                self,
                ops::{AddOp, ConstantOp, ReturnOp},
                types::FuncType,
            },
        },
        ir::r#type::{TypeHandle, TypedHandle},
        printable::Printable,
        utils::apint::APInt,
    };
    use std::num::NonZero;

    fn test_context() -> Context {
        let mut ctx = Context::new();
        llvm::register(&mut ctx);
        ctx
    }

    fn i64_type(ctx: &mut Context) -> TypeHandle {
        builtin::types::IntegerType::get(ctx, 64, Signedness::Signless).into()
    }

    #[test]
    fn inlines_internal_callee_and_removes_it() {
        let mut ctx = test_context();
        let module = ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let module_block = module.get_body(&ctx, 0);
        let i64_ty = i64_type(&mut ctx);
        let fn_ty: TypedHandle<FuncType> = FuncType::get(&mut ctx, i64_ty, vec![i64_ty], false);

        // callee: internal, returns arg + 1
        let callee = llvm::ops::FuncOp::new(
            &mut ctx,
            "callee".try_into().unwrap(),
            fn_ty,
            LinkageAttr::Internal,
        );
        callee.get_operation().insert_at_back(module_block, &ctx);
        let callee_entry = callee.get_entry_block(&ctx);
        let callee_arg = callee_entry.deref(&ctx).get_argument(0);
        let int_ty = builtin::types::IntegerType::get(&mut ctx, 64, Signedness::Signless);
        let one = ConstantOp::new_integer(
            &mut ctx,
            IntegerAttr::new(int_ty, APInt::from_u64(1, NonZero::new(64).unwrap())),
        );
        one.get_operation().insert_at_back(callee_entry, &ctx);
        let one_val = one.get_result(&ctx);
        let add = AddOp::new(&mut ctx, callee_arg, one_val);
        add.get_operation().insert_at_back(callee_entry, &ctx);
        let sum = add.get_result(&ctx);
        ReturnOp::new(&mut ctx, Some(sum))
            .get_operation()
            .insert_at_back(callee_entry, &ctx);

        // caller: external, returns callee(arg)
        let caller = llvm::ops::FuncOp::new(
            &mut ctx,
            "caller".try_into().unwrap(),
            fn_ty,
            LinkageAttr::External,
        );
        caller.get_operation().insert_at_back(module_block, &ctx);
        let caller_entry = caller.get_entry_block(&ctx);
        let caller_arg = caller_entry.deref(&ctx).get_argument(0);
        let call = CallOp::new_direct(
            &mut ctx,
            "callee".try_into().unwrap(),
            vec![caller_arg],
            Some(i64_ty),
        );
        call.get_operation().insert_at_back(caller_entry, &ctx);
        let call_res = call.get_operation().deref(&ctx).get_result(0);
        ReturnOp::new(&mut ctx, Some(call_res))
            .get_operation()
            .insert_at_back(caller_entry, &ctx);

        LLVMInlinePass::default()
            .run(module.get_operation(), &mut ctx, PassOptions::default())
            .unwrap();

        let text = format!("{}", module.get_operation().disp(&ctx));
        assert!(!text.contains("llvm.call"), "{text}");
        assert!(text.contains("llvm.add"), "{text}");
        // Dead internal callee removed; only the caller remains.
        assert!(!text.contains("@callee"), "{text}");
        assert!(text.contains("@caller"), "{text}");
    }
}
