//! Direct LLVM-to-AArch64 instruction selection.
//!
//! This remains a single pass because it owns function-wide lowering state:
//! SSA values, virtual-register allocation, stack-slot allocation, literal
//! labels, and machine CFG edge blocks. Independent domains live in sibling
//! modules; this file coordinates function and instruction lowering.

use std::collections::HashMap;

use crate::{
    common_traits::Named,
    context::{Context, Ptr},
    dialects::{
        aarch64::{
            attributes::{AbiLocation, ConditionCode, FunctionAbi, FunctionAbiAttr},
            op_interfaces::Aarch64Opcode,
            ops::{self as aarch64_ops, FuncOp as Aarch64FuncOp},
            registers::{LR, Register, X8, X16},
        },
        builtin::op_interfaces::{OneRegionInterface, OneResultInterface, SymbolOpInterface},
        llvm::{
            attributes::ICmpPredicateAttr,
            op_interfaces::IsDeclaration,
            ops::{
                AddressOfOp, AllocaOp, BitcastOp, BrOp, CStrOp, CallOp, CondBrOp, ConstantOp,
                ExtractValueOp, FuncOp as LlvmFuncOp, GetElementPtrOp, GlobalOp as LlvmGlobalOp,
                ICmpOp, InsertValueOp, IntToPtrOp, LoadOp, PtrToIntOp, ReturnOp, StoreOp, TruncOp,
                UndefOp, UnreachableOp, ZExtOp,
            },
        },
    },
    input_error_noloc,
    ir::{
        basic_block::BasicBlock,
        op::Op,
        operation::Operation,
        r#type::{TypeHandle, Typed},
        value::Value,
    },
    linked_list::{ContainsLinkedList, LinkedList},
    passes::hot_path::{BranchProbability, HotPathInfo},
    result::STAIRResult,
};

use super::isel_control_flow::{branch_edge_target, emit_block_arg_copies, machine_block};
use super::isel_i128::{lower_binary_128, lower_compare_value};
use super::isel_memory_abi::{
    ResultLocation, adapt_value_to_type, aggregate_field_layout, align_to, emit_return_value,
    load_gpr_aggregate_result, load_memory, load_stack_value, lower_gep, result_location_for_type,
    scalar_size_of, stack_align_of, stack_size_of, store_memory, word_ty,
};
use crate::conversion::pass::{AnalysisManager, Pass, PassResult, changed};

use super::{
    attrs::ATTR_KEY_DARWIN_ABI,
    error::Aarch64DarwinErr,
    frontend::{BinaryKind, binary_kind, collect_entry_arguments, module_op, validate_linkage},
    util::module_body,
};

/// Instruction selection: rewrites the module in place, lowering every
/// defined `llvm.func` to an `aarch64.func` and erasing the `llvm` ops.
pub struct LlvmToAarch64IselPass;

impl Pass for LlvmToAarch64IselPass {
    fn name(&self) -> &str {
        "llvm-to-aarch64-isel"
    }

    fn run(
        &mut self,
        root: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> pliron::result::Result<PassResult> {
        let module = module_op(ctx, root)?;
        let body = module_body(ctx, module);
        let llvm_ops: Vec<_> = body.deref(ctx).iter(ctx).collect();

        let mut globals = HashMap::<crate::identifier::Identifier, Vec<u8>>::new();
        for op_ptr in llvm_ops.iter().copied() {
            let op_obj = Operation::get_op_dyn(op_ptr, ctx);
            if let Some(global) = op_obj.downcast_ref::<LlvmGlobalOp>() {
                if let Some(bytes) = global.get_initializer_bytes(ctx) {
                    globals.insert(global.get_symbol_name(ctx), bytes);
                }
            }
        }

        for op_ptr in llvm_ops.iter().copied() {
            let op_obj = Operation::get_op_dyn(op_ptr, ctx);
            if let Some(llvm_func) = op_obj.downcast_ref::<LlvmFuncOp>() {
                if !llvm_func.is_declaration(ctx) {
                    lower_function(ctx, llvm_func, body, &globals)?;
                }
            }
        }

        for op_ptr in llvm_ops {
            Operation::erase(op_ptr, ctx);
        }
        Ok(changed())
    }
}

// Function orchestration ----------------------------------------------------

/// The target-level shape of one function before its instructions are lowered.
///
/// This separates function construction and CFG creation from instruction
/// selection. `FunctionLowerer` owns the mutable value, register, and frame
/// state used while filling this plan.
struct MachineFunctionPlan {
    func: Aarch64FuncOp,
    entry: Ptr<BasicBlock>,
    region: Ptr<crate::ir::region::Region>,
    blocks: Vec<Ptr<BasicBlock>>,
    block_map: HashMap<Ptr<BasicBlock>, Ptr<BasicBlock>>,
    abi: FunctionAbi,
    has_call: bool,
}

impl MachineFunctionPlan {
    fn create(
        ctx: &mut Context,
        llvm_func: &LlvmFuncOp,
        module_body: Ptr<BasicBlock>,
    ) -> STAIRResult<Self> {
        let name = llvm_func.get_symbol_name(ctx);
        let abi = function_abi(ctx, llvm_func)?;
        let linkage = validate_linkage(&name.to_string(), llvm_func.get_linkage(ctx))?;
        let func = Aarch64FuncOp::new(ctx, name, linkage);
        func.get_operation().insert_at_back(module_body, ctx);
        let entry = func.entry_block(ctx);

        let blocks: Vec<_> = llvm_func.get_region(ctx).deref(ctx).iter(ctx).collect();
        let region = func.get_region(ctx);
        let mut block_map = HashMap::new();
        for (index, llvm_block) in blocks.iter().copied().enumerate() {
            let aarch64_block = if index == 0 {
                entry
            } else {
                let label = llvm_block.deref(ctx).unique_name(ctx).to_string();
                let block = BasicBlock::new(ctx, Some(label.try_into().unwrap()), vec![]);
                block.insert_at_back(region, ctx);
                block
            };
            block_map.insert(llvm_block, aarch64_block);
        }

        let has_call = blocks
            .iter()
            .copied()
            .any(|block| block_contains_call(ctx, block));
        Ok(Self {
            func,
            entry,
            region,
            blocks,
            block_map,
            abi,
            has_call,
        })
    }
}

/// The Darwin ABI locations the abi pass recorded on `llvm_func`. Running
/// instruction selection on a function the abi pass has not seen is a
/// pipeline error.
fn function_abi(ctx: &Context, llvm_func: &LlvmFuncOp) -> STAIRResult<FunctionAbi> {
    llvm_func
        .get_operation()
        .deref(ctx)
        .attributes
        .get::<FunctionAbiAttr>(&ATTR_KEY_DARWIN_ABI)
        .map(|attr| attr.0.clone())
        .ok_or_else(|| {
            input_error_noloc!(Aarch64DarwinErr::UnsupportedOp(format!(
                "`{}` has no Darwin ABI locations; the abi pass must run before isel",
                llvm_func.get_symbol_name(ctx)
            )))
        })
}

/// Reverse post-order of the blocks reachable from `entry`. Unlike
/// [topological_order](crate::graph::traversals::region::topological_order),
/// this never roots the traversal at an unreachable block, so a dominating
/// definition's block is guaranteed to come before all of its users' blocks.
fn entry_reverse_post_order(ctx: &Context, entry: Ptr<BasicBlock>) -> Vec<Ptr<BasicBlock>> {
    let mut visited = std::collections::HashSet::new();
    visited.insert(entry);
    let mut post_order = Vec::new();
    let mut stack = vec![(entry, 0usize)];
    while let Some((block, succ_idx)) = stack.last_mut() {
        let succs = block.deref(ctx).succs(ctx);
        if let Some(&succ) = succs.get(*succ_idx) {
            *succ_idx += 1;
            if visited.insert(succ) {
                stack.push((succ, 0));
            }
        } else {
            post_order.push(*block);
            stack.pop();
        }
    }
    post_order.reverse();
    post_order
}

fn block_contains_call(ctx: &Context, block: Ptr<BasicBlock>) -> bool {
    let mut op = block.deref(ctx).get_head();
    while let Some(op_ptr) = op {
        let op_obj = Operation::get_op_dyn(op_ptr, ctx);
        if op_obj.downcast_ref::<CallOp>().is_some() {
            return true;
        }
        op = op_ptr.deref(ctx).get_next();
    }
    false
}

fn lower_function(
    ctx: &mut Context,
    llvm_func: &LlvmFuncOp,
    module_body: Ptr<crate::ir::basic_block::BasicBlock>,
    globals: &HashMap<crate::identifier::Identifier, Vec<u8>>,
) -> STAIRResult<()> {
    // Branch probabilities on the LLVM-level CFG (explicit branch weights
    // plus static loop heuristics). They are transferred onto the machine
    // conditional branches below, the way LLVM's instruction selection copies
    // BranchProbabilityInfo onto MachineBasicBlock successor probabilities.
    let branch_probabilities = HotPathInfo::for_op(llvm_func, ctx);
    let plan = MachineFunctionPlan::create(ctx, llvm_func, module_body)?;
    let MachineFunctionPlan {
        func,
        entry,
        region,
        blocks,
        block_map,
        abi,
        has_call,
    } = plan;

    let mut values = HashMap::<Value, LoweredValue>::new();
    let mut next_vreg = 0usize;
    let mut arg_copies: Vec<(Register, Register)> = Vec::new();
    for (arg, location) in collect_entry_arguments(ctx, llvm_func)?
        .into_iter()
        .zip(abi.args)
    {
        match location {
            AbiLocation::Void => {
                values.insert(arg, LoweredValue::Undef);
            }
            AbiLocation::Stack(offset) => {
                if is_128_bit_integer(ctx, arg.get_type(ctx)) {
                    let lo = fresh_vreg(&mut next_vreg);
                    aarch64_ops::ldr_stack_arg(ctx, lo.clone(), offset).insert_at_back(entry, ctx);
                    let hi = fresh_vreg(&mut next_vreg);
                    aarch64_ops::ldr_stack_arg(ctx, hi.clone(), offset + 8)
                        .insert_at_back(entry, ctx);
                    values.insert(arg, LoweredValue::RegPair(lo, hi));
                } else {
                    let dst = fresh_vreg(&mut next_vreg);
                    aarch64_ops::ldr_stack_arg(ctx, dst.clone(), offset).insert_at_back(entry, ctx);
                    values.insert(arg, LoweredValue::Reg(dst));
                }
            }
            // Copy incoming ABI registers into virtual registers: the raw
            // x0..x7 are clobbered by the first call (or argument setup),
            // while a promoted argument value may live for the whole
            // function. The copies are queued so the entry block keeps its
            // stack-arg-loads/link-register-save prefix, which frame
            // lowering inserts the stack adjustment after.
            AbiLocation::GprPair(lo, hi) => {
                let lo_vreg = fresh_vreg(&mut next_vreg);
                let hi_vreg = fresh_vreg(&mut next_vreg);
                arg_copies.push((lo_vreg, lo));
                arg_copies.push((hi_vreg, hi));
                values.insert(arg, LoweredValue::RegPair(lo_vreg, hi_vreg));
            }
            AbiLocation::Gpr(reg) => {
                let dst = fresh_vreg(&mut next_vreg);
                arg_copies.push((dst, reg));
                values.insert(arg, LoweredValue::Reg(dst));
            }
            AbiLocation::IndirectResult { .. } => {
                return Err(input_error_noloc!(Aarch64DarwinErr::UnsupportedOp(
                    "indirect-result ABI location on a function argument".to_string()
                )));
            }
        }
    }

    if has_call {
        aarch64_ops::str_pre_sp(ctx, LR, 16).insert_at_back(entry, ctx);
    }
    for (dst, src) in arg_copies {
        aarch64_ops::mov(ctx, dst, src).insert_at_back(entry, ctx);
    }

    let llvm_entry = llvm_func.get_entry_block(ctx);
    for block in &blocks {
        if *block == llvm_entry {
            continue;
        }
        for arg in block.deref(ctx).arguments() {
            let dst = fresh_vreg(&mut next_vreg);
            values.insert(arg, LoweredValue::Reg(dst));
        }
    }

    let mut next_literal = 0usize;
    let mut next_edge_block = 0usize;
    let mut stack = StackAllocator::default();
    let sret_result_slot = if let AbiLocation::IndirectResult { reg } = abi.result {
        let sret_ptr_ty = word_ty(ctx);
        let slot = stack.allocate(ctx, sret_ptr_ty)?;
        aarch64_ops::str_sp_offset(ctx, reg, slot.offset).insert_at_back(entry, ctx);
        Some(slot)
    } else {
        None
    };
    // Lower blocks in reverse post-order from the entry, so a value's
    // defining block is visited before every block that uses it (mem2reg
    // introduces direct cross-block SSA uses). Instructions are emitted into
    // each block's own machine block, so this does not change the machine
    // block layout. Unreachable blocks follow in layout order; they can only
    // use values from reachable defs or from each other in layout order.
    let mut ordered = entry_reverse_post_order(ctx, blocks[0]);
    let reachable: std::collections::HashSet<_> = ordered.iter().copied().collect();
    ordered.extend(
        blocks
            .iter()
            .copied()
            .filter(|block| !reachable.contains(block)),
    );
    for block in ordered {
        let insert_block = *block_map.get(&block).ok_or_else(|| {
            input_error_noloc!(Aarch64DarwinErr::UnsupportedOp(
                "missing lowered AArch64 block".to_string()
            ))
        })?;
        let mut op = block.deref(ctx).get_head();
        while let Some(op_ptr) = op {
            op = op_ptr.deref(ctx).get_next();
            let opid = Operation::get_opid(op_ptr, ctx);
            let op_obj = Operation::get_op_dyn(op_ptr, ctx);
            if let Some(alloca) = op_obj.downcast_ref::<AllocaOp>() {
                let slot = stack.allocate(ctx, alloca.get_elem_type(ctx))?;
                values.insert(alloca.get_result(ctx), LoweredValue::StackAddr(slot));
            } else if let Some(constant) = op_obj.downcast_ref::<ConstantOp>() {
                let attr = constant
                    .get_value(ctx)
                    .expect("constant verified to be integer");
                values.insert(
                    constant.get_result(ctx),
                    LoweredValue::Imm(attr.value().to_u128()),
                );
            } else if let Some(cstr) = op_obj.downcast_ref::<CStrOp>() {
                let label = format!(
                    "L_stair_cstr_{}_{}",
                    llvm_func.get_symbol_name(ctx),
                    next_literal
                );
                next_literal += 1;
                values.insert(
                    cstr.get_result(ctx),
                    LoweredValue::CStr {
                        label,
                        bytes: cstr.get_value(ctx).as_bytes().to_vec(),
                    },
                );
            } else if let Some(addr) = op_obj.downcast_ref::<AddressOfOp>() {
                let symbol = addr.get_symbol(ctx);
                if let Some(bytes) = globals.get(&symbol).cloned() {
                    values.insert(
                        addr.get_result(ctx),
                        LoweredValue::CStr {
                            label: symbol.to_string(),
                            bytes,
                        },
                    );
                } else {
                    // Not a byte global: take the address of a function
                    // defined in this module.
                    let dst = fresh_vreg(&mut next_vreg);
                    aarch64_ops::adr_function(ctx, dst.clone(), symbol)
                        .insert_at_back(insert_block, ctx);
                    values.insert(addr.get_result(ctx), LoweredValue::Reg(dst));
                }
            } else if let Some(undef) = op_obj.downcast_ref::<UndefOp>() {
                values.insert(undef.get_result(ctx), LoweredValue::Undef);
            } else if let Some(insert) = op_obj.downcast_ref::<InsertValueOp>() {
                let aggregate = lookup_value(ctx, &values, insert.get_aggregate(ctx))?;
                let value = lookup_value(ctx, &values, insert.get_value(ctx))?;
                values.insert(
                    insert.get_result(ctx),
                    insert_aggregate_value(aggregate, &insert.get_indices(ctx), value)?,
                );
            } else if let Some(extract) = op_obj.downcast_ref::<ExtractValueOp>() {
                let aggregate = lookup_value(ctx, &values, extract.get_aggregate(ctx))?;
                values.insert(
                    extract.get_result(ctx),
                    extract_aggregate_value(aggregate, &extract.get_indices(ctx))?,
                );
            } else if let Some(cast) = op_obj.downcast_ref::<IntToPtrOp>() {
                let value = match lookup_value(ctx, &values, cast.get_input(ctx))? {
                    LoweredValue::Imm(imm) if imm & 1 == 1 => {
                        LoweredValue::TaggedLen((imm >> 1) as u64)
                    }
                    other => {
                        let reg = materialize(
                            ctx,
                            insert_block,
                            other,
                            &mut next_vreg,
                            "llvm.inttoptr input",
                        )?;
                        LoweredValue::Reg(reg)
                    }
                };
                values.insert(cast.get_result(ctx), value);
            } else if let Some(cast) = op_obj.downcast_ref::<PtrToIntOp>() {
                let value = lookup_value(ctx, &values, cast.get_input(ctx))?;
                values.insert(cast.get_result(ctx), value);
            } else if let Some(cast) = op_obj.downcast_ref::<BitcastOp>() {
                let value = lookup_value(ctx, &values, cast.get_input(ctx))?;
                let result = cast.get_result(ctx);
                let value = adapt_value_to_type(ctx, value, result.get_type(ctx))?;
                values.insert(result, value);
            } else if let Some(cast) = op_obj.downcast_ref::<ZExtOp>() {
                let value = lookup_value(ctx, &values, cast.get_input(ctx))?;
                values.insert(cast.get_result(ctx), value);
            } else if let Some(cast) = op_obj.downcast_ref::<TruncOp>() {
                let result = cast.get_result(ctx);
                let value = lookup_value(ctx, &values, cast.get_input(ctx))?;
                if let Some(mask) = integer_trunc_mask(ctx, result.get_type(ctx)) {
                    if let LoweredValue::Imm(imm) = value {
                        values.insert(result, LoweredValue::Imm(imm & mask as u128));
                    } else {
                        let src =
                            materialize(ctx, insert_block, value, &mut next_vreg, "trunc input")?;
                        let mask_reg = fresh_vreg(&mut next_vreg);
                        materialize_u64_immediate(ctx, insert_block, mask_reg, mask);
                        let dst = fresh_vreg(&mut next_vreg);
                        aarch64_ops::binary(
                            ctx,
                            aarch64_ops::AndOp::OPCODE,
                            dst.clone(),
                            src,
                            mask_reg,
                        )
                        .insert_at_back(insert_block, ctx);
                        values.insert(result, LoweredValue::Reg(dst));
                    }
                } else {
                    values.insert(result, value);
                }
            } else if let Some(gep) = op_obj.downcast_ref::<GetElementPtrOp>() {
                let dynamic_indices: Vec<_> =
                    gep.get_operation().deref(ctx).operands().skip(1).collect();
                let address = lower_gep(
                    ctx,
                    insert_block,
                    &values,
                    gep.get_base(ctx),
                    &dynamic_indices,
                    &gep.get_indices(ctx).0,
                    gep.get_source_elem_type(ctx),
                    &mut next_vreg,
                )?;
                values.insert(gep.get_result(ctx), address);
            } else if let Some(load) = op_obj.downcast_ref::<LoadOp>() {
                let result = load.get_result(ctx);
                let addr = lookup_value(ctx, &values, load.get_addr(ctx))?;
                let value = load_memory(
                    ctx,
                    insert_block,
                    addr,
                    result.get_type(ctx),
                    &mut next_vreg,
                )?;
                values.insert(result, value);
            } else if let Some(store) = op_obj.downcast_ref::<StoreOp>() {
                let value = store.get_value(ctx);
                let lowered_value = lookup_value(ctx, &values, value)?;
                let addr = lookup_value(ctx, &values, store.get_addr(ctx))?;
                store_memory(
                    ctx,
                    insert_block,
                    addr,
                    lowered_value,
                    value.get_type(ctx),
                    &mut next_vreg,
                )?;
            } else if let Some(call) = op_obj.downcast_ref::<CallOp>() {
                let callee = call.get_callee(ctx);
                let mut args = call.get_args(ctx);
                let callee_ptr = if callee.is_none() {
                    // Indirect call: operand 0 is the callee function pointer.
                    if args.is_empty() {
                        return Err(input_error_noloc!(Aarch64DarwinErr::UnsupportedOp(
                            "indirect llvm.call has no callee operand".to_string()
                        )));
                    }
                    let callee_value = args.remove(0);
                    let lowered = lookup_value(ctx, &values, callee_value)?;
                    Some(materialize_pointer(
                        ctx,
                        insert_block,
                        lowered,
                        &mut next_vreg,
                        "indirect call target",
                    )?)
                } else {
                    None
                };
                let call_op = call.get_operation();
                let result = if call_op.deref(ctx).get_num_results() > 0 {
                    Some(call_op.deref(ctx).get_result(0))
                } else {
                    None
                };
                let result_location = result
                    .map(|result| result_location_for_type(ctx, result.get_type(ctx)))
                    .transpose()?;
                let indirect_result_slot = match (result, result_location) {
                    (Some(result), Some(ResultLocation::IndirectX8)) => {
                        Some(stack.allocate(ctx, result.get_type(ctx))?)
                    }
                    _ => None,
                };
                let mut arg_regs = Vec::with_capacity(args.len());
                for arg in args {
                    let lowered = lookup_value(ctx, &values, arg)?;
                    if is_128_bit_integer(ctx, arg.get_type(ctx)) {
                        let (lo, hi) = materialize_pair(
                            ctx,
                            insert_block,
                            lowered,
                            arg.get_type(ctx),
                            &mut next_vreg,
                            "call argument",
                        )?;
                        arg_regs.push(lo);
                        arg_regs.push(hi);
                    } else {
                        arg_regs.push(materialize_typed(
                            ctx,
                            insert_block,
                            lowered,
                            arg.get_type(ctx),
                            &mut next_vreg,
                            "call argument",
                        )?);
                    }
                }
                let stack_arg_count = arg_regs.len().saturating_sub(8);
                let outgoing_stack_size = align_to_16((stack_arg_count as u64) * 8);
                if outgoing_stack_size > 0 {
                    aarch64_ops::sub_sp_imm(ctx, outgoing_stack_size)
                        .insert_at_back(insert_block, ctx);
                }
                if let Some(callee_ptr) = &callee_ptr {
                    // x16 is an intra-procedure-call scratch register outside
                    // the allocatable set, so it survives the argument moves.
                    aarch64_ops::mov(ctx, X16, *callee_ptr).insert_at_back(insert_block, ctx);
                }
                for (idx, src) in arg_regs.into_iter().enumerate() {
                    if idx < 8 {
                        aarch64_ops::mov(ctx, Register::gpr(idx as u8), src)
                            .insert_at_back(insert_block, ctx);
                    } else {
                        aarch64_ops::str_sp_offset(ctx, src, ((idx - 8) as u64) * 8)
                            .insert_at_back(insert_block, ctx);
                    }
                }
                if let Some(slot) = indirect_result_slot {
                    // sp is already lowered by the outgoing stack-argument
                    // area here, so compensate to reach the frame slot.
                    aarch64_ops::add_sp_offset(ctx, X8, slot.offset + outgoing_stack_size)
                        .insert_at_back(insert_block, ctx);
                }
                match callee {
                    Some(callee) => {
                        aarch64_ops::call(ctx, callee).insert_at_back(insert_block, ctx);
                    }
                    None => {
                        aarch64_ops::blr(ctx, X16).insert_at_back(insert_block, ctx);
                    }
                }
                if outgoing_stack_size > 0 {
                    aarch64_ops::add_sp_imm(ctx, outgoing_stack_size)
                        .insert_at_back(insert_block, ctx);
                }
                if let Some(result) = result {
                    let lowered = match result_location.unwrap() {
                        ResultLocation::ScalarX0 => {
                            let dst = fresh_vreg(&mut next_vreg);
                            aarch64_ops::mov(ctx, dst, Register::gpr(0))
                                .insert_at_back(insert_block, ctx);
                            LoweredValue::Reg(dst)
                        }
                        ResultLocation::ScalarX0X1 => {
                            let lo = fresh_vreg(&mut next_vreg);
                            aarch64_ops::mov(ctx, lo, Register::gpr(0))
                                .insert_at_back(insert_block, ctx);
                            let hi = fresh_vreg(&mut next_vreg);
                            aarch64_ops::mov(ctx, hi, Register::gpr(1))
                                .insert_at_back(insert_block, ctx);
                            LoweredValue::RegPair(lo, hi)
                        }
                        ResultLocation::DirectGprs(count) => load_gpr_aggregate_result(
                            ctx,
                            insert_block,
                            result.get_type(ctx),
                            count,
                            &mut next_vreg,
                        )?,
                        ResultLocation::IndirectX8 => {
                            let slot = indirect_result_slot.ok_or_else(|| {
                                input_error_noloc!(Aarch64DarwinErr::UnsupportedOp(
                                    "missing indirect result slot".to_string()
                                ))
                            })?;
                            load_stack_value(
                                ctx,
                                insert_block,
                                slot.offset,
                                result.get_type(ctx),
                                &mut next_vreg,
                            )?
                        }
                        ResultLocation::Void => LoweredValue::Undef,
                    };
                    values.insert(result, lowered);
                }
            } else if let Some(kind) = binary_kind(&*op_obj) {
                let op_ref = op_ptr.deref(ctx);
                let result = op_ref.get_result(0);
                let lhs = op_ref.get_operand(0);
                let rhs = op_ref.get_operand(1);
                drop(op_ref);
                let lhs_value = lookup_value(ctx, &values, lhs)?;
                let rhs_value = lookup_value(ctx, &values, rhs)?;
                if is_128_bit_integer(ctx, result.get_type(ctx)) {
                    let pair = lower_binary_128(
                        ctx,
                        insert_block,
                        kind,
                        lhs_value,
                        rhs_value,
                        result.get_type(ctx),
                        &mut next_vreg,
                    )?;
                    values.insert(result, pair);
                    continue;
                }
                if let (LoweredValue::Imm(lhs), LoweredValue::Imm(rhs)) = (&lhs_value, &rhs_value)
                    && let Some(imm) = fold_binary(ctx, kind, *lhs, *rhs, result.get_type(ctx))
                {
                    values.insert(result, LoweredValue::Imm(imm));
                    continue;
                }
                let lhs = materialize_typed(
                    ctx,
                    insert_block,
                    lhs_value,
                    lhs.get_type(ctx),
                    &mut next_vreg,
                    "binary lhs",
                )?;
                let rhs = materialize_typed(
                    ctx,
                    insert_block,
                    rhs_value,
                    rhs.get_type(ctx),
                    &mut next_vreg,
                    "binary rhs",
                )?;
                let dst = fresh_vreg(&mut next_vreg);
                if matches!(kind, BinaryKind::SRem | BinaryKind::URem) {
                    let quotient = fresh_vreg(&mut next_vreg);
                    let div_opcode = if kind == BinaryKind::URem {
                        aarch64_ops::UdivOp::OPCODE
                    } else {
                        aarch64_ops::SdivOp::OPCODE
                    };
                    aarch64_ops::binary(
                        ctx,
                        div_opcode,
                        quotient.clone(),
                        lhs.clone(),
                        rhs.clone(),
                    )
                    .insert_at_back(insert_block, ctx);
                    let product = fresh_vreg(&mut next_vreg);
                    aarch64_ops::binary(
                        ctx,
                        aarch64_ops::MulOp::OPCODE,
                        product.clone(),
                        quotient,
                        rhs,
                    )
                    .insert_at_back(insert_block, ctx);
                    aarch64_ops::binary(ctx, aarch64_ops::SubOp::OPCODE, dst.clone(), lhs, product)
                        .insert_at_back(insert_block, ctx);
                } else {
                    aarch64_ops::binary(ctx, opcode(kind), dst.clone(), lhs, rhs)
                        .insert_at_back(insert_block, ctx);
                }
                let dst = normalize_integer_reg(
                    ctx,
                    insert_block,
                    dst,
                    result.get_type(ctx),
                    &mut next_vreg,
                )?;
                values.insert(result, LoweredValue::Reg(dst));
            } else if let Some(icmp) = op_obj.downcast_ref::<ICmpOp>() {
                let lhs = icmp.get_lhs(ctx);
                let rhs = icmp.get_rhs(ctx);
                values.insert(
                    icmp.get_result(ctx),
                    LoweredValue::Compare(CompareValue {
                        predicate: icmp.get_predicate(ctx),
                        lhs_ty: lhs.get_type(ctx),
                        lhs: Box::new(lookup_value(ctx, &values, lhs)?),
                        rhs_ty: rhs.get_type(ctx),
                        rhs: Box::new(lookup_value(ctx, &values, rhs)?),
                    }),
                );
            } else if let Some(ret) = op_obj.downcast_ref::<ReturnOp>() {
                if let Some(value) = ret.retval(ctx) {
                    emit_return_value(
                        ctx,
                        insert_block,
                        &values,
                        value,
                        abi.result,
                        sret_result_slot,
                        &mut next_vreg,
                    )?;
                }
                if has_call {
                    aarch64_ops::ldr_post_sp(ctx, LR, 16).insert_at_back(insert_block, ctx);
                }
                aarch64_ops::ret(ctx).insert_at_back(insert_block, ctx);
            } else if let Some(_unreachable) = op_obj.downcast_ref::<UnreachableOp>() {
                aarch64_ops::brk(ctx).insert_at_back(insert_block, ctx);
            } else if let Some(br) = op_obj.downcast_ref::<BrOp>() {
                let dest = br.get_dest(ctx);
                let args = br.get_dest_operands(ctx);
                emit_block_arg_copies(ctx, insert_block, &values, dest, &args, &mut next_vreg)?;
                let target = machine_block(&block_map, dest)?;
                aarch64_ops::b(ctx, target).insert_at_back(insert_block, ctx);
            } else if let Some(cond_br) = op_obj.downcast_ref::<CondBrOp>() {
                let true_dest = cond_br.get_true_dest(ctx);
                let true_args = cond_br.get_true_operands(ctx);
                let false_dest = cond_br.get_false_dest(ctx);
                let false_args = cond_br.get_false_operands(ctx);
                let true_target = branch_edge_target(
                    ctx,
                    region,
                    &block_map,
                    &values,
                    true_dest,
                    &true_args,
                    &mut next_vreg,
                    &mut next_edge_block,
                )?;
                let false_target = branch_edge_target(
                    ctx,
                    region,
                    &block_map,
                    &values,
                    false_dest,
                    &false_args,
                    &mut next_vreg,
                    &mut next_edge_block,
                )?;
                // Successor 0 of llvm.cond_br is the true (taken) edge.
                let taken = branch_probabilities
                    .successor_probabilities(block)
                    .first()
                    .copied()
                    .unwrap_or_else(|| BranchProbability::from_ratio(1, 2));
                let (taken_weight, not_taken_weight) =
                    (taken.numerator(), taken.complement().numerator());
                let condition_value = lookup_value(ctx, &values, cond_br.get_condition(ctx))?;
                if let LoweredValue::Compare(compare) = condition_value {
                    if is_128_bit_integer(ctx, compare.lhs_ty) {
                        let condition =
                            lower_compare_value(ctx, insert_block, compare, &mut next_vreg)?;
                        let branch = aarch64_ops::cbnz(ctx, condition, true_target);
                        aarch64_ops::set_branch_weights(
                            ctx,
                            branch,
                            taken_weight,
                            not_taken_weight,
                        );
                        branch.insert_at_back(insert_block, ctx);
                        aarch64_ops::b(ctx, false_target).insert_at_back(insert_block, ctx);
                        continue;
                    }
                    let lhs = materialize_typed(
                        ctx,
                        insert_block,
                        *compare.lhs,
                        compare.lhs_ty,
                        &mut next_vreg,
                        "icmp lhs",
                    )?;
                    let rhs = materialize_typed(
                        ctx,
                        insert_block,
                        *compare.rhs,
                        compare.rhs_ty,
                        &mut next_vreg,
                        "icmp rhs",
                    )?;
                    aarch64_ops::cmp(ctx, lhs, rhs).insert_at_back(insert_block, ctx);
                    let branch = aarch64_ops::b_cond(
                        ctx,
                        condition_code(compare.predicate),
                        true_target,
                    );
                    aarch64_ops::set_branch_weights(ctx, branch, taken_weight, not_taken_weight);
                    branch.insert_at_back(insert_block, ctx);
                } else {
                    let condition = materialize(
                        ctx,
                        insert_block,
                        condition_value,
                        &mut next_vreg,
                        "branch condition",
                    )?;
                    let branch = aarch64_ops::cbnz(ctx, condition, true_target);
                    aarch64_ops::set_branch_weights(ctx, branch, taken_weight, not_taken_weight);
                    branch.insert_at_back(insert_block, ctx);
                }
                aarch64_ops::b(ctx, false_target).insert_at_back(insert_block, ctx);
            } else {
                return Err(input_error_noloc!(Aarch64DarwinErr::UnsupportedOp(
                    opid.to_string()
                )));
            }
        }
    }

    func.set_stack_size(ctx, align_to_16(stack.next_offset));
    Ok(())
}

// Lowered values and materialization ----------------------------------------

pub(super) fn fresh_vreg(next_vreg: &mut usize) -> Register {
    let reg = Register::virtual_gpr(*next_vreg as u32);
    *next_vreg += 1;
    reg
}

#[derive(Clone, Debug)]
pub(super) enum LoweredValue {
    Reg(Register),
    RegPair(Register, Register),
    Imm(u128),
    CStr { label: String, bytes: Vec<u8> },
    StackAddr(StackSlot),
    Address { base: Register, offset: u64 },
    Aggregate(Vec<Option<LoweredValue>>),
    Compare(CompareValue),
    TaggedLen(u64),
    Undef,
}

#[derive(Clone, Debug)]
pub(super) struct CompareValue {
    pub(super) predicate: ICmpPredicateAttr,
    pub(super) lhs_ty: TypeHandle,
    pub(super) lhs: Box<LoweredValue>,
    pub(super) rhs_ty: TypeHandle,
    pub(super) rhs: Box<LoweredValue>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct StackSlot {
    pub(super) offset: u64,
}

#[derive(Default)]
struct StackAllocator {
    next_offset: u64,
}

impl StackAllocator {
    fn allocate(
        &mut self,
        ctx: &Context,
        ty: TypeHandle,
    ) -> STAIRResult<StackSlot> {
        let size = stack_size_of(ctx, ty)?;
        let align = stack_align_of(ctx, ty)?;
        self.next_offset = align_to(self.next_offset, align);
        let slot = StackSlot {
            offset: self.next_offset,
        };
        self.next_offset += size;
        Ok(slot)
    }
}

pub(super) fn lookup_value(
    ctx: &Context,
    values: &HashMap<Value, LoweredValue>,
    value: Value,
) -> STAIRResult<LoweredValue> {
    values.get(&value).cloned().ok_or_else(|| {
        input_error_noloc!(Aarch64DarwinErr::UndefinedValue(
            value.unique_name(ctx).to_string()
        ))
    })
}

fn insert_aggregate_value(
    aggregate: LoweredValue,
    indices: &[u32],
    value: LoweredValue,
) -> STAIRResult<LoweredValue> {
    let Some((index, rest)) = indices.split_first() else {
        return Ok(value);
    };
    let mut fields = match aggregate {
        LoweredValue::Aggregate(fields) => fields,
        LoweredValue::Undef => Vec::new(),
        other => {
            return Err(input_error_noloc!(Aarch64DarwinErr::UnsupportedOp(
                format!("llvm.insertvalue into non-aggregate {other:?}")
            )));
        }
    };
    let index = *index as usize;
    if fields.len() <= index {
        fields.resize(index + 1, None);
    }
    let current = fields[index].take().unwrap_or(LoweredValue::Undef);
    fields[index] = Some(insert_aggregate_value(current, rest, value)?);
    Ok(LoweredValue::Aggregate(fields))
}

fn extract_aggregate_value(aggregate: LoweredValue, indices: &[u32]) -> STAIRResult<LoweredValue> {
    let Some((index, rest)) = indices.split_first() else {
        return Ok(aggregate);
    };
    let fields = match aggregate {
        LoweredValue::Aggregate(fields) => fields,
        LoweredValue::Undef => return Ok(LoweredValue::Undef),
        other => {
            return Err(input_error_noloc!(Aarch64DarwinErr::UnsupportedOp(
                format!("llvm.extractvalue from non-aggregate {other:?}")
            )));
        }
    };
    let field = fields
        .get(*index as usize)
        .and_then(|field| field.clone())
        .ok_or_else(|| {
            input_error_noloc!(Aarch64DarwinErr::UndefinedValue(
                "extract from unset aggregate field".to_string()
            ))
        })?;
    extract_aggregate_value(field, rest)
}

pub(super) fn lookup_reg(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    values: &HashMap<Value, LoweredValue>,
    value: Value,
    next_vreg: &mut usize,
) -> STAIRResult<Register> {
    let ty = value.get_type(ctx);
    let lowered = lookup_value(ctx, values, value)?;
    materialize_typed(ctx, entry, lowered, ty, next_vreg, "SSA value")
}

pub(super) fn materialize_typed(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    value: LoweredValue,
    ty: TypeHandle,
    next_vreg: &mut usize,
    context: &str,
) -> STAIRResult<Register> {
    if is_128_bit_integer(ctx, ty) {
        let (lo, _) = materialize_pair(ctx, entry, value, ty, next_vreg, context)?;
        return Ok(lo);
    }
    let reg = materialize(ctx, entry, value, next_vreg, context)?;
    normalize_integer_reg(ctx, entry, reg, ty, next_vreg)
}

pub(super) fn materialize_pair(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    value: LoweredValue,
    ty: TypeHandle,
    next_vreg: &mut usize,
    context: &str,
) -> STAIRResult<(Register, Register)> {
    match value {
        LoweredValue::RegPair(lo, hi) => Ok((lo, hi)),
        LoweredValue::Aggregate(fields) if fields.len() == 2 => {
            let lo = fields
                .first()
                .and_then(|field| field.clone())
                .ok_or_else(|| {
                    input_error_noloc!(Aarch64DarwinErr::UndefinedValue(
                        "missing low 128-bit lane".to_string()
                    ))
                })?;
            let hi = fields
                .get(1)
                .and_then(|field| field.clone())
                .ok_or_else(|| {
                    input_error_noloc!(Aarch64DarwinErr::UndefinedValue(
                        "missing high 128-bit lane".to_string()
                    ))
                })?;
            let lo = materialize(ctx, entry, lo, next_vreg, context)?;
            let hi = materialize(ctx, entry, hi, next_vreg, context)?;
            Ok((lo, hi))
        }
        LoweredValue::Imm(imm) => {
            let lo = fresh_vreg(next_vreg);
            materialize_u64_immediate(ctx, entry, lo, imm as u64);
            let hi = fresh_vreg(next_vreg);
            materialize_u64_immediate(ctx, entry, hi, (imm >> 64) as u64);
            Ok((lo, hi))
        }
        LoweredValue::Reg(lo) => {
            let hi = fresh_vreg(next_vreg);
            if integer_width_and_signedness(ctx, ty)
                .map(|(_, signed)| signed)
                .unwrap_or(false)
            {
                let sign = fresh_vreg(next_vreg);
                materialize_u64_immediate(ctx, entry, sign, 63);
                aarch64_ops::binary(
                    ctx,
                    aarch64_ops::LsrOp::OPCODE,
                    hi.clone(),
                    lo.clone(),
                    sign,
                )
                .insert_at_back(entry, ctx);
                let mask = fresh_vreg(next_vreg);
                materialize_u64_immediate(ctx, entry, mask, 0u64.wrapping_sub(1));
                aarch64_ops::binary(
                    ctx,
                    aarch64_ops::MulOp::OPCODE,
                    hi.clone(),
                    hi.clone(),
                    mask,
                )
                .insert_at_back(entry, ctx);
            } else {
                materialize_u64_immediate(ctx, entry, hi, 0);
            }
            Ok((lo, hi))
        }
        other => {
            let lo = materialize(ctx, entry, other, next_vreg, context)?;
            let hi = fresh_vreg(next_vreg);
            materialize_u64_immediate(ctx, entry, hi, 0);
            Ok((lo, hi))
        }
    }
}

pub(super) fn materialize(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    value: LoweredValue,
    next_vreg: &mut usize,
    context: &str,
) -> STAIRResult<Register> {
    match value {
        LoweredValue::Reg(reg) => Ok(reg),
        LoweredValue::RegPair(lo, _) => Ok(lo),
        LoweredValue::Imm(imm) => {
            let dst = fresh_vreg(next_vreg);
            materialize_u64_immediate(ctx, entry, dst, imm as u64);
            Ok(dst)
        }
        LoweredValue::CStr { label, bytes, .. } => {
            let dst = fresh_vreg(next_vreg);
            aarch64_ops::adr_literal(ctx, dst.clone(), label, bytes).insert_at_back(entry, ctx);
            Ok(dst)
        }
        LoweredValue::TaggedLen(len) => {
            let dst = fresh_vreg(next_vreg);
            materialize_u64_immediate(ctx, entry, dst, (len << 1) | 1);
            Ok(dst)
        }
        LoweredValue::Address { base, offset } => {
            if offset == 0 {
                Ok(base)
            } else {
                let offset_reg = fresh_vreg(next_vreg);
                materialize_u64_immediate(ctx, entry, offset_reg, offset);
                let dst = fresh_vreg(next_vreg);
                aarch64_ops::binary(
                    ctx,
                    aarch64_ops::AddOp::OPCODE,
                    dst.clone(),
                    base,
                    offset_reg,
                )
                .insert_at_back(entry, ctx);
                Ok(dst)
            }
        }
        LoweredValue::StackAddr(slot) => {
            let dst = fresh_vreg(next_vreg);
            aarch64_ops::add_sp_offset(ctx, dst.clone(), slot.offset).insert_at_back(entry, ctx);
            Ok(dst)
        }
        LoweredValue::Compare(compare) => lower_compare_value(ctx, entry, compare, next_vreg),
        LoweredValue::Aggregate(mut fields) if fields.len() == 1 => {
            let field = fields.pop().flatten().ok_or_else(|| {
                input_error_noloc!(Aarch64DarwinErr::UndefinedValue(
                    "materialize from unset aggregate field".to_string()
                ))
            })?;
            materialize(ctx, entry, field, next_vreg, context)
        }
        // An undef value (e.g. mem2reg promoting a slot that is not
        // initialized on every path) may be materialized as any value; use a
        // defined zero so downstream passes see a normal register.
        LoweredValue::Undef => {
            let dst = fresh_vreg(next_vreg);
            materialize_u64_immediate(ctx, entry, dst, 0);
            Ok(dst)
        }
        LoweredValue::Aggregate(_) => Err(input_error_noloc!(Aarch64DarwinErr::UnsupportedOp(
            format!("cannot materialize {context}: {value:?}")
        ))),
    }
}

pub(super) fn materialize_pointer(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    value: LoweredValue,
    next_vreg: &mut usize,
    context: &str,
) -> STAIRResult<Register> {
    match value {
        LoweredValue::Aggregate(fields) => {
            let data = fields
                .first()
                .and_then(|field| field.clone())
                .ok_or_else(|| {
                    input_error_noloc!(Aarch64DarwinErr::UndefinedValue(format!(
                        "missing data pointer for {context}"
                    )))
                })?;
            materialize(ctx, entry, data, next_vreg, context)
        }
        other => materialize(ctx, entry, other, next_vreg, context),
    }
}

pub(super) fn materialize_u64_immediate(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    dst: Register,
    imm: u64,
) {
    if imm <= 0xffff {
        aarch64_ops::mov_imm(ctx, dst, imm).insert_at_back(entry, ctx);
        return;
    }

    let first_shift = (0..4)
        .map(|chunk| chunk * 16)
        .find(|shift| ((imm >> shift) & 0xffff) != 0)
        .unwrap_or(0);
    aarch64_ops::movz(ctx, dst, (imm >> first_shift) & 0xffff, first_shift)
        .insert_at_back(entry, ctx);

    for shift in (0..4).map(|chunk| chunk * 16) {
        if shift == first_shift {
            continue;
        }
        let chunk = (imm >> shift) & 0xffff;
        if chunk == 0 {
            continue;
        }
        aarch64_ops::movk(ctx, dst, chunk, shift).insert_at_back(entry, ctx);
    }
}

// Integer arithmetic and comparisons ----------------------------------------

fn integer_trunc_mask(ctx: &Context, ty: TypeHandle) -> Option<u64> {
    let ty_ref = ty.deref(ctx);
    let int_ty = ty_ref.downcast_ref::<crate::dialects::builtin::types::IntegerType>()?;
    let width = int_ty.width();
    if width >= 64 {
        None
    } else {
        Some((1u64 << width) - 1)
    }
}

pub(super) fn normalize_integer_reg(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    reg: Register,
    ty: TypeHandle,
    next_vreg: &mut usize,
) -> STAIRResult<Register> {
    let Some((width, signed)) = integer_width_and_signedness(ctx, ty) else {
        return Ok(reg);
    };
    if width >= 64 {
        return Ok(reg);
    }

    let mask = (1u64 << width) - 1;
    let mask_reg = fresh_vreg(next_vreg);
    materialize_u64_immediate(ctx, entry, mask_reg, mask);
    let masked = fresh_vreg(next_vreg);
    aarch64_ops::binary(
        ctx,
        aarch64_ops::AndOp::OPCODE,
        masked.clone(),
        reg,
        mask_reg,
    )
    .insert_at_back(entry, ctx);

    if !signed {
        return Ok(masked);
    }

    let sign_bit_reg = fresh_vreg(next_vreg);
    materialize_u64_immediate(ctx, entry, sign_bit_reg, 1u64 << (width - 1));
    let flipped = fresh_vreg(next_vreg);
    aarch64_ops::binary(
        ctx,
        aarch64_ops::XorOp::OPCODE,
        flipped.clone(),
        masked,
        sign_bit_reg.clone(),
    )
    .insert_at_back(entry, ctx);
    let extended = fresh_vreg(next_vreg);
    aarch64_ops::binary(
        ctx,
        aarch64_ops::SubOp::OPCODE,
        extended.clone(),
        flipped,
        sign_bit_reg,
    )
    .insert_at_back(entry, ctx);
    Ok(extended)
}

fn integer_width_and_signedness(
    ctx: &Context,
    ty: TypeHandle,
) -> Option<(u32, bool)> {
    let ty_ref = ty.deref(ctx);
    let int_ty = ty_ref.downcast_ref::<crate::dialects::builtin::types::IntegerType>()?;
    Some((int_ty.width(), int_ty.is_signed()))
}

pub(super) fn is_128_bit_integer(ctx: &Context, ty: TypeHandle) -> bool {
    integer_width_and_signedness(ctx, ty)
        .map(|(width, _)| width == 128)
        .unwrap_or(false)
}

pub(super) fn load_sp_opcode(
    ctx: &Context,
    ty: TypeHandle,
) -> STAIRResult<Aarch64Opcode> {
    Ok(match scalar_size_of(ctx, ty)? {
        1 => aarch64_ops::LdrbSpOffsetOp::OPCODE,
        2 => aarch64_ops::LdrhSpOffsetOp::OPCODE,
        3 | 4 => aarch64_ops::LdrwSpOffsetOp::OPCODE,
        5..=8 => aarch64_ops::LdrSpOffsetOp::OPCODE,
        size => {
            return Err(input_error_noloc!(Aarch64DarwinErr::UnsupportedType(
                format!("scalar load size {size}")
            )));
        }
    })
}

pub(super) fn store_sp_opcode(
    ctx: &Context,
    ty: TypeHandle,
) -> STAIRResult<Aarch64Opcode> {
    Ok(match scalar_size_of(ctx, ty)? {
        1 => aarch64_ops::StrbSpOffsetOp::OPCODE,
        2 => aarch64_ops::StrhSpOffsetOp::OPCODE,
        3 | 4 => aarch64_ops::StrwSpOffsetOp::OPCODE,
        5..=8 => aarch64_ops::StrSpOffsetOp::OPCODE,
        size => {
            return Err(input_error_noloc!(Aarch64DarwinErr::UnsupportedType(
                format!("scalar store size {size}")
            )));
        }
    })
}

pub(super) fn load_reg_opcode(
    ctx: &Context,
    ty: TypeHandle,
) -> STAIRResult<Aarch64Opcode> {
    Ok(match scalar_size_of(ctx, ty)? {
        1 => aarch64_ops::LdrbRegOffsetOp::OPCODE,
        2 => aarch64_ops::LdrhRegOffsetOp::OPCODE,
        3 | 4 => aarch64_ops::LdrwRegOffsetOp::OPCODE,
        5..=8 => aarch64_ops::LdrRegOffsetOp::OPCODE,
        size => {
            return Err(input_error_noloc!(Aarch64DarwinErr::UnsupportedType(
                format!("scalar load size {size}")
            )));
        }
    })
}

pub(super) fn store_reg_opcode(
    ctx: &Context,
    ty: TypeHandle,
) -> STAIRResult<Aarch64Opcode> {
    Ok(match scalar_size_of(ctx, ty)? {
        1 => aarch64_ops::StrbRegOffsetOp::OPCODE,
        2 => aarch64_ops::StrhRegOffsetOp::OPCODE,
        3 | 4 => aarch64_ops::StrwRegOffsetOp::OPCODE,
        5..=8 => aarch64_ops::StrRegOffsetOp::OPCODE,
        size => {
            return Err(input_error_noloc!(Aarch64DarwinErr::UnsupportedType(
                format!("scalar store size {size}")
            )));
        }
    })
}

pub(super) fn is_zero_sized_ty(ctx: &Context, ty: TypeHandle) -> bool {
    ty.deref(ctx)
        .downcast_ref::<crate::dialects::builtin::types::UnitType>()
        .is_some()
}

pub(super) fn is_stack_scalar_ty(ctx: &Context, ty: TypeHandle) -> bool {
    let ty_ref = ty.deref(ctx);
    ty_ref
        .downcast_ref::<crate::dialects::builtin::types::IntegerType>()
        .is_some()
        || ty_ref
            .downcast_ref::<crate::dialects::llvm::types::PointerType>()
            .is_some()
}

pub(super) fn is_aggregate_ty(ctx: &Context, ty: TypeHandle) -> bool {
    let ty_ref = ty.deref(ctx);
    ty_ref
        .downcast_ref::<crate::dialects::llvm::types::ArrayType>()
        .is_some()
        || ty_ref
            .downcast_ref::<crate::dialects::llvm::types::StructType>()
            .is_some()
}

pub(super) fn indexed_element(
    ctx: &Context,
    ty: TypeHandle,
) -> STAIRResult<(TypeHandle, u64)> {
    let ty_ref = ty.deref(ctx);
    if let Some(array_ty) = ty_ref.downcast_ref::<crate::dialects::llvm::types::ArrayType>() {
        let elem_ty = array_ty.elem_type();
        drop(ty_ref);
        let elem_size = stack_size_of(ctx, elem_ty)?;
        if elem_size == 0 {
            return Ok((elem_ty, 0));
        }
        return Ok((elem_ty, align_to(elem_size, stack_align_of(ctx, elem_ty)?)));
    }
    drop(ty_ref);
    Ok((ty, stack_size_of(ctx, ty)?))
}

pub(super) fn struct_field_offset(
    ctx: &Context,
    ty: TypeHandle,
    index: Option<u64>,
) -> STAIRResult<Option<(u64, TypeHandle)>> {
    let Some(index) = index else {
        return Ok(None);
    };
    if ty
        .deref(ctx)
        .downcast_ref::<crate::dialects::llvm::types::StructType>()
        .is_none()
    {
        return Ok(None);
    }
    let layout = aggregate_field_layout(ctx, ty)?;
    let index = index as usize;
    layout.get(index).copied().map(Some).ok_or_else(|| {
        input_error_noloc!(Aarch64DarwinErr::UnsupportedOp(
            "llvm.gep struct field index is out of bounds".to_string()
        ))
    })
}

pub(super) fn struct_fields(
    ctx: &Context,
    ty: TypeHandle,
) -> STAIRResult<Vec<TypeHandle>> {
    let ty_ref = ty.deref(ctx);
    if ty_ref
        .downcast_ref::<crate::dialects::builtin::types::UnitType>()
        .is_some()
    {
        return Ok(Vec::new());
    }
    if let Some(array_ty) = ty_ref.downcast_ref::<crate::dialects::llvm::types::ArrayType>() {
        return Ok(vec![array_ty.elem_type(); array_ty.size() as usize]);
    }
    let Some(struct_ty) = ty_ref.downcast_ref::<crate::dialects::llvm::types::StructType>() else {
        return Err(input_error_noloc!(Aarch64DarwinErr::UnsupportedType(
            format!("{:?}", &*ty_ref)
        )));
    };
    struct_ty
        .fields()
        .map(|fields| fields.to_vec())
        .ok_or_else(|| {
            input_error_noloc!(Aarch64DarwinErr::UnsupportedType(
                "opaque struct stack slot".to_string()
            ))
        })
}

fn align_to_16(bytes: u64) -> u64 {
    (bytes + 15) & !15
}

pub(super) fn fold_binary(
    ctx: &Context,
    kind: BinaryKind,
    lhs: u128,
    rhs: u128,
    result_ty: TypeHandle,
) -> Option<u128> {
    let width = integer_width_and_signedness(ctx, result_ty)
        .map(|(width, _)| width.min(128))
        .unwrap_or(64);
    if width == 128 {
        let result = match kind {
            BinaryKind::Add => lhs.wrapping_add(rhs),
            BinaryKind::Sub => lhs.wrapping_sub(rhs),
            BinaryKind::Mul => lhs.wrapping_mul(rhs),
            BinaryKind::SDiv => {
                (rhs != 0).then_some((lhs as i128).wrapping_div(rhs as i128) as u128)?
            }
            BinaryKind::UDiv => (rhs != 0).then_some(lhs.wrapping_div(rhs))?,
            BinaryKind::SRem => {
                (rhs != 0).then_some((lhs as i128).wrapping_rem(rhs as i128) as u128)?
            }
            BinaryKind::URem => (rhs != 0).then_some(lhs.wrapping_rem(rhs))?,
            BinaryKind::And => lhs & rhs,
            BinaryKind::Or => lhs | rhs,
            BinaryKind::Xor => lhs ^ rhs,
            BinaryKind::Shl => lhs.wrapping_shl(rhs as u32),
            BinaryKind::Shr => lhs.wrapping_shr(rhs as u32),
        };
        return Some(result);
    }
    let mask = if width == 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    let lhs = (lhs as u64) & mask;
    let rhs = (rhs as u64) & mask;
    let result = match kind {
        BinaryKind::Add => lhs.wrapping_add(rhs),
        BinaryKind::Sub => lhs.wrapping_sub(rhs),
        BinaryKind::Mul => lhs.wrapping_mul(rhs),
        BinaryKind::SDiv => {
            let lhs = sign_extend_immediate(lhs, width) as i64;
            let rhs = sign_extend_immediate(rhs, width) as i64;
            if rhs == 0 {
                return None;
            }
            lhs.wrapping_div(rhs) as u64
        }
        BinaryKind::UDiv => {
            if rhs == 0 {
                return None;
            }
            lhs.wrapping_div(rhs)
        }
        BinaryKind::SRem => {
            let lhs = sign_extend_immediate(lhs, width) as i64;
            let rhs = sign_extend_immediate(rhs, width) as i64;
            if rhs == 0 {
                return None;
            }
            lhs.wrapping_rem(rhs) as u64
        }
        BinaryKind::URem => {
            if rhs == 0 {
                return None;
            }
            lhs.wrapping_rem(rhs)
        }
        BinaryKind::And => lhs & rhs,
        BinaryKind::Or => lhs | rhs,
        BinaryKind::Xor => lhs ^ rhs,
        BinaryKind::Shl => lhs.wrapping_shl(rhs as u32),
        BinaryKind::Shr => lhs.wrapping_shr(rhs as u32),
    };
    Some((result & mask) as u128)
}

fn sign_extend_immediate(value: u64, width: u32) -> u64 {
    if width >= 64 {
        return value;
    }
    let sign_bit = 1u64 << (width - 1);
    let mask = (1u64 << width) - 1;
    let value = value & mask;
    if value & sign_bit == 0 {
        value
    } else {
        value | !mask
    }
}

pub(super) fn opcode(kind: BinaryKind) -> Aarch64Opcode {
    match kind {
        BinaryKind::Add => aarch64_ops::AddOp::OPCODE,
        BinaryKind::Sub => aarch64_ops::SubOp::OPCODE,
        BinaryKind::Mul => aarch64_ops::MulOp::OPCODE,
        BinaryKind::SDiv => aarch64_ops::SdivOp::OPCODE,
        BinaryKind::UDiv => aarch64_ops::UdivOp::OPCODE,
        BinaryKind::SRem | BinaryKind::URem => {
            unreachable!("remainders lower through division and subtraction")
        }
        BinaryKind::And => aarch64_ops::AndOp::OPCODE,
        BinaryKind::Or => aarch64_ops::OrOp::OPCODE,
        BinaryKind::Xor => aarch64_ops::XorOp::OPCODE,
        BinaryKind::Shl => aarch64_ops::ShlOp::OPCODE,
        BinaryKind::Shr => aarch64_ops::LsrOp::OPCODE,
    }
}

/// The AArch64 condition code that tests `predicate` after a `cmp`.
pub(super) fn condition_code(predicate: ICmpPredicateAttr) -> ConditionCode {
    match predicate {
        ICmpPredicateAttr::EQ => ConditionCode::Eq,
        ICmpPredicateAttr::NE => ConditionCode::Ne,
        ICmpPredicateAttr::ULT => ConditionCode::Lo,
        ICmpPredicateAttr::UGE => ConditionCode::Hs,
        ICmpPredicateAttr::ULE => ConditionCode::Ls,
        ICmpPredicateAttr::UGT => ConditionCode::Hi,
        ICmpPredicateAttr::SLT => ConditionCode::Lt,
        ICmpPredicateAttr::SGE => ConditionCode::Ge,
        ICmpPredicateAttr::SLE => ConditionCode::Le,
        ICmpPredicateAttr::SGT => ConditionCode::Gt,
    }
}
