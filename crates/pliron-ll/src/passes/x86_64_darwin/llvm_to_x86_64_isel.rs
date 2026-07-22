//! Direct LLVM-to-x86-64 instruction selection.
//!
//! This remains a single pass because it owns function-wide lowering state:
//! SSA values, virtual-register allocation, stack-slot allocation, literal
//! labels, and machine CFG edge blocks. Independent domains live in sibling
//! modules; this file coordinates function and instruction lowering.

use std::collections::HashMap;

use pliron::builtin::op_interfaces::{AtMostOneRegionInterface as _, BranchOpInterface as _, CallOpInterface as _, OneOpdInterface as _};
use pliron_llvm::op_interfaces::{PointerTypeResult as _};

use crate::ll::ops::CStrOp;

use crate::{
    common_traits::Named,
    context::{Context, Ptr},
    dialects::{
        x86_64::{
            attributes::{AbiLocation, ConditionCode, FunctionAbi, FunctionAbiAttr},
            op_interfaces::X86_64Opcode,
            ops::{self as x86_64_ops, FuncOp as X86_64FuncOp},
            registers::{R11, RAX, RDI, RDX, Register},
        },
        builtin::{
            attributes::IntegerAttr,
            op_interfaces::{
                CallOpCallable, OneRegionInterface, OneResultInterface, SymbolOpInterface,
            },
        },
        llvm::{
            attributes::ICmpPredicateAttr,
            op_interfaces::IsDeclaration,
            ops::{
                AddressOfOp, AllocaOp, BitcastOp, BrOp, CallOp, CondBrOp, ConstantOp,
                ExtractValueOp, FuncOp as LlvmFuncOp, GetElementPtrOp, GlobalOp as LlvmGlobalOp,
                ICmpOp, InsertValueOp, IntToPtrOp, LoadOp, PoisonOp, PtrToIntOp, ReturnOp, StoreOp,
                TruncOp,
                UndefOp, UnreachableOp, ZExtOp,
            },
        },
    },
    input_error_noloc,
    ir::{basic_block::BasicBlock, op::Op, operation::Operation, r#type::Typed, value::Value},
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
use crate::r#type::TypeHandle;

use super::{
    attrs::ATTR_KEY_DARWIN_ABI,
    error::X86_64DarwinErr,
    frontend::{ARG_GPRS, BinaryKind, binary_kind, collect_entry_arguments, module_op, validate_linkage},
    util::module_body,
};

/// Instruction selection: rewrites the module in place, lowering every
/// defined `llvm.func` to an `x86_64.func` and erasing the `llvm` ops.
pub struct LlvmToX86_64IselPass;

impl Pass for LlvmToX86_64IselPass {
    fn name(&self) -> &str {
        "llvm-to-x86-64-isel"
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
                if let Some(bytes) = crate::ll::global_initializer_bytes(ctx, global) {
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
    func: X86_64FuncOp,
    entry: Ptr<BasicBlock>,
    region: Ptr<crate::ir::region::Region>,
    blocks: Vec<Ptr<BasicBlock>>,
    block_map: HashMap<Ptr<BasicBlock>, Ptr<BasicBlock>>,
    abi: FunctionAbi,
}

impl MachineFunctionPlan {
    fn create(
        ctx: &mut Context,
        llvm_func: &LlvmFuncOp,
        module_body: Ptr<BasicBlock>,
    ) -> STAIRResult<Self> {
        let name = llvm_func.get_symbol_name(ctx);
        let abi = function_abi(ctx, llvm_func)?;
        let linkage = validate_linkage(&name.to_string(), llvm_func.get_attr_llvm_function_linkage(ctx).expect("llvm function without linkage").clone())?;
        let func = X86_64FuncOp::new(ctx, name, linkage);
        func.get_operation().insert_at_back(module_body, ctx);
        let entry = func.entry_block(ctx);

        let blocks: Vec<_> = llvm_func.get_region(ctx).expect("llvm.func definition must have a body").deref(ctx).iter(ctx).collect();
        let region = func.get_region(ctx);
        let mut block_map = HashMap::new();
        for (index, llvm_block) in blocks.iter().copied().enumerate() {
            let x86_64_block = if index == 0 {
                entry
            } else {
                let label = llvm_block.deref(ctx).unique_name(ctx).to_string();
                let block = BasicBlock::new(ctx, Some(label.try_into().unwrap()), vec![]);
                block.insert_at_back(region, ctx);
                block
            };
            block_map.insert(llvm_block, x86_64_block);
        }

        Ok(Self {
            func,
            entry,
            region,
            blocks,
            block_map,
            abi,
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
            input_error_noloc!(X86_64DarwinErr::UnsupportedOp(format!(
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
    let branch_probabilities = HotPathInfo::for_region(llvm_func.get_region(ctx).expect("llvm.func definition must have a body"), ctx);
    let plan = MachineFunctionPlan::create(ctx, llvm_func, module_body)?;
    let MachineFunctionPlan {
        func,
        entry,
        region,
        blocks,
        block_map,
        abi,
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
                    x86_64_ops::ldr_stack_arg(ctx, lo.clone(), offset).insert_at_back(entry, ctx);
                    let hi = fresh_vreg(&mut next_vreg);
                    x86_64_ops::ldr_stack_arg(ctx, hi.clone(), offset + 8)
                        .insert_at_back(entry, ctx);
                    values.insert(arg, LoweredValue::RegPair(lo, hi));
                } else {
                    let dst = fresh_vreg(&mut next_vreg);
                    x86_64_ops::ldr_stack_arg(ctx, dst.clone(), offset).insert_at_back(entry, ctx);
                    values.insert(arg, LoweredValue::Reg(dst));
                }
            }
            // Copy incoming ABI registers into virtual registers: the raw
            // rdi..r9 are clobbered by the first call (or argument setup),
            // while a promoted argument value may live for the whole
            // function. The copies are queued so the entry block keeps its
            // stack-arg-loads prefix, which frame lowering inserts the
            // prologue after.
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
            AbiLocation::IndirectResult => {
                return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedOp(
                    "indirect-result ABI location on a function argument".to_string()
                )));
            }
        }
    }

    // Unlike AArch64, x86-64 has no link register to save: `call` pushes the
    // return address, and frame lowering keeps the stack 16-byte aligned.
    for (dst, src) in arg_copies {
        x86_64_ops::mov(ctx, dst, src).insert_at_back(entry, ctx);
    }

    let llvm_entry = llvm_func
        .get_entry_block(ctx)
        .expect("llvm.func definition must have a body");
    for block in &blocks {
        if *block == llvm_entry {
            continue;
        }
        for arg in block.deref(ctx).arguments() {
            let lowered = block_arg_value(ctx, arg.get_type(ctx), &mut next_vreg)?;
            values.insert(arg, lowered);
        }
    }

    let mut next_literal = 0usize;
    let mut next_edge_block = 0usize;
    let mut stack = StackAllocator::default();
    let sret_result_slot = if abi.result == AbiLocation::IndirectResult {
        let sret_ptr_ty = word_ty(ctx);
        let slot = stack.allocate(ctx, sret_ptr_ty)?;
        x86_64_ops::str_sp_offset(ctx, RDI, slot.offset).insert_at_back(entry, ctx);
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
            input_error_noloc!(X86_64DarwinErr::UnsupportedOp(
                "missing lowered x86-64 block".to_string()
            ))
        })?;
        let mut op = block.deref(ctx).get_head();
        while let Some(op_ptr) = op {
            op = op_ptr.deref(ctx).get_next();
            let opid = Operation::get_opid(op_ptr, ctx);
            let op_obj = Operation::get_op_dyn(op_ptr, ctx);
            if let Some(alloca) = op_obj.downcast_ref::<AllocaOp>() {
                let slot = stack.allocate(ctx, alloca.result_pointee_type(ctx))?;
                values.insert(alloca.get_result(ctx), LoweredValue::StackAddr(slot));
            } else if let Some(constant) = op_obj.downcast_ref::<ConstantOp>() {
                let attr = constant.get_value(ctx);
                let attr = attr
                    .downcast_ref::<IntegerAttr>()
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
                let symbol = addr.get_global_name(ctx);
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
                    x86_64_ops::adr_function(ctx, dst.clone(), symbol)
                        .insert_at_back(insert_block, ctx);
                    values.insert(addr.get_result(ctx), LoweredValue::Reg(dst));
                }
            } else if let Some(undef) = op_obj.downcast_ref::<UndefOp>() {
                values.insert(undef.get_result(ctx), LoweredValue::Undef);
            } else if let Some(poison) = op_obj.downcast_ref::<PoisonOp>() {
                values.insert(poison.get_result(ctx), LoweredValue::Undef);
            } else if let Some(insert) = op_obj.downcast_ref::<InsertValueOp>() {
                let aggregate = lookup_value(ctx, &values, insert.get_operation().deref(ctx).get_operand(0))?;
                let value = lookup_value(ctx, &values, insert.get_operation().deref(ctx).get_operand(1))?;
                values.insert(
                    insert.get_result(ctx),
                    insert_aggregate_value(aggregate, &insert.indices(ctx), value)?,
                );
            } else if let Some(extract) = op_obj.downcast_ref::<ExtractValueOp>() {
                let aggregate = lookup_value(ctx, &values, extract.get_operation().deref(ctx).get_operand(0))?;
                values.insert(
                    extract.get_result(ctx),
                    extract_aggregate_value(aggregate, &extract.indices(ctx))?,
                );
            } else if let Some(cast) = op_obj.downcast_ref::<IntToPtrOp>() {
                let value = match lookup_value(ctx, &values, cast.get_operand(ctx))? {
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
                let value = lookup_value(ctx, &values, cast.get_operand(ctx))?;
                values.insert(cast.get_result(ctx), value);
            } else if let Some(cast) = op_obj.downcast_ref::<BitcastOp>() {
                let value = lookup_value(ctx, &values, cast.get_operand(ctx))?;
                let result = cast.get_result(ctx);
                let value = adapt_value_to_type(ctx, value, result.get_type(ctx))?;
                values.insert(result, value);
            } else if let Some(cast) = op_obj.downcast_ref::<ZExtOp>() {
                let value = lookup_value(ctx, &values, cast.get_operand(ctx))?;
                values.insert(cast.get_result(ctx), value);
            } else if let Some(cast) = op_obj.downcast_ref::<TruncOp>() {
                let result = cast.get_result(ctx);
                let value = lookup_value(ctx, &values, cast.get_operand(ctx))?;
                if let Some(mask) = integer_trunc_mask(ctx, result.get_type(ctx)) {
                    if let LoweredValue::Imm(imm) = value {
                        values.insert(result, LoweredValue::Imm(imm & mask as u128));
                    } else {
                        let src =
                            materialize(ctx, insert_block, value, &mut next_vreg, "trunc input")?;
                        let mask_reg = fresh_vreg(&mut next_vreg);
                        materialize_u64_immediate(ctx, insert_block, mask_reg, mask);
                        let dst = fresh_vreg(&mut next_vreg);
                        x86_64_ops::binary(
                            ctx,
                            x86_64_ops::AndOp::OPCODE,
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
                let address = lower_gep(
                    ctx,
                    insert_block,
                    &values,
                    gep.get_operand_src_ptr(ctx),
                    &gep.indices(ctx),
                    gep.src_elem_type(ctx),
                    &mut next_vreg,
                )?;
                values.insert(gep.get_result(ctx), address);
            } else if let Some(load) = op_obj.downcast_ref::<LoadOp>() {
                let result = load.get_result(ctx);
                let addr = lookup_value(ctx, &values, load.get_operand_address(ctx))?;
                let value = load_memory(
                    ctx,
                    insert_block,
                    addr,
                    result.get_type(ctx),
                    &mut next_vreg,
                )?;
                values.insert(result, value);
            } else if let Some(store) = op_obj.downcast_ref::<StoreOp>() {
                let value = store.get_operand_value(ctx);
                let lowered_value = lookup_value(ctx, &values, value)?;
                let addr = lookup_value(ctx, &values, store.get_operand_address(ctx))?;
                store_memory(
                    ctx,
                    insert_block,
                    addr,
                    lowered_value,
                    value.get_type(ctx),
                    &mut next_vreg,
                )?;
            } else if let Some(call) = op_obj.downcast_ref::<CallOp>() {
                let callee = call.callee(ctx);
                let args = call.args(ctx);
                let callee_ptr = if let CallOpCallable::Indirect(callee_value) = &callee {
                    let lowered = lookup_value(ctx, &values, *callee_value)?;
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
                    (Some(result), Some(ResultLocation::IndirectRdi)) => {
                        Some(stack.allocate(ctx, result.get_type(ctx))?)
                    }
                    _ => None,
                };
                // An indirect result consumes rdi as the hidden first
                // argument; the visible arguments shift down by one register.
                let arg_reg_base = usize::from(indirect_result_slot.is_some());
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
                let stack_arg_count = (arg_regs.len() + arg_reg_base).saturating_sub(ARG_GPRS.len());
                let outgoing_stack_size = align_to_16((stack_arg_count as u64) * 8);
                if outgoing_stack_size > 0 {
                    x86_64_ops::sub_sp_imm(ctx, outgoing_stack_size)
                        .insert_at_back(insert_block, ctx);
                }
                if let Some(callee_ptr) = &callee_ptr {
                    // r11 is a scratch register outside the allocatable set,
                    // so it survives the argument moves.
                    x86_64_ops::mov(ctx, R11, *callee_ptr).insert_at_back(insert_block, ctx);
                }
                for (idx, src) in arg_regs.into_iter().enumerate() {
                    let slot = idx + arg_reg_base;
                    if slot < ARG_GPRS.len() {
                        x86_64_ops::mov(ctx, ARG_GPRS[slot], src)
                            .insert_at_back(insert_block, ctx);
                    } else {
                        x86_64_ops::str_sp_offset(ctx, src, ((slot - ARG_GPRS.len()) as u64) * 8)
                            .insert_at_back(insert_block, ctx);
                    }
                }
                if let Some(slot) = indirect_result_slot {
                    // sp is already lowered by the outgoing stack-argument
                    // area here, so compensate to reach the frame slot.
                    x86_64_ops::add_sp_offset(ctx, RDI, slot.offset + outgoing_stack_size)
                        .insert_at_back(insert_block, ctx);
                }
                match callee {
                    CallOpCallable::Direct(callee) => {
                        x86_64_ops::call(ctx, callee).insert_at_back(insert_block, ctx);
                    }
                    CallOpCallable::Indirect(_) => {
                        x86_64_ops::call_reg(ctx, R11).insert_at_back(insert_block, ctx);
                    }
                }
                if outgoing_stack_size > 0 {
                    x86_64_ops::add_sp_imm(ctx, outgoing_stack_size)
                        .insert_at_back(insert_block, ctx);
                }
                if let Some(result) = result {
                    let lowered = match result_location.unwrap() {
                        ResultLocation::ScalarRax => {
                            let dst = fresh_vreg(&mut next_vreg);
                            x86_64_ops::mov(ctx, dst, RAX).insert_at_back(insert_block, ctx);
                            LoweredValue::Reg(dst)
                        }
                        ResultLocation::ScalarRaxRdx => {
                            let lo = fresh_vreg(&mut next_vreg);
                            x86_64_ops::mov(ctx, lo, RAX).insert_at_back(insert_block, ctx);
                            let hi = fresh_vreg(&mut next_vreg);
                            x86_64_ops::mov(ctx, hi, RDX).insert_at_back(insert_block, ctx);
                            LoweredValue::RegPair(lo, hi)
                        }
                        ResultLocation::DirectGprs(count) => load_gpr_aggregate_result(
                            ctx,
                            insert_block,
                            result.get_type(ctx),
                            count,
                            &mut next_vreg,
                        )?,
                        ResultLocation::IndirectRdi => {
                            let slot = indirect_result_slot.ok_or_else(|| {
                                input_error_noloc!(X86_64DarwinErr::UnsupportedOp(
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
                        x86_64_ops::UdivOp::OPCODE
                    } else {
                        x86_64_ops::SdivOp::OPCODE
                    };
                    x86_64_ops::binary(
                        ctx,
                        div_opcode,
                        quotient.clone(),
                        lhs.clone(),
                        rhs.clone(),
                    )
                    .insert_at_back(insert_block, ctx);
                    let product = fresh_vreg(&mut next_vreg);
                    x86_64_ops::binary(
                        ctx,
                        x86_64_ops::MulOp::OPCODE,
                        product.clone(),
                        quotient,
                        rhs,
                    )
                    .insert_at_back(insert_block, ctx);
                    x86_64_ops::binary(ctx, x86_64_ops::SubOp::OPCODE, dst.clone(), lhs, product)
                        .insert_at_back(insert_block, ctx);
                } else {
                    x86_64_ops::binary(ctx, opcode(kind), dst.clone(), lhs, rhs)
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
                let lhs = icmp.get_operation().deref(ctx).get_operand(0);
                let rhs = icmp.get_operation().deref(ctx).get_operand(1);
                values.insert(
                    icmp.get_result(ctx),
                    LoweredValue::Compare(CompareValue {
                        predicate: icmp.predicate(ctx),
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
                x86_64_ops::ret(ctx).insert_at_back(insert_block, ctx);
            } else if let Some(_unreachable) = op_obj.downcast_ref::<UnreachableOp>() {
                x86_64_ops::trap(ctx).insert_at_back(insert_block, ctx);
            } else if let Some(br) = op_obj.downcast_ref::<BrOp>() {
                let dest = br.get_operation().deref(ctx).get_successor(0);
                let args = br.successor_operands(ctx, 0);
                emit_block_arg_copies(ctx, insert_block, &values, dest, &args, &mut next_vreg)?;
                let target = machine_block(&block_map, dest)?;
                x86_64_ops::jmp(ctx, target).insert_at_back(insert_block, ctx);
            } else if let Some(cond_br) = op_obj.downcast_ref::<CondBrOp>() {
                let true_dest = cond_br.get_operation().deref(ctx).get_successor(0);
                let true_args = cond_br.successor_operands(ctx, 0);
                let false_dest = cond_br.get_operation().deref(ctx).get_successor(1);
                let false_args = cond_br.successor_operands(ctx, 1);
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
                let condition_value = lookup_value(ctx, &values, cond_br.get_operand_condition(ctx))?;
                if let LoweredValue::Compare(compare) = condition_value {
                    if is_128_bit_integer(ctx, compare.lhs_ty) {
                        let condition =
                            lower_compare_value(ctx, insert_block, compare, &mut next_vreg)?;
                        let branch = x86_64_ops::test_jnz(ctx, condition, true_target);
                        x86_64_ops::set_branch_weights(
                            ctx,
                            branch,
                            taken_weight,
                            not_taken_weight,
                        );
                        branch.insert_at_back(insert_block, ctx);
                        x86_64_ops::jmp(ctx, false_target).insert_at_back(insert_block, ctx);
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
                    x86_64_ops::cmp(ctx, lhs, rhs).insert_at_back(insert_block, ctx);
                    let branch = x86_64_ops::jcc(
                        ctx,
                        condition_code(compare.predicate),
                        true_target,
                    );
                    x86_64_ops::set_branch_weights(ctx, branch, taken_weight, not_taken_weight);
                    branch.insert_at_back(insert_block, ctx);
                } else {
                    let condition = materialize(
                        ctx,
                        insert_block,
                        condition_value,
                        &mut next_vreg,
                        "branch condition",
                    )?;
                    let branch = x86_64_ops::test_jnz(ctx, condition, true_target);
                    x86_64_ops::set_branch_weights(ctx, branch, taken_weight, not_taken_weight);
                    branch.insert_at_back(insert_block, ctx);
                }
                x86_64_ops::jmp(ctx, false_target).insert_at_back(insert_block, ctx);
            } else {
                return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedOp(
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
        input_error_noloc!(X86_64DarwinErr::UndefinedValue(
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
            return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedOp(
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
            return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedOp(
                format!("llvm.extractvalue from non-aggregate {other:?}")
            )));
        }
    };
    // A field that was never inserted is an uninitialized read; through
    // memory this would have been an undef load, so it is undef here too.
    let field = fields
        .get(*index as usize)
        .and_then(|field| field.clone())
        .unwrap_or(LoweredValue::Undef);
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
    // Reconcile packed-scalar and field-wise aggregate representations of
    // the value with the type the use site expects.
    let value = adapt_value_to_type(ctx, value, ty)?;
    if let LoweredValue::Aggregate(fields) = &value {
        if fields.len() > 1 {
            return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedOp(format!(
                "cannot materialize {context}: multi-field aggregate of type {}",
                pliron::printable::Printable::disp(&ty, ctx)
            ))));
        }
    }
    if is_128_bit_integer(ctx, ty) {
        let (lo, _) = materialize_pair(ctx, entry, value, ty, next_vreg, context)?;
        return Ok(lo);
    }
    let context = format!(
        "{context} (type {})",
        pliron::printable::Printable::disp(&ty, ctx)
    );
    let reg = materialize(ctx, entry, value, next_vreg, &context)?;
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
                    input_error_noloc!(X86_64DarwinErr::UndefinedValue(
                        "missing low 128-bit lane".to_string()
                    ))
                })?;
            let hi = fields
                .get(1)
                .and_then(|field| field.clone())
                .ok_or_else(|| {
                    input_error_noloc!(X86_64DarwinErr::UndefinedValue(
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
                x86_64_ops::binary(
                    ctx,
                    x86_64_ops::LsrOp::OPCODE,
                    hi.clone(),
                    lo.clone(),
                    sign,
                )
                .insert_at_back(entry, ctx);
                let mask = fresh_vreg(next_vreg);
                materialize_u64_immediate(ctx, entry, mask, 0u64.wrapping_sub(1));
                x86_64_ops::binary(
                    ctx,
                    x86_64_ops::MulOp::OPCODE,
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
            x86_64_ops::adr_literal(ctx, dst.clone(), label, bytes).insert_at_back(entry, ctx);
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
                x86_64_ops::binary(
                    ctx,
                    x86_64_ops::AddOp::OPCODE,
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
            x86_64_ops::add_sp_offset(ctx, dst.clone(), slot.offset).insert_at_back(entry, ctx);
            Ok(dst)
        }
        LoweredValue::Compare(compare) => lower_compare_value(ctx, entry, compare, next_vreg),
        LoweredValue::Aggregate(mut fields) if fields.len() == 1 => {
            let field = fields.pop().flatten().ok_or_else(|| {
                input_error_noloc!(X86_64DarwinErr::UndefinedValue(
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
        LoweredValue::Aggregate(_) => Err(input_error_noloc!(X86_64DarwinErr::UnsupportedOp(
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
                    input_error_noloc!(X86_64DarwinErr::UndefinedValue(format!(
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
    // movabs carries a full 64-bit immediate in one instruction.
    x86_64_ops::mov_imm(ctx, dst, imm).insert_at_back(entry, ctx);
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
    x86_64_ops::binary(
        ctx,
        x86_64_ops::AndOp::OPCODE,
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
    x86_64_ops::binary(
        ctx,
        x86_64_ops::XorOp::OPCODE,
        flipped.clone(),
        masked,
        sign_bit_reg.clone(),
    )
    .insert_at_back(entry, ctx);
    let extended = fresh_vreg(next_vreg);
    x86_64_ops::binary(
        ctx,
        x86_64_ops::SubOp::OPCODE,
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
) -> STAIRResult<X86_64Opcode> {
    Ok(match scalar_size_of(ctx, ty)? {
        1 => x86_64_ops::LdrbSpOffsetOp::OPCODE,
        2 => x86_64_ops::LdrhSpOffsetOp::OPCODE,
        3 | 4 => x86_64_ops::LdrwSpOffsetOp::OPCODE,
        5..=8 => x86_64_ops::LdrSpOffsetOp::OPCODE,
        size => {
            return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedType(
                format!("scalar load size {size}")
            )));
        }
    })
}

pub(super) fn store_sp_opcode(
    ctx: &Context,
    ty: TypeHandle,
) -> STAIRResult<X86_64Opcode> {
    Ok(match scalar_size_of(ctx, ty)? {
        1 => x86_64_ops::StrbSpOffsetOp::OPCODE,
        2 => x86_64_ops::StrhSpOffsetOp::OPCODE,
        3 | 4 => x86_64_ops::StrwSpOffsetOp::OPCODE,
        5..=8 => x86_64_ops::StrSpOffsetOp::OPCODE,
        size => {
            return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedType(
                format!("scalar store size {size}")
            )));
        }
    })
}

pub(super) fn load_reg_opcode(
    ctx: &Context,
    ty: TypeHandle,
) -> STAIRResult<X86_64Opcode> {
    Ok(match scalar_size_of(ctx, ty)? {
        1 => x86_64_ops::LdrbRegOffsetOp::OPCODE,
        2 => x86_64_ops::LdrhRegOffsetOp::OPCODE,
        3 | 4 => x86_64_ops::LdrwRegOffsetOp::OPCODE,
        5..=8 => x86_64_ops::LdrRegOffsetOp::OPCODE,
        size => {
            return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedType(
                format!("scalar load size {size}")
            )));
        }
    })
}

pub(super) fn store_reg_opcode(
    ctx: &Context,
    ty: TypeHandle,
) -> STAIRResult<X86_64Opcode> {
    Ok(match scalar_size_of(ctx, ty)? {
        1 => x86_64_ops::StrbRegOffsetOp::OPCODE,
        2 => x86_64_ops::StrhRegOffsetOp::OPCODE,
        3 | 4 => x86_64_ops::StrwRegOffsetOp::OPCODE,
        5..=8 => x86_64_ops::StrRegOffsetOp::OPCODE,
        size => {
            return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedType(
                format!("scalar store size {size}")
            )));
        }
    })
}


/// The [LoweredValue] shape of a block argument of type `ty`: one fresh
/// virtual register per scalar leaf — a plain register for single-register
/// scalars, a register pair for 128-bit integers, and a recursive
/// [LoweredValue::Aggregate] for structs and arrays.
pub(super) fn block_arg_value(
    ctx: &Context,
    ty: TypeHandle,
    next_vreg: &mut usize,
) -> STAIRResult<LoweredValue> {
    if is_zero_sized_ty(ctx, ty) {
        return Ok(LoweredValue::Undef);
    }
    if is_128_bit_integer(ctx, ty) {
        return Ok(LoweredValue::RegPair(
            fresh_vreg(next_vreg),
            fresh_vreg(next_vreg),
        ));
    }
    if is_aggregate_ty(ctx, ty) {
        let fields = struct_fields(ctx, ty)?;
        let mut lowered = Vec::with_capacity(fields.len());
        for field in fields {
            lowered.push(Some(block_arg_value(ctx, field, next_vreg)?));
        }
        return Ok(LoweredValue::Aggregate(lowered));
    }
    Ok(LoweredValue::Reg(fresh_vreg(next_vreg)))
}

/// The registers backing a block argument's [LoweredValue], in leaf order.
pub(super) fn block_arg_registers(value: &LoweredValue, out: &mut Vec<Register>) {
    match value {
        LoweredValue::Reg(reg) => out.push(reg.clone()),
        LoweredValue::RegPair(lo, hi) => {
            out.push(lo.clone());
            out.push(hi.clone());
        }
        LoweredValue::Aggregate(fields) => {
            for field in fields.iter().flatten() {
                block_arg_registers(field, out);
            }
        }
        LoweredValue::Undef => {}
        other => unreachable!("block arguments lower to registers, got {other:?}"),
    }
}

pub(super) fn is_zero_sized_ty(ctx: &Context, ty: TypeHandle) -> bool {
    let ty_ref = ty.deref(ctx);
    ty_ref
        .downcast_ref::<crate::dialects::builtin::types::UnitType>()
        .is_some()
        || ty_ref
            .downcast_ref::<crate::dialects::llvm::types::VoidType>()
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
        input_error_noloc!(X86_64DarwinErr::UnsupportedOp(
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
        return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedType(
            format!("{:?}", &*ty_ref)
        )));
    };
    if struct_ty.is_opaque() {
        return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedType(
            "opaque struct stack slot".to_string()
        )));
    }
    Ok(struct_ty.fields().collect())
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

pub(super) fn opcode(kind: BinaryKind) -> X86_64Opcode {
    match kind {
        BinaryKind::Add => x86_64_ops::AddOp::OPCODE,
        BinaryKind::Sub => x86_64_ops::SubOp::OPCODE,
        BinaryKind::Mul => x86_64_ops::MulOp::OPCODE,
        BinaryKind::SDiv => x86_64_ops::SdivOp::OPCODE,
        BinaryKind::UDiv => x86_64_ops::UdivOp::OPCODE,
        BinaryKind::SRem | BinaryKind::URem => {
            unreachable!("remainders lower through division and subtraction")
        }
        BinaryKind::And => x86_64_ops::AndOp::OPCODE,
        BinaryKind::Or => x86_64_ops::OrOp::OPCODE,
        BinaryKind::Xor => x86_64_ops::XorOp::OPCODE,
        BinaryKind::Shl => x86_64_ops::ShlOp::OPCODE,
        BinaryKind::Shr => x86_64_ops::LsrOp::OPCODE,
    }
}

/// The x86 condition code (the `cc` in `jcc`/`setcc`) that tests `predicate`
/// after a `cmp`.
pub(super) fn condition_code(predicate: ICmpPredicateAttr) -> ConditionCode {
    match predicate {
        ICmpPredicateAttr::EQ => ConditionCode::E,
        ICmpPredicateAttr::NE => ConditionCode::Ne,
        ICmpPredicateAttr::ULT => ConditionCode::B,
        ICmpPredicateAttr::UGE => ConditionCode::Ae,
        ICmpPredicateAttr::ULE => ConditionCode::Be,
        ICmpPredicateAttr::UGT => ConditionCode::A,
        ICmpPredicateAttr::SLT => ConditionCode::L,
        ICmpPredicateAttr::SGE => ConditionCode::Ge,
        ICmpPredicateAttr::SLE => ConditionCode::Le,
        ICmpPredicateAttr::SGT => ConditionCode::G,
    }
}
