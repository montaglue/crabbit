//! Direct LLVM-to-AArch64 instruction selection.
//!
//! This remains a single pass because it owns function-wide lowering state:
//! SSA values, virtual-register allocation, stack-slot allocation, literal
//! labels, and machine CFG edge blocks. Independent domains live in sibling
//! modules; this file coordinates function and instruction lowering.

use crate::dialects::builtin::ops::ConstantOp;
use std::collections::HashMap;

use pliron::builtin::op_interfaces::{AtMostOneRegionInterface as _, BranchOpInterface as _, CallOpInterface as _, OneOpdInterface as _};
use pliron_llvm::op_interfaces::{PointerTypeResult as _};

use crate::ll::ops::CStrOp;

use crate::{
    common_traits::Named,
    context::{Context, Ptr},
    dialects::{
        aarch64::{
            attributes::{AbiLocation, ConditionCode, FunctionAbi, FunctionAbiAttr},
            encoding::{fmov_imm8_for_f32_bits, fmov_imm8_for_f64_bits},
            op_interfaces::Aarch64Opcode,
            ops::{self as aarch64_ops, FuncOp as Aarch64FuncOp},
            registers::{LR, Register, RegisterClass, X8, X16},
        },
        builtin::{
            attributes::IntegerAttr,
            op_interfaces::{
                CallOpCallable, OneRegionInterface, OneResultInterface, SymbolOpInterface,
            },
        },
        llvm::{
            attributes::{FCmpPredicateAttr, ICmpPredicateAttr},
            op_interfaces::IsDeclaration,
            ops::{
                AddressOfOp, AllocaOp, BitcastOp, BrOp, CallOp, CondBrOp, ExtractValueOp, FAddOp,
                FCmpOp, FDivOp, FMulOp,
                FNegOp, FPExtOp, FPToSIOp, FPToUIOp, FPTruncOp, FRemOp, FSubOp,
                FuncOp as LlvmFuncOp,
                GetElementPtrOp, GlobalOp as LlvmGlobalOp,
                ICmpOp, InsertValueOp, IntToPtrOp, LoadOp, PoisonOp, PtrToIntOp, ReturnOp,
                SIToFPOp, StoreOp,
                TruncOp,
                SExtOp, UIToFPOp, UndefOp, UnreachableOp, ZExtOp,
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
    attrs::ATTR_KEY_AARCH64_ABI,
    error::Aarch64Err,
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
        let mut data_globals = std::collections::HashSet::<crate::identifier::Identifier>::new();
        let mut tls_globals = std::collections::HashSet::<crate::identifier::Identifier>::new();
        for op_ptr in llvm_ops.iter().copied() {
            let op_obj = Operation::get_op_dyn(op_ptr, ctx);
            if let Some(global) = op_obj.downcast_ref::<LlvmGlobalOp>() {
                if crate::ll::global_is_thread_local(ctx, global) {
                    // Thread-local (defined in `.tdata`/`.tbss` or an extern
                    // TLS declaration): addressed through the thread pointer,
                    // never by an ordinary data-section address.
                    tls_globals.insert(global.get_symbol_name(ctx));
                } else if let Some(bytes) = crate::ll::global_initializer_bytes(ctx, global) {
                    globals.insert(global.get_symbol_name(ctx), bytes);
                } else {
                    // A data-section definition (`ll.data`) or an extern
                    // declaration with no initializer: both are addressed
                    // via adrp+add, the latter resolving through an
                    // undefined symbol-table entry.
                    data_globals.insert(global.get_symbol_name(ctx));
                }
            }
        }

        for op_ptr in llvm_ops.iter().copied() {
            let op_obj = Operation::get_op_dyn(op_ptr, ctx);
            if let Some(llvm_func) = op_obj.downcast_ref::<LlvmFuncOp>() {
                if !llvm_func.is_declaration(ctx) {
                    lower_function(ctx, llvm_func, body, &globals, &data_globals, &tls_globals)?;
                }
            }
        }

        for op_ptr in llvm_ops {
            // Data-section and TLS globals stay in the module: the object
            // writers read them when laying out `.rodata`/`.data`/`.tdata`.
            let op_obj = Operation::get_op_dyn(op_ptr, ctx);
            if let Some(global) = op_obj.downcast_ref::<LlvmGlobalOp>()
                && (data_globals.contains(&global.get_symbol_name(ctx))
                    || tls_globals.contains(&global.get_symbol_name(ctx)))
            {
                continue;
            }
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
        let linkage = validate_linkage(&name.to_string(), llvm_func.get_attr_llvm_function_linkage(ctx).expect("llvm function without linkage").clone())?;
        let func = Aarch64FuncOp::new(ctx, name, linkage);
        func.get_operation().insert_at_back(module_body, ctx);
        let entry = func.entry_block(ctx);

        let blocks: Vec<_> = llvm_func.get_region(ctx).expect("llvm.func definition must have a body").deref(ctx).iter(ctx).collect();
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
        .get::<FunctionAbiAttr>(&ATTR_KEY_AARCH64_ABI)
        .map(|attr| attr.0.clone())
        .ok_or_else(|| {
            input_error_noloc!(Aarch64Err::UnsupportedOp(format!(
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
        // `llvm.frem` lowers to a call to fmod/fmodf, so it clobbers the
        // link register like any explicit call. Calls to the intrinsic
        // declarations that select to inline instructions do not.
        if let Some(call) = op_obj.downcast_ref::<CallOp>() {
            let inline_intrinsic = match call.callee(ctx) {
                CallOpCallable::Direct(name) => {
                    saturating_fp_to_int_intrinsic(name.as_ref()).is_some()
                        || fp_math_intrinsic(name.as_ref()).is_some()
                }
                CallOpCallable::Indirect(_) => false,
            };
            if !inline_intrinsic {
                return true;
            }
        } else if op_obj.downcast_ref::<FRemOp>().is_some() {
            return true;
        }
        op = op_ptr.deref(ctx).get_next();
    }
    false
}

/// The bytes of outgoing stack-argument area a call with `args` needs,
/// mirroring the register/stack split the call lowering performs: integers
/// and pointers draw from x0-x7, FP scalars from v0-v7, 128-bit integers
/// take two GPR slots, and overflow goes to 8-byte stack slots.
fn outgoing_stack_bytes_for_call(ctx: &Context, args: &[Value]) -> u64 {
    let mut next_gpr = 0u8;
    let mut next_fpr = 0u8;
    let mut stack = 0u64;
    let take_gpr = |next_gpr: &mut u8, stack: &mut u64| {
        if *next_gpr < 8 {
            *next_gpr += 1;
        } else {
            *stack += 8;
        }
    };
    for arg in args {
        let ty = arg.get_type(ctx);
        if fp_kind(ctx, ty).is_some() {
            if next_fpr < 8 {
                next_fpr += 1;
            } else {
                stack += 8;
            }
        } else if is_128_bit_integer(ctx, ty) {
            take_gpr(&mut next_gpr, &mut stack);
            take_gpr(&mut next_gpr, &mut stack);
        } else {
            take_gpr(&mut next_gpr, &mut stack);
        }
    }
    stack
}

/// The largest outgoing stack-argument area any call in `blocks` needs.
/// Like LLVM's fixed outgoing-argument area, it is reserved at the bottom
/// of the frame (offset 0 from sp) for the whole function, so no call site
/// has to move sp: a spill reload between argument setup and the `bl`
/// would otherwise read through a shifted sp.
fn max_outgoing_stack_bytes(ctx: &Context, blocks: &[Ptr<BasicBlock>]) -> u64 {
    let mut max = 0u64;
    for block in blocks {
        let mut op = block.deref(ctx).get_head();
        while let Some(op_ptr) = op {
            let op_obj = Operation::get_op_dyn(op_ptr, ctx);
            if let Some(call) = op_obj.downcast_ref::<CallOp>() {
                let args = call.args(ctx);
                max = max.max(outgoing_stack_bytes_for_call(ctx, &args));
            }
            op = op_ptr.deref(ctx).get_next();
        }
    }
    align_to_16(max)
}

fn lower_function(
    ctx: &mut Context,
    llvm_func: &LlvmFuncOp,
    module_body: Ptr<crate::ir::basic_block::BasicBlock>,
    globals: &HashMap<crate::identifier::Identifier, Vec<u8>>,
    data_globals: &std::collections::HashSet<crate::identifier::Identifier>,
    tls_globals: &std::collections::HashSet<crate::identifier::Identifier>,
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
        has_call,
    } = plan;

    // Backward-attribution stamping is live iff the op-id pass ran on this
    // function (docs/PROFILE-FEEDBACK-BACKWARD.md).
    let stamping = llvm_func
        .get_entry_block(ctx)
        .and_then(|entry_block| entry_block.deref(ctx).get_head())
        .is_some_and(|first| super::opmap::op_id(ctx, first).is_some());
    let mut stamper = super::opmap::IselStamper::new(ctx, region, stamping);

    let mut values = HashMap::<Value, LoweredValue>::new();
    let mut next_vreg = 0usize;
    let mut arg_copies: Vec<(Register, Register)> = Vec::new();
    // The link-register save leads the entry block; frame lowering puts the
    // frame allocation right after it and rebases the incoming stack
    // argument loads below past both.
    if has_call {
        aarch64_ops::str_pre_sp(ctx, LR, 16).insert_at_back(entry, ctx);
    }
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
            // function. The copies are queued after the stack-argument
            // loads so those stay a contiguous prefix.
            AbiLocation::GprPair(lo, hi) => {
                let lo_vreg = fresh_vreg(&mut next_vreg);
                let hi_vreg = fresh_vreg(&mut next_vreg);
                arg_copies.push((lo_vreg, lo));
                arg_copies.push((hi_vreg, hi));
                values.insert(arg, LoweredValue::RegPair(lo_vreg, hi_vreg));
            }
            AbiLocation::Gpr(reg) => {
                // The register class of the incoming ABI register decides
                // the promoted argument's file: FP arguments arrive in
                // v0-v7 and stay in FP virtual registers.
                let dst = match reg.class() {
                    RegisterClass::Fpr64 => fresh_fpr(&mut next_vreg, FpKind::F64),
                    RegisterClass::Fpr32 => fresh_fpr(&mut next_vreg, FpKind::F32),
                    _ => fresh_vreg(&mut next_vreg),
                };
                arg_copies.push((dst, reg));
                values.insert(arg, LoweredValue::Reg(dst));
            }
            AbiLocation::IndirectResult { .. } => {
                return Err(input_error_noloc!(Aarch64Err::UnsupportedOp(
                    "indirect-result ABI location on a function argument".to_string()
                )));
            }
        }
    }

    for (dst, src) in arg_copies {
        emit_move(ctx, entry, dst, src)?;
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
    // Frame slots start above the outgoing stack-argument area.
    let mut stack = StackAllocator::new(max_outgoing_stack_bytes(ctx, &blocks));
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
    stamper.preamble(ctx, entry);
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
            input_error_noloc!(Aarch64Err::UnsupportedOp(
                "missing lowered AArch64 block".to_string()
            ))
        })?;
        let mut op = block.deref(ctx).get_head();
        while let Some(op_ptr) = op {
            op = op_ptr.deref(ctx).get_next();
            // Close the previous source op's attribution bracket and open
            // this one's: everything emitted until the next `begin` derives
            // from this op.
            stamper.begin(ctx, insert_block, super::opmap::op_id(ctx, op_ptr));
            let opid = Operation::get_opid(op_ptr, ctx);
            let op_obj = Operation::get_op_dyn(op_ptr, ctx);
            if let Some(alloca) = op_obj.downcast_ref::<AllocaOp>() {
                let slot = stack.allocate(ctx, alloca.result_pointee_type(ctx))?;
                values.insert(alloca.get_result(ctx), LoweredValue::StackAddr(slot));
            } else if let Some(constant) = op_obj.downcast_ref::<ConstantOp>() {
                let attr = constant.get_value(ctx);
                // FP constants carry their IEEE bit pattern in `Imm`; the
                // result type at each use decides the register file.
                let imm = if let Some(fp32) = attr
                    .downcast_ref::<crate::dialects::builtin::attributes::FPSingleAttr>()
                {
                    pliron::utils::apfloat::Float::to_bits(fp32.0)
                } else if let Some(fp64) = attr
                    .downcast_ref::<crate::dialects::builtin::attributes::FPDoubleAttr>()
                {
                    pliron::utils::apfloat::Float::to_bits(fp64.0)
                } else {
                    let attr = attr
                        .downcast_ref::<IntegerAttr>()
                        .ok_or_else(|| {
                            input_error_noloc!(Aarch64Err::UnsupportedOp(format!(
                                "constant of unsupported attribute kind {attr:?}"
                            )))
                        })?;
                    attr.value().to_u128()
                };
                values.insert(constant.get_result(ctx), LoweredValue::Imm(imm));
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
                if tls_globals.contains(&symbol) {
                    // A thread-local global: local-exec model — the address is
                    // the thread pointer plus a link-time-constant offset,
                    // materialized as `mrs` + two TPREL adds.
                    let dst = fresh_vreg(&mut next_vreg);
                    aarch64_ops::mrs_tpidr(ctx, dst)
                        .insert_at_back(insert_block, ctx);
                    aarch64_ops::add_tprel_hi12(ctx, dst, dst, symbol.clone())
                        .insert_at_back(insert_block, ctx);
                    aarch64_ops::add_tprel_lo12_nc(ctx, dst, dst, symbol)
                        .insert_at_back(insert_block, ctx);
                    values.insert(addr.get_result(ctx), LoweredValue::Reg(dst));
                } else if data_globals.contains(&symbol) {
                    // A data-section global: materialize its address with an
                    // adrp+add pair the linker resolves through page
                    // relocations.
                    let dst = fresh_vreg(&mut next_vreg);
                    aarch64_ops::adrp(ctx, dst, symbol.clone())
                        .insert_at_back(insert_block, ctx);
                    aarch64_ops::add_lo12(ctx, dst, dst, symbol)
                        .insert_at_back(insert_block, ctx);
                    values.insert(addr.get_result(ctx), LoweredValue::Reg(dst));
                } else if let Some(bytes) = globals.get(&symbol).cloned() {
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
            } else if let Some(cast) = op_obj.downcast_ref::<SExtOp>() {
                // Registers hold values zero-extended to 64 bits, so extend
                // the source's W low bits with (x ^ 2^(W-1)) - 2^(W-1) and
                // re-mask to the destination width to restore the invariant.
                // A 128-bit destination gets an explicit sign-extended high
                // half: the pair fallback in `materialize_pair` sees only the
                // signless post-mir-lower type and would zero it.
                let result = cast.get_result(ctx);
                let operand = cast.get_operand(ctx);
                let value = lookup_value(ctx, &values, operand)?;
                let result_is_i128 = is_128_bit_integer(ctx, result.get_type(ctx));
                match integer_trunc_mask(ctx, operand.get_type(ctx)) {
                    None => {
                        // 64-bit source: the register already carries the
                        // full sign pattern.
                        if result_is_i128 {
                            let lo = materialize(
                                ctx,
                                insert_block,
                                value,
                                &mut next_vreg,
                                "sext input",
                            )?;
                            let hi = sign_extend_high_half(
                                ctx,
                                insert_block,
                                lo.clone(),
                                &mut next_vreg,
                            );
                            values.insert(result, LoweredValue::RegPair(lo, hi));
                        } else {
                            values.insert(result, value);
                        }
                    }
                    Some(src_mask) => {
                        let sign_bit = (src_mask >> 1) + 1;
                        let dst_mask = integer_trunc_mask(ctx, result.get_type(ctx));
                        if let LoweredValue::Imm(imm) = value {
                            let extended =
                                ((imm as u64) ^ sign_bit).wrapping_sub(sign_bit);
                            let masked = if result_is_i128 {
                                extended as i64 as i128 as u128
                            } else {
                                dst_mask.map_or(extended, |mask| extended & mask) as u128
                            };
                            values.insert(result, LoweredValue::Imm(masked));
                        } else {
                            let src = materialize(
                                ctx,
                                insert_block,
                                value,
                                &mut next_vreg,
                                "sext input",
                            )?;
                            let sign_reg = fresh_vreg(&mut next_vreg);
                            materialize_u64_immediate(ctx, insert_block, sign_reg.clone(), sign_bit);
                            let flipped = fresh_vreg(&mut next_vreg);
                            aarch64_ops::binary(
                                ctx,
                                aarch64_ops::XorOp::OPCODE,
                                flipped.clone(),
                                src,
                                sign_reg.clone(),
                            )
                            .insert_at_back(insert_block, ctx);
                            let extended = fresh_vreg(&mut next_vreg);
                            aarch64_ops::binary(
                                ctx,
                                aarch64_ops::SubOp::OPCODE,
                                extended.clone(),
                                flipped,
                                sign_reg,
                            )
                            .insert_at_back(insert_block, ctx);
                            if let Some(mask) = dst_mask {
                                let mask_reg = fresh_vreg(&mut next_vreg);
                                materialize_u64_immediate(ctx, insert_block, mask_reg.clone(), mask);
                                let dst = fresh_vreg(&mut next_vreg);
                                aarch64_ops::binary(
                                    ctx,
                                    aarch64_ops::AndOp::OPCODE,
                                    dst.clone(),
                                    extended,
                                    mask_reg,
                                )
                                .insert_at_back(insert_block, ctx);
                                values.insert(result, LoweredValue::Reg(dst));
                            } else if result_is_i128 {
                                let hi = sign_extend_high_half(
                                    ctx,
                                    insert_block,
                                    extended.clone(),
                                    &mut next_vreg,
                                );
                                values.insert(result, LoweredValue::RegPair(extended, hi));
                            } else {
                                values.insert(result, LoweredValue::Reg(extended));
                            }
                        }
                    }
                }
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
                } else if let LoweredValue::RegPair(lo, _) = value {
                    // 128-bit source, 64-bit result: the low half is the
                    // whole value; keeping the pair would smuggle the old
                    // high half into later widening uses.
                    values.insert(result, LoweredValue::Reg(lo));
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
                // Rust's saturating float-to-int `as` casts arrive as calls
                // to mir-lower's `llvm_fpto{s,u}i_sat_*` intrinsic
                // declarations; `fcvtzs`/`fcvtzu` implement them exactly
                // (saturation at the bounds, NaN to 0), so inline them
                // instead of emitting an unresolvable call.
                if let CallOpCallable::Direct(name) = &callee
                    && let Some((signed, _width)) =
                        saturating_fp_to_int_intrinsic(name.as_ref())
                {
                    let [arg] = args.as_slice() else {
                        return Err(input_error_noloc!(Aarch64Err::UnsupportedOp(format!(
                            "saturating float-to-int intrinsic `{name}` expects one argument"
                        ))));
                    };
                    let result = call.get_operation().deref(ctx).get_result(0);
                    let lowered_value = lookup_value(ctx, &values, *arg)?;
                    let lowered = fp_to_int(
                        ctx,
                        insert_block,
                        lowered_value,
                        arg.get_type(ctx),
                        result.get_type(ctx),
                        signed,
                        &mut next_vreg,
                    )?;
                    values.insert(result, lowered);
                    continue;
                }
                // Float math with a single-instruction lowering arrives as
                // calls to `llvm_<op>_f{32,64}` declarations (sqrt, fabs,
                // the rounding family, min/max); select the FP instruction
                // instead of emitting an unresolvable call.
                if let CallOpCallable::Direct(name) = &callee
                    && let Some(intrinsic) = fp_math_intrinsic(name.as_ref())
                {
                    let result = call.get_operation().deref(ctx).get_result(0);
                    let fp = fp_kind(ctx, result.get_type(ctx)).ok_or_else(|| {
                        input_error_noloc!(Aarch64Err::UnsupportedType(format!(
                            "float math intrinsic `{name}` on non-scalar-float type {}",
                            pliron::printable::Printable::disp(&result.get_type(ctx), ctx)
                        )))
                    })?;
                    let mut operands = Vec::with_capacity(args.len());
                    for arg in &args {
                        let lowered = lookup_value(ctx, &values, *arg)?;
                        operands.push(materialize_fp(
                            ctx,
                            insert_block,
                            lowered,
                            fp,
                            &mut next_vreg,
                            "float math intrinsic argument",
                        )?);
                    }
                    let dst = fresh_fpr(&mut next_vreg, fp);
                    match (intrinsic, operands.as_slice()) {
                        (FpMathIntrinsic::Unary(d, s), [src]) => {
                            let opcode = if fp == FpKind::F64 { d } else { s };
                            aarch64_ops::unary(ctx, opcode, dst.clone(), src.clone())
                                .insert_at_back(insert_block, ctx);
                        }
                        (FpMathIntrinsic::Binary(d, s), [lhs, rhs]) => {
                            let opcode = if fp == FpKind::F64 { d } else { s };
                            aarch64_ops::binary(ctx, opcode, dst.clone(), lhs.clone(), rhs.clone())
                                .insert_at_back(insert_block, ctx);
                        }
                        _ => {
                            return Err(input_error_noloc!(Aarch64Err::UnsupportedOp(format!(
                                "float math intrinsic `{name}` with {} arguments",
                                args.len()
                            ))));
                        }
                    }
                    values.insert(result, LoweredValue::Reg(dst));
                    continue;
                }
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
                    (Some(result), Some(ResultLocation::IndirectX8)) => {
                        Some(stack.allocate(ctx, result.get_type(ctx))?)
                    }
                    _ => None,
                };
                // Assign argument locations in a single walk mirroring
                // `assign_abi`: integers and pointers draw from x0-x7, FP
                // scalars from v0-v7, and overflow goes to 8-byte outgoing
                // stack slots in argument order.
                let mut arg_moves: Vec<(Register, CallArgDst)> = Vec::new();
                let mut next_gpr = 0u8;
                let mut next_fpr = 0u8;
                let mut next_stack = 0u64;
                for arg in args {
                    let lowered = lookup_value(ctx, &values, arg)?;
                    let arg_ty = arg.get_type(ctx);
                    if let Some(fp) = fp_kind(ctx, arg_ty) {
                        let src = materialize_fp(
                            ctx,
                            insert_block,
                            lowered,
                            fp,
                            &mut next_vreg,
                            "call argument",
                        )?;
                        if next_fpr < 8 {
                            let dst = match fp {
                                FpKind::F64 => Register::fpr64(next_fpr),
                                FpKind::F32 => Register::fpr32(next_fpr),
                            };
                            next_fpr += 1;
                            arg_moves.push((src, CallArgDst::Fpr(dst)));
                        } else {
                            arg_moves.push((src, CallArgDst::Stack(next_stack)));
                            next_stack += 8;
                        }
                        continue;
                    }
                    let push_gpr = |src: Register,
                                        arg_moves: &mut Vec<(Register, CallArgDst)>,
                                        next_gpr: &mut u8,
                                        next_stack: &mut u64| {
                        if *next_gpr < 8 {
                            arg_moves.push((src, CallArgDst::Gpr(*next_gpr)));
                            *next_gpr += 1;
                        } else {
                            arg_moves.push((src, CallArgDst::Stack(*next_stack)));
                            *next_stack += 8;
                        }
                    };
                    if is_128_bit_integer(ctx, arg_ty) {
                        let (lo, hi) = materialize_pair(
                            ctx,
                            insert_block,
                            lowered,
                            arg_ty,
                            &mut next_vreg,
                            "call argument",
                        )?;
                        push_gpr(lo, &mut arg_moves, &mut next_gpr, &mut next_stack);
                        push_gpr(hi, &mut arg_moves, &mut next_gpr, &mut next_stack);
                    } else {
                        let src = materialize_typed(
                            ctx,
                            insert_block,
                            lowered,
                            arg_ty,
                            &mut next_vreg,
                            "call argument",
                        )?;
                        push_gpr(src, &mut arg_moves, &mut next_gpr, &mut next_stack);
                    }
                }
                // The stack slots land in the function's reserved outgoing
                // area at the bottom of the frame; sp does not move.
                debug_assert!(align_to_16(next_stack) <= stack.outgoing_area_bytes());
                if let Some(callee_ptr) = &callee_ptr {
                    // x16 is an intra-procedure-call scratch register outside
                    // the allocatable set, so it survives the argument moves.
                    aarch64_ops::mov(ctx, X16, *callee_ptr).insert_at_back(insert_block, ctx);
                }
                for (src, dst) in arg_moves {
                    match dst {
                        CallArgDst::Gpr(number) => {
                            aarch64_ops::mov(ctx, Register::gpr(number), src)
                                .insert_at_back(insert_block, ctx);
                        }
                        CallArgDst::Fpr(reg) => {
                            emit_move(ctx, insert_block, reg, src)?;
                        }
                        CallArgDst::Stack(offset) => {
                            let opcode = match src.class() {
                                RegisterClass::Fpr64 => aarch64_ops::StrdSpOffsetOp::OPCODE,
                                RegisterClass::Fpr32 => aarch64_ops::StrsSpOffsetOp::OPCODE,
                                _ => aarch64_ops::StrSpOffsetOp::OPCODE,
                            };
                            aarch64_ops::str_sp_offset_sized(ctx, opcode, src, offset)
                                .insert_at_back(insert_block, ctx);
                        }
                    }
                }
                if let Some(slot) = indirect_result_slot {
                    aarch64_ops::add_sp_offset(ctx, X8, slot.offset)
                        .insert_at_back(insert_block, ctx);
                }
                match callee {
                    CallOpCallable::Direct(callee) => {
                        aarch64_ops::call(ctx, callee).insert_at_back(insert_block, ctx);
                    }
                    CallOpCallable::Indirect(_) => {
                        aarch64_ops::blr(ctx, X16).insert_at_back(insert_block, ctx);
                    }
                }
                if let Some(result) = result {
                    let lowered = match result_location.unwrap() {
                        ResultLocation::Fpr(fp) => {
                            let dst = fresh_fpr(&mut next_vreg, fp);
                            let src = match fp {
                                FpKind::F64 => Register::fpr64(0),
                                FpKind::F32 => Register::fpr32(0),
                            };
                            emit_move(ctx, insert_block, dst, src)?;
                            LoweredValue::Reg(dst)
                        }
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
                                input_error_noloc!(Aarch64Err::UnsupportedOp(
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
            } else if let Some(kind) = float_binary_kind(&*op_obj) {
                let op_ref = op_ptr.deref(ctx);
                let result = op_ref.get_result(0);
                let lhs = op_ref.get_operand(0);
                let rhs = op_ref.get_operand(1);
                drop(op_ref);
                let fp = fp_kind(ctx, result.get_type(ctx)).ok_or_else(|| {
                    input_error_noloc!(Aarch64Err::UnsupportedType(format!(
                        "float arithmetic on non-scalar-float type {}",
                        pliron::printable::Printable::disp(&result.get_type(ctx), ctx)
                    )))
                })?;
                let lhs_value = lookup_value(ctx, &values, lhs)?;
                let rhs_value = lookup_value(ctx, &values, rhs)?;
                let lhs = materialize_fp(ctx, insert_block, lhs_value, fp, &mut next_vreg, "fp lhs")?;
                let rhs = materialize_fp(ctx, insert_block, rhs_value, fp, &mut next_vreg, "fp rhs")?;
                if kind == FloatBinaryKind::Rem {
                    // No hardware remainder: call the C runtime's
                    // fmod/fmodf through the standard FP argument registers.
                    let (a0, a1, callee) = match fp {
                        FpKind::F64 => (Register::fpr64(0), Register::fpr64(1), "fmod"),
                        FpKind::F32 => (Register::fpr32(0), Register::fpr32(1), "fmodf"),
                    };
                    emit_move(ctx, insert_block, a0, lhs)?;
                    emit_move(ctx, insert_block, a1, rhs)?;
                    aarch64_ops::call(ctx, callee.try_into().unwrap())
                        .insert_at_back(insert_block, ctx);
                    let dst = fresh_fpr(&mut next_vreg, fp);
                    emit_move(ctx, insert_block, dst, a0)?;
                    values.insert(result, LoweredValue::Reg(dst));
                    continue;
                }
                let opcode = match (kind, fp) {
                    (FloatBinaryKind::Add, FpKind::F64) => aarch64_ops::FaddDOp::OPCODE,
                    (FloatBinaryKind::Add, FpKind::F32) => aarch64_ops::FaddSOp::OPCODE,
                    (FloatBinaryKind::Sub, FpKind::F64) => aarch64_ops::FsubDOp::OPCODE,
                    (FloatBinaryKind::Sub, FpKind::F32) => aarch64_ops::FsubSOp::OPCODE,
                    (FloatBinaryKind::Mul, FpKind::F64) => aarch64_ops::FmulDOp::OPCODE,
                    (FloatBinaryKind::Mul, FpKind::F32) => aarch64_ops::FmulSOp::OPCODE,
                    (FloatBinaryKind::Div, FpKind::F64) => aarch64_ops::FdivDOp::OPCODE,
                    (FloatBinaryKind::Div, FpKind::F32) => aarch64_ops::FdivSOp::OPCODE,
                    (FloatBinaryKind::Rem, _) => unreachable!("handled above"),
                };
                let dst = fresh_fpr(&mut next_vreg, fp);
                aarch64_ops::binary(ctx, opcode, dst, lhs, rhs).insert_at_back(insert_block, ctx);
                values.insert(result, LoweredValue::Reg(dst));
            } else if let Some(fneg) = op_obj.downcast_ref::<FNegOp>() {
                let result = fneg.get_result(ctx);
                let fp = fp_kind(ctx, result.get_type(ctx)).ok_or_else(|| {
                    input_error_noloc!(Aarch64Err::UnsupportedType(
                        "fneg on non-scalar-float type".to_string()
                    ))
                })?;
                let value = lookup_value(ctx, &values, fneg.get_operand(ctx))?;
                let src = materialize_fp(ctx, insert_block, value, fp, &mut next_vreg, "fneg input")?;
                let opcode = match fp {
                    FpKind::F64 => aarch64_ops::FnegDOp::OPCODE,
                    FpKind::F32 => aarch64_ops::FnegSOp::OPCODE,
                };
                let dst = fresh_fpr(&mut next_vreg, fp);
                aarch64_ops::unary(ctx, opcode, dst, src).insert_at_back(insert_block, ctx);
                values.insert(result, LoweredValue::Reg(dst));
            } else if let Some(fcmp) = op_obj.downcast_ref::<FCmpOp>() {
                // Unlike icmp (kept symbolic so branches can fuse cmp with
                // b.cond), fcmp lowers eagerly to a 0/1 GPR at its program
                // point: nzcv cannot be carried across blocks anyway.
                let lhs = fcmp.get_operation().deref(ctx).get_operand(0);
                let rhs = fcmp.get_operation().deref(ctx).get_operand(1);
                let bit = lower_fcmp(
                    ctx,
                    insert_block,
                    &values,
                    fcmp.predicate(ctx),
                    lhs,
                    rhs,
                    &mut next_vreg,
                )?;
                values.insert(fcmp.get_result(ctx), bit);
            } else if let Some(cast) = op_obj.downcast_ref::<SIToFPOp>() {
                let (value, result) = (cast.get_operand(ctx), cast.get_result(ctx));
                let lowered = int_to_fp(
                    ctx, insert_block, &values, value, result, true, &mut next_vreg,
                )?;
                values.insert(result, lowered);
            } else if let Some(cast) = op_obj.downcast_ref::<UIToFPOp>() {
                let (value, result) = (cast.get_operand(ctx), cast.get_result(ctx));
                let lowered = int_to_fp(
                    ctx, insert_block, &values, value, result, false, &mut next_vreg,
                )?;
                values.insert(result, lowered);
            } else if let Some(cast) = op_obj.downcast_ref::<FPToSIOp>() {
                let (value, result) = (cast.get_operand(ctx), cast.get_result(ctx));
                let src_ty = value.get_type(ctx);
                let lowered_value = lookup_value(ctx, &values, value)?;
                let lowered = fp_to_int(
                    ctx,
                    insert_block,
                    lowered_value,
                    src_ty,
                    result.get_type(ctx),
                    true,
                    &mut next_vreg,
                )?;
                values.insert(result, lowered);
            } else if let Some(cast) = op_obj.downcast_ref::<FPToUIOp>() {
                let (value, result) = (cast.get_operand(ctx), cast.get_result(ctx));
                let src_ty = value.get_type(ctx);
                let lowered_value = lookup_value(ctx, &values, value)?;
                let lowered = fp_to_int(
                    ctx,
                    insert_block,
                    lowered_value,
                    src_ty,
                    result.get_type(ctx),
                    false,
                    &mut next_vreg,
                )?;
                values.insert(result, lowered);
            } else if let Some(cast) = op_obj.downcast_ref::<FPExtOp>() {
                let value = lookup_value(ctx, &values, cast.get_operand(ctx))?;
                let src = materialize_fp(
                    ctx, insert_block, value, FpKind::F32, &mut next_vreg, "fpext input",
                )?;
                let dst = fresh_fpr(&mut next_vreg, FpKind::F64);
                aarch64_ops::unary(ctx, aarch64_ops::FcvtDSOp::OPCODE, dst, src)
                    .insert_at_back(insert_block, ctx);
                values.insert(cast.get_result(ctx), LoweredValue::Reg(dst));
            } else if let Some(cast) = op_obj.downcast_ref::<FPTruncOp>() {
                let value = lookup_value(ctx, &values, cast.get_operand(ctx))?;
                let src = materialize_fp(
                    ctx, insert_block, value, FpKind::F64, &mut next_vreg, "fptrunc input",
                )?;
                let dst = fresh_fpr(&mut next_vreg, FpKind::F32);
                aarch64_ops::unary(ctx, aarch64_ops::FcvtSDOp::OPCODE, dst, src)
                    .insert_at_back(insert_block, ctx);
                values.insert(cast.get_result(ctx), LoweredValue::Reg(dst));
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
                // Arithmetic shift right and signed division/remainder need
                // sign-extended operands: the 64-bit `sdiv`/`asr` forms are
                // used for every width, and the (signless) type does not
                // carry the signedness, the op does.
                let signed_operands =
                    matches!(kind, BinaryKind::AShr | BinaryKind::SDiv | BinaryKind::SRem);
                let lhs_ty = if signed_operands {
                    signed_variant_ty(ctx, lhs.get_type(ctx))
                } else {
                    lhs.get_type(ctx)
                };
                let rhs_ty = if matches!(kind, BinaryKind::SDiv | BinaryKind::SRem) {
                    signed_variant_ty(ctx, rhs.get_type(ctx))
                } else {
                    rhs.get_type(ctx)
                };
                let lhs = materialize_typed(
                    ctx,
                    insert_block,
                    lhs_value,
                    lhs_ty,
                    &mut next_vreg,
                    "binary lhs",
                )?;
                let rhs = materialize_typed(
                    ctx,
                    insert_block,
                    rhs_value,
                    rhs_ty,
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
                if has_call {
                    aarch64_ops::ldr_post_sp(ctx, LR, 16).insert_at_back(insert_block, ctx);
                }
                aarch64_ops::ret(ctx).insert_at_back(insert_block, ctx);
            } else if let Some(_unreachable) = op_obj.downcast_ref::<UnreachableOp>() {
                aarch64_ops::brk(ctx).insert_at_back(insert_block, ctx);
            } else if let Some(br) = op_obj.downcast_ref::<BrOp>() {
                let dest = br.get_operation().deref(ctx).get_successor(0);
                let args = br.successor_operands(ctx, 0);
                emit_block_arg_copies(ctx, insert_block, &values, dest, &args, &mut next_vreg)?;
                let target = machine_block(&block_map, dest)?;
                aarch64_ops::b(ctx, target).insert_at_back(insert_block, ctx);
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
                    let lhs_ty =
                        compare_operand_ty(ctx, compare.predicate.clone(), compare.lhs_ty);
                    let rhs_ty =
                        compare_operand_ty(ctx, compare.predicate.clone(), compare.rhs_ty);
                    let lhs = materialize_typed(
                        ctx,
                        insert_block,
                        *compare.lhs,
                        lhs_ty,
                        &mut next_vreg,
                        "icmp lhs",
                    )?;
                    let rhs = materialize_typed(
                        ctx,
                        insert_block,
                        *compare.rhs,
                        rhs_ty,
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
                return Err(input_error_noloc!(Aarch64Err::UnsupportedOp(
                    opid.to_string()
                )));
            }
        }
    }
    stamper.finish(ctx);

    func.set_stack_size(ctx, align_to_16(stack.next_offset));
    Ok(())
}

// Lowered values and materialization ----------------------------------------

pub(super) fn fresh_vreg(next_vreg: &mut usize) -> Register {
    let reg = Register::virtual_gpr(*next_vreg as u32);
    *next_vreg += 1;
    reg
}

/// The scalar floating-point width of a value: `f64` maps to the `d`
/// register file, `f32` to the `s` file.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum FpKind {
    F32,
    F64,
}

impl FpKind {
    pub(super) fn class(self) -> RegisterClass {
        match self {
            Self::F32 => RegisterClass::Fpr32,
            Self::F64 => RegisterClass::Fpr64,
        }
    }
}

pub(super) fn fp_kind(ctx: &Context, ty: TypeHandle) -> Option<FpKind> {
    let ty_ref = ty.deref(ctx);
    if ty_ref
        .downcast_ref::<crate::dialects::builtin::types::FP32Type>()
        .is_some()
    {
        return Some(FpKind::F32);
    }
    if ty_ref
        .downcast_ref::<crate::dialects::builtin::types::FP64Type>()
        .is_some()
    {
        return Some(FpKind::F64);
    }
    None
}

pub(super) fn fresh_fpr(next_vreg: &mut usize, kind: FpKind) -> Register {
    let reg = match kind {
        FpKind::F64 => Register::virtual_fpr64(*next_vreg as u32),
        FpKind::F32 => Register::virtual_fpr32(*next_vreg as u32),
    };
    *next_vreg += 1;
    reg
}

/// A register-class-aware copy: GPR-to-GPR uses `mov`, FP-to-FP the matching
/// `fmov` form, and cross-file copies the bit-preserving `fmov` between the
/// register files.
pub(super) fn emit_move(
    ctx: &mut Context,
    block: Ptr<crate::ir::basic_block::BasicBlock>,
    dst: Register,
    src: Register,
) -> STAIRResult<()> {
    use RegisterClass::{Fpr32, Fpr64, Gpr64};
    let op = match (dst.class(), src.class()) {
        (Gpr64, Gpr64) => aarch64_ops::mov(ctx, dst, src),
        (Fpr64, Fpr64) => aarch64_ops::fmov_rr(ctx, aarch64_ops::FmovDOp::OPCODE, dst, src),
        (Fpr32, Fpr32) => aarch64_ops::fmov_rr(ctx, aarch64_ops::FmovSOp::OPCODE, dst, src),
        (Fpr64, Gpr64) => aarch64_ops::unary(ctx, aarch64_ops::FmovDXOp::OPCODE, dst, src),
        (Gpr64, Fpr64) => aarch64_ops::unary(ctx, aarch64_ops::FmovXDOp::OPCODE, dst, src),
        (Fpr32, Gpr64) => aarch64_ops::unary(ctx, aarch64_ops::FmovSWOp::OPCODE, dst, src),
        (Gpr64, Fpr32) => aarch64_ops::unary(ctx, aarch64_ops::FmovWSOp::OPCODE, dst, src),
        (dst_class, src_class) => {
            return Err(input_error_noloc!(Aarch64Err::UnsupportedOp(format!(
                "register copy between incompatible classes {dst_class:?} <- {src_class:?}"
            ))));
        }
    };
    op.insert_at_back(block, ctx);
    Ok(())
}

/// Materialize the FP value with bit pattern `bits` into a fresh FP register:
/// an `fmov` immediate when the pattern is VFPExpandImm-representable, a GPR
/// materialization plus cross-file `fmov` for cheap patterns (zero or a
/// single 16-bit chunk), and a literal-pool load otherwise.
fn materialize_fp_constant(
    ctx: &mut Context,
    block: Ptr<crate::ir::basic_block::BasicBlock>,
    bits: u64,
    kind: FpKind,
    next_vreg: &mut usize,
) -> Register {
    let dst = fresh_fpr(next_vreg, kind);
    let imm8 = match kind {
        FpKind::F64 => fmov_imm8_for_f64_bits(bits),
        FpKind::F32 => fmov_imm8_for_f32_bits(bits as u32),
    };
    if let Some(imm8) = imm8 {
        let opcode = match kind {
            FpKind::F64 => aarch64_ops::FmovImmDOp::OPCODE,
            FpKind::F32 => aarch64_ops::FmovImmSOp::OPCODE,
        };
        aarch64_ops::fmov_imm(ctx, opcode, dst, imm8 as u64).insert_at_back(block, ctx);
        return dst;
    }
    // A pattern whose GPR materialization is at most one instruction (zero,
    // or a single 16-bit chunk) is cheaper through the integer file than
    // through the literal pool.
    let chunks = (0..4)
        .filter(|chunk| (bits >> (chunk * 16)) & 0xffff != 0)
        .count();
    if chunks <= 1 {
        let gpr = fresh_vreg(next_vreg);
        materialize_u64_immediate(ctx, block, gpr, bits);
        let opcode = match kind {
            FpKind::F64 => aarch64_ops::FmovDXOp::OPCODE,
            FpKind::F32 => aarch64_ops::FmovSWOp::OPCODE,
        };
        aarch64_ops::unary(ctx, opcode, dst, gpr).insert_at_back(block, ctx);
        return dst;
    }
    // Arbitrary bit pattern: place it in the literal pool and load it. The
    // label is content-addressed, so repeated constants share one entry.
    let (label, bytes, load_opcode) = match kind {
        FpKind::F64 => (
            format!("L_stair_fp64_{bits:016x}"),
            bits.to_le_bytes().to_vec(),
            aarch64_ops::LdrdRegOffsetOp::OPCODE,
        ),
        FpKind::F32 => (
            format!("L_stair_fp32_{:08x}", bits as u32),
            (bits as u32).to_le_bytes().to_vec(),
            aarch64_ops::LdrsRegOffsetOp::OPCODE,
        ),
    };
    let addr = fresh_vreg(next_vreg);
    aarch64_ops::adr_literal(ctx, addr, label, bytes).insert_at_back(block, ctx);
    aarch64_ops::ldr_reg_offset_sized(ctx, load_opcode, dst, addr, 0).insert_at_back(block, ctx);
    dst
}

/// Materialize `value` into an FP register of `kind`'s class. Values already
/// in the right FP class pass through; GPR-resident values (bitcasts, packed
/// aggregate fields, stack-argument loads) cross the register files with
/// `fmov`; immediates carry their IEEE bit pattern.
pub(super) fn materialize_fp(
    ctx: &mut Context,
    block: Ptr<crate::ir::basic_block::BasicBlock>,
    value: LoweredValue,
    kind: FpKind,
    next_vreg: &mut usize,
    context: &str,
) -> STAIRResult<Register> {
    match value {
        LoweredValue::Reg(reg) if reg.class() == kind.class() => Ok(reg),
        LoweredValue::Reg(reg) if reg.is_fpr() => Err(input_error_noloc!(
            Aarch64Err::UnsupportedOp(format!(
                "cannot materialize {context}: FP register {reg} has the wrong width for {kind:?}"
            ))
        )),
        LoweredValue::Imm(bits) => Ok(materialize_fp_constant(
            ctx,
            block,
            bits as u64,
            kind,
            next_vreg,
        )),
        LoweredValue::Undef => {
            Ok(materialize_fp_constant(ctx, block, 0, kind, next_vreg))
        }
        other => {
            // Anything else (a GPR register, an aggregate wrapper, ...)
            // materializes to its 64-bit bit pattern first and then crosses
            // into the FP file.
            let gpr = materialize(ctx, block, other, next_vreg, context)?;
            let dst = fresh_fpr(next_vreg, kind);
            let opcode = match kind {
                FpKind::F64 => aarch64_ops::FmovDXOp::OPCODE,
                FpKind::F32 => aarch64_ops::FmovSWOp::OPCODE,
            };
            aarch64_ops::unary(ctx, opcode, dst, gpr).insert_at_back(block, ctx);
            Ok(dst)
        }
    }
}

/// Where one materialized call argument goes: an integer argument register,
/// an FP argument register, or an 8-byte outgoing stack slot.
enum CallArgDst {
    Gpr(u8),
    Fpr(Register),
    Stack(u64),
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

struct StackAllocator {
    /// The outgoing stack-argument area reserved below every frame slot.
    outgoing_area: u64,
    next_offset: u64,
}

impl StackAllocator {
    fn new(outgoing_area: u64) -> Self {
        Self {
            outgoing_area,
            next_offset: outgoing_area,
        }
    }

    fn outgoing_area_bytes(&self) -> u64 {
        self.outgoing_area
    }

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
        input_error_noloc!(Aarch64Err::UndefinedValue(
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
            return Err(input_error_noloc!(Aarch64Err::UnsupportedOp(
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
            return Err(input_error_noloc!(Aarch64Err::UnsupportedOp(
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
            return Err(input_error_noloc!(Aarch64Err::UnsupportedOp(format!(
                "cannot materialize {context}: multi-field aggregate of type {}",
                pliron::printable::Printable::disp(&ty, ctx)
            ))));
        }
    }
    if is_128_bit_integer(ctx, ty) {
        let (lo, _) = materialize_pair(ctx, entry, value, ty, next_vreg, context)?;
        return Ok(lo);
    }
    if let Some(kind) = fp_kind(ctx, ty) {
        return materialize_fp(ctx, entry, value, kind, next_vreg, context);
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
                    input_error_noloc!(Aarch64Err::UndefinedValue(
                        "missing low 128-bit lane".to_string()
                    ))
                })?;
            let hi = fields
                .get(1)
                .and_then(|field| field.clone())
                .ok_or_else(|| {
                    input_error_noloc!(Aarch64Err::UndefinedValue(
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
            let hi = if integer_width_and_signedness(ctx, ty)
                .map(|(_, signed)| signed)
                .unwrap_or(false)
            {
                sign_extend_high_half(ctx, entry, lo.clone(), next_vreg)
            } else {
                let hi = fresh_vreg(next_vreg);
                materialize_u64_immediate(ctx, entry, hi.clone(), 0);
                hi
            };
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

/// The sign-extension high half of a 64-bit register: `(lo >> 63) * -1`,
/// i.e. all zeros or all ones depending on `lo`'s sign bit.
fn sign_extend_high_half(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    lo: Register,
    next_vreg: &mut usize,
) -> Register {
    let sign = fresh_vreg(next_vreg);
    materialize_u64_immediate(ctx, entry, sign.clone(), 63);
    let hi = fresh_vreg(next_vreg);
    aarch64_ops::binary(ctx, aarch64_ops::LsrOp::OPCODE, hi.clone(), lo, sign)
        .insert_at_back(entry, ctx);
    let mask = fresh_vreg(next_vreg);
    materialize_u64_immediate(ctx, entry, mask.clone(), 0u64.wrapping_sub(1));
    aarch64_ops::binary(
        ctx,
        aarch64_ops::MulOp::OPCODE,
        hi.clone(),
        hi.clone(),
        mask,
    )
    .insert_at_back(entry, ctx);
    hi
}

pub(super) fn materialize(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    value: LoweredValue,
    next_vreg: &mut usize,
    context: &str,
) -> STAIRResult<Register> {
    match value {
        // A value living in the FP file used as an integer (bitcast, packed
        // aggregate field): move its bit pattern across the register files.
        LoweredValue::Reg(reg) if reg.is_fpr() => {
            let dst = fresh_vreg(next_vreg);
            let opcode = if reg.class() == RegisterClass::Fpr64 {
                aarch64_ops::FmovXDOp::OPCODE
            } else {
                aarch64_ops::FmovWSOp::OPCODE
            };
            aarch64_ops::unary(ctx, opcode, dst, reg).insert_at_back(entry, ctx);
            Ok(dst)
        }
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
                input_error_noloc!(Aarch64Err::UndefinedValue(
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
        LoweredValue::Aggregate(_) => Err(input_error_noloc!(Aarch64Err::UnsupportedOp(
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
                    input_error_noloc!(Aarch64Err::UndefinedValue(format!(
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
            return Err(input_error_noloc!(Aarch64Err::UnsupportedType(
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
            return Err(input_error_noloc!(Aarch64Err::UnsupportedType(
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
            return Err(input_error_noloc!(Aarch64Err::UnsupportedType(
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
            return Err(input_error_noloc!(Aarch64Err::UnsupportedType(
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
    if let Some(kind) = fp_kind(ctx, ty) {
        return Ok(LoweredValue::Reg(fresh_fpr(next_vreg, kind)));
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
        || ty_ref
            .downcast_ref::<crate::dialects::builtin::types::FP32Type>()
            .is_some()
        || ty_ref
            .downcast_ref::<crate::dialects::builtin::types::FP64Type>()
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
        input_error_noloc!(Aarch64Err::UnsupportedOp(
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
        return Err(input_error_noloc!(Aarch64Err::UnsupportedType(
            format!("{:?}", &*ty_ref)
        )));
    };
    if struct_ty.is_opaque() {
        return Err(input_error_noloc!(Aarch64Err::UnsupportedType(
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
            BinaryKind::AShr => (lhs as i128).wrapping_shr(rhs as u32) as u128,
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
        BinaryKind::AShr => {
            (sign_extend_immediate(lhs, width) as i64).wrapping_shr(rhs as u32) as u64
        }
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
        BinaryKind::AShr => aarch64_ops::AsrOp::OPCODE,
    }
}

/// The AArch64 condition code that tests `predicate` after a `cmp`.
/// The register type to materialize a compare operand as. Signed predicates
/// must see sign-extended operands, but mir-lower emits signless integer
/// types (LLVM convention) where the *predicate*, not the type, carries the
/// comparison's signedness. Re-type sub-64-bit operands as signed so
/// [normalize_integer_reg] sign-extends them.
pub(super) fn compare_operand_ty(
    ctx: &mut Context,
    predicate: ICmpPredicateAttr,
    ty: TypeHandle,
) -> TypeHandle {
    if !matches!(
        predicate,
        ICmpPredicateAttr::SLT
            | ICmpPredicateAttr::SLE
            | ICmpPredicateAttr::SGT
            | ICmpPredicateAttr::SGE
    ) {
        return ty;
    }
    let width = {
        let ty_ref = ty.deref(ctx);
        match ty_ref.downcast_ref::<crate::dialects::builtin::types::IntegerType>() {
            Some(int_ty) if int_ty.width() < 64 => int_ty.width(),
            _ => return ty,
        }
    };
    crate::dialects::builtin::types::IntegerType::get(
        ctx,
        width,
        crate::dialects::builtin::types::Signedness::Signed,
    )
    .into()
}

// Floating-point lowering helpers ---------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FloatBinaryKind {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
}

fn float_binary_kind(op: &dyn Op) -> Option<FloatBinaryKind> {
    if op.downcast_ref::<FAddOp>().is_some() {
        Some(FloatBinaryKind::Add)
    } else if op.downcast_ref::<FSubOp>().is_some() {
        Some(FloatBinaryKind::Sub)
    } else if op.downcast_ref::<FMulOp>().is_some() {
        Some(FloatBinaryKind::Mul)
    } else if op.downcast_ref::<FDivOp>().is_some() {
        Some(FloatBinaryKind::Div)
    } else if op.downcast_ref::<FRemOp>().is_some() {
        Some(FloatBinaryKind::Rem)
    } else {
        None
    }
}

/// The condition code holding after `fcmp` for a predicate that maps to a
/// single test. On AArch64 an unordered comparison sets nzcv to `0011`
/// (C and V), which makes exactly `one`/`ueq` unrepresentable as one code.
fn fcmp_condition_code(predicate: &FCmpPredicateAttr) -> Option<ConditionCode> {
    Some(match predicate {
        FCmpPredicateAttr::OEQ => ConditionCode::Eq,
        FCmpPredicateAttr::OGT => ConditionCode::Gt,
        FCmpPredicateAttr::OGE => ConditionCode::Ge,
        FCmpPredicateAttr::OLT => ConditionCode::Mi,
        FCmpPredicateAttr::OLE => ConditionCode::Ls,
        FCmpPredicateAttr::ORD => ConditionCode::Vc,
        FCmpPredicateAttr::UNE => ConditionCode::Ne,
        FCmpPredicateAttr::UGT => ConditionCode::Hi,
        FCmpPredicateAttr::UGE => ConditionCode::Pl,
        FCmpPredicateAttr::ULT => ConditionCode::Lt,
        FCmpPredicateAttr::ULE => ConditionCode::Le,
        FCmpPredicateAttr::UNO => ConditionCode::Vs,
        _ => return None,
    })
}

/// Lower an `llvm.fcmp` to a 0/1 GPR value.
fn lower_fcmp(
    ctx: &mut Context,
    block: Ptr<crate::ir::basic_block::BasicBlock>,
    values: &HashMap<Value, LoweredValue>,
    predicate: FCmpPredicateAttr,
    lhs: Value,
    rhs: Value,
    next_vreg: &mut usize,
) -> STAIRResult<LoweredValue> {
    if matches!(predicate, FCmpPredicateAttr::False) {
        return Ok(LoweredValue::Imm(0));
    }
    if matches!(predicate, FCmpPredicateAttr::True) {
        return Ok(LoweredValue::Imm(1));
    }
    let fp = fp_kind(ctx, lhs.get_type(ctx)).ok_or_else(|| {
        input_error_noloc!(Aarch64Err::UnsupportedType(format!(
            "fcmp on non-scalar-float type {}",
            pliron::printable::Printable::disp(&lhs.get_type(ctx), ctx)
        )))
    })?;
    let lhs_value = lookup_value(ctx, values, lhs)?;
    let rhs_value = lookup_value(ctx, values, rhs)?;
    let lhs = materialize_fp(ctx, block, lhs_value, fp, next_vreg, "fcmp lhs")?;
    let rhs = materialize_fp(ctx, block, rhs_value, fp, next_vreg, "fcmp rhs")?;
    let fcmp_opcode = match fp {
        FpKind::F64 => aarch64_ops::FcmpDOp::OPCODE,
        FpKind::F32 => aarch64_ops::FcmpSOp::OPCODE,
    };
    aarch64_ops::fcmp(ctx, fcmp_opcode, lhs, rhs).insert_at_back(block, ctx);
    if let Some(cond) = fcmp_condition_code(&predicate) {
        let dst = fresh_vreg(next_vreg);
        aarch64_ops::cset(ctx, dst, cond).insert_at_back(block, ctx);
        return Ok(LoweredValue::Reg(dst));
    }
    // `one` (lt-or-gt, ordered) and `ueq` (eq-or-unordered) need two tests
    // of the same nzcv, combined with `orr`.
    let (first, second) = match predicate {
        FCmpPredicateAttr::ONE => (ConditionCode::Mi, ConditionCode::Gt),
        FCmpPredicateAttr::UEQ => (ConditionCode::Eq, ConditionCode::Vs),
        _ => unreachable!("all other predicates map to a single condition code"),
    };
    let first_bit = fresh_vreg(next_vreg);
    aarch64_ops::cset(ctx, first_bit, first).insert_at_back(block, ctx);
    let second_bit = fresh_vreg(next_vreg);
    aarch64_ops::cset(ctx, second_bit, second).insert_at_back(block, ctx);
    let dst = fresh_vreg(next_vreg);
    aarch64_ops::binary(ctx, aarch64_ops::OrOp::OPCODE, dst, first_bit, second_bit)
        .insert_at_back(block, ctx);
    Ok(LoweredValue::Reg(dst))
}

/// Lower `llvm.sitofp` / `llvm.uitofp`. Sources up to 64 bits go through
/// `scvtf`/`ucvtf` on the 64-bit register (sub-64-bit sources are first
/// sign- or zero-extended to 64 bits, which preserves their value).
fn int_to_fp(
    ctx: &mut Context,
    block: Ptr<crate::ir::basic_block::BasicBlock>,
    values: &HashMap<Value, LoweredValue>,
    value: Value,
    result: Value,
    signed: bool,
    next_vreg: &mut usize,
) -> STAIRResult<LoweredValue> {
    let src_ty = value.get_type(ctx);
    if is_128_bit_integer(ctx, src_ty) {
        // Unreachable from the crabbit importer, which lowers 128-bit
        // int-to-float casts to the `__floattidf`-family libcalls before
        // isel; kept as a guard for hand-built LLVM-dialect input.
        return Err(input_error_noloc!(Aarch64Err::UnsupportedOp(
            "128-bit integer to floating-point conversion (lower to the __floattidf-family \
             libcalls before isel)"
                .to_string()
        )));
    }
    let fp = fp_kind(ctx, result.get_type(ctx)).ok_or_else(|| {
        input_error_noloc!(Aarch64Err::UnsupportedType(
            "int-to-float conversion to a non-scalar-float type".to_string()
        ))
    })?;
    let lowered = lookup_value(ctx, values, value)?;
    // The predicate-style signedness lives on the op, not the (signless)
    // type: re-type sub-64-bit sources so materialization extends correctly.
    let src_ty = if signed {
        signed_variant_ty(ctx, src_ty)
    } else {
        src_ty
    };
    let src = materialize_typed(ctx, block, lowered, src_ty, next_vreg, "int-to-float input")?;
    let opcode = match (signed, fp) {
        (true, FpKind::F64) => aarch64_ops::ScvtfDXOp::OPCODE,
        (true, FpKind::F32) => aarch64_ops::ScvtfSXOp::OPCODE,
        (false, FpKind::F64) => aarch64_ops::UcvtfDXOp::OPCODE,
        (false, FpKind::F32) => aarch64_ops::UcvtfSXOp::OPCODE,
    };
    let dst = fresh_fpr(next_vreg, fp);
    aarch64_ops::unary(ctx, opcode, dst, src).insert_at_back(block, ctx);
    Ok(LoweredValue::Reg(dst))
}

/// Lower a float-to-int conversion through `fcvtzs`/`fcvtzu`, which saturate
/// at the destination width and convert NaN to 0 — exactly Rust's `as`-cast
/// semantics (mir-lower routes those here as `llvm_fptosi_sat_*` /
/// `llvm_fptoui_sat_*` calls). Only 32- and 64-bit destinations have a
/// hardware form with the right saturation bounds.
fn fp_to_int(
    ctx: &mut Context,
    block: Ptr<crate::ir::basic_block::BasicBlock>,
    value: LoweredValue,
    src_ty: TypeHandle,
    result_ty: TypeHandle,
    signed: bool,
    next_vreg: &mut usize,
) -> STAIRResult<LoweredValue> {
    let fp = fp_kind(ctx, src_ty).ok_or_else(|| {
        input_error_noloc!(Aarch64Err::UnsupportedType(
            "float-to-int conversion from a non-scalar-float type".to_string()
        ))
    })?;
    let width = integer_width_and_signedness(ctx, result_ty)
        .map(|(width, _)| width)
        .ok_or_else(|| {
            input_error_noloc!(Aarch64Err::UnsupportedType(
                "float-to-int conversion to a non-integer type".to_string()
            ))
        })?;
    let use_64 = match width {
        64 => true,
        32 => false,
        // i128 destinations are unreachable from the crabbit importer, which
        // lowers 128-bit float-to-int casts to the `__fixdfti`-family
        // libcalls plus an explicit clamp before isel.
        other => {
            return Err(input_error_noloc!(Aarch64Err::UnsupportedOp(format!(
                "saturating float-to-int conversion to i{other} (only 32- and 64-bit \
                 destinations have a matching fcvtz form; lower i128 to the \
                 __fixdfti-family libcalls before isel)"
            ))));
        }
    };
    let src = materialize_fp(ctx, block, value, fp, next_vreg, "float-to-int input")?;
    let opcode = match (signed, use_64, fp) {
        (true, true, FpKind::F64) => aarch64_ops::FcvtzsXDOp::OPCODE,
        (true, false, FpKind::F64) => aarch64_ops::FcvtzsWDOp::OPCODE,
        (true, true, FpKind::F32) => aarch64_ops::FcvtzsXSOp::OPCODE,
        (true, false, FpKind::F32) => aarch64_ops::FcvtzsWSOp::OPCODE,
        (false, true, FpKind::F64) => aarch64_ops::FcvtzuXDOp::OPCODE,
        (false, false, FpKind::F64) => aarch64_ops::FcvtzuWDOp::OPCODE,
        (false, true, FpKind::F32) => aarch64_ops::FcvtzuXSOp::OPCODE,
        (false, false, FpKind::F32) => aarch64_ops::FcvtzuWSOp::OPCODE,
    };
    let dst = fresh_vreg(next_vreg);
    aarch64_ops::unary(ctx, opcode, dst, src).insert_at_back(block, ctx);
    Ok(LoweredValue::Reg(dst))
}

/// A float math intrinsic with a one-instruction AArch64 lowering, as the
/// `(d form, s form)` opcode pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FpMathIntrinsic {
    Unary(aarch64_ops::Aarch64Opcode, aarch64_ops::Aarch64Opcode),
    Binary(aarch64_ops::Aarch64Opcode, aarch64_ops::Aarch64Opcode),
}

/// The `llvm_<op>_f{32,64}` declarations the importer routes float math
/// through (mirroring LLVM's `llvm.<op>.f32` intrinsics): `sqrt`, `fabs`,
/// `floor`/`ceil`/`trunc`/`round`/`rint` and `minnum`/`maxnum`/`minimum`/
/// `maximum`. Transcendentals (`exp2` and friends) never come here: the
/// importer emits libm calls for those.
pub(super) fn fp_math_intrinsic(name: &str) -> Option<FpMathIntrinsic> {
    use FpMathIntrinsic::{Binary, Unary};
    use aarch64_ops::Aarch64Opcode as Opc;
    let op = name.strip_prefix("llvm_")?;
    let op = op
        .strip_suffix("_f32")
        .or_else(|| op.strip_suffix("_f64"))?;
    Some(match op {
        "sqrt" => Unary(Opc::FsqrtD, Opc::FsqrtS),
        "fabs" => Unary(Opc::FabsD, Opc::FabsS),
        "floor" => Unary(Opc::FrintmD, Opc::FrintmS),
        "ceil" => Unary(Opc::FrintpD, Opc::FrintpS),
        "trunc" => Unary(Opc::FrintzD, Opc::FrintzS),
        // Rust `round`: half away from zero.
        "round" => Unary(Opc::FrintaD, Opc::FrintaS),
        // Rust `round_ties_even`.
        "rint" => Unary(Opc::FrintnD, Opc::FrintnS),
        // IEEE minNum/maxNum (Rust `min`/`max`): a NaN operand loses.
        "minnum" => Binary(Opc::FminnmD, Opc::FminnmS),
        "maxnum" => Binary(Opc::FmaxnmD, Opc::FmaxnmS),
        // IEEE-2019 minimum/maximum (Rust `minimum`/`maximum`): NaN wins.
        "minimum" => Binary(Opc::FminD, Opc::FminS),
        "maximum" => Binary(Opc::FmaxD, Opc::FmaxS),
        _ => return None,
    })
}

/// The `llvm_fptosi_sat_iN_fM` / `llvm_fptoui_sat_iN_fM` intrinsic family
/// mir-lower declares for Rust's saturating float-to-int `as` casts. Parses
/// the destination width and signedness; the source float type comes from
/// the argument.
pub(super) fn saturating_fp_to_int_intrinsic(name: &str) -> Option<(bool, u32)> {
    let (signed, rest) = if let Some(rest) = name.strip_prefix("llvm_fptosi_sat_i") {
        (true, rest)
    } else if let Some(rest) = name.strip_prefix("llvm_fptoui_sat_i") {
        (false, rest)
    } else {
        return None;
    };
    let (width, float_suffix) = rest.split_once('_')?;
    if !matches!(float_suffix, "f16" | "f32" | "f64") {
        return None;
    }
    Some((signed, width.parse().ok()?))
}

/// The signed spelling of a sub-64-bit integer type, so materialization
/// sign-extends; wider and non-integer types pass through unchanged.
pub(super) fn signed_variant_ty(ctx: &mut Context, ty: TypeHandle) -> TypeHandle {
    let width = {
        let ty_ref = ty.deref(ctx);
        match ty_ref.downcast_ref::<crate::dialects::builtin::types::IntegerType>() {
            Some(int_ty) if int_ty.width() < 64 => int_ty.width(),
            _ => return ty,
        }
    };
    crate::dialects::builtin::types::IntegerType::get(
        ctx,
        width,
        crate::dialects::builtin::types::Signedness::Signed,
    )
    .into()
}

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
