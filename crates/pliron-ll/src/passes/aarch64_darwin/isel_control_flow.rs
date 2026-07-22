use std::collections::HashMap;
use crate::r#type::TypeHandle;
use super::isel_memory_abi::adapt_value_to_type;

use crate::{
    context::{Context, Ptr},
    dialects::aarch64::{ops as aarch64_ops, registers::Register},
    input_error_noloc,
    ir::{basic_block::BasicBlock, r#type::Typed, value::Value},
    result::STAIRResult,
};

use super::{
    error::Aarch64DarwinErr,
    llvm_to_aarch64_isel::{
        LoweredValue, block_arg_registers, fresh_vreg, is_128_bit_integer, is_aggregate_ty,
        is_zero_sized_ty, lookup_value, materialize_pair, materialize_typed, struct_fields,
    },
};

/// Lowers CFG edges after instruction selection has assigned every LLVM block
/// a machine block and every block argument a virtual register.
pub(super) fn branch_edge_target(
    ctx: &mut Context,
    region: Ptr<crate::ir::region::Region>,
    block_map: &HashMap<Ptr<BasicBlock>, Ptr<BasicBlock>>,
    values: &HashMap<Value, LoweredValue>,
    dest: Ptr<BasicBlock>,
    args: &[Value],
    next_vreg: &mut usize,
    next_edge_block: &mut usize,
) -> STAIRResult<Ptr<BasicBlock>> {
    if args.is_empty() {
        return machine_block(block_map, dest);
    }

    let label = format!("edge{}", *next_edge_block);
    *next_edge_block += 1;
    let edge = BasicBlock::new(ctx, Some(label.try_into().unwrap()), vec![]);
    edge.insert_at_back(region, ctx);
    emit_block_arg_copies(ctx, edge, values, dest, args, next_vreg)?;
    let target = machine_block(block_map, dest)?;
    aarch64_ops::b(ctx, target).insert_at_back(edge, ctx);
    Ok(edge)
}

/// Materializes incoming SSA values into a destination block's virtual
/// registers. Conditional edges use a dedicated edge block so these copies do
/// not execute on the other edge.
pub(super) fn emit_block_arg_copies(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    values: &HashMap<Value, LoweredValue>,
    dest: Ptr<BasicBlock>,
    args: &[Value],
    next_vreg: &mut usize,
) -> STAIRResult<()> {
    let dest_args: Vec<_> = dest.deref(ctx).arguments().collect();
    if dest_args.len() != args.len() {
        return Err(input_error_noloc!(Aarch64DarwinErr::UnsupportedOp(
            format!(
                "branch operand count {} does not match target block argument count {}",
                args.len(),
                dest_args.len()
            )
        )));
    }

    // The copies are conceptually parallel: on a backedge, a source may be
    // another destination's current value (the classic phi swap). First
    // materialize every source, then emit the moves in dependency order,
    // breaking cycles through a scratch register.
    let mut pending: Vec<(Register, Option<Register>)> = Vec::new();
    for (arg, dest_arg) in args.iter().copied().zip(dest_args) {
        let src = lookup_value(ctx, values, arg)?;
        let mut dst_regs = Vec::new();
        block_arg_registers(&lookup_value(ctx, values, dest_arg)?, &mut dst_regs);
        let mut src_regs = Vec::new();
        flatten_incoming(
            ctx,
            insert_block,
            src,
            dest_arg.get_type(ctx),
            next_vreg,
            &mut src_regs,
        )?;
        if dst_regs.len() != src_regs.len() {
            return Err(input_error_noloc!(Aarch64DarwinErr::UnsupportedOp(format!(
                "block argument register count mismatch: {} incoming vs {} target",
                src_regs.len(),
                dst_regs.len()
            ))));
        }
        for (dst, src) in dst_regs.into_iter().zip(src_regs) {
            // An undef incoming leaf (e.g. from mem2reg promoting a slot
            // that is not initialized on this path) is never read; give the
            // vreg a defined value so liveness stays consistent.
            if src.as_ref() != Some(&dst) {
                pending.push((dst, src));
            }
        }
    }

    while !pending.is_empty() {
        let free = pending.iter().position(|(dst, _)| {
            !pending.iter().any(|(_, src)| *src == Some(*dst))
        });
        if let Some(free) = free {
            let (dst, src) = pending.remove(free);
            match src {
                Some(src) => aarch64_ops::mov(ctx, dst, src).insert_at_back(insert_block, ctx),
                None => aarch64_ops::mov_imm(ctx, dst, 0).insert_at_back(insert_block, ctx),
            };
        } else {
            // Every remaining destination is still read by another copy:
            // save one destination's current value and redirect its readers.
            let dst = pending[0].0;
            let scratch = fresh_vreg(next_vreg);
            aarch64_ops::mov(ctx, scratch, dst).insert_at_back(insert_block, ctx);
            for (_, src) in pending.iter_mut() {
                if *src == Some(dst) {
                    *src = Some(scratch);
                }
            }
        }
    }
    Ok(())
}

/// The machine block lowered for an LLVM block.
pub(super) fn machine_block(
    block_map: &HashMap<Ptr<BasicBlock>, Ptr<BasicBlock>>,
    target: Ptr<BasicBlock>,
) -> STAIRResult<Ptr<BasicBlock>> {
    block_map.get(&target).copied().ok_or_else(|| {
        input_error_noloc!(Aarch64DarwinErr::UnsupportedOp(
            "branch target block was not lowered".to_string()
        ))
    })
}

/// Materialize an incoming block-argument value as one register per scalar
/// leaf of `ty`, in the same leaf order as
/// [block_arg_registers](llvm_to_aarch64_isel::block_arg_registers). `None`
/// marks an undef leaf.
fn flatten_incoming(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    src: LoweredValue,
    ty: TypeHandle,
    next_vreg: &mut usize,
    out: &mut Vec<Option<Register>>,
) -> STAIRResult<()> {
    if is_zero_sized_ty(ctx, ty) {
        return Ok(());
    }
    // Reconcile packed-scalar and field-wise representations of the same
    // value before flattening.
    let src = adapt_value_to_type(ctx, src, ty)?;
    if is_128_bit_integer(ctx, ty) {
        if matches!(src, LoweredValue::Undef) {
            out.push(None);
            out.push(None);
            return Ok(());
        }
        let (lo, hi) = materialize_pair(ctx, insert_block, src, ty, next_vreg, "block argument")?;
        out.push(Some(lo));
        out.push(Some(hi));
        return Ok(());
    }
    if is_aggregate_ty(ctx, ty) {
        let fields = struct_fields(ctx, ty)?;
        let field_values: Vec<Option<LoweredValue>> = match src {
            LoweredValue::Undef => vec![None; fields.len()],
            LoweredValue::Aggregate(values) => {
                let mut values = values;
                values.resize(fields.len(), None);
                values
            }
            other => {
                return Err(input_error_noloc!(Aarch64DarwinErr::UnsupportedOp(format!(
                    "aggregate block argument fed by non-aggregate {other:?}"
                ))));
            }
        };
        for (field_ty, field_value) in fields.into_iter().zip(field_values) {
            flatten_incoming(
                ctx,
                insert_block,
                field_value.unwrap_or(LoweredValue::Undef),
                field_ty,
                next_vreg,
                out,
            )?;
        }
        return Ok(());
    }
    if matches!(src, LoweredValue::Undef) {
        out.push(None);
        return Ok(());
    }
    let reg = materialize_typed(ctx, insert_block, src, ty, next_vreg, "block argument")?;
    out.push(Some(reg));
    Ok(())
}
