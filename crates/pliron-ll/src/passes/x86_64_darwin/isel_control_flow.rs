use std::collections::HashMap;

use crate::{
    context::{Context, Ptr},
    dialects::x86_64::{ops as x86_64_ops, registers::Register},
    input_error_noloc,
    ir::{basic_block::BasicBlock, r#type::Typed, value::Value},
    result::STAIRResult,
};

use super::{
    error::X86_64DarwinErr,
    llvm_to_x86_64_isel::{LoweredValue, fresh_vreg, lookup_value, materialize_typed},
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
    x86_64_ops::jmp(ctx, target).insert_at_back(edge, ctx);
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
        return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedOp(
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
        let dst = match lookup_value(ctx, values, dest_arg)? {
            LoweredValue::Reg(reg) => reg,
            _ => {
                return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedOp(
                    "x86-64 block argument target must be a register".to_string()
                )));
            }
        };
        if matches!(src, LoweredValue::Undef) {
            // An undef incoming value (e.g. from mem2reg promoting a slot
            // that is not initialized on this path) is never read; give the
            // vreg a defined value so liveness stays consistent.
            pending.push((dst, None));
            continue;
        }
        let src = materialize_typed(
            ctx,
            insert_block,
            src,
            dest_arg.get_type(ctx),
            next_vreg,
            "block argument",
        )?;
        if src != dst {
            pending.push((dst, Some(src)));
        }
    }

    while !pending.is_empty() {
        let free = pending.iter().position(|(dst, _)| {
            !pending.iter().any(|(_, src)| *src == Some(*dst))
        });
        if let Some(free) = free {
            let (dst, src) = pending.remove(free);
            match src {
                Some(src) => x86_64_ops::mov(ctx, dst, src).insert_at_back(insert_block, ctx),
                None => x86_64_ops::mov_imm(ctx, dst, 0).insert_at_back(insert_block, ctx),
            };
        } else {
            // Every remaining destination is still read by another copy:
            // save one destination's current value and redirect its readers.
            let dst = pending[0].0;
            let scratch = fresh_vreg(next_vreg);
            x86_64_ops::mov(ctx, scratch, dst).insert_at_back(insert_block, ctx);
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
        input_error_noloc!(X86_64DarwinErr::UnsupportedOp(
            "branch target block was not lowered".to_string()
        ))
    })
}
