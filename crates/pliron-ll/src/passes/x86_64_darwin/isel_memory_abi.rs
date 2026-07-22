use std::collections::HashMap;

use crate::{
    context::{Context, Ptr},
    dialects::{
        x86_64::{
            attributes::AbiLocation,
            ops as x86_64_ops,
            registers::{RAX, Register},
        },
        llvm::ops::GepIndex,
    },
    input_error_noloc,
    ir::{r#type::Typed, value::Value},
    result::STAIRResult,
};

use super::{error::X86_64DarwinErr, frontend::RESULT_GPRS, llvm_to_x86_64_isel::*};
use crate::r#type::TypeHandle;

/// Lowers addresses, memory access, aggregate ABI values, and stack layout.
pub(super) fn lower_gep(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    values: &HashMap<Value, LoweredValue>,
    base: Value,
    indices: &[GepIndex],
    source_elem_type: TypeHandle,
    next_vreg: &mut usize,
) -> STAIRResult<LoweredValue> {
    let base = lookup_value(ctx, values, base)?;
    let base = materialize_pointer(ctx, entry, base, next_vreg, "gep base")?;
    let mut current_ty = source_elem_type;
    let mut constant_offset = 0u64;
    let mut dynamic_offset = None::<Register>;

    for index in indices {
        let (index_const, index_reg) = match index {
            GepIndex::Constant(value) => (Some(*value as u64), None),
            GepIndex::Value(value) => {
                let reg = lookup_reg(ctx, entry, values, *value, next_vreg)?;
                (None, Some(reg))
            }
        };

        if let Some((field_offset, field_ty)) = struct_field_offset(ctx, current_ty, index_const)? {
            constant_offset += field_offset;
            current_ty = field_ty;
            continue;
        }

        let (element_ty, element_size) = indexed_element(ctx, current_ty)?;
        if let Some(value) = index_const {
            constant_offset += value * element_size;
        } else if let Some(reg) = index_reg {
            let scaled = if element_size == 1 {
                reg
            } else {
                let scale = fresh_vreg(next_vreg);
                materialize_u64_immediate(ctx, entry, scale, element_size);
                let scaled = fresh_vreg(next_vreg);
                x86_64_ops::binary(ctx, x86_64_ops::MulOp::OPCODE, scaled.clone(), reg, scale)
                    .insert_at_back(entry, ctx);
                scaled
            };
            dynamic_offset = Some(if let Some(acc) = dynamic_offset {
                let dst = fresh_vreg(next_vreg);
                x86_64_ops::binary(ctx, x86_64_ops::AddOp::OPCODE, dst.clone(), acc, scaled)
                    .insert_at_back(entry, ctx);
                dst
            } else {
                scaled
            });
        }
        current_ty = element_ty;
    }

    let base = if let Some(offset) = dynamic_offset {
        let dst = fresh_vreg(next_vreg);
        x86_64_ops::binary(ctx, x86_64_ops::AddOp::OPCODE, dst.clone(), base, offset)
            .insert_at_back(entry, ctx);
        dst
    } else {
        base
    };

    Ok(LoweredValue::Address {
        base,
        offset: constant_offset,
    })
}

pub(super) fn load_memory(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    addr: LoweredValue,
    ty: TypeHandle,
    next_vreg: &mut usize,
) -> STAIRResult<LoweredValue> {
    if stack_size_of(ctx, ty)? == 0 {
        return Ok(LoweredValue::Undef);
    }
    match addr {
        LoweredValue::StackAddr(slot) => load_stack_value(ctx, entry, slot.offset, ty, next_vreg),
        other => {
            let (base, offset) = materialize_address(ctx, entry, other, next_vreg, "load address")?;
            load_register_address_value(ctx, entry, base, offset, ty, next_vreg)
        }
    }
}

// Calls and ABI result values ------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ResultLocation {
    Void,
    ScalarRax,
    ScalarRaxRdx,
    DirectGprs(usize),
    IndirectRdi,
}

pub(super) fn result_location_for_type(
    ctx: &Context,
    ty: TypeHandle,
) -> STAIRResult<ResultLocation> {
    if stack_size_of(ctx, ty)? == 0 {
        return Ok(ResultLocation::Void);
    }
    if is_128_bit_integer(ctx, ty) {
        return Ok(ResultLocation::ScalarRaxRdx);
    }
    if is_aggregate_ty(ctx, ty) {
        let size = stack_size_of(ctx, ty)?;
        if size <= 8 {
            Ok(ResultLocation::DirectGprs(1))
        } else if size <= 16 {
            Ok(ResultLocation::DirectGprs(2))
        } else {
            Ok(ResultLocation::IndirectRdi)
        }
    } else {
        Ok(ResultLocation::ScalarRax)
    }
}

pub(super) fn load_gpr_aggregate_result(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    ty: TypeHandle,
    count: usize,
    next_vreg: &mut usize,
) -> STAIRResult<LoweredValue> {
    let mut next_reg = 0usize;
    let value = load_direct_aggregate_from_gprs(ctx, entry, ty, count, &mut next_reg, next_vreg)?;
    Ok(value)
}

pub(super) fn emit_return_value(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    values: &HashMap<Value, LoweredValue>,
    value: Value,
    result: AbiLocation,
    sret_result_slot: Option<StackSlot>,
    next_vreg: &mut usize,
) -> STAIRResult<()> {
    match result {
        AbiLocation::Void => Ok(()),
        AbiLocation::Gpr(reg) => {
            let lowered = lookup_value(ctx, values, value)?;
            let src = if is_aggregate_ty(ctx, value.get_type(ctx)) {
                pack_aggregate_to_reg(ctx, entry, lowered, value.get_type(ctx), 0, next_vreg)?
            } else {
                materialize_typed(
                    ctx,
                    entry,
                    lowered,
                    value.get_type(ctx),
                    next_vreg,
                    "return value",
                )?
            };
            x86_64_ops::mov(ctx, reg, src).insert_at_back(entry, ctx);
            Ok(())
        }
        AbiLocation::GprPair(lo_reg, hi_reg) => {
            let lowered = lookup_value(ctx, values, value)?;
            if is_128_bit_integer(ctx, value.get_type(ctx)) {
                let (lo, hi) = materialize_pair(
                    ctx,
                    entry,
                    lowered,
                    value.get_type(ctx),
                    next_vreg,
                    "return value",
                )?;
                x86_64_ops::mov(ctx, lo_reg, lo).insert_at_back(entry, ctx);
                x86_64_ops::mov(ctx, hi_reg, hi).insert_at_back(entry, ctx);
                Ok(())
            } else {
                emit_direct_aggregate_return(ctx, entry, lowered, value.get_type(ctx), next_vreg)
            }
        }
        AbiLocation::IndirectResult => {
            let lowered = lookup_value(ctx, values, value)?;
            let sret_result_slot = sret_result_slot.ok_or_else(|| {
                input_error_noloc!(X86_64DarwinErr::UnsupportedOp(
                    "missing saved sret result pointer".to_string()
                ))
            })?;
            let dst = fresh_vreg(next_vreg);
            x86_64_ops::ldr_sp_offset(ctx, dst.clone(), sret_result_slot.offset)
                .insert_at_back(entry, ctx);
            store_register_address_value(
                ctx,
                entry,
                dst,
                0,
                lowered,
                value.get_type(ctx),
                next_vreg,
            )?;
            // The System V AMD64 ABI returns the sret pointer in rax.
            x86_64_ops::mov(ctx, RAX, dst).insert_at_back(entry, ctx);
            Ok(())
        }
        AbiLocation::Stack(_) => Err(input_error_noloc!(X86_64DarwinErr::UnsupportedOp(
            "stack ABI location on a function result".to_string()
        ))),
    }
}

fn emit_direct_aggregate_return(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    value: LoweredValue,
    ty: TypeHandle,
    next_vreg: &mut usize,
) -> STAIRResult<()> {
    let mut next_reg = 0usize;
    emit_direct_aggregate_to_gprs(ctx, entry, value, ty, &mut next_reg, next_vreg)?;
    Ok(())
}

fn load_direct_aggregate_from_gprs(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    ty: TypeHandle,
    count: usize,
    next_reg: &mut usize,
    next_vreg: &mut usize,
) -> STAIRResult<LoweredValue> {
    if !is_aggregate_ty(ctx, ty) {
        if is_128_bit_integer(ctx, ty) {
            if *next_reg + 1 >= count {
                return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedType(
                    "128-bit aggregate return field exceeds Darwin ABI location".to_string(),
                )));
            }
            let lo = fresh_vreg(next_vreg);
            x86_64_ops::mov(ctx, lo, RESULT_GPRS[*next_reg]).insert_at_back(entry, ctx);
            *next_reg += 1;
            let hi = fresh_vreg(next_vreg);
            x86_64_ops::mov(ctx, hi, RESULT_GPRS[*next_reg]).insert_at_back(entry, ctx);
            *next_reg += 1;
            return Ok(LoweredValue::RegPair(lo, hi));
        }
        if *next_reg >= count {
            return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedType(
                "aggregate return uses more GPR fields than Darwin ABI location".to_string()
            )));
        }
        let dst = fresh_vreg(next_vreg);
        x86_64_ops::mov(ctx, dst, RESULT_GPRS[*next_reg]).insert_at_back(entry, ctx);
        *next_reg += 1;
        return Ok(LoweredValue::Reg(dst));
    }

    if count == 1 && stack_size_of(ctx, ty)? <= 8 {
        let base = fresh_vreg(next_vreg);
        x86_64_ops::mov(ctx, base, RESULT_GPRS[*next_reg]).insert_at_back(entry, ctx);
        *next_reg += 1;
        return unpack_aggregate_from_reg(ctx, entry, ty, base, 0, next_vreg);
    }

    if count == 2 && stack_size_of(ctx, ty)? <= 16 {
        let lo = fresh_vreg(next_vreg);
        x86_64_ops::mov(ctx, lo, RESULT_GPRS[*next_reg]).insert_at_back(entry, ctx);
        *next_reg += 1;
        let hi = fresh_vreg(next_vreg);
        x86_64_ops::mov(ctx, hi, RESULT_GPRS[*next_reg]).insert_at_back(entry, ctx);
        *next_reg += 1;
        return unpack_aggregate_from_reg_pair(ctx, entry, ty, lo, hi, 0, next_vreg);
    }

    let fields = struct_fields(ctx, ty)?;
    let mut values = Vec::with_capacity(fields.len());
    for field_ty in fields {
        values.push(Some(load_direct_aggregate_from_gprs(
            ctx, entry, field_ty, count, next_reg, next_vreg,
        )?));
    }
    Ok(LoweredValue::Aggregate(values))
}

/// Unpack an aggregate laid out across two 8-byte registers by memory layout,
/// mirroring the Darwin x86-64 ABI for 9..=16 byte aggregates.
fn unpack_aggregate_from_reg_pair(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    ty: TypeHandle,
    lo: Register,
    hi: Register,
    byte_offset: u64,
    next_vreg: &mut usize,
) -> STAIRResult<LoweredValue> {
    if is_stack_scalar_ty(ctx, ty) {
        if is_128_bit_integer(ctx, ty) {
            if byte_offset != 0 {
                return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedType(
                    "128-bit field at nonzero offset in register-pair aggregate".to_string()
                )));
            }
            return Ok(LoweredValue::RegPair(lo, hi));
        }
        let size = scalar_size_of(ctx, ty)?;
        if byte_offset < 8 && byte_offset + size > 8 {
            return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedType(
                "scalar field straddles register boundary in aggregate".to_string()
            )));
        }
        return if byte_offset < 8 {
            unpack_aggregate_from_reg(ctx, entry, ty, lo, byte_offset, next_vreg)
        } else {
            unpack_aggregate_from_reg(ctx, entry, ty, hi, byte_offset - 8, next_vreg)
        };
    }

    let layout = aggregate_field_layout(ctx, ty)?;
    let mut values = Vec::with_capacity(layout.len());
    for (field_offset, field_ty) in layout {
        values.push(Some(unpack_aggregate_from_reg_pair(
            ctx,
            entry,
            field_ty,
            lo,
            hi,
            byte_offset + field_offset,
            next_vreg,
        )?));
    }
    Ok(LoweredValue::Aggregate(values))
}

fn emit_direct_aggregate_to_gprs(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    value: LoweredValue,
    ty: TypeHandle,
    next_reg: &mut usize,
    next_vreg: &mut usize,
) -> STAIRResult<()> {
    if !is_aggregate_ty(ctx, ty) {
        if is_128_bit_integer(ctx, ty) {
            if *next_reg + 1 >= 2 {
                return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedType(
                    "direct aggregate return has more than two GPR fields".to_string(),
                )));
            }
            let (lo, hi) =
                materialize_pair(ctx, entry, value, ty, next_vreg, "aggregate return field")?;
            x86_64_ops::mov(ctx, RESULT_GPRS[*next_reg], lo).insert_at_back(entry, ctx);
            *next_reg += 1;
            x86_64_ops::mov(ctx, RESULT_GPRS[*next_reg], hi).insert_at_back(entry, ctx);
            *next_reg += 1;
            return Ok(());
        }
        if *next_reg >= 2 {
            return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedType(
                "direct aggregate return has more than two GPR fields".to_string()
            )));
        }
        let src = materialize(ctx, entry, value, next_vreg, "aggregate return field")?;
        x86_64_ops::mov(ctx, RESULT_GPRS[*next_reg], src).insert_at_back(entry, ctx);
        *next_reg += 1;
        return Ok(());
    }

    if stack_size_of(ctx, ty)? <= 8 {
        if *next_reg >= 2 {
            return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedType(
                "direct aggregate return has more than two GPR fields".to_string()
            )));
        }
        let packed = pack_aggregate_to_reg(ctx, entry, value, ty, 0, next_vreg)?;
        x86_64_ops::mov(ctx, RESULT_GPRS[*next_reg], packed).insert_at_back(entry, ctx);
        *next_reg += 1;
        return Ok(());
    }

    if stack_size_of(ctx, ty)? <= 16 {
        if *next_reg != 0 {
            return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedType(
                "direct aggregate return has more than two GPR fields".to_string()
            )));
        }
        let mut acc: (Option<Register>, Option<Register>) = (None, None);
        pack_aggregate_to_reg_pair(ctx, entry, value, ty, 0, &mut acc, next_vreg)?;
        for (reg, half) in [acc.0, acc.1].into_iter().enumerate() {
            let src = match half {
                Some(src) => src,
                None => {
                    let zero = fresh_vreg(next_vreg);
                    materialize_u64_immediate(ctx, entry, zero, 0);
                    zero
                }
            };
            x86_64_ops::mov(ctx, RESULT_GPRS[*next_reg + reg], src).insert_at_back(entry, ctx);
        }
        *next_reg += 2;
        return Ok(());
    }

    let fields = struct_fields(ctx, ty)?;
    let LoweredValue::Aggregate(values) = value else {
        return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedOp(
            "direct aggregate return requires aggregate value".to_string()
        )));
    };
    for (index, field_ty) in fields.into_iter().enumerate() {
        let field = values
            .get(index)
            .and_then(|value| value.clone())
            .ok_or_else(|| {
                input_error_noloc!(X86_64DarwinErr::UndefinedValue(
                    "return from unset aggregate field".to_string()
                ))
            })?;
        emit_direct_aggregate_to_gprs(ctx, entry, field, field_ty, next_reg, next_vreg)?;
    }
    Ok(())
}

/// Pack an aggregate into two 8-byte register halves by memory layout,
/// mirroring the Darwin x86-64 ABI for 9..=16 byte aggregates. Missing
/// (undef) fields leave their half untouched.
fn pack_aggregate_to_reg_pair(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    value: LoweredValue,
    ty: TypeHandle,
    byte_offset: u64,
    acc: &mut (Option<Register>, Option<Register>),
    next_vreg: &mut usize,
) -> STAIRResult<()> {
    if stack_size_of(ctx, ty)? == 0 || matches!(value, LoweredValue::Undef) {
        return Ok(());
    }
    if is_stack_scalar_ty(ctx, ty) {
        if is_128_bit_integer(ctx, ty) {
            if byte_offset != 0 {
                return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedType(
                    "128-bit field at nonzero offset in register-pair aggregate".to_string()
                )));
            }
            let (lo, hi) =
                materialize_pair(ctx, entry, value, ty, next_vreg, "aggregate return field")?;
            merge_packed_half(ctx, entry, &mut acc.0, lo, next_vreg);
            merge_packed_half(ctx, entry, &mut acc.1, hi, next_vreg);
            return Ok(());
        }
        let size = scalar_size_of(ctx, ty)?;
        if byte_offset < 8 && byte_offset + size > 8 {
            return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedType(
                "scalar field straddles register boundary in aggregate".to_string()
            )));
        }
        let (half, rel_offset) = if byte_offset < 8 {
            (&mut acc.0, byte_offset)
        } else {
            (&mut acc.1, byte_offset - 8)
        };
        let packed = pack_aggregate_to_reg(ctx, entry, value, ty, rel_offset, next_vreg)?;
        merge_packed_half(ctx, entry, half, packed, next_vreg);
        return Ok(());
    }

    let LoweredValue::Aggregate(values) = value else {
        return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedOp(
            "packed aggregate return requires aggregate value".to_string()
        )));
    };
    let layout = aggregate_field_layout(ctx, ty)?;
    for (index, (field_offset, field_ty)) in layout.into_iter().enumerate() {
        if let Some(field) = values.get(index).and_then(|field| field.clone()) {
            pack_aggregate_to_reg_pair(
                ctx,
                entry,
                field,
                field_ty,
                byte_offset + field_offset,
                acc,
                next_vreg,
            )?;
        }
    }
    Ok(())
}

fn merge_packed_half(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    half: &mut Option<Register>,
    value: Register,
    next_vreg: &mut usize,
) {
    *half = Some(match half.take() {
        Some(existing) => {
            let joined = fresh_vreg(next_vreg);
            x86_64_ops::binary(
                ctx,
                x86_64_ops::OrOp::OPCODE,
                joined.clone(),
                existing,
                value,
            )
            .insert_at_back(entry, ctx);
            joined
        }
        None => value,
    });
}

fn unpack_aggregate_from_reg(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    ty: TypeHandle,
    base: Register,
    byte_offset: u64,
    next_vreg: &mut usize,
) -> STAIRResult<LoweredValue> {
    if is_stack_scalar_ty(ctx, ty) {
        let size = scalar_size_of(ctx, ty)?;
        if size == 8 && byte_offset == 0 {
            return Ok(LoweredValue::Reg(base));
        }
        let mut value = base;
        if byte_offset > 0 {
            let shift = fresh_vreg(next_vreg);
            materialize_u64_immediate(ctx, entry, shift, byte_offset * 8);
            let shifted = fresh_vreg(next_vreg);
            x86_64_ops::binary(
                ctx,
                x86_64_ops::LsrOp::OPCODE,
                shifted.clone(),
                value,
                shift,
            )
            .insert_at_back(entry, ctx);
            value = shifted;
        }
        if size < 8 {
            let mask = fresh_vreg(next_vreg);
            materialize_u64_immediate(ctx, entry, mask, (1u64 << (size * 8)) - 1);
            let masked = fresh_vreg(next_vreg);
            x86_64_ops::binary(ctx, x86_64_ops::AndOp::OPCODE, masked.clone(), value, mask)
                .insert_at_back(entry, ctx);
            value = masked;
        }
        return Ok(LoweredValue::Reg(value));
    }

    let layout = aggregate_field_layout(ctx, ty)?;
    let mut values = Vec::with_capacity(layout.len());
    for (field_offset, field_ty) in layout {
        values.push(Some(unpack_aggregate_from_reg(
            ctx,
            entry,
            field_ty,
            base,
            byte_offset + field_offset,
            next_vreg,
        )?));
    }
    Ok(LoweredValue::Aggregate(values))
}

fn pack_aggregate_to_reg(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    value: LoweredValue,
    ty: TypeHandle,
    byte_offset: u64,
    next_vreg: &mut usize,
) -> STAIRResult<Register> {
    if is_stack_scalar_ty(ctx, ty) {
        if matches!(value, LoweredValue::Undef) {
            let zero = fresh_vreg(next_vreg);
            materialize_u64_immediate(ctx, entry, zero, 0);
            return Ok(zero);
        }
        let mut reg = materialize(ctx, entry, value, next_vreg, "packed aggregate field")?;
        if scalar_size_of(ctx, ty)? < 8 {
            let mask = fresh_vreg(next_vreg);
            let bits = scalar_size_of(ctx, ty)? * 8;
            materialize_u64_immediate(ctx, entry, mask, (1u64 << bits) - 1);
            let masked = fresh_vreg(next_vreg);
            x86_64_ops::binary(ctx, x86_64_ops::AndOp::OPCODE, masked.clone(), reg, mask)
                .insert_at_back(entry, ctx);
            reg = masked;
        }
        if byte_offset > 0 {
            let shift = fresh_vreg(next_vreg);
            materialize_u64_immediate(ctx, entry, shift, byte_offset * 8);
            let shifted = fresh_vreg(next_vreg);
            x86_64_ops::binary(ctx, x86_64_ops::ShlOp::OPCODE, shifted.clone(), reg, shift)
                .insert_at_back(entry, ctx);
            reg = shifted;
        }
        return Ok(reg);
    }

    let layout = aggregate_field_layout(ctx, ty)?;
    let values = match value {
        LoweredValue::Aggregate(values) => values,
        LoweredValue::Undef => vec![None; layout.len()],
        _ => {
            return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedOp(
                "packed aggregate return requires aggregate value".to_string()
            )));
        }
    };
    let mut acc = None::<Register>;
    for (index, (field_offset, field_ty)) in layout.into_iter().enumerate() {
        if let Some(field) = values.get(index).and_then(|field| field.clone()) {
            let field_reg = pack_aggregate_to_reg(
                ctx,
                entry,
                field,
                field_ty,
                byte_offset + field_offset,
                next_vreg,
            )?;
            acc = Some(if let Some(acc) = acc {
                let joined = fresh_vreg(next_vreg);
                x86_64_ops::binary(
                    ctx,
                    x86_64_ops::OrOp::OPCODE,
                    joined.clone(),
                    acc,
                    field_reg,
                )
                .insert_at_back(entry, ctx);
                joined
            } else {
                field_reg
            });
        }
    }
    if let Some(acc) = acc {
        Ok(acc)
    } else {
        let zero = fresh_vreg(next_vreg);
        materialize_u64_immediate(ctx, entry, zero, 0);
        Ok(zero)
    }
}

/// Re-shape a lowered value to the field structure of `ty`.
///
/// `llvm.bitcast` (from `mir.cast` between layout-compatible types) can
/// change the nominal struct shape without changing the bytes: wrapping a
/// value in a single-field struct, unwrapping it, or dropping fat-pointer
/// metadata. The lowered value must follow the target type's shape, or
/// later loads/stores misinterpret registers as addresses.
pub(super) fn adapt_value_to_type(
    ctx: &Context,
    value: LoweredValue,
    ty: TypeHandle,
) -> STAIRResult<LoweredValue> {
    if matches!(value, LoweredValue::Undef) {
        return Ok(LoweredValue::Undef);
    }
    if is_stack_scalar_ty(ctx, ty) {
        return match value {
            LoweredValue::Aggregate(fields) => {
                let inner = fields
                    .into_iter()
                    .flatten()
                    .next()
                    .unwrap_or(LoweredValue::Undef);
                adapt_value_to_type(ctx, inner, ty)
            }
            other => Ok(other),
        };
    }
    if !is_aggregate_ty(ctx, ty) {
        return Ok(value);
    }
    let layout = aggregate_field_layout(ctx, ty)?;
    match value {
        LoweredValue::Aggregate(fields) if fields.len() == layout.len() => {
            let mut adapted = Vec::with_capacity(fields.len());
            for (field, (_, field_ty)) in fields.into_iter().zip(layout) {
                adapted.push(match field {
                    Some(field) => Some(adapt_value_to_type(ctx, field, field_ty)?),
                    None => None,
                });
            }
            Ok(LoweredValue::Aggregate(adapted))
        }
        LoweredValue::Aggregate(fields) if fields.len() == 1 => {
            let inner = fields
                .into_iter()
                .next()
                .flatten()
                .unwrap_or(LoweredValue::Undef);
            adapt_value_to_type(ctx, inner, ty)
        }
        other => {
            // Wrap a scalar-like (or mismatched aggregate) value into the
            // target shape; fields with no source data (e.g. dropped
            // fat-pointer metadata) become undef.
            let mut adapted = Vec::with_capacity(layout.len());
            let mut remaining = Some(other);
            for (_, field_ty) in layout {
                adapted.push(match remaining.take() {
                    Some(value) => Some(adapt_value_to_type(ctx, value, field_ty)?),
                    None => Some(LoweredValue::Undef),
                });
            }
            Ok(LoweredValue::Aggregate(adapted))
        }
    }
}

pub(super) fn load_stack_value(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    offset: u64,
    ty: TypeHandle,
    next_vreg: &mut usize,
) -> STAIRResult<LoweredValue> {
    if stack_size_of(ctx, ty)? == 0 {
        return Ok(LoweredValue::Undef);
    }
    if is_stack_scalar_ty(ctx, ty) {
        if is_128_bit_integer(ctx, ty) {
            let lo = fresh_vreg(next_vreg);
            x86_64_ops::ldr_sp_offset(ctx, lo.clone(), offset).insert_at_back(entry, ctx);
            let hi = fresh_vreg(next_vreg);
            x86_64_ops::ldr_sp_offset(ctx, hi.clone(), offset + 8).insert_at_back(entry, ctx);
            return Ok(LoweredValue::RegPair(lo, hi));
        }
        let dst = fresh_vreg(next_vreg);
        x86_64_ops::ldr_sp_offset_sized(ctx, load_sp_opcode(ctx, ty)?, dst.clone(), offset)
            .insert_at_back(entry, ctx);
        let dst = normalize_integer_reg(ctx, entry, dst, ty, next_vreg)?;
        return Ok(LoweredValue::Reg(dst));
    }

    let layout = aggregate_field_layout(ctx, ty)?;
    let mut values = Vec::with_capacity(layout.len());
    for (field_offset, field_ty) in layout {
        values.push(Some(load_stack_value(
            ctx,
            entry,
            offset + field_offset,
            field_ty,
            next_vreg,
        )?));
    }
    Ok(LoweredValue::Aggregate(values))
}

pub(super) fn store_memory(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    addr: LoweredValue,
    value: LoweredValue,
    ty: TypeHandle,
    next_vreg: &mut usize,
) -> STAIRResult<()> {
    if stack_size_of(ctx, ty)? == 0 {
        return Ok(());
    }
    if matches!(value, LoweredValue::Undef) {
        return Ok(());
    }
    match addr {
        LoweredValue::StackAddr(slot) => {
            store_stack_value(ctx, entry, slot.offset, value, ty, next_vreg)
        }
        other => {
            let (base, offset) =
                materialize_address(ctx, entry, other, next_vreg, "store address")?;
            store_register_address_value(ctx, entry, base, offset, value, ty, next_vreg)
        }
    }
}

fn store_stack_value(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    offset: u64,
    value: LoweredValue,
    ty: TypeHandle,
    next_vreg: &mut usize,
) -> STAIRResult<()> {
    if stack_size_of(ctx, ty)? == 0 {
        return Ok(());
    }
    if matches!(value, LoweredValue::Undef) {
        return Ok(());
    }
    if is_stack_scalar_ty(ctx, ty) {
        if matches!(value, LoweredValue::Aggregate(_)) {
            return store_flattened_aggregate_chunks(ctx, entry, None, offset, value, next_vreg);
        }
        if is_128_bit_integer(ctx, ty) {
            let (lo, hi) = materialize_pair(ctx, entry, value, ty, next_vreg, "store value")?;
            x86_64_ops::str_sp_offset(ctx, lo, offset).insert_at_back(entry, ctx);
            x86_64_ops::str_sp_offset(ctx, hi, offset + 8).insert_at_back(entry, ctx);
            return Ok(());
        }
        let src = materialize(ctx, entry, value, next_vreg, "store value")?;
        x86_64_ops::str_sp_offset_sized(ctx, store_sp_opcode(ctx, ty)?, src, offset)
            .insert_at_back(entry, ctx);
        return Ok(());
    }

    let value = load_aggregate_copy_source(ctx, entry, value, ty, next_vreg)?;
    if !matches!(value, LoweredValue::Aggregate(_)) && stack_size_of(ctx, ty)? <= 8 {
        let word_ty = word_ty(ctx);
        return store_stack_value(ctx, entry, offset, value, word_ty, next_vreg);
    }
    let LoweredValue::Aggregate(values) = value else {
        return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedOp(
            format!(
                "aggregate store requires aggregate value, got {value:?} for type {} size {}",
                ty.deref(ctx).disp(ctx),
                stack_size_of(ctx, ty)?
            )
        )));
    };
    let layout = aggregate_field_layout(ctx, ty)?;
    for (index, (field_offset, field_ty)) in layout.into_iter().enumerate() {
        if let Some(field) = values.get(index).and_then(|value| value.clone()) {
            store_stack_value(
                ctx,
                entry,
                offset + field_offset,
                field,
                field_ty,
                next_vreg,
            )?;
        }
    }
    Ok(())
}

fn materialize_address(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    addr: LoweredValue,
    next_vreg: &mut usize,
    context: &str,
) -> STAIRResult<(Register, u64)> {
    match addr {
        LoweredValue::Reg(reg) => Ok((reg, 0)),
        LoweredValue::Address { base, offset } => Ok((base, offset)),
        other => Ok((
            materialize_pointer(ctx, entry, other, next_vreg, context)?,
            0,
        )),
    }
}

fn load_register_address_value(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    base: Register,
    offset: u64,
    ty: TypeHandle,
    next_vreg: &mut usize,
) -> STAIRResult<LoweredValue> {
    if stack_size_of(ctx, ty)? == 0 {
        return Ok(LoweredValue::Undef);
    }
    if is_stack_scalar_ty(ctx, ty) {
        if is_128_bit_integer(ctx, ty) {
            let lo = fresh_vreg(next_vreg);
            x86_64_ops::ldr_reg_offset_sized(
                ctx,
                x86_64_ops::LdrRegOffsetOp::OPCODE,
                lo.clone(),
                base,
                offset,
            )
            .insert_at_back(entry, ctx);
            let hi = fresh_vreg(next_vreg);
            x86_64_ops::ldr_reg_offset_sized(
                ctx,
                x86_64_ops::LdrRegOffsetOp::OPCODE,
                hi.clone(),
                base,
                offset + 8,
            )
            .insert_at_back(entry, ctx);
            return Ok(LoweredValue::RegPair(lo, hi));
        }
        let dst = fresh_vreg(next_vreg);
        x86_64_ops::ldr_reg_offset_sized(
            ctx,
            load_reg_opcode(ctx, ty)?,
            dst.clone(),
            base,
            offset,
        )
        .insert_at_back(entry, ctx);
        let dst = normalize_integer_reg(ctx, entry, dst, ty, next_vreg)?;
        return Ok(LoweredValue::Reg(dst));
    }

    let layout = aggregate_field_layout(ctx, ty)?;
    let mut values = Vec::with_capacity(layout.len());
    for (field_offset, field_ty) in layout {
        values.push(Some(load_register_address_value(
            ctx,
            entry,
            base,
            offset + field_offset,
            field_ty,
            next_vreg,
        )?));
    }
    Ok(LoweredValue::Aggregate(values))
}

fn store_register_address_value(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    base: Register,
    offset: u64,
    value: LoweredValue,
    ty: TypeHandle,
    next_vreg: &mut usize,
) -> STAIRResult<()> {
    if stack_size_of(ctx, ty)? == 0 {
        return Ok(());
    }
    if matches!(value, LoweredValue::Undef) {
        return Ok(());
    }
    if is_stack_scalar_ty(ctx, ty) {
        if matches!(value, LoweredValue::Aggregate(_)) {
            return store_flattened_aggregate_chunks(
                ctx,
                entry,
                Some(base),
                offset,
                value,
                next_vreg,
            );
        }
        if is_128_bit_integer(ctx, ty) {
            let (lo, hi) = materialize_pair(ctx, entry, value, ty, next_vreg, "store value")?;
            x86_64_ops::str_reg_offset_sized(
                ctx,
                x86_64_ops::StrRegOffsetOp::OPCODE,
                lo,
                base,
                offset,
            )
            .insert_at_back(entry, ctx);
            x86_64_ops::str_reg_offset_sized(
                ctx,
                x86_64_ops::StrRegOffsetOp::OPCODE,
                hi,
                base,
                offset + 8,
            )
            .insert_at_back(entry, ctx);
            return Ok(());
        }
        let src = materialize(ctx, entry, value, next_vreg, "store value")?;
        x86_64_ops::str_reg_offset_sized(ctx, store_reg_opcode(ctx, ty)?, src, base, offset)
            .insert_at_back(entry, ctx);
        return Ok(());
    }

    let value = load_aggregate_copy_source(ctx, entry, value, ty, next_vreg)?;
    if !matches!(value, LoweredValue::Aggregate(_)) && stack_size_of(ctx, ty)? <= 8 {
        let word_ty = word_ty(ctx);
        return store_register_address_value(ctx, entry, base, offset, value, word_ty, next_vreg);
    }
    let LoweredValue::Aggregate(values) = value else {
        return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedOp(
            format!(
                "aggregate store requires aggregate value, got {value:?} for type {} size {}",
                ty.deref(ctx).disp(ctx),
                stack_size_of(ctx, ty)?
            )
        )));
    };
    let layout = aggregate_field_layout(ctx, ty)?;
    for (index, (field_offset, field_ty)) in layout.into_iter().enumerate() {
        if let Some(field) = values.get(index).and_then(|value| value.clone()) {
            store_register_address_value(
                ctx,
                entry,
                base,
                offset + field_offset,
                field,
                field_ty,
                next_vreg,
            )?;
        }
    }
    Ok(())
}

fn load_aggregate_copy_source(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    value: LoweredValue,
    ty: TypeHandle,
    next_vreg: &mut usize,
) -> STAIRResult<LoweredValue> {
    match value {
        LoweredValue::Aggregate(_) | LoweredValue::Undef => Ok(value),
        LoweredValue::StackAddr(slot) => load_stack_value(ctx, entry, slot.offset, ty, next_vreg),
        LoweredValue::Address { base, offset } => {
            load_register_address_value(ctx, entry, base, offset, ty, next_vreg)
        }
        LoweredValue::Reg(base) if stack_size_of(ctx, ty)? > 8 => {
            load_register_address_value(ctx, entry, base, 0, ty, next_vreg)
        }
        other => Ok(other),
    }
}

fn store_flattened_aggregate_chunks(
    ctx: &mut Context,
    entry: Ptr<crate::ir::basic_block::BasicBlock>,
    base: Option<Register>,
    offset: u64,
    value: LoweredValue,
    next_vreg: &mut usize,
) -> STAIRResult<()> {
    let chunks = flatten_aggregate_value(value)?;
    let word_ty = word_ty(ctx);
    for (index, chunk) in chunks.into_iter().enumerate() {
        let chunk_offset = offset + (index as u64 * 8);
        if let Some(base) = base {
            store_register_address_value(
                ctx,
                entry,
                base,
                chunk_offset,
                chunk,
                word_ty,
                next_vreg,
            )?;
        } else {
            store_stack_value(ctx, entry, chunk_offset, chunk, word_ty, next_vreg)?;
        }
    }
    Ok(())
}

// Aggregate and stack layout -------------------------------------------------

pub(super) fn word_ty(ctx: &mut Context) -> TypeHandle {
    crate::dialects::builtin::types::IntegerType::get(
        ctx,
        64,
        crate::dialects::builtin::types::Signedness::Unsigned,
    )
    .into()
}

fn flatten_aggregate_value(value: LoweredValue) -> STAIRResult<Vec<LoweredValue>> {
    match value {
        LoweredValue::Aggregate(fields) => {
            let mut flattened = Vec::new();
            for field in fields {
                if let Some(field) = field {
                    flattened.extend(flatten_aggregate_value(field)?);
                } else {
                    flattened.push(LoweredValue::Undef);
                }
            }
            Ok(flattened)
        }
        other => Ok(vec![other]),
    }
}

pub(super) fn stack_size_of(
    ctx: &Context,
    ty: TypeHandle,
) -> STAIRResult<u64> {
    if is_zero_sized_ty(ctx, ty) {
        return Ok(0);
    }
    if is_stack_scalar_ty(ctx, ty) {
        return scalar_size_of(ctx, ty);
    }
    let ty_ref = ty.deref(ctx);
    if let Some(array_ty) = ty_ref.downcast_ref::<crate::dialects::llvm::types::ArrayType>() {
        let elem_ty = array_ty.elem_type();
        let len = array_ty.size();
        drop(ty_ref);
        let elem_size = stack_size_of(ctx, elem_ty)?;
        if elem_size == 0 {
            return Ok(0);
        }
        let stride = align_to(elem_size, stack_align_of(ctx, elem_ty)?);
        return Ok(stride * len);
    }
    if ty_ref
        .downcast_ref::<crate::dialects::llvm::types::StructType>()
        .is_none()
    {
        return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedType(
            format!("non-stack type {}", ty_ref.disp(ctx))
        )));
    }
    drop(ty_ref);

    let layout = aggregate_field_layout(ctx, ty)?;
    let Some((last_offset, last_ty)) = layout.last().copied() else {
        return Ok(0);
    };
    let unpadded_size = last_offset + stack_size_of(ctx, last_ty)?;
    Ok(align_to(unpadded_size, stack_align_of(ctx, ty)?))
}

pub(super) fn stack_align_of(
    ctx: &Context,
    ty: TypeHandle,
) -> STAIRResult<u64> {
    if is_zero_sized_ty(ctx, ty) {
        return Ok(1);
    }
    if is_stack_scalar_ty(ctx, ty) {
        return Ok(scalar_size_of(ctx, ty)?.min(8).max(1));
    }
    let ty_ref = ty.deref(ctx);
    if let Some(array_ty) = ty_ref.downcast_ref::<crate::dialects::llvm::types::ArrayType>() {
        let elem_ty = array_ty.elem_type();
        drop(ty_ref);
        return stack_align_of(ctx, elem_ty);
    }
    drop(ty_ref);
    let mut align = 1;
    for field in struct_fields(ctx, ty)? {
        align = align.max(stack_align_of(ctx, field)?);
    }
    Ok(align)
}

pub(super) fn align_to(value: u64, align: u64) -> u64 {
    if align <= 1 {
        value
    } else {
        (value + align - 1) & !(align - 1)
    }
}

pub(super) fn aggregate_field_layout(
    ctx: &Context,
    ty: TypeHandle,
) -> STAIRResult<Vec<(u64, TypeHandle)>> {
    if is_zero_sized_ty(ctx, ty) {
        return Ok(Vec::new());
    }

    let ty_ref = ty.deref(ctx);
    if let Some(array_ty) = ty_ref.downcast_ref::<crate::dialects::llvm::types::ArrayType>() {
        let elem_ty = array_ty.elem_type();
        let len = array_ty.size();
        drop(ty_ref);
        let elem_size = stack_size_of(ctx, elem_ty)?;
        if elem_size == 0 {
            return Ok(Vec::new());
        }
        let stride = align_to(elem_size, stack_align_of(ctx, elem_ty)?);
        return Ok((0..len).map(|index| (index * stride, elem_ty)).collect());
    }

    let Some(struct_ty) = ty_ref.downcast_ref::<crate::dialects::llvm::types::StructType>() else {
        return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedType(
            format!("non-aggregate layout type {}", ty_ref.disp(ctx))
        )));
    };
    if struct_ty.is_opaque() {
        return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedType(
            "opaque struct layout".to_string()
        )));
    }
    let fields: Vec<_> = struct_ty.fields().collect();
    drop(ty_ref);

    let mut offset = 0u64;
    let mut layout = Vec::with_capacity(fields.len());
    for field_ty in fields {
        let field_size = stack_size_of(ctx, field_ty)?;
        if field_size == 0 {
            layout.push((offset, field_ty));
            continue;
        }
        offset = align_to(offset, stack_align_of(ctx, field_ty)?);
        layout.push((offset, field_ty));
        offset += field_size;
    }
    Ok(layout)
}

pub(super) fn scalar_size_of(
    ctx: &Context,
    ty: TypeHandle,
) -> STAIRResult<u64> {
    let ty_ref = ty.deref(ctx);
    if let Some(int_ty) = ty_ref.downcast_ref::<crate::dialects::builtin::types::IntegerType>() {
        let size = (int_ty.width() as u64).div_ceil(8).max(1);
        return Ok(size);
    }
    if ty_ref
        .downcast_ref::<crate::dialects::llvm::types::PointerType>()
        .is_some()
    {
        return Ok(8);
    }
    Err(input_error_noloc!(X86_64DarwinErr::UnsupportedType(
        format!("non-scalar memory type {}", ty_ref.disp(ctx))
    )))
}
