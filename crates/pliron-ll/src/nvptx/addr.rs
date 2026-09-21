//! Address folding for memory operands: `[reg + imm]` instead of a fresh
//! address register per access.
//!
//! PTX addressing accepts a register plus a constant byte offset in every
//! state space. After the mid-end fully unrolls a tile loop, the per-clone
//! addresses differ only in a constant, so materializing each of them into
//! its own register is exactly the register pressure that cost gemm_tiled
//! its occupancy (`ptxas` pre-hoists every address it is handed). This
//! module decomposes an address SSA value into
//!
//! ```text
//! root  +  Σ termᵢ.value · termᵢ.scale  +  offset
//! ```
//!
//! by walking GEP chains and reassociating constant contributions out of
//! the index expressions. Emission then keeps ONE base register per
//! distinct `(root, terms)` and folds `offset` into the memory operand.
//!
//! Reassociation soundness:
//! - 64-bit `add/sub/mul/shl` with a constant operand peel unconditionally:
//!   address arithmetic is modulo 2^64, and `(x + c)·s = x·s + c·s` holds
//!   exactly under wrapping.
//! - 32-bit expressions under a `zext` to 64 bits peel only when a small
//!   unsigned value-range analysis proves the 32-bit operation cannot wrap
//!   (`zext(x +₃₂ c) = zext(x) + c` requires `x + c < 2^32`). Ranges come
//!   from the launch-geometry special registers (`%tid.*` < the hardware
//!   block-dimension limits, `%ctaid.y/z` < 65535, …), constants, and a few
//!   monotone combinators. `sext` never peels.
//!
//! Space discipline: a folded address must be usable with the SAME
//! representation as the value it replaces. `Shared`-proven values hold raw
//! shared-window addresses, so a fold rooted at a `Shared` value is only
//! admitted when the folded result is itself still `Shared`-proven (a
//! demoted chain keeps the eager `cvta` path). Global and generic addresses
//! are numerically identical, so they mix freely.

use std::collections::HashMap;

use pliron::builtin::op_interfaces::{
    CallOpCallable, CallOpInterface as _, OneOpdInterface as _, OneResultInterface as _,
};

use crate::{
    context::{Context, Ptr},
    dialects::{
        builtin::attributes::IntegerAttr,
        llvm::ops::{
            AddOp, AndOp, BitcastOp, CallOp, GepIndex, GetElementPtrOp, IntToPtrOp, LShrOp,
            MulOp, PtrToIntOp, SelectOp, ShlOp, SubOp, TruncOp, ZExtOp,
        },
        llvm::types::{ArrayType, StructType},
    },
    ir::{basic_block::BasicBlock, operation::Operation, r#type::Typed, value::Value},
    linked_list::ContainsLinkedList,
};

use super::{
    PtrSpace, align_of_ty, align_to, nvvm_intrinsic_name, sext_from, size_of_ty,
    sreg_for_callee, struct_field_offset, width_of,
};

/// How a term's (≤32-bit) value widens to the 64-bit address register.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum TermKind {
    /// Already 64 bits wide.
    I64,
    /// Zero-extend a `.b32` register (`cvt.u64.u32`). Also correct for
    /// sub-32-bit sources, which the register invariant keeps zero-extended.
    Zext32,
    /// Sign-extend from `width` bits (`cvt.s64.s32` after re-sign-extension
    /// for sub-32 widths) — LLVM GEP index semantics for narrow indices.
    Sext(u32),
}

/// One dynamic contribution to an address: `widen(value) * scale` bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Term {
    pub value: Value,
    pub scale: i64,
    pub kind: TermKind,
}

/// A decomposed address: `root + Σ terms + offset` bytes, all modulo 2^64.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct FoldedAddr {
    pub root: Value,
    pub terms: Vec<Term>,
    pub offset: i64,
}

impl FoldedAddr {
    /// The base-register cache key: everything but the constant offset.
    pub fn base_key(&self) -> (Value, Vec<Term>) {
        (self.root, self.terms.clone())
    }

    /// A power-of-two byte alignment the dynamic part (`Σ terms`) provably
    /// keeps, capped at 16 (the largest PTX vector access). The root's own
    /// alignment is the caller's problem.
    pub fn dynamic_align(&self) -> i64 {
        self.terms
            .iter()
            .map(|term| {
                let scale = term.scale.unsigned_abs();
                if scale == 0 { 16 } else { (1i64 << scale.trailing_zeros().min(4)).min(16) }
            })
            .min()
            .unwrap_or(16)
    }
}

/// Follow register-aliasing casts (bitcast / inttoptr / ptrtoint) to the
/// value they alias: emission maps them to the same register, so the
/// source is the better root (it unifies base keys and lets the vector
/// planner see through `&raw mut SHARED as *mut T` to the `addressof`).
/// A cast that crosses the raw-shared representation boundary (a demoted
/// alias materializes a `cvta` at its definition) is NOT an alias here.
fn unwrap_aliases(ctx: &Context, value: Value, spaces: &HashMap<Value, PtrSpace>) -> Value {
    let shared = |v: &Value| spaces.get(v).copied() == Some(PtrSpace::Shared);
    let mut value = value;
    loop {
        let Some(def) = value.defining_op() else {
            return value;
        };
        let op_obj = Operation::get_op_dyn(def, ctx);
        let operand = if let Some(cast) = op_obj.downcast_ref::<BitcastOp>() {
            cast.get_operand(ctx)
        } else if let Some(cast) = op_obj.downcast_ref::<IntToPtrOp>() {
            cast.get_operand(ctx)
        } else if let Some(cast) = op_obj.downcast_ref::<PtrToIntOp>() {
            cast.get_operand(ctx)
        } else {
            return value;
        };
        if shared(&value) != shared(&operand) {
            return value;
        }
        value = operand;
    }
}

/// Offsets folded into memory operands must be immediate-encodable (PTX
/// address offsets are 32-bit signed); the margin below i32::MAX keeps
/// aggregate-leaf and vector-lane additions in range too.
const MAX_FOLD_OFFSET: i64 = 1 << 30;
/// More terms than this and folding no longer pays for its `mad` chain.
const MAX_TERMS: usize = 4;
/// Recursion budget for the linear decomposition of one index.
const PEEL_DEPTH: u32 = 8;

/// The IR-side folder: integer constants and value ranges for one function.
pub(super) struct AddrFolder {
    /// Integer `ConstantOp` results: (bits masked to width, width).
    consts: HashMap<Value, (u128, u32)>,
    /// Memoized unsigned upper bounds (`None` = unbounded / unknown).
    range_memo: HashMap<Value, Option<u64>>,
}

impl AddrFolder {
    /// Scan the function once for integer constants; everything else is
    /// resolved lazily.
    pub fn new(ctx: &Context, blocks: &[Ptr<BasicBlock>]) -> Self {
        let mut consts = HashMap::new();
        for block in blocks {
            for op_ptr in block.deref(ctx).iter(ctx) {
                let op_obj = Operation::get_op_dyn(op_ptr, ctx);
                let Some(constant) =
                    op_obj.downcast_ref::<crate::dialects::builtin::ops::ConstantOp>()
                else {
                    continue;
                };
                let result = constant.get_result(ctx);
                let Ok(width) = width_of(ctx, result.get_type(ctx)) else {
                    continue;
                };
                if let Some(int) = constant.get_value(ctx).downcast_ref::<IntegerAttr>() {
                    let mask = if width >= 128 {
                        u128::MAX
                    } else {
                        (1u128 << width) - 1
                    };
                    consts.insert(result, (int.value().to_u128() & mask, width));
                }
            }
        }
        AddrFolder {
            consts,
            range_memo: HashMap::new(),
        }
    }

    fn const_of(&self, value: Value) -> Option<(u128, u32)> {
        self.consts.get(&value).copied()
    }

    /// Fold one GEP into `root + terms + offset`, absorbing an
    /// already-folded base. `None` = keep the eager emission path (the
    /// caller loses nothing but this optimization).
    pub fn fold_gep(
        &mut self,
        ctx: &Context,
        gep: &GetElementPtrOp,
        plans: &HashMap<Value, FoldedAddr>,
        spaces: &HashMap<Value, PtrSpace>,
    ) -> Option<FoldedAddr> {
        let base_val = gep.get_operand_src_ptr(ctx);
        let (root, mut terms, mut offset) = match plans.get(&base_val) {
            Some(base_plan) => (
                base_plan.root,
                base_plan.terms.clone(),
                base_plan.offset,
            ),
            None => (unwrap_aliases(ctx, base_val, spaces), Vec::new(), 0i64),
        };

        let indices = gep.indices(ctx);
        let mut current_ty = gep.src_elem_type(ctx);
        for (position, index) in indices.iter().enumerate() {
            // Mirror of emit_gep: first index strides over the source
            // element type, later indices descend into it.
            let elem_ty = if position == 0 {
                current_ty
            } else {
                let ty_ref = current_ty.deref(ctx);
                if let Some(array_ty) = ty_ref.downcast_ref::<ArrayType>() {
                    array_ty.elem_type()
                } else if let Some(struct_ty) = ty_ref.downcast_ref::<StructType>() {
                    if struct_ty.is_opaque() {
                        return None;
                    }
                    let field_index = match index {
                        GepIndex::Constant(value) => *value as u64,
                        GepIndex::Value(value) => {
                            let (imm, _) = self.const_of(*value)?;
                            imm as u64
                        }
                    };
                    let (field_offset, field_ty) =
                        struct_field_offset(ctx, struct_ty, field_index).ok()?;
                    drop(ty_ref);
                    offset = offset.wrapping_add(field_offset as i64);
                    current_ty = field_ty;
                    continue;
                } else {
                    return None;
                }
            };
            let elem_size = align_to(
                size_of_ty(ctx, elem_ty).ok()?,
                align_of_ty(ctx, elem_ty).ok()?,
            ) as i64;
            match index {
                GepIndex::Constant(value) => {
                    offset = offset.wrapping_add((*value as i64).wrapping_mul(elem_size));
                }
                GepIndex::Value(value) => {
                    let width = width_of(ctx, value.get_type(ctx)).ok()?;
                    if let Some((imm, imm_width)) = self.const_of(*value) {
                        offset = offset
                            .wrapping_add(sext_from(imm, imm_width).wrapping_mul(elem_size));
                    } else if width == 64 {
                        offset = offset
                            .wrapping_add(self.peel64(ctx, *value, elem_size, &mut terms));
                    } else if width <= 32 {
                        // Narrow indices sign-extend (LLVM GEP semantics);
                        // no reassociation under a sign extension.
                        terms.push(Term {
                            value: *value,
                            scale: elem_size,
                            kind: TermKind::Sext(width),
                        });
                    } else {
                        return None;
                    }
                }
            }
            current_ty = elem_ty;
        }

        // Merge repeated (value, kind) terms; a zero net scale vanishes.
        let mut merged: Vec<Term> = Vec::with_capacity(terms.len());
        for term in terms {
            if term.scale == 0 {
                continue;
            }
            match merged
                .iter_mut()
                .find(|m| m.value == term.value && m.kind == term.kind)
            {
                Some(existing) => existing.scale = existing.scale.wrapping_add(term.scale),
                None => merged.push(term),
            }
        }
        merged.retain(|term| term.scale != 0);
        let terms = merged;
        if terms.len() > MAX_TERMS || offset.abs() > MAX_FOLD_OFFSET {
            return None;
        }
        // Representation discipline: raw-shared folds stay valid only while
        // the whole chain is Shared-proven (demoted results take the eager
        // cvta path instead).
        let generic = PtrSpace::Generic;
        let root_space = spaces.get(&root).copied().unwrap_or(generic);
        let result_space = spaces
            .get(&gep.get_result(ctx))
            .copied()
            .unwrap_or(generic);
        let base_space = spaces.get(&base_val).copied().unwrap_or(generic);
        if (root_space == PtrSpace::Shared) != (result_space == PtrSpace::Shared)
            || (root_space == PtrSpace::Shared) != (base_space == PtrSpace::Shared)
        {
            return None;
        }
        Some(FoldedAddr {
            root,
            terms,
            offset,
        })
    }

    /// Peel constants out of a 64-bit index expression along a single
    /// spine (appending the one residual term, if any, to `terms`) and
    /// return the constant byte contribution. Peels are exact under
    /// mod-2^64 arithmetic, so no range proofs are needed at this width.
    ///
    /// Deliberately NOT a full linear decomposition: splitting a
    /// dynamic+dynamic 64-bit add into two terms re-materializes the
    /// loop-invariant half inside every block that touches the address,
    /// and it trades the `add`/`shl` shapes `ptxas`'s induction-variable
    /// and uniform-register machinery recognizes for longer per-access
    /// chains — measured on gemm_control's k-loop as 30→38 registers and
    /// a 1.38× slowdown before this was reverted. The 32-bit
    /// decomposition under a `zext` ([Self::decompose_zext32]) does split
    /// sums: those shapes are the unrolled tile indices this folding
    /// exists for, and their invariant halves are shared through the
    /// per-block base cache instead.
    fn peel64(
        &mut self,
        ctx: &Context,
        value: Value,
        scale: i64,
        terms: &mut Vec<Term>,
    ) -> i64 {
        let mut value = value;
        let mut scale = scale;
        let mut offset = 0i64;
        for _ in 0..PEEL_DEPTH {
            if scale == 0 {
                return offset;
            }
            if let Some((imm, width)) = self.const_of(value) {
                return offset.wrapping_add(sext_from(imm, width).wrapping_mul(scale));
            }
            let Some(def) = value.defining_op() else { break };
            let op_obj = Operation::get_op_dyn(def, ctx);
            let operands = |i: usize| def.deref(ctx).get_operand(i);
            if op_obj.downcast_ref::<AddOp>().is_some() {
                let (a, b) = (operands(0), operands(1));
                if let Some((imm, width)) = self.const_of(b) {
                    offset = offset.wrapping_add(sext_from(imm, width).wrapping_mul(scale));
                    value = a;
                } else if let Some((imm, width)) = self.const_of(a) {
                    offset = offset.wrapping_add(sext_from(imm, width).wrapping_mul(scale));
                    value = b;
                } else {
                    break;
                }
            } else if op_obj.downcast_ref::<SubOp>().is_some() {
                let (a, b) = (operands(0), operands(1));
                if let Some((imm, width)) = self.const_of(b) {
                    offset = offset.wrapping_sub(sext_from(imm, width).wrapping_mul(scale));
                    value = a;
                } else {
                    break;
                }
            } else if op_obj.downcast_ref::<MulOp>().is_some() {
                let (a, b) = (operands(0), operands(1));
                if let Some((imm, width)) = self.const_of(b) {
                    scale = scale.wrapping_mul(sext_from(imm, width));
                    value = a;
                } else if let Some((imm, width)) = self.const_of(a) {
                    scale = scale.wrapping_mul(sext_from(imm, width));
                    value = b;
                } else {
                    break;
                }
            } else if op_obj.downcast_ref::<ShlOp>().is_some() {
                let (a, b) = (operands(0), operands(1));
                match self.const_of(b) {
                    Some((imm, _)) if imm < 64 => {
                        scale = scale.wrapping_shl(imm as u32);
                        value = a;
                    }
                    _ => break,
                }
            } else if let Some(zext) = op_obj.downcast_ref::<ZExtOp>() {
                let inner = zext.get_operand(ctx);
                let Ok(inner_width) = width_of(ctx, inner.get_type(ctx)) else {
                    break;
                };
                if inner_width > 32 || inner_width == 1 {
                    break;
                }
                if inner_width == 32 {
                    return offset.wrapping_add(self.decompose_zext32(
                        ctx,
                        inner,
                        scale,
                        PEEL_DEPTH,
                        terms,
                    ));
                }
                // Narrower sources are zero-extended in their register
                // already; the term widens with a plain cvt.u64.u32.
                terms.push(Term {
                    value: inner,
                    scale,
                    kind: TermKind::Zext32,
                });
                return offset;
            } else {
                break;
            }
        }
        terms.push(Term {
            value,
            scale,
            kind: TermKind::I64,
        });
        offset
    }

    /// Decompose a 32-bit unsigned expression under a `zext` to 64 bits.
    /// Every reassociation needs a range proof that the 32-bit operation
    /// cannot wrap (`zext(a + b) = zext(a) + zext(b)` only when
    /// `a + b < 2^32`, and likewise for constant multiplies/shifts);
    /// without one the expression stays a single opaque term.
    fn decompose_zext32(
        &mut self,
        ctx: &Context,
        value: Value,
        scale: i64,
        depth: u32,
        terms: &mut Vec<Term>,
    ) -> i64 {
        if scale == 0 {
            return 0;
        }
        if let Some((imm, _)) = self.const_of(value) {
            // Zero-extended constant: folds entirely.
            return (imm as u32 as i64).wrapping_mul(scale);
        }
        let opaque = |terms: &mut Vec<Term>| {
            terms.push(Term {
                value,
                scale,
                kind: TermKind::Zext32,
            });
            0
        };
        if depth == 0 {
            return opaque(terms);
        }
        let Some(def) = value.defining_op() else {
            return opaque(terms);
        };
        let op_obj = Operation::get_op_dyn(def, ctx);
        let operands = |i: usize| def.deref(ctx).get_operand(i);
        if op_obj.downcast_ref::<AddOp>().is_some() {
            let (a, b) = (operands(0), operands(1));
            let (Some(max_a), Some(max_b)) =
                (self.range_max(ctx, a), self.range_max(ctx, b))
            else {
                return opaque(terms);
            };
            if max_a
                .checked_add(max_b)
                .is_none_or(|sum| sum > u32::MAX as u64)
            {
                return opaque(terms);
            }
            let lhs = self.decompose_zext32(ctx, a, scale, depth - 1, terms);
            lhs.wrapping_add(self.decompose_zext32(ctx, b, scale, depth - 1, terms))
        } else if op_obj.downcast_ref::<MulOp>().is_some() {
            let (a, b) = (operands(0), operands(1));
            let (other, constant) = if let Some((imm, _)) = self.const_of(b) {
                (a, imm as u32 as u64)
            } else if let Some((imm, _)) = self.const_of(a) {
                (b, imm as u32 as u64)
            } else {
                return opaque(terms);
            };
            let Some(max) = self.range_max(ctx, other) else {
                return opaque(terms);
            };
            if max
                .checked_mul(constant)
                .is_none_or(|prod| prod > u32::MAX as u64)
            {
                return opaque(terms);
            }
            self.decompose_zext32(
                ctx,
                other,
                scale.wrapping_mul(constant as i64),
                depth - 1,
                terms,
            )
        } else if op_obj.downcast_ref::<ShlOp>().is_some() {
            let (a, b) = (operands(0), operands(1));
            let Some((imm, _)) = self.const_of(b) else {
                return opaque(terms);
            };
            if imm >= 32 {
                return opaque(terms);
            }
            let Some(max) = self.range_max(ctx, a) else {
                return opaque(terms);
            };
            if max
                .checked_shl(imm as u32)
                .is_none_or(|shifted| shifted > u32::MAX as u64)
            {
                return opaque(terms);
            }
            self.decompose_zext32(ctx, a, scale.wrapping_shl(imm as u32), depth - 1, terms)
        } else {
            opaque(terms)
        }
    }

    /// A provable unsigned upper bound of a 32-bit value, or `None`.
    /// Sources: constants, the launch-geometry special registers, and a few
    /// monotone combinators over them.
    fn range_max(&mut self, ctx: &Context, value: Value) -> Option<u64> {
        if let Some(cached) = self.range_memo.get(&value) {
            return *cached;
        }
        // Cycle guard: block arguments (the only possible cycles run
        // through them) return None anyway, but seed the memo defensively.
        self.range_memo.insert(value, None);
        let bound = self.range_max_inner(ctx, value);
        self.range_memo.insert(value, bound);
        bound
    }

    fn range_max_inner(&mut self, ctx: &Context, value: Value) -> Option<u64> {
        if let Some((imm, width)) = self.const_of(value) {
            if width > 32 {
                return None;
            }
            return Some(imm as u32 as u64);
        }
        let def = value.defining_op()?;
        let op_obj = Operation::get_op_dyn(def, ctx);
        let operands = |i: usize| def.deref(ctx).get_operand(i);
        if let Some(call) = op_obj.downcast_ref::<CallOp>() {
            let CallOpCallable::Direct(name) = call.callee(ctx) else {
                return None;
            };
            let canonical = nvvm_intrinsic_name(name.as_ref());
            // Hardware launch-geometry limits (CUDA occupancy tables):
            // block dims ≤ (1024, 1024, 64), grid y/z dims ≤ 65535.
            return match sreg_for_callee(&canonical)? {
                "%tid.x" | "%tid.y" => Some(1023),
                "%tid.z" => Some(63),
                "%ntid.x" | "%ntid.y" => Some(1024),
                "%ntid.z" => Some(64),
                "%ctaid.y" | "%ctaid.z" => Some(65534),
                "%nctaid.y" | "%nctaid.z" => Some(65535),
                "%laneid" => Some(31),
                "WARP_SZ" => Some(32),
                _ => None,
            };
        }
        let bound = if op_obj.downcast_ref::<AddOp>().is_some() {
            let a = self.range_max(ctx, operands(0))?;
            let b = self.range_max(ctx, operands(1))?;
            a.checked_add(b)?
        } else if op_obj.downcast_ref::<MulOp>().is_some() {
            let a = self.range_max(ctx, operands(0))?;
            let b = self.range_max(ctx, operands(1))?;
            a.checked_mul(b)?
        } else if op_obj.downcast_ref::<ShlOp>().is_some() {
            let (imm, _) = self.const_of(operands(1))?;
            if imm >= 32 {
                return None;
            }
            self.range_max(ctx, operands(0))?.checked_shl(imm as u32)?
        } else if op_obj.downcast_ref::<LShrOp>().is_some() {
            let (imm, _) = self.const_of(operands(1))?;
            if imm >= 32 {
                return None;
            }
            self.range_max(ctx, operands(0))? >> (imm as u32)
        } else if op_obj.downcast_ref::<AndOp>().is_some() {
            // and(x, c) ≤ c even when x is unbounded.
            let a = self.range_max(ctx, operands(0));
            let b = self.const_of(operands(1)).map(|(imm, _)| imm as u32 as u64);
            match (a, b) {
                (Some(a), Some(b)) => a.min(b),
                (Some(a), None) => a,
                (None, Some(b)) => b,
                (None, None) => return None,
            }
        } else if op_obj.downcast_ref::<SelectOp>().is_some() {
            let a = self.range_max(ctx, operands(1))?;
            let b = self.range_max(ctx, operands(2))?;
            a.max(b)
        } else if op_obj.downcast_ref::<ZExtOp>().is_some() {
            let inner = operands(0);
            let width = width_of(ctx, inner.get_type(ctx)).ok()?;
            if width > 32 {
                return None;
            }
            self.range_max(ctx, inner)
                .unwrap_or((1u64 << width) - 1)
                .min((1u64 << width) - 1)
        } else if op_obj.downcast_ref::<TruncOp>().is_some() {
            // Truncation preserves the bound only when it provably fits.
            let width = width_of(ctx, value.get_type(ctx)).ok()?;
            let inner = self.range_max(ctx, operands(0))?;
            if width >= 64 || inner < (1u64 << width) {
                inner
            } else {
                return None;
            }
        } else {
            return None;
        };
        // Bounds are only meaningful for 32-bit wrap proofs.
        if bound > u32::MAX as u64 {
            None
        } else {
            Some(bound)
        }
    }
}
