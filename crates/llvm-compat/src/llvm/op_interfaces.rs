//! Op interfaces specific to the LLVM dialect.

use pliron::derive::op_interface;
use thiserror::Error;

use crate::{
    context::Context,
    dialects::builtin::{
        op_interfaces::{OneResultInterface, SameOperandsAndResultType},
        types::IntegerType,
    },
    dict_key,
    ir::op::Op,
    ir::r#type::Typed,
    ir::value::Value,
    result::STAIRResult,
    verify_err,
};

use super::{
    attributes::{AlignmentAttr, IntegerOverflowFlagsAttr},
    types::PointerType,
};

// ============================================================================
// BinArithOp
// ============================================================================

/// Base interface for binary arithmetic operations.
/// Requires same operand and result types, and exactly one result.
#[op_interface]
pub trait BinArithOp: SameOperandsAndResultType + OneResultInterface {
    /// Get the left-hand side operand.
    fn get_lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Get the right-hand side operand.
    fn get_rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }

    fn verify(_op: &dyn Op, _ctx: &Context) -> STAIRResult<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}

// ============================================================================
// IntBinArithOp
// ============================================================================

#[derive(Error, Debug)]
#[error("Integer binary arith op operands/result must be signless integer type")]
pub struct IntBinArithOpVerifyErr;

/// Binary arithmetic operation on signless integers.
#[op_interface]
pub trait IntBinArithOp: BinArithOp {
    fn verify(op: &dyn Op, ctx: &Context) -> STAIRResult<()>
    where
        Self: Sized,
    {
        let op_ref = op.get_operation().deref(ctx);
        let ty = op_ref.get_operand(0).get_type(ctx);
        let ty_ref = ty.deref(ctx);
        // Check it's a signless integer type
        if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>()
            && int_ty.is_signless()
        {
            return Ok(());
        }
        verify_err!(op.loc(ctx), IntBinArithOpVerifyErr)
    }
}

// ============================================================================
// IntBinArithOpWithOverflowFlag
// ============================================================================

dict_key!(
    /// Key for the integer overflow flags attribute.
    ATTR_KEY_INTEGER_OVERFLOW_FLAGS, "llvm_integer_overflow_flags"
);

/// Integer binary arithmetic with overflow flags (nsw/nuw).
#[op_interface]
pub trait IntBinArithOpWithOverflowFlag: IntBinArithOp {
    /// Get the overflow flags.
    fn integer_overflow_flag(&self, ctx: &Context) -> IntegerOverflowFlagsAttr {
        let op_ref = self.get_operation().deref(ctx);
        op_ref
            .attributes
            .get::<IntegerOverflowFlagsAttr>(&ATTR_KEY_INTEGER_OVERFLOW_FLAGS)
            .cloned()
            .unwrap_or_default()
    }

    /// Set the overflow flags.
    fn set_integer_overflow_flag(&self, ctx: &Context, flag: IntegerOverflowFlagsAttr) {
        let mut op_ref = self.get_operation().deref_mut(ctx);
        op_ref
            .attributes
            .set(ATTR_KEY_INTEGER_OVERFLOW_FLAGS.clone(), flag);
    }

    fn verify(_op: &dyn Op, _ctx: &Context) -> STAIRResult<()>
    where
        Self: Sized,
    {
        // The attribute is optional (defaults to no flags), so nothing to verify.
        Ok(())
    }
}

// ============================================================================
// FloatBinArithOp
// ============================================================================

#[derive(Error, Debug)]
#[error("Float binary arith op operands/result must be a floating-point type")]
pub struct FloatBinArithOpVerifyErr;

/// Binary arithmetic operation on floating-point types.
#[op_interface]
pub trait FloatBinArithOp: BinArithOp {
    fn verify(op: &dyn Op, ctx: &Context) -> STAIRResult<()>
    where
        Self: Sized,
    {
        use crate::dialects::builtin::type_interfaces::FloatTypeInterface;
        use crate::ir::r#type::type_impls;

        let op_ref = op.get_operation().deref(ctx);
        let ty = op_ref.get_operand(0).get_type(ctx);
        if type_impls::<dyn FloatTypeInterface>(&*ty.deref(ctx)) {
            return Ok(());
        }
        verify_err!(op.loc(ctx), FloatBinArithOpVerifyErr)
    }
}

// ============================================================================
// CastOpInterface
// ============================================================================

/// Interface for cast operations: one operand, one result, possibly different types.
#[op_interface]
pub trait CastOpInterface: OneResultInterface {
    /// Get the operand being cast.
    fn get_operand(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    fn verify(_op: &dyn Op, _ctx: &Context) -> STAIRResult<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}

// ============================================================================
// PointerTypeResult
// ============================================================================

#[derive(Error, Debug)]
#[error("Op result must be an LLVM pointer type")]
pub struct PointerTypeResultVerifyErr;

/// Interface for operations whose single result is a pointer type.
#[op_interface]
pub trait PointerTypeResult: OneResultInterface {
    fn verify(op: &dyn Op, ctx: &Context) -> STAIRResult<()>
    where
        Self: Sized,
    {
        let result_ty = op.get_operation().deref(ctx).get_result(0).get_type(ctx);
        if result_ty.deref(ctx).downcast_ref::<PointerType>().is_some() {
            return Ok(());
        }
        verify_err!(op.loc(ctx), PointerTypeResultVerifyErr)
    }
}

// ============================================================================
// IsDeclaration
// ============================================================================

/// Interface for operations that distinguish declarations from definitions.
#[op_interface]
pub trait IsDeclaration {
    /// Returns true if this op is a declaration (no body), false if a definition.
    fn is_declaration(&self, ctx: &Context) -> bool;

    fn verify(_op: &dyn Op, _ctx: &Context) -> STAIRResult<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}

// ============================================================================
// AlignableOpInterface
// ============================================================================

dict_key!(
    /// Key for the alignment attribute.
    ATTR_KEY_LLVM_ALIGNMENT, "llvm_alignment"
);

/// Interface for operations that can have an alignment attribute.
#[op_interface]
pub trait AlignableOpInterface {
    /// Get the alignment in bytes, if set.
    fn alignment(&self, ctx: &Context) -> Option<u32> {
        let op_ref = self.get_operation().deref(ctx);
        op_ref
            .attributes
            .get::<AlignmentAttr>(&ATTR_KEY_LLVM_ALIGNMENT)
            .map(|a| a.0)
    }

    /// Set the alignment in bytes.
    fn set_alignment(&self, ctx: &Context, alignment: u32) {
        let mut op_ref = self.get_operation().deref_mut(ctx);
        op_ref
            .attributes
            .set(ATTR_KEY_LLVM_ALIGNMENT.clone(), AlignmentAttr(alignment));
    }

    fn verify(_op: &dyn Op, _ctx: &Context) -> STAIRResult<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}
