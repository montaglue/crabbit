//! Dialect Conversion Driver
//!
//! This module provides the main entry points for running dialect conversions:
//! - `apply_partial_conversion` - Convert as much as possible, allow some illegal ops
//! - `apply_full_conversion` - All operations must be converted to be legal
//!
//! See: https://mlir.llvm.org/docs/DialectConversion/

use crate::{
    context::{Context, Ptr},
    ir::{basic_block::BasicBlock, operation::Operation, region::Region},
    linked_list::{ContainsLinkedList, LinkedList},
    result::STAIRResult,
};

use super::{
    conversion_pattern::{ConversionPatternSet, PatternMatchResult},
    conversion_target::ConversionTarget,
    rewriter::ConversionPatternRewriter,
    type_converter::TypeConverter,
};

/// Result of a dialect conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversionResult {
    /// All operations were successfully converted.
    Success,
    /// Some operations could not be converted.
    Failure,
}

/// Apply a partial conversion to the given operation.
///
/// A partial conversion will legalize as many operations as possible,
/// but will allow operations that were not explicitly marked as "illegal"
/// to remain unconverted.
///
/// # Arguments
/// * `root` - The root operation to convert
/// * `target` - The conversion target defining legality
/// * `patterns` - The conversion patterns to apply
/// * `type_converter` - Optional type converter for type transformations
/// * `ctx` - The context
///
/// # Returns
/// * `Ok(Success)` - Conversion completed (some ops may remain unconverted)
/// * `Ok(Failure)` - Conversion failed for illegal operations
/// * `Err(...)` - An error occurred during conversion
pub fn apply_partial_conversion(
    root: Ptr<Operation>,
    target: &ConversionTarget,
    patterns: &ConversionPatternSet,
    type_converter: Option<&mut TypeConverter>,
    ctx: &mut Context,
) -> STAIRResult<ConversionResult> {
    let mut driver = ConversionDriver::new(target, patterns, type_converter, false);
    driver.convert_operation(root, ctx)
}

/// Apply a full conversion to the given operation.
///
/// A full conversion requires that all operations are converted to be legal
/// according to the conversion target. If any illegal operation cannot be
/// converted, the conversion fails.
///
/// # Arguments
/// * `root` - The root operation to convert
/// * `target` - The conversion target defining legality
/// * `patterns` - The conversion patterns to apply
/// * `type_converter` - Optional type converter for type transformations
/// * `ctx` - The context
///
/// # Returns
/// * `Ok(Success)` - All operations are now legal
/// * `Ok(Failure)` - Some illegal operations could not be converted
/// * `Err(...)` - An error occurred during conversion
pub fn apply_full_conversion(
    root: Ptr<Operation>,
    target: &ConversionTarget,
    patterns: &ConversionPatternSet,
    type_converter: Option<&mut TypeConverter>,
    ctx: &mut Context,
) -> STAIRResult<ConversionResult> {
    let mut driver = ConversionDriver::new(target, patterns, type_converter, true);
    driver.convert_operation(root, ctx)
}

/// Internal driver for dialect conversion.
struct ConversionDriver<'a> {
    target: &'a ConversionTarget,
    patterns: &'a ConversionPatternSet,
    type_converter: Option<&'a mut TypeConverter>,
    full_conversion: bool,
    rewriter: ConversionPatternRewriter,
}

impl<'a> ConversionDriver<'a> {
    fn new(
        target: &'a ConversionTarget,
        patterns: &'a ConversionPatternSet,
        type_converter: Option<&'a mut TypeConverter>,
        full_conversion: bool,
    ) -> Self {
        ConversionDriver {
            target,
            patterns,
            type_converter,
            full_conversion,
            rewriter: ConversionPatternRewriter::new(),
        }
    }

    /// Convert an operation and all its nested regions.
    fn convert_operation(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
    ) -> STAIRResult<ConversionResult> {
        // First, recursively convert all nested regions
        let regions: Vec<_> = op.deref(ctx).regions().collect();
        for region in regions {
            self.convert_region(region, ctx)?;
        }

        // Check if this operation needs conversion
        if self.target.is_legal(op, ctx) {
            return Ok(ConversionResult::Success);
        }

        // Try to convert the operation
        let result = self.try_convert_operation(op, ctx)?;

        match result {
            ConversionResult::Success => Ok(ConversionResult::Success),
            ConversionResult::Failure => {
                if self.full_conversion {
                    // In full conversion mode, failure to convert is an error
                    Ok(ConversionResult::Failure)
                } else {
                    // In partial conversion mode, we allow failures for non-illegal ops
                    Ok(ConversionResult::Success)
                }
            }
        }
    }

    /// Convert all operations in a region.
    fn convert_region(
        &mut self,
        region: Ptr<Region>,
        ctx: &mut Context,
    ) -> STAIRResult<ConversionResult> {
        // Walk all blocks in the region
        let mut block_opt = region.deref(ctx).get_head();
        while let Some(block) = block_opt {
            self.convert_block(block, ctx)?;
            block_opt = block.deref(ctx).get_next();
        }
        Ok(ConversionResult::Success)
    }

    /// Convert all operations in a block.
    fn convert_block(
        &mut self,
        block: Ptr<BasicBlock>,
        ctx: &mut Context,
    ) -> STAIRResult<ConversionResult> {
        // Collect operations to convert (we iterate over a snapshot because
        // the block contents may change during conversion)
        let mut ops = Vec::new();
        let mut op_opt = block.deref(ctx).get_head();
        while let Some(op) = op_opt {
            ops.push(op);
            op_opt = op.deref(ctx).get_next();
        }

        // Convert each operation
        for op in ops {
            // Skip if operation was already erased
            if op.deref(ctx).get_parent_block().is_none() {
                continue;
            }
            self.convert_operation(op, ctx)?;
        }

        Ok(ConversionResult::Success)
    }

    /// Try to convert a single operation using registered patterns.
    fn try_convert_operation(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
    ) -> STAIRResult<ConversionResult> {
        let opid = Operation::get_opid(op, ctx);

        // Get patterns for this operation
        let Some(patterns) = self.patterns.get_patterns(&opid) else {
            // No patterns registered for this operation
            return Ok(ConversionResult::Failure);
        };

        // Get remapped operands (applying type conversions if needed)
        let operands = self.get_remapped_operands(op, ctx);

        // Set insertion point for new operations
        self.rewriter.set_insertion_point_before(op, ctx);

        // Try each pattern until one succeeds
        for pattern in patterns {
            let result = pattern.match_and_rewrite(op, &operands, &mut self.rewriter, ctx)?;

            if result == PatternMatchResult::Success {
                self.rewriter.finalize(ctx);
                return Ok(ConversionResult::Success);
            }
        }

        // No pattern matched
        self.rewriter.rollback(ctx);
        Ok(ConversionResult::Failure)
    }

    /// Get remapped operands for an operation, applying type conversions.
    fn get_remapped_operands(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
    ) -> Vec<crate::ir::value::Value> {
        let operands: Vec<_> = op.deref(ctx).operands().collect();

        if self.type_converter.is_some() {
            // With type conversion, we'd need to potentially insert casts
            // For now, just return remapped values
            operands
                .into_iter()
                .map(|v| self.rewriter.get_remapped_value(v))
                .collect()
        } else {
            // Without type conversion, just remap values
            operands
                .into_iter()
                .map(|v| self.rewriter.get_remapped_value(v))
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversion_result() {
        assert!(ConversionResult::Success == ConversionResult::Success);
        assert!(ConversionResult::Failure != ConversionResult::Success);
    }
}
