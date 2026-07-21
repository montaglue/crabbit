//! Conversion Patterns
//!
//! ConversionPatterns define how to rewrite illegal operations into legal ones.
//! Unlike regular rewrite patterns, conversion patterns receive remapped operands
//! that have already been converted to target types.
//!
//! See: https://mlir.llvm.org/docs/DialectConversion/

use crate::{
    context::{Context, Ptr},
    ir::{op::OpId, operation::Operation, value::Value},
    result::STAIRResult,
};

use super::rewriter::ConversionPatternRewriter;

/// Result of applying a conversion pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternMatchResult {
    /// Pattern matched and conversion succeeded.
    Success,
    /// Pattern did not match this operation.
    Failure,
}

impl PatternMatchResult {
    pub fn is_success(&self) -> bool {
        matches!(self, PatternMatchResult::Success)
    }
}

/// Trait for conversion patterns.
///
/// A conversion pattern specifies how to convert an illegal operation
/// to one or more legal operations.
///
/// # Key differences from regular rewrite patterns
///
/// 1. **Remapped operands**: The `operands` parameter contains operands
///    that have already been converted to target types (if a TypeConverter
///    is registered).
///
/// 2. **ConversionPatternRewriter**: Uses a special rewriter that tracks
///    conversions and can rollback on failure.
///
/// # Example
/// ```ignore
/// struct SourceIntLiteralToIndexConstant;
///
/// impl ConversionPattern for SourceIntLiteralToIndexConstant {
///     fn get_root_kind(&self) -> OpId {
///         SourceIntLiteralOp::get_opid_static()
///     }
///
///     fn match_and_rewrite(
///         &self,
///         op: Ptr<Operation>,
///         operands: &[Value],
///         rewriter: &mut ConversionPatternRewriter,
///         ctx: &mut Context,
///     ) -> STAIRResult<PatternMatchResult> {
///         let source_op = SourceIntLiteralOp { op };
///         let value = source_op.get_attr_int_value(ctx).unwrap().0;
///
///         let index_const = IndexConstantOp::new(ctx, value);
///         rewriter.replace_op(op, vec![index_const.get_result(ctx)], ctx);
///
///         Ok(PatternMatchResult::Success)
///     }
/// }
/// ```
pub trait ConversionPattern: Send + Sync {
    /// Get the OpId that this pattern matches.
    ///
    /// The pattern's `match_and_rewrite` will only be called for operations
    /// with this OpId.
    fn get_root_kind(&self) -> OpId;

    /// Get the benefit of this pattern.
    ///
    /// Higher benefit patterns are tried first. Default is 1.
    fn get_benefit(&self) -> usize {
        1
    }

    /// Match and rewrite the operation.
    ///
    /// # Arguments
    /// * `op` - The operation to convert
    /// * `operands` - The remapped operands (already converted to target types)
    /// * `rewriter` - The rewriter for modifying IR
    /// * `ctx` - The context
    ///
    /// # Returns
    /// * `Ok(Success)` - Conversion succeeded
    /// * `Ok(Failure)` - Pattern did not match (try other patterns)
    /// * `Err(...)` - Conversion failed with an error
    fn match_and_rewrite(
        &self,
        op: Ptr<Operation>,
        operands: &[Value],
        rewriter: &mut ConversionPatternRewriter,
        ctx: &mut Context,
    ) -> STAIRResult<PatternMatchResult>;
}

/// Type alias for boxed conversion patterns.
pub type ConversionPatternObj = Box<dyn ConversionPattern>;

/// Collection of conversion patterns organized by root operation.
pub struct ConversionPatternSet {
    /// Patterns indexed by the operation they match.
    patterns: rustc_hash::FxHashMap<OpId, Vec<ConversionPatternObj>>,
}

impl ConversionPatternSet {
    /// Create a new empty pattern set.
    pub fn new() -> Self {
        ConversionPatternSet {
            patterns: rustc_hash::FxHashMap::default(),
        }
    }

    /// Add a pattern to the set.
    pub fn add<P: ConversionPattern + 'static>(&mut self, pattern: P) {
        let root_kind = pattern.get_root_kind();
        self.patterns
            .entry(root_kind)
            .or_default()
            .push(Box::new(pattern));
    }

    /// Add a boxed pattern to the set.
    pub fn add_boxed(&mut self, pattern: ConversionPatternObj) {
        let root_kind = pattern.get_root_kind();
        self.patterns.entry(root_kind).or_default().push(pattern);
    }

    /// Get patterns that match a given operation.
    ///
    /// Patterns are returned sorted by benefit (highest first).
    pub fn get_patterns(&self, opid: &OpId) -> Option<&[ConversionPatternObj]> {
        self.patterns.get(opid).map(|v| v.as_slice())
    }

    /// Sort all patterns by benefit (highest first).
    pub fn finalize(&mut self) {
        for patterns in self.patterns.values_mut() {
            patterns.sort_by(|a, b| b.get_benefit().cmp(&a.get_benefit()));
        }
    }

    /// Check if there are any patterns for a given operation.
    pub fn has_patterns_for(&self, opid: &OpId) -> bool {
        self.patterns.contains_key(opid)
    }

    /// Get all registered OpIds.
    pub fn get_registered_ops(&self) -> impl Iterator<Item = &OpId> {
        self.patterns.keys()
    }
}

impl Default for ConversionPatternSet {
    fn default() -> Self {
        Self::new()
    }
}

/// Compatibility alias: the vendored core's [Op](crate::op::Op) trait now
/// provides `from_operation` directly.
pub use crate::op::Op as FromOperation;

/// Helper trait for defining type-safe conversion patterns.
///
/// This trait allows patterns to be defined with concrete Op types
/// rather than raw Operation pointers.
pub trait TypedConversionPattern<OpType>: Send + Sync
where
    OpType: FromOperation,
{
    /// Get the benefit of this pattern.
    fn get_benefit(&self) -> usize {
        1
    }

    /// Match and rewrite the operation.
    fn match_and_rewrite(
        &self,
        op: OpType,
        operands: &[Value],
        rewriter: &mut ConversionPatternRewriter,
        ctx: &mut Context,
    ) -> STAIRResult<PatternMatchResult>;
}

/// Wrapper to convert a TypedConversionPattern into a ConversionPattern.
pub struct TypedPatternWrapper<OpType, P>
where
    OpType: FromOperation + 'static,
    P: TypedConversionPattern<OpType>,
{
    pattern: P,
    _phantom: std::marker::PhantomData<OpType>,
}

impl<OpType, P> TypedPatternWrapper<OpType, P>
where
    OpType: FromOperation + 'static,
    P: TypedConversionPattern<OpType>,
{
    pub fn new(pattern: P) -> Self {
        TypedPatternWrapper {
            pattern,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<OpType, P> ConversionPattern for TypedPatternWrapper<OpType, P>
where
    OpType: FromOperation + Send + Sync + 'static,
    P: TypedConversionPattern<OpType> + 'static,
{
    fn get_root_kind(&self) -> OpId {
        OpType::get_opid_static()
    }

    fn get_benefit(&self) -> usize {
        self.pattern.get_benefit()
    }

    fn match_and_rewrite(
        &self,
        op: Ptr<Operation>,
        operands: &[Value],
        rewriter: &mut ConversionPatternRewriter,
        ctx: &mut Context,
    ) -> STAIRResult<PatternMatchResult> {
        let typed_op = OpType::from_operation(op);
        self.pattern
            .match_and_rewrite(typed_op, operands, rewriter, ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_set_creation() {
        let set = ConversionPatternSet::new();
        assert!(set.patterns.is_empty());
    }

    #[test]
    fn test_pattern_match_result() {
        assert!(PatternMatchResult::Success.is_success());
        assert!(!PatternMatchResult::Failure.is_success());
    }
}
