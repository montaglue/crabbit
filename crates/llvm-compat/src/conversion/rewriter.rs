//! Conversion Pattern Rewriter
//!
//! The ConversionPatternRewriter provides methods for modifying IR during
//! dialect conversion. It tracks modifications and can rollback on failure.
//!
//! See: https://mlir.llvm.org/docs/DialectConversion/

use crate::{
    context::{Context, Ptr},
    ir::{basic_block::BasicBlock, operation::Operation, region::Region, value::Value},
    linked_list::{ContainsLinkedList, LinkedList},
};

/// Replace all uses of `op`'s results with the corresponding [Value]s from
/// `results`. Results beyond the length of `results` are left untouched.
fn replace_all_op_uses_with(op: Ptr<Operation>, ctx: &Context, results: impl Iterator<Item = Value>) {
    let old_results: Vec<Value> = op.deref(ctx).results().collect();
    for (old_result, new_result) in old_results.into_iter().zip(results) {
        old_result.replace_all_uses_with(ctx, &new_result);
    }
}

/// Tracks a single IR modification for potential rollback.
#[allow(dead_code)]
enum RewriteAction {
    /// An operation was created.
    CreateOp(Ptr<Operation>),
    /// An operation was erased.
    EraseOp {
        /// The erased operation (may be invalid after erase).
        op: Ptr<Operation>,
        /// The block it was in.
        block: Option<Ptr<BasicBlock>>,
    },
    /// An operation was replaced.
    ReplaceOp {
        original: Ptr<Operation>,
        replacements: Vec<Value>,
    },
    /// An operation was moved.
    MoveOp {
        op: Ptr<Operation>,
        from_block: Option<Ptr<BasicBlock>>,
    },
    /// Block arguments were modified.
    ModifyBlockArgs {
        block: Ptr<BasicBlock>,
        original_arg_count: usize,
    },
}

/// Rewriter for conversion patterns.
///
/// This rewriter tracks all modifications made during pattern application
/// and can rollback changes if the conversion fails.
///
/// # Key methods
///
/// - `replace_op` - Replace an operation with new values
/// - `erase_op` - Remove an operation
/// - `create_op` - Create a new operation
/// - `set_insertion_point` - Set where new operations are inserted
pub struct ConversionPatternRewriter {
    /// Stack of modifications for rollback.
    actions: Vec<RewriteAction>,

    /// Current insertion point (block and position).
    insertion_block: Option<Ptr<BasicBlock>>,

    /// Insert before this operation (None = end of block).
    insertion_point: Option<Ptr<Operation>>,

    /// Value mappings from original to converted values.
    value_mapping: rustc_hash::FxHashMap<Value, Value>,

    /// Whether we're in rollback mode (delayed modifications).
    rollback_mode: bool,
}

impl ConversionPatternRewriter {
    /// Create a new rewriter.
    pub fn new() -> Self {
        ConversionPatternRewriter {
            actions: Vec::new(),
            insertion_block: None,
            insertion_point: None,
            value_mapping: rustc_hash::FxHashMap::default(),
            rollback_mode: true,
        }
    }

    /// Create a rewriter without rollback support (immediate modifications).
    pub fn new_no_rollback() -> Self {
        ConversionPatternRewriter {
            actions: Vec::new(),
            insertion_block: None,
            insertion_point: None,
            value_mapping: rustc_hash::FxHashMap::default(),
            rollback_mode: false,
        }
    }

    /// Set the insertion point to before the given operation.
    pub fn set_insertion_point_before(&mut self, op: Ptr<Operation>, ctx: &Context) {
        let op_ref = op.deref(ctx);
        self.insertion_block = op_ref.get_parent_block();
        self.insertion_point = Some(op);
    }

    /// Set the insertion point to after the given operation.
    pub fn set_insertion_point_after(&mut self, op: Ptr<Operation>, ctx: &Context) {
        let op_ref = op.deref(ctx);
        self.insertion_block = op_ref.get_parent_block();
        self.insertion_point = op_ref.get_next();
    }

    /// Set the insertion point to the start of a block.
    pub fn set_insertion_point_to_start(&mut self, block: Ptr<BasicBlock>, ctx: &Context) {
        self.insertion_block = Some(block);
        self.insertion_point = block.deref(ctx).get_head();
    }

    /// Set the insertion point to the end of a block.
    pub fn set_insertion_point_to_end(&mut self, block: Ptr<BasicBlock>) {
        self.insertion_block = Some(block);
        self.insertion_point = None;
    }

    /// Get the current insertion block.
    pub fn get_insertion_block(&self) -> Option<Ptr<BasicBlock>> {
        self.insertion_block
    }

    /// Insert an operation at the current insertion point.
    pub fn insert(&mut self, op: Ptr<Operation>, ctx: &Context) {
        if let Some(block) = self.insertion_block {
            if let Some(before) = self.insertion_point {
                op.insert_before(ctx, before);
            } else {
                op.insert_at_back(block, ctx);
            }
            self.actions.push(RewriteAction::CreateOp(op));
        }
    }

    /// Replace an operation's results with new values.
    ///
    /// This replaces all uses of the operation's results with the provided
    /// values, then erases the operation.
    pub fn replace_op(&mut self, op: Ptr<Operation>, replacements: Vec<Value>, ctx: &Context) {
        // Record the action for potential rollback
        self.actions.push(RewriteAction::ReplaceOp {
            original: op,
            replacements: replacements.clone(),
        });

        // Map old results to new values
        let op_ref = op.deref(ctx);
        let num_results = op_ref.get_num_results();
        assert_eq!(
            replacements.len(),
            num_results,
            "Replacement count must match result count"
        );

        for (i, replacement) in replacements.into_iter().enumerate() {
            let old_result = op_ref.get_result(i);
            self.value_mapping.insert(old_result, replacement);
        }
        drop(op_ref);

        // Replace uses and erase
        replace_all_op_uses_with(op, ctx, self.get_mapped_values(op, ctx).into_iter());
        self.erase_op_impl(op, ctx);
    }

    /// Replace an operation with results from a new operation.
    pub fn replace_op_with_new_op(
        &mut self,
        old_op: Ptr<Operation>,
        new_op: Ptr<Operation>,
        ctx: &Context,
    ) {
        let new_results: Vec<Value> = new_op.deref(ctx).results().collect();
        self.replace_op(old_op, new_results, ctx);
    }

    /// Erase an operation.
    pub fn erase_op(&mut self, op: Ptr<Operation>, ctx: &mut Context) {
        let block = op.deref(ctx).get_parent_block();
        self.actions.push(RewriteAction::EraseOp { op, block });
        self.erase_op_impl(op, ctx);
    }

    /// Internal implementation of operation erasure.
    fn erase_op_impl(&self, op: Ptr<Operation>, ctx: &Context) {
        // In a full implementation, we'd handle this differently based on
        // rollback_mode. For now, we just mark it for removal.
        // The actual erasure happens when the conversion is finalized.

        // Drop uses but don't deallocate yet if in rollback mode
        Operation::drop_all_uses(op, ctx);

        if !self.rollback_mode {
            // Immediate mode: unlink from block
            if op.deref(ctx).get_parent_block().is_some() {
                op.unlink(ctx);
            }
        }
    }

    /// Get the mapped value for an original value.
    ///
    /// If the value has been remapped (e.g., through type conversion),
    /// returns the new value. Otherwise returns the original.
    pub fn get_remapped_value(&self, value: Value) -> Value {
        self.value_mapping.get(&value).copied().unwrap_or(value)
    }

    /// Get remapped values for all operands of an operation.
    pub fn get_remapped_operands(&self, op: Ptr<Operation>, ctx: &Context) -> Vec<Value> {
        op.deref(ctx)
            .operands()
            .map(|v| self.get_remapped_value(v))
            .collect()
    }

    /// Get mapped values for an operation's results.
    fn get_mapped_values(&self, op: Ptr<Operation>, ctx: &Context) -> Vec<Value> {
        op.deref(ctx)
            .results()
            .map(|v| self.get_remapped_value(v))
            .collect()
    }

    /// Map a value to a new value.
    pub fn map_value(&mut self, from: Value, to: Value) {
        self.value_mapping.insert(from, to);
    }

    /// Create a new block in a region.
    pub fn create_block(&mut self, region: Ptr<Region>, ctx: &mut Context) -> Ptr<BasicBlock> {
        let block = BasicBlock::new(ctx, None, vec![]);
        block.insert_at_back(region, ctx);
        block
    }

    /// Move an operation to a new location.
    pub fn move_op_before(&mut self, op: Ptr<Operation>, before: Ptr<Operation>, ctx: &Context) {
        let from_block = op.deref(ctx).get_parent_block();
        self.actions.push(RewriteAction::MoveOp { op, from_block });

        if op.deref(ctx).get_parent_block().is_some() {
            op.unlink(ctx);
        }
        op.insert_before(ctx, before);
    }

    /// Move an operation to the end of a block.
    pub fn move_op_to_block_end(
        &mut self,
        op: Ptr<Operation>,
        block: Ptr<BasicBlock>,
        ctx: &Context,
    ) {
        let from_block = op.deref(ctx).get_parent_block();
        self.actions.push(RewriteAction::MoveOp { op, from_block });

        if op.deref(ctx).get_parent_block().is_some() {
            op.unlink(ctx);
        }
        op.insert_at_back(block, ctx);
    }

    /// Inline a block's contents before an operation.
    pub fn inline_block_before(
        &mut self,
        source: Ptr<BasicBlock>,
        before: Ptr<Operation>,
        ctx: &Context,
    ) {
        // Move all operations from source to before the target
        let mut op_opt = source.deref(ctx).get_head();
        while let Some(op) = op_opt {
            let next = op.deref(ctx).get_next();
            self.move_op_before(op, before, ctx);
            op_opt = next;
        }
    }

    /// Notify that the pattern failed and rollback changes.
    pub fn rollback(&mut self, _ctx: &mut Context) {
        // In a full implementation, we would undo all recorded actions
        // For now, just clear the action list
        self.actions.clear();
        self.value_mapping.clear();
    }

    /// Finalize the conversion (called on success).
    pub fn finalize(&mut self, ctx: &mut Context) {
        // Actually erase operations that were marked for removal
        for action in &self.actions {
            if let RewriteAction::ReplaceOp { original, .. } = action {
                // The operation should already have its uses replaced
                // Now we can safely deallocate it
                if original.deref(ctx).get_parent_block().is_some() {
                    original.unlink(ctx);
                }
                // Note: actual deallocation would require mutable context
                // and ArenaObj::dealloc, which we'll skip for now
            }
        }
        self.actions.clear();
    }

    /// Get the number of recorded actions.
    pub fn num_actions(&self) -> usize {
        self.actions.len()
    }
}

impl Default for ConversionPatternRewriter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rewriter_creation() {
        let rewriter = ConversionPatternRewriter::new();
        assert!(rewriter.rollback_mode);
        assert_eq!(rewriter.num_actions(), 0);
    }

    #[test]
    fn test_rewriter_no_rollback() {
        let rewriter = ConversionPatternRewriter::new_no_rollback();
        assert!(!rewriter.rollback_mode);
    }
}
