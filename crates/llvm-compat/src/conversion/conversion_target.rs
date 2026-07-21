//! Conversion Target
//!
//! The ConversionTarget defines what operations are considered "legal" after
//! a dialect conversion. This follows MLIR's dialect conversion framework.
//!
//! See: https://mlir.llvm.org/docs/DialectConversion/

use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    context::{Context, Ptr},
    ir::{dialect::DialectName, op::OpId, operation::Operation},
};

/// Legality status for an operation or dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegalityAction {
    /// All instances of the operation/dialect are legal.
    Legal,
    /// The operation/dialect is illegal and must be converted.
    Illegal,
    /// Legality is determined dynamically per-instance.
    Dynamic,
}

/// Callback type for dynamic legality checking.
pub type DynamicLegalityFn = Box<dyn Fn(Ptr<Operation>, &Context) -> bool + Send + Sync>;

/// Defines what is considered legal during dialect conversion.
///
/// Operations can be marked as:
/// - **Legal**: Every instance is valid
/// - **Illegal**: Must be converted
/// - **Dynamic**: Legality checked per-instance via callback
///
/// # Example
/// ```ignore
/// let mut target = ConversionTarget::new();
/// target.add_legal_dialect("builtin");
/// target.add_legal_dialect("scf");
/// target.add_legal_dialect("cf");
/// target.add_legal_dialect("index");
/// target.add_illegal_dialect("source");
/// ```
pub struct ConversionTarget {
    /// Per-dialect legality settings.
    dialect_legality: FxHashMap<DialectName, LegalityAction>,

    /// Per-operation legality overrides (takes precedence over dialect).
    op_legality: FxHashMap<OpId, LegalityAction>,

    /// Dynamic legality callbacks for operations marked as Dynamic.
    dynamic_legality_fns: FxHashMap<OpId, DynamicLegalityFn>,

    /// Operations that are known to be legal regardless of dialect settings.
    legal_ops: FxHashSet<OpId>,

    /// Operations that are known to be illegal regardless of dialect settings.
    illegal_ops: FxHashSet<OpId>,
}

impl ConversionTarget {
    /// Create a new empty conversion target.
    pub fn new() -> Self {
        ConversionTarget {
            dialect_legality: FxHashMap::default(),
            op_legality: FxHashMap::default(),
            dynamic_legality_fns: FxHashMap::default(),
            legal_ops: FxHashSet::default(),
            illegal_ops: FxHashSet::default(),
        }
    }

    /// Mark an entire dialect as legal.
    ///
    /// All operations in this dialect will be considered legal unless
    /// explicitly marked otherwise.
    pub fn add_legal_dialect(&mut self, dialect: impl Into<DialectName>) {
        self.dialect_legality
            .insert(dialect.into(), LegalityAction::Legal);
    }

    /// Mark an entire dialect as illegal.
    ///
    /// All operations in this dialect must be converted.
    pub fn add_illegal_dialect(&mut self, dialect: impl Into<DialectName>) {
        self.dialect_legality
            .insert(dialect.into(), LegalityAction::Illegal);
    }

    /// Mark a specific operation as legal.
    ///
    /// This overrides any dialect-level setting.
    pub fn add_legal_op(&mut self, opid: OpId) {
        self.op_legality.insert(opid.clone(), LegalityAction::Legal);
        self.legal_ops.insert(opid);
    }

    /// Mark a specific operation as illegal.
    ///
    /// This overrides any dialect-level setting.
    pub fn add_illegal_op(&mut self, opid: OpId) {
        self.op_legality
            .insert(opid.clone(), LegalityAction::Illegal);
        self.illegal_ops.insert(opid);
    }

    /// Mark a specific operation as dynamically legal.
    ///
    /// The provided callback will be called for each instance to determine legality.
    pub fn add_dynamically_legal_op<F>(&mut self, opid: OpId, callback: F)
    where
        F: Fn(Ptr<Operation>, &Context) -> bool + Send + Sync + 'static,
    {
        self.op_legality
            .insert(opid.clone(), LegalityAction::Dynamic);
        self.dynamic_legality_fns.insert(opid, Box::new(callback));
    }

    /// Check if an operation is legal according to this target.
    pub fn is_legal(&self, op: Ptr<Operation>, ctx: &Context) -> bool {
        let opid = Operation::get_opid(op, ctx);

        // Check operation-specific legality first
        if let Some(action) = self.op_legality.get(&opid) {
            return match action {
                LegalityAction::Legal => true,
                LegalityAction::Illegal => false,
                LegalityAction::Dynamic => {
                    if let Some(callback) = self.dynamic_legality_fns.get(&opid) {
                        callback(op, ctx)
                    } else {
                        // No callback registered, assume legal
                        true
                    }
                }
            };
        }

        // Fall back to dialect-level legality
        if let Some(action) = self.dialect_legality.get(&opid.dialect) {
            return match action {
                LegalityAction::Legal => true,
                LegalityAction::Illegal => false,
                LegalityAction::Dynamic => true, // Dialect-level dynamic defaults to legal
            };
        }

        // Unknown operations are assumed legal (conservative)
        true
    }

    /// Check if an operation is illegal according to this target.
    pub fn is_illegal(&self, op: Ptr<Operation>, ctx: &Context) -> bool {
        !self.is_legal(op, ctx)
    }

    /// Get all operations that are explicitly marked as illegal.
    pub fn get_illegal_ops(&self) -> &FxHashSet<OpId> {
        &self.illegal_ops
    }

    /// Check if a dialect is marked as illegal.
    pub fn is_dialect_illegal(&self, dialect: &DialectName) -> bool {
        self.dialect_legality
            .get(dialect)
            .map(|a| *a == LegalityAction::Illegal)
            .unwrap_or(false)
    }
}

impl Default for ConversionTarget {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversion_target_dialect_legality() {
        let mut target = ConversionTarget::new();
        target.add_legal_dialect(DialectName::try_new("builtin").unwrap());
        target.add_illegal_dialect(DialectName::try_new("source").unwrap());

        assert!(!target.is_dialect_illegal(&DialectName::try_new("builtin").unwrap()));
        assert!(target.is_dialect_illegal(&DialectName::try_new("source").unwrap()));
        assert!(!target.is_dialect_illegal(&DialectName::try_new("unknown").unwrap()));
    }
}
