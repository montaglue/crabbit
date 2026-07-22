//! Type Converter
//!
//! The TypeConverter handles type transformations during dialect conversion.
//! It manages how types from the source dialect map to types in the target dialect.
//!
//! See: https://mlir.llvm.org/docs/DialectConversion/

use rustc_hash::FxHashMap;

use crate::{
    context::Context,
    ir::r#type::TypeHandle,
};

/// Function type for converting a single type to one or more target types.
pub type TypeConversionFn =
    Box<dyn Fn(TypeHandle, &mut Context) -> Option<Vec<TypeHandle>> + Send + Sync>;

/// Function type for materializing a conversion (creating cast operations).
/// Returns the materialized value or None if materialization failed.
pub type MaterializationFn = Box<
    dyn Fn(
            &mut Context,
            TypeHandle,
            Vec<crate::ir::value::Value>,
        ) -> Option<crate::ir::value::Value>
        + Send
        + Sync,
>;

/// Handles type conversions during dialect conversion.
///
/// The TypeConverter serves several purposes:
/// 1. Define how source types map to target types (1-to-1 or 1-to-N)
/// 2. Provide materialization callbacks to create cast operations
/// 3. Convert block argument types in regions
///
/// # Example
/// ```ignore
/// let mut converter = TypeConverter::new();
///
/// // Source integer types -> index type
/// converter.add_conversion(|ty, ctx| {
///     if ty.deref(ctx).is::<SourceIntType>() {
///         Some(vec![IndexType::get_or_create(ctx).into()])
///     } else {
///         None
///     }
/// });
/// ```
pub struct TypeConverter {
    /// Type conversion functions, tried in order.
    conversions: Vec<TypeConversionFn>,

    /// Cache of already converted types.
    conversion_cache: FxHashMap<TypeHandle, Vec<TypeHandle>>,

    /// Source materialization callback.
    /// Called when a converted value needs to be converted back to the original type.
    source_materialization: Option<MaterializationFn>,

    /// Target materialization callback.
    /// Called when a value needs to be converted to the target type.
    target_materialization: Option<MaterializationFn>,

    /// Argument materialization callback.
    /// Called for block argument conversions.
    argument_materialization: Option<MaterializationFn>,
}

impl TypeConverter {
    /// Create a new empty type converter.
    pub fn new() -> Self {
        TypeConverter {
            conversions: Vec::new(),
            conversion_cache: FxHashMap::default(),
            source_materialization: None,
            target_materialization: None,
            argument_materialization: None,
        }
    }

    /// Add a type conversion function.
    ///
    /// Conversion functions are tried in order until one returns `Some`.
    /// A conversion function takes a source type and returns:
    /// - `Some(vec![...])` - The target type(s) to convert to
    /// - `None` - This converter doesn't handle this type
    pub fn add_conversion<F>(&mut self, conversion: F)
    where
        F: Fn(TypeHandle, &mut Context) -> Option<Vec<TypeHandle>> + Send + Sync + 'static,
    {
        self.conversions.push(Box::new(conversion));
    }

    /// Set the source materialization callback.
    ///
    /// This is called when a converted value needs to be used where the
    /// original type is expected (e.g., passing to an unconverted operation).
    pub fn set_source_materialization<F>(&mut self, materialization: F)
    where
        F: Fn(
                &mut Context,
                TypeHandle,
                Vec<crate::ir::value::Value>,
            ) -> Option<crate::ir::value::Value>
            + Send
            + Sync
            + 'static,
    {
        self.source_materialization = Some(Box::new(materialization));
    }

    /// Set the target materialization callback.
    ///
    /// This is called when an unconverted value needs to be used where
    /// the target type is expected.
    pub fn set_target_materialization<F>(&mut self, materialization: F)
    where
        F: Fn(
                &mut Context,
                TypeHandle,
                Vec<crate::ir::value::Value>,
            ) -> Option<crate::ir::value::Value>
            + Send
            + Sync
            + 'static,
    {
        self.target_materialization = Some(Box::new(materialization));
    }

    /// Set the argument materialization callback.
    ///
    /// This is called when converting block arguments.
    pub fn set_argument_materialization<F>(&mut self, materialization: F)
    where
        F: Fn(
                &mut Context,
                TypeHandle,
                Vec<crate::ir::value::Value>,
            ) -> Option<crate::ir::value::Value>
            + Send
            + Sync
            + 'static,
    {
        self.argument_materialization = Some(Box::new(materialization));
    }

    /// Convert a type to its target representation.
    ///
    /// Returns `None` if no conversion is registered for this type,
    /// meaning the type should remain unchanged.
    pub fn convert_type(
        &mut self,
        ty: TypeHandle,
        ctx: &mut Context,
    ) -> Option<Vec<TypeHandle>> {
        // Check cache first
        if let Some(cached) = self.conversion_cache.get(&ty) {
            return Some(cached.clone());
        }

        // Try each conversion function
        for conversion in &self.conversions {
            if let Some(result) = conversion(ty, ctx) {
                self.conversion_cache.insert(ty, result.clone());
                return Some(result);
            }
        }

        None
    }

    /// Check if a type needs conversion.
    #[allow(unused_variables)]
    pub fn is_type_legal(&self, ty: TypeHandle, ctx: &Context) -> bool {
        // A type is legal if no conversion is registered for it
        // This is a simplified check - in MLIR, legal types are explicitly tracked
        // We can't actually check without mutable context, so we just return true
        // In a full implementation, we'd track legal types separately
        true
    }

    /// Convert a list of types.
    pub fn convert_types(
        &mut self,
        types: &[TypeHandle],
        ctx: &mut Context,
    ) -> Option<Vec<TypeHandle>> {
        let mut result = Vec::new();
        for &ty in types {
            if let Some(converted) = self.convert_type(ty, ctx) {
                result.extend(converted);
            } else {
                // Type doesn't need conversion, keep as-is
                result.push(ty);
            }
        }
        Some(result)
    }

    /// Materialize a source conversion (target -> source).
    pub fn materialize_source_conversion(
        &self,
        ctx: &mut Context,
        result_type: TypeHandle,
        inputs: Vec<crate::ir::value::Value>,
    ) -> Option<crate::ir::value::Value> {
        self.source_materialization
            .as_ref()
            .and_then(|f| f(ctx, result_type, inputs))
    }

    /// Materialize a target conversion (source -> target).
    pub fn materialize_target_conversion(
        &self,
        ctx: &mut Context,
        result_type: TypeHandle,
        inputs: Vec<crate::ir::value::Value>,
    ) -> Option<crate::ir::value::Value> {
        self.target_materialization
            .as_ref()
            .and_then(|f| f(ctx, result_type, inputs))
    }

    /// Materialize an argument conversion.
    pub fn materialize_argument_conversion(
        &self,
        ctx: &mut Context,
        result_type: TypeHandle,
        inputs: Vec<crate::ir::value::Value>,
    ) -> Option<crate::ir::value::Value> {
        self.argument_materialization
            .as_ref()
            .and_then(|f| f(ctx, result_type, inputs))
    }
}

impl Default for TypeConverter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_converter_creation() {
        let converter = TypeConverter::new();
        assert!(converter.conversions.is_empty());
    }
}
