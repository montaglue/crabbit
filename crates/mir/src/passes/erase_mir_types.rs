//! The types `mir-lower` leaves behind, and their LLVM-dialect spelling.
//!
//! crabbit's importer builds aggregate values with llvm-dialect ops
//! (`llvm.undef`, `llvm.insert_value`, `llvm.extract_value`, `llvm.bitcast`)
//! over llvm aggregate types whose fields are MIR-typed: `mir.ptr` pointers,
//! signed and unsigned integers, named structs. `mir-lower` converts the
//! types of the ops it rewrites (every `mir.*` op, function signatures) and
//! passes llvm ops through untouched, so the MIR types survive in those
//! ops' result types, inside their aggregate types, in block arguments, and
//! in types held by attributes (callee types of calls to foreign symbols,
//! global types, function types). [erase] rewrites them in place, right
//! after the lowering
//! ([LowerDialectMirPass](super::lower_dialect_mir::LowerDialectMirPass)),
//! which leaves a module in the LLVM dialect alone.
//!
//! [llvm_type] is the one mapping, and it is the lowering's own
//! ([mir_lower::convert::types::convert_type]: `llvm.ptr` for `mir.ptr`,
//! signless integers, unnamed structs), so what the lowering made and what
//! it left meet; the importer spells its declarations with it too.

use std::collections::HashMap;

use mir_lower::{convert::types::convert_type, type_conversion_interface::MirConvertibleType};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{IntegerAttr, TypeAttr},
        type_interfaces::FunctionTypeInterface as _,
        types::{IntegerType, Signedness},
    },
    context::{Context, Ptr},
    identifier::Identifier,
    linked_list::ContainsLinkedList as _,
    operation::Operation,
    result::Result,
    r#type::{TypeHandle, Typed as _, type_impls},
};
use pliron_llvm::types::{FuncType, StructType};

/// Rewrite every MIR type left under `root` to its [llvm_type]; returns the
/// number of values and attributes rewritten.
pub fn erase(ctx: &mut Context, root: Ptr<Operation>) -> Result<usize> {
    Eraser::default().visit_op(ctx, root)
}

/// The LLVM-dialect spelling of `ty`, as the lowering spells it. A type
/// with nothing to rewrite is returned as is.
pub fn llvm_type(ctx: &mut Context, ty: TypeHandle) -> Result<TypeHandle> {
    Eraser::default().erase(ctx, ty)
}

/// Rewrites types, memoized per source type.
#[derive(Default)]
struct Eraser {
    memo: HashMap<TypeHandle, TypeHandle>,
}

impl Eraser {
    /// Rewrite the result types, block-argument types and attribute-held
    /// types under `op` (including `op`'s own); returns the number rewritten.
    fn visit_op(&mut self, ctx: &mut Context, op: Ptr<Operation>) -> Result<usize> {
        let mut rewritten = 0;
        let results: Vec<pliron::value::Value> = {
            let op_ref = op.deref(ctx);
            (0..op_ref.get_num_results())
                .map(|i| op_ref.get_result(i))
                .collect()
        };
        for result in results {
            rewritten += self.retype(ctx, result)?;
        }
        rewritten += self.rewrite_attributes(ctx, op)?;
        let blocks: Vec<Ptr<BasicBlock>> = op
            .deref(ctx)
            .regions()
            .flat_map(|region| region.deref(ctx).iter(ctx))
            .collect();
        for block in blocks {
            let args: Vec<pliron::value::Value> = block.deref(ctx).arguments().collect();
            for arg in args {
                rewritten += self.retype(ctx, arg)?;
            }
            let ops: Vec<Ptr<Operation>> = block.deref(ctx).iter(ctx).collect();
            for inner in ops {
                rewritten += self.visit_op(ctx, inner)?;
            }
        }
        Ok(rewritten)
    }

    /// Types held by `op`'s attributes: [TypeAttr]s (callee, function,
    /// global, element types) and the types of [IntegerAttr]s.
    fn rewrite_attributes(&mut self, ctx: &mut Context, op: Ptr<Operation>) -> Result<usize> {
        let held: Vec<(Identifier, TypeHandle, Option<IntegerAttr>)> = {
            let attributes = &op.deref(ctx).attributes;
            attributes
                .0
                .iter()
                .filter_map(|(key, _)| {
                    if let Some(attr) = attributes.get::<TypeAttr>(key) {
                        Some((key.clone(), attr.get_type(ctx), None))
                    } else {
                        attributes
                            .get::<IntegerAttr>(key)
                            .map(|attr| (key.clone(), attr.get_type().into(), Some(attr.clone())))
                    }
                })
                .collect()
        };
        let mut rewritten = 0;
        for (key, ty, integer) in held {
            let erased = self.erase(ctx, ty)?;
            if erased == ty {
                continue;
            }
            match integer {
                None => op.deref_mut(ctx).attributes.set(key, TypeAttr::new(erased)),
                Some(attr) => {
                    let width = attr.get_type().deref(ctx).width();
                    let signless = IntegerType::get(ctx, width, Signedness::Signless);
                    op.deref_mut(ctx)
                        .attributes
                        .set(key, IntegerAttr::new(signless, attr.value()));
                }
            }
            rewritten += 1;
        }
        Ok(rewritten)
    }

    fn retype(&mut self, ctx: &mut Context, value: pliron::value::Value) -> Result<usize> {
        let ty = value.get_type(ctx);
        let erased = self.erase(ctx, ty)?;
        if erased == ty {
            return Ok(0);
        }
        value.set_type(ctx, erased);
        Ok(1)
    }

    fn erase(&mut self, ctx: &mut Context, ty: TypeHandle) -> Result<TypeHandle> {
        if let Some(done) = self.memo.get(&ty) {
            return Ok(*done);
        }
        let erased = self.erase_uncached(ctx, ty)?;
        self.memo.insert(ty, erased);
        Ok(erased)
    }

    fn erase_uncached(&mut self, ctx: &mut Context, ty: TypeHandle) -> Result<TypeHandle> {
        let (func, opaque, convertible) = {
            let ty_ref = ty.deref(ctx);
            let func = ty_ref.downcast_ref::<FuncType>().map(|func| {
                (
                    func.result_type(),
                    func.arg_types().to_vec(),
                    func.is_var_arg(),
                )
            });
            let opaque = ty_ref
                .downcast_ref::<StructType>()
                .is_some_and(|st| st.is_opaque());
            (func, opaque, type_impls::<dyn MirConvertibleType>(&*ty_ref))
        };
        // Function types are not the lowering's to convert here (it builds
        // signatures itself): their parts are.
        if let Some((result, args, variadic)) = func {
            let erased_result = self.erase(ctx, result)?;
            let erased_args = args
                .iter()
                .map(|arg| self.erase(ctx, *arg))
                .collect::<Result<Vec<_>>>()?;
            return Ok(if erased_result == result && erased_args == args {
                ty
            } else {
                FuncType::get(ctx, erased_result, erased_args, variadic).into()
            });
        }
        // An opaque struct has no body to convert.
        if opaque || !convertible {
            return Ok(ty);
        }
        convert_type(ctx, ty).map_err(|e| pliron::input_error_noloc!("{e}"))
    }
}
