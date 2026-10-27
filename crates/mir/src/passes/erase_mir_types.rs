//! `erase-mir-types`: after `lower-dialect-mir`, substitute `llvm.ptr` for
//! every `mir.ptr` left in the module's types.
//!
//! crabbit's importer builds aggregate values with llvm-dialect ops
//! (`llvm.undef`, `llvm.insert_value`, `llvm.extract_value`, `llvm.bitcast`)
//! over llvm aggregate types whose pointer fields are `mir.ptr`. `mir-lower`
//! converts the types of the ops it rewrites (every `mir.*` op, function
//! signatures) and passes llvm ops through untouched, so `mir.ptr` survives
//! in those ops' result types, inside their aggregate types, in block
//! arguments, and in the callee types of calls to foreign symbols. The
//! representation is the same — the lowering maps `mir.ptr`
//! to the generic `llvm.ptr` for the ops it does convert — so this pass
//! rewrites the remaining types in place, which leaves a module in the LLVM
//! dialect alone. Named structs with a rewritten field take the name the
//! importer gives their declaration-side mirror (`<name>__llvm`), so the two
//! spellings meet.

use std::collections::HashMap;

use dialect_mir::types::MirPtrType;
use pliron::{
    basic_block::BasicBlock,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList as _,
    operation::Operation,
    builtin::{op_interfaces::CallOpInterface as _, type_interfaces::FunctionTypeInterface as _},
    r#type::{TypeHandle, Typed as _},
    result::Result,
};
use pliron_ll::conversion::pass::{AnalysisManager, Pass, PassResult, changed, unchanged};
use pliron_llvm::{
    ops::CallOp,
    types::{ArrayType, FuncType, PointerType, StructType},
};

/// The pass name.
pub const PASS_NAME: &str = "erase-mir-types";

pub struct EraseMirTypesPass;

impl Pass for EraseMirTypesPass {
    fn name(&self) -> &str {
        PASS_NAME
    }

    fn run(
        &mut self,
        root: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut eraser = Eraser::default();
        let rewritten = eraser.visit_op(ctx, root);
        Ok(if rewritten > 0 { changed() } else { unchanged() })
    }
}

/// Rewrites types, memoized per source type.
#[derive(Default)]
struct Eraser {
    memo: HashMap<TypeHandle, TypeHandle>,
}

impl Eraser {
    /// Rewrite the result and block-argument types under `op` (including
    /// `op`'s own results); returns the number of values retyped.
    fn visit_op(&mut self, ctx: &mut Context, op: Ptr<Operation>) -> usize {
        let mut rewritten = 0;
        let results: Vec<pliron::value::Value> = {
            let op_ref = op.deref(ctx);
            (0..op_ref.get_num_results()).map(|i| op_ref.get_result(i)).collect()
        };
        for result in results {
            rewritten += self.retype(ctx, result);
        }
        // A call's callee type is an attribute, not a value type: the
        // lowering builds it from the `mir.call`'s foreign signature.
        if let Some(call) = Operation::get_op::<CallOp>(op, ctx) {
            let ty = call.callee_type(ctx);
            let erased = self.erase(ctx, ty);
            if erased != ty {
                call.set_callee_type(ctx, erased);
                rewritten += 1;
            }
        }
        let blocks: Vec<Ptr<BasicBlock>> = op
            .deref(ctx)
            .regions()
            .flat_map(|region| region.deref(ctx).iter(ctx))
            .collect();
        for block in blocks {
            let args: Vec<pliron::value::Value> = block.deref(ctx).arguments().collect();
            for arg in args {
                rewritten += self.retype(ctx, arg);
            }
            let ops: Vec<Ptr<Operation>> = block.deref(ctx).iter(ctx).collect();
            for inner in ops {
                rewritten += self.visit_op(ctx, inner);
            }
        }
        rewritten
    }

    fn retype(&mut self, ctx: &mut Context, value: pliron::value::Value) -> usize {
        let ty = value.get_type(ctx);
        let erased = self.erase(ctx, ty);
        if erased == ty {
            return 0;
        }
        value.set_type(ctx, erased);
        1
    }

    /// `ty` with `llvm.ptr` for every `mir.ptr` inside it.
    fn erase(&mut self, ctx: &mut Context, ty: TypeHandle) -> TypeHandle {
        if let Some(done) = self.memo.get(&ty) {
            return *done;
        }
        let erased = self.erase_uncached(ctx, ty);
        self.memo.insert(ty, erased);
        erased
    }

    fn erase_uncached(&mut self, ctx: &mut Context, ty: TypeHandle) -> TypeHandle {
        enum Shape {
            MirPtr,
            Array(TypeHandle, u64),
            Struct(Option<pliron::identifier::Identifier>, Vec<TypeHandle>, pliron_llvm::types::StructLayout),
            Func(TypeHandle, Vec<TypeHandle>, bool),
            Other,
        }
        let shape = {
            let ty_ref = ty.deref(ctx);
            if ty_ref.is::<MirPtrType>() {
                Shape::MirPtr
            } else if let Some(array) = ty_ref.downcast_ref::<ArrayType>() {
                Shape::Array(array.elem_type(), array.size())
            } else if let Some(st) = ty_ref.downcast_ref::<StructType>() {
                if st.is_opaque() {
                    Shape::Other
                } else {
                    Shape::Struct(st.name(), st.fields().collect(), st.layout())
                }
            } else if let Some(func) = ty_ref.downcast_ref::<FuncType>() {
                Shape::Func(func.result_type(), func.arg_types().to_vec(), func.is_var_arg())
            } else {
                Shape::Other
            }
        };
        match shape {
            Shape::MirPtr => PointerType::get(ctx, 0).into(),
            Shape::Array(elem, size) => {
                let erased = self.erase(ctx, elem);
                if erased == elem {
                    ty
                } else {
                    ArrayType::get(ctx, erased, size).into()
                }
            }
            Shape::Struct(name, fields, layout) => {
                let erased: Vec<TypeHandle> = fields.iter().map(|f| self.erase(ctx, *f)).collect();
                if erased == fields {
                    return ty;
                }
                match name {
                    Some(name) => {
                        let mirror = format!("{name}__llvm")
                            .try_into()
                            .expect("suffixing a legal identifier keeps it legal");
                        StructType::get_named(ctx, mirror, Some((erased, layout)))
                            .expect("the __llvm mirror name maps 1:1 to this field set")
                            .into()
                    }
                    None => StructType::get_unnamed(ctx, (erased, layout)).into(),
                }
            }
            Shape::Func(result, args, variadic) => {
                let erased_result = self.erase(ctx, result);
                let erased_args: Vec<TypeHandle> = args.iter().map(|a| self.erase(ctx, *a)).collect();
                if erased_result == result && erased_args == args {
                    ty
                } else {
                    FuncType::get(ctx, erased_result, erased_args, variadic).into()
                }
            }
            Shape::Other => ty,
        }
    }
}
