//! Convert Rust MIR dialect operations to the LLVM dialect.

use std::cmp::Ordering;

use awint::bw;

use crate::{
    context::{Context, Ptr},
    dialects::{
        builtin::{
            attributes::{IntegerAttr, TypeAttr},
            op_interfaces::{OneRegionInterface, OneResultInterface, SymbolOpInterface},
            type_interfaces::FunctionTypeInterface,
            types::{FunctionType, IntegerType, Signedness},
        },
        llvm::{
            attributes::{GepIndexAttr, GepIndicesAttr, ICmpPredicateAttr, LinkageAttr},
            ops as llvm_ops,
            types::{ArrayType, FuncType, PointerType, StructType, VoidType},
        },
        mir::{ops as mir_ops, types::PtrType as MirPtrType},
    },
    ir::{
        basic_block::BasicBlock,
        dialect::DialectName,
        op::Op,
        operation::Operation,
        r#type::{TypeHandle, TypedHandle, Typed},
        value::Value,
    },
    linked_list::ContainsLinkedList,
    conversion::{
        conversion_pattern::{ConversionPattern, ConversionPatternSet, PatternMatchResult},
        conversion_target::ConversionTarget,
        dialect_conversion::apply_partial_conversion,
        pass::{AnalysisManager, Pass, PassResult, changed},
        rewriter::ConversionPatternRewriter,
    },
    result::STAIRResult,
    utils::apint::APInt,
};

// ============================================================================
// Type conversion
// ============================================================================

fn convert_type(ctx: &mut Context, ty: TypeHandle) -> TypeHandle {
    let ty_ref = ty.deref(ctx);
    if ty_ref.is::<MirPtrType>() {
        drop(ty_ref);
        return PointerType::get(ctx).into();
    }
    if let Some(array_ty) = ty_ref.downcast_ref::<ArrayType>() {
        let elem = array_ty.elem_type();
        let size = array_ty.size();
        drop(ty_ref);
        let elem = convert_type(ctx, elem);
        return ArrayType::get(ctx, elem, size).into();
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<StructType>() {
        let name = struct_ty.name().cloned();
        let fields = struct_ty.fields().map(|fields| fields.to_vec());
        drop(ty_ref);
        let fields = fields.map(|fields| {
            fields
                .into_iter()
                .map(|field| convert_type(ctx, field))
                .collect::<Vec<_>>()
        });
        return match name {
            Some(name) => {
                let name = format!("{name}__llvm").try_into().unwrap();
                StructType::get_named(ctx, name, fields).unwrap().into()
            }
            None => StructType::get_unnamed(ctx, fields.unwrap_or_default()).into(),
        };
    }
    if let Some(func_ty) = ty_ref.downcast_ref::<FunctionType>() {
        let arg_types = func_ty.arg_types();
        let results = func_ty.res_types();
        drop(ty_ref);

        let inputs = arg_types
            .into_iter()
            .map(|arg| convert_type(ctx, arg))
            .collect();
        let result_ty = if results.is_empty() {
            VoidType::get(ctx).into()
        } else {
            convert_type(ctx, results[0])
        };
        return FuncType::get(ctx, result_ty, inputs, false).into();
    }
    ty
}

fn convert_function_type(ctx: &mut Context, ty: TypeHandle) -> TypedHandle<FuncType> {
    let ty_ref = ty.deref(ctx);
    let func_ty = ty_ref
        .downcast_ref::<FunctionType>()
        .expect("mir.func must carry a builtin.function type");
    let arg_types = func_ty.arg_types();
    let results = func_ty.res_types();
    drop(ty_ref);

    let inputs = arg_types
        .into_iter()
        .map(|arg| convert_type(ctx, arg))
        .collect();
    let result_ty = if results.is_empty() {
        VoidType::get(ctx).into()
    } else {
        convert_type(ctx, results[0])
    };
    FuncType::get(ctx, result_ty, inputs, false)
}

fn convert_block_arg_types(root: Ptr<Operation>, ctx: &mut Context) {
    let regions: Vec<_> = root.deref(ctx).regions().collect();
    for region in regions {
        let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
        for block in blocks {
            // Block argument types can't be mutated in place, so append a
            // freshly typed argument per old one, move its uses over, then
            // drop the old arguments.
            let old_args: Vec<Value> = block.deref(ctx).arguments().collect();
            let new_types: Vec<TypeHandle> = old_args
                .iter()
                .map(|arg| convert_type(ctx, arg.get_type(ctx)))
                .collect();
            let new_args: Vec<Value> = new_types
                .into_iter()
                .map(|new_ty| {
                    let new_idx = BasicBlock::push_argument(block, ctx, new_ty);
                    block.deref(ctx).get_argument(new_idx)
                })
                .collect();
            for (old_arg, new_arg) in old_args.iter().zip(&new_args) {
                old_arg.replace_all_uses_with(ctx, new_arg);
            }
            for _ in 0..old_args.len() {
                BasicBlock::remove_argument(block, ctx, 0);
            }
        }
    }
}

// ============================================================================
// Function conversion
// ============================================================================

struct FuncToLLVM;

impl ConversionPattern for FuncToLLVM {
    fn get_root_kind(&self) -> crate::ir::op::OpId {
        mir_ops::FuncOp::get_opid_static()
    }

    fn match_and_rewrite(
        &self,
        op: Ptr<Operation>,
        _operands: &[Value],
        rewriter: &mut ConversionPatternRewriter,
        ctx: &mut Context,
    ) -> STAIRResult<PatternMatchResult> {
        let mir_func = mir_ops::FuncOp::from_operation(op);
        let name = mir_func.get_symbol_name(ctx);
        let llvm_func_ty = convert_function_type(ctx, mir_func.get_func_type(ctx));
        // A `llvm_linkage` attribute pre-set on the mir.func (e.g. by the
        // Rust importer for upstream instances) is carried over.
        let linkage = op
            .deref(ctx)
            .attributes
            .get::<LinkageAttr>(&llvm_ops::ATTR_KEY_LLVM_LINKAGE)
            .cloned()
            .unwrap_or(LinkageAttr::External);

        let new_op = Operation::new(
            ctx,
            llvm_ops::FuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );
        let old_region = mir_func.get_region(ctx);
        crate::ir::region::Region::move_to_op(old_region, new_op, ctx);

        let llvm_func = llvm_ops::FuncOp::from_operation(new_op);
        llvm_func.set_symbol_name(ctx, name);
        {
            let mut op_ref = llvm_func.get_operation().deref_mut(ctx);
            op_ref.attributes.set(
                llvm_ops::ATTR_KEY_LLVM_FUNC_TYPE.clone(),
                TypeAttr::new(llvm_func_ty.into()),
            );
            op_ref
                .attributes
                .set(llvm_ops::ATTR_KEY_LLVM_LINKAGE.clone(), linkage);
        }
        convert_block_arg_types(new_op, ctx);

        rewriter.insert(new_op, ctx);
        rewriter.replace_op(op, vec![], ctx);
        Ok(PatternMatchResult::Success)
    }
}

// ============================================================================
// Simple one-to-one operation conversions
// ============================================================================

macro_rules! def_binary_conversion {
    ($name:ident, $src:ty, $dst:ty) => {
        struct $name;

        impl ConversionPattern for $name {
            fn get_root_kind(&self) -> crate::ir::op::OpId {
                <$src>::get_opid_static()
            }

            fn match_and_rewrite(
                &self,
                op: Ptr<Operation>,
                operands: &[Value],
                rewriter: &mut ConversionPatternRewriter,
                ctx: &mut Context,
            ) -> STAIRResult<PatternMatchResult> {
                let new_op = <$dst>::new(ctx, operands[0], operands[1]);
                rewriter.insert(new_op.get_operation(), ctx);
                rewriter.replace_op(op, vec![new_op.get_result(ctx)], ctx);
                Ok(PatternMatchResult::Success)
            }
        }
    };
}

macro_rules! def_cmp_conversion {
    ($name:ident, $src:ty, $pred:expr) => {
        struct $name;

        impl ConversionPattern for $name {
            fn get_root_kind(&self) -> crate::ir::op::OpId {
                <$src>::get_opid_static()
            }

            fn match_and_rewrite(
                &self,
                op: Ptr<Operation>,
                operands: &[Value],
                rewriter: &mut ConversionPatternRewriter,
                ctx: &mut Context,
            ) -> STAIRResult<PatternMatchResult> {
                let new_op = llvm_ops::ICmpOp::new(ctx, $pred, operands[0], operands[1]);
                rewriter.insert(new_op.get_operation(), ctx);
                rewriter.replace_op(op, vec![new_op.get_result(ctx)], ctx);
                Ok(PatternMatchResult::Success)
            }
        }
    };
}

def_binary_conversion!(AddToLLVM, mir_ops::AddOp, llvm_ops::AddOp);
def_binary_conversion!(SubToLLVM, mir_ops::SubOp, llvm_ops::SubOp);
def_binary_conversion!(MulToLLVM, mir_ops::MulOp, llvm_ops::MulOp);
def_binary_conversion!(ShlToLLVM, mir_ops::ShlOp, llvm_ops::ShlOp);
def_binary_conversion!(ShrToLLVM, mir_ops::ShrOp, llvm_ops::LShrOp);
def_binary_conversion!(BitAndToLLVM, mir_ops::BitAndOp, llvm_ops::AndOp);
def_binary_conversion!(BitOrToLLVM, mir_ops::BitOrOp, llvm_ops::OrOp);
def_binary_conversion!(BitXorToLLVM, mir_ops::BitXorOp, llvm_ops::XorOp);

struct DivToLLVM;

impl ConversionPattern for DivToLLVM {
    fn get_root_kind(&self) -> crate::ir::op::OpId {
        mir_ops::DivOp::get_opid_static()
    }

    fn match_and_rewrite(
        &self,
        op: Ptr<Operation>,
        operands: &[Value],
        rewriter: &mut ConversionPatternRewriter,
        ctx: &mut Context,
    ) -> STAIRResult<PatternMatchResult> {
        let signedness = integer_signedness(ctx, operands[0])?;
        let new_op = if signedness == Signedness::Unsigned {
            llvm_ops::UDivOp::new(ctx, operands[0], operands[1]).get_operation()
        } else {
            llvm_ops::SDivOp::new(ctx, operands[0], operands[1]).get_operation()
        };
        let result = new_op.deref(ctx).get_result(0);
        rewriter.insert(new_op, ctx);
        rewriter.replace_op(op, vec![result], ctx);
        Ok(PatternMatchResult::Success)
    }
}

struct RemToLLVM;

impl ConversionPattern for RemToLLVM {
    fn get_root_kind(&self) -> crate::ir::op::OpId {
        mir_ops::RemOp::get_opid_static()
    }

    fn match_and_rewrite(
        &self,
        op: Ptr<Operation>,
        operands: &[Value],
        rewriter: &mut ConversionPatternRewriter,
        ctx: &mut Context,
    ) -> STAIRResult<PatternMatchResult> {
        let signedness = integer_signedness(ctx, operands[0])?;
        let new_op = if signedness == Signedness::Unsigned {
            llvm_ops::URemOp::new(ctx, operands[0], operands[1]).get_operation()
        } else {
            llvm_ops::SRemOp::new(ctx, operands[0], operands[1]).get_operation()
        };
        let result = new_op.deref(ctx).get_result(0);
        rewriter.insert(new_op, ctx);
        rewriter.replace_op(op, vec![result], ctx);
        Ok(PatternMatchResult::Success)
    }
}

fn integer_signedness(ctx: &Context, value: Value) -> STAIRResult<Signedness> {
    let ty = value.get_type(ctx);
    let ty_ref = ty.deref(ctx);
    let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() else {
        return Ok(Signedness::Signless);
    };
    Ok(int_ty.signedness())
}

def_cmp_conversion!(EqToLLVM, mir_ops::EqOp, ICmpPredicateAttr::EQ);
def_cmp_conversion!(NeToLLVM, mir_ops::NeOp, ICmpPredicateAttr::NE);

macro_rules! def_ordered_cmp_conversion {
    ($name:ident, $src:ty, $signed_pred:expr, $unsigned_pred:expr) => {
        struct $name;

        impl ConversionPattern for $name {
            fn get_root_kind(&self) -> crate::ir::op::OpId {
                <$src>::get_opid_static()
            }

            fn match_and_rewrite(
                &self,
                op: Ptr<Operation>,
                operands: &[Value],
                rewriter: &mut ConversionPatternRewriter,
                ctx: &mut Context,
            ) -> STAIRResult<PatternMatchResult> {
                let pred = if integer_signedness(ctx, operands[0])? == Signedness::Unsigned {
                    $unsigned_pred
                } else {
                    $signed_pred
                };
                let new_op = llvm_ops::ICmpOp::new(ctx, pred, operands[0], operands[1]);
                rewriter.insert(new_op.get_operation(), ctx);
                rewriter.replace_op(op, vec![new_op.get_result(ctx)], ctx);
                Ok(PatternMatchResult::Success)
            }
        }
    };
}

def_ordered_cmp_conversion!(
    LtToLLVM,
    mir_ops::LtOp,
    ICmpPredicateAttr::SLT,
    ICmpPredicateAttr::ULT
);
def_ordered_cmp_conversion!(
    LeToLLVM,
    mir_ops::LeOp,
    ICmpPredicateAttr::SLE,
    ICmpPredicateAttr::ULE
);
def_ordered_cmp_conversion!(
    GtToLLVM,
    mir_ops::GtOp,
    ICmpPredicateAttr::SGT,
    ICmpPredicateAttr::UGT
);
def_ordered_cmp_conversion!(
    GeToLLVM,
    mir_ops::GeOp,
    ICmpPredicateAttr::SGE,
    ICmpPredicateAttr::UGE
);

struct ConstantToLLVM;

impl ConversionPattern for ConstantToLLVM {
    fn get_root_kind(&self) -> crate::ir::op::OpId {
        mir_ops::ConstantOp::get_opid_static()
    }

    fn match_and_rewrite(
        &self,
        op: Ptr<Operation>,
        _operands: &[Value],
        rewriter: &mut ConversionPatternRewriter,
        ctx: &mut Context,
    ) -> STAIRResult<PatternMatchResult> {
        let attr = mir_ops::ConstantOp::from_operation(op).get_value(ctx).unwrap();
        let new_op = llvm_ops::ConstantOp::new_integer(ctx, attr);
        rewriter.insert(new_op.get_operation(), ctx);
        rewriter.replace_op(op, vec![new_op.get_result(ctx)], ctx);
        Ok(PatternMatchResult::Success)
    }
}

struct CStrToLLVM;

impl ConversionPattern for CStrToLLVM {
    fn get_root_kind(&self) -> crate::ir::op::OpId {
        mir_ops::CStrOp::get_opid_static()
    }

    fn match_and_rewrite(
        &self,
        op: Ptr<Operation>,
        _operands: &[Value],
        rewriter: &mut ConversionPatternRewriter,
        ctx: &mut Context,
    ) -> STAIRResult<PatternMatchResult> {
        let new_op = llvm_ops::CStrOp::new(ctx, mir_ops::CStrOp::from_operation(op).get_value(ctx));
        rewriter.insert(new_op.get_operation(), ctx);
        rewriter.replace_op(op, vec![new_op.get_result(ctx)], ctx);
        Ok(PatternMatchResult::Success)
    }
}

struct AddressOfToLLVM;

impl ConversionPattern for AddressOfToLLVM {
    fn get_root_kind(&self) -> crate::ir::op::OpId {
        mir_ops::AddressOfOp::get_opid_static()
    }

    fn match_and_rewrite(
        &self,
        op: Ptr<Operation>,
        _operands: &[Value],
        rewriter: &mut ConversionPatternRewriter,
        ctx: &mut Context,
    ) -> STAIRResult<PatternMatchResult> {
        let new_op = llvm_ops::AddressOfOp::new(ctx, mir_ops::AddressOfOp::from_operation(op).get_symbol(ctx));
        rewriter.insert(new_op.get_operation(), ctx);
        rewriter.replace_op(op, vec![new_op.get_result(ctx)], ctx);
        Ok(PatternMatchResult::Success)
    }
}

struct UndefToLLVM;

impl ConversionPattern for UndefToLLVM {
    fn get_root_kind(&self) -> crate::ir::op::OpId {
        mir_ops::UndefOp::get_opid_static()
    }

    fn match_and_rewrite(
        &self,
        op: Ptr<Operation>,
        _operands: &[Value],
        rewriter: &mut ConversionPatternRewriter,
        ctx: &mut Context,
    ) -> STAIRResult<PatternMatchResult> {
        let result_ty = {
            let op_ref = op.deref(ctx);
            op_ref.get_result(0).get_type(ctx)
        };
        let result_ty = convert_type(ctx, result_ty);
        let new_op = llvm_ops::UndefOp::new(ctx, result_ty);
        rewriter.insert(new_op.get_operation(), ctx);
        rewriter.replace_op(op, vec![new_op.get_result(ctx)], ctx);
        Ok(PatternMatchResult::Success)
    }
}

struct ExtractValueToLLVM;

impl ConversionPattern for ExtractValueToLLVM {
    fn get_root_kind(&self) -> crate::ir::op::OpId {
        mir_ops::ExtractValueOp::get_opid_static()
    }

    fn match_and_rewrite(
        &self,
        op: Ptr<Operation>,
        operands: &[Value],
        rewriter: &mut ConversionPatternRewriter,
        ctx: &mut Context,
    ) -> STAIRResult<PatternMatchResult> {
        let mir_op = mir_ops::ExtractValueOp::from_operation(op);
        let result_ty = {
            let op_ref = op.deref(ctx);
            op_ref.get_result(0).get_type(ctx)
        };
        let result_ty = convert_type(ctx, result_ty);
        let new_op =
            llvm_ops::ExtractValueOp::new(ctx, operands[0], mir_op.get_indices(ctx), result_ty);
        rewriter.insert(new_op.get_operation(), ctx);
        rewriter.replace_op(op, vec![new_op.get_result(ctx)], ctx);
        Ok(PatternMatchResult::Success)
    }
}

struct InsertValueToLLVM;

impl ConversionPattern for InsertValueToLLVM {
    fn get_root_kind(&self) -> crate::ir::op::OpId {
        mir_ops::InsertValueOp::get_opid_static()
    }

    fn match_and_rewrite(
        &self,
        op: Ptr<Operation>,
        operands: &[Value],
        rewriter: &mut ConversionPatternRewriter,
        ctx: &mut Context,
    ) -> STAIRResult<PatternMatchResult> {
        let mir_op = mir_ops::InsertValueOp::from_operation(op);
        let new_op =
            llvm_ops::InsertValueOp::new(ctx, operands[0], operands[1], mir_op.get_indices(ctx));
        rewriter.insert(new_op.get_operation(), ctx);
        rewriter.replace_op(op, vec![new_op.get_result(ctx)], ctx);
        Ok(PatternMatchResult::Success)
    }
}

struct CastToLLVM;

impl ConversionPattern for CastToLLVM {
    fn get_root_kind(&self) -> crate::ir::op::OpId {
        mir_ops::CastOp::get_opid_static()
    }

    fn match_and_rewrite(
        &self,
        op: Ptr<Operation>,
        operands: &[Value],
        rewriter: &mut ConversionPatternRewriter,
        ctx: &mut Context,
    ) -> STAIRResult<PatternMatchResult> {
        let input_ty = convert_type(ctx, operands[0].get_type(ctx));
        let result_ty = {
            let op_ref = op.deref(ctx);
            op_ref.get_result(0).get_type(ctx)
        };
        let result_ty = convert_type(ctx, result_ty);

        if input_ty == result_ty {
            rewriter.replace_op(op, vec![operands[0]], ctx);
            return Ok(PatternMatchResult::Success);
        }

        let input_is_int = input_ty.deref(ctx).is::<IntegerType>();
        let input_is_ptr = input_ty.deref(ctx).is::<PointerType>();
        let result_is_int = result_ty.deref(ctx).is::<IntegerType>();
        let result_is_ptr = result_ty.deref(ctx).is::<PointerType>();

        let new_op = if input_is_int && result_is_ptr {
            llvm_ops::IntToPtrOp::new(ctx, operands[0], result_ty).get_operation()
        } else if input_is_ptr && result_is_int {
            llvm_ops::PtrToIntOp::new(ctx, operands[0], result_ty).get_operation()
        } else if input_is_int && result_is_int {
            let input_width = input_ty
                .deref(ctx)
                .downcast_ref::<IntegerType>()
                .unwrap()
                .width();
            let result_width = result_ty
                .deref(ctx)
                .downcast_ref::<IntegerType>()
                .unwrap()
                .width();
            match input_width.cmp(&result_width) {
                Ordering::Equal => {
                    rewriter.replace_op(op, vec![operands[0]], ctx);
                    return Ok(PatternMatchResult::Success);
                }
                Ordering::Less => {
                    llvm_ops::ZExtOp::new(ctx, operands[0], result_ty).get_operation()
                }
                Ordering::Greater => {
                    llvm_ops::TruncOp::new(ctx, operands[0], result_ty).get_operation()
                }
            }
        } else {
            llvm_ops::BitcastOp::new(ctx, operands[0], result_ty).get_operation()
        };

        let replacement = new_op.deref(ctx).get_result(0);
        rewriter.insert(new_op, ctx);
        rewriter.replace_op(op, vec![replacement], ctx);
        Ok(PatternMatchResult::Success)
    }
}

struct AllocaToLLVM;

impl ConversionPattern for AllocaToLLVM {
    fn get_root_kind(&self) -> crate::ir::op::OpId {
        mir_ops::AllocaOp::get_opid_static()
    }

    fn match_and_rewrite(
        &self,
        op: Ptr<Operation>,
        _operands: &[Value],
        rewriter: &mut ConversionPatternRewriter,
        ctx: &mut Context,
    ) -> STAIRResult<PatternMatchResult> {
        let elem_ty = convert_type(ctx, mir_ops::AllocaOp::from_operation(op).get_elem_type(ctx));
        let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
        let one = IntegerAttr::new(i64_ty, APInt::from_u64(1, bw(64)));
        let one_op = llvm_ops::ConstantOp::new_integer(ctx, one);
        rewriter.insert(one_op.get_operation(), ctx);

        let new_op = llvm_ops::AllocaOp::new(ctx, one_op.get_result(ctx), elem_ty);
        rewriter.insert(new_op.get_operation(), ctx);
        rewriter.replace_op(op, vec![new_op.get_result(ctx)], ctx);
        Ok(PatternMatchResult::Success)
    }
}

struct LoadToLLVM;

impl ConversionPattern for LoadToLLVM {
    fn get_root_kind(&self) -> crate::ir::op::OpId {
        mir_ops::LoadOp::get_opid_static()
    }

    fn match_and_rewrite(
        &self,
        op: Ptr<Operation>,
        operands: &[Value],
        rewriter: &mut ConversionPatternRewriter,
        ctx: &mut Context,
    ) -> STAIRResult<PatternMatchResult> {
        let result_ty = {
            let op_ref = op.deref(ctx);
            op_ref.get_result(0).get_type(ctx)
        };
        let result_ty = convert_type(ctx, result_ty);
        let new_op = llvm_ops::LoadOp::new(ctx, operands[0], result_ty);
        rewriter.insert(new_op.get_operation(), ctx);
        rewriter.replace_op(op, vec![new_op.get_result(ctx)], ctx);
        Ok(PatternMatchResult::Success)
    }
}

struct StoreToLLVM;

impl ConversionPattern for StoreToLLVM {
    fn get_root_kind(&self) -> crate::ir::op::OpId {
        mir_ops::StoreOp::get_opid_static()
    }

    fn match_and_rewrite(
        &self,
        op: Ptr<Operation>,
        operands: &[Value],
        rewriter: &mut ConversionPatternRewriter,
        ctx: &mut Context,
    ) -> STAIRResult<PatternMatchResult> {
        let new_op = llvm_ops::StoreOp::new(ctx, operands[0], operands[1]);
        rewriter.insert(new_op.get_operation(), ctx);
        rewriter.replace_op(op, vec![], ctx);
        Ok(PatternMatchResult::Success)
    }
}

struct PtrOffsetToLLVM;

impl ConversionPattern for PtrOffsetToLLVM {
    fn get_root_kind(&self) -> crate::ir::op::OpId {
        mir_ops::PtrOffsetOp::get_opid_static()
    }

    fn match_and_rewrite(
        &self,
        op: Ptr<Operation>,
        operands: &[Value],
        rewriter: &mut ConversionPatternRewriter,
        ctx: &mut Context,
    ) -> STAIRResult<PatternMatchResult> {
        let i8_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
        let new_op = llvm_ops::GetElementPtrOp::new(
            ctx,
            operands[0],
            vec![operands[1]],
            GepIndicesAttr(vec![GepIndexAttr::OperandIdx(0)]),
            i8_ty,
        );
        rewriter.insert(new_op.get_operation(), ctx);
        rewriter.replace_op(op, vec![new_op.get_result(ctx)], ctx);
        Ok(PatternMatchResult::Success)
    }
}

struct ReturnToLLVM;

impl ConversionPattern for ReturnToLLVM {
    fn get_root_kind(&self) -> crate::ir::op::OpId {
        mir_ops::ReturnOp::get_opid_static()
    }

    fn match_and_rewrite(
        &self,
        op: Ptr<Operation>,
        operands: &[Value],
        rewriter: &mut ConversionPatternRewriter,
        ctx: &mut Context,
    ) -> STAIRResult<PatternMatchResult> {
        let new_op = llvm_ops::ReturnOp::new(ctx, operands.first().copied());
        rewriter.insert(new_op.get_operation(), ctx);
        rewriter.replace_op(op, vec![], ctx);
        Ok(PatternMatchResult::Success)
    }
}

struct GotoToLLVM;

impl ConversionPattern for GotoToLLVM {
    fn get_root_kind(&self) -> crate::ir::op::OpId {
        mir_ops::GotoOp::get_opid_static()
    }

    fn match_and_rewrite(
        &self,
        op: Ptr<Operation>,
        operands: &[Value],
        rewriter: &mut ConversionPatternRewriter,
        ctx: &mut Context,
    ) -> STAIRResult<PatternMatchResult> {
        let mir_op = mir_ops::GotoOp::from_operation(op);
        let new_op = llvm_ops::BrOp::new(ctx, mir_op.get_dest(ctx), operands.to_vec());
        rewriter.insert(new_op.get_operation(), ctx);
        rewriter.replace_op(op, vec![], ctx);
        Ok(PatternMatchResult::Success)
    }
}

struct CondBrToLLVM;

impl ConversionPattern for CondBrToLLVM {
    fn get_root_kind(&self) -> crate::ir::op::OpId {
        mir_ops::CondBrOp::get_opid_static()
    }

    fn match_and_rewrite(
        &self,
        op: Ptr<Operation>,
        _operands: &[Value],
        rewriter: &mut ConversionPatternRewriter,
        ctx: &mut Context,
    ) -> STAIRResult<PatternMatchResult> {
        let mir_op = mir_ops::CondBrOp::from_operation(op);
        let condition = rewriter.get_remapped_value(mir_op.get_condition(ctx));
        let true_operands = mir_op
            .get_true_operands(ctx)
            .into_iter()
            .map(|value| rewriter.get_remapped_value(value))
            .collect();
        let false_operands = mir_op
            .get_false_operands(ctx)
            .into_iter()
            .map(|value| rewriter.get_remapped_value(value))
            .collect();
        let new_op = llvm_ops::CondBrOp::new(
            ctx,
            condition,
            mir_op.get_true_dest(ctx),
            true_operands,
            mir_op.get_false_dest(ctx),
            false_operands,
        );
        rewriter.insert(new_op.get_operation(), ctx);
        rewriter.replace_op(op, vec![], ctx);
        Ok(PatternMatchResult::Success)
    }
}

struct UnreachableToLLVM;

impl ConversionPattern for UnreachableToLLVM {
    fn get_root_kind(&self) -> crate::ir::op::OpId {
        mir_ops::UnreachableOp::get_opid_static()
    }

    fn match_and_rewrite(
        &self,
        op: Ptr<Operation>,
        _operands: &[Value],
        rewriter: &mut ConversionPatternRewriter,
        ctx: &mut Context,
    ) -> STAIRResult<PatternMatchResult> {
        let new_op = llvm_ops::UnreachableOp::new(ctx);
        rewriter.insert(new_op.get_operation(), ctx);
        rewriter.replace_op(op, vec![], ctx);
        Ok(PatternMatchResult::Success)
    }
}

struct CallToLLVM;

impl ConversionPattern for CallToLLVM {
    fn get_root_kind(&self) -> crate::ir::op::OpId {
        mir_ops::CallOp::get_opid_static()
    }

    fn match_and_rewrite(
        &self,
        op: Ptr<Operation>,
        operands: &[Value],
        rewriter: &mut ConversionPatternRewriter,
        ctx: &mut Context,
    ) -> STAIRResult<PatternMatchResult> {
        let mir_op = mir_ops::CallOp::from_operation(op);
        let result_type = if op.deref(ctx).get_num_results() > 0 {
            let result_ty = {
                let op_ref = op.deref(ctx);
                op_ref.get_result(0).get_type(ctx)
            };
            Some(convert_type(ctx, result_ty))
        } else {
            None
        };
        let new_op = match mir_op.get_callee_opt(ctx) {
            Some(callee) => {
                llvm_ops::CallOp::new_direct(ctx, callee, operands.to_vec(), result_type)
            }
            None => {
                let (callee, args) = operands.split_first().ok_or_else(|| {
                    crate::input_error_noloc!("indirect mir.call has no callee operand")
                })?;
                llvm_ops::CallOp::new_indirect(ctx, *callee, args.to_vec(), result_type)
            }
        };
        let replacements = new_op.get_operation().deref(ctx).results().collect();
        rewriter.insert(new_op.get_operation(), ctx);
        rewriter.replace_op(op, replacements, ctx);
        Ok(PatternMatchResult::Success)
    }
}

/// Populate MIR-to-LLVM conversion patterns.
pub fn populate_mir_to_llvm_patterns(patterns: &mut ConversionPatternSet) {
    patterns.add(FuncToLLVM);
    patterns.add(ConstantToLLVM);
    patterns.add(CStrToLLVM);
    patterns.add(AddressOfToLLVM);
    patterns.add(UndefToLLVM);
    patterns.add(AllocaToLLVM);
    patterns.add(LoadToLLVM);
    patterns.add(StoreToLLVM);
    patterns.add(PtrOffsetToLLVM);
    patterns.add(AddToLLVM);
    patterns.add(SubToLLVM);
    patterns.add(MulToLLVM);
    patterns.add(ShlToLLVM);
    patterns.add(ShrToLLVM);
    patterns.add(DivToLLVM);
    patterns.add(RemToLLVM);
    patterns.add(BitAndToLLVM);
    patterns.add(BitOrToLLVM);
    patterns.add(BitXorToLLVM);
    patterns.add(EqToLLVM);
    patterns.add(NeToLLVM);
    patterns.add(LtToLLVM);
    patterns.add(LeToLLVM);
    patterns.add(GtToLLVM);
    patterns.add(GeToLLVM);
    patterns.add(CastToLLVM);
    patterns.add(ExtractValueToLLVM);
    patterns.add(InsertValueToLLVM);
    patterns.add(ReturnToLLVM);
    patterns.add(GotoToLLVM);
    patterns.add(CondBrToLLVM);
    patterns.add(UnreachableToLLVM);
    patterns.add(CallToLLVM);
}

/// Pass that converts the MIR dialect to the LLVM dialect.
pub struct ConvertMirToLLVMPass;

impl Pass for ConvertMirToLLVMPass {
    fn name(&self) -> &str {
        "convert-mir-to-llvm"
    }

    fn run(&mut self, root: Ptr<Operation>, ctx: &mut Context, _analyses: &mut AnalysisManager) -> pliron::result::Result<PassResult> {
        let mut target = ConversionTarget::new();
        target.add_illegal_dialect(DialectName::try_new("mir").unwrap());
        target.add_legal_dialect(DialectName::try_new("llvm").unwrap());
        target.add_legal_dialect(DialectName::try_new("builtin").unwrap());

        let mut patterns = ConversionPatternSet::new();
        populate_mir_to_llvm_patterns(&mut patterns);
        patterns.finalize();

        apply_partial_conversion(root, &target, &patterns, None, ctx)?;
        Ok(changed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dialects::{builtin, llvm, mir},
        printable::Printable,
    };

    fn test_context() -> Context {
        let mut ctx = Context::new();
        mir::register(&mut ctx);
        llvm::register(&mut ctx);
        ctx
    }

    #[test]
    fn converts_scalar_add_function() {
        let mut ctx = test_context();
        let i64_ty: TypeHandle = IntegerType::get(&mut ctx, 64, Signedness::Signless).into();
        let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let module_body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();

        let fn_ty = FunctionType::get(&mut ctx, vec![i64_ty, i64_ty], vec![i64_ty]);
        let func = mir_ops::FuncOp::new(&mut ctx, "add".try_into().unwrap(), fn_ty);
        func.get_operation().insert_at_back(module_body, &ctx);

        let entry = func.get_entry_block(&ctx);
        let args: Vec<_> = entry.deref(&ctx).arguments().collect();
        let add = mir_ops::AddOp::new(&mut ctx, args[0], args[1]);
        add.get_operation().insert_at_back(entry, &ctx);
        let add_result = add.get_result(&ctx);
        let ret = mir_ops::ReturnOp::new(&mut ctx, Some(add_result));
        ret.get_operation().insert_at_back(entry, &ctx);

        ConvertMirToLLVMPass
            .run(module.get_operation(), &mut ctx, &mut AnalysisManager::default())
            .unwrap();

        let text = format!("{}", module.get_operation().disp(&ctx));
        assert!(text.contains("llvm.func external @add"));
        assert!(text.contains("llvm.add"));
        assert!(text.contains("llvm.return"));
        assert!(!text.contains("mir."));
    }

    #[test]
    fn converts_alloca_load_store() {
        let mut ctx = test_context();
        let i64_ty: TypeHandle = IntegerType::get(&mut ctx, 64, Signedness::Signless).into();
        let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let module_body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();

        let fn_ty = FunctionType::get(&mut ctx, vec![i64_ty], vec![i64_ty]);
        let func = mir_ops::FuncOp::new(&mut ctx, "slot".try_into().unwrap(), fn_ty);
        func.get_operation().insert_at_back(module_body, &ctx);

        let entry = func.get_entry_block(&ctx);
        let arg = entry.deref(&ctx).get_argument(0);
        let alloca = mir_ops::AllocaOp::new(&mut ctx, i64_ty);
        alloca.get_operation().insert_at_back(entry, &ctx);
        let slot = alloca.get_result(&ctx);
        let store = mir_ops::StoreOp::new(&mut ctx, arg, slot);
        store.get_operation().insert_at_back(entry, &ctx);
        let load = mir_ops::LoadOp::new(&mut ctx, slot, i64_ty);
        load.get_operation().insert_at_back(entry, &ctx);
        let loaded = load.get_result(&ctx);
        let ret = mir_ops::ReturnOp::new(&mut ctx, Some(loaded));
        ret.get_operation().insert_at_back(entry, &ctx);

        ConvertMirToLLVMPass
            .run(module.get_operation(), &mut ctx, &mut AnalysisManager::default())
            .unwrap();

        let text = format!("{}", module.get_operation().disp(&ctx));
        assert!(text.contains("llvm.alloca"));
        assert!(text.contains("llvm.store"));
        assert!(text.contains("llvm.load"));
        assert!(!text.contains("mir."));
    }

    #[test]
    fn converts_casts_by_source_and_destination_kind() {
        let mut ctx = test_context();
        let i32_ty: TypeHandle = IntegerType::get(&mut ctx, 32, Signedness::Signless).into();
        let i64_ty: TypeHandle = IntegerType::get(&mut ctx, 64, Signedness::Signless).into();
        let si64_ty: TypeHandle = IntegerType::get(&mut ctx, 64, Signedness::Signed).into();
        let ui64_ty: TypeHandle = IntegerType::get(&mut ctx, 64, Signedness::Unsigned).into();
        let mir_ptr_ty: TypeHandle = MirPtrType::get(&mut ctx, i32_ty, false).into();
        let fp32_ty: TypeHandle = builtin::types::FP32Type::get(&ctx).into();
        let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let module_body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();

        let fn_ty = FunctionType::get(&mut ctx, vec![i32_ty, i64_ty, si64_ty, mir_ptr_ty], vec![]);
        let func = mir_ops::FuncOp::new(&mut ctx, "casts".try_into().unwrap(), fn_ty);
        func.get_operation().insert_at_back(module_body, &ctx);

        let entry = func.get_entry_block(&ctx);
        let args: Vec<_> = entry.deref(&ctx).arguments().collect();
        for cast in [
            mir_ops::CastOp::new(&mut ctx, args[0], i64_ty),
            mir_ops::CastOp::new(&mut ctx, args[1], i32_ty),
            mir_ops::CastOp::new(&mut ctx, args[2], ui64_ty),
            mir_ops::CastOp::new(&mut ctx, args[0], fp32_ty),
            mir_ops::CastOp::new(&mut ctx, args[3], fp32_ty),
            mir_ops::CastOp::new(&mut ctx, args[3], i64_ty),
        ] {
            cast.get_operation().insert_at_back(entry, &ctx);
        }
        mir_ops::ReturnOp::new(&mut ctx, None)
            .get_operation()
            .insert_at_back(entry, &ctx);

        ConvertMirToLLVMPass
            .run(module.get_operation(), &mut ctx, &mut AnalysisManager::default())
            .unwrap();

        let text = format!("{}", module.get_operation().disp(&ctx));
        assert_eq!(text.matches("llvm.zext").count(), 1);
        assert_eq!(text.matches("llvm.trunc").count(), 1);
        assert_eq!(text.matches("llvm.bitcast").count(), 2);
        assert_eq!(text.matches("llvm.ptrtoint").count(), 1);
        // Block arguments are rebuilt (not mutated in place) during type
        // conversion, so their printed SSA names aren't stable; check the
        // conversion signatures instead.
        assert!(
            text.contains(" : builtin.integer i32 to builtin.integer i64")
        );
        assert!(
            text.contains(" : builtin.integer i64 to builtin.integer i32")
        );
        assert!(!text.contains("mir.cast"));
        assert!(!text.contains("mir.ptr"));
    }

    #[test]
    fn converts_unsigned_division_and_remainder() {
        let mut ctx = test_context();
        let ui64_ty: TypeHandle = IntegerType::get(&mut ctx, 64, Signedness::Unsigned).into();
        let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let module_body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();

        let fn_ty = FunctionType::get(&mut ctx, vec![ui64_ty, ui64_ty], vec![]);
        let func = mir_ops::FuncOp::new(&mut ctx, "unsigned_ops".try_into().unwrap(), fn_ty);
        func.get_operation().insert_at_back(module_body, &ctx);

        let entry = func.get_entry_block(&ctx);
        let args: Vec<_> = entry.deref(&ctx).arguments().collect();
        mir_ops::DivOp::new(&mut ctx, args[0], args[1])
            .get_operation()
            .insert_at_back(entry, &ctx);
        mir_ops::RemOp::new(&mut ctx, args[0], args[1])
            .get_operation()
            .insert_at_back(entry, &ctx);
        mir_ops::ReturnOp::new(&mut ctx, None)
            .get_operation()
            .insert_at_back(entry, &ctx);

        ConvertMirToLLVMPass
            .run(module.get_operation(), &mut ctx, &mut AnalysisManager::default())
            .unwrap();

        let text = format!("{}", module.get_operation().disp(&ctx));
        assert!(text.contains("llvm.udiv"));
        assert!(text.contains("llvm.urem"));
        assert!(!text.contains("mir.div"));
        assert!(!text.contains("mir.rem"));
    }

    #[test]
    fn converts_calls_with_and_without_results() {
        let mut ctx = test_context();
        let i64_ty: TypeHandle = IntegerType::get(&mut ctx, 64, Signedness::Signless).into();
        let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let module_body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();

        let fn_ty = FunctionType::get(&mut ctx, vec![], vec![i64_ty]);
        let func = mir_ops::FuncOp::new(&mut ctx, "calls".try_into().unwrap(), fn_ty);
        func.get_operation().insert_at_back(module_body, &ctx);

        let entry = func.get_entry_block(&ctx);
        let result_call = mir_ops::CallOp::new_direct(
            &mut ctx,
            "returns_i64".try_into().unwrap(),
            vec![],
            Some(i64_ty),
        );
        let result = result_call.get_operation().deref(&ctx).get_result(0);
        result_call.get_operation().insert_at_back(entry, &ctx);
        mir_ops::CallOp::new_direct(
            &mut ctx,
            "returns_nothing".try_into().unwrap(),
            vec![],
            None,
        )
        .get_operation()
        .insert_at_back(entry, &ctx);
        mir_ops::ReturnOp::new(&mut ctx, Some(result))
            .get_operation()
            .insert_at_back(entry, &ctx);

        ConvertMirToLLVMPass
            .run(module.get_operation(), &mut ctx, &mut AnalysisManager::default())
            .unwrap();

        let text = format!("{}", module.get_operation().disp(&ctx));
        assert!(text.contains(" = llvm.call @returns_i64() : builtin.integer i64"));
        assert!(text.contains("llvm.call @returns_nothing()"));
        assert_eq!(text.matches("llvm.call").count(), 2);
        assert!(!text.contains("mir.call"));
    }

    #[test]
    fn has_stable_pass_name() {
        assert_eq!(ConvertMirToLLVMPass.name(), "convert-mir-to-llvm");
    }

    #[test]
    fn converts_aggregate_ops_and_int_to_ptr_cast() {
        let mut ctx = test_context();
        let i64_ty: TypeHandle = IntegerType::get(&mut ctx, 64, Signedness::Signless).into();
        let ptr_ty: TypeHandle = PointerType::get(&mut ctx).into();
        let aggregate_ty: TypeHandle =
            llvm::types::StructType::get_unnamed(&mut ctx, vec![ptr_ty, i64_ty]).into();
        let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let module_body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();

        let fn_ty = FunctionType::get(&mut ctx, vec![i64_ty], vec![i64_ty]);
        let func = mir_ops::FuncOp::new(&mut ctx, "aggregate".try_into().unwrap(), fn_ty);
        func.get_operation().insert_at_back(module_body, &ctx);

        let entry = func.get_entry_block(&ctx);
        let arg = entry.deref(&ctx).get_argument(0);
        let undef = mir_ops::UndefOp::new(&mut ctx, aggregate_ty);
        undef.get_operation().insert_at_back(entry, &ctx);
        let cast = mir_ops::CastOp::new(&mut ctx, arg, ptr_ty);
        cast.get_operation().insert_at_back(entry, &ctx);
        let cast_result = cast.get_result(&ctx);
        let undef_result = undef.get_result(&ctx);
        let with_ptr = mir_ops::InsertValueOp::new(&mut ctx, cast_result, undef_result, vec![0]);
        with_ptr.get_operation().insert_at_back(entry, &ctx);
        let with_ptr_result = with_ptr.get_result(&ctx);
        let with_i64 = mir_ops::InsertValueOp::new(&mut ctx, arg, with_ptr_result, vec![1]);
        with_i64.get_operation().insert_at_back(entry, &ctx);
        let with_i64_result = with_i64.get_result(&ctx);
        let extract = mir_ops::ExtractValueOp::new(&mut ctx, with_i64_result, vec![1], i64_ty);
        extract.get_operation().insert_at_back(entry, &ctx);
        let extract_result = extract.get_result(&ctx);
        let ret = mir_ops::ReturnOp::new(&mut ctx, Some(extract_result));
        ret.get_operation().insert_at_back(entry, &ctx);

        ConvertMirToLLVMPass
            .run(module.get_operation(), &mut ctx, &mut AnalysisManager::default())
            .unwrap();

        let text = format!("{}", module.get_operation().disp(&ctx));
        assert!(text.contains("llvm.undef"));
        assert!(text.contains("llvm.inttoptr"));
        assert!(text.contains("llvm.insertvalue"));
        assert!(text.contains("llvm.extractvalue"));
        assert!(!text.contains("mir."));
    }
}
