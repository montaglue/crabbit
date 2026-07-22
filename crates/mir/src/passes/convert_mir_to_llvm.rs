//! Convert Rust MIR dialect operations to the LLVM dialect.
//!
//! Each `mir` op implements [pliron_llvm]'s own [ToLLVMDialect] interface;
//! the conversion itself is driven by pliron's
//! [dialect_conversion](pliron::irbuild::dialect_conversion) framework.

use std::cmp::Ordering;

use awint::bw;

use pliron::{
    builtin::{
        attributes::IntegerAttr,
        op_interfaces::{CallOpCallable, OneRegionInterface, OneResultInterface,
            SymbolOpInterface},
        type_interfaces::FunctionTypeInterface,
        types::{FunctionType, IntegerType, Signedness},
    },
    context::{Context, Ptr},
    derive::op_interface_impl,
    dict_key, input_error_noloc,
    irbuild::{
        dialect_conversion::{self, DialectConversion, DialectConversionRewriter, OperandsInfo},
        inserter::Inserter,
        rewriter::Rewriter,
    },
    op::{Op, op_cast, op_impls},
    operation::Operation,
    region::Region,
    result::Result,
    r#type::{TypeHandle, TypedHandle, Typed},
    utils::apint::APInt,
    value::Value,
};

use pliron_llvm::{
    ToLLVMDialect,
    attributes::{ICmpPredicateAttr, IntegerOverflowFlagsAttr, LinkageAttr},
    ops::{self as llvm_ops, GepIndex},
    op_interfaces::{
        BinArithOp as _, CastOpInterface as _, CastOpWithNNegInterface as _,
        IntBinArithOpWithOverflowFlag as _,
    },
    types::{ArrayType, FuncType, PointerType, StructType, VoidType},
};

use pliron_ll::ll::ops::CStrOp as LlCStrOp;

use crate::mir::{ops as mir_ops, types::PtrType as MirPtrType};

dict_key!(
    /// Machine-level linkage recorded on a `mir.func` by the importer,
    /// carried onto the `llvm.func` during conversion.
    ATTR_KEY_MIR_FUNC_LINKAGE, "mir_func_linkage"
);

// ============================================================================
// Type conversion
// ============================================================================

fn convert_type(ctx: &mut Context, ty: TypeHandle) -> TypeHandle {
    let ty_ref = ty.deref(ctx);
    if ty_ref.is::<MirPtrType>() {
        drop(ty_ref);
        return PointerType::get(ctx, 0).into();
    }
    if let Some(array_ty) = ty_ref.downcast_ref::<ArrayType>() {
        let elem = array_ty.elem_type();
        let size = array_ty.size();
        drop(ty_ref);
        let elem = convert_type(ctx, elem);
        return ArrayType::get(ctx, elem, size).into();
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<StructType>() {
        let name = struct_ty.name();
        let fields = (!struct_ty.is_opaque()).then(|| struct_ty.fields().collect::<Vec<_>>());
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

/// Whether `ty` contains a `mir` type and hence needs [convert_type].
fn type_needs_conversion(ctx: &Context, ty: TypeHandle) -> bool {
    let ty_ref = ty.deref(ctx);
    if ty_ref.is::<MirPtrType>() || ty_ref.is::<FunctionType>() {
        return true;
    }
    if let Some(array_ty) = ty_ref.downcast_ref::<ArrayType>() {
        let elem = array_ty.elem_type();
        drop(ty_ref);
        return type_needs_conversion(ctx, elem);
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<StructType>() {
        if struct_ty.is_opaque() {
            return false;
        }
        let fields: Vec<_> = struct_ty.fields().collect();
        drop(ty_ref);
        return fields
            .into_iter()
            .any(|field| type_needs_conversion(ctx, field));
    }
    false
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
    use pliron::basic_block::BasicBlock;
    use pliron::linked_list::ContainsLinkedList;

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
// Shared rewrite helpers
// ============================================================================

/// Insert `new_op` before the op being rewritten and replace `old` with it.
fn replace_with(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    old: Ptr<Operation>,
    new_op: Ptr<Operation>,
) {
    rewriter.insert_operation(ctx, new_op);
    rewriter.replace_operation(ctx, old, new_op);
}

fn result_type_of(ctx: &mut Context, op: Ptr<Operation>) -> TypeHandle {
    let ty = op.deref(ctx).get_result(0).get_type(ctx);
    convert_type(ctx, ty)
}

fn integer_signedness(ctx: &Context, value: Value) -> Signedness {
    let ty = value.get_type(ctx);
    let ty_ref = ty.deref(ctx);
    let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() else {
        return Signedness::Signless;
    };
    int_ty.signedness()
}

// ============================================================================
// Function conversion
// ============================================================================

#[op_interface_impl]
impl ToLLVMDialect for mir_ops::FuncOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let name = self.get_symbol_name(ctx);
        let llvm_func_ty = convert_function_type(ctx, self.get_func_type(ctx));
        // A linkage pre-set on the mir.func (e.g. by the Rust importer for
        // upstream instances) is carried over; external otherwise.
        let linkage = op
            .deref(ctx)
            .attributes
            .get::<LinkageAttr>(&ATTR_KEY_MIR_FUNC_LINKAGE)
            .cloned()
            .unwrap_or(LinkageAttr::ExternalLinkage);

        let llvm_func = llvm_ops::FuncOp::new(ctx, name, llvm_func_ty);
        Region::move_to_op(self.get_region(ctx), llvm_func.get_operation(), ctx);
        llvm_func.set_attr_llvm_function_linkage(ctx, linkage);
        convert_block_arg_types(llvm_func.get_operation(), ctx);

        replace_with(ctx, rewriter, op, llvm_func.get_operation());
        Ok(())
    }
}

// ============================================================================
// Simple one-to-one operation conversions
// ============================================================================

macro_rules! binary_to_llvm {
    ($($src:ty => $dst:ty),* $(,)?) => {
        $(
            #[op_interface_impl]
            impl ToLLVMDialect for $src {
                fn rewrite(
                    &self,
                    ctx: &mut Context,
                    rewriter: &mut DialectConversionRewriter,
                    _operands_info: &OperandsInfo,
                ) -> Result<()> {
                    let op = self.get_operation();
                    let (lhs, rhs) = {
                        let op_ref = op.deref(ctx);
                        (op_ref.get_operand(0), op_ref.get_operand(1))
                    };
                    let new_op = <$dst>::new(ctx, lhs, rhs);
                    replace_with(ctx, rewriter, op, new_op.get_operation());
                    Ok(())
                }
            }
        )*
    };
}

binary_to_llvm!(
    mir_ops::ShrOp => llvm_ops::LShrOp,
    mir_ops::BitAndOp => llvm_ops::AndOp,
    mir_ops::BitOrOp => llvm_ops::OrOp,
    mir_ops::BitXorOp => llvm_ops::XorOp,
);

macro_rules! overflow_binary_to_llvm {
    ($($src:ty => $dst:ty),* $(,)?) => {
        $(
            #[op_interface_impl]
            impl ToLLVMDialect for $src {
                fn rewrite(
                    &self,
                    ctx: &mut Context,
                    rewriter: &mut DialectConversionRewriter,
                    _operands_info: &OperandsInfo,
                ) -> Result<()> {
                    let op = self.get_operation();
                    let (lhs, rhs) = {
                        let op_ref = op.deref(ctx);
                        (op_ref.get_operand(0), op_ref.get_operand(1))
                    };
                    let new_op = <$dst>::new_with_overflow_flag(
                        ctx,
                        lhs,
                        rhs,
                        IntegerOverflowFlagsAttr::default(),
                    );
                    replace_with(ctx, rewriter, op, new_op.get_operation());
                    Ok(())
                }
            }
        )*
    };
}

overflow_binary_to_llvm!(
    mir_ops::AddOp => llvm_ops::AddOp,
    mir_ops::SubOp => llvm_ops::SubOp,
    mir_ops::MulOp => llvm_ops::MulOp,
    mir_ops::ShlOp => llvm_ops::ShlOp,
);

macro_rules! signed_binary_to_llvm {
    ($($src:ty => ($signed:ty, $unsigned:ty)),* $(,)?) => {
        $(
            #[op_interface_impl]
            impl ToLLVMDialect for $src {
                fn rewrite(
                    &self,
                    ctx: &mut Context,
                    rewriter: &mut DialectConversionRewriter,
                    _operands_info: &OperandsInfo,
                ) -> Result<()> {
                    let op = self.get_operation();
                    let (lhs, rhs) = {
                        let op_ref = op.deref(ctx);
                        (op_ref.get_operand(0), op_ref.get_operand(1))
                    };
                    let new_op = if integer_signedness(ctx, lhs) == Signedness::Unsigned {
                        <$unsigned>::new(ctx, lhs, rhs).get_operation()
                    } else {
                        <$signed>::new(ctx, lhs, rhs).get_operation()
                    };
                    replace_with(ctx, rewriter, op, new_op);
                    Ok(())
                }
            }
        )*
    };
}

signed_binary_to_llvm!(
    mir_ops::DivOp => (llvm_ops::SDivOp, llvm_ops::UDivOp),
    mir_ops::RemOp => (llvm_ops::SRemOp, llvm_ops::URemOp),
);

macro_rules! cmp_to_llvm {
    ($($src:ty => ($signed_pred:expr, $unsigned_pred:expr)),* $(,)?) => {
        $(
            #[op_interface_impl]
            impl ToLLVMDialect for $src {
                fn rewrite(
                    &self,
                    ctx: &mut Context,
                    rewriter: &mut DialectConversionRewriter,
                    _operands_info: &OperandsInfo,
                ) -> Result<()> {
                    let op = self.get_operation();
                    let (lhs, rhs) = {
                        let op_ref = op.deref(ctx);
                        (op_ref.get_operand(0), op_ref.get_operand(1))
                    };
                    let pred = if integer_signedness(ctx, lhs) == Signedness::Unsigned {
                        $unsigned_pred
                    } else {
                        $signed_pred
                    };
                    let new_op = llvm_ops::ICmpOp::new(ctx, pred, lhs, rhs);
                    replace_with(ctx, rewriter, op, new_op.get_operation());
                    Ok(())
                }
            }
        )*
    };
}

cmp_to_llvm!(
    mir_ops::EqOp => (ICmpPredicateAttr::EQ, ICmpPredicateAttr::EQ),
    mir_ops::NeOp => (ICmpPredicateAttr::NE, ICmpPredicateAttr::NE),
    mir_ops::LtOp => (ICmpPredicateAttr::SLT, ICmpPredicateAttr::ULT),
    mir_ops::LeOp => (ICmpPredicateAttr::SLE, ICmpPredicateAttr::ULE),
    mir_ops::GtOp => (ICmpPredicateAttr::SGT, ICmpPredicateAttr::UGT),
    mir_ops::GeOp => (ICmpPredicateAttr::SGE, ICmpPredicateAttr::UGE),
);

#[op_interface_impl]
impl ToLLVMDialect for mir_ops::ConstantOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let attr = self.get_value(ctx).unwrap();
        let new_op = llvm_ops::ConstantOp::new(ctx, Box::new(attr));
        replace_with(ctx, rewriter, op, new_op.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for mir_ops::CStrOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let new_op = LlCStrOp::new(ctx, self.get_value(ctx));
        replace_with(ctx, rewriter, op, new_op.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for mir_ops::AddressOfOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let new_op = llvm_ops::AddressOfOp::new(ctx, self.get_symbol(ctx), 0);
        replace_with(ctx, rewriter, op, new_op.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for mir_ops::UndefOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let result_ty = result_type_of(ctx, op);
        let new_op = llvm_ops::UndefOp::new(ctx, result_ty);
        replace_with(ctx, rewriter, op, new_op.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for mir_ops::ExtractValueOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let aggregate = op.deref(ctx).get_operand(0);
        let new_op = llvm_ops::ExtractValueOp::new(ctx, aggregate, self.get_indices(ctx))?;
        replace_with(ctx, rewriter, op, new_op.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for mir_ops::InsertValueOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let value = self.get_value(ctx);
        let aggregate = self.get_aggregate(ctx);
        let new_op = llvm_ops::InsertValueOp::new(ctx, aggregate, value, self.get_indices(ctx));
        replace_with(ctx, rewriter, op, new_op.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for mir_ops::CastOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let input = op.deref(ctx).get_operand(0);
        let input_ty = convert_type(ctx, input.get_type(ctx));
        let result_ty = result_type_of(ctx, op);

        if input_ty == result_ty {
            rewriter.replace_operation_with_values(ctx, op, vec![input]);
            return Ok(());
        }

        let input_is_int = input_ty.deref(ctx).is::<IntegerType>();
        let input_is_ptr = input_ty.deref(ctx).is::<PointerType>();
        let result_is_int = result_ty.deref(ctx).is::<IntegerType>();
        let result_is_ptr = result_ty.deref(ctx).is::<PointerType>();

        let new_op = if input_is_int && result_is_ptr {
            llvm_ops::IntToPtrOp::new(ctx, input, result_ty).get_operation()
        } else if input_is_ptr && result_is_int {
            llvm_ops::PtrToIntOp::new(ctx, input, result_ty).get_operation()
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
                    rewriter.replace_operation_with_values(ctx, op, vec![input]);
                    return Ok(());
                }
                Ordering::Less => {
                    llvm_ops::ZExtOp::new_with_nneg(ctx, input, result_ty, false).get_operation()
                }
                Ordering::Greater => llvm_ops::TruncOp::new(ctx, input, result_ty).get_operation(),
            }
        } else {
            llvm_ops::BitcastOp::new(ctx, input, result_ty).get_operation()
        };

        replace_with(ctx, rewriter, op, new_op);
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for mir_ops::AllocaOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let elem_ty = convert_type(ctx, self.get_elem_type(ctx));
        let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
        let one = IntegerAttr::new(i64_ty, APInt::from_u64(1, bw(64)));
        let one_op = llvm_ops::ConstantOp::new(ctx, Box::new(one));
        rewriter.insert_operation(ctx, one_op.get_operation());

        let new_op = llvm_ops::AllocaOp::new(ctx, elem_ty, one_op.get_result(ctx));
        replace_with(ctx, rewriter, op, new_op.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for mir_ops::LoadOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let addr = op.deref(ctx).get_operand(0);
        let result_ty = result_type_of(ctx, op);
        let new_op = llvm_ops::LoadOp::new(ctx, addr, result_ty);
        replace_with(ctx, rewriter, op, new_op.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for mir_ops::StoreOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let (value, addr) = {
            let op_ref = op.deref(ctx);
            (op_ref.get_operand(0), op_ref.get_operand(1))
        };
        let new_op = llvm_ops::StoreOp::new(ctx, value, addr);
        replace_with(ctx, rewriter, op, new_op.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for mir_ops::PtrOffsetOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let (base, offset) = {
            let op_ref = op.deref(ctx);
            (op_ref.get_operand(0), op_ref.get_operand(1))
        };
        let i8_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
        let new_op = llvm_ops::GetElementPtrOp::new(ctx, base, vec![GepIndex::Value(offset)], i8_ty);
        replace_with(ctx, rewriter, op, new_op.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for mir_ops::ReturnOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let value = {
            let op_ref = op.deref(ctx);
            (op_ref.get_num_operands() > 0).then(|| op_ref.get_operand(0))
        };
        let new_op = llvm_ops::ReturnOp::new(ctx, value);
        replace_with(ctx, rewriter, op, new_op.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for mir_ops::GotoOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let dest = self.get_dest(ctx);
        let dest_operands = self.get_dest_operands(ctx);
        let new_op = llvm_ops::BrOp::new(ctx, dest, dest_operands);
        replace_with(ctx, rewriter, op, new_op.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for mir_ops::CondBrOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let condition = self.get_condition(ctx);
        let true_dest = self.get_true_dest(ctx);
        let true_operands = self.get_true_operands(ctx);
        let false_dest = self.get_false_dest(ctx);
        let false_operands = self.get_false_operands(ctx);
        let new_op = llvm_ops::CondBrOp::new(
            ctx,
            condition,
            true_dest,
            true_operands,
            false_dest,
            false_operands,
        );
        replace_with(ctx, rewriter, op, new_op.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for mir_ops::UnreachableOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let new_op = llvm_ops::UnreachableOp::new(ctx);
        replace_with(ctx, rewriter, op, new_op.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for mir_ops::CallOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let operands: Vec<Value> = op.deref(ctx).operands().collect();
        let result_type = if op.deref(ctx).get_num_results() > 0 {
            result_type_of(ctx, op)
        } else {
            VoidType::get(ctx).into()
        };

        let (callee, args) = match self.get_callee_opt(ctx) {
            Some(callee) => (CallOpCallable::Direct(callee), operands),
            None => {
                let (callee, args) = operands.split_first().ok_or_else(|| {
                    input_error_noloc!("indirect mir.call has no callee operand")
                })?;
                (CallOpCallable::Indirect(*callee), args.to_vec())
            }
        };
        let arg_types = args.iter().map(|arg| {
            let ty = arg.get_type(ctx);
            convert_type(ctx, ty)
        });
        let arg_types: Vec<_> = arg_types.collect();
        let callee_ty = FuncType::get(ctx, result_type, arg_types, false);

        let new_op = llvm_ops::CallOp::new(ctx, callee, callee_ty, args);
        rewriter.insert_operation(ctx, new_op.get_operation());
        // An `llvm.call` always has one result (void-typed for `void`
        // functions); a `mir.call` to a void function has none.
        let replacements: Vec<Value> = new_op
            .get_operation()
            .deref(ctx)
            .results()
            .take(op.deref(ctx).get_num_results())
            .collect();
        rewriter.replace_operation_with_values(ctx, op, replacements);
        Ok(())
    }
}

// ============================================================================
// The conversion and its pass
// ============================================================================

/// Dialect conversion from the `mir` dialect to the `llvm` dialect: converts
/// every op implementing [ToLLVMDialect] and every type containing a `mir`
/// type.
#[derive(Default)]
pub struct MirToLLVMConversion;

impl DialectConversion for MirToLLVMConversion {
    fn can_convert_op(&self, ctx: &Context, op: Ptr<Operation>) -> bool {
        let op_dyn = Operation::get_op_dyn(op, ctx);
        op_impls::<dyn ToLLVMDialect>(op_dyn.op_ref())
    }

    fn can_convert_type(&self, ctx: &Context, ty: TypeHandle) -> bool {
        type_needs_conversion(ctx, ty)
    }

    fn convert_type(&mut self, ctx: &mut Context, ty: TypeHandle) -> Result<TypeHandle> {
        Ok(convert_type(ctx, ty))
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op_dyn = Operation::get_op_dyn(op, ctx);
        if let Some(to_llvm) = op_cast::<dyn ToLLVMDialect>(op_dyn.op_ref()) {
            to_llvm.rewrite(ctx, rewriter, operands_info)?;
        }
        Ok(())
    }
}

/// The `convert-mir-to-llvm` [Pass].
pub fn convert_mir_to_llvm_pass() -> dialect_conversion::PassWrapper<MirToLLVMConversion> {
    dialect_conversion::PassWrapper::new("convert-mir-to-llvm", MirToLLVMConversion)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pliron::{
        builtin,
        pass::Pass as _,
        linked_list::ContainsLinkedList,
        pass::AnalysisManager,
        printable::Printable,
    };

    fn test_context() -> Context {
        let mut ctx = Context::new();
        crate::mir::register(&mut ctx);
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

        convert_mir_to_llvm_pass()
            .run(module.get_operation(), &mut ctx, &mut AnalysisManager::default())
            .unwrap();

        let text = format!("{}", module.get_operation().disp(&ctx));
        assert!(text.contains("llvm.func @add"), "{text}");
        assert!(text.contains("ExternalLinkage"), "{text}");
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

        convert_mir_to_llvm_pass()
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

        convert_mir_to_llvm_pass()
            .run(module.get_operation(), &mut ctx, &mut AnalysisManager::default())
            .unwrap();

        let text = format!("{}", module.get_operation().disp(&ctx));
        assert_eq!(text.matches("llvm.zext").count(), 1);
        assert_eq!(text.matches("llvm.trunc").count(), 1);
        assert_eq!(text.matches("llvm.bitcast").count(), 2);
        assert_eq!(text.matches("llvm.ptrtoint").count(), 1);
        // The zext/trunc pair covers both widening and narrowing; exact
        // printed signatures are pliron's cast format and asserted above by
        // op counts.
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

        convert_mir_to_llvm_pass()
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

        convert_mir_to_llvm_pass()
            .run(module.get_operation(), &mut ctx, &mut AnalysisManager::default())
            .unwrap();

        let text = format!("{}", module.get_operation().disp(&ctx));
        assert!(text.contains("llvm.call"), "{text}");
        assert!(text.contains("@returns_i64"), "{text}");
        assert!(text.contains("@returns_nothing"), "{text}");
        assert_eq!(text.matches("llvm.call").count(), 2);
        assert!(!text.contains("mir.call"));
    }

    #[test]
    fn has_stable_pass_name() {
        assert_eq!(convert_mir_to_llvm_pass().name(), "convert-mir-to-llvm");
    }

    #[test]
    fn converts_aggregate_ops_and_int_to_ptr_cast() {
        let mut ctx = test_context();
        let i64_ty: TypeHandle = IntegerType::get(&mut ctx, 64, Signedness::Signless).into();
        let ptr_ty: TypeHandle = PointerType::get(&mut ctx, 0).into();
        let aggregate_ty: TypeHandle =
            StructType::get_unnamed(&mut ctx, vec![ptr_ty, i64_ty]).into();
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

        convert_mir_to_llvm_pass()
            .run(module.get_operation(), &mut ctx, &mut AnalysisManager::default())
            .unwrap();

        let text = format!("{}", module.get_operation().disp(&ctx));
        assert!(text.contains("llvm.undef"));
        assert!(text.contains("llvm.inttoptr"));
        assert!(text.contains("llvm.insert_value"));
        assert!(text.contains("llvm.extract_value"));
        assert!(!text.contains("mir."));
    }
}
