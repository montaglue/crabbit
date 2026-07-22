use pliron::derive::op_interface;
use pliron::derive::op_interface_impl;

use crate::{
    context::{Context, Ptr},
    dialects::{
        x86_64::{
            attributes::{AbiLocation, FunctionAbi},
            registers::{self, Register},
        },
        builtin::{
            op_interfaces::OneRegionInterface,
            ops::ModuleOp,
            types::{FP32Type, FP64Type, IntegerType, UnitType},
        },
        llvm::{
            attributes::LinkageAttr as LlvmLinkageAttr,
            ops::{
                AddOp, AddressOfOp, AllocaOp, AndOp, BitcastOp, BrOp, CStrOp, CallOp, CondBrOp,
                ConstantOp, ExtractValueOp, FuncOp, GetElementPtrOp, ICmpOp, InsertValueOp,
                IntToPtrOp, LShrOp, LoadOp, MulOp, OrOp, PtrToIntOp, ReturnOp, SDivOp, SRemOp,
                ShlOp, StoreOp, SubOp, TruncOp, UDivOp, URemOp, UndefOp, UnreachableOp, XorOp,
                ZExtOp,
            },
            types::{FuncType, PointerType, VoidType},
        },
    },
    input_error_noloc,
    ir::{
        op::{Op, op_cast},
        operation::Operation,
        r#type::TypeHandle,
        value::Value,
    },
    linked_list::{ContainsLinkedList, LinkedList},
    result::STAIRResult,
};

use super::{error::X86_64DarwinErr, util::cast_operation};
use llvm_compat::ll::{LinkageAttr};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AbiClass {
    Int { size: u64, align: u64 },
    Aggregate { size: u64, align: u64 },
    Void,
}

/// System V AMD64 integer argument registers, in assignment order.
pub(super) const ARG_GPRS: [Register; 6] = [
    registers::RDI,
    registers::RSI,
    registers::RDX,
    registers::RCX,
    registers::R8,
    registers::R9,
];

/// System V AMD64 integer result registers, in assignment order.
pub(super) const RESULT_GPRS: [Register; 2] = [registers::RAX, registers::RDX];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BinaryKind {
    Add,
    Sub,
    Mul,
    SDiv,
    UDiv,
    SRem,
    URem,
    And,
    Or,
    Xor,
    Shl,
    Shr,
}

#[op_interface]
pub(super) trait X86_64DarwinValidOpInterface {
    fn verify(_op: &dyn Op, _ctx: &Context) -> STAIRResult<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}

#[op_interface]
pub(super) trait X86_64DarwinBinaryOpInterface {
    fn binary_kind(&self) -> BinaryKind;

    fn verify(_op: &dyn Op, _ctx: &Context) -> STAIRResult<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}

pub(super) fn module_op(ctx: &Context, root: Ptr<Operation>) -> STAIRResult<ModuleOp> {
    cast_operation::<ModuleOp>(ctx, root)
        .ok_or_else(|| input_error_noloc!(X86_64DarwinErr::NotModule))
}

/// Map an LLVM linkage to its machine-level linkage, rejecting the ones the
/// Darwin backend does not support.
pub(super) fn validate_linkage(name: &str, linkage: LlvmLinkageAttr) -> STAIRResult<LinkageAttr> {
    match linkage {
        LlvmLinkageAttr::External => Ok(LinkageAttr::External),
        LlvmLinkageAttr::Internal => Ok(LinkageAttr::Internal),
        LlvmLinkageAttr::Private => Ok(LinkageAttr::Private),
        other => Err(input_error_noloc!(X86_64DarwinErr::UnsupportedLinkage(
            name.to_string(),
            other
        ))),
    }
}

pub(super) fn validate_function_type(
    ctx: &Context,
    name: &str,
    ty: TypeHandle,
) -> STAIRResult<()> {
    let (args, result) = function_abi_classes(ctx, ty)?;
    assign_darwin_abi(name, &args, result)?;
    Ok(())
}

pub(super) fn validate_body(ctx: &Context, func: &FuncOp) -> STAIRResult<()> {
    for block in func.get_region(ctx).deref(ctx).iter(ctx) {
        let mut op = block.deref(ctx).get_head();
        while let Some(op_ptr) = op {
            let op_obj = Operation::get_op_dyn(op_ptr, ctx);
            if op_cast::<dyn X86_64DarwinValidOpInterface>(&*op_obj).is_none() {
                return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedOp(
                    Operation::get_opid(op_ptr, ctx).to_string()
                )));
            }
            op = op_ptr.deref(ctx).get_next();
        }
    }
    Ok(())
}

pub(super) fn binary_kind(op: &dyn Op) -> Option<BinaryKind> {
    op_cast::<dyn X86_64DarwinBinaryOpInterface>(op).map(|binary| binary.binary_kind())
}

macro_rules! impl_valid_op {
    ($($op:ty),* $(,)?) => {
        $(
            #[op_interface_impl]
            impl X86_64DarwinValidOpInterface for $op {}
        )*
    };
}

macro_rules! impl_binary_op {
    ($($op:ty => $kind:expr),* $(,)?) => {
        $(
            #[op_interface_impl]
            impl X86_64DarwinValidOpInterface for $op {}

            #[op_interface_impl]
            impl X86_64DarwinBinaryOpInterface for $op {
                fn binary_kind(&self) -> BinaryKind {
                    $kind
                }
            }
        )*
    };
}

impl_valid_op!(
    ConstantOp,
    AllocaOp,
    LoadOp,
    StoreOp,
    CallOp,
    CStrOp,
    AddressOfOp,
    UndefOp,
    InsertValueOp,
    ExtractValueOp,
    IntToPtrOp,
    PtrToIntOp,
    BitcastOp,
    ZExtOp,
    TruncOp,
    GetElementPtrOp,
    ICmpOp,
    BrOp,
    CondBrOp,
    ReturnOp,
    UnreachableOp,
);

impl_binary_op!(
    AddOp => BinaryKind::Add,
    SubOp => BinaryKind::Sub,
    MulOp => BinaryKind::Mul,
    SDivOp => BinaryKind::SDiv,
    UDivOp => BinaryKind::UDiv,
    SRemOp => BinaryKind::SRem,
    URemOp => BinaryKind::URem,
    AndOp => BinaryKind::And,
    OrOp => BinaryKind::Or,
    XorOp => BinaryKind::Xor,
    ShlOp => BinaryKind::Shl,
    LShrOp => BinaryKind::Shr,
);

pub(super) fn function_abi_classes(
    ctx: &Context,
    ty: TypeHandle,
) -> STAIRResult<(Vec<AbiClass>, AbiClass)> {
    let ty_ref = ty.deref(ctx);
    let func_ty = ty_ref
        .downcast_ref::<FuncType>()
        .expect("llvm.func must carry llvm.func type");
    let args = func_ty.arg_types_slice().to_vec();
    let result = func_ty.result_type();
    drop(ty_ref);

    let mut classes = Vec::with_capacity(args.len());
    for arg in args {
        classes.push(abi_class(ctx, arg)?);
    }
    Ok((classes, abi_class(ctx, result)?))
}

pub(super) fn abi_class(ctx: &Context, ty: TypeHandle) -> STAIRResult<AbiClass> {
    let ty_ref = ty.deref(ctx);
    if ty_ref.downcast_ref::<VoidType>().is_some() || ty_ref.downcast_ref::<UnitType>().is_some() {
        return Ok(AbiClass::Void);
    }
    if ty_ref.downcast_ref::<PointerType>().is_some()
        || ty_ref.downcast_ref::<IntegerType>().is_some()
    {
        drop(ty_ref);
        let (size, align) = abi_type_layout(ctx, ty)?;
        return Ok(AbiClass::Int { size, align });
    }
    if ty_ref.downcast_ref::<FP32Type>().is_some() || ty_ref.downcast_ref::<FP64Type>().is_some() {
        return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedType(
            "floating-point ABI lowering is not implemented".to_string()
        )));
    }
    if ty_ref
        .downcast_ref::<crate::dialects::llvm::types::StructType>()
        .is_some()
        || ty_ref
            .downcast_ref::<crate::dialects::llvm::types::ArrayType>()
            .is_some()
    {
        drop(ty_ref);
        let (size, align) = abi_type_layout(ctx, ty)?;
        return Ok(AbiClass::Aggregate { size, align });
    }
    Err(input_error_noloc!(X86_64DarwinErr::UnsupportedType(
        format!("{:?}", &*ty_ref)
    )))
}

pub(super) fn assign_darwin_abi(
    name: &str,
    args: &[AbiClass],
    result: AbiClass,
) -> STAIRResult<FunctionAbi> {
    let result = match result {
        AbiClass::Int { size, .. } if size <= 8 => AbiLocation::Gpr(RESULT_GPRS[0]),
        AbiClass::Int { size, .. } if size <= 16 => {
            AbiLocation::GprPair(RESULT_GPRS[0], RESULT_GPRS[1])
        }
        AbiClass::Int { size, .. } => {
            return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedType(
                format!("integer result of {size} bytes in `{name}`")
            )));
        }
        AbiClass::Aggregate { size, .. } if size == 0 => AbiLocation::Void,
        AbiClass::Aggregate { size, .. } if size <= 8 => AbiLocation::Gpr(RESULT_GPRS[0]),
        AbiClass::Aggregate { size, .. } if size <= 16 => {
            AbiLocation::GprPair(RESULT_GPRS[0], RESULT_GPRS[1])
        }
        AbiClass::Aggregate { .. } => AbiLocation::IndirectResult,
        AbiClass::Void => AbiLocation::Void,
    };

    // An indirect result consumes rdi as the hidden first argument.
    let mut gpr: usize = if result == AbiLocation::IndirectResult {
        1
    } else {
        0
    };
    let arg_gprs = ARG_GPRS.len();
    let mut stack_offset = 0u64;
    let mut locations = Vec::with_capacity(args.len());
    for arg in args {
        match arg {
            AbiClass::Int { size, .. } if *size <= 8 => {
                if gpr < arg_gprs {
                    locations.push(AbiLocation::Gpr(ARG_GPRS[gpr]));
                    gpr += 1;
                } else {
                    locations.push(AbiLocation::Stack(stack_offset));
                    stack_offset += 8;
                }
            }
            AbiClass::Int { size, align } if *size <= 16 => {
                if gpr + 2 <= arg_gprs {
                    locations.push(AbiLocation::GprPair(ARG_GPRS[gpr], ARG_GPRS[gpr + 1]));
                    gpr += 2;
                } else {
                    stack_offset = align_to(stack_offset, *align);
                    locations.push(AbiLocation::Stack(stack_offset));
                    stack_offset += 16;
                }
            }
            AbiClass::Int { size, .. } => {
                return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedType(
                    format!("integer argument of {size} bytes in `{name}`")
                )));
            }
            AbiClass::Void => {
                locations.push(AbiLocation::Void);
            }
            AbiClass::Aggregate { size: 0, .. } => {
                locations.push(AbiLocation::Void);
            }
            AbiClass::Aggregate { .. } => {
                return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedType(
                    format!("aggregate argument in `{name}`")
                )));
            }
        }
    }
    Ok(FunctionAbi {
        args: locations,
        result,
    })
}

fn abi_type_layout(ctx: &Context, ty: TypeHandle) -> STAIRResult<(u64, u64)> {
    let ty_ref = ty.deref(ctx);
    if ty_ref.downcast_ref::<UnitType>().is_some() {
        return Ok((0, 1));
    }
    if ty_ref.downcast_ref::<PointerType>().is_some() || ty_ref.downcast_ref::<FP64Type>().is_some()
    {
        return Ok((8, 8));
    }
    if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
        let size = (int_ty.width() as u64).div_ceil(8).max(1);
        return Ok((size, size.next_power_of_two().min(16)));
    }
    if ty_ref.downcast_ref::<FP32Type>().is_some() {
        return Ok((4, 4));
    }
    if let Some(array_ty) = ty_ref.downcast_ref::<crate::dialects::llvm::types::ArrayType>() {
        let elem_ty = array_ty.elem_type();
        let len = array_ty.size();
        drop(ty_ref);
        let (element_size, alignment) = abi_type_layout(ctx, elem_ty)?;
        return Ok((align_to(element_size, alignment) * len, alignment));
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<crate::dialects::llvm::types::StructType>() {
        let fields = struct_ty.fields().ok_or_else(|| {
            input_error_noloc!(X86_64DarwinErr::UnsupportedType(
                "opaque aggregate ABI type".to_string()
            ))
        })?;
        let fields = fields.to_vec();
        drop(ty_ref);
        let mut size = 0;
        let mut alignment = 1;
        for field in fields {
            let (field_size, field_alignment) = abi_type_layout(ctx, field)?;
            size = align_to(size, field_alignment);
            size += field_size;
            alignment = alignment.max(field_alignment);
        }
        return Ok((align_to(size, alignment), alignment));
    }
    Err(input_error_noloc!(X86_64DarwinErr::UnsupportedType(
        format!("{:?}", &*ty_ref)
    )))
}

fn align_to(value: u64, align: u64) -> u64 {
    if align <= 1 {
        value
    } else {
        (value + align - 1) & !(align - 1)
    }
}

pub(super) fn collect_entry_arguments(ctx: &Context, func: &FuncOp) -> STAIRResult<Vec<Value>> {
    let entry = func.get_entry_block(ctx);
    Ok(entry.deref(ctx).arguments().collect())
}
