//! Constructor-compatible shims over cuda-oxide's `dialect-mir` ops (plus raw
//! [pliron_llvm] ops for aggregate values and symbol addresses), mirroring the
//! old `cmir` op constructors so the importer body above stays line-compatible
//! with `importer.rs` while emitting the new dialect.
//!
//! Conventions established by the port:
//! - Pointers are `MirPtrType` in the generic (0) address space; pointees are
//!   advisory (the sibling branch relaxes dialect-mir to opaque-pointer mode).
//! - `ptr_offset` keeps the old byte-addressing semantics via the
//!   `ptr_offset_unit = "byte"` attribute understood by `mir-lower`.
//! - Aggregate values (undef/extract/insert) and symbol addresses stay in the
//!   llvm dialect: `mir-lower` passes llvm ops through untouched and its type
//!   converter accepts llvm aggregate types.
use crate::attribute::AttrObj;
use crate::context::{Context, Ptr};
use crate::dialects::builtin::attributes::{
    FPDoubleAttr, FPSingleAttr, IntegerAttr, StringAttr, TypeAttr,
};
use crate::dialects::builtin::op_interfaces::{
    CallOpCallable, OperandSegmentInterface, SymbolOpInterface,
};
use crate::dialects::builtin::type_interfaces::FunctionTypeInterface;
use crate::dialects::builtin::types::{FP32Type, FP64Type, FunctionType, IntegerType, Signedness};
use crate::dialects::dialect_mir::attributes::MirCastKindAttr;
use crate::dialects::dialect_mir::ops as dm_ops;
use crate::dialects::dialect_mir::types as dm_types;
use crate::dialects::llvm as pllvm;
use crate::dialects::llvm::op_interfaces::BinArithOp as _;
use crate::dialects::llvm::op_interfaces::CastOpInterface as _;
use crate::identifier::Identifier;
use crate::ir::basic_block::BasicBlock;
use crate::ir::operation::Operation;
use crate::ir::r#type::{TypeHandle, Typed, TypedHandle};
use crate::ir::value::Value;
use crate::linked_list::ContainsLinkedList;
use crate::op::Op;
use crate::region::Region;

pub fn func_linkage_key() -> Identifier {
    "mir_func_linkage".try_into().expect("static identifier literal")
}

pub mod types {
    use super::*;

    /// Old `cmir.ptr` constructor surface over [dm_types::MirPtrType].
    pub struct PtrType;

    impl PtrType {
        pub fn get(
            ctx: &mut Context,
            elem: TypeHandle,
            mutable: bool,
        ) -> TypedHandle<dm_types::MirPtrType> {
            dm_types::MirPtrType::get(
                ctx,
                elem,
                mutable,
                dm_types::address_space::GENERIC,
            )
        }
    }
}

macro_rules! shim_common {
    ($Shim:ident) => {
        impl $Shim {
            pub fn get_operation(&self) -> Ptr<Operation> {
                self.op
            }

            #[allow(dead_code)]
            pub fn get_result(&self, ctx: &Context) -> Value {
                self.op.deref(ctx).get_result(0)
            }
        }
    };
}

/// Whether a value's static type is a pointer (either dialect).
/// dialect-mir arithmetic/compare lowering resolves signedness from
/// integer operand types and rejects raw llvm pointers, so pointer
/// operands take the llvm twin op instead (the old pipeline's exact
/// output for pointer arithmetic).
fn is_ptr_value(ctx: &Context, value: Value) -> bool {
    let ty = value.get_type(ctx);
    let ty_ref = ty.deref(ctx);
    ty_ref.is::<dm_types::MirPtrType>() || ty_ref.is::<pllvm::types::PointerType>()
}

macro_rules! shim_binop {
    ($Shim:ident, $Theirs:ident, $LlvmTwin:ident) => {
        pub struct $Shim {
            op: Ptr<Operation>,
        }

        impl $Shim {
            pub fn new(ctx: &mut Context, lhs: Value, rhs: Value) -> Self {
                if super::is_ptr_value(ctx, lhs) || super::is_ptr_value(ctx, rhs) {
                    let twin = pllvm::ops::$LlvmTwin::new(ctx, lhs, rhs);
                    return Self {
                        op: twin.get_operation(),
                    };
                }
                let ty = lhs.get_type(ctx);
                let op = Operation::new(
                    ctx,
                    dm_ops::$Theirs::get_concrete_op_info(),
                    vec![ty],
                    vec![lhs, rhs],
                    vec![],
                    0,
                );
                Self { op }
            }
        }
        shim_common!($Shim);
    };
}

macro_rules! shim_cmpop {
    ($Shim:ident, $Theirs:ident, $PtrPred:ident) => {
        pub struct $Shim {
            op: Ptr<Operation>,
        }

        impl $Shim {
            pub fn new(ctx: &mut Context, lhs: Value, rhs: Value) -> Self {
                if super::is_ptr_value(ctx, lhs) || super::is_ptr_value(ctx, rhs) {
                    // Pointers compare unsigned.
                    let icmp = pllvm::ops::ICmpOp::new(
                        ctx,
                        pllvm::attributes::ICmpPredicateAttr::$PtrPred,
                        lhs,
                        rhs,
                    );
                    return Self {
                        op: icmp.get_operation(),
                    };
                }
                let i1: TypeHandle =
                    IntegerType::get(ctx, 1, Signedness::Signless).into();
                let op = Operation::new(
                    ctx,
                    dm_ops::$Theirs::get_concrete_op_info(),
                    vec![i1],
                    vec![lhs, rhs],
                    vec![],
                    0,
                );
                Self { op }
            }
        }
        shim_common!($Shim);
    };
}

pub mod ops {
    use super::*;

    pub struct FuncOp {
        op: Ptr<Operation>,
    }

    impl FuncOp {
        pub fn new(
            ctx: &mut Context,
            name: Identifier,
            ty: TypedHandle<FunctionType>,
        ) -> Self {
            let op = Operation::new(
                ctx,
                dm_ops::MirFuncOp::get_concrete_op_info(),
                vec![],
                vec![],
                vec![],
                1,
            );
            let func = dm_ops::MirFuncOp::new(ctx, op, TypeAttr::new(ty.into()));
            func.set_symbol_name(ctx, name);
            let arg_types = ty.deref(ctx).arg_types();
            let region = op.deref(ctx).get_region(0);
            let entry =
                BasicBlock::new(ctx, Some("entry".try_into().expect("static identifier literal")), arg_types);
            entry.insert_at_front(region, ctx);
            FuncOp { op }
        }

        pub fn get_entry_block(&self, ctx: &Context) -> Ptr<BasicBlock> {
            self.op
                .deref(ctx)
                .get_region(0)
                .deref(ctx)
                .get_head()
                .expect("mir.func must have an entry block")
        }

        pub fn get_region(&self, ctx: &Context) -> Ptr<Region> {
            self.op.deref(ctx).get_region(0)
        }
    }
    shim_common!(FuncOp);

    pub struct ConstantOp {
        op: Ptr<Operation>,
    }

    impl ConstantOp {
        pub fn new(ctx: &mut Context, attr: AttrObj) -> Self {
            if let Some(int_attr) = (*attr).downcast_ref::<IntegerAttr>() {
                return Self::new_integer(ctx, int_attr.clone());
            }
            if let Some(fp32) = (*attr).downcast_ref::<FPSingleAttr>() {
                let fp32 = fp32.clone();
                let ty: TypeHandle = FP32Type::get(ctx).into();
                let op = Operation::new(
                    ctx,
                    dm_ops::MirFloatConstantOp::get_concrete_op_info(),
                    vec![ty],
                    vec![],
                    vec![],
                    0,
                );
                dm_ops::MirFloatConstantOp::new(op)
                    .set_attr_float_value(ctx, fp32);
                return ConstantOp { op };
            }
            if let Some(fp64) = (*attr).downcast_ref::<FPDoubleAttr>() {
                let fp64 = fp64.clone();
                let ty: TypeHandle = FP64Type::get(ctx).into();
                let op = Operation::new(
                    ctx,
                    dm_ops::MirFloatConstantOp::get_concrete_op_info(),
                    vec![ty],
                    vec![],
                    vec![],
                    0,
                );
                dm_ops::MirFloatConstantOp::new(op)
                    .set_attr_float_value_f64(ctx, fp64);
                return ConstantOp { op };
            }
            panic!("oxide constant shim: unsupported constant attribute kind");
        }

        pub fn new_integer(ctx: &mut Context, attr: IntegerAttr) -> Self {
            let ty: TypeHandle = attr.get_type().into();
            let op = Operation::new(
                ctx,
                dm_ops::MirConstantOp::get_concrete_op_info(),
                vec![ty],
                vec![],
                vec![],
                0,
            );
            dm_ops::MirConstantOp::new(op).set_attr_value(ctx, attr);
            ConstantOp { op }
        }
    }
    shim_common!(ConstantOp);

    pub struct UndefOp {
        op: Ptr<Operation>,
    }

    impl UndefOp {
        pub fn new(ctx: &mut Context, result_ty: TypeHandle) -> Self {
            let undef = pllvm::ops::UndefOp::new(ctx, result_ty);
            UndefOp {
                op: undef.get_operation(),
            }
        }
    }
    shim_common!(UndefOp);

    pub struct AddressOfOp {
        op: Ptr<Operation>,
    }

    impl AddressOfOp {
        pub fn new(
            ctx: &mut Context,
            symbol: Identifier,
            _result_ty: TypeHandle,
        ) -> Self {
            let addr = pllvm::ops::AddressOfOp::new(ctx, symbol, 0);
            AddressOfOp {
                op: addr.get_operation(),
            }
        }
    }
    shim_common!(AddressOfOp);

    pub struct AllocaOp {
        op: Ptr<Operation>,
    }

    impl AllocaOp {
        pub fn new(ctx: &mut Context, elem_type: TypeHandle) -> Self {
            let ptr_ty: TypeHandle =
                super::types::PtrType::get(ctx, elem_type, true).into();
            let op = Operation::new(
                ctx,
                dm_ops::MirAllocaOp::get_concrete_op_info(),
                vec![ptr_ty],
                vec![],
                vec![],
                0,
            );
            AllocaOp { op }
        }
    }
    shim_common!(AllocaOp);

    pub struct LoadOp {
        op: Ptr<Operation>,
    }

    impl LoadOp {
        pub fn new(ctx: &mut Context, addr: Value, result_type: TypeHandle) -> Self {
            let op = Operation::new(
                ctx,
                dm_ops::MirLoadOp::get_concrete_op_info(),
                vec![result_type],
                vec![addr],
                vec![],
                0,
            );
            LoadOp { op }
        }
    }
    shim_common!(LoadOp);

    pub struct StoreOp {
        op: Ptr<Operation>,
    }

    impl StoreOp {
        /// Old cmir operand order is `(value, addr)`; dialect-mir stores
        /// take `[ptr, value]`.
        pub fn new(ctx: &mut Context, value: Value, addr: Value) -> Self {
            let op = Operation::new(
                ctx,
                dm_ops::MirStoreOp::get_concrete_op_info(),
                vec![],
                vec![addr, value],
                vec![],
                0,
            );
            StoreOp { op }
        }
    }
    shim_common!(StoreOp);

    pub struct PtrOffsetOp {
        op: Ptr<Operation>,
    }

    impl PtrOffsetOp {
        /// Byte-addressed pointer arithmetic. Emitted directly as an
        /// `llvm.getelementptr` over `i8` (the exact op both the old and
        /// the dialect-mir lowering would produce for byte offsets):
        /// upstream `mir.ptr_offset` is element-scaled, which is not
        /// this op's contract.
        pub fn new(ctx: &mut Context, base: Value, byte_offset: Value) -> Self {
            let i8_ty: TypeHandle =
                IntegerType::get(ctx, 8, Signedness::Unsigned).into();
            let gep = pllvm::ops::GetElementPtrOp::new(
                ctx,
                base,
                vec![pllvm::ops::GepIndex::Value(byte_offset)],
                i8_ty,
            );
            PtrOffsetOp {
                op: gep.get_operation(),
            }
        }
    }
    shim_common!(PtrOffsetOp);

    pub struct CastOp {
        op: Ptr<Operation>,
    }

    impl CastOp {
        pub fn new(ctx: &mut Context, input: Value, result_type: TypeHandle) -> Self {
            let input_ty = input.get_type(ctx);
            let Some(kind) = cast_kind(ctx, input_ty, result_type) else {
                // Aggregate-involving casts keep the old cmir semantics: a
                // lax llvm.bitcast (their mir.cast Transmute size-checks).
                let bitcast = pllvm::ops::BitcastOp::new(ctx, input, result_type);
                return CastOp {
                    op: bitcast.get_operation(),
                };
            };
            let op = Operation::new(
                ctx,
                dm_ops::MirCastOp::get_concrete_op_info(),
                vec![result_type],
                vec![input],
                vec![],
                0,
            );
            dm_ops::MirCastOp::new(op).set_attr_cast_kind(ctx, kind);
            CastOp { op }
        }
    }
    shim_common!(CastOp);

    pub struct ExtractValueOp {
        op: Ptr<Operation>,
    }

    impl ExtractValueOp {
        pub fn new(
            ctx: &mut Context,
            aggregate: Value,
            indices: Vec<u32>,
            _result_type: TypeHandle,
        ) -> Self {
            let extract = pllvm::ops::ExtractValueOp::new(ctx, aggregate, indices)
                .expect("oxide extractvalue shim: invalid indices for aggregate");
            ExtractValueOp {
                op: extract.get_operation(),
            }
        }
    }
    shim_common!(ExtractValueOp);

    pub struct InsertValueOp {
        op: Ptr<Operation>,
    }

    impl InsertValueOp {
        /// Old cmir operand order is `(value, aggregate, ..)`; the llvm op
        /// takes `(aggregate, value, ..)`.
        pub fn new(
            ctx: &mut Context,
            value: Value,
            aggregate: Value,
            indices: Vec<u32>,
        ) -> Self {
            let insert = pllvm::ops::InsertValueOp::new(ctx, aggregate, value, indices);
            InsertValueOp {
                op: insert.get_operation(),
            }
        }
    }
    shim_common!(InsertValueOp);

    pub struct ReturnOp {
        op: Ptr<Operation>,
    }

    impl ReturnOp {
        pub fn new(ctx: &mut Context, retval: Option<Value>) -> Self {
            let op = Operation::new(
                ctx,
                dm_ops::MirReturnOp::get_concrete_op_info(),
                vec![],
                retval.into_iter().collect(),
                vec![],
                0,
            );
            ReturnOp { op }
        }
    }
    shim_common!(ReturnOp);

    pub struct GotoOp {
        op: Ptr<Operation>,
    }

    impl GotoOp {
        pub fn new(
            ctx: &mut Context,
            dest: Ptr<BasicBlock>,
            dest_operands: Vec<Value>,
        ) -> Self {
            let op = Operation::new(
                ctx,
                dm_ops::MirGotoOp::get_concrete_op_info(),
                vec![],
                dest_operands,
                vec![dest],
                0,
            );
            GotoOp { op }
        }
    }
    shim_common!(GotoOp);

    pub struct CondBrOp {
        op: Ptr<Operation>,
    }

    impl CondBrOp {
        pub fn new(
            ctx: &mut Context,
            condition: Value,
            true_dest: Ptr<BasicBlock>,
            true_operands: Vec<Value>,
            false_dest: Ptr<BasicBlock>,
            false_operands: Vec<Value>,
        ) -> Self {
            let (operands, sizes) = dm_ops::MirCondBranchOp::compute_segment_sizes(
                vec![vec![condition], true_operands, false_operands],
            );
            let op = Operation::new(
                ctx,
                dm_ops::MirCondBranchOp::get_concrete_op_info(),
                vec![],
                operands,
                vec![true_dest, false_dest],
                0,
            );
            dm_ops::MirCondBranchOp::new(op).set_operand_segment_sizes(ctx, sizes);
            CondBrOp { op }
        }
    }
    shim_common!(CondBrOp);

    pub struct UnreachableOp {
        op: Ptr<Operation>,
    }

    impl UnreachableOp {
        pub fn new(ctx: &mut Context) -> Self {
            let unreachable = dm_ops::MirUnreachableOp::new(ctx);
            UnreachableOp {
                op: unreachable.get_operation(),
            }
        }
    }
    shim_common!(UnreachableOp);

    pub struct CallOp {
        op: Ptr<Operation>,
    }

    impl CallOp {
        pub fn new_direct(
            ctx: &mut Context,
            callee: Identifier,
            args: Vec<Value>,
            result_type: Option<TypeHandle>,
        ) -> Self {
            let op = Operation::new(
                ctx,
                dm_ops::MirCallOp::get_concrete_op_info(),
                result_type.into_iter().collect(),
                args,
                vec![],
                0,
            );
            dm_ops::MirCallOp::new(op)
                .set_attr_callee(ctx, StringAttr::new(callee.to_string()));
            CallOp { op }
        }

        /// Indirect call through a function-pointer value. dialect-mir
        /// has no indirect `mir.call` form, so this is emitted directly
        /// as an indirect `llvm.call` (which mir-lower passes through,
        /// remapping operands as their defining ops convert).
        pub fn new_indirect(
            ctx: &mut Context,
            callee: Value,
            args: Vec<Value>,
            result_type: Option<TypeHandle>,
        ) -> Self {
            // The FuncType is snapshotted now, before dialect
            // conversion rewrites values: any dialect-mir pointer type
            // must be recorded as an opaque llvm pointer or the isel
            // later reads foreign types out of the call's ABI signature.
            let sanitize = |ctx: &mut Context, ty: TypeHandle| -> TypeHandle {
                if ty.deref(ctx).is::<dm_types::MirPtrType>() {
                    pllvm::types::PointerType::get(ctx, 0).into()
                } else {
                    ty
                }
            };
            let result_ty = result_type
                .unwrap_or_else(|| pllvm::types::VoidType::get(ctx).into());
            let result_ty = sanitize(ctx, result_ty);
            let arg_tys: Vec<TypeHandle> = args
                .iter()
                .map(|a| a.get_type(ctx))
                .collect::<Vec<_>>()
                .into_iter()
                .map(|t| sanitize(ctx, t))
                .collect();
            let func_ty = pllvm::types::FuncType::get(ctx, result_ty, arg_tys, false);
            let call = pllvm::ops::CallOp::new(
                ctx,
                CallOpCallable::Indirect(callee),
                func_ty,
                args,
            );
            CallOp {
                op: call.get_operation(),
            }
        }
    }
    shim_common!(CallOp);

    shim_binop!(AddOp, MirAddOp, AddOp);
    shim_binop!(SubOp, MirSubOp, SubOp);
    shim_binop!(MulOp, MirMulOp, MulOp);
    shim_binop!(DivOp, MirDivOp, UDivOp);
    shim_binop!(RemOp, MirRemOp, URemOp);
    shim_binop!(BitAndOp, MirBitAndOp, AndOp);
    shim_binop!(BitOrOp, MirBitOrOp, OrOp);
    shim_binop!(BitXorOp, MirBitXorOp, XorOp);
    shim_binop!(ShlOp, MirShlOp, ShlOp);

    /// The importer's own synthesized shifts (intrinsic expansions,
    /// overflow decompositions) are always logical, matching the
    /// unsigned bit-pattern math they implement.
    pub struct ShrOp {
        op: Ptr<Operation>,
    }

    impl ShrOp {
        pub fn new(ctx: &mut Context, lhs: Value, rhs: Value) -> Self {
            let lshr = pllvm::ops::LShrOp::new(ctx, lhs, rhs);
            ShrOp {
                op: lshr.get_operation(),
            }
        }
    }
    shim_common!(ShrOp);

    // Rust-level `>>` goes through dialect-mir's `mir.shr`, which
    // mir-lower turns into `llvm.ashr` for signed operands and
    // `llvm.lshr` for unsigned ones — the aarch64 isel lowers both
    // (asr since the FP round; x86_64 still rejects `ashr` with a
    // precise error).
    shim_binop!(SignAwareShrOp, MirShrOp, LShrOp);

    pub struct NegOp {
        op: Ptr<Operation>,
    }

    impl NegOp {
        pub fn new(ctx: &mut Context, input: Value) -> Self {
            let ty = input.get_type(ctx);
            let op = Operation::new(
                ctx,
                dm_ops::MirNegOp::get_concrete_op_info(),
                vec![ty],
                vec![input],
                vec![],
                0,
            );
            NegOp { op }
        }
    }
    shim_common!(NegOp);

    shim_cmpop!(EqOp, MirEqOp, EQ);
    shim_cmpop!(NeOp, MirNeOp, NE);
    shim_cmpop!(LtOp, MirLtOp, ULT);
    shim_cmpop!(LeOp, MirLeOp, ULE);
    shim_cmpop!(GtOp, MirGtOp, UGT);
    shim_cmpop!(GeOp, MirGeOp, UGE);
}

/// Pick the dialect-mir cast kind for a `(input, result)` type pair, the
/// way the old `cmir.cast` conversion classified casts structurally.
fn cast_kind(
    ctx: &Context,
    input: TypeHandle,
    result: TypeHandle,
) -> Option<MirCastKindAttr> {
    #[derive(PartialEq)]
    enum Kind {
        Int,
        Float,
        Pointer,
        Other,
    }
    let classify = |ty: TypeHandle| -> Kind {
        let ty_ref = ty.deref(ctx);
        if ty_ref.is::<IntegerType>() {
            Kind::Int
        } else if ty_ref.is::<crate::dialects::builtin::types::FP32Type>()
            || ty_ref.is::<crate::dialects::builtin::types::FP64Type>()
        {
            Kind::Float
        } else if ty_ref.is::<dm_types::MirPtrType>()
            || ty_ref.is::<pllvm::types::PointerType>()
        {
            Kind::Pointer
        } else {
            Kind::Other
        }
    };
    match (classify(input), classify(result)) {
        (Kind::Int, Kind::Int) => Some(MirCastKindAttr::IntToInt),
        (Kind::Int, Kind::Float) => Some(MirCastKindAttr::IntToFloat),
        (Kind::Float, Kind::Int) => Some(MirCastKindAttr::FloatToInt),
        (Kind::Float, Kind::Float) => Some(MirCastKindAttr::FloatToFloat),
        (Kind::Pointer, Kind::Pointer) => Some(MirCastKindAttr::PtrToPtr),
        (Kind::Pointer, Kind::Int) => Some(MirCastKindAttr::PointerExposeAddress),
        (Kind::Int, Kind::Pointer) => Some(MirCastKindAttr::PointerWithExposedProvenance),
        _ => None,
    }
}
