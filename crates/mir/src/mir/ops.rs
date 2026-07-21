//! Operations defined by the Rust MIR dialect.

use combine::{Parser, optional, parser::char::char, sep_by, token, value};
use pliron::derive::{def_op, derive_op_interface_impl};
use pliron::derive::derive_attr_get_set;

use llvm_compat::op_interfaces::FunctionLikeInterface;

use crate::{
    common_traits::Named,
    context::{Context, Ptr},
    dialects::{
        builtin::{
            attributes::{IdentifierAttr, IntegerAttr, StringAttr, TypeAttr},
            op_interfaces::{
                self, BranchOpInterface, IsTerminatorInterface,
                IsolatedFromAboveInterface, OneRegionInterface, OneResultInterface,
                OperandSegmentInterface, SameOperandsAndResultType, SameOperandsType,
                SameResultsType, SymbolOpInterface, NOpdsInterface, NResultsInterface,
            },
            type_interfaces::FunctionTypeInterface,
            types::FunctionType,
        },
        mir::{attributes::InsertExtractValueIndicesAttr, types::PtrType},
    },
    dict_key,
    identifier::Identifier,
    impl_verify_succ, input_err,
    ir::{
        attribute::{AttrObj, attr_cast},
        basic_block::BasicBlock,
        irfmt::{
            parsers::{
                attr_parser, block_opd_parser, process_parsed_ssa_defs, spaced, ssa_opd_parser,
                type_parser,
            },
            printers::op::region,
        },
        location::{Located, Location},
        op::{Op, OpObj},
        operation::Operation,
        region::Region,
        r#type::{TypeHandle, TypedHandle, Typed},
        value::Value,
    },
    linked_list::ContainsLinkedList,
    parsable::{Parsable, ParseResult, StateStream},
    printable::{self, Printable},
};

dict_key!(ATTR_KEY_MIR_ALLOCA_ELEM_TYPE, "mir_alloca_elem_type");
dict_key!(ATTR_KEY_MIR_CONSTANT_VALUE, "mir_constant_value");
dict_key!(ATTR_KEY_MIR_CALLEE, "mir_callee");
dict_key!(ATTR_KEY_MIR_CSTR_VALUE, "mir_cstr_value");
dict_key!(ATTR_KEY_MIR_SYMBOL, "mir_symbol");
dict_key!(
    ATTR_KEY_MIR_EXTRACTVALUE_INDICES,
    "mir_extractvalue_indices"
);
dict_key!(ATTR_KEY_MIR_INSERTVALUE_INDICES, "mir_insertvalue_indices");

// ============================================================================
// Function operation
// ============================================================================

/// Rust MIR function.
#[def_op("mir.func")]
#[derive_op_interface_impl(
    OneRegionInterface,
    FunctionLikeInterface,
    SymbolOpInterface,
    IsolatedFromAboveInterface,
    NOpdsInterface<0>,
    NResultsInterface<0>
)]
#[derive_attr_get_set(mir_func_type : TypeAttr)]
pub struct FuncOp;

impl FuncOp {
    pub fn new(ctx: &mut Context, name: Identifier, ty: TypedHandle<FunctionType>) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 1);
        let arg_types = ty.deref(ctx).arg_types();
        let region = op.deref_mut(ctx).get_region(0);
        let entry = BasicBlock::new(ctx, Some("entry".try_into().unwrap()), arg_types);
        entry.insert_at_front(region, ctx);

        let func = FuncOp { op };
        func.set_symbol_name(ctx, name);
        func.set_attr_mir_func_type(ctx, TypeAttr::new(ty.into()));
        func
    }

    pub fn get_func_type(&self, ctx: &Context) -> TypeHandle {
        self.get_attr_mir_func_type(ctx).unwrap().get_type(ctx)
    }

    pub fn get_entry_block(&self, ctx: &Context) -> Ptr<BasicBlock> {
        self.get_region(ctx).deref(ctx).get_head().unwrap()
    }
}

impl Typed for FuncOp {
    fn get_type(&self, ctx: &Context) -> TypeHandle {
        self.get_func_type(ctx)
    }
}

impl Printable for FuncOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(
            f,
            "{} @{} : ",
            self.get_opid().disp(ctx),
            self.get_symbol_name(ctx)
        )?;
        self.get_func_type(ctx).fmt(ctx, state, f)?;
        write!(f, " ")?;
        region(self).fmt(ctx, state, f)
    }
}

impl Parsable for FuncOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        if !results.is_empty() {
            input_err!(
                state_stream.loc(),
                op_interfaces::NResultsVerifyErr(0, results.len())
            )?
        }

        let op = Operation::new(
            state_stream.state.ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );
        let mut parser = (
            spaced(token('@').with(Identifier::parser(()))),
            spaced(token(':')).with(spaced(type_parser())),
            spaced(Region::parser(op)),
        );

        parser
            .parse_stream(state_stream)
            .map(|(name, func_type, _region)| -> OpObj {
                let ctx = &mut *state_stream.state.ctx;
                let func = FuncOp { op };
                func.set_symbol_name(ctx, name);
                func.set_attr_mir_func_type(ctx, TypeAttr::new(func_type));
                OpObj::new(func)
            })
            .into()
    }
}

impl_verify_succ!(FuncOp);

// ============================================================================
// Constants and memory
// ============================================================================

/// Integer constant.
#[def_op("mir.constant")]
#[derive_op_interface_impl(OneResultInterface, NOpdsInterface<0>)]
pub struct ConstantOp;

impl ConstantOp {
    pub fn new(ctx: &mut Context, attr: AttrObj) -> Self {
        let ty =
            attr_cast::<dyn crate::dialects::builtin::attr_interfaces::TypedAttrInterface>(&*attr)
                .expect("mir.constant requires a typed attribute")
                .get_type(ctx);
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![ty], vec![], vec![], 0);
        let constant = ConstantOp { op };
        constant
            .get_operation()
            .deref_mut(ctx)
            .attributes
            .0
            .insert(ATTR_KEY_MIR_CONSTANT_VALUE.clone(), attr);
        constant
    }

    pub fn new_integer(ctx: &mut Context, attr: IntegerAttr) -> Self {
        Self::new(ctx, attr.into())
    }

    pub fn get_value(&self, ctx: &Context) -> Option<IntegerAttr> {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<IntegerAttr>(&ATTR_KEY_MIR_CONSTANT_VALUE)
            .cloned()
    }

    pub fn get_attr(&self, ctx: &Context) -> Option<AttrObj> {
        self.get_operation()
            .deref(ctx)
            .attributes
            .0
            .get(&ATTR_KEY_MIR_CONSTANT_VALUE)
            .cloned()
    }
}

impl Printable for ConstantOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        self.get_result(ctx).fmt(ctx, state, f)?;
        write!(f, " = {} ", self.get_opid().disp(ctx))?;
        if let Some(value) = self.get_attr(ctx) {
            value.fmt(ctx, state, f)?;
        }
        Ok(())
    }
}

impl Parsable for ConstantOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let mut parser = spaced(attr_parser());
        parser
            .parse_stream(state_stream)
            .map(|attr| -> OpObj {
                let ctx = &mut *state_stream.state.ctx;
                let op = ConstantOp::new(ctx, attr);
                if !results.is_empty() {
                    process_parsed_ssa_defs(state_stream, &results, op.get_operation()).ok();
                }
                OpObj::new(op)
            })
            .into()
    }
}

impl_verify_succ!(ConstantOp);

/// Pointer to a null-terminated string literal.
#[def_op("mir.cstr")]
#[derive_op_interface_impl(OneResultInterface, NOpdsInterface<0>)]
pub struct CStrOp;

impl CStrOp {
    pub fn new(ctx: &mut Context, value: String) -> Self {
        let i8_ty: TypeHandle = crate::dialects::builtin::types::IntegerType::get(
            ctx,
            8,
            crate::dialects::builtin::types::Signedness::Unsigned,
        )
        .into();
        let ptr_ty: TypeHandle = PtrType::get(ctx, i8_ty, false).into();
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![ptr_ty],
            vec![],
            vec![],
            0,
        );
        let cstr = CStrOp { op };
        cstr.get_operation()
            .deref_mut(ctx)
            .attributes
            .set(ATTR_KEY_MIR_CSTR_VALUE.clone(), StringAttr::new(value));
        cstr
    }

    pub fn get_value(&self, ctx: &Context) -> String {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<StringAttr>(&ATTR_KEY_MIR_CSTR_VALUE)
            .cloned()
            .map(String::from)
            .unwrap()
    }
}

impl Printable for CStrOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        self.get_result(ctx).fmt(ctx, state, f)?;
        write!(f, " = {} ", self.get_opid().disp(ctx))?;
        StringAttr::new(self.get_value(ctx)).fmt(ctx, state, f)
    }
}

impl Parsable for CStrOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let mut parser = spaced(StringAttr::parser(()));
        parser
            .parse_stream(state_stream)
            .map(|attr| -> OpObj {
                let ctx = &mut *state_stream.state.ctx;
                let op = CStrOp::new(ctx, attr.into());
                if !results.is_empty() {
                    process_parsed_ssa_defs(state_stream, &results, op.get_operation()).ok();
                }
                OpObj::new(op)
            })
            .into()
    }
}

impl_verify_succ!(CStrOp);

/// Address of a symbol visible to later object emission.
#[def_op("mir.addressof")]
#[derive_op_interface_impl(OneResultInterface, NOpdsInterface<0>)]
pub struct AddressOfOp;

impl AddressOfOp {
    pub fn new(ctx: &mut Context, symbol: Identifier, result_ty: TypeHandle) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_ty],
            vec![],
            vec![],
            0,
        );
        let addr = AddressOfOp { op };
        addr.get_operation()
            .deref_mut(ctx)
            .attributes
            .set(ATTR_KEY_MIR_SYMBOL.clone(), IdentifierAttr::new(symbol));
        addr
    }

    pub fn get_symbol(&self, ctx: &Context) -> Identifier {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<IdentifierAttr>(&ATTR_KEY_MIR_SYMBOL)
            .cloned()
            .unwrap()
            .into()
    }
}

impl Printable for AddressOfOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        self.get_result(ctx).fmt(ctx, state, f)?;
        write!(
            f,
            " = {} @{} : ",
            self.get_opid().disp(ctx),
            self.get_symbol(ctx)
        )?;
        self.get_result(ctx).get_type(ctx).fmt(ctx, state, f)
    }
}

impl Parsable for AddressOfOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let mut parser = spaced(token('@').with(Identifier::parser(())))
            .skip(spaced(token(':')))
            .and(spaced(type_parser()));

        parser
            .parse_stream(state_stream)
            .map(|(symbol, result_ty)| -> OpObj {
                let ctx = &mut *state_stream.state.ctx;
                let op = AddressOfOp::new(ctx, symbol, result_ty);
                if !results.is_empty() {
                    process_parsed_ssa_defs(state_stream, &results, op.get_operation()).ok();
                }
                OpObj::new(op)
            })
            .into()
    }
}

impl_verify_succ!(AddressOfOp);

/// Undefined aggregate or scalar value used as a construction seed.
#[def_op("mir.undef")]
#[derive_op_interface_impl(OneResultInterface, NOpdsInterface<0>)]
pub struct UndefOp;

impl UndefOp {
    pub fn new(ctx: &mut Context, result_ty: TypeHandle) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_ty],
            vec![],
            vec![],
            0,
        );
        UndefOp { op }
    }
}

impl Printable for UndefOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        self.get_result(ctx).fmt(ctx, state, f)?;
        write!(f, " = {} : ", self.get_opid().disp(ctx))?;
        self.get_result(ctx).get_type(ctx).fmt(ctx, state, f)
    }
}

impl Parsable for UndefOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let mut parser = spaced(token(':')).with(spaced(type_parser()));
        parser
            .parse_stream(state_stream)
            .map(|ty| -> OpObj {
                let ctx = &mut *state_stream.state.ctx;
                let op = UndefOp::new(ctx, ty);
                if !results.is_empty() {
                    process_parsed_ssa_defs(state_stream, &results, op.get_operation()).ok();
                }
                OpObj::new(op)
            })
            .into()
    }
}

impl_verify_succ!(UndefOp);

/// Stack slot for a MIR local/place.
#[def_op("mir.alloca")]
#[derive_op_interface_impl(OneResultInterface, NOpdsInterface<0>)]
pub struct AllocaOp;

impl AllocaOp {
    pub fn new(ctx: &mut Context, elem_type: TypeHandle) -> Self {
        let ptr_ty: TypeHandle = PtrType::get(ctx, elem_type, true).into();
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![ptr_ty],
            vec![],
            vec![],
            0,
        );
        let alloca = AllocaOp { op };
        alloca.get_operation().deref_mut(ctx).attributes.set(
            ATTR_KEY_MIR_ALLOCA_ELEM_TYPE.clone(),
            TypeAttr::new(elem_type),
        );
        alloca
    }

    pub fn get_elem_type(&self, ctx: &Context) -> TypeHandle {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<TypeAttr>(&ATTR_KEY_MIR_ALLOCA_ELEM_TYPE)
            .unwrap()
            .get_type(ctx)
    }
}

impl Printable for AllocaOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        self.get_result(ctx).fmt(ctx, state, f)?;
        write!(f, " = {} : ", self.get_opid().disp(ctx))?;
        self.get_elem_type(ctx).fmt(ctx, state, f)
    }
}

impl Parsable for AllocaOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let mut parser = spaced(token(':')).with(spaced(type_parser()));
        parser
            .parse_stream(state_stream)
            .map(|elem_type| -> OpObj {
                let ctx = &mut *state_stream.state.ctx;
                let op = AllocaOp::new(ctx, elem_type);
                if !results.is_empty() {
                    process_parsed_ssa_defs(state_stream, &results, op.get_operation()).ok();
                }
                OpObj::new(op)
            })
            .into()
    }
}

impl_verify_succ!(AllocaOp);

/// Load from a MIR pointer/place.
#[def_op("mir.load")]
#[derive_op_interface_impl(OneResultInterface)]
pub struct LoadOp;

impl LoadOp {
    pub fn new(ctx: &mut Context, addr: Value, result_type: TypeHandle) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_type],
            vec![addr],
            vec![],
            0,
        );
        LoadOp { op }
    }

    pub fn get_addr(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}

impl Printable for LoadOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        self.get_result(ctx).fmt(ctx, state, f)?;
        write!(f, " = {} ", self.get_opid().disp(ctx))?;
        self.get_addr(ctx).fmt(ctx, state, f)?;
        write!(f, " : ")?;
        self.get_result(ctx).get_type(ctx).fmt(ctx, state, f)
    }
}

impl Parsable for LoadOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let mut parser = spaced(ssa_opd_parser())
            .skip(spaced(token(':')))
            .and(spaced(type_parser()));
        parser
            .parse_stream(state_stream)
            .map(|(addr, result_type)| -> OpObj {
                let ctx = &mut *state_stream.state.ctx;
                let op = LoadOp::new(ctx, addr, result_type);
                if !results.is_empty() {
                    process_parsed_ssa_defs(state_stream, &results, op.get_operation()).ok();
                }
                OpObj::new(op)
            })
            .into()
    }
}

impl_verify_succ!(LoadOp);

/// Store to a MIR pointer/place.
#[def_op("mir.store")]
#[derive_op_interface_impl(NResultsInterface<0>)]
pub struct StoreOp;

impl StoreOp {
    pub fn new(ctx: &mut Context, value: Value, addr: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![value, addr],
            vec![],
            0,
        );
        StoreOp { op }
    }

    pub fn get_value(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    pub fn get_addr(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}

impl Printable for StoreOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "{} ", self.get_opid().disp(ctx))?;
        self.get_value(ctx).fmt(ctx, state, f)?;
        write!(f, ", ")?;
        self.get_addr(ctx).fmt(ctx, state, f)?;
        write!(f, " : ")?;
        self.get_value(ctx).get_type(ctx).fmt(ctx, state, f)
    }
}

impl Parsable for StoreOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        if !results.is_empty() {
            input_err!(
                state_stream.loc(),
                op_interfaces::NResultsVerifyErr(0, results.len())
            )?
        }
        let mut parser = spaced(ssa_opd_parser())
            .skip(spaced(token(',')))
            .and(spaced(ssa_opd_parser()))
            .skip(spaced(token(':')))
            .skip(spaced(type_parser()));
        parser
            .parse_stream(state_stream)
            .map(|(value, addr)| -> OpObj {
                OpObj::new(StoreOp::new(state_stream.state.ctx, value, addr))
            })
            .into()
    }
}

impl_verify_succ!(StoreOp);

/// Compute a pointer advanced by a byte offset.
#[def_op("mir.ptr_offset")]
#[derive_op_interface_impl(OneResultInterface)]
pub struct PtrOffsetOp;

impl PtrOffsetOp {
    pub fn new(ctx: &mut Context, base: Value, byte_offset: Value) -> Self {
        let ptr_ty = base.get_type(ctx);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![ptr_ty],
            vec![base, byte_offset],
            vec![],
            0,
        );
        PtrOffsetOp { op }
    }

    pub fn get_base(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    pub fn get_byte_offset(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}

impl Printable for PtrOffsetOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        self.get_result(ctx).fmt(ctx, state, f)?;
        write!(f, " = {} ", self.get_opid().disp(ctx))?;
        self.get_base(ctx).fmt(ctx, state, f)?;
        write!(f, ", ")?;
        self.get_byte_offset(ctx).fmt(ctx, state, f)
    }
}

impl Parsable for PtrOffsetOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let mut parser = spaced(ssa_opd_parser())
            .skip(spaced(token(',')))
            .and(spaced(ssa_opd_parser()));
        parser
            .parse_stream(state_stream)
            .map(|(base, byte_offset)| -> OpObj {
                let ctx = &mut *state_stream.state.ctx;
                let op = PtrOffsetOp::new(ctx, base, byte_offset);
                if !results.is_empty() {
                    process_parsed_ssa_defs(state_stream, &results, op.get_operation()).ok();
                }
                OpObj::new(op)
            })
            .into()
    }
}

impl_verify_succ!(PtrOffsetOp);

// ============================================================================
// Arithmetic and comparison operations
// ============================================================================

macro_rules! def_mir_binary_op {
    ($name:ident, $opid:literal) => {
        #[def_op($opid)]
        #[derive_op_interface_impl(
            OneResultInterface,
            SameOperandsType,
            SameResultsType,
            SameOperandsAndResultType
        )]
        pub struct $name;

        impl $name {
            pub fn new(ctx: &mut Context, lhs: Value, rhs: Value) -> Self {
                let result_type = lhs.get_type(ctx);
                let op = Operation::new(
                    ctx,
                    Self::get_concrete_op_info(),
                    vec![result_type],
                    vec![lhs, rhs],
                    vec![],
                    0,
                );
                $name { op }
            }

            pub fn get_lhs(&self, ctx: &Context) -> Value {
                self.get_operation().deref(ctx).get_operand(0)
            }

            pub fn get_rhs(&self, ctx: &Context) -> Value {
                self.get_operation().deref(ctx).get_operand(1)
            }
        }

        impl Printable for $name {
            fn fmt(
                &self,
                ctx: &Context,
                state: &printable::State,
                f: &mut core::fmt::Formatter<'_>,
            ) -> core::fmt::Result {
                self.get_result(ctx).fmt(ctx, state, f)?;
                write!(f, " = {} ", self.get_opid().disp(ctx))?;
                self.get_lhs(ctx).fmt(ctx, state, f)?;
                write!(f, ", ")?;
                self.get_rhs(ctx).fmt(ctx, state, f)?;
                write!(f, " : ")?;
                self.get_lhs(ctx).get_type(ctx).fmt(ctx, state, f)
            }
        }

        impl Parsable for $name {
            type Arg = Vec<(Identifier, Location)>;
            type Parsed = OpObj;

            fn parse<'a>(
                state_stream: &mut StateStream<'a>,
                results: Self::Arg,
            ) -> ParseResult<'a, Self::Parsed> {
                let mut parser = spaced(ssa_opd_parser())
                    .skip(spaced(token(',')))
                    .and(spaced(ssa_opd_parser()))
                    .skip(spaced(token(':')))
                    .skip(spaced(type_parser()));
                parser
                    .parse_stream(state_stream)
                    .map(|(lhs, rhs)| -> OpObj {
                        let ctx = &mut *state_stream.state.ctx;
                        let op = $name::new(ctx, lhs, rhs);
                        if !results.is_empty() {
                            process_parsed_ssa_defs(state_stream, &results, op.get_operation()).ok();
                        }
                        OpObj::new(op)
                    })
                    .into()
            }
        }

        impl_verify_succ!($name);
    };
}

macro_rules! def_mir_cmp_op {
    ($name:ident, $opid:literal) => {
        #[def_op($opid)]
        #[derive_op_interface_impl(OneResultInterface, SameOperandsType)]
        pub struct $name;

        impl $name {
            pub fn new(ctx: &mut Context, lhs: Value, rhs: Value) -> Self {
                let i1 = crate::dialects::builtin::types::IntegerType::get(
                    ctx,
                    1,
                    crate::dialects::builtin::types::Signedness::Signless,
                );
                let op = Operation::new(
                    ctx,
                    Self::get_concrete_op_info(),
                    vec![i1.into()],
                    vec![lhs, rhs],
                    vec![],
                    0,
                );
                $name { op }
            }

            pub fn get_lhs(&self, ctx: &Context) -> Value {
                self.get_operation().deref(ctx).get_operand(0)
            }

            pub fn get_rhs(&self, ctx: &Context) -> Value {
                self.get_operation().deref(ctx).get_operand(1)
            }
        }

        impl Printable for $name {
            fn fmt(
                &self,
                ctx: &Context,
                state: &printable::State,
                f: &mut core::fmt::Formatter<'_>,
            ) -> core::fmt::Result {
                self.get_result(ctx).fmt(ctx, state, f)?;
                write!(f, " = {} ", self.get_opid().disp(ctx))?;
                self.get_lhs(ctx).fmt(ctx, state, f)?;
                write!(f, ", ")?;
                self.get_rhs(ctx).fmt(ctx, state, f)?;
                write!(f, " : ")?;
                self.get_lhs(ctx).get_type(ctx).fmt(ctx, state, f)
            }
        }

        impl Parsable for $name {
            type Arg = Vec<(Identifier, Location)>;
            type Parsed = OpObj;

            fn parse<'a>(
                state_stream: &mut StateStream<'a>,
                results: Self::Arg,
            ) -> ParseResult<'a, Self::Parsed> {
                let mut parser = spaced(ssa_opd_parser())
                    .skip(spaced(token(',')))
                    .and(spaced(ssa_opd_parser()))
                    .skip(spaced(token(':')))
                    .skip(spaced(type_parser()));
                parser
                    .parse_stream(state_stream)
                    .map(|(lhs, rhs)| -> OpObj {
                        let ctx = &mut *state_stream.state.ctx;
                        let op = $name::new(ctx, lhs, rhs);
                        if !results.is_empty() {
                            process_parsed_ssa_defs(state_stream, &results, op.get_operation()).ok();
                        }
                        OpObj::new(op)
                    })
                    .into()
            }
        }

        impl_verify_succ!($name);
    };
}

def_mir_binary_op!(AddOp, "mir.add");
def_mir_binary_op!(SubOp, "mir.sub");
def_mir_binary_op!(MulOp, "mir.mul");
def_mir_binary_op!(ShlOp, "mir.shl");
def_mir_binary_op!(ShrOp, "mir.shr");
def_mir_binary_op!(DivOp, "mir.div");
def_mir_binary_op!(RemOp, "mir.rem");
def_mir_binary_op!(BitAndOp, "mir.bitand");
def_mir_binary_op!(BitOrOp, "mir.bitor");
def_mir_binary_op!(BitXorOp, "mir.bitxor");

def_mir_cmp_op!(EqOp, "mir.eq");
def_mir_cmp_op!(NeOp, "mir.ne");
def_mir_cmp_op!(LtOp, "mir.lt");
def_mir_cmp_op!(LeOp, "mir.le");
def_mir_cmp_op!(GtOp, "mir.gt");
def_mir_cmp_op!(GeOp, "mir.ge");

/// Cast between scalar MIR types.
#[def_op("mir.cast")]
#[derive_op_interface_impl(OneResultInterface)]
pub struct CastOp;

impl CastOp {
    pub fn new(ctx: &mut Context, input: Value, result_type: TypeHandle) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_type],
            vec![input],
            vec![],
            0,
        );
        CastOp { op }
    }

    pub fn get_input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}

impl Printable for CastOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        self.get_result(ctx).fmt(ctx, state, f)?;
        write!(f, " = {} ", self.get_opid().disp(ctx))?;
        self.get_input(ctx).fmt(ctx, state, f)?;
        write!(f, " : ")?;
        self.get_input(ctx).get_type(ctx).fmt(ctx, state, f)?;
        write!(f, " to ")?;
        self.get_result(ctx).get_type(ctx).fmt(ctx, state, f)
    }
}

impl Parsable for CastOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let mut parser = spaced(ssa_opd_parser())
            .skip(spaced(token(':')))
            .skip(spaced(type_parser()))
            .skip(spaced(token('t')))
            .skip(spaced(token('o')))
            .and(spaced(type_parser()));
        parser
            .parse_stream(state_stream)
            .map(|(input, result_type)| -> OpObj {
                let ctx = &mut *state_stream.state.ctx;
                let op = CastOp::new(ctx, input, result_type);
                if !results.is_empty() {
                    process_parsed_ssa_defs(state_stream, &results, op.get_operation()).ok();
                }
                OpObj::new(op)
            })
            .into()
    }
}

impl_verify_succ!(CastOp);

/// Extract a field from an aggregate MIR value.
#[def_op("mir.extractvalue")]
#[derive_op_interface_impl(OneResultInterface)]
pub struct ExtractValueOp;

impl ExtractValueOp {
    pub fn new(
        ctx: &mut Context,
        aggregate: Value,
        indices: Vec<u32>,
        result_type: TypeHandle,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_type],
            vec![aggregate],
            vec![],
            0,
        );
        let extract = ExtractValueOp { op };
        extract.get_operation().deref_mut(ctx).attributes.set(
            ATTR_KEY_MIR_EXTRACTVALUE_INDICES.clone(),
            InsertExtractValueIndicesAttr(indices),
        );
        extract
    }

    pub fn get_aggregate(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    pub fn get_indices(&self, ctx: &Context) -> Vec<u32> {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<InsertExtractValueIndicesAttr>(&ATTR_KEY_MIR_EXTRACTVALUE_INDICES)
            .map(|attr| attr.0.clone())
            .unwrap_or_default()
    }
}

impl Printable for ExtractValueOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        self.get_result(ctx).fmt(ctx, state, f)?;
        write!(f, " = {} ", self.get_opid().disp(ctx))?;
        self.get_aggregate(ctx).fmt(ctx, state, f)?;
        write!(f, " ")?;
        InsertExtractValueIndicesAttr(self.get_indices(ctx)).fmt(ctx, state, f)?;
        write!(f, " : ")?;
        self.get_aggregate(ctx).get_type(ctx).fmt(ctx, state, f)
    }
}

impl Parsable for ExtractValueOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let mut parser = spaced(ssa_opd_parser())
            .and(spaced(InsertExtractValueIndicesAttr::parser(())))
            .skip(spaced(token(':')))
            .and(spaced(type_parser()));
        parser
            .parse_stream(state_stream)
            .map(|((aggregate, indices), result_type)| -> OpObj {
                let ctx = &mut *state_stream.state.ctx;
                let op = ExtractValueOp::new(ctx, aggregate, indices.0, result_type);
                if !results.is_empty() {
                    process_parsed_ssa_defs(state_stream, &results, op.get_operation()).ok();
                }
                OpObj::new(op)
            })
            .into()
    }
}

impl_verify_succ!(ExtractValueOp);

/// Insert a value into an aggregate MIR value.
#[def_op("mir.insertvalue")]
#[derive_op_interface_impl(OneResultInterface)]
pub struct InsertValueOp;

impl InsertValueOp {
    pub fn new(ctx: &mut Context, value: Value, aggregate: Value, indices: Vec<u32>) -> Self {
        let result_type = aggregate.get_type(ctx);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_type],
            vec![value, aggregate],
            vec![],
            0,
        );
        let insert = InsertValueOp { op };
        insert.get_operation().deref_mut(ctx).attributes.set(
            ATTR_KEY_MIR_INSERTVALUE_INDICES.clone(),
            InsertExtractValueIndicesAttr(indices),
        );
        insert
    }

    pub fn get_value(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    pub fn get_aggregate(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }

    pub fn get_indices(&self, ctx: &Context) -> Vec<u32> {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<InsertExtractValueIndicesAttr>(&ATTR_KEY_MIR_INSERTVALUE_INDICES)
            .map(|attr| attr.0.clone())
            .unwrap_or_default()
    }
}

impl Printable for InsertValueOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        self.get_result(ctx).fmt(ctx, state, f)?;
        write!(f, " = {} ", self.get_opid().disp(ctx))?;
        self.get_value(ctx).fmt(ctx, state, f)?;
        write!(f, ", ")?;
        self.get_aggregate(ctx).fmt(ctx, state, f)?;
        write!(f, " ")?;
        InsertExtractValueIndicesAttr(self.get_indices(ctx)).fmt(ctx, state, f)?;
        write!(f, " : ")?;
        self.get_aggregate(ctx).get_type(ctx).fmt(ctx, state, f)
    }
}

impl Parsable for InsertValueOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let mut parser = spaced(ssa_opd_parser())
            .skip(spaced(token(',')))
            .and(spaced(ssa_opd_parser()))
            .and(spaced(InsertExtractValueIndicesAttr::parser(())))
            .skip(spaced(token(':')))
            .skip(spaced(type_parser()));
        parser
            .parse_stream(state_stream)
            .map(|((value, aggregate), indices)| -> OpObj {
                let ctx = &mut *state_stream.state.ctx;
                let op = InsertValueOp::new(ctx, value, aggregate, indices.0);
                if !results.is_empty() {
                    process_parsed_ssa_defs(state_stream, &results, op.get_operation()).ok();
                }
                OpObj::new(op)
            })
            .into()
    }
}

impl_verify_succ!(InsertValueOp);

// ============================================================================
// Control flow
// ============================================================================

/// MIR return.
#[def_op("mir.return")]
#[derive_op_interface_impl(IsTerminatorInterface, NResultsInterface<0>)]
pub struct ReturnOp;

impl ReturnOp {
    pub fn new(ctx: &mut Context, retval: Option<Value>) -> Self {
        let operands = retval.into_iter().collect();
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], operands, vec![], 0);
        ReturnOp { op }
    }

    pub fn get_return_value(&self, ctx: &Context) -> Option<Value> {
        self.get_operation().deref(ctx).operands().next()
    }
}

impl Printable for ReturnOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "{}", self.get_opid().disp(ctx))?;
        if let Some(value) = self.get_return_value(ctx) {
            write!(f, " ")?;
            value.fmt(ctx, state, f)?;
            write!(f, " : ")?;
            value.get_type(ctx).fmt(ctx, state, f)?;
        }
        Ok(())
    }
}

impl Parsable for ReturnOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        if !results.is_empty() {
            input_err!(
                state_stream.loc(),
                op_interfaces::NResultsVerifyErr(0, results.len())
            )?
        }
        let mut parser = optional(
            spaced(ssa_opd_parser())
                .skip(spaced(token(':')))
                .skip(spaced(type_parser())),
        );
        parser
            .parse_stream(state_stream)
            .map(|retval| -> OpObj { OpObj::new(ReturnOp::new(state_stream.state.ctx, retval)) })
            .into()
    }
}

impl_verify_succ!(ReturnOp);

/// MIR unconditional branch.
#[def_op("mir.goto")]
#[derive_op_interface_impl(IsTerminatorInterface, NResultsInterface<0>)]
pub struct GotoOp;

impl BranchOpInterface for GotoOp {
    fn add_successor_operand(&self, ctx: &mut Context, _succ_idx: usize, operand: Value) -> usize {
        Operation::push_operand(self.get_operation(), ctx, operand)
    }

    fn remove_successor_operand(&self, ctx: &mut Context, _succ_idx: usize, opd_idx: usize) -> Value {
        Operation::remove_operand(self.get_operation(), ctx, opd_idx)
    }

    fn successor_operands(&self, ctx: &Context, _succ_idx: usize) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().collect()
    }
}

impl GotoOp {
    pub fn new(ctx: &mut Context, dest: Ptr<BasicBlock>, dest_operands: Vec<Value>) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            dest_operands,
            vec![dest],
            0,
        );
        GotoOp { op }
    }

    pub fn get_dest(&self, ctx: &Context) -> Ptr<BasicBlock> {
        self.get_operation().deref(ctx).get_successor(0)
    }

    pub fn get_dest_operands(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().collect()
    }
}

impl Printable for GotoOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(
            f,
            "{} ^{}",
            self.get_opid().disp(ctx),
            self.get_dest(ctx).deref(ctx).unique_name(ctx)
        )?;
        print_successor_operands(ctx, state, f, &self.get_dest_operands(ctx))
    }
}

impl Parsable for GotoOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        if !results.is_empty() {
            input_err!(
                state_stream.loc(),
                op_interfaces::NResultsVerifyErr(0, results.len())
            )?
        }
        let mut parser = spaced(block_opd_parser())
            .and(optional(spaced(successor_operands_parser())));
        parser
            .parse_stream(state_stream)
            .map(|(dest, operands)| -> OpObj {
                let ctx = &mut *state_stream.state.ctx;
                OpObj::new(GotoOp::new(ctx, dest, operands.unwrap_or_default()))
            })
            .into()
    }
}

impl_verify_succ!(GotoOp);

/// MIR conditional branch.
#[def_op("mir.cond_br")]
#[derive_op_interface_impl(IsTerminatorInterface, NResultsInterface<0>, OperandSegmentInterface)]
pub struct CondBrOp;

impl BranchOpInterface for CondBrOp {
    fn add_successor_operand(&self, ctx: &mut Context, succ_idx: usize, operand: Value) -> usize {
        // The successor operands start at segment 1; segment 0 is the condition.
        self.push_to_segment(ctx, succ_idx + 1, operand)
    }

    fn remove_successor_operand(&self, ctx: &mut Context, succ_idx: usize, opd_idx: usize) -> Value {
        // The successor operands start at segment 1; segment 0 is the condition.
        self.remove_from_segment(ctx, succ_idx + 1, opd_idx)
    }

    fn successor_operands(&self, ctx: &Context, succ_idx: usize) -> Vec<Value> {
        self.get_segment(ctx, succ_idx + 1)
    }
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
        let (operands, sizes) =
            Self::compute_segment_sizes(vec![vec![condition], true_operands, false_operands]);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            operands,
            vec![true_dest, false_dest],
            0,
        );
        let cond_br = CondBrOp { op };
        cond_br.set_operand_segment_sizes(ctx, sizes);
        cond_br
    }

    pub fn get_condition(&self, ctx: &Context) -> Value {
        self.get_segment(ctx, 0).into_iter().next().unwrap()
    }

    pub fn get_true_dest(&self, ctx: &Context) -> Ptr<BasicBlock> {
        self.get_operation().deref(ctx).get_successor(0)
    }

    pub fn get_false_dest(&self, ctx: &Context) -> Ptr<BasicBlock> {
        self.get_operation().deref(ctx).get_successor(1)
    }

    pub fn get_true_operands(&self, ctx: &Context) -> Vec<Value> {
        self.get_segment(ctx, 1)
    }

    pub fn get_false_operands(&self, ctx: &Context) -> Vec<Value> {
        self.get_segment(ctx, 2)
    }
}

impl Printable for CondBrOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "{} ", self.get_opid().disp(ctx))?;
        self.get_condition(ctx).fmt(ctx, state, f)?;
        write!(
            f,
            ", ^{}",
            self.get_true_dest(ctx).deref(ctx).unique_name(ctx)
        )?;
        print_successor_operands(ctx, state, f, &self.get_true_operands(ctx))?;
        write!(
            f,
            ", ^{}",
            self.get_false_dest(ctx).deref(ctx).unique_name(ctx)
        )?;
        print_successor_operands(ctx, state, f, &self.get_false_operands(ctx))
    }
}

impl Parsable for CondBrOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        if !results.is_empty() {
            input_err!(
                state_stream.loc(),
                op_interfaces::NResultsVerifyErr(0, results.len())
            )?
        }
        let block_parser = || {
            spaced(block_opd_parser())
                .and(optional(spaced(successor_operands_parser())))
        };
        let mut parser = spaced(ssa_opd_parser())
            .skip(spaced(token(',')))
            .and(block_parser())
            .skip(spaced(token(',')))
            .and(block_parser());
        parser
            .parse_stream(state_stream)
            .map(
                |((cond, (true_dest, true_opds)), (false_dest, false_opds))| -> OpObj {
                    let ctx = &mut *state_stream.state.ctx;
                    OpObj::new(CondBrOp::new(
                        ctx,
                        cond,
                        true_dest,
                        true_opds.unwrap_or_default(),
                        false_dest,
                        false_opds.unwrap_or_default(),
                    ))
                },
            )
            .into()
    }
}

impl_verify_succ!(CondBrOp);

/// MIR unreachable terminator.
#[def_op("mir.unreachable")]
#[derive_op_interface_impl(IsTerminatorInterface, NResultsInterface<0>, NOpdsInterface<0>)]
pub struct UnreachableOp;

impl UnreachableOp {
    pub fn new(ctx: &mut Context) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        UnreachableOp { op }
    }
}

impl Printable for UnreachableOp {
    fn fmt(
        &self,
        ctx: &Context,
        _state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "{}", self.get_opid().disp(ctx))
    }
}

impl Parsable for UnreachableOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        if !results.is_empty() {
            input_err!(
                state_stream.loc(),
                op_interfaces::NResultsVerifyErr(0, results.len())
            )?
        }
        let mut parser = value(());
        parser
            .parse_stream(state_stream)
            .map(|_| -> OpObj { OpObj::new(UnreachableOp::new(state_stream.state.ctx)) })
            .into()
    }
}

impl_verify_succ!(UnreachableOp);

// ============================================================================
// Calls
// ============================================================================

/// Direct MIR call.
#[def_op("mir.call")]
pub struct CallOp;

impl CallOp {
    pub fn new_direct(
        ctx: &mut Context,
        callee: Identifier,
        args: Vec<Value>,
        result_type: Option<TypeHandle>,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            result_type.into_iter().collect(),
            args,
            vec![],
            0,
        );
        let call = CallOp { op };
        call.get_operation()
            .deref_mut(ctx)
            .attributes
            .set(ATTR_KEY_MIR_CALLEE.clone(), IdentifierAttr::new(callee));
        call
    }

    /// Indirect MIR call: the callee function pointer is operand 0 and the
    /// call arguments follow it.
    pub fn new_indirect(
        ctx: &mut Context,
        callee: Value,
        args: Vec<Value>,
        result_type: Option<TypeHandle>,
    ) -> Self {
        let mut operands = Vec::with_capacity(args.len() + 1);
        operands.push(callee);
        operands.extend(args);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            result_type.into_iter().collect(),
            operands,
            vec![],
            0,
        );
        CallOp { op }
    }

    pub fn get_callee(&self, ctx: &Context) -> Identifier {
        self.get_callee_opt(ctx).unwrap()
    }

    /// The callee symbol for direct calls; `None` for indirect calls.
    pub fn get_callee_opt(&self, ctx: &Context) -> Option<Identifier> {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<IdentifierAttr>(&ATTR_KEY_MIR_CALLEE)
            .cloned()
            .map(Into::into)
    }

    pub fn get_args(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().collect()
    }
}

impl Printable for CallOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        let op_ref = self.get_operation().deref(ctx);
        if op_ref.get_num_results() > 0 {
            op_ref.get_result(0).fmt(ctx, state, f)?;
            write!(f, " = ")?;
        }
        drop(op_ref);

        write!(f, "{} ", self.get_opid().disp(ctx))?;
        if let Some(callee) = self.get_callee_opt(ctx) {
            write!(f, "@{}(", callee)?;
        } else {
            write!(f, "(")?;
        }
        let args = self.get_args(ctx);
        for (i, arg) in args.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            arg.fmt(ctx, state, f)?;
        }
        write!(f, ")")?;

        let op_ref = self.get_operation().deref(ctx);
        if op_ref.get_num_results() > 0 {
            write!(f, " : ")?;
            op_ref.get_result(0).get_type(ctx).fmt(ctx, state, f)?;
        }
        Ok(())
    }
}

impl Parsable for CallOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let args_parser = char('(')
            .with(sep_by(spaced(ssa_opd_parser()), spaced(char(','))))
            .skip(char(')'));
        let mut parser = spaced(token('@').with(Identifier::parser(())))
            .and(spaced(args_parser))
            .and(optional(spaced(token(':')).with(spaced(type_parser()))));

        parser
            .parse_stream(state_stream)
            .map(|((callee, args), result_type)| -> OpObj {
                let ctx = &mut *state_stream.state.ctx;
                let op = CallOp::new_direct(ctx, callee, args, result_type);
                if !results.is_empty() && op.get_operation().deref(ctx).get_num_results() > 0 {
                    let result = op.get_operation().deref(ctx).get_result(0);
                    process_parsed_ssa_defs(state_stream, &results, op.get_operation()).ok();
                }
                OpObj::new(op)
            })
            .into()
    }
}

impl_verify_succ!(CallOp);

fn successor_operands_parser<'a>() -> impl Parser<StateStream<'a>, Output = Vec<Value>> {
    token('(')
        .with(sep_by(
            spaced(ssa_opd_parser())
                .skip(spaced(token(':')))
                .skip(spaced(type_parser())),
            spaced(token(',')),
        ))
        .skip(token(')'))
}

fn print_successor_operands(
    ctx: &Context,
    state: &printable::State,
    f: &mut core::fmt::Formatter<'_>,
    operands: &[Value],
) -> core::fmt::Result {
    if operands.is_empty() {
        return Ok(());
    }
    write!(f, "(")?;
    for (i, opd) in operands.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        opd.fmt(ctx, state, f)?;
        write!(f, " : ")?;
        opd.get_type(ctx).fmt(ctx, state, f)?;
    }
    write!(f, ")")
}

pub fn register(ctx: &mut Context) {
}

#[cfg(test)]
mod tests {
    use combine::Parser;

    use crate::{
        context::Context,
        dialects::{builtin, mir},
        ir::{
            location,
            operation::{Operation, OperationParserConfig},
        },
        parsable::{Parsable, State, state_stream_from_iterator},
        printable::Printable,
    };

    #[test]
    fn parses_printed_constant_attribute_form() {
        let mut ctx = Context::new();
        mir::register(&mut ctx);

        let input = "builtin.module @m {
  ^entry():
    op0 = mir.constant builtin.integer <35: ui64>
}";
        let state_stream = state_stream_from_iterator(
            input.chars(),
            State::new(&mut ctx, location::Source::InMemory),
        );
        let config = OperationParserConfig {
            look_for_outlined_attrs: false,
        };

        let parsed = <Operation as Parsable>::parser(config)
            .parse(state_stream)
            .unwrap()
            .0;
        assert!(
            parsed
                .disp(&ctx)
                .to_string()
                .contains("mir.constant builtin.integer <35: ui64>")
        );
    }
}
