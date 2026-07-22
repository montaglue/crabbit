//! Operations of the `ll` dialect: crabbit-specific extensions over the
//! upstream `pliron-llvm` dialect. The CFG stays in pliron's block-argument
//! form throughout; there is no phi operation.

use combine::Parser;
use pliron::derive::{def_op, derive_op_interface_impl};
use pliron::{
    builtin::{attributes::StringAttr, op_interfaces::OneResultInterface},
    context::Context,
    identifier::Identifier,
    impl_verify_succ,
    irfmt::parsers::{process_parsed_ssa_defs, spaced},
    location::Location,
    op::{Op, OpObj},
    operation::Operation,
    parsable::{Parsable, ParseResult, StateStream},
    printable::{self, Printable},
    r#type::TypeHandle,
};

use pliron_llvm::types::PointerType;

pliron::dict_key!(ATTR_KEY_LL_CSTR_VALUE, "ll_cstr_value");

/// A NUL-terminatable C string literal materialized as a pointer, used by the
/// MIR importer for string constants before they get a layout in the object's
/// literal pool.
#[def_op("ll.cstr")]
#[derive_op_interface_impl(OneResultInterface)]
pub struct CStrOp;

impl CStrOp {
    pub fn new(ctx: &mut Context, value: String) -> Self {
        let ptr_ty: TypeHandle = PointerType::get(ctx, 0).into();
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
            .set(ATTR_KEY_LL_CSTR_VALUE.clone(), StringAttr::new(value));
        cstr
    }

    pub fn get_value(&self, ctx: &Context) -> String {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<StringAttr>(&ATTR_KEY_LL_CSTR_VALUE)
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
