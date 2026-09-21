//! Operations of the `ll` dialect: crabbit-specific extensions over the
//! upstream `pliron-llvm` dialect. The CFG stays in pliron's block-argument
//! form throughout; there is no phi operation.

use combine::{Parser, token};
use pliron::derive::{def_op, derive_op_interface_impl, verify_succ};
use pliron::{
    builtin::{
        attributes::StringAttr,
        op_interfaces::{NOpdsInterface, NResultsInterface, OneResultInterface},
    },
    context::Context,
    identifier::Identifier,
    irfmt::parsers::{process_parsed_ssa_defs, spaced, ssa_opd_parser, type_parser},
    location::{Located, Location},
    op::{Op, OpObj},
    operation::Operation,
    parsable::{Parsable, ParseResult, StateStream},
    printable::{self, Printable},
    r#type::{TypeHandle, Typed},
    value::Value,
};

use pliron_llvm::types::PointerType;

use super::attributes::{VBinOpKindAttr, VReduceKindAttr};

pliron::dict_key!(ATTR_KEY_LL_CSTR_VALUE, "ll_cstr_value");
pliron::dict_key!(ATTR_KEY_LL_VBINOP_KIND, "ll_vbinop_kind");
pliron::dict_key!(ATTR_KEY_LL_VREDUCE_KIND, "ll_vreduce_kind");

/// A NUL-terminatable C string literal materialized as a pointer, used by the
/// MIR importer for string constants before they get a layout in the object's
/// literal pool.
#[verify_succ]
#[def_op("ll.cstr")]
#[derive_op_interface_impl(NResultsInterface<1>, OneResultInterface)]
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

// ---------------------------------------------------------------------------
// Crabbit-internal vector ops (docs/MIDEND-PLAN.md, CPU SIMD). The importer
// never produces these; only the llvm-vectorize pass does, and only the
// machine backends consume them. Vector values use the upstream
// [pliron_llvm::types::VectorType] (fixed 128-bit shapes: f32x4, i32x4,
// f64x2, i64x2).
// ---------------------------------------------------------------------------

/// The fixed [VectorType](pliron_llvm::types::VectorType) of a value, if it
/// has one.
pub fn vector_type_of(ctx: &Context, ty: TypeHandle) -> Option<(TypeHandle, u32)> {
    let ty_ref = ty.deref(ctx);
    let vec = ty_ref.downcast_ref::<pliron_llvm::types::VectorType>()?;
    if vec.is_scalable() {
        return None;
    }
    Some((vec.elem_type(), vec.num_elements()))
}

/// A unit-stride 128-bit vector load: reads `lanes × elem` bytes at the
/// pointer operand.
///
/// Form: `<res> = ll.vload <ptr> : <vector type>`
#[verify_succ]
#[def_op("ll.vload")]
#[derive_op_interface_impl(NResultsInterface<1>, OneResultInterface, NOpdsInterface<1>)]
pub struct VLoadOp;

impl VLoadOp {
    pub fn new(ctx: &mut Context, addr: Value, vec_ty: TypeHandle) -> Self {
        VLoadOp {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![vec_ty],
                vec![addr],
                vec![],
                0,
            ),
        }
    }
}

impl Printable for VLoadOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        self.get_result(ctx).fmt(ctx, state, f)?;
        write!(f, " = {} ", self.get_opid().disp(ctx))?;
        self.op.deref(ctx).get_operand(0).fmt(ctx, state, f)?;
        write!(f, " : ")?;
        self.get_result(ctx).get_type(ctx).fmt(ctx, state, f)
    }
}

impl Parsable for VLoadOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let mut parser = spaced(ssa_opd_parser())
            .skip(spaced(token(':')))
            .and(spaced(type_parser()));
        let ((addr, ty), commit) = parser.parse_stream(state_stream).into_result()?;
        let ctx = &mut *state_stream.state.ctx;
        let op = VLoadOp::new(ctx, addr, ty);
        process_parsed_ssa_defs(state_stream, &results, op.get_operation())?;
        Ok((OpObj::new(op), commit))
    }
}

/// A unit-stride 128-bit vector store: writes the vector operand's bytes at
/// the pointer operand.
///
/// Form: `ll.vstore <value>, <ptr>`
#[verify_succ]
#[def_op("ll.vstore")]
#[derive_op_interface_impl(NResultsInterface<0>, NOpdsInterface<2>)]
pub struct VStoreOp;

impl VStoreOp {
    pub fn new(ctx: &mut Context, value: Value, addr: Value) -> Self {
        VStoreOp {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                vec![value, addr],
                vec![],
                0,
            ),
        }
    }

    pub fn value_operand(&self, ctx: &Context) -> Value {
        self.op.deref(ctx).get_operand(0)
    }

    pub fn address_operand(&self, ctx: &Context) -> Value {
        self.op.deref(ctx).get_operand(1)
    }
}

impl Printable for VStoreOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "{} ", self.get_opid().disp(ctx))?;
        self.op.deref(ctx).get_operand(0).fmt(ctx, state, f)?;
        write!(f, ", ")?;
        self.op.deref(ctx).get_operand(1).fmt(ctx, state, f)
    }
}

impl Parsable for VStoreOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        if !results.is_empty() {
            pliron::input_err!(state_stream.loc(), "ll.vstore produces no results")?
        }
        let mut parser = spaced(ssa_opd_parser())
            .skip(spaced(token(',')))
            .and(spaced(ssa_opd_parser()));
        let ((value, addr), commit) = parser.parse_stream(state_stream).into_result()?;
        let ctx = &mut *state_stream.state.ctx;
        let op = VStoreOp::new(ctx, value, addr);
        Ok((OpObj::new(op), commit))
    }
}

/// A lane-wise binary operation on two vectors of the same type; the result
/// has the operands' type. Integer kinds wrap per lane; FP kinds are the
/// IEEE scalar semantics applied per lane (no reassociation is implied).
///
/// Form: `<res> = ll.vbinop <kind> <lhs>, <rhs>`
#[verify_succ]
#[def_op("ll.vbinop")]
#[derive_op_interface_impl(NResultsInterface<1>, OneResultInterface, NOpdsInterface<2>)]
pub struct VBinOp;

impl VBinOp {
    pub fn new(ctx: &mut Context, kind: VBinOpKindAttr, lhs: Value, rhs: Value) -> Self {
        let result_ty = lhs.get_type(ctx);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_ty],
            vec![lhs, rhs],
            vec![],
            0,
        );
        op.deref_mut(ctx)
            .attributes
            .set(ATTR_KEY_LL_VBINOP_KIND.clone(), kind);
        VBinOp { op }
    }

    pub fn kind(&self, ctx: &Context) -> VBinOpKindAttr {
        *self
            .op
            .deref(ctx)
            .attributes
            .get::<VBinOpKindAttr>(&ATTR_KEY_LL_VBINOP_KIND)
            .expect("ll.vbinop without its kind attribute")
    }
}

impl Printable for VBinOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        self.get_result(ctx).fmt(ctx, state, f)?;
        write!(f, " = {} {} ", self.get_opid().disp(ctx), self.kind(ctx))?;
        self.op.deref(ctx).get_operand(0).fmt(ctx, state, f)?;
        write!(f, ", ")?;
        self.op.deref(ctx).get_operand(1).fmt(ctx, state, f)
    }
}

impl Parsable for VBinOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let mut parser = spaced(VBinOpKindAttr::parser(()))
            .and(spaced(ssa_opd_parser()))
            .skip(spaced(token(',')))
            .and(spaced(ssa_opd_parser()));
        let (((kind, lhs), rhs), commit) = parser.parse_stream(state_stream).into_result()?;
        let ctx = &mut *state_stream.state.ctx;
        let op = VBinOp::new(ctx, kind, lhs, rhs);
        process_parsed_ssa_defs(state_stream, &results, op.get_operation())?;
        Ok((OpObj::new(op), commit))
    }
}

/// Broadcast a scalar into every lane of a vector.
///
/// Form: `<res> = ll.vsplat <scalar> : <vector type>`
#[verify_succ]
#[def_op("ll.vsplat")]
#[derive_op_interface_impl(NResultsInterface<1>, OneResultInterface, NOpdsInterface<1>)]
pub struct VSplatOp;

impl VSplatOp {
    pub fn new(ctx: &mut Context, scalar: Value, vec_ty: TypeHandle) -> Self {
        VSplatOp {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![vec_ty],
                vec![scalar],
                vec![],
                0,
            ),
        }
    }
}

impl Printable for VSplatOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        self.get_result(ctx).fmt(ctx, state, f)?;
        write!(f, " = {} ", self.get_opid().disp(ctx))?;
        self.op.deref(ctx).get_operand(0).fmt(ctx, state, f)?;
        write!(f, " : ")?;
        self.get_result(ctx).get_type(ctx).fmt(ctx, state, f)
    }
}

impl Parsable for VSplatOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let mut parser = spaced(ssa_opd_parser())
            .skip(spaced(token(':')))
            .and(spaced(type_parser()));
        let ((scalar, ty), commit) = parser.parse_stream(state_stream).into_result()?;
        let ctx = &mut *state_stream.state.ctx;
        let op = VSplatOp::new(ctx, scalar, ty);
        process_parsed_ssa_defs(state_stream, &results, op.get_operation())?;
        Ok((OpObj::new(op), commit))
    }
}

/// Horizontal reduction of a vector to one scalar of its element type.
///
/// Form: `<res> = ll.vreduce <kind> <vec>`
#[verify_succ]
#[def_op("ll.vreduce")]
#[derive_op_interface_impl(NResultsInterface<1>, OneResultInterface, NOpdsInterface<1>)]
pub struct VReduceOp;

impl VReduceOp {
    /// Panics if `vec`'s type is not a fixed vector type.
    pub fn new(ctx: &mut Context, kind: VReduceKindAttr, vec: Value) -> Self {
        let vec_ty = vec.get_type(ctx);
        let (elem_ty, _) = vector_type_of(ctx, vec_ty)
            .expect("ll.vreduce operand must have a fixed vector type");
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![elem_ty],
            vec![vec],
            vec![],
            0,
        );
        op.deref_mut(ctx)
            .attributes
            .set(ATTR_KEY_LL_VREDUCE_KIND.clone(), kind);
        VReduceOp { op }
    }

    pub fn kind(&self, ctx: &Context) -> VReduceKindAttr {
        *self
            .op
            .deref(ctx)
            .attributes
            .get::<VReduceKindAttr>(&ATTR_KEY_LL_VREDUCE_KIND)
            .expect("ll.vreduce without its kind attribute")
    }
}

impl Printable for VReduceOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        self.get_result(ctx).fmt(ctx, state, f)?;
        write!(f, " = {} {} ", self.get_opid().disp(ctx), self.kind(ctx))?;
        self.op.deref(ctx).get_operand(0).fmt(ctx, state, f)
    }
}

impl Parsable for VReduceOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let loc = state_stream.loc();
        let mut parser = spaced(VReduceKindAttr::parser(())).and(spaced(ssa_opd_parser()));
        let ((kind, vec), commit) = parser.parse_stream(state_stream).into_result()?;
        let ctx = &mut *state_stream.state.ctx;
        if vector_type_of(ctx, vec.get_type(ctx)).is_none() {
            pliron::input_err!(loc, "ll.vreduce operand must have a fixed vector type")?
        }
        let op = VReduceOp::new(ctx, kind, vec);
        process_parsed_ssa_defs(state_stream, &results, op.get_operation())?;
        Ok((OpObj::new(op), commit))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pliron::{
        builtin::types::{FP32Type, Signedness, IntegerType},
        location,
        parsable::{self, state_stream_from_iterator},
    };
    use pliron_llvm::{
        attributes::LinkageAttr as LlvmLinkageAttr,
        ops::{FuncOp, ReturnOp},
        types::{FuncType, PointerType, VectorType, VectorTypeKind},
    };

    /// Print-metadata-insensitive form: SSA value names and block labels
    /// are renamed to first-appearance indices (pliron re-uniques both on
    /// every parse), mirroring the emit_ir_roundtrip gate's canonicalizer.
    fn canonicalize(text: &str) -> String {
        // Drop the reprint's `outlined_attributes:` footer and `!n`
        // location refs: per-op source locations are parse metadata.
        let text = match text.find("\noutlined_attributes:") {
            Some(idx) => &text[..idx],
            None => text,
        };
        let mut out = String::with_capacity(text.len());
        let mut names: std::collections::HashMap<String, usize> = Default::default();
        let bytes = text.as_bytes();
        let is_ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
        let mut i = 0;
        while i < bytes.len() {
            let b = bytes[i];
            if b == b'!' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() {
                let mut end = i + 1;
                while end < bytes.len() && bytes[end].is_ascii_digit() {
                    end += 1;
                }
                // Drop the ref and one preceding space if present.
                if out.ends_with(' ') {
                    out.pop();
                }
                i = end;
                continue;
            }
            let is_value = (b == b'v')
                && (i == 0 || !is_ident(bytes[i - 1]))
                && i + 1 < bytes.len()
                && bytes[i + 1].is_ascii_digit();
            if b == b'^' || is_value {
                let start = i + usize::from(b == b'^');
                let mut end = start;
                while end < bytes.len() && is_ident(bytes[end]) {
                    end += 1;
                }
                let name = text[start..end].to_string();
                let next = names.len();
                let id = *names.entry(name).or_insert(next);
                out.push_str(if b == b'^' { "^bb" } else { "val" });
                out.push_str(&id.to_string());
                i = end;
            } else {
                out.push(b as char);
                i += 1;
            }
        }
        out.trim_end().to_string()
    }

    fn parse_op_text(ctx: &mut Context, text: &str) -> pliron::context::Ptr<Operation> {
        let state_stream = state_stream_from_iterator(
            text.chars(),
            parsable::State::new(ctx, location::Source::InMemory),
        );
        <Operation as Parsable>::parser(pliron::operation::OperationParserConfig {
            look_for_outlined_attrs: false,
        })
        .parse(state_stream)
        .unwrap_or_else(|err| panic!("failed to parse:\n{text}\nerror: {err}"))
        .0
    }

    /// Every vector op prints into a form that parses back and re-prints
    /// identically (modulo pliron's parse-time value/label re-uniquing):
    /// the textual IR is a faithful serialization.
    #[test]
    fn vector_ops_round_trip_through_text() {
        let mut ctx = Context::new();
        let f32_ty: TypeHandle = FP32Type::get(&mut ctx).into();
        let i32_ty: TypeHandle = IntegerType::get(&mut ctx, 32, Signedness::Signless).into();
        let ptr_ty: TypeHandle = PointerType::get(&mut ctx, 0).into();
        let f32x4: TypeHandle =
            VectorType::get(&mut ctx, f32_ty, 4, VectorTypeKind::Fixed).into();
        let i32x4: TypeHandle =
            VectorType::get(&mut ctx, i32_ty, 4, VectorTypeKind::Fixed).into();
        let fn_ty = FuncType::get(&mut ctx, f32_ty, vec![ptr_ty, f32_ty, i32_ty], false);
        let func = FuncOp::new(&mut ctx, "vec".try_into().unwrap(), fn_ty);
        func.set_attr_llvm_function_linkage(&ctx, LlvmLinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(&mut ctx);
        let entry = func.get_entry_block(&ctx).unwrap();
        let (ptr, scalar, iscalar) = {
            let entry_ref = entry.deref(&ctx);
            (
                entry_ref.get_argument(0),
                entry_ref.get_argument(1),
                entry_ref.get_argument(2),
            )
        };

        let load = VLoadOp::new(&mut ctx, ptr, f32x4);
        load.get_operation().insert_at_back(entry, &ctx);
        let load_v = load.get_result(&ctx);
        let splat = VSplatOp::new(&mut ctx, scalar, f32x4);
        splat.get_operation().insert_at_back(entry, &ctx);
        let splat_v = splat.get_result(&ctx);
        let mul = VBinOp::new(&mut ctx, VBinOpKindAttr::FMul, load_v, splat_v);
        mul.get_operation().insert_at_back(entry, &ctx);
        let mul_v = mul.get_result(&ctx);
        let store = VStoreOp::new(&mut ctx, mul_v, ptr);
        store.get_operation().insert_at_back(entry, &ctx);
        // Integer shapes too: splat + wrapping add + integer reduce.
        let isplat = VSplatOp::new(&mut ctx, iscalar, i32x4);
        isplat.get_operation().insert_at_back(entry, &ctx);
        let isplat_v = isplat.get_result(&ctx);
        let iadd = VBinOp::new(&mut ctx, VBinOpKindAttr::Add, isplat_v, isplat_v);
        iadd.get_operation().insert_at_back(entry, &ctx);
        let iadd_v = iadd.get_result(&ctx);
        let ired = VReduceOp::new(&mut ctx, VReduceKindAttr::Add, iadd_v);
        ired.get_operation().insert_at_back(entry, &ctx);
        let red = VReduceOp::new(&mut ctx, VReduceKindAttr::FAdd, mul_v);
        red.get_operation().insert_at_back(entry, &ctx);
        let red_v = red.get_result(&ctx);
        ReturnOp::new(&mut ctx, Some(red_v))
            .get_operation()
            .insert_at_back(entry, &ctx);

        let state = printable::State::default();
        let text1 = func.get_operation().print(&ctx, &state).to_string();
        for needle in [
            "ll.vload",
            "ll.vstore",
            "ll.vbinop fmul",
            "ll.vbinop add",
            "ll.vsplat",
            "ll.vreduce fadd",
            "ll.vreduce add",
        ] {
            assert!(text1.contains(needle), "missing `{needle}` in:\n{text1}");
        }

        let parsed = parse_op_text(&mut ctx, &text1);
        let state = printable::State::default();
        let text2 = parsed.print(&ctx, &state).to_string();
        assert_eq!(
            canonicalize(&text1),
            canonicalize(&text2),
            "print → parse → print must be a fixpoint modulo renaming:\n{text1}\n----\n{text2}"
        );
    }
}
