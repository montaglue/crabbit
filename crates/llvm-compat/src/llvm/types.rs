//! Types defined in the LLVM dialect.

use combine::{
    Parser, between, choice, optional,
    parser::char::{char, string},
    sep_by,
};
use pliron::derive::def_type;
use pliron::derive::{format_type, type_interface_impl};

use crate::{
    context::{Context, Ptr},
    dialects::builtin::type_interfaces::FunctionTypeInterface,
    identifier::Identifier,
    impl_verify_succ,
    ir::irfmt::parsers::{int_parser, spaced, type_parser},
    ir::r#type::{Type, TypeHandle, TypedHandle},
    parsable::{Parsable, ParseResult, StateStream},
    printable::{self, Printable},
    result::STAIRResult,
    verify_err_noloc,
};

// ============================================================================
// VoidType
// ============================================================================

/// LLVM void type, used as function return type for functions that return nothing.
#[def_type("llvm.void")]
#[format_type]
#[derive(Hash, PartialEq, Eq, Debug)]
pub struct VoidType;

impl VoidType {
    /// Register type in dialect and instantiate the singleton instance.
    pub fn register_and_instantiate(ctx: &mut Context) {
        Type::register_instance(Self {}, ctx);
    }

    /// Get the singleton void type.
    pub fn get(ctx: &Context) -> TypedHandle<Self> {
        Type::get_instance(Self {}, ctx).expect("VoidType singleton not instantiated")
    }
}

impl_verify_succ!(VoidType);

// ============================================================================
// PointerType
// ============================================================================

/// LLVM opaque pointer type.
/// In modern LLVM (opaque pointers), all pointers are type-erased.
#[def_type("llvm.ptr")]
#[format_type]
#[derive(Hash, PartialEq, Eq, Debug)]
pub struct PointerType;

impl PointerType {
    /// Register type in dialect and instantiate the singleton instance.
    pub fn register_and_instantiate(ctx: &mut Context) {
        Type::register_instance(Self {}, ctx);
    }

    /// Get the singleton pointer type.
    pub fn get(ctx: &Context) -> TypedHandle<Self> {
        Type::get_instance(Self {}, ctx).expect("PointerType singleton not instantiated")
    }
}

impl_verify_succ!(PointerType);

// ============================================================================
// ArrayType
// ============================================================================

/// LLVM array type: a fixed-size array of elements of the same type.
#[def_type("llvm.array")]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct ArrayType {
    elem: TypeHandle,
    size: u64,
}

impl ArrayType {
    /// Get or create a new array type.
    pub fn get(ctx: &mut Context, elem: TypeHandle, size: u64) -> TypedHandle<Self> {
        Type::register_instance(ArrayType { elem, size }, ctx)
    }

    /// Get, if it already exists, an array type.
    pub fn get_existing(ctx: &Context, elem: TypeHandle, size: u64) -> Option<TypedHandle<Self>> {
        Type::get_instance(ArrayType { elem, size }, ctx)
    }

    /// Get the element type.
    pub fn elem_type(&self) -> TypeHandle {
        self.elem
    }

    /// Get the array size.
    pub fn size(&self) -> u64 {
        self.size
    }
}

impl Printable for ArrayType {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "<{} x ", self.size)?;
        self.elem.fmt(ctx, state, f)?;
        write!(f, ">")
    }
}

impl Parsable for ArrayType {
    type Arg = ();
    type Parsed = TypedHandle<Self>;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        // Parse: <size x elem_type>
        let mut parser = between(
            spaced(char('<')),
            spaced(char('>')),
            spaced(int_parser::<u64>())
                .skip(spaced(char('x')))
                .and(spaced(type_parser())),
        );

        parser
            .parse_stream(state_stream)
            .map(|(size, elem)| ArrayType::get(state_stream.state.ctx, elem, size))
            .into()
    }
}

impl_verify_succ!(ArrayType);

// ============================================================================
// VectorType
// ============================================================================

/// Whether the vector is fixed-size or scalable (RISC-V style).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VectorTypeKind {
    Fixed,
    Scalable,
}

/// LLVM vector type: a vector of elements, either fixed-size or scalable.
#[def_type("llvm.vector")]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct VectorType {
    elem_ty: TypeHandle,
    num_elems: u32,
    kind: VectorTypeKind,
}

impl VectorType {
    /// Get or create a new vector type.
    pub fn get(
        ctx: &mut Context,
        elem_ty: TypeHandle,
        num_elems: u32,
        kind: VectorTypeKind,
    ) -> TypedHandle<Self> {
        Type::register_instance(
            VectorType {
                elem_ty,
                num_elems,
                kind,
            },
            ctx,
        )
    }

    /// Get the element type.
    pub fn elem_type(&self) -> TypeHandle {
        self.elem_ty
    }

    /// Get the number of elements.
    pub fn num_elements(&self) -> u32 {
        self.num_elems
    }

    /// Whether this is a scalable vector.
    pub fn is_scalable(&self) -> bool {
        matches!(self.kind, VectorTypeKind::Scalable)
    }

    /// Get the vector kind.
    pub fn kind(&self) -> VectorTypeKind {
        self.kind
    }
}

impl Printable for VectorType {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "<")?;
        if self.is_scalable() {
            write!(f, "vscale x ")?;
        }
        write!(f, "{} x ", self.num_elems)?;
        self.elem_ty.fmt(ctx, state, f)?;
        write!(f, ">")
    }
}

impl Parsable for VectorType {
    type Arg = ();
    type Parsed = TypedHandle<Self>;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        // Parse: <[vscale x] num_elems x elem_type>
        let kind_parser = optional(spaced(string("vscale")).skip(spaced(char('x')))).map(|v| {
            if v.is_some() {
                VectorTypeKind::Scalable
            } else {
                VectorTypeKind::Fixed
            }
        });

        let mut parser = between(
            spaced(char('<')),
            spaced(char('>')),
            kind_parser
                .and(spaced(int_parser::<u32>()))
                .skip(spaced(char('x')))
                .and(spaced(type_parser())),
        );

        parser
            .parse_stream(state_stream)
            .map(|((kind, num_elems), elem_ty)| {
                VectorType::get(state_stream.state.ctx, elem_ty, num_elems, kind)
            })
            .into()
    }
}

impl_verify_succ!(VectorType);

// ============================================================================
// FuncType
// ============================================================================

/// LLVM function type: result type, argument types, and optional variadic flag.
#[def_type("llvm.func")]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct FuncType {
    res: TypeHandle,
    args: Vec<TypeHandle>,
    is_var_arg: bool,
}

impl FuncType {
    /// Get or create a new function type.
    pub fn get(
        ctx: &mut Context,
        res: TypeHandle,
        args: Vec<TypeHandle>,
        is_var_arg: bool,
    ) -> TypedHandle<Self> {
        Type::register_instance(
            FuncType {
                res,
                args,
                is_var_arg,
            },
            ctx,
        )
    }

    /// Get, if it already exists, a function type.
    pub fn get_existing(
        ctx: &Context,
        res: TypeHandle,
        args: Vec<TypeHandle>,
        is_var_arg: bool,
    ) -> Option<TypedHandle<Self>> {
        Type::get_instance(
            FuncType {
                res,
                args,
                is_var_arg,
            },
            ctx,
        )
    }

    /// Get the result type.
    pub fn result_type(&self) -> TypeHandle {
        self.res
    }

    /// Get the argument types.
    pub fn arg_types_slice(&self) -> &[TypeHandle] {
        &self.args
    }

    /// Whether this function type is variadic.
    pub fn is_var_arg(&self) -> bool {
        self.is_var_arg
    }
}

#[type_interface_impl]
impl FunctionTypeInterface for FuncType {
    fn arg_types(&self) -> Vec<TypeHandle> {
        self.args.clone()
    }

    fn res_types(&self) -> Vec<TypeHandle> {
        vec![self.res]
    }
}

impl Printable for FuncType {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "<")?;
        self.res.fmt(ctx, state, f)?;
        write!(f, " (")?;
        for (i, arg) in self.args.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            arg.fmt(ctx, state, f)?;
        }
        if self.is_var_arg {
            if !self.args.is_empty() {
                write!(f, ", ")?;
            }
            write!(f, "...")?;
        }
        write!(f, ")>")
    }
}

impl Parsable for FuncType {
    type Arg = ();
    type Parsed = TypedHandle<Self>;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        // Parse: <res_type (arg_types[, ...])>
        let args_and_vararg = between(
            spaced(char('(')),
            spaced(char(')')),
            optional(sep_by::<Vec<_>, _, _, _>(
                spaced(choice((
                    string("...").map(|_| None),
                    type_parser().map(Some),
                ))),
                spaced(char(',')),
            )),
        );

        let mut parser = between(
            spaced(char('<')),
            spaced(char('>')),
            spaced(type_parser()).and(args_and_vararg),
        );

        parser
            .parse_stream(state_stream)
            .map(|(res, args_opt)| {
                let args_list = args_opt.unwrap_or_default();
                let mut args = Vec::new();
                let mut is_var_arg = false;
                for item in args_list {
                    match item {
                        Some(ty) => args.push(ty),
                        None => is_var_arg = true,
                    }
                }
                FuncType::get(state_stream.state.ctx, res, args, is_var_arg)
            })
            .into()
    }
}

impl_verify_succ!(FuncType);

// ============================================================================
// StructType
// ============================================================================

/// LLVM struct type: named or unnamed, possibly opaque (no body), possibly recursive.
///
/// Named structs are uniqued only by name, enabling recursive types via the
/// opaque-then-fill pattern. Anonymous structs are uniqued by their fields.
#[def_type("llvm.struct")]
#[derive(Debug)]
pub struct StructType {
    name: Option<Identifier>,
    fields: Option<Vec<TypeHandle>>,
}

// Named structs are uniqued only by name. Anonymous structs by fields.
impl Hash for StructType {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        if self.name.is_none() {
            self.fields.hash(state);
        }
    }
}

impl PartialEq for StructType {
    fn eq(&self, other: &Self) -> bool {
        match (&self.name, &other.name) {
            (Some(a), Some(b)) => a == b,
            (None, None) => self.fields == other.fields,
            _ => false,
        }
    }
}

impl Eq for StructType {}

impl StructType {
    /// Get or create a named StructType.
    /// If `fields` is `None`, it indicates an opaque struct.
    /// A body can be added to opaque structs by calling this again later.
    pub fn get_named(
        ctx: &mut Context,
        name: Identifier,
        fields: Option<Vec<TypeHandle>>,
    ) -> STAIRResult<TypedHandle<Self>> {
        let self_ptr = Type::register_instance(
            StructType {
                name: Some(name.clone()),
                fields: None,
            },
            ctx,
        );
        // If fields are provided, try to set them.
        if let Some(fields) = fields {
            let mut self_ref = self_ptr.to_handle().deref_mut(ctx);
            let self_ref = self_ref.downcast_mut::<StructType>().unwrap();
            if let Some(existing_fields) = &self_ref.fields {
                if existing_fields != &fields {
                    return verify_err_noloc!(StructTypeErr::ExistingFieldsMismatch(
                        name.to_string()
                    ));
                }
            } else {
                self_ref.fields = Some(fields);
            }
        }
        Ok(self_ptr)
    }

    /// Get or create a new unnamed (anonymous) struct.
    /// These are finalized upon creation, and uniqued based on the fields.
    pub fn get_unnamed(ctx: &mut Context, fields: Vec<TypeHandle>) -> TypedHandle<Self> {
        Type::register_instance(
            StructType {
                name: None,
                fields: Some(fields),
            },
            ctx,
        )
    }

    /// If a named struct already exists, get a pointer to it.
    pub fn get_existing_named(ctx: &Context, name: &Identifier) -> Option<TypedHandle<Self>> {
        Type::get_instance(
            StructType {
                name: Some(name.clone()),
                fields: None,
            },
            ctx,
        )
    }

    /// Is this a named struct?
    pub fn is_named(&self) -> bool {
        self.name.is_some()
    }

    /// Is this an opaque struct (named but no body)?
    pub fn is_opaque(&self) -> bool {
        self.name.is_some() && self.fields.is_none()
    }

    /// Get the struct name (if named).
    pub fn name(&self) -> Option<&Identifier> {
        self.name.as_ref()
    }

    /// Get the fields (if the body is set).
    pub fn fields(&self) -> Option<&[TypeHandle]> {
        self.fields.as_deref()
    }

    /// Get the number of fields.
    pub fn num_fields(&self) -> Option<usize> {
        self.fields.as_ref().map(|f| f.len())
    }

    /// Get a specific field type by index.
    pub fn field_type(&self, index: usize) -> Option<TypeHandle> {
        self.fields.as_ref().and_then(|f| f.get(index).copied())
    }
}

use std::hash::Hash;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum StructTypeErr {
    #[error("Named struct '{0}' already has a different body")]
    ExistingFieldsMismatch(String),
    #[error("Anonymous struct cannot be opaque")]
    AnonymousOpaque,
}

impl Printable for StructType {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "<")?;
        if let Some(name) = &self.name {
            write!(f, "\"{}\"", name)?;
            if let Some(fields) = &self.fields {
                write!(f, " {{")?;
                for (i, field) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    field.fmt(ctx, state, f)?;
                }
                write!(f, "}}")?;
            }
        } else if let Some(fields) = &self.fields {
            write!(f, "{{")?;
            for (i, field) in fields.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                field.fmt(ctx, state, f)?;
            }
            write!(f, "}}")?;
        }
        write!(f, ">")
    }
}

impl Parsable for StructType {
    type Arg = ();
    type Parsed = TypedHandle<Self>;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        use crate::ir::irfmt::parsers::quoted_string_parser;

        let make_fields_parser = || {
            between(
                spaced(char('{')),
                spaced(char('}')),
                sep_by::<Vec<_>, _, _, _>(spaced(type_parser()), spaced(char(','))),
            )
        };

        // Try named struct: <"name" {fields}> or <"name">
        // Or anonymous struct: <{fields}>
        let mut parser = between(
            spaced(char('<')),
            spaced(char('>')),
            choice((
                // Named: "name" optionally followed by {fields}
                spaced(quoted_string_parser())
                    .and(optional(spaced(make_fields_parser())))
                    .map(|(name, fields)| (Some(name), fields)),
                // Anonymous: {fields}
                make_fields_parser().map(|fields| (None::<String>, Some(fields))),
            )),
        );

        parser
            .parse_stream(state_stream)
            .map(|(name, fields)| {
                let ctx = &mut *state_stream.state.ctx;
                match name {
                    Some(name_str) => {
                        let ident: Identifier = name_str.try_into().unwrap();
                        StructType::get_named(ctx, ident, fields).unwrap()
                    }
                    None => StructType::get_unnamed(ctx, fields.unwrap_or_default()),
                }
            })
            .into()
    }
}

impl crate::common_traits::Verify for StructType {
    fn verify(&self, _ctx: &Context) -> STAIRResult<()> {
        if self.name.is_none() && self.fields.is_none() {
            return verify_err_noloc!(StructTypeErr::AnonymousOpaque);
        }
        Ok(())
    }
}

// ============================================================================
// Registration
// ============================================================================

pub fn register(ctx: &mut Context) {
    VoidType::register_and_instantiate(ctx);
    PointerType::register_and_instantiate(ctx);
}
