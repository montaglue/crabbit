//! rustc MIR importer targeting cuda-oxide's `dialect-mir` (the `mir`
//! dialect) instead of crabbit's own `cmir`.
//!
//! Forked from [importer](crate::importer); emission sites are being swapped
//! from `cmir` ops to `dialect_mir` ops one construct at a time. Selected at
//! runtime via `CRABBIT_PIPELINE=oxide`. Once every emission site is swapped
//! and the fixtures are green under the oxide pipeline, this file replaces
//! `importer.rs` and the `cmir` dialect is deleted.

use std::num::NonZero;

use rustc_abi::{Size, TagEncoding, VariantIdx, Variants};
use rustc_middle::{
    mir::{
        self as rustc_mir, BasicBlock as RustBasicBlock, BinOp, Body, ConstOperand, Local, Operand,
        Place, Rvalue, StatementKind, TerminatorKind,
    },
    ty::{EarlyBinder, Instance, InstanceKind, Ty, TyCtxt, TypeVisitableExt},
};
use crate::{
    context::{Context, Ptr},
    dialects::{
        aarch64, builtin,
        builtin::op_interfaces::OneRegionInterface,
        builtin::{
            attributes::{FPDoubleAttr, FPSingleAttr, IdentifierAttr, IntegerAttr},
            op_interfaces::{ATTR_KEY_SYM_NAME, SymbolOpInterface},
            types::{FP32Type, FP64Type, FunctionType, IntegerType, Signedness, UnitType},
        },
        llvm::{self, attributes::LinkageAttr},
        macho, x86_64,
    },
    identifier::Legaliser,
    ir::op::{Op, op_cast},
    ir::{
        basic_block::BasicBlock,
        operation::Operation,
        r#type::{TypeHandle, TypedHandle, Typed},
        value::Value,
    },
    linked_list::ContainsLinkedList,
    utils::apint::APInt,
};

// The emission shims stand in for the old cmir op paths; the body below is
// line-compatible with `importer.rs` modulo this alias.
use self::ox as mir_dialect;

// Submodules (rustc MIR -> dialect-mir import, split by concern):
mod entry;       // crate walk, kernel detection, function import, decls
mod globals;     // static/TLS/allocation globals, allocator shims
mod strings;     // string/byte-string constant materialization + parsers
mod stmt;        // statements, terminators, calls, SIMD, dyn dispatch
mod intrinsics;  // the intrinsic lowering match + bit/float-math helpers
mod abi;         // parameter/argument ABI classification
mod arith;       // i128 + overflow arithmetic, bool/cast helpers
mod rvalue;      // Rvalue lowering, casts, unsizing, fn-pointer reification
mod constants;   // constant/allocation import (scalar, pointer, enum, str)
mod place;       // place load/store, projections, address computation
mod layout;      // field offsets, size/align computation
mod helpers;     // slot/block lookup, cached types, ty predicates
mod convert;     // rustc Ty -> dialect type conversion, enum layout
mod ox;          // constructor shims over dialect-mir / llvm ops
#[cfg(test)]
mod tests;

pub use entry::{create_context, import_crate};
use entry::*;
use globals::*;
use strings::*;
use stmt::*;
use intrinsics::*;
use abi::*;
use arith::*;
use rvalue::*;
use constants::*;
use place::*;
use layout::*;
use helpers::*;
use convert::*;

/// A single unsupported MIR body or construct discovered during import.
#[derive(Debug, Clone)]
pub struct ImportError {
    pub item: String,
    pub reason: String,
}

/// Result of importing a rustc crate into crabbit MIR.
pub struct ImportedCrate {
    pub ctx: Context,
    pub module: Ptr<Operation>,
    pub kernel_module: Ptr<Operation>,
    pub kernel_count: usize,
    pub unsupported: Vec<ImportError>,
}

pub const KERNEL_EXPORT_PREFIX: &str = "__crabbit_kernel_";

struct FunctionImportState<'tcx> {
    module_body: Ptr<BasicBlock>,
    blocks: Vec<Ptr<BasicBlock>>,
    local_slots: Vec<Option<(Value, TypeHandle)>>,
    instance: Option<Instance<'tcx>>,
}
