//! Native machine-level dialects and backend pipelines on pliron:
//! aarch64/x86_64 darwin — instruction selection, register allocation,
//! encoding, and Mach-O emission — plus the `ll` extension dialect and the
//! LLVM-level mid-end passes over [pliron_llvm]'s dialect.

pub mod aarch64;
pub mod conversion;
pub mod ll;
pub mod macho;
pub mod passes;
pub mod x86_64;

pub mod dialects {
    pub use crate::{aarch64, ll, macho, x86_64};
    pub use pliron::builtin;
    pub use pliron_llvm as llvm;
}
// ---- compatibility re-exports over the pliron core (cleanup pending) ----
pub use pliron::{
    attribute, basic_block, builtin, common_traits, context, debug_info, dialect,
    graph, identifier, irbuild, irfmt, linked_list, location, op, operation, opts,
    parsable, printable, region, storage_uniquer, symbol_table, r#type,
    uniqued_any, utils, value,
};
pub mod result {
    pub use pliron::result::*;
    /// Old stair name for [Result].
    pub type STAIRResult<T> = pliron::result::Result<T>;
}
pub mod ir {
    pub use pliron::{
        attribute, basic_block, dialect, irfmt, location, op, operation, region, value,
    };
    pub use pliron::r#type;
}
pub use pliron::{
    arg_err, arg_err_noloc, arg_error, arg_error_noloc, create_err, create_error,
    dict_key, impl_verify_succ, indented_block, input_err, input_err_noloc,
    input_error, input_error_noloc, type_to_trait, verify_err, verify_err_noloc,
    verify_error, verify_error_noloc,
};
