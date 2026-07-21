//! Temporary home for STAIR's LLVM dialect, conversion framework and
//! LLVM-level passes, until they are replaced by / merged into pliron-llvm
//! and pliron's irbuild + pass manager.

pub mod conversion;
pub mod ll;
pub mod llvm;
pub mod op_interfaces;
pub mod passes;

pub mod dialects {
    pub use crate::llvm;
    pub use pliron::builtin;
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
