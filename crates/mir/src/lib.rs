//! crabbit's adapter over cuda-oxide's `mir` dialect (dialect-mir): the
//! [LowerDialectMirPass](passes::lower_dialect_mir::LowerDialectMirPass)
//! wrapper plus re-exports. A plain rlib so tooling (the backward-PGO
//! tool) can link it without the rustc-plugin dylib machinery.

pub mod passes;

pub use dialect_mir;
pub use mir_lower;

pub mod dialects {
    pub use dialect_mir;
    pub use pliron::builtin;
    pub use pliron_llvm as llvm;
    pub use pliron_ll::{aarch64, macho, x86_64};
}
