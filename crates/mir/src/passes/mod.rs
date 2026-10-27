//! The `mir`-level pass stage: passes over a module of dialect-mir
//! functions between the importer and the lowering, and the passes around
//! the lowering itself.
//!
//! - [mir_census]: pure analysis (opt-in measurement), before the lowering.
//! - [lower_dialect_mir]: cuda-oxide's `mir-lower` as a pass.
//! - [linkage]: function linkage across the lowering (module attribute in,
//!   `llvm.func` linkage out), right after it.
//! - [function_addresses]: the importer's deferred function addresses
//!   (`ll.func_addr`) become `llvm.addressof`.
//! - [erase_mir_types]: `llvm.ptr` for the `mir.ptr` the lowering leaves in
//!   the importer's llvm-dialect aggregate ops, last.
//!
//! Constraints, verified against the pinned `mir-lower`: its conversion
//! claims every `mir.*`/`nvvm.*` op and fails on one it has no arm for;
//! attributes on `mir` ops are not carried onto the lowered ops; the
//! `builtin.module` op and its attributes survive. So passes before the
//! lowering must not introduce `mir.*` kinds it lacks, and anything that
//! has to cross the lowering travels on the module.

use pliron_ll::conversion::pass::Passes;

pub mod erase_mir_types;
pub mod function_addresses;
pub mod gate;
pub mod linkage;
pub mod lower_dialect_mir;
pub mod mir_census;

/// The passes before the lowering (the measurement stage).
pub fn add_mir_passes(passes: &mut Passes) {
    add_mir_passes_labelled(passes, None);
}

/// [add_mir_passes] with `label` on the census records.
pub fn add_mir_passes_labelled(passes: &mut Passes, label: Option<&str>) {
    passes.add_pass(mir_census::MirCensusPass::new(label));
}
