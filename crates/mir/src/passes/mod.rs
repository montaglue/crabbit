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

use pliron_ll::conversion::pass::Pass;

pub mod erase_mir_types;
pub mod function_addresses;
pub mod gate;
pub mod linkage;
pub mod lower_dialect_mir;
pub mod mir_census;

/// The default order of the stage from imported `mir` to the LLVM dialect:
/// the measurement pass, the lowering, and the passes that complete it for
/// crabbit's module (linkage stamp, deferred function addresses, residual
/// `mir.ptr` types). crabbit-config builds pipelines from this order.
pub const DEFAULT_ORDER: &[&str] = &[
    mir_census::PASS_NAME,
    "lower-dialect-mir",
    linkage::PASS_NAME,
    function_addresses::PASS_NAME,
    erase_mir_types::PASS_NAME,
];

/// The pass of this stage called `name`; `label` is the census record's
/// `crate` field. `None` for a name that is not one of this stage's passes.
pub fn construct(name: &str, label: Option<&str>) -> Option<Box<dyn Pass>> {
    Some(match name {
        mir_census::PASS_NAME => Box::new(mir_census::MirCensusPass::new(label)),
        "lower-dialect-mir" => Box::new(lower_dialect_mir::LowerDialectMirPass),
        linkage::PASS_NAME => Box::new(linkage::ApplyModuleLinkagePass),
        function_addresses::PASS_NAME => {
            Box::new(function_addresses::MaterializeFunctionAddressesPass)
        }
        erase_mir_types::PASS_NAME => Box::new(erase_mir_types::EraseMirTypesPass),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every name in [DEFAULT_ORDER] constructs a pass reporting that name.
    #[test]
    fn default_order_constructs_passes_by_their_names() {
        for name in DEFAULT_ORDER {
            let pass = construct(name, Some("label"))
                .unwrap_or_else(|| panic!("`{name}` does not construct"));
            assert_eq!(pass.name(), *name);
        }
    }
}
