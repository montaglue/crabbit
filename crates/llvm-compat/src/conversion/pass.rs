//! Pass infrastructure: pliron's own [pliron::pass] module, re-exported under
//! the crate's compatibility facade. Passes implement [Pass] directly and
//! pipelines are [Passes]; there is no parallel pass manager here.
//!
//! `Pass::run` mutates the operation it is given and keeps its identity.
//! Transformations that used to swap the root operation (instruction
//! selection) now rewrite the module in place; producing a different IR
//! altogether (Mach-O objects) is a translation out of the pass pipeline,
//! not a pass.

pub use pliron::pass::*;

/// A [PassResult] reporting that the IR changed — the common case for every
/// pass here, none of which currently participate in analysis caching.
pub fn changed() -> PassResult {
    let mut result = PassResult::default();
    result.ir_changed = pliron::irbuild::IRStatus::Changed;
    result
}

/// A [PassResult] reporting that the IR is untouched (all analyses are
/// preserved) — what verification-only passes return.
pub fn unchanged() -> PassResult {
    PassResult::default()
}
