//! Re-export shim: the engine-selection code moved to the rustc-free
//! `research-config` crate so the resident driver/server shares it (see
//! docs/SERVERD-PLAN.md); everything in crabbit keeps this path.
pub use research_config::*;
