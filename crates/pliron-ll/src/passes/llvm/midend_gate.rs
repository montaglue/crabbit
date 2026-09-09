//! Ablation gate for the research mid-end passes (docs/MIDEND-PLAN.md):
//! `CRABBIT_MIDEND_DISABLE=gvn,licm,divmagic` names passes to skip, read
//! per pass run (like [crate::codegen_opts]) so A/B needs no rebuild.
//! Unknown names are ignored (the variable is read by each pass for its
//! own name only).

/// True when `name` appears in `CRABBIT_MIDEND_DISABLE` (comma-separated).
pub fn midend_disabled(name: &str) -> bool {
    std::env::var("CRABBIT_MIDEND_DISABLE")
        .map(|v| v.split(',').any(|entry| entry.trim() == name))
        .unwrap_or(false)
}
