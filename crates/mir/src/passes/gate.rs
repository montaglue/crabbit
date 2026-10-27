//! Ablation gate for the `mir`-level passes (the stage between the importer
//! and `lower-dialect-mir`): `CRABBIT_MIR_DISABLE=census,...` names passes
//! to skip, read per pass run so A/B needs no rebuild. Same semantics as
//! the mid-end's `CRABBIT_MIDEND_DISABLE`
//! ([pliron_ll::passes::llvm::midend_gate]): unknown names are ignored,
//! each pass checks only its own name.

/// The environment variable naming the mir-level passes to skip.
pub const DISABLE_ENV: &str = "CRABBIT_MIR_DISABLE";

/// True when `name` appears in `CRABBIT_MIR_DISABLE` (comma-separated).
pub fn mir_disabled(name: &str) -> bool {
    std::env::var(DISABLE_ENV)
        .map(|v| v.split(',').any(|entry| entry.trim() == name))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    #[test]
    fn splits_on_commas_and_trims() {
        // Pure parsing check against the same predicate the gate applies,
        // without touching the process environment (tests run in parallel).
        let matches = |value: &str, name: &str| value.split(',').any(|entry| entry.trim() == name);
        assert!(matches("census", "census"));
        assert!(matches(" census , other", "census"));
        assert!(!matches("censu", "census"));
        assert!(!matches("", "census"));
    }
}
