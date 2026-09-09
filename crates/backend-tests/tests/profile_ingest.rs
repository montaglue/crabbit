//! Gate for the profile-feedback ingestion tool
//! (docs/PROFILE-FEEDBACK-PLAN.md): runs the python unit tests of
//! `scripts/perf-harness/profile_ingest.py` against the checked-in
//! synthetic perf-script fixture, so `cargo test --workspace` covers the
//! perf-side half of the loop too.

use std::path::Path;
use std::process::Command;

#[test]
fn profile_ingest_unit_tests_pass() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root");
    let test_script = repo_root.join("scripts/perf-harness/test_profile_ingest.py");
    let output = Command::new("python3")
        .arg(&test_script)
        .output()
        .expect("python3 must be runnable");
    assert!(
        output.status.success(),
        "profile_ingest tests failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}
