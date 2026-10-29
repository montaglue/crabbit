//! Where crabbit writes its traces; the trace format is `pliron-trace`.

use std::path::PathBuf;

pub use pliron_trace::*;

/// Where traces go when no project root is known (see [`trace_dir`]).
pub const DEFAULT_TRACE_DIR: &str = "/tmp/crabbit-events";
/// The trace directory of a project, under its root.
pub const PROJECT_TRACE_DIR: &str = ".crabbit-traces";

pub fn project_trace_path(project: &str, version: &str) -> PathBuf {
    trace_dir()
        .join(project)
        .join(format!("{version}.{TRACE_EXTENSION}"))
}

/// The directory traces are written to: `CRABBIT_TRACE_DIR`, else
/// `.crabbit-traces/` at the root of the project being compiled (which is
/// where pliron-inspect, opening that project, looks for them), else
/// [`DEFAULT_TRACE_DIR`].
pub fn trace_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("CRABBIT_TRACE_DIR").filter(|d| !d.is_empty()) {
        return PathBuf::from(dir);
    }
    match project_root() {
        Some(root) => root.join(PROJECT_TRACE_DIR),
        None => PathBuf::from(DEFAULT_TRACE_DIR),
    }
}

/// The root of the cargo project being compiled: from the
/// `CARGO_MANIFEST_DIR` cargo hands to rustc, the nearest ancestor whose
/// `Cargo.toml` declares a `[workspace]`, else the package directory.
pub fn project_root() -> Option<PathBuf> {
    let manifest_dir = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR")?);
    for dir in manifest_dir.ancestors() {
        let manifest = dir.join("Cargo.toml");
        if manifest.is_file()
            && std::fs::read_to_string(&manifest).is_ok_and(|t| t.contains("[workspace"))
        {
            return Some(dir.to_path_buf());
        }
    }
    Some(manifest_dir)
}
