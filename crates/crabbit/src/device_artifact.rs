//! A crate's device code embedded in its host binary, in the artifact
//! format cuda-oxide's host API loads kernels from (`oxide-artifacts`):
//! one bundle named after the cargo package, carrying the PTX and its
//! kernel entries, in the `.oxart` section of an extra object linked next
//! to the host object. The object also defines the package's artifact
//! anchor, the symbol `#[cuda_module]`'s generated `load()` references,
//! so `kernels::load(&ctx)` finds the bundle in the executable or shared
//! object it was linked into (docs/KERNEL-ABI.md).

use std::path::{Path, PathBuf};

use oxide_artifacts::{
    ArtifactBundleSpec, ArtifactEntryKind, ArtifactEntrySpec, ArtifactPayloadKind,
    ArtifactPayloadSpec, build_artifact_blob, build_host_object_for_target,
};

use crate::EmitError;

/// Write the artifact object embedding `ptx` (built for `sm_<sm>`, with
/// the `.entry` kernels `entries`) next to `object`, for the host
/// `host_target`, and return its path. The bundle is named, and anchored,
/// as cuda-oxide's own backend does it: after the cargo package
/// (`CARGO_PKG_NAME`, `CARGO_PKG_VERSION`), after `crate_name` when the
/// crate is not built by cargo (`#[cuda_module]` then references no
/// anchor and the loader searches the executable by bundle name).
pub(crate) fn write_artifact_object(
    object: &Path,
    crate_name: &str,
    host_target: &str,
    sm: u32,
    ptx: &str,
    entries: &[String],
) -> Result<PathBuf, EmitError> {
    let err = |reason: String| EmitError::Pipeline {
        stage: "device artifact",
        reason,
    };
    let package = std::env::var("CARGO_PKG_NAME").unwrap_or_else(|_| crate_name.to_string());
    let version = std::env::var("CARGO_PKG_VERSION").unwrap_or_default();
    let target = format!("sm_{sm}");
    let payload_name = format!("{package}.ptx");
    let mut spec = ArtifactBundleSpec::new(&package, &target).with_payload(
        ArtifactPayloadSpec::new(ArtifactPayloadKind::Ptx, &payload_name, ptx.as_bytes()),
    );
    for entry in entries {
        spec = spec.with_entry(ArtifactEntrySpec::new(entry, ArtifactEntryKind::Kernel));
    }
    let blob = build_artifact_blob(&spec).map_err(|error| err(error.to_string()))?;
    let anchor = reserved_oxide_symbols::artifact_anchor_symbol(&package, &version);
    let bytes = build_host_object_for_target(&blob, host_target, Some(&anchor))
        .map_err(|error| err(error.to_string()))?;
    let path = object.with_extension("oxart.o");
    std::fs::write(&path, bytes).map_err(crate::io_err("write device artifact object", &path))?;
    Ok(path)
}
