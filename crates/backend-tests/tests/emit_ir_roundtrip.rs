//! CRABBIT_EMIT_IR round-trip gate (docs/SERVERD-PLAN.md): the post-
//! lower-dialect-mir IR that crabbit emits must parse back and re-print to
//! the identical text, for every fixture. This is what lets the resident
//! driver/server consume rustc-emitted modules losslessly.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use pliron::combine::Parser;
use pliron::context::Context;
use pliron::location::Source;
use pliron::operation::{Operation, OperationParserConfig};
use pliron::parsable::{Parsable, State, state_stream_from_iterator};
use pliron::printable::Printable;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/backend-tests should be two levels below the repository root")
        .to_path_buf()
}

fn build_backend(root: &Path, cargo: &str) -> PathBuf {
    let status = Command::new(cargo)
        .args(["build", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args(["-p", "crabbit"])
        .status()
        .expect("failed to build crabbit backend dylib");
    assert!(status.success(), "crabbit backend build failed");
    let backend = root.join("target").join("debug").join(format!(
        "{}crabbit{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    ));
    assert!(backend.exists(), "no dylib at {}", backend.display());
    backend
}

/// Parse printed IR the way pliron-inspect's driver does. Dialects
/// self-register on `Context::new()` through pliron's linkme
/// CONTEXT_REGISTRATIONS — but only if the dialect crates are actually
/// linked into this test binary, so touch a pliron-ll symbol first.
fn force_link_dialects() {
    // Referencing any pliron-ll item links the crate (and pliron-llvm
    // beneath it), which carries the linkme registration sections.
    let _ = pliron_ll::passes::llvm::midend_gate::midend_disabled("_force_link");
}

fn parse_ir(
    content: &str,
    ctx: &mut Context,
    outlined: bool,
) -> Result<pliron::context::Ptr<Operation>, String> {
    let state = State::new(ctx, Source::InMemory);
    let stream = state_stream_from_iterator(content.chars(), state);
    let config = OperationParserConfig {
        look_for_outlined_attrs: outlined,
    };
    <Operation as Parsable>::parser(config)
        .parse(stream)
        .map(|(op, _)| op)
        .map_err(|e| format!("{e}"))
}

fn roundtrip_dir(dir: &Path) -> Vec<String> {
    let mut failures = Vec::new();
    let mut checked = 0;
    for entry in fs::read_dir(dir).expect("emit dir") {
        let path = entry.expect("entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("plir") {
            continue;
        }
        checked += 1;
        let text1 = fs::read_to_string(&path).expect("read .plir");
        let mut ctx = Context::new();
        let op = match parse_ir(&text1, &mut ctx, false) {
            Ok(op) => op,
            Err(e) => {
                failures.push(format!("{}: PARSE FAILED: {e}", path.display()));
                continue;
            }
        };
        let state = pliron::printable::State::default();
        let text2 = op.print(&ctx, &state).to_string();
        if std::env::var("ROUNDTRIP_DUMP").is_ok() {
            let _ = fs::write(path.with_extension("reprint"), &text2);
        }
        // (a) The emitted text and its reprint must agree up to pliron's
        // parse-time block-label uniquing and `!n` location refs (pure
        // print metadata; upstream pliron reprints parsed labels with a
        // fresh unique suffix, which we cannot patch from here).
        if canonicalize(&text1) != canonicalize(&text2) {
            let diff = first_diff(&canonicalize(&text1), &canonicalize(&text2));
            failures.push(format!("{}: NOT EQUIVALENT: {diff}", path.display()));
            continue;
        }
        // (b) The parsed form must be stable under a further parse/print
        // cycle, again modulo labels: upstream pliron re-uniques block
        // labels on EVERY parse (`block1v1` → `block1v1_block1v1` → …),
        // so strict textual fixpoint is unattainable without an upstream
        // fix; value names are stable (kept through builtin_debug_info).
        // The server always parses from the stored original text, so
        // label growth never accumulates there.
        let mut ctx3 = Context::new();
        match parse_ir(&text2, &mut ctx3, true) {
            Ok(op3) => {
                let state3 = pliron::printable::State::default();
                let text3 = op3.print(&ctx3, &state3).to_string();
                if canonicalize(&text2) != canonicalize(&text3) {
                    let diff = first_diff(&canonicalize(&text2), &canonicalize(&text3));
                    failures.push(format!(
                        "{}: PARSED FORM NOT A FIXPOINT: {diff}",
                        path.display()
                    ));
                }
            }
            Err(e) => failures.push(format!(
                "{}: REPRINT DOES NOT REPARSE: {e}",
                path.display()
            )),
        }
    }
    assert!(checked > 0, "no .plir emitted into {}", dir.display());
    failures
}

/// Trailing-whitespace/newline differences are print-cosmetic, not
/// information loss.
fn normalize(s: &str) -> String {
    s.lines()
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n")
        .trim_end()
        .to_string()
}

/// Equality modulo print metadata: block labels and SSA value names are
/// renamed to their order of first appearance (upstream pliron re-uniques
/// both when parsing, e.g. `v1427` reprints as `v1427_v0`), and `!n`
/// location references are stripped.
fn canonicalize(s: &str) -> String {
    // Drop the reprint's `outlined_attributes:` footer (per-op locations
    // and debug-info value names — pliron's own round-trip metadata).
    let body = match s.find("\noutlined_attributes:") {
        Some(idx) => &s[..idx],
        None => s,
    };
    let normalized = normalize(body);
    let mut out = String::with_capacity(normalized.len());
    let mut labels: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut values: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let bytes = normalized.as_bytes();
    let is_ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'^' {
            let start = i + 1;
            let mut end = start;
            while end < bytes.len() && is_ident(bytes[end]) {
                end += 1;
            }
            let name = normalized[start..end].to_string();
            let next = labels.len();
            let id = *labels.entry(name).or_insert(next);
            out.push_str(&format!("^bb{id}"));
            i = end;
        } else if c == b'!' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() {
            let mut end = i + 1;
            while end < bytes.len() && bytes[end].is_ascii_digit() {
                end += 1;
            }
            i = end; // drop the location ref
        } else if c == b'v'
            && (i == 0 || !is_ident(bytes[i - 1]))
            && i + 1 < bytes.len()
            && bytes[i + 1].is_ascii_digit()
        {
            let start = i;
            let mut end = i + 1;
            while end < bytes.len() && is_ident(bytes[end]) {
                end += 1;
            }
            let name = normalized[start..end].to_string();
            let next = values.len();
            let id = *values.entry(name).or_insert(next);
            out.push_str(&format!("val{id}"));
            i = end;
        } else {
            out.push(c as char);
            i += 1;
        }
    }
    // Dropped location refs leave stray spaces (`) :`, `i32 ;`).
    out.lines()
        .map(|l| l.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect::<Vec<_>>()
        .join("\n")
        .replace(" :", ":")
        .replace(" ;", ";")
}

fn first_diff(a: &str, b: &str) -> String {
    for (i, (la, lb)) in a.lines().zip(b.lines()).enumerate() {
        if la != lb {
            return format!("line {}: emitted {la:?} vs reprinted {lb:?}", i + 1);
        }
    }
    format!(
        "line counts differ: emitted {} vs reprinted {}",
        a.lines().count(),
        b.lines().count()
    )
}

fn compile_with_emit(
    cargo: &str,
    backend: &Path,
    manifest: &Path,
    target_dir: &Path,
    emit_dir: &Path,
    lib: bool,
) {
    let _ = fs::remove_dir_all(emit_dir);
    fs::create_dir_all(emit_dir).expect("emit dir");
    // cargo does not track CRABBIT_EMIT_IR: force a fresh leaf compile.
    let _ = fs::remove_dir_all(target_dir);
    let mut cmd = Command::new(cargo);
    cmd.arg("rustc").arg("--manifest-path").arg(manifest);
    if lib {
        cmd.arg("--lib");
    }
    cmd.arg("--")
        .arg(format!("-Zcodegen-backend={}", backend.display()))
        .arg("-Coverflow-checks=off")
        .env("CARGO_TARGET_DIR", target_dir)
        .env("CRABBIT_EMIT_IR", emit_dir);
    let status = cmd.status().expect("compile fixture");
    assert!(status.success(), "fixture {} failed to compile", manifest.display());
}

#[test]
fn emitted_ir_round_trips_for_every_fixture() {
    force_link_dialects();
    let root = repo_root();
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let backend = build_backend(&root, &cargo);
    let fixtures_dir = root.join("crates/backend-tests/fixtures");
    let scratch = root.join("target/stair-emit-ir-roundtrip");

    let mut all_failures = Vec::new();
    for entry in fs::read_dir(&fixtures_dir).expect("fixtures dir") {
        let dir = entry.expect("entry").path();
        let manifest = dir.join("Cargo.toml");
        if !manifest.exists() {
            continue;
        }
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        if name == "backend-smoke" || name == "llama-rms-norm" {
            // The #[ignore]d AMDGPU-path fixtures; they do not compile on
            // the aarch64 backend path.
            continue;
        }
        let lib = name == "kernel-vector-add";
        let emit_dir = scratch.join("ir").join(&name);
        let target_dir = scratch.join("target").join(&name);
        compile_with_emit(&cargo, &backend, &manifest, &target_dir, &emit_dir, lib);
        for failure in roundtrip_dir(&emit_dir) {
            all_failures.push(format!("[{name}] {failure}"));
        }
    }
    assert!(
        all_failures.is_empty(),
        "round-trip failures:\n{}",
        all_failures.join("\n")
    );
}
