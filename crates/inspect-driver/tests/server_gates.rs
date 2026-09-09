//! Resident-server gates (docs/SERVERD-PLAN.md): pipeline equivalence
//! (server object bytes == rustc-path object bytes for the same config),
//! per-pass progress, and deterministic cancellation, driven through the
//! real `crabbit-analysisd` stdio protocol.

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::{Duration, Instant};

use pliron_inspect_protocol::server::base64_decode;
use serde_json::{Value, json};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/inspect-driver is two levels below the repo root")
        .to_path_buf()
}

struct ServerProc {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl ServerProc {
    fn spawn(workers: usize) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_crabbit-analysisd"))
            .args(["--workers", &workers.to_string()])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("spawn crabbit-analysisd");
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        ServerProc { child, stdin, stdout }
    }

    fn call(&mut self, cmd: Value) -> Value {
        writeln!(self.stdin, "{cmd}").expect("write command");
        let mut line = String::new();
        self.stdout.read_line(&mut line).expect("read response");
        serde_json::from_str(&line).expect("response is JSON")
    }
}

impl Drop for ServerProc {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Compile `fixture` with the crabbit dylib under `config` env, capturing
/// the emitted post-lowering IR and the rustc-path object bytes.
fn compile_fixture(
    root: &Path,
    backend: &Path,
    fixture: &str,
    config: &BTreeMap<String, String>,
    tag: &str,
) -> (String, Vec<u8>) {
    let scratch = root.join("target/stair-server-gates").join(fixture).join(tag);
    let _ = fs::remove_dir_all(&scratch);
    let emit_dir = scratch.join("ir");
    fs::create_dir_all(&emit_dir).unwrap();
    let manifest = root
        .join("crates/backend-tests/fixtures")
        .join(fixture)
        .join("Cargo.toml");
    let mut cmd = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()));
    cmd.arg("rustc")
        .arg("--manifest-path")
        .arg(&manifest)
        .arg("--")
        .arg(format!("-Zcodegen-backend={}", backend.display()))
        .arg("-Coverflow-checks=off")
        .arg("-Csave-temps")
        .env("CARGO_TARGET_DIR", scratch.join("target"))
        .env("CRABBIT_EMIT_IR", &emit_dir);
    for (k, v) in config {
        cmd.env(k, v);
    }
    let status = cmd.status().expect("compile fixture");
    assert!(status.success(), "{fixture} failed to compile under {tag}");

    let plir = fs::read_dir(&emit_dir)
        .unwrap()
        .filter_map(|e| Some(e.ok()?.path()))
        .find(|p| p.extension().is_some_and(|e| e == "plir"))
        .expect("emitted .plir");
    let text = fs::read_to_string(plir).unwrap();

    let mut objects: Vec<PathBuf> = walk(&scratch.join("target"))
        .into_iter()
        .filter(|p| {
            p.extension().is_some_and(|e| e == "o")
                && p.file_name()
                    .is_some_and(|n| n.to_string_lossy().contains("stair_rust"))
        })
        .collect();
    objects.sort();
    assert_eq!(
        objects.len(),
        1,
        "expected exactly one stair_rust object for {fixture}, got {objects:?}"
    );
    (text, fs::read(&objects[0]).unwrap())
}

fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            out.extend(walk(&path));
        } else {
            out.push(path);
        }
    }
    out
}

fn build_backend(root: &Path) -> PathBuf {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let status = Command::new(cargo)
        .args(["build", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args(["-p", "crabbit"])
        .status()
        .expect("build crabbit");
    assert!(status.success());
    root.join("target/debug/libcrabbit.so")
}

fn wait_done(server: &mut ServerProc, run_id: u64) -> Value {
    let deadline = Instant::now() + Duration::from_secs(600);
    loop {
        let status = server.call(json!({"cmd": "run_status", "runId": run_id}));
        match status["status"].as_str() {
            Some("done") | Some("failed") | Some("cancelled") => return status,
            _ => {
                assert!(Instant::now() < deadline, "run {run_id} timed out: {status}");
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }
}

fn run_to_artifact(
    server: &mut ServerProc,
    module_id: u64,
    config: &BTreeMap<String, String>,
) -> Vec<u8> {
    let start = server.call(json!({
        "cmd": "start_run", "moduleId": module_id,
        "target": "aarch64-linux", "config": config,
    }));
    let run_id = start["runId"].as_u64().unwrap_or_else(|| panic!("start_run: {start}"));
    assert!(start["passes"].as_array().is_some_and(|p| p.len() > 20));
    let status = wait_done(server, run_id);
    assert_eq!(status["status"], "done", "run failed: {status}");
    let artifact = server.call(json!({"cmd": "run_artifact", "runId": run_id}));
    base64_decode(artifact["artifactBase64"].as_str().unwrap_or_else(|| panic!("{artifact}")))
        .expect("artifact base64")
}

#[test]
fn server_objects_match_the_rustc_path_and_runs_report_and_cancel() {
    let root = repo_root();
    let backend = build_backend(&root);
    let baseline = BTreeMap::new();
    let eregalloc: BTreeMap<String, String> = [
        ("CRABBIT_REGALLOC", "eregalloc"),
        ("CRABBIT_REGALLOC_ORACLE", "c2"),
        ("CRABBIT_BLOCK_FREQ", "spectral"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect();

    let mut server = ServerProc::spawn(1);
    let health = server.call(json!({"cmd": "server_health"}));
    assert_eq!(health["ok"], true, "{health}");
    let targets = server.call(json!({"cmd": "list_targets"}));
    assert!(
        targets["targets"].as_array().is_some_and(|t| t.iter().any(|v| v == "aarch64-linux")),
        "{targets}"
    );

    // --- Equivalence: two fixtures × two configs -------------------------
    for fixture in ["hello-world-aarch64", "pure-rust-aarch64"] {
        for (tag, config) in [("baseline", &baseline), ("eregalloc-c2-spectral", &eregalloc)] {
            let (text, rustc_object) = compile_fixture(&root, &backend, fixture, config, tag);
            let loaded = server.call(json!({
                "cmd": "load_module", "name": format!("{fixture}-{tag}"), "text": text,
            }));
            let module_id = loaded["moduleId"].as_u64().unwrap_or_else(|| panic!("{loaded}"));
            let server_object = run_to_artifact(&mut server, module_id, config);
            assert_eq!(
                server_object, rustc_object,
                "object bytes differ for {fixture} under {tag}"
            );
        }
    }

    // --- Progress + deterministic cancel (1 worker: B queues behind A) --
    let (big_text, _) = compile_fixture(&root, &backend, "sudoku-solver", &baseline, "progress");
    let loaded = server.call(json!({"cmd": "load_module", "name": "big", "text": big_text}));
    let module_id = loaded["moduleId"].as_u64().unwrap();
    let run_a = server.call(json!({
        "cmd": "start_run", "moduleId": module_id, "target": "aarch64-linux", "config": {},
    }))["runId"]
        .as_u64()
        .unwrap();
    let run_b = server.call(json!({
        "cmd": "start_run", "moduleId": module_id, "target": "aarch64-linux", "config": {},
    }))["runId"]
        .as_u64()
        .unwrap();
    // B is queued behind A on the single worker; cancel it now.
    let cancelled = server.call(json!({"cmd": "cancel_run", "runId": run_b}));
    assert_eq!(cancelled["ok"], true, "{cancelled}");
    // Progress on A: observe a strictly-partial pass list at least once.
    let mut saw_partial = false;
    loop {
        let status = server.call(json!({"cmd": "run_status", "runId": run_a}));
        let done = status["passes"].as_array().map(|p| p.len()).unwrap_or(0);
        let total = status["totalPasses"].as_u64().unwrap_or(0) as usize;
        if status["status"] == "running" && done > 0 && done < total {
            saw_partial = true;
        }
        if matches!(status["status"].as_str(), Some("done" | "failed" | "cancelled")) {
            assert_eq!(status["status"], "done", "{status}");
            assert_eq!(done, total, "all passes reported: {status}");
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(saw_partial, "never observed mid-run progress on the big module");
    let status_b = wait_done(&mut server, run_b);
    assert_eq!(status_b["status"], "cancelled", "{status_b}");

    // --- Replay: IR after pass 0 differs from IR after the last pass ----
    let ir0 = server.call(json!({"cmd": "run_ir", "runId": run_a, "pass": 0}));
    let ir_last = server.call(json!({"cmd": "run_ir", "runId": run_a}));
    let (ir0, ir_last) = (
        ir0["ir"].as_str().unwrap_or_else(|| panic!("{ir0}")).to_string(),
        ir_last["ir"].as_str().unwrap_or_else(|| panic!("{ir_last}")).to_string(),
    );
    assert_ne!(ir0, ir_last);
    assert!(ir_last.contains("aarch64"), "final IR should be machine IR");
}
