//! Compile the fixture crates for x86_64-apple-darwin through the crabbit
//! codegen backend and execute the produced binaries. On Apple Silicon hosts
//! the binaries run under Rosetta 2; on Intel hosts they run natively.

#![cfg(target_os = "macos")]

use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, ExitStatus, Output, Stdio},
};

const TARGET: &str = "x86_64-apple-darwin";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/backend-tests should be two levels below the repository root")
        .to_path_buf()
}

fn fixture_manifest(root: &Path, fixture: &str) -> PathBuf {
    root.join("crates")
        .join("backend-tests")
        .join("fixtures")
        .join(fixture)
        .join("Cargo.toml")
}

fn backend_path(root: &Path) -> PathBuf {
    root.join("target").join("debug").join("libcrabbit.dylib")
}

#[derive(Clone, Copy)]
enum FixtureProfile {
    Debug,
    Release,
}

impl FixtureProfile {
    const ALL: [Self; 2] = [Self::Debug, Self::Release];

    fn name(self) -> &'static str {
        match self {
            Self::Debug => "debug",
            Self::Release => "release",
        }
    }

    fn cargo_release_arg(self) -> Option<&'static str> {
        match self {
            Self::Debug => None,
            Self::Release => Some("--release"),
        }
    }
}

fn build_backend(root: &Path, cargo: &str) -> PathBuf {
    let backend_status = Command::new(cargo)
        .args(["build", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args(["-p", "crabbit"])
        .status()
        .expect("failed to build crabbit backend dylib");
    assert!(backend_status.success(), "crabbit backend build failed");

    let backend = backend_path(root);
    assert!(
        backend.exists(),
        "expected backend dylib at {}",
        backend.display()
    );
    backend
}

fn fixture_target_dir(root: &Path, fixture: &str, profile: FixtureProfile) -> PathBuf {
    root.join("target").join(format!(
        "stair-backend-tests-x86-64-{fixture}-{}",
        profile.name()
    ))
}

fn compile_fixture(
    cargo: &str,
    fixture: &Path,
    bin: &str,
    backend: &Path,
    target_dir: &Path,
    profile: FixtureProfile,
) -> ExitStatus {
    if target_dir.exists() {
        fs::remove_dir_all(target_dir).expect("failed to clear fixture target directory");
    }
    let mut command = Command::new(cargo);
    command
        .arg("rustc")
        .arg("--manifest-path")
        .arg(fixture)
        .arg("--bin")
        .arg(bin)
        .arg("--target")
        .arg(TARGET);
    if let Some(arg) = profile.cargo_release_arg() {
        command.arg(arg);
    }
    command
        .arg("--")
        .arg(format!("-Zcodegen-backend={}", backend.display()))
        .arg("-Coverflow-checks=off")
        .env("CARGO_TARGET_DIR", target_dir);

    command
        .status()
        .unwrap_or_else(|_| panic!("failed to compile {bin} fixture in {} mode", profile.name()))
}

fn executable_path(target_dir: &Path, profile: FixtureProfile, bin: &str) -> PathBuf {
    target_dir
        .join(TARGET)
        .join(profile.name())
        .join(format!("{bin}{}", std::env::consts::EXE_SUFFIX))
}

/// Compile `fixture`/`bin` in the given profile and return the executable.
fn compile(fixture: &str, bin: &str, profile: FixtureProfile) -> PathBuf {
    let root = repo_root();
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let backend = build_backend(&root, &cargo);
    let manifest = fixture_manifest(&root, fixture);
    let target_dir = fixture_target_dir(&root, fixture, profile);

    let status = compile_fixture(&cargo, &manifest, bin, &backend, &target_dir, profile);
    assert!(
        status.success(),
        "{fixture} did not compile for {TARGET} with crabbit dylib in {} mode",
        profile.name()
    );

    let executable = executable_path(&target_dir, profile, bin);
    assert!(
        executable.exists(),
        "{fixture} did not produce executable {}",
        executable.display()
    );
    let arch = Command::new("file")
        .arg(&executable)
        .output()
        .expect("failed to run file");
    assert!(
        String::from_utf8_lossy(&arch.stdout).contains("x86_64"),
        "{fixture} produced a non-x86_64 binary"
    );
    executable
}

fn run_with_stdin(executable: &Path, args: &[&str], input: &str) -> Output {
    let mut child = Command::new(executable)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn fixture for stdin run");
    child
        .stdin
        .take()
        .expect("child stdin was not piped")
        .write_all(input.as_bytes())
        .expect("failed to write to child stdin");
    child
        .wait_with_output()
        .expect("failed to wait for fixture stdin run")
}

fn assert_success(output: &Output, context: &str) {
    assert!(
        output.status.success(),
        "{context} exited unsuccessfully: {}\nstdout: {}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn pure_rust_crate_compiles_and_runs_for_x86_64() {
    for profile in FixtureProfile::ALL {
        let executable = compile("pure-rust-aarch64", "pure-rust-aarch64", profile);
        let output = Command::new(&executable)
            .output()
            .expect("failed to run pure Rust executable");
        assert_success(&output, "pure Rust executable");
    }
}

#[test]
fn structs_crate_compiles_and_runs_for_x86_64() {
    for profile in FixtureProfile::ALL {
        let executable = compile("structs-aarch64", "structs-aarch64", profile);
        let output = Command::new(&executable)
            .output()
            .expect("failed to run structs executable");
        assert_success(&output, "structs executable");
    }
}

#[test]
fn hello_world_crate_compiles_and_prints_for_x86_64() {
    for profile in FixtureProfile::ALL {
        let executable = compile("hello-world-aarch64", "hello-world-aarch64", profile);
        let output = Command::new(&executable)
            .output()
            .expect("failed to run hello-world executable");
        assert_success(&output, "hello-world executable");
        assert_eq!(output.stdout, b"Hello, world!\n");
    }
}

#[test]
fn itoa_crate_compiles_and_self_checks_for_x86_64() {
    for profile in FixtureProfile::ALL {
        let executable = compile("itoa-aarch64", "itoa-aarch64", profile);
        let output = Command::new(&executable)
            .output()
            .expect("failed to run itoa executable");
        assert_success(&output, "itoa executable");
    }
}

#[test]
fn stdin_crate_reads_standard_input_for_x86_64() {
    // Deliberately ragged: leading/trailing spaces and a blank line make the
    // `trim`/`split_whitespace` searchers load-bearing.
    const INPUT: &str = "  10 20\n30 40  \n\n5\n";

    for profile in FixtureProfile::ALL {
        let executable = compile("stdin-aarch64", "stdin-aarch64", profile);

        let output = run_with_stdin(&executable, &["read_to_string"], INPUT);
        assert_success(&output, "stdin fixture read_to_string");
        assert_eq!(
            String::from_utf8_lossy(&output.stdout),
            "bytes=19\ntrimmed=[10 20\n30 40  \n\n5]\nsum=105\n"
        );

        let output = run_with_stdin(&executable, &["read_line"], INPUT);
        assert_success(&output, "stdin fixture read_line");
        assert_eq!(String::from_utf8_lossy(&output.stdout), "first=[10 20]\n");

        let output = run_with_stdin(&executable, &["lines"], INPUT);
        assert_success(&output, "stdin fixture lines");
        assert_eq!(
            String::from_utf8_lossy(&output.stdout),
            "lines=3\ntotal=105\n"
        );
    }
}

#[test]
fn sudoku_crate_prints_solution_for_x86_64() {
    for profile in FixtureProfile::ALL {
        let executable = compile("sudoku-aarch64", "sudoku-aarch64", profile);
        let output = Command::new(&executable)
            .output()
            .expect("failed to run sudoku executable");
        assert_success(&output, "sudoku executable");
        assert_eq!(
            output.stdout,
            b"534678912\n672195348\n198342567\n859761423\n426853791\n713924856\n961537284\n287419635\n345286179\n"
        );
    }
}

#[test]
fn sudoku_solver_solves_puzzles_for_x86_64() {
    const PUZZLE: &str =
        "530070000600195000098000060800060003400803001700020006060000280000419005000080079";
    const SOLUTION: &str = "5 3 4 6 7 8 9 1 2\n6 7 2 1 9 5 3 4 8\n1 9 8 3 4 2 5 6 7\n\
                            8 5 9 7 6 1 4 2 3\n4 2 6 8 5 3 7 9 1\n7 1 3 9 2 4 8 5 6\n\
                            9 6 1 5 3 7 2 8 4\n2 8 7 4 1 9 6 3 5\n3 4 5 2 8 6 1 7 9\n";

    for profile in FixtureProfile::ALL {
        let executable = compile("sudoku-solver", "stair-sudoku-solver", profile);

        let output = Command::new(&executable)
            .arg(PUZZLE)
            .output()
            .expect("failed to run sudoku solver executable");
        assert_success(&output, "sudoku solver");
        assert_eq!(String::from_utf8_lossy(&output.stdout), SOLUTION);

        let output = run_with_stdin(&executable, &[], &format!("  {PUZZLE}\n"));
        assert_success(&output, "sudoku solver reading stdin");
        assert_eq!(String::from_utf8_lossy(&output.stdout), SOLUTION);

        let mut duplicate = String::from("11");
        duplicate.push_str(&"0".repeat(79));
        let output = Command::new(&executable)
            .arg(&duplicate)
            .output()
            .expect("failed to run sudoku solver executable");
        assert_eq!(output.status.code(), Some(1));
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("duplicate 1 at row 1, column 2"),
            "unexpected stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
