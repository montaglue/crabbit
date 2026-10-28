use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, ExitStatus, Stdio},
};

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

fn backend_path(root: &std::path::Path) -> PathBuf {
    // CRABBIT_TEST_BACKEND points the whole suite at an alternative
    // backend dylib — e.g. one built with extensions (docs/EXTENDING.md),
    // so the fixtures check that its passes keep every program correct.
    if let Ok(path) = std::env::var("CRABBIT_TEST_BACKEND") {
        if !path.is_empty() {
            return PathBuf::from(path);
        }
    }
    let file_name = if cfg!(target_os = "macos") {
        "libcrabbit.dylib"
    } else if cfg!(target_os = "windows") {
        "crabbit.dll"
    } else {
        "libcrabbit.so"
    };
    root.join("target").join("debug").join(file_name)
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

    fn keeps_intermediate_objects(self) -> bool {
        matches!(self, Self::Debug)
    }
}

fn object_files(root: &Path) -> Vec<PathBuf> {
    let mut objects = Vec::new();
    let Ok(entries) = fs::read_dir(root) else {
        return objects;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            objects.extend(object_files(&path));
        } else if path.extension().is_some_and(|extension| extension == "o")
            && path.metadata().is_ok_and(|metadata| metadata.len() > 0)
        {
            objects.push(path);
        }
    }
    objects
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
    root.join("target")
        .join(format!("crabbit-backend-tests-{fixture}-{}", profile.name()))
}

fn clear_target_dir(target_dir: &Path, context: &str) {
    if target_dir.exists() {
        fs::remove_dir_all(target_dir)
            .unwrap_or_else(|_| panic!("failed to clear {context} target directory"));
    }
}

fn compile_fixture(
    cargo: &str,
    fixture: &Path,
    bin: &str,
    backend: &Path,
    target_dir: &Path,
    profile: FixtureProfile,
    rustc_args: &[&str],
) -> ExitStatus {
    let mut command = Command::new(cargo);
    command
        .arg("rustc")
        .arg("--manifest-path")
        .arg(fixture)
        .arg("--bin")
        .arg(bin);
    if let Some(arg) = profile.cargo_release_arg() {
        command.arg(arg);
    }
    command
        .arg("--")
        .args(rustc_args)
        .arg(format!("-Zcodegen-backend={}", backend.display()))
        .arg("-Coverflow-checks=off")
        .env("CARGO_TARGET_DIR", target_dir);
    if !cfg!(target_os = "macos") {
        // macOS keeps CGU objects after linking (they carry the unpacked
        // debuginfo); other OSes delete them, so keep temps for the
        // intermediate-object assertions.
        command.arg("-Csave-temps");
    }

    command
        .status()
        .unwrap_or_else(|_| panic!("failed to compile {bin} fixture in {} mode", profile.name()))
}

fn executable_path(target_dir: &Path, profile: FixtureProfile, bin: &str) -> PathBuf {
    target_dir
        .join(profile.name())
        .join(format!("{bin}{}", std::env::consts::EXE_SUFFIX))
}

/// Runtime scalar transmutes (`f32::from_bits` etc.). Regression fixture
/// for the structural-cast-classification miscompile the real-kernel
/// corpus caught (Transmute imported as IntToFloat): the binary asserts
/// the reinterpretations itself, so a regression fails the run.
#[test]
fn transmute_aarch64_crate_compiles_and_runs_with_codegen_dylib() {
    let root = repo_root();
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let backend = build_backend(&root, &cargo);

    let fixture = fixture_manifest(&root, "transmute-aarch64");
    for profile in FixtureProfile::ALL {
        let target_dir = fixture_target_dir(&root, "transmute-aarch64", profile);
        clear_target_dir(&target_dir, "transmute");

        let fixture_status = compile_fixture(
            &cargo,
            &fixture,
            "transmute-aarch64",
            &backend,
            &target_dir,
            profile,
            &[],
        );
        assert!(
            fixture_status.success(),
            "transmute fixture did not compile with crabbit dylib in {} mode",
            profile.name()
        );

        let executable = executable_path(&target_dir, profile, "transmute-aarch64");
        let output = Command::new(&executable)
            .output()
            .expect("failed to run transmute executable");
        assert!(
            output.status.success(),
            "transmute executable failed in {} mode: {}\n{}",
            profile.name(),
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&output.stdout).trim(),
            "transmute ok",
            "unexpected transmute fixture output in {} mode",
            profile.name()
        );
    }
}

#[test]
fn pure_rust_aarch64_crate_compiles_and_runs_with_codegen_dylib() {
    let root = repo_root();
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let backend = build_backend(&root, &cargo);

    let fixture = fixture_manifest(&root, "pure-rust-aarch64");
    for profile in FixtureProfile::ALL {
        let target_dir = fixture_target_dir(&root, "pure-rust-aarch64", profile);
        clear_target_dir(&target_dir, "pure Rust");

        let fixture_status = compile_fixture(
            &cargo,
            &fixture,
            "pure-rust-aarch64",
            &backend,
            &target_dir,
            profile,
            &[],
        );

        assert!(
            fixture_status.success(),
            "pure Rust fixture did not compile with crabbit dylib in {} mode",
            profile.name()
        );

        if profile.keeps_intermediate_objects() {
            let objects = object_files(&target_dir);
            assert!(
                !objects.is_empty(),
                "pure Rust fixture did not produce a non-empty object file under {}",
                target_dir.display()
            );
        }

        let executable = executable_path(&target_dir, profile, "pure-rust-aarch64");
        assert!(
            executable.exists(),
            "pure Rust fixture did not produce executable {}",
            executable.display()
        );

        let run_status = Command::new(&executable)
            .status()
            .expect("failed to run pure Rust executable");
        assert!(
            run_status.success(),
            "pure Rust executable exited unsuccessfully in {} mode: {run_status}",
            profile.name()
        );
    }
}

#[test]
fn structs_aarch64_crate_compiles_and_runs_with_codegen_dylib() {
    let root = repo_root();
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let backend = build_backend(&root, &cargo);

    let fixture = fixture_manifest(&root, "structs-aarch64");
    for profile in FixtureProfile::ALL {
        let target_dir = fixture_target_dir(&root, "structs-aarch64", profile);
        clear_target_dir(&target_dir, "structs");

        let fixture_status = compile_fixture(
            &cargo,
            &fixture,
            "structs-aarch64",
            &backend,
            &target_dir,
            profile,
            &[],
        );

        assert!(
            fixture_status.success(),
            "structs fixture did not compile with crabbit dylib in {} mode",
            profile.name()
        );

        let executable = executable_path(&target_dir, profile, "structs-aarch64");
        let run_status = Command::new(&executable)
            .status()
            .expect("failed to run structs executable");
        assert!(
            run_status.success(),
            "structs executable exited unsuccessfully in {} mode: {run_status}",
            profile.name()
        );
    }
}

#[test]
fn hello_world_aarch64_crate_compiles_and_prints_with_codegen_dylib() {
    let root = repo_root();
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let backend = build_backend(&root, &cargo);

    let fixture = fixture_manifest(&root, "hello-world-aarch64");
    for profile in FixtureProfile::ALL {
        let target_dir = fixture_target_dir(&root, "hello-world-aarch64", profile);
        clear_target_dir(&target_dir, "hello-world");

        let fixture_status = compile_fixture(
            &cargo,
            &fixture,
            "hello-world-aarch64",
            &backend,
            &target_dir,
            profile,
            &[],
        );

        assert!(
            fixture_status.success(),
            "hello-world fixture did not compile with crabbit dylib in {} mode",
            profile.name()
        );

        let executable = executable_path(&target_dir, profile, "hello-world-aarch64");
        let output = Command::new(&executable)
            .output()
            .expect("failed to run hello-world executable");
        assert!(
            output.status.success(),
            "hello-world executable exited unsuccessfully in {} mode: {}",
            profile.name(),
            output.status
        );
        assert_eq!(output.stdout, b"Hello, world!\n");
    }
}

#[test]
fn fp_aarch64_crate_computes_floats_and_dynamic_shifts_with_codegen_dylib() {
    let root = repo_root();
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let backend = build_backend(&root, &cargo);

    let fixture = fixture_manifest(&root, "fp-aarch64");
    for profile in FixtureProfile::ALL {
        let target_dir = fixture_target_dir(&root, "fp-aarch64", profile);
        clear_target_dir(&target_dir, "fp");

        let fixture_status = compile_fixture(
            &cargo,
            &fixture,
            "fp-aarch64",
            &backend,
            &target_dir,
            profile,
            &[],
        );

        assert!(
            fixture_status.success(),
            "fp fixture did not compile with crabbit dylib in {} mode",
            profile.name()
        );

        let executable = executable_path(&target_dir, profile, "fp-aarch64");
        let output = Command::new(&executable)
            .output()
            .expect("failed to run fp executable");
        assert!(
            output.status.success(),
            "fp executable exited unsuccessfully in {} mode: {}\nstdout: {}\nstderr: {}",
            profile.name(),
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(output.stdout, b"fp ok\n");
    }
}

#[test]
fn int128_aarch64_crate_computes_128_bit_overflow_and_collects_with_codegen_dylib() {
    let root = repo_root();
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let backend = build_backend(&root, &cargo);

    let fixture = fixture_manifest(&root, "int128-aarch64");
    for profile in FixtureProfile::ALL {
        let target_dir = fixture_target_dir(&root, "int128-aarch64", profile);
        clear_target_dir(&target_dir, "int128");

        let fixture_status = compile_fixture(
            &cargo,
            &fixture,
            "int128-aarch64",
            &backend,
            &target_dir,
            profile,
            &[],
        );

        assert!(
            fixture_status.success(),
            "int128 fixture did not compile with crabbit dylib in {} mode",
            profile.name()
        );

        let executable = executable_path(&target_dir, profile, "int128-aarch64");
        let output = Command::new(&executable)
            .output()
            .expect("failed to run int128 executable");
        assert!(
            output.status.success(),
            "int128 executable exited unsuccessfully in {} mode: {}\nstdout: {}\nstderr: {}",
            profile.name(),
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(output.stdout, b"int128 ok\n");
    }
}

#[test]
fn hashmap_aarch64_crate_builds_and_reads_back_with_codegen_dylib() {
    let root = repo_root();
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let backend = build_backend(&root, &cargo);

    let fixture = fixture_manifest(&root, "hashmap-aarch64");
    for profile in FixtureProfile::ALL {
        let target_dir = fixture_target_dir(&root, "hashmap-aarch64", profile);
        clear_target_dir(&target_dir, "hashmap");

        let fixture_status = compile_fixture(
            &cargo,
            &fixture,
            "hashmap-aarch64",
            &backend,
            &target_dir,
            profile,
            &[],
        );

        assert!(
            fixture_status.success(),
            "hashmap fixture did not compile with crabbit dylib in {} mode",
            profile.name()
        );

        // HashMap::new goes through std's `#[thread_local]` RandomState KEYS,
        // so a correct run proves the ELF TLS path end to end.
        let executable = executable_path(&target_dir, profile, "hashmap-aarch64");
        let output = Command::new(&executable)
            .output()
            .expect("failed to run hashmap executable");
        assert!(
            output.status.success(),
            "hashmap executable exited unsuccessfully in {} mode: {}\nstdout: {}\nstderr: {}",
            profile.name(),
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(output.stdout, b"hashmap ok len=2 crab=bitte\n");
    }
}

/// Feed `input` to `executable` on stdin and return its captured stdout.
fn run_with_stdin(executable: &Path, args: &[&str], input: &str) -> std::process::Output {
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

#[test]
fn stdin_aarch64_crate_reads_standard_input_with_codegen_dylib() {
    let root = repo_root();
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let backend = build_backend(&root, &cargo);

    // Deliberately ragged: leading/trailing spaces and a blank line make the
    // `trim`/`split_whitespace` searchers load-bearing.
    const INPUT: &str = "  10 20\n30 40  \n\n5\n";

    let fixture = fixture_manifest(&root, "stdin-aarch64");
    for profile in FixtureProfile::ALL {
        let target_dir = fixture_target_dir(&root, "stdin-aarch64", profile);
        clear_target_dir(&target_dir, "stdin");

        let fixture_status = compile_fixture(
            &cargo,
            &fixture,
            "stdin-aarch64",
            &backend,
            &target_dir,
            profile,
            &[],
        );
        assert!(
            fixture_status.success(),
            "stdin fixture did not compile with crabbit dylib in {} mode",
            profile.name()
        );

        let executable = executable_path(&target_dir, profile, "stdin-aarch64");

        // `Stdin::read_to_string` plus `str::trim`.
        let output = run_with_stdin(&executable, &["read_to_string"], INPUT);
        assert!(
            output.status.success(),
            "stdin fixture read_to_string failed in {} mode: {}\nstderr: {}",
            profile.name(),
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&output.stdout),
            "bytes=19\ntrimmed=[10 20\n30 40  \n\n5]\nsum=105\n"
        );

        // `Stdin::read_line` reads only the first line.
        let output = run_with_stdin(&executable, &["read_line"], INPUT);
        assert!(
            output.status.success(),
            "stdin fixture read_line failed in {} mode: {}\nstderr: {}",
            profile.name(),
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(String::from_utf8_lossy(&output.stdout), "first=[10 20]\n");

        // `stdin().lock().lines()` imports the generic `Lines`/`BufRead` code,
        // which reaches the atomic intrinsics behind `Stdin`'s mutex.
        let output = run_with_stdin(&executable, &["lines"], INPUT);
        assert!(
            output.status.success(),
            "stdin fixture lines failed in {} mode: {}\nstderr: {}",
            profile.name(),
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&output.stdout),
            "lines=3\ntotal=105\n"
        );
    }
}

#[test]
fn sudoku_aarch64_crate_compiles_and_prints_solution_with_codegen_dylib() {
    let root = repo_root();
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let backend = build_backend(&root, &cargo);

    let fixture = fixture_manifest(&root, "sudoku-aarch64");
    for profile in FixtureProfile::ALL {
        let target_dir = fixture_target_dir(&root, "sudoku-aarch64", profile);
        clear_target_dir(&target_dir, "sudoku");

        let fixture_status = compile_fixture(
            &cargo,
            &fixture,
            "sudoku-aarch64",
            &backend,
            &target_dir,
            profile,
            &[],
        );

        assert!(
            fixture_status.success(),
            "sudoku fixture did not compile with crabbit dylib in {} mode",
            profile.name()
        );

        let executable = executable_path(&target_dir, profile, "sudoku-aarch64");
        let output = Command::new(&executable)
            .output()
            .expect("failed to run sudoku executable");
        assert!(
            output.status.success(),
            "sudoku executable exited unsuccessfully in {} mode: {}",
            profile.name(),
            output.status
        );
        assert_eq!(
            output.stdout,
            b"534678912\n672195348\n198342567\n859761423\n426853791\n713924856\n961537284\n287419635\n345286179\n"
        );
    }
}

#[test]
fn sudoku_solver_crate_solves_puzzles_from_arguments_with_codegen_dylib() {
    let root = repo_root();
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let backend = build_backend(&root, &cargo);

    const PUZZLE: &str =
        "530070000600195000098000060800060003400803001700020006060000280000419005000080079";
    const SOLUTION: &str = "5 3 4 6 7 8 9 1 2\n6 7 2 1 9 5 3 4 8\n1 9 8 3 4 2 5 6 7\n\
                            8 5 9 7 6 1 4 2 3\n4 2 6 8 5 3 7 9 1\n7 1 3 9 2 4 8 5 6\n\
                            9 6 1 5 3 7 2 8 4\n2 8 7 4 1 9 6 3 5\n3 4 5 2 8 6 1 7 9\n";

    let fixture = fixture_manifest(&root, "sudoku-solver");
    for profile in FixtureProfile::ALL {
        let target_dir = fixture_target_dir(&root, "sudoku-solver", profile);
        clear_target_dir(&target_dir, "sudoku solver");

        let fixture_status = compile_fixture(
            &cargo,
            &fixture,
            "crabbit-sudoku-solver",
            &backend,
            &target_dir,
            profile,
            &[],
        );
        assert!(
            fixture_status.success(),
            "sudoku solver fixture did not compile with crabbit dylib in {} mode",
            profile.name()
        );

        let executable = executable_path(&target_dir, profile, "crabbit-sudoku-solver");
        let output = Command::new(&executable)
            .arg(PUZZLE)
            .output()
            .expect("failed to run sudoku solver executable");
        assert!(
            output.status.success(),
            "sudoku solver exited unsuccessfully in {} mode: {}\nstderr: {}",
            profile.name(),
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(String::from_utf8_lossy(&output.stdout), SOLUTION);

        // Reading the puzzle from stdin exercises the enum-heavy std path:
        // `io::stdin().read_to_string` and `str::trim` (whose searcher relies
        // on real enum layouts and fat `*const str` metadata). The surrounding
        // whitespace makes `trim` load-bearing.
        let output = run_with_stdin(&executable, &[], &format!("  {PUZZLE}\n"));
        assert!(
            output.status.success(),
            "sudoku solver exited unsuccessfully in {} mode reading stdin: {}\nstderr: {}",
            profile.name(),
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(String::from_utf8_lossy(&output.stdout), SOLUTION);

        // Rejecting invalid input exercises `format!` with runtime arguments
        // and the error path through `Result<_, String>`.
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

/// Compile a mixed fixture (a host binary whose crate also holds cuda-oxide
/// `#[kernel]` functions) with the crabbit dylib: the host executable
/// builds, links and runs; the kernels land in the PTX sidecar
/// (`CRABBIT_PTX_OUT`) and not in the host object.
fn mixed_kernel_binary_fixture(
    fixture: &str,
    bin: &str,
    target_dir_name: &str,
    expected_ptx: &[&str],
    kernel_symbol: &str,
) -> (String, String) {
    let root = repo_root();
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let backend_status = Command::new(&cargo)
        .args(["build", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args(["-p", "crabbit"])
        .status()
        .expect("failed to build crabbit backend dylib");
    assert!(backend_status.success(), "crabbit backend build failed");
    let backend = backend_path(&root);
    assert!(backend.exists(), "expected backend dylib at {}", backend.display());

    let manifest = fixture_manifest(&root, fixture);
    let target_dir = root.join("target").join(target_dir_name);
    if target_dir.exists() {
        fs::remove_dir_all(&target_dir).expect("failed to clear the fixture target directory");
    }
    let ptx_out = target_dir.join("kernels.ptx");
    // Release, like every kernel build: cuda-device's checked accessors
    // (`DisjointSlice::get_mut`, `ThreadIndex`) leave `Option` aggregates
    // in debug code that the NVPTX translator does not lower; the
    // optimizer scalarizes them (docs/KERNEL-ABI.md).
    let mut command = Command::new(&cargo);
    command
        .arg("rustc")
        .arg("--manifest-path")
        .arg(&manifest)
        .arg("--bin")
        .arg(bin)
        .arg("--release")
        .arg("--")
        .arg(format!("-Zcodegen-backend={}", backend.display()))
        .arg("-Coverflow-checks=off")
        .env("CARGO_TARGET_DIR", &target_dir)
        .env("CRABBIT_PTX_OUT", &ptx_out);
    if !cfg!(target_os = "macos") {
        // Keep the intermediate objects for the symbol assertions.
        command.arg("-Csave-temps");
    }
    let fixture_status = command.status().expect("failed to compile the fixture");
    assert!(fixture_status.success(), "{fixture} did not compile with the crabbit dylib");

    let executable = target_dir
        .join("release")
        .join(format!("{bin}{}", std::env::consts::EXE_SUFFIX));
    assert!(executable.exists(), "{fixture} did not produce executable {}", executable.display());
    let run_output = Command::new(&executable)
        .output()
        .expect("failed to run the fixture executable");
    assert!(run_output.status.success(), "{fixture} executable exited unsuccessfully");
    let stdout = String::from_utf8_lossy(&run_output.stdout).to_string();
    assert!(
        stdout.contains("hello from crabbit"),
        "expected println output from the {fixture} executable, got:\n{stdout}"
    );

    let ptx = fs::read_to_string(&ptx_out).expect("CRABBIT_PTX_OUT was not written");
    for expected in expected_ptx {
        assert!(ptx.contains(expected), "expected `{expected}` in the PTX of {fixture}:\n{ptx}");
    }

    let objects = object_files(&target_dir);
    assert!(!objects.is_empty(), "{fixture} left no object file under {}", target_dir.display());
    if let Ok(output) = Command::new("nm").args(&objects).output()
        && output.status.success()
    {
        let symbols = String::from_utf8_lossy(&output.stdout);
        assert!(symbols.contains("main"), "expected main symbol in the host object, got:\n{symbols}");
        // The kernel is never *defined* on the host. (rustc's own
        // `symbols.o` lists every exported symbol of the crate as an
        // undefined reference, the renamed `#[kernel]` included.)
        let defined_kernel = symbols.lines().any(|line| {
            let mut fields = line.split_whitespace();
            let (Some(_addr), Some(kind), Some(name)) = (fields.next(), fields.next(), fields.next())
            else {
                return false;
            };
            matches!(kind, "T" | "t" | "W" | "w") && name.contains(kernel_symbol)
        });
        assert!(
            !defined_kernel,
            "kernel `{kernel_symbol}` should not be defined in the host object, got:\n{symbols}"
        );
    }
    (stdout, ptx)
}

#[test]
fn backend_smoke_crate_compiles_with_codegen_dylib() {
    let (stdout, _ptx) = mixed_kernel_binary_fixture(
        "backend-smoke",
        "crabbit-backend-smoke",
        "crabbit-backend-tests-smoke",
        &[
            ".visible .entry add_kernel(",
            ".visible .entry slice_kernel(",
            "add.s64",
            "%tid.x",
        ],
        "add_kernel",
    );
    assert!(stdout.contains("hello from crabbit: 42"), "{stdout}");
}

#[test]
fn llama_rms_norm_crate_compiles_with_codegen_dylib() {
    mixed_kernel_binary_fixture(
        "llama-rms-norm",
        "crabbit-llama-rms-norm",
        "crabbit-backend-tests-llama-rms-norm",
        &[
            ".visible .entry llama_rms_norm_f32(",
            "rsqrt.approx.f32",
            "bar.sync 0;",
            ".shared .align 4 .b8 ",
            "%ctaid.x",
        ],
        "llama_rms_norm_f32",
    );
}
