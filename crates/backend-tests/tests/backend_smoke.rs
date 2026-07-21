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

fn stair_files(root: &Path) -> Vec<PathBuf> {
    let mut dumps = Vec::new();
    let Ok(entries) = fs::read_dir(root) else {
        return dumps;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            dumps.extend(stair_files(&path));
        } else if path
            .extension()
            .is_some_and(|extension| extension == "stair")
            && path.metadata().is_ok_and(|metadata| metadata.len() > 0)
        {
            dumps.push(path);
        }
    }
    dumps
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
        .join(format!("stair-backend-tests-{fixture}-{}", profile.name()))
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

    command
        .status()
        .unwrap_or_else(|_| panic!("failed to compile {bin} fixture in {} mode", profile.name()))
}

fn executable_path(target_dir: &Path, profile: FixtureProfile, bin: &str) -> PathBuf {
    target_dir
        .join(profile.name())
        .join(format!("{bin}{}", std::env::consts::EXE_SUFFIX))
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
            "stair-sudoku-solver",
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

        let executable = executable_path(&target_dir, profile, "stair-sudoku-solver");
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

#[test]
#[ignore = "AMDGPU kernel path is not supported by the AArch64 Darwin STAIR object backend yet"]
fn backend_smoke_crate_compiles_with_codegen_dylib() {
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
    assert!(
        backend.exists(),
        "expected backend dylib at {}",
        backend.display()
    );

    let fixture = fixture_manifest(&root, "backend-smoke");
    let target_dir = root.join("target").join("stair-backend-tests-smoke");
    if target_dir.exists() {
        fs::remove_dir_all(&target_dir).expect("failed to clear backend smoke target directory");
    }

    let fixture_status = Command::new(&cargo)
        .arg("rustc")
        .arg("--manifest-path")
        .arg(fixture)
        .arg("--bin")
        .arg("stair-backend-smoke")
        .arg("--")
        .arg("--emit=obj")
        .arg(format!("-Zcodegen-backend={}", backend.display()))
        .arg("-Coverflow-checks=off")
        .env("CARGO_TARGET_DIR", &target_dir)
        .status()
        .expect("failed to compile backend smoke fixture");

    assert!(
        fixture_status.success(),
        "backend smoke fixture did not compile with crabbit dylib"
    );

    let objects = object_files(&target_dir);
    assert!(
        !objects.is_empty(),
        "backend smoke fixture did not produce a non-empty object file under {}",
        target_dir.display()
    );

    let executable = target_dir.join("debug").join(format!(
        "stair-backend-smoke{}",
        std::env::consts::EXE_SUFFIX
    ));
    assert!(
        executable.exists(),
        "backend smoke fixture did not produce executable {}",
        executable.display()
    );
    let stair_dump = executable.with_extension("stair");
    assert!(
        stair_dump.exists(),
        "backend smoke fixture did not produce kernel dump {}",
        stair_dump.display()
    );
    let stair_text = fs::read_to_string(&stair_dump).expect("failed to read kernel dump");
    assert!(
        stair_text.contains("amdgpu.kernel"),
        "expected amdgpu.kernel in kernel dump, got:\n{stair_text}"
    );
    assert!(
        stair_text.contains("gfx906"),
        "expected MI50 gfx906 metadata in kernel dump, got:\n{stair_text}"
    );
    assert!(
        stair_text.contains("add_kernel"),
        "expected kernel symbol in kernel dump, got:\n{stair_text}"
    );
    assert!(
        stair_text.contains("slice_kernel"),
        "expected safe slice kernel symbol in kernel dump, got:\n{stair_text}"
    );

    let run_output = Command::new(&executable)
        .output()
        .expect("failed to run backend smoke executable");
    assert!(
        run_output.status.success(),
        "backend smoke executable exited unsuccessfully"
    );
    let stdout = String::from_utf8_lossy(&run_output.stdout);
    assert!(
        stdout.contains("hello from stair"),
        "expected println output from backend smoke executable, got:\n{stdout}"
    );

    if let Ok(output) = Command::new("nm").args(&objects).output()
        && output.status.success()
    {
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("main"),
            "expected main symbol, got:\n{stdout}"
        );
        assert!(
            !stdout.contains("add_kernel"),
            "kernel symbol should be omitted from host object, got:\n{stdout}"
        );
    }
}

#[test]
#[ignore = "AMDGPU kernel path is not supported by the AArch64 Darwin STAIR object backend yet"]
fn llama_rms_norm_crate_compiles_with_codegen_dylib() {
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
    assert!(
        backend.exists(),
        "expected backend dylib at {}",
        backend.display()
    );

    let fixture = fixture_manifest(&root, "llama-rms-norm");
    let target_dir = root
        .join("target")
        .join("stair-backend-tests-llama-rms-norm");
    if target_dir.exists() {
        fs::remove_dir_all(&target_dir).expect("failed to clear RMSNorm target directory");
    }

    let fixture_status = Command::new(&cargo)
        .arg("rustc")
        .arg("--manifest-path")
        .arg(fixture)
        .arg("--bin")
        .arg("stair-llama-rms-norm")
        .arg("--")
        .arg("--emit=obj")
        .arg(format!("-Zcodegen-backend={}", backend.display()))
        .arg("-Coverflow-checks=off")
        .env("CARGO_TARGET_DIR", &target_dir)
        .status()
        .expect("failed to compile RMSNorm fixture");

    assert!(
        fixture_status.success(),
        "RMSNorm fixture did not compile with crabbit dylib"
    );

    let objects = object_files(&target_dir);
    assert!(
        !objects.is_empty(),
        "RMSNorm fixture did not produce a non-empty object file under {}",
        target_dir.display()
    );

    let executable = target_dir.join("debug").join(format!(
        "stair-llama-rms-norm{}",
        std::env::consts::EXE_SUFFIX
    ));
    assert!(
        executable.exists(),
        "RMSNorm fixture did not produce executable {}",
        executable.display()
    );

    let dumps = stair_files(&target_dir);
    assert!(
        !dumps.is_empty(),
        "RMSNorm fixture did not produce a .stair kernel dump under {}",
        target_dir.display()
    );
    let stable_dump = executable.with_extension("stair");
    assert!(
        stable_dump.exists(),
        "RMSNorm fixture did not produce stable kernel dump {}",
        stable_dump.display()
    );
    let stair_text = dumps
        .iter()
        .map(|dump| fs::read_to_string(dump).expect("failed to read RMSNorm kernel dump"))
        .collect::<Vec<_>>()
        .join("\n");

    for expected in [
        "amdgpu.kernel @llama_rms_norm_f32",
        "gfx906",
        "block_id_x",
        "thread_id_x",
        "block_dim_x",
        "reduce_sum_f32",
        "rsqrt_f32",
        "amdgpu.cast",
        "amdgpu.load",
        "amdgpu.store",
    ] {
        assert!(
            stair_text.contains(expected),
            "expected `{expected}` in RMSNorm kernel dump, got:\n{stair_text}"
        );
    }

    let run_output = Command::new(&executable)
        .output()
        .expect("failed to run RMSNorm executable");
    assert!(
        run_output.status.success(),
        "RMSNorm executable exited unsuccessfully"
    );
    let stdout = String::from_utf8_lossy(&run_output.stdout);
    assert!(
        stdout.contains("hello from stair"),
        "expected println output from RMSNorm executable, got:\n{stdout}"
    );

    if let Ok(output) = Command::new("nm").args(&objects).output()
        && output.status.success()
    {
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("main"),
            "expected main symbol in RMSNorm object, got:\n{stdout}"
        );
        assert!(
            !stdout.contains("llama_rms_norm_f32"),
            "kernel symbol should be omitted from RMSNorm host object, got:\n{stdout}"
        );
    }
}
