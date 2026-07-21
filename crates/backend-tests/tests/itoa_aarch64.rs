use std::{
    ffi::{OsStr, OsString},
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/backend-tests should be two levels below the repository root")
        .to_path_buf()
}

fn backend_path(root: &Path) -> PathBuf {
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

    fn cargo_release_arg(self) -> Option<OsString> {
        match self {
            Self::Debug => None,
            Self::Release => Some(OsString::from("--release")),
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

    objects.sort();
    objects
}

fn command_line(program: &str, args: &[OsString]) -> String {
    let mut rendered = shell_word(program);
    for arg in args {
        rendered.push(' ');
        rendered.push_str(&shell_word(&arg.to_string_lossy()));
    }
    rendered
}

fn shell_word(value: &str) -> String {
    if value
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '/' | '.' | '_' | '-' | '=' | ':'))
    {
        value.to_string()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}

fn run_command(program: &str, args: Vec<OsString>, envs: &[(&str, &OsStr)]) -> CommandResult {
    let output = Command::new(program)
        .args(&args)
        .envs(envs.iter().map(|(key, value)| (*key, value)))
        .output()
        .unwrap_or_else(|err| panic!("failed to run {}: {err}", command_line(program, &args)));

    CommandResult {
        command: command_line(program, &args),
        output,
    }
}

struct CommandResult {
    command: String,
    output: Output,
}

impl CommandResult {
    fn assert_success(&self, context: &str) {
        assert!(
            self.output.status.success(),
            "{context} failed\n{}",
            self.failure_report("")
        );
    }

    fn failure_report(&self, extra: &str) -> String {
        let mut report = String::new();
        let _ = writeln!(report, "command: {}", self.command);
        let _ = writeln!(report, "status: {}", status_summary(&self.output));
        write_output(&mut report, "stdout", &self.output.stdout);
        write_output(&mut report, "stderr", &self.output.stderr);
        if !extra.is_empty() {
            let _ = writeln!(report, "{extra}");
        }
        report
    }
}

fn status_summary(output: &Output) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = output.status.signal() {
            return format!("terminated by signal {signal}");
        }
    }

    match output.status.code() {
        Some(code) => format!("exited with code {code}"),
        None => output.status.to_string(),
    }
}

fn write_output(report: &mut String, label: &str, bytes: &[u8]) {
    let text = String::from_utf8_lossy(bytes);
    if text.is_empty() {
        let _ = writeln!(report, "{label}: <empty>");
    } else {
        let _ = writeln!(report, "{label}:\n{text}");
    }
}

fn build_backend(root: &Path, cargo: &str) -> PathBuf {
    let args = vec![
        OsString::from("build"),
        OsString::from("--manifest-path"),
        root.join("Cargo.toml").into_os_string(),
        OsString::from("-p"),
        OsString::from("crabbit"),
    ];
    let result = run_command(cargo, args, &[]);
    result.assert_success("crabbit backend build");

    let backend = backend_path(root);
    assert!(
        backend.exists(),
        "expected backend dylib at {}",
        backend.display()
    );
    backend
}

fn symbol_snippet(objects: &[PathBuf]) -> String {
    if objects.is_empty() {
        return String::new();
    }

    let output = Command::new("nm").args(objects).output();
    let Ok(output) = output else {
        return "nm: failed to launch\n".to_string();
    };

    let mut report = String::new();
    let _ = writeln!(report, "nm status: {}", status_summary(&output));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let snippet = stdout.lines().take(80).collect::<Vec<_>>().join("\n");
    if snippet.is_empty() {
        let _ = writeln!(report, "nm stdout: <empty>");
    } else {
        let _ = writeln!(report, "nm stdout first 80 lines:\n{snippet}");
    }
    if !stderr.is_empty() {
        let _ = writeln!(report, "nm stderr:\n{stderr}");
    }
    report
}

fn lldb_backtrace(executable: &Path) -> String {
    if !cfg!(target_os = "macos") {
        return String::new();
    }

    let output = Command::new("lldb")
        .arg("--batch")
        .arg("-o")
        .arg("run")
        .arg("-o")
        .arg("thread backtrace all")
        .arg("--")
        .arg(executable)
        .output();

    let Ok(output) = output else {
        return "lldb: failed to launch\n".to_string();
    };

    let mut report = String::new();
    let _ = writeln!(report, "lldb status: {}", status_summary(&output));
    write_output(&mut report, "lldb stdout", &output.stdout);
    write_output(&mut report, "lldb stderr", &output.stderr);
    report
}

fn run_failure_details(executable: &Path, target_dir: &Path, objects: &[PathBuf]) -> String {
    let mut details = String::new();
    let _ = writeln!(details, "executable: {}", executable.display());
    let _ = writeln!(details, "target dir: {}", target_dir.display());
    let _ = writeln!(details, "object files:");
    for object in objects {
        let _ = writeln!(details, "  {}", object.display());
    }

    let symbols = symbol_snippet(objects);
    if !symbols.is_empty() {
        let _ = writeln!(details, "\n{symbols}");
    }

    let lldb = lldb_backtrace(executable);
    if !lldb.is_empty() {
        let _ = writeln!(details, "\n{lldb}");
    }

    details
}

#[test]
fn itoa_aarch64_crate_compiles_objects_and_runs_with_codegen_dylib() {
    let root = repo_root();
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let backend = build_backend(&root, &cargo);

    let fixture = root
        .join("crates")
        .join("backend-tests")
        .join("fixtures")
        .join("itoa-aarch64")
        .join("Cargo.toml");
    for profile in FixtureProfile::ALL {
        let target_dir = root.join("target").join(format!(
            "stair-backend-tests-itoa-aarch64-{}",
            profile.name()
        ));
        if target_dir.exists() {
            fs::remove_dir_all(&target_dir).expect("failed to clear itoa fixture target directory");
        }

        let mut build_args = vec![
            OsString::from("rustc"),
            OsString::from("--manifest-path"),
            fixture.clone().into_os_string(),
            OsString::from("--bin"),
            OsString::from("itoa-aarch64"),
        ];
        if let Some(arg) = profile.cargo_release_arg() {
            build_args.push(arg);
        }
        build_args.extend([
            OsString::from("--"),
            OsString::from(format!("-Zcodegen-backend={}", backend.display())),
            OsString::from("-Coverflow-checks=off"),
        ]);
        let compile = run_command(
            &cargo,
            build_args,
            &[("CARGO_TARGET_DIR", target_dir.as_os_str())],
        );
        compile.assert_success(&format!("itoa fixture {} compile", profile.name()));

        let objects = object_files(&target_dir);
        if profile.keeps_intermediate_objects() {
            assert!(
                !objects.is_empty(),
                "itoa fixture did not produce a non-empty object file under {}",
                target_dir.display()
            );
        }

        let executable = target_dir
            .join(profile.name())
            .join(format!("itoa-aarch64{}", std::env::consts::EXE_SUFFIX));
        assert!(
            executable.exists(),
            "itoa fixture did not produce executable {}",
            executable.display()
        );

        let run = run_command(&executable.to_string_lossy(), Vec::new(), &[]);
        assert!(
            run.output.status.success(),
            "itoa fixture executable failed in {} mode\n{}",
            profile.name(),
            run.failure_report(&run_failure_details(&executable, &target_dir, &objects))
        );
    }
}
