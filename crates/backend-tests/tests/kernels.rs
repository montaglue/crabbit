//! End-to-end kernel path (docs/KERNEL-ABI.md): a `#![no_std]` device crate
//! with `__crabbit_kernel_*` functions is compiled by the crabbit dylib, the
//! PTX sidecar is assembled by `ptxas` (skipped without a CUDA toolkit), and
//! — when a CUDA driver + GPU are present — executed through the driver API
//! by a small C harness that checks the results exactly.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
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
    } else {
        "libcrabbit.so"
    };
    root.join("target").join("debug").join(file_name)
}

fn build_backend(root: &Path, cargo: &str) -> PathBuf {
    let status = Command::new(cargo)
        .args(["build", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args(["-p", "crabbit"])
        .status()
        .expect("failed to build crabbit backend dylib");
    assert!(status.success(), "crabbit backend build failed");
    let backend = backend_path(root);
    assert!(backend.exists(), "expected backend dylib at {}", backend.display());
    backend
}

fn find_tool(candidates: &[&str]) -> Option<PathBuf> {
    candidates
        .iter()
        .map(PathBuf::from)
        .find(|p| Command::new(p).arg("--version").output().is_ok())
}

#[test]
fn kernel_fixture_compiles_to_ptx_and_runs_on_gpu() {
    let root = repo_root();
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let backend = build_backend(&root, &cargo);
    let fixture_dir = root
        .join("crates/backend-tests/fixtures/kernel-vector-add");
    let target_dir = root.join("target/crabbit-backend-tests-kernel-vector-add");
    if target_dir.exists() {
        fs::remove_dir_all(&target_dir).expect("clear kernel fixture target dir");
    }
    fs::create_dir_all(&target_dir).unwrap();
    let ptx_out = target_dir.join("kernels.ptx");
    let ll_out = target_dir.join("kernels.ll");

    let status = Command::new(&cargo)
        .arg("rustc")
        .arg("--manifest-path")
        .arg(fixture_dir.join("Cargo.toml"))
        .args(["--lib", "--release", "--"])
        .arg(format!("-Zcodegen-backend={}", backend.display()))
        .args(["-Coverflow-checks=off", "-Csave-temps"])
        .env("CARGO_TARGET_DIR", &target_dir)
        .env("CRABBIT_PTX_OUT", &ptx_out)
        .env("CRABBIT_LL_OUT", &ll_out)
        .status()
        .expect("failed to run cargo for the kernel fixture");
    assert!(status.success(), "kernel fixture did not compile with the crabbit dylib");

    // The CRABBIT_PTX_OUT copy and the sidecar next to the object.
    assert!(ptx_out.exists(), "CRABBIT_PTX_OUT was not written");
    let ptx = fs::read_to_string(&ptx_out).unwrap();
    assert!(ptx.contains(".visible .entry vector_add("), "{ptx}");
    assert!(ptx.contains(".visible .entry block_sum_f32("), "{ptx}");
    assert!(ptx.contains(".shared .align 4 .b8 "), "{ptx}");
    assert!(ptx.contains("bar.sync 0;"), "{ptx}");
    assert!(ptx.contains("add.rn.f32"), "{ptx}");
    let sidecars: Vec<PathBuf> = fs::read_dir(target_dir.join("release/deps"))
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "ptx"))
        .collect();
    assert!(!sidecars.is_empty(), "no .ptx sidecar next to the object");
    assert_eq!(fs::read_to_string(&sidecars[0]).unwrap(), ptx);

    // The llvm-export leg (rust-llvm arm): kernels + intrinsics + shared
    // memory in textual LLVM IR.
    let ll = fs::read_to_string(&ll_out).expect("CRABBIT_LL_OUT was not written");
    assert!(ll.contains("define ptx_kernel void @vector_add("), "{ll}");
    assert!(ll.contains("declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()"), "{ll}");
    assert!(ll.contains("addrspace(3) global [1024 x i8] zeroinitializer, align 4"), "{ll}");
    assert!(ll.contains("call void @llvm.nvvm.barrier0()"), "{ll}");

    let Some(ptxas) = find_tool(&["ptxas", "/usr/local/cuda/bin/ptxas"]) else {
        eprintln!("ptxas not found; skipping assembly + GPU execution");
        return;
    };
    let output = Command::new(&ptxas)
        .args(["-arch=sm_121", "-o"])
        .arg(target_dir.join("kernels.cubin"))
        .arg(&ptx_out)
        .output()
        .expect("run ptxas");
    assert!(
        output.status.success(),
        "ptxas rejected the fixture PTX:\n{}\n{ptx}",
        String::from_utf8_lossy(&output.stderr)
    );

    // GPU execution through the driver API.
    let cuda_include = Path::new("/usr/local/cuda/include");
    let libcuda_present = ["/usr/lib/aarch64-linux-gnu/libcuda.so.1", "/usr/lib/x86_64-linux-gnu/libcuda.so.1", "/usr/lib64/libcuda.so.1"]
        .iter()
        .any(|p| Path::new(p).exists());
    if !cuda_include.join("cuda.h").exists() || !libcuda_present {
        eprintln!("no CUDA driver/headers; skipping GPU execution");
        return;
    }
    let harness = target_dir.join("run_kernels");
    let gcc = Command::new("gcc")
        .arg("-O1")
        .arg("-o")
        .arg(&harness)
        .arg(fixture_dir.join("harness/run_kernels.c"))
        .arg(format!("-I{}", cuda_include.display()))
        .arg("-lcuda")
        .output()
        .expect("run gcc");
    assert!(
        gcc.status.success(),
        "harness build failed:\n{}",
        String::from_utf8_lossy(&gcc.stderr)
    );
    let run = Command::new(&harness).arg(&ptx_out).output().expect("run harness");
    let stderr = String::from_utf8_lossy(&run.stderr);
    if stderr.contains("cuInit(0) failed") || stderr.contains("cuDeviceGet") {
        eprintln!("no usable GPU: {stderr}; skipping GPU execution");
        return;
    }
    assert!(
        run.status.success(),
        "kernels produced wrong results on the GPU:\n{stderr}\n{}",
        String::from_utf8_lossy(&run.stdout)
    );
    let stdout = String::from_utf8_lossy(&run.stdout);
    assert!(stdout.contains("vector_add ok") && stdout.contains("block_sum_f32 ok"), "{stdout}");

    // Same kernels through LLVM's NVPTX backend, when an llc is around
    // (LLVM 18 knows sm_90 at most; the PTX it emits still runs on sm_121).
    let llc_home = std::env::var("HOME").map(|h| format!("{h}/.local/opt/llvm18/bin/llc")).unwrap_or_default();
    let Some(llc) = find_tool(&["llc", llc_home.as_str()]) else {
        eprintln!("llc not found; skipping the rust-llvm arm");
        return;
    };
    let llvm_ptx = target_dir.join("kernels-llvm.ptx");
    let out = Command::new(&llc)
        .args(["-march=nvptx64", "-mcpu=sm_90", "-O3", "-o"])
        .arg(&llvm_ptx)
        .arg(&ll_out)
        .output()
        .expect("run llc");
    assert!(out.status.success(), "llc rejected the exported IR:\n{}\n{ll}", String::from_utf8_lossy(&out.stderr));
    let run = Command::new(&harness).arg(&llvm_ptx).output().expect("run harness (llvm arm)");
    assert!(
        run.status.success(),
        "LLVM-compiled kernels produced wrong results on the GPU:\n{}",
        String::from_utf8_lossy(&run.stderr)
    );
}
