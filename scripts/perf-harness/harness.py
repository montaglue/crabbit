#!/usr/bin/env python3
"""crabbit A/B perf & metrics harness.

Compiles a corpus (runnable fixtures + arrow-rs library crates) under a matrix
of named env-var configurations against the crabbit rustc codegen backend, and
compares hard numbers: runtime wall time, compile wall time, text size,
instruction counts, and spill-traffic proxies (sp-relative loads/stores,
mov-immediate counts) from objdump disassembly.

Config matrix is generic: a config is just {"name": ..., "env": {...}}, so new
research env vars (e.g. CRABBIT_REGALLOC) are simply more keys in configs.json.

Fixture output mismatches vs the known-good expectations (lifted from
crates/backend-tests/tests/backend_smoke.rs) or vs the baseline config's output
are flagged loudly as MISCOMPILE and excluded from perf comparison.

Usage:
  python3 scripts/perf-harness/harness.py [--configs configs.json] [--runs 10]
      [--workdir DIR] [--fixtures a,b,...] [--arrow-crates a,b,...]
      [--skip-fixtures] [--skip-arrow] [--skip-backend-build] [--no-hyperfine]

Outputs: scripts/perf-harness/results/<timestamp>/{results.csv,summary.md,meta.json}
(summary.md is also printed to stdout).
"""

import argparse
import hashlib
import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
TOOLCHAIN = "nightly-2026-04-03"
# A durable checkout (the original lived in a session scratchpad that may be
# cleaned): apache/arrow-rs v59.2.0 copied under the harness workdir.
DEFAULT_ARROW_RS = str(Path.home() / ".cache/crabbit-perf-harness/arrow-rs")
ARROW_CANDIDATES = ["arrow-schema", "arrow-buffer", "arrow-data", "arrow-array"]

# ---------------------------------------------------------------------------
# Fixture corpus (expectations lifted verbatim from backend_smoke.rs)
# ---------------------------------------------------------------------------

STDIN_INPUT = "  10 20\n30 40  \n\n5\n"
SUDOKU_PUZZLE = (
    "530070000600195000098000060800060003400803001"
    "700020006060000280000419005000080079"
)
SUDOKU_SOLUTION = (
    "5 3 4 6 7 8 9 1 2\n6 7 2 1 9 5 3 4 8\n1 9 8 3 4 2 5 6 7\n"
    "8 5 9 7 6 1 4 2 3\n4 2 6 8 5 3 7 9 1\n7 1 3 9 2 4 8 5 6\n"
    "9 6 1 5 3 7 2 8 4\n2 8 7 4 1 9 6 3 5\n3 4 5 2 8 6 1 7 9\n"
)
SUDOKU_BAD = "11" + "0" * 79


class RunCase:
    def __init__(self, args=None, stdin=None, expected_stdout=None,
                 expected_exit=0, stderr_contains=None, label=""):
        self.args = args or []
        self.stdin = stdin
        # None => no hardcoded expectation; gate on exit code + equality with
        # the baseline config's captured stdout.
        self.expected_stdout = expected_stdout
        self.expected_exit = expected_exit
        self.stderr_contains = stderr_contains
        self.label = label


class Fixture:
    def __init__(self, name, bin_name=None, runs=None, manifest=None,
                 package=None, src_main=None, kind="fixture"):
        self.name = name
        self.bin = bin_name or name
        self.runs = runs or [RunCase()]
        # Where cargo finds it: crabbit's backend-tests fixture by default,
        # or an explicit workspace manifest + package (kernel-corpus CPU
        # variants).
        self.manifest = manifest or (REPO / "crates" / "backend-tests"
                                     / "fixtures" / name / "Cargo.toml")
        self.package = package
        self.src_main = src_main or (self.manifest.parent / "src" / "main.rs")
        self.kind = kind


FIXTURES = [
    Fixture("pure-rust-aarch64"),
    Fixture("structs-aarch64"),
    Fixture("hello-world-aarch64",
            runs=[RunCase(expected_stdout="Hello, world!\n")]),
    Fixture("fp-aarch64", runs=[RunCase(expected_stdout="fp ok\n")]),
    Fixture("int128-aarch64", runs=[RunCase(expected_stdout="int128 ok\n")]),
    Fixture("hashmap-aarch64",
            runs=[RunCase(expected_stdout="hashmap ok len=2 crab=bitte\n")]),
    Fixture("itoa-aarch64", bin_name="itoa-aarch64"),
    Fixture("oc-course"),
    Fixture("stdin-aarch64", runs=[
        RunCase(args=["read_to_string"], stdin=STDIN_INPUT, label="read_to_string",
                expected_stdout="bytes=19\ntrimmed=[10 20\n30 40  \n\n5]\nsum=105\n"),
        RunCase(args=["read_line"], stdin=STDIN_INPUT, label="read_line",
                expected_stdout="first=[10 20]\n"),
        RunCase(args=["lines"], stdin=STDIN_INPUT, label="lines",
                expected_stdout="lines=3\ntotal=105\n"),
    ]),
    Fixture("sudoku-aarch64", runs=[RunCase(expected_stdout=(
        "534678912\n672195348\n198342567\n859761423\n426853791\n"
        "713924856\n961537284\n287419635\n345286179\n"))]),
    Fixture("sudoku-solver", bin_name="crabbit-sudoku-solver", runs=[
        RunCase(args=[SUDOKU_PUZZLE], expected_stdout=SUDOKU_SOLUTION,
                label="argv"),
        RunCase(stdin=f"  {SUDOKU_PUZZLE}\n", expected_stdout=SUDOKU_SOLUTION,
                label="stdin"),
        RunCase(args=[SUDOKU_BAD], expected_exit=1,
                stderr_contains="duplicate 1 at row 1, column 2",
                label="reject"),
    ]),
]

# AMDGPU-path fixtures (backend-smoke, llama-rms-norm) are #[ignore]d in the
# smoke suite and are skipped here too.

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def log(msg):
    print(f"[harness] {msg}", flush=True)


def base_env(extra=None):
    env = dict(os.environ)
    env["RUSTUP_TOOLCHAIN"] = TOOLCHAIN
    # Make sure stale policy vars in the ambient environment never leak in.
    for key in list(env):
        if key.startswith("CRABBIT_"):
            del env[key]
    if extra:
        env.update(extra)
    return env


def compile_failure(results, cname, what, proc):
    """Persist a failed compile's stderr next to results.csv; return the
    one-line miscompile detail."""
    errlog = results.csv_path.parent / f"compile-fail-{cname}-{what}.log"
    errlog.write_text(proc.stderr)
    last = proc.stderr.strip().splitlines()[-1][:300] if proc.stderr.strip() \
        else "(no stderr)"
    return f"COMPILE FAILED (stderr in {errlog.name}): {last}"


# CPU pinning for every timed run (`--pin-cpu N`; empty = unpinned).
PIN_PREFIX = []


def run_cmd(cmd, env=None, cwd=None, timeout=3600, stdin_data=None):
    return subprocess.run(
        cmd, env=env, cwd=cwd, timeout=timeout, input=stdin_data,
        capture_output=True, text=True)


def touch(path):
    os.utime(path, None)


class Results:
    """Long-format (config, target, metric, value) accumulator.

    Rows are checkpointed to results.csv immediately so a killed run keeps
    everything measured so far, and re-invocations with the same --outdir
    append (existing rows are loaded back for the summary)."""

    def __init__(self, csv_path=None):
        self.rows = []  # (config, target, metric, value)
        self.miscompiles = []  # (config, target, detail)
        self.notes = []
        self.csv_path = csv_path
        if csv_path and Path(csv_path).exists():
            with open(csv_path, newline="") as f:
                for row in csv.reader(f):
                    if len(row) != 4 or row[0] == "config":
                        continue
                    config, target, metric, value = row
                    if metric == "MISCOMPILE":
                        self.miscompiles.append((config, target, value))
                    else:
                        try:
                            value = int(value)
                        except ValueError:
                            value = float(value)
                        self.rows.append((config, target, metric, value))
        elif csv_path:
            self._append_row(["config", "target", "metric", "value"])

    def _append_row(self, row):
        if self.csv_path:
            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow(row)

    def add(self, config, target, metric, value):
        self.rows.append((config, target, metric, value))
        self._append_row([config, target, metric, value])

    def miscompile(self, config, target, detail):
        self.miscompiles.append((config, target, detail))
        self._append_row([config, target, "MISCOMPILE", detail.splitlines()[0]])
        log(f"!!! MISCOMPILE [{config}] {target}: {detail.splitlines()[0][:200]}")

    def get(self, config, target, metric):
        for c, t, m, v in self.rows:
            if c == config and t == target and m == metric:
                return v
        return None


# ---------------------------------------------------------------------------
# Object-file static metrics
# ---------------------------------------------------------------------------

INSN_RE = re.compile(r"^\s*[0-9a-f]+:\s+[0-9a-f]{8}\s+(\S+)\s*(.*)$")
LOADS = {"ldr", "ldur", "ldrb", "ldrh", "ldrsb", "ldrsh", "ldrsw"}
STORES = {"str", "stur", "strb", "strh"}
LOAD_PAIRS = {"ldp", "ldnp", "ldpsw"}
STORE_PAIRS = {"stp", "stnp"}
MOV_IMM = {"mov", "movz", "movn", "movk"}


SYMBOL_RE = re.compile(r"^[0-9a-f]+ <(.*)>:$")
# Per-function ("kfn_") metrics are collected for symbols whose demangled
# name matches this: the corpus CPU variants keep their loop nest in
# `fn kernel(...)`, which crabbit's inliner leaves as its own symbol.
KERNEL_FN_RE = re.compile(r"::kernel$")


def classify(mnem, ops, m):
    sp_ref = "[sp" in ops
    if mnem in LOADS and sp_ref:
        m["ldr_sp"] += 1
    elif mnem in STORES and sp_ref:
        m["str_sp"] += 1
    elif mnem in LOAD_PAIRS and sp_ref:
        m["ldp_sp"] += 1
    elif mnem in STORE_PAIRS and sp_ref:
        m["stp_sp"] += 1
    elif mnem in MOV_IMM and "#" in ops:
        m["mov_imm"] += 1


def analyze_objects(objects):
    """Aggregate static metrics over a list of .o files: whole-object
    counts, plus `kfn_*` counts restricted to the kernel function(s)
    (absent when no symbol matches KERNEL_FN_RE)."""
    m = {"obj_bytes": 0, "text_bytes": 0, "insns": 0, "ldr_sp": 0,
         "str_sp": 0, "ldp_sp": 0, "stp_sp": 0, "mov_imm": 0,
         "objects": len(objects)}
    k = {"insns": 0, "ldr_sp": 0, "str_sp": 0, "ldp_sp": 0, "stp_sp": 0,
         "mov_imm": 0, "symbols": 0}
    for obj in objects:
        m["obj_bytes"] += obj.stat().st_size
        hdr = run_cmd(["objdump", "-h", str(obj)])
        for line in hdr.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 4 and parts[0].isdigit() and \
                    parts[1].startswith(".text"):
                m["text_bytes"] += int(parts[2], 16)
        dis = run_cmd(["objdump", "-d", "-C", str(obj)])
        in_kernel = False
        for line in dis.stdout.splitlines():
            sym = SYMBOL_RE.match(line)
            if sym:
                in_kernel = bool(KERNEL_FN_RE.search(sym.group(1)))
                if in_kernel:
                    k["symbols"] += 1
                continue
            match = INSN_RE.match(line)
            if not match:
                continue
            mnem, ops = match.group(1), match.group(2)
            if mnem in (".word", ".inst", ".byte", ".short"):
                continue
            m["insns"] += 1
            classify(mnem, ops, m)
            if in_kernel:
                k["insns"] += 1
                classify(mnem, ops, k)
    if k["symbols"]:
        for key, value in k.items():
            m[f"kfn_{key}"] = value
    return m


def find_objects(root, crate_prefix, newer_than):
    """Find non-empty .o files for a crate written after `newer_than`."""
    prefix = crate_prefix.replace("-", "_")
    out = []
    for path in Path(root).rglob("*.o"):
        try:
            st = path.stat()
        except OSError:
            continue
        if st.st_size > 0 and st.st_mtime >= newer_than and \
                path.name.startswith(prefix):
            out.append(path)
    return sorted(out)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def time_with_python(exe, case, runs, warmups=2):
    times = []
    for i in range(warmups + runs):
        start = time.monotonic()
        try:
            proc = subprocess.run(
                PIN_PREFIX + [str(exe)] + case.args, input=case.stdin,
                text=True, capture_output=True, timeout=600)
        except subprocess.TimeoutExpired:
            return None, "run exceeded 600s during timing"
        elapsed = time.monotonic() - start
        code = proc.returncode
        expected = case.expected_exit
        if code != expected:
            return None, f"exit code {code} (expected {expected}) during timing"
        if i >= warmups:
            times.append(elapsed)
    return times, None


def time_with_hyperfine(exe, case, runs, tmpdir):
    """Use hyperfine when available and the case needs no stdin."""
    export = Path(tmpdir) / "hyperfine.json"
    cmd = " ".join(PIN_PREFIX + [str(exe)] + [f"'{a}'" for a in case.args])
    proc = run_cmd(["hyperfine", "--warmup", "2", "--runs", str(runs),
                    "--style", "none", "--export-json", str(export), cmd])
    if proc.returncode != 0 or not export.exists():
        return None, f"hyperfine failed: {proc.stderr[:200]}"
    data = json.loads(export.read_text())["results"][0]
    return data["times"], None


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


def compile_fixture(fixture, backend, target_dir, env_extra, timed=True):
    env = base_env(env_extra)
    env["CARGO_TARGET_DIR"] = str(target_dir)
    cmd = ["cargo", "rustc", "--manifest-path", str(fixture.manifest)]
    if fixture.package:
        cmd += ["-p", fixture.package]
    cmd += ["--bin", fixture.bin, "--release", "--",
           f"-Zcodegen-backend={backend}", "-Coverflow-checks=off",
           "-Csave-temps"]
    start = time.monotonic()
    proc = run_cmd(cmd, env=env, cwd=str(REPO))
    elapsed = time.monotonic() - start
    ok = proc.returncode == 0
    return ok, elapsed, proc


def compile_arrow_crate(crate, backend, arrow_rs, target_dir, env_extra):
    env = base_env(env_extra)
    env["CARGO_TARGET_DIR"] = str(target_dir)
    cmd = ["cargo", "rustc", "-p", crate, "--lib", "--",
           f"-Zcodegen-backend={backend}", "-Coverflow-checks=off",
           "-Csave-temps"]
    start = time.monotonic()
    proc = run_cmd(cmd, env=env, cwd=str(arrow_rs), timeout=3600)
    elapsed = time.monotonic() - start
    return proc.returncode == 0, elapsed, proc


# ---------------------------------------------------------------------------
# Correctness gate
# ---------------------------------------------------------------------------


def check_run(proc_out, case, baseline_stdout):
    """Return (ok, detail). baseline_stdout is the baseline config's captured
    stdout for cases with no hardcoded expectation (None on baseline itself)."""
    stdout, stderr, code = proc_out
    if code != case.expected_exit:
        return False, (f"exit code {code}, expected {case.expected_exit}\n"
                       f"stderr: {stderr[:500]}")
    if case.expected_stdout is not None and stdout != case.expected_stdout:
        return False, (f"stdout mismatch\nexpected: {case.expected_stdout!r}\n"
                       f"actual:   {stdout!r}")
    if case.stderr_contains and case.stderr_contains not in stderr:
        return False, f"stderr missing {case.stderr_contains!r}: {stderr[:500]}"
    if case.expected_stdout is None and baseline_stdout is not None \
            and stdout != baseline_stdout:
        return False, (f"stdout differs from baseline config\n"
                       f"baseline: {baseline_stdout!r}\nactual:   {stdout!r}")
    return True, ""


# ---------------------------------------------------------------------------
# Main phases
# ---------------------------------------------------------------------------


def build_backend():
    # Research dylib first: the engine axes (CRABBIT_REGALLOC=eregalloc,
    # cmt-provider frequencies) only exist in the composition build, which
    # needs the private sibling checkouts (crates/crabbit-research).
    research = REPO / "crates" / "crabbit-research"
    if (research / "Cargo.toml").exists():
        log("building research backend (crates/crabbit-research)...")
        proc = run_cmd(["cargo", "build"], env=base_env(), cwd=str(research),
                       timeout=3600)
        backend = research / "target" / "debug" / "libcrabbit_research.so"
        if proc.returncode == 0 and backend.exists():
            return backend
        log("WARNING: crabbit-research build failed (missing private "
            "research checkouts?); falling back to the engine-less "
            "libcrabbit.so -- eregalloc/cmt configs will error")
    log("building crabbit backend (cargo build -p crabbit)...")
    proc = run_cmd(["cargo", "build", "-p", "crabbit"], env=base_env(),
                   cwd=str(REPO), timeout=3600)
    if proc.returncode != 0:
        sys.exit(f"backend build failed:\n{proc.stderr[-4000:]}")
    backend = REPO / "target" / "debug" / "libcrabbit.so"
    if not backend.exists():
        sys.exit(f"backend dylib missing at {backend}")
    return backend


def corpus_targets(corpus_root, kernel_names):
    """kernel-corpus CPU variants as Fixtures: `kernels/<k>/cpu` bins named
    `<k>-cpu`, run for both spec sizes (exact checksums from spec.json when
    present, otherwise gated on baseline equality), timed on `--size large`
    with the spec's repeat count."""
    corpus_root = Path(corpus_root)
    manifest = corpus_root / "Cargo.toml"
    targets = []
    if not manifest.exists():
        return targets
    for spec_path in sorted((corpus_root / "kernels").glob("*/spec.json")):
        kdir = spec_path.parent
        name = kdir.name
        if kernel_names and name not in kernel_names:
            continue
        if not (kdir / "cpu" / "Cargo.toml").exists():
            continue
        spec = json.loads(spec_path.read_text())
        # The package/bin name comes from the crate manifest (hyphenated),
        # not the kernel directory (underscored).
        cargo_toml = (kdir / "cpu" / "Cargo.toml").read_text()
        m = re.search(r'^name\s*=\s*"([^"]+)"', cargo_toml, re.M)
        package = m.group(1) if m else f"{name}-cpu"
        checksums = spec.get("checksum") or {}
        cases = []
        # Timed case first (large), correctness on both sizes.
        for size in ("large", "small"):
            expected = None
            if size in checksums:
                expected = f"checksum {checksums[size]}\n"
            cases.append(RunCase(args=["--size", size], label=size,
                                 expected_stdout=expected))
        targets.append(Fixture(name, bin_name=package, runs=cases,
                               manifest=manifest, package=package,
                               src_main=kdir / "cpu" / "src" / "main.rs",
                               kind="corpus"))
    return targets


def run_fixture_phase(configs, backend, workdir, runs, results, use_hyperfine,
                      fixture_names, fixtures=None, subdir="fixtures"):
    baseline_name = configs[0]["name"]
    baseline_stdout = {}  # (fixture, case_idx) -> stdout
    if fixtures is None:
        fixtures = [f for f in FIXTURES
                    if not fixture_names or f.name in fixture_names]
    for fixture in fixtures:
        target_dir = workdir / subdir / fixture.name
        target_dir.mkdir(parents=True, exist_ok=True)
        src_main = fixture.src_main
        warmed = False
        for config in configs:
            cname, cenv = config["name"], config["env"]
            target = f"{fixture.kind}:{fixture.name}"
            log(f"[{cname}] compiling {fixture.name}...")
            if not warmed:
                ok, _, proc = compile_fixture(fixture, backend, target_dir,
                                              cenv)
                if not ok:
                    results.miscompile(
                        cname, target, "(warmup) " + compile_failure(
                            results, cname, fixture.name, proc))
                    break
                warmed = True
            # Force recompile of the leaf crate (cargo does not track env
            # vars), then time just the leaf compile.
            touch(src_main)
            marker = time.time() - 1
            ok, compile_s, proc = compile_fixture(fixture, backend,
                                                  target_dir, cenv)
            if not ok:
                results.miscompile(cname, target,
                                   compile_failure(results, cname, fixture.name, proc))
                continue
            results.add(cname, target, "compile_s", round(compile_s, 3))

            # Static metrics from this fixture's own objects (-Csave-temps).
            objs = find_objects(target_dir / "release",
                               fixture.bin, marker)
            if objs:
                for k, v in analyze_objects(objs).items():
                    results.add(cname, target, k, v)

            exe = target_dir / "release" / fixture.bin
            if not exe.exists():
                results.miscompile(cname, target, f"missing executable {exe}")
                continue

            # Correctness gate over all run cases.
            miscompiled = False
            for idx, case in enumerate(fixture.runs):
                try:
                    proc2 = subprocess.run(
                        [str(exe)] + case.args, input=case.stdin, text=True,
                        capture_output=True, timeout=600)
                except subprocess.TimeoutExpired:
                    label = case.label or f"case{idx}"
                    results.miscompile(cname, target,
                                       f"[{label}] run exceeded 600s")
                    miscompiled = True
                    continue
                key = (fixture.name, idx)
                base_out = None
                if cname == baseline_name:
                    baseline_stdout[key] = proc2.stdout
                else:
                    base_out = baseline_stdout.get(key)
                ok, detail = check_run(
                    (proc2.stdout, proc2.stderr, proc2.returncode),
                    case, base_out)
                if not ok:
                    label = case.label or f"case{idx}"
                    results.miscompile(cname, target, f"[{label}] {detail}")
                    miscompiled = True
            results.add(cname, target, "correct", 0 if miscompiled else 1)
            if miscompiled:
                continue  # perf numbers for a miscompile are meaningless

            # Timing: first run case only.
            case = fixture.runs[0]
            times = None
            if use_hyperfine and case.stdin is None:
                times, err = time_with_hyperfine(exe, case, runs, workdir)
                if err:
                    log(f"    {err}; falling back to python timer")
                    times = None
            if times is None:
                times, err = time_with_python(exe, case, runs)
            if times is None:
                results.miscompile(cname, target, f"timing failed: {err}")
                continue
            results.add(cname, target, "run_median_s",
                        round(statistics.median(times), 6))
            results.add(cname, target, "run_min_s", round(min(times), 6))


def run_arrow_phase(configs, backend, arrow_rs, workdir, results,
                    crate_names, probe=True):
    arrow_rs = Path(arrow_rs)
    if not arrow_rs.exists():
        results.notes.append(f"arrow-rs checkout not found at {arrow_rs}; "
                             "skipping compile-only corpus")
        log(results.notes[-1])
        return
    target_dir = workdir / "arrow-target"
    target_dir.mkdir(parents=True, exist_ok=True)
    candidates = crate_names or ARROW_CANDIDATES
    baseline_env = configs[0]["env"]
    usable = []
    for crate in candidates:
        lib_rs = arrow_rs / crate / "src" / "lib.rs"
        if not lib_rs.exists():
            results.notes.append(f"{crate}: no src/lib.rs, skipped")
            continue
        if not probe:
            usable.append(crate)
            continue
        log(f"probing {crate} (baseline warmup compile)...")
        ok, elapsed, proc = compile_arrow_crate(crate, backend, arrow_rs,
                                                target_dir, baseline_env)
        if ok:
            usable.append(crate)
            log(f"  {crate}: compiles ({elapsed:.1f}s warm-up incl. deps)")
        else:
            tail = proc.stderr[-600:].replace("\n", " | ")
            results.notes.append(f"{crate}: does NOT compile with crabbit "
                                 f"(skipped). tail: {tail}")
            log(f"  {crate}: FAILED, excluded from matrix")

    for crate in usable:
        lib_rs = arrow_rs / crate / "src" / "lib.rs"
        for config in configs:
            cname, cenv = config["name"], config["env"]
            target = f"arrow:{crate}"
            log(f"[{cname}] compiling {crate}...")
            touch(lib_rs)
            marker = time.time() - 1
            ok, compile_s, proc = compile_arrow_crate(
                crate, backend, arrow_rs, target_dir, cenv)
            if not ok:
                results.miscompile(cname, target,
                                   compile_failure(results, cname, crate, proc))
                continue
            results.add(cname, target, "compile_s", round(compile_s, 3))
            objs = find_objects(target_dir / "debug" / "deps", crate, marker)
            if not objs:
                results.notes.append(
                    f"[{cname}] {crate}: no fresh .o files found under "
                    f"{target_dir}/debug/deps")
                continue
            for k, v in analyze_objects(objs).items():
                results.add(cname, target, k, v)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

METRIC_ORDER = ["run_median_s", "run_min_s", "compile_s", "text_bytes",
                "insns", "ldr_sp", "str_sp", "ldp_sp", "stp_sp", "mov_imm",
                "obj_bytes", "objects", "correct"]
LOWER_IS_BETTER = {"run_median_s", "run_min_s", "compile_s", "text_bytes",
                   "insns", "ldr_sp", "str_sp", "ldp_sp", "stp_sp",
                   "obj_bytes"}


def fmt_val(metric, v):
    if v is None:
        return "-"
    if metric.endswith("_s"):
        return f"{v:.4f}" if v < 1 else f"{v:.2f}"
    return str(v)


def render_summary(configs, results, runs):
    lines = []
    cfg_names = [c["name"] for c in configs]
    base = cfg_names[0]
    lines.append("# crabbit A/B harness results\n")
    lines.append(f"Toolchain `{TOOLCHAIN}`, backend `target/debug/libcrabbit.so`, "
                 f"{runs} timed runs per fixture (release profile); arrow "
                 "crates compiled debug, leaf-crate-only recompiles timed.\n")
    lines.append("## Configs\n")
    for c in configs:
        env_desc = ", ".join(f"{k}={v}" for k, v in c["env"].items()) or "(none)"
        lines.append(f"- **{c['name']}**: {env_desc}")
    lines.append("")

    if results.miscompiles:
        lines.append("## CORRECTNESS FAILURES / MISCOMPILES\n")
        for cname, target, detail in results.miscompiles:
            first = detail.splitlines()[0]
            lines.append(f"- **MISCOMPILE** `[{cname}] {target}`: {first}")
        lines.append("")
    else:
        lines.append("## Correctness gate: ALL PASS "
                     "(no output differences across configs)\n")

    targets = []
    for _, t, _, _ in results.rows:
        if t not in targets:
            targets.append(t)

    for target in targets:
        lines.append(f"## {target}\n")
        header = "| metric | " + " | ".join(
            [base] + [f"{c} (Δ vs {base})" for c in cfg_names[1:]]) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(cfg_names) + 1))
        for metric in METRIC_ORDER:
            base_v = results.get(base, target, metric)
            vals = [results.get(c, target, metric) for c in cfg_names]
            if all(v is None for v in vals):
                continue
            cells = [fmt_val(metric, base_v)]
            for c, v in zip(cfg_names[1:], vals[1:]):
                if v is None:
                    cells.append("-")
                elif base_v in (None, 0) or metric == "correct":
                    cells.append(fmt_val(metric, v))
                else:
                    delta = (v - base_v) / base_v * 100.0
                    cells.append(f"{fmt_val(metric, v)} ({delta:+.1f}%)")
            lines.append(f"| {metric} | " + " | ".join(cells) + " |")
        lines.append("")

    if results.notes:
        lines.append("## Notes\n")
        for n in results.notes:
            lines.append(f"- {n}")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", default=str(HERE / "configs.json"))
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--workdir",
                    default=os.environ.get("CRABBIT_HARNESS_WORKDIR",
                                           str(Path.home()
                                               / ".cache/crabbit-perf-harness")))
    ap.add_argument("--arrow-rs", default=DEFAULT_ARROW_RS)
    ap.add_argument("--fixtures", default="",
                    help="comma-separated fixture names (default: all)")
    ap.add_argument("--arrow-crates", default="",
                    help="comma-separated arrow crate names (default: probe "
                         + ",".join(ARROW_CANDIDATES) + ")")
    ap.add_argument("--skip-fixtures", action="store_true")
    ap.add_argument("--corpus", default=str(Path.home()
                                            / "projects/montaglue/kernel-corpus"),
                    help="kernel-corpus checkout; its kernels/*/cpu variants "
                         "are a third corpus (skip with --skip-corpus)")
    ap.add_argument("--kernels", default="",
                    help="comma-separated corpus kernel names (default: all)")
    ap.add_argument("--skip-corpus", action="store_true")
    ap.add_argument("--pin-cpu", default="",
                    help="taskset CPU id for every timed run (e.g. 7 = a "
                         "Cortex-X925 core on the DGX Spark)")
    ap.add_argument("--skip-arrow", action="store_true")
    ap.add_argument("--skip-backend-build", action="store_true")
    ap.add_argument("--backend", default="",
                    help="use this libcrabbit.so instead of building/locating "
                         "target/debug/libcrabbit.so")
    ap.add_argument("--no-hyperfine", action="store_true")
    ap.add_argument("--no-probe", action="store_true",
                    help="trust --arrow-crates without a warmup probe compile")
    ap.add_argument("--outdir", default="",
                    help="results dir to create or APPEND to (default: "
                         "results/<timestamp>); rows checkpoint incrementally")
    args = ap.parse_args()

    if args.pin_cpu:
        if shutil.which("taskset") is None:
            sys.exit("--pin-cpu needs taskset")
        PIN_PREFIX[:] = ["taskset", "-c", str(args.pin_cpu)]
    configs = json.loads(Path(args.configs).read_text())
    assert configs and all("name" in c and "env" in c for c in configs), \
        "configs.json must be a list of {name, env} objects"
    log("config matrix: " + ", ".join(c["name"] for c in configs))
    log(f"note: config '{configs[0]['name']}' is the baseline "
        "(first entry in the matrix)")

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    if args.backend:
        backend = Path(args.backend).resolve()
        if not backend.exists():
            sys.exit(f"--backend dylib not found at {backend}")
    elif args.skip_backend_build:
        backend = REPO / "target" / "debug" / "libcrabbit.so"
        if not backend.exists():
            sys.exit(f"--skip-backend-build but no dylib at {backend}")
    else:
        backend = build_backend()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    outdir = (Path(args.outdir) if args.outdir else HERE / "results" / stamp).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Snapshot the backend into the results dir so a `cargo build` during a
    # long sweep cannot change the compiler under the run (every row of a
    # sweep must come from one backend build). The sha256 goes in meta.json.
    snapshot = outdir / "libcrabbit.so"
    if not snapshot.exists():
        shutil.copy2(backend, snapshot)
    backend_sha = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    backend = snapshot
    log(f"backend: {backend} (sha256 {backend_sha[:16]})")

    use_hyperfine = shutil.which("hyperfine") is not None \
        and not args.no_hyperfine
    log(f"hyperfine: {'available' if use_hyperfine else 'not found, using python timer'}")

    results = Results(csv_path=outdir / "results.csv")

    fixture_names = [f for f in args.fixtures.split(",") if f]
    arrow_names = [c for c in args.arrow_crates.split(",") if c]

    if not args.skip_fixtures:
        run_fixture_phase(configs, backend, workdir, args.runs, results,
                          use_hyperfine, fixture_names)
    if not args.skip_corpus:
        targets = corpus_targets(args.corpus,
                                 [k for k in args.kernels.split(",") if k])
        if targets:
            log(f"corpus CPU variants: {', '.join(t.name for t in targets)}")
            run_fixture_phase(configs, backend, workdir, args.runs, results,
                              use_hyperfine, [], fixtures=targets,
                              subdir="corpus")
        else:
            results.notes.append(f"no corpus CPU variants found under {args.corpus}")
            log(results.notes[-1])
    if not args.skip_arrow:
        run_arrow_phase(configs, backend, args.arrow_rs, workdir, results,
                        arrow_names, probe=not args.no_probe)

    summary = render_summary(configs, results, args.runs)
    (outdir / "summary.md").write_text(summary)
    (outdir / "meta.json").write_text(json.dumps({
        "timestamp": stamp, "toolchain": TOOLCHAIN, "runs": args.runs,
        "configs": configs, "workdir": str(workdir),
        "arrow_rs": args.arrow_rs, "hyperfine": use_hyperfine,
        "backend_sha256": backend_sha, "pin_cpu": args.pin_cpu,
        "corpus": args.corpus,
        "miscompiles": len(results.miscompiles),
    }, indent=2))
    print()
    print(summary)
    log(f"results written to {outdir}")
    if results.miscompiles:
        log(f"{len(results.miscompiles)} MISCOMPILE/FAILURE entries -- see above")
        sys.exit(2)


if __name__ == "__main__":
    main()
