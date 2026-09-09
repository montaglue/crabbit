# crabbit-serverd: resident compilation service

User direction (2026-08-31): flip the architecture — instead of dumping
traces for later viewing, make crabbit a server that owns compilation.
Progress and resource control become server features; inspection reads
live in-process state ("sending fee" gone); and everything downstream of
the front half becomes re-runnable without re-invoking rustc.

## Why (what it collapses)

- Sweeps: the harness re-runs rustc per config although all configs share
  import+mid-end. Server: load module once, run the machine pipeline per
  config. 
- Profile feedback: frequencies are consumed at the RA position only —
  profile iteration = re-run RA→encode, seconds not minutes.
- Inspection: per-pass IR served from memory (replay or snapshot), diffs
  node-accurate, progress/pause/cancel/parallelism owned by the server.

## Architecture

rustc + `-Zcodegen-backend` stays the front door for real builds.
- crabbit gains `CRABBIT_EMIT_IR=<dir>` writing, per CGU, the module as
  printed IR immediately after LowerDialectMir (pure llvm dialect, so the
  server never needs the mir dialect or rustc), plus the kernel module.
- New crate `crates/serverd` (bin `crabbit-serverd`) — depends on
  pliron-ll, eregalloc-passes, cmt-spectral-cfg; NOT on `crabbit` (that
  crate is rustc_private-tainted). The engine/config selection code moves
  from `crabbit::regalloc_engine` into a new tiny shared crate
  `crates/research-config` (no rustc deps) consumed by both, so server
  and backend can never drift on flag semantics. Mid-end pass list: same
  move or re-export from pliron-ll.
- API (HTTP+JSON, minimal deps — tiny_http or axum, agent's choice):
  POST /modules (.plir text) → id (parsed+verified);
  POST /runs {module, target-triple, config axes (regalloc/oracle/freq/
    midend-disable/profile-json), inspect: bool} → run id; runs execute
    on a bounded worker pool (resource control), pass-by-pass loop with
    per-pass progress events;
  GET /runs/:id → status/progress (pass i/N, per-pass wall time);
  GET /runs/:id/ir/:pass → IR text at that point (from per-pass capture
    when inspect=true, else deterministic REPLAY of the first k passes);
  GET /runs/:id/artifact → object bytes (and .ptx/.blockmap.json);
  GET /modules/:id/diff?run_a,run_b,pass → later.
- Versioning strategy v1: replay-on-demand + optional in-memory per-pass
  text capture. v2: Context::clone_ir via a small upstream pliron PR
  (Clone on Operation/BasicBlock/Region); v3: the COW journal
  (docs earlier discussion) if memory demands it.

## Gates

- Round-trip fidelity test: every backend-tests fixture's emitted IR
  (CRABBIT_EMIT_IR) parses back and re-prints identically (print→parse→
  print fixpoint); any op/attr that fails gets its Parsable fixed.
- Pipeline equivalence: for a fixture CGU, the server's object bytes ==
  the rustc-path object bytes for the same config (baseline AND
  eregalloc-c2+spectral).
- Progress/cancel: a run on arrow-schema's largest CGU reports per-pass
  progress and can be cancelled mid-pipeline.
- No commits. Server is Linux/localhost, no auth, binds 127.0.0.1 only.
