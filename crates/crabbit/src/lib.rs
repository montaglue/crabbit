#![feature(rustc_private)]

extern crate rustc_abi;
extern crate rustc_codegen_ssa;
extern crate rustc_data_structures;
extern crate rustc_hir;
extern crate rustc_middle;
extern crate rustc_session;
extern crate rustc_span;
extern crate rustc_symbol_mangling;

#[allow(unused_extern_crates)]
extern crate rustc_driver;

// The MIR importer targeting cuda-oxide's dialect-mir. The legacy cmir
// importer and dialect are retired; their sources are detached from the
// build pending deletion.
#[path = "importer_oxide.rs"]
pub mod importer;
pub mod regalloc_engine;
pub mod kernel_llvm_export;

use rustc_codegen_ssa::target_features::cfg_target_feature;
use rustc_codegen_ssa::traits::CodegenBackend;
use rustc_codegen_ssa::{CompiledModule, CompiledModules, CrateInfo, TargetConfig};
use rustc_data_structures::fx::FxIndexMap;
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_session::config::{OutputFilenames, OutputType};


use crate::{
    conversion::pass::{AnalysisManager, PMConfig, Passes},
    printable::Printable,
    trace::{StairTraceFile, StairTraceMeta},
};
use pliron_ll::{
    targets::{self, TargetBackend},
    triple::Triple,
};
use std::{
    any::Any,
    collections::BTreeMap,
    time::{SystemTime, UNIX_EPOCH},
};

struct StairBackend;

impl CodegenBackend for StairBackend {
    fn name(&self) -> &'static str {
        "crabbit"
    }

    fn target_cpu(&self, sess: &Session) -> String {
        sess.target.cpu.as_ref().to_owned()
    }

    /// Report the ABI-mandated baseline features (e.g. `neon` on AArch64) as
    /// enabled. The backend generates code under the target's default ABI and
    /// never disables a baseline feature; without this, rustc cannot see that
    /// the ABI-required features are on and warns on every compile that
    /// `neon` "must be enabled to ensure that the ABI of the current target
    /// can be implemented correctly".
    fn target_config(&self, sess: &Session) -> TargetConfig {
        let abi_required = sess.target.abi_required_features();
        let (target_features, unstable_target_features) = cfg_target_feature::<0>(
            sess,
            |_feature| Default::default(),
            |feature| abi_required.required.contains(&feature),
        );
        TargetConfig {
            target_features,
            unstable_target_features,
            // The backend has no verified f16/f128 support; report them
            // unreliable so `cfg(target_has_reliable_f16)`-style gates stay
            // honest for code compiled through it.
            has_reliable_f16: false,
            has_reliable_f16_math: false,
            has_reliable_f128: false,
            has_reliable_f128_math: false,
        }
    }

    fn codegen_crate<'tcx>(&self, tcx: TyCtxt<'tcx>, _crate_info: &CrateInfo) -> Box<dyn Any> {
        // The importer names symbols via `def_path_str`, which outside this
        // guard consults the diagnostic path-trimming machinery: names could
        // come out trimmed, and computing `trimmed_def_paths` on a
        // warning-free compile is an ICE ("diagnostics were expected but
        // none were emitted"). Force full, stable paths instead.
        rustc_middle::ty::print::with_no_trimmed_paths!(Box::new(importer::import_crate(tcx)))
    }

    fn join_codegen(
        &self,
        ongoing_codegen: Box<dyn Any>,
        sess: &Session,
        outputs: &OutputFilenames,
    ) -> (CompiledModules, FxIndexMap<WorkProductId, WorkProduct>) {
        let mut imported = ongoing_codegen
            .downcast::<importer::ImportedCrate>()
            .expect("crabbit backend received unexpected codegen payload");

        if !imported.unsupported.is_empty() {
            let details = imported
                .unsupported
                .iter()
                .map(|error| format!("{}: {}", error.item, error.reason))
                .collect::<Vec<_>>()
                .join("\n");
            sess.dcx().fatal(format!(
                "crabbit cannot import the requested MIR subset:\n{details}"
            ));
        }

        let object = emit_object(sess, outputs, &mut imported).unwrap_or_else(|error| {
            sess.dcx()
                .fatal(format!("crabbit codegen failed: {error}"))
        });

        let module = CompiledModule {
            name: "stair_rust".to_string(),
            kind: rustc_codegen_ssa::ModuleKind::Regular,
            object: Some(object),
            dwarf_object: None,
            bytecode: None,
            assembly: None,
            llvm_ir: None,
            links_from_incr_cache: Vec::new(),
        };

        (
            CompiledModules {
                modules: vec![module],
                allocator_module: None,
            },
            FxIndexMap::default(),
        )
    }
}

/// Resolves the session's LLVM target triple against pliron-ll's backend
/// registry, the way LLVM's `TargetRegistry::lookupTarget` resolves a
/// `Target` from a triple.
fn backend_for_session(sess: &Session) -> Result<&'static TargetBackend, String> {
    let triple = Triple::parse(&sess.target.llvm_target);
    targets::lookup(&triple).ok_or_else(|| {
        format!(
            "no crabbit object backend is registered for target `{}` (parsed as `{triple}`); \
             registered backends: {}",
            sess.target.llvm_target,
            targets::registered_names().collect::<Vec<_>>().join(", ")
        )
    })
}

/// The full MIR-to-machine-code pipeline for `target`, as pliron [Passes].
/// The CFG stays in pliron's block-argument form throughout; pliron's own
/// [Mem2RegPass] promotes the importer's alloca-per-local pattern to SSA
/// values directly in that form.
fn pipeline(target: &TargetBackend, already_lowered: bool) -> Result<Passes, String> {
    let mut passes = Passes::default();
    if already_lowered {
        // CRABBIT_EMIT_IR ran lower-dialect-mir standalone (to print the
        // module in its pure-LLVM-dialect form); continue from there.
        pliron_ll::passes::llvm::add_llvm_midend_passes(
            &mut passes,
            &pliron_ll::target_profile::TargetProfile::host_cpu(),
        );
    } else {
        add_midend_passes(&mut passes, &pliron_ll::target_profile::TargetProfile::host_cpu());
    }
    // The machine pipeline, with the register allocator swapped for the
    // engine chosen by CRABBIT_REGALLOC (see [regalloc_engine]; the
    // engine→allocator mapping lives in research-config so its cargo
    // features gate it in one place).
    let engine = regalloc_engine::RegallocEngine::from_env()?;
    passes.add_pass(engine.machine_pipeline(target)?);
    Ok(passes)
}

/// The target-independent mid-end: `mir` → LLVM dialect, then inlining,
/// simplification, SROA and mem2reg (twice, see below). Shared by the host
/// pipeline and the kernel pipeline.
fn add_midend_passes(passes: &mut Passes, profile: &pliron_ll::target_profile::TargetProfile) {
    passes.add_pass(crabbit_mir::passes::lower_dialect_mir::LowerDialectMirPass);
    // The LLVM-dialect mid-end proper is shared with the resident
    // driver/server (which starts from post-lowering IR), so its pass list
    // lives in pliron-ll and cannot drift between the two entry points.
    pliron_ll::passes::llvm::add_llvm_midend_passes(passes, profile);
}

/// The kernel (`rust_kernels` module) pipeline: the mid-end only. PTX has
/// virtual registers and ptxas does the machine work, so translation to
/// PTX text ([pliron_ll::nvptx::write_ptx_from_ir]) happens outside the
/// pass pipeline, like the object writers.
fn kernel_pipeline(already_lowered: bool) -> Passes {
    let mut passes = Passes::default();
    let profile = pliron_ll::target_profile::TargetProfile::gpu_kernel();
    if already_lowered {
        pliron_ll::passes::llvm::add_llvm_midend_passes(&mut passes, &profile);
    } else {
        add_midend_passes(&mut passes, &profile);
    }
    passes
}

/// `CRABBIT_EMIT_IR=<dir>`: emit the module as printed IR immediately
/// after `lower-dialect-mir` (pure LLVM dialect — parseable without the
/// mir dialect or rustc), for the resident driver/server. Runs the lower
/// pass standalone on `module` and writes `<dir>/<stem>.plir`; the caller
/// must then continue with an `already_lowered` pipeline.
fn emit_lowered_ir(
    ctx: &mut pliron::context::Context,
    module: pliron::context::Ptr<pliron::operation::Operation>,
    dir: &std::path::Path,
    stem: &str,
) -> Result<(), String> {
    let mut lower = Passes::default();
    lower.add_pass(crabbit_mir::passes::lower_dialect_mir::LowerDialectMirPass);
    lower
        .run(module, ctx, &mut AnalysisManager::default())
        .map_err(|error| format!("lower-dialect-mir (CRABBIT_EMIT_IR) failed: {error}"))?;
    std::fs::create_dir_all(dir)
        .map_err(|error| format!("failed to create CRABBIT_EMIT_IR dir: {error}"))?;
    let path = dir.join(format!("{stem}.plir"));
    std::fs::write(&path, module.disp(ctx).to_string())
        .map_err(|error| format!("failed to write `{}`: {error}", path.display()))
}

fn emit_ir_dir() -> Option<std::path::PathBuf> {
    std::env::var("CRABBIT_EMIT_IR")
        .ok()
        .filter(|v| !v.is_empty())
        .map(std::path::PathBuf::from)
}

/// Lower the imported `rust_kernels` module and write its PTX next to
/// `object` (`<stem>.ptx`) and to `CRABBIT_PTX_OUT` when set. See
/// docs/KERNEL-ABI.md.
fn emit_kernels(
    imported: &mut importer::ImportedCrate,
    object: &std::path::Path,
    tracing: bool,
    ir_stem: &str,
) -> Result<std::path::PathBuf, String> {
    let mut analyses = AnalysisManager::default();
    let emit_dir = emit_ir_dir();
    if let Some(dir) = &emit_dir {
        emit_lowered_ir(
            &mut imported.ctx,
            imported.kernel_module,
            dir,
            &format!("{ir_stem}-kernels"),
        )?;
    }
    let mut pipeline = kernel_pipeline(emit_dir.is_some());
    let mut dump_dir = None;
    if tracing {
        let dir = std::env::temp_dir().join(format!("stair-kernel-pass-dumps-{}", trace_version()));
        std::fs::create_dir_all(&dir)
            .map_err(|error| format!("failed to create kernel pass dump directory: {error}"))?;
        let mut config = PMConfig::default();
        config.print_after_all = true;
        config.ir_printing_dir = Some(dir.clone());
        pipeline.set_config(config);
        dump_dir = Some(dir);
    }
    let run_result = pipeline.run(imported.kernel_module, &mut imported.ctx, &mut analyses);
    if let Some(dir) = dump_dir {
        // Keep the kernel dumps where CRABBIT_TRACE users can find them;
        // they are small (kernel modules are) and named after the pass.
        let keep = object.with_extension("kernel-trace");
        let _ = std::fs::remove_dir_all(&keep);
        let _ = std::fs::rename(&dir, &keep).or_else(|_| {
            std::fs::create_dir_all(&keep).and_then(|_| {
                for (name, dump) in collect_pass_dumps(&dir) {
                    std::fs::write(keep.join(format!("{name}.plir")), dump)?;
                }
                std::fs::remove_dir_all(&dir)
            })
        });
    }
    run_result.map_err(|error| format!("kernel pipeline failed: {error}"))?;

    let mut target = pliron_ll::nvptx::PtxTarget::default();
    if let Ok(sm) = std::env::var("CRABBIT_PTX_SM")
        && !sm.is_empty()
    {
        target.sm = sm
            .parse()
            .map_err(|_| format!("CRABBIT_PTX_SM must be an integer SM number, got `{sm}`"))?;
    }
    let (ptx, linemap) =
        pliron_ll::nvptx::write_ptx_and_linemap_from_ir(&imported.ctx, imported.kernel_module, &target)
            .map_err(|error| format!("NVPTX emission failed: {error}"))?;

    let sidecar = object.with_extension("ptx");
    std::fs::write(&sidecar, &ptx).map_err(|error| {
        format!("failed to write PTX sidecar `{}`: {error}", sidecar.display())
    })?;
    // The GPU analogue of the machine op-map: `<ptx>.linemap.json` next to
    // every PTX we write (docs/PROFILE-FEEDBACK-BACKWARD.md, ncu leg).
    let write_linemap = |ptx_path: &std::path::Path| -> Result<(), String> {
        if let Some(json) = &linemap {
            let path = ptx_path.with_extension("ptx.linemap.json");
            std::fs::write(&path, json).map_err(|error| {
                format!("failed to write PTX linemap `{}`: {error}", path.display())
            })?;
        }
        Ok(())
    };
    write_linemap(&sidecar)?;
    if let Ok(out) = std::env::var("CRABBIT_PTX_OUT")
        && !out.is_empty()
    {
        std::fs::write(&out, &ptx)
            .map_err(|error| format!("failed to write CRABBIT_PTX_OUT `{out}`: {error}"))?;
        write_linemap(std::path::Path::new(&out))?;
    }
    if let Ok(out) = std::env::var("CRABBIT_LL_OUT")
        && !out.is_empty()
    {
        let ll = kernel_llvm_export::export_kernel_module(&mut imported.ctx, imported.kernel_module)
            .map_err(|error| format!("LLVM IR export of the kernel module failed: {error}"))?;
        std::fs::write(&out, ll)
            .map_err(|error| format!("failed to write CRABBIT_LL_OUT `{out}`: {error}"))?;
    }
    Ok(sidecar)
}

/// The `(pass name, IR dump)` pairs pliron's `print_after_all` hook wrote
/// into `dir` (as `{count}-after-{name}.plir`), in execution order.
fn collect_pass_dumps(dir: &std::path::Path) -> Vec<(String, String)> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut dumps: Vec<(usize, String, String)> = entries
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let stem = path.file_stem()?.to_str()?;
            let (count, name) = stem.split_once("-after-")?;
            let count: usize = count.parse().ok()?;
            let contents = std::fs::read_to_string(&path).ok()?;
            Some((count, name.to_string(), contents))
        })
        .collect();
    dumps.sort_by_key(|(count, ..)| *count);
    dumps
        .into_iter()
        .map(|(_, name, dump)| (name, dump))
        .collect()
}

fn emit_object(
    sess: &Session,
    outputs: &OutputFilenames,
    imported: &mut importer::ImportedCrate,
) -> Result<std::path::PathBuf, String> {
    let target = backend_for_session(sess)?;

    // Per-pass IR dumps and the trace file cost O(passes × module text) in
    // formatting and I/O — gigabytes on a large crate — so tracing is
    // opt-in via CRABBIT_TRACE.
    let tracing = std::env::var("CRABBIT_TRACE").is_ok_and(|value| !value.is_empty() && value != "0");

    let mut analyses = AnalysisManager::default();
    // The trace's "initial" entry must be captured BEFORE any lowering:
    // with CRABBIT_EMIT_IR set, `emit_lowered_ir` runs lower-dialect-mir
    // in place, and capturing afterwards would present post-lowering IR
    // as "initial" while the recorded pipeline omitted the pass that ran.
    let mut initial_dump = if tracing {
        Some(imported.module.disp(&imported.ctx).to_string())
    } else {
        None
    };
    let mut emitted_lowering_dump = None;
    let emit_dir = emit_ir_dir();
    if let Some(dir) = &emit_dir {
        emit_lowered_ir(
            &mut imported.ctx,
            imported.module,
            dir,
            &format!("{}-stair_rust", trace_project(sess)),
        )?;
        if tracing {
            // Record the stage the emit path ran outside the pipeline, so
            // the trace still shows every pass that touched the module.
            emitted_lowering_dump =
                Some(imported.module.disp(&imported.ctx).to_string());
        }
    }
    let mut pipeline = pipeline(target, emit_dir.is_some())?;
    let mut dump_dir = None;
    let version = trace_version();
    if tracing {
        // Per-pass IR dumps come from pliron's own PMConfig printing hooks;
        // the trace file is assembled from the dumped files after the run,
        // so a failed pipeline still leaves a trace up to the failing pass.
        let dir = std::env::temp_dir().join(format!("stair-pass-dumps-{version}"));
        std::fs::create_dir_all(&dir)
            .map_err(|error| format!("failed to create pass dump directory: {error}"))?;
        let mut config = PMConfig::default();
        config.print_after_all = true;
        config.ir_printing_dir = Some(dir.clone());
        pipeline.set_config(config);
        dump_dir = Some(dir);
    } else {
        initial_dump = None;
    }

    let run_result = pipeline.run(imported.module, &mut imported.ctx, &mut analyses);

    if let Some(dump_dir) = dump_dir {
        let project = trace_project(sess);
        let dumps = collect_pass_dumps(&dump_dir);
        let _ = std::fs::remove_dir_all(&dump_dir);

        let mut trace = StairTraceFile::new(StairTraceMeta {
            name: project.clone(),
            kind: "compiler-run".to_string(),
            entry: None,
            source: None,
            pipeline: emitted_lowering_dump
                .iter()
                .map(|_| "lower-dialect-mir(emit-ir)".to_string())
                .chain(dumps.iter().map(|(name, _)| name.clone()))
                .collect(),
            target: Some(sess.target.llvm_target.to_string()),
            note: Some(format!("version {version}")),
            extra: BTreeMap::new(),
        });
        trace.push_dump("initial", initial_dump.take().unwrap_or_default());
        if let Some(dump) = emitted_lowering_dump.take() {
            trace.push_dump("lower-dialect-mir(emit-ir)", dump);
        }
        for (name, dump) in dumps {
            trace.push_dump(name, dump);
        }
        let trace_path = trace::project_trace_path(&project, &version);
        if let Err(error) = &run_result {
            let _ = trace.write(&trace_path);
            return Err(error.to_string());
        }
        trace
            .write(&trace_path)
            .map_err(|error| error.to_string())?;
    }
    if let Err(error) = run_result {
        return Err(error.to_string());
    }

    // No invocation-temp component: the object must outlive the rustc
    // invocation (backend-tests inspect it after the build).
    let object = outputs.temp_path_for_cgu(OutputType::Object, "stair_rust", None);
    if let Some(parent) = object.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create object output directory: {error}"))?;
    }
    let bytes = target
        .write_object(&mut imported.ctx, imported.module)
        .map_err(|error| error.to_string())?;
    std::fs::write(&object, bytes).map_err(|error| {
        format!(
            "failed to write object file `{}`: {error}",
            object.display()
        )
    })?;
    // CRABBIT_BLOCKMAP=1: the profile-feedback sidecar mapping final .text
    // byte ranges back to RA-position block ids, next to the object
    // (docs/PROFILE-FEEDBACK-PLAN.md). The ids were stamped by the
    // aarch64-blockmap-ids pass; a module without them writes nothing.
    if pliron_ll::passes::aarch64::blockmap::blockmap_enabled()
        && let Some(json) =
            pliron_ll::passes::aarch64::blockmap::blockmap_json_from_ir(&imported.ctx, imported.module)
    {
        let sidecar = object.with_extension("blockmap.json");
        std::fs::write(&sidecar, json).map_err(|error| {
            format!(
                "failed to write blockmap sidecar `{}`: {error}",
                sidecar.display()
            )
        })?;
    }
    if imported.kernel_count > 0 {
        emit_kernels(imported, &object, tracing, &format!("{}-stair_rust", trace_project(sess)))?;
    }
    Ok(object)
}

fn trace_project(sess: &Session) -> String {
    let crate_name = sess
        .opts
        .crate_name
        .as_ref()
        .map(ToString::to_string)
        .unwrap_or_else(|| "crate".to_string());
    sanitize_trace_name(&crate_name)
}

// Timestamp-first so lexicographic order of version filenames matches
// chronological order within a project folder.
fn trace_version() -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default();
    format!("{timestamp}-{}", std::process::id())
}

fn sanitize_trace_name(name: &str) -> String {
    let mut out = String::new();
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "crate".to_string()
    } else {
        out
    }
}

// SAFETY: rustc loads custom codegen backends by looking up this exact exported symbol.
#[unsafe(no_mangle)]
pub fn __rustc_codegen_backend() -> Box<dyn CodegenBackend> {
    Box::new(StairBackend)
}

pub mod trace;


pub mod passes {
    pub use crabbit_mir::passes::lower_dialect_mir;
    pub use pliron_ll::passes::{
        aarch64, aarch64_darwin, aarch64_linux, dominance_frontier, hot_path, llvm, verify,
        x86_64_darwin,
    };
}

pub use pliron_ll::conversion;

pub mod dialects {
    // cuda-oxide's MIR port, registered as dialect `mir`.
    pub use crabbit_mir::dialect_mir;
    pub use pliron_llvm as llvm;
    pub use pliron::builtin;
    pub use pliron_ll::{aarch64, ll, macho, x86_64};
}
// ---- compatibility re-exports over the pliron core (cleanup pending) ----
pub use pliron::{
    attribute, basic_block, builtin, common_traits, context, debug_info, dialect,
    graph, identifier, irbuild, irfmt, linked_list, location, op, operation, opts,
    parsable, printable, region, storage_uniquer, symbol_table, r#type,
    uniqued_any, utils, value,
};
pub mod result {
    pub use pliron::result::*;
    /// Old stair name for [Result].
    pub type STAIRResult<T> = pliron::result::Result<T>;
}
pub mod ir {
    pub use pliron::{
        attribute, basic_block, dialect, irfmt, location, op, operation, region, value,
    };
    pub use pliron::r#type;
}
pub use pliron::{
    arg_err, arg_err_noloc, arg_error, arg_error_noloc, create_err, create_error,
    dict_key, indented_block, input_err, input_err_noloc,
    input_error, input_error_noloc, type_to_trait, verify_err, verify_err_noloc,
    verify_error, verify_error_noloc,
};
