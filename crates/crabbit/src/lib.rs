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

pub mod importer;

use rustc_codegen_ssa::traits::CodegenBackend;
use rustc_codegen_ssa::{CompiledModule, CompiledModules, CrateInfo};
use rustc_data_structures::fx::FxIndexMap;
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_session::config::{OutputFilenames, OutputType};
use pliron::opts::mem2reg::Mem2RegPass;

use crate::{
    conversion::pass::{AnalysisManager, PMConfig, Pass, Passes},
    passes::llvm::inline::LLVMInlinePass,
    passes::llvm::simplify::LLVMSimplifyPass,
    passes::llvm::simplify_cfg::LLVMSimplifyCfgPass,
    passes::llvm::sroa::LLVMSroaPass,
    passes::{aarch64_darwin, convert_mir_to_llvm::convert_mir_to_llvm_pass, x86_64_darwin},
    printable::Printable,
    trace::{StairTraceFile, StairTraceMeta},
};
use std::{
    any::Any,
    collections::BTreeMap,
    time::{SystemTime, UNIX_EPOCH},
};

struct StairBackend;

impl CodegenBackend for StairBackend {
    fn name(&self) -> &'static str {
        "stair"
    }

    fn target_cpu(&self, sess: &Session) -> String {
        sess.target.cpu.as_ref().to_owned()
    }

    fn codegen_crate<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Box<dyn Any> {
        Box::new(importer::import_crate(tcx))
    }

    fn join_codegen(
        &self,
        ongoing_codegen: Box<dyn Any>,
        sess: &Session,
        outputs: &OutputFilenames,
        _crate_info: &CrateInfo,
    ) -> (CompiledModules, FxIndexMap<WorkProductId, WorkProduct>) {
        let mut imported = ongoing_codegen
            .downcast::<importer::ImportedCrate>()
            .expect("stair-rust backend received unexpected codegen payload");

        if !imported.unsupported.is_empty() {
            let details = imported
                .unsupported
                .iter()
                .map(|error| format!("{}: {}", error.item, error.reason))
                .collect::<Vec<_>>()
                .join("\n");
            sess.dcx().fatal(format!(
                "stair-rust cannot import the requested MIR subset:\n{details}"
            ));
        }

        let object = emit_object(sess, outputs, &mut imported).unwrap_or_else(|error| {
            sess.dcx()
                .fatal(format!("stair-rust codegen failed: {error}"))
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

/// The Darwin object backend selected by the session's target architecture.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ObjectTarget {
    Aarch64Darwin,
    X86_64Darwin,
}

impl ObjectTarget {
    fn for_session(sess: &Session) -> Result<Self, String> {
        if sess.target.os.to_string() != "macos" {
            return Err(format!(
                "the STAIR object backend only supports macOS targets, got `{}`",
                sess.target.llvm_target
            ));
        }
        match sess.target.arch.to_string().as_str() {
            "aarch64" => Ok(Self::Aarch64Darwin),
            "x86_64" => Ok(Self::X86_64Darwin),
            other => Err(format!(
                "the STAIR object backend does not support the `{other}` architecture"
            )),
        }
    }

    fn pipeline(self) -> Passes {
        match self {
            Self::Aarch64Darwin => aarch64_darwin::pipeline(),
            Self::X86_64Darwin => x86_64_darwin::pipeline(),
        }
    }

    fn write_object(
        self,
        ctx: &mut crate::context::Context,
        root: crate::context::Ptr<crate::ir::operation::Operation>,
    ) -> crate::result::STAIRResult<Vec<u8>> {
        match self {
            Self::Aarch64Darwin => aarch64_darwin::write_macho_object_from_ir(ctx, root),
            Self::X86_64Darwin => x86_64_darwin::write_macho_object_from_ir(ctx, root),
        }
    }
}

/// The full MIR-to-machine-code pipeline for `target`, as pliron [Passes].
/// The CFG stays in pliron's block-argument form throughout; pliron's own
/// [Mem2RegPass] promotes the importer's alloca-per-local pattern to SSA
/// values directly in that form.
fn pipeline(target: ObjectTarget) -> Passes {
    let mut passes = Passes::default();
    passes.add_pass(convert_mir_to_llvm_pass());
    // Inline the module-internal call graph, then fold/clean and merge the
    // inlined blocks. simplify runs again after the CFG cleanup because
    // merging blocks turns cross-block load/store chains into block-local
    // ones.
    passes.add_pass(LLVMInlinePass::default());
    passes.add_pass(LLVMSimplifyPass);
    passes.add_pass(LLVMSimplifyCfgPass);
    // Split small struct allocas into scalars so mem2reg can promote the
    // pieces, then promote and clean up.
    passes.add_pass(LLVMSroaPass);
    passes.add_pass(LLVMSimplifyPass);
    passes.add_pass(Mem2RegPass);
    passes.add_pass(LLVMSimplifyPass);
    passes.add_pass(LLVMSimplifyCfgPass);
    passes.add_pass(LLVMSimplifyPass);
    // Second round: promoting pointer slots exposes direct accesses to
    // aggregates that were previously reached through those pointers
    // (e.g. a loop iterator updated via `&mut`), so split and promote
    // once more.
    passes.add_pass(LLVMSroaPass);
    passes.add_pass(LLVMSimplifyPass);
    passes.add_pass(Mem2RegPass);
    passes.add_pass(LLVMSimplifyPass);
    passes.add_pass(LLVMSimplifyCfgPass);
    passes.add_pass(LLVMSimplifyPass);
    passes.add_pass(target.pipeline());
    passes
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
    let target = ObjectTarget::for_session(sess)?;
    if imported.kernel_count > 0 {
        return Err("Darwin STAIR object emission does not support kernels yet".to_string());
    }

    let project = trace_project(sess);
    let version = trace_version();

    // Per-pass IR dumps come from pliron's own PMConfig printing hooks; the
    // trace file is assembled from the dumped files after the run, so a
    // failed pipeline still leaves a trace up to the failing pass.
    let dump_dir = std::env::temp_dir().join(format!("stair-pass-dumps-{version}"));
    std::fs::create_dir_all(&dump_dir)
        .map_err(|error| format!("failed to create pass dump directory: {error}"))?;
    let mut analyses = AnalysisManager::default();
    let mut config = PMConfig::default();
    config.print_after_all = true;
    config.ir_printing_dir = Some(dump_dir.clone());
    analyses.set_config(config);

    let initial_dump = imported.module.disp(&imported.ctx).to_string();
    let run_result = pipeline(target).run(imported.module, &mut imported.ctx, &mut analyses);

    let dumps = collect_pass_dumps(&dump_dir);
    let _ = std::fs::remove_dir_all(&dump_dir);

    let mut trace = StairTraceFile::new(StairTraceMeta {
        name: project.clone(),
        kind: "compiler-run".to_string(),
        entry: None,
        source: None,
        pipeline: dumps.iter().map(|(name, _)| name.clone()).collect(),
        target: Some(sess.target.llvm_target.to_string()),
        note: Some(format!("version {version}")),
        extra: BTreeMap::new(),
    });
    trace.push_dump("initial", initial_dump);
    for (name, dump) in dumps {
        trace.push_dump(name, dump);
    }
    let trace_path = trace::project_trace_path(&project, &version);
    if let Err(error) = run_result {
        let _ = trace.write(&trace_path);
        return Err(error.to_string());
    }
    trace
        .write(&trace_path)
        .map_err(|error| error.to_string())?;

    let object = outputs.temp_path_for_cgu(OutputType::Object, "stair_rust");
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

pub use crabbit_mir::mir;

pub mod passes {
    pub use crabbit_mir::passes::convert_mir_to_llvm;
    pub use pliron_ll::passes::{
        aarch64_darwin, dominance_frontier, hot_path, llvm, verify, x86_64_darwin,
    };
}

pub use pliron_ll::conversion;

pub mod dialects {
    pub use crate::mir;
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
    dict_key, impl_verify_succ, indented_block, input_err, input_err_noloc,
    input_error, input_error_noloc, type_to_trait, verify_err, verify_err_noloc,
    verify_error, verify_error_noloc,
};
