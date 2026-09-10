//! Library shared by the crabbit tooling binaries (`crabbit-inspect-driver`
//! one-shot CLI and `crabbit-analysisd` resident analysis server), linking
//! the crabbit dialect stack:
//! mir (cuda-oxide's dialect-mir), llvm, ll, aarch64, x86_64, macho — plus
//! the mid-level pass pipeline.

use crabbit_mir::passes::lower_dialect_mir::LowerDialectMirPass;
use pliron::context::{Context, Ptr};
use pliron::operation::Operation;
use pliron_inspect_driver::DriverHooks;
use pliron_ll::conversion::pass::{AnalysisManager, Mem2RegPass, Pass};
use pliron_ll::passes::llvm::{
    inline::LLVMInlinePass, pin_type_punned_slots::LLVMPinTypePunnedSlotsPass,
    simplify::LLVMSimplifyPass, simplify_cfg::LLVMSimplifyCfgPass,
    sroa::LLVMSroaPass,
};
use pliron_ll::passes::verify::VerifyPass;

pub struct CrabbitHooks {
    // `Pass::run` takes `&mut self` (pliron 0.17) while `DriverHooks` hands
    // out `&self`; RefCell bridges the two.
    passes: Vec<std::cell::RefCell<Box<dyn Pass>>>,
}

impl CrabbitHooks {
    pub fn new() -> Self {
        let passes: Vec<Box<dyn Pass>> = vec![
            Box::new(VerifyPass::new()),
            Box::new(LowerDialectMirPass),
            Box::new(LLVMPinTypePunnedSlotsPass),
            Box::new(Mem2RegPass),
            Box::new(LLVMInlinePass::default()),
            Box::new(LLVMSimplifyPass),
            Box::new(LLVMSimplifyCfgPass),
            Box::new(LLVMSroaPass),
        ];
        CrabbitHooks {
            passes: passes.into_iter().map(std::cell::RefCell::new).collect(),
        }
    }
}

impl DriverHooks for CrabbitHooks {
    fn pass_names(&self) -> Vec<String> {
        self.passes
            .iter()
            .map(|p| p.borrow().name().to_string())
            .collect()
    }

    fn run_pass(
        &self,
        name: &str,
        root: Ptr<Operation>,
        ctx: &mut Context,
    ) -> Result<Ptr<Operation>, String> {
        let Some(pass) = self
            .passes
            .iter()
            .find(|p| p.borrow().name() == name)
        else {
            return Err(format!("unknown pass: {name}"));
        };
        let mut analyses = AnalysisManager::default();
        pass.borrow_mut()
            .run(root, ctx, &mut analyses)
            .map(|_| root)
            .map_err(|e| format!("{e}"))
    }
}

// ---------------------------------------------------------------------------
// Resident server mode (docs/SERVERD-PLAN.md): the same driver binary grown
// into the compilation service. `--serve` speaks the server extension of the
// stdio protocol; `--http 127.0.0.1:PORT` adds the HTTP shim.
// ---------------------------------------------------------------------------

/// Registry target names the server accepts, with the triple each one
/// resolves through `pliron_ll::targets::lookup`, plus the pseudo-target
/// for kernel modules (mid-end + PTX translation, no machine pipeline).
const SERVE_TARGETS: &[(&str, &str)] = &[
    ("aarch64-linux", "aarch64-unknown-linux-gnu"),
    ("aarch64-darwin", "aarch64-apple-darwin"),
    ("x86_64-darwin", "x86_64-apple-darwin"),
];
const NVPTX_TARGET: &str = "nvptx-kernel";

pub struct ServeHooks;

impl ServeHooks {
    fn backend(target: &str) -> Result<&'static pliron_ll::targets::TargetBackend, String> {
        let (_, triple) = SERVE_TARGETS
            .iter()
            .find(|(name, _)| *name == target)
            .ok_or_else(|| format!("unknown target `{target}`"))?;
        pliron_ll::targets::lookup(&pliron_ll::triple::Triple::parse(triple))
            .ok_or_else(|| format!("no backend registered for `{target}`"))
    }

    /// The full pipeline for `target` (flattened for per-pass reporting).
    /// Must be called inside `with_env_config` — the engine selection and
    /// several passes read `CRABBIT_*` at construction or run time.
    fn build_pipeline(target: &str) -> Result<pliron_ll::conversion::pass::Passes, String> {
        let mut passes = pliron_ll::conversion::pass::Passes::default();
        let profile = if target == NVPTX_TARGET {
            pliron_ll::target_profile::TargetProfile::gpu_kernel()
        } else {
            pliron_ll::target_profile::TargetProfile::host_cpu()
        };
        pliron_ll::passes::llvm::add_llvm_midend_passes(&mut passes, &profile);
        if target == NVPTX_TARGET {
            return Ok(passes);
        }
        let backend = Self::backend(target)?;
        let engine = research_config::RegallocEngine::from_env()?;
        passes.extend(engine.machine_pipeline(backend)?);
        Ok(passes)
    }
}

impl DriverHooks for ServeHooks {
    fn list_targets(&self) -> Vec<String> {
        SERVE_TARGETS
            .iter()
            .map(|(name, _)| name.to_string())
            .chain([NVPTX_TARGET.to_string()])
            .collect()
    }

    fn pipeline_pass_names(
        &self,
        target: &str,
        config: &std::collections::BTreeMap<String, String>,
    ) -> Result<Vec<String>, String> {
        research_config::with_env_config(config, || {
            Self::build_pipeline(target).map(|p| p.names())
        })?
    }

    fn attribution_ops(
        &self,
        module_text: &str,
        target: &str,
        config: &std::collections::BTreeMap<String, String>,
        boundary_pass: &str,
    ) -> Result<pliron_inspect_driver::AttributionOps, String> {
        // Force attribution stamping on for the replay: stamping only adds
        // attributes, so the transformations (and therefore the boundary
        // IR) match the original run exactly. Unconditional insert — a
        // stored config carrying an explicit "0" must not silently turn
        // the replay into an empty attribution table.
        let mut config = config.clone();
        config.insert("CRABBIT_PROFILE_MAP".to_string(), "1".to_string());
        research_config::with_env_config(&config, || {
            let hooks = analysis_hooks_factory()();
            let mut ctx = hooks.create_context();
            let root = pliron_inspect_driver::parse_ir(module_text, &mut ctx)
                .map_err(|e| format!("attribution parse failed: {e}"))?;
            let mut passes = Self::build_pipeline(target)?;
            let mut analyses = AnalysisManager::default();
            let mut hit = false;
            passes
                .run_observed(root, &mut ctx, &mut analyses, &mut |_, name, _, _| {
                    use pliron_ll::conversion::pass::PassControl;
                    if name == boundary_pass {
                        hit = true;
                        PassControl::Stop
                    } else {
                        PassControl::Continue
                    }
                })
                .map_err(|e| format!("attribution replay failed: {e}"))?;
            if !hit {
                return Err(format!(
                    "boundary pass `{boundary_pass}` is not in the `{target}` pipeline"
                ));
            }
            use pliron::printable::Printable;
            let state = pliron::printable::State::default();
            let ir = root.print(&ctx, &state).to_string();
            let ir_lines: Vec<&str> = ir.lines().collect();
            let mut cursor = 0usize;
            let mut functions = Vec::new();
            let module = pliron_ll::ir::operation::Operation::get_op_dyn(root, &ctx);
            let module = module
                .downcast_ref::<pliron_ll::dialects::builtin::ops::ModuleOp>()
                .ok_or_else(|| "attribution root is not a module".to_string())?;
            use pliron::builtin::op_interfaces::OneRegionInterface as _;
            let body = module
                .get_region(&ctx)
                .deref(&ctx)
                .get_head()
                .ok_or_else(|| "module has no body".to_string())?;
            use pliron_ll::linked_list::ContainsLinkedList as _;
            let funcs: Vec<_> = body.deref(&ctx).iter(&ctx).collect();
            for func_ptr in funcs {
                let func_obj =
                    pliron_ll::ir::operation::Operation::get_op_dyn(func_ptr, &ctx);
                let Some(func) = func_obj
                    .downcast_ref::<pliron_ll::dialects::llvm::ops::FuncOp>()
                else {
                    continue;
                };
                let Some(region) = pliron::builtin::op_interfaces::AtMostOneRegionInterface::get_region(func, &ctx) else {
                    continue;
                };
                let symbol = pliron::builtin::op_interfaces::SymbolOpInterface::get_symbol_name(
                    func, &ctx,
                )
                .to_string();
                drop(func_obj);
                let mut table = Vec::new();
                for block in region.deref(&ctx).iter(&ctx) {
                    for op in block.deref(&ctx).iter(&ctx) {
                        let Some(id) = pliron_ll::passes::aarch64::opmap::op_id(&ctx, op)
                        else {
                            continue;
                        };
                        let snippet = {
                            let state = pliron::printable::State::default();
                            let text = op.print(&ctx, &state).to_string();
                            let mut line = text.lines().next().unwrap_or("").trim().to_string();
                            line.truncate(160);
                            line
                        };
                        // Ops print in program order; search forward from
                        // the previous match for the snippet's line.
                        let line = ir_lines[cursor.min(ir_lines.len())..]
                            .iter()
                            .position(|candidate| candidate.trim() == snippet)
                            .map(|offset| {
                                let absolute = cursor + offset;
                                cursor = absolute + 1;
                                absolute as u32
                            });
                        table.push((id, line, snippet));
                    }
                }
                functions.push((symbol, table));
            }
            Ok(pliron_inspect_driver::AttributionOps { ir, functions })
        })?
    }

    fn run_pipeline(
        &self,
        ctx: &mut Context,
        root: Ptr<Operation>,
        target: &str,
        config: &std::collections::BTreeMap<String, String>,
        upto: Option<usize>,
        capture: bool,
        events: &mut dyn FnMut(pliron_inspect_driver::PipelineEvent) -> bool,
    ) -> Result<(), String> {
        research_config::with_env_config(config, || {
            let mut passes = Self::build_pipeline(target)?;
            let mut analyses = AnalysisManager::default();
            let mut last = std::time::Instant::now();
            passes
                .run_observed(root, ctx, &mut analyses, &mut |index, name, ctx, op| {
                    use pliron_ll::conversion::pass::PassControl;
                    let micros = last.elapsed().as_micros() as u64;
                    last = std::time::Instant::now();
                    let ir = capture.then(|| {
                        use pliron::printable::Printable;
                        let state = pliron::printable::State::default();
                        op.print(ctx, &state).to_string()
                    });
                    let keep_going = events(pliron_inspect_driver::PipelineEvent {
                        index,
                        name: name.to_string(),
                        micros,
                        ir,
                    });
                    let stop_after = upto.is_some_and(|u| index + 1 >= u);
                    if keep_going && !stop_after {
                        PassControl::Continue
                    } else {
                        PassControl::Stop
                    }
                })
                .map(|_| ())
                .map_err(|e| format!("{e}"))
        })?
    }

    fn write_artifact(
        &self,
        ctx: &mut Context,
        root: Ptr<Operation>,
        target: &str,
    ) -> Result<Vec<u8>, String> {
        if target == NVPTX_TARGET {
            let ptx = pliron_ll::nvptx::write_ptx_from_ir(
                ctx,
                root,
                &pliron_ll::nvptx::PtxTarget::default(),
            )
            .map_err(|e| format!("{e}"))?;
            return Ok(ptx.into_bytes());
        }
        let backend = Self::backend(target)?;
        backend
            .write_object(ctx, root)
            .map_err(|e| format!("{e}"))
    }

    fn artifact_sidecars(
        &self,
        ctx: &mut Context,
        root: Ptr<Operation>,
        target: &str,
        config: &std::collections::BTreeMap<String, String>,
    ) -> Vec<(String, Vec<u8>)> {
        if target == NVPTX_TARGET {
            // GPU leg of the backward-attribution design: the PTX linemap
            // sidecar, when the run's config asked for attribution.
            let enabled = ["CRABBIT_PTX_LINEMAP", "CRABBIT_PROFILE_MAP", "CRABBIT_BLOCKMAP"]
                .iter()
                .any(|k| config.get(*k).is_some_and(|v| !v.is_empty() && v != "0"));
            if !enabled {
                return vec![];
            }
            return match pliron_ll::nvptx::write_ptx_with_forced_linemap(
                ctx,
                root,
                &pliron_ll::nvptx::PtxTarget::default(),
            ) {
                Ok((_, json)) => {
                    vec![("ptx.linemap.json".to_string(), json.into_bytes())]
                }
                Err(_) => vec![],
            };
        }
        // The blockmap sidecar exists when the run's config enabled it
        // (the ids were stamped during the pipeline, under the same env).
        // Mirror `blockmap_enabled()` exactly: either variable, non-empty,
        // not "0" — the stamping and the sidecar must never diverge.
        let set = |var: &str| {
            config.get(var).is_some_and(|v| !v.is_empty() && v != "0")
        };
        let enabled = set("CRABBIT_BLOCKMAP") || set("CRABBIT_PROFILE_MAP");
        if !enabled {
            return vec![];
        }
        match pliron_ll::passes::aarch64::blockmap::blockmap_json_from_ir(ctx, root) {
            Some(json) => vec![("blockmap.json".to_string(), json.into_bytes())],
            None => vec![],
        }
    }
}


/// Hooks factory for the resident analysis server.
pub fn analysis_hooks_factory() -> pliron_inspect_driver::HooksFactory {
    std::sync::Arc::new(|| Box::new(ServeHooks))
}
