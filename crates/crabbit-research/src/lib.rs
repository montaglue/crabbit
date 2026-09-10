//! The research composition of crabbit: the public backend crate linked
//! together with the private research engines, exporting the same
//! `__rustc_codegen_backend` entry rustc looks for. Point
//! `-Zcodegen-backend` at `libcrabbit_research.so` and every
//! `CRABBIT_REGALLOC=eregalloc` configuration works; the public
//! `libcrabbit.so` reports such configs as not linked.
//!
//! Engine construction lives here (not in `research-config`) because the
//! engines are private sibling checkouts the public workspace must never
//! resolve — see research-config's crate docs for the mechanics. The
//! environment-variable semantics are unchanged from the integration
//! contract (eregalloc/INTEGRATION-CONTRACT.md):
//!
//! | variable | values |
//! |---|---|
//! | `CRABBIT_REGALLOC_ORACLE` | `c0` (syntactic), `c2` (saturated e-graph) |
//! | `CRABBIT_BLOCK_FREQ` | `uniform`, `spectral`, `profile`, `cmt-provider` |
//! | `CRABBIT_MEASURED_COSTS` | op_costs.json; lifted per-decision costs replace the oracle's estimates |
#![feature(rustc_private)]

extern crate rustc_codegen_ssa;
extern crate rustc_driver;

use eregalloc_passes::{BlockFreqSource, EregallocRegisterAllocatePass, OracleKind};
use pliron::{context::Context, context::Ptr, region::Region};
use pliron_ll::conversion::pass::DynPass;
use rustc_codegen_ssa::traits::CodegenBackend;

/// CMT's region-level spectral block frequencies, in the exact shape of
/// `BlockFreqSource::Provider`. Selected by `CRABBIT_BLOCK_FREQ=cmt-provider`
/// (explicitly experimental: the pre-placement machine CFG uses implicit
/// fallthrough, which a `block.succs` analysis cannot see — `spectral`
/// runs the same Perron computation over the allocator's own successor
/// lists and is the contract's recommendation at the RA position).
pub fn cmt_block_frequencies(ctx: &Context, region: Ptr<Region>) -> Vec<f64> {
    cmt_spectral_cfg::block_frequencies(ctx, region)
}

/// The `eregalloc` engine factory: parses the engine's own environment
/// (oracle, frequency source, measured costs) and constructs the
/// allocator pass. Runs where `RegallocEngine::machine_pipeline` builds
/// the pipeline, so errors surface exactly like other config errors.
fn eregalloc_factory() -> Result<DynPass, String> {
    let oracle = match env("CRABBIT_REGALLOC_ORACLE").as_deref() {
        None | Some("c0") => OracleKind::Syntactic,
        Some("c2") => OracleKind::EGraph,
        Some(other) => {
            return Err(format!(
                "unknown value `{other}` for CRABBIT_REGALLOC_ORACLE; expected one of: c0, c2"
            ));
        }
    };
    let source = match env("CRABBIT_BLOCK_FREQ").as_deref() {
        None | Some("uniform") => BlockFreqSource::Uniform,
        Some("spectral") => BlockFreqSource::Spectral,
        Some("profile") => BlockFreqSource::Provider(research_config::profile_block_frequencies),
        Some("cmt-provider") => BlockFreqSource::Provider(cmt_block_frequencies),
        Some(other) => {
            return Err(format!(
                "unknown value `{other}` for CRABBIT_BLOCK_FREQ; expected one of: uniform, spectral, profile, cmt-provider"
            ));
        }
    };
    let mut pass = EregallocRegisterAllocatePass::new(oracle, source);
    if env("CRABBIT_MEASURED_COSTS").is_some() {
        pass = pass.with_measured_costs_provider(measured_costs_for_symbol);
    }
    Ok(DynPass::new(pass))
}

/// Register every engine this dylib links. Idempotent; called from the
/// rustc entry point below and reusable by future research binaries
/// (e.g. a research analysisd).
pub fn register_engines() {
    research_config::register_engine("eregalloc", eregalloc_factory);
}

// SAFETY: rustc loads custom codegen backends by looking up this exact
// exported symbol. crabbit-core is an rlib and exports no entry of its
// own.
#[unsafe(no_mangle)]
pub fn __rustc_codegen_backend() -> Box<dyn CodegenBackend> {
    register_engines();
    crabbit_core::create_backend()
}

/// Per-symbol measured restore costs from the `CRABBIT_MEASURED_COSTS`
/// op_costs.json (docs/PROFILE-FEEDBACK-BACKWARD.md, experiment E1): the
/// `lifted` section's integer keys are RA-boundary attribution ids — the
/// same ids the eregalloc allocator reads off machine defs'
/// `ll.derived_from`. Missing/unreadable file, missing symbol, or
/// non-integer keys never error: the allocator falls back to estimates.
/// Path-keyed lazy cache.
fn measured_costs_for_symbol(symbol: &str) -> Option<eregalloc_passes::MeasuredCosts> {
    use std::sync::{Mutex, OnceLock};
    type Costs = std::collections::HashMap<String, eregalloc_passes::MeasuredCosts>;
    static CACHE: OnceLock<Mutex<Option<(String, std::sync::Arc<Costs>)>>> = OnceLock::new();
    let path = env("CRABBIT_MEASURED_COSTS")?;
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    let mut cached = cache.lock().unwrap_or_else(|poison| poison.into_inner());
    let costs = match cached.as_ref() {
        Some((cached_path, costs)) if *cached_path == path => costs.clone(),
        _ => {
            let loaded: Costs = std::fs::read_to_string(&path)
                .ok()
                .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
                .and_then(|value| {
                    let object = value.as_object()?;
                    Some(
                        object
                            .iter()
                            .filter_map(|(symbol, kinds)| {
                                let lifted = kinds.get("lifted")?.as_object()?;
                                let map: eregalloc_passes::MeasuredCosts = lifted
                                    .iter()
                                    .filter_map(|(key, count)| {
                                        Some((key.parse::<i64>().ok()?, count.as_u64()?))
                                    })
                                    .collect();
                                Some((symbol.clone(), map))
                            })
                            .collect(),
                    )
                })
                .unwrap_or_else(|| {
                    eprintln!(
                        "crabbit: CRABBIT_MEASURED_COSTS={path} unreadable or not op_costs.json; using estimates"
                    );
                    Costs::default()
                });
            let costs = std::sync::Arc::new(loaded);
            *cached = Some((path, costs.clone()));
            costs
        }
    };
    costs.get(symbol).cloned().filter(|map| !map.is_empty())
}

fn env(var: &str) -> Option<String> {
    std::env::var(var).ok().filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The full engine path through the registry: register, resolve via
    /// env, and construct a real allocator pass.
    #[test]
    fn registered_eregalloc_constructs_an_allocator() {
        register_engines();
        unsafe { std::env::set_var("CRABBIT_REGALLOC", "eregalloc") };
        let engine = research_config::RegallocEngine::from_env().unwrap();
        let backend = pliron_ll::targets::lookup(&pliron_ll::triple::Triple::parse(
            "aarch64-unknown-linux-gnu",
        ))
        .unwrap();
        engine.machine_pipeline(backend).unwrap();
        unsafe { std::env::remove_var("CRABBIT_REGALLOC") };
    }
}
