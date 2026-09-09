//! Register-allocation engine selection (eregalloc/INTEGRATION-CONTRACT.md).
//!
//! | variable | values | default | effect |
//! |---|---|---|---|
//! | `CRABBIT_REGALLOC` | `linear`, `eregalloc` | `linear` | `linear` keeps the backend's own allocator; `eregalloc` swaps in `EregallocRegisterAllocatePass` |
//! | `CRABBIT_REGALLOC_ORACLE` | `c0`, `c2` | `c0` | `c0` → syntactic oracle, `c2` → saturated e-graph oracle; only read under `eregalloc` |
//! | `CRABBIT_BLOCK_FREQ` | `uniform`, `spectral`, `profile` | `uniform` | under `eregalloc`: the pass's frequency source (`spectral` is the fallthrough-aware machine-CFG Perron computation; `profile` reads measured frequencies from the `CRABBIT_PROFILE` JSON via a [BlockFreqSource::Provider], with per-function uniform fallback — see docs/PROFILE-FEEDBACK-PLAN.md) |
//!
//! Under `eregalloc`, `CRABBIT_SPILL_POLICY` / `CRABBIT_RESTORE_ESTIMATE`
//! are superseded (the oracle scoring and plan-driven remat are the engine)
//! and are ignored; `CRABBIT_BLOCK_FREQ` keeps its meaning. Under `linear`
//! all three `CodegenOpts` variables keep their exact meaning and the
//! eregalloc-only variables are ignored.
//!
//! This lives outside both `pliron-ll` (eregalloc-passes depends on
//! pliron-ll — package cycle) and the rustc_private-tainted `crabbit`
//! crate, so the resident driver/server can share it.
//!
//! `cmt_spectral_cfg::block_frequencies` is the region-level CMT analysis
//! and matches `BlockFreqSource::Provider` exactly; it is kept reachable
//! ([cmt_block_frequencies]) for explicit-edge CFGs, but the pre-placement
//! machine CFG at the RA position uses implicit fallthrough, which a
//! `block.succs` analysis cannot see — so `spectral` routes to
//! `BlockFreqSource::Spectral` (same math over the allocator's own
//! successor lists) as the contract prescribes.

#[cfg(feature = "eregalloc")]
use eregalloc_passes::{BlockFreqSource, EregallocRegisterAllocatePass, OracleKind};
use pliron::{context::Context, context::Ptr, region::Region};
use pliron_ll::{conversion::pass::Passes, targets::TargetBackend};

/// CMT's region-level spectral block frequencies, in the exact shape of
/// `BlockFreqSource::Provider`. Selected by `CRABBIT_BLOCK_FREQ=cmt-provider`
/// (an explicitly experimental value: see the module docs for why it
/// under-connects the machine CFG at the RA position).
#[cfg(feature = "cmt")]
pub fn cmt_block_frequencies(ctx: &Context, region: Ptr<Region>) -> Vec<f64> {
    cmt_spectral_cfg::block_frequencies(ctx, region)
}

/// Measured block frequencies from the `CRABBIT_PROFILE` JSON, in the
/// exact shape of [BlockFreqSource::Provider]. Called once per machine
/// function with its region at RA time; the profile's per-symbol vectors
/// are indexed by RA-order block position, which is precisely this
/// region's block order at that point. Any miss — no profile configured,
/// function symbol absent (stale profile), wrong-length vector — returns
/// uniform; a misconfigured profile must never fail allocation. (The
/// contract's own wrong-length guard would also catch a bad vector, but
/// falling back here keeps the two engines' behavior identical.)
pub fn profile_block_frequencies(ctx: &Context, region: Ptr<Region>) -> Vec<f64> {
    use pliron::linked_list::ContainsLinkedList;
    let blocks = region.deref(ctx).iter(ctx).count();
    let uniform = || vec![1.0; blocks];
    let func_op = region.deref(ctx).get_parent_op();
    let Some(func) =
        pliron::operation::Operation::get_op_dyn(func_op, ctx).downcast::<pliron_ll::aarch64::ops::FuncOp>()
    else {
        return uniform();
    };
    use pliron::builtin::op_interfaces::SymbolOpInterface;
    let symbol = func.get_symbol_name(ctx).to_string();
    pliron_ll::passes::profile_freq::frequencies_for(&symbol, blocks).unwrap_or_else(uniform)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegallocEngine {
    Linear,
    #[cfg(feature = "eregalloc")]
    Eregalloc { oracle: OracleKind, freq: FreqChoice },
}

/// The message for a config value whose engine was compiled out.
#[cfg(not(all(feature = "eregalloc", feature = "cmt")))]
fn feature_missing(feature: &str) -> String {
    format!("crabbit was built without the `{feature}` feature")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FreqChoice {
    Uniform,
    Spectral,
    Profile,
    #[cfg(feature = "cmt")]
    CmtProvider,
}

impl RegallocEngine {
    pub fn from_env() -> Result<Self, String> {
        let engine = env("CRABBIT_REGALLOC");
        match engine.as_deref() {
            None | Some("linear") => Ok(RegallocEngine::Linear),
            #[cfg(not(feature = "eregalloc"))]
            Some("eregalloc") => Err(feature_missing("eregalloc")),
            #[cfg(feature = "eregalloc")]
            Some("eregalloc") => {
                let oracle = match env("CRABBIT_REGALLOC_ORACLE").as_deref() {
                    None | Some("c0") => OracleKind::Syntactic,
                    Some("c2") => OracleKind::EGraph,
                    Some(other) => return Err(unknown("CRABBIT_REGALLOC_ORACLE", other, "c0, c2")),
                };
                let freq = match env("CRABBIT_BLOCK_FREQ").as_deref() {
                    None | Some("uniform") => FreqChoice::Uniform,
                    Some("spectral") => FreqChoice::Spectral,
                    Some("profile") => FreqChoice::Profile,
                    #[cfg(feature = "cmt")]
                    Some("cmt-provider") => FreqChoice::CmtProvider,
                    #[cfg(not(feature = "cmt"))]
                    Some("cmt-provider") => return Err(feature_missing("cmt")),
                    Some(other) => {
                        return Err(unknown(
                            "CRABBIT_BLOCK_FREQ",
                            other,
                            "uniform, spectral, profile, cmt-provider",
                        ));
                    }
                };
                Ok(RegallocEngine::Eregalloc { oracle, freq })
            }
            Some(other) => Err(unknown("CRABBIT_REGALLOC", other, "linear, eregalloc")),
        }
    }

    /// The allocator pass to substitute, or `None` for the backend default.
    #[cfg(feature = "eregalloc")]
    pub fn allocator(self) -> Option<EregallocRegisterAllocatePass> {
        match self {
            RegallocEngine::Linear => None,
            RegallocEngine::Eregalloc { oracle, freq } => {
                let source = match freq {
                    FreqChoice::Uniform => BlockFreqSource::Uniform,
                    FreqChoice::Spectral => BlockFreqSource::Spectral,
                    FreqChoice::Profile => BlockFreqSource::Provider(profile_block_frequencies),
                    #[cfg(feature = "cmt")]
                    FreqChoice::CmtProvider => BlockFreqSource::Provider(cmt_block_frequencies),
                };
                let mut pass = EregallocRegisterAllocatePass::new(oracle, source);
                if env("CRABBIT_MEASURED_COSTS").is_some() {
                    pass = pass.with_measured_costs_provider(measured_costs_for_symbol);
                }
                Some(pass)
            }
        }
    }

    /// The machine pipeline for `target` under this engine — the one place
    /// that knows which allocator the engine substitutes, so callers (the
    /// rustc backend and the resident server) never name engine types and
    /// feature gating stays local to this crate.
    pub fn machine_pipeline(self, target: &TargetBackend) -> Result<Passes, String> {
        match self {
            RegallocEngine::Linear => Ok(target.pipeline()),
            #[cfg(feature = "eregalloc")]
            engine @ RegallocEngine::Eregalloc { .. } => {
                let allocator = engine
                    .allocator()
                    .expect("Eregalloc engine always substitutes an allocator");
                target.pipeline_with_allocator(allocator).ok_or_else(|| {
                    format!(
                        "CRABBIT_REGALLOC=eregalloc is not supported by the `{}` backend (no swappable allocator)",
                        target.name
                    )
                })
            }
        }
    }
}

/// Run `f` with the given `CRABBIT_*` environment configuration set,
/// restoring the previous values afterwards (also on panic). Process
/// environment is global, so this serializes: every configured pipeline
/// section in a resident server runs under one lock. Only `CRABBIT_*`
/// keys are accepted — the config channel must not become an arbitrary
/// environment injector.
pub fn with_env_config<R>(
    config: &std::collections::BTreeMap<String, String>,
    f: impl FnOnce() -> R,
) -> Result<R, String> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    for key in config.keys() {
        if !key.starts_with("CRABBIT_") {
            return Err(format!("config key `{key}` is not a CRABBIT_* variable"));
        }
    }
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    struct Restore(Vec<(String, Option<String>)>);
    impl Drop for Restore {
        fn drop(&mut self) {
            for (key, value) in &self.0 {
                match value {
                    Some(value) => unsafe { std::env::set_var(key, value) },
                    None => unsafe { std::env::remove_var(key) },
                }
            }
        }
    }
    let _restore = Restore(
        config
            .keys()
            .map(|k| (k.clone(), std::env::var(k).ok()))
            .collect(),
    );
    for (key, value) in config {
        unsafe { std::env::set_var(key, value) };
    }
    Ok(f())
}

fn unknown(var: &str, value: &str, expected: &str) -> String {
    format!("unknown value `{value}` for {var}; expected one of: {expected}")
}

/// Per-symbol measured restore costs from the `CRABBIT_MEASURED_COSTS`
/// op_costs.json (docs/PROFILE-FEEDBACK-BACKWARD.md, experiment E1): the
/// `lifted` section's integer keys are RA-boundary attribution ids — the
/// same ids the eregalloc allocator reads off machine defs'
/// `ll.derived_from`. Missing/unreadable file, missing symbol, or
/// non-integer keys never error: the allocator falls back to estimates.
/// Path-keyed lazy cache, mirroring [profile_block_frequencies]'s.
#[cfg(feature = "eregalloc")]
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

    // Env-var tests share process state; serialize them.
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn with_env(vars: &[(&str, &str)], f: impl FnOnce()) {
        let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        for var in [
            "CRABBIT_REGALLOC",
            "CRABBIT_REGALLOC_ORACLE",
            "CRABBIT_BLOCK_FREQ",
            "CRABBIT_PROFILE",
            "CRABBIT_MEASURED_COSTS",
        ] {
            unsafe { std::env::remove_var(var) };
        }
        for (k, v) in vars {
            unsafe { std::env::set_var(k, v) };
        }
        f();
        for (k, _) in vars {
            unsafe { std::env::remove_var(k) };
        }
    }

    #[test]
    fn defaults_to_linear() {
        with_env(&[], || {
            assert_eq!(RegallocEngine::from_env().unwrap(), RegallocEngine::Linear);
            #[cfg(feature = "eregalloc")]
            assert!(RegallocEngine::Linear.allocator().is_none());
        });
    }

    #[cfg(feature = "eregalloc")]
    #[test]
    fn eregalloc_variants_parse() {
        with_env(&[("CRABBIT_REGALLOC", "eregalloc")], || {
            assert_eq!(
                RegallocEngine::from_env().unwrap(),
                RegallocEngine::Eregalloc { oracle: OracleKind::Syntactic, freq: FreqChoice::Uniform }
            );
        });
        with_env(
            &[("CRABBIT_REGALLOC", "eregalloc"), ("CRABBIT_REGALLOC_ORACLE", "c2"), ("CRABBIT_BLOCK_FREQ", "spectral")],
            || {
                let engine = RegallocEngine::from_env().unwrap();
                assert_eq!(
                    engine,
                    RegallocEngine::Eregalloc { oracle: OracleKind::EGraph, freq: FreqChoice::Spectral }
                );
                assert!(engine.allocator().is_some());
            },
        );
    }

    #[cfg(feature = "eregalloc")]
    #[test]
    fn profile_freq_choice_parses_and_uses_the_provider() {
        with_env(
            &[("CRABBIT_REGALLOC", "eregalloc"), ("CRABBIT_BLOCK_FREQ", "profile")],
            || {
                let engine = RegallocEngine::from_env().unwrap();
                assert_eq!(
                    engine,
                    RegallocEngine::Eregalloc {
                        oracle: OracleKind::Syntactic,
                        freq: FreqChoice::Profile
                    }
                );
                assert!(engine.allocator().is_some());
            },
        );
    }

    /// The provider must fall back to uniform — never error — on a stale
    /// or missing profile (docs/PROFILE-FEEDBACK-PLAN.md).
    #[cfg(feature = "eregalloc")]
    #[test]
    fn profile_provider_falls_back_to_uniform_without_a_usable_profile() {
        use pliron::builtin::op_interfaces::OneRegionInterface;
        let mut ctx = pliron::context::Context::new();
        pliron_ll::aarch64::register(&mut ctx);
        let func = pliron_ll::aarch64::ops::FuncOp::new(
            &mut ctx,
            "no_such_symbol_in_any_profile".try_into().unwrap(),
            pliron_ll::ll::LinkageAttr::External,
        );
        let region = func.get_region(&ctx);
        with_env(&[("CRABBIT_PROFILE", "/nonexistent/profile.json")], || {
            assert_eq!(profile_block_frequencies(&ctx, region), vec![1.0]);
        });
        with_env(&[], || {
            assert_eq!(profile_block_frequencies(&ctx, region), vec![1.0]);
        });
    }

    #[cfg(not(feature = "eregalloc"))]
    #[test]
    fn disabled_eregalloc_feature_reports_clearly() {
        with_env(&[("CRABBIT_REGALLOC", "eregalloc")], || {
            assert_eq!(
                RegallocEngine::from_env().unwrap_err(),
                "crabbit was built without the `eregalloc` feature"
            );
        });
    }

    #[cfg(all(feature = "eregalloc", not(feature = "cmt")))]
    #[test]
    fn disabled_cmt_feature_reports_clearly() {
        with_env(
            &[("CRABBIT_REGALLOC", "eregalloc"), ("CRABBIT_BLOCK_FREQ", "cmt-provider")],
            || {
                assert_eq!(
                    RegallocEngine::from_env().unwrap_err(),
                    "crabbit was built without the `cmt` feature"
                );
            },
        );
    }

    #[test]
    fn oracle_is_ignored_under_linear_and_unknown_values_error() {
        with_env(&[("CRABBIT_REGALLOC_ORACLE", "bogus")], || {
            assert_eq!(RegallocEngine::from_env().unwrap(), RegallocEngine::Linear);
        });
        with_env(&[("CRABBIT_REGALLOC", "bogus")], || {
            let err = RegallocEngine::from_env().unwrap_err();
            assert!(err.contains("CRABBIT_REGALLOC") && err.contains("linear, eregalloc"), "{err}");
        });
        #[cfg(feature = "eregalloc")]
        with_env(&[("CRABBIT_REGALLOC", "eregalloc"), ("CRABBIT_REGALLOC_ORACLE", "c1")], || {
            assert!(RegallocEngine::from_env().unwrap_err().contains("c0, c2"));
        });
    }
    #[cfg(feature = "eregalloc")]
    #[test]
    fn measured_costs_loads_lifted_ints_per_symbol() {
        let dir = std::env::temp_dir().join(format!("rc-mcosts-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("op_costs.json");
        std::fs::write(
            &path,
            r#"{"f": {"machine": {"0": 3}, "lifted": {"2": 5, "7": 1, "isel:abi": 4}},
                "g": {"lifted": {}}}"#,
        )
        .unwrap();
        with_env(&[("CRABBIT_MEASURED_COSTS", path.to_str().unwrap())], || {
            let f = measured_costs_for_symbol("f").expect("f has costs");
            assert_eq!(f.get(&2), Some(&5));
            assert_eq!(f.get(&7), Some(&1));
            assert_eq!(f.len(), 2, "root keys are skipped: {f:?}");
            assert!(measured_costs_for_symbol("g").is_none(), "empty map -> None");
            assert!(measured_costs_for_symbol("missing").is_none());
        });
        let _ = std::fs::remove_dir_all(&dir);
    }
}
