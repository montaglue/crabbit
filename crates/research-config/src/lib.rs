//! Register-allocation engine selection (eregalloc/INTEGRATION-CONTRACT.md).
//!
//! | variable | values | default | effect |
//! |---|---|---|---|
//! | `CRABBIT_REGALLOC` | `linear`, or a linked engine (`eregalloc`) | `linear` | `linear` keeps the backend's own allocator; other names resolve through the [engine registry](register_engine) — the public backend links no engines, `libcrabbit_research.so` registers `eregalloc` |
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
//! Engine availability is a LINK-TIME property, not a feature: the
//! research engines live in private sibling repositories, and cargo
//! resolves even optional dependencies into the lockfile, so the public
//! workspace cannot reference them at all. Instead the workspace-excluded
//! `crates/crabbit-research` composition dylib registers its engines here
//! ([register_engine]) before delegating to the ordinary backend; the
//! `eregalloc`/`cmt` cargo features remain declared (default-off) only so
//! existing `--features` invocations stay valid — they are no-ops.

use pliron::{context::Context, context::Ptr, region::Region};
use pliron_ll::{
    conversion::pass::{DynPass, Passes},
    targets::TargetBackend,
};

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

/// A factory for an externally linked register-allocation engine: reads
/// its own `CRABBIT_*` configuration from the environment (it runs at the
/// same point [RegallocEngine::from_env] does) and returns the allocator
/// pass to substitute at the backend's swappable-allocator slot.
pub type EngineFactory = fn() -> Result<DynPass, String>;

fn engine_registry() -> &'static std::sync::Mutex<std::collections::BTreeMap<&'static str, EngineFactory>>
{
    static REGISTRY: std::sync::OnceLock<
        std::sync::Mutex<std::collections::BTreeMap<&'static str, EngineFactory>>,
    > = std::sync::OnceLock::new();
    REGISTRY.get_or_init(|| std::sync::Mutex::new(std::collections::BTreeMap::new()))
}

/// Register an engine under the name `CRABBIT_REGALLOC` selects it by.
/// Called by a composition dylib (crates/crabbit-research) at backend
/// load, before any compilation consults [RegallocEngine::from_env].
/// Re-registering a name replaces the factory (idempotent loads).
pub fn register_engine(name: &'static str, factory: EngineFactory) {
    engine_registry()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .insert(name, factory);
}

fn engine_factory(name: &str) -> Option<(&'static str, EngineFactory)> {
    engine_registry()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .get_key_value(name)
        .map(|(n, f)| (*n, *f))
}

#[derive(Clone, Copy, Debug)]
pub enum RegallocEngine {
    Linear,
    /// An engine resolved through the registry; carries its registered
    /// name (diagnostics) and factory.
    External {
        name: &'static str,
        factory: EngineFactory,
    },
}


/// Errors from engine selection and environment configuration.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("unknown value `{value}` for {var}; expected one of: {expected}")]
    UnknownValue {
        var: &'static str,
        value: String,
        expected: &'static str,
    },
    /// The engine exists but is compiled only into the research
    /// composition dylib, not the backend currently running.
    #[error(
        "the `{engine}` engine is not linked into this backend: it lives in \
         crates/crabbit-research (which needs the private research checkouts); \
         build that crate and point -Zcodegen-backend at libcrabbit_research.so"
    )]
    EngineNotLinked { engine: String },
    #[error("engine `{engine}` failed to construct its allocator: {reason}")]
    EngineFactory { engine: String, reason: String },
    #[error(
        "CRABBIT_REGALLOC={engine} is not supported by the `{backend}` backend \
         (no swappable allocator)"
    )]
    NoSwappableAllocator { engine: String, backend: String },
    #[error("config key `{key}` is not a CRABBIT_* variable")]
    NonCrabbitKey { key: String },
}

/// Compared by identity/name only: factories are `fn` pointers, whose
/// address comparison the compiler rightly flags as meaningless.
impl PartialEq for RegallocEngine {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (RegallocEngine::Linear, RegallocEngine::Linear) => true,
            (
                RegallocEngine::External { name: a, .. },
                RegallocEngine::External { name: b, .. },
            ) => a == b,
            _ => false,
        }
    }
}
impl Eq for RegallocEngine {}

impl RegallocEngine {
    pub fn from_env() -> Result<Self, ConfigError> {
        let engine = env("CRABBIT_REGALLOC");
        match engine.as_deref() {
            None | Some("linear") => Ok(RegallocEngine::Linear),
            Some(name) => match engine_factory(name) {
                Some((name, factory)) => Ok(RegallocEngine::External { name, factory }),
                None if name == "eregalloc" => Err(ConfigError::EngineNotLinked {
                    engine: name.to_string(),
                }),
                None => Err(ConfigError::UnknownValue {
                    var: "CRABBIT_REGALLOC",
                    value: name.to_string(),
                    expected: "linear, eregalloc",
                }),
            },
        }
    }

    /// The machine pipeline for `target` under this engine — the one place
    /// that knows which allocator the engine substitutes, so callers (the
    /// rustc backend and the resident server) never name engine types.
    pub fn machine_pipeline(self, target: &TargetBackend) -> Result<Passes, ConfigError> {
        match self {
            RegallocEngine::Linear => Ok(target.pipeline()),
            RegallocEngine::External { name, factory } => {
                let allocator = factory().map_err(|reason| ConfigError::EngineFactory {
                    engine: name.to_string(),
                    reason,
                })?;
                target.pipeline_with_allocator(allocator).ok_or_else(|| {
                    ConfigError::NoSwappableAllocator {
                        engine: name.to_string(),
                        backend: target.name.to_string(),
                    }
                })
            }
        }
    }
}

/// Set or clear an environment variable under this crate's env-mutation
/// discipline.
///
/// SAFETY argument, once for every call site in this crate: `set_var`/
/// `remove_var` are unsafe because concurrent readers in other threads
/// could observe a torn environment. Every mutation in this crate happens
/// either while holding [`with_env_config`]'s process-wide lock (including
/// the restore in its guard's `Drop`, which runs before the lock is
/// released) or inside this crate's lock-serialized tests; no other thread
/// reads the environment at those points.
fn apply_env(key: &str, value: Option<&str>) {
    match value {
        Some(value) => unsafe { std::env::set_var(key, value) },
        None => unsafe { std::env::remove_var(key) },
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
) -> Result<R, ConfigError> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    for key in config.keys() {
        if !key.starts_with("CRABBIT_") {
            return Err(ConfigError::NonCrabbitKey { key: key.clone() });
        }
    }
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    struct Restore(Vec<(String, Option<String>)>);
    impl Drop for Restore {
        fn drop(&mut self) {
            for (key, value) in &self.0 {
                // Still under with_env_config's lock (the guard drops
                // before the lock); see apply_env's SAFETY.
                apply_env(key, value.as_deref());
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
        apply_env(key, Some(value));
    }
    Ok(f())
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
            apply_env(var, None);
        }
        for (k, v) in vars {
            apply_env(k, Some(v));
        }
        f();
        for (k, _) in vars {
            apply_env(k, None);
        }
    }

    #[test]
    fn defaults_to_linear() {
        with_env(&[], || {
            assert_eq!(RegallocEngine::from_env().unwrap(), RegallocEngine::Linear);
        });
    }

    #[test]
    fn unlinked_eregalloc_reports_the_composition_dylib() {
        with_env(&[("CRABBIT_REGALLOC", "eregalloc")], || {
            // No engine registered in this crate's own tests: the message
            // must point at the composition dylib, not at a cargo feature.
            let err = RegallocEngine::from_env().unwrap_err().to_string();
            assert!(
                err.contains("crabbit-research") && err.contains("libcrabbit_research.so"),
                "{err}"
            );
        });
    }

    #[test]
    fn registered_engines_resolve_and_their_factory_errors_surface() {
        fn failing_factory() -> Result<DynPass, String> {
            Err("factory ran".to_string())
        }
        register_engine("test-engine", failing_factory);
        with_env(&[("CRABBIT_REGALLOC", "test-engine")], || {
            let engine = RegallocEngine::from_env().unwrap();
            assert!(matches!(
                engine,
                RegallocEngine::External { name: "test-engine", .. }
            ));
            let backend = pliron_ll::targets::lookup(&pliron_ll::triple::Triple::parse(
                "aarch64-unknown-linux-gnu",
            ))
            .unwrap();
            // Passes has no Debug: destructure instead of unwrap_err.
            match engine.machine_pipeline(backend) {
                Err(error) => {
                    assert!(error.to_string().contains("factory ran"), "{error}")
                }
                Ok(_) => panic!("factory error must propagate"),
            }
        });
    }

    #[test]
    fn oracle_is_ignored_under_linear_and_unknown_values_error() {
        with_env(&[("CRABBIT_REGALLOC_ORACLE", "bogus")], || {
            assert_eq!(RegallocEngine::from_env().unwrap(), RegallocEngine::Linear);
        });
        with_env(&[("CRABBIT_REGALLOC", "bogus")], || {
            let err = RegallocEngine::from_env().unwrap_err().to_string();
            assert!(
                err.contains("CRABBIT_REGALLOC") && err.contains("linear, eregalloc"),
                "{err}"
            );
        });
    }

    /// The provider must fall back to uniform — never error — on a stale
    /// or missing profile (docs/PROFILE-FEEDBACK-PLAN.md).
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

    #[test]
    fn with_env_config_rejects_non_crabbit_keys_and_restores() {
        let mut config = std::collections::BTreeMap::new();
        config.insert("PATH".to_string(), "hijack".to_string());
        assert!(with_env_config(&config, || ()).is_err());
        let mut config = std::collections::BTreeMap::new();
        config.insert("CRABBIT_REGALLOC".to_string(), "linear".to_string());
        with_env_config(&config, || {
            assert_eq!(std::env::var("CRABBIT_REGALLOC").unwrap(), "linear");
        })
        .unwrap();
    }
}
