//! crabbit's compilation configuration and pass-pipeline API.
//!
//! **Configuration.** A [CrabbitConfig] describes one compilation: the host
//! target (the triple rustc compiles the crate for), the device target (the
//! kernel module, `nvptx64-nvidia-cuda`) and the crate name. Passes are
//! built from it for one [Side]: a pass used on both sides reads
//! [CrabbitConfig::target], a host- or device-only pass reads its field.
//!
//! **Passes by name.** Every pass has a name. crabbit's own passes are known
//! to the crates that define them (`crabbit_mir::passes::construct`,
//! `pliron_ll::passes::llvm::construct`, the target backends' machine
//! passes); any other crate adds passes with [register_pass]. A registered
//! name shadows a crabbit pass of the same name.
//!
//! **Pipelines as orders.** Each [Stage] of a pipeline is an order of pass
//! names. crabbit has a default order per stage and side
//! ([default_order]); an order function registered with [register_order]
//! receives that default and may return its own — inserting, removing,
//! reordering or replacing passes, crabbit's own included. [build] turns
//! the effective order ([order]) into a pipeline.
//!
//! **Extending crabbit.** Registration happens in the codegen-backend dylib
//! rustc loads: a crate that links crabbit-core, registers its passes and
//! its order in `__rustc_codegen_backend`, then returns
//! `crabbit_core::create_backend()` (docs/EXTENDING.md).

use pliron_ll::conversion::pass::{Pass, PassList};
use pliron_ll::targets::{self, TargetBackend};

pub use pliron_ll::target_profile::{PipelineTarget, TargetProfile};
pub use pliron_ll::triple::Triple;

/// One compilation's configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CrabbitConfig {
    /// The host module's target: the triple rustc compiles for.
    pub host: PipelineTarget,
    /// The kernel module's target.
    pub device: PipelineTarget,
    /// The crate being compiled (labels measurement records), when known.
    pub crate_name: Option<String>,
}

impl CrabbitConfig {
    /// A configuration for `host` (the host profile follows from the
    /// triple) with the NVPTX device target.
    pub fn new(host: Triple) -> Self {
        CrabbitConfig {
            host: PipelineTarget::host(host),
            device: PipelineTarget::nvptx_kernel(),
            crate_name: None,
        }
    }

    pub fn with_crate_name(mut self, name: impl Into<String>) -> Self {
        self.crate_name = Some(name.into());
        self
    }

    /// The target of `side`.
    pub fn target(&self, side: Side) -> &PipelineTarget {
        match side {
            Side::Host => &self.host,
            Side::Device => &self.device,
        }
    }
}

/// Which module a pipeline compiles: the host module or the kernel module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Side {
    Host,
    Device,
}

/// A stage of a pipeline; each stage is one order of pass names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    /// From the importer's `mir` dialect to the LLVM dialect: crabbit's mir
    /// passes, `lower-dialect-mir` and the passes completing it.
    MirToLlvm,
    /// The LLVM-dialect mid-end.
    Midend,
    /// The host's machine pipeline: LLVM dialect to encoded machine code.
    /// The device side has none — PTX is written from the mid-end's output
    /// and ptxas does the machine work.
    Machine,
}

/// Builds a registered pass for one side of a configuration. A pass that
/// reads further settings (environment, files) reports a bad one here.
pub type PassFactory = fn(&CrabbitConfig, Side) -> Result<Box<dyn Pass>, String>;

/// Decides the order of one stage: receives crabbit's default order and
/// returns its own, or `None` to keep the default.
pub type OrderFn =
    fn(Stage, Side, &CrabbitConfig, &[&'static str]) -> Result<Option<Vec<&'static str>>, String>;

/// Errors from building pipelines and from configuration input.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error(
        "the device side has no machine pipeline (kernels are translated to PTX after the mid-end)"
    )]
    NoDeviceMachineStage,
    #[error(
        "no crabbit backend is registered for host triple `{triple}`; registered backends: {registered}"
    )]
    UnsupportedHost { triple: String, registered: String },
    #[error("the registered pass order failed for {stage:?} ({side:?}): {reason}")]
    Order {
        stage: Stage,
        side: Side,
        reason: String,
    },
    #[error("unknown pass `{name}` in the {stage:?} order ({side:?}); known passes: {known}")]
    UnknownPass {
        name: String,
        stage: Stage,
        side: Side,
        known: String,
    },
    #[error("pass `{name}` failed to construct: {reason}")]
    PassFactory { name: String, reason: String },
    #[error("config key `{key}` is not a CRABBIT_* variable")]
    NonCrabbitKey { key: String },
}

#[derive(Default)]
struct Registry {
    passes: Vec<(&'static str, PassFactory)>,
    order: Option<OrderFn>,
}

fn registry() -> std::sync::MutexGuard<'static, Registry> {
    static REGISTRY: std::sync::OnceLock<std::sync::Mutex<Registry>> = std::sync::OnceLock::new();
    REGISTRY
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
}

/// Register the pass called `name`. Registering a name again replaces its
/// factory (repeated loads are idempotent); a name crabbit already uses
/// shadows crabbit's pass everywhere that name appears in an order.
pub fn register_pass(name: &'static str, factory: PassFactory) {
    let mut registry = registry();
    match registry
        .passes
        .iter_mut()
        .find(|(existing, _)| *existing == name)
    {
        Some(entry) => entry.1 = factory,
        None => registry.passes.push((name, factory)),
    }
}

/// Register the order function. A process has one: the backend dylib
/// combines the order edits of everything it links into one function.
/// Registering again replaces it.
pub fn register_order(order: OrderFn) {
    registry().order = Some(order);
}

/// The names of the registered passes, in registration order.
pub fn registered_passes() -> Vec<&'static str> {
    registry().passes.iter().map(|(name, _)| *name).collect()
}

fn host_backend(cfg: &CrabbitConfig) -> Result<&'static TargetBackend, ConfigError> {
    targets::lookup(&cfg.host.triple).ok_or_else(|| ConfigError::UnsupportedHost {
        triple: cfg.host.triple.to_string(),
        registered: targets::registered_names().collect::<Vec<_>>().join(", "),
    })
}

/// crabbit's default order of `stage` for `side`.
pub fn default_order(
    stage: Stage,
    side: Side,
    cfg: &CrabbitConfig,
) -> Result<Vec<&'static str>, ConfigError> {
    Ok(match (stage, side) {
        (Stage::MirToLlvm, _) => crabbit_mir::passes::DEFAULT_ORDER.to_vec(),
        (Stage::Midend, _) => pliron_ll::passes::llvm::DEFAULT_ORDER.to_vec(),
        (Stage::Machine, Side::Host) => host_backend(cfg)?.default_order.to_vec(),
        (Stage::Machine, Side::Device) => return Err(ConfigError::NoDeviceMachineStage),
    })
}

/// The effective order of `stage` for `side`: the registered order
/// function's answer, or crabbit's default.
pub fn order(
    stage: Stage,
    side: Side,
    cfg: &CrabbitConfig,
) -> Result<Vec<&'static str>, ConfigError> {
    let default = default_order(stage, side, cfg)?;
    // Copy the function out: it may build pipelines itself.
    let order_fn = registry().order;
    match order_fn.map(|order_fn| order_fn(stage, side, cfg, &default)) {
        None | Some(Ok(None)) => Ok(default),
        Some(Ok(Some(order))) => Ok(order),
        Some(Err(reason)) => Err(ConfigError::Order {
            stage,
            side,
            reason,
        }),
    }
}

/// The pass called `name` in `stage` for `side`: a registered pass first,
/// then crabbit's own.
pub fn construct(
    name: &str,
    stage: Stage,
    side: Side,
    cfg: &CrabbitConfig,
) -> Result<Box<dyn Pass>, ConfigError> {
    let registered = registry()
        .passes
        .iter()
        .find(|(existing, _)| *existing == name)
        .map(|(_, factory)| *factory);
    if let Some(factory) = registered {
        return factory(cfg, side).map_err(|reason| ConfigError::PassFactory {
            name: name.to_string(),
            reason,
        });
    }
    crabbit_mir::passes::construct(name, cfg.crate_name.as_deref())
        .or_else(|| pliron_ll::passes::llvm::construct(name, cfg.target(side)))
        .or_else(|| match side {
            Side::Host => targets::lookup(&cfg.host.triple)?.construct(name),
            Side::Device => None,
        })
        .ok_or_else(|| ConfigError::UnknownPass {
            name: name.to_string(),
            stage,
            side,
            known: known_passes(side, cfg).join(", "),
        })
}

/// Every name [construct] resolves for `side` (for diagnostics).
fn known_passes(side: Side, cfg: &CrabbitConfig) -> Vec<&'static str> {
    let mut names = registered_passes();
    names.extend(crabbit_mir::passes::DEFAULT_ORDER);
    names.extend(pliron_ll::passes::llvm::DEFAULT_ORDER);
    if side == Side::Host
        && let Ok(backend) = host_backend(cfg)
    {
        names.extend(backend.default_order);
    }
    names.sort_unstable();
    names.dedup();
    names
}

/// The passes of `stage` for `side`: [order] built with [construct], in
/// order (`pliron_ll::conversion::pass::into_passes` makes them pliron's
/// `Passes`).
pub fn build(stage: Stage, side: Side, cfg: &CrabbitConfig) -> Result<PassList, ConfigError> {
    order(stage, side, cfg)?
        .into_iter()
        .map(|name| construct(name, stage, side, cfg))
        .collect()
}

/// Set or clear an environment variable under this crate's env-mutation
/// discipline.
///
/// SAFETY argument, once for every call site in this crate: `set_var`/
/// `remove_var` are unsafe because concurrent readers in other threads
/// could observe a torn environment. Every mutation in this crate happens
/// while holding [`with_env_config`]'s process-wide lock (including the
/// restore in its guard's `Drop`, which runs before the lock is released);
/// no other thread reads the environment at those points.
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
