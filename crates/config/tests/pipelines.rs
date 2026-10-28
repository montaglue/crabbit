//! The pipeline API: default orders, registered passes, and an order
//! function that inserts, removes and replaces crabbit's own passes. One
//! test function: the registry is process-global, so registrations and
//! the pipelines that observe them must not interleave across threads.

use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

use crabbit_config::{
    ConfigError, CrabbitConfig, Side, Stage, Triple, build, default_order, order, register_order,
    register_pass, registered_passes,
};
use pliron_ll::{
    context::{Context, Ptr},
    conversion::pass::{AnalysisManager, Pass, PassResult, into_passes, unchanged},
    dialects::builtin,
    ir::{op::Op, operation::Operation},
};

static MARKER_RUNS: AtomicUsize = AtomicUsize::new(0);
/// What the registered order function does (0 = edits, 1 = unknown name,
/// 2 = error, 3 = failing pass).
static ORDER_MODE: AtomicU8 = AtomicU8::new(0);

/// A no-op pass identified by its name; counts the marker's runs.
struct Named(&'static str);

impl Pass for Named {
    fn name(&self) -> &str {
        self.0
    }

    fn run(
        &mut self,
        _op: Ptr<Operation>,
        _ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> pliron::result::Result<PassResult> {
        if self.0 == "test-marker" {
            MARKER_RUNS.fetch_add(1, Ordering::SeqCst);
        }
        Ok(unchanged())
    }
}

fn marker(_: &CrabbitConfig, side: Side) -> Result<Box<dyn Pass>, String> {
    assert_eq!(
        side,
        Side::Device,
        "the order only places the marker in device pipelines"
    );
    Ok(Box::new(Named("test-marker")))
}

fn allocator(cfg: &CrabbitConfig, _: Side) -> Result<Box<dyn Pass>, String> {
    assert!(cfg.host.triple.to_string().starts_with("aarch64"));
    Ok(Box::new(Named("test-allocator")))
}

fn failing(_: &CrabbitConfig, _: Side) -> Result<Box<dyn Pass>, String> {
    Err("bad setting".to_string())
}

fn edits(
    stage: Stage,
    side: Side,
    _: &CrabbitConfig,
    default: &[&'static str],
) -> Result<Option<Vec<&'static str>>, String> {
    match ORDER_MODE.load(Ordering::SeqCst) {
        1 => return Ok(Some(vec!["no-such-pass"])),
        2 => return Err("order refused".to_string()),
        3 => return Ok(Some(vec!["test-failing"])),
        _ => {}
    }
    Ok(match (stage, side) {
        // Insert before unroll and remove licm: crabbit's own passes are
        // as editable as the extension's.
        (Stage::Midend, Side::Device) => {
            let mut order: Vec<_> = default
                .iter()
                .copied()
                .filter(|name| *name != "llvm-licm")
                .collect();
            let unroll = order
                .iter()
                .position(|name| *name == "llvm-unroll")
                .unwrap();
            order.insert(unroll, "test-marker");
            Some(order)
        }
        // Replace the register allocator.
        (Stage::Machine, Side::Host) => Some(
            default
                .iter()
                .map(|name| {
                    if *name == "aarch64-register-allocate" {
                        "test-allocator"
                    } else {
                        *name
                    }
                })
                .collect(),
        ),
        _ => None,
    })
}

fn names(stage: Stage, side: Side, cfg: &CrabbitConfig) -> Vec<String> {
    build(stage, side, cfg)
        .unwrap()
        .iter()
        .map(|pass| pass.name().to_string())
        .collect()
}

#[test]
fn orders_insert_remove_and_replace_passes() {
    let cfg = CrabbitConfig::new(Triple::parse("aarch64-unknown-linux-gnu")).with_crate_name("t");

    // Defaults come from the crates that define the passes.
    assert_eq!(
        default_order(Stage::MirToLlvm, Side::Host, &cfg).unwrap(),
        crabbit_mir::passes::DEFAULT_ORDER
    );
    assert_eq!(
        default_order(Stage::Midend, Side::Device, &cfg).unwrap(),
        pliron_ll::passes::llvm::DEFAULT_ORDER
    );
    assert_eq!(
        default_order(Stage::Machine, Side::Host, &cfg).unwrap(),
        pliron_ll::passes::aarch64::default_order(pliron_ll::passes::aarch64::TargetOs::Linux)
    );
    assert!(matches!(
        default_order(Stage::Machine, Side::Device, &cfg),
        Err(ConfigError::NoDeviceMachineStage)
    ));
    let riscv = CrabbitConfig::new(Triple::parse("riscv64-unknown-linux-gnu"));
    assert!(matches!(
        default_order(Stage::Machine, Side::Host, &riscv),
        Err(ConfigError::UnsupportedHost { .. })
    ));

    // Without registrations every stage builds its default order.
    for (stage, side) in [
        (Stage::MirToLlvm, Side::Host),
        (Stage::MirToLlvm, Side::Device),
        (Stage::Midend, Side::Host),
        (Stage::Midend, Side::Device),
        (Stage::Machine, Side::Host),
    ] {
        let default: Vec<String> = default_order(stage, side, &cfg)
            .unwrap()
            .into_iter()
            .map(String::from)
            .collect();
        assert_eq!(names(stage, side, &cfg), default, "{stage:?} {side:?}");
    }

    register_pass("test-marker", marker);
    register_pass("test-allocator", allocator);
    register_pass("test-failing", failing);
    register_order(edits);

    let device = names(Stage::Midend, Side::Device, &cfg);
    let at = device
        .iter()
        .position(|name| name == "test-marker")
        .expect("marker inserted");
    assert_eq!(device[at + 1], "llvm-unroll", "{device:?}");
    assert!(
        !device.iter().any(|name| name == "llvm-licm"),
        "licm removed: {device:?}"
    );
    // The host mid-end keeps crabbit's default.
    assert_eq!(
        names(Stage::Midend, Side::Host, &cfg),
        pliron_ll::passes::llvm::DEFAULT_ORDER.to_vec()
    );
    let machine = names(Stage::Machine, Side::Host, &cfg);
    assert!(
        machine.iter().any(|name| name == "test-allocator"),
        "{machine:?}"
    );
    assert!(
        !machine
            .iter()
            .any(|name| name == "aarch64-register-allocate"),
        "{machine:?}"
    );
    assert_eq!(
        order(Stage::MirToLlvm, Side::Host, &cfg).unwrap(),
        crabbit_mir::passes::DEFAULT_ORDER
    );

    // A registered pass runs when its pipeline does.
    let mut ctx = Context::new();
    let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
    into_passes(build(Stage::Midend, Side::Device, &cfg).unwrap())
        .run(
            module.get_operation(),
            &mut ctx,
            &mut AnalysisManager::default(),
        )
        .unwrap();
    assert_eq!(MARKER_RUNS.load(Ordering::SeqCst), 1);

    // Unknown names, refused orders and failing factories are errors.
    ORDER_MODE.store(1, Ordering::SeqCst);
    match build(Stage::Midend, Side::Host, &cfg) {
        Err(ConfigError::UnknownPass { name, known, .. }) => {
            assert_eq!(name, "no-such-pass");
            assert!(
                known.contains("llvm-gvn") && known.contains("test-marker"),
                "{known}"
            );
        }
        other => panic!(
            "expected UnknownPass, got {:?}",
            other.map(|passes| passes.len())
        ),
    }
    ORDER_MODE.store(2, Ordering::SeqCst);
    assert!(matches!(
        order(Stage::Midend, Side::Host, &cfg),
        Err(ConfigError::Order { .. })
    ));
    ORDER_MODE.store(3, Ordering::SeqCst);
    match build(Stage::Midend, Side::Host, &cfg) {
        Err(ConfigError::PassFactory { name, reason }) => {
            assert_eq!(
                (name.as_str(), reason.as_str()),
                ("test-failing", "bad setting")
            );
        }
        other => panic!(
            "expected PassFactory, got {:?}",
            other.map(|passes| passes.len())
        ),
    }
    ORDER_MODE.store(0, Ordering::SeqCst);

    // Registering a name again replaces it.
    register_pass("test-marker", marker);
    assert_eq!(
        registered_passes(),
        ["test-marker", "test-allocator", "test-failing"]
    );
}
