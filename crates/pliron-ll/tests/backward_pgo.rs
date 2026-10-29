//! Backward PGO through the real pipeline: a function with a GVN-dedup
//! pair and a constant divide goes through the mid-end and the machine
//! pipeline with every pass's IR recorded, as a traced compilation
//! records it; synthetic cycles on the machine code of (a) the merged add
//! and (b) the divide's magic expansion must come back, at the source
//! level, as (a) a 50/50 split across both original adds and (b) the
//! original `sdiv` — and the total must hold at every level.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::num::NonZero;

use pliron::builtin::op_interfaces::{OneRegionInterface, OneResultInterface};
use pliron::context::Context;
use pliron::printable::Printable as _;
use pliron_bpgo::{BackwardPasses, CYCLES, profile, walk};
use pliron_ll::conversion::pass::{AnalysisManager, PMConfig, Pass as _, Passes};
use pliron_ll::dialects::builtin::attributes::IntegerAttr;
use pliron_ll::dialects::builtin::ops::{ConstantOp, ModuleOp};
use pliron_ll::dialects::builtin::types::{IntegerType, Signedness};
use pliron_ll::dialects::llvm::op_interfaces::{
    BinArithOp as _, IntBinArithOpWithOverflowFlag as _,
};
use pliron_ll::dialects::llvm::{
    attributes::LinkageAttr,
    ops::{AddOp, FuncOp, ReturnOp, SDivOp},
    types::FuncType,
};
use pliron_ll::dialects::{aarch64, macho};
use pliron_ll::ir::op::Op as _;
use pliron_ll::linked_list::ContainsLinkedList;
use pliron_ll::passes::aarch64::backward::{Samples, attach, timeline};
use pliron_ll::passes::aarch64::blockmap::Aarch64BlockmapIdPass;
use pliron_ll::passes::aarch64::opmap::{self, Aarch64OpIdPass, unstamped_op_count};
use pliron_ll::passes::aarch64::{
    TargetOs, aarch64_asm_lower::Aarch64AsmLowerPass,
    aarch64_block_placement::Aarch64BlockPlacementPass,
    aarch64_branch_relax::Aarch64BranchRelaxPass, aarch64_encode::Aarch64EncodePass,
    aarch64_frame_lower::Aarch64FrameLowerPass, aarch64_legalize::Aarch64LegalizePass,
    aarch64_machine_cfg_cleanup::Aarch64MachineCfgCleanupPass,
    aarch64_post_ra_opts::Aarch64PostRaOptsPass,
    aarch64_register_allocate::Aarch64RegisterAllocatePass,
    aarch64_target_opts_pre_ra::Aarch64TargetOptsPreRaPass, llvm_aarch64_abi::LlvmAarch64AbiPass,
    llvm_to_aarch64_isel::LlvmToAarch64IselPass, verify_llvm_for_aarch64::VerifyLlvmForAarch64Pass,
};
use pliron_ll::passes::llvm::default_midend;
use pliron_ll::target_profile::{PipelineTarget, TargetProfile};
use pliron_ll::utils::apint::APInt;
use pliron_trace::{Trace, TraceMeta};

fn context() -> Context {
    let mut ctx = Context::new();
    aarch64::register(&mut ctx);
    macho::register(&mut ctx);
    ctx
}

/// `f(a, b) = (a + b) / 3 + (a + b)`, the second `a + b` a duplicate GVN
/// merges into the first. Source ids in program order: add1=0, add2=1,
/// const3=2, sdiv=3, add_ret=4, ret=5.
fn build(ctx: &mut Context) -> ModuleOp {
    let module = ModuleOp::new(ctx, "test".try_into().unwrap());
    let body = module.get_region(ctx).deref(ctx).get_head().unwrap();
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let func_ty = FuncType::get(
        ctx,
        i64_ty.into(),
        vec![i64_ty.into(), i64_ty.into()],
        false,
    );
    let func = FuncOp::new(ctx, "audit".try_into().unwrap(), func_ty);
    func.set_attr_llvm_function_linkage(ctx, LinkageAttr::ExternalLinkage);
    func.get_or_create_entry_block(ctx);
    func.get_operation().insert_at_back(body, ctx);
    let entry = func.get_entry_block(ctx).unwrap();
    let args: Vec<_> = entry.deref(ctx).arguments().collect();
    let add1 = AddOp::new_with_overflow_flag(ctx, args[0], args[1], Default::default());
    add1.get_operation().insert_at_back(entry, ctx);
    let add2 = AddOp::new_with_overflow_flag(ctx, args[0], args[1], Default::default());
    add2.get_operation().insert_at_back(entry, ctx);
    let three = ConstantOp::new(
        ctx,
        Box::new(IntegerAttr::new(
            i64_ty,
            APInt::from_u64(3, NonZero::new(64).unwrap()),
        )),
    );
    three.get_operation().insert_at_back(entry, ctx);
    let (add1_v, add2_v, three_v) = (
        add1.get_result(ctx),
        add2.get_result(ctx),
        three.get_result(ctx),
    );
    let sdiv = SDivOp::new(ctx, add1_v, three_v);
    sdiv.get_operation().insert_at_back(entry, ctx);
    let sdiv_v = sdiv.get_result(ctx);
    let add_ret = AddOp::new_with_overflow_flag(ctx, sdiv_v, add2_v, Default::default());
    add_ret.get_operation().insert_at_back(entry, ctx);
    let add_ret_v = add_ret.get_result(ctx);
    ReturnOp::new(ctx, Some(add_ret_v))
        .get_operation()
        .insert_at_back(entry, ctx);
    module
}

/// Compile `audit` through the mid-end and the aarch64 pipeline, recording
/// the IR after every pass (pliron's `print_after_all`, as a traced
/// compilation does). Returns the trace and the final module.
fn traced_compile(
    ctx: &mut Context,
) -> (Trace, pliron::context::Ptr<pliron::operation::Operation>) {
    let module = build(ctx);
    let root = module.get_operation();
    let dumps = tempfile::tempdir().unwrap();
    let mut analyses = AnalysisManager::default();
    analyses.set_config(PMConfig {
        print_after_all: true,
        ir_printing_dir: Some(dumps.path().to_path_buf()),
        ..Default::default()
    });
    let mut trace = Trace::new(TraceMeta {
        name: "audit".into(),
        kind: "compiler-run".into(),
        entry: None,
        source: None,
        pipeline: Vec::new(),
        target: None,
        note: None,
        extra: BTreeMap::new(),
    });
    trace.push_snapshot("initial", root.disp(ctx).to_string());

    let target = PipelineTarget {
        triple: pliron_ll::triple::Triple::parse("aarch64-unknown-linux-gnu"),
        profile: TargetProfile::host_cpu(),
    };
    default_midend(&target)
        .run(root, ctx, &mut analyses)
        .unwrap();
    let mut machine = Passes::default();
    machine.add_pass(VerifyLlvmForAarch64Pass::new(TargetOs::Linux));
    machine.add_pass(LlvmAarch64AbiPass::new(TargetOs::Linux));
    machine.add_pass(Aarch64OpIdPass);
    machine.add_pass(LlvmToAarch64IselPass);
    machine.add_pass(Aarch64LegalizePass);
    machine.add_pass(Aarch64MachineCfgCleanupPass);
    machine.add_pass(Aarch64TargetOptsPreRaPass);
    machine.add_pass(Aarch64BlockmapIdPass);
    machine.add_pass(Aarch64RegisterAllocatePass);
    machine.add_pass(Aarch64FrameLowerPass);
    machine.add_pass(Aarch64PostRaOptsPass);
    machine.add_pass(Aarch64BlockPlacementPass);
    machine.add_pass(Aarch64BranchRelaxPass);
    machine.add_pass(Aarch64AsmLowerPass);
    machine.add_pass(Aarch64EncodePass);
    machine.run(root, ctx, &mut analyses).unwrap();

    let mut files: Vec<(usize, String, String)> = std::fs::read_dir(dumps.path())
        .unwrap()
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let stem = path.file_stem()?.to_str()?.to_string();
            let (count, name) = stem.split_once("-after-")?;
            Some((
                count.parse().ok()?,
                name.to_string(),
                std::fs::read_to_string(&path).ok()?,
            ))
        })
        .collect();
    files.sort_by_key(|(count, ..)| *count);
    for (_, name, ir) in files {
        trace.meta.pipeline.push(name.clone());
        trace.push_snapshot(name, ir);
    }
    (trace, root)
}

fn backward_passes() -> BackwardPasses {
    let mut passes = BackwardPasses::new();
    pliron_ll::passes::aarch64::backward::register(&mut passes);
    pliron_ll::passes::llvm::backward::register(&mut passes);
    passes
}

/// The source ids (`ll.op_id`) of the ops of a snapshot, by position.
fn ids_by_position(text: &str) -> HashMap<(u32, u32), u32> {
    let mut ctx = context();
    let root = walk::parse(&mut ctx, text).unwrap();
    walk::ops_under(&ctx, root)
        .into_iter()
        .filter_map(|op| Some((walk::position(&ctx, op)?, opmap::op_id(&ctx, op)?)))
        .collect()
}

#[test]
fn cycles_come_back_to_the_source_ops_through_every_pass() {
    unsafe { std::env::set_var("CRABBIT_PROFILE_MAP", "1") };
    let mut ctx = context();
    let (trace, final_root) = traced_compile(&mut ctx);
    assert_eq!(
        unstamped_op_count(&ctx, final_root).unwrap(),
        0,
        "every final op is attributed"
    );
    let final_derived: Vec<i64> = walk::ops_under(&ctx, final_root)
        .into_iter()
        .filter_map(|op| opmap::derived_from(&ctx, op))
        .collect();
    assert!(
        !final_derived.contains(&opmap::roots::UNATTRIBUTED),
        "no machine op may stay unattributed"
    );
    assert!(
        final_derived.contains(&opmap::roots::ISEL_ABI),
        "argument copies are rooted at the ABI lowering"
    );

    // The boundary ids of the LLVM ops the sdiv was expanded into (the
    // IR instruction selection received names its sources on each op).
    let boundary = trace
        .snapshots
        .iter()
        .position(|s| s.label == "aarch64-op-ids")
        .unwrap();
    let mut bctx = context();
    let broot = walk::parse(&mut bctx, &trace.snapshots[boundary].ir).unwrap();
    let expansion: HashSet<i64> = walk::ops_under(&bctx, broot)
        .into_iter()
        .filter(|&op| {
            opmap::effective_sources(&bctx, op) == vec![3] && opmap::op_id(&bctx, op) != Some(3)
        })
        .filter_map(|op| opmap::op_id(&bctx, op).map(i64::from))
        .collect();
    assert!(!expansion.is_empty(), "the sdiv is expanded into new ops");

    // Synthetic measurement: 600 cycles on the merged add's machine code,
    // 500 on the divide's expansion.
    let (add_cycles, div_cycles) = (600.0, 500.0);
    let passes = backward_passes();
    let result = walk::walk(&trace, &passes, &mut |ctx, root| {
        let ops = walk::ops_under(ctx, root);
        let on = |want: &dyn Fn(i64) -> bool| -> Vec<_> {
            ops.iter()
                .copied()
                .filter(|&op| opmap::derived_from(ctx, op).is_some_and(want))
                .collect()
        };
        let adds = on(&|d| d == 0);
        let divs = on(&|d| expansion.contains(&d));
        assert!(!adds.is_empty() && !divs.is_empty(), "machine code of both");
        for &op in &adds {
            profile::add(ctx, op, CYCLES, add_cycles / adds.len() as f64);
        }
        for &op in &divs {
            profile::add(ctx, op, CYCLES, div_cycles / divs.len() as f64);
        }
        Ok(())
    })
    .unwrap();

    // The total holds at every level the profile reaches.
    for (index, ops) in &result.snapshots {
        let total: f64 = ops
            .iter()
            .flat_map(|op| op.metrics.iter())
            .filter(|(name, _)| name.as_str() == CYCLES || name.ends_with(".cycles"))
            .map(|(_, value)| value)
            .sum();
        assert!(
            (total - (add_cycles + div_cycles)).abs() < 1e-6,
            "snapshot {index} ({}): {total} cycles",
            trace.snapshots[*index].label
        );
    }

    // At the source level (the first snapshot with ids): the adds split
    // the merged add's cycles, the sdiv has its expansion's.
    let source = trace
        .snapshots
        .iter()
        .position(|s| s.label == "llvm-op-ids")
        .unwrap();
    let ids = ids_by_position(&trace.snapshots[source].ir);
    let by_id: HashMap<u32, f64> = result.snapshots[&source]
        .iter()
        .filter_map(|op| Some((*ids.get(&(op.line, op.column))?, *op.metrics.get(CYCLES)?)))
        .collect();
    let near = |id: u32, want: f64| by_id.get(&id).is_some_and(|got| (got - want).abs() < 1e-6);
    assert!(
        near(0, add_cycles / 2.0),
        "add1 gets half the merged add: {by_id:?}"
    );
    assert!(
        near(1, add_cycles / 2.0),
        "add2 gets the other half: {by_id:?}"
    );
    assert!(
        near(3, div_cycles),
        "the sdiv gets its expansion: {by_id:?}"
    );
}

#[test]
fn perf_samples_land_on_the_instruction_at_their_offset() {
    unsafe { std::env::set_var("CRABBIT_PROFILE_MAP", "1") };
    let mut ctx = context();
    let (_, root) = traced_compile(&mut ctx);
    let script = "\
armv8_pmuv3_1/cycles/u:      aaaa00001000 audit+0x0
armv8_pmuv3_1/cycles/u:      aaaa00001000 audit+0x0
      cycles:u:      aaaa00001004 audit+0x4
instructions:u:      aaaa00001000 audit+0x0
cycles:u:      ffff00000000 [unknown]
";
    let samples = Samples::from_perf_script(script, 10);
    assert_eq!(samples.unknown, 1);
    let attached = attach(&ctx, root, &samples);
    assert_eq!(attached.matched.get("cycles"), Some(&3));

    // The function's first two instructions, in layout order.
    let func = walk::ops_under(&ctx, root)
        .into_iter()
        .find(|&op| {
            pliron::operation::Operation::get_op_dyn(op, &ctx)
                .downcast_ref::<pliron_ll::dialects::aarch64::ops::FuncOp>()
                .is_some()
        })
        .unwrap();
    let instructions: Vec<_> = walk::ops_under(&ctx, func)
        .into_iter()
        .skip(1)
        .filter(|&op| {
            let obj = pliron::operation::Operation::get_op_dyn(op, &ctx);
            pliron::op::op_cast::<
                dyn pliron_ll::dialects::aarch64::op_interfaces::BinarySerializableOpInterface,
            >(&*obj)
            .is_some_and(|s| s.byte_len(&ctx) > 0)
        })
        .collect();
    let cycles = |op| profile::get(&ctx, op).and_then(|p| p.0.get(CYCLES).copied());
    assert_eq!(cycles(instructions[0]), Some(20.0));
    assert_eq!(cycles(instructions[1]), Some(10.0));
    assert_eq!(cycles(instructions[2]), None);
}

#[test]
fn timeline_puts_each_cycle_sample_on_its_ops_line() {
    unsafe { std::env::set_var("CRABBIT_PROFILE_MAP", "1") };
    let mut ctx = context();
    let (trace, _) = traced_compile(&mut ctx);
    // The last snapshot as the tool reads it: parsed, its ops at their lines.
    let mut parsed = context();
    let root = walk::parse(&mut parsed, &trace.snapshots.last().unwrap().ir).unwrap();
    let script = "\
100.000001000: armv8_pmuv3_1/cycles/u:      aaaa00001000 audit+0x0
100.000002500: armv8_pmuv3_1/instructions/u:      aaaa00001000 audit+0x0
100.000003000:       cycles:u:      aaaa00001004 audit+0x4
100.000004000:       cycles:u:      ffff00000000 [unknown]
";
    let samples = Samples::from_perf_script(script, 10);
    let timeline = timeline(&parsed, root, &samples);
    assert_eq!(timeline.start, 100_000_001_000);
    // Only cycle samples, each after the one before.
    assert_eq!(timeline.time, vec![0, 2000, 1000]);
    let (first, second, unknown) = (timeline.line[0], timeline.line[1], timeline.line[2]);
    assert!(first > 0 && second > first, "{:?}", timeline.line);
    assert_eq!(unknown, 0);
}

#[test]
fn profile_consumers_read_a_profiled_trace() {
    use pliron_ll::passes::aarch64::backward::{
        ALLOCATOR_INPUT, profiled_snapshot, samples_by_derived,
    };
    unsafe { std::env::set_var("CRABBIT_PROFILE_MAP", "1") };
    let mut ctx = context();
    let (trace, _) = traced_compile(&mut ctx);
    // 1000 cycles and 50 runs on every machine op of the add.
    let result = walk::walk(&trace, &backward_passes(), &mut |ctx, root| {
        for op in walk::ops_under(ctx, root) {
            if opmap::derived_from(ctx, op) == Some(0) {
                profile::add(ctx, op, CYCLES, 1000.0);
                profile::add(ctx, op, pliron_bpgo::EXECUTIONS, 50.0);
            }
        }
        Ok(())
    })
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("audit.bpgo.crt");
    pliron_bpgo::annotate(&trace, &result, serde_json::json!({ "period": 10 }))
        .write(&path)
        .unwrap();

    // The allocator's measured costs: the add's code, in samples.
    let snapshot = profiled_snapshot(&path, ALLOCATOR_INPUT).unwrap();
    let samples = samples_by_derived(&snapshot);
    let audit = &samples["audit"];
    let adds: u64 = audit.get(&0).copied().unwrap_or(0);
    assert!(adds > 0, "{audit:?}");
    assert_eq!(
        audit.values().sum::<u64>(),
        adds,
        "only the add was measured: {audit:?}"
    );

    // Block frequencies: the single-block function runs as often as its entry.
    unsafe { std::env::set_var(pliron_ll::passes::profile_freq::PROFILE_ENV, &path) };
    let blocks = pliron_ll::passes::aarch64::backward::block_executions(&snapshot)["audit"].len();
    assert_eq!(
        pliron_ll::passes::profile_freq::frequencies_for("audit", blocks),
        Some(vec![1.0; blocks])
    );
}
