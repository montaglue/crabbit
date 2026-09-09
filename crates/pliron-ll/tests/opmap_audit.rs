//! E3-lite attribution-fidelity audit (docs/PROFILE-FEEDBACK-BACKWARD.md):
//! samples synthesized onto a known machine-op range must lift onto the
//! correct LLVM-level op id through the real pipeline + the real
//! ingestion tool. Also the "every final op is attributed" audit.

use pliron::builtin::op_interfaces::{OneRegionInterface, OneResultInterface};
use pliron_ll::conversion::pass::{AnalysisManager, Passes};
use pliron_ll::dialects::builtin::ops::ModuleOp;
use pliron_ll::dialects::builtin::types::{IntegerType, Signedness};
use pliron_ll::dialects::llvm::op_interfaces::IntBinArithOpWithOverflowFlag as _;
use pliron_ll::dialects::llvm::{
    attributes::LinkageAttr,
    ops::{AddOp, FuncOp, ReturnOp},
    types::FuncType,
};
use pliron_ll::dialects::{aarch64, macho};
use pliron_ll::ir::op::Op as _;
use pliron_ll::linked_list::ContainsLinkedList;
use pliron_ll::passes::aarch64::blockmap::{
    assign_blockmap_ids, blockmap_json_from_ir, collect_blockmap,
};
use pliron_ll::passes::aarch64::opmap::{assign_op_ids, roots, unstamped_op_count};
use pliron_ll::passes::aarch64::{
    TargetOs, aarch64_asm_lower::Aarch64AsmLowerPass,
    aarch64_block_placement::Aarch64BlockPlacementPass,
    aarch64_branch_relax::Aarch64BranchRelaxPass, aarch64_encode::Aarch64EncodePass,
    aarch64_frame_lower::Aarch64FrameLowerPass, aarch64_legalize::Aarch64LegalizePass,
    aarch64_machine_cfg_cleanup::Aarch64MachineCfgCleanupPass,
    aarch64_post_ra_opts::Aarch64PostRaOptsPass,
    aarch64_register_allocate::Aarch64RegisterAllocatePass,
    aarch64_target_opts_pre_ra::Aarch64TargetOptsPreRaPass,
    llvm_aarch64_abi::LlvmAarch64AbiPass, llvm_to_aarch64_isel::LlvmToAarch64IselPass,
    verify_llvm_for_aarch64::VerifyLlvmForAarch64Pass,
};
use pliron::context::Context;

fn context() -> Context {
    let mut ctx = Context::new();
    aarch64::register(&mut ctx);
    macho::register(&mut ctx);
    ctx
}

/// `f(a, b) = a + b` (arguments, so the add cannot constant-fold and its
/// machine instruction is emitted in the add's own attribution bracket):
/// op ids in program order are add=0, ret=1.
fn build_module(ctx: &mut Context) -> ModuleOp {
    let module = ModuleOp::new(ctx, "test".try_into().unwrap());
    let body = module.get_region(ctx).deref(ctx).get_head().unwrap();
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let func_ty = FuncType::get(ctx, i64_ty.into(), vec![i64_ty.into(), i64_ty.into()], false);
    let func = FuncOp::new(ctx, "audited".try_into().unwrap(), func_ty);
    func.set_attr_llvm_function_linkage(ctx, LinkageAttr::ExternalLinkage);
    func.get_or_create_entry_block(ctx);
    func.get_operation().insert_at_back(body, ctx);
    let entry = func.get_entry_block(ctx).unwrap();
    let args: Vec<_> = entry.deref(ctx).arguments().collect();
    let add = AddOp::new_with_overflow_flag(ctx, args[0], args[1], Default::default());
    add.get_operation().insert_at_back(entry, ctx);
    let sum = add.get_result(ctx);
    ReturnOp::new(ctx, Some(sum))
        .get_operation()
        .insert_at_back(entry, ctx);
    module
}

#[test]
fn samples_on_the_add_lift_onto_the_add_llvm_op_id() {
    let mut ctx = context();
    let module = build_module(&mut ctx);
    let root = module.get_operation();
    let mut analyses = AnalysisManager::default();

    // Front half + op-id stamping at the exact pipeline position.
    let mut prefix = Passes::default();
    prefix.add_pass(VerifyLlvmForAarch64Pass::new(TargetOs::Linux));
    prefix.add_pass(LlvmAarch64AbiPass::new(TargetOs::Linux));
    prefix.run(root, &mut ctx, &mut analyses).unwrap();
    assign_op_ids(&mut ctx, root).unwrap();

    let mut middle = Passes::default();
    middle.add_pass(LlvmToAarch64IselPass);
    middle.add_pass(Aarch64LegalizePass);
    middle.add_pass(Aarch64MachineCfgCleanupPass);
    middle.add_pass(Aarch64TargetOptsPreRaPass);
    middle.run(root, &mut ctx, &mut analyses).unwrap();
    assign_blockmap_ids(&mut ctx, root).unwrap();

    let mut suffix = Passes::default();
    suffix.add_pass(Aarch64RegisterAllocatePass);
    suffix.add_pass(Aarch64FrameLowerPass);
    suffix.add_pass(Aarch64PostRaOptsPass);
    suffix.add_pass(Aarch64BlockPlacementPass);
    suffix.add_pass(Aarch64BranchRelaxPass);
    suffix.add_pass(Aarch64AsmLowerPass);
    suffix.add_pass(Aarch64EncodePass);
    suffix.run(root, &mut ctx, &mut analyses).unwrap();

    // Audit: every final machine op carries an attribution.
    assert_eq!(
        unstamped_op_count(&ctx, root).unwrap(),
        0,
        "every op in the final IR must be attributed"
    );

    // The add (llvm op id 0) has exactly one machine range; the ABI
    // argument copies are isel:abi-rooted.
    let map = collect_blockmap(&ctx, root).unwrap();
    let ranges = &map["audited"];
    let ops: Vec<_> = ranges.iter().flat_map(|range| range.ops.iter()).collect();
    assert!(!ops.is_empty(), "op stamping must produce op ranges");
    assert!(
        ops.iter()
            .all(|op| op.derived_from != roots::UNATTRIBUTED),
        "no op may stay unattributed: {ops:?}"
    );
    assert!(
        ops.iter().any(|op| op.derived_from == roots::ISEL_ABI),
        "argument copies must be isel:abi-rooted: {ops:?}"
    );
    for source in [0i64, 1] {
        assert!(
            ops.iter().any(|op| op.derived_from == source),
            "llvm op id {source} lost its machine code: {ops:?}"
        );
    }
    let add_ranges: Vec<_> = ops.iter().filter(|op| op.derived_from == 0).collect();
    assert_eq!(add_ranges.len(), 1, "one add instruction: {add_ranges:?}");
    let add_range = add_ranges[0];

    // Synthesize perf-script samples inside the add's byte range and run
    // the real ingestion tool.
    let json = blockmap_json_from_ir(&ctx, root).expect("sidecar payload");
    let dir = std::env::temp_dir().join(format!(
        "opmap-audit-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let blockmap_path = dir.join("audit.blockmap.json");
    std::fs::write(&blockmap_path, &json).unwrap();
    let samples = 5u64;
    let mut script = String::new();
    for index in 0..samples {
        // Function base is 0 (single function), so symbol offset ==
        // absolute offset; stay strictly inside the range.
        let offset = add_range.start + (index % (add_range.end - add_range.start));
        script.push_str(&format!("\t0000aaaa0000 audited+0x{offset:x}\n"));
    }
    let script_path = dir.join("audit.perf-script.txt");
    std::fs::write(&script_path, script).unwrap();
    let profile_path = dir.join("profile.json");

    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let ingest = repo_root.join("scripts/perf-harness/profile_ingest.py");
    let output = match std::process::Command::new("python3")
        .arg(&ingest)
        .arg("--blockmap")
        .arg(&blockmap_path)
        .arg("--perf-script")
        .arg(&script_path)
        .arg("-o")
        .arg(&profile_path)
        .output()
    {
        Ok(output) => output,
        Err(error) => {
            eprintln!("skipping ingest leg: python3 unavailable ({error})");
            let _ = std::fs::remove_dir_all(&dir);
            return;
        }
    };
    assert!(
        output.status.success(),
        "ingest failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let op_costs: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("profile.op_costs.json")).unwrap(),
    )
    .unwrap();
    // The backward hop: every synthesized sample lands on llvm op id 2.
    assert_eq!(
        op_costs["audited"]["lifted"]["0"],
        serde_json::json!(samples),
        "op_costs: {op_costs}"
    );
    assert_eq!(
        op_costs["audited"]["lifted"]
            .as_object()
            .unwrap()
            .values()
            .filter_map(|value| value.as_u64())
            .sum::<u64>(),
        samples
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// E3 proper (docs/PROFILE-FEEDBACK-BACKWARD.md): the TWO-hop lift.
/// A function with a GVN-dedup pair and a constant divide goes through the
/// real mid-end (head stamping + adjoints) and the machine pipeline;
/// synthetic samples on (a) the surviving merged add and (b) one op of the
/// divide's magic expansion must land, at source level, on (a) a 50/50
/// split across BOTH original adds and (b) the original sdiv op id.
#[test]
fn two_hop_lift_recovers_source_ops_with_gvn_split() {
    use pliron_ll::dialects::builtin::attributes::IntegerAttr;
    use pliron_ll::dialects::builtin::ops::ConstantOp;
    use pliron_ll::dialects::llvm::op_interfaces::BinArithOp as _;
    use pliron_ll::dialects::llvm::ops::SDivOp;
    use pliron_ll::passes::aarch64::opmap::assign_boundary_ids;
    use pliron_ll::passes::llvm::add_llvm_midend_passes;
    use pliron_ll::target_profile::TargetProfile;
    use pliron_ll::utils::apint::APInt;
    use std::num::NonZero;

    unsafe { std::env::set_var("CRABBIT_PROFILE_MAP", "1") };

    let mut ctx = context();
    let module = ModuleOp::new(&mut ctx, "test".try_into().unwrap());
    let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
    let i64_ty = IntegerType::get(&mut ctx, 64, Signedness::Signless);
    let func_ty = FuncType::get(
        &mut ctx,
        i64_ty.into(),
        vec![i64_ty.into(), i64_ty.into()],
        false,
    );
    let func = FuncOp::new(&mut ctx, "audit2".try_into().unwrap(), func_ty);
    func.set_attr_llvm_function_linkage(&ctx, LinkageAttr::ExternalLinkage);
    func.get_or_create_entry_block(&mut ctx);
    func.get_operation().insert_at_back(body, &ctx);
    let entry = func.get_entry_block(&ctx).unwrap();
    let args: Vec<_> = entry.deref(&ctx).arguments().collect();
    // Source ids in program order: add1=0, add2=1, const3=2, sdiv=3,
    // add_ret=4, ret=5.
    let add1 = AddOp::new_with_overflow_flag(&mut ctx, args[0], args[1], Default::default());
    add1.get_operation().insert_at_back(entry, &ctx);
    let add2 = AddOp::new_with_overflow_flag(&mut ctx, args[0], args[1], Default::default());
    add2.get_operation().insert_at_back(entry, &ctx);
    let three = ConstantOp::new(
        &mut ctx,
        Box::new(IntegerAttr::new(
            i64_ty,
            APInt::from_u64(3, NonZero::new(64).unwrap()),
        )),
    );
    three.get_operation().insert_at_back(entry, &ctx);
    let add1_v = add1.get_result(&ctx);
    let add2_v = add2.get_result(&ctx);
    let three_v = three.get_result(&ctx);
    let sdiv = SDivOp::new(&mut ctx, add1_v, three_v);
    sdiv.get_operation().insert_at_back(entry, &ctx);
    let sdiv_v = sdiv.get_result(&ctx);
    let add_ret =
        AddOp::new_with_overflow_flag(&mut ctx, sdiv_v, add2_v, Default::default());
    add_ret.get_operation().insert_at_back(entry, &ctx);
    let add_ret_v = add_ret.get_result(&ctx);
    ReturnOp::new(&mut ctx, Some(add_ret_v))
        .get_operation()
        .insert_at_back(entry, &ctx);

    let root = module.get_operation();
    let mut analyses = AnalysisManager::default();

    // Real mid-end (head stamping + adjoints), then the machine front.
    let mut midend = Passes::default();
    add_llvm_midend_passes(&mut midend, &TargetProfile::host_cpu());
    midend.run(root, &mut ctx, &mut analyses).unwrap();

    let mut prefix = Passes::default();
    prefix.add_pass(VerifyLlvmForAarch64Pass::new(TargetOs::Linux));
    prefix.add_pass(LlvmAarch64AbiPass::new(TargetOs::Linux));
    prefix.run(root, &mut ctx, &mut analyses).unwrap();
    assign_boundary_ids(&mut ctx, root).unwrap();

    let mut middle = Passes::default();
    middle.add_pass(LlvmToAarch64IselPass);
    middle.add_pass(Aarch64LegalizePass);
    middle.add_pass(Aarch64MachineCfgCleanupPass);
    middle.add_pass(Aarch64TargetOptsPreRaPass);
    middle.run(root, &mut ctx, &mut analyses).unwrap();
    assign_blockmap_ids(&mut ctx, root).unwrap();

    let mut suffix = Passes::default();
    suffix.add_pass(Aarch64RegisterAllocatePass);
    suffix.add_pass(Aarch64FrameLowerPass);
    suffix.add_pass(Aarch64PostRaOptsPass);
    suffix.add_pass(Aarch64BlockPlacementPass);
    suffix.add_pass(Aarch64BranchRelaxPass);
    suffix.add_pass(Aarch64AsmLowerPass);
    suffix.add_pass(Aarch64EncodePass);
    suffix.run(root, &mut ctx, &mut analyses).unwrap();

    assert_eq!(unstamped_op_count(&ctx, root).unwrap(), 0);

    let json = blockmap_json_from_ir(&ctx, root).expect("sidecar payload");
    let sidecar: serde_json::Value = serde_json::from_str(&json).unwrap();
    let midend_table = &sidecar["__midend__"]["audit2"];
    assert!(
        midend_table.is_object(),
        "mid-end boundary table must exist: {json}"
    );
    // The surviving add is preserved id 0 with merged parents [0, 1].
    assert_eq!(
        midend_table["0"],
        serde_json::json!([0, 1]),
        "gvn merge adjoint: {midend_table}"
    );
    // Fresh ids mapping to the sdiv's source id 3 (the expansion ops;
    // several — not every one necessarily survives to its own machine
    // instruction, so match any of them below).
    let expansion_ids: Vec<String> = midend_table
        .as_object()
        .unwrap()
        .iter()
        .filter(|(_, v)| v.as_array() == Some(&vec![serde_json::json!(3)]))
        .map(|(k, _)| k.clone())
        .collect();
    assert!(!expansion_ids.is_empty(), "expansion ops derived from the sdiv");

    // Find machine ranges attributed to the add (0) and to the expansion.
    let ranges = sidecar["audit2"].as_array().unwrap();
    let mut add_range = None;
    let mut exp_range = None;
    for block in ranges {
        for op in block["ops"].as_array().into_iter().flatten() {
            let derived = op["derived_from"].as_i64().unwrap();
            if derived == 0 && add_range.is_none() {
                add_range = Some((op["start"].as_u64().unwrap(), op["end"].as_u64().unwrap()));
            }
            if expansion_ids.contains(&derived.to_string()) && exp_range.is_none() {
                exp_range = Some((op["start"].as_u64().unwrap(), op["end"].as_u64().unwrap()));
            }
        }
    }
    let add_range = add_range.expect("a machine op derived from the merged add");
    let exp_range = exp_range.expect("a machine op derived from the expansion");

    let dir = std::env::temp_dir().join(format!(
        "opmap-audit2-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let blockmap_path = dir.join("audit2.blockmap.json");
    std::fs::write(&blockmap_path, &json).unwrap();
    let add_samples = 6u64;
    let exp_samples = 5u64;
    let mut script = String::new();
    for index in 0..add_samples {
        let offset = add_range.0 + (index % (add_range.1 - add_range.0));
        script.push_str(&format!("\t0000aaaa0000 audit2+0x{offset:x}\n"));
    }
    for index in 0..exp_samples {
        let offset = exp_range.0 + (index % (exp_range.1 - exp_range.0));
        script.push_str(&format!("\t0000aaaa0000 audit2+0x{offset:x}\n"));
    }
    let script_path = dir.join("audit2.perf-script.txt");
    std::fs::write(&script_path, script).unwrap();
    let profile_path = dir.join("profile.json");
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let ingest = repo_root.join("scripts/perf-harness/profile_ingest.py");
    let output = match std::process::Command::new("python3")
        .arg(&ingest)
        .arg("--blockmap")
        .arg(&blockmap_path)
        .arg("--perf-script")
        .arg(&script_path)
        .arg("-o")
        .arg(&profile_path)
        .output()
    {
        Ok(output) => output,
        Err(error) => {
            eprintln!("skipping ingest leg: python3 unavailable ({error})");
            let _ = std::fs::remove_dir_all(&dir);
            return;
        }
    };
    assert!(
        output.status.success(),
        "ingest failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let op_costs: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("profile.op_costs.json")).unwrap(),
    )
    .unwrap();
    let source = &op_costs["audit2"]["source"];
    // Hop 2, merge: the merged add's samples split 50/50 across both
    // original adds (source ids 0 and 1).
    assert_eq!(source["0"], serde_json::json!(add_samples as f64 / 2.0), "{op_costs}");
    assert_eq!(source["1"], serde_json::json!(add_samples as f64 / 2.0), "{op_costs}");
    // Hop 2, expansion: the magic-sequence samples land on the sdiv (3).
    assert_eq!(source["3"], serde_json::json!(exp_samples as f64), "{op_costs}");
    let _ = std::fs::remove_dir_all(&dir);
}
