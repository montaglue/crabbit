//! cuda-oxide's device API in crabbit's importer: the `#[kernel]` macro's
//! symbol convention and the `cuda_device` functions whose bodies are
//! placeholders (`unreachable!`) that the compiler replaces.
//!
//! cuda-oxide's own importer intercepts these calls by their Rust path and
//! builds `nvvm` dialect ops; this does the same for the subset crabbit's
//! kernels use, with the same ops, so `mir-lower` lowers them exactly as it
//! lowers cuda-oxide's output. Shared memory keeps crabbit's convention
//! underneath: a `SharedArray` static is an ordinary global in the `.shared`
//! section (`docs/KERNEL-ABI.md`), its pointer methods are address
//! arithmetic on that global, and `DynamicSharedArray` is the module's
//! `extern .shared` window.

use super::*;
use crate::dialects::nvvm::ops as nvvm;

/// The section a `SharedArray` static lives in.
pub(super) const SHARED_SECTION: &str = ".shared";

/// Whether `name` is a `#[kernel]` entry's symbol.
pub(super) fn is_cuda_oxide_kernel_symbol(name: &str) -> bool {
    name.starts_with(reserved_oxide_symbols::KERNEL_PREFIX)
}

/// The PTX entry name of a `#[kernel]` symbol: the prefix stripped, so the
/// comparison arms (crabbit, LLVM, nvcc) share one `.entry` name.
pub(super) fn cuda_oxide_entry_name(symbol: &str) -> Option<&str> {
    symbol.strip_prefix(reserved_oxide_symbols::KERNEL_PREFIX)
}

const DEVICE_CRATE: &str = "cuda_device";
const SHARED_ARRAY: &str = "SharedArray";
const DYNAMIC_SHARED_ARRAY: &str = "DynamicSharedArray";
const DISJOINT_SLICE: &str = "DisjointSlice";

/// Whether a kernel parameter of type `ty` crosses the launch boundary as
/// its scalar fields, one `.param` each, instead of one aggregate `.param`:
/// slice pointers (`&[T]`, `&mut [T]`, `*const [T]`, `*mut [T]`) as
/// `(ptr, len)`, and `DisjointSlice<T, S>` as `(ptr, len)` followed by its
/// index space's runtime layout. This is cuda-oxide's kernel-entry ABI:
/// its host launchers push a slice's pointer and length as two driver
/// arguments (`cuda_host::push_kernel_device_slice`).
pub(super) fn is_flattened_kernel_param<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
    use rustc_middle::ty::TyKind;
    match runtime_ty(ty).kind() {
        TyKind::Ref(_, pointee, _) | TyKind::RawPtr(pointee, _) => {
            matches!(pointee.kind(), TyKind::Slice(_))
        }
        TyKind::Adt(adt, _) => {
            tcx.crate_name(adt.did().krate).as_str() == DEVICE_CRATE
                && tcx.item_name(adt.did()).as_str() == DISJOINT_SLICE
        }
        _ => false,
    }
}

/// The storage a `SharedArray<T, N, ALIGN>` static needs: `(bytes, align)`,
/// `None` for any other type.
pub(super) fn shared_array_storage<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> Result<Option<(u64, u64)>, String> {
    let rustc_middle::ty::TyKind::Adt(adt, args) = runtime_ty(ty).kind() else {
        return Ok(None);
    };
    if tcx.crate_name(adt.did().krate).as_str() != DEVICE_CRATE
        || tcx.item_name(adt.did()).as_str() != SHARED_ARRAY
    {
        return Ok(None);
    }
    let elem = args.type_at(0);
    let count = args
        .const_at(1)
        .try_to_target_usize(tcx)
        .ok_or("SharedArray length is not a constant")?;
    let requested_align = args
        .const_at(2)
        .try_to_target_usize(tcx)
        .ok_or("SharedArray alignment is not a constant")?;
    let elem_size = layout_size_of_ty(tcx, elem)?;
    let elem_align = layout_align_of_ty(tcx, elem)?;
    Ok(Some((
        elem_size * count,
        elem_align.max(requested_align).max(1),
    )))
}

/// (crate, type name) of an ADT type.
fn adt_ident<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<(String, String)> {
    match runtime_ty(ty).kind() {
        rustc_middle::ty::TyKind::Adt(adt, _) => Some((
            tcx.crate_name(adt.did().krate).to_string(),
            tcx.item_name(adt.did()).to_string(),
        )),
        _ => None,
    }
}

/// How a cuda-device item is identified: by the crate, the item's own
/// name, and either the module it sits in or the type it is a method of.
/// `def_path_str` is not used because the crate re-exports most items at
/// its root and the printed path follows the re-export.
struct DeviceItem {
    name: String,
    /// `Some("thread")` for `cuda_device::thread::threadIdx_x`.
    module: Option<String>,
    /// `Some("SharedArray")` for a method of `SharedArray<..>`.
    self_type: Option<String>,
}

fn device_item<'tcx>(tcx: TyCtxt<'tcx>, def_id: rustc_hir::def_id::DefId) -> Option<DeviceItem> {
    if tcx.crate_name(def_id.krate).as_str() != DEVICE_CRATE {
        return None;
    }
    let name = tcx.item_name(def_id).to_string();
    let parent = tcx.parent(def_id);
    let (module, self_type) = match tcx.def_kind(parent) {
        rustc_hir::def::DefKind::Impl { .. } => {
            let self_ty = tcx
                .type_of(parent)
                .instantiate_identity()
                .skip_normalization();
            (None, adt_ident(tcx, self_ty).map(|(_, name)| name))
        }
        rustc_hir::def::DefKind::Mod => (Some(tcx.item_name(parent).to_string()), None),
        _ => (None, None),
    };
    Some(DeviceItem {
        name,
        module,
        self_type,
    })
}

/// Lower a call to cuda-device's API. Returns `true` when the call was
/// handled and only the branch to the return target remains.
#[allow(clippy::too_many_arguments)]
pub(super) fn lower_cuda_device_call<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    func: &Operand<'tcx>,
    args: &[rustc_span::Spanned<Operand<'tcx>>],
    destination: &Place<'tcx>,
) -> Result<bool, String> {
    // Only device code is rewritten; the host copy of a device helper keeps
    // cuda-device's placeholder bodies (never run).
    if !is_device_module(ctx, state.module_body) {
        return Ok(false);
    }
    let Some(instance) = call_instance(tcx, state, body, func) else {
        return Ok(false);
    };
    if !matches!(instance.def, InstanceKind::Item(_)) {
        return Ok(false);
    }
    let Some(item) = device_item(tcx, instance.def_id()) else {
        return Ok(false);
    };
    let path = tcx.def_path_str(instance.def_id());
    let operand = |ctx: &mut Context, index: usize| -> Result<Value, String> {
        let Some(arg) = args.get(index) else {
            return Err(format!("{path}: missing argument {index}"));
        };
        import_operand(tcx, ctx, state, insert_block, body, &arg.node)
    };
    let store = |ctx: &mut Context, value: Value| -> Result<(), String> {
        let want = destination_storage_ty(tcx, ctx, state, body, destination)?;
        let value = match want {
            Some(ty) => cast_value_to_type(ctx, insert_block, value, ty),
            None => value,
        };
        store_place(tcx, ctx, state, insert_block, body, destination, value)
    };
    let module = item.module.as_deref();
    let name = item.name.as_str();

    // Special registers (`cuda_device::thread::threadIdx_x` and friends).
    if module == Some("thread") {
        let result_ty = u32_ty(ctx);
        if let Some(op) = sreg_op(ctx, name, result_ty) {
            op.insert_at_back(insert_block, ctx);
            let value = op.deref(ctx).get_result(0);
            store(ctx, value)?;
            return Ok(true);
        }
        if name == "sync_threads" {
            let op = Operation::new(
                ctx,
                nvvm::Barrier0Op::get_concrete_op_info(),
                vec![],
                vec![],
                vec![],
                0,
            );
            op.insert_at_back(insert_block, ctx);
            return Ok(true);
        }
    }

    if module == Some("warp") && name == "lane_id" {
        let result_ty = u32_ty(ctx);
        let op = Operation::new(
            ctx,
            nvvm::ReadPtxSregLaneIdOp::get_concrete_op_info(),
            vec![result_ty],
            vec![],
            vec![],
            0,
        );
        op.insert_at_back(insert_block, ctx);
        let value = op.deref(ctx).get_result(0);
        store(ctx, value)?;
        return Ok(true);
    }
    if module == Some("warp") && name == "sync_mask" {
        let mask = operand(ctx, 0)?;
        let op = nvvm::BarWarpSyncOp::build(ctx, mask);
        op.insert_at_back(insert_block, ctx);
        return Ok(true);
    }

    // Scalar float math (`cuda_device::float::ex2_approx_f32` and friends).
    if module == Some("float") {
        use crate::dialects::nvvm::ops::{
            ScalarMathFormatAttr as Format, ScalarMathOperationAttr as Op,
            ScalarMathPrecisionAttr as Precision, ScalarMathSubnormalAttr as Subnormal,
        };
        let scalar = match name {
            "ex2_approx_f32" => Some((Op::Ex2, Precision::Approx)),
            "lg2_approx_f32" => Some((Op::Lg2, Precision::Approx)),
            "rsqrt_approx_f32" => Some((Op::Rsqrt, Precision::Approx)),
            "rcp_approx_f32" => Some((Op::Rcp, Precision::Approx)),
            "sin_approx_f32" => Some((Op::Sin, Precision::Approx)),
            "cos_approx_f32" => Some((Op::Cos, Precision::Approx)),
            "sqrt_rn_f32" => Some((Op::Sqrt, Precision::Rn)),
            "sqrt_approx_f32" => Some((Op::Sqrt, Precision::Approx)),
            _ => None,
        };
        if let Some((operation, precision)) = scalar {
            let x = operand(ctx, 0)?;
            let op = nvvm::ScalarMathOp::build(
                ctx,
                x,
                Format::F32,
                operation,
                precision,
                Subnormal::Preserve,
            );
            op.insert_at_back(insert_block, ctx);
            let value = op.deref(ctx).get_result(0);
            store(ctx, value)?;
            return Ok(true);
        }
        if name == "fma_rn_f32" {
            let a = operand(ctx, 0)?;
            let b = operand(ctx, 1)?;
            let c = operand(ctx, 2)?;
            let value = device_placeholder_call(
                ctx,
                insert_block,
                dialect_mir_placeholder("fmaf32"),
                vec![a, b, c],
                Some(a.get_type(ctx)),
            );
            store(ctx, value.expect("fma has a result"))?;
            return Ok(true);
        }
    }

    // Warp shuffles (`_sync` forms: mask, value, lane or delta) and votes.
    if module == Some("warp") {
        let shuffle: Option<fn(&mut Context, Value, Value, Value) -> Ptr<Operation>> = match name {
            "shuffle_down_f32_sync" => Some(nvvm::ShflSyncDownF32Op::build),
            "shuffle_up_f32_sync" => Some(nvvm::ShflSyncUpF32Op::build),
            "shuffle_xor_f32_sync" => Some(nvvm::ShflSyncBflyF32Op::build),
            "shuffle_f32_sync" => Some(nvvm::ShflSyncIdxF32Op::build),
            "shuffle_down_sync" => Some(nvvm::ShflSyncDownI32Op::build),
            "shuffle_up_sync" => Some(nvvm::ShflSyncUpI32Op::build),
            "shuffle_xor_sync" => Some(nvvm::ShflSyncBflyI32Op::build),
            "shuffle_sync" => Some(nvvm::ShflSyncIdxI32Op::build),
            _ => None,
        };
        if let Some(build) = shuffle {
            let mask = operand(ctx, 0)?;
            let value = operand(ctx, 1)?;
            let lane = operand(ctx, 2)?;
            let op = build(ctx, mask, value, lane);
            op.insert_at_back(insert_block, ctx);
            let value = op.deref(ctx).get_result(0);
            store(ctx, value)?;
            return Ok(true);
        }
        let vote: Option<fn(&mut Context, Value, Value) -> Ptr<Operation>> = match name {
            "ballot_sync" => Some(nvvm::VoteSyncBallotOp::build),
            "all_sync" => Some(nvvm::VoteSyncAllOp::build),
            "any_sync" => Some(nvvm::VoteSyncAnyOp::build),
            _ => None,
        };
        if let Some(build) = vote {
            let mask = operand(ctx, 0)?;
            let predicate = operand(ctx, 1)?;
            let i1: TypeHandle = IntegerType::get(ctx, 1, Signedness::Signless).into();
            let predicate = cast_value_to_type(ctx, insert_block, predicate, i1);
            let op = build(ctx, mask, predicate);
            op.insert_at_back(insert_block, ctx);
            let value = op.deref(ctx).get_result(0);
            store(ctx, value)?;
            return Ok(true);
        }
    }

    // Atomics: `DeviceAtomicI32::fetch_add(&self, val, ordering)` and the
    // other read-modify-write methods, on the `#[repr(transparent)]` cell
    // (the receiver is the pointer).
    if let Some(self_type) = item.self_type.as_deref()
        && self_type.starts_with("DeviceAtomic")
        && let Some(kind) = atomic_rmw_kind(name)
    {
        use crate::dialects::nvvm::ops::{AtomicOrdering, AtomicScope};
        let ptr = operand(ctx, 0)?;
        let val = operand(ctx, 1)?;
        let ordering = match args
            .get(2)
            .and_then(|arg| constant_discriminant(tcx, state, body, &arg.node))
        {
            Some(0) => AtomicOrdering::Relaxed,
            Some(1) => AtomicOrdering::Acquire,
            Some(2) => AtomicOrdering::Release,
            Some(3) => AtomicOrdering::AcqRel,
            Some(_) => AtomicOrdering::SeqCst,
            None => return Err(format!("{path}: the ordering must be a constant")),
        };
        let result_ty = val.get_type(ctx);
        let op = nvvm::NvvmAtomicRmwOp::build(
            ctx,
            ptr,
            val,
            result_ty,
            kind,
            ordering,
            AtomicScope::Device,
        )
        .get_operation();
        op.insert_at_back(insert_block, ctx);
        let value = op.deref(ctx).get_result(0);
        store(ctx, value)?;
        return Ok(true);
    }

    // `SharedArray` pointer methods and indexing: address arithmetic on the
    // static's global (the receiver is a pointer to it).
    if item.self_type.as_deref() == Some(SHARED_ARRAY) {
        let receiver_ty = mono_ty(
            tcx,
            state,
            args.first()
                .map(|a| a.node.ty(body, tcx))
                .ok_or("SharedArray method without receiver")?,
        );
        let elem_ty = shared_array_elem_ty(tcx, receiver_ty)
            .ok_or_else(|| format!("{path}: receiver is not a pointer to SharedArray"))?;
        match name {
            "as_ptr" | "as_mut_ptr" | "as_raw_mut_ptr" => {
                let base = operand(ctx, 0)?;
                store(ctx, base)?;
                return Ok(true);
            }
            "index" | "index_mut" => {
                let base = operand(ctx, 0)?;
                let index = operand(ctx, 1)?;
                let elem_size = layout_size_of_ty(tcx, elem_ty)?;
                let scaled = scale_index(ctx, insert_block, index, elem_size)?;
                let addr = mir_dialect::ops::PtrOffsetOp::new(ctx, base, scaled);
                addr.get_operation().insert_at_back(insert_block, ctx);
                store(ctx, addr.get_result(ctx))?;
                return Ok(true);
            }
            _ => return Err(format!("unsupported cuda_device call: {path}")),
        }
    }

    // `DynamicSharedArray`: the launch-sized `extern .shared` window.
    if item.self_type.as_deref() == Some(DYNAMIC_SHARED_ARRAY) {
        let base = dynamic_shared_base(ctx, state.module_body, insert_block);
        match name {
            "get" | "get_raw" => {
                store(ctx, base)?;
                return Ok(true);
            }
            "offset" => {
                let byte_offset = operand(ctx, 0)?;
                let addr = mir_dialect::ops::PtrOffsetOp::new(ctx, base, byte_offset);
                addr.get_operation().insert_at_back(insert_block, ctx);
                store(ctx, addr.get_result(ctx))?;
                return Ok(true);
            }
            _ => return Err(format!("unsupported cuda_device call: {path}")),
        }
    }

    // Anything else under `cuda_device::` is ordinary Rust (index helpers,
    // launch-contract types) and imports as a call.
    Ok(false)
}

/// Whether `module_body` is the body of the `rust_kernels` module: code that
/// runs on the device.
pub(super) fn is_device_module(ctx: &Context, module_body: Ptr<BasicBlock>) -> bool {
    module_body
        .deref(ctx)
        .get_parent_op(ctx)
        .and_then(|op| Operation::get_op::<builtin::ops::ModuleOp>(op, ctx))
        .is_some_and(|module| module.get_symbol_name(ctx).as_ref() == "rust_kernels")
}

/// The dialect-mir placeholder symbol for a core float/integer intrinsic
/// (`dialect_mir::rust_intrinsics`): `mir-lower` lowers the call to the
/// device instruction, and the dialect's verifier knows the schema.
pub(super) fn dialect_mir_placeholder(suffix: &str) -> crate::identifier::Identifier {
    format!(
        "{}{suffix}",
        crate::dialects::dialect_mir::rust_intrinsics::PLACEHOLDER_PREFIX
    )
    .try_into()
    .expect("placeholder names are legal identifiers")
}

/// The placeholder a core intrinsic maps to on the device, by rustc's
/// intrinsic name.
pub(super) fn device_core_intrinsic_placeholder(name: &str) -> Option<&'static str> {
    Some(match name {
        "sqrtf32" => "sqrtf32",
        "sqrtf64" => "sqrtf64",
        "fmaf32" => "fmaf32",
        "fmaf64" => "fmaf64",
        "exp2f32" => "exp2f32",
        "exp2f64" => "exp2f64",
        "expf32" => "expf32",
        "log2f32" => "log2f32",
        "logf32" => "logf32",
        "maxnumf32" => "maximum_number_nsz_f32",
        "maxnumf64" => "maximum_number_nsz_f64",
        "minnumf32" => "minimum_number_nsz_f32",
        "minnumf64" => "minimum_number_nsz_f64",
        "floorf32" => "floorf32",
        "ceilf32" => "ceilf32",
        "truncf32" => "truncf32",
        "roundf32" => "roundf32",
        "fabsf32" | "fabsf64" => "fabs",
        _ => return None,
    })
}

/// A `mir.call` to a dialect-mir placeholder (inserted).
pub(super) fn device_placeholder_call(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    callee: crate::identifier::Identifier,
    args: Vec<Value>,
    result_ty: Option<TypeHandle>,
) -> Option<Value> {
    let call = mir_dialect::ops::CallOp::new_direct(ctx, callee, args, result_ty);
    call.get_operation().insert_at_back(insert_block, ctx);
    result_ty.map(|_| call.get_result(ctx))
}

fn atomic_rmw_kind(name: &str) -> Option<crate::dialects::nvvm::ops::AtomicRmwKind> {
    use crate::dialects::nvvm::ops::AtomicRmwKind as Kind;
    Some(match name {
        "fetch_add" => Kind::Add,
        "fetch_sub" => Kind::Sub,
        "fetch_and" => Kind::And,
        "fetch_or" => Kind::Or,
        "fetch_xor" => Kind::Xor,
        "fetch_min" => Kind::Min,
        "fetch_max" => Kind::Max,
        "swap" => Kind::Xchg,
        _ => return None,
    })
}

/// The discriminant of a constant fieldless-enum operand (`AtomicOrdering`).
fn constant_discriminant<'tcx>(
    tcx: TyCtxt<'tcx>,
    state: &FunctionImportState<'tcx>,
    body: &Body<'tcx>,
    operand: &Operand<'tcx>,
) -> Option<u128> {
    let Operand::Constant(constant) = operand else {
        return None;
    };
    let constant = mono_const(tcx, state, constant.const_);
    let ty = mono_ty(tcx, state, operand.ty(body, tcx));
    let value =
        constant.try_eval_scalar_int(tcx, rustc_middle::ty::TypingEnv::fully_monomorphized())?;
    let rustc_middle::ty::TyKind::Adt(adt, _) = ty.kind() else {
        return Some(value.to_bits_unchecked());
    };
    // A fieldless enum constant is its tag; map it to the variant index.
    let tag = value.to_bits_unchecked();
    adt.discriminants(tcx)
        .find(|(_, discr)| discr.val == tag)
        .map(|(index, _)| index.as_u32() as u128)
        .or(Some(tag))
}

fn u32_ty(ctx: &mut Context) -> TypeHandle {
    IntegerType::get(ctx, 32, Signedness::Unsigned).into()
}

/// The storage type of `destination`, `None` for a ZST.
fn destination_storage_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    body: &Body<'tcx>,
    destination: &Place<'tcx>,
) -> Result<Option<TypeHandle>, String> {
    let ty = mono_ty(tcx, state, destination.ty(body, tcx).ty);
    convert_storage_ty(tcx, ctx, ty)
}

/// `T` of a pointer or reference to `SharedArray<T, N, ALIGN>`.
fn shared_array_elem_ty<'tcx>(tcx: TyCtxt<'tcx>, receiver: Ty<'tcx>) -> Option<Ty<'tcx>> {
    let pointee = match runtime_ty(receiver).kind() {
        rustc_middle::ty::TyKind::Ref(_, inner, _) | rustc_middle::ty::TyKind::RawPtr(inner, _) => {
            *inner
        }
        _ => receiver,
    };
    match runtime_ty(pointee).kind() {
        rustc_middle::ty::TyKind::Adt(adt, args)
            if tcx.crate_name(adt.did().krate).as_str() == DEVICE_CRATE
                && tcx.item_name(adt.did()).as_str() == SHARED_ARRAY =>
        {
            Some(args.type_at(0))
        }
        _ => None,
    }
}

/// The `nvvm.read_ptx_sreg_*` op (detached) for a `cuda_device::thread`
/// accessor name.
fn sreg_op(ctx: &mut Context, name: &str, result_ty: TypeHandle) -> Option<Ptr<Operation>> {
    macro_rules! sreg {
        ($op:ident) => {
            Operation::new(
                ctx,
                nvvm::$op::get_concrete_op_info(),
                vec![result_ty],
                vec![],
                vec![],
                0,
            )
        };
    }
    Some(match name {
        "threadIdx_x" => sreg!(ReadPtxSregTidXOp),
        "threadIdx_y" => sreg!(ReadPtxSregTidYOp),
        "threadIdx_z" => sreg!(ReadPtxSregTidZOp),
        "blockDim_x" => sreg!(ReadPtxSregNtidXOp),
        "blockDim_y" => sreg!(ReadPtxSregNtidYOp),
        "blockDim_z" => sreg!(ReadPtxSregNtidZOp),
        "blockIdx_x" => sreg!(ReadPtxSregCtaidXOp),
        "blockIdx_y" => sreg!(ReadPtxSregCtaidYOp),
        "blockIdx_z" => sreg!(ReadPtxSregCtaidZOp),
        "gridDim_x" => sreg!(ReadPtxSregNctaidXOp),
        "gridDim_y" => sreg!(ReadPtxSregNctaidYOp),
        "gridDim_z" => sreg!(ReadPtxSregNctaidZOp),
        _ => return None,
    })
}

/// A call to crabbit's `crabbit.dyn.shared.base` intrinsic (docs/KERNEL-ABI.md).
fn dynamic_shared_base(
    ctx: &mut Context,
    module_body: Ptr<BasicBlock>,
    insert_block: Ptr<BasicBlock>,
) -> Value {
    let symbol: crate::identifier::Identifier = "crabbit_dyn_shared_base"
        .try_into()
        .expect("static identifier literal");
    let ptr = ptr_ty(ctx);
    declare_external_function(ctx, module_body, symbol.clone(), vec![], Some(ptr));
    let call = mir_dialect::ops::CallOp::new_direct(ctx, symbol, vec![], Some(ptr));
    call.get_operation().insert_at_back(insert_block, ctx);
    call.get_result(ctx)
}
