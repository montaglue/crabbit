//! rustc MIR importer targeting cuda-oxide's `dialect-mir` (the `mir`
//! dialect) instead of crabbit's own `cmir`.
//!
//! Forked from [importer](crate::importer); emission sites are being swapped
//! from `cmir` ops to `dialect_mir` ops one construct at a time. Selected at
//! runtime via `CRABBIT_PIPELINE=oxide`. Once every emission site is swapped
//! and the fixtures are green under the oxide pipeline, this file replaces
//! `importer.rs` and the `cmir` dialect is deleted.

use std::num::NonZero;

use rustc_abi::{Size, TagEncoding, VariantIdx, Variants};
use rustc_middle::{
    mir::{
        self as rustc_mir, BasicBlock as RustBasicBlock, BinOp, Body, ConstOperand, Local, Operand,
        Place, Rvalue, StatementKind, TerminatorKind,
    },
    ty::{EarlyBinder, Instance, InstanceKind, Ty, TyCtxt, TypeVisitableExt},
};
use crate::{
    context::{Context, Ptr},
    dialects::{
        aarch64, builtin,
        builtin::op_interfaces::OneRegionInterface,
        builtin::{
            attributes::{FPDoubleAttr, FPSingleAttr, IdentifierAttr, IntegerAttr},
            op_interfaces::{ATTR_KEY_SYM_NAME, SymbolOpInterface},
            types::{FP32Type, FP64Type, FunctionType, IntegerType, Signedness, UnitType},
        },
        llvm::{self, attributes::LinkageAttr},
        macho, x86_64,
    },
    identifier::Legaliser,
    ir::op::{Op, op_cast},
    ir::{
        basic_block::BasicBlock,
        operation::Operation,
        r#type::{TypeHandle, TypedHandle, Typed},
        value::Value,
    },
    linked_list::ContainsLinkedList,
    utils::apint::APInt,
};

// The emission shims stand in for the old cmir op paths; the body below is
// line-compatible with `importer.rs` modulo this alias.
use self::ox as stair_mir;

/// A single unsupported MIR body or construct discovered during import.
#[derive(Debug, Clone)]
pub struct ImportError {
    pub item: String,
    pub reason: String,
}

/// Result of importing a rustc crate into STAIR MIR.
pub struct ImportedCrate {
    pub ctx: Context,
    pub module: Ptr<Operation>,
    pub kernel_module: Ptr<Operation>,
    pub kernel_count: usize,
    pub unsupported: Vec<ImportError>,
}

pub const KERNEL_EXPORT_PREFIX: &str = "__stair_kernel_";

/// The zero-sized "unit" representation: an llvm-native `[0 x u8]`.
/// dialect-mir's type converter recurses into aggregate fields and only
/// knows llvm/builtin scalar and Mir types, so Rust `()`/`!`/fn-item types
/// must not surface as `builtin.unit` anywhere a mir op's operand or result
/// type can reach the lowering.
fn unit_ty(ctx: &mut Context) -> TypeHandle {
    let u8_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
    llvm::types::ArrayType::get(ctx, u8_ty, 0).into()
}

/// Whether a converted type is the zero-sized unit representation (or a
/// legacy `builtin.unit`, matched defensively).
fn is_unit_converted_ty(ctx: &Context, ty: TypeHandle) -> bool {
    let ty_ref = ty.deref(ctx);
    if ty_ref.is::<UnitType>() {
        return true;
    }
    ty_ref
        .downcast_ref::<llvm::types::ArrayType>()
        .is_some_and(|array| array.size() == 0)
}

/// Create a STAIR context with all dialects needed by the Rust MIR path.
pub fn create_context() -> Context {
    let mut ctx = Context::new();
    aarch64::register(&mut ctx);
    x86_64::register(&mut ctx);
    crate::dialects::dialect_mir::register(&mut ctx);
    macho::register(&mut ctx);
    ctx
}

/// Import all MIR body owners visible to rustc.
pub fn import_crate<'tcx>(tcx: TyCtxt<'tcx>) -> ImportedCrate {
    let mut ctx = create_context();
    let module_name = "rust_crate".try_into().unwrap();
    let module_op = builtin::ops::ModuleOp::new(&mut ctx, module_name);
    let module_body = module_op.get_region(&ctx).deref(&ctx).get_head().unwrap();
    let module = module_op.get_operation();
    let kernel_module_op =
        builtin::ops::ModuleOp::new(&mut ctx, "rust_kernels".try_into().unwrap());
    let kernel_module_body = kernel_module_op
        .get_region(&ctx)
        .deref(&ctx)
        .get_head()
        .unwrap();
    let kernel_module = kernel_module_op.get_operation();

    let mut legaliser = Legaliser::default();
    let mut unsupported = Vec::new();
    let mut kernel_count = 0;

    let mut body_owners: Vec<_> = tcx.hir_body_owners().collect();
    for item in tcx.hir_free_items() {
        let def_id = item.owner_id.def_id;
        if !body_owners.contains(&def_id) && is_kernel_def_id(tcx, def_id.to_def_id()) {
            body_owners.push(def_id);
        }
    }

    for owner in body_owners {
        if !is_codegen_body(tcx, owner.to_def_id()) {
            continue;
        }
        if tcx
            .generics_of(owner.to_def_id())
            .requires_monomorphization(tcx)
        {
            continue;
        }
        let name = tcx.def_path_str(owner.to_def_id());
        let is_kernel = is_kernel_def_id(tcx, owner.to_def_id());
        let symbol = if is_kernel {
            kernel_symbol(tcx, &mut legaliser, owner.to_def_id())
        } else {
            function_symbol(tcx, &mut legaliser, owner.to_def_id())
        };
        let body = tcx.optimized_mir(owner);
        let import_body = if is_kernel {
            kernel_count += 1;
            kernel_module_body
        } else {
            module_body
        };
        match import_function(tcx, &mut ctx, import_body, symbol, body, is_kernel, None) {
            Ok(()) => {}
            Err(reason) => unsupported.push(ImportError { item: name, reason }),
        }
    }

    if let Some((entry_def_id, _)) = tcx.entry_fn(())
        && entry_def_id.is_local()
    {
        if is_kernel_def_id(tcx, entry_def_id) {
            unsupported.push(ImportError {
                item: tcx.def_path_str(entry_def_id),
                reason: "the Rust entry point cannot be marked #[kernel]".to_string(),
            });
            return ImportedCrate {
                ctx,
                module,
                kernel_module,
                kernel_count,
                unsupported,
            };
        }
        let rust_main = function_symbol(tcx, &mut legaliser, entry_def_id);
        if let Err(reason) = import_entry_wrapper(&mut ctx, module_body, rust_main) {
            unsupported.push(ImportError {
                item: tcx.def_path_str(entry_def_id),
                reason,
            });
        }
    }

    declare_default_allocator_shims(tcx, &mut ctx, module_body);

    ImportedCrate {
        ctx,
        module,
        kernel_module,
        kernel_count,
        unsupported,
    }
}

fn is_codegen_body(tcx: TyCtxt<'_>, def_id: rustc_hir::def_id::DefId) -> bool {
    matches!(
        tcx.def_kind(def_id),
        rustc_hir::def::DefKind::Fn
            | rustc_hir::def::DefKind::AssocFn
            | rustc_hir::def::DefKind::Ctor(_, _)
    )
}

fn is_kernel_def_id<'tcx>(tcx: TyCtxt<'tcx>, def_id: rustc_hir::def_id::DefId) -> bool {
    tcx.codegen_fn_attrs(def_id)
        .symbol_name
        .is_some_and(|name| name.as_str().starts_with(KERNEL_EXPORT_PREFIX))
}

fn function_symbol<'tcx>(
    tcx: TyCtxt<'tcx>,
    legaliser: &mut Legaliser,
    def_id: rustc_hir::def_id::DefId,
) -> crate::identifier::Identifier {
    legaliser.legalise(tcx.symbol_name(Instance::mono(tcx, def_id)).name)
}

fn kernel_symbol<'tcx>(
    tcx: TyCtxt<'tcx>,
    legaliser: &mut Legaliser,
    def_id: rustc_hir::def_id::DefId,
) -> crate::identifier::Identifier {
    legaliser.legalise(&tcx.def_path_str(def_id))
}

fn import_entry_wrapper(
    ctx: &mut Context,
    module_body: Ptr<BasicBlock>,
    rust_main: crate::identifier::Identifier,
) -> Result<(), String> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signed);
    let fn_ty = FunctionType::get(ctx, vec![], vec![i32_ty.into()]);
    let func = stair_mir::ops::FuncOp::new(ctx, "main".try_into().unwrap(), fn_ty);
    let entry = func.get_entry_block(ctx);

    let call = stair_mir::ops::CallOp::new_direct(ctx, rust_main, vec![], None);
    call.get_operation().insert_at_back(entry, ctx);

    let zero = stair_mir::ops::ConstantOp::new_integer(
        ctx,
        IntegerAttr::new(i32_ty, APInt::from_u32(0, NonZero::new(32).unwrap())),
    );
    zero.get_operation().insert_at_back(entry, ctx);

    let ret = stair_mir::ops::ReturnOp::new(ctx, Some(zero.get_result(ctx)));
    ret.get_operation().insert_at_back(entry, ctx);

    func.get_operation().insert_at_back(module_body, ctx);
    Ok(())
}

struct FunctionImportState<'tcx> {
    module_body: Ptr<BasicBlock>,
    blocks: Vec<Ptr<BasicBlock>>,
    local_slots: Vec<Option<(Value, TypeHandle)>>,
    instance: Option<Instance<'tcx>>,
}

fn mono_ty<'tcx>(tcx: TyCtxt<'tcx>, state: &FunctionImportState<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
    state.instance.map_or(ty, |instance| {
        instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            rustc_middle::ty::TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(ty),
        )
    })
}

fn import_function<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    module_body: Ptr<BasicBlock>,
    symbol: crate::identifier::Identifier,
    body: &Body<'tcx>,
    is_kernel: bool,
    instance: Option<Instance<'tcx>>,
) -> Result<(), String> {
    if symbol_exists(ctx, module_body, &symbol.to_string()) {
        return Ok(());
    }

    if !is_kernel && contains_kernel_call(tcx, body) {
        return Err("direct host calls to #[kernel] functions are not implemented yet".to_string());
    }

    let mut inputs = Vec::new();
    let mut input_groups = Vec::new();
    for local in body.args_iter() {
        let ty = mono_ty(
            tcx,
            &FunctionImportState {
                module_body,
                blocks: Vec::new(),
                local_slots: Vec::new(),
                instance,
            },
            body.local_decls[local].ty,
        );
        let storage_ty = convert_ty(tcx, ctx, ty)?;
        if body.spread_arg == Some(local) {
            // "rust-call" bodies receive the elements of the trailing tuple
            // as separate ABI arguments; callers untuple to match.
            let rustc_middle::ty::TyKind::Tuple(elem_tys) = runtime_ty(ty).kind() else {
                return Err(format!("spread argument is not a tuple: {ty:?}"));
            };
            let mut elements = Vec::new();
            for elem_ty in elem_tys.iter() {
                let elem_storage_ty = convert_ty(tcx, ctx, elem_ty)?;
                let elem_abi_ty = convert_immediate_ty(tcx, ctx, elem_ty)?;
                let abi = arg_abi_for_ty(ctx, elem_abi_ty)?;
                match &abi {
                    ArgAbi::Leaves(leaves) => inputs.extend(leaves.iter().map(|(_, ty)| *ty)),
                    ArgAbi::Indirect => inputs.push(llvm_ptr_ty(ctx)),
                }
                elements.push((elem_storage_ty, abi));
            }
            input_groups.push((local, storage_ty, ParamAbi::Spread(elements)));
            continue;
        }
        let abi_ty = convert_immediate_ty(tcx, ctx, ty)?;
        let group = arg_abi_for_ty(ctx, abi_ty)?;
        match &group {
            ArgAbi::Leaves(leaves) => inputs.extend(leaves.iter().map(|(_, ty)| *ty)),
            ArgAbi::Indirect => inputs.push(llvm_ptr_ty(ctx)),
        }
        input_groups.push((local, storage_ty, ParamAbi::Single(group)));
    }

    let return_ty = mono_ty(
        tcx,
        &FunctionImportState {
            module_body,
            blocks: Vec::new(),
            local_slots: Vec::new(),
            instance,
        },
        body.local_decls[rustc_mir::RETURN_PLACE].ty,
    );
    let results = convert_return_ty(tcx, ctx, return_ty)?;
    let fn_ty = FunctionType::get(ctx, inputs, results);
    let func = stair_mir::ops::FuncOp::new(ctx, symbol, fn_ty);
    func.get_operation().insert_at_back(module_body, ctx);
    let entry = func.get_entry_block(ctx);

    let region = func.get_region(ctx);
    let mut blocks = vec![entry];
    for (bb, _) in body.basic_blocks.iter_enumerated() {
        if bb == rustc_mir::START_BLOCK {
            continue;
        }
        let block = BasicBlock::new(
            ctx,
            Some(format!("bb{}", bb.index()).try_into().unwrap()),
            vec![],
        );
        block.insert_at_back(region, ctx);
        blocks.push(block);
    }

    let mut state = FunctionImportState {
        module_body,
        blocks,
        local_slots: vec![None; body.local_decls.len()],
        instance,
    };

    let mut insert_block = entry;
    for (local, decl) in body.local_decls.iter_enumerated() {
        let Some(ty) = convert_storage_ty(tcx, ctx, mono_ty(tcx, &state, decl.ty))? else {
            continue;
        };
        let alloca = stair_mir::ops::AllocaOp::new(ctx, ty);
        alloca.get_operation().insert_at_back(insert_block, ctx);
        state.local_slots[local.index()] = Some((alloca.get_result(ctx), ty));
    }

    let mut block_arg_idx = 0usize;
    for (local, aggregate_ty, group) in input_groups {
        let Some(slot) = local_slot_opt(&state, local)? else {
            continue;
        };
        let arg = match group {
            ParamAbi::Single(group) => reassemble_abi_arg(
                ctx,
                entry,
                insert_block,
                aggregate_ty,
                group,
                &mut block_arg_idx,
            ),
            ParamAbi::Spread(elements) => {
                let tuple_rust_ty = mono_ty(tcx, &state, body.local_decls[local].ty);
                let undef = stair_mir::ops::UndefOp::new(ctx, aggregate_ty);
                undef.get_operation().insert_at_back(insert_block, ctx);
                let mut current = undef.get_result(ctx);
                for (idx, (elem_ty, abi)) in elements.into_iter().enumerate() {
                    let element = reassemble_abi_arg(
                        ctx,
                        entry,
                        insert_block,
                        elem_ty,
                        abi,
                        &mut block_arg_idx,
                    );
                    let element = cast_value_to_type(ctx, insert_block, element, elem_ty);
                    let index = converted_field_index(tcx, tuple_rust_ty, idx)?;
                    let insert =
                        stair_mir::ops::InsertValueOp::new(ctx, element, current, vec![index]);
                    insert.get_operation().insert_at_back(insert_block, ctx);
                    current = insert.get_result(ctx);
                }
                current
            }
        };
        let arg = normalize_bool_for_storage(
            tcx,
            ctx,
            &state,
            insert_block,
            body.local_decls[local].ty,
            arg,
        )?;
        let store = stair_mir::ops::StoreOp::new(ctx, arg, slot);
        store.get_operation().insert_at_back(insert_block, ctx);
    }

    for (bb, data) in body.basic_blocks.iter_enumerated() {
        insert_block = block_for(&state, bb)?;
        for statement in &data.statements {
            import_statement(tcx, ctx, &state, insert_block, body, &statement.kind)?;
        }
        let terminator = data.terminator();
        import_terminator(tcx, ctx, &state, insert_block, body, &terminator.kind)?;
    }
    Ok(())
}

/// An `llvm.func` declaration (no body).
fn new_llvm_func_decl(
    ctx: &mut Context,
    name: crate::identifier::Identifier,
    ty: TypedHandle<llvm::types::FuncType>,
    linkage: LinkageAttr,
) -> llvm::ops::FuncOp {
    let func = llvm::ops::FuncOp::new(ctx, name, ty);
    func.set_attr_llvm_function_linkage(ctx, linkage);
    func
}

/// An `llvm.func` definition with an entry block carrying the argument values.
fn new_llvm_func_def(
    ctx: &mut Context,
    name: crate::identifier::Identifier,
    ty: TypedHandle<llvm::types::FuncType>,
    linkage: LinkageAttr,
) -> llvm::ops::FuncOp {
    let func = new_llvm_func_decl(ctx, name, ty, linkage);
    func.get_or_create_entry_block(ctx);
    func
}

fn declare_external_function(
    ctx: &mut Context,
    module_body: Ptr<BasicBlock>,
    symbol: crate::identifier::Identifier,
    args: Vec<TypeHandle>,
    result: Option<TypeHandle>,
) {
    if module_has_llvm_function(ctx, module_body, &symbol) {
        return;
    }

    let result = result
        .map(|ty| llvm_decl_type(ctx, ty))
        .unwrap_or_else(|| llvm::types::VoidType::get(ctx).into());
    let args = args
        .into_iter()
        .map(|arg| llvm_decl_type(ctx, arg))
        .collect();
    let func_ty = llvm::types::FuncType::get(ctx, result, args, false);
    let func = new_llvm_func_decl(ctx, symbol, func_ty, LinkageAttr::ExternalLinkage);
    func.get_operation().insert_at_back(module_body, ctx);
}

fn module_has_llvm_function(
    ctx: &Context,
    module_body: Ptr<BasicBlock>,
    symbol: &crate::identifier::Identifier,
) -> bool {
    module_body.deref(ctx).iter(ctx).any(|op| {
        Operation::get_opid(op, ctx) == llvm::ops::FuncOp::get_opid_static()
            && op
                .deref(ctx)
                .attributes
                .get::<IdentifierAttr>(&ATTR_KEY_SYM_NAME)
                .is_some_and(|attr| crate::identifier::Identifier::from(attr.clone()) == *symbol)
    })
}

fn declare_static_global<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    module_body: Ptr<BasicBlock>,
    symbol: crate::identifier::Identifier,
    def_id: rustc_span::def_id::DefId,
) -> Result<(), String> {
    if symbol_exists(ctx, module_body, &symbol.to_string()) {
        return Ok(());
    }
    // A local static is defined here from its evaluated initializer; a foreign
    // one is only declared (a `llvm.global` with no initializer), so its
    // address resolves through an undefined symbol-table entry at link time.
    if def_id.is_local()
        && let Ok(alloc) = tcx.eval_static_initializer(def_id)
    {
        return emit_allocation_global(
            tcx,
            ctx,
            module_body,
            symbol,
            alloc.inner(),
            LinkageAttr::ExternalLinkage,
            false,
        );
    }
    let byte_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
    let global_ty = llvm::types::ArrayType::get(ctx, byte_ty, 0).into();
    let global = llvm::ops::GlobalOp::new(ctx, symbol, global_ty);
    global.set_attr_llvm_global_linkage(ctx, LinkageAttr::ExternalLinkage);
    global.get_operation().insert_at_back(module_body, ctx);
    Ok(())
}

/// Declare or define the storage of a `#[thread_local]` static (the target
/// of [Rvalue::ThreadLocalRef]). Same shape as [declare_static_global] —
/// a local static is defined from its evaluated initializer, a foreign one
/// (e.g. std's `RandomState::new::KEYS`, exported from libstd's `.tdata`)
/// is only declared — but the emitted `llvm.global` carries the
/// [ll.tls](pliron_ll::ll::TlsAttr) marker, so the backend materializes its
/// address through the thread pointer and places the initializer in the TLS
/// template sections.
fn declare_thread_local_global<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    module_body: Ptr<BasicBlock>,
    symbol: crate::identifier::Identifier,
    def_id: rustc_span::def_id::DefId,
) -> Result<(), String> {
    if symbol_exists(ctx, module_body, &symbol.to_string()) {
        return Ok(());
    }
    if def_id.is_local()
        && let Ok(alloc) = tcx.eval_static_initializer(def_id)
    {
        return emit_allocation_global(
            tcx,
            ctx,
            module_body,
            symbol,
            alloc.inner(),
            LinkageAttr::ExternalLinkage,
            true,
        );
    }
    let byte_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
    let global_ty = llvm::types::ArrayType::get(ctx, byte_ty, 0).into();
    let global = llvm::ops::GlobalOp::new(ctx, symbol, global_ty);
    global.set_attr_llvm_global_linkage(ctx, LinkageAttr::ExternalLinkage);
    pliron_ll::ll::set_global_thread_local(ctx, &global);
    global.get_operation().insert_at_back(module_body, ctx);
    Ok(())
}

/// Define `symbol` as an [ll.data](pliron_ll::ll::DataAttr) global carrying
/// `alloc`'s raw bytes and pointer relocations, mirroring rustc's
/// `Allocation` (bytes + provenance). The global is inserted before its
/// relocation targets are resolved so cyclic allocation graphs (e.g. a static
/// referencing itself through another static) terminate.
fn emit_allocation_global<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    module_body: Ptr<BasicBlock>,
    symbol: crate::identifier::Identifier,
    alloc: &rustc_mir::interpret::Allocation,
    linkage: LinkageAttr,
    thread_local: bool,
) -> Result<(), String> {
    if symbol_exists(ctx, module_body, &symbol.to_string()) {
        return Ok(());
    }
    let byte_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
    let global_ty = llvm::types::ArrayType::get(ctx, byte_ty, alloc.len() as u64).into();
    let global = llvm::ops::GlobalOp::new(ctx, symbol, global_ty);
    global.set_attr_llvm_global_linkage(ctx, linkage);
    if thread_local {
        pliron_ll::ll::set_global_thread_local(ctx, &global);
    }
    global.get_operation().insert_at_back(module_body, ctx);

    // Uninitialized ranges (e.g. padding) read as whatever the raw buffer
    // holds, matching what rustc's own codegen emits for globals.
    let bytes = alloc
        .inspect_with_uninit_and_ptr_outside_interpreter(0..alloc.len())
        .to_vec();
    let ptr_size = tcx.data_layout.pointer_size().bytes() as usize;
    let mut relocs = Vec::new();
    for (offset, provenance) in alloc.provenance().ptrs().iter() {
        let slot = offset.bytes() as usize;
        // rustc encodes a provenance-carrying slot as `target_offset` raw
        // bytes (the pointer value is `target_base + target_offset`, and the
        // base is only re-added at relocation time), so the stored value is
        // exactly the relocation addend — the same derivation as
        // rustc_codegen_llvm's `const_alloc_to_llvm`.
        let addend = rustc_mir::interpret::read_target_uint(
            tcx.data_layout.endian,
            &bytes[slot..slot + ptr_size],
        )
        .map_err(|error| format!("cannot read relocation pointer bytes: {error}"))?
            as u64;
        // A type-id "pointer" is not an address: its slot bytes already hold
        // the hash segment, so no relocation is needed.
        if matches!(
            tcx.global_alloc(provenance.alloc_id()),
            rustc_mir::interpret::GlobalAlloc::TypeId { .. }
        ) {
            continue;
        }
        let target = data_global_for_alloc(tcx, ctx, module_body, provenance.alloc_id())?;
        relocs.push(pliron_ll::ll::DataReloc {
            offset: slot as u64,
            symbol: target.to_string(),
            addend: addend as i64,
        });
    }
    pliron_ll::ll::set_global_data(
        ctx,
        &global,
        pliron_ll::ll::DataAttr {
            bytes,
            align: alloc.align.bytes(),
            mutable: alloc.mutability.is_mut(),
            relocs,
        },
    );
    Ok(())
}

/// The module symbol whose address is the runtime address of `alloc_id`,
/// emitting the backing global (and, recursively, everything it points to)
/// on first use.
fn data_global_for_alloc<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    module_body: Ptr<BasicBlock>,
    alloc_id: rustc_mir::interpret::AllocId,
) -> Result<crate::identifier::Identifier, String> {
    let mut legaliser = Legaliser::default();
    match tcx.global_alloc(alloc_id) {
        rustc_mir::interpret::GlobalAlloc::Memory(alloc) => {
            let symbol = legaliser.legalise(&format!("__crabbit_{alloc_id:?}"));
            emit_allocation_global(
                tcx,
                ctx,
                module_body,
                symbol.clone(),
                alloc.inner(),
                LinkageAttr::InternalLinkage,
                false,
            )?;
            Ok(symbol)
        }
        rustc_mir::interpret::GlobalAlloc::Static(def_id) => {
            let symbol = legaliser.legalise(tcx.symbol_name(Instance::mono(tcx, def_id)).name);
            declare_static_global(tcx, ctx, module_body, symbol.clone(), def_id)?;
            Ok(symbol)
        }
        rustc_mir::interpret::GlobalAlloc::Function { instance } => {
            let symbol = legaliser.legalise(tcx.symbol_name(instance).name);
            if should_import_instance(tcx, instance) {
                import_upstream_instance(tcx, ctx, module_body, instance).map_err(|error| {
                    format!("while importing fn-pointer target {symbol}: {error}")
                })?;
                // A data relocation resolves through the symbol table, unlike
                // module-internal calls (which encode resolves directly), so
                // the imported copy must be visible there — but *weak*: an
                // LLVM-built rlib may export the same monomorphization
                // (share-generics) as a strong global, and both copies
                // implement the same function.
                set_function_linkage(ctx, module_body, &symbol, LinkageAttr::WeakODRLinkage);
                Ok(symbol)
            } else {
                // No importable MIR: forward through a local thunk (the same
                // scheme as reify_fn_pointer), exported so the relocation can
                // target it.
                let thunk = emit_fn_ptr_thunk(tcx, ctx, module_body, symbol, instance)?;
                set_function_linkage(ctx, module_body, &thunk, LinkageAttr::ExternalLinkage);
                Ok(thunk)
            }
        }
        rustc_mir::interpret::GlobalAlloc::VTable(ty, dyn_ty) => {
            let vtable_alloc_id = tcx.vtable_allocation((
                ty,
                dyn_ty.principal().map(|principal| {
                    tcx.instantiate_bound_regions_with_erased(principal)
                }),
            ));
            let rustc_mir::interpret::GlobalAlloc::Memory(alloc) =
                tcx.global_alloc(vtable_alloc_id)
            else {
                return Err(format!(
                    "vtable allocation for {ty:?} is not a memory allocation"
                ));
            };
            let symbol = legaliser.legalise(&format!("__crabbit_{vtable_alloc_id:?}"));
            emit_allocation_global(
                tcx,
                ctx,
                module_body,
                symbol.clone(),
                alloc.inner(),
                LinkageAttr::InternalLinkage,
                false,
            )?;
            Ok(symbol)
        }
        rustc_mir::interpret::GlobalAlloc::TypeId { ty } => Err(format!(
            "unsupported MIR constant: type-id allocation for {ty:?} has no runtime address"
        )),
    }
}

/// Hex rendering of a payload for use in a symbol name; the payload itself is
/// carried as bytes.
fn hex_suffix(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn declare_anonymous_byte_global(
    ctx: &mut Context,
    module_body: Ptr<BasicBlock>,
    bytes: &[u8],
) -> crate::identifier::Identifier {
    let suffix = if bytes.is_empty() {
        "empty".to_string()
    } else {
        hex_suffix(bytes)
    };
    let mut legaliser = Legaliser::default();
    let symbol = legaliser.legalise(&format!("L_stair_bytes_{suffix}"));
    if symbol_exists(ctx, module_body, &symbol.to_string()) {
        return symbol;
    }

    let byte_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
    let global_ty = llvm::types::ArrayType::get(ctx, byte_ty, bytes.len() as u64).into();
    let global = llvm::ops::GlobalOp::new(ctx, symbol.clone(), global_ty);
    global.set_attr_llvm_global_linkage(ctx, LinkageAttr::PrivateLinkage);
    pliron_ll::ll::set_global_initializer_bytes(ctx, &global, bytes.to_vec());
    global.get_operation().insert_at_back(module_body, ctx);
    symbol
}

fn llvm_decl_type(ctx: &mut Context, ty: TypeHandle) -> TypeHandle {
    let ty_ref = ty.deref(ctx);
    if ty_ref.is::<crate::dialects::dialect_mir::types::MirPtrType>() {
        drop(ty_ref);
        return llvm::types::PointerType::get(ctx, 0).into();
    }
    if let Some(array_ty) = ty_ref.downcast_ref::<llvm::types::ArrayType>() {
        let elem = array_ty.elem_type();
        let size = array_ty.size();
        drop(ty_ref);
        let elem = llvm_decl_type(ctx, elem);
        return llvm::types::ArrayType::get(ctx, elem, size).into();
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<llvm::types::StructType>() {
        let name = struct_ty.name();
        let fields = (!struct_ty.is_opaque()).then(|| struct_ty.fields().collect::<Vec<_>>());
        drop(ty_ref);
        let fields = fields.map(|fields| {
            fields
                .into_iter()
                .map(|field| llvm_decl_type(ctx, field))
                .collect::<Vec<_>>()
        });
        return match name {
            Some(name) => {
                let name = format!("{name}__llvm").try_into().unwrap();
                llvm::types::StructType::get_named(ctx, name, fields)
                    .unwrap()
                    .into()
            }
            None => llvm::types::StructType::get_unnamed(ctx, fields.unwrap_or_default()).into(),
        };
    }
    ty
}

fn declare_default_allocator_shims<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    module_body: Ptr<BasicBlock>,
) {
    let ptr_ty: TypeHandle = llvm::types::PointerType::get(ctx, 0).into();
    let usize_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let void_ty: TypeHandle = llvm::types::VoidType::get(ctx).into();

    let malloc_ty = llvm::types::FuncType::get(ctx, ptr_ty, vec![usize_ty], false);
    let free_ty = llvm::types::FuncType::get(ctx, void_ty, vec![ptr_ty], false);
    let realloc_ty = llvm::types::FuncType::get(ctx, ptr_ty, vec![ptr_ty, usize_ty], false);
    let calloc_ty = llvm::types::FuncType::get(ctx, ptr_ty, vec![usize_ty, usize_ty], false);
    let rust_alloc_ty = llvm::types::FuncType::get(ctx, ptr_ty, vec![usize_ty, usize_ty], false);
    let rust_dealloc_ty =
        llvm::types::FuncType::get(ctx, void_ty, vec![ptr_ty, usize_ty, usize_ty], false);
    let rust_realloc_ty = llvm::types::FuncType::get(
        ctx,
        ptr_ty,
        vec![ptr_ty, usize_ty, usize_ty, usize_ty],
        false,
    );
    let no_alloc_ty = llvm::types::FuncType::get(ctx, void_ty, vec![], false);

    for (name, ty) in [
        ("malloc", malloc_ty),
        ("free", free_ty),
        ("realloc", realloc_ty),
        ("calloc", calloc_ty),
    ] {
        let decl = new_llvm_func_decl(ctx, name.try_into().unwrap(), ty, LinkageAttr::ExternalLinkage);
        decl.get_operation().insert_at_back(module_body, ctx);
    }

    define_allocator_call(
        ctx,
        module_body,
        allocator_symbol(tcx, "__rust_alloc"),
        "malloc".try_into().unwrap(),
        rust_alloc_ty,
        |args, _one| vec![args[0]],
        Some(ptr_ty),
    );
    define_allocator_call(
        ctx,
        module_body,
        allocator_symbol(tcx, "__rust_dealloc"),
        "free".try_into().unwrap(),
        rust_dealloc_ty,
        |args, _one| vec![args[0]],
        None,
    );
    define_allocator_call(
        ctx,
        module_body,
        allocator_symbol(tcx, "__rust_realloc"),
        "realloc".try_into().unwrap(),
        rust_realloc_ty,
        |args, _one| vec![args[0], args[3]],
        Some(ptr_ty),
    );
    define_allocator_call(
        ctx,
        module_body,
        allocator_symbol(tcx, "__rust_alloc_zeroed"),
        "calloc".try_into().unwrap(),
        rust_alloc_ty,
        |args, one| vec![one, args[0]],
        Some(ptr_ty),
    );

    let no_alloc = new_llvm_func_def(
        ctx,
        allocator_symbol(tcx, "__rust_no_alloc_shim_is_unstable_v2"),
        no_alloc_ty,
        LinkageAttr::ExternalLinkage,
    );
    let ret = llvm::ops::ReturnOp::new(ctx, None);
    let no_alloc_entry = no_alloc
        .get_entry_block(ctx)
        .expect("definition has an entry block");
    ret.get_operation().insert_at_back(no_alloc_entry, ctx);
    no_alloc.get_operation().insert_at_back(module_body, ctx);
}

fn define_allocator_call<F>(
    ctx: &mut Context,
    module_body: Ptr<BasicBlock>,
    name: crate::identifier::Identifier,
    callee: crate::identifier::Identifier,
    func_ty: TypedHandle<llvm::types::FuncType>,
    select_args: F,
    result_ty: Option<TypeHandle>,
) where
    F: FnOnce(&[Value], Value) -> Vec<Value>,
{
    let func = new_llvm_func_def(ctx, name, func_ty, LinkageAttr::ExternalLinkage);
    let entry = func
        .get_entry_block(ctx)
        .expect("definition has an entry block");
    let args: Vec<_> = entry.deref(ctx).arguments().collect();
    let usize_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let one = integer_constant(ctx, usize_ty, 1)
        .expect("usize constant should be representable")
        .get_operation();
    let one_value = one.deref(ctx).get_result(0);
    one.insert_at_back(entry, ctx);
    let call_args = select_args(&args, one_value);
    let call_result_ty =
        result_ty.unwrap_or_else(|| llvm::types::VoidType::get(ctx).into());
    let call_arg_types: Vec<TypeHandle> = call_args
        .iter()
        .map(|arg| {
            use pliron::r#type::Typed;
            arg.get_type(ctx)
        })
        .collect();
    let callee_ty = llvm::types::FuncType::get(ctx, call_result_ty, call_arg_types, false);
    let call = llvm::ops::CallOp::new(
        ctx,
        pliron::builtin::op_interfaces::CallOpCallable::Direct(callee),
        callee_ty,
        call_args,
    );
    let ret_value = result_ty.map(|_| call.get_operation().deref(ctx).get_result(0));
    call.get_operation().insert_at_back(entry, ctx);
    let ret = llvm::ops::ReturnOp::new(ctx, ret_value);
    ret.get_operation().insert_at_back(entry, ctx);
    func.get_operation().insert_at_back(module_body, ctx);
}

fn allocator_symbol<'tcx>(tcx: TyCtxt<'tcx>, name: &str) -> crate::identifier::Identifier {
    rustc_symbol_mangling::mangle_internal_symbol(tcx, name)
        .as_str()
        .try_into()
        .unwrap()
}

fn literal_string_constant<'tcx>(
    tcx: TyCtxt<'tcx>,
    _body: &Body<'tcx>,
    constant: &ConstOperand<'tcx>,
) -> Option<String> {
    if !matches!(
        constant.const_.ty().kind(),
        rustc_middle::ty::TyKind::Ref(_, inner, _) if matches!(inner.kind(), rustc_middle::ty::TyKind::Str)
    ) {
        return None;
    }

    if let Ok(value) = constant
        .const_
        .eval(tcx, _body.typing_env(tcx), constant.span)
        && let Some(bytes) = value.try_get_slice_bytes_for_diagnostics(tcx)
        && let Ok(value) = std::str::from_utf8(bytes)
    {
        return Some(value.to_string());
    }

    let debug = format!("{:?}", constant.const_);
    if let Some(value) = parse_debug_string_const(&debug) {
        return Some(value);
    }

    let snippet = tcx.sess.source_map().span_to_snippet(constant.span).ok()?;
    parse_rust_string_literal(snippet.trim())
}

/// Evaluate a (monomorphized) `&str` constant that is not a source literal,
/// e.g. a promoted `type_name` string, into its UTF-8 contents.
fn evaluated_str_constant<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: rustc_middle::ty::TypingEnv<'tcx>,
    span: rustc_span::Span,
    constant: rustc_mir::Const<'tcx>,
) -> Option<String> {
    let value = constant.eval(tcx, typing_env, span).ok()?;
    let bytes = value.try_get_slice_bytes_for_diagnostics(tcx)?;
    std::str::from_utf8(bytes).ok().map(str::to_string)
}

fn literal_byte_string_constant<'tcx>(
    tcx: TyCtxt<'tcx>,
    constant: &ConstOperand<'tcx>,
) -> Option<Vec<u8>> {
    let snippet = tcx.sess.source_map().span_to_snippet(constant.span).ok()?;
    parse_rust_byte_string_literal(snippet.trim())
}

fn evaluated_byte_string_constant<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: rustc_middle::ty::TypingEnv<'tcx>,
    span: rustc_span::Span,
    constant: rustc_mir::Const<'tcx>,
    len: u64,
) -> Option<Vec<u8>> {
    let value = constant.eval(tcx, typing_env, span).ok()?;
    match value {
        rustc_mir::ConstValue::Scalar(ptr) => {
            let ptr = ptr
                .to_pointer(&tcx)
                .discard_err()?
                .into_pointer_or_addr()
                .ok()?;
            let (provenance, offset) = ptr.prov_and_relative_offset();
            allocation_bytes(tcx, provenance.alloc_id(), offset, len)
        }
        rustc_mir::ConstValue::Slice { alloc_id, meta } => {
            allocation_bytes(tcx, alloc_id, Size::ZERO, meta)
        }
        rustc_mir::ConstValue::Indirect { alloc_id, offset } => {
            let alloc = tcx.global_alloc(alloc_id);
            let rustc_mir::interpret::GlobalAlloc::Memory(alloc) = alloc else {
                return None;
            };
            let ptr_size = tcx.data_layout.pointer_size();
            let ptr = alloc
                .inner()
                .read_scalar(
                    &tcx,
                    rustc_mir::interpret::alloc_range(offset, ptr_size),
                    true,
                )
                .ok()?
                .to_pointer(&tcx)
                .discard_err()?
                .into_pointer_or_addr()
                .ok()?;
            let (provenance, offset) = ptr.prov_and_relative_offset();
            allocation_bytes(tcx, provenance.alloc_id(), offset, len)
        }
        rustc_mir::ConstValue::ZeroSized => Some(Vec::new()),
    }
}

fn allocation_bytes<'tcx>(
    tcx: TyCtxt<'tcx>,
    alloc_id: rustc_mir::interpret::AllocId,
    offset: Size,
    len: u64,
) -> Option<Vec<u8>> {
    let alloc = tcx.global_alloc(alloc_id);
    let rustc_mir::interpret::GlobalAlloc::Memory(alloc) = alloc else {
        return None;
    };
    let range = rustc_mir::interpret::alloc_range(offset, Size::from_bytes(len));
    alloc
        .inner()
        .get_bytes_strip_provenance(&tcx, range)
        .ok()
        .map(|bytes| bytes.to_vec())
}

fn parse_debug_string_const(debug: &str) -> Option<String> {
    let start = debug.find("const \"")? + "const ".len();
    parse_rust_string_literal(&debug[start..])
}

fn parse_rust_string_literal(input: &str) -> Option<String> {
    let mut chars = input.chars();
    if chars.next()? != '"' {
        return None;
    }

    let mut out = String::new();
    let mut escaped = false;
    for ch in chars {
        if escaped {
            match ch {
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                '0' => out.push('\0'),
                '\\' => out.push('\\'),
                '"' => out.push('"'),
                other => out.push(other),
            }
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            return Some(out);
        } else {
            out.push(ch);
        }
    }
    None
}

fn parse_rust_byte_string_literal(input: &str) -> Option<Vec<u8>> {
    let mut chars = input.chars();
    if chars.next()? != 'b' || chars.next()? != '"' {
        return None;
    }

    let mut out = Vec::new();
    let mut escaped = false;
    for ch in chars {
        if escaped {
            match ch {
                'n' => out.push(b'\n'),
                'r' => out.push(b'\r'),
                't' => out.push(b'\t'),
                '0' => out.push(b'\0'),
                '\\' => out.push(b'\\'),
                '"' => out.push(b'"'),
                '\'' => out.push(b'\''),
                other if other.is_ascii() => out.push(other as u8),
                _ => return None,
            }
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            return Some(out);
        } else if ch.is_ascii() {
            out.push(ch as u8);
        } else {
            return None;
        }
    }
    None
}

fn is_arguments_from_str_call<'tcx>(tcx: TyCtxt<'tcx>, func: &Operand<'tcx>) -> bool {
    let Operand::Constant(constant) = func else {
        return false;
    };
    let rustc_middle::ty::TyKind::FnDef(def_id, _) = constant.const_.ty().kind() else {
        return false;
    };
    let path = tcx.def_path_str(*def_id);
    path.contains("fmt::Arguments") && path.ends_with("from_str")
}

fn contains_kernel_call<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) -> bool {
    body.basic_blocks.iter().any(|data| {
        matches!(
            &data.terminator().kind,
            TerminatorKind::Call { func, .. } if is_call_to_kernel(tcx, func)
        )
    })
}

fn is_call_to_kernel<'tcx>(tcx: TyCtxt<'tcx>, func: &Operand<'tcx>) -> bool {
    let Operand::Constant(constant) = func else {
        return false;
    };
    let rustc_middle::ty::TyKind::FnDef(def_id, _) = constant.const_.ty().kind() else {
        return false;
    };
    is_kernel_def_id(tcx, *def_id)
}

fn import_statement<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    statement: &StatementKind<'tcx>,
) -> Result<(), String> {
    match statement {
        StatementKind::Assign(boxed) => {
            let (place, rvalue) = &**boxed;
            if place.projection.is_empty() && place_slot_opt(state, place)?.is_none() {
                // The destination is zero-sized (no slot). MIR rvalues are
                // pure, so the whole assignment can be dropped.
                return Ok(());
            }
            let value = import_rvalue(tcx, ctx, state, insert_block, body, rvalue)?;
            store_place(tcx, ctx, state, insert_block, body, place, value)?;
            Ok(())
        }
        StatementKind::StorageLive(_) | StatementKind::StorageDead(_) | StatementKind::Nop => {
            Ok(())
        }
        StatementKind::SetDiscriminant {
            place,
            variant_index,
        } => {
            let enum_ty = mono_ty(tcx, state, place.ty(body, tcx).ty);
            if !is_enum_ty(enum_ty) {
                return Ok(());
            }
            let addr = place_addr(tcx, ctx, state, insert_block, body, place)?;
            write_enum_tag(tcx, ctx, insert_block, enum_ty, *variant_index, addr)
        }
        StatementKind::Intrinsic(intrinsic)
            if matches!(&**intrinsic, rustc_mir::NonDivergingIntrinsic::Assume(_)) =>
        {
            Ok(())
        }
        StatementKind::Intrinsic(intrinsic)
            if matches!(
                &**intrinsic,
                rustc_mir::NonDivergingIntrinsic::CopyNonOverlapping(_)
            ) =>
        {
            let rustc_mir::NonDivergingIntrinsic::CopyNonOverlapping(copy) = &**intrinsic else {
                unreachable!("copy_nonoverlapping checked above");
            };
            let dst = import_operand(tcx, ctx, state, insert_block, body, &copy.dst)?;
            let src = import_operand(tcx, ctx, state, insert_block, body, &copy.src)?;
            let count = import_operand(tcx, ctx, state, insert_block, body, &copy.count)?;
            let src_ty = mono_ty(tcx, state, copy.src.ty(body, tcx));
            let elem_size = layout_size_of_ty(tcx, pointee_ty(src_ty)?)?;
            let byte_count = scale_index(ctx, insert_block, count, elem_size)?;

            let memcpy: crate::identifier::Identifier = "memcpy".try_into().unwrap();
            let ptr_ty = llvm_ptr_ty(ctx);
            let usize_ty: TypeHandle = usize_ty(ctx).into();
            declare_external_function(
                ctx,
                state.module_body,
                memcpy.clone(),
                vec![ptr_ty, ptr_ty, usize_ty],
                Some(ptr_ty),
            );
            let call = stair_mir::ops::CallOp::new_direct(
                ctx,
                memcpy,
                vec![dst, src, byte_count],
                Some(ptr_ty),
            );
            call.get_operation().insert_at_back(insert_block, ctx);
            Ok(())
        }
        other => Err(format!("unsupported MIR statement: {other:?}")),
    }
}

fn import_terminator<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    terminator: &TerminatorKind<'tcx>,
) -> Result<(), String> {
    match terminator {
        TerminatorKind::Return => {
            let retval = if convert_return_ty(
                tcx,
                ctx,
                mono_ty(tcx, state, body.local_decls[rustc_mir::RETURN_PLACE].ty),
            )?
            .is_empty()
            {
                None
            } else {
                Some(load_place(
                    tcx,
                    ctx,
                    state,
                    insert_block,
                    body,
                    rustc_mir::RETURN_PLACE.into(),
                )?)
            };
            let ret = stair_mir::ops::ReturnOp::new(ctx, retval);
            ret.get_operation().insert_at_back(insert_block, ctx);
            Ok(())
        }
        TerminatorKind::Goto { target } => {
            let dest = block_for(state, *target)?;
            let goto = stair_mir::ops::GotoOp::new(ctx, dest, vec![]);
            goto.get_operation().insert_at_back(insert_block, ctx);
            Ok(())
        }
        TerminatorKind::Unreachable => {
            let unreachable = stair_mir::ops::UnreachableOp::new(ctx);
            unreachable
                .get_operation()
                .insert_at_back(insert_block, ctx);
            Ok(())
        }
        TerminatorKind::UnwindResume => {
            let unreachable = stair_mir::ops::UnreachableOp::new(ctx);
            unreachable
                .get_operation()
                .insert_at_back(insert_block, ctx);
            Ok(())
        }
        TerminatorKind::SwitchInt { discr, targets } => {
            let discr = import_operand(tcx, ctx, state, insert_block, body, discr)?;
            let otherwise = targets.otherwise();
            let cases = targets.iter().collect::<Vec<_>>();
            if cases.is_empty() {
                let goto = stair_mir::ops::GotoOp::new(ctx, block_for(state, otherwise)?, vec![]);
                goto.get_operation().insert_at_back(insert_block, ctx);
                return Ok(());
            }

            let mut compare_block = insert_block;
            for (idx, (expected, target)) in cases.iter().copied().enumerate() {
                let next_block = if idx + 1 == cases.len() {
                    block_for(state, otherwise)?
                } else {
                    let block = BasicBlock::new(
                        ctx,
                        Some(format!("switch{}", idx + 1).try_into().unwrap()),
                        vec![],
                    );
                    let region = insert_block
                        .deref(ctx)
                        .get_parent_region()
                        .ok_or_else(|| "MIR switch block is not in a region".to_string())?;
                    block.insert_at_back(region, ctx);
                    block
                };
                let expected = integer_constant(ctx, discr.get_type(ctx), expected)?;
                expected.get_operation().insert_at_back(compare_block, ctx);
                let cmp = stair_mir::ops::EqOp::new(ctx, discr, expected.get_result(ctx));
                cmp.get_operation().insert_at_back(compare_block, ctx);
                let branch = stair_mir::ops::CondBrOp::new(
                    ctx,
                    cmp.get_result(ctx),
                    block_for(state, target)?,
                    vec![],
                    next_block,
                    vec![],
                );
                branch.get_operation().insert_at_back(compare_block, ctx);
                compare_block = next_block;
            }
            Ok(())
        }
        TerminatorKind::Call {
            func,
            args,
            destination,
            target,
            ..
        } => {
            if is_arguments_from_str_call(tcx, func) {
                let Some(target) = target else {
                    return Err("unsupported MIR call without return target".to_string());
                };
                lower_arguments_from_str_call(
                    tcx,
                    ctx,
                    state,
                    insert_block,
                    body,
                    args,
                    destination,
                )?;
                let goto = stair_mir::ops::GotoOp::new(ctx, block_for(state, *target)?, vec![]);
                goto.get_operation().insert_at_back(insert_block, ctx);
                return Ok(());
            }

            if is_unreachable_unchecked_call(tcx, state, body, func) {
                stair_mir::ops::UnreachableOp::new(ctx)
                    .get_operation()
                    .insert_at_back(insert_block, ctx);
                return Ok(());
            }

            if is_noop_intrinsic_call(tcx, state, body, func) {
                let Some(target) = target else {
                    return Err(
                        "unsupported MIR no-op intrinsic call without return target".to_string()
                    );
                };
                let goto = stair_mir::ops::GotoOp::new(ctx, block_for(state, *target)?, vec![]);
                goto.get_operation().insert_at_back(insert_block, ctx);
                return Ok(());
            }

            if lower_known_intrinsic_call(
                tcx,
                ctx,
                state,
                insert_block,
                body,
                func,
                args,
                destination,
            )? {
                let Some(target) = target else {
                    stair_mir::ops::UnreachableOp::new(ctx)
                        .get_operation()
                        .insert_at_back(insert_block, ctx);
                    return Ok(());
                };
                let goto = stair_mir::ops::GotoOp::new(ctx, block_for(state, *target)?, vec![]);
                goto.get_operation().insert_at_back(insert_block, ctx);
                return Ok(());
            }

            // A virtual call: the callee resolves to a vtable slot instead of
            // a symbol, so the function pointer is loaded from the receiver's
            // vtable at `index * ptr_size` (the index already counts the
            // drop/size/align header, as in rustc_codegen_ssa's
            // `VirtualIndex`), and the receiver's data half becomes the first
            // argument.
            let virtual_slot = call_instance(tcx, state, body, func).and_then(|instance| {
                match instance.def {
                    InstanceKind::Virtual(_, index) => Some(index as u64),
                    _ => None,
                }
            });

            // An indirect call: the callee is a function-pointer value rather
            // than a (constant or zero-sized function-item) `FnDef`.
            let callee_value = if call_fn_def(tcx, state, body, func).is_some() {
                None
            } else {
                Some(import_operand(tcx, ctx, state, insert_block, body, func)?)
            };

            // Callees with the "rust-call" ABI take their trailing tuple
            // argument untupled (matching `spread_arg` on the body).
            let untuple_last = callee_value.is_none()
                && call_fn_def(tcx, state, body, func).is_some_and(|(def_id, _)| {
                    tcx.fn_sig(def_id).skip_binder().skip_binder().abi
                        == rustc_abi::ExternAbi::RustCall
                });

            let mut call_args = Vec::with_capacity(args.len());
            let mut vtable_ptr = None;
            for (idx, arg) in args.iter().enumerate() {
                if idx == 0 && virtual_slot.is_some() {
                    let receiver =
                        import_operand(tcx, ctx, state, insert_block, body, &arg.node)?;
                    let (data, vtable) = split_dyn_receiver(ctx, insert_block, receiver)?;
                    vtable_ptr = Some(vtable);
                    lower_abi_call_arg(ctx, insert_block, data, &mut call_args)?;
                    continue;
                }
                let value = import_operand(tcx, ctx, state, insert_block, body, &arg.node)?;
                if untuple_last && idx + 1 == args.len() {
                    let tuple_ty = mono_ty(tcx, state, arg.node.ty(body, tcx));
                    let rustc_middle::ty::TyKind::Tuple(elem_tys) = runtime_ty(tuple_ty).kind()
                    else {
                        return Err(format!(
                            "rust-call trailing argument is not a tuple: {tuple_ty:?}"
                        ));
                    };
                    for (elem_idx, elem_ty) in elem_tys.iter().enumerate() {
                        let index = converted_field_index(tcx, tuple_ty, elem_idx)?;
                        let elem_conv = convert_ty(tcx, ctx, elem_ty)?;
                        let element =
                            stair_mir::ops::ExtractValueOp::new(ctx, value, vec![index], elem_conv);
                        element.get_operation().insert_at_back(insert_block, ctx);
                        lower_abi_call_arg(
                            ctx,
                            insert_block,
                            element.get_result(ctx),
                            &mut call_args,
                        )?;
                    }
                    continue;
                }
                lower_abi_call_arg(ctx, insert_block, value, &mut call_args)?;
            }

            let result_type = place_slot_opt(state, destination)?
                .map(|_| {
                    let result_tys = convert_return_ty(
                        tcx,
                        ctx,
                        mono_ty(tcx, state, body.local_decls[destination.local].ty),
                    )?;
                    match result_tys.as_slice() {
                        [] => Ok(None),
                        [ty] => Ok(Some(*ty)),
                        _ => Err("unsupported multi-value MIR call result".to_string()),
                    }
                })
                .transpose()?
                .flatten();

            // Results of calls into prebuilt std code use the real rustc
            // enum layout, which differs from this importer's discriminant-
            // first representation; such results are returned as raw bytes
            // and decoded afterwards.
            let mut external_enum_result = None;
            let mut result_type = result_type;

            let call = if let Some(slot) = virtual_slot {
                let vtable = vtable_ptr
                    .ok_or_else(|| "virtual call without a receiver argument".to_string())?;
                let usize_ty: TypeHandle = usize_ty(ctx).into();
                let ptr_size = tcx.data_layout.pointer_size().bytes();
                let offset = integer_constant(ctx, usize_ty, (slot * ptr_size) as u128)?;
                offset.get_operation().insert_at_back(insert_block, ctx);
                let fn_slot =
                    stair_mir::ops::PtrOffsetOp::new(ctx, vtable, offset.get_result(ctx));
                fn_slot.get_operation().insert_at_back(insert_block, ctx);
                let ptr_ty = llvm_ptr_ty(ctx);
                let fn_ptr = stair_mir::ops::LoadOp::new(ctx, fn_slot.get_result(ctx), ptr_ty);
                fn_ptr.get_operation().insert_at_back(insert_block, ctx);
                stair_mir::ops::CallOp::new_indirect(
                    ctx,
                    fn_ptr.get_result(ctx),
                    call_args,
                    result_type,
                )
            } else if let Some(callee_value) = callee_value {
                stair_mir::ops::CallOp::new_indirect(ctx, callee_value, call_args, result_type)
            } else {
                let callee = call_callee(tcx, state, body, func)?;
                let external_call = is_upstream_call(tcx, state, body, func);
                let imported = if let Some(instance) = call_instance(tcx, state, body, func)
                    && should_import_instance(tcx, instance)
                {
                    import_upstream_instance(tcx, ctx, state.module_body, instance).map_err(
                        |error| {
                            let mut legaliser = Legaliser::default();
                            let symbol = legaliser.legalise(tcx.symbol_name(instance).name);
                            format!("while importing instance {symbol}: {error}")
                        },
                    )?;
                    true
                } else {
                    false
                };
                if !imported && external_call {
                    let rust_ret_ty = mono_ty(tcx, state, body.local_decls[destination.local].ty);
                    let layout_sensitive = matches!(
                        runtime_ty(rust_ret_ty).kind(),
                        rustc_middle::ty::TyKind::Adt(_, _)
                            | rustc_middle::ty::TyKind::Closure(_, _)
                            | rustc_middle::ty::TyKind::Array(_, _)
                    ) || matches!(
                        runtime_ty(rust_ret_ty).kind(),
                        rustc_middle::ty::TyKind::Tuple(fields) if !fields.is_empty()
                    );
                    if result_type.is_some() && layout_sensitive {
                        let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
                        let real_size = rustc_layout_size_of_ty(tcx, typing_env, rust_ret_ty)?;
                        // Use a word-array blob so the decoded loads stay
                        // 8-byte aligned.
                        let word_ty: TypeHandle =
                            IntegerType::get(ctx, 64, Signedness::Unsigned).into();
                        let blob_ty: TypeHandle =
                            llvm::types::ArrayType::get(ctx, word_ty, real_size.div_ceil(8)).into();
                        result_type = Some(blob_ty);
                        external_enum_result = Some((rust_ret_ty, blob_ty));
                    }
                    let arg_types = call_args
                        .iter()
                        .map(|arg| arg.get_type(ctx))
                        .collect::<Vec<_>>();
                    declare_external_function(
                        ctx,
                        state.module_body,
                        callee.clone(),
                        arg_types,
                        result_type,
                    );
                }
                stair_mir::ops::CallOp::new_direct(ctx, callee, call_args, result_type)
            };
            call.get_operation().insert_at_back(insert_block, ctx);

            if let Some((rust_ret_ty, blob_ty)) = external_enum_result {
                let blob = call.get_operation().deref(ctx).get_result(0);
                let blob_slot = stair_mir::ops::AllocaOp::new(ctx, blob_ty);
                blob_slot.get_operation().insert_at_back(insert_block, ctx);
                let store = stair_mir::ops::StoreOp::new(ctx, blob, blob_slot.get_result(ctx));
                store.get_operation().insert_at_back(insert_block, ctx);
                let value = load_value_from_real_layout(
                    tcx,
                    ctx,
                    insert_block,
                    rust_ret_ty,
                    blob_slot.get_result(ctx),
                )?;
                store_place(tcx, ctx, state, insert_block, body, destination, value)?;
            } else if let Some(slot) = place_slot_opt(state, destination)?
                && call.get_operation().deref(ctx).get_num_results() > 0
            {
                let result = call.get_operation().deref(ctx).get_result(0);
                let result = normalize_bool_for_storage(
                    tcx,
                    ctx,
                    state,
                    insert_block,
                    body.local_decls[destination.local].ty,
                    result,
                )?;
                stair_mir::ops::StoreOp::new(ctx, result, slot)
                    .get_operation()
                    .insert_at_back(insert_block, ctx);
            }

            let Some(target) = target else {
                stair_mir::ops::UnreachableOp::new(ctx)
                    .get_operation()
                    .insert_at_back(insert_block, ctx);
                return Ok(());
            };
            let goto = stair_mir::ops::GotoOp::new(ctx, block_for(state, *target)?, vec![]);
            goto.get_operation().insert_at_back(insert_block, ctx);
            Ok(())
        }
        TerminatorKind::Assert { target, .. } => {
            let goto = stair_mir::ops::GotoOp::new(ctx, block_for(state, *target)?, vec![]);
            goto.get_operation().insert_at_back(insert_block, ctx);
            Ok(())
        }
        TerminatorKind::Drop { place, target, .. } => {
            let ty = mono_ty(tcx, state, place.ty(body, tcx).ty);
            let typing_env = state.instance.map_or_else(
                || body.typing_env(tcx),
                |_| rustc_middle::ty::TypingEnv::fully_monomorphized(),
            );
            if ty.needs_drop(tcx, typing_env) {
                let drop_instance = Instance::resolve_drop_in_place(tcx, ty);
                let mut legaliser = Legaliser::default();
                let symbol = legaliser.legalise(tcx.symbol_name(drop_instance).name);
                import_upstream_instance(tcx, ctx, state.module_body, drop_instance)
                    .map_err(|error| format!("while importing drop glue for {ty:?}: {error}"))?;
                let addr = place_addr(tcx, ctx, state, insert_block, body, place)?;
                let mut call_args = Vec::new();
                lower_abi_call_arg(ctx, insert_block, addr, &mut call_args)?;
                let call = stair_mir::ops::CallOp::new_direct(ctx, symbol, call_args, None);
                call.get_operation().insert_at_back(insert_block, ctx);
            }
            let goto = stair_mir::ops::GotoOp::new(ctx, block_for(state, *target)?, vec![]);
            goto.get_operation().insert_at_back(insert_block, ctx);
            Ok(())
        }
        other => Err(format!("unsupported MIR terminator: {other:?}")),
    }
}

fn call_callee<'tcx>(
    tcx: TyCtxt<'tcx>,
    state: &FunctionImportState<'tcx>,
    body: &Body<'tcx>,
    func: &Operand<'tcx>,
) -> Result<crate::identifier::Identifier, String> {
    call_symbol(tcx, state, body, func)
}

/// The `FnDef` (def id, generic args) of a call, whether the callee operand
/// is a constant or a zero-sized function-item value read from a place.
fn call_fn_def<'tcx>(
    tcx: TyCtxt<'tcx>,
    state: &FunctionImportState<'tcx>,
    body: &Body<'tcx>,
    func: &Operand<'tcx>,
) -> Option<(
    rustc_hir::def_id::DefId,
    rustc_middle::ty::GenericArgsRef<'tcx>,
)> {
    let func_ty = mono_ty(tcx, state, func.ty(body, tcx));
    let rustc_middle::ty::TyKind::FnDef(def_id, args) = runtime_ty(func_ty).kind() else {
        return None;
    };
    Some((*def_id, args))
}

fn call_instance<'tcx>(
    tcx: TyCtxt<'tcx>,
    state: &FunctionImportState<'tcx>,
    body: &Body<'tcx>,
    func: &Operand<'tcx>,
) -> Option<Instance<'tcx>> {
    let (def_id, args) = call_fn_def(tcx, state, body, func)?;
    let args = mono_generic_args(tcx, state, args);
    Instance::try_resolve(tcx, body.typing_env(tcx), def_id, args)
        .ok()
        .flatten()
}

fn mono_generic_args<'tcx>(
    tcx: TyCtxt<'tcx>,
    state: &FunctionImportState<'tcx>,
    args: rustc_middle::ty::GenericArgsRef<'tcx>,
) -> rustc_middle::ty::GenericArgsRef<'tcx> {
    state.instance.map_or(args, |instance| {
        instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            rustc_middle::ty::TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(args),
        )
    })
}

/// Require a SIMD vector type to be exactly 8 bytes (a NEON D register),
/// the only shape the scalar SWAR lowering handles.
fn require_8_byte_vector<'tcx>(
    tcx: TyCtxt<'tcx>,
    name: &str,
    vector_ty: Ty<'tcx>,
) -> Result<(), String> {
    let size = layout_size_of_ty(tcx, vector_ty)?;
    if size != 8 {
        return Err(format!(
            "unsupported {name} intrinsic vector size: {size} bytes (only 64-bit vectors are \
             lowered)"
        ));
    }
    Ok(())
}

/// The u64 bit pattern of an 8-byte SIMD vector value. Scalars are cast
/// directly; aggregate representations round-trip through a stack slot.
fn simd_value_bits(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    value: Value,
) -> Result<Value, String> {
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    if value.get_type(ctx).deref(ctx).is::<IntegerType>() {
        return Ok(cast_value_to_type(ctx, insert_block, value, u64_ty));
    }
    let slot = stair_mir::ops::AllocaOp::new(ctx, u64_ty);
    slot.get_operation().insert_at_back(insert_block, ctx);
    let slot = slot.get_result(ctx);
    let store = stair_mir::ops::StoreOp::new(ctx, value, slot);
    store.get_operation().insert_at_back(insert_block, ctx);
    let load = stair_mir::ops::LoadOp::new(ctx, slot, u64_ty);
    load.get_operation().insert_at_back(insert_block, ctx);
    Ok(load.get_result(ctx))
}

/// Store a u64 bit pattern into `destination`, whose type is an 8-byte SIMD
/// vector: spill the bits and reload them with the destination's real
/// layout, then store the place.
fn store_simd_bits<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    destination: &Place<'tcx>,
    bits: Value,
) -> Result<(), String> {
    let dest_rust_ty = mono_ty(tcx, state, destination.ty(body, tcx).ty);
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let slot = stair_mir::ops::AllocaOp::new(ctx, u64_ty);
    slot.get_operation().insert_at_back(insert_block, ctx);
    let slot = slot.get_result(ctx);
    let store = stair_mir::ops::StoreOp::new(ctx, bits, slot);
    store.get_operation().insert_at_back(insert_block, ctx);
    let value = load_value_from_real_layout(tcx, ctx, insert_block, dest_rust_ty, slot)?;
    store_place(tcx, ctx, state, insert_block, body, destination, value)
}

/// Split a `dyn` method receiver into its (data pointer, vtable pointer)
/// halves. Mirrors rustc_codegen_ssa's virtual-call handling: peel
/// `DispatchFromDyn` newtype wrappers (`Pin<&mut dyn ..>`, `Box<dyn ..>`
/// converted as single-field structs) until the fat pointer itself, then
/// take its two fields.
fn split_dyn_receiver(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    mut receiver: Value,
) -> Result<(Value, Value), String> {
    loop {
        let ty = receiver.get_type(ctx);
        let (num_fields, field_tys) = {
            let ty_ref = ty.deref(ctx);
            let Some(struct_ty) = ty_ref.downcast_ref::<llvm::types::StructType>() else {
                return Err(format!(
                    "unsupported dyn receiver type: {}",
                    ty_ref.disp(ctx)
                ));
            };
            let fields: Vec<TypeHandle> = (0..struct_ty.num_fields())
                .map(|index| struct_ty.field_type(index))
                .collect();
            (fields.len(), fields)
        };
        match num_fields {
            2 => {
                let data = emit_op(
                    stair_mir::ops::ExtractValueOp::new(ctx, receiver, vec![0], field_tys[0])
                        .get_operation(),
                    ctx,
                    insert_block,
                );
                let vtable = emit_op(
                    stair_mir::ops::ExtractValueOp::new(ctx, receiver, vec![1], field_tys[1])
                        .get_operation(),
                    ctx,
                    insert_block,
                );
                return Ok((data, vtable));
            }
            1 => {
                receiver = emit_op(
                    stair_mir::ops::ExtractValueOp::new(ctx, receiver, vec![0], field_tys[0])
                        .get_operation(),
                    ctx,
                    insert_block,
                );
            }
            other => {
                return Err(format!(
                    "unsupported dyn receiver shape: {other}-field struct"
                ));
            }
        }
    }
}

fn should_import_instance<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> bool {
    match instance.def {
        // Tuple-variant constructor bodies are synthesized by `instance_mir`.
        InstanceKind::Item(def_id) => {
            tcx.is_mir_available(def_id)
                || matches!(tcx.def_kind(def_id), rustc_hir::def::DefKind::Ctor(_, _))
        }
        // Shim MIR is synthesized on demand by `tcx.instance_mir`.
        InstanceKind::DropGlue(_, Some(_))
        | InstanceKind::ClosureOnceShim { .. }
        | InstanceKind::CloneShim(_, _)
        | InstanceKind::FnPtrShim(_, _)
        // Vtable and reify shims (receiver/ABI adapters referenced from
        // vtables and fn-pointer casts) exist in no upstream object; their
        // MIR is synthesized like the other shims.
        | InstanceKind::VTableShim(_)
        | InstanceKind::ReifyShim(_, _) => true,
        _ => false,
    }
}

fn is_unreachable_unchecked_call<'tcx>(
    tcx: TyCtxt<'tcx>,
    state: &FunctionImportState<'tcx>,
    body: &Body<'tcx>,
    func: &Operand<'tcx>,
) -> bool {
    call_instance(tcx, state, body, func).is_some_and(|instance| {
        tcx.def_path_str(instance.def.def_id())
            .contains("unreachable_unchecked")
    })
}

fn is_noop_intrinsic_call<'tcx>(
    tcx: TyCtxt<'tcx>,
    state: &FunctionImportState<'tcx>,
    body: &Body<'tcx>,
    func: &Operand<'tcx>,
) -> bool {
    call_instance(tcx, state, body, func).is_some_and(|instance| {
        let path = tcx.def_path_str(instance.def.def_id());
        path.contains("core::intrinsics::assert_inhabited")
    }) || call_fn_def(tcx, state, body, func).is_some_and(|(def_id, _)| {
        tcx.def_path_str(def_id)
            .contains("core::intrinsics::assert_inhabited")
    }) || call_symbol(tcx, state, body, func)
        .is_ok_and(|symbol| symbol.as_ref().contains("assert_inhabited"))
}

/// Lower well-known codegen intrinsic calls that have no MIR body. Returns
/// `true` when the call was handled and only the branch to the return target
/// still needs to be emitted.
#[allow(clippy::too_many_arguments)]
fn lower_known_intrinsic_call<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    func: &Operand<'tcx>,
    args: &[rustc_span::Spanned<Operand<'tcx>>],
    destination: &Place<'tcx>,
) -> Result<bool, String> {
    let Some(instance) = call_instance(tcx, state, body, func) else {
        return Ok(false);
    };
    let name = match instance.def {
        InstanceKind::Intrinsic(def_id) => tcx.item_name(def_id).to_string(),
        // Both explicit `ptr::drop_in_place` calls and drop-glue shims
        // resolve to this instance kind (the shim was named `ptr::drop_glue`
        // on newer nightlies; matching the kind covers both spellings).
        InstanceKind::DropGlue(..) => "drop_glue".to_string(),
        _ => return Ok(false),
    };

    match name.as_str() {
        // Pure optimization hints with no runtime semantics.
        "cold_path" => Ok(true),
        // `atomic_load`/`atomic_store` are `(*const T) -> T` and
        // `(*mut T, T) -> ()`, with the ordering as a const generic rather than
        // a runtime argument.
        //
        // These lower to plain loads and stores. On AArch64 a naturally aligned
        // access up to 8 bytes is single-copy atomic, so the *atomicity* holds;
        // what is dropped is the ordering, because no barrier is emitted. That
        // is sound only for the single-threaded programs this backend targets
        // (std reaches these through the uncontended `Mutex` behind
        // `io::stdin()`). Honouring the ordering needs real barrier/acquire
        // instructions, and read-modify-write atomics (`atomic_cxchg`,
        // `atomic_rmw*`) are deliberately not lowered here — they still fail
        // loudly as unresolved symbols rather than being silently mis-compiled.
        "atomic_load" => {
            if args.len() != 1 {
                return Err(format!(
                    "unsupported atomic_load intrinsic arity: {}",
                    args.len()
                ));
            }
            let ptr = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let value_ty = mono_ty(tcx, state, instance.args.type_at(0));
            let Some(load_ty) = convert_storage_ty(tcx, ctx, value_ty)? else {
                return Ok(true);
            };
            let load = stair_mir::ops::LoadOp::new(ctx, ptr, load_ty);
            load.get_operation().insert_at_back(insert_block, ctx);
            store_place(
                tcx,
                ctx,
                state,
                insert_block,
                body,
                destination,
                load.get_result(ctx),
            )?;
            Ok(true)
        }
        "atomic_store" => {
            if args.len() != 2 {
                return Err(format!(
                    "unsupported atomic_store intrinsic arity: {}",
                    args.len()
                ));
            }
            let ptr = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let value_ty = mono_ty(tcx, state, instance.args.type_at(0));
            let Some(store_ty) = convert_storage_ty(tcx, ctx, value_ty)? else {
                return Ok(true);
            };
            let value = import_operand(tcx, ctx, state, insert_block, body, &args[1].node)?;
            let value = normalize_bool_for_storage(tcx, ctx, state, insert_block, value_ty, value)?;
            let value = cast_value_to_type(ctx, insert_block, value, store_ty);
            let store = stair_mir::ops::StoreOp::new(ctx, value, ptr);
            store.get_operation().insert_at_back(insert_block, ctx);
            Ok(true)
        }
        // A genuinely atomic exchange, lowered to LLVM's aarch64 outline-
        // atomic helper (`__aarch64_swpN_acq_rel`) from compiler-builtins,
        // which dispatches to LSE `SWPAL` or an LL/SC loop at runtime. The
        // ordering is strengthened to acquire+release regardless of the
        // requested one, which is always sound. The helpers only exist in
        // aarch64-linux's compiler-builtins; on other targets the intrinsic
        // stays unhandled and fails loudly at link time (macOS std never
        // reaches it — its `Mutex` is `os_unfair_lock`-based).
        "atomic_xchg" => {
            if !(tcx.sess.target.arch.to_string() == "aarch64"
                && tcx.sess.target.os.to_string() == "linux")
            {
                return Ok(false);
            }
            if args.len() != 2 {
                return Err(format!(
                    "unsupported atomic_xchg intrinsic arity: {}",
                    args.len()
                ));
            }
            let value_ty = mono_ty(tcx, state, instance.args.type_at(0));
            let Some(storage_ty) = convert_storage_ty(tcx, ctx, value_ty)? else {
                return Ok(true);
            };
            let size = layout_size_of_ty(tcx, value_ty)?;
            if !matches!(size, 1 | 2 | 4 | 8) {
                return Err(format!(
                    "unsupported atomic_xchg operand size: {size} bytes"
                ));
            }
            let helper: crate::identifier::Identifier =
                format!("__aarch64_swp{size}_acq_rel").try_into().unwrap();
            let ptr_ty = llvm_ptr_ty(ctx);
            declare_external_function(
                ctx,
                state.module_body,
                helper.clone(),
                vec![storage_ty, ptr_ty],
                Some(storage_ty),
            );
            let ptr = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let value = import_operand(tcx, ctx, state, insert_block, body, &args[1].node)?;
            let value = normalize_bool_for_storage(tcx, ctx, state, insert_block, value_ty, value)?;
            let value = cast_value_to_type(ctx, insert_block, value, storage_ty);
            let mut call_args = Vec::new();
            lower_abi_call_arg(ctx, insert_block, value, &mut call_args)?;
            lower_abi_call_arg(ctx, insert_block, ptr, &mut call_args)?;
            let call =
                stair_mir::ops::CallOp::new_direct(ctx, helper, call_args, Some(storage_ty));
            call.get_operation().insert_at_back(insert_block, ctx);
            let old = call.get_result(ctx);
            store_place(tcx, ctx, state, insert_block, body, destination, old)?;
            Ok(true)
        }
        "write_bytes" => {
            if args.len() != 3 {
                return Err(format!(
                    "unsupported write_bytes intrinsic arity: {}",
                    args.len()
                ));
            }
            let dst = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let val = import_operand(tcx, ctx, state, insert_block, body, &args[1].node)?;
            let count = import_operand(tcx, ctx, state, insert_block, body, &args[2].node)?;
            let elem = instance.args.type_at(0);
            let elem_size = layout_size_of_ty(tcx, mono_ty(tcx, state, elem))?;
            let byte_count = scale_index(ctx, insert_block, count, elem_size)?;
            let val_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signed).into();
            let val = cast_value_to_type(ctx, insert_block, val, val_ty);

            let memset: crate::identifier::Identifier = "memset".try_into().unwrap();
            let ptr_ty = llvm_ptr_ty(ctx);
            let usize_ty: TypeHandle = usize_ty(ctx).into();
            declare_external_function(
                ctx,
                state.module_body,
                memset.clone(),
                vec![ptr_ty, val_ty, usize_ty],
                Some(ptr_ty),
            );
            let call = stair_mir::ops::CallOp::new_direct(
                ctx,
                memset,
                vec![dst, val, byte_count],
                Some(ptr_ty),
            );
            call.get_operation().insert_at_back(insert_block, ctx);
            Ok(true)
        }
        "ptr_offset_from_unsigned" | "ptr_offset_from" => {
            if args.len() != 2 {
                return Err(format!(
                    "unsupported ptr_offset_from intrinsic arity: {}",
                    args.len()
                ));
            }
            let lhs = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let rhs = import_operand(tcx, ctx, state, insert_block, body, &args[1].node)?;
            let usize_ty: TypeHandle = usize_ty(ctx).into();
            let lhs = cast_value_to_type(ctx, insert_block, lhs, usize_ty);
            let rhs = cast_value_to_type(ctx, insert_block, rhs, usize_ty);
            let diff = stair_mir::ops::SubOp::new(ctx, lhs, rhs);
            diff.get_operation().insert_at_back(insert_block, ctx);
            let mut result = diff.get_operation().deref(ctx).get_result(0);
            let elem = instance.args.type_at(0);
            let elem_size = layout_size_of_ty(tcx, mono_ty(tcx, state, elem))?;
            if elem_size > 1 {
                let size = integer_constant(ctx, usize_ty, elem_size as u128)?;
                size.get_operation().insert_at_back(insert_block, ctx);
                let div = stair_mir::ops::DivOp::new(ctx, result, size.get_result(ctx));
                div.get_operation().insert_at_back(insert_block, ctx);
                result = div.get_operation().deref(ctx).get_result(0);
            }
            let dest_ty =
                convert_immediate_ty(tcx, ctx, mono_ty(tcx, state, destination.ty(body, tcx).ty))?;
            let result = cast_value_to_type(ctx, insert_block, result, dest_ty);
            store_place(tcx, ctx, state, insert_block, body, destination, result)?;
            Ok(true)
        }
        "ctpop" => {
            if args.len() != 1 {
                return Err(format!("unsupported ctpop intrinsic arity: {}", args.len()));
            }
            let input = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let x = widen_to_u64(ctx, insert_block, input)?;
            let total = emit_popcount64(ctx, insert_block, x)?;
            let dest_ty =
                convert_immediate_ty(tcx, ctx, mono_ty(tcx, state, destination.ty(body, tcx).ty))?;
            let result = cast_value_to_type(ctx, insert_block, total, dest_ty);
            store_place(tcx, ctx, state, insert_block, body, destination, result)?;
            Ok(true)
        }
        "cttz" | "cttz_nonzero" => {
            if args.len() != 1 {
                return Err(format!("unsupported cttz intrinsic arity: {}", args.len()));
            }
            let input = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let width = input
                .get_type(ctx)
                .deref(ctx)
                .downcast_ref::<IntegerType>()
                .map(|ty| ty.width())
                .ok_or_else(|| "cttz on non-integer type".to_string())?;
            let total = if width == 128 {
                emit_cttz128(ctx, insert_block, input)?
            } else if width > 64 {
                return Err(format!("unsupported {width}-bit cttz intrinsic"));
            } else {
                let x = widen_to_u64(ctx, insert_block, input)?;
                emit_cttz64(ctx, insert_block, x, width)?
            };
            let dest_ty =
                convert_immediate_ty(tcx, ctx, mono_ty(tcx, state, destination.ty(body, tcx).ty))?;
            let result = cast_value_to_type(ctx, insert_block, total, dest_ty);
            store_place(tcx, ctx, state, insert_block, body, destination, result)?;
            Ok(true)
        }
        "ctlz" | "ctlz_nonzero" => {
            if args.len() != 1 {
                return Err(format!("unsupported ctlz intrinsic arity: {}", args.len()));
            }
            let input = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let width = input
                .get_type(ctx)
                .deref(ctx)
                .downcast_ref::<IntegerType>()
                .map(|ty| ty.width())
                .ok_or_else(|| "ctlz on non-integer type".to_string())?;
            let total = if width == 128 {
                emit_ctlz128(ctx, insert_block, input)?
            } else if width > 64 {
                return Err(format!("unsupported {width}-bit ctlz intrinsic"));
            } else {
                let x = widen_to_u64(ctx, insert_block, input)?;
                emit_ctlz64(ctx, insert_block, x, width)?
            };
            let dest_ty =
                convert_immediate_ty(tcx, ctx, mono_ty(tcx, state, destination.ty(body, tcx).ty))?;
            let result = cast_value_to_type(ctx, insert_block, total, dest_ty);
            store_place(tcx, ctx, state, insert_block, body, destination, result)?;
            Ok(true)
        }
        "rotate_left" | "rotate_right" => {
            if args.len() != 2 {
                return Err(format!(
                    "unsupported rotate intrinsic arity: {}",
                    args.len()
                ));
            }
            let input = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let width = input
                .get_type(ctx)
                .deref(ctx)
                .downcast_ref::<IntegerType>()
                .map(|ty| ty.width())
                .ok_or_else(|| "rotate on non-integer type".to_string())?;
            if width > 64 || !width.is_power_of_two() {
                return Err(format!("unsupported {width}-bit rotate intrinsic"));
            }
            let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
            let constant = |ctx: &mut Context, bits: u128| -> Result<Value, String> {
                let op = integer_constant(ctx, u64_ty, bits)?;
                op.get_operation().insert_at_back(insert_block, ctx);
                Ok(op.get_result(ctx))
            };
            let width_mask = (u64::MAX >> (64 - width as usize)) as u128;
            let mut x = widen_to_u64(ctx, insert_block, input)?;
            if width < 64 {
                // Widening may sign-extend; the rotate must only see the low
                // `width` bits.
                let mask = constant(ctx, width_mask)?;
                x = emit_op(
                    stair_mir::ops::BitAndOp::new(ctx, x, mask).get_operation(),
                    ctx,
                    insert_block,
                );
            }
            let shift = import_operand(tcx, ctx, state, insert_block, body, &args[1].node)?;
            let shift = widen_to_u64(ctx, insert_block, shift)?;
            // Rotation is modular: k = shift & (width - 1); the opposite
            // shift is (width - k) & (width - 1), which is 0 when k is 0 so
            // both halves degenerate to the untouched value.
            let modulus_mask = constant(ctx, (width - 1) as u128)?;
            let k = emit_op(
                stair_mir::ops::BitAndOp::new(ctx, shift, modulus_mask).get_operation(),
                ctx,
                insert_block,
            );
            let width_value = constant(ctx, width as u128)?;
            let complement = emit_op(
                stair_mir::ops::SubOp::new(ctx, width_value, k).get_operation(),
                ctx,
                insert_block,
            );
            let inv = emit_op(
                stair_mir::ops::BitAndOp::new(ctx, complement, modulus_mask).get_operation(),
                ctx,
                insert_block,
            );
            let (left_amount, right_amount) = if name == "rotate_left" {
                (k, inv)
            } else {
                (inv, k)
            };
            let left = emit_op(
                stair_mir::ops::ShlOp::new(ctx, x, left_amount).get_operation(),
                ctx,
                insert_block,
            );
            let right = emit_op(
                stair_mir::ops::ShrOp::new(ctx, x, right_amount).get_operation(),
                ctx,
                insert_block,
            );
            let mut rotated = emit_op(
                stair_mir::ops::BitOrOp::new(ctx, left, right).get_operation(),
                ctx,
                insert_block,
            );
            if width < 64 {
                let mask = constant(ctx, width_mask)?;
                rotated = emit_op(
                    stair_mir::ops::BitAndOp::new(ctx, rotated, mask).get_operation(),
                    ctx,
                    insert_block,
                );
            }
            let dest_ty =
                convert_immediate_ty(tcx, ctx, mono_ty(tcx, state, destination.ty(body, tcx).ty))?;
            let result = cast_value_to_type(ctx, insert_block, rotated, dest_ty);
            store_place(tcx, ctx, state, insert_block, body, destination, result)?;
            Ok(true)
        }
        // `copy(src, dst, count)` allows overlap (memmove);
        // `copy_nonoverlapping` does not (memcpy).
        "copy" | "copy_nonoverlapping" => {
            if args.len() != 3 {
                return Err(format!(
                    "unsupported copy intrinsic arity: {}",
                    args.len()
                ));
            }
            let src = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let dst = import_operand(tcx, ctx, state, insert_block, body, &args[1].node)?;
            let count = import_operand(tcx, ctx, state, insert_block, body, &args[2].node)?;
            let elem = instance.args.type_at(0);
            let elem_size = layout_size_of_ty(tcx, mono_ty(tcx, state, elem))?;
            let byte_count = scale_index(ctx, insert_block, count, elem_size)?;
            let helper: crate::identifier::Identifier = if name == "copy" {
                "memmove".try_into().unwrap()
            } else {
                "memcpy".try_into().unwrap()
            };
            let ptr_ty = llvm_ptr_ty(ctx);
            let usize_ty: TypeHandle = usize_ty(ctx).into();
            declare_external_function(
                ctx,
                state.module_body,
                helper.clone(),
                vec![ptr_ty, ptr_ty, usize_ty],
                Some(ptr_ty),
            );
            let call = stair_mir::ops::CallOp::new_direct(
                ctx,
                helper,
                vec![dst, src, byte_count],
                Some(ptr_ty),
            );
            call.get_operation().insert_at_back(insert_block, ctx);
            Ok(true)
        }
        "typed_swap_nonoverlapping" => {
            if args.len() != 2 {
                return Err(format!(
                    "unsupported typed_swap_nonoverlapping intrinsic arity: {}",
                    args.len()
                ));
            }
            let size = layout_size_of_ty(tcx, mono_ty(tcx, state, instance.args.type_at(0)))?;
            if size == 0 {
                return Ok(true);
            }
            let x = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let y = import_operand(tcx, ctx, state, insert_block, body, &args[1].node)?;
            // Swap through a stack temporary with three memcpys; the operands
            // are guaranteed non-overlapping.
            let byte_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
            let temp_ty: TypeHandle = llvm::types::ArrayType::get(ctx, byte_ty, size).into();
            let temp = stair_mir::ops::AllocaOp::new(ctx, temp_ty);
            temp.get_operation().insert_at_back(insert_block, ctx);
            let temp_ptr = temp.get_result(ctx);
            let memcpy: crate::identifier::Identifier = "memcpy".try_into().unwrap();
            let ptr_ty = llvm_ptr_ty(ctx);
            let usize_ty: TypeHandle = usize_ty(ctx).into();
            declare_external_function(
                ctx,
                state.module_body,
                memcpy.clone(),
                vec![ptr_ty, ptr_ty, usize_ty],
                Some(ptr_ty),
            );
            let size_value = integer_constant(ctx, usize_ty, size as u128)?;
            size_value.get_operation().insert_at_back(insert_block, ctx);
            let size_value = size_value.get_result(ctx);
            for (to, from) in [(temp_ptr, x), (x, y), (y, temp_ptr)] {
                let call = stair_mir::ops::CallOp::new_direct(
                    ctx,
                    memcpy.clone(),
                    vec![to, from, size_value],
                    Some(ptr_ty),
                );
                call.get_operation().insert_at_back(insert_block, ctx);
            }
            Ok(true)
        }
        // NEON 64-bit vector intrinsics reached through std's hashbrown group
        // scan (`uint8x8_t`/`int8x8_t`: 8 byte lanes in 64 bits). The backend
        // has no vector registers, so the lanes are computed as scalar SWAR /
        // per-lane code on the vector's u64 bit pattern; the D-register
        // vectors are 8 bytes, so a u64 round-trips their layout exactly.
        "simd_splat" => {
            if args.len() != 1 {
                return Err(format!(
                    "unsupported simd_splat intrinsic arity: {}",
                    args.len()
                ));
            }
            let vector_ty = mono_ty(tcx, state, instance.args.type_at(0));
            require_8_byte_vector(tcx, "simd_splat", vector_ty)?;
            let lane = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            if lane
                .get_type(ctx)
                .deref(ctx)
                .downcast_ref::<IntegerType>()
                .map(|ty| ty.width())
                != Some(8)
            {
                return Err("unsupported simd_splat lane type (expected u8)".to_string());
            }
            let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
            let lane = widen_to_u64(ctx, insert_block, lane)?;
            let byte_mask = integer_constant(ctx, u64_ty, 0xff)?;
            byte_mask.get_operation().insert_at_back(insert_block, ctx);
            let lane = emit_op(
                stair_mir::ops::BitAndOp::new(ctx, lane, byte_mask.get_result(ctx))
                    .get_operation(),
                ctx,
                insert_block,
            );
            let spread = integer_constant(ctx, u64_ty, 0x0101_0101_0101_0101)?;
            spread.get_operation().insert_at_back(insert_block, ctx);
            let bits = emit_op(
                stair_mir::ops::MulOp::new(ctx, lane, spread.get_result(ctx)).get_operation(),
                ctx,
                insert_block,
            );
            store_simd_bits(tcx, ctx, state, insert_block, body, destination, bits)?;
            Ok(true)
        }
        "simd_or" => {
            if args.len() != 2 {
                return Err(format!(
                    "unsupported simd_or intrinsic arity: {}",
                    args.len()
                ));
            }
            let vector_ty = mono_ty(tcx, state, instance.args.type_at(0));
            require_8_byte_vector(tcx, "simd_or", vector_ty)?;
            let lhs = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let rhs = import_operand(tcx, ctx, state, insert_block, body, &args[1].node)?;
            let lhs = simd_value_bits(ctx, insert_block, lhs)?;
            let rhs = simd_value_bits(ctx, insert_block, rhs)?;
            let bits = emit_op(
                stair_mir::ops::BitOrOp::new(ctx, lhs, rhs).get_operation(),
                ctx,
                insert_block,
            );
            store_simd_bits(tcx, ctx, state, insert_block, body, destination, bits)?;
            Ok(true)
        }
        "simd_extract" | "simd_extract_dyn" => {
            if args.len() != 2 {
                return Err(format!(
                    "unsupported simd_extract intrinsic arity: {}",
                    args.len()
                ));
            }
            let vector_ty = mono_ty(tcx, state, instance.args.type_at(0));
            require_8_byte_vector(tcx, "simd_extract", vector_ty)?;
            // Only the single-lane `uint64x1_t -> u64` form: the lane index
            // can only be zero, so the extract is the bit pattern itself.
            let lane_size = layout_size_of_ty(
                tcx,
                mono_ty(tcx, state, instance.args.type_at(1)),
            )?;
            if lane_size != 8 {
                return Err(format!(
                    "unsupported simd_extract lane size: {lane_size} bytes (expected a \
                     single-lane 64-bit vector)"
                ));
            }
            let vector = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let bits = simd_value_bits(ctx, insert_block, vector)?;
            let dest_ty =
                convert_immediate_ty(tcx, ctx, mono_ty(tcx, state, destination.ty(body, tcx).ty))?;
            let result = cast_value_to_type(ctx, insert_block, bits, dest_ty);
            store_place(tcx, ctx, state, insert_block, body, destination, result)?;
            Ok(true)
        }
        "simd_eq" => {
            if args.len() != 2 {
                return Err(format!(
                    "unsupported simd_eq intrinsic arity: {}",
                    args.len()
                ));
            }
            let vector_ty = mono_ty(tcx, state, instance.args.type_at(0));
            require_8_byte_vector(tcx, "simd_eq", vector_ty)?;
            let lhs = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let rhs = import_operand(tcx, ctx, state, insert_block, body, &args[1].node)?;
            let lhs = simd_value_bits(ctx, insert_block, lhs)?;
            let rhs = simd_value_bits(ctx, insert_block, rhs)?;
            // Byte-wise equality mask via the SWAR zero-byte trick on the
            // xor: `(t - 0x01..) & !t & 0x80..` marks equal lanes with 0x80,
            // then the mark is smeared down to fill the lane with 0xff.
            let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
            let constant = |ctx: &mut Context, bits: u128| -> Result<Value, String> {
                let op = integer_constant(ctx, u64_ty, bits)?;
                op.get_operation().insert_at_back(insert_block, ctx);
                Ok(op.get_result(ctx))
            };
            let t = emit_op(
                stair_mir::ops::BitXorOp::new(ctx, lhs, rhs).get_operation(),
                ctx,
                insert_block,
            );
            let low_ones = constant(ctx, 0x0101_0101_0101_0101)?;
            let minus = emit_op(
                stair_mir::ops::SubOp::new(ctx, t, low_ones).get_operation(),
                ctx,
                insert_block,
            );
            let all_ones = constant(ctx, u64::MAX as u128)?;
            let not_t = emit_op(
                stair_mir::ops::BitXorOp::new(ctx, t, all_ones).get_operation(),
                ctx,
                insert_block,
            );
            let and1 = emit_op(
                stair_mir::ops::BitAndOp::new(ctx, minus, not_t).get_operation(),
                ctx,
                insert_block,
            );
            let high_bits = constant(ctx, 0x8080_8080_8080_8080)?;
            let mut mask = emit_op(
                stair_mir::ops::BitAndOp::new(ctx, and1, high_bits).get_operation(),
                ctx,
                insert_block,
            );
            for shift in [1u128, 2, 4] {
                let amount = constant(ctx, shift)?;
                let shifted = emit_op(
                    stair_mir::ops::ShrOp::new(ctx, mask, amount).get_operation(),
                    ctx,
                    insert_block,
                );
                mask = emit_op(
                    stair_mir::ops::BitOrOp::new(ctx, mask, shifted).get_operation(),
                    ctx,
                    insert_block,
                );
            }
            store_simd_bits(tcx, ctx, state, insert_block, body, destination, mask)?;
            Ok(true)
        }
        "simd_lt" | "simd_ge" => {
            if args.len() != 2 {
                return Err(format!(
                    "unsupported {name} intrinsic arity: {}",
                    args.len()
                ));
            }
            let vector_ty = mono_ty(tcx, state, instance.args.type_at(0));
            require_8_byte_vector(tcx, name.as_str(), vector_ty)?;
            let lhs = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let rhs = import_operand(tcx, ctx, state, insert_block, body, &args[1].node)?;
            let lhs = simd_value_bits(ctx, insert_block, lhs)?;
            let rhs = simd_value_bits(ctx, insert_block, rhs)?;
            // Signed per-lane compare, one i8 lane at a time (correctness
            // over speed; the group scan only runs on 8 lanes).
            let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
            let i8_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Signed).into();
            let constant = |ctx: &mut Context, bits: u128| -> Result<Value, String> {
                let op = integer_constant(ctx, u64_ty, bits)?;
                op.get_operation().insert_at_back(insert_block, ctx);
                Ok(op.get_result(ctx))
            };
            let zero = constant(ctx, 0)?;
            let mut result = zero;
            for lane in 0..8u32 {
                let amount = constant(ctx, (lane * 8) as u128)?;
                let lhs_lane = emit_op(
                    stair_mir::ops::ShrOp::new(ctx, lhs, amount).get_operation(),
                    ctx,
                    insert_block,
                );
                let lhs_lane = cast_value_to_type(ctx, insert_block, lhs_lane, i8_ty);
                let rhs_lane = emit_op(
                    stair_mir::ops::ShrOp::new(ctx, rhs, amount).get_operation(),
                    ctx,
                    insert_block,
                );
                let rhs_lane = cast_value_to_type(ctx, insert_block, rhs_lane, i8_ty);
                let cond = if name == "simd_lt" {
                    stair_mir::ops::LtOp::new(ctx, lhs_lane, rhs_lane).get_operation()
                } else {
                    stair_mir::ops::GeOp::new(ctx, lhs_lane, rhs_lane).get_operation()
                };
                let cond = emit_op(cond, ctx, insert_block);
                let cond = cast_value_to_type(ctx, insert_block, cond, u64_ty);
                // true -> 0xff in this lane: (0 - cond) & 0xff, shifted home.
                let neg = emit_op(
                    stair_mir::ops::SubOp::new(ctx, zero, cond).get_operation(),
                    ctx,
                    insert_block,
                );
                let byte_mask = constant(ctx, 0xff)?;
                let lane_mask = emit_op(
                    stair_mir::ops::BitAndOp::new(ctx, neg, byte_mask).get_operation(),
                    ctx,
                    insert_block,
                );
                let placed = emit_op(
                    stair_mir::ops::ShlOp::new(ctx, lane_mask, amount).get_operation(),
                    ctx,
                    insert_block,
                );
                result = emit_op(
                    stair_mir::ops::BitOrOp::new(ctx, result, placed).get_operation(),
                    ctx,
                    insert_block,
                );
            }
            store_simd_bits(tcx, ctx, state, insert_block, body, destination, result)?;
            Ok(true)
        }
        "saturating_add" | "saturating_sub" => {
            if args.len() != 2 {
                return Err(format!(
                    "unsupported saturating intrinsic arity: {}",
                    args.len()
                ));
            }
            let lhs = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let rhs = import_operand(tcx, ctx, state, insert_block, body, &args[1].node)?;
            if !is_unsigned_integer_value(ctx, lhs) {
                return Err(format!("unsupported signed saturating intrinsic: {name}"));
            }
            let int_ty = lhs.get_type(ctx);
            let result = if name == "saturating_add" {
                // r = a + b; if r < a saturate to all-ones: r | (0 - (r < a)).
                let sum = stair_mir::ops::AddOp::new(ctx, lhs, rhs).get_operation();
                sum.insert_at_back(insert_block, ctx);
                let sum = sum.deref(ctx).get_result(0);
                let overflow = stair_mir::ops::LtOp::new(ctx, sum, lhs).get_operation();
                overflow.insert_at_back(insert_block, ctx);
                let overflow = overflow.deref(ctx).get_result(0);
                let overflow = cast_value_to_type(ctx, insert_block, overflow, int_ty);
                let zero = integer_constant(ctx, int_ty, 0)?;
                zero.get_operation().insert_at_back(insert_block, ctx);
                let mask =
                    stair_mir::ops::SubOp::new(ctx, zero.get_result(ctx), overflow).get_operation();
                mask.insert_at_back(insert_block, ctx);
                let mask = mask.deref(ctx).get_result(0);
                let saturated = stair_mir::ops::BitOrOp::new(ctx, sum, mask).get_operation();
                saturated.insert_at_back(insert_block, ctx);
                saturated.deref(ctx).get_result(0)
            } else {
                // r = (a - b) & (0 - (a >= b)): zero when the subtraction
                // would underflow.
                let diff = stair_mir::ops::SubOp::new(ctx, lhs, rhs).get_operation();
                diff.insert_at_back(insert_block, ctx);
                let diff = diff.deref(ctx).get_result(0);
                let no_borrow = stair_mir::ops::GeOp::new(ctx, lhs, rhs).get_operation();
                no_borrow.insert_at_back(insert_block, ctx);
                let no_borrow = no_borrow.deref(ctx).get_result(0);
                let no_borrow = cast_value_to_type(ctx, insert_block, no_borrow, int_ty);
                let zero = integer_constant(ctx, int_ty, 0)?;
                zero.get_operation().insert_at_back(insert_block, ctx);
                let mask = stair_mir::ops::SubOp::new(ctx, zero.get_result(ctx), no_borrow)
                    .get_operation();
                mask.insert_at_back(insert_block, ctx);
                let mask = mask.deref(ctx).get_result(0);
                let saturated = stair_mir::ops::BitAndOp::new(ctx, diff, mask).get_operation();
                saturated.insert_at_back(insert_block, ctx);
                saturated.deref(ctx).get_result(0)
            };
            store_place(tcx, ctx, state, insert_block, body, destination, result)?;
            Ok(true)
        }
        "arith_offset" => {
            if args.len() != 2 {
                return Err(format!(
                    "unsupported arith_offset intrinsic arity: {}",
                    args.len()
                ));
            }
            let ptr = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let count = import_operand(tcx, ctx, state, insert_block, body, &args[1].node)?;
            let elem = normalize_ty(tcx, mono_ty(tcx, state, instance.args.type_at(0)));
            let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
            let elem_size = rustc_layout_size_of_ty(tcx, typing_env, elem)?;
            let byte_offset = scale_index(ctx, insert_block, count, elem_size)?;
            let offset = stair_mir::ops::PtrOffsetOp::new(ctx, ptr, byte_offset);
            offset.get_operation().insert_at_back(insert_block, ctx);
            store_place(
                tcx,
                ctx,
                state,
                insert_block,
                body,
                destination,
                offset.get_result(ctx),
            )?;
            Ok(true)
        }
        "size_of_val" | "align_of_val" => {
            if args.len() != 1 {
                return Err(format!(
                    "unsupported {name} intrinsic arity: {}",
                    args.len()
                ));
            }
            let elem = normalize_ty(tcx, mono_ty(tcx, state, instance.args.type_at(0)));
            let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
            let usize_ty: TypeHandle = usize_ty(ctx).into();
            let result = match runtime_ty(elem).kind() {
                rustc_middle::ty::TyKind::Dynamic(_, _) => {
                    // Fat pointer {data, vtable}; the real vtable stores size
                    // at +8 and align at +16.
                    let fat = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
                    let ptr_ty = llvm_ptr_ty(ctx);
                    let vtable = stair_mir::ops::ExtractValueOp::new(ctx, fat, vec![1], ptr_ty);
                    vtable.get_operation().insert_at_back(insert_block, ctx);
                    let offset = if name == "size_of_val" { 8 } else { 16 };
                    let addr = ptr_offset_const(ctx, insert_block, vtable.get_result(ctx), offset)?;
                    let load = stair_mir::ops::LoadOp::new(ctx, addr, usize_ty);
                    load.get_operation().insert_at_back(insert_block, ctx);
                    load.get_result(ctx)
                }
                rustc_middle::ty::TyKind::Slice(inner) => {
                    let fat = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
                    let len = stair_mir::ops::ExtractValueOp::new(ctx, fat, vec![1], usize_ty);
                    len.get_operation().insert_at_back(insert_block, ctx);
                    if name == "size_of_val" {
                        let elem_size = rustc_layout_size_of_ty(tcx, typing_env, *inner)?;
                        scale_index(ctx, insert_block, len.get_result(ctx), elem_size)?
                    } else {
                        let align = rustc_layout_align_of_ty(tcx, typing_env, *inner)?;
                        let constant = integer_constant(ctx, usize_ty, align as u128)?;
                        constant.get_operation().insert_at_back(insert_block, ctx);
                        constant.get_result(ctx)
                    }
                }
                _ => {
                    let bytes = if name == "size_of_val" {
                        rustc_layout_size_of_ty(tcx, typing_env, elem)?
                    } else {
                        rustc_layout_align_of_ty(tcx, typing_env, elem)?
                    };
                    let constant = integer_constant(ctx, usize_ty, bytes as u128)?;
                    constant.get_operation().insert_at_back(insert_block, ctx);
                    constant.get_result(ctx)
                }
            };
            store_place(tcx, ctx, state, insert_block, body, destination, result)?;
            Ok(true)
        }
        "drop_glue" => {
            let ty = mono_ty(tcx, state, instance.args.type_at(0));
            let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
            if !ty.needs_drop(tcx, typing_env) {
                return Ok(true);
            }
            if args.len() != 1 {
                return Err(format!(
                    "unsupported drop_glue intrinsic arity: {}",
                    args.len()
                ));
            }
            let drop_instance = Instance::resolve_drop_in_place(tcx, ty);
            let mut legaliser = Legaliser::default();
            let symbol = legaliser.legalise(tcx.symbol_name(drop_instance).name);
            import_upstream_instance(tcx, ctx, state.module_body, drop_instance)
                .map_err(|error| format!("while importing drop glue for {ty:?}: {error}"))?;
            let addr = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
            let mut call_args = Vec::new();
            lower_abi_call_arg(ctx, insert_block, addr, &mut call_args)?;
            let call = stair_mir::ops::CallOp::new_direct(ctx, symbol, call_args, None);
            call.get_operation().insert_at_back(insert_block, ctx);
            Ok(true)
        }
        _ => Ok(false),
    }
}

/// Zero-extend an integer value of width <= 64 to u64.
fn widen_to_u64(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    value: Value,
) -> Result<Value, String> {
    let width = value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .map(|ty| ty.width())
        .ok_or_else(|| "bit intrinsic on non-integer type".to_string())?;
    if width > 64 {
        return Err("unsupported 128-bit bit intrinsic".to_string());
    }
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    Ok(cast_value_to_type(ctx, insert_block, value, u64_ty))
}

/// Branch-free trailing-zero count of the low `width` bits of a u64 value:
/// `popcount(((x & -x) - 1) & width_mask)`. For `x == 0` the mask makes this
/// exactly `width`, as the `cttz` intrinsic requires.
fn emit_cttz64(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    x: Value,
    width: u32,
) -> Result<Value, String> {
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let zero = integer_constant(ctx, u64_ty, 0)?;
    zero.get_operation().insert_at_back(insert_block, ctx);
    let neg = emit_op(
        stair_mir::ops::SubOp::new(ctx, zero.get_result(ctx), x).get_operation(),
        ctx,
        insert_block,
    );
    let low_bit = emit_op(
        stair_mir::ops::BitAndOp::new(ctx, x, neg).get_operation(),
        ctx,
        insert_block,
    );
    let one = integer_constant(ctx, u64_ty, 1)?;
    one.get_operation().insert_at_back(insert_block, ctx);
    let below = emit_op(
        stair_mir::ops::SubOp::new(ctx, low_bit, one.get_result(ctx)).get_operation(),
        ctx,
        insert_block,
    );
    let width_mask = integer_constant(ctx, u64_ty, u64::MAX as u128 >> (64 - width as usize))?;
    width_mask.get_operation().insert_at_back(insert_block, ctx);
    let masked = emit_op(
        stair_mir::ops::BitAndOp::new(ctx, below, width_mask.get_result(ctx)).get_operation(),
        ctx,
        insert_block,
    );
    emit_popcount64(ctx, insert_block, masked)
}

/// 128-bit trailing-zero count on 64-bit halves:
/// `cttz(x) = cttz64(lo) + (lo == 0 ? cttz64(hi) : 0)`. `cttz64` already
/// yields 64 for a zero half, so the total is 128 for `x == 0`.
fn emit_cttz128(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    input: Value,
) -> Result<Value, String> {
    let u128_ty: TypeHandle = IntegerType::get(ctx, 128, Signedness::Unsigned).into();
    let bits = cast_value_to_type(ctx, insert_block, input, u128_ty);
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let lo = cast_value_to_type(ctx, insert_block, bits, u64_ty);
    let sixty_four = integer_constant(ctx, u128_ty, 64)?;
    sixty_four.get_operation().insert_at_back(insert_block, ctx);
    let hi_wide = emit_op(
        stair_mir::ops::ShrOp::new(ctx, bits, sixty_four.get_result(ctx)).get_operation(),
        ctx,
        insert_block,
    );
    let hi = cast_value_to_type(ctx, insert_block, hi_wide, u64_ty);

    let lo_count = emit_cttz64(ctx, insert_block, lo, 64)?;
    let hi_count = emit_cttz64(ctx, insert_block, hi, 64)?;

    // (lo == 0 ? hi_count : 0) as `(0 - (lo == 0)) & hi_count`, branch-free.
    let zero = integer_constant(ctx, u64_ty, 0)?;
    zero.get_operation().insert_at_back(insert_block, ctx);
    let lo_is_zero = emit_op(
        stair_mir::ops::EqOp::new(ctx, lo, zero.get_result(ctx)).get_operation(),
        ctx,
        insert_block,
    );
    let lo_is_zero = cast_value_to_type(ctx, insert_block, lo_is_zero, u64_ty);
    let mask = emit_op(
        stair_mir::ops::SubOp::new(ctx, zero.get_result(ctx), lo_is_zero).get_operation(),
        ctx,
        insert_block,
    );
    let extra = emit_op(
        stair_mir::ops::BitAndOp::new(ctx, mask, hi_count).get_operation(),
        ctx,
        insert_block,
    );
    Ok(emit_op(
        stair_mir::ops::AddOp::new(ctx, lo_count, extra).get_operation(),
        ctx,
        insert_block,
    ))
}

/// Branch-free leading-zero count of the low `width` bits of `x` (a u64
/// value with any widening garbage cleared as part of the MSB alignment).
fn emit_ctlz64(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    mut x: Value,
    width: u32,
) -> Result<Value, String> {
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    // Align the operand's MSB with bit 63 so a 64-bit count is exact
    // (and clears any widening garbage above `width` on the way).
    if width < 64 {
        let up = integer_constant(ctx, u64_ty, (64 - width) as u128)?;
        up.get_operation().insert_at_back(insert_block, ctx);
        x = emit_op(
            stair_mir::ops::ShlOp::new(ctx, x, up.get_result(ctx)).get_operation(),
            ctx,
            insert_block,
        );
    }
    // Smear the highest set bit rightward; the complement then has
    // ones exactly in the leading-zero positions.
    for shift in [1u128, 2, 4, 8, 16, 32] {
        let amount = integer_constant(ctx, u64_ty, shift)?;
        amount.get_operation().insert_at_back(insert_block, ctx);
        let shifted = emit_op(
            stair_mir::ops::ShrOp::new(ctx, x, amount.get_result(ctx)).get_operation(),
            ctx,
            insert_block,
        );
        x = emit_op(
            stair_mir::ops::BitOrOp::new(ctx, x, shifted).get_operation(),
            ctx,
            insert_block,
        );
    }
    let ones = integer_constant(ctx, u64_ty, u64::MAX as u128)?;
    ones.get_operation().insert_at_back(insert_block, ctx);
    let inverted = emit_op(
        stair_mir::ops::BitXorOp::new(ctx, x, ones.get_result(ctx)).get_operation(),
        ctx,
        insert_block,
    );
    emit_popcount64(ctx, insert_block, inverted)
}

/// Branch-free 128-bit leading-zero count, mirroring [emit_cttz128] with the
/// halves' roles swapped: `ctlz(x) = ctlz64(hi) + (hi == 0 ? ctlz64(lo) : 0)`
/// (`ctlz64` of a zero half yields 64, so the total reaches 128 for zero).
fn emit_ctlz128(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    input: Value,
) -> Result<Value, String> {
    let u128_ty: TypeHandle = IntegerType::get(ctx, 128, Signedness::Unsigned).into();
    let bits = cast_value_to_type(ctx, insert_block, input, u128_ty);
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let lo = cast_value_to_type(ctx, insert_block, bits, u64_ty);
    let sixty_four = integer_constant(ctx, u128_ty, 64)?;
    sixty_four.get_operation().insert_at_back(insert_block, ctx);
    let hi_wide = emit_op(
        stair_mir::ops::ShrOp::new(ctx, bits, sixty_four.get_result(ctx)).get_operation(),
        ctx,
        insert_block,
    );
    let hi = cast_value_to_type(ctx, insert_block, hi_wide, u64_ty);

    let hi_count = emit_ctlz64(ctx, insert_block, hi, 64)?;
    let lo_count = emit_ctlz64(ctx, insert_block, lo, 64)?;

    // (hi == 0 ? lo_count : 0) as `(0 - (hi == 0)) & lo_count`, branch-free.
    let zero = integer_constant(ctx, u64_ty, 0)?;
    zero.get_operation().insert_at_back(insert_block, ctx);
    let hi_is_zero = emit_op(
        stair_mir::ops::EqOp::new(ctx, hi, zero.get_result(ctx)).get_operation(),
        ctx,
        insert_block,
    );
    let hi_is_zero = cast_value_to_type(ctx, insert_block, hi_is_zero, u64_ty);
    let mask = emit_op(
        stair_mir::ops::SubOp::new(ctx, zero.get_result(ctx), hi_is_zero).get_operation(),
        ctx,
        insert_block,
    );
    let extra = emit_op(
        stair_mir::ops::BitAndOp::new(ctx, mask, lo_count).get_operation(),
        ctx,
        insert_block,
    );
    Ok(emit_op(
        stair_mir::ops::AddOp::new(ctx, hi_count, extra).get_operation(),
        ctx,
        insert_block,
    ))
}

/// Branch-free 64-bit population count.
fn emit_popcount64(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    mut x: Value,
) -> Result<Value, String> {
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let constant = |ctx: &mut Context, bits: u128| -> Result<Value, String> {
        let op = integer_constant(ctx, u64_ty, bits)?;
        op.get_operation().insert_at_back(insert_block, ctx);
        Ok(op.get_result(ctx))
    };
    fn emit(ctx: &Context, insert_block: Ptr<BasicBlock>, op: Ptr<Operation>) -> Value {
        op.insert_at_back(insert_block, ctx);
        op.deref(ctx).get_result(0)
    }

    let one = constant(ctx, 1)?;
    let shifted = stair_mir::ops::ShrOp::new(ctx, x, one).get_operation();
    let shifted = emit(ctx, insert_block, shifted);
    let mask55 = constant(ctx, 0x5555_5555_5555_5555)?;
    let and55 = stair_mir::ops::BitAndOp::new(ctx, shifted, mask55).get_operation();
    let and55 = emit(ctx, insert_block, and55);
    let sub = stair_mir::ops::SubOp::new(ctx, x, and55).get_operation();
    x = emit(ctx, insert_block, sub);

    let mask33 = constant(ctx, 0x3333_3333_3333_3333)?;
    let low_pairs = stair_mir::ops::BitAndOp::new(ctx, x, mask33).get_operation();
    let low_pairs = emit(ctx, insert_block, low_pairs);
    let two = constant(ctx, 2)?;
    let shr2 = stair_mir::ops::ShrOp::new(ctx, x, two).get_operation();
    let shr2 = emit(ctx, insert_block, shr2);
    let high_pairs = stair_mir::ops::BitAndOp::new(ctx, shr2, mask33).get_operation();
    let high_pairs = emit(ctx, insert_block, high_pairs);
    let pair_sum = stair_mir::ops::AddOp::new(ctx, low_pairs, high_pairs).get_operation();
    x = emit(ctx, insert_block, pair_sum);

    let four = constant(ctx, 4)?;
    let shr4 = stair_mir::ops::ShrOp::new(ctx, x, four).get_operation();
    let shr4 = emit(ctx, insert_block, shr4);
    let nibble_sum = stair_mir::ops::AddOp::new(ctx, x, shr4).get_operation();
    let nibble_sum = emit(ctx, insert_block, nibble_sum);
    let mask0f = constant(ctx, 0x0f0f_0f0f_0f0f_0f0f)?;
    let nibbles = stair_mir::ops::BitAndOp::new(ctx, nibble_sum, mask0f).get_operation();
    let nibbles = emit(ctx, insert_block, nibbles);

    let ones = constant(ctx, 0x0101_0101_0101_0101)?;
    let spread = stair_mir::ops::MulOp::new(ctx, nibbles, ones).get_operation();
    let spread = emit(ctx, insert_block, spread);
    let fifty_six = constant(ctx, 56)?;
    let total = stair_mir::ops::ShrOp::new(ctx, spread, fifty_six).get_operation();
    Ok(emit(ctx, insert_block, total))
}

fn import_upstream_instance<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    module_body: Ptr<BasicBlock>,
    instance: Instance<'tcx>,
) -> Result<(), String> {
    let mut legaliser = Legaliser::default();
    let symbol = legaliser.legalise(tcx.symbol_name(instance).name);
    if symbol_exists(ctx, module_body, &symbol.to_string()) {
        return Ok(());
    }
    let body = tcx.instance_mir(instance.def);
    import_function(
        tcx,
        ctx,
        module_body,
        symbol.clone(),
        body,
        false,
        Some(instance),
    )?;
    // Shared-generic instances also exist in prebuilt std rlibs; keep our
    // copy internal so the object does not export colliding symbols.
    set_internal_linkage(ctx, module_body, &symbol);
    Ok(())
}

fn set_internal_linkage(
    ctx: &mut Context,
    module_body: Ptr<BasicBlock>,
    symbol: &crate::identifier::Identifier,
) {
    if std::env::var_os("STAIR_DEBUG_EXPORT_ALL").is_some() {
        return;
    }
    set_function_linkage(ctx, module_body, symbol, LinkageAttr::InternalLinkage);
}

fn set_function_linkage(
    ctx: &mut Context,
    module_body: Ptr<BasicBlock>,
    symbol: &crate::identifier::Identifier,
    linkage: LinkageAttr,
) {
    let func = module_body.deref(ctx).iter(ctx).find(|op| {
        let op_obj = Operation::get_op_dyn(*op, ctx);
        op_cast::<dyn SymbolOpInterface>(&*op_obj)
            .is_some_and(|symbol_op| symbol_op.get_symbol_name(ctx) == *symbol)
    });
    if let Some(func) = func {
        if let Some(llvm_func) = Operation::get_op::<llvm::ops::FuncOp>(func, ctx) {
            llvm_func.set_attr_llvm_function_linkage(ctx, linkage);
        } else {
            // A `mir.func`: mir-lower's propagate_linkage_attr carries this
            // key onto the lowered `llvm.func`.
            func.deref_mut(ctx)
                .attributes
                .set(ox::func_linkage_key(), linkage);
        }
    }
}

fn symbol_exists(ctx: &Context, module_body: Ptr<BasicBlock>, symbol: &str) -> bool {
    module_body.deref(ctx).iter(ctx).any(|op| {
        let op_obj = Operation::get_op_dyn(op, ctx);
        op_cast::<dyn SymbolOpInterface>(&*op_obj)
            .is_some_and(|symbol_op| symbol_op.get_symbol_name(ctx).as_ref() == symbol)
    })
}

fn is_upstream_call<'tcx>(
    tcx: TyCtxt<'tcx>,
    state: &FunctionImportState<'tcx>,
    body: &Body<'tcx>,
    func: &Operand<'tcx>,
) -> bool {
    let Some((def_id, _)) = call_fn_def(tcx, state, body, func) else {
        return false;
    };
    !def_id.is_local()
}

/// How one MIR parameter maps to ABI-level arguments.
enum ParamAbi {
    Single(ArgAbi),
    /// "rust-call" spread argument: each tuple element lowered separately.
    Spread(Vec<(TypeHandle, ArgAbi)>),
}

/// Rebuild one Rust-level value from its incoming ABI arguments.
fn reassemble_abi_arg(
    ctx: &mut Context,
    entry: Ptr<BasicBlock>,
    insert_block: Ptr<BasicBlock>,
    value_ty: TypeHandle,
    abi: ArgAbi,
    block_arg_idx: &mut usize,
) -> Value {
    match abi {
        ArgAbi::Indirect => {
            let ptr = entry.deref(ctx).get_argument(*block_arg_idx);
            *block_arg_idx += 1;
            let load = stair_mir::ops::LoadOp::new(ctx, ptr, value_ty);
            load.get_operation().insert_at_back(insert_block, ctx);
            load.get_result(ctx)
        }
        ArgAbi::Leaves(group) if group.len() == 1 && group[0].0.is_empty() => {
            let arg = entry.deref(ctx).get_argument(*block_arg_idx);
            *block_arg_idx += 1;
            arg
        }
        ArgAbi::Leaves(group) => {
            let undef = stair_mir::ops::UndefOp::new(ctx, value_ty);
            undef.get_operation().insert_at_back(insert_block, ctx);
            let mut current = undef.get_result(ctx);
            for (indices, _) in group {
                let field = entry.deref(ctx).get_argument(*block_arg_idx);
                *block_arg_idx += 1;
                let insert = stair_mir::ops::InsertValueOp::new(ctx, field, current, indices);
                insert.get_operation().insert_at_back(insert_block, ctx);
                current = insert.get_result(ctx);
            }
            current
        }
    }
}

/// How one Rust-level argument maps to ABI-level arguments.
enum ArgAbi {
    /// Flattened into scalar leaves passed directly.
    Leaves(Vec<(Vec<u32>, TypeHandle)>),
    /// Copied to a stack temporary and passed by pointer (aggregates larger
    /// than two registers or containing arrays), matching the AArch64 rule
    /// for composites bigger than 16 bytes.
    Indirect,
}

fn arg_abi_for_ty(ctx: &Context, ty: TypeHandle) -> Result<ArgAbi, String> {
    // Zero-sized types (`()`, `!`, captureless closures) have no ABI
    // presence: no signature input, no call argument, no entry-block
    // argument. This must stay symmetric across import_function signatures,
    // entry reassembly, and lower_abi_call_arg — an asymmetry shifts every
    // later parameter one register over.
    if let Ok(0) = stair_ty_size(ctx, ty) {
        return Ok(ArgAbi::Leaves(Vec::new()));
    }
    let ty_ref = ty.deref(ctx);
    let is_aggregate =
        ty_ref.is::<llvm::types::StructType>() || ty_ref.is::<llvm::types::ArrayType>();
    drop(ty_ref);
    if is_aggregate {
        if stair_ty_size(ctx, ty)? > 16 {
            return Ok(ArgAbi::Indirect);
        }
        if let Ok(leaves) = simple_abi_leaves_for_ty(ctx, ty) {
            return Ok(ArgAbi::Leaves(leaves));
        }
        return Ok(ArgAbi::Indirect);
    }
    Ok(ArgAbi::Leaves(simple_abi_leaves_for_ty(ctx, ty)?))
}

/// Byte size of a converted STAIR type, mirroring the AArch64 lowering's
/// `stack_size_of` alignment rules.
fn stair_ty_size(ctx: &Context, ty: TypeHandle) -> Result<u64, String> {
    let ty_ref = ty.deref(ctx);
    if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
        return Ok((int_ty.width() as u64).div_ceil(8).max(1));
    }
    if ty_ref.is::<llvm::types::PointerType>() || ty_ref.is::<crate::dialects::dialect_mir::types::MirPtrType>() {
        return Ok(8);
    }
    if ty_ref.is::<FP32Type>() {
        return Ok(4);
    }
    if ty_ref.is::<FP64Type>() {
        return Ok(8);
    }
    if ty_ref.is::<UnitType>() {
        return Ok(0);
    }
    if let Some(array_ty) = ty_ref.downcast_ref::<llvm::types::ArrayType>() {
        let elem = array_ty.elem_type();
        let len = array_ty.size();
        drop(ty_ref);
        let stride = align_to(stair_ty_size(ctx, elem)?, stair_ty_align(ctx, elem)?);
        return Ok(stride * len);
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<llvm::types::StructType>() {
        if struct_ty.is_opaque() {
            return Err("opaque struct in ABI sizing".to_string());
        }
        let fields: Vec<_> = struct_ty.fields().collect();
        drop(ty_ref);
        let mut size = 0u64;
        let mut align = 1u64;
        for field in fields {
            let field_size = stair_ty_size(ctx, field)?;
            if field_size == 0 {
                continue;
            }
            let field_align = stair_ty_align(ctx, field)?;
            size = align_to(size, field_align) + field_size;
            align = align.max(field_align);
        }
        return Ok(align_to(size, align));
    }
    Err(format!("unsupported ABI type: {:?}", &*ty_ref))
}

fn stair_ty_align(ctx: &Context, ty: TypeHandle) -> Result<u64, String> {
    let ty_ref = ty.deref(ctx);
    if ty_ref.is::<UnitType>() {
        return Ok(1);
    }
    if let Some(array_ty) = ty_ref.downcast_ref::<llvm::types::ArrayType>() {
        let elem = array_ty.elem_type();
        drop(ty_ref);
        return stair_ty_align(ctx, elem);
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<llvm::types::StructType>() {
        if struct_ty.is_opaque() {
            return Err("opaque struct in ABI sizing".to_string());
        }
        let fields: Vec<_> = struct_ty.fields().collect();
        drop(ty_ref);
        let mut align = 1u64;
        for field in fields {
            if stair_ty_size(ctx, field)? == 0 {
                continue;
            }
            align = align.max(stair_ty_align(ctx, field)?);
        }
        return Ok(align);
    }
    drop(ty_ref);
    Ok(stair_ty_size(ctx, ty)?.min(8).max(1))
}

fn lower_abi_call_arg(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    value: Value,
    out: &mut Vec<Value>,
) -> Result<(), String> {
    let value_ty = value.get_type(ctx);
    match arg_abi_for_ty(ctx, value_ty)? {
        ArgAbi::Indirect => {
            let slot = stair_mir::ops::AllocaOp::new(ctx, value_ty);
            slot.get_operation().insert_at_back(insert_block, ctx);
            let store = stair_mir::ops::StoreOp::new(ctx, value, slot.get_result(ctx));
            store.get_operation().insert_at_back(insert_block, ctx);
            out.push(slot.get_result(ctx));
            Ok(())
        }
        ArgAbi::Leaves(leaves) => {
            if leaves.len() == 1 && leaves[0].0.is_empty() {
                out.push(value);
                return Ok(());
            }
            for (indices, result_ty) in leaves {
                let field = stair_mir::ops::ExtractValueOp::new(ctx, value, indices, result_ty);
                field.get_operation().insert_at_back(insert_block, ctx);
                out.push(field.get_result(ctx));
            }
            Ok(())
        }
    }
}

fn lower_overflow_binary(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    op: BinOp,
    lhs: Value,
    rhs: Value,
) -> Result<Value, String> {
    if is_unsigned_integer_value(ctx, lhs) {
        lower_unsigned_overflow_binary(ctx, insert_block, op, lhs, rhs)
    } else {
        lower_signed_overflow_binary(ctx, insert_block, op, lhs, rhs)
    }
}

fn lower_unsigned_overflow_binary(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    op: BinOp,
    lhs: Value,
    rhs: Value,
) -> Result<Value, String> {
    if matches!(op, BinOp::MulWithOverflow) {
        return lower_unsigned_mul_overflow(ctx, insert_block, lhs, rhs);
    }

    let wrapped = match op {
        BinOp::AddWithOverflow => stair_mir::ops::AddOp::new(ctx, lhs, rhs).get_operation(),
        BinOp::SubWithOverflow => stair_mir::ops::SubOp::new(ctx, lhs, rhs).get_operation(),
        other => return Err(format!("unsupported MIR overflow binary op: {other:?}")),
    };
    wrapped.insert_at_back(insert_block, ctx);
    let wrapped_value = wrapped.deref(ctx).get_result(0);

    let overflow = match op {
        BinOp::AddWithOverflow => {
            stair_mir::ops::LtOp::new(ctx, wrapped_value, lhs).get_operation()
        }
        BinOp::SubWithOverflow => stair_mir::ops::LtOp::new(ctx, lhs, rhs).get_operation(),
        _ => unreachable!(),
    };
    overflow.insert_at_back(insert_block, ctx);
    let overflow_value = overflow.deref(ctx).get_result(0);
    Ok(pack_overflow_result(
        ctx,
        insert_block,
        wrapped_value,
        overflow_value,
    ))
}

fn lower_signed_overflow_binary(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    op: BinOp,
    lhs: Value,
    rhs: Value,
) -> Result<Value, String> {
    let ty = lhs.get_type(ctx);
    if !ty
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|int_ty| int_ty.signedness() == Signedness::Signed)
    {
        return Err(format!(
            "unsupported non-integer MIR overflow binary op: {op:?}"
        ));
    }

    if matches!(op, BinOp::MulWithOverflow) {
        return lower_signed_mul_overflow(ctx, insert_block, lhs, rhs);
    }

    let wrapped = match op {
        BinOp::AddWithOverflow => stair_mir::ops::AddOp::new(ctx, lhs, rhs).get_operation(),
        BinOp::SubWithOverflow => stair_mir::ops::SubOp::new(ctx, lhs, rhs).get_operation(),
        other => return Err(format!("unsupported MIR overflow binary op: {other:?}")),
    };
    wrapped.insert_at_back(insert_block, ctx);
    let wrapped_value = wrapped.deref(ctx).get_result(0);

    // Add overflows iff the operands share a sign and the result's differs:
    // the sign bit of `(res ^ lhs) & (res ^ rhs)`. Sub overflows iff the
    // operands' signs differ and the result's sign differs from lhs's: the
    // sign bit of `(lhs ^ rhs) & (lhs ^ res)`.
    let (xor_a, xor_b) = match op {
        BinOp::AddWithOverflow => ((wrapped_value, lhs), (wrapped_value, rhs)),
        BinOp::SubWithOverflow => ((lhs, rhs), (lhs, wrapped_value)),
        _ => unreachable!(),
    };
    let a = stair_mir::ops::BitXorOp::new(ctx, xor_a.0, xor_a.1).get_operation();
    a.insert_at_back(insert_block, ctx);
    let a = a.deref(ctx).get_result(0);
    let b = stair_mir::ops::BitXorOp::new(ctx, xor_b.0, xor_b.1).get_operation();
    b.insert_at_back(insert_block, ctx);
    let b = b.deref(ctx).get_result(0);
    let sign = stair_mir::ops::BitAndOp::new(ctx, a, b).get_operation();
    sign.insert_at_back(insert_block, ctx);
    let sign = sign.deref(ctx).get_result(0);

    let zero = integer_constant(ctx, ty, 0)?;
    zero.get_operation().insert_at_back(insert_block, ctx);
    // A signed lt: `ty` is a Signed integer type, so dialect-mir picks slt.
    let overflow = stair_mir::ops::LtOp::new(ctx, sign, zero.get_result(ctx)).get_operation();
    overflow.insert_at_back(insert_block, ctx);
    let overflow_value = overflow.deref(ctx).get_result(0);
    Ok(pack_overflow_result(
        ctx,
        insert_block,
        wrapped_value,
        overflow_value,
    ))
}

fn lower_signed_mul_overflow(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    lhs: Value,
    rhs: Value,
) -> Result<Value, String> {
    let ty = lhs.get_type(ctx);
    let width = ty
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .map(|int_ty| int_ty.width())
        .ok_or_else(|| "MIR mul-with-overflow on non-integer type".to_string())?;
    if width == 128 {
        return lower_signed_mul_overflow_128(ctx, insert_block, lhs, rhs);
    }
    if width > 64 {
        return Err(format!(
            "unsupported {width}-bit MIR mul-with-overflow"
        ));
    }

    // Multiply at double width (sext: the operands are Signed), then check
    // that the product survives a round-trip through the narrow type.
    let wide_ty: TypeHandle = IntegerType::get(ctx, width * 2, Signedness::Signed).into();
    let wide_lhs = cast_value_to_type(ctx, insert_block, lhs, wide_ty);
    let wide_rhs = cast_value_to_type(ctx, insert_block, rhs, wide_ty);
    let wide_mul = stair_mir::ops::MulOp::new(ctx, wide_lhs, wide_rhs);
    wide_mul.get_operation().insert_at_back(insert_block, ctx);
    let wide_value = wide_mul.get_operation().deref(ctx).get_result(0);
    let wrapped_value = cast_value_to_type(ctx, insert_block, wide_value, ty);
    let widened_back = cast_value_to_type(ctx, insert_block, wrapped_value, wide_ty);
    let overflow = stair_mir::ops::NeOp::new(ctx, wide_value, widened_back);
    overflow.get_operation().insert_at_back(insert_block, ctx);
    let overflow_value = overflow.get_result(ctx);
    Ok(pack_overflow_result(
        ctx,
        insert_block,
        wrapped_value,
        overflow_value,
    ))
}

/// 128-bit signed mul-with-overflow. The wrapped product's bits are the same
/// as the unsigned wrapping product; the exact 256-bit product's high half is
/// recovered from the unsigned high half with the standard sign fixup
/// `smulh(a, b) = umulh(a, b) - (a < 0 ? b : 0) - (b < 0 ? a : 0)` (all mod
/// 2^128), and overflow holds iff that high half differs from the sign
/// extension of the wrapped low half.
fn lower_signed_mul_overflow_128(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    lhs: Value,
    rhs: Value,
) -> Result<Value, String> {
    let u128_ty: TypeHandle = IntegerType::get(ctx, 128, Signedness::Unsigned).into();
    let lhs_bits = cast_value_to_type(ctx, insert_block, lhs, u128_ty);
    let rhs_bits = cast_value_to_type(ctx, insert_block, rhs, u128_ty);

    let wrapped = emit_op(
        stair_mir::ops::MulOp::new(ctx, lhs, rhs).get_operation(),
        ctx,
        insert_block,
    );

    let unsigned_high = emit_umulh128(ctx, insert_block, lhs_bits, rhs_bits)?;
    // (a < 0 ? b : 0) as `sign_mask(a) & b`, branch-free.
    let lhs_mask = emit_sign_mask128(ctx, insert_block, lhs_bits)?;
    let rhs_mask = emit_sign_mask128(ctx, insert_block, rhs_bits)?;
    let lhs_fix = emit_op(
        stair_mir::ops::BitAndOp::new(ctx, lhs_mask, rhs_bits).get_operation(),
        ctx,
        insert_block,
    );
    let rhs_fix = emit_op(
        stair_mir::ops::BitAndOp::new(ctx, rhs_mask, lhs_bits).get_operation(),
        ctx,
        insert_block,
    );
    let signed_high = emit_op(
        stair_mir::ops::SubOp::new(ctx, unsigned_high, lhs_fix).get_operation(),
        ctx,
        insert_block,
    );
    let signed_high = emit_op(
        stair_mir::ops::SubOp::new(ctx, signed_high, rhs_fix).get_operation(),
        ctx,
        insert_block,
    );

    // Sign extension of the wrapped low half: all-ones iff it is negative.
    let wrapped_bits = cast_value_to_type(ctx, insert_block, wrapped, u128_ty);
    let expected_high = emit_sign_mask128(ctx, insert_block, wrapped_bits)?;
    let overflow = emit_op(
        stair_mir::ops::NeOp::new(ctx, signed_high, expected_high).get_operation(),
        ctx,
        insert_block,
    );
    Ok(pack_overflow_result(ctx, insert_block, wrapped, overflow))
}

/// Package a wrapped result and its overflow flag into the `(value, i8)`
/// struct the importer uses for the MIR `*WithOverflow` result tuple.
fn pack_overflow_result(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    wrapped_value: Value,
    overflow_value: Value,
) -> Value {
    let overflow_ty = bool_storage_ty(ctx);
    let overflow_value = cast_value_to_type(ctx, insert_block, overflow_value, overflow_ty);

    let result_ty =
        llvm::types::StructType::get_unnamed(ctx, vec![wrapped_value.get_type(ctx), overflow_ty])
            .into();
    let undef = stair_mir::ops::UndefOp::new(ctx, result_ty);
    undef.get_operation().insert_at_back(insert_block, ctx);
    let with_value =
        stair_mir::ops::InsertValueOp::new(ctx, wrapped_value, undef.get_result(ctx), vec![0]);
    with_value.get_operation().insert_at_back(insert_block, ctx);
    let with_overflow = stair_mir::ops::InsertValueOp::new(
        ctx,
        overflow_value,
        with_value.get_result(ctx),
        vec![1],
    );
    with_overflow
        .get_operation()
        .insert_at_back(insert_block, ctx);
    with_overflow.get_result(ctx)
}

/// Insert a freshly built operation and return its first result value.
fn emit_op(op: Ptr<Operation>, ctx: &Context, insert_block: Ptr<BasicBlock>) -> Value {
    op.insert_at_back(insert_block, ctx);
    op.deref(ctx).get_result(0)
}

/// The high 128 bits of the exact 256-bit unsigned product `lhs * rhs` (both
/// u128), via 64-bit half decomposition. With `a = a_hi·2^64 + a_lo` (halves
/// < 2^64) the partial products `a_lo·b_lo`, `a_hi·b_lo`, `a_lo·b_hi`, and
/// `a_hi·b_hi` are all exact in u128, as is the carry accumulation below, so
/// no wider type is needed. This is the standard compiler-builtins-style
/// decomposition for 128-bit mul-with-overflow, where no wider type exists
/// to widen into.
fn emit_umulh128(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    lhs: Value,
    rhs: Value,
) -> Result<Value, String> {
    let u128_ty: TypeHandle = IntegerType::get(ctx, 128, Signedness::Unsigned).into();
    let constant = |ctx: &mut Context, bits: u128| -> Result<Value, String> {
        let op = integer_constant(ctx, u128_ty, bits)?;
        op.get_operation().insert_at_back(insert_block, ctx);
        Ok(op.get_result(ctx))
    };
    let mask = constant(ctx, u64::MAX as u128)?;
    let sixty_four = constant(ctx, 64)?;

    let a_lo = emit_op(
        stair_mir::ops::BitAndOp::new(ctx, lhs, mask).get_operation(),
        ctx,
        insert_block,
    );
    let a_hi = emit_op(
        stair_mir::ops::ShrOp::new(ctx, lhs, sixty_four).get_operation(),
        ctx,
        insert_block,
    );
    let b_lo = emit_op(
        stair_mir::ops::BitAndOp::new(ctx, rhs, mask).get_operation(),
        ctx,
        insert_block,
    );
    let b_hi = emit_op(
        stair_mir::ops::ShrOp::new(ctx, rhs, sixty_four).get_operation(),
        ctx,
        insert_block,
    );

    let lo_lo = emit_op(
        stair_mir::ops::MulOp::new(ctx, a_lo, b_lo).get_operation(),
        ctx,
        insert_block,
    );
    let cross_a = emit_op(
        stair_mir::ops::MulOp::new(ctx, a_hi, b_lo).get_operation(),
        ctx,
        insert_block,
    );
    let cross_b = emit_op(
        stair_mir::ops::MulOp::new(ctx, a_lo, b_hi).get_operation(),
        ctx,
        insert_block,
    );
    let hi_hi = emit_op(
        stair_mir::ops::MulOp::new(ctx, a_hi, b_hi).get_operation(),
        ctx,
        insert_block,
    );

    // carry = ((cross_a & M) + (cross_b & M) + (lo_lo >> 64)) >> 64: the
    // carry out of bit 127 of the full product (sum of three values < 2^64,
    // exact in u128).
    let cross_a_lo = emit_op(
        stair_mir::ops::BitAndOp::new(ctx, cross_a, mask).get_operation(),
        ctx,
        insert_block,
    );
    let cross_b_lo = emit_op(
        stair_mir::ops::BitAndOp::new(ctx, cross_b, mask).get_operation(),
        ctx,
        insert_block,
    );
    let lo_lo_hi = emit_op(
        stair_mir::ops::ShrOp::new(ctx, lo_lo, sixty_four).get_operation(),
        ctx,
        insert_block,
    );
    let mid_sum = emit_op(
        stair_mir::ops::AddOp::new(ctx, cross_a_lo, cross_b_lo).get_operation(),
        ctx,
        insert_block,
    );
    let mid_sum = emit_op(
        stair_mir::ops::AddOp::new(ctx, mid_sum, lo_lo_hi).get_operation(),
        ctx,
        insert_block,
    );
    let carry = emit_op(
        stair_mir::ops::ShrOp::new(ctx, mid_sum, sixty_four).get_operation(),
        ctx,
        insert_block,
    );

    // high = hi_hi + (cross_a >> 64) + (cross_b >> 64) + carry; the true
    // high half is < 2^128, and no intermediate sum wraps.
    let cross_a_hi = emit_op(
        stair_mir::ops::ShrOp::new(ctx, cross_a, sixty_four).get_operation(),
        ctx,
        insert_block,
    );
    let cross_b_hi = emit_op(
        stair_mir::ops::ShrOp::new(ctx, cross_b, sixty_four).get_operation(),
        ctx,
        insert_block,
    );
    let high = emit_op(
        stair_mir::ops::AddOp::new(ctx, hi_hi, cross_a_hi).get_operation(),
        ctx,
        insert_block,
    );
    let high = emit_op(
        stair_mir::ops::AddOp::new(ctx, high, cross_b_hi).get_operation(),
        ctx,
        insert_block,
    );
    let high = emit_op(
        stair_mir::ops::AddOp::new(ctx, high, carry).get_operation(),
        ctx,
        insert_block,
    );
    Ok(high)
}

/// `0 - (value >> 127)` on u128: all-ones when the top bit of `value` is
/// set, zero otherwise.
fn emit_sign_mask128(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    value: Value,
) -> Result<Value, String> {
    let u128_ty: TypeHandle = IntegerType::get(ctx, 128, Signedness::Unsigned).into();
    let shift = integer_constant(ctx, u128_ty, 127)?;
    shift.get_operation().insert_at_back(insert_block, ctx);
    let sign_bit = emit_op(
        stair_mir::ops::ShrOp::new(ctx, value, shift.get_result(ctx)).get_operation(),
        ctx,
        insert_block,
    );
    let zero = integer_constant(ctx, u128_ty, 0)?;
    zero.get_operation().insert_at_back(insert_block, ctx);
    Ok(emit_op(
        stair_mir::ops::SubOp::new(ctx, zero.get_result(ctx), sign_bit).get_operation(),
        ctx,
        insert_block,
    ))
}

fn lower_unsigned_mul_overflow(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    lhs: Value,
    rhs: Value,
) -> Result<Value, String> {
    let ty = lhs.get_type(ctx);
    let width = ty
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .map(|int_ty| int_ty.width())
        .ok_or_else(|| "MIR mul-with-overflow on non-integer type".to_string())?;
    if width == 128 {
        // No wider type exists: the product wraps in-register and overflow
        // is exactly "the true high 128 bits are non-zero".
        let wrapped = emit_op(
            stair_mir::ops::MulOp::new(ctx, lhs, rhs).get_operation(),
            ctx,
            insert_block,
        );
        let high = emit_umulh128(ctx, insert_block, lhs, rhs)?;
        let u128_ty: TypeHandle = IntegerType::get(ctx, 128, Signedness::Unsigned).into();
        let zero = integer_constant(ctx, u128_ty, 0)?;
        zero.get_operation().insert_at_back(insert_block, ctx);
        let overflow = emit_op(
            stair_mir::ops::NeOp::new(ctx, high, zero.get_result(ctx)).get_operation(),
            ctx,
            insert_block,
        );
        return Ok(pack_overflow_result(ctx, insert_block, wrapped, overflow));
    }
    if width > 64 {
        return Err(format!(
            "unsupported {width}-bit MIR mul-with-overflow"
        ));
    }

    let wide_ty: TypeHandle = IntegerType::get(ctx, width * 2, Signedness::Unsigned).into();
    let wide_lhs = cast_value_to_type(ctx, insert_block, lhs, wide_ty);
    let wide_rhs = cast_value_to_type(ctx, insert_block, rhs, wide_ty);
    let wide_mul = stair_mir::ops::MulOp::new(ctx, wide_lhs, wide_rhs);
    wide_mul.get_operation().insert_at_back(insert_block, ctx);
    let wide_value = wide_mul.get_operation().deref(ctx).get_result(0);
    let wrapped_value = cast_value_to_type(ctx, insert_block, wide_value, ty);

    let narrow_max = integer_constant(ctx, wide_ty, u128::MAX >> (128 - width as usize))?;
    narrow_max.get_operation().insert_at_back(insert_block, ctx);
    let overflow = stair_mir::ops::GtOp::new(ctx, wide_value, narrow_max.get_result(ctx));
    overflow.get_operation().insert_at_back(insert_block, ctx);
    let overflow_value = overflow.get_operation().deref(ctx).get_result(0);
    Ok(pack_overflow_result(
        ctx,
        insert_block,
        wrapped_value,
        overflow_value,
    ))
}

/// Lower MIR's three-way compare to `(lhs > rhs) as i8 - (lhs < rhs) as i8`.
/// That byte is exactly `core::cmp::Ordering`'s direct tag (-1/0/1), so it is
/// materialized as the enum blob the usual way: written into a stack
/// temporary and loaded back as the real in-memory bytes.
fn lower_three_way_cmp<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    ordering_ty: Ty<'tcx>,
    lhs: Value,
    rhs: Value,
) -> Result<Value, String> {
    // dialect-mir compares resolve signedness from the operand types.
    let gt = stair_mir::ops::GtOp::new(ctx, lhs, rhs);
    gt.get_operation().insert_at_back(insert_block, ctx);
    let lt = stair_mir::ops::LtOp::new(ctx, lhs, rhs);
    lt.get_operation().insert_at_back(insert_block, ctx);
    let byte_ty = bool_storage_ty(ctx);
    let gt = cast_value_to_type(ctx, insert_block, gt.get_result(ctx), byte_ty);
    let lt = cast_value_to_type(ctx, insert_block, lt.get_result(ctx), byte_ty);
    let tag = stair_mir::ops::SubOp::new(ctx, gt, lt).get_operation();
    tag.insert_at_back(insert_block, ctx);
    let tag = tag.deref(ctx).get_result(0);

    let blob_ty = convert_ty(tcx, ctx, ordering_ty)?;
    let slot = stair_mir::ops::AllocaOp::new(ctx, blob_ty);
    slot.get_operation().insert_at_back(insert_block, ctx);
    let slot = slot.get_result(ctx);
    let store = stair_mir::ops::StoreOp::new(ctx, tag, slot);
    store.get_operation().insert_at_back(insert_block, ctx);
    let load = stair_mir::ops::LoadOp::new(ctx, slot, blob_ty);
    load.get_operation().insert_at_back(insert_block, ctx);
    Ok(load.get_result(ctx))
}

fn is_unsigned_integer_value(ctx: &Context, value: Value) -> bool {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|ty| ty.signedness() != Signedness::Signed)
}

fn bool_immediate_ty(ctx: &mut Context) -> TypeHandle {
    IntegerType::get(ctx, 1, Signedness::Signless).into()
}

fn bool_storage_ty(ctx: &mut Context) -> TypeHandle {
    IntegerType::get(ctx, 8, Signedness::Signless).into()
}

fn is_bool_ty<'tcx>(ty: Ty<'tcx>) -> bool {
    matches!(runtime_ty(ty).kind(), rustc_middle::ty::TyKind::Bool)
}

fn cast_value_to_type(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    value: Value,
    target_ty: TypeHandle,
) -> Value {
    if value.get_type(ctx) == target_ty {
        return value;
    }
    let cast = stair_mir::ops::CastOp::new(ctx, value, target_ty);
    cast.get_operation().insert_at_back(insert_block, ctx);
    cast.get_result(ctx)
}

fn aggregate_field_type(
    ctx: &Context,
    aggregate_ty: TypeHandle,
    index: usize,
) -> Result<TypeHandle, String> {
    let ty_ref = aggregate_ty.deref(ctx);
    if let Some(array_ty) = ty_ref.downcast_ref::<llvm::types::ArrayType>() {
        return Ok(array_ty.elem_type());
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<llvm::types::StructType>() {
        if index >= struct_ty.num_fields() {
            return Err(format!(
                "aggregate field index {index} is out of bounds for {}",
                ty_ref.disp(ctx)
            ));
        }
        return Ok(struct_ty.field_type(index));
    }
    Err(format!("unsupported aggregate type: {}", ty_ref.disp(ctx)))
}

fn normalize_bool_for_storage<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    rust_ty: Ty<'tcx>,
    value: Value,
) -> Result<Value, String> {
    let rust_ty = mono_ty(tcx, state, rust_ty);
    if !is_bool_ty(rust_ty) {
        return Ok(value);
    }
    let storage_ty = bool_storage_ty(ctx);
    Ok(cast_value_to_type(ctx, insert_block, value, storage_ty))
}

fn normalize_bool_for_immediate<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    rust_ty: Ty<'tcx>,
    value: Value,
) -> Result<Value, String> {
    let rust_ty = mono_ty(tcx, state, rust_ty);
    if !is_bool_ty(rust_ty) {
        return Ok(value);
    }
    let immediate_ty = bool_immediate_ty(ctx);
    Ok(cast_value_to_type(ctx, insert_block, value, immediate_ty))
}

fn simple_abi_leaves_for_ty(
    ctx: &Context,
    ty: TypeHandle,
) -> Result<Vec<(Vec<u32>, TypeHandle)>, String> {
    let mut leaves = Vec::new();
    collect_simple_abi_fields(ctx, ty, Vec::new(), &mut leaves)?;
    Ok(leaves)
}

fn collect_simple_abi_fields(
    ctx: &Context,
    ty: TypeHandle,
    prefix: Vec<u32>,
    out: &mut Vec<(Vec<u32>, TypeHandle)>,
) -> Result<(), String> {
    if is_simple_abi_scalar_ty(ctx, ty) {
        out.push((prefix, ty));
        return Ok(());
    }
    if ty.deref(ctx).is::<UnitType>() {
        return Ok(());
    }

    let fields = {
        let ty_ref = ty.deref(ctx);
        let Some(struct_ty) = ty_ref.downcast_ref::<llvm::types::StructType>() else {
            return Err(format!(
                "unsupported aggregate argument type for external call ABI: {:?}",
                ty_ref
            ));
        };
        if struct_ty.is_opaque() {
            return Err("opaque struct argument cannot be ABI-lowered".to_string());
        }
        struct_ty.fields().collect::<Vec<_>>()
    };

    for (index, field_ty) in fields.into_iter().enumerate() {
        let mut field_prefix = prefix.clone();
        field_prefix.push(index as u32);
        collect_simple_abi_fields(ctx, field_ty, field_prefix, out)?;
    }
    Ok(())
}

fn lower_arguments_from_str_call<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    args: &[rustc_span::Spanned<Operand<'tcx>>],
    destination: &Place<'tcx>,
) -> Result<(), String> {
    if args.len() != 1 {
        return Err(format!(
            "unsupported Arguments::from_str argument count: {}",
            args.len()
        ));
    }

    let str_value = import_operand(tcx, ctx, state, insert_block, body, &args[0].node)?;
    let ptr_ty = llvm_ptr_ty(ctx);
    let fmt_ty = fmt_arguments_ty(ctx);

    let template = stair_mir::ops::ExtractValueOp::new(ctx, str_value, vec![0], ptr_ty);
    template.get_operation().insert_at_back(insert_block, ctx);

    let usize_ty: TypeHandle = usize_ty(ctx).into();
    let len = stair_mir::ops::ExtractValueOp::new(ctx, str_value, vec![1], usize_ty);
    len.get_operation().insert_at_back(insert_block, ctx);

    let one = stair_mir::ops::ConstantOp::new_integer(
        ctx,
        IntegerAttr::new(
            TypedHandle::from_handle(usize_ty, ctx).unwrap(),
            APInt::from_u64(1, NonZero::new(64).unwrap()),
        ),
    );
    one.get_operation().insert_at_back(insert_block, ctx);

    let shifted = stair_mir::ops::ShlOp::new(ctx, len.get_result(ctx), one.get_result(ctx));
    shifted.get_operation().insert_at_back(insert_block, ctx);
    let encoded = stair_mir::ops::BitOrOp::new(ctx, shifted.get_result(ctx), one.get_result(ctx));
    encoded.get_operation().insert_at_back(insert_block, ctx);

    let args_ptr = stair_mir::ops::CastOp::new(ctx, encoded.get_result(ctx), ptr_ty);
    args_ptr.get_operation().insert_at_back(insert_block, ctx);

    let undef = stair_mir::ops::UndefOp::new(ctx, fmt_ty);
    undef.get_operation().insert_at_back(insert_block, ctx);
    let with_template = stair_mir::ops::InsertValueOp::new(
        ctx,
        template.get_result(ctx),
        undef.get_result(ctx),
        vec![0],
    );
    with_template
        .get_operation()
        .insert_at_back(insert_block, ctx);
    let args = stair_mir::ops::InsertValueOp::new(
        ctx,
        args_ptr.get_result(ctx),
        with_template.get_result(ctx),
        vec![1],
    );
    args.get_operation().insert_at_back(insert_block, ctx);

    store_place(
        tcx,
        ctx,
        state,
        insert_block,
        body,
        destination,
        args.get_result(ctx),
    )
}

fn call_symbol<'tcx>(
    tcx: TyCtxt<'tcx>,
    state: &FunctionImportState<'tcx>,
    body: &Body<'tcx>,
    func: &Operand<'tcx>,
) -> Result<crate::identifier::Identifier, String> {
    let Some((def_id, args)) = call_fn_def(tcx, state, body, func) else {
        return Err(format!("unsupported MIR call callee: {func:?}"));
    };
    let mut legaliser = Legaliser::default();
    let args = mono_generic_args(tcx, state, args);
    if let Ok(Some(instance)) = Instance::try_resolve(tcx, body.typing_env(tcx), def_id, args) {
        if let Some(symbol) = known_codegen_symbol(tcx, instance) {
            return Ok(legaliser.legalise(symbol));
        }
        return Ok(legaliser.legalise(tcx.symbol_name(instance).name));
    }
    Ok(legaliser.legalise(&tcx.def_path_str(def_id)))
}

fn known_codegen_symbol<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> Option<&'static str> {
    let path = tcx.def_path_str(instance.def.def_id());
    if path.contains("compare_bytes") {
        return Some("memcmp");
    }
    None
}

fn import_rvalue<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    rvalue: &Rvalue<'tcx>,
) -> Result<Value, String> {
    match rvalue {
        Rvalue::Use(operand) => import_operand(tcx, ctx, state, insert_block, body, operand),
        Rvalue::Ref(_, _, place) | Rvalue::RawPtr(_, place) => {
            place_addr(tcx, ctx, state, insert_block, body, place)
        }
        Rvalue::ThreadLocalRef(def_id) => {
            // The address of the current thread's copy of a `#[thread_local]`
            // static: emitted like a static's address, but the global carries
            // the `ll.tls` marker so the backend goes through the thread
            // pointer instead of a data-section relocation.
            let mut legaliser = Legaliser::default();
            let symbol = legaliser.legalise(tcx.symbol_name(Instance::mono(tcx, *def_id)).name);
            declare_thread_local_global(tcx, ctx, state.module_body, symbol.clone(), *def_id)?;
            let ty = convert_immediate_ty(tcx, ctx, mono_ty(tcx, state, rvalue.ty(body, tcx)))?;
            let op = stair_mir::ops::AddressOfOp::new(ctx, symbol, ty);
            op.get_operation().insert_at_back(insert_block, ctx);
            Ok(op.get_result(ctx))
        }
        Rvalue::BinaryOp(op, operands) => {
            let lhs = import_operand(tcx, ctx, state, insert_block, body, &operands.0)?;
            let rhs = import_operand(tcx, ctx, state, insert_block, body, &operands.1)?;
            if matches!(op, BinOp::Offset) {
                let lhs_ty = mono_ty(tcx, state, operands.0.ty(body, tcx));
                let elem_size = layout_size_of_ty(tcx, pointee_ty(lhs_ty)?)?;
                let byte_offset = scale_index(ctx, insert_block, rhs, elem_size)?;
                let offset = stair_mir::ops::PtrOffsetOp::new(ctx, lhs, byte_offset);
                offset.get_operation().insert_at_back(insert_block, ctx);
                return Ok(offset.get_result(ctx));
            }
            if matches!(
                op,
                BinOp::AddWithOverflow | BinOp::SubWithOverflow | BinOp::MulWithOverflow
            ) {
                return lower_overflow_binary(ctx, insert_block, *op, lhs, rhs);
            }
            if matches!(op, BinOp::Cmp) {
                let ordering_ty = mono_ty(tcx, state, rvalue.ty(body, tcx));
                return lower_three_way_cmp(tcx, ctx, insert_block, ordering_ty, lhs, rhs);
            }
            let op = match op {
                BinOp::Add | BinOp::AddUnchecked => {
                    stair_mir::ops::AddOp::new(ctx, lhs, rhs).get_operation()
                }
                BinOp::Sub | BinOp::SubUnchecked => {
                    stair_mir::ops::SubOp::new(ctx, lhs, rhs).get_operation()
                }
                BinOp::Mul | BinOp::MulUnchecked => {
                    stair_mir::ops::MulOp::new(ctx, lhs, rhs).get_operation()
                }
                BinOp::Shr | BinOp::ShrUnchecked => {
                    stair_mir::ops::SignAwareShrOp::new(ctx, lhs, rhs).get_operation()
                }
                BinOp::Shl | BinOp::ShlUnchecked => {
                    stair_mir::ops::ShlOp::new(ctx, lhs, rhs).get_operation()
                }
                BinOp::Div => stair_mir::ops::DivOp::new(ctx, lhs, rhs).get_operation(),
                BinOp::Rem => stair_mir::ops::RemOp::new(ctx, lhs, rhs).get_operation(),
                BinOp::BitAnd => stair_mir::ops::BitAndOp::new(ctx, lhs, rhs).get_operation(),
                BinOp::BitOr => stair_mir::ops::BitOrOp::new(ctx, lhs, rhs).get_operation(),
                BinOp::BitXor => stair_mir::ops::BitXorOp::new(ctx, lhs, rhs).get_operation(),
                BinOp::Eq => stair_mir::ops::EqOp::new(ctx, lhs, rhs).get_operation(),
                BinOp::Ne => stair_mir::ops::NeOp::new(ctx, lhs, rhs).get_operation(),
                BinOp::Lt => stair_mir::ops::LtOp::new(ctx, lhs, rhs).get_operation(),
                BinOp::Le => stair_mir::ops::LeOp::new(ctx, lhs, rhs).get_operation(),
                BinOp::Gt => stair_mir::ops::GtOp::new(ctx, lhs, rhs).get_operation(),
                BinOp::Ge => stair_mir::ops::GeOp::new(ctx, lhs, rhs).get_operation(),
                other => return Err(format!("unsupported MIR binary op: {other:?}")),
            };
            op.insert_at_back(insert_block, ctx);
            Ok(op.deref(ctx).get_result(0))
        }
        Rvalue::Cast(kind, operand, ty) => {
            if matches!(
                kind,
                rustc_mir::CastKind::PointerCoercion(
                    rustc_middle::ty::adjustment::PointerCoercion::ReifyFnPointer(_),
                    _
                )
            ) {
                return reify_fn_pointer(tcx, ctx, state, insert_block, body, operand);
            }
            if matches!(kind, rustc_mir::CastKind::Transmute) {
                return import_transmute(tcx, ctx, state, insert_block, body, operand, *ty);
            }
            if let Some(value) =
                lower_pointer_unsize_cast(tcx, ctx, state, insert_block, body, kind, operand, *ty)?
            {
                return Ok(value);
            }
            let input = import_operand(tcx, ctx, state, insert_block, body, operand)?;
            let result_type = convert_immediate_ty(tcx, ctx, mono_ty(tcx, state, *ty))?;
            let cast = stair_mir::ops::CastOp::new(ctx, input, result_type);
            cast.get_operation().insert_at_back(insert_block, ctx);
            Ok(cast.get_result(ctx))
        }
        Rvalue::UnaryOp(rustc_mir::UnOp::Neg, operand) => {
            let input = import_operand(tcx, ctx, state, insert_block, body, operand)?;
            let neg = stair_mir::ops::NegOp::new(ctx, input);
            neg.get_operation().insert_at_back(insert_block, ctx);
            Ok(neg.get_result(ctx))
        }
        Rvalue::UnaryOp(rustc_mir::UnOp::Not, operand) => {
            let input = import_operand(tcx, ctx, state, insert_block, body, operand)?;
            let input_ty = input.get_type(ctx);
            let width = input_ty
                .deref(ctx)
                .downcast_ref::<IntegerType>()
                .map(|ty| ty.width())
                .ok_or_else(|| format!("unsupported non-integer MIR not: {rvalue:?}"))?;
            if width == 1 {
                let false_value = integer_constant(ctx, input_ty, 0)?;
                false_value
                    .get_operation()
                    .insert_at_back(insert_block, ctx);
                let eq = stair_mir::ops::EqOp::new(ctx, input, false_value.get_result(ctx));
                eq.get_operation().insert_at_back(insert_block, ctx);
                return Ok(eq.get_result(ctx));
            }
            let ones = integer_constant(ctx, input_ty, u128::MAX >> (128 - width as usize))?;
            ones.get_operation().insert_at_back(insert_block, ctx);
            let xor = stair_mir::ops::BitXorOp::new(ctx, input, ones.get_result(ctx));
            xor.get_operation().insert_at_back(insert_block, ctx);
            Ok(xor.get_result(ctx))
        }
        Rvalue::UnaryOp(rustc_mir::UnOp::PtrMetadata, operand) => {
            let value = import_operand(tcx, ctx, state, insert_block, body, operand)?;
            let value_ty = value.get_type(ctx);
            let result_ty = {
                let value_ty_ref = value_ty.deref(ctx);
                let Some(struct_ty) = value_ty_ref.downcast_ref::<llvm::types::StructType>() else {
                    return Err(format!(
                        "unsupported MIR pointer metadata operand type: {}",
                        value_ty_ref.disp(ctx)
                    ));
                };
                if struct_ty.num_fields() < 2 {
                    return Err(format!(
                        "unsupported MIR pointer metadata operand shape: {}",
                        value_ty_ref.disp(ctx)
                    ));
                }
                let result_ty = struct_ty.field_type(1);
                result_ty
            };
            let extract = stair_mir::ops::ExtractValueOp::new(ctx, value, vec![1], result_ty);
            extract.get_operation().insert_at_back(insert_block, ctx);
            Ok(extract.get_result(ctx))
        }
        Rvalue::Discriminant(place) => {
            let enum_ty = mono_ty(tcx, state, place.ty(body, tcx).ty);
            if !is_enum_ty(enum_ty) {
                // A non-enum place (e.g. a struct or single-variant ADT) has a
                // discriminant that is always zero.
                let discr_ty = convert_ty(tcx, ctx, enum_ty.discriminant_ty(tcx))?;
                let zero = integer_constant(ctx, discr_ty, 0)?;
                zero.get_operation().insert_at_back(insert_block, ctx);
                return Ok(zero.get_result(ctx));
            }
            let slot = place_addr(tcx, ctx, state, insert_block, body, place)?;
            read_enum_discriminant(tcx, ctx, insert_block, enum_ty, slot)
        }
        Rvalue::Aggregate(kind, operands) => {
            let operand_tys = operands
                .iter()
                .map(|operand| mono_ty(tcx, state, operand.ty(body, tcx)))
                .collect::<Vec<_>>();
            let aggregate_ty = aggregate_result_ty(tcx, ctx, state, kind, &operand_tys)?;
            if let rustc_mir::AggregateKind::RawPtr(pointee_ty, mutability) = **kind {
                if operands.len() != 2 {
                    return Err(format!(
                        "unsupported MIR raw pointer aggregate operand count: {}",
                        operands.len()
                    ));
                }
                let mut operands = operands.iter();
                let data_operand = operands.next().unwrap();
                let metadata_operand = operands.next().unwrap();
                let data = import_operand(tcx, ctx, state, insert_block, body, data_operand)?;
                let metadata =
                    import_operand(tcx, ctx, state, insert_block, body, metadata_operand)?;
                let pointee_ty = mono_ty(tcx, state, pointee_ty);
                if matches!(
                    runtime_ty(pointee_ty).kind(),
                    rustc_middle::ty::TyKind::Slice(_)
                ) {
                    let ptr_ty = llvm_ptr_ty(ctx);
                    let data = if data.get_type(ctx) == ptr_ty {
                        data
                    } else {
                        let cast = stair_mir::ops::CastOp::new(ctx, data, ptr_ty);
                        cast.get_operation().insert_at_back(insert_block, ctx);
                        cast.get_result(ctx)
                    };
                    let undef = stair_mir::ops::UndefOp::new(ctx, aggregate_ty);
                    undef.get_operation().insert_at_back(insert_block, ctx);
                    let with_data = stair_mir::ops::InsertValueOp::new(
                        ctx,
                        data,
                        undef.get_result(ctx),
                        vec![0],
                    );
                    with_data.get_operation().insert_at_back(insert_block, ctx);
                    let with_metadata = stair_mir::ops::InsertValueOp::new(
                        ctx,
                        metadata,
                        with_data.get_result(ctx),
                        vec![1],
                    );
                    with_metadata
                        .get_operation()
                        .insert_at_back(insert_block, ctx);
                    return Ok(with_metadata.get_result(ctx));
                }
                let _ = mutability;
                return Ok(data);
            }
            let undef = stair_mir::ops::UndefOp::new(ctx, aggregate_ty);
            undef.get_operation().insert_at_back(insert_block, ctx);
            let mut current = undef.get_result(ctx);

            if aggregate_is_union(tcx, kind) {
                if let Some(operand) = operands.iter().next()
                    && layout_size_of_ty(tcx, mono_ty(tcx, state, operand.ty(body, tcx)))? != 0
                {
                    return import_operand(tcx, ctx, state, insert_block, body, operand);
                }
                return Ok(current);
            }

            if let rustc_mir::AggregateKind::Adt(def_id, variant_idx, _, _, _) = **kind
                && is_fmt_rt_argument_type(tcx, def_id)
            {
                let variant = tcx.adt_def(def_id).variant(variant_idx).name.to_string();
                let ptr_ty = llvm_ptr_ty(ctx);
                let (value, formatter) = match variant.as_str() {
                    "Placeholder" => {
                        let mut operands = operands.iter();
                        let value_operand = operands.next().ok_or_else(|| {
                            "fmt Placeholder aggregate is missing its value".to_string()
                        })?;
                        let formatter_operand = operands.next().ok_or_else(|| {
                            "fmt Placeholder aggregate is missing its formatter".to_string()
                        })?;
                        let value =
                            import_operand(tcx, ctx, state, insert_block, body, value_operand)?;
                        let formatter =
                            import_operand(tcx, ctx, state, insert_block, body, formatter_operand)?;
                        (
                            cast_value_to_type(ctx, insert_block, value, ptr_ty),
                            cast_value_to_type(ctx, insert_block, formatter, ptr_ty),
                        )
                    }
                    "Count" => {
                        let count_operand = operands.iter().next().ok_or_else(|| {
                            "fmt Count aggregate is missing its count".to_string()
                        })?;
                        let count =
                            import_operand(tcx, ctx, state, insert_block, body, count_operand)?;
                        let usize_ty: TypeHandle = usize_ty(ctx).into();
                        let count = cast_value_to_type(ctx, insert_block, count, usize_ty);
                        let null = integer_constant(ctx, usize_ty, 0)?;
                        null.get_operation().insert_at_back(insert_block, ctx);
                        let null =
                            cast_value_to_type(ctx, insert_block, null.get_result(ctx), ptr_ty);
                        (cast_value_to_type(ctx, insert_block, count, ptr_ty), null)
                    }
                    other => {
                        return Err(format!("unsupported fmt ArgumentType variant: {other}"));
                    }
                };
                let undef = stair_mir::ops::UndefOp::new(ctx, aggregate_ty);
                undef.get_operation().insert_at_back(insert_block, ctx);
                let with_value =
                    stair_mir::ops::InsertValueOp::new(ctx, value, undef.get_result(ctx), vec![0]);
                with_value.get_operation().insert_at_back(insert_block, ctx);
                let with_formatter = stair_mir::ops::InsertValueOp::new(
                    ctx,
                    formatter,
                    with_value.get_result(ctx),
                    vec![1],
                );
                with_formatter
                    .get_operation()
                    .insert_at_back(insert_block, ctx);
                return Ok(with_formatter.get_result(ctx));
            }

            if let rustc_mir::AggregateKind::Adt(def_id, variant_idx, args, _, _) = **kind
                && tcx.adt_def(def_id).is_enum()
            {
                let enum_ty = mono_ty(tcx, state, Ty::new_adt(tcx, tcx.adt_def(def_id), args));
                let blob_ty = convert_ty(tcx, ctx, enum_ty)?;
                if is_unit_converted_ty(ctx, blob_ty) {
                    // Zero-sized enum (uninhabited or a single ZST variant).
                    return Ok(current);
                }

                // Materialize the enum in a stack temporary using its real
                // layout, then load it back: an enum value simply *is* its real
                // in-memory bytes. This keeps it interoperable with prebuilt std
                // code that reads and writes enums by reference.
                let slot = stair_mir::ops::AllocaOp::new(ctx, blob_ty);
                slot.get_operation().insert_at_back(insert_block, ctx);
                let slot = slot.get_result(ctx);
                write_enum_tag(tcx, ctx, insert_block, enum_ty, variant_idx, slot)?;

                let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
                let cx = rustc_middle::ty::layout::LayoutCx::new(tcx, typing_env);
                let layout = tcx
                    .layout_of(typing_env.as_query_input(enum_ty))
                    .map_err(|error| format!("no layout for {enum_ty:?}: {error:?}"))?;
                let variant_layout = layout.for_variant(&cx, variant_idx);
                for (idx, operand) in operands.iter().enumerate() {
                    let field_ty = mono_ty(tcx, state, operand.ty(body, tcx));
                    if layout_size_of_ty(tcx, field_ty)? == 0 {
                        continue;
                    }
                    let offset = variant_layout.fields.offset(idx).bytes();
                    let field_addr = if offset != 0 {
                        ptr_offset_const(ctx, insert_block, slot, offset)?
                    } else {
                        slot
                    };
                    let value = import_operand(tcx, ctx, state, insert_block, body, operand)?;
                    let value =
                        normalize_bool_for_storage(tcx, ctx, state, insert_block, field_ty, value)?;
                    let conv_field_ty = convert_ty(tcx, ctx, field_ty)?;
                    let value = cast_value_to_type(ctx, insert_block, value, conv_field_ty);
                    let store = stair_mir::ops::StoreOp::new(ctx, value, field_addr);
                    store.get_operation().insert_at_back(insert_block, ctx);
                }

                let load = stair_mir::ops::LoadOp::new(ctx, slot, blob_ty);
                load.get_operation().insert_at_back(insert_block, ctx);
                return Ok(load.get_result(ctx));
            }

            let rust_aggregate_ty = match **kind {
                rustc_mir::AggregateKind::Adt(def_id, _, adt_args, _, _) => Some(mono_ty(
                    tcx,
                    state,
                    Ty::new_adt(tcx, tcx.adt_def(def_id), adt_args),
                )),
                rustc_mir::AggregateKind::Closure(def_id, closure_args) => Some(mono_ty(
                    tcx,
                    state,
                    Ty::new_closure(tcx, def_id, closure_args),
                )),
                rustc_mir::AggregateKind::Tuple => Some(Ty::new_tup(tcx, &operand_tys)),
                _ => None,
            };
            for (idx, operand) in operands.iter().enumerate() {
                let index = match rust_aggregate_ty {
                    Some(aggregate) => converted_field_index(tcx, aggregate, idx)?,
                    None => idx as u32,
                };
                let value = import_operand(tcx, ctx, state, insert_block, body, operand)?;
                let field_ty = aggregate_field_type(ctx, current.get_type(ctx), index as usize)?;
                let value = cast_value_to_type(ctx, insert_block, value, field_ty);
                let insert = stair_mir::ops::InsertValueOp::new(ctx, value, current, vec![index]);
                insert.get_operation().insert_at_back(insert_block, ctx);
                current = insert.get_result(ctx);
            }

            Ok(current)
        }
        Rvalue::Repeat(operand, len) => {
            let value = import_operand(tcx, ctx, state, insert_block, body, operand)?;
            let elem_ty = mono_ty(tcx, state, operand.ty(body, tcx));
            let elem_ty = convert_ty(tcx, ctx, elem_ty)?;
            let len = array_len(tcx, mono_ty_const(tcx, state, *len))?;
            let aggregate_ty = llvm::types::ArrayType::get(ctx, elem_ty, len).into();
            let undef = stair_mir::ops::UndefOp::new(ctx, aggregate_ty);
            undef.get_operation().insert_at_back(insert_block, ctx);
            let mut current = undef.get_result(ctx);
            let value = cast_value_to_type(ctx, insert_block, value, elem_ty);
            for idx in 0..len {
                let insert =
                    stair_mir::ops::InsertValueOp::new(ctx, value, current, vec![idx as u32]);
                insert.get_operation().insert_at_back(insert_block, ctx);
                current = insert.get_result(ctx);
            }
            Ok(current)
        }
        other => Err(format!("unsupported MIR rvalue: {other:?}")),
    }
}

/// Lower `ReifyFnPointer`: materialize the address of the target function.
/// When the target's MIR is available it is imported into this module and
/// addressed directly; otherwise a local thunk that forwards to the external
/// symbol is synthesized, so no data relocations are required.
fn reify_fn_pointer<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    operand: &Operand<'tcx>,
) -> Result<Value, String> {
    let fn_ty = mono_ty(tcx, state, operand.ty(body, tcx));
    let rustc_middle::ty::TyKind::FnDef(def_id, args) = fn_ty.kind() else {
        return Err(format!("unsupported reified fn pointer source: {fn_ty:?}"));
    };
    let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
    let instance = Instance::resolve_for_fn_ptr(tcx, typing_env, *def_id, args)
        .ok_or_else(|| format!("cannot resolve reified fn pointer target: {fn_ty:?}"))?;
    let mut legaliser = Legaliser::default();
    let symbol = legaliser.legalise(tcx.symbol_name(instance).name);

    let target_symbol = if should_import_instance(tcx, instance) {
        import_upstream_instance(tcx, ctx, state.module_body, instance)
            .map_err(|error| format!("while importing reified fn {symbol}: {error}"))?;
        symbol
    } else {
        emit_fn_ptr_thunk(tcx, ctx, state.module_body, symbol, instance)?
    };

    let ptr_ty = llvm_ptr_ty(ctx);
    let op = stair_mir::ops::AddressOfOp::new(ctx, target_symbol, ptr_ty);
    op.get_operation().insert_at_back(insert_block, ctx);
    Ok(op.get_result(ctx))
}

/// Create a local function that forwards all arguments to an external symbol.
/// Taking its address only needs pc-relative addressing within the module.
fn emit_fn_ptr_thunk<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    module_body: Ptr<BasicBlock>,
    symbol: crate::identifier::Identifier,
    instance: Instance<'tcx>,
) -> Result<crate::identifier::Identifier, String> {
    let mut legaliser = Legaliser::default();
    let thunk_symbol = legaliser.legalise(&format!("{symbol}__stair_fnptr_thunk"));
    if symbol_exists(ctx, module_body, &thunk_symbol.to_string()) {
        return Ok(thunk_symbol);
    }

    let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
    let fn_sig = tcx
        .fn_sig(instance.def.def_id())
        .instantiate(tcx, instance.args);
    let fn_sig = tcx.normalize_erasing_regions(typing_env, fn_sig);
    let fn_sig = tcx.instantiate_bound_regions_with_erased(fn_sig);

    let mut inputs = Vec::new();
    for input_ty in fn_sig.inputs() {
        let abi_ty = convert_immediate_ty(tcx, ctx, *input_ty)?;
        match arg_abi_for_ty(ctx, abi_ty)? {
            ArgAbi::Leaves(leaves) => inputs.extend(leaves.iter().map(|(_, ty)| *ty)),
            ArgAbi::Indirect => inputs.push(llvm_ptr_ty(ctx)),
        }
    }
    let results = convert_return_ty(tcx, ctx, fn_sig.output())?;
    let result_ty = match results.as_slice() {
        [] => None,
        [ty] => Some(*ty),
        _ => return Err("unsupported multi-value reified fn result".to_string()),
    };

    declare_external_function(ctx, module_body, symbol.clone(), inputs.clone(), result_ty);

    let fn_ty = FunctionType::get(ctx, inputs, results);
    let func = stair_mir::ops::FuncOp::new(ctx, thunk_symbol.clone(), fn_ty);
    func.get_operation().insert_at_back(module_body, ctx);
    let entry = func.get_entry_block(ctx);
    let args: Vec<Value> = entry.deref(ctx).arguments().collect();
    let call = stair_mir::ops::CallOp::new_direct(ctx, symbol, args, result_ty);
    call.get_operation().insert_at_back(entry, ctx);
    let ret_val = result_ty.map(|_| call.get_operation().deref(ctx).get_result(0));
    let ret = stair_mir::ops::ReturnOp::new(ctx, ret_val);
    ret.get_operation().insert_at_back(entry, ctx);
    set_internal_linkage(ctx, module_body, &thunk_symbol);
    Ok(thunk_symbol)
}

/// Lower `CastKind::Transmute`. Transmute semantics are defined on the real
/// rustc layout, which matches this importer's representation for scalars
/// and (memory-ordered) structs but not for enums.
fn import_transmute<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    operand: &Operand<'tcx>,
    target: Ty<'tcx>,
) -> Result<Value, String> {
    let src_ty = mono_ty(tcx, state, operand.ty(body, tcx));
    let dst_ty = mono_ty(tcx, state, target);
    let value = import_operand(tcx, ctx, state, insert_block, body, operand)?;

    if !is_enum_ty(src_ty) && !is_enum_ty(dst_ty) {
        let result_ty = convert_immediate_ty(tcx, ctx, dst_ty)?;
        if value.get_type(ctx) == result_ty {
            return Ok(value);
        }
        // A direct cast is only meaningful between scalar (integer, pointer,
        // float) representations. Aggregate reinterpretations (e.g.
        // `[T; N] -> [MaybeUninit<T>; N]`) must go through memory below: a
        // struct-to-struct "bitcast" does not remap the value's pieces.
        let is_scalar = |ctx: &Context, ty: TypeHandle| {
            let ty_ref = ty.deref(ctx);
            ty_ref.is::<IntegerType>()
                || ty_ref.is::<llvm::types::PointerType>()
                || ty_ref.is::<FP32Type>()
                || ty_ref.is::<FP64Type>()
        };
        if is_scalar(ctx, value.get_type(ctx)) && is_scalar(ctx, result_ty) {
            let cast = stair_mir::ops::CastOp::new(ctx, value, result_ty);
            cast.get_operation().insert_at_back(insert_block, ctx);
            return Ok(cast.get_result(ctx));
        }
    }

    // Any other transmute is a pure reinterpretation of bytes. Since every
    // converted type shares rustc's real memory layout, spill the source
    // value and reload it as the destination type.
    let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
    let src_size = rustc_layout_size_of_ty(tcx, typing_env, src_ty)?;
    let dst_size = rustc_layout_size_of_ty(tcx, typing_env, dst_ty)?;
    let word_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let words = src_size.max(dst_size).div_ceil(8).max(1);
    let blob_ty: TypeHandle = llvm::types::ArrayType::get(ctx, word_ty, words).into();
    let slot = stair_mir::ops::AllocaOp::new(ctx, blob_ty);
    slot.get_operation().insert_at_back(insert_block, ctx);
    let slot = slot.get_result(ctx);
    let store = stair_mir::ops::StoreOp::new(ctx, value, slot);
    store.get_operation().insert_at_back(insert_block, ctx);
    load_value_from_real_layout(tcx, ctx, insert_block, dst_ty, slot)
}

fn lower_pointer_unsize_cast<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    kind: &rustc_mir::CastKind,
    operand: &Operand<'tcx>,
    target_ty: Ty<'tcx>,
) -> Result<Option<Value>, String> {
    if !matches!(
        kind,
        rustc_mir::CastKind::PointerCoercion(
            rustc_middle::ty::adjustment::PointerCoercion::Unsize,
            _
        )
    ) {
        return Ok(None);
    }

    let source_ty = mono_ty(tcx, state, operand.ty(body, tcx));
    let target_ty = mono_ty(tcx, state, target_ty);
    let Some(metadata) = unsize_metadata(tcx, source_ty, target_ty)? else {
        return Ok(None);
    };

    let input = import_operand(tcx, ctx, state, insert_block, body, operand)?;
    let ptr_ty = llvm_ptr_ty(ctx);

    // The metadata half of the resulting fat pointer.
    let metadata_value: Value = match metadata {
        UnsizeMetadata::SliceLen(len) => {
            let usize_ty: TypeHandle = usize_ty(ctx).into();
            let len_op = integer_constant(ctx, usize_ty, len as u128)?;
            len_op.get_operation().insert_at_back(insert_block, ctx);
            len_op.get_result(ctx)
        }
        UnsizeMetadata::Vtable {
            concrete,
            principal,
        } => {
            // The same vtable rustc's own codegen would reference: emit its
            // allocation as an internal data global and take its address.
            let vtable_alloc_id = tcx.vtable_allocation((concrete, principal));
            let rustc_mir::interpret::GlobalAlloc::Memory(alloc) =
                tcx.global_alloc(vtable_alloc_id)
            else {
                return Err(format!(
                    "vtable allocation for {concrete:?} is not a memory allocation"
                ));
            };
            let mut legaliser = Legaliser::default();
            let symbol = legaliser.legalise(&format!("__crabbit_{vtable_alloc_id:?}"));
            emit_allocation_global(
                tcx,
                ctx,
                state.module_body,
                symbol.clone(),
                alloc.inner(),
                LinkageAttr::InternalLinkage,
                false,
            )?;
            let op = stair_mir::ops::AddressOfOp::new(ctx, symbol, ptr_ty);
            op.get_operation().insert_at_back(insert_block, ctx);
            op.get_result(ctx)
        }
        UnsizeMetadata::ReuseSource => {
            // dyn -> dyn with the same principal: a NOP coercion — the fat
            // pointer (data and vtable) is unchanged. Only the converted
            // type may differ nominally, so reinterpret through a spill
            // when it does.
            let result_ty = convert_ty(tcx, ctx, target_ty)?;
            if input.get_type(ctx) == result_ty {
                return Ok(Some(input));
            }
            let input_ty = input.get_type(ctx);
            let spill = stair_mir::ops::AllocaOp::new(ctx, input_ty);
            spill.get_operation().insert_at_back(insert_block, ctx);
            let spill = spill.get_result(ctx);
            let store = stair_mir::ops::StoreOp::new(ctx, input, spill);
            store.get_operation().insert_at_back(insert_block, ctx);
            let load = stair_mir::ops::LoadOp::new(ctx, spill, result_ty);
            load.get_operation().insert_at_back(insert_block, ctx);
            return Ok(Some(load.get_result(ctx)));
        }
    };

    if runtime_ty(source_ty).is_box() {
        // `Box<[T; N]> -> Box<[T]>`: the source value is a thin pointer at
        // byte offset 0 of its (nested) struct layout, and the target's real
        // layout is `{ ptr, len }`. The converted struct types nest the
        // pointer several levels deep (Unique -> NonNull -> ptr), so
        // assemble the fat value through memory at real byte offsets rather
        // than by insertvalue paths.
        let input_ty = input.get_type(ctx);
        let spill = stair_mir::ops::AllocaOp::new(ctx, input_ty);
        spill.get_operation().insert_at_back(insert_block, ctx);
        let spill = spill.get_result(ctx);
        let store = stair_mir::ops::StoreOp::new(ctx, input, spill);
        store.get_operation().insert_at_back(insert_block, ctx);
        let load_ptr = stair_mir::ops::LoadOp::new(ctx, spill, ptr_ty);
        load_ptr.get_operation().insert_at_back(insert_block, ctx);
        let data_ptr = load_ptr.get_result(ctx);

        let result_ty = convert_ty(tcx, ctx, target_ty)?;
        let out = stair_mir::ops::AllocaOp::new(ctx, result_ty);
        out.get_operation().insert_at_back(insert_block, ctx);
        let out = out.get_result(ctx);
        let store_ptr = stair_mir::ops::StoreOp::new(ctx, data_ptr, out);
        store_ptr.get_operation().insert_at_back(insert_block, ctx);
        let meta_addr = ptr_offset_const(ctx, insert_block, out, 8)?;
        let store_meta = stair_mir::ops::StoreOp::new(ctx, metadata_value, meta_addr);
        store_meta.get_operation().insert_at_back(insert_block, ctx);
        let load = stair_mir::ops::LoadOp::new(ctx, out, result_ty);
        load.get_operation().insert_at_back(insert_block, ctx);
        return Ok(Some(load.get_result(ctx)));
    }

    let data_ptr = if input.get_type(ctx) == ptr_ty {
        input
    } else {
        let cast = stair_mir::ops::CastOp::new(ctx, input, ptr_ty);
        cast.get_operation().insert_at_back(insert_block, ctx);
        cast.get_result(ctx)
    };

    let result_ty = convert_ty(tcx, ctx, target_ty)?;
    let undef = stair_mir::ops::UndefOp::new(ctx, result_ty);
    undef.get_operation().insert_at_back(insert_block, ctx);
    let with_ptr =
        stair_mir::ops::InsertValueOp::new(ctx, data_ptr, undef.get_result(ctx), vec![0]);
    with_ptr.get_operation().insert_at_back(insert_block, ctx);
    let with_meta = stair_mir::ops::InsertValueOp::new(
        ctx,
        metadata_value,
        with_ptr.get_result(ctx),
        vec![1],
    );
    with_meta.get_operation().insert_at_back(insert_block, ctx);
    Ok(Some(with_meta.get_result(ctx)))
}

/// What the fat pointer's metadata half is for an unsize coercion; the same
/// classification as rustc_codegen_ssa's `unsized_info`.
enum UnsizeMetadata<'tcx> {
    /// `[T; N] -> [T]` (possibly through a struct tail): the slice length.
    SliceLen(u64),
    /// `T -> dyn Trait`: the vtable of `concrete` for `principal`.
    Vtable {
        concrete: Ty<'tcx>,
        principal: Option<rustc_middle::ty::ExistentialTraitRef<'tcx>>,
    },
    /// `dyn Trait -> dyn Trait` with the same principal (adding/removing
    /// auto traits): the source's metadata is reused unchanged.
    ReuseSource,
}

fn unsized_slice_len<'tcx>(
    tcx: TyCtxt<'tcx>,
    source_ty: Ty<'tcx>,
    target_ty: Ty<'tcx>,
) -> Result<Option<u64>, String> {
    match unsize_metadata(tcx, source_ty, target_ty)? {
        Some(UnsizeMetadata::SliceLen(len)) => Ok(Some(len)),
        _ => Ok(None),
    }
}

fn unsize_metadata<'tcx>(
    tcx: TyCtxt<'tcx>,
    source_ty: Ty<'tcx>,
    target_ty: Ty<'tcx>,
) -> Result<Option<UnsizeMetadata<'tcx>>, String> {
    let source_ty = runtime_ty(source_ty);
    let target_ty = runtime_ty(target_ty);
    let (source_inner, target_inner) = match (source_ty.kind(), target_ty.kind()) {
        (
            rustc_middle::ty::TyKind::Ref(_, source_inner, _),
            rustc_middle::ty::TyKind::Ref(_, target_inner, _),
        )
        | (
            rustc_middle::ty::TyKind::RawPtr(source_inner, _),
            rustc_middle::ty::TyKind::RawPtr(target_inner, _),
        ) => (*source_inner, *target_inner),
        _ if source_ty.is_box() && target_ty.is_box() => {
            (source_ty.expect_boxed_ty(), target_ty.expect_boxed_ty())
        }
        _ => return Ok(None),
    };

    // Unsizing may act through a struct's (recursive) tail field, e.g.
    // `&PolymorphicIter<[T; N]> -> &PolymorphicIter<[T]>` in core's array
    // iterator: the fat pointer's metadata is still the slice length. The
    // *lockstep* tails stop peeling as soon as the sides diverge, so a
    // sized `Sq(u64) -> dyn Shape` coercion keeps `Sq` as the concrete
    // type instead of drilling to its last field (rustc_codegen_ssa's
    // `unsized_info` does the same).
    let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
    let (source_tail, target_tail) = tcx.struct_lockstep_tails_for_codegen(
        runtime_ty(source_inner),
        runtime_ty(target_inner),
        typing_env,
    );
    match (
        runtime_ty(source_tail).kind(),
        runtime_ty(target_tail).kind(),
    ) {
        (rustc_middle::ty::TyKind::Array(_, len), rustc_middle::ty::TyKind::Slice(_)) => {
            Ok(Some(UnsizeMetadata::SliceLen(array_len(tcx, *len)?)))
        }
        (
            rustc_middle::ty::TyKind::Dynamic(data_a, _),
            rustc_middle::ty::TyKind::Dynamic(data_b, _),
        ) => {
            let b_principal = data_b.principal_def_id();
            if data_a.principal_def_id() == b_principal || b_principal.is_none() {
                Ok(Some(UnsizeMetadata::ReuseSource))
            } else {
                Err(format!(
                    "unsupported trait upcasting coercion: {source_tail:?} -> {target_tail:?}"
                ))
            }
        }
        (_, rustc_middle::ty::TyKind::Dynamic(data, _)) => Ok(Some(UnsizeMetadata::Vtable {
            concrete: runtime_ty(source_tail),
            principal: data
                .principal()
                .map(|principal| tcx.instantiate_bound_regions_with_erased(principal)),
        })),
        _ => Ok(None),
    }
}

fn aggregate_result_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    kind: &rustc_mir::AggregateKind<'tcx>,
    operand_tys: &[Ty<'tcx>],
) -> Result<TypeHandle, String> {
    match kind {
        rustc_mir::AggregateKind::Array(elem_ty) => {
            let elem_ty = convert_ty(tcx, ctx, mono_ty(tcx, state, *elem_ty))?;
            Ok(llvm::types::ArrayType::get(ctx, elem_ty, operand_tys.len() as u64).into())
        }
        rustc_mir::AggregateKind::Adt(def_id, variant_idx, args, _, None) => {
            let ty = Ty::new_adt(tcx, tcx.adt_def(*def_id), args);
            let rustc_middle::ty::TyKind::Adt(adt_def, _) = ty.kind() else {
                unreachable!("Ty::new_adt must produce an ADT type");
            };
            if !adt_def.is_struct() && !adt_def.is_union() && !adt_def.is_enum() {
                return Err(format!("unsupported aggregate ADT kind: {kind:?}"));
            }
            let _ = variant_idx;
            convert_ty(tcx, ctx, mono_ty(tcx, state, ty))
        }
        rustc_mir::AggregateKind::Adt(def_id, variant_idx, args, _, Some(_)) => {
            let _ = variant_idx;
            convert_ty(
                tcx,
                ctx,
                mono_ty(tcx, state, Ty::new_adt(tcx, tcx.adt_def(*def_id), args)),
            )
        }
        rustc_mir::AggregateKind::Tuple => {
            let fields = operand_tys
                .iter()
                .map(|operand_ty| convert_ty(tcx, ctx, mono_ty(tcx, state, *operand_ty)))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(llvm::types::StructType::get_unnamed(ctx, fields).into())
        }
        rustc_mir::AggregateKind::RawPtr(pointee_ty, mutability) => {
            let pointee_ty = mono_ty(tcx, state, *pointee_ty);
            let ptr_ty = Ty::new_ptr(tcx, pointee_ty, *mutability);
            convert_ty(tcx, ctx, ptr_ty)
        }
        rustc_mir::AggregateKind::Closure(def_id, args) => {
            let ty = Ty::new_closure(tcx, *def_id, args);
            convert_ty(tcx, ctx, mono_ty(tcx, state, ty))
        }
        _ => Err(format!("unsupported MIR aggregate kind: {kind:?}")),
    }
}

fn import_operand<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    operand: &Operand<'tcx>,
) -> Result<Value, String> {
    match operand {
        Operand::Copy(place) | Operand::Move(place) => {
            load_place(tcx, ctx, state, insert_block, body, *place)
        }
        Operand::Constant(constant) => {
            import_constant(tcx, ctx, state, insert_block, body, constant)
        }
        Operand::RuntimeChecks(check) => {
            let ty = bool_immediate_ty(ctx);
            let value = if check.value(tcx.sess) { 1 } else { 0 };
            let op = integer_constant(ctx, ty, value)?;
            op.get_operation().insert_at_back(insert_block, ctx);
            Ok(op.get_result(ctx))
        }
    }
}

fn import_constant<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    constant: &ConstOperand<'tcx>,
) -> Result<Value, String> {
    let const_ = mono_const(tcx, state, constant.const_);
    let typing_env = state.instance.map_or_else(
        || body.typing_env(tcx),
        |_| rustc_middle::ty::TypingEnv::fully_monomorphized(),
    );
    if is_str_ref_ty(const_.ty()) {
        return import_str_constant(
            tcx,
            ctx,
            state.module_body,
            insert_block,
            typing_env,
            body,
            constant,
            const_,
        );
    }
    if let Some(len) = byte_array_ref_len(tcx, const_.ty())
        && let Some(bytes) = literal_byte_string_constant(tcx, constant)
            .or_else(|| evaluated_byte_string_constant(tcx, typing_env, constant.span, const_, len))
    {
        let ty = convert_ty(tcx, ctx, const_.ty())?;
        let symbol = declare_anonymous_byte_global(ctx, state.module_body, &bytes);
        let op = stair_mir::ops::AddressOfOp::new(ctx, symbol, ty);
        op.get_operation().insert_at_back(insert_block, ctx);
        return Ok(op.get_result(ctx));
    }

    let ty = convert_immediate_ty(tcx, ctx, const_.ty())?;
    if let Some(def_id) = constant.check_static_ptr(tcx) {
        let mut legaliser = Legaliser::default();
        let symbol = legaliser.legalise(tcx.symbol_name(Instance::mono(tcx, def_id)).name);
        declare_static_global(tcx, ctx, state.module_body, symbol.clone(), def_id)?;
        let op = stair_mir::ops::AddressOfOp::new(ctx, symbol, ty);
        op.get_operation().insert_at_back(insert_block, ctx);
        return Ok(op.get_result(ctx));
    }
    if layout_size_of_ty(tcx, const_.ty())? == 0 {
        let op = stair_mir::ops::UndefOp::new(ctx, ty);
        op.get_operation().insert_at_back(insert_block, ctx);
        return Ok(op.get_result(ctx));
    }
    if let Some(value) = import_initialized_maybe_uninit_u8_constant(
        tcx,
        ctx,
        insert_block,
        typing_env,
        constant.span,
        const_,
        ty,
    )? {
        return Ok(value);
    }
    if contains_maybe_uninit_ty(tcx, const_.ty()) {
        let op = stair_mir::ops::UndefOp::new(ctx, ty);
        op.get_operation().insert_at_back(insert_block, ctx);
        return Ok(op.get_result(ctx));
    }
    if is_enum_ty(const_.ty()) {
        return import_enum_constant(
            tcx,
            ctx,
            state,
            insert_block,
            typing_env,
            constant.span,
            const_,
            ty,
        );
    }
    let Some(bits) = const_.try_eval_bits(tcx, typing_env) else {
        if let Some(value) = import_memory_constant(
            tcx,
            ctx,
            state,
            insert_block,
            typing_env,
            constant.span,
            const_,
            ty,
        )? {
            return Ok(value);
        }
        if let Some(value) = import_pointer_to_plain_data_constant(
            tcx,
            ctx,
            state,
            insert_block,
            typing_env,
            constant.span,
            const_,
            ty,
        )? {
            return Ok(value);
        }
        if let Some(value) = import_pointer_constant(
            tcx,
            ctx,
            state,
            insert_block,
            typing_env,
            constant.span,
            const_,
            ty,
        )? {
            return Ok(value);
        }
        return Err(unsupported_constant_reason(
            tcx,
            typing_env,
            constant.span,
            const_,
        ));
    };
    match constant_from_bits(ctx, ty, bits) {
        Ok(op) => {
            op.get_operation().insert_at_back(insert_block, ctx);
            Ok(op.get_result(ctx))
        }
        Err(_) => {
            // Newtype wrappers around a scalar (Cap, NonZero inner types,
            // ...) evaluate to scalar bits but convert to a struct type.
            if let Some(value) = scalar_constant_in_aggregate(ctx, insert_block, ty, bits)? {
                return Ok(value);
            }
            let op = stair_mir::ops::UndefOp::new(ctx, ty);
            op.get_operation().insert_at_back(insert_block, ctx);
            Ok(op.get_result(ctx))
        }
    }
}

/// Import a constant whose evaluated value lives in a memory allocation
/// without pointer provenance, by emitting an anonymous byte global and
/// loading the converted value type from it.
#[allow(clippy::too_many_arguments)]
fn import_memory_constant<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    typing_env: rustc_middle::ty::TypingEnv<'tcx>,
    span: rustc_span::Span,
    constant: rustc_mir::Const<'tcx>,
    ty: TypeHandle,
) -> Result<Option<Value>, String> {
    let Ok(value) = constant.eval(tcx, typing_env, span) else {
        return Ok(None);
    };
    let rustc_mir::ConstValue::Indirect { alloc_id, offset } = value else {
        return Ok(None);
    };
    let rustc_mir::interpret::GlobalAlloc::Memory(alloc) = tcx.global_alloc(alloc_id) else {
        return Ok(None);
    };
    let alloc = alloc.inner();
    if !alloc.provenance().ptrs().is_empty() {
        return Ok(None);
    }
    // Uninitialized ranges (e.g. enum padding) read as whatever the raw
    // buffer holds, matching what rustc's own codegen emits for globals.
    let bytes = alloc
        .inspect_with_uninit_and_ptr_outside_interpreter(offset.bytes() as usize..alloc.len())
        .to_vec();
    let symbol = declare_anonymous_byte_global(ctx, state.module_body, &bytes);
    let ptr_ty = llvm_ptr_ty(ctx);
    let addr = stair_mir::ops::AddressOfOp::new(ctx, symbol, ptr_ty);
    addr.get_operation().insert_at_back(insert_block, ctx);
    let load = stair_mir::ops::LoadOp::new(ctx, addr.get_result(ctx), ty);
    load.get_operation().insert_at_back(insert_block, ctx);
    Ok(Some(load.get_result(ctx)))
}

/// Explain why a constant could not be imported. Pointer-carrying constants
/// are lowered through `ll.data` globals with data relocations, so a constant
/// that still fails here could not be evaluated or has an unhandled value
/// shape; report the MIR dump.
fn unsupported_constant_reason<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: rustc_middle::ty::TypingEnv<'tcx>,
    span: rustc_span::Span,
    constant: rustc_mir::Const<'tcx>,
) -> String {
    match constant.eval(tcx, typing_env, span) {
        Ok(value) => format!("unsupported MIR constant value {value:?}: {constant:?}"),
        Err(_) => format!("unsupported MIR constant (evaluation failed): {constant:?}"),
    }
}

/// Import an evaluated constant that is a thin pointer/reference into a
/// read-only allocation without pointer provenance (e.g. the promoted
/// `&ControlFlow<(), ()>` constants inside `Iterator::all`/`any`, or promoted
/// `&[T; 0]` empty-array refs), or a fat `&[T]` slice ref whose data carries
/// no provenance. The pointee bytes become an anonymous byte global and the
/// constant becomes its address (plus the length for slice refs).
#[allow(clippy::too_many_arguments)]
fn import_pointer_to_plain_data_constant<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    typing_env: rustc_middle::ty::TypingEnv<'tcx>,
    span: rustc_span::Span,
    constant: rustc_mir::Const<'tcx>,
    ty: TypeHandle,
) -> Result<Option<Value>, String> {
    let pointee = match runtime_ty(constant.ty()).kind() {
        rustc_middle::ty::TyKind::Ref(_, inner, _)
        | rustc_middle::ty::TyKind::RawPtr(inner, _) => runtime_ty(*inner),
        _ => return Ok(None),
    };
    let Ok(value) = constant.eval(tcx, typing_env, span) else {
        return Ok(None);
    };
    match value {
        rustc_mir::ConstValue::Scalar(rustc_mir::interpret::Scalar::Ptr(ptr, _)) => {
            let (provenance, offset) = ptr.prov_and_relative_offset();
            let rustc_mir::interpret::GlobalAlloc::Memory(alloc) =
                tcx.global_alloc(provenance.alloc_id())
            else {
                return Ok(None);
            };
            let alloc = alloc.inner();
            if !alloc.provenance().ptrs().is_empty() {
                return Ok(None);
            }
            let bytes = alloc
                .inspect_with_uninit_and_ptr_outside_interpreter(
                    offset.bytes() as usize..alloc.len(),
                )
                .to_vec();
            let symbol = declare_anonymous_byte_global(ctx, state.module_body, &bytes);
            let op = stair_mir::ops::AddressOfOp::new(ctx, symbol, ty);
            op.get_operation().insert_at_back(insert_block, ctx);
            Ok(Some(op.get_result(ctx)))
        }
        rustc_mir::ConstValue::Slice { alloc_id, meta } => {
            let rustc_middle::ty::TyKind::Slice(elem) = pointee.kind() else {
                return Ok(None);
            };
            let elem_size = layout_size_of_ty(tcx, *elem)?;
            let Some(bytes) =
                allocation_bytes(tcx, alloc_id, Size::ZERO, meta.saturating_mul(elem_size))
            else {
                return Ok(None);
            };
            let symbol = declare_anonymous_byte_global(ctx, state.module_body, &bytes);
            let ptr_ty = llvm_ptr_ty(ctx);
            let data = stair_mir::ops::AddressOfOp::new(ctx, symbol, ptr_ty);
            data.get_operation().insert_at_back(insert_block, ctx);
            let usize_ty: TypeHandle = usize_ty(ctx).into();
            let len = integer_constant(ctx, usize_ty, meta as u128)?;
            len.get_operation().insert_at_back(insert_block, ctx);
            let undef = stair_mir::ops::UndefOp::new(ctx, ty);
            undef.get_operation().insert_at_back(insert_block, ctx);
            let with_ptr = stair_mir::ops::InsertValueOp::new(
                ctx,
                data.get_result(ctx),
                undef.get_result(ctx),
                vec![0],
            );
            with_ptr.get_operation().insert_at_back(insert_block, ctx);
            let with_len = stair_mir::ops::InsertValueOp::new(
                ctx,
                len.get_result(ctx),
                with_ptr.get_result(ctx),
                vec![1],
            );
            with_len.get_operation().insert_at_back(insert_block, ctx);
            Ok(Some(with_len.get_result(ctx)))
        }
        _ => Ok(None),
    }
}

/// Import an evaluated constant whose value carries pointer provenance, by
/// materializing the backing allocations as [ll.data](pliron_ll::ll::DataAttr)
/// globals (with data relocations for the pointer slots) and taking their
/// addresses:
///
/// - a `Scalar::Ptr` becomes `llvm.addressof` of the target's global (plus a
///   byte offset for interior pointers),
/// - a `Slice` becomes the `{ptr, len}` fat-pointer pair over the data global,
/// - an `Indirect` (by-ref) value is loaded back out of its relocated global,
///   so pointer slots inside it hold real addresses at runtime.
#[allow(clippy::too_many_arguments)]
fn import_pointer_constant<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    typing_env: rustc_middle::ty::TypingEnv<'tcx>,
    span: rustc_span::Span,
    constant: rustc_mir::Const<'tcx>,
    ty: TypeHandle,
) -> Result<Option<Value>, String> {
    let Ok(value) = constant.eval(tcx, typing_env, span) else {
        return Ok(None);
    };
    match value {
        rustc_mir::ConstValue::Scalar(rustc_mir::interpret::Scalar::Ptr(ptr, _)) => {
            let (provenance, offset) = ptr.prov_and_relative_offset();
            // A type-id "pointer" is not an address: its offset bytes are a
            // hash segment, materialized as a plain integer.
            if matches!(
                tcx.global_alloc(provenance.alloc_id()),
                rustc_mir::interpret::GlobalAlloc::TypeId { .. }
            ) {
                let usize_ty: TypeHandle = usize_ty(ctx).into();
                let op = integer_constant(ctx, usize_ty, offset.bytes() as u128)?;
                op.get_operation().insert_at_back(insert_block, ctx);
                return Ok(Some(cast_value_to_type(
                    ctx,
                    insert_block,
                    op.get_result(ctx),
                    ty,
                )));
            }
            let symbol =
                data_global_for_alloc(tcx, ctx, state.module_body, provenance.alloc_id())?;
            let ptr_ty = llvm_ptr_ty(ctx);
            let addr = stair_mir::ops::AddressOfOp::new(ctx, symbol, ptr_ty);
            addr.get_operation().insert_at_back(insert_block, ctx);
            let mut value = addr.get_result(ctx);
            if offset.bytes() != 0 {
                value = ptr_offset_const(ctx, insert_block, value, offset.bytes())?;
            }
            Ok(Some(cast_value_to_type(ctx, insert_block, value, ty)))
        }
        rustc_mir::ConstValue::Slice { alloc_id, meta } => {
            let symbol = data_global_for_alloc(tcx, ctx, state.module_body, alloc_id)?;
            let ptr_ty = llvm_ptr_ty(ctx);
            let data = stair_mir::ops::AddressOfOp::new(ctx, symbol, ptr_ty);
            data.get_operation().insert_at_back(insert_block, ctx);
            let usize_ty: TypeHandle = usize_ty(ctx).into();
            let len = integer_constant(ctx, usize_ty, meta as u128)?;
            len.get_operation().insert_at_back(insert_block, ctx);
            let undef = stair_mir::ops::UndefOp::new(ctx, ty);
            undef.get_operation().insert_at_back(insert_block, ctx);
            let with_ptr = stair_mir::ops::InsertValueOp::new(
                ctx,
                data.get_result(ctx),
                undef.get_result(ctx),
                vec![0],
            );
            with_ptr.get_operation().insert_at_back(insert_block, ctx);
            let with_len = stair_mir::ops::InsertValueOp::new(
                ctx,
                len.get_result(ctx),
                with_ptr.get_result(ctx),
                vec![1],
            );
            with_len.get_operation().insert_at_back(insert_block, ctx);
            Ok(Some(with_len.get_result(ctx)))
        }
        rustc_mir::ConstValue::Indirect { alloc_id, offset } => {
            let symbol = data_global_for_alloc(tcx, ctx, state.module_body, alloc_id)?;
            let ptr_ty = llvm_ptr_ty(ctx);
            let addr = stair_mir::ops::AddressOfOp::new(ctx, symbol, ptr_ty);
            addr.get_operation().insert_at_back(insert_block, ctx);
            let mut addr_value = addr.get_result(ctx);
            if offset.bytes() != 0 {
                addr_value = ptr_offset_const(ctx, insert_block, addr_value, offset.bytes())?;
            }
            let load = stair_mir::ops::LoadOp::new(ctx, addr_value, ty);
            load.get_operation().insert_at_back(insert_block, ctx);
            Ok(Some(load.get_result(ctx)))
        }
        _ => Ok(None),
    }
}

/// Load a value stored with the real rustc layout (e.g. written by prebuilt
/// std code) and rebuild it in this importer's representation, following the
/// real field offsets recursively.
fn load_value_from_real_layout<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    rust_ty: Ty<'tcx>,
    addr: Value,
) -> Result<Value, String> {
    use rustc_middle::ty::TyKind;
    let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
    let rust_ty = normalize_ty(tcx, runtime_ty(rust_ty));

    let load_direct = |ctx: &mut Context, ty: TypeHandle| -> Value {
        let load = stair_mir::ops::LoadOp::new(ctx, addr, ty);
        load.get_operation().insert_at_back(insert_block, ctx);
        load.get_result(ctx)
    };

    let field_based = |ctx: &mut Context, fields: Vec<Ty<'tcx>>| -> Result<Value, String> {
        let our_ty = convert_ty(tcx, ctx, rust_ty)?;
        let layout = tcx
            .layout_of(typing_env.as_query_input(rust_ty))
            .map_err(|error| format!("no layout for {rust_ty:?}: {error:?}"))?;
        let undef = stair_mir::ops::UndefOp::new(ctx, our_ty);
        undef.get_operation().insert_at_back(insert_block, ctx);
        let mut current = undef.get_result(ctx);
        for (idx, field_ty) in fields.into_iter().enumerate() {
            if layout_size_of_ty(tcx, field_ty)? == 0 {
                continue;
            }
            let offset = layout.fields.offset(idx).bytes();
            let field_addr = if offset != 0 {
                ptr_offset_const(ctx, insert_block, addr, offset)?
            } else {
                addr
            };
            let value = load_value_from_real_layout(tcx, ctx, insert_block, field_ty, field_addr)?;
            let index = converted_field_index(tcx, rust_ty, idx)?;
            let insert = stair_mir::ops::InsertValueOp::new(ctx, value, current, vec![index]);
            insert.get_operation().insert_at_back(insert_block, ctx);
            current = insert.get_result(ctx);
        }
        Ok(current)
    };

    match rust_ty.kind() {
        TyKind::Bool
        | TyKind::Char
        | TyKind::Int(_)
        | TyKind::Uint(_)
        | TyKind::Float(_)
        | TyKind::FnPtr(_, _)
        | TyKind::Ref(_, _, _)
        | TyKind::RawPtr(_, _) => {
            // Scalars and references; fat references share the real
            // {ptr, meta} two-word layout.
            let our_ty = convert_ty(tcx, ctx, rust_ty)?;
            Ok(load_direct(ctx, our_ty))
        }
        TyKind::Tuple(fields) => field_based(ctx, fields.iter().collect()),
        TyKind::Closure(_, args) => {
            field_based(ctx, args.as_closure().upvar_tys().iter().collect())
        }
        TyKind::Adt(adt_def, _) if is_fmt_rt_argument_type(tcx, adt_def.did()) => {
            let our_ty = convert_ty(tcx, ctx, rust_ty)?;
            Ok(load_direct(ctx, our_ty))
        }
        TyKind::Adt(adt_def, args) if adt_def.is_struct() => {
            let fields = adt_def
                .non_enum_variant()
                .fields
                .iter()
                .map(|field| field.ty(tcx, args))
                .collect::<Vec<_>>();
            field_based(ctx, fields)
        }
        TyKind::Adt(adt_def, _) if adt_def.is_union() => {
            let our_ty = convert_ty(tcx, ctx, rust_ty)?;
            Ok(load_direct(ctx, our_ty))
        }
        TyKind::Adt(adt_def, _) if adt_def.is_enum() => {
            // An enum value is simply its real in-memory bytes, so its converted
            // type already matches the source layout: load it directly.
            let our_ty = convert_ty(tcx, ctx, rust_ty)?;
            Ok(load_direct(ctx, our_ty))
        }
        TyKind::Array(elem, len) => {
            let our_ty = convert_ty(tcx, ctx, rust_ty)?;
            let len = array_len(tcx, *len)?;
            let elem_stride = rustc_layout_size_of_ty(tcx, typing_env, *elem)?;
            let undef = stair_mir::ops::UndefOp::new(ctx, our_ty);
            undef.get_operation().insert_at_back(insert_block, ctx);
            let mut current = undef.get_result(ctx);
            for idx in 0..len {
                let offset = idx * elem_stride;
                let elem_addr = if offset != 0 {
                    ptr_offset_const(ctx, insert_block, addr, offset)?
                } else {
                    addr
                };
                let value = load_value_from_real_layout(tcx, ctx, insert_block, *elem, elem_addr)?;
                let insert =
                    stair_mir::ops::InsertValueOp::new(ctx, value, current, vec![idx as u32]);
                insert.get_operation().insert_at_back(insert_block, ctx);
                current = insert.get_result(ctx);
            }
            Ok(current)
        }
        other => Err(format!("unsupported real-layout load: {other:?}")),
    }
}

fn import_enum_constant<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    typing_env: rustc_middle::ty::TypingEnv<'tcx>,
    span: rustc_span::Span,
    constant: rustc_mir::Const<'tcx>,
    ty: TypeHandle,
) -> Result<Value, String> {
    // An enum value is its real in-memory bytes, so materialize the constant's
    // real layout into an anonymous global and load the enum blob from it. This
    // handles C-like, niche-encoded, and data-carrying enum constants uniformly.
    let rust_ty = normalize_ty(tcx, runtime_ty(constant.ty()));
    let size = rustc_layout_size_of_ty(tcx, typing_env, rust_ty)?;
    if size == 0 {
        let op = stair_mir::ops::UndefOp::new(ctx, ty);
        op.get_operation().insert_at_back(insert_block, ctx);
        return Ok(op.get_result(ctx));
    }
    let Some(bytes) = enum_constant_bytes(tcx, typing_env, span, constant, size) else {
        // The variant payload carries pointers (e.g. `Some(&STATIC)`): fall
        // back to the relocated-global path, which loads the enum blob out of
        // an `ll.data` global whose pointer slots the linker fills in.
        return import_pointer_constant(
            tcx, ctx, state, insert_block, typing_env, span, constant, ty,
        )?
        .ok_or_else(|| format!("unsupported MIR enum constant: {:?}", constant));
    };
    let symbol = declare_anonymous_byte_global(ctx, state.module_body, &bytes);
    let ptr_ty = llvm_ptr_ty(ctx);
    let addr = stair_mir::ops::AddressOfOp::new(ctx, symbol, ptr_ty);
    addr.get_operation().insert_at_back(insert_block, ctx);
    let load = stair_mir::ops::LoadOp::new(ctx, addr.get_result(ctx), ty);
    load.get_operation().insert_at_back(insert_block, ctx);
    Ok(load.get_result(ctx))
}

/// The `size` real-layout bytes of an enum constant, or `None` when it cannot be
/// evaluated to plain bytes (e.g. it carries pointer relocations).
fn enum_constant_bytes<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: rustc_middle::ty::TypingEnv<'tcx>,
    span: rustc_span::Span,
    constant: rustc_mir::Const<'tcx>,
    size: u64,
) -> Option<Vec<u8>> {
    match constant.eval(tcx, typing_env, span).ok()? {
        rustc_mir::ConstValue::Scalar(scalar) => {
            let scalar = scalar.try_to_scalar_int().ok()?;
            let mut bytes = scalar.to_uint(scalar.size()).to_le_bytes().to_vec();
            bytes.resize(size as usize, 0);
            Some(bytes)
        }
        rustc_mir::ConstValue::Indirect { alloc_id, offset } => {
            let rustc_mir::interpret::GlobalAlloc::Memory(alloc) = tcx.global_alloc(alloc_id)
            else {
                return None;
            };
            let alloc = alloc.inner();
            if !alloc.provenance().ptrs().is_empty() {
                return None;
            }
            let start = offset.bytes() as usize;
            Some(
                alloc
                    .inspect_with_uninit_and_ptr_outside_interpreter(start..start + size as usize)
                    .to_vec(),
            )
        }
        _ => None,
    }
}

fn import_initialized_maybe_uninit_u8_constant<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    typing_env: rustc_middle::ty::TypingEnv<'tcx>,
    span: rustc_span::Span,
    constant: rustc_mir::Const<'tcx>,
    ty: TypeHandle,
) -> Result<Option<Value>, String> {
    if !is_maybe_uninit_u8_ty(tcx, constant.ty()) {
        return Ok(None);
    }
    let Some(bytes) = evaluated_constant_bytes(tcx, typing_env, span, constant, 1) else {
        return Ok(None);
    };
    let Some(byte) = bytes.first().copied() else {
        return Ok(None);
    };

    let byte_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
    let byte = integer_constant(ctx, byte_ty, byte as u128)?;
    byte.get_operation().insert_at_back(insert_block, ctx);

    let array_ty = llvm::types::ArrayType::get(ctx, byte_ty, 1).into();
    let array_undef = stair_mir::ops::UndefOp::new(ctx, array_ty);
    array_undef
        .get_operation()
        .insert_at_back(insert_block, ctx);
    let array = stair_mir::ops::InsertValueOp::new(
        ctx,
        byte.get_result(ctx),
        array_undef.get_result(ctx),
        vec![0],
    );
    array.get_operation().insert_at_back(insert_block, ctx);

    let outer_undef = stair_mir::ops::UndefOp::new(ctx, ty);
    outer_undef
        .get_operation()
        .insert_at_back(insert_block, ctx);
    let outer = stair_mir::ops::InsertValueOp::new(
        ctx,
        array.get_result(ctx),
        outer_undef.get_result(ctx),
        vec![0],
    );
    outer.get_operation().insert_at_back(insert_block, ctx);
    Ok(Some(outer.get_result(ctx)))
}

fn evaluated_constant_bytes<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: rustc_middle::ty::TypingEnv<'tcx>,
    span: rustc_span::Span,
    constant: rustc_mir::Const<'tcx>,
    len: u64,
) -> Option<Vec<u8>> {
    match constant.eval(tcx, typing_env, span).ok()? {
        rustc_mir::ConstValue::Indirect { alloc_id, offset } => {
            allocation_bytes(tcx, alloc_id, offset, len)
        }
        rustc_mir::ConstValue::Scalar(scalar) => scalar
            .try_to_scalar_int()
            .ok()
            .map(|scalar| scalar.to_uint(scalar.size()).to_le_bytes()[..len as usize].to_vec()),
        _ => None,
    }
}

fn rustc_layout_size_of_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: rustc_middle::ty::TypingEnv<'tcx>,
    ty: Ty<'tcx>,
) -> Result<u64, String> {
    tcx.layout_of(typing_env.as_query_input(ty))
        .map(|layout| layout.size.bytes())
        .map_err(|error| format!("unsupported Rust type layout in MIR importer: {ty:?}: {error:?}"))
}

fn rustc_layout_align_of_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: rustc_middle::ty::TypingEnv<'tcx>,
    ty: Ty<'tcx>,
) -> Result<u64, String> {
    tcx.layout_of(typing_env.as_query_input(ty))
        .map(|layout| layout.align.abi.bytes())
        .map_err(|error| format!("unsupported Rust type layout in MIR importer: {ty:?}: {error:?}"))
}

fn mono_const<'tcx>(
    tcx: TyCtxt<'tcx>,
    state: &FunctionImportState<'tcx>,
    constant: rustc_mir::Const<'tcx>,
) -> rustc_mir::Const<'tcx> {
    state.instance.map_or(constant, |instance| {
        instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            rustc_middle::ty::TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(constant),
        )
    })
}

/// Monomorphize a type-level constant (e.g. an array length that mentions a
/// generic const parameter) the same way `mono_ty`/`mono_const` do.
fn mono_ty_const<'tcx>(
    tcx: TyCtxt<'tcx>,
    state: &FunctionImportState<'tcx>,
    constant: rustc_middle::ty::Const<'tcx>,
) -> rustc_middle::ty::Const<'tcx> {
    state.instance.map_or(constant, |instance| {
        instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            rustc_middle::ty::TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(constant),
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn import_str_constant<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    module_body: Ptr<BasicBlock>,
    insert_block: Ptr<BasicBlock>,
    typing_env: rustc_middle::ty::TypingEnv<'tcx>,
    body: &Body<'tcx>,
    constant: &ConstOperand<'tcx>,
    const_: rustc_mir::Const<'tcx>,
) -> Result<Value, String> {
    let value = literal_string_constant(tcx, body, constant)
        .or_else(|| evaluated_str_constant(tcx, typing_env, constant.span, const_))
        .ok_or_else(|| format!("unsupported string constant: {:?}", const_))?;
    // The old `cmir.cstr` becomes a private NUL-terminated byte global plus
    // an address-of; the NUL matches the old ll.cstr literal emission and is
    // excluded from the `{ptr, len}` str-ref length below.
    let mut bytes = value.clone().into_bytes();
    bytes.push(0);
    let symbol = declare_anonymous_byte_global(ctx, module_body, &bytes);
    let ptr_ty = llvm_ptr_ty(ctx);
    let ptr = stair_mir::ops::AddressOfOp::new(ctx, symbol, ptr_ty);
    ptr.get_operation().insert_at_back(insert_block, ctx);

    let usize_ty = usize_ty(ctx);
    let len = stair_mir::ops::ConstantOp::new_integer(
        ctx,
        IntegerAttr::new(
            usize_ty,
            APInt::from_u64(value.len() as u64, NonZero::new(64).unwrap()),
        ),
    );
    len.get_operation().insert_at_back(insert_block, ctx);

    let str_ref_ty = str_ref_ty(ctx);
    let undef = stair_mir::ops::UndefOp::new(ctx, str_ref_ty);
    undef.get_operation().insert_at_back(insert_block, ctx);
    let with_ptr = stair_mir::ops::InsertValueOp::new(
        ctx,
        ptr.get_result(ctx),
        undef.get_result(ctx),
        vec![0],
    );
    with_ptr.get_operation().insert_at_back(insert_block, ctx);
    let with_len = stair_mir::ops::InsertValueOp::new(
        ctx,
        len.get_result(ctx),
        with_ptr.get_result(ctx),
        vec![1],
    );
    with_len.get_operation().insert_at_back(insert_block, ctx);
    Ok(with_len.get_result(ctx))
}

/// Materialize scalar constant bits into a struct type that (recursively)
/// wraps exactly one sized scalar field.
fn scalar_constant_in_aggregate(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    ty: TypeHandle,
    bits: u128,
) -> Result<Option<Value>, String> {
    if let Ok(op) = constant_from_bits(ctx, ty, bits) {
        op.get_operation().insert_at_back(insert_block, ctx);
        return Ok(Some(op.get_result(ctx)));
    }
    let fields = {
        let ty_ref = ty.deref(ctx);
        let Some(struct_ty) = ty_ref.downcast_ref::<llvm::types::StructType>() else {
            return Ok(None);
        };
        if struct_ty.is_opaque() {
            return Ok(None);
        }
        struct_ty.fields().collect::<Vec<_>>()
    };
    let mut sized = fields.iter().enumerate().filter(|(_, field)| {
        stair_ty_size(ctx, **field)
            .map(|size| size > 0)
            .unwrap_or(true)
    });
    let Some((index, field_ty)) = sized.next() else {
        return Ok(None);
    };
    if sized.next().is_some() {
        return Ok(None);
    }
    let field_ty = *field_ty;
    let Some(inner) = scalar_constant_in_aggregate(ctx, insert_block, field_ty, bits)? else {
        return Ok(None);
    };
    let undef = stair_mir::ops::UndefOp::new(ctx, ty);
    undef.get_operation().insert_at_back(insert_block, ctx);
    let wrap =
        stair_mir::ops::InsertValueOp::new(ctx, inner, undef.get_result(ctx), vec![index as u32]);
    wrap.get_operation().insert_at_back(insert_block, ctx);
    Ok(Some(wrap.get_result(ctx)))
}

fn constant_from_bits(
    ctx: &mut Context,
    ty: TypeHandle,
    bits: u128,
) -> Result<stair_mir::ops::ConstantOp, String> {
    if ty.deref(ctx).downcast_ref::<FP32Type>().is_some() {
        return Ok(stair_mir::ops::ConstantOp::new(
            ctx,
            FPSingleAttr::from(f32::from_bits(bits as u32)).into(),
        ));
    }
    if ty.deref(ctx).downcast_ref::<FP64Type>().is_some() {
        return Ok(stair_mir::ops::ConstantOp::new(
            ctx,
            FPDoubleAttr::from(f64::from_bits(bits as u64)).into(),
        ));
    }
    integer_constant(ctx, ty, bits)
}

fn integer_constant(
    ctx: &mut Context,
    ty: TypeHandle,
    bits: u128,
) -> Result<stair_mir::ops::ConstantOp, String> {
    let ty_ref = ty.deref(ctx);
    let int_ty = ty_ref
        .downcast_ref::<IntegerType>()
        .ok_or_else(|| "MIR integer constant has non-integer type".to_string())?;
    let width = int_ty.width();
    let int_ty: TypedHandle<IntegerType> = TypedHandle::from_handle(ty, ctx).unwrap();
    drop(ty_ref);
    Ok(stair_mir::ops::ConstantOp::new_integer(
        ctx,
        IntegerAttr::new(
            int_ty,
            APInt::from_u128(bits, NonZero::new(width as usize).unwrap()),
        ),
    ))
}

fn load_place<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    place: Place<'tcx>,
) -> Result<Value, String> {
    if let Some(value) = try_load_field_projection(tcx, ctx, state, insert_block, body, place)? {
        return Ok(value);
    }

    let rust_ty = mono_ty(tcx, state, place.ty(body, tcx).ty);
    if convert_storage_ty(tcx, ctx, rust_ty)?.is_none() {
        let unit_ty = convert_ty(tcx, ctx, rust_ty)?;
        let undef = stair_mir::ops::UndefOp::new(ctx, unit_ty);
        undef.get_operation().insert_at_back(insert_block, ctx);
        return Ok(undef.get_result(ctx));
    }

    let slot = place_addr(tcx, ctx, state, insert_block, body, &place)?;
    let ty = convert_ty(tcx, ctx, rust_ty)?;
    let load = stair_mir::ops::LoadOp::new(ctx, slot, ty);
    load.get_operation().insert_at_back(insert_block, ctx);
    normalize_bool_for_immediate(tcx, ctx, state, insert_block, rust_ty, load.get_result(ctx))
}

fn try_load_field_projection<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    place: Place<'tcx>,
) -> Result<Option<Value>, String> {
    if place.projection.is_empty()
        || !place
            .projection
            .iter()
            .all(|elem| matches!(elem, rustc_mir::ProjectionElem::Field(_, _)))
        || projection_touches_union(tcx, state, body, place)?
    {
        return Ok(None);
    }

    let slot = local_slot(state, place.local)?;
    let aggregate_ty = convert_ty(
        tcx,
        ctx,
        mono_ty(tcx, state, body.local_decls[place.local].ty),
    )?;
    let load = stair_mir::ops::LoadOp::new(ctx, slot, aggregate_ty);
    load.get_operation().insert_at_back(insert_block, ctx);
    let mut current = load.get_result(ctx);
    let mut current_rust_ty = mono_ty(tcx, state, body.local_decls[place.local].ty);

    for elem in place.projection {
        let rustc_mir::ProjectionElem::Field(field, field_ty) = elem else {
            unreachable!("field-only projection checked above");
        };
        let index = converted_field_index(tcx, current_rust_ty, field.index())?;
        let result_ty = convert_ty(tcx, ctx, mono_ty(tcx, state, field_ty))?;
        let extract = stair_mir::ops::ExtractValueOp::new(ctx, current, vec![index], result_ty);
        extract.get_operation().insert_at_back(insert_block, ctx);
        current = extract.get_result(ctx);
        current_rust_ty = mono_ty(tcx, state, field_ty);
    }

    let rust_ty = mono_ty(tcx, state, place.ty(body, tcx).ty);
    normalize_bool_for_immediate(tcx, ctx, state, insert_block, rust_ty, current).map(Some)
}

/// The index of a source-order field within the converted struct type.
fn converted_field_index<'tcx>(
    tcx: TyCtxt<'tcx>,
    aggregate_ty: Ty<'tcx>,
    source_idx: usize,
) -> Result<u32, String> {
    let aggregate_ty = normalize_ty(tcx, runtime_ty(aggregate_ty));
    match struct_like_source_fields(tcx, aggregate_ty) {
        Some(fields) => Ok(field_memory_position(
            tcx,
            aggregate_ty,
            fields.len(),
            source_idx,
        )),
        None => Ok(source_idx as u32),
    }
}

fn store_place<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    place: &Place<'tcx>,
    value: Value,
) -> Result<(), String> {
    let rust_ty = mono_ty(tcx, state, place.ty(body, tcx).ty);
    let value = normalize_bool_for_storage(tcx, ctx, state, insert_block, rust_ty, value)?;
    if !place.projection.is_empty()
        && place
            .projection
            .iter()
            .all(|elem| matches!(elem, rustc_mir::ProjectionElem::Field(_, _)))
        && !projection_touches_union(tcx, state, body, *place)?
    {
        return store_field_projection(tcx, ctx, state, insert_block, body, place, value);
    }

    let addr = place_addr(tcx, ctx, state, insert_block, body, place)?;
    let store = stair_mir::ops::StoreOp::new(ctx, value, addr);
    store.get_operation().insert_at_back(insert_block, ctx);
    Ok(())
}

fn store_field_projection<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    place: &Place<'tcx>,
    value: Value,
) -> Result<(), String> {
    let slot = local_slot(state, place.local)?;
    let aggregate_ty = convert_ty(
        tcx,
        ctx,
        mono_ty(tcx, state, body.local_decls[place.local].ty),
    )?;
    let load = stair_mir::ops::LoadOp::new(ctx, slot, aggregate_ty);
    load.get_operation().insert_at_back(insert_block, ctx);

    let mut current = load.get_result(ctx);
    let mut current_rust_ty = mono_ty(tcx, state, body.local_decls[place.local].ty);
    let mut parents = Vec::<(u32, Value)>::new();
    for elem in place
        .projection
        .iter()
        .take(place.projection.len().saturating_sub(1))
    {
        let rustc_mir::ProjectionElem::Field(field, field_ty) = elem else {
            unreachable!("field-only projection checked above");
        };
        let index = converted_field_index(tcx, current_rust_ty, field.index())?;
        let result_ty = convert_ty(tcx, ctx, mono_ty(tcx, state, field_ty))?;
        parents.push((index, current));
        let extract = stair_mir::ops::ExtractValueOp::new(ctx, current, vec![index], result_ty);
        extract.get_operation().insert_at_back(insert_block, ctx);
        current = extract.get_result(ctx);
        current_rust_ty = mono_ty(tcx, state, field_ty);
    }

    let Some(rustc_mir::ProjectionElem::Field(field, _)) = place.projection.last() else {
        unreachable!("non-empty field-only projection checked above");
    };
    let last_index = converted_field_index(tcx, current_rust_ty, field.index())?;
    let insert = stair_mir::ops::InsertValueOp::new(ctx, value, current, vec![last_index]);
    insert.get_operation().insert_at_back(insert_block, ctx);
    let mut updated = insert.get_result(ctx);

    for (index, parent) in parents.into_iter().rev() {
        let insert = stair_mir::ops::InsertValueOp::new(ctx, updated, parent, vec![index]);
        insert.get_operation().insert_at_back(insert_block, ctx);
        updated = insert.get_result(ctx);
    }

    let store = stair_mir::ops::StoreOp::new(ctx, updated, slot);
    store.get_operation().insert_at_back(insert_block, ctx);
    Ok(())
}

/// The "address" of a slot-less local (type `()` or `!`): a dangling pointer
/// at the type's alignment, the same convention rustc's codegen uses for ZST
/// places. The pointee is never read or written through it.
fn dangling_zst_addr<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    local: Local,
) -> Result<Value, String> {
    let ty = mono_ty(tcx, state, body.local_decls[local].ty);
    let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
    let align = rustc_layout_align_of_ty(tcx, typing_env, runtime_ty(ty))
        .unwrap_or(1)
        .max(1);
    let usize_ty: TypeHandle = usize_ty(ctx).into();
    let addr = integer_constant(ctx, usize_ty, align as u128)?;
    addr.get_operation().insert_at_back(insert_block, ctx);
    let ptr_ty = llvm_ptr_ty(ctx);
    Ok(cast_value_to_type(
        ctx,
        insert_block,
        addr.get_result(ctx),
        ptr_ty,
    ))
}

fn place_addr<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    state: &FunctionImportState<'tcx>,
    insert_block: Ptr<BasicBlock>,
    body: &Body<'tcx>,
    place: &Place<'tcx>,
) -> Result<Value, String> {
    let base = match local_slot_opt(state, place.local)? {
        Some(slot) => slot,
        // A local of type `()` or `!` gets no storage slot, but MIR may
        // still take its address (e.g. `&core::ptr::metadata(p)` where the
        // metadata is `()`). Mirror rustc's ZST codegen: a dangling,
        // suitably aligned pointer.
        None => dangling_zst_addr(tcx, ctx, state, insert_block, body, place.local)?,
    };
    if place.projection.is_empty() {
        return Ok(base);
    }

    let mut addr = base;
    let mut current_ty = mono_ty(tcx, state, body.local_decls[place.local].ty);
    let mut current_variant = None;
    for elem in place.projection {
        match elem {
            rustc_mir::ProjectionElem::Deref => {
                let ptr_ty = convert_ty(tcx, ctx, mono_ty(tcx, state, current_ty))?;
                let load = stair_mir::ops::LoadOp::new(ctx, addr, ptr_ty);
                load.get_operation().insert_at_back(insert_block, ctx);
                addr = load.get_result(ctx);
                current_ty = mono_ty(tcx, state, pointee_ty(current_ty)?);
                current_variant = None;
            }
            rustc_mir::ProjectionElem::Field(field, field_ty) => {
                let offset = field_offset_of(tcx, current_ty, current_variant, field.index())?;
                let field_ty = mono_ty(tcx, state, field_ty);
                let addr_is_fat = addr
                    .get_type(ctx)
                    .deref(ctx)
                    .is::<llvm::types::StructType>();
                if addr_is_fat {
                    // The base is a fat `{ptr, meta}` value (a Deref of a
                    // reference to an unsized ADT). Sized fields live at byte
                    // offsets off the data pointer; the unsized tail field
                    // keeps the metadata.
                    let fat_ty = addr.get_type(ctx);
                    let ptr_ty = llvm_ptr_ty(ctx);
                    let data = stair_mir::ops::ExtractValueOp::new(ctx, addr, vec![0], ptr_ty);
                    data.get_operation().insert_at_back(insert_block, ctx);
                    let mut data = data.get_result(ctx);
                    if offset != 0 {
                        data = ptr_offset_const(ctx, insert_block, data, offset)?;
                    }
                    let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
                    if runtime_ty(field_ty).is_sized(tcx, typing_env) {
                        addr = data;
                    } else {
                        let usize_ty: TypeHandle = usize_ty(ctx).into();
                        let meta =
                            stair_mir::ops::ExtractValueOp::new(ctx, addr, vec![1], usize_ty);
                        meta.get_operation().insert_at_back(insert_block, ctx);
                        let undef = stair_mir::ops::UndefOp::new(ctx, fat_ty);
                        undef.get_operation().insert_at_back(insert_block, ctx);
                        let with_ptr = stair_mir::ops::InsertValueOp::new(
                            ctx,
                            data,
                            undef.get_result(ctx),
                            vec![0],
                        );
                        with_ptr.get_operation().insert_at_back(insert_block, ctx);
                        let with_meta = stair_mir::ops::InsertValueOp::new(
                            ctx,
                            meta.get_result(ctx),
                            with_ptr.get_result(ctx),
                            vec![1],
                        );
                        with_meta.get_operation().insert_at_back(insert_block, ctx);
                        addr = with_meta.get_result(ctx);
                    }
                } else if offset != 0 {
                    addr = ptr_offset_const(ctx, insert_block, addr, offset)?;
                }
                current_ty = field_ty;
                current_variant = None;
            }
            rustc_mir::ProjectionElem::Downcast(_, variant) => {
                current_variant = Some(variant);
            }
            rustc_mir::ProjectionElem::Index(index_local) => {
                let index = load_place(tcx, ctx, state, insert_block, body, index_local.into())?;
                let elem_size = indexed_elem_size(tcx, current_ty)?;
                let byte_offset = scale_index(ctx, insert_block, index, elem_size)?;
                let offset = stair_mir::ops::PtrOffsetOp::new(ctx, addr, byte_offset);
                offset.get_operation().insert_at_back(insert_block, ctx);
                addr = offset.get_result(ctx);
                current_ty = mono_ty(tcx, state, indexed_elem_ty(current_ty)?);
                current_variant = None;
            }
            rustc_mir::ProjectionElem::ConstantIndex {
                offset,
                min_length,
                from_end,
            } => {
                let index = if from_end {
                    min_length - offset
                } else {
                    offset
                };
                let elem_size = indexed_elem_size(tcx, current_ty)?;
                let byte_offset = index * elem_size;
                if byte_offset != 0 {
                    addr = ptr_offset_const(ctx, insert_block, addr, byte_offset)?;
                }
                current_ty = mono_ty(tcx, state, indexed_elem_ty(current_ty)?);
                current_variant = None;
            }
            rustc_mir::ProjectionElem::Subslice { from, to, from_end } => {
                match runtime_ty(current_ty).kind() {
                    rustc_middle::ty::TyKind::Array(elem, len) => {
                        // `array[from..]` starts `from` elements in; the result
                        // is a shorter array at a constant byte offset.
                        let elem_ty = mono_ty(tcx, state, *elem);
                        let elem_size = indexed_elem_size(tcx, current_ty)?;
                        let byte_offset = from * elem_size;
                        if byte_offset != 0 {
                            addr = ptr_offset_const(ctx, insert_block, addr, byte_offset)?;
                        }
                        let new_len = if from_end {
                            array_len(tcx, mono_ty_const(tcx, state, *len))?
                                .checked_sub(from + to)
                                .ok_or_else(|| {
                                    format!("MIR Subslice out of bounds: {elem:?} {from}..-{to}")
                                })?
                        } else {
                            to - from
                        };
                        current_ty = Ty::new_array(tcx, elem_ty, new_len);
                    }
                    rustc_middle::ty::TyKind::Slice(_) => {
                        // A slice place's "address" is the fat `{ptr, len}`
                        // value the preceding Deref loaded: rebase the data
                        // pointer by `from` elements and shrink the length.
                        let elem_size = indexed_elem_size(tcx, current_ty)?;
                        let fat_ty = addr.get_type(ctx);
                        let ptr_ty = llvm_ptr_ty(ctx);
                        let usize_ty: TypeHandle = usize_ty(ctx).into();
                        let data = stair_mir::ops::ExtractValueOp::new(ctx, addr, vec![0], ptr_ty);
                        data.get_operation().insert_at_back(insert_block, ctx);
                        let mut data = data.get_result(ctx);
                        if from != 0 {
                            data = ptr_offset_const(ctx, insert_block, data, from * elem_size)?;
                        }
                        let new_len = if from_end {
                            let len =
                                stair_mir::ops::ExtractValueOp::new(ctx, addr, vec![1], usize_ty);
                            len.get_operation().insert_at_back(insert_block, ctx);
                            let dropped = integer_constant(ctx, usize_ty, (from + to) as u128)?;
                            dropped.get_operation().insert_at_back(insert_block, ctx);
                            let sub = stair_mir::ops::SubOp::new(
                                ctx,
                                len.get_result(ctx),
                                dropped.get_result(ctx),
                            );
                            sub.get_operation().insert_at_back(insert_block, ctx);
                            sub.get_result(ctx)
                        } else {
                            let len = integer_constant(ctx, usize_ty, (to - from) as u128)?;
                            len.get_operation().insert_at_back(insert_block, ctx);
                            len.get_result(ctx)
                        };
                        let undef = stair_mir::ops::UndefOp::new(ctx, fat_ty);
                        undef.get_operation().insert_at_back(insert_block, ctx);
                        let with_ptr = stair_mir::ops::InsertValueOp::new(
                            ctx,
                            data,
                            undef.get_result(ctx),
                            vec![0],
                        );
                        with_ptr.get_operation().insert_at_back(insert_block, ctx);
                        let with_len = stair_mir::ops::InsertValueOp::new(
                            ctx,
                            new_len,
                            with_ptr.get_result(ctx),
                            vec![1],
                        );
                        with_len.get_operation().insert_at_back(insert_block, ctx);
                        addr = with_len.get_result(ctx);
                        // The place stays the same slice type.
                    }
                    other => {
                        return Err(format!(
                            "unsupported MIR Subslice base type: {other:?}"
                        ));
                    }
                }
                current_variant = None;
            }
            rustc_mir::ProjectionElem::OpaqueCast(ty)
            | rustc_mir::ProjectionElem::UnwrapUnsafeBinder(ty) => {
                current_ty = mono_ty(tcx, state, ty);
                current_variant = None;
            }
        }
    }

    Ok(addr)
}

fn ptr_offset_const(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    base: Value,
    offset: u64,
) -> Result<Value, String> {
    let usize_ty: TypeHandle = usize_ty(ctx).into();
    let offset_op = integer_constant(ctx, usize_ty, offset as u128)?;
    offset_op.get_operation().insert_at_back(insert_block, ctx);
    let ptr_offset = stair_mir::ops::PtrOffsetOp::new(ctx, base, offset_op.get_result(ctx));
    ptr_offset.get_operation().insert_at_back(insert_block, ctx);
    Ok(ptr_offset.get_result(ctx))
}

fn scale_index(
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    index: Value,
    element_size: u64,
) -> Result<Value, String> {
    if element_size == 1 {
        return Ok(index);
    }
    let scale = integer_constant(ctx, index.get_type(ctx), element_size as u128)?;
    scale.get_operation().insert_at_back(insert_block, ctx);
    let mul = stair_mir::ops::MulOp::new(ctx, index, scale.get_result(ctx));
    mul.get_operation().insert_at_back(insert_block, ctx);
    Ok(mul.get_result(ctx))
}

fn pointee_ty<'tcx>(ty: Ty<'tcx>) -> Result<Ty<'tcx>, String> {
    match runtime_ty(ty).kind() {
        rustc_middle::ty::TyKind::Ref(_, inner, _) | rustc_middle::ty::TyKind::RawPtr(inner, _) => {
            Ok(*inner)
        }
        other => Err(format!("MIR deref of non-pointer type: {other:?}")),
    }
}

fn indexed_elem_ty<'tcx>(ty: Ty<'tcx>) -> Result<Ty<'tcx>, String> {
    match runtime_ty(ty).kind() {
        rustc_middle::ty::TyKind::Array(elem, _) | rustc_middle::ty::TyKind::Slice(elem) => {
            Ok(*elem)
        }
        other => Err(format!("MIR index of non-array type: {other:?}")),
    }
}

fn aggregate_is_union<'tcx>(tcx: TyCtxt<'tcx>, kind: &rustc_mir::AggregateKind<'tcx>) -> bool {
    match kind {
        rustc_mir::AggregateKind::Adt(def_id, _, _, _, _) => tcx.adt_def(*def_id).is_union(),
        _ => false,
    }
}

fn projection_touches_union<'tcx>(
    tcx: TyCtxt<'tcx>,
    state: &FunctionImportState<'tcx>,
    body: &Body<'tcx>,
    place: Place<'tcx>,
) -> Result<bool, String> {
    let mut current_ty = mono_ty(tcx, state, body.local_decls[place.local].ty);
    let mut current_variant = None;
    for elem in place.projection {
        match elem {
            rustc_mir::ProjectionElem::Field(_, field_ty) => {
                if matches!(
                    runtime_ty(current_ty).kind(),
                    rustc_middle::ty::TyKind::Adt(adt_def, _) if adt_def.is_union()
                ) {
                    return Ok(true);
                }
                current_ty = mono_ty(tcx, state, field_ty);
                current_variant = None;
            }
            rustc_mir::ProjectionElem::Downcast(_, variant) => {
                current_variant = Some(variant);
            }
            rustc_mir::ProjectionElem::Deref => {
                current_ty = mono_ty(tcx, state, pointee_ty(current_ty)?);
                current_variant = None;
            }
            rustc_mir::ProjectionElem::Index(_) => {
                current_ty = mono_ty(tcx, state, indexed_elem_ty(current_ty)?);
                current_variant = None;
            }
            rustc_mir::ProjectionElem::ConstantIndex { .. } => {
                current_ty = mono_ty(tcx, state, indexed_elem_ty(current_ty)?);
                current_variant = None;
            }
            rustc_mir::ProjectionElem::OpaqueCast(ty)
            | rustc_mir::ProjectionElem::UnwrapUnsafeBinder(ty) => {
                current_ty = mono_ty(tcx, state, ty);
                current_variant = None;
            }
            _ => {}
        }
        let _ = current_variant;
    }
    Ok(false)
}

fn indexed_elem_size<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Result<u64, String> {
    layout_size_of_ty(tcx, indexed_elem_ty(ty)?)
}

fn field_offset_of<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    variant: Option<VariantIdx>,
    index: usize,
) -> Result<u64, String> {
    let ty = runtime_ty(ty);
    let fields = match ty.kind() {
        rustc_middle::ty::TyKind::Adt(adt_def, _) if adt_def.is_enum() => {
            let variant = variant.ok_or_else(|| {
                format!("MIR enum field projection without variant downcast: {ty:?}.{index}")
            })?;
            if index >= adt_def.variant(variant).fields.len() {
                return Err(format!(
                    "MIR enum field index {index} out of bounds for variant {:?}",
                    adt_def.variant(variant).name
                ));
            }
            // Enums use the real rustc layout, so the field offset comes
            // straight from the variant's layout.
            let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
            let layout = tcx
                .layout_of(typing_env.as_query_input(ty))
                .map_err(|error| format!("no layout for {ty:?}: {error:?}"))?;
            let cx = rustc_middle::ty::layout::LayoutCx::new(tcx, typing_env);
            let variant_layout = layout.for_variant(&cx, variant);
            return Ok(variant_layout.fields.offset(index).bytes());
        }
        rustc_middle::ty::TyKind::Adt(adt_def, args) if adt_def.is_struct() => adt_def
            .non_enum_variant()
            .fields
            .iter()
            .map(|field| field.ty(tcx, args))
            .collect::<Vec<_>>(),
        rustc_middle::ty::TyKind::Adt(adt_def, args) if adt_def.is_union() => {
            let fields = adt_def
                .non_enum_variant()
                .fields
                .iter()
                .map(|field| field.ty(tcx, args))
                .collect::<Vec<_>>();
            if index >= fields.len() {
                return Err(format!(
                    "MIR union field index {index} out of bounds for {} fields",
                    fields.len()
                ));
            }
            return Ok(0);
        }
        rustc_middle::ty::TyKind::Tuple(fields) => fields.iter().collect::<Vec<_>>(),
        rustc_middle::ty::TyKind::Closure(_, args) => {
            args.as_closure().upvar_tys().iter().collect::<Vec<_>>()
        }
        other => {
            return Err(format!(
                "MIR field projection of non-aggregate type: {other:?}"
            ));
        }
    };
    if index >= fields.len() {
        return Err(format!(
            "MIR field index {index} out of bounds for {} fields",
            fields.len()
        ));
    }
    let order = struct_memory_order(tcx, ty, fields.len());
    let ordered: Vec<_> = order.iter().map(|&idx| fields[idx]).collect();
    let offsets = struct_like_field_offsets(tcx, &ordered)?;
    let position = field_memory_position(tcx, ty, fields.len(), index);
    Ok(offsets[position as usize])
}

/// Field offsets of a struct-like sequence of fields, aligned exactly the
/// way the AArch64 lowering lays out converted `llvm.struct` types.
fn struct_like_field_offsets<'tcx>(
    tcx: TyCtxt<'tcx>,
    fields: &[Ty<'tcx>],
) -> Result<Vec<u64>, String> {
    let mut offset = 0u64;
    let mut offsets = Vec::with_capacity(fields.len());
    for field in fields {
        let size = layout_size_of_ty(tcx, *field)?;
        if size == 0 {
            offsets.push(offset);
            continue;
        }
        offset = align_to(offset, layout_align_of_ty(tcx, *field)?);
        offsets.push(offset);
        offset += size;
    }
    Ok(offsets)
}

fn struct_like_size<'tcx>(tcx: TyCtxt<'tcx>, fields: &[Ty<'tcx>]) -> Result<u64, String> {
    let offsets = struct_like_field_offsets(tcx, fields)?;
    let mut align = 1u64;
    let mut unpadded = 0u64;
    for (field, offset) in fields.iter().zip(offsets) {
        let size = layout_size_of_ty(tcx, *field)?;
        if size == 0 {
            continue;
        }
        align = align.max(layout_align_of_ty(tcx, *field)?);
        unpadded = unpadded.max(offset + size);
    }
    Ok(align_to(unpadded, align))
}

fn align_to(value: u64, align: u64) -> u64 {
    if align <= 1 {
        value
    } else {
        (value + align - 1) & !(align - 1)
    }
}

/// Alignment of the converted STAIR type for a Rust type, mirroring the
/// AArch64 lowering's `stack_align_of` rules.
fn layout_align_of_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Result<u64, String> {
    let ty = normalize_ty(tcx, runtime_ty(ty));
    match ty.kind() {
        rustc_middle::ty::TyKind::Bool => Ok(1),
        rustc_middle::ty::TyKind::Char => Ok(4),
        rustc_middle::ty::TyKind::Str | rustc_middle::ty::TyKind::Dynamic(_, _) => Ok(1),
        rustc_middle::ty::TyKind::Int(kind) => {
            Ok((int_width(*kind) as u64).div_ceil(8).min(8).max(1))
        }
        rustc_middle::ty::TyKind::Uint(kind) => {
            Ok((uint_width(*kind) as u64).div_ceil(8).min(8).max(1))
        }
        rustc_middle::ty::TyKind::Float(rustc_middle::ty::FloatTy::F32) => Ok(4),
        rustc_middle::ty::TyKind::Float(_) => Ok(8),
        rustc_middle::ty::TyKind::FnPtr(_, _)
        | rustc_middle::ty::TyKind::Ref(_, _, _)
        | rustc_middle::ty::TyKind::RawPtr(_, _) => Ok(8),
        rustc_middle::ty::TyKind::FnDef(_, _) | rustc_middle::ty::TyKind::Never => Ok(1),
        rustc_middle::ty::TyKind::Tuple(fields) => {
            let fields: Vec<_> = fields.iter().collect();
            struct_like_align(tcx, &fields)
        }
        rustc_middle::ty::TyKind::Array(elem, _) | rustc_middle::ty::TyKind::Slice(elem) => {
            layout_align_of_ty(tcx, *elem)
        }
        rustc_middle::ty::TyKind::Closure(_, args) => {
            let fields: Vec<_> = args.as_closure().upvar_tys().iter().collect();
            struct_like_align(tcx, &fields)
        }
        rustc_middle::ty::TyKind::Adt(adt_def, _)
            if is_fmt_rt_argument_type(tcx, adt_def.did()) =>
        {
            Ok(8)
        }
        rustc_middle::ty::TyKind::Adt(adt_def, args) if adt_def.is_struct() => {
            let fields: Vec<_> = adt_def
                .non_enum_variant()
                .fields
                .iter()
                .map(|field| field.ty(tcx, args))
                .collect();
            struct_like_align(tcx, &fields)
        }
        // Unions convert to a byte-array wrapper, which the lowering treats
        // as alignment 1.
        rustc_middle::ty::TyKind::Adt(adt_def, _) if adt_def.is_union() => Ok(1),
        rustc_middle::ty::TyKind::Adt(adt_def, _) if adt_def.is_enum() => enum_stack_align(tcx, ty),
        other => Err(format!(
            "unsupported Rust type alignment in MIR importer: {other:?}"
        )),
    }
}

fn struct_like_align<'tcx>(tcx: TyCtxt<'tcx>, fields: &[Ty<'tcx>]) -> Result<u64, String> {
    let mut align = 1u64;
    for field in fields {
        if layout_size_of_ty(tcx, *field)? == 0 {
            continue;
        }
        align = align.max(layout_align_of_ty(tcx, *field)?);
    }
    Ok(align)
}

fn layout_size_of_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Result<u64, String> {
    let ty = normalize_ty(tcx, runtime_ty(ty));
    match ty.kind() {
        rustc_middle::ty::TyKind::Bool => Ok(1),
        rustc_middle::ty::TyKind::Char => Ok(4),
        rustc_middle::ty::TyKind::Str => Ok(0),
        rustc_middle::ty::TyKind::Dynamic(_, _) => Ok(0),
        rustc_middle::ty::TyKind::Int(kind) => Ok((int_width(*kind) as u64).div_ceil(8)),
        rustc_middle::ty::TyKind::Uint(kind) => Ok((uint_width(*kind) as u64).div_ceil(8)),
        rustc_middle::ty::TyKind::Float(rustc_middle::ty::FloatTy::F32) => Ok(4),
        rustc_middle::ty::TyKind::Float(rustc_middle::ty::FloatTy::F64) => Ok(8),
        rustc_middle::ty::TyKind::FnPtr(_, _) => Ok(8),
        rustc_middle::ty::TyKind::FnDef(_, _) | rustc_middle::ty::TyKind::Never => Ok(0),
        rustc_middle::ty::TyKind::Ref(_, inner, _) | rustc_middle::ty::TyKind::RawPtr(inner, _) => {
            // Fat references convert to a two-word {ptr, meta} struct.
            Ok(if pointee_unsized_tail(tcx, *inner).is_some() {
                16
            } else {
                8
            })
        }
        rustc_middle::ty::TyKind::Tuple(_) => {
            let fields = memory_ordered_fields(tcx, ty)?;
            struct_like_size(tcx, &fields)
        }
        rustc_middle::ty::TyKind::Array(elem, len) => {
            let len = array_len(tcx, *len)?;
            let elem_size = layout_size_of_ty(tcx, *elem)?;
            let stride = align_to(elem_size, layout_align_of_ty(tcx, *elem)?);
            Ok(stride * len)
        }
        rustc_middle::ty::TyKind::Slice(elem) => layout_size_of_ty(tcx, *elem),
        rustc_middle::ty::TyKind::Adt(adt_def, _)
            if is_fmt_rt_argument_type(tcx, adt_def.did()) =>
        {
            Ok(16)
        }
        rustc_middle::ty::TyKind::Adt(adt_def, _) if adt_def.is_struct() => {
            let fields = memory_ordered_fields(tcx, ty)?;
            struct_like_size(tcx, &fields)
        }
        rustc_middle::ty::TyKind::Adt(adt_def, args) if adt_def.is_union() => adt_def
            .non_enum_variant()
            .fields
            .iter()
            .try_fold(0u64, |size, field| {
                Ok(size.max(layout_size_of_ty(tcx, field.ty(tcx, args))?))
            }),
        rustc_middle::ty::TyKind::Adt(adt_def, _) if adt_def.is_enum() => {
            let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
            rustc_layout_size_of_ty(tcx, typing_env, ty)
        }
        rustc_middle::ty::TyKind::Closure(_, _) => {
            let fields = memory_ordered_fields(tcx, ty)?;
            struct_like_size(tcx, &fields)
        }
        other => Err(format!(
            "unsupported Rust type layout in MIR importer: {other:?}"
        )),
    }
}

/// The struct-like fields of `ty` in converted (memory) order.
fn memory_ordered_fields<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Result<Vec<Ty<'tcx>>, String> {
    let source = struct_like_source_fields(tcx, ty)
        .ok_or_else(|| format!("expected struct-like type: {ty:?}"))?;
    let order = struct_memory_order(tcx, ty, source.len());
    Ok(order.into_iter().map(|idx| source[idx]).collect())
}

fn array_len<'tcx>(tcx: TyCtxt<'tcx>, len: rustc_middle::ty::Const<'tcx>) -> Result<u64, String> {
    if let Some(value) = len.try_to_target_usize(tcx) {
        return Ok(value);
    }
    let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
    let normalized = tcx
        .try_normalize_erasing_regions(typing_env, len)
        .map_err(|error| format!("unsupported non-constant array length: {len:?}: {error:?}"))?;
    normalized
        .try_to_target_usize(tcx)
        .ok_or_else(|| format!("unsupported non-constant array length: {len:?}"))
}

fn local_slot<'tcx>(state: &FunctionImportState<'tcx>, local: Local) -> Result<Value, String> {
    local_slot_opt(state, local)?.ok_or_else(|| format!("missing slot for MIR local {local:?}"))
}

fn local_slot_opt<'tcx>(
    state: &FunctionImportState<'tcx>,
    local: Local,
) -> Result<Option<Value>, String> {
    state
        .local_slots
        .get(local.index())
        .map(|slot| slot.as_ref().map(|(slot, _)| *slot))
        .ok_or_else(|| format!("unknown MIR local {local:?}"))
}

fn place_slot_opt<'tcx>(
    state: &FunctionImportState<'tcx>,
    place: &Place<'tcx>,
) -> Result<Option<Value>, String> {
    if !place.projection.is_empty() {
        return Err(format!(
            "unsupported MIR place projection: {:?}",
            place.projection
        ));
    }
    local_slot_opt(state, place.local)
}

fn block_for<'tcx>(
    state: &FunctionImportState<'tcx>,
    block: RustBasicBlock,
) -> Result<Ptr<BasicBlock>, String> {
    state
        .blocks
        .get(block.index())
        .copied()
        .ok_or_else(|| format!("missing STAIR block for MIR block {block:?}"))
}

fn usize_ty(ctx: &mut Context) -> TypedHandle<IntegerType> {
    IntegerType::get(ctx, 64, Signedness::Unsigned)
}

fn llvm_ptr_ty(ctx: &mut Context) -> TypeHandle {
    llvm::types::PointerType::get(ctx, 0).into()
}

fn str_ref_ty(ctx: &mut Context) -> TypeHandle {
    let ptr = llvm_ptr_ty(ctx);
    let usize: TypeHandle = usize_ty(ctx).into();
    llvm::types::StructType::get_unnamed(ctx, vec![ptr, usize]).into()
}

fn slice_ref_ty(ctx: &mut Context, _elem: TypeHandle) -> TypeHandle {
    let ptr = llvm_ptr_ty(ctx);
    let usize: TypeHandle = usize_ty(ctx).into();
    llvm::types::StructType::get_unnamed(ctx, vec![ptr, usize]).into()
}

fn trait_object_ref_ty(ctx: &mut Context) -> TypeHandle {
    let ptr = llvm_ptr_ty(ctx);
    llvm::types::StructType::get_unnamed(ctx, vec![ptr, ptr]).into()
}

fn fmt_arguments_ty(ctx: &mut Context) -> TypeHandle {
    let ptr = llvm_ptr_ty(ctx);
    llvm::types::StructType::get_unnamed(ctx, vec![ptr, ptr]).into()
}

fn device_slice_ty(ctx: &mut Context, elem: TypeHandle, mutable: bool) -> TypeHandle {
    let _ = (elem, mutable);
    let ptr: TypeHandle = llvm_ptr_ty(ctx);
    let usize: TypeHandle = usize_ty(ctx).into();
    llvm::types::StructType::get_unnamed(ctx, vec![ptr, usize]).into()
}

/// Memory-order permutation of a struct-like type's fields per the real
/// rustc layout: element `i` is the source index of the field at memory
/// position `i`. Identity when no layout is available.
fn struct_memory_order<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>, field_count: usize) -> Vec<usize> {
    let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
    let Ok(layout) = tcx.layout_of(typing_env.as_query_input(ty)) else {
        return (0..field_count).collect();
    };
    match &layout.fields {
        rustc_abi::FieldsShape::Arbitrary {
            in_memory_order, ..
        } if in_memory_order.len() == field_count => in_memory_order
            .iter()
            .map(|source| source.as_usize())
            .collect(),
        _ => (0..field_count).collect(),
    }
}

/// Position of a source-order field within the converted (memory-ordered)
/// struct type.
fn field_memory_position<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    field_count: usize,
    source_idx: usize,
) -> u32 {
    let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
    let Ok(layout) = tcx.layout_of(typing_env.as_query_input(ty)) else {
        return source_idx as u32;
    };
    match &layout.fields {
        rustc_abi::FieldsShape::Arbitrary {
            in_memory_order, ..
        } if in_memory_order.len() == field_count => in_memory_order
            .iter()
            .position(|source| source.as_usize() == source_idx)
            .map(|position| position as u32)
            .unwrap_or(source_idx as u32),
        _ => source_idx as u32,
    }
}

/// The struct-like source fields of a type, if it is field-reordered by the
/// converted representation (structs, tuples, closures — not enums).
fn struct_like_source_fields<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<Vec<Ty<'tcx>>> {
    match runtime_ty(ty).kind() {
        rustc_middle::ty::TyKind::Tuple(fields) => Some(fields.iter().collect()),
        rustc_middle::ty::TyKind::Closure(_, args) => {
            Some(args.as_closure().upvar_tys().iter().collect())
        }
        rustc_middle::ty::TyKind::Adt(adt_def, args)
            if adt_def.is_struct() && !is_fmt_rt_argument_type(tcx, adt_def.did()) =>
        {
            Some(
                adt_def
                    .non_enum_variant()
                    .fields
                    .iter()
                    .map(|field| field.ty(tcx, args))
                    .collect(),
            )
        }
        _ => None,
    }
}

fn runtime_ty<'tcx>(mut ty: Ty<'tcx>) -> Ty<'tcx> {
    while let rustc_middle::ty::TyKind::Pat(base, _) = ty.kind() {
        ty = *base;
    }
    ty
}

/// Resolve associated-type projections (e.g. `AtomicPrimitive::Storage` inside
/// std internals) that reach type conversion unnormalized via field types.
fn normalize_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
    if !ty.has_aliases() {
        return ty;
    }
    let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
    tcx.try_normalize_erasing_regions(typing_env, ty)
        .unwrap_or(ty)
}

fn is_simple_abi_scalar_ty(ctx: &Context, ty: TypeHandle) -> bool {
    let ty_ref = ty.deref(ctx);
    ty_ref.is::<IntegerType>()
        || ty_ref.is::<FP32Type>()
        || ty_ref.is::<FP64Type>()
        || ty_ref.is::<llvm::types::PointerType>()
        || ty_ref.is::<crate::dialects::dialect_mir::types::MirPtrType>()
}

fn is_str_ref_ty<'tcx>(ty: Ty<'tcx>) -> bool {
    let ty = runtime_ty(ty);
    matches!(
        ty.kind(),
        rustc_middle::ty::TyKind::Ref(_, inner, _)
            if matches!(inner.kind(), rustc_middle::ty::TyKind::Str)
    )
}

fn byte_array_ref_len<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<u64> {
    let ty = runtime_ty(ty);
    let rustc_middle::ty::TyKind::Ref(_, inner, _) = ty.kind() else {
        return None;
    };
    let rustc_middle::ty::TyKind::Array(elem, len) = runtime_ty(*inner).kind() else {
        return None;
    };
    if matches!(
        runtime_ty(*elem).kind(),
        rustc_middle::ty::TyKind::Uint(rustc_middle::ty::UintTy::U8)
    ) {
        array_len(tcx, *len).ok()
    } else {
        None
    }
}

/// For a pointer/reference pointee, the unsized tail type (`[T]`, `str`, or
/// `dyn Trait`) that determines the fat-pointer metadata, or `None` when the
/// pointee is sized (thin pointer). Peels ADT struct tails, so e.g.
/// `&PolymorphicIter<[T]>` (core's array iterator internals) is fat with a
/// slice-length metadata.
fn pointee_unsized_tail<'tcx>(tcx: TyCtxt<'tcx>, pointee: Ty<'tcx>) -> Option<Ty<'tcx>> {
    let pointee = runtime_ty(pointee);
    match pointee.kind() {
        rustc_middle::ty::TyKind::Slice(_)
        | rustc_middle::ty::TyKind::Str
        | rustc_middle::ty::TyKind::Dynamic(_, _) => Some(pointee),
        _ => {
            let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
            if pointee.is_sized(tcx, typing_env) {
                return None;
            }
            let tail = tcx.struct_tail_for_codegen(pointee, typing_env);
            match runtime_ty(tail).kind() {
                rustc_middle::ty::TyKind::Slice(_)
                | rustc_middle::ty::TyKind::Str
                | rustc_middle::ty::TyKind::Dynamic(_, _) => Some(runtime_ty(tail)),
                _ => None,
            }
        }
    }
}

fn contains_maybe_uninit_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
    let ty = runtime_ty(ty);
    if format!("{ty:?}").contains("MaybeUninit") {
        return true;
    }
    match ty.kind() {
        rustc_middle::ty::TyKind::Adt(adt_def, args) => {
            tcx.def_path_str(adt_def.did())
                .contains("mem::maybe_uninit::MaybeUninit")
                || args.iter().any(|arg| {
                    arg.as_type()
                        .is_some_and(|ty| contains_maybe_uninit_ty(tcx, ty))
                })
        }
        rustc_middle::ty::TyKind::Array(elem, _)
        | rustc_middle::ty::TyKind::Slice(elem)
        | rustc_middle::ty::TyKind::Ref(_, elem, _)
        | rustc_middle::ty::TyKind::RawPtr(elem, _) => contains_maybe_uninit_ty(tcx, *elem),
        rustc_middle::ty::TyKind::Tuple(fields) => fields
            .iter()
            .any(|field| contains_maybe_uninit_ty(tcx, field)),
        _ => false,
    }
}

fn is_maybe_uninit_u8_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
    let ty = runtime_ty(ty);
    let rustc_middle::ty::TyKind::Adt(adt_def, args) = ty.kind() else {
        return false;
    };
    if !tcx.def_path_str(adt_def.did()).contains("MaybeUninit")
        && !format!("{ty:?}").contains("MaybeUninit")
    {
        return false;
    }
    args.iter().any(|arg| {
        arg.as_type().is_some_and(|ty| {
            matches!(
                runtime_ty(ty).kind(),
                rustc_middle::ty::TyKind::Uint(rustc_middle::ty::UintTy::U8)
            )
        })
    })
}

fn is_enum_ty<'tcx>(ty: Ty<'tcx>) -> bool {
    matches!(
        runtime_ty(ty).kind(),
        rustc_middle::ty::TyKind::Adt(adt_def, _) if adt_def.is_enum()
    )
}

fn convert_return_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    ty: Ty<'tcx>,
) -> Result<Vec<TypeHandle>, String> {
    let ty = runtime_ty(ty);
    match ty.kind() {
        rustc_middle::ty::TyKind::Tuple(fields) if fields.is_empty() => Ok(Vec::new()),
        rustc_middle::ty::TyKind::Never => Ok(Vec::new()),
        rustc_middle::ty::TyKind::Bool => Ok(vec![bool_immediate_ty(ctx)]),
        _ => Ok(vec![convert_ty(tcx, ctx, ty)?]),
    }
}

fn convert_storage_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    ty: Ty<'tcx>,
) -> Result<Option<TypeHandle>, String> {
    let ty = runtime_ty(ty);
    match ty.kind() {
        rustc_middle::ty::TyKind::Tuple(fields) if fields.is_empty() => Ok(None),
        rustc_middle::ty::TyKind::Never => Ok(None),
        _ => Ok(Some(convert_ty(tcx, ctx, ty)?)),
    }
}

fn convert_immediate_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    ty: Ty<'tcx>,
) -> Result<TypeHandle, String> {
    if is_bool_ty(ty) {
        return Ok(bool_immediate_ty(ctx));
    }
    convert_ty(tcx, ctx, ty)
}

fn convert_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    ty: Ty<'tcx>,
) -> Result<TypeHandle, String> {
    use rustc_middle::ty::TyKind;

    let ty = normalize_ty(tcx, runtime_ty(ty));
    let ty: TypedHandle<IntegerType> = match ty.kind() {
        TyKind::Bool => return Ok(bool_storage_ty(ctx)),
        TyKind::Str => {
            let byte: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
            return Ok(llvm::types::ArrayType::get(ctx, byte, 0).into());
        }
        TyKind::Dynamic(_, _) => {
            let byte: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
            return Ok(llvm::types::ArrayType::get(ctx, byte, 0).into());
        }
        TyKind::Int(kind) => IntegerType::get(ctx, int_width(*kind), Signedness::Signed),
        TyKind::Uint(kind) => IntegerType::get(ctx, uint_width(*kind), Signedness::Unsigned),
        TyKind::Char => IntegerType::get(ctx, 32, Signedness::Unsigned),
        TyKind::Float(rustc_middle::ty::FloatTy::F32) => return Ok(FP32Type::get(ctx).into()),
        TyKind::Float(rustc_middle::ty::FloatTy::F64) => return Ok(FP64Type::get(ctx).into()),
        TyKind::FnPtr(_, _) => return Ok(llvm_ptr_ty(ctx)),
        TyKind::FnDef(_, _) => return Ok(unit_ty(ctx)),
        TyKind::Never => return Ok(unit_ty(ctx)),
        TyKind::Tuple(fields) if fields.is_empty() => return Ok(unit_ty(ctx)),
        TyKind::Tuple(fields) => {
            let source: Vec<_> = fields.iter().collect();
            let order = struct_memory_order(tcx, ty, source.len());
            let fields = order
                .into_iter()
                .map(|idx| convert_ty(tcx, ctx, source[idx]))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(llvm::types::StructType::get_unnamed(ctx, fields).into());
        }
        TyKind::Array(elem, len) => {
            let elem = convert_ty(tcx, ctx, *elem)?;
            let len = array_len(tcx, *len)?;
            return Ok(llvm::types::ArrayType::get(ctx, elem, len).into());
        }
        TyKind::Slice(elem) => {
            let elem = convert_ty(tcx, ctx, *elem)?;
            return Ok(llvm::types::ArrayType::get(ctx, elem, 0).into());
        }
        TyKind::Ref(_, inner, _) | TyKind::RawPtr(inner, _) => {
            return match pointee_unsized_tail(tcx, *inner) {
                Some(tail) => match runtime_ty(tail).kind() {
                    TyKind::Str => Ok(str_ref_ty(ctx)),
                    TyKind::Slice(elem) => {
                        let elem = convert_ty(tcx, ctx, *elem)?;
                        Ok(slice_ref_ty(ctx, elem))
                    }
                    TyKind::Dynamic(_, _) => Ok(trait_object_ref_ty(ctx)),
                    other => Err(format!("unsupported unsized pointee tail: {other:?}")),
                },
                None => Ok(llvm_ptr_ty(ctx)),
            };
        }
        TyKind::Adt(_, _)
            if format!("{ty:?}").starts_with("std::fmt::Arguments")
                || format!("{ty:?}").starts_with("core::fmt::Arguments") =>
        {
            return Ok(fmt_arguments_ty(ctx));
        }
        TyKind::Adt(adt_def, args) if adt_def.is_struct() => {
            let name = type_symbol(tcx, ty);
            // `get_named` with no fields returns the type if the name is
            // already registered, and reserves it (as an opaque struct)
            // otherwise; only proceed to build the body for opaque results.
            let reserved = llvm::types::StructType::get_named(ctx, name.clone(), None)
                .map_err(|error| error.to_string())?;
            if !reserved.deref(ctx).is_opaque() {
                return Ok(reserved.into());
            }
            let source: Vec<_> = adt_def
                .non_enum_variant()
                .fields
                .iter()
                .map(|field| field.ty(tcx, args))
                .collect();
            let order = struct_memory_order(tcx, ty, source.len());
            let fields = order
                .into_iter()
                .map(|idx| convert_ty(tcx, ctx, source[idx]))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(llvm::types::StructType::get_named(ctx, name, Some(fields))
                .map_err(|error| error.to_string())?
                .into());
        }
        TyKind::Adt(adt_def, args) if adt_def.is_union() => {
            let name = type_symbol(tcx, ty);
            // `get_named` with no fields returns the type if the name is
            // already registered, and reserves it (as an opaque struct)
            // otherwise; only proceed to build the body for opaque results.
            let reserved = llvm::types::StructType::get_named(ctx, name.clone(), None)
                .map_err(|error| error.to_string())?;
            if !reserved.deref(ctx).is_opaque() {
                return Ok(reserved.into());
            }
            let size = layout_size_of_ty(tcx, ty)?;
            let byte_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
            let storage: TypeHandle = if size == 0 {
                unit_ty(ctx)
            } else {
                llvm::types::ArrayType::get(ctx, byte_ty, size).into()
            };
            let _ = args;
            return Ok(
                llvm::types::StructType::get_named(ctx, name, Some(vec![storage]))
                    .map_err(|error| error.to_string())?
                    .into(),
            );
        }
        TyKind::Adt(adt_def, _) if is_fmt_rt_argument_type(tcx, adt_def.did()) => {
            // `core::fmt::rt::ArgumentType` is read by prebuilt std code, so
            // it must use the real niche layout: {value ptr, formatter fn ptr}.
            return Ok(fmt_arguments_ty(ctx));
        }
        TyKind::Adt(adt_def, _) if adt_def.is_enum() => {
            return enum_blob_ty(tcx, ctx, ty);
        }
        TyKind::Adt(_, args)
            if is_stair_device_wrapper_ty(ty, "DeviceSliceMut")
                || is_stair_device_wrapper_ty(ty, "DisjointSlice") =>
        {
            let elem = args[0].expect_ty();
            let elem = convert_ty(tcx, ctx, elem)?;
            return Ok(device_slice_ty(ctx, elem, true));
        }
        TyKind::Adt(_, args) if is_stair_device_wrapper_ty(ty, "DeviceSlice") => {
            let elem = args[0].expect_ty();
            let elem = convert_ty(tcx, ctx, elem)?;
            return Ok(device_slice_ty(ctx, elem, false));
        }
        TyKind::Closure(_, args) => {
            let name = type_symbol(tcx, ty);
            // `get_named` with no fields returns the type if the name is
            // already registered, and reserves it (as an opaque struct)
            // otherwise; only proceed to build the body for opaque results.
            let reserved = llvm::types::StructType::get_named(ctx, name.clone(), None)
                .map_err(|error| error.to_string())?;
            if !reserved.deref(ctx).is_opaque() {
                return Ok(reserved.into());
            }
            let source: Vec<_> = args.as_closure().upvar_tys().iter().collect();
            let order = struct_memory_order(tcx, ty, source.len());
            let fields = order
                .into_iter()
                .map(|idx| convert_ty(tcx, ctx, source[idx]))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(llvm::types::StructType::get_named(ctx, name, Some(fields))
                .map_err(|error| error.to_string())?
                .into());
        }
        other => return Err(format!("unsupported Rust type in MIR importer: {other:?}")),
    };
    Ok(ty.into())
}

/// The opaque, real-layout blob type for an enum: a named struct wrapping an
/// `[N x iM]` array sized and aligned to the enum's real rustc layout. Enum
/// values are simply their real bytes, so every discriminant read, variant
/// field access, and construction goes through real byte offsets (see
/// [`read_enum_discriminant`], [`field_offset_of`], and the `Rvalue::Aggregate`
/// enum path). This is what lets enums interoperate with prebuilt std code that
/// reads and writes them by reference.
fn enum_blob_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    ty: Ty<'tcx>,
) -> Result<TypeHandle, String> {
    let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
    let size = rustc_layout_size_of_ty(tcx, typing_env, ty)?;
    if size == 0 {
        return Ok(unit_ty(ctx));
    }
    let name = type_symbol(tcx, ty);
    let reserved = llvm::types::StructType::get_named(ctx, name.clone(), None)
        .map_err(|error| error.to_string())?;
    if !reserved.deref(ctx).is_opaque() {
        return Ok(reserved.into());
    }
    let align = enum_stack_align(tcx, ty)?;
    let elem: TypeHandle = IntegerType::get(ctx, (align * 8) as u32, Signedness::Unsigned).into();
    let blob: TypeHandle = llvm::types::ArrayType::get(ctx, elem, size / align).into();
    Ok(
        llvm::types::StructType::get_named(ctx, name, Some(vec![blob]))
            .map_err(|error| error.to_string())?
            .into(),
    )
}

/// The backend stack alignment (capped at the 8-byte maximum the AArch64
/// lowering supports) of an enum's real rustc layout.
fn enum_stack_align<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Result<u64, String> {
    let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
    Ok(rustc_layout_align_of_ty(tcx, typing_env, ty)?.clamp(1, 8))
}

/// Read an enum's discriminant from its real in-memory layout at `addr`,
/// producing a value of the converted discriminant type. Mirrors what rustc's
/// own codegen emits for `Rvalue::Discriminant`: a direct tag load for `Direct`
/// encoding, and the niche-decode arithmetic for `Niche` encoding.
fn read_enum_discriminant<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    enum_ty: Ty<'tcx>,
    addr: Value,
) -> Result<Value, String> {
    let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
    let enum_ty = normalize_ty(tcx, runtime_ty(enum_ty));
    let rustc_middle::ty::TyKind::Adt(adt_def, _) = enum_ty.kind() else {
        return Err(format!("MIR discriminant of non-ADT type: {enum_ty:?}"));
    };
    let layout = tcx
        .layout_of(typing_env.as_query_input(enum_ty))
        .map_err(|error| format!("no layout for {enum_ty:?}: {error:?}"))?;
    let discr_ty = convert_ty(tcx, ctx, enum_ty.discriminant_ty(tcx))?;
    let discr_width = discr_ty
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .map(|ty| ty.width())
        .ok_or_else(|| "enum discriminant is not an integer".to_string())?;
    let discr_mask = if discr_width >= 128 {
        u128::MAX
    } else {
        (1u128 << discr_width) - 1
    };

    match &layout.variants {
        Variants::Empty => {
            // An uninhabited enum (e.g. `Result<Infallible, !>`) has no
            // values, so this read can never execute. Mirror rustc's cg_ssa,
            // which emits an arbitrary (poison) value: a zero constant of the
            // discriminant type, matching the non-enum branch of
            // `Rvalue::Discriminant`.
            let op = integer_constant(ctx, discr_ty, 0)?;
            op.get_operation().insert_at_back(insert_block, ctx);
            Ok(op.get_result(ctx))
        }
        Variants::Single { index } => {
            let value = adt_def.discriminant_for_variant(tcx, *index).val & discr_mask;
            let op = integer_constant(ctx, discr_ty, value)?;
            op.get_operation().insert_at_back(insert_block, ctx);
            Ok(op.get_result(ctx))
        }
        Variants::Multiple {
            tag,
            tag_encoding,
            tag_field,
            ..
        } => {
            let tag_offset = layout.fields.offset(tag_field.as_usize()).bytes();
            let tag_bits = tag.size(&tcx).bits() as u32;
            let tag_ty: TypeHandle = IntegerType::get(ctx, tag_bits, Signedness::Unsigned).into();
            let tag_addr = if tag_offset != 0 {
                ptr_offset_const(ctx, insert_block, addr, tag_offset)?
            } else {
                addr
            };
            let load = stair_mir::ops::LoadOp::new(ctx, tag_addr, tag_ty);
            load.get_operation().insert_at_back(insert_block, ctx);
            let tag_value = load.get_result(ctx);
            match tag_encoding {
                TagEncoding::Direct => {
                    Ok(cast_value_to_type(ctx, insert_block, tag_value, discr_ty))
                }
                TagEncoding::Niche {
                    untagged_variant,
                    niche_variants,
                    niche_start,
                } => {
                    let tag_mask = if tag_bits >= 128 {
                        u128::MAX
                    } else {
                        (1u128 << tag_bits) - 1
                    };
                    let relative_max =
                        niche_variants.end().as_u32() - niche_variants.start().as_u32();

                    // relative_tag = tag - niche_start, wrapping in the tag width
                    // so the comparison below matches rustc's niche decoding.
                    let niche_start_const = integer_constant(ctx, tag_ty, niche_start & tag_mask)?;
                    niche_start_const
                        .get_operation()
                        .insert_at_back(insert_block, ctx);
                    let relative_tag = stair_mir::ops::SubOp::new(
                        ctx,
                        tag_value,
                        niche_start_const.get_result(ctx),
                    )
                    .get_operation();
                    relative_tag.insert_at_back(insert_block, ctx);
                    let relative_tag = relative_tag.deref(ctx).get_result(0);

                    // is_niche = relative_tag <= relative_max (unsigned).
                    let relative_max_const =
                        integer_constant(ctx, tag_ty, relative_max as u128 & tag_mask)?;
                    relative_max_const
                        .get_operation()
                        .insert_at_back(insert_block, ctx);
                    let is_niche = stair_mir::ops::LeOp::new(
                        ctx,
                        relative_tag,
                        relative_max_const.get_result(ctx),
                    );
                    is_niche.get_operation().insert_at_back(insert_block, ctx);
                    let is_niche =
                        cast_value_to_type(ctx, insert_block, is_niche.get_result(ctx), discr_ty);

                    // For niche encoding the discriminant equals the variant
                    // index, so the tagged discriminant is
                    // `niche_variants.start() + relative_tag`.
                    let relative_discr =
                        cast_value_to_type(ctx, insert_block, relative_tag, discr_ty);
                    let niche_base = integer_constant(
                        ctx,
                        discr_ty,
                        niche_variants.start().as_u32() as u128 & discr_mask,
                    )?;
                    niche_base.get_operation().insert_at_back(insert_block, ctx);
                    let tagged =
                        stair_mir::ops::AddOp::new(ctx, niche_base.get_result(ctx), relative_discr)
                            .get_operation();
                    tagged.insert_at_back(insert_block, ctx);
                    let tagged = tagged.deref(ctx).get_result(0);

                    // discr = untagged + (tagged - untagged) * is_niche.
                    let untagged = adt_def.discriminant_for_variant(tcx, *untagged_variant).val;
                    let base = integer_constant(ctx, discr_ty, untagged & discr_mask)?;
                    base.get_operation().insert_at_back(insert_block, ctx);
                    let base = base.get_result(ctx);
                    let diff = stair_mir::ops::SubOp::new(ctx, tagged, base).get_operation();
                    diff.insert_at_back(insert_block, ctx);
                    let diff = diff.deref(ctx).get_result(0);
                    let scaled = stair_mir::ops::MulOp::new(ctx, diff, is_niche).get_operation();
                    scaled.insert_at_back(insert_block, ctx);
                    let scaled = scaled.deref(ctx).get_result(0);
                    let discr = stair_mir::ops::AddOp::new(ctx, base, scaled).get_operation();
                    discr.insert_at_back(insert_block, ctx);
                    Ok(discr.deref(ctx).get_result(0))
                }
            }
        }
    }
}

/// Write the tag for `variant` into an enum stored with its real layout at
/// `addr`. A no-op when the variant needs no tag (a single-variant enum or the
/// untagged niche variant), mirroring rustc's `tag_for_variant`.
fn write_enum_tag<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    insert_block: Ptr<BasicBlock>,
    enum_ty: Ty<'tcx>,
    variant: VariantIdx,
    addr: Value,
) -> Result<(), String> {
    let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
    let enum_ty = normalize_ty(tcx, runtime_ty(enum_ty));
    let layout = tcx
        .layout_of(typing_env.as_query_input(enum_ty))
        .map_err(|error| format!("no layout for {enum_ty:?}: {error:?}"))?;
    let Variants::Multiple { tag, tag_field, .. } = &layout.variants else {
        return Ok(());
    };
    let Some(tag_value) = tcx.tag_for_variant(typing_env.as_query_input((enum_ty, variant))) else {
        return Ok(());
    };
    let tag_bits = tag.size(&tcx).bits() as u32;
    let tag_ty: TypeHandle = IntegerType::get(ctx, tag_bits, Signedness::Unsigned).into();
    let bits = tag_value.to_uint(tag_value.size());
    let tag_const = integer_constant(ctx, tag_ty, bits)?;
    tag_const.get_operation().insert_at_back(insert_block, ctx);
    let tag_offset = layout.fields.offset(tag_field.as_usize()).bytes();
    let tag_addr = if tag_offset != 0 {
        ptr_offset_const(ctx, insert_block, addr, tag_offset)?
    } else {
        addr
    };
    let store = stair_mir::ops::StoreOp::new(ctx, tag_const.get_result(ctx), tag_addr);
    store.get_operation().insert_at_back(insert_block, ctx);
    Ok(())
}

fn type_symbol<'tcx>(_tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> crate::identifier::Identifier {
    // `{ty:?}` prints paths of local items without the crate name, so e.g.
    // arrow-buffer's own `bytes::Bytes` and the external `bytes` crate's
    // `Bytes` would print identically and alias one converted struct (with
    // whichever layout was converted first). `with_resolve_crate_name`
    // prefixes every path (local ones included) with the real crate name,
    // in generic-argument position too, keeping the symbols distinct.
    let printed = rustc_middle::ty::print::with_resolve_crate_name!(format!("{ty:?}"));
    let mut legaliser = Legaliser::default();
    legaliser.legalise(&printed)
}

fn is_fmt_rt_argument_type<'tcx>(tcx: TyCtxt<'tcx>, def_id: rustc_hir::def_id::DefId) -> bool {
    tcx.def_path_str(def_id).contains("fmt::rt::ArgumentType")
}

fn is_stair_device_wrapper_ty<'tcx>(ty: Ty<'tcx>, name: &str) -> bool {
    let ty = format!("{ty:?}");
    ty.starts_with(&format!("stair_device::{name}"))
        || ty.starts_with(&format!("stair_device::slice::{name}"))
}

fn int_width(kind: rustc_middle::ty::IntTy) -> u32 {
    use rustc_middle::ty::IntTy;
    match kind {
        IntTy::I8 => 8,
        IntTy::I16 => 16,
        IntTy::I32 => 32,
        IntTy::I64 => 64,
        IntTy::I128 => 128,
        IntTy::Isize => 64,
    }
}

fn uint_width(kind: rustc_middle::ty::UintTy) -> u32 {
    use rustc_middle::ty::UintTy;
    match kind {
        UintTy::U8 => 8,
        UintTy::U16 => 16,
        UintTy::U32 => 32,
        UintTy::U64 => 64,
        UintTy::U128 => 128,
        UintTy::Usize => 64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_field_collection_flattens_simple_structs() {
        let mut ctx = create_context();
        let ptr_ty: TypeHandle = llvm::types::PointerType::get(&mut ctx, 0).into();
        let i32_ty: TypeHandle = IntegerType::get(&mut ctx, 32, Signedness::Signed).into();
        let i64_ty: TypeHandle = IntegerType::get(&mut ctx, 64, Signedness::Unsigned).into();
        let nested_ty: TypeHandle =
            llvm::types::StructType::get_unnamed(&mut ctx, vec![i32_ty, ptr_ty]).into();
        let aggregate_ty: TypeHandle =
            llvm::types::StructType::get_unnamed(&mut ctx, vec![ptr_ty, i64_ty, nested_ty]).into();

        let mut fields = Vec::new();
        collect_simple_abi_fields(&ctx, aggregate_ty, Vec::new(), &mut fields).unwrap();

        let indices = fields
            .iter()
            .map(|(indices, _)| indices.clone())
            .collect::<Vec<_>>();
        assert_eq!(indices, vec![vec![0], vec![1], vec![2, 0], vec![2, 1]]);
        assert_eq!(fields[0].1, ptr_ty);
        assert_eq!(fields[1].1, i64_ty);
        assert_eq!(fields[2].1, i32_ty);
        assert_eq!(fields[3].1, ptr_ty);
    }
}

/// Constructor-compatible shims over cuda-oxide's `dialect-mir` ops (plus raw
/// [pliron_llvm] ops for aggregate values and symbol addresses), mirroring the
/// old `cmir` op constructors so the importer body above stays line-compatible
/// with `importer.rs` while emitting the new dialect.
///
/// Conventions established by the port:
/// - Pointers are `MirPtrType` in the generic (0) address space; pointees are
///   advisory (the sibling branch relaxes dialect-mir to opaque-pointer mode).
/// - `ptr_offset` keeps the old byte-addressing semantics via the
///   `ptr_offset_unit = "byte"` attribute understood by `mir-lower`.
/// - Aggregate values (undef/extract/insert) and symbol addresses stay in the
///   llvm dialect: `mir-lower` passes llvm ops through untouched and its type
///   converter accepts llvm aggregate types.
mod ox {
    use crate::attribute::AttrObj;
    use crate::context::{Context, Ptr};
    use crate::dialects::builtin::attributes::{
        FPDoubleAttr, FPSingleAttr, IntegerAttr, StringAttr, TypeAttr,
    };
    use crate::dialects::builtin::op_interfaces::{
        CallOpCallable, OperandSegmentInterface, SymbolOpInterface,
    };
    use crate::dialects::builtin::type_interfaces::FunctionTypeInterface;
    use crate::dialects::builtin::types::{FP32Type, FP64Type, FunctionType, IntegerType, Signedness};
    use crate::dialects::dialect_mir::attributes::MirCastKindAttr;
    use crate::dialects::dialect_mir::ops as dm_ops;
    use crate::dialects::dialect_mir::types as dm_types;
    use crate::dialects::llvm as pllvm;
    use crate::dialects::llvm::op_interfaces::BinArithOp as _;
    use crate::dialects::llvm::op_interfaces::CastOpInterface as _;
    use crate::identifier::Identifier;
    use crate::ir::basic_block::BasicBlock;
    use crate::ir::operation::Operation;
    use crate::ir::r#type::{TypeHandle, Typed, TypedHandle};
    use crate::ir::value::Value;
    use crate::linked_list::ContainsLinkedList;
    use crate::op::Op;
    use crate::region::Region;

    pub fn byte_ptr_unit_key() -> Identifier {
        "ptr_offset_unit".try_into().unwrap()
    }

    pub fn func_linkage_key() -> Identifier {
        "mir_func_linkage".try_into().unwrap()
    }

    pub mod types {
        use super::*;

        /// Old `cmir.ptr` constructor surface over [dm_types::MirPtrType].
        pub struct PtrType;

        impl PtrType {
            pub fn get(
                ctx: &mut Context,
                elem: TypeHandle,
                mutable: bool,
            ) -> TypedHandle<dm_types::MirPtrType> {
                dm_types::MirPtrType::get(
                    ctx,
                    elem,
                    mutable,
                    dm_types::address_space::GENERIC,
                )
            }
        }
    }

    macro_rules! shim_common {
        ($Shim:ident) => {
            impl $Shim {
                pub fn get_operation(&self) -> Ptr<Operation> {
                    self.op
                }

                #[allow(dead_code)]
                pub fn get_result(&self, ctx: &Context) -> Value {
                    self.op.deref(ctx).get_result(0)
                }
            }
        };
    }

    /// Whether a value's static type is a pointer (either dialect).
    /// dialect-mir arithmetic/compare lowering resolves signedness from
    /// integer operand types and rejects raw llvm pointers, so pointer
    /// operands take the llvm twin op instead (the old pipeline's exact
    /// output for pointer arithmetic).
    fn is_ptr_value(ctx: &Context, value: Value) -> bool {
        let ty = value.get_type(ctx);
        let ty_ref = ty.deref(ctx);
        ty_ref.is::<dm_types::MirPtrType>() || ty_ref.is::<pllvm::types::PointerType>()
    }

    macro_rules! shim_binop {
        ($Shim:ident, $Theirs:ident, $LlvmTwin:ident) => {
            pub struct $Shim {
                op: Ptr<Operation>,
            }

            impl $Shim {
                pub fn new(ctx: &mut Context, lhs: Value, rhs: Value) -> Self {
                    if super::is_ptr_value(ctx, lhs) || super::is_ptr_value(ctx, rhs) {
                        let twin = pllvm::ops::$LlvmTwin::new(ctx, lhs, rhs);
                        return Self {
                            op: twin.get_operation(),
                        };
                    }
                    let ty = lhs.get_type(ctx);
                    let op = Operation::new(
                        ctx,
                        dm_ops::$Theirs::get_concrete_op_info(),
                        vec![ty],
                        vec![lhs, rhs],
                        vec![],
                        0,
                    );
                    Self { op }
                }
            }
            shim_common!($Shim);
        };
    }

    macro_rules! shim_cmpop {
        ($Shim:ident, $Theirs:ident, $PtrPred:ident) => {
            pub struct $Shim {
                op: Ptr<Operation>,
            }

            impl $Shim {
                pub fn new(ctx: &mut Context, lhs: Value, rhs: Value) -> Self {
                    if super::is_ptr_value(ctx, lhs) || super::is_ptr_value(ctx, rhs) {
                        // Pointers compare unsigned.
                        let icmp = pllvm::ops::ICmpOp::new(
                            ctx,
                            pllvm::attributes::ICmpPredicateAttr::$PtrPred,
                            lhs,
                            rhs,
                        );
                        return Self {
                            op: icmp.get_operation(),
                        };
                    }
                    let i1: TypeHandle =
                        IntegerType::get(ctx, 1, Signedness::Signless).into();
                    let op = Operation::new(
                        ctx,
                        dm_ops::$Theirs::get_concrete_op_info(),
                        vec![i1],
                        vec![lhs, rhs],
                        vec![],
                        0,
                    );
                    Self { op }
                }
            }
            shim_common!($Shim);
        };
    }

    pub mod ops {
        use super::*;

        pub struct FuncOp {
            op: Ptr<Operation>,
        }

        impl FuncOp {
            pub fn new(
                ctx: &mut Context,
                name: Identifier,
                ty: TypedHandle<FunctionType>,
            ) -> Self {
                let op = Operation::new(
                    ctx,
                    dm_ops::MirFuncOp::get_concrete_op_info(),
                    vec![],
                    vec![],
                    vec![],
                    1,
                );
                let func = dm_ops::MirFuncOp::new(ctx, op, TypeAttr::new(ty.into()));
                func.set_symbol_name(ctx, name);
                let arg_types = ty.deref(ctx).arg_types();
                let region = op.deref(ctx).get_region(0);
                let entry =
                    BasicBlock::new(ctx, Some("entry".try_into().unwrap()), arg_types);
                entry.insert_at_front(region, ctx);
                FuncOp { op }
            }

            pub fn get_entry_block(&self, ctx: &Context) -> Ptr<BasicBlock> {
                self.op
                    .deref(ctx)
                    .get_region(0)
                    .deref(ctx)
                    .get_head()
                    .expect("mir.func must have an entry block")
            }

            pub fn get_region(&self, ctx: &Context) -> Ptr<Region> {
                self.op.deref(ctx).get_region(0)
            }
        }
        shim_common!(FuncOp);

        pub struct ConstantOp {
            op: Ptr<Operation>,
        }

        impl ConstantOp {
            pub fn new(ctx: &mut Context, attr: AttrObj) -> Self {
                if let Some(int_attr) = (*attr).downcast_ref::<IntegerAttr>() {
                    return Self::new_integer(ctx, int_attr.clone());
                }
                if let Some(fp32) = (*attr).downcast_ref::<FPSingleAttr>() {
                    let fp32 = fp32.clone();
                    let ty: TypeHandle = FP32Type::get(ctx).into();
                    let op = Operation::new(
                        ctx,
                        dm_ops::MirFloatConstantOp::get_concrete_op_info(),
                        vec![ty],
                        vec![],
                        vec![],
                        0,
                    );
                    dm_ops::MirFloatConstantOp::new(op)
                        .set_attr_float_value(ctx, fp32);
                    return ConstantOp { op };
                }
                if let Some(fp64) = (*attr).downcast_ref::<FPDoubleAttr>() {
                    let fp64 = fp64.clone();
                    let ty: TypeHandle = FP64Type::get(ctx).into();
                    let op = Operation::new(
                        ctx,
                        dm_ops::MirFloatConstantOp::get_concrete_op_info(),
                        vec![ty],
                        vec![],
                        vec![],
                        0,
                    );
                    dm_ops::MirFloatConstantOp::new(op)
                        .set_attr_float_value_f64(ctx, fp64);
                    return ConstantOp { op };
                }
                panic!("oxide constant shim: unsupported constant attribute kind");
            }

            pub fn new_integer(ctx: &mut Context, attr: IntegerAttr) -> Self {
                let ty: TypeHandle = attr.get_type().into();
                let op = Operation::new(
                    ctx,
                    dm_ops::MirConstantOp::get_concrete_op_info(),
                    vec![ty],
                    vec![],
                    vec![],
                    0,
                );
                dm_ops::MirConstantOp::new(op).set_attr_value(ctx, attr);
                ConstantOp { op }
            }
        }
        shim_common!(ConstantOp);

        pub struct UndefOp {
            op: Ptr<Operation>,
        }

        impl UndefOp {
            pub fn new(ctx: &mut Context, result_ty: TypeHandle) -> Self {
                let undef = pllvm::ops::UndefOp::new(ctx, result_ty);
                UndefOp {
                    op: undef.get_operation(),
                }
            }
        }
        shim_common!(UndefOp);

        pub struct AddressOfOp {
            op: Ptr<Operation>,
        }

        impl AddressOfOp {
            pub fn new(
                ctx: &mut Context,
                symbol: Identifier,
                _result_ty: TypeHandle,
            ) -> Self {
                let addr = pllvm::ops::AddressOfOp::new(ctx, symbol, 0);
                AddressOfOp {
                    op: addr.get_operation(),
                }
            }
        }
        shim_common!(AddressOfOp);

        pub struct AllocaOp {
            op: Ptr<Operation>,
        }

        impl AllocaOp {
            pub fn new(ctx: &mut Context, elem_type: TypeHandle) -> Self {
                let ptr_ty: TypeHandle =
                    super::types::PtrType::get(ctx, elem_type, true).into();
                let op = Operation::new(
                    ctx,
                    dm_ops::MirAllocaOp::get_concrete_op_info(),
                    vec![ptr_ty],
                    vec![],
                    vec![],
                    0,
                );
                AllocaOp { op }
            }
        }
        shim_common!(AllocaOp);

        pub struct LoadOp {
            op: Ptr<Operation>,
        }

        impl LoadOp {
            pub fn new(ctx: &mut Context, addr: Value, result_type: TypeHandle) -> Self {
                let op = Operation::new(
                    ctx,
                    dm_ops::MirLoadOp::get_concrete_op_info(),
                    vec![result_type],
                    vec![addr],
                    vec![],
                    0,
                );
                LoadOp { op }
            }
        }
        shim_common!(LoadOp);

        pub struct StoreOp {
            op: Ptr<Operation>,
        }

        impl StoreOp {
            /// Old cmir operand order is `(value, addr)`; dialect-mir stores
            /// take `[ptr, value]`.
            pub fn new(ctx: &mut Context, value: Value, addr: Value) -> Self {
                let op = Operation::new(
                    ctx,
                    dm_ops::MirStoreOp::get_concrete_op_info(),
                    vec![],
                    vec![addr, value],
                    vec![],
                    0,
                );
                StoreOp { op }
            }
        }
        shim_common!(StoreOp);

        pub struct PtrOffsetOp {
            op: Ptr<Operation>,
        }

        impl PtrOffsetOp {
            /// Byte-addressed pointer arithmetic. Emitted directly as an
            /// `llvm.getelementptr` over `i8` (the exact op both the old and
            /// the dialect-mir lowering would produce for byte offsets):
            /// upstream `mir.ptr_offset` is element-scaled, which is not
            /// this op's contract.
            pub fn new(ctx: &mut Context, base: Value, byte_offset: Value) -> Self {
                let i8_ty: TypeHandle =
                    IntegerType::get(ctx, 8, Signedness::Unsigned).into();
                let gep = pllvm::ops::GetElementPtrOp::new(
                    ctx,
                    base,
                    vec![pllvm::ops::GepIndex::Value(byte_offset)],
                    i8_ty,
                );
                PtrOffsetOp {
                    op: gep.get_operation(),
                }
            }
        }
        shim_common!(PtrOffsetOp);

        pub struct CastOp {
            op: Ptr<Operation>,
        }

        impl CastOp {
            pub fn new(ctx: &mut Context, input: Value, result_type: TypeHandle) -> Self {
                let input_ty = input.get_type(ctx);
                let Some(kind) = cast_kind(ctx, input_ty, result_type) else {
                    // Aggregate-involving casts keep the old cmir semantics: a
                    // lax llvm.bitcast (their mir.cast Transmute size-checks).
                    let bitcast = pllvm::ops::BitcastOp::new(ctx, input, result_type);
                    return CastOp {
                        op: bitcast.get_operation(),
                    };
                };
                let op = Operation::new(
                    ctx,
                    dm_ops::MirCastOp::get_concrete_op_info(),
                    vec![result_type],
                    vec![input],
                    vec![],
                    0,
                );
                dm_ops::MirCastOp::new(op).set_attr_cast_kind(ctx, kind);
                CastOp { op }
            }
        }
        shim_common!(CastOp);

        pub struct ExtractValueOp {
            op: Ptr<Operation>,
        }

        impl ExtractValueOp {
            pub fn new(
                ctx: &mut Context,
                aggregate: Value,
                indices: Vec<u32>,
                _result_type: TypeHandle,
            ) -> Self {
                let extract = pllvm::ops::ExtractValueOp::new(ctx, aggregate, indices)
                    .expect("oxide extractvalue shim: invalid indices for aggregate");
                ExtractValueOp {
                    op: extract.get_operation(),
                }
            }
        }
        shim_common!(ExtractValueOp);

        pub struct InsertValueOp {
            op: Ptr<Operation>,
        }

        impl InsertValueOp {
            /// Old cmir operand order is `(value, aggregate, ..)`; the llvm op
            /// takes `(aggregate, value, ..)`.
            pub fn new(
                ctx: &mut Context,
                value: Value,
                aggregate: Value,
                indices: Vec<u32>,
            ) -> Self {
                let insert = pllvm::ops::InsertValueOp::new(ctx, aggregate, value, indices);
                InsertValueOp {
                    op: insert.get_operation(),
                }
            }
        }
        shim_common!(InsertValueOp);

        pub struct ReturnOp {
            op: Ptr<Operation>,
        }

        impl ReturnOp {
            pub fn new(ctx: &mut Context, retval: Option<Value>) -> Self {
                let op = Operation::new(
                    ctx,
                    dm_ops::MirReturnOp::get_concrete_op_info(),
                    vec![],
                    retval.into_iter().collect(),
                    vec![],
                    0,
                );
                ReturnOp { op }
            }
        }
        shim_common!(ReturnOp);

        pub struct GotoOp {
            op: Ptr<Operation>,
        }

        impl GotoOp {
            pub fn new(
                ctx: &mut Context,
                dest: Ptr<BasicBlock>,
                dest_operands: Vec<Value>,
            ) -> Self {
                let op = Operation::new(
                    ctx,
                    dm_ops::MirGotoOp::get_concrete_op_info(),
                    vec![],
                    dest_operands,
                    vec![dest],
                    0,
                );
                GotoOp { op }
            }
        }
        shim_common!(GotoOp);

        pub struct CondBrOp {
            op: Ptr<Operation>,
        }

        impl CondBrOp {
            pub fn new(
                ctx: &mut Context,
                condition: Value,
                true_dest: Ptr<BasicBlock>,
                true_operands: Vec<Value>,
                false_dest: Ptr<BasicBlock>,
                false_operands: Vec<Value>,
            ) -> Self {
                let (operands, sizes) = dm_ops::MirCondBranchOp::compute_segment_sizes(
                    vec![vec![condition], true_operands, false_operands],
                );
                let op = Operation::new(
                    ctx,
                    dm_ops::MirCondBranchOp::get_concrete_op_info(),
                    vec![],
                    operands,
                    vec![true_dest, false_dest],
                    0,
                );
                dm_ops::MirCondBranchOp::new(op).set_operand_segment_sizes(ctx, sizes);
                CondBrOp { op }
            }
        }
        shim_common!(CondBrOp);

        pub struct UnreachableOp {
            op: Ptr<Operation>,
        }

        impl UnreachableOp {
            pub fn new(ctx: &mut Context) -> Self {
                let unreachable = dm_ops::MirUnreachableOp::new(ctx);
                UnreachableOp {
                    op: unreachable.get_operation(),
                }
            }
        }
        shim_common!(UnreachableOp);

        pub struct CallOp {
            op: Ptr<Operation>,
        }

        impl CallOp {
            pub fn new_direct(
                ctx: &mut Context,
                callee: Identifier,
                args: Vec<Value>,
                result_type: Option<TypeHandle>,
            ) -> Self {
                let op = Operation::new(
                    ctx,
                    dm_ops::MirCallOp::get_concrete_op_info(),
                    result_type.into_iter().collect(),
                    args,
                    vec![],
                    0,
                );
                dm_ops::MirCallOp::new(op)
                    .set_attr_callee(ctx, StringAttr::new(callee.to_string()));
                CallOp { op }
            }

            /// Indirect call through a function-pointer value. dialect-mir
            /// has no indirect `mir.call` form, so this is emitted directly
            /// as an indirect `llvm.call` (which mir-lower passes through,
            /// remapping operands as their defining ops convert).
            pub fn new_indirect(
                ctx: &mut Context,
                callee: Value,
                args: Vec<Value>,
                result_type: Option<TypeHandle>,
            ) -> Self {
                // The FuncType is snapshotted now, before dialect
                // conversion rewrites values: any dialect-mir pointer type
                // must be recorded as an opaque llvm pointer or the isel
                // later reads foreign types out of the call's ABI signature.
                let sanitize = |ctx: &mut Context, ty: TypeHandle| -> TypeHandle {
                    if ty.deref(ctx).is::<dm_types::MirPtrType>() {
                        pllvm::types::PointerType::get(ctx, 0).into()
                    } else {
                        ty
                    }
                };
                let result_ty = result_type
                    .unwrap_or_else(|| pllvm::types::VoidType::get(ctx).into());
                let result_ty = sanitize(ctx, result_ty);
                let arg_tys: Vec<TypeHandle> = args
                    .iter()
                    .map(|a| a.get_type(ctx))
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(|t| sanitize(ctx, t))
                    .collect();
                let func_ty = pllvm::types::FuncType::get(ctx, result_ty, arg_tys, false);
                let call = pllvm::ops::CallOp::new(
                    ctx,
                    CallOpCallable::Indirect(callee),
                    func_ty,
                    args,
                );
                CallOp {
                    op: call.get_operation(),
                }
            }
        }
        shim_common!(CallOp);

        shim_binop!(AddOp, MirAddOp, AddOp);
        shim_binop!(SubOp, MirSubOp, SubOp);
        shim_binop!(MulOp, MirMulOp, MulOp);
        shim_binop!(DivOp, MirDivOp, UDivOp);
        shim_binop!(RemOp, MirRemOp, URemOp);
        shim_binop!(BitAndOp, MirBitAndOp, AndOp);
        shim_binop!(BitOrOp, MirBitOrOp, OrOp);
        shim_binop!(BitXorOp, MirBitXorOp, XorOp);
        shim_binop!(ShlOp, MirShlOp, ShlOp);

        /// The importer's own synthesized shifts (intrinsic expansions,
        /// overflow decompositions) are always logical, matching the
        /// unsigned bit-pattern math they implement.
        pub struct ShrOp {
            op: Ptr<Operation>,
        }

        impl ShrOp {
            pub fn new(ctx: &mut Context, lhs: Value, rhs: Value) -> Self {
                let lshr = pllvm::ops::LShrOp::new(ctx, lhs, rhs);
                ShrOp {
                    op: lshr.get_operation(),
                }
            }
        }
        shim_common!(ShrOp);

        // Rust-level `>>` goes through dialect-mir's `mir.shr`, which
        // mir-lower turns into `llvm.ashr` for signed operands and
        // `llvm.lshr` for unsigned ones — the aarch64 isel lowers both
        // (asr since the FP round; x86_64 still rejects `ashr` with a
        // precise error).
        shim_binop!(SignAwareShrOp, MirShrOp, LShrOp);

        pub struct NegOp {
            op: Ptr<Operation>,
        }

        impl NegOp {
            pub fn new(ctx: &mut Context, input: Value) -> Self {
                let ty = input.get_type(ctx);
                let op = Operation::new(
                    ctx,
                    dm_ops::MirNegOp::get_concrete_op_info(),
                    vec![ty],
                    vec![input],
                    vec![],
                    0,
                );
                NegOp { op }
            }
        }
        shim_common!(NegOp);

        shim_cmpop!(EqOp, MirEqOp, EQ);
        shim_cmpop!(NeOp, MirNeOp, NE);
        shim_cmpop!(LtOp, MirLtOp, ULT);
        shim_cmpop!(LeOp, MirLeOp, ULE);
        shim_cmpop!(GtOp, MirGtOp, UGT);
        shim_cmpop!(GeOp, MirGeOp, UGE);
    }

    /// Pick the dialect-mir cast kind for a `(input, result)` type pair, the
    /// way the old `cmir.cast` conversion classified casts structurally.
    fn cast_kind(
        ctx: &Context,
        input: TypeHandle,
        result: TypeHandle,
    ) -> Option<MirCastKindAttr> {
        #[derive(PartialEq)]
        enum Kind {
            Int,
            Float,
            Pointer,
            Other,
        }
        let classify = |ty: TypeHandle| -> Kind {
            let ty_ref = ty.deref(ctx);
            if ty_ref.is::<IntegerType>() {
                Kind::Int
            } else if ty_ref.is::<crate::dialects::builtin::types::FP32Type>()
                || ty_ref.is::<crate::dialects::builtin::types::FP64Type>()
            {
                Kind::Float
            } else if ty_ref.is::<dm_types::MirPtrType>()
                || ty_ref.is::<pllvm::types::PointerType>()
            {
                Kind::Pointer
            } else {
                Kind::Other
            }
        };
        match (classify(input), classify(result)) {
            (Kind::Int, Kind::Int) => Some(MirCastKindAttr::IntToInt),
            (Kind::Int, Kind::Float) => Some(MirCastKindAttr::IntToFloat),
            (Kind::Float, Kind::Int) => Some(MirCastKindAttr::FloatToInt),
            (Kind::Float, Kind::Float) => Some(MirCastKindAttr::FloatToFloat),
            (Kind::Pointer, Kind::Pointer) => Some(MirCastKindAttr::PtrToPtr),
            (Kind::Pointer, Kind::Int) => Some(MirCastKindAttr::PointerExposeAddress),
            (Kind::Int, Kind::Pointer) => Some(MirCastKindAttr::PointerWithExposedProvenance),
            _ => None,
        }
    }
}
