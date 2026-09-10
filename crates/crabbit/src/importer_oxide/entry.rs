use super::*;

/// The zero-sized "unit" representation: an llvm-native `[0 x u8]`.
/// dialect-mir's type converter recurses into aggregate fields and only
/// knows llvm/builtin scalar and Mir types, so Rust `()`/`!`/fn-item types
/// must not surface as `builtin.unit` anywhere a mir op's operand or result
/// type can reach the lowering.
pub(super) fn unit_ty(ctx: &mut Context) -> TypeHandle {
    let u8_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
    llvm::types::ArrayType::get(ctx, u8_ty, 0).into()
}

/// Whether a converted type is the zero-sized unit representation (or a
/// legacy `builtin.unit`, matched defensively).
pub(super) fn is_unit_converted_ty(ctx: &Context, ty: TypeHandle) -> bool {
    let ty_ref = ty.deref(ctx);
    if ty_ref.is::<UnitType>() {
        return true;
    }
    ty_ref
        .downcast_ref::<llvm::types::ArrayType>()
        .is_some_and(|array| array.size() == 0)
}

/// Create a crabbit context with all dialects needed by the Rust MIR path.
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

pub(super) fn is_codegen_body(tcx: TyCtxt<'_>, def_id: rustc_hir::def_id::DefId) -> bool {
    matches!(
        tcx.def_kind(def_id),
        rustc_hir::def::DefKind::Fn
            | rustc_hir::def::DefKind::AssocFn
            | rustc_hir::def::DefKind::Ctor(_, _)
    )
}

pub(super) fn is_kernel_def_id<'tcx>(tcx: TyCtxt<'tcx>, def_id: rustc_hir::def_id::DefId) -> bool {
    kernel_export_symbol(tcx, def_id).is_some()
}

/// The exported (unmangled) symbol of `def_id` when it names a kernel:
/// an `#[export_name = "__crabbit_kernel_…"]` or a `#[no_mangle]` item whose
/// name carries [KERNEL_EXPORT_PREFIX] (docs/KERNEL-ABI.md).
pub(super) fn kernel_export_symbol<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: rustc_hir::def_id::DefId,
) -> Option<String> {
    let attrs = tcx.codegen_fn_attrs(def_id);
    let name = if let Some(name) = attrs.symbol_name {
        name.to_string()
    } else if attrs
        .flags
        .contains(rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags::NO_MANGLE)
    {
        tcx.item_name(def_id).to_string()
    } else {
        return None;
    };
    name.starts_with(KERNEL_EXPORT_PREFIX).then_some(name)
}

pub(super) fn function_symbol<'tcx>(
    tcx: TyCtxt<'tcx>,
    legaliser: &mut Legaliser,
    def_id: rustc_hir::def_id::DefId,
) -> crate::identifier::Identifier {
    legaliser.legalise(tcx.symbol_name(Instance::mono(tcx, def_id)).name)
}

/// A kernel's PTX entry name: its exported symbol with [KERNEL_EXPORT_PREFIX]
/// stripped, so every toolchain arm (crabbit PTX, LLVM PTX, nvcc) shares
/// one `.entry` name per kernel.
pub(super) fn kernel_symbol<'tcx>(
    tcx: TyCtxt<'tcx>,
    legaliser: &mut Legaliser,
    def_id: rustc_hir::def_id::DefId,
) -> crate::identifier::Identifier {
    let exported = kernel_export_symbol(tcx, def_id)
        .unwrap_or_else(|| tcx.def_path_str(def_id));
    let entry = exported
        .strip_prefix(KERNEL_EXPORT_PREFIX)
        .filter(|rest| !rest.is_empty())
        .unwrap_or(&exported);
    legaliser.legalise(entry)
}

pub(super) fn import_entry_wrapper(
    ctx: &mut Context,
    module_body: Ptr<BasicBlock>,
    rust_main: crate::identifier::Identifier,
) -> Result<(), String> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signed);
    let fn_ty = FunctionType::get(ctx, vec![], vec![i32_ty.into()]);
    let func = mir_dialect::ops::FuncOp::new(ctx, "main".try_into().unwrap(), fn_ty);
    let entry = func.get_entry_block(ctx);

    let call = mir_dialect::ops::CallOp::new_direct(ctx, rust_main, vec![], None);
    call.get_operation().insert_at_back(entry, ctx);

    let zero = mir_dialect::ops::ConstantOp::new_integer(
        ctx,
        IntegerAttr::new(i32_ty, APInt::from_u32(0, NonZero::new(32).unwrap())),
    );
    zero.get_operation().insert_at_back(entry, ctx);

    let ret = mir_dialect::ops::ReturnOp::new(ctx, Some(zero.get_result(ctx)));
    ret.get_operation().insert_at_back(entry, ctx);

    func.get_operation().insert_at_back(module_body, ctx);
    Ok(())
}

pub(super) fn mono_ty<'tcx>(tcx: TyCtxt<'tcx>, state: &FunctionImportState<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
    state.instance.map_or(ty, |instance| {
        instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            rustc_middle::ty::TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(ty),
        )
    })
}

pub(super) fn import_function<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    module_body: Ptr<BasicBlock>,
    symbol: crate::identifier::Identifier,
    body: &Body<'tcx>,
    is_kernel: bool,
    instance: Option<Instance<'tcx>>,
) -> Result<(), String> {
    if symbol_exists(ctx, module_body, symbol.as_ref()) {
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
    let func = mir_dialect::ops::FuncOp::new(ctx, symbol, fn_ty);
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
        let alloca = mir_dialect::ops::AllocaOp::new(ctx, ty);
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
                let undef = mir_dialect::ops::UndefOp::new(ctx, aggregate_ty);
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
                        mir_dialect::ops::InsertValueOp::new(ctx, element, current, vec![index]);
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
        let store = mir_dialect::ops::StoreOp::new(ctx, arg, slot);
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
pub(super) fn new_llvm_func_decl(
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
pub(super) fn new_llvm_func_def(
    ctx: &mut Context,
    name: crate::identifier::Identifier,
    ty: TypedHandle<llvm::types::FuncType>,
    linkage: LinkageAttr,
) -> llvm::ops::FuncOp {
    let func = new_llvm_func_decl(ctx, name, ty, linkage);
    func.get_or_create_entry_block(ctx);
    func
}

pub(super) fn declare_external_function(
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

pub(super) fn module_has_llvm_function(
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
