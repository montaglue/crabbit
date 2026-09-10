use super::*;

pub(super) fn declare_static_global<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    module_body: Ptr<BasicBlock>,
    symbol: crate::identifier::Identifier,
    def_id: rustc_span::def_id::DefId,
) -> Result<(), String> {
    if symbol_exists(ctx, module_body, symbol.as_ref()) {
        return Ok(());
    }
    // A local static is defined here from its evaluated initializer; a foreign
    // one is only declared (a `llvm.global` with no initializer), so its
    // address resolves through an undefined symbol-table entry at link time.
    if def_id.is_local()
        && let Ok(alloc) = tcx.eval_static_initializer(def_id)
    {
        emit_allocation_global(
            tcx,
            ctx,
            module_body,
            symbol.clone(),
            alloc.inner(),
            LinkageAttr::ExternalLinkage,
            false,
        )?;
        // `#[link_section]` rides along as `ll_section`; the NVPTX
        // translator reads `.shared` off it (docs/KERNEL-ABI.md).
        if let Some(section) = tcx.codegen_fn_attrs(def_id).link_section {
            set_global_section_by_symbol(ctx, module_body, &symbol, section.as_str());
        }
        return Ok(());
    }
    let byte_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
    let global_ty = llvm::types::ArrayType::get(ctx, byte_ty, 0).into();
    let global = llvm::ops::GlobalOp::new(ctx, symbol, global_ty);
    global.set_attr_llvm_global_linkage(ctx, LinkageAttr::ExternalLinkage);
    global.get_operation().insert_at_back(module_body, ctx);
    Ok(())
}

pub(super) fn set_global_section_by_symbol(
    ctx: &mut Context,
    module_body: Ptr<BasicBlock>,
    symbol: &crate::identifier::Identifier,
    section: &str,
) {
    let global = module_body.deref(ctx).iter(ctx).find(|op| {
        let op_obj = Operation::get_op_dyn(*op, ctx);
        op_cast::<dyn SymbolOpInterface>(&*op_obj)
            .is_some_and(|symbol_op| symbol_op.get_symbol_name(ctx) == *symbol)
    });
    if let Some(global) = global
        && let Some(global) = Operation::get_op::<llvm::ops::GlobalOp>(global, ctx)
    {
        pliron_ll::ll::set_global_section(ctx, &global, section);
    }
}

/// Declare or define the storage of a `#[thread_local]` static (the target
/// of [Rvalue::ThreadLocalRef]). Same shape as [declare_static_global] —
/// a local static is defined from its evaluated initializer, a foreign one
/// (e.g. std's `RandomState::new::KEYS`, exported from libstd's `.tdata`)
/// is only declared — but the emitted `llvm.global` carries the
/// [ll.tls](pliron_ll::ll::TlsAttr) marker, so the backend materializes its
/// address through the thread pointer and places the initializer in the TLS
/// template sections.
pub(super) fn declare_thread_local_global<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    module_body: Ptr<BasicBlock>,
    symbol: crate::identifier::Identifier,
    def_id: rustc_span::def_id::DefId,
) -> Result<(), String> {
    if symbol_exists(ctx, module_body, symbol.as_ref()) {
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
pub(super) fn emit_allocation_global<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctx: &mut Context,
    module_body: Ptr<BasicBlock>,
    symbol: crate::identifier::Identifier,
    alloc: &rustc_mir::interpret::Allocation,
    linkage: LinkageAttr,
    thread_local: bool,
) -> Result<(), String> {
    if symbol_exists(ctx, module_body, symbol.as_ref()) {
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
pub(super) fn data_global_for_alloc<'tcx>(
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
pub(super) fn hex_suffix(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub(super) fn declare_anonymous_byte_global(
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
    let symbol = legaliser.legalise(&format!("L_crabbit_bytes_{suffix}"));
    if symbol_exists(ctx, module_body, symbol.as_ref()) {
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

pub(super) fn llvm_decl_type(ctx: &mut Context, ty: TypeHandle) -> TypeHandle {
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
                let name = format!("{name}__llvm").try_into().expect("suffixing an existing legal identifier keeps it legal");
                llvm::types::StructType::get_named(ctx, name, fields)
                    .expect("the __llvm mirror name maps 1:1 to this field set")
                    .into()
            }
            None => llvm::types::StructType::get_unnamed(ctx, fields.unwrap_or_default()).into(),
        };
    }
    ty
}

pub(super) fn declare_default_allocator_shims<'tcx>(
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
        let decl = new_llvm_func_decl(ctx, name.try_into().expect("static identifier literal"), ty, LinkageAttr::ExternalLinkage);
        decl.get_operation().insert_at_back(module_body, ctx);
    }

    define_allocator_call(
        ctx,
        module_body,
        allocator_symbol(tcx, "__rust_alloc"),
        "malloc".try_into().expect("static identifier literal"),
        rust_alloc_ty,
        |args, _one| vec![args[0]],
        Some(ptr_ty),
    );
    define_allocator_call(
        ctx,
        module_body,
        allocator_symbol(tcx, "__rust_dealloc"),
        "free".try_into().expect("static identifier literal"),
        rust_dealloc_ty,
        |args, _one| vec![args[0]],
        None,
    );
    define_allocator_call(
        ctx,
        module_body,
        allocator_symbol(tcx, "__rust_realloc"),
        "realloc".try_into().expect("static identifier literal"),
        rust_realloc_ty,
        |args, _one| vec![args[0], args[3]],
        Some(ptr_ty),
    );
    define_allocator_call(
        ctx,
        module_body,
        allocator_symbol(tcx, "__rust_alloc_zeroed"),
        "calloc".try_into().expect("static identifier literal"),
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

pub(super) fn define_allocator_call<F>(
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

pub(super) fn allocator_symbol<'tcx>(tcx: TyCtxt<'tcx>, name: &str) -> crate::identifier::Identifier {
    rustc_symbol_mangling::mangle_internal_symbol(tcx, name)
        .as_str()
        .try_into()
        .expect("mangled symbols are legal identifiers")
}
