//! Textual LLVM IR export of the `rust_kernels` module (`CRABBIT_LL_OUT`),
//! through cuda-oxide's `llvm-export` crate. This is the "rust-llvm"
//! comparison arm of the kernel corpus: the same front half as the native
//! PTX path (importer + mid-end), LLVM's NVPTX backend as the back half
//! (`llc -march=nvptx64`).
//!
//! The exporter recognizes kernels by a `gpu_kernel` string attribute on
//! the `llvm.func` and prints them with the `ptx_kernel` calling convention
//! (no `!nvvm.annotations` needed for llc); intrinsic callees named
//! `llvm_nvvm_…` are decoded back to `llvm.nvvm.…`.
//!
//! `.shared` statics are exported as `addrspace(3)` globals with a
//! zeroinitializer; the exporter renders `llvm.addressof` of such a global
//! through an `addrspacecast` to the generic address space, which is
//! exactly what the native path's `cvta.shared` does.

use pliron::{
    builtin::{attributes::StringAttr, op_interfaces::OneRegionInterface, ops::ModuleOp},
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
};
use pliron::builtin::op_interfaces::{AtMostOneRegionInterface as _, OneResultInterface, SymbolOpInterface};
use pliron_llvm::{
    attributes::LinkageAttr,
    op_interfaces::{CastOpInterface as _, IsDeclaration},
    ops::{AddrSpaceCastOp, AddressOfOp, FuncOp, GlobalOp},
    types::PointerType,
};

pub fn export_kernel_module(ctx: &mut Context, root: Ptr<Operation>) -> Result<String, String> {
    let module = Operation::get_op::<ModuleOp>(root, ctx)
        .ok_or_else(|| "kernel module root is not a builtin.module".to_string())?;
    let body = module.get_region(ctx).deref(ctx).get_head().unwrap();
    let ops: Vec<Ptr<Operation>> = body.deref(ctx).iter(ctx).collect();
    let kernel_key: pliron::identifier::Identifier = "gpu_kernel".try_into().unwrap();
    let mut shared: Vec<String> = Vec::new();
    let mut globals_in_1: Vec<String> = Vec::new();
    for op in ops.iter().copied() {
        if let Some(func) = Operation::get_op::<FuncOp>(op, ctx) {
            let external = func
                .get_attr_llvm_function_linkage(ctx)
                .is_none_or(|linkage| matches!(*linkage, LinkageAttr::ExternalLinkage));
            if !func.is_declaration(ctx) && external {
                op.deref_mut(ctx)
                    .attributes
                    .set(kernel_key.clone(), StringAttr::new("kernel".to_string()));
            }
        } else if let Some(global) = Operation::get_op::<GlobalOp>(op, ctx) {
            let data = pliron_ll::ll::global_data(ctx, &global);
            let name = global.get_symbol_name(ctx).to_string();
            let is_shared =
                pliron_ll::ll::global_section(ctx, &global).as_deref() == Some(".shared");
            // The exporter keeps alignment and byte initializers in its own
            // attributes (cuda-oxide's `GlobalOpExt`): translate crabbit's
            // `ll.data`. Externally-linked globals are printed as `external`
            // declarations there, so definitions become internal.
            let align_key: pliron::identifier::Identifier =
                "cuda_oxide_global_alignment".try_into().unwrap();
            let hex_key: pliron::identifier::Identifier =
                "cuda_oxide_global_initializer_hex".try_into().unwrap();
            if let Some(data) = &data {
                if !data.relocs.is_empty() {
                    return Err(format!(
                        "global `{name}` holds pointers (relocations are not supported in kernels)"
                    ));
                }
                op.deref_mut(ctx).attributes.set(
                    align_key.clone(),
                    pliron_llvm::attributes::AlignmentAttr(data.align.max(1) as u32),
                );
                global.set_attr_llvm_global_linkage(ctx, LinkageAttr::InternalLinkage);
                if !is_shared && data.bytes.iter().any(|&b| b != 0) {
                    let hex: String = data.bytes.iter().map(|b| format!("{b:02x}")).collect();
                    op.deref_mut(ctx)
                        .attributes
                        .set(hex_key.clone(), StringAttr::new(hex));
                }
            }
            if is_shared {
                // Shared memory: addrspace(3); the exporter zero-initializes a
                // global without a byte initializer.
                global.set_address_space(ctx, 3);
                drop_data_initializer(ctx, &global);
                shared.push(name);
            } else if data.is_some() {
                // `.global` data: addrspace(1) so llc places it in global
                // memory; the initializer is carried by the hex attribute.
                global.set_address_space(ctx, 1);
                drop_data_initializer(ctx, &global);
                globals_in_1.push(name);
            }
        }
    }
    // `llvm.addressof` of a shared global must yield a `ptr addrspace(3)`;
    // cast it to the generic space for the existing uses.
    for op in ops {
        if let Some(func) = Operation::get_op::<FuncOp>(op, ctx)
            && !func.is_declaration(ctx)
        {
            let region = func.get_region(ctx).unwrap();
            let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
            for block in blocks {
                let inner: Vec<Ptr<Operation>> = block.deref(ctx).iter(ctx).collect();
                for inner_op in inner {
                    let Some(addr) = Operation::get_op::<AddressOfOp>(inner_op, ctx) else {
                        continue;
                    };
                    let name = addr.get_global_name(ctx).to_string();
                    let space = if shared.contains(&name) {
                        3
                    } else if globals_in_1.contains(&name) {
                        1
                    } else {
                        continue;
                    };
                    let shared_addr = AddressOfOp::new(ctx, name.try_into().unwrap(), space);
                    shared_addr.get_operation().insert_before(ctx, inner_op);
                    let generic_ty = PointerType::get(ctx, 0).into();
                    let shared_val = shared_addr.get_result(ctx);
                    let cast = AddrSpaceCastOp::new(ctx, shared_val, generic_ty);
                    cast.get_operation().insert_before(ctx, inner_op);
                    let old = addr.get_result(ctx);
                    let new = cast.get_result(ctx);
                    old.replace_all_uses_with(ctx, &new);
                    Operation::erase(inner_op, ctx);
                }
            }
        }
    }
    llvm_export::export::export_module_to_string(ctx, &module)
}

/// Drop crabbit's `ll.data` initializer value (the exporter does not know
/// the attribute; bytes and alignment were translated above).
fn drop_data_initializer(ctx: &mut Context, global: &GlobalOp) {
    let init_key: pliron::identifier::Identifier = "global_initializer".try_into().unwrap();
    global
        .get_operation()
        .deref_mut(ctx)
        .attributes
        .0
        .remove(&init_key);
}
