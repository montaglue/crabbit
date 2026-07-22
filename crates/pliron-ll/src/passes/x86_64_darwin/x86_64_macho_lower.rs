use crate::{
    context::{Context, Ptr},
    dialects::{
        x86_64::op_interfaces::FixupKind,
        x86_64::ops::FuncOp,
        builtin::{op_interfaces::SymbolOpInterface},
        macho::ops::{ObjectOp, Relocation, Symbol},
    },
    ir::operation::Operation,
    linked_list::ContainsLinkedList,
    result::STAIRResult,
};

use super::{
    attrs::{ATTR_KEY_X86_64_ENCODED, ATTR_KEY_X86_64_FIXUPS, ATTR_KEY_X86_64_MODULE_LITERALS},
    frontend::module_op,
    util::{cast_operation, darwin_symbol, get_bytes_attr, get_fixups_attr, identifier, module_body},
};
use llvm_compat::ll::LinkageAttr;

const MACHO_X86_64_RELOC_BRANCH: u8 = 2;

/// Translates a fully-encoded x86-64 module into a Mach-O `macho.object`
/// operation. This is a translation out of the pass pipeline (the way
/// `mlir-translate` sits outside `mlir-opt`), not a [pliron::pass::Pass]:
/// it produces a new operation instead of transforming the module.
pub fn x86_64_macho_lower(ctx: &mut Context, root: Ptr<Operation>) -> STAIRResult<ObjectOp> {
    let module = module_op(ctx, root)?;
    let body = module_body(ctx, module);
    let mut text = Vec::new();
    let mut symbols = Vec::new();
    let funcs: Vec<_> = body.deref(ctx).iter(ctx).collect();
    for op in funcs {
        let Some(func) = cast_operation::<FuncOp>(ctx, op) else {
            continue;
        };
        let offset = text.len() as u64;
        let encoded =
            get_bytes_attr(op, ctx, ATTR_KEY_X86_64_ENCODED.as_str()).unwrap_or_default();
        text.extend_from_slice(&encoded);
        if func.linkage(ctx) == LinkageAttr::External {
            symbols.push(Symbol {
                name: darwin_symbol(&func.get_symbol_name(ctx).to_string()),
                offset,
                external: true,
                defined: true,
            });
        }
    }
    let literals =
        get_bytes_attr(root, ctx, ATTR_KEY_X86_64_MODULE_LITERALS.as_str()).unwrap_or_default();
    text.extend_from_slice(&literals);
    let relocations = external_branch_relocations(ctx, root, &mut symbols);
    Ok(ObjectOp::new_with_relocations(
        ctx,
        identifier("x86_64_darwin_object"),
        text,
        symbols,
        relocations,
    ))
}

fn external_branch_relocations(
    ctx: &Context,
    root: Ptr<Operation>,
    symbols: &mut Vec<Symbol>,
) -> Vec<Relocation> {
    let fixups = get_fixups_attr(root, ctx, ATTR_KEY_X86_64_FIXUPS.as_str()).unwrap_or_default();
    fixups
        .into_iter()
        .map(|fixup| {
            let symbol = darwin_symbol(&fixup.symbol);
            if !symbols.iter().any(|existing| existing.name == symbol) {
                symbols.push(Symbol {
                    name: symbol.clone(),
                    offset: 0,
                    external: true,
                    defined: false,
                });
            }
            let kind = match fixup.kind {
                FixupKind::Branch32 => MACHO_X86_64_RELOC_BRANCH,
            };
            Relocation {
                offset: fixup.offset,
                symbol,
                pcrel: true,
                length: 2,
                extern_: true,
                kind,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use llvm_compat::ll::LinkageAttr;
    use crate::{
        context::Context,
        dialects::{
            x86_64::{
                self,
                op_interfaces::{BinaryFixup, FixupKind},
            },
            builtin::{self, op_interfaces::OneRegionInterface},
            macho,
        },
        ir::op::Op,
        linked_list::ContainsLinkedList,
    };

    use super::{
        super::util::{set_bytes_attr, set_fixups_attr},
        ATTR_KEY_X86_64_ENCODED, ATTR_KEY_X86_64_FIXUPS, ATTR_KEY_X86_64_MODULE_LITERALS,
        x86_64_macho_lower,
    };

    fn context() -> Context {
        let mut ctx = Context::new();
        x86_64::register(&mut ctx);
        macho::register(&mut ctx);
        ctx
    }

    #[test]
    fn lowers_call26_relocations_and_reuses_existing_symbols() {
        let mut ctx = context();
        let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
        let target = x86_64::ops::FuncOp::new(&mut ctx, "target".try_into().unwrap(), LinkageAttr::External);
        target.get_operation().insert_at_back(body, &ctx);
        set_bytes_attr(
            target.get_operation(),
            &mut ctx,
            ATTR_KEY_X86_64_ENCODED.as_str(),
            vec![0x00, 0x00, 0x00, 0x00],
        );
        let other = x86_64::ops::FuncOp::new(&mut ctx, "other".try_into().unwrap(), LinkageAttr::External);
        other.get_operation().insert_at_back(body, &ctx);
        set_bytes_attr(
            other.get_operation(),
            &mut ctx,
            ATTR_KEY_X86_64_ENCODED.as_str(),
            vec![],
        );
        set_bytes_attr(
            module.get_operation(),
            &mut ctx,
            ATTR_KEY_X86_64_MODULE_LITERALS.as_str(),
            vec![],
        );
        set_fixups_attr(
            module.get_operation(),
            &mut ctx,
            ATTR_KEY_X86_64_FIXUPS.as_str(),
            vec![
                BinaryFixup {
                    offset: 0,
                    symbol: "target".to_string(),
                    kind: FixupKind::Branch32,
                },
                BinaryFixup {
                    offset: 4,
                    symbol: "callee".to_string(),
                    kind: FixupKind::Branch32,
                },
            ],
        );

        let object = x86_64_macho_lower(&mut ctx, module.get_operation()).unwrap();
        let relocations = object.relocations(&ctx);
        assert_eq!(relocations.len(), 2);
        assert_eq!(relocations[0].symbol, "_target");
        assert_eq!(relocations[1].symbol, "_callee");
        let symbols = object.symbols(&ctx);
        assert_eq!(
            symbols
                .iter()
                .filter(|symbol| symbol.name == "_target")
                .count(),
            1
        );
        assert!(symbols.iter().any(|symbol| symbol.name == "_callee"));
    }
}
