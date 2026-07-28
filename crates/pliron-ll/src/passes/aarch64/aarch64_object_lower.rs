use crate::ll::LinkageAttr;

use crate::{
    context::{Context, Ptr},
    dialects::{
        aarch64::op_interfaces::{BinaryFixup, FixupKind},
        aarch64::ops::FuncOp,
        builtin::{op_interfaces::SymbolOpInterface},
        macho::ops::{ObjectOp, Relocation, Symbol},
    },
    ir::operation::Operation,
    linked_list::ContainsLinkedList,
    result::STAIRResult,
};

use super::{
    attrs::{ATTR_KEY_AARCH64_ENCODED, ATTR_KEY_AARCH64_FIXUPS, ATTR_KEY_AARCH64_MODULE_LITERALS},
    frontend::module_op,
    target::TargetOs,
    util::{cast_operation, get_bytes_attr, get_fixups_attr, identifier, module_body},
};

const MACHO_ARM64_RELOC_BRANCH26: u8 = 2;

/// The container-format-independent pieces of an emitted aarch64 module:
/// the text section, its symbols (already mangled for the target OS), and
/// the branch fixups that must become relocations. Each object-container
/// writer maps these into its own format.
pub(super) struct ObjectParts {
    pub text: Vec<u8>,
    pub symbols: Vec<Symbol>,
    pub fixups: Vec<BinaryFixup>,
}

/// Collects the encoded functions, module literals, symbols, and pending
/// fixups of a fully-encoded aarch64 module, mangling symbol names for `os`.
pub(super) fn collect_object_parts(
    ctx: &Context,
    root: Ptr<Operation>,
    os: TargetOs,
) -> STAIRResult<ObjectParts> {
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
            get_bytes_attr(op, ctx, ATTR_KEY_AARCH64_ENCODED.as_str()).unwrap_or_default();
        text.extend_from_slice(&encoded);
        if func.linkage(ctx) == LinkageAttr::External {
            symbols.push(Symbol {
                name: os.symbol_name(&func.get_symbol_name(ctx).to_string()),
                offset,
                external: true,
                defined: true,
            });
        }
    }
    let literals =
        get_bytes_attr(root, ctx, ATTR_KEY_AARCH64_MODULE_LITERALS.as_str()).unwrap_or_default();
    text.extend_from_slice(&literals);
    let mut fixups =
        get_fixups_attr(root, ctx, ATTR_KEY_AARCH64_FIXUPS.as_str()).unwrap_or_default();
    for fixup in &mut fixups {
        fixup.symbol = os.symbol_name(&fixup.symbol);
        if !symbols.iter().any(|existing| existing.name == fixup.symbol) {
            symbols.push(Symbol {
                name: fixup.symbol.clone(),
                offset: 0,
                external: true,
                defined: false,
            });
        }
    }
    Ok(ObjectParts {
        text,
        symbols,
        fixups,
    })
}

/// Translates a fully-encoded aarch64 module into a Mach-O `macho.object`
/// operation. This is a translation out of the pass pipeline (the way
/// `mlir-translate` sits outside `mlir-opt`), not a [pliron::pass::Pass]:
/// it produces a new operation instead of transforming the module.
pub fn aarch64_macho_lower(ctx: &mut Context, root: Ptr<Operation>) -> STAIRResult<ObjectOp> {
    let parts = collect_object_parts(ctx, root, TargetOs::Darwin)?;
    let relocations = parts
        .fixups
        .iter()
        .map(|fixup| {
            let kind = match fixup.kind {
                FixupKind::Call26 => MACHO_ARM64_RELOC_BRANCH26,
            };
            Relocation {
                offset: fixup.offset,
                symbol: fixup.symbol.clone(),
                pcrel: true,
                length: 2,
                extern_: true,
                kind,
            }
        })
        .collect();
    Ok(ObjectOp::new_with_relocations(
        ctx,
        identifier("aarch64_object"),
        parts.text,
        parts.symbols,
        relocations,
    ))
}

#[cfg(test)]
mod tests {
    use crate::ll::LinkageAttr;
    use crate::{
        context::Context,
        dialects::{
            aarch64::{
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
        ATTR_KEY_AARCH64_ENCODED, ATTR_KEY_AARCH64_FIXUPS, ATTR_KEY_AARCH64_MODULE_LITERALS,
        aarch64_macho_lower,
    };

    fn context() -> Context {
        let mut ctx = Context::new();
        aarch64::register(&mut ctx);
        macho::register(&mut ctx);
        ctx
    }

    #[test]
    fn lowers_call26_relocations_and_reuses_existing_symbols() {
        let mut ctx = context();
        let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
        let target = aarch64::ops::FuncOp::new(&mut ctx, "target".try_into().unwrap(), LinkageAttr::External);
        target.get_operation().insert_at_back(body, &ctx);
        set_bytes_attr(
            target.get_operation(),
            &mut ctx,
            ATTR_KEY_AARCH64_ENCODED.as_str(),
            vec![0x00, 0x00, 0x00, 0x00],
        );
        let other = aarch64::ops::FuncOp::new(&mut ctx, "other".try_into().unwrap(), LinkageAttr::External);
        other.get_operation().insert_at_back(body, &ctx);
        set_bytes_attr(
            other.get_operation(),
            &mut ctx,
            ATTR_KEY_AARCH64_ENCODED.as_str(),
            vec![],
        );
        set_bytes_attr(
            module.get_operation(),
            &mut ctx,
            ATTR_KEY_AARCH64_MODULE_LITERALS.as_str(),
            vec![],
        );
        set_fixups_attr(
            module.get_operation(),
            &mut ctx,
            ATTR_KEY_AARCH64_FIXUPS.as_str(),
            vec![
                BinaryFixup {
                    offset: 0,
                    symbol: "target".to_string(),
                    kind: FixupKind::Call26,
                },
                BinaryFixup {
                    offset: 4,
                    symbol: "callee".to_string(),
                    kind: FixupKind::Call26,
                },
            ],
        );

        let object = aarch64_macho_lower(&mut ctx, module.get_operation()).unwrap();
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
