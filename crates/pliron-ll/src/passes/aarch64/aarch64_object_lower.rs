use crate::ll::{DataReloc, LinkageAttr};

use crate::{
    context::{Context, Ptr},
    dialects::{
        aarch64::op_interfaces::{BinaryFixup, FixupKind},
        aarch64::ops::FuncOp,
        builtin::{op_interfaces::SymbolOpInterface},
        llvm::ops::GlobalOp as LlvmGlobalOp,
        macho::ops::{ObjectOp, Relocation, Symbol},
    },
    input_error_noloc,
    ir::operation::Operation,
    linked_list::ContainsLinkedList,
    result::STAIRResult,
};

use super::{
    attrs::{ATTR_KEY_AARCH64_ENCODED, ATTR_KEY_AARCH64_FIXUPS, ATTR_KEY_AARCH64_MODULE_LITERALS},
    error::Aarch64Err,
    frontend::module_op,
    target::TargetOs,
    util::{cast_operation, get_bytes_attr, get_fixups_attr, identifier, module_body},
};

const MACHO_ARM64_RELOC_BRANCH26: u8 = 2;

/// A symbol defined inside a [DataSection], at `offset` bytes into it.
pub(super) struct DataSymbol {
    pub name: String,
    pub offset: u64,
    pub size: u64,
}

/// One laid-out data section: the concatenated (aligned) initializers of the
/// module's `ll.data` globals, the symbol each global defines, and the
/// pointer-slot relocations with section-relative offsets.
#[derive(Default)]
pub(super) struct DataSection {
    pub align: u64,
    pub bytes: Vec<u8>,
    pub symbols: Vec<DataSymbol>,
    pub relocs: Vec<DataReloc>,
}

impl DataSection {
    fn append_global(&mut self, name: String, init: &crate::ll::DataAttr, os: TargetOs) {
        let align = init.align.max(1);
        let offset = (self.bytes.len() as u64).next_multiple_of(align);
        self.bytes.resize(offset as usize, 0);
        self.align = self.align.max(align);
        self.symbols.push(DataSymbol {
            name,
            offset,
            size: init.bytes.len() as u64,
        });
        for reloc in &init.relocs {
            self.relocs.push(DataReloc {
                offset: offset + reloc.offset,
                symbol: os.symbol_name(&reloc.symbol),
                addend: reloc.addend,
            });
        }
        self.bytes.extend_from_slice(&init.bytes);
    }

    fn defines(&self, name: &str) -> bool {
        self.symbols.iter().any(|symbol| symbol.name == name)
    }
}

/// The container-format-independent pieces of an emitted aarch64 module:
/// the text section, its symbols (already mangled for the target OS), the
/// branch fixups that must become relocations, and the data sections.
/// Each object-container writer maps these into its own format.
///
/// Immutable globals split by whether they contain pointer slots: a slot is
/// filled by the dynamic linker at load time, so its section must be
/// writable until relocation — that is `.data.rel.ro` (read-only after
/// relocation), while genuinely constant bytes stay in `.rodata`.
///
/// Thread-local globals (`ll.tls`) go to the TLS template sections instead:
/// `.tdata` when the initializer has content, `.tbss` when it is all zeros
/// (a `.tbss` [DataSection] keeps its zero `bytes` for size and offset
/// accounting, but object writers emit no file content for it).
pub(super) struct ObjectParts {
    pub text: Vec<u8>,
    pub symbols: Vec<Symbol>,
    pub fixups: Vec<BinaryFixup>,
    /// Names among `symbols` exported with weak binding (`STB_WEAK`): a
    /// strong definition of the same symbol elsewhere wins at link time.
    pub weak_symbols: std::collections::HashSet<String>,
    pub rodata: Option<DataSection>,
    pub relro: Option<DataSection>,
    pub data: Option<DataSection>,
    pub tdata: Option<DataSection>,
    pub tbss: Option<DataSection>,
}

impl ObjectParts {
    pub fn data_sections(&self) -> impl Iterator<Item = &DataSection> {
        [&self.rodata, &self.relro, &self.data, &self.tdata, &self.tbss]
            .into_iter()
            .flatten()
    }
}

/// Collects the encoded functions, module literals, data globals, symbols,
/// and pending fixups of a fully-encoded aarch64 module, mangling symbol
/// names for `os`.
pub(super) fn collect_object_parts(
    ctx: &Context,
    root: Ptr<Operation>,
    os: TargetOs,
) -> STAIRResult<ObjectParts> {
    let module = module_op(ctx, root)?;
    let body = module_body(ctx, module);
    let mut text = Vec::new();
    let mut symbols = Vec::new();
    let mut weak_symbols = std::collections::HashSet::new();
    let mut rodata = DataSection::default();
    let mut relro = DataSection::default();
    let mut data = DataSection::default();
    let mut tdata = DataSection::default();
    let mut tbss = DataSection::default();
    let ops: Vec<_> = body.deref(ctx).iter(ctx).collect();
    for op in ops {
        if let Some(func) = cast_operation::<FuncOp>(ctx, op) {
            let offset = text.len() as u64;
            let encoded =
                get_bytes_attr(op, ctx, ATTR_KEY_AARCH64_ENCODED.as_ref()).unwrap_or_default();
            text.extend_from_slice(&encoded);
            let linkage = func.linkage(ctx);
            if matches!(linkage, LinkageAttr::External | LinkageAttr::Weak) {
                let name = os.symbol_name(func.get_symbol_name(ctx).as_ref());
                if linkage == LinkageAttr::Weak {
                    weak_symbols.insert(name.clone());
                }
                symbols.push(Symbol {
                    name,
                    offset,
                    external: true,
                    defined: true,
                });
            }
        } else if let Some(global) = cast_operation::<LlvmGlobalOp>(ctx, op)
            && let Some(init) = crate::ll::global_data(ctx, &global)
        {
            let name = os.symbol_name(global.get_symbol_name(ctx).as_ref());
            let section = if crate::ll::global_is_thread_local(ctx, &global) {
                // TLS template: all-zero pointer-free initializers need no
                // file content (`.tbss`); anything else is `.tdata`.
                if init.relocs.is_empty() && init.bytes.iter().all(|byte| *byte == 0) {
                    &mut tbss
                } else {
                    &mut tdata
                }
            } else if init.mutable {
                &mut data
            } else if init.relocs.is_empty() {
                &mut rodata
            } else {
                &mut relro
            };
            section.append_global(name, &init, os);
        }
    }
    let literals =
        get_bytes_attr(root, ctx, ATTR_KEY_AARCH64_MODULE_LITERALS.as_ref()).unwrap_or_default();
    text.extend_from_slice(&literals);
    let mut fixups =
        get_fixups_attr(root, ctx, ATTR_KEY_AARCH64_FIXUPS.as_ref()).unwrap_or_default();
    // Every relocation target needs a symbol-table entry: a defined text or
    // data symbol when there is one, otherwise an undefined symbol for the
    // linker to resolve.
    let ensure_symbol = |symbols: &mut Vec<Symbol>, sections: [&DataSection; 5], name: &str| {
        if symbols.iter().any(|existing| existing.name == name)
            || sections.iter().any(|section| section.defines(name))
        {
            return;
        }
        symbols.push(Symbol {
            name: name.to_string(),
            offset: 0,
            external: true,
            defined: false,
        });
    };
    for fixup in &mut fixups {
        fixup.symbol = os.symbol_name(&fixup.symbol);
        let symbol = fixup.symbol.clone();
        ensure_symbol(&mut symbols, [&rodata, &relro, &data, &tdata, &tbss], &symbol);
    }
    for section in [&rodata, &relro, &data, &tdata, &tbss] {
        for reloc in &section.relocs {
            ensure_symbol(
                &mut symbols,
                [&rodata, &relro, &data, &tdata, &tbss],
                &reloc.symbol,
            );
        }
    }
    Ok(ObjectParts {
        text,
        symbols,
        fixups,
        weak_symbols,
        rodata: (!rodata.symbols.is_empty()).then_some(rodata),
        relro: (!relro.symbols.is_empty()).then_some(relro),
        data: (!data.symbols.is_empty()).then_some(data),
        tdata: (!tdata.symbols.is_empty()).then_some(tdata),
        tbss: (!tbss.symbols.is_empty()).then_some(tbss),
    })
}

/// Translates a fully-encoded aarch64 module into a Mach-O `macho.object`
/// operation. This is a translation out of the pass pipeline (the way
/// `mlir-translate` sits outside `mlir-opt`), not a [pliron::pass::Pass]:
/// it produces a new operation instead of transforming the module.
pub fn aarch64_macho_lower(ctx: &mut Context, root: Ptr<Operation>) -> STAIRResult<ObjectOp> {
    let parts = collect_object_parts(ctx, root, TargetOs::Darwin)?;
    if parts.data_sections().next().is_some() {
        return Err(input_error_noloc!(Aarch64Err::UnsupportedOp(
            "module defines `ll.data` globals, and Mach-O data sections are not implemented; \
             lower the module for an ELF target instead"
                .to_string()
        )));
    }
    let mut relocations = Vec::with_capacity(parts.fixups.len());
    for fixup in &parts.fixups {
        let kind = match fixup.kind {
            FixupKind::Call26 => MACHO_ARM64_RELOC_BRANCH26,
            FixupKind::AdrpPage21 | FixupKind::AddLo12 => {
                return Err(input_error_noloc!(Aarch64Err::UnsupportedOp(
                    "Mach-O page relocations for data globals are not implemented".to_string()
                )));
            }
            FixupKind::TprelHi12 | FixupKind::TprelLo12Nc => {
                return Err(input_error_noloc!(Aarch64Err::UnsupportedOp(
                    "Mach-O thread-local relocations are not implemented; \
                     lower the module for an ELF target instead"
                        .to_string()
                )));
            }
        };
        relocations.push(Relocation {
            offset: fixup.offset,
            symbol: fixup.symbol.clone(),
            pcrel: true,
            length: 2,
            extern_: true,
            kind,
        });
    }
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
            ATTR_KEY_AARCH64_ENCODED.as_ref(),
            vec![0x00, 0x00, 0x00, 0x00],
        );
        let other = aarch64::ops::FuncOp::new(&mut ctx, "other".try_into().unwrap(), LinkageAttr::External);
        other.get_operation().insert_at_back(body, &ctx);
        set_bytes_attr(
            other.get_operation(),
            &mut ctx,
            ATTR_KEY_AARCH64_ENCODED.as_ref(),
            vec![],
        );
        set_bytes_attr(
            module.get_operation(),
            &mut ctx,
            ATTR_KEY_AARCH64_MODULE_LITERALS.as_ref(),
            vec![],
        );
        set_fixups_attr(
            module.get_operation(),
            &mut ctx,
            ATTR_KEY_AARCH64_FIXUPS.as_ref(),
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

    fn data_global(
        ctx: &mut Context,
        body: crate::context::Ptr<crate::ir::basic_block::BasicBlock>,
        name: &str,
        data: crate::ll::DataAttr,
    ) {
        use crate::dialects::llvm::ops::GlobalOp;
        let i64_ty = crate::dialects::builtin::types::IntegerType::get(
            ctx,
            64,
            crate::dialects::builtin::types::Signedness::Signless,
        );
        let global = GlobalOp::new(ctx, name.try_into().unwrap(), i64_ty.into());
        global.set_attr_llvm_global_linkage(
            ctx,
            crate::dialects::llvm::attributes::LinkageAttr::ExternalLinkage,
        );
        crate::ll::set_global_data(ctx, &global, data);
        global.get_operation().insert_at_back(body, ctx);
    }

    #[test]
    fn collects_data_globals_into_sections() {
        use super::{TargetOs, collect_object_parts};
        use crate::ll::{DataAttr, DataReloc};

        let mut ctx = context();
        let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
        // Constant, pointer-free: `.rodata`. Deliberately 4 bytes so the next
        // 8-aligned global starts at offset 8, not 4.
        data_global(
            &mut ctx,
            body,
            "plain",
            DataAttr {
                bytes: vec![1, 2, 3, 4],
                align: 4,
                mutable: false,
                relocs: vec![],
            },
        );
        data_global(
            &mut ctx,
            body,
            "plain_tail",
            DataAttr {
                bytes: vec![5; 8],
                align: 8,
                mutable: false,
                relocs: vec![],
            },
        );
        // Constant with a pointer slot: `.data.rel.ro`. One reloc targets a
        // defined data global, the other an undefined symbol.
        data_global(
            &mut ctx,
            body,
            "holder",
            DataAttr {
                bytes: vec![0; 16],
                align: 8,
                mutable: false,
                relocs: vec![
                    DataReloc {
                        offset: 0,
                        symbol: "plain".to_string(),
                        addend: 4,
                    },
                    DataReloc {
                        offset: 8,
                        symbol: "elsewhere".to_string(),
                        addend: 0,
                    },
                ],
            },
        );
        // Mutable: `.data`.
        data_global(
            &mut ctx,
            body,
            "counter",
            DataAttr {
                bytes: vec![7; 8],
                align: 8,
                mutable: true,
                relocs: vec![],
            },
        );

        let parts = collect_object_parts(&ctx, module.get_operation(), TargetOs::Linux).unwrap();
        let rodata = parts.rodata.as_ref().unwrap();
        assert_eq!(rodata.align, 8);
        assert_eq!(rodata.bytes.len(), 16);
        assert_eq!(rodata.symbols[0].name, "plain");
        assert_eq!(rodata.symbols[0].offset, 0);
        assert_eq!(rodata.symbols[0].size, 4);
        assert_eq!(rodata.symbols[1].name, "plain_tail");
        assert_eq!(rodata.symbols[1].offset, 8);
        let relro = parts.relro.as_ref().unwrap();
        assert_eq!(relro.symbols[0].name, "holder");
        assert_eq!(relro.relocs.len(), 2);
        assert_eq!(relro.relocs[0].symbol, "plain");
        assert_eq!(relro.relocs[0].addend, 4);
        let data = parts.data.as_ref().unwrap();
        assert_eq!(data.symbols[0].name, "counter");
        // Only the genuinely undefined reloc target gets an undefined symbol.
        assert_eq!(parts.symbols.len(), 1);
        assert_eq!(parts.symbols[0].name, "elsewhere");
        assert!(!parts.symbols[0].defined);
    }

    #[test]
    fn macho_lowering_rejects_data_globals() {
        use crate::ll::DataAttr;

        let mut ctx = context();
        let module = builtin::ops::ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let body = module.get_region(&ctx).deref(&ctx).get_head().unwrap();
        data_global(
            &mut ctx,
            body,
            "answer",
            DataAttr {
                bytes: vec![42, 0, 0, 0, 0, 0, 0, 0],
                align: 8,
                mutable: false,
                relocs: vec![],
            },
        );

        let err = match aarch64_macho_lower(&mut ctx, module.get_operation()) {
            Ok(_) => panic!("data globals unexpectedly lowered to Mach-O"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("Mach-O data sections"));
    }
}
