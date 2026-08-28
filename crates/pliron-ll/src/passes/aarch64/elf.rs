//! Minimal ELF64 relocatable-object writer for the aarch64 backend, the
//! Linux counterpart of the Mach-O writer in [super::macho]. It emits a
//! single `.text` section plus `.rela.text`, `.symtab`, the string tables,
//! and an empty `.note.GNU-stack` (so linkers don't assume an executable
//! stack). Modules with `ll.data` globals additionally get `.rodata`,
//! `.data.rel.ro`, and/or `.data` sections (with `.rela` twins for their
//! pointer slots); a module without them produces byte-identical output to
//! the text-only writer. Thread-local globals (`ll.tls`) get `.tdata` /
//! `.tbss` (`SHF_TLS`, symbols `STT_TLS`); gathering them into `PT_TLS` is
//! the linker's job, so an ET_REL object only carries the sections,
//! symbols, and TPREL relocations.

use crate::dialects::aarch64::op_interfaces::FixupKind;

use super::aarch64_object_lower::{DataSection, ObjectParts};

const EM_AARCH64: u16 = 183;
const ET_REL: u16 = 1;
const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const SHT_RELA: u32 = 4;
const SHT_NOBITS: u32 = 8;
const SHF_WRITE: u64 = 0x1;
const SHF_ALLOC: u64 = 0x2;
const SHF_EXECINSTR: u64 = 0x4;
const SHF_TLS: u64 = 0x400;
const STB_GLOBAL: u8 = 1;
const STB_WEAK: u8 = 2;
const STT_OBJECT: u8 = 1;
const STT_FUNC: u8 = 2;
const STT_TLS: u8 = 6;
const SHN_UNDEF: u16 = 0;
const R_AARCH64_ABS64: u32 = 257;
const R_AARCH64_ADR_PREL_PG_HI21: u32 = 275;
const R_AARCH64_ADD_ABS_LO12_NC: u32 = 277;
const R_AARCH64_CALL26: u32 = 283;
const R_AARCH64_TLSLE_ADD_TPREL_HI12: u32 = 549;
const R_AARCH64_TLSLE_ADD_TPREL_LO12_NC: u32 = 551;

/// One row of the symbol table after unifying text and data symbols; the
/// index of a row (+1 for the null entry) is what relocations reference.
struct ElfSymbol {
    name: String,
    info: u8,
    shndx: u16,
    value: u64,
    size: u64,
}

pub(super) fn write_elf_object(parts: &ObjectParts) -> Vec<u8> {
    // Section header table layout (index 0 is the SHT_NULL entry). The data
    // sections and their `.rela` twins exist only when the module has data
    // globals, so indices are computed rather than fixed.
    let rodata = parts.rodata.as_ref();
    let relro = parts.relro.as_ref();
    let data = parts.data.as_ref();
    let tdata = parts.tdata.as_ref();
    let tbss = parts.tbss.as_ref();
    let sec_text: u16 = 1;
    let _sec_rela_text: u16 = 2;
    let mut next_section = 3u16;
    let mut claim = |present: bool| {
        present.then(|| {
            let index = next_section;
            next_section += 1;
            index
        })
    };
    let sec_rodata = claim(rodata.is_some());
    let sec_rela_rodata = claim(rodata.is_some_and(|section| !section.relocs.is_empty()));
    let sec_relro = claim(relro.is_some());
    let sec_rela_relro = claim(relro.is_some_and(|section| !section.relocs.is_empty()));
    let sec_data = claim(data.is_some());
    let sec_rela_data = claim(data.is_some_and(|section| !section.relocs.is_empty()));
    let sec_tdata = claim(tdata.is_some());
    let sec_rela_tdata = claim(tdata.is_some_and(|section| !section.relocs.is_empty()));
    // `.tbss` globals are all-zero and pointer-free by construction, so it
    // never has a `.rela` twin.
    let sec_tbss = claim(tbss.is_some());
    let sec_symtab = claim(true).unwrap();
    let sec_strtab = claim(true).unwrap();
    let _sec_note = claim(true).unwrap();
    let sec_shstrtab = claim(true).unwrap();
    let section_count = next_section;

    // An undefined symbol referenced by a TPREL relocation must still be
    // STT_TLS, or linkers reject the TLS relocation against it.
    let tls_fixup_targets: std::collections::HashSet<&str> = parts
        .fixups
        .iter()
        .filter(|fixup| {
            matches!(fixup.kind, FixupKind::TprelHi12 | FixupKind::TprelLo12Nc)
        })
        .map(|fixup| fixup.symbol.as_str())
        .collect();

    // Text symbols first (so a text-only module keeps its symbol numbering),
    // then each data section's symbols.
    let mut elf_symbols: Vec<ElfSymbol> = parts
        .symbols
        .iter()
        .map(|symbol| ElfSymbol {
            name: symbol.name.clone(),
            info: (if parts.weak_symbols.contains(&symbol.name) {
                STB_WEAK
            } else {
                STB_GLOBAL
            } << 4)
                | if symbol.defined {
                    STT_FUNC
                } else if tls_fixup_targets.contains(symbol.name.as_str()) {
                    STT_TLS
                } else {
                    0
                },
            shndx: if symbol.defined { sec_text } else { SHN_UNDEF },
            value: if symbol.defined { symbol.offset } else { 0 },
            size: 0,
        })
        .collect();
    for (section, index, symbol_type) in [
        (rodata, sec_rodata, STT_OBJECT),
        (relro, sec_relro, STT_OBJECT),
        (data, sec_data, STT_OBJECT),
        (tdata, sec_tdata, STT_TLS),
        (tbss, sec_tbss, STT_TLS),
    ] {
        let (Some(section), Some(index)) = (section, index) else {
            continue;
        };
        for symbol in &section.symbols {
            elf_symbols.push(ElfSymbol {
                name: symbol.name.clone(),
                info: (STB_GLOBAL << 4) | symbol_type,
                shndx: index,
                value: symbol.offset,
                size: symbol.size,
            });
        }
    }
    let symbol_index = |name: &str| -> u64 {
        elf_symbols
            .iter()
            .position(|symbol| symbol.name == name)
            .expect("collect_object_parts registers a symbol for every relocation")
            as u64
            + 1 // + the null symbol
    };

    let mut strtab: Vec<u8> = vec![0];
    let sym_name_offsets: Vec<u32> = elf_symbols
        .iter()
        .map(|symbol| {
            let offset = strtab.len() as u32;
            strtab.extend_from_slice(symbol.name.as_bytes());
            strtab.push(0);
            offset
        })
        .collect();

    // Symbol table: one null entry, then every symbol (all global, so
    // sh_info — the index of the first non-local symbol — is 1).
    let mut symtab = Vec::with_capacity((elf_symbols.len() + 1) * 24);
    symtab.resize(24, 0);
    for (symbol, name_offset) in elf_symbols.iter().zip(&sym_name_offsets) {
        symtab.extend_from_slice(&name_offset.to_le_bytes());
        symtab.push(symbol.info);
        symtab.push(0); // st_other: default visibility
        symtab.extend_from_slice(&symbol.shndx.to_le_bytes());
        symtab.extend_from_slice(&symbol.value.to_le_bytes());
        symtab.extend_from_slice(&symbol.size.to_le_bytes());
    }

    let mut rela_text = Vec::with_capacity(parts.fixups.len() * 24);
    for fixup in &parts.fixups {
        let r_type = match fixup.kind {
            FixupKind::Call26 => R_AARCH64_CALL26,
            FixupKind::AdrpPage21 => R_AARCH64_ADR_PREL_PG_HI21,
            FixupKind::AddLo12 => R_AARCH64_ADD_ABS_LO12_NC,
            FixupKind::TprelHi12 => R_AARCH64_TLSLE_ADD_TPREL_HI12,
            FixupKind::TprelLo12Nc => R_AARCH64_TLSLE_ADD_TPREL_LO12_NC,
        };
        rela_text.extend_from_slice(&u64::from(fixup.offset).to_le_bytes());
        rela_text.extend_from_slice(
            &((symbol_index(&fixup.symbol) << 32) | u64::from(r_type)).to_le_bytes(),
        );
        rela_text.extend_from_slice(&0i64.to_le_bytes());
    }
    let data_rela = |section: &DataSection| -> Vec<u8> {
        let mut rela = Vec::with_capacity(section.relocs.len() * 24);
        for reloc in &section.relocs {
            rela.extend_from_slice(&reloc.offset.to_le_bytes());
            rela.extend_from_slice(
                &((symbol_index(&reloc.symbol) << 32) | u64::from(R_AARCH64_ABS64)).to_le_bytes(),
            );
            rela.extend_from_slice(&reloc.addend.to_le_bytes());
        }
        rela
    };
    let rela_rodata = rodata.map(&data_rela).unwrap_or_default();
    let rela_relro = relro.map(&data_rela).unwrap_or_default();
    let rela_data = data.map(&data_rela).unwrap_or_default();
    let rela_tdata = tdata.map(&data_rela).unwrap_or_default();

    let mut shstrtab: Vec<u8> = vec![0];
    let shstr = |table: &mut Vec<u8>, name: &str| -> u32 {
        let offset = table.len() as u32;
        table.extend_from_slice(name.as_bytes());
        table.push(0);
        offset
    };
    let text_name = shstr(&mut shstrtab, ".text");
    let rela_text_name = shstr(&mut shstrtab, ".rela.text");
    let rodata_name = sec_rodata.map(|_| shstr(&mut shstrtab, ".rodata"));
    let rela_rodata_name = sec_rela_rodata.map(|_| shstr(&mut shstrtab, ".rela.rodata"));
    let relro_name = sec_relro.map(|_| shstr(&mut shstrtab, ".data.rel.ro"));
    let rela_relro_name = sec_rela_relro.map(|_| shstr(&mut shstrtab, ".rela.data.rel.ro"));
    let data_name = sec_data.map(|_| shstr(&mut shstrtab, ".data"));
    let rela_data_name = sec_rela_data.map(|_| shstr(&mut shstrtab, ".rela.data"));
    let tdata_name = sec_tdata.map(|_| shstr(&mut shstrtab, ".tdata"));
    let rela_tdata_name = sec_rela_tdata.map(|_| shstr(&mut shstrtab, ".rela.tdata"));
    let tbss_name = sec_tbss.map(|_| shstr(&mut shstrtab, ".tbss"));
    let symtab_name = shstr(&mut shstrtab, ".symtab");
    let strtab_name = shstr(&mut shstrtab, ".strtab");
    let note_gnu_stack_name = shstr(&mut shstrtab, ".note.GNU-stack");
    let shstrtab_name = shstr(&mut shstrtab, ".shstrtab");

    let header_size = 64u64;
    let text_offset = align(header_size, 4);
    let mut cursor = text_offset + parts.text.len() as u64;
    let mut place = |section: Option<&DataSection>| {
        section.map(|section| {
            cursor = align(cursor, section.align.max(1));
            let offset = cursor;
            cursor += section.bytes.len() as u64;
            offset
        })
    };
    let rodata_offset = place(rodata);
    let relro_offset = place(relro);
    let data_offset = place(data);
    let tdata_offset = place(tdata);
    // `.tbss` is SHT_NOBITS: it has a (conventional) file offset but no file
    // content, so the cursor does not advance past it.
    let tbss_offset = tbss.map(|section| align(cursor, section.align.max(1)));
    let rela_text_offset = align(cursor, 8);
    let rela_rodata_offset = rela_text_offset + rela_text.len() as u64;
    let rela_relro_offset = rela_rodata_offset + rela_rodata.len() as u64;
    let rela_data_offset = rela_relro_offset + rela_relro.len() as u64;
    let rela_tdata_offset = rela_data_offset + rela_data.len() as u64;
    let symtab_offset = align(rela_tdata_offset + rela_tdata.len() as u64, 8);
    let strtab_offset = symtab_offset + symtab.len() as u64;
    let note_offset = strtab_offset + strtab.len() as u64;
    let shstrtab_offset = note_offset;
    let shoff = align(shstrtab_offset + shstrtab.len() as u64, 8);

    let mut out = Vec::with_capacity(shoff as usize + usize::from(section_count) * 64);

    // ELF header.
    out.extend_from_slice(&[0x7f, b'E', b'L', b'F', 2, 1, 1, 0]); // 64-bit LE SysV
    out.resize(16, 0);
    out.extend_from_slice(&ET_REL.to_le_bytes());
    out.extend_from_slice(&EM_AARCH64.to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes()); // e_version
    out.extend_from_slice(&0u64.to_le_bytes()); // e_entry
    out.extend_from_slice(&0u64.to_le_bytes()); // e_phoff
    out.extend_from_slice(&shoff.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // e_flags
    out.extend_from_slice(&(header_size as u16).to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // e_phentsize
    out.extend_from_slice(&0u16.to_le_bytes()); // e_phnum
    out.extend_from_slice(&64u16.to_le_bytes()); // e_shentsize
    out.extend_from_slice(&section_count.to_le_bytes());
    out.extend_from_slice(&sec_shstrtab.to_le_bytes());

    pad_to(&mut out, text_offset);
    out.extend_from_slice(&parts.text);
    for (section, offset) in [
        (rodata, rodata_offset),
        (relro, relro_offset),
        (data, data_offset),
        (tdata, tdata_offset),
    ] {
        if let (Some(section), Some(offset)) = (section, offset) {
            pad_to(&mut out, offset);
            out.extend_from_slice(&section.bytes);
        }
    }
    pad_to(&mut out, rela_text_offset);
    out.extend_from_slice(&rela_text);
    out.extend_from_slice(&rela_rodata);
    out.extend_from_slice(&rela_relro);
    out.extend_from_slice(&rela_data);
    out.extend_from_slice(&rela_tdata);
    pad_to(&mut out, symtab_offset);
    out.extend_from_slice(&symtab);
    out.extend_from_slice(&strtab);
    out.extend_from_slice(&shstrtab);
    pad_to(&mut out, shoff);

    let section = |out: &mut Vec<u8>,
                       name: u32,
                       sh_type: u32,
                       flags: u64,
                       offset: u64,
                       size: u64,
                       link: u32,
                       info: u32,
                       addralign: u64,
                       entsize: u64| {
        out.extend_from_slice(&name.to_le_bytes());
        out.extend_from_slice(&sh_type.to_le_bytes());
        out.extend_from_slice(&flags.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes()); // sh_addr
        out.extend_from_slice(&offset.to_le_bytes());
        out.extend_from_slice(&size.to_le_bytes());
        out.extend_from_slice(&link.to_le_bytes());
        out.extend_from_slice(&info.to_le_bytes());
        out.extend_from_slice(&addralign.to_le_bytes());
        out.extend_from_slice(&entsize.to_le_bytes());
    };

    section(&mut out, 0, 0, 0, 0, 0, 0, 0, 0, 0); // SHT_NULL
    section(
        &mut out,
        text_name,
        SHT_PROGBITS,
        SHF_ALLOC | SHF_EXECINSTR,
        text_offset,
        parts.text.len() as u64,
        0,
        0,
        4,
        0,
    );
    section(
        &mut out,
        rela_text_name,
        SHT_RELA,
        0,
        rela_text_offset,
        rela_text.len() as u64,
        u32::from(sec_symtab),
        u32::from(sec_text),
        8,
        24,
    );
    for (data_section, target, name, offset, flags, rela, rela_name, rela_offset) in [
        (
            rodata,
            sec_rodata,
            rodata_name,
            rodata_offset,
            SHF_ALLOC,
            &rela_rodata,
            rela_rodata_name,
            rela_rodata_offset,
        ),
        // Writable so the dynamic linker can fill the pointer slots; the
        // `.data.rel.ro` name is what makes linkers move it into the
        // read-only-after-relocation segment.
        (
            relro,
            sec_relro,
            relro_name,
            relro_offset,
            SHF_ALLOC | SHF_WRITE,
            &rela_relro,
            rela_relro_name,
            rela_relro_offset,
        ),
        (
            data,
            sec_data,
            data_name,
            data_offset,
            SHF_ALLOC | SHF_WRITE,
            &rela_data,
            rela_data_name,
            rela_data_offset,
        ),
        // The TLS initialization image; linkers gather SHF_TLS sections into
        // the PT_TLS segment each thread's block is instantiated from.
        (
            tdata,
            sec_tdata,
            tdata_name,
            tdata_offset,
            SHF_ALLOC | SHF_WRITE | SHF_TLS,
            &rela_tdata,
            rela_tdata_name,
            rela_tdata_offset,
        ),
    ] {
        let Some(data_section) = data_section else {
            continue;
        };
        section(
            &mut out,
            name.unwrap(),
            SHT_PROGBITS,
            flags,
            offset.unwrap(),
            data_section.bytes.len() as u64,
            0,
            0,
            data_section.align.max(1),
            0,
        );
        if let Some(rela_name) = rela_name {
            section(
                &mut out,
                rela_name,
                SHT_RELA,
                0,
                rela_offset,
                rela.len() as u64,
                u32::from(sec_symtab),
                u32::from(target.unwrap()),
                8,
                24,
            );
        }
    }
    if let Some(tbss_section) = tbss {
        // Zero-initialized TLS: SHT_NOBITS, so sh_size counts toward the
        // thread block but the file carries no bytes.
        section(
            &mut out,
            tbss_name.unwrap(),
            SHT_NOBITS,
            SHF_ALLOC | SHF_WRITE | SHF_TLS,
            tbss_offset.unwrap(),
            tbss_section.bytes.len() as u64,
            0,
            0,
            tbss_section.align.max(1),
            0,
        );
    }
    section(
        &mut out,
        symtab_name,
        SHT_SYMTAB,
        0,
        symtab_offset,
        symtab.len() as u64,
        u32::from(sec_strtab),
        1, // first (and only) local symbol is the null entry
        8,
        24,
    );
    section(
        &mut out,
        strtab_name,
        SHT_STRTAB,
        0,
        strtab_offset,
        strtab.len() as u64,
        0,
        0,
        1,
        0,
    );
    section(
        &mut out,
        note_gnu_stack_name,
        SHT_PROGBITS,
        0,
        note_offset,
        0,
        0,
        0,
        1,
        0,
    );
    section(
        &mut out,
        shstrtab_name,
        SHT_STRTAB,
        0,
        shstrtab_offset,
        shstrtab.len() as u64,
        0,
        0,
        1,
        0,
    );

    out
}

fn align(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) & !(alignment - 1)
}

fn pad_to(out: &mut Vec<u8>, len: u64) {
    if (out.len() as u64) < len {
        out.resize(len as usize, 0);
    }
}

#[cfg(test)]
mod tests {
    use crate::dialects::aarch64::op_interfaces::{BinaryFixup, FixupKind};
    use crate::dialects::macho::ops::Symbol;
    use crate::ll::DataReloc;

    use super::super::aarch64_object_lower::{DataSection, DataSymbol, ObjectParts};
    use super::write_elf_object;

    fn header_u16(bytes: &[u8], offset: usize) -> u16 {
        u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap())
    }

    /// The section headers of an emitted object as (name, sh_type, flags,
    /// size, link, info) rows.
    fn section_headers(bytes: &[u8]) -> Vec<(String, u32, u64, u64, u32, u32)> {
        let shoff = u64::from_le_bytes(bytes[40..48].try_into().unwrap()) as usize;
        let shnum = header_u16(bytes, 60) as usize;
        let shstrndx = header_u16(bytes, 62) as usize;
        let strtab_offset = {
            let header = &bytes[shoff + shstrndx * 64..];
            u64::from_le_bytes(header[24..32].try_into().unwrap()) as usize
        };
        (0..shnum)
            .map(|index| {
                let header = &bytes[shoff + index * 64..shoff + (index + 1) * 64];
                let name_offset =
                    strtab_offset + u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
                let name_end = bytes[name_offset..]
                    .iter()
                    .position(|byte| *byte == 0)
                    .unwrap()
                    + name_offset;
                (
                    String::from_utf8(bytes[name_offset..name_end].to_vec()).unwrap(),
                    u32::from_le_bytes(header[4..8].try_into().unwrap()),
                    u64::from_le_bytes(header[8..16].try_into().unwrap()),
                    u64::from_le_bytes(header[32..40].try_into().unwrap()),
                    u32::from_le_bytes(header[40..44].try_into().unwrap()),
                    u32::from_le_bytes(header[44..48].try_into().unwrap()),
                )
            })
            .collect()
    }

    #[test]
    fn text_only_layout_is_unchanged() {
        let parts = ObjectParts {
            text: vec![0xc0, 0x03, 0x5f, 0xd6],
            symbols: vec![Symbol {
                name: "main".to_string(),
                offset: 0,
                external: true,
                defined: true,
            }],
            fixups: vec![],
            weak_symbols: Default::default(),
            rodata: None,
            relro: None,
            data: None,
            tdata: None,
            tbss: None,
        };
        let bytes = write_elf_object(&parts);
        let names: Vec<String> = section_headers(&bytes)
            .into_iter()
            .map(|(name, ..)| name)
            .collect();
        assert_eq!(
            names,
            ["", ".text", ".rela.text", ".symtab", ".strtab", ".note.GNU-stack", ".shstrtab"]
        );
        assert_eq!(header_u16(&bytes, 62), 6); // e_shstrndx
    }

    #[test]
    fn data_sections_get_headers_symbols_and_abs64_relocations() {
        let parts = ObjectParts {
            text: vec![0xc0, 0x03, 0x5f, 0xd6],
            symbols: vec![Symbol {
                name: "main".to_string(),
                offset: 0,
                external: true,
                defined: true,
            }],
            weak_symbols: Default::default(),
            fixups: vec![BinaryFixup {
                offset: 0,
                symbol: "holder".to_string(),
                kind: FixupKind::AdrpPage21,
            }],
            rodata: Some(DataSection {
                align: 8,
                bytes: 42u64.to_le_bytes().to_vec(),
                symbols: vec![DataSymbol {
                    name: "answer".to_string(),
                    offset: 0,
                    size: 8,
                }],
                relocs: vec![],
            }),
            relro: Some(DataSection {
                align: 8,
                bytes: vec![0; 8],
                symbols: vec![DataSymbol {
                    name: "holder".to_string(),
                    offset: 0,
                    size: 8,
                }],
                relocs: vec![DataReloc {
                    offset: 0,
                    symbol: "answer".to_string(),
                    addend: 4,
                }],
            }),
            data: None,
            tdata: None,
            tbss: None,
        };
        let bytes = write_elf_object(&parts);
        let headers = section_headers(&bytes);
        let names: Vec<&str> = headers.iter().map(|(name, ..)| name.as_str()).collect();
        assert_eq!(
            names,
            [
                "",
                ".text",
                ".rela.text",
                ".rodata",
                ".data.rel.ro",
                ".rela.data.rel.ro",
                ".symtab",
                ".strtab",
                ".note.GNU-stack",
                ".shstrtab"
            ]
        );
        // .rodata is read-only allocatable; .data.rel.ro stays writable for
        // the dynamic linker.
        assert_eq!(headers[3].2, 0x2);
        assert_eq!(headers[4].2, 0x3);
        // .rela.data.rel.ro: SHT_RELA linked to .symtab, targeting section 4.
        assert_eq!(headers[5].1, 4);
        assert_eq!(headers[5].4, 6);
        assert_eq!(headers[5].5, 4);

        // Its single entry: r_offset 0, symbol `answer` (index 2: null,
        // main, answer), type R_AARCH64_ABS64 (257), addend 4.
        let shoff = u64::from_le_bytes(bytes[40..48].try_into().unwrap()) as usize;
        let rela_offset = {
            let header = &bytes[shoff + 5 * 64..];
            u64::from_le_bytes(header[24..32].try_into().unwrap()) as usize
        };
        let rela = &bytes[rela_offset..rela_offset + 24];
        assert_eq!(u64::from_le_bytes(rela[0..8].try_into().unwrap()), 0);
        assert_eq!(
            u64::from_le_bytes(rela[8..16].try_into().unwrap()),
            (2 << 32) | 257
        );
        assert_eq!(i64::from_le_bytes(rela[16..24].try_into().unwrap()), 4);
    }
}
