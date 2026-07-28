//! Minimal ELF64 relocatable-object writer for the aarch64 backend, the
//! Linux counterpart of the Mach-O writer in [super::macho]. It emits a
//! single `.text` section plus `.rela.text`, `.symtab`, the string tables,
//! and an empty `.note.GNU-stack` (so linkers don't assume an executable
//! stack).

use crate::dialects::aarch64::op_interfaces::FixupKind;

use super::aarch64_object_lower::ObjectParts;

const EM_AARCH64: u16 = 183;
const ET_REL: u16 = 1;
const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const SHT_RELA: u32 = 4;
const SHF_ALLOC: u64 = 0x2;
const SHF_EXECINSTR: u64 = 0x4;
const STB_GLOBAL: u8 = 1;
const STT_FUNC: u8 = 2;
const SHN_UNDEF: u16 = 0;
const R_AARCH64_CALL26: u32 = 283;

// Section header table layout (fixed order; index 0 is the SHT_NULL entry,
// remaining indices are used in sh_link/sh_info).
const SEC_TEXT: u16 = 1;
const SEC_SYMTAB: u16 = 3;
const SEC_STRTAB: u16 = 4;
const SEC_SHSTRTAB: u16 = 6;
const SECTION_COUNT: u16 = 7;

pub(super) fn write_elf_object(parts: &ObjectParts) -> Vec<u8> {
    let mut strtab: Vec<u8> = vec![0];
    let sym_name_offsets: Vec<u32> = parts
        .symbols
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
    let mut symtab = Vec::with_capacity((parts.symbols.len() + 1) * 24);
    symtab.resize(24, 0);
    for (symbol, name_offset) in parts.symbols.iter().zip(&sym_name_offsets) {
        symtab.extend_from_slice(&name_offset.to_le_bytes());
        symtab.push((STB_GLOBAL << 4) | if symbol.defined { STT_FUNC } else { 0 });
        symtab.push(0); // st_other: default visibility
        symtab.extend_from_slice(
            &if symbol.defined { SEC_TEXT } else { SHN_UNDEF }.to_le_bytes(),
        );
        symtab.extend_from_slice(&if symbol.defined { symbol.offset } else { 0 }.to_le_bytes());
        symtab.extend_from_slice(&0u64.to_le_bytes()); // st_size
    }

    let mut rela = Vec::with_capacity(parts.fixups.len() * 24);
    for fixup in &parts.fixups {
        let sym_index = parts
            .symbols
            .iter()
            .position(|symbol| symbol.name == fixup.symbol)
            .expect("collect_object_parts registers a symbol for every fixup")
            as u64
            + 1; // + the null symbol
        let r_type = match fixup.kind {
            FixupKind::Call26 => R_AARCH64_CALL26,
        };
        rela.extend_from_slice(&u64::from(fixup.offset).to_le_bytes());
        rela.extend_from_slice(&((sym_index << 32) | u64::from(r_type)).to_le_bytes());
        rela.extend_from_slice(&0i64.to_le_bytes());
    }

    let mut shstrtab: Vec<u8> = vec![0];
    let shstr = |table: &mut Vec<u8>, name: &str| -> u32 {
        let offset = table.len() as u32;
        table.extend_from_slice(name.as_bytes());
        table.push(0);
        offset
    };
    let text_name = shstr(&mut shstrtab, ".text");
    let rela_text_name = shstr(&mut shstrtab, ".rela.text");
    let symtab_name = shstr(&mut shstrtab, ".symtab");
    let strtab_name = shstr(&mut shstrtab, ".strtab");
    let note_gnu_stack_name = shstr(&mut shstrtab, ".note.GNU-stack");
    let shstrtab_name = shstr(&mut shstrtab, ".shstrtab");

    let header_size = 64u64;
    let text_offset = align(header_size, 4);
    let rela_offset = align(text_offset + parts.text.len() as u64, 8);
    let symtab_offset = align(rela_offset + rela.len() as u64, 8);
    let strtab_offset = symtab_offset + symtab.len() as u64;
    let note_offset = strtab_offset + strtab.len() as u64;
    let shstrtab_offset = note_offset;
    let shoff = align(shstrtab_offset + shstrtab.len() as u64, 8);

    let mut out = Vec::with_capacity(shoff as usize + usize::from(SECTION_COUNT) * 64);

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
    out.extend_from_slice(&SECTION_COUNT.to_le_bytes());
    out.extend_from_slice(&SEC_SHSTRTAB.to_le_bytes());

    pad_to(&mut out, text_offset);
    out.extend_from_slice(&parts.text);
    pad_to(&mut out, rela_offset);
    out.extend_from_slice(&rela);
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
        rela_offset,
        rela.len() as u64,
        u32::from(SEC_SYMTAB),
        u32::from(SEC_TEXT),
        8,
        24,
    );
    section(
        &mut out,
        symtab_name,
        SHT_SYMTAB,
        0,
        symtab_offset,
        symtab.len() as u64,
        u32::from(SEC_STRTAB),
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
