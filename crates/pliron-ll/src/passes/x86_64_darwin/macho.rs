use std::collections::HashMap;

use crate::{
    context::Context,
    dialects::macho::ops::{ObjectOp, Symbol},
};

const MH_MAGIC_64: u32 = 0xfeedfacf;
const CPU_TYPE_X86_64: u32 = 0x01000007;
const CPU_SUBTYPE_X86_64_ALL: u32 = 3;
const MH_OBJECT: u32 = 1;
const LC_SEGMENT_64: u32 = 0x19;
const LC_SYMTAB: u32 = 0x2;
const LC_BUILD_VERSION: u32 = 0x32;
const PLATFORM_MACOS: u32 = 1;
const VM_PROT_READ: u32 = 0x1;
const VM_PROT_WRITE: u32 = 0x2;
const VM_PROT_EXECUTE: u32 = 0x4;
const N_EXT: u8 = 0x01;
const N_UNDF: u8 = 0x00;
const N_SECT: u8 = 0x0e;
const S_ATTR_PURE_INSTRUCTIONS: u32 = 0x80000000;
const S_ATTR_SOME_INSTRUCTIONS: u32 = 0x00000400;

pub fn write_macho_object(ctx: &Context, object: ObjectOp) -> Vec<u8> {
    let text = object.text(ctx);
    let symbols = object.symbols(ctx);
    let relocations = object.relocations(ctx);
    let nsects = 1u32;
    let segment_cmd_size = 72 + 80 * nsects;
    let symtab_cmd_size = 24u32;
    let build_version_cmd_size = 24u32;
    let sizeofcmds = segment_cmd_size + symtab_cmd_size + build_version_cmd_size;
    let header_size = 32u32;
    let text_offset = align(header_size + sizeofcmds, 4);
    let text_size = text.len() as u32;
    let reloff = align(text_offset + text_size, 4);
    let nreloc = relocations.len() as u32;
    let symoff = align(reloff + nreloc * 8, 8);
    let nsyms = symbols.len() as u32;
    let stroff = symoff + nsyms * 16;
    let string_table = build_string_table(&symbols);
    let strsize = string_table.len() as u32;

    let mut out = Vec::with_capacity((stroff + strsize) as usize);

    write_u32(&mut out, MH_MAGIC_64);
    write_u32(&mut out, CPU_TYPE_X86_64);
    write_u32(&mut out, CPU_SUBTYPE_X86_64_ALL);
    write_u32(&mut out, MH_OBJECT);
    write_u32(&mut out, 3);
    write_u32(&mut out, sizeofcmds);
    write_u32(&mut out, 0);
    write_u32(&mut out, 0);

    write_u32(&mut out, LC_SEGMENT_64);
    write_u32(&mut out, segment_cmd_size);
    write_fixed_str(&mut out, "", 16);
    write_u64(&mut out, 0);
    write_u64(&mut out, text_size as u64);
    write_u64(&mut out, text_offset as u64);
    write_u64(&mut out, text_size as u64);
    write_u32(&mut out, VM_PROT_READ | VM_PROT_WRITE | VM_PROT_EXECUTE);
    write_u32(&mut out, VM_PROT_READ | VM_PROT_EXECUTE);
    write_u32(&mut out, nsects);
    write_u32(&mut out, 0);

    write_fixed_str(&mut out, "__text", 16);
    write_fixed_str(&mut out, "__TEXT", 16);
    write_u64(&mut out, 0);
    write_u64(&mut out, text_size as u64);
    write_u32(&mut out, text_offset);
    write_u32(&mut out, 2);
    write_u32(&mut out, reloff);
    write_u32(&mut out, nreloc);
    write_u32(
        &mut out,
        S_ATTR_PURE_INSTRUCTIONS | S_ATTR_SOME_INSTRUCTIONS,
    );
    write_u32(&mut out, 0);
    write_u32(&mut out, 0);
    write_u32(&mut out, 0);

    write_u32(&mut out, LC_SYMTAB);
    write_u32(&mut out, symtab_cmd_size);
    write_u32(&mut out, symoff);
    write_u32(&mut out, nsyms);
    write_u32(&mut out, stroff);
    write_u32(&mut out, strsize);

    write_u32(&mut out, LC_BUILD_VERSION);
    write_u32(&mut out, build_version_cmd_size);
    write_u32(&mut out, PLATFORM_MACOS);
    write_u32(&mut out, encoded_version(11, 0, 0));
    write_u32(&mut out, 0);
    write_u32(&mut out, 0);

    pad_to(&mut out, text_offset as usize);
    out.extend_from_slice(&text);
    pad_to(&mut out, reloff as usize);

    let symbol_indices = symbol_indices(&symbols);
    for relocation in &relocations {
        write_u32(&mut out, relocation.offset);
        let symbolnum = symbol_indices
            .get(&relocation.symbol)
            .copied()
            .unwrap_or_default()
            & 0x00ff_ffff;
        let info = symbolnum
            | ((relocation.pcrel as u32) << 24)
            | (((relocation.length as u32) & 0x3) << 25)
            | ((relocation.extern_ as u32) << 27)
            | (((relocation.kind as u32) & 0xf) << 28);
        write_u32(&mut out, info);
    }
    pad_to(&mut out, symoff as usize);

    let mut strx = 1u32;
    for symbol in &symbols {
        write_u32(&mut out, strx);
        write_u8(
            &mut out,
            if symbol.defined { N_SECT } else { N_UNDF } | if symbol.external { N_EXT } else { 0 },
        );
        write_u8(&mut out, if symbol.defined { 1 } else { 0 });
        write_u16(&mut out, 0);
        write_u64(&mut out, if symbol.defined { symbol.offset } else { 0 });
        strx += symbol.name.len() as u32 + 1;
    }

    pad_to(&mut out, stroff as usize);
    out.extend_from_slice(&string_table);
    out
}

fn symbol_indices(symbols: &[Symbol]) -> HashMap<String, u32> {
    symbols
        .iter()
        .enumerate()
        .map(|(idx, symbol)| (symbol.name.clone(), idx as u32))
        .collect()
}

fn build_string_table(symbols: &[crate::dialects::macho::ops::Symbol]) -> Vec<u8> {
    let mut table = vec![0];
    for symbol in symbols {
        table.extend_from_slice(symbol.name.as_bytes());
        table.push(0);
    }
    table
}

fn align(value: u32, alignment: u32) -> u32 {
    (value + alignment - 1) & !(alignment - 1)
}

fn encoded_version(major: u32, minor: u32, patch: u32) -> u32 {
    (major << 16) | (minor << 8) | patch
}

fn pad_to(out: &mut Vec<u8>, len: usize) {
    if out.len() < len {
        out.resize(len, 0);
    }
}

fn write_fixed_str(out: &mut Vec<u8>, value: &str, width: usize) {
    let bytes = value.as_bytes();
    let len = bytes.len().min(width);
    out.extend_from_slice(&bytes[..len]);
    out.resize(out.len() + (width - len), 0);
}

fn write_u8(out: &mut Vec<u8>, value: u8) {
    out.push(value);
}

fn write_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn write_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn write_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}
