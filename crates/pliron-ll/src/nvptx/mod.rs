//! NVPTX kernel emission: translates `llvm`-dialect functions directly to
//! PTX text, with no NVVM or LLVM library behind it.
//!
//! PTX is a virtual ISA with unlimited typed virtual registers; `ptxas` owns
//! register allocation, scheduling, and encoding. Emission is therefore a
//! translation that sits outside the pass pipeline, the way the Mach-O/ELF
//! writers do, rather than a machine pipeline of its own.
//!
//! Value representation: `i1` lives in a `.pred` register, `i8`/`i16`/`i32`
//! in a `.b32` register, and `i64`/pointers in a `.b64` register. Sub-32-bit
//! integers are kept zero-extended in their `.b32` register; operations that
//! are sensitive to the upper bits (signed compares, signed division,
//! arithmetic shift) re-sign-extend explicitly, and operations that can
//! overflow the narrow width mask afterwards.
//!
//! State spaces: a per-kernel pointer-provenance analysis (the `spaces`
//! submodule) proves addresses global or shared, and loads/stores/atomics
//! through a proven address are space-qualified (`ld.global`, `st.shared`, `atom.global.…`);
//! unproven or mixed provenance keeps the generic forms. The representation
//! invariant: a value proven `.shared` holds the RAW shared-window address —
//! its `llvm.addressof` skips the `cvta.shared.u64` — and every flow of such
//! a value into a non-shared slot (a demoted gep result or aliasing cast, a
//! block-argument copy, a select arm) inserts the `cvta.shared.u64` at that
//! boundary instead. Global addresses need no conversion either way: the
//! global window is identity-mapped in generic space (kernel pointer params
//! arrive as global addresses and have always been used with generic
//! `ld`/`st` here), so one register serves both the `.global`-qualified and
//! the generic forms.
//!
//! FP contraction: an `fadd` whose operand is a single-use `fmul` of the
//! same type contracts to `fma.rn` (the fused mul emits nothing), mirroring
//! nvcc's default `-fmad=true`. Like nvcc's, the contracted result may
//! differ from the two-instruction rounding; kernels needing exact
//! `mul`+`add` semantics should use explicit intrinsics.
//!
//! Immediate operands: integer constants are not materialized at their
//! definition. They fold into the using instruction as PTX immediates
//! (arithmetic/logic/compare right operands, shift amounts, select arms,
//! block-argument copies, GEP indices); a use that needs a register (an
//! address, a stored value, an intrinsic argument) materializes a fresh
//! `mov` at that use, which always dominates itself.
//!
//! Memory operands fold constant byte offsets: an address decomposed by
//! the `addr` submodule into `root + Σ dynamic·scale + const` is emitted
//! as `ld/st [%base + const]` with ONE base register per distinct
//! `(root, dynamics)` in a block, instead of a fresh register per access.
//! This works in every state space (`.shared`, `.global`, generic) and is
//! what keeps a fully unrolled tile loop's address file at two registers
//! rather than thirty-two (`ptxas` pre-hoists whatever addresses PTX
//! materializes; see the soundness notes in `addr`). GEPs whose every use
//! folds away — and the pure integer chains feeding only such GEPs — emit
//! nothing at all.
//!
//! Vectorized shared loads: within one block, loads from the same folded
//! base at consecutive constant offsets and with no intervening store,
//! call, or barrier group into `ld.shared.v4.f32/u32` (or `.v2`) when the
//! 16-byte (8 for v2.32) alignment of the group start is provable: the
//! shared global's declaration alignment is raised as needed, and every
//! dynamic term's scale must keep the required alignment. GLOBAL loads
//! are deliberately NOT vectorized: kernel parameters are raw `*const T`
//! pointers whose runtime value only promises the element alignment
//! (a caller may pass `buf.add(1)`), so a 16-byte-aligned-base proof is
//! not available; a misaligned vector access is a hard fault. Stores are
//! not vectorized either — per-thread store patterns in the corpus never
//! form same-thread consecutive groups, so there is no win to buy.
//!
//! `ld.global.nc` (the read-only data cache): a global load takes `.nc`
//! when the per-root write-set analysis ([spaces::readonly_values]) proves
//! the loaded memory is written by NO store, atomic, or opaque call
//! anywhere in the kernel body. Every thread runs this same body, so the
//! proof covers the whole launch — exactly `.nc`'s requirement (the
//! read-only cache does not snoop stores within a launch; it is
//! invalidated between launches). Provenance roots are kernel pointer
//! parameters, module globals, allocas, and the dyn-shared window;
//! derivations the analysis cannot track (loaded pointers, opaque integer
//! round-trips) poison to Unknown, and a store through an Unknown address
//! — or any call that could write memory — dirties every root. An earlier
//! revision refused `.nc` outright on the grounds that nvcc requires
//! `const` + `__restrict__`; that is true of nvcc (verified with
//! `nvcc -arch=sm_121 -ptx` on the real corpus: iq3_xxs_dequant 26/26,
//! q4_0_dequant 6/6 and comp_pool 73/73 global loads take `.nc`, all
//! through `const T* __restrict__` params, while adamw's and
//! soft_max_f32's plain `const T*` params get none), but the conclusion
//! — skip — was wrong: the promise `__restrict__` supplies in C is
//! supplied here by the kernel ABI itself. Its raw pointers derive from
//! the `&[T]`/`&mut [T]` borrows the launch split (docs/KERNEL-ABI.md),
//! so a written region overlapping a read-only parameter is launch-side
//! UB — Rust gives every read-only parameter what nvcc must be told per
//! parameter. On the `__restrict__` kernels above, this per-root
//! criterion reproduces nvcc's `.nc` placements exactly. Entry kernels
//! only — a device `.func` cannot see its callers' stores to module
//! globals. Kill switch: `CRABBIT_NVPTX_NC=0`.
//!
//! Sub-word load combining: narrow (u8/u16) integer loads from the SAME
//! folded base whose constant offsets fully cover one aligned 4-byte
//! window combine into a single 32-bit load plus one extract per original
//! load (`prmt.b32 d, w, 0, 0x444j` per byte — ptxas keeps it as PRMT,
//! the exact shape nvcc emits for byte-table reads — and `and`/`shr` for
//! halves). Soundness: the window must be read-only-proven (the `.nc`
//! analysis above), which subsumes every intervening-write question, so a
//! group may span blocks and interleave with unrelated stores — the shape
//! iq3_xxs_dequant's fully unrolled codebook reads take; the leader (the
//! group's first load in program order) must dominate every member, since
//! its window register feeds their extracts; and the 4-byte alignment
//! proof needs every dynamic term scale ≡ 0 (mod 4) plus a root whose
//! declaration this module owns — a module global, its `.align` raised to
//! 4 — because raw-pointer parameters promise only element alignment.
//! Windows only partially covered by actual loads stay scalar (the
//! combined load must never touch a byte the kernel didn't). Kill switch:
//! `CRABBIT_NVPTX_BYTECOMB=0`.
//!
//! Allocas that survive mem2reg (small arrays, address-taken locals)
//! lower to per-function `.local` arrays; the resulting address is
//! converted with `cvta.local.u64` once at the definition and then flows
//! as an ordinary generic address (local residue is cold by construction,
//! so the generic access forms are fine there).
//!
//! Device functions: internal-linkage definitions that survive the
//! inliner (too big for its budget, or address-diverse shapes it skips)
//! are emitted as PTX `.func`s with the `.param` ABI — scalar params in
//! `.param` space, at most one scalar result — and calls to them become
//! `call.uni` sequences. Their pointer parameters carry no provenance
//! (callers may pass any state space), so their memory accesses use the
//! generic forms unless provenance is proven locally.

mod addr;
mod spaces;

use std::collections::{BTreeMap, HashMap, HashSet};

use addr::{AddrFolder, FoldedAddr, Term, TermKind};
use spaces::PtrSpace;

use thiserror::Error;

use pliron::builtin::op_interfaces::{
    AtMostOneRegionInterface as _, BranchOpInterface as _, CallOpCallable, CallOpInterface as _,
    OneOpdInterface as _, OneRegionInterface, OneResultInterface, SymbolOpInterface,
};
use pliron_llvm::op_interfaces::{IsDeclaration, PointerTypeResult};

use crate::{
    context::{Context, Ptr},
    dialects::{
        builtin::{
            attributes::{FPDoubleAttr, FPSingleAttr, IntegerAttr},
            ops::ModuleOp,
            types::{FP32Type, FP64Type, IntegerType},
        },
        llvm::{
            attributes::{FCmpPredicateAttr, ICmpPredicateAttr},
            ops::{
                AShrOp, AddOp, AddressOfOp, AllocaOp, AndOp, BitcastOp, BrOp, CallOp, CondBrOp,
                FAddOp, FCmpOp, FDivOp, FMulOp, FNegOp, FPExtOp, FPToSIOp, FPToUIOp, FPTruncOp,
                ExtractValueOp, FRemOp, FSubOp, FuncOp as LlvmFuncOp, GepIndex, GetElementPtrOp,
                GlobalOp, ICmpOp, InsertValueOp, IntToPtrOp, LShrOp, LoadOp, MulOp, OrOp,
                PoisonOp, PtrToIntOp, ReturnOp,
                SDivOp, SExtOp, SIToFPOp, SRemOp, SelectOp, ShlOp, StoreOp, SubOp, TruncOp,
                UDivOp, UIToFPOp, URemOp, UndefOp, UnreachableOp, XorOp, ZExtOp,
            },
            types::{ArrayType, PointerType, StructType, VoidType},
        },
    },
    input_error_noloc,
    ir::{basic_block::BasicBlock, op::Op, operation::Operation, r#type::Typed, value::Value},
    linked_list::ContainsLinkedList,
    printable::Printable,
    r#type::TypeHandle,
    result::CrabbitResult,
};

#[derive(Debug, Error)]
pub enum NvptxErr {
    #[error("expected builtin.module as NVPTX emission root")]
    NotModule,
    #[error("unsupported LLVM operation for NVPTX emission: {0}")]
    UnsupportedOp(String),
    #[error("unsupported LLVM type for NVPTX emission: {0}")]
    UnsupportedType(String),
    #[error("value was used before NVPTX emission defined it: {0}")]
    UndefinedValue(String),
    #[error("unsupported global for NVPTX emission: {0}")]
    UnsupportedGlobal(String),
}

/// The PTX module header parameters: target SM and PTX ISA version.
///
/// The default targets `sm_121` (GB10 / DGX Spark) with PTX ISA 8.8, the
/// first ISA revision that knows that SM.
pub struct PtxTarget {
    pub sm: u32,
    pub ptx_isa: (u32, u32),
}

impl Default for PtxTarget {
    fn default() -> Self {
        PtxTarget {
            sm: 121,
            ptx_isa: (8, 8),
        }
    }
}

/// Translate every defined `llvm.func` in the module rooted at `root` into a
/// PTX `.entry` kernel and return the complete PTX module text.
pub fn write_ptx_from_ir(
    ctx: &Context,
    root: Ptr<Operation>,
    target: &PtxTarget,
) -> CrabbitResult<String> {
    Ok(write_ptx_and_linemap_from_ir(ctx, root, target)?.0)
}

/// True when PTX linemap emission is requested (`CRABBIT_PTX_LINEMAP` or
/// the umbrella `CRABBIT_PROFILE_MAP`). The linemap never alters the
/// emitted PTX; the gate only controls whether the sidecar JSON is built.
pub fn linemap_enabled() -> bool {
    ["CRABBIT_PTX_LINEMAP", "CRABBIT_PROFILE_MAP"].iter().any(|var| {
        std::env::var(var).is_ok_and(|value| !value.is_empty() && value != "0")
    })
}

/// [write_ptx_from_ir], additionally returning the PTX linemap JSON (the
/// GPU analogue of the machine op-map, docs/PROFILE-FEEDBACK-BACKWARD.md)
/// when [linemap_enabled]: per `.entry`, a map from ABSOLUTE 1-based line
/// numbers in the returned PTX text to the source-level llvm op ids that
/// line lowers (`[id, …]`) or a synthetic root (`"ptx:entry"`,
/// `"ptx:label"`, `"ptx:decl"`, `"ptx:unattributed"`). `midend` is `null`
/// by design: the kernel pipeline carries a single (source) numbering —
/// mid-end adjoints are resolved at emission via `effective_sources`.
pub fn write_ptx_and_linemap_from_ir(
    ctx: &Context,
    root: Ptr<Operation>,
    target: &PtxTarget,
) -> CrabbitResult<(String, Option<String>)> {
    write_ptx_linemap_inner(ctx, root, target, linemap_enabled())
}

/// [write_ptx_and_linemap_from_ir] with the linemap forced on regardless
/// of environment — for callers (the analysis server) whose enablement
/// comes from per-run config rather than process env.
pub fn write_ptx_with_forced_linemap(
    ctx: &Context,
    root: Ptr<Operation>,
    target: &PtxTarget,
) -> CrabbitResult<(String, String)> {
    let (ptx, map) = write_ptx_linemap_inner(ctx, root, target, true)?;
    Ok((ptx, map.expect("linemap forced on")))
}

fn write_ptx_linemap_inner(
    ctx: &Context,
    root: Ptr<Operation>,
    target: &PtxTarget,
    want_linemap: bool,
) -> CrabbitResult<(String, Option<String>)> {
    let root_op = Operation::get_op_dyn(root, ctx);
    let module = root_op
        .downcast_ref::<ModuleOp>()
        .ok_or_else(|| input_error_noloc!(NvptxErr::NotModule))?;
    let body = module
        .get_region(ctx)
        .deref(ctx)
        .get_head()
        .expect("builtin.module has a body block");

    // Module-level globals first (in the OUTPUT): their state space
    // decides how `llvm.addressof` materializes the address inside
    // kernels. Their text renders only after the functions, though,
    // because a vectorized shared load may raise a global's declared
    // alignment; the raise never changes the declaration's line count.
    let mut globals: HashMap<String, GlobalSpace> = HashMap::new();
    let mut global_ptrs: Vec<Ptr<Operation>> = Vec::new();
    let no_raises: HashMap<String, u64> = HashMap::new();
    for op_ptr in body.deref(ctx).iter(ctx) {
        let op_obj = Operation::get_op_dyn(op_ptr, ctx);
        if let Some(global) = op_obj.downcast_ref::<GlobalOp>() {
            let (name, space, _) = emit_global(ctx, global, &no_raises)?;
            globals.insert(name, space);
            global_ptrs.push(op_ptr);
        }
    }

    // Device functions are the internal-linkage definitions the inliner
    // left behind; they are emitted as `.func`s and everything else that
    // is defined becomes a `.entry` kernel. Forward-declare every device
    // function first so call sites and bodies may appear in any order.
    let mut device_sigs: HashMap<String, DeviceSig> = HashMap::new();
    let mut decls = String::new();
    for op_ptr in body.deref(ctx).iter(ctx) {
        let op_obj = Operation::get_op_dyn(op_ptr, ctx);
        if let Some(func) = op_obj.downcast_ref::<LlvmFuncOp>() {
            if func.is_declaration(ctx) || !is_internal(ctx, func) {
                continue;
            }
            let name = func.get_symbol_name(ctx).to_string();
            let sig = device_signature(ctx, func)?;
            decls.push_str(&format!("{};\n", device_func_header(&name, &sig)));
            device_sigs.insert(name, sig);
        }
    }

    let mut align_raises: HashMap<String, u64> = HashMap::new();
    let mut used_dyn_shared = false;
    let mut funcs: Vec<(String, String, Vec<LineTag>)> = Vec::new();
    for op_ptr in body.deref(ctx).iter(ctx) {
        let op_obj = Operation::get_op_dyn(op_ptr, ctx);
        if let Some(func) = op_obj.downcast_ref::<LlvmFuncOp>() {
            if func.is_declaration(ctx) {
                continue;
            }
            let kind = if is_internal(ctx, func) {
                FuncKind::Device
            } else {
                FuncKind::Entry
            };
            let (text, tags) = emit_function(
                ctx,
                func,
                &globals,
                &device_sigs,
                kind,
                &mut align_raises,
                &mut used_dyn_shared,
            )?;
            funcs.push((func.get_symbol_name(ctx).to_string(), text, tags));
        }
    }

    // Compose the module in the fixed order: header, globals (with any
    // alignment raises applied), device forward declarations, functions.
    let mut out = String::new();
    let mut line = 3usize; // header lines below
    let mut entries: Vec<(String, usize, usize, Vec<LineTag>)> = Vec::new();
    out.push_str(&format!(
        ".version {}.{}\n.target sm_{}\n.address_size 64\n",
        target.ptx_isa.0, target.ptx_isa.1, target.sm
    ));
    for op_ptr in global_ptrs {
        let op_obj = Operation::get_op_dyn(op_ptr, ctx);
        let global = op_obj
            .downcast_ref::<GlobalOp>()
            .expect("collected as a GlobalOp above");
        let (_, _, text) = emit_global(ctx, global, &align_raises)?;
        line += text.matches('\n').count();
        out.push_str(&text);
    }
    if used_dyn_shared {
        // The single dynamic shared-memory window; its size is the
        // launch's `sharedMemBytes`.
        out.push_str(&format!(
            ".extern .shared .align {DYN_SHARED_ALIGN} .b8 {DYN_SHARED_SYM}[];\n"
        ));
        line += 1;
    }
    line += decls.matches('\n').count();
    out.push_str(&decls);
    for (name, text, tags) in funcs {
        out.push('\n');
        line += 1;
        let start = line + 1;
        line += text.matches('\n').count();
        entries.push((name, start, line, tags));
        out.push_str(&text);
    }
    let linemap = if want_linemap {
        let mut map = serde_json::Map::new();
        for (name, start, end, tags) in entries {
            let mut lines = serde_json::Map::new();
            for (offset, tag) in tags.iter().enumerate() {
                let value = match tag {
                    LineTag::Ops(ids) => serde_json::json!(ids),
                    LineTag::Root(root) => serde_json::json!(root),
                };
                lines.insert((start + offset).to_string(), value);
            }
            map.insert(
                name,
                serde_json::json!({
                    "start": start,
                    "end": end,
                    "lines": lines,
                    "midend": serde_json::Value::Null,
                }),
            );
        }
        Some(serde_json::Value::Object(map).to_string())
    } else {
        None
    };
    Ok((out, linemap))
}

/// The PTX state space a module global lives in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GlobalSpace {
    /// `.global`: device memory, statically initialized.
    Global,
    /// `.shared`: per-CTA shared memory (a `#[link_section = ".shared"]`
    /// static, docs/KERNEL-ABI.md).
    Shared,
}

/// A module global as a PTX variable declaration. `align_raises` carries
/// per-symbol minimum alignments demanded by vectorized accesses (raising
/// a definition's alignment is always legal).
fn emit_global(
    ctx: &Context,
    global: &GlobalOp,
    align_raises: &HashMap<String, u64>,
) -> CrabbitResult<(String, GlobalSpace, String)> {
    let name = global.get_symbol_name(ctx).to_string();
    let Some(data) = crate::ll::global_data(ctx, global) else {
        return Err(input_error_noloc!(NvptxErr::UnsupportedGlobal(format!(
            "`{name}` has no data initializer (extern globals are not supported in kernels)"
        ))));
    };
    if !data.relocs.is_empty() {
        return Err(input_error_noloc!(NvptxErr::UnsupportedGlobal(format!(
            "`{name}` holds pointers (relocations are not supported in kernels)"
        ))));
    }
    let align = data
        .align
        .max(1)
        .max(align_raises.get(&name).copied().unwrap_or(1));
    let len = data.bytes.len().max(1);
    let section = crate::ll::global_section(ctx, global);
    match section.as_deref() {
        Some(".shared") => {
            if data.bytes.iter().any(|&b| b != 0) {
                return Err(input_error_noloc!(NvptxErr::UnsupportedGlobal(format!(
                    "`{name}` is in `.shared` but has a non-zero initializer \
                     (shared memory cannot be statically initialized)"
                ))));
            }
            Ok((
                name.clone(),
                GlobalSpace::Shared,
                format!(".shared .align {align} .b8 {name}[{len}];\n"),
            ))
        }
        Some(other) => Err(input_error_noloc!(NvptxErr::UnsupportedGlobal(format!(
            "`{name}` is in link section `{other}` (only `.shared` is meaningful in kernels)"
        )))),
        None => {
            let init: Vec<String> = data.bytes.iter().map(|b| b.to_string()).collect();
            let text = if data.bytes.is_empty() {
                format!(".global .align {align} .b8 {name}[{len}];\n")
            } else {
                format!(
                    ".global .align {align} .b8 {name}[{len}] = {{{}}};\n",
                    init.join(", ")
                )
            };
            Ok((name, GlobalSpace::Global, text))
        }
    }
}

// Register model ------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum RegClass {
    Pred,
    B32,
    B64,
    F32,
    F64,
}

impl RegClass {
    fn prefix(self) -> &'static str {
        match self {
            RegClass::Pred => "%p",
            RegClass::B32 => "%r",
            RegClass::B64 => "%rd",
            RegClass::F32 => "%f",
            RegClass::F64 => "%fd",
        }
    }

    fn decl(self) -> &'static str {
        match self {
            RegClass::Pred => ".pred",
            RegClass::B32 => ".b32",
            RegClass::B64 => ".b64",
            RegClass::F32 => ".f32",
            RegClass::F64 => ".f64",
        }
    }

    /// The `mov` type suffix that copies a full register of this class.
    fn mov_suffix(self) -> &'static str {
        match self {
            RegClass::Pred => "pred",
            RegClass::B32 => "b32",
            RegClass::B64 => "b64",
            RegClass::F32 => "f32",
            RegClass::F64 => "f64",
        }
    }

    fn index(self) -> usize {
        match self {
            RegClass::Pred => 0,
            RegClass::B32 => 1,
            RegClass::B64 => 2,
            RegClass::F32 => 3,
            RegClass::F64 => 4,
        }
    }

    fn is_float(self) -> bool {
        matches!(self, RegClass::F32 | RegClass::F64)
    }

    /// The PTX type of a float class (`f32`/`f64`).
    fn float_ty(self) -> &'static str {
        match self {
            RegClass::F32 => "f32",
            RegClass::F64 => "f64",
            _ => unreachable!("float_ty on an integer register class"),
        }
    }
}

const REG_CLASSES: [RegClass; 5] = [
    RegClass::Pred,
    RegClass::B32,
    RegClass::B64,
    RegClass::F32,
    RegClass::F64,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Reg {
    class: RegClass,
    index: usize,
}

impl std::fmt::Display for Reg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.class.prefix(), self.index)
    }
}

/// The register class holding a value of `ty`, or an error for types PTX
/// emission does not model yet.
fn classify(ctx: &Context, ty: TypeHandle) -> CrabbitResult<RegClass> {
    let ty_ref = ty.deref(ctx);
    if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
        return match int_ty.width() {
            1 => Ok(RegClass::Pred),
            8 | 16 | 32 => Ok(RegClass::B32),
            64 => Ok(RegClass::B64),
            width => Err(input_error_noloc!(NvptxErr::UnsupportedType(format!(
                "i{width}"
            )))),
        };
    }
    if ty_ref.downcast_ref::<PointerType>().is_some() {
        return Ok(RegClass::B64);
    }
    if ty_ref.downcast_ref::<FP32Type>().is_some() {
        return Ok(RegClass::F32);
    }
    if ty_ref.downcast_ref::<FP64Type>().is_some() {
        return Ok(RegClass::F64);
    }
    Err(input_error_noloc!(NvptxErr::UnsupportedType(
        ty.disp(ctx).to_string()
    )))
}

/// The bit width of `ty`; pointers count as 64.
fn width_of(ctx: &Context, ty: TypeHandle) -> CrabbitResult<u32> {
    let ty_ref = ty.deref(ctx);
    if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
        return Ok(int_ty.width());
    }
    if ty_ref.downcast_ref::<PointerType>().is_some() {
        return Ok(64);
    }
    if ty_ref.downcast_ref::<FP32Type>().is_some() {
        return Ok(32);
    }
    if ty_ref.downcast_ref::<FP64Type>().is_some() {
        return Ok(64);
    }
    Err(input_error_noloc!(NvptxErr::UnsupportedType(
        ty.disp(ctx).to_string()
    )))
}

/// Byte size of `ty` when indexed through or loaded/stored: the GEP stride.
/// Struct/array layout mirrors the aarch64 backend's rules (natural
/// alignment, fields padded to their alignment, total size padded to the
/// aggregate's alignment), so both backends agree with the importer's
/// field indices.
fn size_of_ty(ctx: &Context, ty: TypeHandle) -> CrabbitResult<u64> {
    let ty_ref = ty.deref(ctx);
    if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
        return Ok((int_ty.width() as u64).div_ceil(8).max(1));
    }
    if ty_ref.downcast_ref::<PointerType>().is_some() {
        return Ok(8);
    }
    if ty_ref.downcast_ref::<FP32Type>().is_some() {
        return Ok(4);
    }
    if ty_ref.downcast_ref::<FP64Type>().is_some() {
        return Ok(8);
    }
    if let Some(array_ty) = ty_ref.downcast_ref::<ArrayType>() {
        let elem_ty = array_ty.elem_type();
        let len = array_ty.size();
        drop(ty_ref);
        let elem = size_of_ty(ctx, elem_ty)?;
        let stride = align_to(elem, align_of_ty(ctx, elem_ty)?);
        return Ok(stride * len);
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<StructType>() {
        if struct_ty.is_opaque() {
            return Err(input_error_noloc!(NvptxErr::UnsupportedType(
                "opaque struct in kernel".to_string()
            )));
        }
        let fields: Vec<TypeHandle> = struct_ty.fields().collect();
        drop(ty_ref);
        let mut offset = 0u64;
        let mut align = 1u64;
        for field in fields {
            let field_align = align_of_ty(ctx, field)?;
            offset = align_to(offset, field_align) + size_of_ty(ctx, field)?;
            align = align.max(field_align);
        }
        return Ok(align_to(offset, align));
    }
    Err(input_error_noloc!(NvptxErr::UnsupportedType(format!(
        "no NVPTX size for {}",
        ty.disp(ctx)
    ))))
}

/// Natural alignment of `ty`: scalars align to their size (capped at 8),
/// arrays to their element, structs to their most-aligned field.
fn align_of_ty(ctx: &Context, ty: TypeHandle) -> CrabbitResult<u64> {
    let ty_ref = ty.deref(ctx);
    if let Some(array_ty) = ty_ref.downcast_ref::<ArrayType>() {
        let elem_ty = array_ty.elem_type();
        drop(ty_ref);
        return align_of_ty(ctx, elem_ty);
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<StructType>() {
        if struct_ty.is_opaque() {
            return Err(input_error_noloc!(NvptxErr::UnsupportedType(
                "opaque struct in kernel".to_string()
            )));
        }
        let fields: Vec<TypeHandle> = struct_ty.fields().collect();
        drop(ty_ref);
        let mut align = 1u64;
        for field in fields {
            align = align.max(align_of_ty(ctx, field)?);
        }
        return Ok(align);
    }
    drop(ty_ref);
    Ok(size_of_ty(ctx, ty)?.clamp(1, 8))
}

fn align_to(value: u64, align: u64) -> u64 {
    if align <= 1 { value } else { (value + align - 1) & !(align - 1) }
}

/// Whether `ty` is an aggregate (array or struct): a value of it lives as
/// a set of scalar leaf registers, not one register.
fn is_aggregate(ctx: &Context, ty: TypeHandle) -> bool {
    let ty_ref = ty.deref(ctx);
    ty_ref.downcast_ref::<ArrayType>().is_some() || ty_ref.downcast_ref::<StructType>().is_some()
}

/// Append the scalar leaves of `ty` (depth-first, layout order) as
/// `(byte offset from the aggregate start, leaf type)`.
fn agg_leaves(
    ctx: &Context,
    ty: TypeHandle,
    base: u64,
    out: &mut Vec<(u64, TypeHandle)>,
) -> CrabbitResult<()> {
    let ty_ref = ty.deref(ctx);
    if let Some(array_ty) = ty_ref.downcast_ref::<ArrayType>() {
        let elem_ty = array_ty.elem_type();
        let len = array_ty.size();
        drop(ty_ref);
        let stride = align_to(size_of_ty(ctx, elem_ty)?, align_of_ty(ctx, elem_ty)?);
        for i in 0..len {
            agg_leaves(ctx, elem_ty, base + i * stride, out)?;
        }
        return Ok(());
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<StructType>() {
        if struct_ty.is_opaque() {
            return Err(input_error_noloc!(NvptxErr::UnsupportedType(
                "opaque struct value in kernel".to_string()
            )));
        }
        let fields: Vec<TypeHandle> = struct_ty.fields().collect();
        drop(ty_ref);
        let mut offset = 0u64;
        for field in fields {
            offset = align_to(offset, align_of_ty(ctx, field)?);
            agg_leaves(ctx, field, base + offset, out)?;
            offset += size_of_ty(ctx, field)?;
        }
        return Ok(());
    }
    drop(ty_ref);
    out.push((base, ty));
    Ok(())
}

/// Number of scalar leaves of `ty`.
fn agg_leaf_count(ctx: &Context, ty: TypeHandle) -> CrabbitResult<usize> {
    let mut leaves = Vec::new();
    agg_leaves(ctx, ty, 0, &mut leaves)?;
    Ok(leaves.len())
}

/// Resolve an insert/extract_value index path: the linearized leaf index
/// where the addressed sub-value starts, and that sub-value's type.
fn agg_path(
    ctx: &Context,
    ty: TypeHandle,
    path: &[u32],
) -> CrabbitResult<(usize, TypeHandle)> {
    let mut leaf_start = 0usize;
    let mut current = ty;
    for &index in path {
        let ty_ref = current.deref(ctx);
        if let Some(array_ty) = ty_ref.downcast_ref::<ArrayType>() {
            let elem_ty = array_ty.elem_type();
            let len = array_ty.size();
            drop(ty_ref);
            if index as u64 >= len {
                return Err(input_error_noloc!(NvptxErr::UnsupportedOp(
                    "insert/extract_value array index out of bounds".to_string()
                )));
            }
            leaf_start += index as usize * agg_leaf_count(ctx, elem_ty)?;
            current = elem_ty;
        } else if let Some(struct_ty) = ty_ref.downcast_ref::<StructType>() {
            let fields: Vec<TypeHandle> = struct_ty.fields().collect();
            drop(ty_ref);
            if index as usize >= fields.len() {
                return Err(input_error_noloc!(NvptxErr::UnsupportedOp(
                    "insert/extract_value struct index out of bounds".to_string()
                )));
            }
            for field in &fields[..index as usize] {
                leaf_start += agg_leaf_count(ctx, *field)?;
            }
            current = fields[index as usize];
        } else {
            return Err(input_error_noloc!(NvptxErr::UnsupportedOp(
                "insert/extract_value indexes into a non-aggregate".to_string()
            )));
        }
    }
    Ok((leaf_start, current))
}

/// The byte offset and type of struct field `index`, or an error when
/// `index` is out of bounds.
fn struct_field_offset(
    ctx: &Context,
    struct_ty: &StructType,
    index: u64,
) -> CrabbitResult<(u64, TypeHandle)> {
    let fields: Vec<TypeHandle> = struct_ty.fields().collect();
    let mut offset = 0u64;
    for (i, field) in fields.iter().copied().enumerate() {
        offset = align_to(offset, align_of_ty(ctx, field)?);
        if i as u64 == index {
            return Ok((offset, field));
        }
        offset += size_of_ty(ctx, field)?;
    }
    Err(input_error_noloc!(NvptxErr::UnsupportedOp(format!(
        "gep struct field index {index} is out of bounds"
    ))))
}

// NVVM intrinsic calls ------------------------------------------------------

/// The canonical (underscored, prefix-stripped) form of an NVVM intrinsic
/// callee. pliron identifiers cannot carry dots, so intrinsics arrive as
/// `llvm_nvvm_…` / `int_nvvm_…` (the latter is cuda-oxide's catalog naming);
/// dotted LLVM names are normalized too.
fn nvvm_intrinsic_name(name: &str) -> String {
    let name = name.replace('.', "_");
    name.strip_prefix("llvm_")
        .or_else(|| name.strip_prefix("int_"))
        .unwrap_or(&name)
        .to_string()
}

/// Special-register intrinsics: canonical callee name → PTX special register.
fn sreg_for_callee(name: &str) -> Option<&'static str> {
    Some(match name {
        "nvvm_read_ptx_sreg_tid_x" => "%tid.x",
        "nvvm_read_ptx_sreg_tid_y" => "%tid.y",
        "nvvm_read_ptx_sreg_tid_z" => "%tid.z",
        "nvvm_read_ptx_sreg_ntid_x" => "%ntid.x",
        "nvvm_read_ptx_sreg_ntid_y" => "%ntid.y",
        "nvvm_read_ptx_sreg_ntid_z" => "%ntid.z",
        "nvvm_read_ptx_sreg_ctaid_x" => "%ctaid.x",
        "nvvm_read_ptx_sreg_ctaid_y" => "%ctaid.y",
        "nvvm_read_ptx_sreg_ctaid_z" => "%ctaid.z",
        "nvvm_read_ptx_sreg_nctaid_x" => "%nctaid.x",
        "nvvm_read_ptx_sreg_nctaid_y" => "%nctaid.y",
        "nvvm_read_ptx_sreg_nctaid_z" => "%nctaid.z",
        "nvvm_read_ptx_sreg_laneid" => "%laneid",
        "nvvm_read_ptx_sreg_warpsize" => "WARP_SZ",
        _ => return None,
    })
}

/// Warp shuffle intrinsics (`llvm.nvvm.shfl.sync.{down,up,bfly,idx}.{i32,f32}`,
/// the plain value-returning forms): canonical callee name → (PTX mode,
/// value register class). All lower to `shfl.sync.<mode>.b32 d, a, b, c,
/// membermask;` — the f32 variants ride the same `.b32` instruction with
/// `.f32` registers (PTX's untyped-b32 operand rule). NVVM operand order is
/// `(membermask, a, b, c)` with the packed `c = cval | (segmask << 8)`
/// convention (full-warp width: `c = 0x1f` for down/bfly/idx, `0` for up).
fn shfl_intrinsic(canonical: &str) -> Option<(&'static str, RegClass)> {
    Some(match canonical {
        "nvvm_shfl_sync_down_i32" => ("down", RegClass::B32),
        "nvvm_shfl_sync_down_f32" => ("down", RegClass::F32),
        "nvvm_shfl_sync_up_i32" => ("up", RegClass::B32),
        "nvvm_shfl_sync_up_f32" => ("up", RegClass::F32),
        "nvvm_shfl_sync_bfly_i32" => ("bfly", RegClass::B32),
        "nvvm_shfl_sync_bfly_f32" => ("bfly", RegClass::F32),
        "nvvm_shfl_sync_idx_i32" => ("idx", RegClass::B32),
        "nvvm_shfl_sync_idx_f32" => ("idx", RegClass::F32),
        _ => return None,
    })
}

/// Warp vote intrinsics: canonical callee name → (PTX vote mode, whether it
/// is the `.ballot.b32` form). NVVM operand order is `(membermask, pred)`;
/// PTX is `vote.sync.<mode>.{pred,b32} d, a, membermask;`.
fn vote_intrinsic(canonical: &str) -> Option<(&'static str, bool)> {
    Some(match canonical {
        "nvvm_vote_ballot_sync" => ("ballot", true),
        "nvvm_vote_all_sync" => ("all", false),
        "nvvm_vote_any_sync" => ("any", false),
        "nvvm_vote_uni_sync" => ("uni", false),
        _ => return None,
    })
}

/// The dynamic shared-memory window (docs/KERNEL-ABI.md, `extern __shared__`
/// analogue): a call to this crabbit-convention intrinsic yields the base
/// address of the module's single `.extern .shared` array, whose size comes
/// from the launch's `sharedMemBytes`.
const DYN_SHARED_INTRINSIC: &str = "crabbit_dyn_shared_base";
/// The PTX symbol the dynamic shared window is declared under.
const DYN_SHARED_SYM: &str = "__crabbit_dyn_shared";
/// Declared alignment of the dynamic shared window. The driver aligns the
/// runtime base at least this much; 16 covers every vector access.
const DYN_SHARED_ALIGN: u64 = 16;

/// Atomic intrinsics: canonical callee name → (scope suffix, PTX op, type,
/// result class) for `atom{.scope}{.space}.op.type d, [a], b`. The `_gen_`
/// names take generic pointers; emission narrows the space when provenance
/// proves the address global or shared.
fn atomic_intrinsic(canonical: &str) -> Option<(&'static str, &'static str, &'static str, RegClass)> {
    Some(match canonical {
        "nvvm_atomic_add_gen_i" => ("", "add", "u32", RegClass::B32),
        "nvvm_atomic_add_gen_i_cta" => (".cta", "add", "u32", RegClass::B32),
        "nvvm_atomic_add_gen_i_sys" => (".sys", "add", "u32", RegClass::B32),
        "nvvm_atomic_add_gen_f" => ("", "add", "f32", RegClass::F32),
        "nvvm_atomic_add_gen_f_cta" => (".cta", "add", "f32", RegClass::F32),
        "nvvm_atomic_add_gen_ll" => ("", "add", "u64", RegClass::B64),
        "nvvm_atomic_add_gen_d" => ("", "add", "f64", RegClass::F64),
        "nvvm_atomic_max_gen_i" => ("", "max", "s32", RegClass::B32),
        "nvvm_atomic_min_gen_i" => ("", "min", "s32", RegClass::B32),
        "nvvm_atomic_max_gen_ui" => ("", "max", "u32", RegClass::B32),
        "nvvm_atomic_min_gen_ui" => ("", "min", "u32", RegClass::B32),
        "nvvm_atomic_and_gen_i" => ("", "and", "b32", RegClass::B32),
        "nvvm_atomic_or_gen_i" => ("", "or", "b32", RegClass::B32),
        "nvvm_atomic_xor_gen_i" => ("", "xor", "b32", RegClass::B32),
        "nvvm_atomic_exch_gen_i" => ("", "exch", "b32", RegClass::B32),
        _ => return None,
    })
}

/// Math intrinsics: canonical callee name → (PTX mnemonic, register class,
/// arity). All pure: register in, register out, no memory effects.
fn math_intrinsic(canonical: &str) -> Option<(&'static str, RegClass, usize)> {
    Some(match canonical {
        "nvvm_sqrt_rn_f" | "nvvm_sqrt_f" | "sqrt_f32" => ("sqrt.rn.f32", RegClass::F32, 1),
        "nvvm_sqrt_approx_f" => ("sqrt.approx.f32", RegClass::F32, 1),
        "nvvm_rsqrt_approx_f" => ("rsqrt.approx.f32", RegClass::F32, 1),
        "nvvm_ex2_approx_f" | "exp2_f32" => ("ex2.approx.f32", RegClass::F32, 1),
        "nvvm_lg2_approx_f" | "log2_f32" => ("lg2.approx.f32", RegClass::F32, 1),
        "nvvm_sin_approx_f" => ("sin.approx.f32", RegClass::F32, 1),
        "nvvm_cos_approx_f" => ("cos.approx.f32", RegClass::F32, 1),
        "nvvm_rcp_approx_f" => ("rcp.approx.f32", RegClass::F32, 1),
        "nvvm_fabs_f" | "fabs_f32" => ("abs.f32", RegClass::F32, 1),
        "nvvm_floor_f" | "floor_f32" => ("cvt.rmi.f32.f32", RegClass::F32, 1),
        "nvvm_ceil_f" | "ceil_f32" => ("cvt.rpi.f32.f32", RegClass::F32, 1),
        "nvvm_round_f" | "nvvm_rint_f" => ("cvt.rni.f32.f32", RegClass::F32, 1),
        "nvvm_trunc_f" | "trunc_f32" => ("cvt.rzi.f32.f32", RegClass::F32, 1),
        "nvvm_fmax_f" | "maxnum_f32" => ("max.f32", RegClass::F32, 2),
        "nvvm_fmin_f" | "minnum_f32" => ("min.f32", RegClass::F32, 2),
        "nvvm_fma_rn_f" | "fma_f32" => ("fma.rn.f32", RegClass::F32, 3),
        "nvvm_sqrt_rn_d" | "sqrt_f64" => ("sqrt.rn.f64", RegClass::F64, 1),
        "nvvm_rsqrt_approx_d" => ("rsqrt.approx.f64", RegClass::F64, 1),
        "nvvm_fabs_d" | "fabs_f64" => ("abs.f64", RegClass::F64, 1),
        "nvvm_floor_d" | "floor_f64" => ("cvt.rmi.f64.f64", RegClass::F64, 1),
        "nvvm_ceil_d" | "ceil_f64" => ("cvt.rpi.f64.f64", RegClass::F64, 1),
        "nvvm_fmax_d" | "maxnum_f64" => ("max.f64", RegClass::F64, 2),
        "nvvm_fmin_d" | "minnum_f64" => ("min.f64", RegClass::F64, 2),
        "nvvm_fma_rn_d" | "fma_f64" => ("fma.rn.f64", RegClass::F64, 3),
        _ => return None,
    })
}

/// How a call may write memory, for the per-root write-set analysis
/// ([spaces::readonly_values]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CallMemEffect {
    /// Provably writes through no pointer: special registers, barriers and
    /// membars (ordering, not writing), shuffles, votes, math, and the
    /// dyn-shared window base.
    None,
    /// An atomic RMW: writes exactly through its pointer operand (index 0).
    AtomicPtrArg,
    /// Anything else — device `.func`s, unrecognized intrinsics, indirect
    /// callees: may write through any pointer it can reach.
    Unknown,
}

fn call_mem_effect(canonical: &str) -> CallMemEffect {
    if atomic_intrinsic(canonical).is_some() {
        return CallMemEffect::AtomicPtrArg;
    }
    let pure = sreg_for_callee(canonical).is_some()
        || shfl_intrinsic(canonical).is_some()
        || vote_intrinsic(canonical).is_some()
        || math_intrinsic(canonical).is_some()
        || canonical == DYN_SHARED_INTRINSIC
        || matches!(
            canonical,
            "nvvm_barrier0"
                | "nvvm_bar_warp_sync"
                | "nvvm_membar_gl"
                | "nvvm_membar_cta"
                | "nvvm_membar_sys"
        );
    if pure {
        CallMemEffect::None
    } else {
        CallMemEffect::Unknown
    }
}

// Device functions ------------------------------------------------------

/// Whether `func` has internal/private linkage: a device helper rather
/// than a kernel entry point.
fn is_internal(ctx: &Context, func: &LlvmFuncOp) -> bool {
    func.get_attr_llvm_function_linkage(ctx).is_some_and(|linkage| {
        matches!(
            *linkage,
            crate::dialects::llvm::attributes::LinkageAttr::InternalLinkage
                | crate::dialects::llvm::attributes::LinkageAttr::PrivateLinkage
        )
    })
}

/// The `.param`-ABI signature of a device `.func`: parameter register
/// classes and the (at most one, scalar) result class.
struct DeviceSig {
    params: Vec<RegClass>,
    ret: Option<RegClass>,
}

/// The `.param` declaration type for a value of `class` (sub-32-bit
/// integers travel widened to `.b32`, matching the register invariant).
fn param_decl_ty(class: RegClass) -> CrabbitResult<&'static str> {
    Ok(match class {
        RegClass::B32 => "b32",
        RegClass::B64 => "b64",
        RegClass::F32 => "f32",
        RegClass::F64 => "f64",
        RegClass::Pred => {
            return Err(input_error_noloc!(NvptxErr::UnsupportedType(
                "i1 device-function parameter or result".to_string()
            )));
        }
    })
}

fn device_signature(ctx: &Context, func: &LlvmFuncOp) -> CrabbitResult<DeviceSig> {
    let func_ty = func.get_type(ctx);
    let (args, result) = {
        use pliron::builtin::type_interfaces::FunctionTypeInterface as _;
        let func_ty = func_ty.deref(ctx);
        (func_ty.arg_types(), func_ty.result_type())
    };
    let mut params = Vec::new();
    for arg in args {
        let class = classify(ctx, arg)?;
        param_decl_ty(class)?;
        params.push(class);
    }
    let ret = if result.deref(ctx).downcast_ref::<VoidType>().is_some() {
        None
    } else {
        let class = classify(ctx, result)?;
        param_decl_ty(class)?;
        Some(class)
    };
    Ok(DeviceSig { params, ret })
}

/// The `.func` header shared by the forward declaration and the body:
/// `.func (.param .b32 name_ret) name (.param .b64 name_param_0, …)`.
fn device_func_header(name: &str, sig: &DeviceSig) -> String {
    let mut header = String::from(".func ");
    if let Some(ret) = sig.ret {
        header.push_str(&format!(
            "(.param .{} {name}_ret) ",
            param_decl_ty(ret).expect("signature already validated")
        ));
    }
    header.push_str(name);
    let params: Vec<String> = sig
        .params
        .iter()
        .enumerate()
        .map(|(i, class)| {
            format!(
                ".param .{} {name}_param_{i}",
                param_decl_ty(*class).expect("signature already validated")
            )
        })
        .collect();
    header.push_str(&format!(" ({})", params.join(", ")));
    header
}

/// Whether a function is emitted as a grid-visible `.entry` kernel or an
/// internal `.func` with the `.param` call ABI.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FuncKind {
    Entry,
    Device,
}

// Function emission ---------------------------------------------------------

/// One emitted PTX line's attribution (docs/PROFILE-FEEDBACK-BACKWARD.md,
/// GPU leg): the SOURCE-level op ids the line lowers ([opmap::
/// effective_sources] — the kernel pipeline has a single numbering, so the
/// mid-end adjoints are already resolved here), or a named synthetic root
/// for lines that lower no op.
///
/// [opmap::effective_sources]: crate::passes::aarch64::opmap::effective_sources
#[derive(Clone, Debug)]
enum LineTag {
    Ops(Vec<i64>),
    Root(&'static str),
}

/// A compile-time constant value: not materialized at its definition, but
/// folded into using instructions as a PTX immediate, or `mov`ed into a
/// fresh register at uses that need one.
#[derive(Clone, Copy, Debug)]
struct ConstVal {
    class: RegClass,
    /// Bits, already masked to `width` (zero-extended narrow invariant).
    imm: u128,
    width: u32,
}

struct FuncEmitter<'c> {
    ctx: &'c Context,
    globals: &'c HashMap<String, GlobalSpace>,
    /// Device `.func` signatures for call emission.
    device_sigs: &'c HashMap<String, DeviceSig>,
    /// For a device `.func` body: the return `.param` name and class.
    ret_param: Option<(String, RegClass)>,
    /// Pointer provenance for the current kernel ([spaces::infer_spaces]):
    /// which state space each address value is proven to live in.
    spaces: HashMap<Value, PtrSpace>,
    /// FP contraction plan ([plan_fp_contraction]): `fmul`s fused into
    /// their single `fadd` use (they emit nothing) …
    fused_muls: HashSet<Ptr<Operation>>,
    /// … and, per contracted `fadd`, the `fma.rn` operands `(a, b, c)`.
    fma_operands: HashMap<Ptr<Operation>, (Value, Value, Value)>,
    /// Folded address decompositions per GEP result ([addr::FoldedAddr]);
    /// memory operands through these fold their constant offset.
    addr_plan: HashMap<Value, FoldedAddr>,
    /// Ops that emit nothing: GEPs whose every use folds into a memory
    /// operand (or another folded GEP), plus the pure integer chains
    /// feeding only such GEPs ([plan_addresses]).
    skip_ops: HashSet<Ptr<Operation>>,
    /// Per-block cache: folded base `(root, terms)` → the register
    /// holding `root + Σ terms`. Cleared at each block (a register from
    /// a non-dominating block must never be reused).
    base_regs: HashMap<(Value, Vec<Term>), Reg>,
    /// Per-block cache of widened (64-bit) term values.
    wide_terms: HashMap<Term, Reg>,
    /// Vector-load plan for the current block ([Self::plan_vector_loads]):
    /// group leaders emit one `ld.shared.v2/v4`, covered members nothing.
    vec_lead: HashMap<Ptr<Operation>, VecGroup>,
    vec_covered: HashSet<Ptr<Operation>>,
    /// Values proven to address only never-written memory
    /// ([spaces::readonly_values]): the `.nc` and load-combining witness.
    readonly: HashSet<Value>,
    /// Whether `.nc` may be emitted: entry kernels with
    /// `CRABBIT_NVPTX_NC` unset or non-zero.
    nc_enabled: bool,
    /// Sub-word combining plan ([Self::plan_byte_combines]): the groups,
    /// each member load's group + lane, and per group the window register
    /// once its leader has loaded it.
    byte_groups: Vec<ByteCombineGroup>,
    byte_members: HashMap<Ptr<Operation>, ByteCombineMember>,
    byte_window_regs: Vec<Option<Reg>>,
    /// Shared globals whose declared alignment must be raised for a
    /// vectorized access's alignment proof (symbol → minimum bytes).
    align_raises: HashMap<String, u64>,
    values: HashMap<Value, Reg>,
    /// Constant results ([ConstVal]); disjoint from `values`.
    consts: HashMap<Value, ConstVal>,
    /// Aggregate (array/struct) SSA values as their scalar leaves in
    /// layout order ([agg_leaves]); `None` leaves are undef.
    aggs: HashMap<Value, Vec<Option<Reg>>>,
    /// `.local` array declarations for allocas, hoisted into the header.
    local_decls: Vec<String>,
    /// Alloca op → the `.local` symbol it addresses.
    local_syms: HashMap<Ptr<Operation>, String>,
    /// Whether this function called [DYN_SHARED_INTRINSIC]: the module then
    /// declares the `.extern .shared` window.
    used_dyn_shared: bool,
    /// Distinct `.param` scope counter for `call.uni` sequences.
    next_call: usize,
    reg_counts: [usize; 5],
    /// Straight-line body text: labels and instructions.
    code: String,
    /// Edge blocks for conditional branches with block arguments; appended
    /// after the main body (control flow in PTX is fully explicit, so block
    /// order does not matter).
    edges: String,
    next_edge: usize,
    block_labels: HashMap<Ptr<BasicBlock>, String>,
    /// One entry per line of [Self::code] / [Self::edges], in order. The
    /// recorder never changes the emitted text (asserted by the on/off
    /// identity test); it only rides along.
    tags: Vec<LineTag>,
    edge_tags: Vec<LineTag>,
    current: LineTag,
}

/// One planned vector load: the group's folded base, its members'
/// (absolute byte offset, load result) in ascending offset order, and the
/// scalar lane type.
#[derive(Clone)]
struct VecGroup {
    plan: FoldedAddr,
    parts: Vec<(i64, Value)>,
    class: RegClass,
    ld_ty: &'static str,
}

/// One sub-word combine group (module docs): its leader loads the aligned
/// 4-byte window (`plan.offset` = window start) once; every member —
/// leader included — extracts its own lane at its own program position.
#[derive(Clone)]
struct ByteCombineGroup {
    plan: FoldedAddr,
    leader: Ptr<Operation>,
    /// The leader load's address value: carries the group's state space
    /// and its read-only proof.
    addr: Value,
}

/// A load covered by a sub-word combine group.
#[derive(Clone, Copy)]
struct ByteCombineMember {
    group: usize,
    /// This load's byte offset within the 4-byte window.
    lane: i64,
    /// The loaded width in bits (8 or 16).
    width: u32,
}

fn emit_function(
    ctx: &Context,
    func: &LlvmFuncOp,
    globals: &HashMap<String, GlobalSpace>,
    device_sigs: &HashMap<String, DeviceSig>,
    kind: FuncKind,
    align_raises: &mut HashMap<String, u64>,
    used_dyn_shared: &mut bool,
) -> CrabbitResult<(String, Vec<LineTag>)> {
    let name = func.get_symbol_name(ctx).to_string();
    let region = func
        .get_region(ctx)
        .expect("llvm.func definition must have a body");
    let entry = region
        .deref(ctx)
        .get_head()
        .expect("llvm.func definition must have an entry block");
    let blocks = entry_reverse_post_order(ctx, entry);

    let (fused_muls, fma_operands) = plan_fp_contraction(ctx, &blocks);
    let ret_param = match kind {
        FuncKind::Entry => None,
        FuncKind::Device => device_sigs
            .get(&name)
            .and_then(|sig| sig.ret)
            .map(|class| (format!("{name}_ret"), class)),
    };
    let spaces = spaces::infer_spaces(ctx, entry, &blocks, globals, kind == FuncKind::Entry);
    let (addr_plan, skip_ops) = plan_addresses(ctx, &blocks, &spaces);
    let readonly = spaces::readonly_values(ctx, entry, &blocks, kind == FuncKind::Entry);
    // `.nc` asserts launch-lifetime invariance, which a device `.func`'s
    // local analysis cannot see past its callers (module docs).
    let nc_enabled = kind == FuncKind::Entry
        && !std::env::var("CRABBIT_NVPTX_NC").is_ok_and(|v| v == "0");
    let mut emitter = FuncEmitter {
        ctx,
        globals,
        device_sigs,
        ret_param,
        spaces,
        fused_muls,
        fma_operands,
        addr_plan,
        skip_ops,
        base_regs: HashMap::new(),
        wide_terms: HashMap::new(),
        vec_lead: HashMap::new(),
        vec_covered: HashSet::new(),
        readonly,
        nc_enabled,
        byte_groups: Vec::new(),
        byte_members: HashMap::new(),
        byte_window_regs: Vec::new(),
        align_raises: HashMap::new(),
        values: HashMap::new(),
        consts: HashMap::new(),
        aggs: HashMap::new(),
        local_decls: Vec::new(),
        local_syms: HashMap::new(),
        used_dyn_shared: false,
        next_call: 0,
        reg_counts: [0; 5],
        code: String::new(),
        edges: String::new(),
        next_edge: 0,
        block_labels: HashMap::new(),
        tags: Vec::new(),
        edge_tags: Vec::new(),
        current: LineTag::Root("ptx:entry"),
    };
    emitter.plan_local_allocas(&blocks, &name)?;
    emitter.plan_byte_combines(&blocks, region)?;

    // Kernel parameters are the entry block's arguments.
    let params: Vec<Value> = entry.deref(ctx).arguments().collect();
    let mut param_decls = Vec::new();
    let mut param_loads = Vec::new();
    for (i, param) in params.iter().copied().enumerate() {
        let param_ty = param.get_type(ctx);
        if kind == FuncKind::Entry && is_aggregate(ctx, param_ty) {
            // A by-value aggregate kernel parameter: one `.param .align A
            // .b8 name_param_i[size]` in the natural layout ([size_of_ty]/
            // [align_of_ty], the same rules struct GEPs use), its scalar
            // leaves loaded with `ld.param` at their field offsets.
            let align = align_of_ty(ctx, param_ty)?;
            let size = size_of_ty(ctx, param_ty)?.max(1);
            param_decls.push(format!(
                "\t.param .align {align} .b8 {name}_param_{i}[{size}]"
            ));
            let mut leaves = Vec::new();
            agg_leaves(ctx, param_ty, 0, &mut leaves)?;
            let mut regs = Vec::with_capacity(leaves.len());
            for (offset, leaf_ty) in leaves {
                let class = classify(ctx, leaf_ty)?;
                let width = width_of(ctx, leaf_ty)?;
                if width == 1 {
                    return Err(input_error_noloc!(NvptxErr::UnsupportedType(
                        "i1 field in an aggregate kernel parameter".to_string()
                    )));
                }
                let reg = emitter.fresh(class);
                let at = if offset == 0 {
                    format!("[{name}_param_{i}]")
                } else {
                    format!("[{name}_param_{i}+{offset}]")
                };
                let load_ty = if class.is_float() {
                    class.float_ty().to_string()
                } else {
                    // Sub-word fields zero-extend into the wider register,
                    // the narrow-value invariant.
                    format!("u{width}")
                };
                param_loads.push(format!("\tld.param.{load_ty} {reg}, {at};\n"));
                regs.push(Some(reg));
            }
            emitter.aggs.insert(param, regs);
            continue;
        }
        let class = classify(ctx, param_ty)?;
        let (decl_ty, load_ty) = match class {
            RegClass::B32 => ("u32", "u32"),
            RegClass::B64 => ("u64", "u64"),
            RegClass::F32 => ("f32", "f32"),
            RegClass::F64 => ("f64", "f64"),
            RegClass::Pred => {
                return Err(input_error_noloc!(NvptxErr::UnsupportedType(
                    "i1 kernel parameter".to_string()
                )));
            }
        };
        let reg = emitter.fresh(class);
        param_decls.push(format!("\t.param .{decl_ty} {name}_param_{i}"));
        param_loads.push(format!("\tld.param.{load_ty} {reg}, [{name}_param_{i}];\n"));
        emitter.values.insert(param, reg);
    }

    // Pre-assign labels and block-argument registers so forward branches can
    // name them before their block is visited.
    for (i, block) in blocks.iter().copied().enumerate() {
        emitter.block_labels.insert(block, format!("$L_bb{i}"));
        if block != entry {
            let args: Vec<Value> = block.deref(ctx).arguments().collect();
            for arg in args {
                let class = classify(ctx, arg.get_type(ctx))?;
                let reg = emitter.fresh(class);
                emitter.values.insert(arg, reg);
            }
        }
    }

    for load in param_loads {
        emitter.code.push_str(&load);
        emitter.tags.push(LineTag::Root("ptx:entry"));
    }
    for block in blocks.iter().copied() {
        if block != entry {
            let label = emitter.block_labels[&block].clone();
            emitter.code.push_str(&format!("{label}:\n"));
            emitter.tags.push(LineTag::Root("ptx:label"));
        }
        emitter.emit_block(block)?;
    }

    let mut out = String::new();
    let mut tags: Vec<LineTag> = Vec::new();
    let decl = |out: &mut String, tags: &mut Vec<LineTag>, text: &str| {
        tags.extend(std::iter::repeat_n(
            LineTag::Root("ptx:decl"),
            text.matches('\n').count(),
        ));
        out.push_str(text);
    };
    match kind {
        FuncKind::Entry => {
            decl(&mut out, &mut tags, &format!(".visible .entry {name}(\n"));
            decl(&mut out, &mut tags, &param_decls.join(",\n"));
            decl(&mut out, &mut tags, "\n)\n{\n");
        }
        FuncKind::Device => {
            let sig = device_sigs
                .get(&name)
                .expect("device function signature collected before emission");
            decl(
                &mut out,
                &mut tags,
                &format!("{}\n{{\n", device_func_header(&name, sig)),
            );
        }
    }
    for local in &emitter.local_decls {
        decl(&mut out, &mut tags, &format!("\t{local}\n"));
    }
    for class in REG_CLASSES {
        let count = emitter.reg_counts[class.index()];
        if count > 0 {
            decl(
                &mut out,
                &mut tags,
                &format!("\t.reg {} {}<{}>;\n", class.decl(), class.prefix(), count),
            );
        }
    }
    decl(&mut out, &mut tags, "\n");
    out.push_str(&emitter.code);
    tags.extend(emitter.tags);
    out.push_str(&emitter.edges);
    tags.extend(emitter.edge_tags);
    decl(&mut out, &mut tags, "}\n");
    debug_assert_eq!(out.matches('\n').count(), tags.len());
    for (symbol, align) in emitter.align_raises {
        let entry = align_raises.entry(symbol).or_insert(align);
        *entry = (*entry).max(align);
    }
    *used_dyn_shared |= emitter.used_dyn_shared;
    Ok((out, tags))
}

/// The muls fused away by FP contraction, and per contracted `fadd` the
/// `(a, b, c)` of `fma.rn d, a, b, c`.
type FpContractionPlan = (
    HashSet<Ptr<Operation>>,
    HashMap<Ptr<Operation>, (Value, Value, Value)>,
);

/// The FP-contraction plan for a kernel body: for every `fadd` with a
/// single-use `fmul` operand, fuse that mul into an `fma.rn` at the add
/// (nvcc's default `-fmad=true`). Returns the fused muls — which emit
/// nothing — and, per contracted add, the `(a, b, c)` of `fma.rn d, a, b, c`.
/// When both operands are single-use muls only the left one fuses; a mul
/// with any other use (or more than one) stays a plain `mul.rn`.
fn plan_fp_contraction(ctx: &Context, blocks: &[Ptr<BasicBlock>]) -> FpContractionPlan {
    let mut fused: HashSet<Ptr<Operation>> = HashSet::new();
    let mut fmas: HashMap<Ptr<Operation>, (Value, Value, Value)> = HashMap::new();
    for block in blocks {
        for op_ptr in block.deref(ctx).iter(ctx) {
            let op_obj = Operation::get_op_dyn(op_ptr, ctx);
            if op_obj.downcast_ref::<FAddOp>().is_none() {
                continue;
            }
            let (lhs, rhs) = {
                let op_deref = op_ptr.deref(ctx);
                (op_deref.get_operand(0), op_deref.get_operand(1))
            };
            for (product, addend) in [(lhs, rhs), (rhs, lhs)] {
                let Some(def) = product.defining_op() else {
                    continue;
                };
                if product.num_uses(ctx) == 1
                    && !fused.contains(&def)
                    && Operation::get_op_dyn(def, ctx)
                        .downcast_ref::<FMulOp>()
                        .is_some()
                {
                    let (a, b) = {
                        let def_deref = def.deref(ctx);
                        (def_deref.get_operand(0), def_deref.get_operand(1))
                    };
                    fused.insert(def);
                    fmas.insert(op_ptr, (a, b, addend));
                    break;
                }
            }
        }
    }
    (fused, fmas)
}

/// The function-level address plan (module docs: memory-operand folding).
///
/// Pass 1 folds every GEP it can into `root + terms + offset`
/// ([AddrFolder::fold_gep]); a GEP whose base already folded absorbs the
/// base's decomposition, so chains collapse to one plan.
///
/// Pass 2 computes the ops that emit nothing: a folded GEP whose every
/// use is a load/store ADDRESS or the base of another folded GEP (no
/// consumer ever needs its register), then — transitively, in reverse
/// program order — any whitelisted pure integer op whose result feeds
/// only skipped ops and is not itself a plan's root or term (those are
/// read when the base register materializes).
fn plan_addresses(
    ctx: &Context,
    blocks: &[Ptr<BasicBlock>],
    spaces: &HashMap<Value, PtrSpace>,
) -> (HashMap<Value, FoldedAddr>, HashSet<Ptr<Operation>>) {
    // A/B instrument: CRABBIT_NVPTX_ADDRFOLD=0 restores the eager
    // per-access address emission (and with it disables vectorized
    // shared loads, which need the folded offsets).
    if std::env::var("CRABBIT_NVPTX_ADDRFOLD").is_ok_and(|v| v == "0") {
        return (HashMap::new(), HashSet::new());
    }
    let mut folder = AddrFolder::new(ctx, blocks);
    let mut plans: HashMap<Value, FoldedAddr> = HashMap::new();
    let mut ops: Vec<Ptr<Operation>> = Vec::new();
    for block in blocks {
        for op_ptr in block.deref(ctx).iter(ctx) {
            let op_obj = Operation::get_op_dyn(op_ptr, ctx);
            if let Some(gep) = op_obj.downcast_ref::<GetElementPtrOp>()
                && let Some(plan) = folder.fold_gep(ctx, gep, &plans, spaces)
            {
                plans.insert(gep.get_result(ctx), plan);
            }
            ops.push(op_ptr);
        }
    }

    // Values whose registers the base materialization reads even though
    // no IR operand names them at the read site.
    let mut pinned: HashSet<Value> = HashSet::new();
    for plan in plans.values() {
        pinned.insert(plan.root);
        for term in &plan.terms {
            pinned.insert(term.value);
        }
    }

    // Per value: the ops using it, EXCLUDING uses that never need its
    // register (memory-operand addresses of folded values, and the base
    // of a GEP that folded — its plan references the root, not the base).
    let mut users: HashMap<Value, Vec<Ptr<Operation>>> = HashMap::new();
    for &op_ptr in &ops {
        let op_obj = Operation::get_op_dyn(op_ptr, ctx);
        let exempt: Option<Value> = if let Some(load) = op_obj.downcast_ref::<LoadOp>() {
            let addr = load.get_operand_address(ctx);
            plans.contains_key(&addr).then_some(addr)
        } else if let Some(store) = op_obj.downcast_ref::<StoreOp>() {
            let addr = store.get_operand_address(ctx);
            // The stored VALUE always needs a register, even when it is
            // the same SSA value as the address.
            (plans.contains_key(&addr) && addr != store.get_operand_value(ctx)).then_some(addr)
        } else if let Some(gep) = op_obj.downcast_ref::<GetElementPtrOp>() {
            let base = gep.get_operand_src_ptr(ctx);
            plans.contains_key(&gep.get_result(ctx)).then_some(base)
        } else {
            None
        };
        let op = op_ptr.deref(ctx);
        for i in 0..op.get_num_operands() {
            let operand = op.get_operand(i);
            if Some(operand) != exempt {
                users.entry(operand).or_default().push(op_ptr);
            }
        }
    }

    let mut skip: HashSet<Ptr<Operation>> = HashSet::new();
    for &op_ptr in ops.iter().rev() {
        let op_obj = Operation::get_op_dyn(op_ptr, ctx);
        let is_gep = op_obj.downcast_ref::<GetElementPtrOp>().is_some();
        // Pure single-result integer ops whose only cost is the emitted
        // instruction; safe to drop when nothing reads the register.
        let is_pure_int = op_obj.downcast_ref::<AddOp>().is_some()
            || op_obj.downcast_ref::<SubOp>().is_some()
            || op_obj.downcast_ref::<MulOp>().is_some()
            || op_obj.downcast_ref::<ShlOp>().is_some()
            || op_obj.downcast_ref::<LShrOp>().is_some()
            || op_obj.downcast_ref::<AShrOp>().is_some()
            || op_obj.downcast_ref::<AndOp>().is_some()
            || op_obj.downcast_ref::<OrOp>().is_some()
            || op_obj.downcast_ref::<XorOp>().is_some()
            || op_obj.downcast_ref::<ZExtOp>().is_some()
            || op_obj.downcast_ref::<SExtOp>().is_some()
            || op_obj.downcast_ref::<TruncOp>().is_some();
        if !is_gep && !is_pure_int {
            continue;
        }
        let result = op_ptr.deref(ctx).get_result(0);
        if is_gep && !plans.contains_key(&result) {
            continue;
        }
        if is_pure_int && pinned.contains(&result) {
            continue;
        }
        let dead = users
            .get(&result)
            .is_none_or(|us| us.iter().all(|user| skip.contains(user)));
        if dead {
            skip.insert(op_ptr);
        }
    }
    (plans, skip)
}

/// Reverse post-order of the blocks reachable from `entry`, so a dominating
/// definition's block always precedes its users' blocks.
fn entry_reverse_post_order(ctx: &Context, entry: Ptr<BasicBlock>) -> Vec<Ptr<BasicBlock>> {
    let mut visited = std::collections::HashSet::new();
    visited.insert(entry);
    let mut post_order = Vec::new();
    let mut stack = vec![(entry, 0usize)];
    while let Some((block, succ_idx)) = stack.last_mut() {
        let succs = block.deref(ctx).succs(ctx);
        if let Some(&succ) = succs.get(*succ_idx) {
            *succ_idx += 1;
            if visited.insert(succ) {
                stack.push((succ, 0));
            }
        } else {
            post_order.push(*block);
            stack.pop();
        }
    }
    post_order.reverse();
    post_order
}

impl<'c> FuncEmitter<'c> {
    fn fresh(&mut self, class: RegClass) -> Reg {
        let index = self.reg_counts[class.index()];
        self.reg_counts[class.index()] += 1;
        Reg { class, index }
    }

    fn inst(&mut self, text: String) {
        self.code.push('\t');
        self.code.push_str(&text);
        self.code.push('\n');
        self.tags.push(self.current.clone());
    }

    /// The register holding `value`. A constant is materialized into a
    /// fresh register at each use (the use always dominates itself, so
    /// no cross-block dominance question arises; ptxas trivially CSEs the
    /// duplicates that survive immediate folding).
    fn lookup(&mut self, value: Value) -> CrabbitResult<Reg> {
        if let Some(reg) = self.values.get(&value).copied() {
            return Ok(reg);
        }
        if let Some(constant) = self.consts.get(&value).copied() {
            return Ok(self.materialize_const(constant.class, constant.imm));
        }
        Err(input_error_noloc!(NvptxErr::UndefinedValue(format!(
            "{value:?}"
        ))))
    }

    /// `value` as an integer immediate operand, when it is a constant of
    /// an integer register class: `(masked bits, width)`.
    fn int_imm(&self, value: Value) -> Option<(u128, u32)> {
        let constant = self.consts.get(&value).copied()?;
        matches!(constant.class, RegClass::B32 | RegClass::B64)
            .then_some((constant.imm, constant.width))
    }

    /// A 32-bit integer instruction operand: the folded immediate when
    /// `value` is constant, its register otherwise (for operand slots that
    /// accept either form, e.g. shuffle lane counts and membermasks).
    fn b32_operand(&mut self, value: Value) -> CrabbitResult<String> {
        if let Some((imm, _)) = self.int_imm(value) {
            return Ok(format!("{}", imm as u32));
        }
        let reg = self.lookup(value)?;
        if reg.class != RegClass::B32 {
            return Err(input_error_noloc!(NvptxErr::UnsupportedOp(format!(
                "expected a 32-bit integer operand, got {:?}",
                reg.class
            ))));
        }
        Ok(reg.to_string())
    }

    /// An `undef`/`poison` result: aggregates become all-undef leaves,
    /// scalars a zero constant.
    fn emit_undef_like(&mut self, result: Value) -> CrabbitResult<()> {
        let ctx = self.ctx;
        let ty = result.get_type(ctx);
        if is_aggregate(ctx, ty) {
            let count = agg_leaf_count(ctx, ty)?;
            self.aggs.insert(result, vec![None; count]);
            return Ok(());
        }
        let class = classify(ctx, ty)?;
        let width = width_of(ctx, ty)?;
        self.consts.insert(result, ConstVal { class, imm: 0, width });
        Ok(())
    }

    /// Pre-pass over the body: assign every `llvm.alloca` a `.local`
    /// array (declared in the function header) so its use sites only
    /// materialize the address. Sizes must be compile-time constants —
    /// dynamic allocas have no PTX lowering.
    fn plan_local_allocas(
        &mut self,
        blocks: &[Ptr<BasicBlock>],
        func_name: &str,
    ) -> CrabbitResult<()> {
        let ctx = self.ctx;
        // Constants seen so far in emission order; alloca sizes are
        // defined before their alloca, exactly like every other use.
        let mut known_consts: HashMap<Value, u64> = HashMap::new();
        let mut index = 0usize;
        for block in blocks {
            for op_ptr in block.deref(ctx).iter(ctx) {
                let op_obj = Operation::get_op_dyn(op_ptr, ctx);
                if let Some(constant) =
                    op_obj.downcast_ref::<crate::dialects::builtin::ops::ConstantOp>()
                {
                    if let Some(int) = constant.get_value(ctx).downcast_ref::<IntegerAttr>() {
                        known_consts
                            .insert(constant.get_result(ctx), int.value().to_u128() as u64);
                    }
                } else if let Some(alloca) = op_obj.downcast_ref::<AllocaOp>() {
                    let elem_ty = alloca.result_pointee_type(ctx);
                    let count_val = alloca.get_operation().deref(ctx).get_operand(0);
                    let Some(count) = known_consts.get(&count_val).copied() else {
                        return Err(input_error_noloc!(NvptxErr::UnsupportedOp(
                            "llvm.alloca with a non-constant size has no PTX lowering"
                                .to_string()
                        )));
                    };
                    let align = align_of_ty(ctx, elem_ty)?;
                    let stride = align_to(size_of_ty(ctx, elem_ty)?, align);
                    let bytes = stride.saturating_mul(count).max(1);
                    let sym = format!("__crabbit_local_{func_name}_{index}");
                    index += 1;
                    self.local_decls
                        .push(format!(".local .align {align} .b8 {sym}[{bytes}];"));
                    self.local_syms.insert(op_ptr, sym);
                }
            }
        }
        Ok(())
    }

    /// The proven state space of `value`; values the analysis never saw are
    /// generic.
    fn space_of(&self, value: Value) -> PtrSpace {
        self.spaces.get(&value).copied().unwrap_or(PtrSpace::Generic)
    }

    /// The state-space qualifier for a memory access through `addr`:
    /// `.global`/`.shared` when proven, empty (generic) otherwise. A
    /// `.shared`-proven register holds the raw shared address, so the
    /// qualified form is not just profitable there but required.
    fn space_qualifier(&self, addr: Value) -> &'static str {
        match self.space_of(addr) {
            PtrSpace::Global => ".global",
            PtrSpace::Shared => ".shared",
            PtrSpace::Generic => "",
        }
    }

    /// Materialize `imm` (already truncated to its width) into a fresh
    /// register of `class`.
    fn materialize_const(&mut self, class: RegClass, imm: u128) -> Reg {
        match class {
            RegClass::Pred => {
                let tmp = self.fresh(RegClass::B32);
                let pred = self.fresh(RegClass::Pred);
                self.inst(format!("mov.u32 {tmp}, {};", (imm & 1) as u32));
                self.inst(format!("setp.ne.b32 {pred}, {tmp}, 0;"));
                pred
            }
            RegClass::B32 => {
                let reg = self.fresh(RegClass::B32);
                self.inst(format!("mov.u32 {reg}, {};", imm as u32));
                reg
            }
            RegClass::B64 => {
                let reg = self.fresh(RegClass::B64);
                self.inst(format!("mov.u64 {reg}, {};", imm as u64));
                reg
            }
            RegClass::F32 => {
                let reg = self.fresh(RegClass::F32);
                self.inst(format!("mov.f32 {reg}, 0f{:08X};", imm as u32));
                reg
            }
            RegClass::F64 => {
                let reg = self.fresh(RegClass::F64);
                self.inst(format!("mov.f64 {reg}, 0d{:016X};", imm as u64));
                reg
            }
        }
    }

    /// Zero out the bits above `width` in a `.b32` register, in place.
    fn mask32(&mut self, reg: Reg, width: u32) {
        if width < 32 {
            let mask = (1u32 << width) - 1;
            self.inst(format!("and.b32 {reg}, {reg}, {mask};"));
        }
    }

    /// A `.b32` register holding `reg` sign-extended from `width` to 32 bits.
    /// Sub-32-bit values are kept zero-extended, so signed consumers pass
    /// through here first.
    fn signed32(&mut self, reg: Reg, width: u32) -> Reg {
        if width >= 32 {
            return reg;
        }
        let out = self.fresh(RegClass::B32);
        self.inst(format!("cvt.s32.s{width} {out}, {reg};"));
        out
    }

    // Block and op emission --------------------------------------------

    fn emit_block(&mut self, block: Ptr<BasicBlock>) -> CrabbitResult<()> {
        let ops: Vec<Ptr<Operation>> = block.deref(self.ctx).iter(self.ctx).collect();
        // Folded base registers are block-local (a register defined in a
        // non-dominating block must never be named), as is the plan for
        // grouping shared loads into vector accesses.
        self.base_regs.clear();
        self.wide_terms.clear();
        self.plan_vector_loads(&ops)?;
        for op_ptr in ops {
            self.emit_op(op_ptr)?;
        }
        Ok(())
    }

    fn emit_op(&mut self, op_ptr: Ptr<Operation>) -> CrabbitResult<()> {
        let ctx = self.ctx;
        // Folded away entirely: every consumer reads the fold, not a
        // register ([plan_addresses]).
        if self.skip_ops.contains(&op_ptr) {
            return Ok(());
        }
        // Attribution bracket: every line emitted while lowering this op —
        // including edge-block copies materialized for its branches —
        // belongs to the op's source-level parents.
        let sources = crate::passes::aarch64::opmap::effective_sources(ctx, op_ptr);
        self.current = if sources.is_empty() {
            LineTag::Root("ptx:unattributed")
        } else {
            LineTag::Ops(sources)
        };
        let op_obj = Operation::get_op_dyn(op_ptr, ctx);

        if let Some(constant) = op_obj.downcast_ref::<crate::dialects::builtin::ops::ConstantOp>() {
            let attr = constant.get_value(ctx);
            let result = constant.get_result(ctx);
            let width = width_of(ctx, result.get_type(ctx))?;
            let class = classify(ctx, result.get_type(ctx))?;
            // FP constants carry their IEEE bit pattern.
            let imm = if let Some(fp32) = attr.downcast_ref::<FPSingleAttr>() {
                pliron::utils::apfloat::Float::to_bits(fp32.0)
            } else if let Some(fp64) = attr.downcast_ref::<FPDoubleAttr>() {
                pliron::utils::apfloat::Float::to_bits(fp64.0)
            } else if let Some(int) = attr.downcast_ref::<IntegerAttr>() {
                int.value().to_u128()
            } else {
                return Err(input_error_noloc!(NvptxErr::UnsupportedOp(format!(
                    "constant of unsupported attribute kind {attr:?}"
                ))));
            };
            // Constants emit nothing here: uses fold them as immediates
            // or materialize them on demand ([Self::lookup]).
            self.consts.insert(
                result,
                ConstVal {
                    class,
                    imm: imm & width_mask(width),
                    width,
                },
            );
        } else if let Some(kind) = binary_float_kind(&*op_obj) {
            self.emit_float_binary(op_ptr, kind)?;
        } else if let Some(fneg) = op_obj.downcast_ref::<FNegOp>() {
            let src = self.lookup(fneg.get_operand(ctx))?;
            let dst = self.fresh(src.class);
            self.inst(format!("neg.{} {dst}, {src};", src.class.float_ty()));
            self.values.insert(fneg.get_result(ctx), dst);
        } else if let Some(fcmp) = op_obj.downcast_ref::<FCmpOp>() {
            self.emit_fcmp(fcmp)?;
        } else if let Some(select) = op_obj.downcast_ref::<SelectOp>() {
            self.emit_select(select)?;
        } else if let Some(cast) = op_obj.downcast_ref::<SIToFPOp>() {
            self.emit_int_to_float(cast.get_operand(ctx), cast.get_result(ctx), true)?;
        } else if let Some(cast) = op_obj.downcast_ref::<UIToFPOp>() {
            self.emit_int_to_float(cast.get_operand(ctx), cast.get_result(ctx), false)?;
        } else if let Some(cast) = op_obj.downcast_ref::<FPToSIOp>() {
            self.emit_float_to_int(cast.get_operand(ctx), cast.get_result(ctx), true)?;
        } else if let Some(cast) = op_obj.downcast_ref::<FPToUIOp>() {
            self.emit_float_to_int(cast.get_operand(ctx), cast.get_result(ctx), false)?;
        } else if let Some(cast) = op_obj.downcast_ref::<FPExtOp>() {
            let src = self.lookup(cast.get_operand(ctx))?;
            let dst = self.fresh(RegClass::F64);
            self.inst(format!("cvt.f64.f32 {dst}, {src};"));
            self.values.insert(cast.get_result(ctx), dst);
        } else if let Some(cast) = op_obj.downcast_ref::<FPTruncOp>() {
            let src = self.lookup(cast.get_operand(ctx))?;
            let dst = self.fresh(RegClass::F32);
            self.inst(format!("cvt.rn.f32.f64 {dst}, {src};"));
            self.values.insert(cast.get_result(ctx), dst);
        } else if let Some(addr) = op_obj.downcast_ref::<AddressOfOp>() {
            let name = addr.get_global_name(ctx).to_string();
            let Some(space) = self.globals.get(&name).copied() else {
                return Err(input_error_noloc!(NvptxErr::UnsupportedGlobal(format!(
                    "address of `{name}`, which is not a global of the kernel module"
                ))));
            };
            let result = addr.get_result(ctx);
            let dst = self.fresh(RegClass::B64);
            self.inst(format!("mov.u64 {dst}, {name};"));
            match space {
                // A shared address stays RAW while provenance proves every
                // consumer takes the `.shared`-qualified path; it converts
                // to generic only when some use escapes the analysis.
                GlobalSpace::Shared if self.space_of(result) == PtrSpace::Shared => {}
                GlobalSpace::Shared => self.inst(format!("cvta.shared.u64 {dst}, {dst};")),
                GlobalSpace::Global => self.inst(format!("cvta.global.u64 {dst}, {dst};")),
            }
            self.values.insert(result, dst);
        } else if let Some(undef) = op_obj.downcast_ref::<UndefOp>() {
            let result = undef.get_result(ctx);
            self.emit_undef_like(result)?;
        } else if let Some(poison) = op_obj.downcast_ref::<PoisonOp>() {
            let result = poison.get_result(ctx);
            self.emit_undef_like(result)?;
        } else if let Some(insert) = op_obj.downcast_ref::<InsertValueOp>() {
            let (aggregate, inserted) = {
                let op = insert.get_operation().deref(ctx);
                (op.get_operand(0), op.get_operand(1))
            };
            let result = insert.get_result(ctx);
            let indices = insert.indices(ctx);
            let agg_ty = aggregate.get_type(ctx);
            let (start, sub_ty) = agg_path(ctx, agg_ty, &indices)?;
            let mut leaves = self
                .aggs
                .get(&aggregate)
                .cloned()
                .ok_or_else(|| {
                    input_error_noloc!(NvptxErr::UndefinedValue(format!("{aggregate:?}")))
                })?;
            if is_aggregate(ctx, sub_ty) {
                // Splicing a sub-aggregate copies its leaves.
                let sub = self.aggs.get(&inserted).cloned().ok_or_else(|| {
                    input_error_noloc!(NvptxErr::UndefinedValue(format!("{inserted:?}")))
                })?;
                leaves[start..start + sub.len()].clone_from_slice(&sub);
            } else {
                leaves[start] = Some(self.lookup(inserted)?);
            }
            self.aggs.insert(result, leaves);
        } else if let Some(extract) = op_obj.downcast_ref::<ExtractValueOp>() {
            let aggregate = extract.get_operation().deref(ctx).get_operand(0);
            let result = extract.get_result(ctx);
            let indices = extract.indices(ctx);
            let agg_ty = aggregate.get_type(ctx);
            let (start, sub_ty) = agg_path(ctx, agg_ty, &indices)?;
            let leaves = self.aggs.get(&aggregate).cloned().ok_or_else(|| {
                input_error_noloc!(NvptxErr::UndefinedValue(format!("{aggregate:?}")))
            })?;
            if is_aggregate(ctx, sub_ty) {
                let count = agg_leaf_count(ctx, sub_ty)?;
                self.aggs
                    .insert(result, leaves[start..start + count].to_vec());
            } else {
                let class = classify(ctx, sub_ty)?;
                let reg = match leaves[start] {
                    Some(reg) => reg,
                    // Extracting an undef leaf: any value works; use 0.
                    None => self.materialize_const(class, 0),
                };
                self.values.insert(result, reg);
            }
        } else if let Some(kind) = binary_int_kind(&*op_obj) {
            self.emit_binary(op_ptr, kind)?;
        } else if let Some(icmp) = op_obj.downcast_ref::<ICmpOp>() {
            self.emit_icmp(icmp)?;
        } else if let Some(zext) = op_obj.downcast_ref::<ZExtOp>() {
            self.emit_zext(zext)?;
        } else if let Some(sext) = op_obj.downcast_ref::<SExtOp>() {
            self.emit_sext(sext)?;
        } else if let Some(trunc) = op_obj.downcast_ref::<TruncOp>() {
            self.emit_trunc(trunc)?;
        } else if let Some(cast) = op_obj.downcast_ref::<BitcastOp>() {
            self.emit_reg_alias(cast.get_operand(ctx), cast.get_result(ctx))?;
        } else if let Some(cast) = op_obj.downcast_ref::<IntToPtrOp>() {
            self.emit_reg_alias(cast.get_operand(ctx), cast.get_result(ctx))?;
        } else if let Some(cast) = op_obj.downcast_ref::<PtrToIntOp>() {
            self.emit_reg_alias(cast.get_operand(ctx), cast.get_result(ctx))?;
        } else if let Some(gep) = op_obj.downcast_ref::<GetElementPtrOp>() {
            self.emit_gep(gep)?;
        } else if let Some(load) = op_obj.downcast_ref::<LoadOp>() {
            self.emit_load(load)?;
        } else if let Some(store) = op_obj.downcast_ref::<StoreOp>() {
            self.emit_store(store)?;
        } else if let Some(call) = op_obj.downcast_ref::<CallOp>() {
            self.emit_call(call)?;
        } else if let Some(br) = op_obj.downcast_ref::<BrOp>() {
            let dest = br.get_operation().deref(ctx).get_successor(0);
            let args = br.successor_operands(ctx, 0);
            // A single outgoing edge: the copies can sit inline before the
            // branch.
            self.emit_block_arg_copies(dest, &args)?;
            let label = self.block_labels[&dest].clone();
            self.inst(format!("bra {label};"));
        } else if let Some(cond_br) = op_obj.downcast_ref::<CondBrOp>() {
            self.emit_cond_br(cond_br)?;
        } else if let Some(ret) = op_obj.downcast_ref::<ReturnOp>() {
            match (ret.retval(ctx), self.ret_param.clone()) {
                (None, None) => self.inst("ret;".to_string()),
                (Some(value), Some((ret_name, class))) => {
                    let reg = self.lookup(value)?;
                    let ty = param_decl_ty(class)?;
                    if reg.class != class {
                        return Err(input_error_noloc!(NvptxErr::UnsupportedOp(
                            "return value class does not match the function signature"
                                .to_string()
                        )));
                    }
                    self.inst(format!("st.param.{ty} [{ret_name}], {reg};"));
                    self.inst("ret;".to_string());
                }
                (Some(_), None) => {
                    return Err(input_error_noloc!(NvptxErr::UnsupportedOp(
                        "kernel functions must return void".to_string()
                    )));
                }
                (None, Some(_)) => {
                    return Err(input_error_noloc!(NvptxErr::UnsupportedOp(
                        "return without a value in a function returning one".to_string()
                    )));
                }
            }
        } else if op_obj.downcast_ref::<UnreachableOp>().is_some() {
            self.inst("trap;".to_string());
        } else if op_obj.downcast_ref::<AllocaOp>().is_some() {
            let sym = self
                .local_syms
                .get(&op_ptr)
                .cloned()
                .expect("plan_local_allocas visited every alloca");
            let result = op_ptr.deref(ctx).get_result(0);
            // Materialize the generic address once: `.local` symbols are
            // window addresses, converted with cvta.local. The residue of
            // mem2reg is cold, so the generic ld/st forms downstream are
            // acceptable.
            let dst = self.fresh(RegClass::B64);
            self.inst(format!("mov.u64 {dst}, {sym};"));
            self.inst(format!("cvta.local.u64 {dst}, {dst};"));
            self.values.insert(result, dst);
        } else {
            return Err(input_error_noloc!(NvptxErr::UnsupportedOp(
                Operation::get_opid(op_ptr, ctx).to_string()
            )));
        }
        Ok(())
    }

    fn emit_binary(&mut self, op_ptr: Ptr<Operation>, kind: BinaryIntKind) -> CrabbitResult<()> {
        let ctx = self.ctx;
        let (lhs, rhs, result) = {
            let op_deref = op_ptr.deref(ctx);
            (
                op_deref.get_operand(0),
                op_deref.get_operand(1),
                op_deref.get_result(0),
            )
        };

        let width = width_of(ctx, result.get_type(ctx))?;
        if width == 1 {
            return Err(input_error_noloc!(NvptxErr::UnsupportedOp(
                "i1 arithmetic is not supported in NVPTX emission".to_string()
            )));
        }
        let class = classify(ctx, result.get_type(ctx))?;
        let bits = if class == RegClass::B64 { 64 } else { 32 };
        let is_shift = matches!(
            kind,
            BinaryIntKind::Shl | BinaryIntKind::LShr | BinaryIntKind::AShr
        );
        let signed_consumer = matches!(
            kind,
            BinaryIntKind::SDiv | BinaryIntKind::SRem | BinaryIntKind::AShr
        );

        let mut a = self.lookup(lhs)?;
        // Signed consumers of sub-32-bit values need real sign bits.
        if signed_consumer {
            a = self.signed32(a, width);
        }

        // The right operand folds to a PTX immediate when constant.
        let b = if let Some((imm, imm_width)) = self.int_imm(rhs) {
            if is_shift {
                // Shift amounts are `.u32` operands in PTX.
                format!("{}", imm as u32)
            } else if signed_consumer && imm_width < 32 {
                // The immediate carries its real sign bits directly.
                format!("{}", sext_from(imm, imm_width) as u32)
            } else if bits == 64 {
                format!("{}", imm as u64)
            } else {
                format!("{}", imm as u32)
            }
        } else {
            let mut b = self.lookup(rhs)?;
            if signed_consumer && kind != BinaryIntKind::AShr {
                b = self.signed32(b, width);
            }
            // 64-bit shift amounts are `.u32` in PTX.
            if is_shift && b.class == RegClass::B64 {
                let amount = self.fresh(RegClass::B32);
                self.inst(format!("cvt.u32.u64 {amount}, {b};"));
                b = amount;
            }
            b.to_string()
        };

        let dst = self.fresh(class);
        let mnemonic = match kind {
            BinaryIntKind::Add => format!("add.s{bits}"),
            BinaryIntKind::Sub => format!("sub.s{bits}"),
            BinaryIntKind::Mul => format!("mul.lo.s{bits}"),
            BinaryIntKind::SDiv => format!("div.s{bits}"),
            BinaryIntKind::UDiv => format!("div.u{bits}"),
            BinaryIntKind::SRem => format!("rem.s{bits}"),
            BinaryIntKind::URem => format!("rem.u{bits}"),
            BinaryIntKind::And => format!("and.b{bits}"),
            BinaryIntKind::Or => format!("or.b{bits}"),
            BinaryIntKind::Xor => format!("xor.b{bits}"),
            BinaryIntKind::Shl => format!("shl.b{bits}"),
            BinaryIntKind::LShr => format!("shr.u{bits}"),
            BinaryIntKind::AShr => format!("shr.s{bits}"),
        };
        self.inst(format!("{mnemonic} {dst}, {a}, {b};"));

        // Keep the zero-extended invariant for narrow results.
        if matches!(
            kind,
            BinaryIntKind::Add
                | BinaryIntKind::Sub
                | BinaryIntKind::Mul
                | BinaryIntKind::Shl
                | BinaryIntKind::SDiv
                | BinaryIntKind::SRem
                | BinaryIntKind::AShr
        ) {
            self.mask32(dst, width);
        }
        self.values.insert(result, dst);
        Ok(())
    }

    fn emit_icmp(&mut self, icmp: &ICmpOp) -> CrabbitResult<()> {
        let ctx = self.ctx;
        let lhs = icmp.get_operation().deref(ctx).get_operand(0);
        let rhs = icmp.get_operation().deref(ctx).get_operand(1);
        let result = icmp.get_result(ctx);
        let predicate = icmp.predicate(ctx);

        let width = width_of(ctx, lhs.get_type(ctx))?;
        if width == 1 {
            // i1 operands live in `.pred` registers (bool compares and MIR's
            // `Not(bool)` reach here as icmp on i1): eq = not(xor), ne = xor,
            // with the constant-side forms folding to a copy or a `not.pred`.
            let invert = match predicate {
                ICmpPredicateAttr::EQ => true,
                ICmpPredicateAttr::NE => false,
                other => {
                    return Err(input_error_noloc!(NvptxErr::UnsupportedOp(format!(
                        "icmp {other:?} on i1 is not supported in NVPTX emission (only eq/ne)"
                    ))));
                }
            };
            let lhs_const = self.consts.get(&lhs).map(|c| (c.imm & 1) != 0);
            let rhs_const = self.consts.get(&rhs).map(|c| (c.imm & 1) != 0);
            let dst = match (lhs_const, rhs_const) {
                (Some(a), Some(b)) => {
                    self.materialize_const(RegClass::Pred, ((a == b) == invert) as u128)
                }
                (Some(k), None) | (None, Some(k)) => {
                    let var = if lhs_const.is_some() { rhs } else { lhs };
                    let reg = self.lookup(var)?;
                    if reg.class != RegClass::Pred {
                        return Err(input_error_noloc!(NvptxErr::UnsupportedOp(format!(
                            "icmp on i1 with a non-pred operand register {:?}",
                            reg.class
                        ))));
                    }
                    if invert == k {
                        reg // p == true / p != false: the predicate itself
                    } else {
                        let d = self.fresh(RegClass::Pred);
                        self.inst(format!("not.pred {d}, {reg};"));
                        d
                    }
                }
                (None, None) => {
                    let a = self.lookup(lhs)?;
                    let b = self.lookup(rhs)?;
                    if a.class != RegClass::Pred || b.class != RegClass::Pred {
                        return Err(input_error_noloc!(NvptxErr::UnsupportedOp(format!(
                            "icmp on i1 with non-pred operand registers {:?}/{:?}",
                            a.class, b.class
                        ))));
                    }
                    let x = self.fresh(RegClass::Pred);
                    self.inst(format!("xor.pred {x}, {a}, {b};"));
                    if invert {
                        let d = self.fresh(RegClass::Pred);
                        self.inst(format!("not.pred {d}, {x};"));
                        d
                    } else {
                        x
                    }
                }
            };
            self.values.insert(result, dst);
            return Ok(());
        }
        let mut a = self.lookup(lhs)?;
        let bits = if a.class == RegClass::B64 { 64 } else { 32 };

        let (cmp, signed) = match predicate {
            ICmpPredicateAttr::EQ => ("eq", None),
            ICmpPredicateAttr::NE => ("ne", None),
            ICmpPredicateAttr::SLT => ("lt", Some(true)),
            ICmpPredicateAttr::SLE => ("le", Some(true)),
            ICmpPredicateAttr::SGT => ("gt", Some(true)),
            ICmpPredicateAttr::SGE => ("ge", Some(true)),
            ICmpPredicateAttr::ULT => ("lt", Some(false)),
            ICmpPredicateAttr::ULE => ("le", Some(false)),
            ICmpPredicateAttr::UGT => ("gt", Some(false)),
            ICmpPredicateAttr::UGE => ("ge", Some(false)),
        };
        if signed == Some(true) {
            a = self.signed32(a, width);
        }
        // The right operand folds to an immediate when constant (with
        // real sign bits for the signed predicates).
        let b = if let Some((imm, imm_width)) = self.int_imm(rhs) {
            if signed == Some(true) && imm_width < 32 {
                format!("{}", sext_from(imm, imm_width) as u32)
            } else if bits == 64 {
                format!("{}", imm as u64)
            } else {
                format!("{}", imm as u32)
            }
        } else {
            let mut b = self.lookup(rhs)?;
            if signed == Some(true) {
                b = self.signed32(b, width);
            }
            b.to_string()
        };
        let ty = match signed {
            None => format!("b{bits}"),
            Some(true) => format!("s{bits}"),
            Some(false) => format!("u{bits}"),
        };
        let pred = self.fresh(RegClass::Pred);
        self.inst(format!("setp.{cmp}.{ty} {pred}, {a}, {b};"));
        self.values.insert(result, pred);
        Ok(())
    }

    fn emit_zext(&mut self, zext: &ZExtOp) -> CrabbitResult<()> {
        let ctx = self.ctx;
        let src_val = zext.get_operand(ctx);
        let result = zext.get_result(ctx);
        let src = self.lookup(src_val)?;
        let dst_class = classify(ctx, result.get_type(ctx))?;
        match (src.class, dst_class) {
            (RegClass::Pred, RegClass::B32) => {
                let dst = self.fresh(RegClass::B32);
                self.inst(format!("selp.b32 {dst}, 1, 0, {src};"));
                self.values.insert(result, dst);
            }
            (RegClass::Pred, RegClass::B64) => {
                let dst = self.fresh(RegClass::B64);
                self.inst(format!("selp.b64 {dst}, 1, 0, {src};"));
                self.values.insert(result, dst);
            }
            (RegClass::B32, RegClass::B64) => {
                let dst = self.fresh(RegClass::B64);
                self.inst(format!("cvt.u64.u32 {dst}, {src};"));
                self.values.insert(result, dst);
            }
            // Sub-32-bit values are already zero-extended in their register.
            (RegClass::B32, RegClass::B32) => {
                self.values.insert(result, src);
            }
            (from, to) => {
                return Err(input_error_noloc!(NvptxErr::UnsupportedOp(format!(
                    "zext from {from:?} to {to:?}"
                ))));
            }
        }
        Ok(())
    }

    fn emit_sext(&mut self, sext: &SExtOp) -> CrabbitResult<()> {
        let ctx = self.ctx;
        let src_val = sext.get_operand(ctx);
        let result = sext.get_result(ctx);
        let src = self.lookup(src_val)?;
        let src_width = width_of(ctx, src_val.get_type(ctx))?;
        let dst_width = width_of(ctx, result.get_type(ctx))?;
        let dst_class = classify(ctx, result.get_type(ctx))?;
        match (src.class, dst_class) {
            (RegClass::B32, RegClass::B64) => {
                let dst = self.fresh(RegClass::B64);
                self.inst(format!("cvt.s64.s{src_width} {dst}, {src};"));
                self.values.insert(result, dst);
            }
            (RegClass::B32, RegClass::B32) => {
                let extended = self.signed32(src, src_width);
                self.mask32(extended, dst_width);
                self.values.insert(result, extended);
            }
            (from, to) => {
                return Err(input_error_noloc!(NvptxErr::UnsupportedOp(format!(
                    "sext from {from:?} to {to:?}"
                ))));
            }
        }
        Ok(())
    }

    fn emit_trunc(&mut self, trunc: &TruncOp) -> CrabbitResult<()> {
        let ctx = self.ctx;
        let src_val = trunc.get_operand(ctx);
        let result = trunc.get_result(ctx);
        let src = self.lookup(src_val)?;
        let dst_width = width_of(ctx, result.get_type(ctx))?;
        let dst_class = classify(ctx, result.get_type(ctx))?;
        match (src.class, dst_class) {
            (RegClass::B64, RegClass::B32) => {
                let dst = self.fresh(RegClass::B32);
                self.inst(format!("cvt.u32.u64 {dst}, {src};"));
                self.mask32(dst, dst_width);
                self.values.insert(result, dst);
            }
            (RegClass::B32, RegClass::B32) => {
                // Truncation within the .b32 register is just the mask.
                if dst_width < 32 {
                    let dst = self.fresh(RegClass::B32);
                    let mask = (1u32 << dst_width) - 1;
                    self.inst(format!("and.b32 {dst}, {src}, {mask};"));
                    self.values.insert(result, dst);
                } else {
                    self.values.insert(result, src);
                }
            }
            (RegClass::B32, RegClass::Pred) | (RegClass::B64, RegClass::Pred) => {
                let bit = self.fresh(RegClass::B32);
                if src.class == RegClass::B64 {
                    self.inst(format!("cvt.u32.u64 {bit}, {src};"));
                } else {
                    self.inst(format!("mov.b32 {bit}, {src};"));
                }
                self.inst(format!("and.b32 {bit}, {bit}, 1;"));
                let pred = self.fresh(RegClass::Pred);
                self.inst(format!("setp.ne.b32 {pred}, {bit}, 0;"));
                self.values.insert(result, pred);
            }
            (from, to) => {
                return Err(input_error_noloc!(NvptxErr::UnsupportedOp(format!(
                    "trunc from {from:?} to {to:?}"
                ))));
            }
        }
        Ok(())
    }

    /// Casts that only reinterpret a register (bitcast, inttoptr, ptrtoint)
    /// alias the operand's register — except across the raw-shared/generic
    /// representation boundary, where the alias becomes the `cvta`.
    fn emit_reg_alias(&mut self, src_val: Value, result: Value) -> CrabbitResult<()> {
        let src = self.lookup(src_val)?;
        let dst_class = classify(self.ctx, result.get_type(self.ctx))?;
        if src.class != dst_class {
            // A same-width int<->float bitcast (`f32::from_bits`/`to_bits`)
            // is a bit-preserving register move: PTX `mov.b32`/`mov.b64`
            // accepts mixed operand register classes.
            let mov = match (src.class, dst_class) {
                (RegClass::B32, RegClass::F32) | (RegClass::F32, RegClass::B32) => Some("mov.b32"),
                (RegClass::B64, RegClass::F64) | (RegClass::F64, RegClass::B64) => Some("mov.b64"),
                _ => None,
            };
            if let Some(mov) = mov {
                let dst = self.fresh(dst_class);
                self.inst(format!("{mov} {dst}, {src};"));
                self.values.insert(result, dst);
                return Ok(());
            }
            return Err(input_error_noloc!(NvptxErr::UnsupportedOp(format!(
                "register-aliasing cast between {:?} and {:?}",
                src.class, dst_class
            ))));
        }
        if src.class == RegClass::B64
            && self.space_of(src_val) == PtrSpace::Shared
            && self.space_of(result) != PtrSpace::Shared
        {
            let dst = self.fresh(RegClass::B64);
            self.inst(format!("cvta.shared.u64 {dst}, {src};"));
            self.values.insert(result, dst);
            return Ok(());
        }
        self.values.insert(result, src);
        Ok(())
    }

    // Folded addressing ---------------------------------------------------

    /// The register holding `plan.root + Σ plan.terms` (NOT the constant
    /// offset), materialized once per block and distinct base.
    fn base_reg_for(&mut self, plan: &FoldedAddr) -> CrabbitResult<Reg> {
        let key = plan.base_key();
        if let Some(&reg) = self.base_regs.get(&key) {
            return Ok(reg);
        }
        let mut reg = self.lookup(plan.root)?;
        for term in &plan.terms {
            let wide = self.term_wide(term)?;
            // Power-of-two scales materialize as shift+add: ptxas lowers
            // those to (hoistable) LEA pairs, while a mad.lo.s64 becomes
            // a genuine 64-bit multiply sequence it neither strength-
            // reduces nor hoists as readily (measured on gemm_control).
            if term.scale == 1 {
                let next = self.fresh(RegClass::B64);
                self.inst(format!("add.s64 {next}, {reg}, {wide};"));
                reg = next;
            } else if term.scale > 1 && (term.scale as u64).is_power_of_two() {
                let shifted = self.fresh(RegClass::B64);
                self.inst(format!(
                    "shl.b64 {shifted}, {wide}, {};",
                    term.scale.trailing_zeros()
                ));
                let next = self.fresh(RegClass::B64);
                self.inst(format!("add.s64 {next}, {reg}, {shifted};"));
                reg = next;
            } else {
                let next = self.fresh(RegClass::B64);
                self.inst(format!("mad.lo.s64 {next}, {wide}, {}, {reg};", term.scale));
                reg = next;
            }
        }
        self.base_regs.insert(key, reg);
        Ok(reg)
    }

    /// A term's value widened to 64 bits per its [TermKind], cached per
    /// block.
    fn term_wide(&mut self, term: &Term) -> CrabbitResult<Reg> {
        if let Some(&reg) = self.wide_terms.get(term) {
            return Ok(reg);
        }
        let reg = self.lookup(term.value)?;
        let wide = match term.kind {
            TermKind::I64 => reg,
            TermKind::Zext32 => {
                let wide = self.fresh(RegClass::B64);
                self.inst(format!("cvt.u64.u32 {wide}, {reg};"));
                wide
            }
            TermKind::Sext(width) => {
                let extended = self.signed32(reg, width);
                let wide = self.fresh(RegClass::B64);
                self.inst(format!("cvt.s64.s32 {wide}, {extended};"));
                wide
            }
        };
        self.wide_terms.insert(*term, wide);
        Ok(wide)
    }

    /// The memory operand for an access through `addr`: base register and
    /// the constant byte offset to fold into the instruction (`[reg+imm]`).
    fn mem_operand(&mut self, addr: Value) -> CrabbitResult<(Reg, i64)> {
        if let Some(plan) = self.addr_plan.get(&addr).cloned() {
            let base = self.base_reg_for(&plan)?;
            return Ok((base, plan.offset));
        }
        Ok((self.lookup(addr)?, 0))
    }

    /// Group this block's foldable `.shared` loads into `v2`/`v4` vector
    /// loads (module docs). Same folded base, consecutive constant
    /// offsets, no store/call (barrier, atomic) between the members, and
    /// a provable group alignment: the shared global's declaration is
    /// raised to the vector size and every dynamic term's scale must be a
    /// multiple of it.
    fn plan_vector_loads(&mut self, ops: &[Ptr<Operation>]) -> CrabbitResult<()> {
        let ctx = self.ctx;
        self.vec_lead.clear();
        self.vec_covered.clear();
        type Members = BTreeMap<i64, (usize, Ptr<Operation>, Value)>;
        type BucketKey = (usize, (Value, Vec<Term>), RegClass);
        let mut buckets: HashMap<BucketKey, (String, Members)> = HashMap::new();
        let mut era = 0usize;
        for (order, &op_ptr) in ops.iter().enumerate() {
            let op_obj = Operation::get_op_dyn(op_ptr, ctx);
            if op_obj.downcast_ref::<StoreOp>().is_some()
                || op_obj.downcast_ref::<CallOp>().is_some()
            {
                // A write or an opaque effect (barrier, atomic, device
                // call): loads across it must not merge.
                era += 1;
                continue;
            }
            let Some(load) = op_obj.downcast_ref::<LoadOp>() else {
                continue;
            };
            if self.skip_ops.contains(&op_ptr) {
                continue;
            }
            let addr_val = load.get_operand_address(ctx);
            let Some(plan) = self.addr_plan.get(&addr_val) else {
                continue;
            };
            if self.space_of(addr_val) != PtrSpace::Shared {
                continue;
            }
            let result = load.get_result(ctx);
            let ty = result.get_type(ctx);
            if is_aggregate(ctx, ty) {
                continue;
            }
            let (Ok(class), Ok(width)) = (classify(ctx, ty), width_of(ctx, ty)) else {
                continue;
            };
            let full_width = match class {
                RegClass::B32 | RegClass::F32 => width == 32,
                RegClass::B64 | RegClass::F64 => width == 64,
                RegClass::Pred => false,
            };
            if !full_width {
                continue;
            }
            // The alignment proof needs a base whose declaration we own:
            // the address of a `.shared` module global.
            let Some(root_def) = plan.root.defining_op() else {
                continue;
            };
            let root_obj = Operation::get_op_dyn(root_def, ctx);
            let Some(address_of) = root_obj.downcast_ref::<AddressOfOp>() else {
                continue;
            };
            let symbol = address_of.get_global_name(ctx).to_string();
            if self.globals.get(&symbol) != Some(&GlobalSpace::Shared) {
                continue;
            }
            let (_, members) = buckets
                .entry((era, plan.base_key(), class))
                .or_insert_with(|| (symbol, Members::new()));
            // A duplicate offset stays a scalar load (ptxas CSEs it).
            members.entry(plan.offset).or_insert((order, op_ptr, result));
        }

        for ((_, (root, terms), class), (symbol, members)) in buckets {
            let elem: i64 = match class {
                RegClass::B32 | RegClass::F32 => 4,
                RegClass::B64 | RegClass::F64 => 8,
                RegClass::Pred => continue,
            };
            let ld_ty = match class {
                RegClass::B32 => "u32",
                RegClass::B64 => "u64",
                RegClass::F32 => "f32",
                RegClass::F64 => "f64",
                RegClass::Pred => continue,
            };
            let plan = FoldedAddr {
                root,
                terms,
                offset: 0,
            };
            let dyn_align = plan.dynamic_align();
            let offsets: Vec<i64> = members.keys().copied().collect();
            let present: HashSet<i64> = offsets.iter().copied().collect();
            let mut taken: HashSet<i64> = HashSet::new();
            for &start in &offsets {
                if taken.contains(&start) {
                    continue;
                }
                let quad = elem == 4
                    && dyn_align >= 16
                    && start % 16 == 0
                    && (1..4).all(|lane| {
                        let off = start + lane * elem;
                        present.contains(&off) && !taken.contains(&off)
                    });
                let pair_align = 2 * elem;
                let pair = !quad
                    && dyn_align >= pair_align
                    && start % pair_align == 0
                    && present.contains(&(start + elem))
                    && !taken.contains(&(start + elem));
                let lanes: i64 = if quad {
                    4
                } else if pair {
                    2
                } else {
                    continue;
                };
                let group_align = (lanes * elem) as u64;
                let mut parts: Vec<(i64, Value)> = Vec::new();
                let mut leader: Option<(usize, Ptr<Operation>)> = None;
                let mut group_ops: Vec<Ptr<Operation>> = Vec::new();
                for lane in 0..lanes {
                    let off = start + lane * elem;
                    taken.insert(off);
                    let (order, op_ptr, result) = members[&off];
                    parts.push((off, result));
                    group_ops.push(op_ptr);
                    if leader.is_none_or(|(best, _)| order < best) {
                        leader = Some((order, op_ptr));
                    }
                }
                let (_, leader_op) = leader.expect("group has members");
                for op_ptr in group_ops {
                    if op_ptr != leader_op {
                        self.vec_covered.insert(op_ptr);
                    }
                }
                self.vec_lead.insert(
                    leader_op,
                    VecGroup {
                        plan: plan.clone(),
                        parts,
                        class,
                        ld_ty,
                    },
                );
                let raise = self.align_raises.entry(symbol.clone()).or_insert(group_align);
                *raise = (*raise).max(group_align);
            }
        }
        Ok(())
    }

    /// Group narrow (u8/u16) loads of read-only-proven memory whose folded
    /// offsets fully cover an aligned 4-byte window into one 32-bit load
    /// plus per-lane extracts (module docs). Function-level, unlike the
    /// vector plan: read-only-ness makes intervening writes irrelevant, so
    /// a group may span blocks — the leader need only dominate every
    /// member (its window register feeds their extracts).
    fn plan_byte_combines(
        &mut self,
        blocks: &[Ptr<BasicBlock>],
        region: Ptr<crate::ir::region::Region>,
    ) -> CrabbitResult<()> {
        if std::env::var("CRABBIT_NVPTX_BYTECOMB").is_ok_and(|v| v == "0") {
            return Ok(());
        }
        let ctx = self.ctx;
        // (base_key, width) → offset → first load at that offset.
        type Member = (usize, Ptr<Operation>, Value, Value, Ptr<BasicBlock>);
        type Members = BTreeMap<i64, Member>;
        type BucketKey = ((Value, Vec<Term>), u32);
        let mut buckets: HashMap<BucketKey, (String, Members)> = HashMap::new();
        let mut order = 0usize;
        for &block in blocks {
            for op_ptr in block.deref(ctx).iter(ctx) {
                order += 1;
                let op_obj = Operation::get_op_dyn(op_ptr, ctx);
                let Some(load) = op_obj.downcast_ref::<LoadOp>() else {
                    continue;
                };
                let addr = load.get_operand_address(ctx);
                let Some(plan) = self.addr_plan.get(&addr) else {
                    continue;
                };
                // Read-only-proven memory only: this is what licenses both
                // ignoring intervening stores and (with `.nc`) the window
                // load itself reading bytes claimed by other paths.
                if !self.readonly.contains(&addr) {
                    continue;
                }
                let result = load.get_result(ctx);
                let ty = result.get_type(ctx);
                let width = match ty.deref(ctx).downcast_ref::<IntegerType>() {
                    Some(int_ty) if matches!(int_ty.width(), 8 | 16) => int_ty.width(),
                    _ => continue,
                };
                // A u16 lane must sit at an even window offset.
                if width == 16 && plan.offset.rem_euclid(2) != 0 {
                    continue;
                }
                // Alignment: the dynamic terms keep 4, and the base itself
                // is provably 4-aligned relative to a module global whose
                // declared alignment we own (raised below).
                if plan.dynamic_align() < 4 {
                    continue;
                }
                let Some(symbol) = self.window_base_symbol(plan.root) else {
                    continue;
                };
                let (_, members) = buckets
                    .entry((plan.base_key(), width))
                    .or_insert_with(|| (symbol, Members::new()));
                // A duplicate offset stays a scalar load (ptxas CSEs it).
                members
                    .entry(plan.offset)
                    .or_insert((order, op_ptr, addr, result, block));
            }
        }
        if buckets.is_empty() {
            return Ok(());
        }

        let dom = crate::passes::llvm::analysis::dominator_tree(ctx, region);
        for (((root, terms), width), (symbol, members)) in buckets {
            let mut taken: HashSet<i64> = HashSet::new();
            let offsets: Vec<i64> = members.keys().copied().collect();
            for &offset in &offsets {
                if taken.contains(&offset) {
                    continue;
                }
                let window = offset.div_euclid(4) * 4;
                let lane_offsets: Vec<i64> = match width {
                    8 => (0..4).map(|j| window + j).collect(),
                    16 => vec![window, window + 2],
                    _ => unreachable!("planner admits widths 8 and 16 only"),
                };
                // Full coverage only: every byte of the window must be one
                // of the group's own loads.
                if !lane_offsets
                    .iter()
                    .all(|off| members.contains_key(off) && !taken.contains(off))
                {
                    continue;
                }
                let group: Vec<(i64, Member)> = lane_offsets
                    .iter()
                    .map(|off| (*off, members[off]))
                    .collect();
                let &(_, (_, leader_op, leader_addr, _, leader_block)) = group
                    .iter()
                    .min_by_key(|(_, (order, ..))| *order)
                    .expect("group has members");
                // The leader's window register must reach every extract.
                let dominated = group.iter().all(|&(_, (_, _, _, _, block))| {
                    dom.contains(&block) && dom.dominates(&leader_block, &block)
                });
                if !dominated {
                    continue;
                }
                let gid = self.byte_groups.len();
                for &(off, (_, op_ptr, _, _, _)) in &group {
                    self.byte_members.insert(
                        op_ptr,
                        ByteCombineMember {
                            group: gid,
                            lane: off - window,
                            width,
                        },
                    );
                    taken.insert(off);
                }
                self.byte_groups.push(ByteCombineGroup {
                    plan: FoldedAddr {
                        root,
                        terms: terms.clone(),
                        offset: window,
                    },
                    leader: leader_op,
                    addr: leader_addr,
                });
                self.byte_window_regs.push(None);
                // The root global's declaration must prove the window's
                // 4-byte alignment.
                let raise = self.align_raises.entry(symbol.clone()).or_insert(4);
                *raise = (*raise).max(4);
            }
        }
        Ok(())
    }

    /// The module global that proves a combine window's 4-byte alignment:
    /// walk `value` down the fold chain — each hop must keep offset ≡ 0
    /// (mod 4) and every dynamic scale a multiple of 4 — to an
    /// `llvm.addressof` of a module global, whose declaration this module
    /// owns (its `.align` is raised to 4 by the planner). A raw-pointer
    /// parameter proves nothing (element alignment only), so a chain
    /// ending anywhere else fails.
    fn window_base_symbol(&self, value: Value) -> Option<String> {
        let ctx = self.ctx;
        let mut current = value;
        loop {
            if let Some(def) = current.defining_op()
                && let Some(address_of) =
                    Operation::get_op_dyn(def, ctx).downcast_ref::<AddressOfOp>()
            {
                let symbol = address_of.get_global_name(ctx).to_string();
                return self.globals.contains_key(&symbol).then_some(symbol);
            }
            let plan = self.addr_plan.get(&current)?;
            if plan.dynamic_align() < 4 || plan.offset.rem_euclid(4) != 0 {
                return None;
            }
            current = plan.root;
        }
    }

    /// The `.nc` qualifier for a load through `addr`: proven-global,
    /// proven-never-written memory in an entry kernel (module docs),
    /// unless `CRABBIT_NVPTX_NC=0` killed it.
    fn nc_suffix(&self, addr: Value) -> &'static str {
        if self.nc_enabled
            && self.space_of(addr) == PtrSpace::Global
            && self.readonly.contains(&addr)
        {
            ".nc"
        } else {
            ""
        }
    }

    fn emit_gep(&mut self, gep: &GetElementPtrOp) -> CrabbitResult<()> {
        let ctx = self.ctx;
        // A folded GEP materializes from its decomposition: the shared
        // base register plus one trailing add for the constant offset
        // (skipped entirely when every use folds — see plan_addresses).
        let result_val = gep.get_result(ctx);
        if let Some(plan) = self.addr_plan.get(&result_val).cloned() {
            let base = self.base_reg_for(&plan)?;
            let reg = if plan.offset == 0 {
                base
            } else {
                let next = self.fresh(RegClass::B64);
                self.inst(format!("add.s64 {next}, {base}, {};", plan.offset));
                next
            };
            self.values.insert(result_val, reg);
            return Ok(());
        }
        let base_val = gep.get_operand_src_ptr(ctx);
        let base = self.lookup(base_val)?;
        let indices = gep.indices(ctx);
        let src_elem_type = gep.src_elem_type(ctx);

        // The address builds incrementally from the base register — no
        // upfront copy: each dynamic index emits one `mad`/`add` into a
        // fresh register, constant offsets fold into one trailing `add`,
        // and a fully constant-zero GEP aliases the base register.
        let mut addr = base;
        let mut constant_offset = 0u64;
        let mut current_ty = src_elem_type;

        for (position, index) in indices.iter().enumerate() {
            // The first index strides over the source element type itself;
            // later indices descend into it.
            let elem_ty = if position == 0 {
                current_ty
            } else {
                let ty_ref = current_ty.deref(ctx);
                if let Some(array_ty) = ty_ref.downcast_ref::<ArrayType>() {
                    array_ty.elem_type()
                } else if let Some(struct_ty) = ty_ref.downcast_ref::<StructType>() {
                    // Struct fields are addressed by constant index only
                    // (LLVM verifies this); the stride is the field's
                    // layout offset, not an element size.
                    if struct_ty.is_opaque() {
                        return Err(input_error_noloc!(NvptxErr::UnsupportedType(
                            "gep into an opaque struct".to_string()
                        )));
                    }
                    let field_index = match index {
                        GepIndex::Constant(value) => *value as u64,
                        GepIndex::Value(value) => {
                            let Some((imm, _)) = self.int_imm(*value) else {
                                return Err(input_error_noloc!(NvptxErr::UnsupportedOp(
                                    "gep struct field index must be constant".to_string()
                                )));
                            };
                            imm as u64
                        }
                    };
                    let (offset, field_ty) = struct_field_offset(ctx, struct_ty, field_index)?;
                    drop(ty_ref);
                    constant_offset = constant_offset.wrapping_add(offset);
                    current_ty = field_ty;
                    continue;
                } else {
                    return Err(input_error_noloc!(NvptxErr::UnsupportedType(format!(
                        "gep into {}",
                        current_ty.disp(ctx)
                    ))));
                }
            };
            let elem_size = align_to(size_of_ty(ctx, elem_ty)?, align_of_ty(ctx, elem_ty)?);
            match index {
                GepIndex::Constant(value) => {
                    constant_offset =
                        constant_offset.wrapping_add((*value as i64 as u64).wrapping_mul(elem_size));
                }
                GepIndex::Value(value) if self.int_imm(*value).is_some() => {
                    // A constant index that arrived as a value: fold it
                    // (dynamic indices are signed, so sign-extend).
                    let (imm, imm_width) = self.int_imm(*value).expect("guard checked");
                    constant_offset = constant_offset
                        .wrapping_add((sext_from(imm, imm_width) as u64).wrapping_mul(elem_size));
                }
                GepIndex::Value(value) => {
                    let index_reg = self.lookup(*value)?;
                    let wide = match index_reg.class {
                        RegClass::B64 => index_reg,
                        RegClass::B32 => {
                            let width = width_of(ctx, value.get_type(ctx))?;
                            let extended = self.signed32(index_reg, width);
                            let wide = self.fresh(RegClass::B64);
                            self.inst(format!("cvt.s64.s32 {wide}, {extended};"));
                            wide
                        }
                        RegClass::Pred | RegClass::F32 | RegClass::F64 => {
                            return Err(input_error_noloc!(NvptxErr::UnsupportedOp(
                                "non-integer gep index".to_string()
                            )));
                        }
                    };
                    let next = self.fresh(RegClass::B64);
                    if elem_size == 1 {
                        self.inst(format!("add.s64 {next}, {addr}, {wide};"));
                    } else {
                        self.inst(format!("mad.lo.s64 {next}, {wide}, {elem_size}, {addr};"));
                    }
                    addr = next;
                }
            }
            current_ty = elem_ty;
        }
        if constant_offset != 0 {
            let next = self.fresh(RegClass::B64);
            self.inst(format!(
                "add.s64 {next}, {addr}, {};",
                constant_offset as i64
            ));
            addr = next;
        }
        let result = gep.get_result(ctx);
        // A raw-shared base yields a raw-shared result; when the result was
        // demoted (some use escapes the analysis) it converts here.
        if self.space_of(base_val) == PtrSpace::Shared
            && self.space_of(result) != PtrSpace::Shared
        {
            let next = self.fresh(RegClass::B64);
            self.inst(format!("cvta.shared.u64 {next}, {addr};"));
            addr = next;
        }
        self.values.insert(result, addr);
        Ok(())
    }

    fn emit_load(&mut self, load: &LoadOp) -> CrabbitResult<()> {
        let ctx = self.ctx;
        let result = load.get_result(ctx);
        let op_ptr = load.get_operation();
        // Sub-word combine plan: the leader loads the whole 4-byte window;
        // every member (leader included) extracts its lane here, at its
        // own position.
        if let Some(member) = self.byte_members.get(&op_ptr).copied() {
            if self.byte_groups[member.group].leader == op_ptr {
                let group = self.byte_groups[member.group].clone();
                let base = self.base_reg_for(&group.plan)?;
                let space = self.space_qualifier(group.addr);
                let nc = self.nc_suffix(group.addr);
                let window = self.fresh(RegClass::B32);
                self.inst(format!(
                    "ld{space}{nc}.u32 {window}, {};",
                    fmt_mem(base, group.plan.offset)
                ));
                self.byte_window_regs[member.group] = Some(window);
            }
            let window = self.byte_window_regs[member.group]
                .expect("byte-combine leader dominates (and so precedes) its members");
            let dst = self.fresh(RegClass::B32);
            match (member.width, member.lane) {
                // prmt selector 0x444j: lane byte j into byte 0, the rest
                // from operand b = 0 — a zero-extended byte in one
                // instruction (PRMT in SASS, nvcc's own byte-table shape).
                (8, lane @ 0..=3) => self.inst(format!(
                    "prmt.b32 {dst}, {window}, 0, 0x{:04X};",
                    0x4440 + lane
                )),
                (16, 0) => self.inst(format!("and.b32 {dst}, {window}, 65535;")),
                (16, 2) => self.inst(format!("shr.u32 {dst}, {window}, 16;")),
                (width, lane) => unreachable!(
                    "byte-combine planner admitted width {width} lane {lane}"
                ),
            }
            self.values.insert(result, dst);
            return Ok(());
        }
        // Vector-load plan: a covered member's value was defined by its
        // group leader's `ld.shared.v2/v4`; the leader emits it here.
        if self.vec_covered.contains(&op_ptr) {
            debug_assert!(
                self.values.contains_key(&result),
                "vector group leader must precede its covered members"
            );
            return Ok(());
        }
        if let Some(group) = self.vec_lead.get(&op_ptr).cloned() {
            let base = self.base_reg_for(&group.plan)?;
            let regs: Vec<Reg> = group.parts.iter().map(|_| self.fresh(group.class)).collect();
            let lanes = regs
                .iter()
                .map(Reg::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            self.inst(format!(
                "ld.shared.v{}.{} {{{lanes}}}, {};",
                group.parts.len(),
                group.ld_ty,
                fmt_mem(base, group.parts[0].0),
            ));
            for ((_, part_result), reg) in group.parts.iter().zip(regs) {
                self.values.insert(*part_result, reg);
            }
            return Ok(());
        }
        // Aggregate loads expand into one load per scalar leaf.
        if is_aggregate(ctx, result.get_type(ctx)) {
            let (base, base_off) = self.mem_operand(load.get_operand_address(ctx))?;
            let space = self.space_qualifier(load.get_operand_address(ctx));
            let nc = self.nc_suffix(load.get_operand_address(ctx));
            let mut leaves = Vec::new();
            agg_leaves(ctx, result.get_type(ctx), 0, &mut leaves)?;
            let mut regs = Vec::with_capacity(leaves.len());
            for (offset, leaf_ty) in leaves {
                let class = classify(ctx, leaf_ty)?;
                let width = width_of(ctx, leaf_ty)?;
                if width == 1 {
                    return Err(input_error_noloc!(NvptxErr::UnsupportedOp(
                        "i1 aggregate field load is not supported in NVPTX emission"
                            .to_string()
                    )));
                }
                let dst = self.fresh(class);
                let at = fmt_mem(base, base_off + offset as i64);
                if class.is_float() {
                    self.inst(format!("ld{space}{nc}.{} {dst}, {at};", class.float_ty()));
                } else {
                    self.inst(format!("ld{space}{nc}.u{width} {dst}, {at};"));
                }
                regs.push(Some(dst));
            }
            self.aggs.insert(result, regs);
            return Ok(());
        }
        let (base, off) = self.mem_operand(load.get_operand_address(ctx))?;
        let width = width_of(ctx, result.get_type(ctx))?;
        if width == 1 {
            return Err(input_error_noloc!(NvptxErr::UnsupportedOp(
                "i1 load is not supported in NVPTX emission".to_string()
            )));
        }
        let class = classify(ctx, result.get_type(ctx))?;
        let space = self.space_qualifier(load.get_operand_address(ctx));
        let nc = self.nc_suffix(load.get_operand_address(ctx));
        let dst = self.fresh(class);
        let at = fmt_mem(base, off);
        if class.is_float() {
            self.inst(format!("ld{space}{nc}.{} {dst}, {at};", class.float_ty()));
        } else {
            // Sub-word loads zero-extend into the wider register, which is
            // exactly the narrow-value invariant.
            self.inst(format!("ld{space}{nc}.u{width} {dst}, {at};"));
        }
        self.values.insert(result, dst);
        Ok(())
    }

    fn emit_store(&mut self, store: &StoreOp) -> CrabbitResult<()> {
        let ctx = self.ctx;
        let value = store.get_operand_value(ctx);
        // Aggregate stores expand into one store per defined scalar leaf
        // (undef leaves store nothing).
        if is_aggregate(ctx, value.get_type(ctx)) {
            let regs = self.aggs.get(&value).cloned().ok_or_else(|| {
                input_error_noloc!(NvptxErr::UndefinedValue(format!("{value:?}")))
            })?;
            let (base, base_off) = self.mem_operand(store.get_operand_address(ctx))?;
            let space = self.space_qualifier(store.get_operand_address(ctx));
            let mut leaves = Vec::new();
            agg_leaves(ctx, value.get_type(ctx), 0, &mut leaves)?;
            for ((offset, leaf_ty), reg) in leaves.into_iter().zip(regs) {
                let Some(reg) = reg else { continue };
                let width = width_of(ctx, leaf_ty)?;
                if width == 1 {
                    return Err(input_error_noloc!(NvptxErr::UnsupportedOp(
                        "i1 aggregate field store is not supported in NVPTX emission"
                            .to_string()
                    )));
                }
                let at = fmt_mem(base, base_off + offset as i64);
                if reg.class.is_float() {
                    self.inst(format!("st{space}.{} {at}, {reg};", reg.class.float_ty()));
                } else {
                    self.inst(format!("st{space}.u{width} {at}, {reg};"));
                }
            }
            return Ok(());
        }
        let width = width_of(ctx, value.get_type(ctx))?;
        if width == 1 {
            return Err(input_error_noloc!(NvptxErr::UnsupportedOp(
                "i1 store is not supported in NVPTX emission".to_string()
            )));
        }
        let src = self.lookup(value)?;
        let (base, off) = self.mem_operand(store.get_operand_address(ctx))?;
        let space = self.space_qualifier(store.get_operand_address(ctx));
        let at = fmt_mem(base, off);
        if src.class.is_float() {
            self.inst(format!("st{space}.{} {at}, {src};", src.class.float_ty()));
        } else {
            self.inst(format!("st{space}.u{width} {at}, {src};"));
        }
        Ok(())
    }

    fn emit_call(&mut self, call: &CallOp) -> CrabbitResult<()> {
        let ctx = self.ctx;
        let callee = match call.callee(ctx) {
            CallOpCallable::Direct(name) => name.to_string(),
            CallOpCallable::Indirect(_) => {
                return Err(input_error_noloc!(NvptxErr::UnsupportedOp(
                    "indirect calls are not supported in kernels yet".to_string()
                )));
            }
        };
        let canonical = nvvm_intrinsic_name(&callee);
        if let Some(sreg) = sreg_for_callee(&canonical) {
            let result = call.get_operation().deref(ctx).get_result(0);
            let dst = self.fresh(RegClass::B32);
            self.inst(format!("mov.u32 {dst}, {sreg};"));
            self.values.insert(result, dst);
            return Ok(());
        }
        let operands: Vec<Value> = {
            let op = call.get_operation().deref(ctx);
            (0..op.get_num_operands()).map(|i| op.get_operand(i)).collect()
        };
        // Direct calls carry only the arguments as operands.
        let args = operands;
        let result = || call.get_operation().deref(ctx).get_result(0);

        match canonical.as_str() {
            "nvvm_barrier0" => {
                self.inst("bar.sync 0;".to_string());
                return Ok(());
            }
            // `__syncwarp(mask)`.
            "nvvm_bar_warp_sync" => {
                if args.len() != 1 {
                    return Err(input_error_noloc!(NvptxErr::UnsupportedOp(format!(
                        "`{callee}` expects (membermask), got {} operands",
                        args.len()
                    ))));
                }
                let mask = self.b32_operand(args[0])?;
                self.inst(format!("bar.warp.sync {mask};"));
                return Ok(());
            }
            "nvvm_membar_gl" => {
                self.inst("membar.gl;".to_string());
                return Ok(());
            }
            "nvvm_membar_cta" => {
                self.inst("membar.cta;".to_string());
                return Ok(());
            }
            "nvvm_membar_sys" => {
                self.inst("membar.sys;".to_string());
                return Ok(());
            }
            _ => {}
        }

        // Warp shuffles: `shfl.sync.<mode>.b32 d, a, b, c, membermask` (the
        // value-returning NVVM forms; b, c and the mask fold as immediates).
        if let Some((mode, class)) = shfl_intrinsic(&canonical) {
            if args.len() != 4 {
                return Err(input_error_noloc!(NvptxErr::UnsupportedOp(format!(
                    "`{callee}` expects (membermask, value, b, c), got {} operands",
                    args.len()
                ))));
            }
            let a = self.lookup(args[1])?;
            if a.class != class {
                return Err(input_error_noloc!(NvptxErr::UnsupportedOp(format!(
                    "`{callee}` value operand is {:?}, expected {class:?}",
                    a.class
                ))));
            }
            let b = self.b32_operand(args[2])?;
            let c = self.b32_operand(args[3])?;
            let mask = self.b32_operand(args[0])?;
            let dst = self.fresh(class);
            self.inst(format!("shfl.sync.{mode}.b32 {dst}, {a}, {b}, {c}, {mask};"));
            self.values.insert(result(), dst);
            return Ok(());
        }

        // Warp votes: `vote.sync.<mode>.{pred,b32} d, a, membermask`.
        if let Some((mode, ballot)) = vote_intrinsic(&canonical) {
            if args.len() != 2 {
                return Err(input_error_noloc!(NvptxErr::UnsupportedOp(format!(
                    "`{callee}` expects (membermask, predicate), got {} operands",
                    args.len()
                ))));
            }
            let pred = self.lookup(args[1])?;
            if pred.class != RegClass::Pred {
                return Err(input_error_noloc!(NvptxErr::UnsupportedOp(format!(
                    "`{callee}` predicate operand must be i1"
                ))));
            }
            let mask = self.b32_operand(args[0])?;
            let (dst, ty) = if ballot {
                (self.fresh(RegClass::B32), "b32")
            } else {
                (self.fresh(RegClass::Pred), "pred")
            };
            self.inst(format!("vote.sync.{mode}.{ty} {dst}, {pred}, {mask};"));
            self.values.insert(result(), dst);
            return Ok(());
        }

        // The dynamic shared-memory window base (docs/KERNEL-ABI.md). The
        // raw `.shared` address stays raw while provenance proves every use
        // shared-qualified, exactly like `llvm.addressof` of a shared global.
        if canonical == DYN_SHARED_INTRINSIC {
            if !args.is_empty() {
                return Err(input_error_noloc!(NvptxErr::UnsupportedOp(format!(
                    "`{callee}` takes no operands, got {}",
                    args.len()
                ))));
            }
            self.used_dyn_shared = true;
            let result = result();
            let dst = self.fresh(RegClass::B64);
            self.inst(format!("mov.u64 {dst}, {DYN_SHARED_SYM};"));
            if self.space_of(result) != PtrSpace::Shared {
                self.inst(format!("cvta.shared.u64 {dst}, {dst};"));
            }
            self.values.insert(result, dst);
            return Ok(());
        }

        // Atomics: `atom{.scope}{.space}.add.<ty> d, [a], b`, the space
        // qualified when the address's provenance is proven.
        if let Some((scope, op, ty, class)) = atomic_intrinsic(&canonical) {
            if args.len() != 2 {
                return Err(input_error_noloc!(NvptxErr::UnsupportedOp(format!(
                    "`{callee}` expects (ptr, value), got {} operands",
                    args.len()
                ))));
            }
            let space = self.space_qualifier(args[0]);
            let addr = self.lookup(args[0])?;
            let value = self.lookup(args[1])?;
            let dst = self.fresh(class);
            self.inst(format!("atom{scope}{space}.{op}.{ty} {dst}, [{addr}], {value};"));
            self.values.insert(result(), dst);
            return Ok(());
        }

        // Math: unary/binary float instructions.
        if let Some((mnemonic, class, arity)) = math_intrinsic(&canonical) {
            if args.len() != arity {
                return Err(input_error_noloc!(NvptxErr::UnsupportedOp(format!(
                    "`{callee}` expects {arity} operand(s), got {}",
                    args.len()
                ))));
            }
            let mut regs = Vec::new();
            for arg in &args {
                regs.push(self.lookup(*arg)?.to_string());
            }
            let dst = self.fresh(class);
            self.inst(format!("{mnemonic} {dst}, {};", regs.join(", ")));
            self.values.insert(result(), dst);
            return Ok(());
        }

        // Device `.func` calls: `.param`-ABI call.uni sequence in its own
        // scope so the `.param` declarations stay call-local.
        if let Some(sig) = self.device_sigs.get(&callee) {
            if args.len() != sig.params.len() {
                return Err(input_error_noloc!(NvptxErr::UnsupportedOp(format!(
                    "call to `{callee}` passes {} argument(s), expected {}",
                    args.len(),
                    sig.params.len()
                ))));
            }
            let params: Vec<RegClass> = sig.params.clone();
            let ret_class = sig.ret;
            let call_id = self.next_call;
            self.next_call += 1;
            self.inst("{".to_string());
            let mut param_names = Vec::new();
            for (i, (arg, class)) in args.iter().zip(params).enumerate() {
                let reg = self.lookup(*arg)?;
                if reg.class != class {
                    return Err(input_error_noloc!(NvptxErr::UnsupportedOp(format!(
                        "call to `{callee}`: argument {i} register class mismatch"
                    ))));
                }
                let ty = param_decl_ty(class)?;
                let param = format!("c{call_id}_param_{i}");
                self.inst(format!(".param .{ty} {param};"));
                self.inst(format!("st.param.{ty} [{param}], {reg};"));
                param_names.push(param);
            }
            let param_list = param_names.join(", ");
            if let Some(ret_class) = ret_class {
                let ty = param_decl_ty(ret_class)?;
                self.inst(format!(".param .{ty} c{call_id}_ret;"));
                self.inst(format!(
                    "call.uni (c{call_id}_ret), {callee}, ({param_list});"
                ));
                let dst = self.fresh(ret_class);
                self.inst(format!("ld.param.{ty} {dst}, [c{call_id}_ret];"));
                self.inst("}".to_string());
                self.values.insert(result(), dst);
            } else {
                self.inst(format!("call.uni {callee}, ({param_list});"));
                self.inst("}".to_string());
            }
            return Ok(());
        }

        Err(input_error_noloc!(NvptxErr::UnsupportedOp(format!(
            "call to `{callee}` (only NVVM intrinsics are supported in kernels; \
             device function calls are not implemented)"
        ))))
    }

    fn emit_float_binary(&mut self, op_ptr: Ptr<Operation>, kind: BinaryFloatKind) -> CrabbitResult<()> {
        let ctx = self.ctx;
        // FP contraction: a fused mul emits nothing; its single fadd use
        // becomes the fma.rn.
        if kind == BinaryFloatKind::FMul && self.fused_muls.contains(&op_ptr) {
            return Ok(());
        }
        if kind == BinaryFloatKind::FAdd
            && let Some(&(a, b, c)) = self.fma_operands.get(&op_ptr) {
                let result = op_ptr.deref(ctx).get_result(0);
                let class = classify(ctx, result.get_type(ctx))?;
                if !class.is_float() {
                    return Err(input_error_noloc!(NvptxErr::UnsupportedOp(
                        "float arithmetic on a non-float type".to_string()
                    )));
                }
                let (a, b, c) = (self.lookup(a)?, self.lookup(b)?, self.lookup(c)?);
                let dst = self.fresh(class);
                self.inst(format!("fma.rn.{} {dst}, {a}, {b}, {c};", class.float_ty()));
                self.values.insert(result, dst);
                return Ok(());
            }
        let (lhs, rhs, result) = {
            let op_deref = op_ptr.deref(ctx);
            (
                op_deref.get_operand(0),
                op_deref.get_operand(1),
                op_deref.get_result(0),
            )
        };
        let a = self.lookup(lhs)?;
        let b = self.lookup(rhs)?;
        let class = classify(ctx, result.get_type(ctx))?;
        if !class.is_float() {
            return Err(input_error_noloc!(NvptxErr::UnsupportedOp(
                "float arithmetic on a non-float type".to_string()
            )));
        }
        let ty = class.float_ty();
        let dst = self.fresh(class);
        let mnemonic = match kind {
            BinaryFloatKind::FAdd => format!("add.rn.{ty}"),
            BinaryFloatKind::FSub => format!("sub.rn.{ty}"),
            BinaryFloatKind::FMul => format!("mul.rn.{ty}"),
            BinaryFloatKind::FDiv => format!("div.rn.{ty}"),
            BinaryFloatKind::FRem => {
                return Err(input_error_noloc!(NvptxErr::UnsupportedOp(
                    "frem has no PTX instruction (use fmod via intrinsics)".to_string()
                )));
            }
        };
        self.inst(format!("{mnemonic} {dst}, {a}, {b};"));
        self.values.insert(result, dst);
        Ok(())
    }

    fn emit_fcmp(&mut self, fcmp: &FCmpOp) -> CrabbitResult<()> {
        let ctx = self.ctx;
        let lhs = fcmp.get_operation().deref(ctx).get_operand(0);
        let rhs = fcmp.get_operation().deref(ctx).get_operand(1);
        let result = fcmp.get_result(ctx);
        let a = self.lookup(lhs)?;
        let b = self.lookup(rhs)?;
        let ty = a.class.float_ty();
        let pred = self.fresh(RegClass::Pred);
        let cmp = match fcmp.predicate(ctx) {
            FCmpPredicateAttr::False => {
                let zero = self.fresh(RegClass::B32);
                self.inst(format!("mov.u32 {zero}, 0;"));
                self.inst(format!("setp.ne.b32 {pred}, {zero}, 0;"));
                self.values.insert(result, pred);
                return Ok(());
            }
            FCmpPredicateAttr::True => {
                let one = self.fresh(RegClass::B32);
                self.inst(format!("mov.u32 {one}, 1;"));
                self.inst(format!("setp.ne.b32 {pred}, {one}, 0;"));
                self.values.insert(result, pred);
                return Ok(());
            }
            FCmpPredicateAttr::OEQ => "eq",
            FCmpPredicateAttr::OGT => "gt",
            FCmpPredicateAttr::OGE => "ge",
            FCmpPredicateAttr::OLT => "lt",
            FCmpPredicateAttr::OLE => "le",
            FCmpPredicateAttr::ONE => "ne",
            FCmpPredicateAttr::ORD => "num",
            FCmpPredicateAttr::UEQ => "equ",
            FCmpPredicateAttr::UGT => "gtu",
            FCmpPredicateAttr::UGE => "geu",
            FCmpPredicateAttr::ULT => "ltu",
            FCmpPredicateAttr::ULE => "leu",
            FCmpPredicateAttr::UNE => "neu",
            FCmpPredicateAttr::UNO => "nan",
        };
        self.inst(format!("setp.{cmp}.{ty} {pred}, {a}, {b};"));
        self.values.insert(result, pred);
        Ok(())
    }

    fn emit_select(&mut self, select: &SelectOp) -> CrabbitResult<()> {
        let ctx = self.ctx;
        let op = select.get_operation();
        let (cond, on_true, on_false) = {
            let op = op.deref(ctx);
            (op.get_operand(0), op.get_operand(1), op.get_operand(2))
        };
        let result = select.get_result(ctx);
        let cond = self.lookup(cond)?;
        if cond.class != RegClass::Pred {
            return Err(input_error_noloc!(NvptxErr::UnsupportedOp(
                "select with a non-i1 condition".to_string()
            )));
        }
        let class = classify(ctx, result.get_type(ctx))?;
        if class == RegClass::Pred {
            let a = self.lookup(on_true)?;
            let b = self.lookup(on_false)?;
            // d = (c & a) | (!c & b)
            let dst = self.fresh(RegClass::Pred);
            let not_c = self.fresh(RegClass::Pred);
            let t1 = self.fresh(RegClass::Pred);
            let t2 = self.fresh(RegClass::Pred);
            self.inst(format!("not.pred {not_c}, {cond};"));
            self.inst(format!("and.pred {t1}, {cond}, {a};"));
            self.inst(format!("and.pred {t2}, {not_c}, {b};"));
            self.inst(format!("or.pred {dst}, {t1}, {t2};"));
            self.values.insert(result, dst);
            return Ok(());
        }
        // Integer arms fold to immediates when constant; register arms
        // that hold raw-shared addresses selected into a non-shared
        // result convert first.
        let mut arms = Vec::with_capacity(2);
        for val in [on_true, on_false] {
            if let Some((imm, _)) = self.int_imm(val) {
                arms.push(if class == RegClass::B64 {
                    format!("{}", imm as u64)
                } else {
                    format!("{}", imm as u32)
                });
                continue;
            }
            let mut reg = self.lookup(val)?;
            if self.space_of(result) != PtrSpace::Shared
                && reg.class == RegClass::B64
                && self.space_of(val) == PtrSpace::Shared
            {
                let tmp = self.fresh(RegClass::B64);
                self.inst(format!("cvta.shared.u64 {tmp}, {reg};"));
                reg = tmp;
            }
            arms.push(reg.to_string());
        }
        let dst = self.fresh(class);
        self.inst(format!(
            "selp.{} {dst}, {}, {}, {cond};",
            class.mov_suffix(),
            arms[0],
            arms[1]
        ));
        self.values.insert(result, dst);
        Ok(())
    }

    fn emit_int_to_float(&mut self, src_val: Value, result: Value, signed: bool) -> CrabbitResult<()> {
        let ctx = self.ctx;
        let src = self.lookup(src_val)?;
        let src_width = width_of(ctx, src_val.get_type(ctx))?;
        let dst_class = classify(ctx, result.get_type(ctx))?;
        if !dst_class.is_float() || src.class.is_float() {
            return Err(input_error_noloc!(NvptxErr::UnsupportedOp(
                "int-to-float cast with wrong operand classes".to_string()
            )));
        }
        let src = match src.class {
            RegClass::Pred => {
                let bit = self.fresh(RegClass::B32);
                self.inst(format!("selp.b32 {bit}, 1, 0, {src};"));
                bit
            }
            RegClass::B32 if signed => self.signed32(src, src_width),
            _ => src,
        };
        let src_ty = match (src.class, signed) {
            (RegClass::B64, true) => "s64",
            (RegClass::B64, false) => "u64",
            (_, true) => "s32",
            (_, false) => "u32",
        };
        let dst = self.fresh(dst_class);
        self.inst(format!("cvt.rn.{}.{src_ty} {dst}, {src};", dst_class.float_ty()));
        self.values.insert(result, dst);
        Ok(())
    }

    fn emit_float_to_int(&mut self, src_val: Value, result: Value, signed: bool) -> CrabbitResult<()> {
        let ctx = self.ctx;
        let src = self.lookup(src_val)?;
        let dst_width = width_of(ctx, result.get_type(ctx))?;
        let dst_class = classify(ctx, result.get_type(ctx))?;
        if !src.class.is_float() || dst_class.is_float() || dst_class == RegClass::Pred {
            return Err(input_error_noloc!(NvptxErr::UnsupportedOp(
                "float-to-int cast with wrong operand classes".to_string()
            )));
        }
        let dst_ty = match (dst_class, signed) {
            (RegClass::B64, true) => "s64",
            (RegClass::B64, false) => "u64",
            (_, true) => "s32",
            (_, false) => "u32",
        };
        let dst = self.fresh(dst_class);
        self.inst(format!("cvt.rzi.{dst_ty}.{} {dst}, {src};", src.class.float_ty()));
        // Narrow results keep the zero-extended invariant (Rust `as` casts
        // saturate, then truncate to the narrow width via `cvt` semantics;
        // PTX cvt already saturates to the destination type).
        if dst_class == RegClass::B32 {
            self.mask32(dst, dst_width);
        }
        self.values.insert(result, dst);
        Ok(())
    }

    fn emit_cond_br(&mut self, cond_br: &CondBrOp) -> CrabbitResult<()> {
        let ctx = self.ctx;
        let condition = self.lookup(cond_br.get_operand_condition(ctx))?;
        let true_dest = cond_br.get_operation().deref(ctx).get_successor(0);
        let true_args = cond_br.successor_operands(ctx, 0);
        let false_dest = cond_br.get_operation().deref(ctx).get_successor(1);
        let false_args = cond_br.successor_operands(ctx, 1);

        let true_label = self.edge_target(true_dest, &true_args)?;
        let false_label = self.edge_target(false_dest, &false_args)?;
        self.inst(format!("@{condition} bra {true_label};"));
        self.inst(format!("bra {false_label};"));
        Ok(())
    }

    /// The label a conditional edge should branch to: the destination block
    /// itself, or a dedicated edge block carrying the block-argument copies
    /// so they do not execute on the other edge.
    fn edge_target(
        &mut self,
        dest: Ptr<BasicBlock>,
        args: &[Value],
    ) -> CrabbitResult<String> {
        let dest_label = self.block_labels[&dest].clone();
        if args.is_empty() {
            return Ok(dest_label);
        }
        let edge_label = format!("$L_edge{}", self.next_edge);
        self.next_edge += 1;

        // Emit the copies into the edge buffer by temporarily swapping it in
        // as the instruction sink.
        let saved = std::mem::take(&mut self.code);
        let saved_tags = std::mem::take(&mut self.tags);
        self.code.push_str(&format!("{edge_label}:\n"));
        self.tags.push(self.current.clone());
        self.emit_block_arg_copies(dest, args)?;
        self.inst(format!("bra {dest_label};"));
        let edge_text = std::mem::replace(&mut self.code, saved);
        let edge_tags = std::mem::replace(&mut self.tags, saved_tags);
        self.edges.push_str(&edge_text);
        self.edge_tags.extend(edge_tags);
        Ok(edge_label)
    }

    /// Copy branch operands into the destination block's argument registers.
    /// The copies are conceptually parallel (a backedge source may be another
    /// destination's current value), so a source that is also some copy's
    /// destination is staged through a fresh temporary; every other copy is
    /// a single direct `mov` (constants move as immediates).
    fn emit_block_arg_copies(
        &mut self,
        dest: Ptr<BasicBlock>,
        args: &[Value],
    ) -> CrabbitResult<()> {
        let dest_args: Vec<Value> = dest.deref(self.ctx).arguments().collect();
        if dest_args.len() != args.len() {
            return Err(input_error_noloc!(NvptxErr::UnsupportedOp(format!(
                "branch operand count {} does not match target block argument count {}",
                args.len(),
                dest_args.len()
            ))));
        }
        // The copy plan: immediates, and register copies with their
        // shared-escape flag (a raw-shared source flowing into a
        // non-shared block argument converts on the copy — the `cvta` IS
        // the copy).
        let mut imm_copies: Vec<(Reg, String)> = Vec::new();
        let mut reg_copies: Vec<(Reg, Reg, bool)> = Vec::new();
        for (arg, dest_arg) in args.iter().copied().zip(dest_args) {
            let dst = self.lookup(dest_arg)?;
            if let Some(constant) = self.consts.get(&arg).copied() {
                let text = match constant.class {
                    RegClass::B32 => format!("mov.u32 {dst}, {};", constant.imm as u32),
                    RegClass::B64 => format!("mov.u64 {dst}, {};", constant.imm as u64),
                    RegClass::F32 => format!("mov.f32 {dst}, 0f{:08X};", constant.imm as u32),
                    RegClass::F64 => format!("mov.f64 {dst}, 0d{:016X};", constant.imm as u64),
                    // Predicates have no immediate form; materialize.
                    RegClass::Pred => {
                        let src = self.lookup(arg)?;
                        format!("mov.pred {dst}, {src};")
                    }
                };
                imm_copies.push((dst, text));
                continue;
            }
            let src = self.lookup(arg)?;
            if src == dst {
                continue;
            }
            let cvta = src.class == RegClass::B64
                && self.space_of(arg) == PtrSpace::Shared
                && self.space_of(dest_arg) != PtrSpace::Shared;
            reg_copies.push((dst, src, cvta));
        }
        let sources: HashSet<Reg> = reg_copies.iter().map(|(_, src, _)| *src).collect();
        // Stage the conflicting reads first, before any destination is
        // written; direct copies write only registers no copy reads.
        let mut staged = Vec::new();
        let mut direct = Vec::new();
        for (dst, src, cvta) in reg_copies {
            if sources.contains(&dst) {
                let tmp = self.fresh(src.class);
                self.inst(format!("mov.{} {tmp}, {src};", src.class.mov_suffix()));
                staged.push((dst, tmp, cvta));
            } else {
                direct.push((dst, src, cvta));
            }
        }
        for (dst, src, cvta) in direct.into_iter().chain(staged) {
            if cvta {
                self.inst(format!("cvta.shared.u64 {dst}, {src};"));
            } else {
                self.inst(format!("mov.{} {dst}, {src};", dst.class.mov_suffix()));
            }
        }
        for (_, text) in imm_copies {
            self.inst(text);
        }
        Ok(())
    }
}

/// A PTX memory operand: `[reg]` or `[reg+imm]` (a negative offset prints
/// as `+-N`, the ISA's canonical spelling).
fn fmt_mem(base: Reg, offset: i64) -> String {
    if offset == 0 {
        format!("[{base}]")
    } else {
        format!("[{base}+{offset}]")
    }
}

/// Sign-extend the low `width` bits of `imm` to 64 bits.
fn sext_from(imm: u128, width: u32) -> i64 {
    if width == 0 || width >= 64 {
        imm as i64
    } else {
        let shift = 64 - width;
        (((imm as u64) << shift) as i64) >> shift
    }
}

fn width_mask(width: u32) -> u128 {
    if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BinaryIntKind {
    Add,
    Sub,
    Mul,
    SDiv,
    UDiv,
    SRem,
    URem,
    And,
    Or,
    Xor,
    Shl,
    LShr,
    AShr,
}

// The variants deliberately keep LLVM's opcode names (fadd, fsub, ...);
// stripping the shared F would leave names that collide conceptually with
// the integer BinaryKind above.
#[allow(clippy::enum_variant_names)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BinaryFloatKind {
    FAdd,
    FSub,
    FMul,
    FDiv,
    FRem,
}

fn binary_float_kind(any: &dyn Op) -> Option<BinaryFloatKind> {
    if any.downcast_ref::<FAddOp>().is_some() {
        Some(BinaryFloatKind::FAdd)
    } else if any.downcast_ref::<FSubOp>().is_some() {
        Some(BinaryFloatKind::FSub)
    } else if any.downcast_ref::<FMulOp>().is_some() {
        Some(BinaryFloatKind::FMul)
    } else if any.downcast_ref::<FDivOp>().is_some() {
        Some(BinaryFloatKind::FDiv)
    } else if any.downcast_ref::<FRemOp>().is_some() {
        Some(BinaryFloatKind::FRem)
    } else {
        None
    }
}

fn binary_int_kind(any: &dyn Op) -> Option<BinaryIntKind> {
    if any.downcast_ref::<AddOp>().is_some() {
        Some(BinaryIntKind::Add)
    } else if any.downcast_ref::<SubOp>().is_some() {
        Some(BinaryIntKind::Sub)
    } else if any.downcast_ref::<MulOp>().is_some() {
        Some(BinaryIntKind::Mul)
    } else if any.downcast_ref::<SDivOp>().is_some() {
        Some(BinaryIntKind::SDiv)
    } else if any.downcast_ref::<UDivOp>().is_some() {
        Some(BinaryIntKind::UDiv)
    } else if any.downcast_ref::<SRemOp>().is_some() {
        Some(BinaryIntKind::SRem)
    } else if any.downcast_ref::<URemOp>().is_some() {
        Some(BinaryIntKind::URem)
    } else if any.downcast_ref::<AndOp>().is_some() {
        Some(BinaryIntKind::And)
    } else if any.downcast_ref::<OrOp>().is_some() {
        Some(BinaryIntKind::Or)
    } else if any.downcast_ref::<XorOp>().is_some() {
        Some(BinaryIntKind::Xor)
    } else if any.downcast_ref::<ShlOp>().is_some() {
        Some(BinaryIntKind::Shl)
    } else if any.downcast_ref::<LShrOp>().is_some() {
        Some(BinaryIntKind::LShr)
    } else if any.downcast_ref::<AShrOp>().is_some() {
        Some(BinaryIntKind::AShr)
    } else {
        None
    }
}
