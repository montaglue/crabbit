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


use std::collections::HashMap;

use thiserror::Error;

use pliron::builtin::op_interfaces::{
    AtMostOneRegionInterface as _, BranchOpInterface as _, CallOpCallable, CallOpInterface as _,
    OneOpdInterface as _, OneRegionInterface, OneResultInterface, SymbolOpInterface,
};
use pliron_llvm::op_interfaces::IsDeclaration;

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
                FRemOp, FSubOp, FuncOp as LlvmFuncOp, GepIndex, GetElementPtrOp, GlobalOp,
                ICmpOp, IntToPtrOp, LShrOp, LoadOp, MulOp, OrOp, PoisonOp, PtrToIntOp, ReturnOp,
                SDivOp, SExtOp, SIToFPOp, SRemOp, SelectOp, ShlOp, StoreOp, SubOp, TruncOp,
                UDivOp, UIToFPOp, URemOp, UndefOp, UnreachableOp, XorOp, ZExtOp,
            },
            types::{ArrayType, PointerType},
        },
    },
    input_error_noloc,
    ir::{basic_block::BasicBlock, op::Op, operation::Operation, r#type::Typed, value::Value},
    linked_list::ContainsLinkedList,
    printable::Printable,
    r#type::TypeHandle,
    result::STAIRResult,
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
) -> STAIRResult<String> {
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
) -> STAIRResult<(String, Option<String>)> {
    write_ptx_linemap_inner(ctx, root, target, linemap_enabled())
}

/// [write_ptx_and_linemap_from_ir] with the linemap forced on regardless
/// of environment — for callers (the analysis server) whose enablement
/// comes from per-run config rather than process env.
pub fn write_ptx_with_forced_linemap(
    ctx: &Context,
    root: Ptr<Operation>,
    target: &PtxTarget,
) -> STAIRResult<(String, String)> {
    let (ptx, map) = write_ptx_linemap_inner(ctx, root, target, true)?;
    Ok((ptx, map.expect("linemap forced on")))
}

fn write_ptx_linemap_inner(
    ctx: &Context,
    root: Ptr<Operation>,
    target: &PtxTarget,
    want_linemap: bool,
) -> STAIRResult<(String, Option<String>)> {
    let root_op = Operation::get_op_dyn(root, ctx);
    let module = root_op
        .downcast_ref::<ModuleOp>()
        .ok_or_else(|| input_error_noloc!(NvptxErr::NotModule))?;
    let body = module
        .get_region(ctx)
        .deref(ctx)
        .get_head()
        .expect("builtin.module has a body block");

    let mut out = String::new();
    let mut line = 3usize; // header lines below
    let mut entries: Vec<(String, usize, usize, Vec<LineTag>)> = Vec::new();
    out.push_str(&format!(
        ".version {}.{}\n.target sm_{}\n.address_size 64\n",
        target.ptx_isa.0, target.ptx_isa.1, target.sm
    ));

    // Module-level globals first: their state space decides how
    // `llvm.addressof` materializes the address inside kernels.
    let mut globals: HashMap<String, GlobalSpace> = HashMap::new();
    for op_ptr in body.deref(ctx).iter(ctx) {
        let op_obj = Operation::get_op_dyn(op_ptr, ctx);
        if let Some(global) = op_obj.downcast_ref::<GlobalOp>() {
            let (name, space, text) = emit_global(ctx, global)?;
            globals.insert(name, space);
            line += text.matches('\n').count();
            out.push_str(&text);
        }
    }

    for op_ptr in body.deref(ctx).iter(ctx) {
        let op_obj = Operation::get_op_dyn(op_ptr, ctx);
        if let Some(func) = op_obj.downcast_ref::<LlvmFuncOp>() {
            if func.is_declaration(ctx) {
                continue;
            }
            // Kernels are the externally visible definitions. Internal
            // definitions are helpers the importer pulled in for inlining;
            // a surviving call to one is reported by `emit_call`.
            if func.get_attr_llvm_function_linkage(ctx)
                .is_some_and(|linkage| {
                    matches!(
                        *linkage,
                        crate::dialects::llvm::attributes::LinkageAttr::InternalLinkage
                            | crate::dialects::llvm::attributes::LinkageAttr::PrivateLinkage
                    )
                })
            {
                continue;
            }
            out.push('\n');
            line += 1;
            let (text, tags) = emit_kernel(ctx, func, &globals)?;
            let start = line + 1;
            line += text.matches('\n').count();
            entries.push((
                func.get_symbol_name(ctx).to_string(),
                start,
                line,
                tags,
            ));
            out.push_str(&text);
        }
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

/// A module global as a PTX variable declaration.
fn emit_global(ctx: &Context, global: &GlobalOp) -> STAIRResult<(String, GlobalSpace, String)> {
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
    let align = data.align.max(1);
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
fn classify(ctx: &Context, ty: TypeHandle) -> STAIRResult<RegClass> {
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
fn width_of(ctx: &Context, ty: TypeHandle) -> STAIRResult<u32> {
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
fn size_of_ty(ctx: &Context, ty: TypeHandle) -> STAIRResult<u64> {
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
        let elem = size_of_ty(ctx, array_ty.elem_type())?;
        return Ok(elem * array_ty.size());
    }
    Err(input_error_noloc!(NvptxErr::UnsupportedType(format!(
        "no NVPTX size for {}",
        ty.disp(ctx)
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

struct FuncEmitter<'c> {
    ctx: &'c Context,
    globals: &'c HashMap<String, GlobalSpace>,
    values: HashMap<Value, Reg>,
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

fn emit_kernel(
    ctx: &Context,
    func: &LlvmFuncOp,
    globals: &HashMap<String, GlobalSpace>,
) -> STAIRResult<(String, Vec<LineTag>)> {
    let name = func.get_symbol_name(ctx).to_string();
    let region = func
        .get_region(ctx)
        .expect("llvm.func definition must have a body");
    let entry = region
        .deref(ctx)
        .get_head()
        .expect("llvm.func definition must have an entry block");
    let blocks = entry_reverse_post_order(ctx, entry);

    let mut emitter = FuncEmitter {
        ctx,
        globals,
        values: HashMap::new(),
        reg_counts: [0; 5],
        code: String::new(),
        edges: String::new(),
        next_edge: 0,
        block_labels: HashMap::new(),
        tags: Vec::new(),
        edge_tags: Vec::new(),
        current: LineTag::Root("ptx:entry"),
    };

    // Kernel parameters are the entry block's arguments.
    let params: Vec<Value> = entry.deref(ctx).arguments().collect();
    let mut param_decls = Vec::new();
    let mut param_loads = Vec::new();
    for (i, param) in params.iter().copied().enumerate() {
        let class = classify(ctx, param.get_type(ctx))?;
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
    decl(&mut out, &mut tags, &format!(".visible .entry {name}(\n"));
    decl(&mut out, &mut tags, &param_decls.join(",\n"));
    decl(&mut out, &mut tags, "\n)\n{\n");
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
    Ok((out, tags))
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

    fn lookup(&self, value: Value) -> STAIRResult<Reg> {
        self.values.get(&value).copied().ok_or_else(|| {
            input_error_noloc!(NvptxErr::UndefinedValue(format!("{value:?}")))
        })
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

    fn emit_block(&mut self, block: Ptr<BasicBlock>) -> STAIRResult<()> {
        let ops: Vec<Ptr<Operation>> = block.deref(self.ctx).iter(self.ctx).collect();
        for op_ptr in ops {
            self.emit_op(op_ptr)?;
        }
        Ok(())
    }

    fn emit_op(&mut self, op_ptr: Ptr<Operation>) -> STAIRResult<()> {
        let ctx = self.ctx;
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
            let reg = self.materialize_const(class, imm & width_mask(width));
            self.values.insert(result, reg);
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
            let dst = self.fresh(RegClass::B64);
            self.inst(format!("mov.u64 {dst}, {name};"));
            match space {
                GlobalSpace::Shared => self.inst(format!("cvta.shared.u64 {dst}, {dst};")),
                GlobalSpace::Global => self.inst(format!("cvta.global.u64 {dst}, {dst};")),
            }
            self.values.insert(addr.get_result(ctx), dst);
        } else if let Some(undef) = op_obj.downcast_ref::<UndefOp>() {
            let result = undef.get_result(ctx);
            let class = classify(ctx, result.get_type(ctx))?;
            let reg = self.materialize_const(class, 0);
            self.values.insert(result, reg);
        } else if let Some(poison) = op_obj.downcast_ref::<PoisonOp>() {
            let result = poison.get_result(ctx);
            let class = classify(ctx, result.get_type(ctx))?;
            let reg = self.materialize_const(class, 0);
            self.values.insert(result, reg);
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
            if ret.get_operation().deref(ctx).get_num_operands() > 0 {
                return Err(input_error_noloc!(NvptxErr::UnsupportedOp(
                    "kernel functions must return void".to_string()
                )));
            }
            self.inst("ret;".to_string());
        } else if op_obj.downcast_ref::<UnreachableOp>().is_some() {
            self.inst("trap;".to_string());
        } else if op_obj.downcast_ref::<AllocaOp>().is_some() {
            return Err(input_error_noloc!(NvptxErr::UnsupportedOp(
                "llvm.alloca survived to NVPTX emission; run mem2reg first \
                 (local-memory allocas are not supported yet)"
                    .to_string()
            )));
        } else {
            return Err(input_error_noloc!(NvptxErr::UnsupportedOp(
                Operation::get_opid(op_ptr, ctx).to_string()
            )));
        }
        Ok(())
    }

    fn emit_binary(&mut self, op_ptr: Ptr<Operation>, kind: BinaryIntKind) -> STAIRResult<()> {
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
        let mut a = self.lookup(lhs)?;
        let mut b = self.lookup(rhs)?;

        // Signed consumers of sub-32-bit values need real sign bits.
        if matches!(
            kind,
            BinaryIntKind::SDiv | BinaryIntKind::SRem | BinaryIntKind::AShr
        ) {
            a = self.signed32(a, width);
            if kind != BinaryIntKind::AShr {
                b = self.signed32(b, width);
            }
        }

        // 64-bit shift amounts are `.u32` in PTX.
        if matches!(
            kind,
            BinaryIntKind::Shl | BinaryIntKind::LShr | BinaryIntKind::AShr
        ) && b.class == RegClass::B64
        {
            let amount = self.fresh(RegClass::B32);
            self.inst(format!("cvt.u32.u64 {amount}, {b};"));
            b = amount;
        }

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

    fn emit_icmp(&mut self, icmp: &ICmpOp) -> STAIRResult<()> {
        let ctx = self.ctx;
        let lhs = icmp.get_operation().deref(ctx).get_operand(0);
        let rhs = icmp.get_operation().deref(ctx).get_operand(1);
        let result = icmp.get_result(ctx);
        let predicate = icmp.predicate(ctx);

        let width = width_of(ctx, lhs.get_type(ctx))?;
        if width == 1 {
            return Err(input_error_noloc!(NvptxErr::UnsupportedOp(
                "icmp on i1 is not supported in NVPTX emission".to_string()
            )));
        }
        let mut a = self.lookup(lhs)?;
        let mut b = self.lookup(rhs)?;
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
        let ty = match signed {
            None => format!("b{bits}"),
            Some(true) => {
                a = self.signed32(a, width);
                b = self.signed32(b, width);
                format!("s{bits}")
            }
            Some(false) => format!("u{bits}"),
        };
        let pred = self.fresh(RegClass::Pred);
        self.inst(format!("setp.{cmp}.{ty} {pred}, {a}, {b};"));
        self.values.insert(result, pred);
        Ok(())
    }

    fn emit_zext(&mut self, zext: &ZExtOp) -> STAIRResult<()> {
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

    fn emit_sext(&mut self, sext: &SExtOp) -> STAIRResult<()> {
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

    fn emit_trunc(&mut self, trunc: &TruncOp) -> STAIRResult<()> {
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
                let dst = self.fresh(RegClass::B32);
                self.inst(format!("mov.b32 {dst}, {src};"));
                self.mask32(dst, dst_width);
                self.values.insert(result, dst);
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
    /// alias the operand's register.
    fn emit_reg_alias(&mut self, src_val: Value, result: Value) -> STAIRResult<()> {
        let src = self.lookup(src_val)?;
        let dst_class = classify(self.ctx, result.get_type(self.ctx))?;
        if src.class != dst_class {
            return Err(input_error_noloc!(NvptxErr::UnsupportedOp(format!(
                "register-aliasing cast between {:?} and {:?}",
                src.class, dst_class
            ))));
        }
        self.values.insert(result, src);
        Ok(())
    }

    fn emit_gep(&mut self, gep: &GetElementPtrOp) -> STAIRResult<()> {
        let ctx = self.ctx;
        let base = self.lookup(gep.get_operand_src_ptr(ctx))?;
        let indices = gep.indices(ctx);
        let src_elem_type = gep.src_elem_type(ctx);

        let addr = self.fresh(RegClass::B64);
        self.inst(format!("mov.u64 {addr}, {base};"));
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
                } else {
                    return Err(input_error_noloc!(NvptxErr::UnsupportedType(format!(
                        "gep into {} (structs are not supported yet)",
                        current_ty.disp(ctx)
                    ))));
                }
            };
            let elem_size = size_of_ty(ctx, elem_ty)?;
            match index {
                GepIndex::Constant(value) => {
                    constant_offset =
                        constant_offset.wrapping_add((*value as i64 as u64).wrapping_mul(elem_size));
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
                    self.inst(format!("mad.lo.s64 {addr}, {wide}, {elem_size}, {addr};"));
                }
            }
            current_ty = elem_ty;
        }
        if constant_offset != 0 {
            self.inst(format!(
                "add.s64 {addr}, {addr}, {};",
                constant_offset as i64
            ));
        }
        self.values.insert(gep.get_result(ctx), addr);
        Ok(())
    }

    fn emit_load(&mut self, load: &LoadOp) -> STAIRResult<()> {
        let ctx = self.ctx;
        let result = load.get_result(ctx);
        let addr = self.lookup(load.get_operand_address(ctx))?;
        let width = width_of(ctx, result.get_type(ctx))?;
        if width == 1 {
            return Err(input_error_noloc!(NvptxErr::UnsupportedOp(
                "i1 load is not supported in NVPTX emission".to_string()
            )));
        }
        let class = classify(ctx, result.get_type(ctx))?;
        let dst = self.fresh(class);
        if class.is_float() {
            self.inst(format!("ld.{} {dst}, [{addr}];", class.float_ty()));
        } else {
            // Sub-word loads zero-extend into the wider register, which is
            // exactly the narrow-value invariant.
            self.inst(format!("ld.u{width} {dst}, [{addr}];"));
        }
        self.values.insert(result, dst);
        Ok(())
    }

    fn emit_store(&mut self, store: &StoreOp) -> STAIRResult<()> {
        let ctx = self.ctx;
        let value = store.get_operand_value(ctx);
        let addr = self.lookup(store.get_operand_address(ctx))?;
        let width = width_of(ctx, value.get_type(ctx))?;
        if width == 1 {
            return Err(input_error_noloc!(NvptxErr::UnsupportedOp(
                "i1 store is not supported in NVPTX emission".to_string()
            )));
        }
        let src = self.lookup(value)?;
        if src.class.is_float() {
            self.inst(format!("st.{} [{addr}], {src};", src.class.float_ty()));
        } else {
            self.inst(format!("st.u{width} [{addr}], {src};"));
        }
        Ok(())
    }

    fn emit_call(&mut self, call: &CallOp) -> STAIRResult<()> {
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

        // Atomics: `atom{.scope}.add.<ty> d, [a], b`.
        let atomic = match canonical.as_str() {
            "nvvm_atomic_add_gen_i" => Some(("", "add", "u32", RegClass::B32)),
            "nvvm_atomic_add_gen_i_cta" => Some((".cta", "add", "u32", RegClass::B32)),
            "nvvm_atomic_add_gen_i_sys" => Some((".sys", "add", "u32", RegClass::B32)),
            "nvvm_atomic_add_gen_f" => Some(("", "add", "f32", RegClass::F32)),
            "nvvm_atomic_add_gen_f_cta" => Some((".cta", "add", "f32", RegClass::F32)),
            "nvvm_atomic_add_gen_ll" => Some(("", "add", "u64", RegClass::B64)),
            "nvvm_atomic_add_gen_d" => Some(("", "add", "f64", RegClass::F64)),
            "nvvm_atomic_max_gen_i" => Some(("", "max", "s32", RegClass::B32)),
            "nvvm_atomic_min_gen_i" => Some(("", "min", "s32", RegClass::B32)),
            "nvvm_atomic_max_gen_ui" => Some(("", "max", "u32", RegClass::B32)),
            "nvvm_atomic_min_gen_ui" => Some(("", "min", "u32", RegClass::B32)),
            "nvvm_atomic_and_gen_i" => Some(("", "and", "b32", RegClass::B32)),
            "nvvm_atomic_or_gen_i" => Some(("", "or", "b32", RegClass::B32)),
            "nvvm_atomic_xor_gen_i" => Some(("", "xor", "b32", RegClass::B32)),
            "nvvm_atomic_exch_gen_i" => Some(("", "exch", "b32", RegClass::B32)),
            _ => None,
        };
        if let Some((scope, op, ty, class)) = atomic {
            if args.len() != 2 {
                return Err(input_error_noloc!(NvptxErr::UnsupportedOp(format!(
                    "`{callee}` expects (ptr, value), got {} operands",
                    args.len()
                ))));
            }
            let addr = self.lookup(args[0])?;
            let value = self.lookup(args[1])?;
            let dst = self.fresh(class);
            self.inst(format!("atom{scope}.{op}.{ty} {dst}, [{addr}], {value};"));
            self.values.insert(result(), dst);
            return Ok(());
        }

        // Math: unary/binary float instructions.
        let math: Option<(&str, RegClass, usize)> = match canonical.as_str() {
            "nvvm_sqrt_rn_f" | "nvvm_sqrt_f" | "sqrt_f32" => Some(("sqrt.rn.f32", RegClass::F32, 1)),
            "nvvm_sqrt_approx_f" => Some(("sqrt.approx.f32", RegClass::F32, 1)),
            "nvvm_rsqrt_approx_f" => Some(("rsqrt.approx.f32", RegClass::F32, 1)),
            "nvvm_ex2_approx_f" | "exp2_f32" => Some(("ex2.approx.f32", RegClass::F32, 1)),
            "nvvm_lg2_approx_f" | "log2_f32" => Some(("lg2.approx.f32", RegClass::F32, 1)),
            "nvvm_sin_approx_f" => Some(("sin.approx.f32", RegClass::F32, 1)),
            "nvvm_cos_approx_f" => Some(("cos.approx.f32", RegClass::F32, 1)),
            "nvvm_rcp_approx_f" => Some(("rcp.approx.f32", RegClass::F32, 1)),
            "nvvm_fabs_f" | "fabs_f32" => Some(("abs.f32", RegClass::F32, 1)),
            "nvvm_floor_f" | "floor_f32" => Some(("cvt.rmi.f32.f32", RegClass::F32, 1)),
            "nvvm_ceil_f" | "ceil_f32" => Some(("cvt.rpi.f32.f32", RegClass::F32, 1)),
            "nvvm_round_f" | "nvvm_rint_f" => Some(("cvt.rni.f32.f32", RegClass::F32, 1)),
            "nvvm_trunc_f" | "trunc_f32" => Some(("cvt.rzi.f32.f32", RegClass::F32, 1)),
            "nvvm_fmax_f" | "maxnum_f32" => Some(("max.f32", RegClass::F32, 2)),
            "nvvm_fmin_f" | "minnum_f32" => Some(("min.f32", RegClass::F32, 2)),
            "nvvm_fma_rn_f" | "fma_f32" => Some(("fma.rn.f32", RegClass::F32, 3)),
            "nvvm_sqrt_rn_d" | "sqrt_f64" => Some(("sqrt.rn.f64", RegClass::F64, 1)),
            "nvvm_rsqrt_approx_d" => Some(("rsqrt.approx.f64", RegClass::F64, 1)),
            "nvvm_fabs_d" | "fabs_f64" => Some(("abs.f64", RegClass::F64, 1)),
            "nvvm_floor_d" | "floor_f64" => Some(("cvt.rmi.f64.f64", RegClass::F64, 1)),
            "nvvm_ceil_d" | "ceil_f64" => Some(("cvt.rpi.f64.f64", RegClass::F64, 1)),
            "nvvm_fmax_d" | "maxnum_f64" => Some(("max.f64", RegClass::F64, 2)),
            "nvvm_fmin_d" | "minnum_f64" => Some(("min.f64", RegClass::F64, 2)),
            "nvvm_fma_rn_d" | "fma_f64" => Some(("fma.rn.f64", RegClass::F64, 3)),
            _ => None,
        };
        if let Some((mnemonic, class, arity)) = math {
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

        Err(input_error_noloc!(NvptxErr::UnsupportedOp(format!(
            "call to `{callee}` (only NVVM intrinsics are supported in kernels; \
             device function calls are not implemented)"
        ))))
    }

    fn emit_float_binary(&mut self, op_ptr: Ptr<Operation>, kind: BinaryFloatKind) -> STAIRResult<()> {
        let ctx = self.ctx;
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

    fn emit_fcmp(&mut self, fcmp: &FCmpOp) -> STAIRResult<()> {
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

    fn emit_select(&mut self, select: &SelectOp) -> STAIRResult<()> {
        let ctx = self.ctx;
        let op = select.get_operation();
        let (cond, on_true, on_false) = {
            let op = op.deref(ctx);
            (op.get_operand(0), op.get_operand(1), op.get_operand(2))
        };
        let result = select.get_result(ctx);
        let cond = self.lookup(cond)?;
        let a = self.lookup(on_true)?;
        let b = self.lookup(on_false)?;
        if cond.class != RegClass::Pred {
            return Err(input_error_noloc!(NvptxErr::UnsupportedOp(
                "select with a non-i1 condition".to_string()
            )));
        }
        let class = classify(ctx, result.get_type(ctx))?;
        let dst = self.fresh(class);
        if class == RegClass::Pred {
            // d = (c & a) | (!c & b)
            let not_c = self.fresh(RegClass::Pred);
            let t1 = self.fresh(RegClass::Pred);
            let t2 = self.fresh(RegClass::Pred);
            self.inst(format!("not.pred {not_c}, {cond};"));
            self.inst(format!("and.pred {t1}, {cond}, {a};"));
            self.inst(format!("and.pred {t2}, {not_c}, {b};"));
            self.inst(format!("or.pred {dst}, {t1}, {t2};"));
        } else {
            self.inst(format!("selp.{} {dst}, {a}, {b}, {cond};", class.mov_suffix()));
        }
        self.values.insert(result, dst);
        Ok(())
    }

    fn emit_int_to_float(&mut self, src_val: Value, result: Value, signed: bool) -> STAIRResult<()> {
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

    fn emit_float_to_int(&mut self, src_val: Value, result: Value, signed: bool) -> STAIRResult<()> {
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

    fn emit_cond_br(&mut self, cond_br: &CondBrOp) -> STAIRResult<()> {
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
    ) -> STAIRResult<String> {
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
    /// destination's current value), so every source is staged through a
    /// fresh temporary first; ptxas coalesces the extra movs.
    fn emit_block_arg_copies(
        &mut self,
        dest: Ptr<BasicBlock>,
        args: &[Value],
    ) -> STAIRResult<()> {
        let dest_args: Vec<Value> = dest.deref(self.ctx).arguments().collect();
        if dest_args.len() != args.len() {
            return Err(input_error_noloc!(NvptxErr::UnsupportedOp(format!(
                "branch operand count {} does not match target block argument count {}",
                args.len(),
                dest_args.len()
            ))));
        }
        let mut staged = Vec::new();
        for (arg, dest_arg) in args.iter().copied().zip(dest_args) {
            let src = self.lookup(arg)?;
            let dst = self.lookup(dest_arg)?;
            if src == dst {
                continue;
            }
            let tmp = self.fresh(src.class);
            let suffix = src.class.mov_suffix();
            self.inst(format!("mov.{suffix} {tmp}, {src};"));
            staged.push((dst, tmp));
        }
        for (dst, tmp) in staged {
            let suffix = dst.class.mov_suffix();
            self.inst(format!("mov.{suffix} {dst}, {tmp};"));
        }
        Ok(())
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
