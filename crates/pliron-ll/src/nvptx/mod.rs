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
        builtin::{attributes::IntegerAttr, ops::ModuleOp, types::IntegerType},
        llvm::{
            attributes::ICmpPredicateAttr,
            ops::{
                AShrOp, AddOp, AllocaOp, AndOp, BitcastOp, BrOp, CallOp, CondBrOp,
                FuncOp as LlvmFuncOp, GepIndex, GetElementPtrOp, ICmpOp, IntToPtrOp, LShrOp,
                LoadOp, MulOp, OrOp, PoisonOp, PtrToIntOp, ReturnOp, SDivOp, SExtOp, SRemOp,
                ShlOp, StoreOp, SubOp, TruncOp, UDivOp, URemOp, UndefOp, UnreachableOp, XorOp,
                ZExtOp,
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
    out.push_str(&format!(
        ".version {}.{}\n.target sm_{}\n.address_size 64\n",
        target.ptx_isa.0, target.ptx_isa.1, target.sm
    ));

    for op_ptr in body.deref(ctx).iter(ctx) {
        let op_obj = Operation::get_op_dyn(op_ptr, ctx);
        if let Some(func) = op_obj.downcast_ref::<LlvmFuncOp>() {
            if !func.is_declaration(ctx) {
                out.push('\n');
                out.push_str(&emit_kernel(ctx, func)?);
            }
        }
    }
    Ok(out)
}

// Register model ------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum RegClass {
    Pred,
    B32,
    B64,
}

impl RegClass {
    fn prefix(self) -> &'static str {
        match self {
            RegClass::Pred => "%p",
            RegClass::B32 => "%r",
            RegClass::B64 => "%rd",
        }
    }

    fn decl(self) -> &'static str {
        match self {
            RegClass::Pred => ".pred",
            RegClass::B32 => ".b32",
            RegClass::B64 => ".b64",
        }
    }

    /// The `mov` type suffix that copies a full register of this class.
    fn mov_suffix(self) -> &'static str {
        match self {
            RegClass::Pred => "pred",
            RegClass::B32 => "b32",
            RegClass::B64 => "b64",
        }
    }

    fn index(self) -> usize {
        match self {
            RegClass::Pred => 0,
            RegClass::B32 => 1,
            RegClass::B64 => 2,
        }
    }
}

const REG_CLASSES: [RegClass; 3] = [RegClass::Pred, RegClass::B32, RegClass::B64];

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
    Err(input_error_noloc!(NvptxErr::UnsupportedType(
        ty.disp(ctx).to_string()
    )))
}

/// The integer bit width of `ty`; pointers count as 64.
fn width_of(ctx: &Context, ty: TypeHandle) -> STAIRResult<u32> {
    let ty_ref = ty.deref(ctx);
    if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
        return Ok(int_ty.width() as u32);
    }
    if ty_ref.downcast_ref::<PointerType>().is_some() {
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

struct FuncEmitter<'c> {
    ctx: &'c Context,
    values: HashMap<Value, Reg>,
    reg_counts: [usize; 3],
    /// Straight-line body text: labels and instructions.
    code: String,
    /// Edge blocks for conditional branches with block arguments; appended
    /// after the main body (control flow in PTX is fully explicit, so block
    /// order does not matter).
    edges: String,
    next_edge: usize,
    block_labels: HashMap<Ptr<BasicBlock>, String>,
}

fn emit_kernel(ctx: &Context, func: &LlvmFuncOp) -> STAIRResult<String> {
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
        values: HashMap::new(),
        reg_counts: [0; 3],
        code: String::new(),
        edges: String::new(),
        next_edge: 0,
        block_labels: HashMap::new(),
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
    }
    for block in blocks.iter().copied() {
        if block != entry {
            let label = emitter.block_labels[&block].clone();
            emitter.code.push_str(&format!("{label}:\n"));
        }
        emitter.emit_block(block)?;
    }

    let mut out = String::new();
    out.push_str(&format!(".visible .entry {name}(\n"));
    out.push_str(&param_decls.join(",\n"));
    out.push_str("\n)\n{\n");
    for class in REG_CLASSES {
        let count = emitter.reg_counts[class.index()];
        if count > 0 {
            out.push_str(&format!(
                "\t.reg {} {}<{}>;\n",
                class.decl(),
                class.prefix(),
                count
            ));
        }
    }
    out.push('\n');
    out.push_str(&emitter.code);
    out.push_str(&emitter.edges);
    out.push_str("}\n");
    Ok(out)
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
        let op_obj = Operation::get_op_dyn(op_ptr, ctx);

        if let Some(constant) = op_obj.downcast_ref::<crate::dialects::builtin::ops::ConstantOp>() {
            let attr = constant.get_value(ctx);
            let attr = attr
                .downcast_ref::<IntegerAttr>()
                .ok_or_else(|| {
                    input_error_noloc!(NvptxErr::UnsupportedOp(
                        "non-integer constant".to_string()
                    ))
                })?;
            let result = constant.get_result(ctx);
            let width = width_of(ctx, result.get_type(ctx))?;
            let class = classify(ctx, result.get_type(ctx))?;
            let imm = attr.value().to_u128() & width_mask(width);
            let reg = self.materialize_const(class, imm);
            self.values.insert(result, reg);
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
                        RegClass::Pred => {
                            return Err(input_error_noloc!(NvptxErr::UnsupportedOp(
                                "i1 gep index".to_string()
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
        // Sub-word loads zero-extend into the wider register, which is
        // exactly the narrow-value invariant.
        self.inst(format!("ld.u{width} {dst}, [{addr}];"));
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
        self.inst(format!("st.u{width} [{addr}], {src};"));
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
        if canonical == "nvvm_barrier0" {
            self.inst("bar.sync 0;".to_string());
            return Ok(());
        }
        Err(input_error_noloc!(NvptxErr::UnsupportedOp(format!(
            "call to `{callee}` (only NVVM intrinsics are supported in kernels yet)"
        ))))
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
        self.code.push_str(&format!("{edge_label}:\n"));
        self.emit_block_arg_copies(dest, args)?;
        self.inst(format!("bra {dest_label};"));
        let edge_text = std::mem::replace(&mut self.code, saved);
        self.edges.push_str(&edge_text);
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
