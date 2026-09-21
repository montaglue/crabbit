use crate::{
    context::{Context, Ptr},
    dialects::{
        aarch64::{
            ops::{
                self as aarch64_ops, ATTR_KEY_AARCH64_RD, ATTR_KEY_AARCH64_RM, ATTR_KEY_AARCH64_RN,
            },
            registers::{Register, RegisterClass},
        },
        builtin::op_interfaces::OneRegionInterface,
    },
    input_error_noloc,
    ir::operation::Operation,
    linked_list::ContainsLinkedList,
    conversion::pass::{AnalysisManager, Pass, PassResult, changed},
    result::CrabbitResult,
};

use super::{error::Aarch64Err, frontend::module_op};

pub struct Aarch64LegalizePass;

impl Pass for Aarch64LegalizePass {
    fn name(&self) -> &str {
        "aarch64-legalize"
    }

    fn run(&mut self, root: Ptr<Operation>, ctx: &mut Context, _analyses: &mut AnalysisManager) -> pliron::result::Result<PassResult> {
        let module = module_op(ctx, root)?;
        let body = module.get_region(ctx).deref(ctx).get_head().unwrap();
        for func in body.deref(ctx).iter(ctx) {
            for region in func.deref(ctx).regions() {
                for block in region.deref(ctx).iter(ctx) {
                    for op in block.deref(ctx).iter(ctx) {
                        if let Some(opcode) = aarch64_ops::opcode(ctx, op) {
                            if opcode == aarch64_ops::MovImmOp::OPCODE
                                && aarch64_ops::imm(ctx, op)
                                    .is_some_and(|imm| imm > u16::MAX as u64)
                            {
                                return Err(input_error_noloc!(Aarch64Err::UnsupportedType(
                                    "materializing immediates wider than 16 bits is not implemented yet"
                                        .into()
                                )));
                            }
                            verify_gpr_operands(ctx, op)?;
                        }
                    }
                }
            }
        }
        Ok(changed())
    }
}

/// The register-file group an instruction expects for one operand key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExpectedClass {
    Gpr,
    /// Either FP class: `d` and `s` spell element sizes of the same file,
    /// and the element size is carried by the opcode.
    Fpr,
    /// The full 128-bit vector view (`q`) of the FP/SIMD file.
    Simd,
}

fn matches_expected(class: RegisterClass, expected: ExpectedClass) -> bool {
    match expected {
        ExpectedClass::Gpr => class == RegisterClass::Gpr64,
        ExpectedClass::Fpr => matches!(class, RegisterClass::Fpr64 | RegisterClass::Fpr32),
        ExpectedClass::Simd => class == RegisterClass::Simd128,
    }
}

/// The expected register-file group of the `(rd, rn, rm)` operands for
/// `opcode`. Defaults to all-GPR; FP instructions override the operands
/// living in the FP file (conversions and cross-file moves mix the two).
fn expected_operand_classes(
    opcode: aarch64_ops::Aarch64Opcode,
) -> (ExpectedClass, ExpectedClass, ExpectedClass) {
    use ExpectedClass::{Fpr, Gpr, Simd};
    use aarch64_ops::Aarch64Opcode as Opc;
    match opcode {
        // NEON vector data-processing: everything in the q view.
        Opc::AddV4s | Opc::SubV4s | Opc::MulV4s | Opc::AddV2d | Opc::SubV2d
        | Opc::FaddV4s | Opc::FsubV4s | Opc::FmulV4s | Opc::FdivV4s
        | Opc::FaddV2d | Opc::FsubV2d | Opc::FmulV2d | Opc::FdivV2d
        | Opc::MovV16b | Opc::FaddpV4s
        | Opc::AndV16b | Opc::OrrV16b | Opc::EorV16b
        | Opc::ShlV4sImm | Opc::UshrV4sImm | Opc::SshrV4sImm
        | Opc::ShlV2dImm | Opc::UshrV2dImm | Opc::SshrV2dImm => (Simd, Simd, Simd),
        // Lane broadcasts: vector destination, scalar source.
        Opc::DupV4sGpr | Opc::DupV2dGpr => (Simd, Gpr, Gpr),
        Opc::DupV4sFpr | Opc::DupV2dFpr => (Simd, Fpr, Fpr),
        // Horizontal reductions: scalar FP destination, vector source.
        Opc::AddvS4s | Opc::AddpD2d | Opc::FaddpS2s | Opc::FaddpD2d => (Fpr, Simd, Simd),
        // q data register, GPR base for 128-bit loads/stores; no rm.
        Opc::StrqSpOffset | Opc::LdrqSpOffset | Opc::StrqRegOffset | Opc::LdrqRegOffset => {
            (Simd, Gpr, Gpr)
        }
        // FP data-processing: everything in the FP file.
        Opc::FaddD | Opc::FaddS | Opc::FsubD | Opc::FsubS | Opc::FmulD | Opc::FmulS
        | Opc::FdivD | Opc::FdivS | Opc::FnegD | Opc::FnegS | Opc::FcmpD | Opc::FcmpS
        | Opc::FabsD | Opc::FabsS | Opc::FsqrtD | Opc::FsqrtS | Opc::FrintmD | Opc::FrintmS
        | Opc::FrintpD | Opc::FrintpS | Opc::FrintzD | Opc::FrintzS | Opc::FrintaD
        | Opc::FrintaS | Opc::FrintnD | Opc::FrintnS | Opc::FminnmD | Opc::FminnmS
        | Opc::FmaxnmD | Opc::FmaxnmS | Opc::FminD | Opc::FminS | Opc::FmaxD | Opc::FmaxS
        | Opc::FcvtDS | Opc::FcvtSD | Opc::FmovD | Opc::FmovS | Opc::FmovImmD
        | Opc::FmovImmS => (Fpr, Fpr, Fpr),
        // FP data register, GPR base for loads/stores; no rm.
        Opc::StrdSpOffset | Opc::LdrdSpOffset | Opc::StrsSpOffset | Opc::LdrsSpOffset
        | Opc::StrdRegOffset | Opc::LdrdRegOffset | Opc::StrsRegOffset | Opc::LdrsRegOffset => {
            (Fpr, Gpr, Gpr)
        }
        // int -> FP conversions and GPR-to-FP moves: FP destination, GPR source.
        Opc::ScvtfDX | Opc::ScvtfSX | Opc::UcvtfDX | Opc::UcvtfSX | Opc::FmovDX
        | Opc::FmovSW => (Fpr, Gpr, Gpr),
        // FP -> int conversions and FP-to-GPR moves: GPR destination, FP source.
        Opc::FcvtzsXD | Opc::FcvtzsWD | Opc::FcvtzsXS | Opc::FcvtzsWS | Opc::FcvtzuXD
        | Opc::FcvtzuWD | Opc::FcvtzuXS | Opc::FcvtzuWS | Opc::FmovXD | Opc::FmovWS => {
            (Gpr, Fpr, Gpr)
        }
        _ => (Gpr, Gpr, Gpr),
    }
}

/// Reject a register operand in the wrong register file before RA/encoding
/// instead of letting it collide with a virtual register name or panic in
/// the encoder's register-number extraction.
fn verify_gpr_operands(ctx: &Context, op: Ptr<Operation>) -> CrabbitResult<()> {
    let mnemonic = aarch64_ops::mnemonic(ctx, op).unwrap_or("<unknown>");
    let Some(opcode) = aarch64_ops::opcode(ctx, op) else {
        return Ok(());
    };
    let (rd_expected, rn_expected, rm_expected) = expected_operand_classes(opcode);
    for (key, expected) in [
        (ATTR_KEY_AARCH64_RD.as_ref(), rd_expected),
        (ATTR_KEY_AARCH64_RN.as_ref(), rn_expected),
        (ATTR_KEY_AARCH64_RM.as_ref(), rm_expected),
    ] {
        let Some(register) = aarch64_ops::reg(ctx, op, key) else {
            continue;
        };
        let class = match register {
            Register::Virtual { class, .. } => class,
            Register::Physical(register) => register.class(),
        };
        if !matches_expected(class, expected) {
            return Err(input_error_noloc!(Aarch64Err::UnsupportedOp(
                format!("{mnemonic} requires a {expected:?} operand for {key}, got `{register}`")
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::ll::LinkageAttr;
    use crate::{
        context::Context,
        dialects::{
            aarch64::{self, ops as aarch64_ops},
            builtin::{self, op_interfaces::OneRegionInterface},
        },
        ir::op::Op,
        linked_list::ContainsLinkedList,
        conversion::pass::{AnalysisManager, Pass},
    };

    use super::{Aarch64LegalizePass, verify_gpr_operands};
    use crate::dialects::aarch64::registers::{PhysicalRegister, Register};

    fn context() -> Context {
        let mut ctx = Context::new();
        aarch64::register(&mut ctx);
        ctx
    }

    fn module_with_inst(
        ctx: &mut Context,
        inst: crate::context::Ptr<crate::ir::operation::Operation>,
    ) -> builtin::ops::ModuleOp {
        let module = builtin::ops::ModuleOp::new(ctx, "test".try_into().unwrap());
        let body = module.get_region(ctx).deref(ctx).get_head().unwrap();
        let func = aarch64::ops::FuncOp::new(ctx, "main".try_into().unwrap(), LinkageAttr::External);
        func.get_operation().insert_at_back(body, ctx);
        inst.insert_at_back(func.entry_block(ctx), ctx);
        module
    }

    #[test]
    fn rejects_simd_spelling_in_a_gpr_instruction() {
        let mut ctx = context();
        let inst = aarch64_ops::mov(&mut ctx, Register::gpr(0), Register::Physical(PhysicalRegister::Simd128(0)));
        assert!(verify_gpr_operands(&ctx, inst).is_err());
    }

    #[test]
    fn vector_ops_require_simd128_operands() {
        use crate::dialects::aarch64::op_interfaces::Aarch64Opcode;
        let mut ctx = context();
        // fadd_v4s with a d-class operand: wrong view of the vector file.
        let bad = aarch64_ops::binary(
            &mut ctx,
            Aarch64Opcode::FaddV4s,
            Register::simd128(0),
            Register::simd128(1),
            Register::fpr64(2),
        );
        assert!(verify_gpr_operands(&ctx, bad).is_err());
        let good = aarch64_ops::binary(
            &mut ctx,
            Aarch64Opcode::FaddV4s,
            Register::simd128(0),
            Register::simd128(1),
            Register::virtual_simd128(7),
        );
        verify_gpr_operands(&ctx, good).unwrap();
        // ldr_q: q data register off a GPR base.
        let load = aarch64_ops::ldr_reg_offset_sized(
            &mut ctx,
            Aarch64Opcode::LdrqRegOffset,
            Register::virtual_simd128(3),
            Register::gpr(1),
            0,
        );
        verify_gpr_operands(&ctx, load).unwrap();
        // addv: scalar FP destination from a vector source.
        let addv = aarch64_ops::unary(
            &mut ctx,
            Aarch64Opcode::AddvS4s,
            Register::fpr32(0),
            Register::virtual_simd128(3),
        );
        verify_gpr_operands(&ctx, addv).unwrap();
    }

    #[test]
    fn accepts_largest_legal_mov_imm() {
        let mut ctx = context();
        let inst = aarch64_ops::mov_imm(&mut ctx, Register::gpr(0), u16::MAX as u64);
        let module = module_with_inst(&mut ctx, inst);

        Aarch64LegalizePass
            .run(module.get_operation(), &mut ctx, &mut AnalysisManager::default())
            .unwrap();
    }

    #[test]
    fn rejects_mov_imm_wider_than_u16() {
        let mut ctx = context();
        let inst = aarch64_ops::mov_imm(&mut ctx, Register::gpr(0), u16::MAX as u64 + 1);
        let module = module_with_inst(&mut ctx, inst);

        assert!(
            Aarch64LegalizePass
                .run(module.get_operation(), &mut ctx, &mut AnalysisManager::default())
                .is_err()
        );
    }

    #[test]
    fn allows_wide_immediate_on_non_mov_imm_instruction() {
        let mut ctx = context();
        let inst = aarch64_ops::svc(&mut ctx, u16::MAX as u64 + 1);
        let module = module_with_inst(&mut ctx, inst);

        Aarch64LegalizePass
            .run(module.get_operation(), &mut ctx, &mut AnalysisManager::default())
            .unwrap();
    }
}
