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
    conversion::pass::{Pass, PassOptions},
    result::STAIRResult,
};

use super::{error::Aarch64DarwinErr, frontend::module_op};

pub struct Aarch64LegalizePass;

impl Pass for Aarch64LegalizePass {
    fn name(&self) -> &str {
        "aarch64-legalize"
    }

    fn run(
        &self,
        root: Ptr<Operation>,
        ctx: &mut Context,
        _options: PassOptions,
    ) -> STAIRResult<Ptr<Operation>> {
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
                                return Err(input_error_noloc!(Aarch64DarwinErr::UnsupportedType(
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
        Ok(root)
    }
}

/// The current instruction set only has GPR encodings. Reject a manually
/// constructed FP/SIMD register operand before RA/encoding instead of letting
/// it collide with a virtual register name or panic in `parse_xreg`.
fn verify_gpr_operands(ctx: &Context, op: Ptr<Operation>) -> STAIRResult<()> {
    let mnemonic = aarch64_ops::mnemonic(ctx, op).unwrap_or("<unknown>");
    for key in [
        ATTR_KEY_AARCH64_RD.as_str(),
        ATTR_KEY_AARCH64_RN.as_str(),
        ATTR_KEY_AARCH64_RM.as_str(),
    ] {
        let Some(register) = aarch64_ops::reg(ctx, op, key) else {
            continue;
        };
        let class = match register {
            Register::Virtual { class, .. } => class,
            Register::Physical(register) => register.class(),
        };
        if class != RegisterClass::Gpr64 {
            return Err(input_error_noloc!(Aarch64DarwinErr::UnsupportedOp(
                format!("{mnemonic} requires a GPR operand, got `{register}`")
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use llvm_compat::ll::LinkageAttr;
    use crate::{
        context::Context,
        dialects::{
            aarch64::{self, ops as aarch64_ops},
            builtin::{self, op_interfaces::OneRegionInterface},
        },
        ir::op::Op,
        linked_list::ContainsLinkedList,
        conversion::pass::{Pass, PassOptions},
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
    fn accepts_largest_legal_mov_imm() {
        let mut ctx = context();
        let inst = aarch64_ops::mov_imm(&mut ctx, Register::gpr(0), u16::MAX as u64);
        let module = module_with_inst(&mut ctx, inst);

        Aarch64LegalizePass
            .run(module.get_operation(), &mut ctx, PassOptions::default())
            .unwrap();
    }

    #[test]
    fn rejects_mov_imm_wider_than_u16() {
        let mut ctx = context();
        let inst = aarch64_ops::mov_imm(&mut ctx, Register::gpr(0), u16::MAX as u64 + 1);
        let module = module_with_inst(&mut ctx, inst);

        assert!(
            Aarch64LegalizePass
                .run(module.get_operation(), &mut ctx, PassOptions::default())
                .is_err()
        );
    }

    #[test]
    fn allows_wide_immediate_on_non_mov_imm_instruction() {
        let mut ctx = context();
        let inst = aarch64_ops::svc(&mut ctx, u16::MAX as u64 + 1);
        let module = module_with_inst(&mut ctx, inst);

        Aarch64LegalizePass
            .run(module.get_operation(), &mut ctx, PassOptions::default())
            .unwrap();
    }
}

use llvm_compat::ll::{LinkageAttr};
