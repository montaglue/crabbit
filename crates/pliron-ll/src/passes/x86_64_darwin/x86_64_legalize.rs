use crate::{
    context::{Context, Ptr},
    dialects::{
        x86_64::{
            ops::{
                self as x86_64_ops, ATTR_KEY_X86_64_RD, ATTR_KEY_X86_64_RM, ATTR_KEY_X86_64_RN,
            },
            registers::{Register, RegisterClass},
        },
        builtin::op_interfaces::OneRegionInterface,
    },
    input_error_noloc,
    ir::operation::Operation,
    linked_list::ContainsLinkedList,
    conversion::pass::{AnalysisManager, Pass, PassResult, changed},
    result::STAIRResult,
};

use super::{error::X86_64DarwinErr, frontend::module_op};

pub struct X86_64LegalizePass;

impl Pass for X86_64LegalizePass {
    fn name(&self) -> &str {
        "x86-64-legalize"
    }

    fn run(&mut self, root: Ptr<Operation>, ctx: &mut Context, _analyses: &mut AnalysisManager) -> pliron::result::Result<PassResult> {
        let module = module_op(ctx, root)?;
        let body = module.get_region(ctx).deref(ctx).get_head().unwrap();
        for func in body.deref(ctx).iter(ctx) {
            for region in func.deref(ctx).regions() {
                for block in region.deref(ctx).iter(ctx) {
                    for op in block.deref(ctx).iter(ctx) {
                        if x86_64_ops::opcode(ctx, op).is_some() {
                            verify_gpr_operands(ctx, op)?;
                        }
                    }
                }
            }
        }
        Ok(changed())
    }
}

/// The current instruction set only has GPR encodings. Reject a manually
/// constructed FP/SIMD register operand before RA/encoding instead of letting
/// it collide with a virtual register name or panic in `parse_xreg`.
fn verify_gpr_operands(ctx: &Context, op: Ptr<Operation>) -> STAIRResult<()> {
    let mnemonic = x86_64_ops::mnemonic(ctx, op).unwrap_or("<unknown>");
    for key in [
        ATTR_KEY_X86_64_RD.as_str(),
        ATTR_KEY_X86_64_RN.as_str(),
        ATTR_KEY_X86_64_RM.as_str(),
    ] {
        let Some(register) = x86_64_ops::reg(ctx, op, key) else {
            continue;
        };
        let class = match register {
            Register::Virtual { class, .. } => class,
            Register::Physical(register) => register.class(),
        };
        if class != RegisterClass::Gpr64 {
            return Err(input_error_noloc!(X86_64DarwinErr::UnsupportedOp(
                format!("{mnemonic} requires a GPR operand, got `{register}`")
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
            x86_64::{self, ops as x86_64_ops},
            builtin::{self, op_interfaces::OneRegionInterface},
        },
        ir::op::Op,
        linked_list::ContainsLinkedList,
        conversion::pass::{AnalysisManager, Pass},
    };

    use super::{X86_64LegalizePass, verify_gpr_operands};
    use crate::dialects::x86_64::registers::{PhysicalRegister, RAX, Register};

    fn context() -> Context {
        let mut ctx = Context::new();
        x86_64::register(&mut ctx);
        ctx
    }

    fn module_with_inst(
        ctx: &mut Context,
        inst: crate::context::Ptr<crate::ir::operation::Operation>,
    ) -> builtin::ops::ModuleOp {
        let module = builtin::ops::ModuleOp::new(ctx, "test".try_into().unwrap());
        let body = module.get_region(ctx).deref(ctx).get_head().unwrap();
        let func = x86_64::ops::FuncOp::new(ctx, "main".try_into().unwrap(), LinkageAttr::External);
        func.get_operation().insert_at_back(body, ctx);
        inst.insert_at_back(func.entry_block(ctx), ctx);
        module
    }

    #[test]
    fn rejects_xmm_spelling_in_a_gpr_instruction() {
        let mut ctx = context();
        let inst = x86_64_ops::mov(&mut ctx, RAX, Register::Physical(PhysicalRegister::Xmm(0)));
        assert!(verify_gpr_operands(&ctx, inst).is_err());
    }

    #[test]
    fn accepts_full_width_mov_imm() {
        // movabs carries a 64-bit immediate; no width legalization needed.
        let mut ctx = context();
        let inst = x86_64_ops::mov_imm(&mut ctx, RAX, u64::MAX);
        let module = module_with_inst(&mut ctx, inst);

        X86_64LegalizePass
            .run(module.get_operation(), &mut ctx, &mut AnalysisManager::default())
            .unwrap();
    }
}
