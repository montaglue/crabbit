//! Function linkage across the `mir` → LLVM dialect lowering.
//!
//! cuda-oxide's `mir-lower` builds every `llvm.func` fresh and copies only
//! the attributes it knows (debug names, kernel facts, inline intent);
//! pliron's dialect conversion copies none when it replaces an op. The
//! `builtin.module` op, by contrast, is never converted and keeps its
//! attributes, and lowered functions keep their symbol names. So the
//! importer records linkage on the module — a [DictAttr] keyed by symbol,
//! under [LINKAGE_ATTR_KEY], holding one [LinkageAttr] per function — and
//! [ApplyModuleLinkagePass], run right after `lower-dialect-mir`, stamps
//! it onto the `llvm.func` ops and removes the dictionary.
//!
//! Every lowered function ends up with an explicit linkage: the recorded
//! one, or external when nothing was recorded (this includes the callee
//! declarations `mir-lower` synthesizes for symbols defined elsewhere). The
//! mid-end relies on that (`inline` reads the attribute unconditionally).
//! Functions the importer emits directly as `llvm.func` already carry
//! their linkage and are left as they are.

use pliron::{
    builtin::{attributes::DictAttr, op_interfaces::SymbolOpInterface as _, ops::ModuleOp},
    context::{Context, Ptr},
    identifier::Identifier,
    linked_list::ContainsLinkedList as _,
    operation::Operation,
    result::Result,
};
use pliron_ll::conversion::pass::{AnalysisManager, Pass, PassResult, changed, unchanged};
use pliron_llvm::{attributes::LinkageAttr, ops::FuncOp};

/// The module attribute holding the symbol → linkage dictionary.
pub const LINKAGE_ATTR_KEY: &str = "crabbit_linkage";
/// The pass name.
pub const PASS_NAME: &str = "apply-module-linkage";

/// [LINKAGE_ATTR_KEY] as an [Identifier].
pub fn linkage_attr_key() -> Identifier {
    LINKAGE_ATTR_KEY.try_into().expect("static identifier literal")
}

/// Record that the function `symbol` of `module` lowers with `linkage`.
/// `module` is the `builtin.module` op the function lives in.
pub fn record_linkage(
    ctx: &mut Context,
    module: Ptr<Operation>,
    symbol: &Identifier,
    linkage: LinkageAttr,
) {
    debug_assert!(
        Operation::get_op::<ModuleOp>(module, ctx).is_some(),
        "linkage is recorded on a builtin.module"
    );
    let key = linkage_attr_key();
    let mut module_ref = module.deref_mut(ctx);
    let mut dict = module_ref
        .attributes
        .get::<DictAttr>(&key)
        .cloned()
        .unwrap_or_else(|| DictAttr(Default::default()));
    dict.0.set(symbol.clone(), linkage);
    module_ref.attributes.set(key, dict);
}

/// The linkage recorded for `symbol` on `module`, if any.
pub fn recorded_linkage(ctx: &Context, module: Ptr<Operation>, symbol: &Identifier) -> Option<LinkageAttr> {
    module
        .deref(ctx)
        .attributes
        .get::<DictAttr>(&linkage_attr_key())
        .and_then(|dict| dict.0.get::<LinkageAttr>(symbol).cloned())
}

/// Stamp the recorded linkage onto the lowered `llvm.func` ops of the
/// module (external where nothing was recorded) and drop the dictionary.
pub struct ApplyModuleLinkagePass;

impl Pass for ApplyModuleLinkagePass {
    fn name(&self) -> &str {
        PASS_NAME
    }

    fn run(
        &mut self,
        root: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let key = linkage_attr_key();
        let recorded = root
            .deref_mut(ctx)
            .attributes
            .0
            .remove(&key)
            .and_then(|attr| attr.downcast::<DictAttr>().ok())
            .map(|dict| *dict);

        let mut functions = Vec::new();
        for block in root.deref(ctx).get_region(0).deref(ctx).iter(ctx) {
            for op in block.deref(ctx).iter(ctx) {
                let Some(func) = Operation::get_op::<FuncOp>(op, ctx) else {
                    continue;
                };
                if func.get_attr_llvm_function_linkage(ctx).is_some() {
                    continue;
                }
                functions.push((func, func.get_symbol_name(ctx)));
            }
        }
        if functions.is_empty() {
            return Ok(if recorded.is_some() { changed() } else { unchanged() });
        }
        for (func, symbol) in functions {
            let linkage = recorded
                .as_ref()
                .and_then(|dict| dict.0.get::<LinkageAttr>(&symbol).cloned())
                .unwrap_or(LinkageAttr::ExternalLinkage);
            func.set_attr_llvm_function_linkage(ctx, linkage);
        }
        Ok(changed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pliron::{builtin::op_interfaces::OneRegionInterface as _, op::Op as _};

    #[test]
    fn records_and_reads_back() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "m".try_into().unwrap()).get_operation();
        let sym: Identifier = "f".try_into().unwrap();
        assert_eq!(recorded_linkage(&ctx, module, &sym), None);
        record_linkage(&mut ctx, module, &sym, LinkageAttr::InternalLinkage);
        assert_eq!(recorded_linkage(&ctx, module, &sym), Some(LinkageAttr::InternalLinkage));
        record_linkage(&mut ctx, module, &sym, LinkageAttr::WeakODRLinkage);
        assert_eq!(recorded_linkage(&ctx, module, &sym), Some(LinkageAttr::WeakODRLinkage));
    }

    #[test]
    fn stamps_recorded_linkage_and_defaults_to_external() {
        let mut ctx = Context::new();
        let module_op = ModuleOp::new(&mut ctx, "m".try_into().unwrap());
        let module = module_op.get_operation();
        let body = module_op.get_region(&ctx).deref(&ctx).get_head().unwrap();
        let void = pliron_llvm::types::VoidType::get(&mut ctx);
        let llvm_fn_ty = pliron_llvm::types::FuncType::get(&mut ctx, void.into(), vec![], false);
        let internal = FuncOp::new(&mut ctx, "internal".try_into().unwrap(), llvm_fn_ty);
        internal.get_operation().insert_at_back(body, &ctx);
        let external = FuncOp::new(&mut ctx, "external".try_into().unwrap(), llvm_fn_ty);
        external.get_operation().insert_at_back(body, &ctx);
        let kept = FuncOp::new(&mut ctx, "kept".try_into().unwrap(), llvm_fn_ty);
        kept.set_attr_llvm_function_linkage(&mut ctx, LinkageAttr::PrivateLinkage);
        kept.get_operation().insert_at_back(body, &ctx);
        record_linkage(&mut ctx, module, &"internal".try_into().unwrap(), LinkageAttr::InternalLinkage);

        ApplyModuleLinkagePass
            .run(module, &mut ctx, &mut AnalysisManager::default())
            .unwrap();

        assert_eq!(internal.get_attr_llvm_function_linkage(&ctx).map(|l| l.clone()), Some(LinkageAttr::InternalLinkage));
        assert_eq!(external.get_attr_llvm_function_linkage(&ctx).map(|l| l.clone()), Some(LinkageAttr::ExternalLinkage));
        assert_eq!(kept.get_attr_llvm_function_linkage(&ctx).map(|l| l.clone()), Some(LinkageAttr::PrivateLinkage));
        assert!(module.deref(&ctx).attributes.get::<DictAttr>(&linkage_attr_key()).is_none());
    }
}
