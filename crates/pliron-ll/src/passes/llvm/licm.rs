//! Loop-invariant code motion (docs/MIDEND-PLAN.md item 3): hoist pure,
//! non-trapping operations whose operands are all defined outside a
//! natural loop into the loop's preheader.
//!
//! v1 restrictions, all deliberate:
//! - Only loops that already HAVE a preheader (unique outside predecessor
//!   branching solely to the header) are touched; preheader creation is
//!   CFG surgery left for later. After llvm-simplify-cfg, loops from
//!   structured Rust code have one.
//! - Only non-trapping pure ops are hoisted (no udiv/sdiv/urem/srem: a
//!   hoisted trap would be speculation, since the preheader runs even
//!   when the header exits before the body). Loads are never hoisted (no
//!   dereferenceability model). Operand-less ops (constants, undef,
//!   addressof, cstr) are left alone: they carry no computation and
//!   hoisting them only extends live ranges — the RA experiments'
//!   independent variable.
//! - Loops are processed innermost first, so invariants chain outward
//!   one level per fixpoint iteration.
//!
//! ADJOINT (backward attribution): hoist = pure move: the op keeps its identity attributes; no adjoint needed.

use rustc_hash::FxHashSet;

use crate::{
    context::{Context, Ptr},
    dialects::llvm::{
        op_interfaces::IsDeclaration,
        ops::{
            AddOp, AndOp, BitcastOp, ExtractValueOp, GetElementPtrOp, ICmpOp, IntToPtrOp,
            LShrOp, MulOp, OrOp, PtrToIntOp, SExtOp, ShlOp, SubOp, TruncOp, XorOp, ZExtOp,
        },
    },
    ir::{
        basic_block::BasicBlock,
        op::{Op, OpId},
        operation::Operation,
        region::Region,
        value::{DefiningEntity, Value},
    },
    linked_list::{ContainsLinkedList, LinkedList as _},
    conversion::pass::{AnalysisManager, Pass, PassResult, changed, unchanged},
};

use super::{
    analysis::{dominator_tree, natural_loops},
    inline::collect_functions,
    midend_gate::midend_disabled,
};
use pliron::builtin::op_interfaces::AtMostOneRegionInterface as _;

const MAX_ITERATIONS: usize = 8;

pub struct LLVMLicmPass;

impl Pass for LLVMLicmPass {
    fn name(&self) -> &str {
        "llvm-licm"
    }

    fn run(
        &mut self,
        root: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> pliron::result::Result<PassResult> {
        if midend_disabled("licm") {
            return Ok(unchanged());
        }
        let hoistable = hoistable_op_ids();
        let mut any = false;
        for func in collect_functions(ctx, root) {
            if func.is_declaration(ctx) {
                continue;
            }
            let region = func
                .get_region(ctx)
                .expect("llvm.func definition must have a body");
            for _ in 0..MAX_ITERATIONS {
                if !hoist_in_region(ctx, region, &hoistable) {
                    break;
                }
                any = true;
            }
        }
        Ok(if any { changed() } else { unchanged() })
    }
}

/// Pure AND non-trapping AND with at least one operand (see module docs).
pub(crate) fn hoistable_op_ids() -> FxHashSet<OpId> {
    let mut ids = FxHashSet::default();
    ids.insert(AddOp::get_opid_static());
    ids.insert(SubOp::get_opid_static());
    ids.insert(MulOp::get_opid_static());
    ids.insert(AndOp::get_opid_static());
    ids.insert(OrOp::get_opid_static());
    ids.insert(XorOp::get_opid_static());
    ids.insert(ShlOp::get_opid_static());
    ids.insert(LShrOp::get_opid_static());
    ids.insert(ICmpOp::get_opid_static());
    ids.insert(ZExtOp::get_opid_static());
    ids.insert(SExtOp::get_opid_static());
    ids.insert(TruncOp::get_opid_static());
    ids.insert(BitcastOp::get_opid_static());
    ids.insert(IntToPtrOp::get_opid_static());
    ids.insert(PtrToIntOp::get_opid_static());
    ids.insert(GetElementPtrOp::get_opid_static());
    ids.insert(ExtractValueOp::get_opid_static());
    ids
}

fn defined_in(ctx: &Context, value: Value, body: &FxHashSet<Ptr<BasicBlock>>) -> bool {
    match value.defining_entity() {
        DefiningEntity::Op(op) => op
            .deref(ctx)
            .get_container()
            .map(|block| body.contains(&block))
            .unwrap_or(false),
        DefiningEntity::Block(block) => body.contains(&block),
    }
}

fn hoist_in_region(
    ctx: &mut Context,
    region: Ptr<Region>,
    hoistable: &FxHashSet<OpId>,
) -> bool {
    let dom = dominator_tree(ctx, region);
    let loops = natural_loops(ctx, &dom);
    let mut changed = false;
    for natural_loop in loops {
        let Some(preheader) = natural_loop.preheader(ctx) else {
            continue;
        };
        let Some(terminator) = preheader.deref(ctx).get_terminator(ctx) else {
            continue;
        };
        // Iterate to a local fixpoint so chains (base address, then the
        // GEP off it) hoist in one pass over the loop.
        loop {
            let mut moved = false;
            let body_blocks: Vec<_> = natural_loop.body.iter().copied().collect();
            for block in body_blocks {
                let ops: Vec<_> = block.deref(ctx).iter(ctx).collect();
                for op in ops {
                    let opid = Operation::get_opid(op, ctx);
                    if !hoistable.contains(&opid) {
                        continue;
                    }
                    let operands: Vec<Value> = {
                        let operation = op.deref(ctx);
                        (0..operation.get_num_operands())
                            .map(|i| operation.get_operand(i))
                            .collect()
                    };
                    if operands.is_empty()
                        || operands
                            .iter()
                            .any(|operand| defined_in(ctx, *operand, &natural_loop.body))
                    {
                        continue;
                    }
                    op.unlink(ctx);
                    op.insert_before(ctx, terminator);
                    moved = true;
                    changed = true;
                }
            }
            if !moved {
                break;
            }
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use pliron::builtin::op_interfaces::OneResultInterface as _;
    use pliron_llvm::op_interfaces::{BinArithOp as _, IntBinArithOpWithOverflowFlag as _};
    use crate::{
        dialects::{
            builtin::types::{IntegerType, Signedness},
            llvm::{
                attributes::LinkageAttr,
                ops::{
                    BrOp, CondBrOp, FuncOp, GepIndex, LoadOp, ReturnOp, SDivOp, StoreOp,
                    UndefOp,
                },
                types::{FuncType, PointerType},
            },
        },
        ir::r#type::{TypeHandle, TypedHandle},
        printable::Printable,
    };

    fn int_ty(ctx: &mut Context, width: u32) -> TypedHandle<IntegerType> {
        IntegerType::get(ctx, width, Signedness::Signless)
    }

    /// while-loop: entry(preheader) → header → {body → header | exit}.
    /// Returns (func, entry, header, body, exit, arg a, arg p).
    fn loop_scaffold(
        ctx: &mut Context,
    ) -> (
        FuncOp,
        Ptr<BasicBlock>,
        Ptr<BasicBlock>,
        Ptr<BasicBlock>,
        Ptr<BasicBlock>,
        Value,
        Value,
    ) {
        let i64_ty: TypeHandle = int_ty(ctx, 64).into();
        let ptr_ty: TypeHandle = PointerType::get(ctx, 0).into();
        let fn_ty = FuncType::get(ctx, i64_ty, vec![i64_ty, ptr_ty], false);
        let func = FuncOp::new(ctx, "l".try_into().unwrap(), fn_ty);
        func.set_attr_llvm_function_linkage(ctx, LinkageAttr::ExternalLinkage);
        func.get_or_create_entry_block(ctx);
        let region = func.get_region(ctx).unwrap();
        let entry = func.get_entry_block(ctx).unwrap();
        let header = BasicBlock::new(ctx, None, vec![]);
        header.insert_at_back(region, ctx);
        let body = BasicBlock::new(ctx, None, vec![]);
        body.insert_at_back(region, ctx);
        let exit = BasicBlock::new(ctx, None, vec![]);
        exit.insert_at_back(region, ctx);
        let a = entry.deref(ctx).get_argument(0);
        let p = entry.deref(ctx).get_argument(1);
        BrOp::new(ctx, header, vec![])
            .get_operation()
            .insert_at_back(entry, ctx);
        let i1: TypeHandle = int_ty(ctx, 1).into();
        let cond = UndefOp::new(ctx, i1);
        cond.get_operation().insert_at_back(header, ctx);
        let cond_v = cond.get_result(ctx);
        CondBrOp::new(ctx, cond_v, body, vec![], exit, vec![])
            .get_operation()
            .insert_at_back(header, ctx);
        BrOp::new(ctx, header, vec![])
            .get_operation()
            .insert_at_back(body, ctx);
        ReturnOp::new(ctx, Some(a))
            .get_operation()
            .insert_at_back(exit, ctx);
        (func, entry, header, body, exit, a, p)
    }

    fn block_ops_text(ctx: &Context, block: Ptr<BasicBlock>) -> String {
        block
            .deref(ctx)
            .iter(ctx)
            .map(|op| format!("{}\n", op.disp(ctx)))
            .collect()
    }

    #[test]
    fn hoists_invariant_chain_to_preheader() {
        let mut ctx = Context::new();
        let (func, entry, _header, body, _exit, a, p) = loop_scaffold(&mut ctx);
        let i64_ty: TypeHandle = int_ty(&mut ctx, 64).into();

        // Invariant chain in the body: t = a + a; addr = gep p[t]; plus a
        // store through it (not hoistable) so the loop isn't empty.
        let add = AddOp::new_with_overflow_flag(&mut ctx, a, a, Default::default());
        let add_v = add.get_result(&ctx);
        let gep = GetElementPtrOp::new(&mut ctx, p, vec![GepIndex::Value(add_v)], i64_ty);
        let gep_v = gep.get_result(&ctx);
        let store = StoreOp::new(&mut ctx, a, gep_v);
        let terminator = body.deref(&ctx).get_terminator(&ctx).unwrap();
        add.get_operation().insert_before(&ctx, terminator);
        gep.get_operation().insert_before(&ctx, terminator);
        store.get_operation().insert_before(&ctx, terminator);

        LLVMLicmPass
            .run(func.get_operation(), &mut ctx, &mut AnalysisManager::default())
            .unwrap();
        let entry_text = block_ops_text(&ctx, entry);
        let body_text = block_ops_text(&ctx, body);
        assert!(entry_text.contains("llvm.add"), "{entry_text}");
        assert!(entry_text.contains("llvm.gep"), "{entry_text}");
        assert!(!body_text.contains("llvm.add"), "{body_text}");
        assert!(!body_text.contains("llvm.gep"), "{body_text}");
        assert!(body_text.contains("llvm.store"), "{body_text}");
    }

    #[test]
    fn does_not_hoist_variant_trapping_or_memory_ops() {
        let mut ctx = Context::new();
        let (func, entry, header, body, _exit, a, p) = loop_scaffold(&mut ctx);
        let i64_ty: TypeHandle = int_ty(&mut ctx, 64).into();

        // A loop-variant value: a load in the header (loads are pinned,
        // so its result is genuinely defined inside the loop every
        // iteration).
        let variant = LoadOp::new(&mut ctx, p, i64_ty);
        let header_first = header.deref(&ctx).get_head().unwrap();
        variant.get_operation().insert_before(&ctx, header_first);
        let variant_v = variant.get_result(&ctx);

        let terminator = body.deref(&ctx).get_terminator(&ctx).unwrap();
        // variant-fed add: must stay.
        let dep = AddOp::new_with_overflow_flag(&mut ctx, variant_v, a, Default::default());
        dep.get_operation().insert_before(&ctx, terminator);
        // invariant sdiv: trapping, must stay.
        let div = SDivOp::new(&mut ctx, a, a);
        div.get_operation().insert_before(&ctx, terminator);
        // invariant load: memory, must stay.
        let load = LoadOp::new(&mut ctx, p, i64_ty);
        load.get_operation().insert_before(&ctx, terminator);

        LLVMLicmPass
            .run(func.get_operation(), &mut ctx, &mut AnalysisManager::default())
            .unwrap();
        let entry_text = block_ops_text(&ctx, entry);
        let body_text = block_ops_text(&ctx, body);
        assert!(!entry_text.contains("llvm.sdiv"), "{entry_text}");
        assert!(!entry_text.contains("llvm.load"), "{entry_text}");
        assert!(body_text.contains("llvm.sdiv"), "{body_text}");
        assert!(body_text.contains("llvm.load"), "{body_text}");
        assert!(body_text.contains("llvm.add"), "{body_text}");
        // The header load feeding it is pinned in the header.
        assert!(block_ops_text(&ctx, header).contains("llvm.load"));
    }
}
