//! Op-level identity and `derived_from` stamping for backward profile
//! attribution (docs/PROFILE-FEEDBACK-BACKWARD.md).
//!
//! Forward direction: [Aarch64OpIdPass] runs immediately before
//! instruction selection and stamps every op of every `llvm.func` with an
//! [OpIdAttr] — its dense per-function index in program order. Instruction
//! selection is the first adjoint: every machine op it emits records the
//! source op's id in a [DerivedFromAttr]. Machine passes that create ops
//! later either propagate the attribute from the op that caused the new
//! code (frame rewrites, spill stores/reloads) or use a synthetic
//! per-pass [roots] id (prologue/epilogue, ABI moves, layout branches) so
//! the overhead is attributed to the pass that created it.
//!
//! The sidecar (see [super::blockmap]) then extends each block range with
//! per-op ranges `{id, derived_from, start, end}` in final layout order,
//! which is everything the ingestion tool needs to lift PC-sample costs
//! from machine instructions onto LLVM-level op ids — the first backward
//! hop of the full design. All stamping is gated on the same environment
//! switch as the blockmap ([super::blockmap::blockmap_enabled]:
//! `CRABBIT_BLOCKMAP` or `CRABBIT_PROFILE_MAP`).

use crate::ll::{DerivedFromAttr, DerivedFromManyAttr, InlinedFromAttr, OpIdAttr};
use pliron::builtin::op_interfaces::AtMostOneRegionInterface as _;

use crate::{
    context::{Context, Ptr},
    dialects::builtin::op_interfaces::OneRegionInterface,
    dialects::llvm::ops::FuncOp as LlvmFuncOp,
    ir::{basic_block::BasicBlock, operation::Operation},
    linked_list::{ContainsLinkedList, LinkedList as _},
    conversion::pass::{AnalysisManager, Pass, PassResult, changed},
};

use super::blockmap::blockmap_enabled;
use super::frontend::module_op;

use crate::dict_key;

dict_key!(ATTR_KEY_AARCH64_OP_ID, "aarch64_op_id");
dict_key!(ATTR_KEY_AARCH64_DERIVED_FROM, "aarch64_derived_from");
dict_key!(ATTR_KEY_DERIVED_FROM_MANY, "ll_derived_from_many");
dict_key!(ATTR_KEY_INLINED_FROM, "ll_inlined_from");
dict_key!(ATTR_KEY_MIDEND_BOUNDARY, "aarch64_midend_boundary");

/// Synthetic `derived_from` roots: code created from nothing is attributed
/// to the pass that created it. Negative so they can never collide with
/// LLVM op ids; the ingestion tool renders them by [root_name].
pub mod roots {
    /// ABI glue emitted by instruction selection before any source op:
    /// link-register save, incoming-argument moves and loads, sret spill.
    pub const ISEL_ABI: i64 = -1;
    /// Register-allocator code whose anchor op carried no attribution.
    pub const REGALLOC: i64 = -2;
    /// Frame prologue/epilogue (`sub sp`/`add sp` and their materialized
    /// offsets).
    pub const FRAME: i64 = -3;
    /// Branches materialized by block placement for what used to be
    /// fallthrough.
    pub const PLACEMENT: i64 = -4;
    /// Final-IR ops that no pass stamped (reported by the sidecar so
    /// nothing is silently dropped; [unstamped_op_count] audits this).
    pub const UNATTRIBUTED: i64 = -5;
    /// Ops created by a mid-end pass that declared no adjoint for them —
    /// visible as a category so imprecise mid-end stamping is loud, not
    /// silent.
    pub const MIDEND: i64 = -6;
}

/// Human-readable name of a synthetic root id, for sidecars/reports.
pub fn root_name(id: i64) -> Option<&'static str> {
    Some(match id {
        roots::ISEL_ABI => "isel:abi",
        roots::REGALLOC => "regalloc",
        roots::FRAME => "frame",
        roots::PLACEMENT => "placement",
        roots::UNATTRIBUTED => "unattributed",
        roots::MIDEND => "midend",
        _ => return None,
    })
}

/// Stamps LLVM-level op ids at the RA boundary when [blockmap_enabled];
/// a no-op otherwise.
///
/// Reconciliation with the mid-end numbering (docs/PROFILE-FEEDBACK-
/// BACKWARD.md): ops that still carry the `ll.op_id` stamped at the
/// mid-end HEAD (and were not inlined) KEEP it — so machine-level
/// `derived_from` refers directly to the pre-mid-end numbering for
/// surviving ops. Ops the mid-end created (or inlined, whose callee-local
/// ids would collide) get FRESH ids past the preserved maximum, and the
/// fresh-id → source-parents mapping (the mid-end adjoint boundary table)
/// is serialized as JSON into a `builtin.string` attribute on the module
/// (key `aarch64_midend_boundary`) for the sidecar and offline lift. A
/// created op with no declared adjoint maps to [roots::MIDEND].
pub struct Aarch64OpIdPass;

impl Pass for Aarch64OpIdPass {
    fn name(&self) -> &str {
        "aarch64-op-ids"
    }

    fn run(
        &mut self,
        root: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> pliron::result::Result<PassResult> {
        if blockmap_enabled() {
            assign_boundary_ids(ctx, root)?;
        }
        Ok(changed())
    }
}

/// RA-boundary id assignment (see [Aarch64OpIdPass]). Public for tests.
/// When no op carries a mid-end id (mid-end stamping was off — the
/// round-1 configuration), this degrades to exactly [assign_op_ids].
pub fn assign_boundary_ids(
    ctx: &mut Context,
    root: Ptr<Operation>,
) -> pliron::result::Result<()> {
    let module = module_op(ctx, root)?;
    let body = module.get_region(ctx).deref(ctx).get_head().unwrap();
    let funcs: Vec<_> = body.deref(ctx).iter(ctx).collect();
    let mut boundary = serde_json::Map::new();
    for func_op in funcs {
        let op_obj = Operation::get_op_dyn(func_op, ctx);
        let Some(func) = op_obj.downcast_ref::<LlvmFuncOp>() else {
            continue;
        };
        let Some(region) = func.get_region(ctx) else {
            continue;
        };
        let symbol = pliron::builtin::op_interfaces::SymbolOpInterface::get_symbol_name(
            func, ctx,
        )
        .to_string();
        let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
        // Pass 1: which ids are preserved; the fresh range starts above.
        let mut preserved = std::collections::HashSet::new();
        let mut max_id: i64 = -1;
        let mut any_midend_ids = false;
        for &block in &blocks {
            for op in block.deref(ctx).iter(ctx) {
                if let Some(id) = op_id(ctx, op) {
                    any_midend_ids = true;
                    if inlined_from(ctx, op).is_none() && preserved.insert(id) {
                        max_id = max_id.max(i64::from(id));
                    }
                }
            }
        }
        if !any_midend_ids {
            // Round-1 configuration: no mid-end stamping ran; number
            // everything densely, nothing to reconcile.
            let mut next = 0u32;
            for &block in &blocks {
                let ops: Vec<_> = block.deref(ctx).iter(ctx).collect();
                for op in ops {
                    op.deref_mut(ctx)
                        .attributes
                        .set(ATTR_KEY_AARCH64_OP_ID.clone(), OpIdAttr(next));
                    next += 1;
                }
            }
            continue;
        }
        let mut next = u32::try_from(max_id + 1).unwrap_or(0);
        let mut table = serde_json::Map::new();
        for &block in &blocks {
            let ops: Vec<_> = block.deref(ctx).iter(ctx).collect();
            for op in ops {
                let keep = op_id(ctx, op)
                    .is_some_and(|_| inlined_from(ctx, op).is_none());
                if keep {
                    // A preserved op that absorbed others (GVN merge)
                    // still needs its multi-parent adjoint in the table,
                    // or the merged identities would be lost.
                    let own = i64::from(op_id(ctx, op).unwrap());
                    let merged = derived_from_many(ctx, op)
                        .or_else(|| derived_from(ctx, op).map(|d| vec![d]));
                    if let Some(sources) = merged
                        && sources != vec![own] {
                            table.insert(
                                own.to_string(),
                                serde_json::Value::Array(
                                    sources.into_iter().map(Into::into).collect(),
                                ),
                            );
                        }
                    continue;
                }
                // Source parents BEFORE the fresh id overwrites op_id.
                let mut sources = effective_sources(ctx, op);
                if sources.is_empty() {
                    sources.push(roots::MIDEND);
                }
                table.insert(
                    next.to_string(),
                    serde_json::Value::Array(
                        sources.into_iter().map(Into::into).collect(),
                    ),
                );
                op.deref_mut(ctx)
                    .attributes
                    .set(ATTR_KEY_AARCH64_OP_ID.clone(), OpIdAttr(next));
                next += 1;
            }
        }
        if !table.is_empty() {
            boundary.insert(symbol, serde_json::Value::Object(table));
        }
    }
    if !boundary.is_empty() {
        let json = serde_json::Value::Object(boundary).to_string();
        root.deref_mut(ctx).attributes.set(
            ATTR_KEY_MIDEND_BOUNDARY.clone(),
            pliron::builtin::attributes::StringAttr::new(json),
        );
    }
    Ok(())
}

/// The mid-end boundary table serialized by [assign_boundary_ids], if any:
/// `{function symbol: {fresh RA-boundary id: [source parents]}}`.
pub fn midend_boundary_json(ctx: &Context, root: Ptr<Operation>) -> Option<String> {
    root.deref(ctx)
        .attributes
        .get::<pliron::builtin::attributes::StringAttr>(&ATTR_KEY_MIDEND_BOUNDARY)
        .map(|attr| attr.as_str().to_string())
}

/// Stamp every op of every `llvm.func` definition with its dense
/// per-function program-order index. Public so tests can drive it without
/// touching the environment.
pub fn assign_op_ids(ctx: &mut Context, root: Ptr<Operation>) -> pliron::result::Result<()> {
    let module = module_op(ctx, root)?;
    let body = module.get_region(ctx).deref(ctx).get_head().unwrap();
    let funcs: Vec<_> = body.deref(ctx).iter(ctx).collect();
    for func_op in funcs {
        let op_obj = Operation::get_op_dyn(func_op, ctx);
        let Some(func) = op_obj.downcast_ref::<LlvmFuncOp>() else {
            continue;
        };
        let Some(region) = func.get_region(ctx) else {
            continue;
        };
        let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
        let mut next = 0u32;
        for block in blocks {
            let ops: Vec<_> = block.deref(ctx).iter(ctx).collect();
            for op in ops {
                op.deref_mut(ctx)
                    .attributes
                    .set(ATTR_KEY_AARCH64_OP_ID.clone(), OpIdAttr(next));
                next += 1;
            }
        }
    }
    Ok(())
}

/// The LLVM-level op id stamped on `op`, if any.
pub fn op_id(ctx: &Context, op: Ptr<Operation>) -> Option<u32> {
    op.deref(ctx)
        .attributes
        .get::<OpIdAttr>(&ATTR_KEY_AARCH64_OP_ID)
        .map(|attr| attr.0)
}

/// Multi-parent `derived_from` (equal weights), if any.
pub fn derived_from_many(ctx: &Context, op: Ptr<Operation>) -> Option<Vec<i64>> {
    op.deref(ctx)
        .attributes
        .get::<DerivedFromManyAttr>(&ATTR_KEY_DERIVED_FROM_MANY)
        .map(|attr| attr.0.clone())
}

/// Record a multi-parent `derived_from` (deduplicated; collapses to the
/// single-parent attr when one id remains).
pub fn set_derived_from_many(ctx: &Context, op: Ptr<Operation>, mut from: Vec<i64>) {
    from.sort_unstable();
    from.dedup();
    if from.len() == 1 {
        set_derived_from(ctx, op, from[0]);
        return;
    }
    op.deref_mut(ctx)
        .attributes
        .set(ATTR_KEY_DERIVED_FROM_MANY.clone(), DerivedFromManyAttr(from));
}

/// The call-site id this op was inlined under, if any.
pub fn inlined_from(ctx: &Context, op: Ptr<Operation>) -> Option<i64> {
    op.deref(ctx)
        .attributes
        .get::<InlinedFromAttr>(&ATTR_KEY_INLINED_FROM)
        .map(|attr| attr.0)
}

/// Record the call-site id on an inlined op.
pub fn set_inlined_from(ctx: &Context, op: Ptr<Operation>, callsite: i64) {
    op.deref_mut(ctx)
        .attributes
        .set(ATTR_KEY_INLINED_FROM.clone(), InlinedFromAttr(callsite));
}

/// The op's effective SOURCE-level parents (pre-mid-end ids or synthetic
/// roots), applying the attribution precedence: an explicit multi-parent
/// or single-parent derivation wins (mid-end adjoints write those eagerly
/// at source level), then an inlined op belongs to its call site, then an
/// op stamped at the mid-end head IS a source op. Empty = unattributed.
pub fn effective_sources(ctx: &Context, op: Ptr<Operation>) -> Vec<i64> {
    if let Some(many) = derived_from_many(ctx, op) {
        return many;
    }
    if let Some(single) = derived_from(ctx, op) {
        return vec![single];
    }
    if let Some(callsite) = inlined_from(ctx, op) {
        return vec![callsite];
    }
    if let Some(id) = op_id(ctx, op) {
        return vec![i64::from(id)];
    }
    Vec::new()
}

/// Whether `op` already carries ANY attribution — the exact complement
/// of what [effective_sources] reads, `inlined_from` included: an op
/// stamped only with its call-site tag is attributed, and re-stamping it
/// with `derived_from_many` (which outranks `inlined_from`) would clobber
/// the call-site dimension of the backward lift.
pub fn has_attribution(ctx: &Context, op: Ptr<Operation>) -> bool {
    derived_from(ctx, op).is_some()
        || derived_from_many(ctx, op).is_some()
        || inlined_from(ctx, op).is_some()
        || op_id(ctx, op).is_some()
}

/// A mid-end pass's adjoint in one call: stamp a newly created `new_op`
/// as derived from `source_op`'s effective sources. No-op when
/// attribution is off (source carries nothing) or the new op is already
/// attributed.
pub fn derive_new_from(ctx: &Context, new_op: Ptr<Operation>, source_op: Ptr<Operation>) {
    if has_attribution(ctx, new_op) {
        return;
    }
    let sources = effective_sources(ctx, source_op);
    if !sources.is_empty() {
        set_derived_from_many(ctx, new_op, sources);
    }
}

/// Chain adjoint for replacement-style rewrites: stamp `value`'s defining
/// op — and transitively the defining ops of its operands — with
/// `source_op`'s effective sources, stopping at any op that already
/// carries attribution (an original op, or one another adjoint stamped).
/// Covers whole freshly-built expansion chains through one hook.
pub fn derive_chain_from(ctx: &Context, value: crate::ir::value::Value, source_op: Ptr<Operation>) {
    let sources = effective_sources(ctx, source_op);
    if sources.is_empty() {
        return;
    }
    let mut work: Vec<Ptr<Operation>> = match value.defining_op() {
        Some(op) => vec![op],
        None => return,
    };
    let mut seen = std::collections::HashSet::new();
    while let Some(op) = work.pop() {
        if !seen.insert(op) {
            continue;
        }
        if has_attribution(ctx, op) {
            continue;
        }
        set_derived_from_many(ctx, op, sources.clone());
        let operands: Vec<_> = op.deref(ctx).operands().collect();
        for operand in operands {
            if let Some(def) = operand.defining_op() {
                work.push(def);
            }
        }
    }
}

/// Bracket helper for expansion-style mid-end rewrites (1→N): stamp every
/// op inserted immediately BEFORE `anchor`, later than `prev_tail`
/// (exclusive; `None` = block head), that lacks attribution, as derived
/// from `anchor`'s effective sources.
pub fn stamp_expansion_before(
    ctx: &Context,
    anchor: Ptr<Operation>,
    prev_tail: Option<Ptr<Operation>>,
) {
    let sources = effective_sources(ctx, anchor);
    if sources.is_empty() {
        return;
    }
    let Some(block) = anchor.deref(ctx).get_container() else {
        return;
    };
    let mut cursor = match prev_tail {
        Some(op) => op.deref(ctx).get_next(),
        None => block.deref(ctx).get_head(),
    };
    while let Some(op) = cursor {
        if op == anchor {
            break;
        }
        cursor = op.deref(ctx).get_next();
        if !has_attribution(ctx, op) {
            set_derived_from_many(ctx, op, sources.clone());
        }
    }
}

/// The `derived_from` recorded on a machine op, if any.
pub fn derived_from(ctx: &Context, op: Ptr<Operation>) -> Option<i64> {
    op.deref(ctx)
        .attributes
        .get::<DerivedFromAttr>(&ATTR_KEY_AARCH64_DERIVED_FROM)
        .map(|attr| attr.0)
}

/// Record `derived_from` on a machine op (overwrites).
pub fn set_derived_from(ctx: &Context, op: Ptr<Operation>, from: i64) {
    op.deref_mut(ctx)
        .attributes
        .set(ATTR_KEY_AARCH64_DERIVED_FROM.clone(), DerivedFromAttr(from));
}

/// Propagate attribution from `anchor` onto a newly created `op`: copies
/// the anchor's `derived_from` when present, else stamps `fallback_root`.
/// A no-op when the anchor carries neither (attribution disabled).
pub fn inherit_derived_from(
    ctx: &Context,
    anchor: Ptr<Operation>,
    op: Ptr<Operation>,
    fallback_root: i64,
) {
    match derived_from(ctx, anchor) {
        Some(from) => set_derived_from(ctx, op, from),
        None => {
            // Only stamp the fallback when attribution is live at all —
            // detectable by the anchor having an op id (pre-erase LLVM op)
            // or by any stamping having happened, which callers know via
            // blockmap_enabled(); re-check cheaply here.
            if blockmap_enabled() {
                set_derived_from(ctx, op, fallback_root);
            }
        }
    }
}

/// Stamp every op after `tail` (exclusive; `None` = whole block) in
/// `block` that lacks a `derived_from` with `from`.
pub fn stamp_ops_after(
    ctx: &Context,
    block: Ptr<BasicBlock>,
    tail: Option<Ptr<Operation>>,
    from: i64,
) {
    let mut cursor = match tail {
        Some(op) => op.deref(ctx).get_next(),
        None => block.deref(ctx).get_head(),
    };
    while let Some(op) = cursor {
        cursor = op.deref(ctx).get_next();
        if derived_from(ctx, op).is_none() {
            set_derived_from(ctx, op, from);
        }
    }
}

/// The number of machine ops in `root`'s functions that carry no
/// `derived_from` — the audit for "every op is attributed". Only
/// meaningful after the full pipeline ran with stamping enabled.
pub fn unstamped_op_count(ctx: &Context, root: Ptr<Operation>) -> pliron::result::Result<usize> {
    let module = module_op(ctx, root)?;
    let body = module.get_region(ctx).deref(ctx).get_head().unwrap();
    let mut count = 0usize;
    for func in body.deref(ctx).iter(ctx) {
        for region in func.deref(ctx).regions() {
            for block in region.deref(ctx).iter(ctx) {
                for op in block.deref(ctx).iter(ctx) {
                    if derived_from(ctx, op).is_none() {
                        count += 1;
                    }
                }
            }
        }
    }
    Ok(count)
}

/// Brackets instruction selection's per-source-op emission: everything a
/// lowering arm appends to the current machine block (or into machine
/// blocks it creates, e.g. cond-branch edge blocks) between two
/// [Self::begin] calls is stamped as derived from that source op. This
/// covers every arm of the isel dispatch with two hooks instead of
/// per-site edits; ABI glue emitted before any source op is stamped
/// [roots::ISEL_ABI] by [Self::preamble].
pub struct IselStamper {
    enabled: bool,
    region: Ptr<crate::ir::region::Region>,
    known_blocks: std::collections::HashSet<Ptr<BasicBlock>>,
    pending: Option<(Ptr<BasicBlock>, Option<Ptr<Operation>>, i64)>,
}

impl IselStamper {
    /// `enabled` is decided by the caller (op ids present on the source
    /// function); everything is a no-op otherwise.
    pub fn new(ctx: &Context, region: Ptr<crate::ir::region::Region>, enabled: bool) -> Self {
        let known_blocks = if enabled {
            region.deref(ctx).iter(ctx).collect()
        } else {
            Default::default()
        };
        IselStamper {
            enabled,
            region,
            known_blocks,
            pending: None,
        }
    }

    /// Stamp everything currently in `entry` (the pre-source ABI glue:
    /// link-register save, argument moves/loads, sret spill) as
    /// [roots::ISEL_ABI].
    pub fn preamble(&self, ctx: &Context, entry: Ptr<BasicBlock>) {
        if self.enabled {
            stamp_ops_after(ctx, entry, None, roots::ISEL_ABI);
        }
    }

    /// Close the previous source op's bracket and open one for the op with
    /// id `source`, whose lowering will append to `insert_block`.
    pub fn begin(
        &mut self,
        ctx: &Context,
        insert_block: Ptr<BasicBlock>,
        source: Option<u32>,
    ) {
        if !self.enabled {
            return;
        }
        self.close(ctx);
        let tail = insert_block.deref(ctx).get_tail();
        let from = source.map(i64::from).unwrap_or(roots::UNATTRIBUTED);
        self.pending = Some((insert_block, tail, from));
    }

    /// Close the final bracket after the last source op was lowered.
    pub fn finish(&mut self, ctx: &Context) {
        if self.enabled {
            self.close(ctx);
        }
    }

    fn close(&mut self, ctx: &Context) {
        let Some((block, tail, from)) = self.pending.take() else {
            return;
        };
        stamp_ops_after(ctx, block, tail, from);
        // Blocks created during this bracket (edge blocks) belong wholly
        // to the bracketed source op.
        let blocks: Vec<_> = self.region.deref(ctx).iter(ctx).collect();
        for candidate in blocks {
            if self.known_blocks.insert(candidate) {
                stamp_ops_after(ctx, candidate, None, from);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialects::builtin::{
        attributes::IntegerAttr,
        ops::ConstantOp,
        types::{IntegerType, Signedness},
    };
    use crate::ir::op::Op as _;
    use crate::utils::apint::APInt;
    use std::num::NonZero;

    fn constant(ctx: &mut Context) -> Ptr<Operation> {
        let ty = IntegerType::get(ctx, 64, Signedness::Signless);
        let attr = IntegerAttr::new(ty, APInt::from_u64(1, NonZero::new(64).unwrap()));
        ConstantOp::new(ctx, Box::new(attr)).get_operation()
    }

    /// An op carrying only `inlined_from` is attributed: the derive
    /// helpers must not clobber the call-site tag with a
    /// `derived_from_many` (which would outrank it in
    /// [effective_sources]).
    #[test]
    fn inlined_from_counts_as_attribution() {
        let mut ctx = Context::new();
        let source = constant(&mut ctx);
        set_derived_from(&ctx, source, 7);
        let inlined = constant(&mut ctx);
        set_inlined_from(&ctx, inlined, 3);

        assert!(has_attribution(&ctx, inlined));
        derive_new_from(&ctx, inlined, source);
        assert_eq!(derived_from_many(&ctx, inlined), None);
        assert_eq!(
            effective_sources(&ctx, inlined),
            vec![3],
            "call-site attribution must survive derive_new_from"
        );
    }
}
