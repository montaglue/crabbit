//! `mir-census`: a pure-analysis pass over a module of dialect-mir functions,
//! run between the importer and `lower-dialect-mir`.
//!
//! The importer currently pre-lowers much of Rust's semantics itself
//! (`undef`+`insert_value` aggregates, bit-trick overflow checks, ...) instead
//! of emitting dialect-mir's Rust-semantic ops (`mir.construct_struct`,
//! `mir.checked_add`, `mir.assert`, ...). Moving towards a thin importer
//! needs a measurement of what the mir-level IR actually contains today,
//! per construct; this pass is that measurement. Per `mir.func` it records
//! an op-kind histogram (full op ids, so the importer's `llvm.*` spellings
//! show up next to the `mir.*` ones), calls grouped by demangled callee and
//! by callee "family", the alloca / assert / checked-arithmetic counts,
//! aggregate construction by `mir.construct_*` versus `undef`+insert chains,
//! natural-loop count and nesting depth, and block/op totals; plus the
//! module-level aggregation.
//!
//! It mutates no IR and never fails a compilation: when
//! `CRABBIT_MIR_CENSUS=<path>` is set, one JSON object per module is
//! appended to that file (JSON Lines); errors and panics inside the census
//! are reported on stderr and the pass returns unchanged. When the variable
//! is unset the walk is skipped entirely — the census is a measurement
//! tool, so an always-on walk (O(ops) hashing plus a dominator tree per
//! function) would be paid on every compile for a result nobody reads.
//! `CRABBIT_MIR_DISABLE=census` skips it too ([super::gate]).
//!
//! The [census] function is the env-free API (tests, tooling); [Census]
//! renders itself as Markdown through [std::fmt::Display].

use std::{
    collections::BTreeMap,
    fmt,
    io::Write as _,
    panic::{AssertUnwindSafe, catch_unwind},
    path::PathBuf,
};

use pliron::{
    builtin::{
        op_interfaces::{CallOpCallable, CallOpInterface as _, SymbolOpInterface as _},
        ops::ModuleOp,
    },
    common_traits::Verify as _,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList as _,
    op::Op as _,
    operation::Operation,
    region::Region,
    result::Result,
};
use pliron_ll::{
    conversion::pass::{AnalysisManager, Pass, PassResult, unchanged},
    passes::llvm::analysis::{dominator_tree, natural_loops},
};
use serde::Serialize;

use super::gate::mir_disabled;

/// The environment variable naming the JSON Lines file the pass appends to.
pub const CENSUS_ENV: &str = "CRABBIT_MIR_CENSUS";
/// The pass name (also the `CRABBIT_MIR_DISABLE` entry, minus the `mir-`).
pub const PASS_NAME: &str = "mir-census";
const GATE_NAME: &str = "census";

/// Aggregate-construction counts. `construct` is dialect-mir's native
/// spelling (`mir.construct_{struct,tuple,array,slice,enum}`); `undef` and
/// `insert` count both dialects' spellings of the pre-lowered form
/// (`mir.undef`/`llvm.undef`, `mir.insert_field`/`llvm.insert_value`), and
/// `undef_insert_chains` is the number of undef ops that feed at least one
/// insert — i.e. aggregate values built the pre-lowered way.
#[derive(Serialize, Default, Clone, Debug, PartialEq, Eq)]
pub struct AggregateCensus {
    pub construct: usize,
    pub undef: usize,
    pub insert: usize,
    pub undef_insert_chains: usize,
}

/// The summable per-function measurements (also the shape of the module
/// totals, where `max_loop_depth` is the maximum over functions).
#[derive(Serialize, Default, Clone, Debug, PartialEq, Eq)]
pub struct Counts {
    pub blocks: usize,
    pub ops: usize,
    /// Full op id (`mir.insert_field`, `llvm.undef`, ...) → count.
    pub op_kinds: BTreeMap<String, usize>,
    /// Demangled callee (generics kept, hashes dropped) → call count.
    /// Direct `mir.call` and `llvm.call` callees; indirect calls are
    /// grouped under `<indirect>`.
    pub calls: BTreeMap<String, usize>,
    /// Callee family (see [family]) → call count.
    pub call_families: BTreeMap<String, usize>,
    pub allocas: usize,
    pub asserts: usize,
    /// `mir.checked_{add,sub,mul}`.
    pub checked_arith: usize,
    pub aggregates: AggregateCensus,
    /// Natural loops (back edges merged per header) over the CFG.
    pub loops: usize,
    /// Deepest loop nesting (1 = a loop not inside another loop).
    pub max_loop_depth: usize,
    /// Verifier failures, `"<op id>: <reason>"` → count: each op verified
    /// on its own (pliron's full per-op check, which includes dialect-mir's
    /// rules), the reason with its type-specific tail cut off
    /// (`verifier_reason`). `mir-lower` stops at the first failure; this is
    /// the whole distance to its typing contract.
    pub verifier_failures: BTreeMap<String, usize>,
}

impl Counts {
    fn total_calls(&self) -> usize {
        self.calls.values().sum()
    }

    fn absorb(&mut self, other: &Counts) {
        self.blocks += other.blocks;
        self.ops += other.ops;
        merge_into(&mut self.op_kinds, &other.op_kinds);
        merge_into(&mut self.calls, &other.calls);
        merge_into(&mut self.call_families, &other.call_families);
        self.allocas += other.allocas;
        self.asserts += other.asserts;
        self.checked_arith += other.checked_arith;
        self.aggregates.construct += other.aggregates.construct;
        self.aggregates.undef += other.aggregates.undef;
        self.aggregates.insert += other.aggregates.insert;
        self.aggregates.undef_insert_chains += other.aggregates.undef_insert_chains;
        self.loops += other.loops;
        self.max_loop_depth = self.max_loop_depth.max(other.max_loop_depth);
        merge_into(&mut self.verifier_failures, &other.verifier_failures);
    }

    fn total_verifier_failures(&self) -> usize {
        self.verifier_failures.values().sum()
    }
}

fn merge_into(into: &mut BTreeMap<String, usize>, from: &BTreeMap<String, usize>) {
    for (key, count) in from {
        *into.entry(key.clone()).or_default() += count;
    }
}

/// One `mir.func`'s census (keyed by its raw symbol in [Census::functions]).
#[derive(Serialize, Default, Clone, Debug, PartialEq, Eq)]
pub struct FunctionCensus {
    /// The symbol demangled (hashes dropped), for reading.
    pub demangled: String,
    #[serde(flatten)]
    pub counts: Counts,
}

/// Module-level aggregation.
#[derive(Serialize, Default, Clone, Debug, PartialEq, Eq)]
pub struct Totals {
    /// Number of `mir.func`s.
    pub functions: usize,
    /// Number of `llvm.func` declarations at module level (the importer's
    /// external/intrinsic declarations).
    pub declarations: usize,
    /// Other top-level op kinds (globals, ...) → count.
    pub other_top_level: BTreeMap<String, usize>,
    /// Op count per dialect (`mir`, `llvm`, ...) over all function bodies.
    pub ops_by_dialect: BTreeMap<String, usize>,
    #[serde(flatten)]
    pub counts: Counts,
}

/// The census of one module.
#[derive(Serialize, Clone, Debug, PartialEq, Eq)]
pub struct Census {
    /// The crate being compiled (rustc's crate name) when the caller knows
    /// it; `None` for a bare module.
    #[serde(rename = "crate")]
    pub crate_name: Option<String>,
    /// The module symbol (`rust_crate`, `rust_kernels`).
    pub module: String,
    pub functions: BTreeMap<String, FunctionCensus>,
    pub totals: Totals,
}

/// Compute the census of `module` (a `builtin.module` holding `mir.func`s).
pub fn census(ctx: &Context, module: Ptr<Operation>) -> Census {
    census_labelled(ctx, module, None)
}

/// [census] with the crate label recorded in [Census::crate_name].
pub fn census_labelled(ctx: &Context, module: Ptr<Operation>, label: Option<&str>) -> Census {
    let module_name = Operation::get_op::<ModuleOp>(module, ctx)
        .map(|m| m.get_symbol_name(ctx).to_string())
        .unwrap_or_else(|| Operation::get_opid(module, ctx).to_string());
    let mut out = Census {
        crate_name: label.map(str::to_string),
        module: module_name,
        functions: BTreeMap::new(),
        totals: Totals::default(),
    };
    let top_level: Vec<Ptr<Operation>> = module
        .deref(ctx)
        .regions()
        .flat_map(|region| region.deref(ctx).iter(ctx))
        .flat_map(|block| block.deref(ctx).iter(ctx))
        .collect();
    for op in top_level {
        if let Some(func) = dialect_mir::ops::MirFuncOp::wrap(ctx, op) {
            let symbol = func.get_symbol_name(ctx).to_string();
            let fc = function_census(ctx, op, &symbol);
            out.totals.counts.absorb(&fc.counts);
            out.totals.functions += 1;
            out.functions.insert(symbol, fc);
            continue;
        }
        let kind = Operation::get_opid(op, ctx).to_string();
        if kind == "llvm.func" {
            out.totals.declarations += 1;
        } else {
            *out.totals.other_top_level.entry(kind).or_default() += 1;
        }
    }
    for (kind, count) in &out.totals.counts.op_kinds {
        let dialect = kind.split('.').next().unwrap_or(kind).to_string();
        *out.totals.ops_by_dialect.entry(dialect).or_default() += count;
    }
    out
}

fn function_census(ctx: &Context, func: Ptr<Operation>, symbol: &str) -> FunctionCensus {
    let mut counts = Counts::default();
    let regions: Vec<Ptr<Region>> = func.deref(ctx).regions().collect();
    for region in &regions {
        visit_region(ctx, *region, &mut counts);
    }
    if let Some(body) = regions.first() {
        let (loops, depth) = loop_metrics(ctx, *body);
        counts.loops = loops;
        counts.max_loop_depth = depth;
    }
    FunctionCensus {
        demangled: demangle(symbol),
        counts,
    }
}

fn visit_region(ctx: &Context, region: Ptr<Region>, counts: &mut Counts) {
    let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
    for block in blocks {
        counts.blocks += 1;
        let ops: Vec<Ptr<Operation>> = block.deref(ctx).iter(ctx).collect();
        for op in ops {
            classify(ctx, op, counts);
            let nested: Vec<Ptr<Region>> = op.deref(ctx).regions().collect();
            for region in nested {
                visit_region(ctx, region, counts);
            }
        }
    }
}

const CONSTRUCT_OPS: &[&str] = &[
    "mir.construct_struct",
    "mir.construct_tuple",
    "mir.construct_array",
    "mir.construct_slice",
    "mir.construct_enum",
];
const UNDEF_OPS: &[&str] = &["mir.undef", "llvm.undef"];
const INSERT_OPS: &[&str] = &["mir.insert_field", "llvm.insert_value"];
const CHECKED_OPS: &[&str] = &["mir.checked_add", "mir.checked_sub", "mir.checked_mul"];

fn classify(ctx: &Context, op: Ptr<Operation>, counts: &mut Counts) {
    let kind = Operation::get_opid(op, ctx).to_string();
    counts.ops += 1;
    *counts.op_kinds.entry(kind.clone()).or_default() += 1;
    if let Some(reason) = verifier_failure(ctx, op) {
        *counts
            .verifier_failures
            .entry(format!("{kind}: {reason}"))
            .or_default() += 1;
    }
    let kind = kind.as_str();
    match kind {
        "mir.alloca" => counts.allocas += 1,
        "mir.assert" => counts.asserts += 1,
        "mir.call" => {
            let callee = Operation::get_op::<dialect_mir::ops::MirCallOp>(op, ctx)
                .and_then(|call| {
                    call.get_attr_callee(ctx)
                        .map(|c| String::from((*c).clone()))
                })
                .unwrap_or_else(|| "<mir.call without callee>".to_string());
            record_call(counts, &callee);
        }
        "llvm.call" => {
            let callee = Operation::get_op::<pliron_llvm::ops::CallOp>(op, ctx)
                .map(|call| match call.callee(ctx) {
                    CallOpCallable::Direct(symbol) => symbol.to_string(),
                    CallOpCallable::Indirect(_) => "<indirect>".to_string(),
                })
                .unwrap_or_else(|| "<indirect>".to_string());
            record_call(counts, &callee);
        }
        _ if CHECKED_OPS.contains(&kind) => counts.checked_arith += 1,
        _ if CONSTRUCT_OPS.contains(&kind) => counts.aggregates.construct += 1,
        _ if INSERT_OPS.contains(&kind) => counts.aggregates.insert += 1,
        _ if UNDEF_OPS.contains(&kind) => {
            counts.aggregates.undef += 1;
            let users: Vec<Ptr<Operation>> = op.deref(ctx).uses().map(|u| u.user_op()).collect();
            if users.iter().any(|user| {
                INSERT_OPS.contains(&Operation::get_opid(*user, ctx).to_string().as_str())
            }) {
                counts.aggregates.undef_insert_chains += 1;
            }
        }
        _ => {}
    }
}

/// The op's own verifier verdict (its interfaces and its op rules; not
/// its regions, which the walk covers op by op), normalized by
/// [verifier_reason]. Also the whole-tree rule dialect-mir applies on top:
/// a `builtin.constant` may not produce a pointer carrier.
fn verifier_failure(ctx: &Context, op: Ptr<Operation>) -> Option<String> {
    let own = if op.deref(ctx).regions().next().is_none() {
        // A leaf op: pliron's whole check (attributes, operands,
        // successors, results, interfaces, the op's rules).
        op.deref(ctx).verify(ctx)
    } else {
        // A region-holding op (`mir.func`): its own rules only; the walk
        // verifies its body op by op.
        let op_obj = Operation::get_op_dyn(op, ctx);
        op_obj.verify_interfaces(ctx).and_then(|()| op_obj.verify(ctx))
    };
    if let Err(error) = own {
        return Some(verifier_reason(&error.err.to_string()));
    }
    if Operation::get_opid(op, ctx) == pliron::builtin::ops::ConstantOp::get_opid_static()
        && let Err(error) = dialect_mir::verification::verify_pointer_kind_producers(ctx, op)
    {
        return Some(verifier_reason(&error.err.to_string()));
    }
    None
}

/// A verifier message reduced to its rule: the type-specific tail
/// (`got: …`, `from X to Y`, `LHS: …`) cut off, so one rule is one
/// histogram key.
fn verifier_reason(message: &str) -> String {
    let line = message.lines().map(str::trim).find(|line| !line.is_empty()).unwrap_or("");
    let cut = [", got", " got:", " from ", ". LHS", ": LHS", ", found", " found "]
        .iter()
        .filter_map(|marker| line.find(marker))
        .min()
        .unwrap_or(line.len());
    // Backtick-quoted symbols (`attached MirCallOp \`sym\` has no …`) are
    // instance data, not part of the rule.
    let mut out = String::new();
    let mut quoted = false;
    for ch in line[..cut].chars() {
        match ch {
            '`' if !quoted => {
                quoted = true;
                out.push_str("`…`");
            }
            '`' => quoted = false,
            _ if quoted => {}
            _ => out.push(ch),
        }
    }
    out.trim_end_matches([' ', ':', ',']).to_string()
}

fn record_call(counts: &mut Counts, callee: &str) {
    let demangled = demangle(callee);
    *counts.call_families.entry(family(&demangled)).or_default() += 1;
    *counts.calls.entry(demangled).or_default() += 1;
}

/// Natural-loop count and maximum nesting depth of a function body, from
/// pliron-ll's CFG analyses (dialect-agnostic: they only read block
/// successors/predecessors). A loop's depth is one plus the number of
/// other loops whose body contains its header.
fn loop_metrics(ctx: &Context, region: Ptr<Region>) -> (usize, usize) {
    if region.deref(ctx).get_entry_block().is_none() {
        return (0, 0);
    }
    let dom = dominator_tree(ctx, region);
    let loops = natural_loops(ctx, &dom);
    let depth = loops
        .iter()
        .map(|l| {
            1 + loops
                .iter()
                .filter(|m| m.header != l.header && m.body.contains(&l.header))
                .count()
        })
        .max()
        .unwrap_or(0);
    (loops.len(), depth)
}

// ============================================================================
// Symbols: demangling and families
// ============================================================================

/// Demangle a Rust symbol (legacy `_ZN…E` or v0 `_R…`), keeping generic
/// arguments but dropping hashes; non-Rust symbols come back unchanged.
pub fn demangle(symbol: &str) -> String {
    match rustc_demangle::try_demangle(symbol) {
        Ok(demangled) => format!("{demangled:#}"),
        Err(_) => symbol.to_string(),
    }
}

/// The "family" of a demangled path: its leading two `::` segments, cut
/// before any generic arguments (`core::iter`, `alloc::vec`, `mycrate::m`).
/// For impl paths (`<Self as Trait>::f`, `<Self>::f`) the self type's path
/// is used when it has one, else the trait's; an inherent method on a
/// primitive, slice or pointer (`<*mut i32>::add`, `<str>::trim_matches`)
/// has neither and is its own family (cut before its generic arguments).
/// Non-Rust symbols (`memcpy`) are their own family.
pub fn family(demangled: &str) -> String {
    const SEGMENTS: usize = 2;
    let s = demangled.trim();
    if let Some(rest) = s.strip_prefix('<') {
        let inner = angle_prefix(rest);
        let (self_ty, trait_path) = match split_top_level(inner, " as ") {
            Some((a, b)) => (a, Some(b)),
            None => (inner, None),
        };
        if is_path(self_ty) {
            return leading_segments(self_ty, SEGMENTS);
        }
        if let Some(trait_path) = trait_path
            && is_path(trait_path)
        {
            return leading_segments(trait_path, SEGMENTS);
        }
        // Neither side has a path (an inherent method on a primitive, slice
        // or pointer): the item itself, cut before its own generic arguments.
        return cut_at_top_level_generics(s).to_string();
    }
    leading_segments(s, SEGMENTS)
}

/// `s` up to the first `::<` outside any `<...>` (the method-level generic
/// arguments of an impl item), or all of it.
fn cut_at_top_level_generics(s: &str) -> &str {
    let mut depth = 0usize;
    for (i, byte) in s.bytes().enumerate() {
        match byte {
            b'<' if depth == 0 && s[..i].ends_with("::") => return &s[..i - 2],
            b'<' => depth += 1,
            b'>' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    s
}

/// The text up to (not including) the `>` that closes an already-opened
/// `<`; all of `rest` if it never closes.
fn angle_prefix(rest: &str) -> &str {
    let mut depth = 0usize;
    for (i, ch) in rest.char_indices() {
        match ch {
            '<' => depth += 1,
            '>' if depth == 0 => return &rest[..i],
            '>' => depth -= 1,
            _ => {}
        }
    }
    rest
}

/// Split at the first occurrence of `sep` outside any `<...>`.
fn split_top_level<'a>(s: &'a str, sep: &str) -> Option<(&'a str, &'a str)> {
    let mut depth = 0usize;
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'<' => depth += 1,
            b'>' => depth = depth.saturating_sub(1),
            _ if depth == 0 && s[i..].starts_with(sep) => {
                return Some((&s[..i], &s[i + sep.len()..]));
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// An identifier followed by `::` — a path, as opposed to a primitive
/// (`u32`, `*mut T`, `[T]`, `&str`, `dyn Trait`, `fn(..)`).
fn is_path(t: &str) -> bool {
    let ident_len = t
        .char_indices()
        .find(|(_, c)| !(c.is_alphanumeric() || *c == '_'))
        .map(|(i, _)| i)
        .unwrap_or(t.len());
    ident_len > 0 && t[ident_len..].starts_with("::")
}

fn leading_segments(path: &str, n: usize) -> String {
    let cut = path.find('<').map(|i| &path[..i]).unwrap_or(path);
    let segments: Vec<&str> = cut
        .split("::")
        .filter(|seg| !seg.is_empty())
        .take(n)
        .collect();
    if segments.is_empty() {
        path.to_string()
    } else {
        segments.join("::")
    }
}

// ============================================================================
// Markdown rendering
// ============================================================================

impl Census {
    /// Render as Markdown: a summary table, the `top` op kinds and callee
    /// families, and a per-function table (functions by op count).
    pub fn to_markdown(&self, top: usize) -> String {
        let mut out = String::new();
        let title = match &self.crate_name {
            Some(name) => format!("{name} ({})", self.module),
            None => self.module.clone(),
        };
        let t = &self.totals;
        let c = &t.counts;
        out.push_str(&format!("# mir-census: {title}\n\n"));
        out.push_str("| metric | value |\n| --- | ---: |\n");
        let by_dialect: Vec<String> = t
            .ops_by_dialect
            .iter()
            .map(|(d, n)| format!("{d}: {n}"))
            .collect();
        let rows: Vec<(&str, String)> = vec![
            ("functions (`mir.func`)", t.functions.to_string()),
            ("declarations (`llvm.func`)", t.declarations.to_string()),
            ("blocks", c.blocks.to_string()),
            ("ops", c.ops.to_string()),
            ("ops by dialect", by_dialect.join(", ")),
            ("calls", c.total_calls().to_string()),
            ("allocas", c.allocas.to_string()),
            ("asserts (`mir.assert`)", c.asserts.to_string()),
            ("checked arithmetic", c.checked_arith.to_string()),
            (
                "aggregates: `mir.construct_*`",
                c.aggregates.construct.to_string(),
            ),
            (
                "aggregates: undef (mir+llvm)",
                c.aggregates.undef.to_string(),
            ),
            (
                "aggregates: insert_field / insert_value",
                c.aggregates.insert.to_string(),
            ),
            (
                "aggregates: undef→insert chains",
                c.aggregates.undef_insert_chains.to_string(),
            ),
            ("loops", c.loops.to_string()),
            ("max loop depth", c.max_loop_depth.to_string()),
            (
                "verifier failures (dialect-mir)",
                c.total_verifier_failures().to_string(),
            ),
        ];
        for (name, value) in rows {
            out.push_str(&format!("| {name} | {value} |\n"));
        }

        out.push_str(&format!(
            "\n## Op kinds (top {top})\n\n| op | count | share |\n| --- | ---: | ---: |\n"
        ));
        for (kind, count) in top_entries(&c.op_kinds, top) {
            let share = if c.ops == 0 {
                0.0
            } else {
                100.0 * *count as f64 / c.ops as f64
            };
            out.push_str(&format!("| `{kind}` | {count} | {share:.1}% |\n"));
        }

        if !c.verifier_failures.is_empty() {
            out.push_str("\n## Verifier failures\n\n| rule | count |\n| --- | ---: |\n");
            let mut failures: Vec<(&String, &usize)> = c.verifier_failures.iter().collect();
            failures.sort_by(|(ka, a), (kb, b)| b.cmp(a).then_with(|| ka.cmp(kb)));
            for (rule, count) in failures {
                out.push_str(&format!("| `{rule}` | {count} |\n"));
            }
        }

        out.push_str(&format!(
            "\n## Callee families (top {top})\n\n| family | calls |\n| --- | ---: |\n"
        ));
        for (fam, count) in top_entries(&c.call_families, top) {
            out.push_str(&format!("| `{fam}` | {count} |\n"));
        }

        out.push_str("\n## Functions\n\n| function | blocks | ops | calls | allocas | asserts | checked | construct | undef→insert | loops | depth |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
        let mut functions: Vec<(&String, &FunctionCensus)> = self.functions.iter().collect();
        functions
            .sort_by(|(sa, a), (sb, b)| b.counts.ops.cmp(&a.counts.ops).then_with(|| sa.cmp(sb)));
        for (_, f) in functions {
            let c = &f.counts;
            out.push_str(&format!(
                "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                f.demangled,
                c.blocks,
                c.ops,
                c.total_calls(),
                c.allocas,
                c.asserts,
                c.checked_arith,
                c.aggregates.construct,
                c.aggregates.undef_insert_chains,
                c.loops,
                c.max_loop_depth
            ));
        }
        out
    }
}

/// The `top` entries of a histogram by count (descending), ties by key.
fn top_entries(map: &BTreeMap<String, usize>, top: usize) -> Vec<(&String, &usize)> {
    let mut entries: Vec<(&String, &usize)> = map.iter().collect();
    entries.sort_by(|(ka, a), (kb, b)| b.cmp(a).then_with(|| ka.cmp(kb)));
    entries.truncate(top);
    entries
}

impl fmt::Display for Census {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_markdown(15))
    }
}

// ============================================================================
// The pass
// ============================================================================

/// The `mir-census` pass: see the module docs. Pure analysis; always
/// returns [unchanged].
pub struct MirCensusPass {
    label: Option<String>,
}

impl MirCensusPass {
    /// `label` is recorded as the census's `crate` field (rustc's crate
    /// name when the caller has it).
    pub fn new(label: Option<&str>) -> Self {
        MirCensusPass {
            label: label.map(str::to_string),
        }
    }
}

impl Default for MirCensusPass {
    fn default() -> Self {
        Self::new(None)
    }
}

/// The census output file from `CRABBIT_MIR_CENSUS` (unset or empty: none).
pub fn census_output_path() -> Option<PathBuf> {
    std::env::var_os(CENSUS_ENV)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

/// Append `census` as one JSON line to `path`. A single `write_all` of the
/// whole line in append mode, so concurrent rustc processes (cargo builds
/// several crates at once) do not interleave inside a record.
pub fn append_json_line(path: &std::path::Path, census: &Census) -> std::io::Result<()> {
    let mut line = serde_json::to_string(census).map_err(std::io::Error::other)?;
    line.push('\n');
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    file.write_all(line.as_bytes())
}

impl Pass for MirCensusPass {
    fn name(&self) -> &str {
        PASS_NAME
    }

    fn run(
        &mut self,
        root: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        if mir_disabled(GATE_NAME) {
            return Ok(unchanged());
        }
        let Some(path) = census_output_path() else {
            return Ok(unchanged());
        };
        // Never fail a compilation over a measurement: a panic inside the
        // walk (a malformed op, an analysis precondition) is caught and
        // reported, as is any I/O error.
        let label = self.label.as_deref();
        match catch_unwind(AssertUnwindSafe(|| census_labelled(&*ctx, root, label))) {
            Ok(census) => {
                if let Err(error) = append_json_line(&path, &census) {
                    eprintln!(
                        "{PASS_NAME}: could not append to {}: {error}",
                        path.display()
                    );
                }
            }
            Err(_) => eprintln!("{PASS_NAME}: census panicked; module skipped"),
        }
        Ok(unchanged())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dialect_mir::{
        attributes::FieldIndexAttr,
        ops::{
            MirAllocaOp, MirAssertOp, MirCallOp, MirCheckedAddOp, MirCondBranchOp, MirFuncOp,
            MirGotoOp, MirInsertFieldOp, MirLtOp, MirReturnOp, MirUndefOp,
        },
        types::{MirPtrType, MirTupleType, address_space},
    };
    use pliron::{
        basic_block::BasicBlock,
        builtin::{
            attributes::{StringAttr, TypeAttr},
            op_interfaces::{OneRegionInterface as _, OperandSegmentInterface as _},
            types::{FunctionType, IntegerType, Signedness},
        },
        op::Op as _,
        r#type::TypeHandle,
    };
    use pliron_ll::conversion::pass::Passes;

    // A legacy-mangled std symbol and a v0 one copied from a real crabbit
    // module (`<*mut i32>::add`, an inherent method on a primitive).
    const LEGACY_STD: &str = "_ZN4core3fmt9Arguments6new_v117h0123456789abcdefE";
    const V0_PTR_ADD: &str =
        "_RNvMNtNtCsbBDxv2Oq2Kj_4core3ptr7mut_ptrOl3addCsbCTi45toqUG_11seidel_mini";

    #[test]
    fn demangles_and_groups_into_families() {
        assert_eq!(demangle(LEGACY_STD), "core::fmt::Arguments::new_v1");
        assert_eq!(family(&demangle(LEGACY_STD)), "core::fmt");
        assert_eq!(demangle(V0_PTR_ADD), "<*mut i32>::add");
        assert_eq!(family(&demangle(V0_PTR_ADD)), "<*mut i32>::add");
        assert_eq!(demangle("memcpy"), "memcpy");
        assert_eq!(family("memcpy"), "memcpy");
        assert_eq!(
            family("core::ptr::drop_in_place::<alloc::vec::Vec<u8>>"),
            "core::ptr"
        );
        assert_eq!(
            family("<alloc::vec::Vec<i32> as core::ops::deref::Deref>::deref"),
            "alloc::vec"
        );
        assert_eq!(family("<u32 as core::fmt::Display>::fmt"), "core::fmt");
        assert_eq!(
            family(
                "core::iter::range::<impl core::iter::traits::iterator::Iterator for core::ops::range::Range<A>>::next"
            ),
            "core::iter"
        );
        assert_eq!(family("sudoku::main::{closure#0}"), "sudoku::main");
        assert_eq!(family("main"), "main");
        assert_eq!(
            family("<dyn core::fmt::Debug as core::fmt::Debug>::fmt"),
            "core::fmt"
        );
        assert_eq!(
            family("<str>::trim_matches::<<char>::is_whitespace>"),
            "<str>::trim_matches"
        );
        assert_eq!(
            family("<[u8]>::split_at_unchecked"),
            "<[u8]>::split_at_unchecked"
        );
        assert_eq!(
            family("<u8 as <[_]>::to_vec_in::ConvertVec>::to_vec::<alloc::alloc::Global>"),
            "<u8 as <[_]>::to_vec_in::ConvertVec>::to_vec"
        );
    }

    struct Builder {
        ctx: Context,
        i1: TypeHandle,
        i32: TypeHandle,
        module: Ptr<Operation>,
        module_body: Ptr<BasicBlock>,
    }

    impl Builder {
        fn new() -> Self {
            let mut ctx = Context::new();
            dialect_mir::register(&mut ctx);
            let i1: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
            let i32: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signed).into();
            let module_op = ModuleOp::new(&mut ctx, "rust_crate".try_into().unwrap());
            let module_body = module_op
                .get_region(&ctx)
                .deref(&ctx)
                .get_head()
                .expect("module has a body block");
            Builder {
                ctx,
                i1,
                i32,
                module: module_op.get_operation(),
                module_body,
            }
        }

        /// `fn name(i32) -> i32` with an entry block; returns (region, entry).
        fn func(&mut self, name: &str) -> (Ptr<Region>, Ptr<BasicBlock>) {
            let ctx = &mut self.ctx;
            let fn_ty = FunctionType::get(ctx, vec![self.i32], vec![self.i32]);
            let op = Operation::new(
                ctx,
                MirFuncOp::get_concrete_op_info(),
                vec![],
                vec![],
                vec![],
                1,
            );
            let func = MirFuncOp::new(ctx, op, TypeAttr::new(fn_ty.into()));
            func.set_symbol_name(ctx, name.try_into().unwrap());
            let region = op.deref(ctx).get_region(0);
            let entry = BasicBlock::new(ctx, Some("entry".try_into().unwrap()), vec![self.i32]);
            entry.insert_at_front(region, ctx);
            op.insert_at_back(self.module_body, ctx);
            (region, entry)
        }

        fn block(&mut self, region: Ptr<Region>, name: &str) -> Ptr<BasicBlock> {
            let block = BasicBlock::new(&mut self.ctx, Some(name.try_into().unwrap()), vec![]);
            block.insert_at_back(region, &self.ctx);
            block
        }

        fn op<O: pliron::op::Op>(
            &mut self,
            results: Vec<TypeHandle>,
            operands: Vec<pliron::value::Value>,
            succs: Vec<Ptr<BasicBlock>>,
            at: Ptr<BasicBlock>,
        ) -> Ptr<Operation> {
            let op = Operation::new(
                &mut self.ctx,
                O::get_concrete_op_info(),
                results,
                operands,
                succs,
                0,
            );
            op.insert_at_back(at, &self.ctx);
            op
        }

        fn arg0(&self, block: Ptr<BasicBlock>) -> pliron::value::Value {
            block.deref(&self.ctx).get_argument(0)
        }

        fn lt(&mut self, block: Ptr<BasicBlock>) -> pliron::value::Value {
            let a = self.arg0_of_func(block);
            let op = self.op::<MirLtOp>(vec![self.i1], vec![a, a], vec![], block);
            op.deref(&self.ctx).get_result(0)
        }

        /// The entry argument of the function containing `block`.
        fn arg0_of_func(&self, block: Ptr<BasicBlock>) -> pliron::value::Value {
            let region = block
                .deref(&self.ctx)
                .get_parent_region()
                .expect("block is in a region");
            let entry = region
                .deref(&self.ctx)
                .get_entry_block()
                .expect("function has an entry");
            self.arg0(entry)
        }

        fn cond_br(&mut self, block: Ptr<BasicBlock>, t: Ptr<BasicBlock>, f: Ptr<BasicBlock>) {
            let cond = self.lt(block);
            let (operands, sizes) =
                MirCondBranchOp::compute_segment_sizes(vec![vec![cond], vec![], vec![]]);
            let op = self.op::<MirCondBranchOp>(vec![], operands, vec![t, f], block);
            MirCondBranchOp::new(op).set_operand_segment_sizes(&self.ctx, sizes);
        }

        fn goto(&mut self, block: Ptr<BasicBlock>, dest: Ptr<BasicBlock>) {
            self.op::<MirGotoOp>(vec![], vec![], vec![dest], block);
        }

        fn ret(&mut self, block: Ptr<BasicBlock>) {
            let a = self.arg0_of_func(block);
            self.op::<MirReturnOp>(vec![], vec![a], vec![], block);
        }
    }

    /// `demo`: entry (alloca, mir.undef+insert_field, llvm.undef+insert_value,
    /// a call to a legacy-mangled std symbol, a call to a v0 one, checked
    /// add, assert) → header; header → (body | exit); body → header.
    /// `nested`: entry → h1; h1 → (h2 | exit); h2 → (h2 | h1).
    fn build() -> Builder {
        let mut b = Builder::new();
        let i32 = b.i32;
        let i1 = b.i1;

        let (region, entry) = b.func("demo");
        let header = b.block(region, "header");
        let body = b.block(region, "body");
        let exit = b.block(region, "exit");
        let a = b.arg0(entry);
        let ptr_ty: TypeHandle =
            MirPtrType::get(&mut b.ctx, i32, true, address_space::GENERIC).into();
        b.op::<MirAllocaOp>(vec![ptr_ty], vec![], vec![], entry);
        // mir.undef + mir.insert_field
        let tuple_ty: TypeHandle = MirTupleType::get(&mut b.ctx, vec![i32, i32]).into();
        let undef = MirUndefOp::new(&mut b.ctx, tuple_ty).get_operation();
        undef.insert_at_back(entry, &b.ctx);
        let undef_v = undef.deref(&b.ctx).get_result(0);
        let insert = b.op::<MirInsertFieldOp>(vec![tuple_ty], vec![undef_v, a], vec![], entry);
        MirInsertFieldOp::new(insert).set_attr_insert_index(&b.ctx, FieldIndexAttr(0));
        // llvm.undef + llvm.insert_value (the importer's spelling)
        let struct_ty: TypeHandle =
            pliron_llvm::types::StructType::get_unnamed(
                &b.ctx,
                (vec![i32], pliron_llvm::types::StructLayout::Unpacked),
            )
            .into();
        let lundef = pliron_llvm::ops::UndefOp::new(&mut b.ctx, struct_ty);
        lundef.get_operation().insert_at_back(entry, &b.ctx);
        let lundef_v = lundef.get_operation().deref(&b.ctx).get_result(0);
        let linsert = pliron_llvm::ops::InsertValueOp::new(&mut b.ctx, lundef_v, a, vec![0]);
        linsert.get_operation().insert_at_back(entry, &b.ctx);
        // a lone mir.undef that feeds nothing: counted as undef, not a chain
        MirUndefOp::new(&mut b.ctx, i32)
            .get_operation()
            .insert_at_back(entry, &b.ctx);
        // calls
        for callee in [LEGACY_STD, LEGACY_STD, V0_PTR_ADD] {
            let call = b.op::<MirCallOp>(vec![i32], vec![a], vec![], entry);
            MirCallOp::new(call).set_attr_callee(&b.ctx, StringAttr::new(callee.to_string()));
        }
        // checked add
        let checked_ty: TypeHandle = MirTupleType::get(&mut b.ctx, vec![i32, i1]).into();
        b.op::<MirCheckedAddOp>(vec![checked_ty], vec![a, a], vec![], entry);
        // assert, then goto header (dialect-mir >= 0.2.1 upstream: `mir.assert`
        // is an in-block check with no successors; the branch is a `mir.goto`)
        let cond = b.lt(entry);
        MirAssertOp::new(&mut b.ctx, cond)
            .get_operation()
            .insert_at_back(entry, &b.ctx);
        b.goto(entry, header);
        b.cond_br(header, body, exit);
        b.goto(body, header);
        b.ret(exit);

        let (region, entry) = b.func("nested");
        let h1 = b.block(region, "h1");
        let h2 = b.block(region, "h2");
        let exit = b.block(region, "exit");
        b.goto(entry, h1);
        b.cond_br(h1, h2, exit);
        b.cond_br(h2, h2, h1);
        b.ret(exit);
        b
    }

    #[test]
    fn census_counts_programmatic_module() {
        let b = build();
        let census = census_labelled(&b.ctx, b.module, Some("unit"));
        assert_eq!(census.crate_name.as_deref(), Some("unit"));
        assert_eq!(census.module, "rust_crate");
        assert_eq!(census.totals.functions, 2);
        assert_eq!(census.totals.declarations, 0);

        let demo = &census.functions["demo"].counts;
        assert_eq!(demo.blocks, 4);
        assert_eq!(demo.allocas, 1);
        assert_eq!(demo.asserts, 1);
        assert_eq!(demo.checked_arith, 1);
        assert_eq!(demo.aggregates.construct, 0);
        assert_eq!(demo.aggregates.undef, 3, "mir.undef x2 + llvm.undef");
        assert_eq!(
            demo.aggregates.insert, 2,
            "mir.insert_field + llvm.insert_value"
        );
        assert_eq!(demo.aggregates.undef_insert_chains, 2);
        assert_eq!(demo.op_kinds["mir.alloca"], 1);
        assert_eq!(demo.op_kinds["mir.insert_field"], 1);
        assert_eq!(demo.op_kinds["llvm.insert_value"], 1);
        assert_eq!(demo.op_kinds["mir.call"], 3);
        assert_eq!(demo.op_kinds["mir.lt"], 2, "assert cond + header cond");
        assert_eq!(demo.op_kinds["mir.cond_br"], 1);
        assert_eq!(demo.op_kinds["mir.goto"], 2, "assert's goto + body → header");
        assert_eq!(demo.op_kinds["mir.return"], 1);
        assert_eq!(demo.ops, demo.op_kinds.values().sum::<usize>());
        assert_eq!(demo.calls["core::fmt::Arguments::new_v1"], 2);
        assert_eq!(demo.calls["<*mut i32>::add"], 1);
        assert_eq!(demo.call_families["core::fmt"], 2);
        assert_eq!(demo.call_families["<*mut i32>::add"], 1);
        assert_eq!(demo.total_calls(), 3);
        assert_eq!((demo.loops, demo.max_loop_depth), (1, 1));

        let nested = &census.functions["nested"].counts;
        assert_eq!(nested.blocks, 4);
        assert_eq!((nested.loops, nested.max_loop_depth), (2, 2));
        assert_eq!(nested.total_calls(), 0);

        let totals = &census.totals.counts;
        assert_eq!(totals.blocks, 8);
        assert_eq!(totals.ops, demo.ops + nested.ops);
        assert_eq!(totals.loops, 3);
        assert_eq!(totals.max_loop_depth, 2);
        assert_eq!(totals.calls.len(), 2);
        assert_eq!(census.totals.ops_by_dialect["llvm"], 2);
        assert_eq!(census.totals.ops_by_dialect["mir"], totals.ops - 2);

        // JSON and Markdown round the same numbers.
        let json: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&census).unwrap()).unwrap();
        assert_eq!(json["crate"], "unit");
        assert_eq!(json["functions"]["demo"]["allocas"], 1);
        assert_eq!(
            json["functions"]["demo"]["aggregates"]["undef_insert_chains"],
            2
        );
        assert_eq!(json["totals"]["functions"], 2);
        assert_eq!(json["totals"]["ops_by_dialect"]["llvm"], 2);
        let md = census.to_string();
        assert!(md.contains("# mir-census: unit (rust_crate)"));
        assert!(md.contains("| `mir.call` | 3 |"));
        assert!(md.contains("| `core::fmt` | 2 |"));
        assert!(md.contains("| `nested` | 4 |"));
    }

    /// Without `CRABBIT_MIR_CENSUS` the pass is a no-op that reports the
    /// IR unchanged (the test never sets the variable, so this holds
    /// regardless of test ordering).
    #[test]
    fn pass_reports_unchanged_and_leaves_ir_alone() {
        let mut b = build();
        let before = census(&b.ctx, b.module);
        let mut passes = Passes::default();
        super::super::add_mir_passes(&mut passes);
        assert_eq!(passes.names(), vec![PASS_NAME.to_string()]);
        let mut analyses = AnalysisManager::default();
        let result = passes
            .run(b.module, &mut b.ctx, &mut analyses)
            .expect("census never fails");
        let _ = result;
        let mut single = MirCensusPass::default();
        let result = single
            .run(b.module, &mut b.ctx, &mut analyses)
            .expect("census never fails");
        assert!(matches!(
            result.ir_changed,
            pliron::irbuild::IRStatus::Unchanged
        ));
        assert_eq!(census(&b.ctx, b.module), before);
    }
}
