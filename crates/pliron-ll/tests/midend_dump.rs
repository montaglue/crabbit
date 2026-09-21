//! Developer aid for the vectorizer work: parse a `CRABBIT_EMIT_IR` module,
//! run the shared LLVM mid-end on it, and print one function's post-midend
//! IR. Inert unless `CRABBIT_MIDEND_DUMP_INPUT` is set:
//!
//! ```sh
//! CRABBIT_MIDEND_DUMP_INPUT=/path/to/module.plir \
//! CRABBIT_MIDEND_DUMP_FUNC=kernel \
//!     cargo test -p pliron-ll --test midend_dump -- --nocapture
//! ```

use pliron::combine::Parser;
use pliron::context::Context;
use pliron::location::Source;
use pliron::operation::{Operation, OperationParserConfig};
use pliron::parsable::{Parsable, State, state_stream_from_iterator};
use pliron::printable::Printable;

#[test]
fn dump_post_midend_ir() {
    let Ok(path) = std::env::var("CRABBIT_MIDEND_DUMP_INPUT") else {
        return;
    };
    let filter = std::env::var("CRABBIT_MIDEND_DUMP_FUNC").unwrap_or_default();
    let content = std::fs::read_to_string(&path).expect("read input module");
    let mut ctx = Context::new();
    let state = State::new(&mut ctx, Source::InMemory);
    let stream = state_stream_from_iterator(content.chars(), state);
    let root = <Operation as Parsable>::parser(OperationParserConfig {
        look_for_outlined_attrs: false,
    })
    .parse(stream)
    .expect("parse input module")
    .0;

    // `CRABBIT_MIDEND_DUMP_PROFILE=gpu` dumps under the kernel pipeline's
    // profile (divergent target, no SIMD) instead of the host CPU's.
    let profile = match std::env::var("CRABBIT_MIDEND_DUMP_PROFILE").as_deref() {
        Ok("gpu") => pliron_ll::target_profile::TargetProfile::gpu_kernel(),
        _ => pliron_ll::target_profile::TargetProfile::host_cpu().with_simd128(true),
    };
    let mut passes = pliron_ll::conversion::pass::Passes::default();
    pliron_ll::passes::llvm::add_llvm_midend_passes(&mut passes, &profile);
    // Run the vectorizer AFTER the shared mid-end, at its intended pipeline
    // position, so its bail/transform decisions are visible here too.
    passes.add_pass(pliron_ll::passes::llvm::vectorize::LLVMVectorizePass::new(
        &profile,
    ));
    passes
        .run(
            root,
            &mut ctx,
            &mut pliron_ll::conversion::pass::AnalysisManager::default(),
        )
        .expect("midend");

    use pliron::builtin::op_interfaces::SymbolOpInterface;
    use pliron::linked_list::ContainsLinkedList;
    let module = pliron::op::Op::get_operation(
        &*Operation::get_op_dyn(root, &ctx),
    );
    let body = module
        .deref(&ctx)
        .get_region(0)
        .deref(&ctx)
        .get_head()
        .unwrap();
    let state = pliron::printable::State::default();
    for op in body.deref(&ctx).iter(&ctx) {
        let op_dyn = Operation::get_op_dyn(op, &ctx);
        if let Some(func) = op_dyn.downcast_ref::<pliron_llvm::ops::FuncOp>() {
            let name = func.get_symbol_name(&ctx).to_string();
            if filter.is_empty() || name.contains(&filter) {
                println!("==== {name}\n{}", op.print(&ctx, &state));
            }
        }
    }
}
