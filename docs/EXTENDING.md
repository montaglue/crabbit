# Extending crabbit

crabbit's pipelines are data. Every pass has a name, and every pipeline
stage is an order of names:

| stage | side | default order |
|---|---|---|
| `MirToLlvm` | host, device | `crabbit_mir::passes::DEFAULT_ORDER` — `mir-census`, `lower-dialect-mir`, and the passes completing the lowering |
| `Midend` | host, device | `pliron_ll::passes::llvm::DEFAULT_ORDER` — inline … gvn, licm, unroll, dse, adce, sink, vectorize |
| `Machine` | host | the host backend's order, e.g. `pliron_ll::passes::aarch64::default_order(os)` — isel … `aarch64-register-allocate` … encode |

(The device side stops after the mid-end: PTX is written from its output
and ptxas does the machine work.)

An extension adds passes and decides orders through crabbit-config
(`crates/config`):

- `register_pass(name, factory)` — a pass under a name. The factory gets
  the `CrabbitConfig` (host target, device target, crate name) and the
  `Side` it builds for. A name crabbit already uses shadows crabbit's pass.
- `register_order(f)` — one function deciding every stage's order. It
  receives crabbit's default and returns its own (`Some`) or keeps the
  default (`None`): insert, remove, reorder, or replace passes, crabbit's
  own included. Replacing `aarch64-register-allocate` swaps the register
  allocator (a replacement must meet its post-conditions, see
  `pliron_ll::passes::aarch64::default_order`).

Passes are [pliron](https://github.com/pliron-org/pliron) passes: implement
`pliron::pass::Pass` (re-exported as `pliron_ll::conversion::pass::Pass`).
A pass reads its own settings — `CRABBIT_*` environment variables are the
convention — and a factory reports a bad one as an error.

## The backend dylib

rustc loads one codegen backend per compilation
(`-Zcodegen-backend=<dylib>`), so extensions run from a backend dylib of
their own: a crate that links crabbit's library statically, registers, and
hands rustc crabbit's backend. crabbit's own `libcrabbit.so`
(`crates/crabbit-backend`) is exactly this with nothing registered.

`Cargo.toml` (crabbit checked out next to your crate; a pinned git
dependency works the same way):

```toml
[package]
name = "my-crabbit"
version = "0.1.0"
edition = "2024"

[lib]
crate-type = ["dylib"]

[dependencies]
crabbit-core = { path = "../crabbit/crates/crabbit" }
crabbit-config = { path = "../crabbit/crates/config" }
pliron-ll = { path = "../crabbit/crates/pliron-ll" }
pliron = "0.20.0"   # exactly the version crabbit resolves
```

Copy crabbit's `rust-toolchain.toml` (the crabbit library uses rustc's
private crates, so the toolchain needs `rustc-dev`) and set
`RUSTC_BOOTSTRAP=1` in `.cargo/config.toml` (`[env]`), as crabbit does.

`src/lib.rs` — a pass that counts the functions of each module, placed
right before `llvm-unroll` in kernel compiles, and an order that drops
crabbit's `llvm-licm` everywhere:

```rust
#![feature(rustc_private)]

extern crate rustc_codegen_ssa;
extern crate rustc_driver;

use crabbit_config::{CrabbitConfig, Side, Stage, register_order, register_pass};
use pliron::{context::{Context, Ptr}, operation::Operation};
use pliron_ll::conversion::pass::{AnalysisManager, Pass, PassResult, unchanged};
use rustc_codegen_ssa::traits::CodegenBackend;

struct CountFunctions;

impl Pass for CountFunctions {
    fn name(&self) -> &str {
        "count-functions"
    }

    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> pliron::result::Result<PassResult> {
        use pliron::linked_list::ContainsLinkedList;
        let region = op.deref(ctx).get_region(0);
        let block = region.deref(ctx).get_head().expect("a module has one block");
        let count = block.deref(ctx).iter(ctx).count();
        eprintln!("count-functions: {count} top-level ops");
        Ok(unchanged())
    }
}

fn count_functions(_: &CrabbitConfig, _: Side) -> Result<Box<dyn Pass>, String> {
    Ok(Box::new(CountFunctions))
}

fn order(
    stage: Stage,
    side: Side,
    _: &CrabbitConfig,
    default: &[&'static str],
) -> Result<Option<Vec<&'static str>>, String> {
    if stage != Stage::Midend {
        return Ok(None);
    }
    let mut order: Vec<_> = default.iter().copied().filter(|name| *name != "llvm-licm").collect();
    if side == Side::Device {
        let unroll = order.iter().position(|name| *name == "llvm-unroll").ok_or("no llvm-unroll")?;
        order.insert(unroll, "count-functions");
    }
    Ok(Some(order))
}

#[unsafe(no_mangle)]
pub fn __rustc_codegen_backend() -> Box<dyn CodegenBackend> {
    register_pass("count-functions", count_functions);
    register_order(order);
    crabbit_core::create_backend()
}
```

Build it (`cargo build` → `target/debug/libmy_crabbit.so`) and compile any
crate with it:

```sh
cargo rustc --release -- -Zcodegen-backend=$PWD/target/debug/libmy_crabbit.so -Coverflow-checks=off
```

## Notes

- **One Rust dylib per process.** crabbit's library is an rlib absorbed by
  your dylib; do not also load `libcrabbit.so`. Several extensions in one
  compilation means one backend crate that registers all of them and
  combines their order edits in its one order function.
- **One pliron.** Your crate must resolve the same `pliron` (and the same
  crabbit source) as the crabbit it links, or the types do not unify.
- **No stable API.** Extensions use crabbit's crates directly (passes,
  dialects, analyses); build against a pinned crabbit revision.
- **Testing.** `CRABBIT_TEST_BACKEND=<your dylib> cargo test -p backend-tests
  --test backend_smoke` (in the crabbit checkout) compiles and runs
  crabbit's fixture programs through your backend.
- **The analysis server** (`crabbit-analysisd`) builds pipelines through
  the same API but sees only what its own process registered; the public
  binary registers nothing.
