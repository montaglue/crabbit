# crabbit kernel ABI convention (corpus v1)

How a Rust-written GPU kernel reaches crabbit's native NVPTX path, and the
conventions the kernel corpus (`~/projects/montaglue/kernel-corpus`) follows.
This is the "minimal `#[no_mangle]` + tiny device-intrinsics shim" route the
kernel-corpus brief allows; cuda-oxide's own `#[kernel]` macro is NOT used
because it mangles kernels as `cuda_oxide_kernel_<hash>_…` and routes
intrinsics through `cuda_intrinsics::__cuda_oxide_intrinsic_abi_v1::iNNNN`
placeholder fns that only `rustc-codegen-cuda` rewrites; crabbit's importer
detects kernels by symbol prefix and needs the intrinsics as plain foreign
declarations.

## Kernel functions

```rust
#![no_std]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __stair_kernel_vector_add(a: *const i32, b: *const i32, out: *mut i32, n: u32) { … }
```

- Symbol prefix `__stair_kernel_` (`KERNEL_EXPORT_PREFIX` in
  `crates/crabbit/src/importer_oxide.rs`) marks a kernel: the importer puts
  its body into the `rust_kernels` module instead of the host module. The
  PTX `.entry` name is the symbol with the prefix STRIPPED (`vector_add`),
  so the three comparison arms (crabbit PTX, LLVM PTX, nvcc) share one
  entry name per kernel.
- Parameters: raw pointers (`*const T` / `*mut T`, T ∈ {i32,u32,i64,u64,f32,f64,u8}),
  and scalars `u32`/`i32`/`u64`/`i64`/`f32`/`f64`. Nothing else (no slices,
  references, structs). Pointer params are global-memory addresses.
- Kernels are leaf functions: no calls except the intrinsics below (device
  helper functions are a later crabbit feature). Use `macro_rules!` or
  write the code inline.
- No panics reachable: index with raw pointer arithmetic (`*a.add(i)`),
  never slices; build device crates with `-Coverflow-checks=off`
  (`overflow-checks = false` in the profile) and `panic = "abort"`.
- Host code must not call a kernel directly (crabbit errors on that).

## Intrinsics: the `stair-device` shim crate

`kernel-corpus/crates/stair-device` declares the NVVM intrinsics as foreign
functions under their real LLVM names:

```rust
unsafe extern "C" {
    #[link_name = "llvm.nvvm.read.ptx.sreg.tid.x"] pub fn tid_x() -> u32;
    #[link_name = "llvm.nvvm.barrier0"]            pub fn barrier0();
    …
}
```

rustc's symbol name for a foreign item is its `link_name`; pliron's
`Legaliser` turns the dots into underscores (`llvm_nvvm_read_ptx_sreg_tid_x`);
`crates/pliron-ll/src/nvptx` normalizes callee names (strips `llvm_`/`int_`,
dots→underscores) and lowers the sreg family to `mov.u32 %r, %tid.x`,
`barrier0` to `bar.sync 0`, etc. Because the names are the genuine LLVM
intrinsic names, the llvm-export route (`.ll` → `llc -march=nvptx64`) sees
`declare … @llvm.nvvm.read.ptx.sreg.tid.x()` once underscores are mapped
back to dots (the exporter/route-(ii) script owns that mapping).

Math on the device uses NVVM math intrinsics, never libm:
`llvm.nvvm.sqrt.rn.f`, `llvm.nvvm.rsqrt.approx.f`, `llvm.nvvm.ex2.approx.f`,
`llvm.nvvm.lg2.approx.f`, `llvm.nvvm.fmax.f`, `llvm.nvvm.fmin.f` (f32) and
the `.d` variants for f64 where they exist. exp(x) = ex2(x · log2 e).

## Shared memory

A `static mut` placed in link section `.shared` is a per-CTA shared array:

```rust
#[unsafe(link_section = ".shared")]
static mut TILE: [f32; 32 * 33] = [0.0; 32 * 33];
```

crabbit emits it as `.shared .align A .b8 TILE[bytes]` and materializes its
address with `mov.u64 %rd, TILE; cvta.shared.u64 %rd, %rd;` so all loads and
stores stay generic. Initializers must be all-zero (shared memory cannot be
statically initialized). Dynamic shared memory is not supported in v1.

## Value types on the device

i1 → `.pred`; i8/i16/i32 → `.b32` (sub-32-bit values kept zero-extended);
i64/ptr → `.b64`; f32 → `.f32`; f64 → `.f64`. i128 is not supported in
kernels.

## What the native translator supports (crates/pliron-ll/src/nvptx)

| area | supported | notes |
|---|---|---|
| integer arith | add sub mul sdiv udiv srem urem and or xor shl lshr ashr, icmp (all preds) | narrow types masked/sign-extended as needed |
| float arith | fadd fsub fmul fdiv (`.rn`), fneg, fcmp (all 16 preds) | **frem unsupported** (no PTX instruction) |
| casts | zext sext trunc bitcast inttoptr ptrtoint, sitofp uitofp fptosi fptoui (`cvt.rzi`), fpext fptrunc | |
| select | `selp` on b32/b64/f32/f64; pred select via and/or/not | |
| memory | load/store of i8/i16/i32/i64/ptr/f32/f64 through generic addresses; gep over ints/arrays (byte-strided) | **no struct GEPs, no allocas** (mem2reg must remove them; local-memory arrays unsupported) |
| globals | `.shared` statics (zero-init) and `.global` byte-initialized data (no pointers/relocs) | address via `mov.u64` + `cvta.{shared,global}` |
| control flow | br / cond_br with block args (edge blocks), return void, unreachable → `trap` | switch / indirectbr unsupported |
| sregs | tid/ntid/ctaid/nctaid .x/.y/.z, laneid, warpsize | |
| barriers | `llvm.nvvm.barrier0` → `bar.sync 0`; `membar.{gl,cta,sys}` | |
| atomics | `llvm.nvvm.atomic.{add,max,min,and,or,xor,exch}.gen.i`, `.add.gen.{f,ll,d}`, `.add.gen.i.{cta,sys}` | generic addressing (`atom.*`) |
| math f32 | sqrt.rn, sqrt.approx, rsqrt.approx, ex2.approx, lg2.approx, sin/cos.approx, rcp.approx, fabs, floor, ceil, round, trunc, fmax, fmin, fma.rn | via `llvm.nvvm.*` link_names (see shim) |
| math f64 | sqrt.rn, rsqrt.approx, fabs, floor, ceil, fmax, fmin, fma.rn | |
| calls | only the intrinsics above | **device function calls unsupported**: every helper must inline (the mid-end inlines module-internal calls; leftover internal definitions are skipped, a leftover call errors by name) |

Everything else errors with the op / callee name.

## LLVM IR export (rust-llvm arm)

`CRABBIT_LL_OUT=<path>` additionally writes the kernel module as textual
LLVM IR through cuda-oxide's `llvm-export` (`crates/crabbit/src/kernel_llvm_export.rs`):
kernels get the `ptx_kernel` calling convention, `llvm_nvvm_*` callees are
decoded back to `llvm.nvvm.*`, `.shared` statics become `addrspace(3)`
zero-initialized globals (with `addrspacecast` to generic at each use),
byte-initialized data becomes `addrspace(1)`. Compile with
`~/.local/opt/llvm18/bin/llc -march=nvptx64 -mcpu=sm_90 -O3` (LLVM 18 does
not know sm_121; its PTX ISA 7.8 output still assembles and runs on the
GB10). Caveat for apples-to-apples: the generic atomics (`llvm.nvvm.atomic.add.gen.i`
etc.) are cuda-oxide/crabbit names, not upstream LLVM intrinsics — kernels
using atomics will not go through llc until they are mapped to `atomicrmw`.

## Build and outputs

A device crate is an ordinary `no_std` `lib` crate compiled with
`-Zcodegen-backend=<libcrabbit.so>`. crabbit writes the host object as
usual and, when the crate contains kernels, a PTX sidecar next to it:
`<object stem>.ptx` in the same directory (i.e. under
`target/<profile>/deps/`), and additionally to the path in
`CRABBIT_PTX_OUT` when that variable is set (one file per crate; the
variable is read at rustc time, so pass it through `cargo rustc` env).
The PTX header defaults to `.target sm_121`, ISA 8.8 (`PtxTarget`);
`CRABBIT_PTX_SM=<n>` overrides the SM.

The host object of a device crate contains nothing but the shim's
declarations; it is never linked into anything.

Concretely (the acceptance fixture, `crates/backend-tests/fixtures/kernel-vector-add`):

```sh
cd crates/backend-tests/fixtures/kernel-vector-add
CRABBIT_PTX_OUT=/tmp/k.ptx CRABBIT_LL_OUT=/tmp/k.ll \
CARGO_TARGET_DIR=/tmp/kernel-target \
cargo rustc --lib --release -- -Zcodegen-backend=$CRABBIT/target/debug/libcrabbit.so -Coverflow-checks=off
ptxas -arch=sm_121 -v /tmp/k.ptx          # native arm
llc -march=nvptx64 -mcpu=sm_90 /tmp/k.ll  # rust-llvm arm
```

`CRABBIT_TRACE=1` keeps the per-pass kernel IR dumps in
`<object stem>.kernel-trace/` next to the object. Both the debug and the
release profile compile the fixture to PTX; the corpus uses `--release`
so the mid-end sees the same IR the CPU arms get.
