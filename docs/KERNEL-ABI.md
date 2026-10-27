# crabbit kernel ABI convention (corpus v2)

How a Rust-written GPU kernel reaches crabbit's native NVPTX path, and the
conventions the kernel corpus (`~/projects/montaglue/kernel-corpus`) follows.
Kernels are written against cuda-oxide's device API: the `#[kernel]`
attribute from `cuda-macros` and the `cuda_device` crate (`thread`, `warp`,
`float`, `atomic`, `SharedArray`, `DynamicSharedArray`), both consumed from
NVIDIA's upstream repository at the revision crabbit pins. A kernel crate
therefore also compiles with cuda-oxide's own compiler.

## Kernel functions

```rust
#![no_std]
use cuda_device::{kernel, thread};

#[kernel]
pub fn vector_add(a: *const i32, b: *const i32, out: *mut i32, n: u32) {
    let i = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
    if i < n {
        unsafe { *out.add(i as usize) = *a.add(i as usize) + *b.add(i as usize) };
    }
}
```

- `#[kernel]` renames the function to cuda-oxide's reserved symbol
  (`KERNEL_PREFIX` from `reserved-oxide-symbols`, currently
  `cuda_oxide_codegen_v1_cuda_oxide_kernel_<hash>_<name>`). The importer
  detects kernels by that prefix and puts their bodies into the
  `rust_kernels` module. The PTX `.entry` name is the symbol with the prefix
  STRIPPED (`vector_add`), so the three comparison arms (crabbit PTX, LLVM
  PTX, nvcc) share one entry name per kernel. `pub unsafe fn` is accepted
  too.
- Device crates set `default-features = false` on `cuda-device` and
  `cuda-macros`: the `host` feature would pull in `cuda-host`, and the host
  side of a crabbit device crate is never linked.
- Parameters: raw pointers (`*const T` / `*mut T`, T ∈
  {i32,u32,i64,u64,f32,f64,u8}), scalars `u32`/`i32`/`u64`/`i64`/`f32`/`f64`,
  and by-value `#[repr(C)]` structs of those (nested structs/arrays
  included). `&[T]` and `DisjointSlice<T>` are accepted and are passed as
  ONE by-value `{ptr, len}` struct parameter each (16 bytes, align 8).
  Pointer params are global-memory addresses. A struct param is one launch
  argument: crabbit emits `.param .align A .b8 name[size]` in the C layout
  and loads the fields with `ld.param`, so the host passes a pointer to the
  C struct in `kernelParams[i]` exactly as it would for an nvcc kernel.
  Kernel parameters are never flattened or passed indirectly. The corpus
  keeps raw pointers plus scalars so the Rust, CUDA C++ and LLVM arms share
  one launch signature; cuda-oxide's own host launchers (which flatten
  slices) are not used.
- Calls: the `cuda_device` functions below always work. Internal Rust
  helpers are inlined by the mid-end; one that survives (e.g. recursion)
  becomes a PTX `.func` with the `.param` ABI (scalar params/result only).
- No panics reachable: index with raw pointer arithmetic (`*a.add(i)`) or
  checked `cuda_device` accessors; build device crates with
  `-Coverflow-checks=off` (`overflow-checks = false` in the profile) and
  `panic = "abort"`.
- Host code must not call a kernel directly (crabbit errors on that).

## The device API the importer understands

cuda-device's functions have placeholder bodies (`unreachable!`) that the
compiler replaces. crabbit's importer intercepts the calls by their
definition (`crates/crabbit/src/importer/cuda_device.rs`) and builds
cuda-oxide's `nvvm` dialect ops, which `mir-lower` lowers exactly as it
lowers cuda-oxide's own output; anything under `cuda_device::` that is not
listed is ordinary Rust and imports as a call.

| cuda_device | op | PTX |
|---|---|---|
| `thread::{threadIdx,blockDim,blockIdx,gridDim}_{x,y,z}()` | `nvvm.read_ptx_sreg_*` | `mov.u32 %r, %tid.x` … |
| `thread::sync_threads()` | `nvvm.barrier0` | `bar.sync 0` |
| `warp::sync_mask(m)`, `warp::lane_id()` | `nvvm.bar_warp_sync`, `nvvm.read_ptx_sreg_laneid` | `bar.warp.sync`, `%laneid` |
| `warp::shuffle{,_up,_down,_xor}{,_f32}_sync(mask, v, lane)` | `nvvm.shfl_sync_{idx,up,down,bfly}_{i32,f32}` | `shfl.sync.*.b32` (full-warp segment) |
| `warp::{ballot,all,any}_sync(mask, pred)` | `nvvm.vote_sync_*` | `vote.sync.*` |
| `float::{ex2_approx,lg2_approx,rsqrt_approx,rcp_approx,sin_approx,cos_approx,sqrt_rn,sqrt_approx}_f32(x)` | `nvvm.scalar_math` | `ex2.approx.f32` … |
| `float::fma_rn_f32(a, b, c)` | placeholder `fmaf32` | `fma.rn.f32` |
| `f32::max/min/sqrt/mul_add/abs/floor/ceil/trunc/round`, `exp2`, `log2` (core intrinsics, in kernels) | dialect-mir placeholders `__cuda_oxide_rust_intrinsic_*` | `max.f32`, `sqrt.rn.f32`, `fma.rn.f32`, … (`mir-lower` names libdevice `__nv_*`; the translator lowers the single-instruction ones) |
| `DeviceAtomic{I32,U32,I64,U64}::from_ptr(p).fetch_{add,sub,and,or,xor,min,max}(v, ordering)`, `.swap` | `nvvm.atomic_rmw` (device scope) | `atom.*` |
| `SharedArray::{as_ptr,as_mut_ptr,as_raw_mut_ptr}`, indexing | address of the `.shared` global | — |
| `DynamicSharedArray::<T>::{get,get_raw,offset}` | crabbit's `crabbit.dyn.shared.base` | `.extern .shared` window |

Intrinsics not in this table are added here when a kernel needs them (the
dialect has ops for all of cuda-device; the translator has to lower each).

## Shared memory

A `static mut` of type `SharedArray<T, N>` is a per-CTA shared array
(cuda-device's type is a zero-sized marker; the importer gives it `N × T`
bytes of storage in the `.shared` section, aligned to `max(align_of::<T>(),
ALIGN)`):

```rust
use cuda_device::SharedArray;

#[kernel]
pub fn softmax(x: *const f32, y: *mut f32, rows: u32, d: u32) {
    static mut RED: SharedArray<f32, 256> = SharedArray::UNINIT;
    let sh = unsafe { SharedArray::as_raw_mut_ptr(&raw mut RED) };
    …
}
```

crabbit emits it as `.shared .align A .b8 NAME[bytes]` and materializes its
address with `mov.u64 %rd, NAME; cvta.shared.u64 %rd, %rd;` so all loads and
stores stay generic; accesses the translator can prove shared become
`ld.shared`/`st.shared`. Shared memory is never statically initialized.

### Dynamic shared memory (`extern __shared__`)

`DynamicSharedArray::<T>::get()` (or `get_raw()` for the `*mut u8` base,
`offset(bytes)` for a partition) yields the base of the module's single
`.extern .shared .align 16 .b8 __crabbit_dyn_shared[];` declaration, whose
size is the launch's `sharedMemBytes` (base alignment 16). The result is a
proven-`.shared` source for provenance. Every call in a kernel returns the
SAME window (PTX has one extern shared array per module) — partition it
manually. Launch side: corpus specs set `dynamic_shared_bytes` (an
expression over the size variables, default 0) and `tools/runner.py`
passes it as `cuLaunchKernel`'s sharedMemBytes; the old `shared_bytes`
spec field remains documentation of static usage. The window is a crabbit
convention (`crabbit.dyn.shared.base`), so kernels using it do not go
through the llvm-export (rust-llvm) arm.

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
| memory | load/store of i8/i16/i32/i64/ptr/f32/f64 (space-qualified when provenance proves it); gep over ints/arrays/structs (natural layout) | allocas that survive mem2reg lower to `.local` arrays (`cvta.local`) |
| globals | `.shared` statics (zero-init) and `.global` byte-initialized data (no pointers/relocs) | address via `mov.u64` + `cvta.{shared,global}` |
| control flow | br / cond_br with block args (edge blocks), return void, unreachable → `trap` | switch / indirectbr unsupported |
| sregs | tid/ntid/ctaid/nctaid .x/.y/.z, laneid, warpsize | |
| barriers | `llvm.nvvm.barrier0` and `llvm.nvvm.barrier.cta.sync.aligned.all` → `bar.sync 0`; `llvm.nvvm.bar.warp.sync` → `bar.warp.sync mask`; `membar.{gl,cta,sys}` | |
| warp shuffle | `llvm.nvvm.shfl.sync.{down,up,bfly,idx}.{i32,f32}` → `shfl.sync.<mode>.b32` | NVVM operand order `(mask, value, b, c)` |
| warp vote | `llvm.nvvm.vote.{ballot,all,any}.sync` → `vote.sync.*` | |
| dynamic shared | `crabbit.dyn.shared.base` → `.extern .shared` window base | crabbit-native arm only |
| struct params | by-value aggregate kernel params as one `.param .align A .b8 name[size]` | leaves via `ld.param` at natural-layout offsets |
| atomics | `llvm.nvvm.atomic.{add,max,min,and,or,xor,exch}.gen.i`, `.add.gen.{f,ll,d}`, `.add.gen.i.{cta,sys}` and the `atomicrmw` forms `mir-lower` emits for `nvvm.atomic_rmw` | generic addressing (`atom.*`) |
| math f32 | sqrt.rn, sqrt.approx, rsqrt.approx, ex2.approx, lg2.approx, sin/cos.approx, rcp.approx, fabs, floor, ceil, round, trunc, fmax, fmin, fma.rn | via `llvm.nvvm.*`, LLVM generic (`llvm.maxnum.f32`) or the single-instruction libdevice names (`__nv_fmaf`, `__nv_fmaxf`, …) |
| math f64 | sqrt.rn, rsqrt.approx, fabs, floor, ceil, fmax, fmin, fma.rn | |
| calls | the intrinsics above, plus internal device helpers that survive the inliner (emitted as PTX `.func`s with the `.param` ABI, e.g. recursion) | scalar params/result only on `.func`s; an unknown callee errors by name |

Everything else errors with the op / callee name.

## LLVM IR export (rust-llvm arm)

`CRABBIT_LL_OUT=<path>` additionally writes the kernel module as textual
LLVM IR through cuda-oxide's `llvm-export` (`crates/crabbit/src/kernel_llvm_export.rs`):
kernels get the `ptx_kernel` calling convention, `.shared` statics become
`addrspace(3)` zero-initialized globals (with `addrspacecast` to generic at
each use), byte-initialized data becomes `addrspace(1)`, and the CTA
barrier is spelled `llvm.nvvm.barrier0` for LLVM 18. Compile with
`~/.local/opt/llvm18/bin/llc -march=nvptx64 -mcpu=sm_90 -O3` (LLVM 18 does
not know sm_121; its PTX ISA 7.8 output still assembles and runs on the
GB10).

## Build and outputs

A device crate is an ordinary `no_std` `lib` crate compiled with
`-Zcodegen-backend=<libcrabbit.so>` on crabbit's toolchain (stable 1.99.1
with `RUSTC_BOOTSTRAP=1`). crabbit writes the host object as usual and,
when the crate contains kernels, a PTX sidecar next to it: `<object
stem>.ptx` in the same directory (under `target/<profile>/deps/`), and
additionally to the path in `CRABBIT_PTX_OUT` when that variable is set
(one file per crate; read at rustc time, so pass it through `cargo rustc`
env). The PTX header defaults to `.target sm_121`, ISA 8.8 (`PtxTarget`);
`CRABBIT_PTX_SM=<n>` overrides the SM.

The host object of a device crate contains the host copies of the kernel
helpers (with cuda-device's placeholder bodies); it is never linked into
anything.

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
