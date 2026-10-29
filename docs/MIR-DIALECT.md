# The `mir` dialect: cuda-oxide's typing contract

crabbit's front end emits NVIDIA's `mir` dialect — `dialect-mir` from
[NVlabs/cuda-oxide](https://github.com/NVlabs/cuda-oxide) — and hands the
module to cuda-oxide's `mir-lower` for the lowering to the LLVM dialect.
Both crates (plus `llvm-export`, the textual LLVM IR writer of the kernel
comparison arm) are plain git dependencies on the upstream repository at
one pinned revision (`Cargo.toml`): no fork, no patches, no vendored copy.
What the lowering lacks for crabbit's module is done by crabbit passes
around it (`crates/mir/src/passes/`) or proposed upstream.

## The contract

`mir_lower::lower_mir_to_llvm` verifies the whole module before it lowers
anything, with no opt-out: every op's own rules plus the whole-tree rule
that a `builtin.constant` may not produce a pointer. The rules the importer
meets (dialect-mir `src/ops/*.rs`, `src/types.rs`):

- Pointers are `mir.ptr<pointee, mutable, address space, kind>`; the kind
  is the pointer's Rust provenance (`Erased`, `SharedRef`, `UniqueRef`,
  `RawConst`, `RawMut`) and part of the type.
- `mir.load` / `mir.store` operands point at exactly the loaded / stored
  type; `mir.alloca` yields a mutable, generic, `Erased` pointer to its slot
  type; `mir.ptr_offset` steps in units of its pointee and keeps pointee,
  kind, mutability and address space.
- `mir.cast` carries rustc's cast kind. Pointer retypes (`PtrToPtr`) may
  keep or forget a kind, never establish one; establishing a kind
  (an integer becoming a raw pointer) needs a `pointer_kind_authority`:
  `RustCast` for a Rust cast, `StaticAddress` for the address of static
  storage.
- `mir.call` to a `mir.func` of the module matches its signature exactly;
  a call to any other symbol carries `external_callee_type` (a builtin
  function type) under the `AbiBoundary` authority.
- The dialect has no host-generic global address producer
  (`mir.global_alloc` is device global/constant memory only), no
  thread-locals, no indirect call form, and function pointers are stand-in
  tokens (cuda-oxide's own importer hashes the symbol).

## How the importer meets it

`crates/crabbit/src/importer/` keeps its byte-addressed, opaque-pointer
walk (every thin pointer is `mir.ptr<u8>`, `Erased`; field and element
addresses are byte offsets through `mir.ptr_offset`; aggregates are LLVM
struct types built with llvm-dialect `undef`/`insert_value`/`extract_value`)
and a typing pass at the end of the import (`importer/typing.rs`) produces
the verifier's form:

- loads, stores, offsets, call arguments and returns get their pointee
  through a representation-only `mir.cast PtrToPtr` (same kind, mutability,
  address space), which `mir-lower` lowers to nothing;
- integer → pointer casts of the walk are `RustCast`-authorized raw
  pointers; calls to `llvm.func` declarations get their `AbiBoundary`
  signature from their own operand and result types;
- the address of a host global or thread-local stays an `llvm.addressof`
  and crosses into the dialect as `llvm.ptrtoint` + `mir.cast
  PointerWithExposedProvenance` (`StaticAddress`); the reverse crossing
  (into indirect `llvm.call`s and pointer compares) is `PointerExposeAddress`
  + `llvm.inttoptr`;
- the address of a function of this module is `ll.func_addr` (crabbit's
  `ll` dialect), since an `llvm.addressof` may not name a `mir.func`;
  `materialize-function-addresses` turns it into the `llvm.addressof`
  right after the lowering.

`lower-dialect-mir` erases, right after the lowering, the MIR types left
inside the llvm aggregate ops the lowering passes through (`mir.ptr`,
signed and unsigned integers, named structs), with `mir-lower`'s own type
mapping. Two crabbit passes then complete the module:
`apply-module-linkage` (the importer records each function's linkage in
the module attribute `crabbit_linkage`; the lowering builds `llvm.func`s
fresh and keeps no custom attributes, while the `builtin.module` op and its
attributes survive) and `materialize-function-addresses`.

## Measuring the distance

The `mir-census` pass (`CRABBIT_MIR_CENSUS=<file>`, `docs/MIR-CENSUS.md`)
verifies every op on its own and records a per-rule histogram
(`verifier_failures`), so the distance to the contract is a number per
module, not the first error `mir-lower` stops at. On 2026-10-27 it went
from 1993 of 21489 ops (sudoku-solver, release; thin pointers typed
`llvm.ptr`, untyped foreign calls, signed `llvm.lshr`) to 0 with the typing
pass; `hello-world-aarch64` and `sudoku-solver` compile and run through the
verified lowering.

## Known gaps to raise upstream

- A host-generic address space for `mir.global_alloc` (or a host global
  producer) and thread-local storage would remove the exposed-provenance
  bridge for global addresses.
- An indirect `mir.call` form and real function addresses would remove
  `ll.func_addr` and the `llvm.call` bridge.
