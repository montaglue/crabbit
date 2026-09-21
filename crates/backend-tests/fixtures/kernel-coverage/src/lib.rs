//! Coverage fixtures for crabbit's native NVPTX path: the kernel shapes the
//! translator gained after the corpus v1 leaf-function subset —
//! struct-typed GEPs through pointer params, allocas that survive mem2reg
//! (dynamically indexed local arrays → `.local`), device-function
//! calls that survive the inliner (recursion → a real PTX `.func`),
//! warp primitives (shfl.sync / vote.sync / bar.warp.sync), dynamic shared
//! memory, and by-value struct kernel parameters.
#![no_std]
#![allow(unsafe_op_in_unsafe_fn)]

use crabbit_device::{FULL_MASK, SHFL_C_DOWN, SHFL_C_UP, raw};

/// A field-addressed pair; `repr(C)` so device and host layouts agree.
#[repr(C)]
pub struct Pair {
    pub a: i32,
    pub b: i32,
}

/// out[i] = Pair { a: p[i].b, b: p[i].a + p[i].b } — struct GEPs on both
/// the load and store sides.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __crabbit_kernel_pair_swap_sum(
    p: *const Pair,
    out: *mut Pair,
    n: u32,
) {
    let i = raw::ctaid_x() * raw::ntid_x() + raw::tid_x();
    if i < n {
        let src = &*p.add(i as usize);
        let dst = &mut *out.add(i as usize);
        dst.a = src.b;
        dst.b = src.a + src.b;
    }
}

/// out[i] = table[x[i] & 7] where table[j] = x[i] * j + j; the dynamic
/// read index keeps the table out of mem2reg's reach, so it lowers to a
/// `.local` array.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __crabbit_kernel_local_table(x: *const u32, out: *mut u32, n: u32) {
    let i = raw::ctaid_x() * raw::ntid_x() + raw::tid_x();
    if i < n {
        let v = *x.add(i as usize);
        let mut table = [0u32; 8];
        let mut j = 0u32;
        while j < 8 {
            table[j as usize] = v.wrapping_mul(j).wrapping_add(j);
            j += 1;
        }
        *out.add(i as usize) = table[(v & 7) as usize];
    }
}

/// Triangular number by recursion: the self-call can never be fully
/// inlined, so a device `.func` (with a real `.param`-ABI call) survives.
fn tri(n: u32) -> u32 {
    if n == 0 { 0 } else { n + tri(n - 1) }
}

/// out[i] = tri(x[i] & 15).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __crabbit_kernel_tri_rec(x: *const u32, out: *mut u32, n: u32) {
    let i = raw::ctaid_x() * raw::ntid_x() + raw::tid_x();
    if i < n {
        *out.add(i as usize) = tri(*x.add(i as usize) & 15);
    }
}

/// Per-warp tree reduction over 32 consecutive f32s via `shfl.down`:
/// lane 0 writes its warp's sum to out[warp]. The launch covers exactly
/// 32*n_warps threads, so every shuffle runs with a full warp. The host
/// check replays the identical pairing tree (offsets 16, 8, 4, 2, 1), so
/// the f32 result is bit-exact, verifying shuffle SEMANTICS end to end.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __crabbit_kernel_warp_reduce(
    x: *const f32,
    out: *mut f32,
    n_warps: u32,
) {
    let i = raw::ctaid_x() * raw::ntid_x() + raw::tid_x();
    let warp = i / 32;
    let lane = i % 32;
    if warp < n_warps {
        let mut v = *x.add(i as usize);
        let mut off = 16u32;
        while off > 0 {
            v += raw::shfl_down_sync_f32(FULL_MASK, v, off, SHFL_C_DOWN);
            off >>= 1;
        }
        if lane == 0 {
            *out.add(warp as usize) = v;
        }
    }
}

/// The remaining warp primitives, one output buffer each: bfly (xor lane 1),
/// idx (broadcast lane 5), up (delta 1; low lanes keep their own value),
/// ballot of "x is odd", and vote all/any packed as (all << 1) | any —
/// with a `bar.warp.sync` between the shuffles and the votes. n is a
/// multiple of 32 so every warp is full.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __crabbit_kernel_warp_ops(
    x: *const u32,
    out_bfly: *mut u32,
    out_idx: *mut u32,
    out_up: *mut u32,
    out_ballot: *mut u32,
    out_vote: *mut u32,
    n: u32,
) {
    let i = raw::ctaid_x() * raw::ntid_x() + raw::tid_x();
    if i < n {
        let v = *x.add(i as usize) as i32;
        let bfly = raw::shfl_bfly_sync_i32(FULL_MASK, v, 1, SHFL_C_DOWN);
        let idx = raw::shfl_idx_sync_i32(FULL_MASK, v, 5, SHFL_C_DOWN);
        let up = raw::shfl_up_sync_i32(FULL_MASK, v, 1, SHFL_C_UP);
        raw::bar_warp_sync(FULL_MASK);
        let odd = (v & 1) == 1;
        let ballot = raw::vote_ballot_sync(FULL_MASK, odd);
        let all = raw::vote_all_sync(FULL_MASK, odd);
        let any = raw::vote_any_sync(FULL_MASK, odd);
        *out_bfly.add(i as usize) = bfly as u32;
        *out_idx.add(i as usize) = idx as u32;
        *out_up.add(i as usize) = up as u32;
        *out_ballot.add(i as usize) = ballot;
        *out_vote.add(i as usize) = ((all as u32) << 1) | (any as u32);
    }
}

/// Dynamic shared memory (`extern __shared__`): each block stages its
/// slice in the window sized by the launch's sharedMemBytes and writes it
/// back block-reversed. n is a multiple of the block size.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __crabbit_kernel_dyn_smem_reverse(
    x: *const f32,
    out: *mut f32,
    n: u32,
) {
    let tid = raw::tid_x();
    let nt = raw::ntid_x();
    let i = raw::ctaid_x() * nt + tid;
    let s = raw::dyn_shared_base() as *mut f32;
    if i < n {
        *s.add(tid as usize) = *x.add(i as usize);
    }
    raw::barrier0();
    if i < n {
        *out.add(i as usize) = *s.add((nt - 1 - tid) as usize);
    }
}

/// A by-value struct kernel parameter: one `.param .align 4 .b8 …[12]` the
/// host fills with the C-layout bytes, fields loaded via `ld.param`.
#[repr(C)]
pub struct Affine {
    pub scale: f32,
    pub bias: f32,
    pub shift: i32,
}

/// out[i] = x[i] * w.scale + w.bias + w.shift as f32.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __crabbit_kernel_affine_apply(
    w: Affine,
    x: *const f32,
    out: *mut f32,
    n: u32,
) {
    let i = raw::ctaid_x() * raw::ntid_x() + raw::tid_x();
    if i < n {
        *out.add(i as usize) = *x.add(i as usize) * w.scale + w.bias + w.shift as f32;
    }
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
