//! Kernel fixtures for crabbit's native NVPTX path (docs/KERNEL-ABI.md):
//! an integer vector add, and a shared-memory f32 block reduction that
//! exercises `.shared` statics, `bar.sync`, and f32 arithmetic.
#![no_std]
#![allow(unsafe_op_in_unsafe_fn)]

use cuda_device::{SharedArray, kernel, thread};

/// out[i] = a[i] + b[i] for i < n.
#[kernel]
pub unsafe fn vector_add(
    a: *const i32,
    b: *const i32,
    out: *mut i32,
    n: u32,
) {
    let i = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
    if i < n {
        *out.add(i as usize) = *a.add(i as usize) + *b.add(i as usize);
    }
}

const BLOCK: usize = 256;

static mut PARTIAL: SharedArray<f32, { BLOCK }> = SharedArray::UNINIT;

/// out[block] = Σ x[block*256 + t] (t < 256, elements beyond n count as 0),
/// via a shared-memory tree reduction.
#[kernel]
pub unsafe fn block_sum_f32(x: *const f32, out: *mut f32, n: u32) {
    let t = thread::threadIdx_x();
    let i = thread::blockIdx_x() * thread::blockDim_x() + t;
    let partial = SharedArray::as_raw_mut_ptr(&raw mut PARTIAL);
    let v = if i < n { *x.add(i as usize) } else { 0.0 };
    *partial.add(t as usize) = v;
    thread::sync_threads();
    let mut stride = BLOCK as u32 / 2;
    while stride > 0 {
        if t < stride {
            let lhs = *partial.add(t as usize);
            let rhs = *partial.add((t + stride) as usize);
            *partial.add(t as usize) = lhs + rhs;
        }
        thread::sync_threads();
        stride /= 2;
    }
    if t == 0 {
        *out.add(thread::blockIdx_x() as usize) = *partial;
    }
}
