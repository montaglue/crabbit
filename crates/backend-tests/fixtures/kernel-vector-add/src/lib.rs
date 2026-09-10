//! Kernel fixtures for crabbit's native NVPTX path (docs/KERNEL-ABI.md):
//! an integer vector add, and a shared-memory f32 block reduction that
//! exercises `.shared` statics, `bar.sync`, and f32 arithmetic.
#![no_std]
#![allow(unsafe_op_in_unsafe_fn)]

use crabbit_device::raw;

/// out[i] = a[i] + b[i] for i < n.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __crabbit_kernel_vector_add(
    a: *const i32,
    b: *const i32,
    out: *mut i32,
    n: u32,
) {
    let i = raw::ctaid_x() * raw::ntid_x() + raw::tid_x();
    if i < n {
        *out.add(i as usize) = *a.add(i as usize) + *b.add(i as usize);
    }
}

const BLOCK: usize = 256;

#[unsafe(link_section = ".shared")]
static mut PARTIAL: [f32; BLOCK] = [0.0; BLOCK];

/// out[block] = Σ x[block*256 + t] (t < 256, elements beyond n count as 0),
/// via a shared-memory tree reduction.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __crabbit_kernel_block_sum_f32(x: *const f32, out: *mut f32, n: u32) {
    let t = raw::tid_x();
    let i = raw::ctaid_x() * raw::ntid_x() + t;
    let partial = &raw mut PARTIAL;
    let v = if i < n { *x.add(i as usize) } else { 0.0 };
    *(partial as *mut f32).add(t as usize) = v;
    raw::barrier0();
    let mut stride = BLOCK as u32 / 2;
    while stride > 0 {
        if t < stride {
            let lhs = *(partial as *mut f32).add(t as usize);
            let rhs = *(partial as *mut f32).add((t + stride) as usize);
            *(partial as *mut f32).add(t as usize) = lhs + rhs;
        }
        raw::barrier0();
        stride /= 2;
    }
    if t == 0 {
        *out.add(raw::ctaid_x() as usize) = *(partial as *mut f32);
    }
}
