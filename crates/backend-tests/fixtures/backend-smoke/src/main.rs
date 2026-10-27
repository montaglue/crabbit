//! Smoke fixture for a mixed crate: an ordinary host binary whose crate
//! also holds cuda-oxide `#[kernel]` functions. The backend compiles the
//! host as usual and writes the kernels to the PTX sidecar
//! (docs/KERNEL-ABI.md); the kernels never enter the host object.
use cuda_device::{DisjointSlice, kernel, thread};

/// out[i] = lhs[i] + rhs[i] for i < n.
#[kernel]
pub fn add_kernel(lhs: *const i64, rhs: *const i64, out: *mut i64, n: u32) {
    let i = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
    if i < n {
        unsafe { *out.add(i as usize) = *lhs.add(i as usize) + *rhs.add(i as usize) };
    }
}

/// output[i] = 2 * input[i], through cuda-device's checked slice types.
#[kernel]
pub fn slice_kernel(input: &[f32], mut output: DisjointSlice<f32>) {
    let i = thread::index_1d();
    let k = i.get();
    if let Some(slot) = output.get_mut(i) {
        *slot = input[k] * 2.0;
    }
}

fn main() {
    println!("hello from crabbit: {}", crabbit_backend_smoke::add_i64(40, 2));
}
