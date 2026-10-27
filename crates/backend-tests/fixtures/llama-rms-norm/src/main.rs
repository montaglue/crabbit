//! Binary backend fixture based on the shape of llama.cpp CUDA RMSNorm.
//!
//! This crate mirrors the backend-smoke fixture: Cargo drives it as a binary,
//! while the cuda-oxide `#[kernel]` function is captured by the crabbit
//! backend into the PTX sidecar instead of the host object.

#![allow(dead_code)]

use cuda_device::{DisjointSlice, SharedArray, float, kernel, thread};

const MAX_BLOCK: usize = 1024;

/// Sum `partial` over the block (block size a power of two ≤ 1024); every
/// thread gets the total.
#[inline(always)]
fn block_reduce_sum_f32(partial: f32) -> f32 {
    static mut RED: SharedArray<f32, MAX_BLOCK> = SharedArray::UNINIT;
    let red = unsafe { SharedArray::as_raw_mut_ptr(&raw mut RED) };
    let tid = thread::threadIdx_x() as usize;
    unsafe { *red.add(tid) = partial };
    thread::sync_threads();
    let mut stride = thread::blockDim_x() as usize / 2;
    while stride > 0 {
        if tid < stride {
            unsafe { *red.add(tid) = *red.add(tid) + *red.add(tid + stride) };
        }
        thread::sync_threads();
        stride /= 2;
    }
    let total = unsafe { *red };
    thread::sync_threads();
    total
}

/// One block normalizes one row.
///
/// Intended launch shape:
/// - `grid.x = nrows`
/// - `block.x = 64` or `1024`, matching the CUDA implementation choice
/// - `weight.len()` is the width of each hidden-state row
///
/// Algorithm:
/// 1. Each thread accumulates `x[col] * x[col]` for its strided columns.
/// 2. The block reduces partial sums into the row sum of squares.
/// 3. The scale is `rsqrt(sum_squares / ncols + eps)`.
/// 4. Each thread writes its strided output columns.
///
/// `weight` is the RMSNorm gamma vector. Passing it here makes this closer to a
/// transformer layer kernel than bare RMS normalization.
#[kernel]
pub fn llama_rms_norm_f32(x: &[f32], weight: &[f32], mut dst: DisjointSlice<f32>, eps: f32) {
    let row = thread::blockIdx_x() as usize;
    let tid = thread::threadIdx_x() as usize;
    let block_size = thread::blockDim_x() as usize;
    let ncols = weight.len();
    let row_base = row * ncols;
    let x = x.as_ptr();
    let w = weight.as_ptr();
    let dst = dst.as_mut_ptr();

    let mut partial_sum = 0.0f32;
    let mut col = tid;
    while col < ncols {
        let xi = unsafe { *x.add(row_base + col) };
        partial_sum += xi * xi;
        col += block_size;
    }

    let sum_squares = block_reduce_sum_f32(partial_sum);
    let scale = float::rsqrt_approx_f32(sum_squares / ncols as f32 + eps);

    let mut col = tid;
    while col < ncols {
        let xi = unsafe { *x.add(row_base + col) };
        let wi = unsafe { *w.add(col) };
        unsafe { *dst.add(row_base + col) = xi * scale * wi };
        col += block_size;
    }
}

fn main() {
    println!("hello from crabbit");
}
