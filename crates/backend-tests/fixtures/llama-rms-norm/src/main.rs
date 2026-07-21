//! Binary backend fixture based on the shape of llama.cpp CUDA RMSNorm.
//!
//! This crate mirrors the backend-smoke fixture: Cargo drives it as a binary,
//! while `#[kernel]` functions are captured by the STAIR backend instead of
//! being emitted into the host object.

#![allow(dead_code)]

use stair_device::{DeviceSlice, DisjointSlice, block, kernel, math, thread};

/// One block normalizes one row.
///
/// Intended launch shape:
/// - `grid.x = nrows`
/// - `block.x = 64` or `1024`, matching the CUDA implementation choice
/// - `ncols` is the width of each hidden-state row
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
pub unsafe fn llama_rms_norm_f32(
    x: DeviceSlice<f32>,
    weight: DeviceSlice<f32>,
    dst: DisjointSlice<f32>,
    eps: f32,
) {
    let row = thread::block_id_x() as usize;
    let tid = thread::thread_id_x() as usize;
    let block_size = thread::block_dim_x() as usize;
    let ncols = weight.len;
    let row_base = row * ncols;

    let mut partial_sum = 0.0f32;
    let mut col = tid;
    while col < ncols {
        let xi = unsafe { *x.ptr.add(row_base + col) };
        partial_sum += xi * xi;
        col += block_size;
    }

    let sum_squares = block::reduce_sum_f32(partial_sum);
    let scale = math::rsqrt_f32(sum_squares / ncols as f32 + eps);

    let mut col = tid;
    while col < ncols {
        let xi = unsafe { *x.ptr.add(row_base + col) };
        let wi = unsafe { *weight.ptr.add(col) };
        unsafe {
            *dst.ptr.add(row_base + col) = xi * scale * wi;
        }
        col += block_size;
    }
}

fn main() {
    println!("hello from stair");
}
