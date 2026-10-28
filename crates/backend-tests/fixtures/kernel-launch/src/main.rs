//! A host program that launches two kernels of its own crate on the GPU,
//! written against cuda-oxide's API: the kernels live in a
//! `#[cuda_module]`, the host loads that module from its own executable
//! and launches through the module's typed methods. crabbit compiles both
//! sides: the kernels to PTX embedded in the executable, `main` to the
//! host object.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    /// `sum[i] = a[i] + b[i]`.
    #[kernel]
    pub fn add(a: &[f32], b: &[f32], mut sum: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if let Some(out) = sum.get_mut(index) {
            *out = a[i] + b[i];
        }
    }

    /// `scaled[i] = factor * x[i]`.
    #[kernel]
    pub fn scale(x: &[f32], factor: f32, mut scaled: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if let Some(out) = scaled.get_mut(index) {
            *out = factor * x[i];
        }
    }
}

const N: usize = 4096;
const FACTOR: f32 = 0.5;

fn main() {
    let ctx = CudaContext::new(0).expect("create a CUDA context on device 0");
    let stream = ctx.default_stream();

    let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..N).map(|i| (3 * i) as f32 + 1.0).collect();
    let a_dev = DeviceBuffer::from_host(&stream, &a).expect("upload a");
    let b_dev = DeviceBuffer::from_host(&stream, &b).expect("upload b");
    let mut sum_dev = DeviceBuffer::<f32>::zeroed(&stream, N).expect("allocate sum");
    let mut scaled_dev = DeviceBuffer::<f32>::zeroed(&stream, N).expect("allocate scaled");

    let module = kernels::load(&ctx).expect("load the embedded kernel module");
    let config = LaunchConfig::for_num_elems(N as u32);
    // SAFETY: one thread per element over N elements; every buffer holds N.
    unsafe {
        module
            .add(&stream, config, &a_dev, &b_dev, &mut sum_dev)
            .expect("launch add");
        module
            .scale(&stream, config, &sum_dev, FACTOR, &mut scaled_dev)
            .expect("launch scale");
    }

    let scaled = scaled_dev.to_host_vec(&stream).expect("download scaled");
    let wrong = (0..N)
        .filter(|&i| scaled[i] != FACTOR * (a[i] + b[i]))
        .count();
    if wrong != 0 {
        eprintln!("kernel-launch: {wrong} of {N} elements wrong (first: {})", scaled[0]);
        std::process::exit(1);
    }
    println!("kernel-launch: add then scale correct on {N} elements");
}
