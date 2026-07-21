use stair_device::{DeviceSlice, DisjointSlice, kernel};

#[kernel]
pub fn add_kernel(lhs: i64, rhs: i64) -> i64 {
    lhs + rhs
}

#[kernel]
pub fn slice_kernel(_input: DeviceSlice<f32>, _output: DisjointSlice<f32>) {
    let _ = stair_device::thread::thread_id_x();
}

fn main() {
    println!("hello from stair");
}
