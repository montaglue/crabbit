# stair-rust Llama RMSNorm Fixture

This crate is intentionally outside the parent workspace. It is a future
backend test fixture, analogous to `backend-smoke`, for a non-trivial LLM kernel.

It is compiled explicitly as a binary by the `stair-rust` backend tests and by
`compile-with-backend.sh`. The fixture exercises raw pointer memory, `f32`,
loops, GPU IDs, reductions, math intrinsics, and host binary emission.

The kernel in `src/main.rs` is an original Rust translation of the shape of a
llama.cpp CUDA RMSNorm kernel: one tensor row per block, thread-strided column
work, sum-of-squares reduction, `rsqrt(mean_square + eps)`, and scaled output.

Run it manually with:

```sh
bash tools/backend-tests/fixtures/llama-rms-norm/compile-with-backend.sh
```
