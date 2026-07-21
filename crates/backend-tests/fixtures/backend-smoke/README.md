# stair-rust Backend Smoke Crate

This crate is intentionally outside the parent workspace. It is compiled with
the `stair-rust` codegen backend dylib to verify that rustc can load the backend
and hand MIR bodies to it.

Run from the repository root:

```sh
bash tools/backend-tests/fixtures/backend-smoke/compile-with-backend.sh
```

The script compiles the binary target with `--emit=obj` and prints the produced
object file path.
