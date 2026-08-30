// Driver-API harness for the kernel-vector-add fixture: loads a PTX module,
// runs `vector_add` and `block_sum_f32`, and checks the results on the host.
// Usage: run_kernels <file.ptx>   (exit 0 on success, prints what failed)
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(call)                                                            \
    do {                                                                       \
        CUresult _r = (call);                                                  \
        if (_r != CUDA_SUCCESS) {                                              \
            const char *name = "?";                                            \
            cuGetErrorName(_r, &name);                                         \
            fprintf(stderr, "%s failed: %s (%d)\n", #call, name, (int)_r);     \
            return 1;                                                          \
        }                                                                      \
    } while (0)

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(n + 1);
    if (fread(buf, 1, n, f) != (size_t)n) { fclose(f); free(buf); return NULL; }
    buf[n] = 0;
    fclose(f);
    return buf;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s file.ptx\n", argv[0]); return 2; }
    char *ptx = read_file(argv[1]);
    if (!ptx) return 2;

    CHECK(cuInit(0));
    CUdevice dev;
    CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK(cuCtxSetCurrent(ctx));
    CUmodule mod;
    char log[8192] = {0};
    CUjit_option opts[] = {CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
    void *vals[] = {log, (void *)(size_t)sizeof log};
    CUresult lr = cuModuleLoadDataEx(&mod, ptx, 2, opts, vals);
    if (lr != CUDA_SUCCESS) {
        fprintf(stderr, "cuModuleLoadDataEx failed (%d): %s\n", (int)lr, log);
        return 1;
    }

    // ---- vector_add -------------------------------------------------------
    {
        const unsigned n = 1000;
        int *a = malloc(n * sizeof *a), *b = malloc(n * sizeof *b), *out = malloc(n * sizeof *out);
        for (unsigned i = 0; i < n; i++) { a[i] = (int)i * 3 - 500; b[i] = 1000 - (int)i * 7; out[i] = -1; }
        CUfunction fn;
        CHECK(cuModuleGetFunction(&fn, mod, "vector_add"));
        CUdeviceptr da, db, dout;
        CHECK(cuMemAlloc(&da, n * sizeof *a));
        CHECK(cuMemAlloc(&db, n * sizeof *b));
        CHECK(cuMemAlloc(&dout, n * sizeof *out));
        CHECK(cuMemcpyHtoD(da, a, n * sizeof *a));
        CHECK(cuMemcpyHtoD(db, b, n * sizeof *b));
        CHECK(cuMemcpyHtoD(dout, out, n * sizeof *out));
        unsigned nn = n;
        void *args[] = {&da, &db, &dout, &nn};
        // 4 blocks of 256 = 1024 threads > n: the bounds check must hold.
        CHECK(cuLaunchKernel(fn, 4, 1, 1, 256, 1, 1, 0, 0, args, 0));
        CHECK(cuCtxSynchronize());
        CHECK(cuMemcpyDtoH(out, dout, n * sizeof *out));
        for (unsigned i = 0; i < n; i++) {
            if (out[i] != a[i] + b[i]) {
                fprintf(stderr, "vector_add mismatch at %u: got %d want %d\n", i, out[i], a[i] + b[i]);
                return 1;
            }
        }
        cuMemFree(da); cuMemFree(db); cuMemFree(dout);
        free(a); free(b); free(out);
        printf("vector_add ok\n");
    }

    // ---- block_sum_f32 ----------------------------------------------------
    {
        const unsigned n = 1000, blocks = 4;
        float *x = malloc(n * sizeof *x), out[4] = {-1, -1, -1, -1};
        for (unsigned i = 0; i < n; i++) x[i] = (float)(i % 17) - 8.0f;
        CUfunction fn;
        CHECK(cuModuleGetFunction(&fn, mod, "block_sum_f32"));
        CUdeviceptr dx, dout;
        CHECK(cuMemAlloc(&dx, n * sizeof *x));
        CHECK(cuMemAlloc(&dout, blocks * sizeof *out));
        CHECK(cuMemcpyHtoD(dx, x, n * sizeof *x));
        CHECK(cuMemcpyHtoD(dout, out, blocks * sizeof *out));
        unsigned nn = n;
        void *args[] = {&dx, &dout, &nn};
        CHECK(cuLaunchKernel(fn, blocks, 1, 1, 256, 1, 1, 0, 0, args, 0));
        CHECK(cuCtxSynchronize());
        CHECK(cuMemcpyDtoH(out, dout, blocks * sizeof *out));
        for (unsigned bidx = 0; bidx < blocks; bidx++) {
            // The inputs are small integers, so the tree order does not
            // change the (exactly representable) sum.
            float want = 0;
            for (unsigned t = 0; t < 256; t++) {
                unsigned i = bidx * 256 + t;
                if (i < n) want += x[i];
            }
            if (out[bidx] != want) {
                fprintf(stderr, "block_sum_f32 mismatch at block %u: got %f want %f\n", bidx, out[bidx], want);
                return 1;
            }
        }
        cuMemFree(dx); cuMemFree(dout);
        free(x);
        printf("block_sum_f32 ok\n");
    }
    cuModuleUnload(mod);
    cuDevicePrimaryCtxRelease(dev);
    return 0;
}
