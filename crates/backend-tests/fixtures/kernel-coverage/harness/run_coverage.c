// Driver-API harness for the kernel-coverage fixture: loads a PTX module,
// runs `pair_swap_sum`, `local_table`, `tri_rec`, the warp-primitive
// kernels (`warp_reduce`, `warp_ops`), `dyn_smem_reverse` (dynamic shared
// memory via sharedMemBytes) and `affine_apply` (by-value struct kernel
// param), and checks results on the host.
// Usage: run_coverage <file.ptx>  (exit 0 on success).
#include <cuda.h>
#include <math.h>
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

static unsigned tri_host(unsigned n) { return n * (n + 1) / 2; }

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

    const unsigned n = 1000;
    const unsigned block = 128, grid = (n + block - 1) / block;

    // ---- pair_swap_sum ----------------------------------------------------
    {
        int *p = malloc(n * 2 * sizeof *p), *out = malloc(n * 2 * sizeof *out);
        for (unsigned i = 0; i < n; i++) { p[2 * i] = (int)i * 3 - 700; p[2 * i + 1] = 400 - (int)i; }
        memset(out, 0xAA, n * 2 * sizeof *out);
        CUfunction fn;
        CHECK(cuModuleGetFunction(&fn, mod, "pair_swap_sum"));
        CUdeviceptr dp, dout;
        CHECK(cuMemAlloc(&dp, n * 2 * sizeof *p));
        CHECK(cuMemAlloc(&dout, n * 2 * sizeof *out));
        CHECK(cuMemcpyHtoD(dp, p, n * 2 * sizeof *p));
        unsigned nn = n;
        void *args[] = {&dp, &dout, &nn};
        CHECK(cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, 0, args, 0));
        CHECK(cuCtxSynchronize());
        CHECK(cuMemcpyDtoH(out, dout, n * 2 * sizeof *out));
        for (unsigned i = 0; i < n; i++) {
            int ea = p[2 * i + 1], eb = p[2 * i] + p[2 * i + 1];
            if (out[2 * i] != ea || out[2 * i + 1] != eb) {
                fprintf(stderr, "pair_swap_sum wrong at %u: got {%d,%d} want {%d,%d}\n",
                        i, out[2 * i], out[2 * i + 1], ea, eb);
                return 1;
            }
        }
        cuMemFree(dp); cuMemFree(dout); free(p); free(out);
        printf("pair_swap_sum ok\n");
    }

    // ---- local_table --------------------------------------------------------
    {
        unsigned *x = malloc(n * sizeof *x), *out = malloc(n * sizeof *out);
        for (unsigned i = 0; i < n; i++) x[i] = i * 2654435761u;
        memset(out, 0xBB, n * sizeof *out);
        CUfunction fn;
        CHECK(cuModuleGetFunction(&fn, mod, "local_table"));
        CUdeviceptr dx, dout;
        CHECK(cuMemAlloc(&dx, n * sizeof *x));
        CHECK(cuMemAlloc(&dout, n * sizeof *out));
        CHECK(cuMemcpyHtoD(dx, x, n * sizeof *x));
        unsigned nn = n;
        void *args[] = {&dx, &dout, &nn};
        CHECK(cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, 0, args, 0));
        CHECK(cuCtxSynchronize());
        CHECK(cuMemcpyDtoH(out, dout, n * sizeof *out));
        for (unsigned i = 0; i < n; i++) {
            unsigned j = x[i] & 7, want = x[i] * j + j;
            if (out[i] != want) {
                fprintf(stderr, "local_table wrong at %u: got %u want %u\n", i, out[i], want);
                return 1;
            }
        }
        cuMemFree(dx); cuMemFree(dout); free(x); free(out);
        printf("local_table ok\n");
    }

    // ---- tri_rec ------------------------------------------------------------
    {
        unsigned *x = malloc(n * sizeof *x), *out = malloc(n * sizeof *out);
        for (unsigned i = 0; i < n; i++) x[i] = i * 40503u;
        memset(out, 0xCC, n * sizeof *out);
        CUfunction fn;
        CHECK(cuModuleGetFunction(&fn, mod, "tri_rec"));
        CUdeviceptr dx, dout;
        CHECK(cuMemAlloc(&dx, n * sizeof *x));
        CHECK(cuMemAlloc(&dout, n * sizeof *out));
        CHECK(cuMemcpyHtoD(dx, x, n * sizeof *x));
        unsigned nn = n;
        void *args[] = {&dx, &dout, &nn};
        CHECK(cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, 0, args, 0));
        CHECK(cuCtxSynchronize());
        CHECK(cuMemcpyDtoH(out, dout, n * sizeof *out));
        for (unsigned i = 0; i < n; i++) {
            unsigned want = tri_host(x[i] & 15);
            if (out[i] != want) {
                fprintf(stderr, "tri_rec wrong at %u: got %u want %u\n", i, out[i], want);
                return 1;
            }
        }
        cuMemFree(dx); cuMemFree(dout); free(x); free(out);
        printf("tri_rec ok\n");
    }

    // ---- warp_reduce: shfl.down tree, bit-exact against the same tree ------
    {
        const unsigned nw = 32, total = nw * 32; // 32 full warps
        float *x = malloc(total * sizeof *x), *out = malloc(nw * sizeof *out);
        for (unsigned i = 0; i < total; i++) x[i] = (float)(i % 37) * 0.25f - 3.0f;
        memset(out, 0xAB, nw * sizeof *out);
        CUfunction fn;
        CHECK(cuModuleGetFunction(&fn, mod, "warp_reduce"));
        CUdeviceptr dx, dout;
        CHECK(cuMemAlloc(&dx, total * sizeof *x));
        CHECK(cuMemAlloc(&dout, nw * sizeof *out));
        CHECK(cuMemcpyHtoD(dx, x, total * sizeof *x));
        unsigned nwarps = nw;
        void *args[] = {&dx, &dout, &nwarps};
        CHECK(cuLaunchKernel(fn, total / 128, 1, 1, 128, 1, 1, 0, 0, args, 0));
        CHECK(cuCtxSynchronize());
        CHECK(cuMemcpyDtoH(out, dout, nw * sizeof *out));
        for (unsigned w = 0; w < nw; w++) {
            float lane[32];
            for (unsigned l = 0; l < 32; l++) lane[l] = x[w * 32 + l];
            for (unsigned off = 16; off > 0; off >>= 1)
                for (unsigned l = 0; l + off < 32; l++) lane[l] += lane[l + off];
            if (out[w] != lane[0]) {
                fprintf(stderr, "warp_reduce wrong at warp %u: got %a want %a\n",
                        w, out[w], lane[0]);
                return 1;
            }
        }
        cuMemFree(dx); cuMemFree(dout); free(x); free(out);
        printf("warp_reduce ok\n");
    }

    // ---- warp_ops: bfly / idx / up shuffles, ballot, vote all/any ----------
    {
        const unsigned total = 1024;
        unsigned *x = malloc(total * sizeof *x);
        unsigned *bfly = malloc(total * sizeof *bfly), *idx = malloc(total * sizeof *idx);
        unsigned *up = malloc(total * sizeof *up), *ballot = malloc(total * sizeof *ballot);
        unsigned *vote = malloc(total * sizeof *vote);
        for (unsigned i = 0; i < total; i++) x[i] = i * 2654435761u;
        CUfunction fn;
        CHECK(cuModuleGetFunction(&fn, mod, "warp_ops"));
        CUdeviceptr dx, db, di, du, dba, dv;
        CHECK(cuMemAlloc(&dx, total * 4)); CHECK(cuMemAlloc(&db, total * 4));
        CHECK(cuMemAlloc(&di, total * 4)); CHECK(cuMemAlloc(&du, total * 4));
        CHECK(cuMemAlloc(&dba, total * 4)); CHECK(cuMemAlloc(&dv, total * 4));
        CHECK(cuMemcpyHtoD(dx, x, total * 4));
        unsigned nn = total;
        void *args[] = {&dx, &db, &di, &du, &dba, &dv, &nn};
        CHECK(cuLaunchKernel(fn, total / 128, 1, 1, 128, 1, 1, 0, 0, args, 0));
        CHECK(cuCtxSynchronize());
        CHECK(cuMemcpyDtoH(bfly, db, total * 4)); CHECK(cuMemcpyDtoH(idx, di, total * 4));
        CHECK(cuMemcpyDtoH(up, du, total * 4)); CHECK(cuMemcpyDtoH(ballot, dba, total * 4));
        CHECK(cuMemcpyDtoH(vote, dv, total * 4));
        for (unsigned w = 0; w < total / 32; w++) {
            const unsigned base = w * 32;
            unsigned bmask = 0, all = 1, any = 0;
            for (unsigned l = 0; l < 32; l++) {
                unsigned odd = x[base + l] & 1;
                bmask |= odd << l;
                all &= odd;
                any |= odd;
            }
            for (unsigned l = 0; l < 32; l++) {
                unsigned i = base + l;
                unsigned ebfly = x[base + (l ^ 1)];
                unsigned eidx = x[base + 5];
                unsigned eup = l == 0 ? x[base] : x[i - 1];
                unsigned evote = (all << 1) | any;
                if (bfly[i] != ebfly || idx[i] != eidx || up[i] != eup ||
                    ballot[i] != bmask || vote[i] != evote) {
                    fprintf(stderr,
                            "warp_ops wrong at %u: bfly %u/%u idx %u/%u up %u/%u "
                            "ballot %08x/%08x vote %u/%u\n",
                            i, bfly[i], ebfly, idx[i], eidx, up[i], eup,
                            ballot[i], bmask, vote[i], evote);
                    return 1;
                }
            }
        }
        cuMemFree(dx); cuMemFree(db); cuMemFree(di); cuMemFree(du);
        cuMemFree(dba); cuMemFree(dv);
        free(x); free(bfly); free(idx); free(up); free(ballot); free(vote);
        printf("warp_ops ok\n");
    }

    // ---- dyn_smem_reverse: dynamic shared memory via sharedMemBytes --------
    {
        const unsigned total = 1024, blk = 128;
        float *x = malloc(total * sizeof *x), *out = malloc(total * sizeof *out);
        for (unsigned i = 0; i < total; i++) x[i] = (float)i * 0.5f - 100.0f;
        memset(out, 0xCD, total * sizeof *out);
        CUfunction fn;
        CHECK(cuModuleGetFunction(&fn, mod, "dyn_smem_reverse"));
        CUdeviceptr dx, dout;
        CHECK(cuMemAlloc(&dx, total * sizeof *x));
        CHECK(cuMemAlloc(&dout, total * sizeof *out));
        CHECK(cuMemcpyHtoD(dx, x, total * sizeof *x));
        unsigned nn = total;
        void *args[] = {&dx, &dout, &nn};
        CHECK(cuLaunchKernel(fn, total / blk, 1, 1, blk, 1, 1,
                             blk * sizeof(float), 0, args, 0));
        CHECK(cuCtxSynchronize());
        CHECK(cuMemcpyDtoH(out, dout, total * sizeof *out));
        for (unsigned i = 0; i < total; i++) {
            float want = x[(i / blk) * blk + (blk - 1 - i % blk)];
            if (out[i] != want) {
                fprintf(stderr, "dyn_smem_reverse wrong at %u: got %f want %f\n",
                        i, out[i], want);
                return 1;
            }
        }
        cuMemFree(dx); cuMemFree(dout); free(x); free(out);
        printf("dyn_smem_reverse ok\n");
    }

    // ---- affine_apply: by-value struct kernel parameter --------------------
    {
        struct Affine { float scale, bias; int shift; } w = {1.5f, -0.25f, 3};
        float *x = malloc(n * sizeof *x), *out = malloc(n * sizeof *out);
        for (unsigned i = 0; i < n; i++) x[i] = (float)i * 0.125f - 40.0f;
        memset(out, 0xEE, n * sizeof *out);
        CUfunction fn;
        CHECK(cuModuleGetFunction(&fn, mod, "affine_apply"));
        CUdeviceptr dx, dout;
        CHECK(cuMemAlloc(&dx, n * sizeof *x));
        CHECK(cuMemAlloc(&dout, n * sizeof *out));
        CHECK(cuMemcpyHtoD(dx, x, n * sizeof *x));
        unsigned nn = n;
        void *args[] = {&w, &dx, &dout, &nn};
        CHECK(cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, 0, args, 0));
        CHECK(cuCtxSynchronize());
        CHECK(cuMemcpyDtoH(out, dout, n * sizeof *out));
        for (unsigned i = 0; i < n; i++) {
            // crabbit contracts x*scale + bias to fma.rn (nvcc -fmad=true).
            float want = fmaf(x[i], w.scale, w.bias) + (float)w.shift;
            if (out[i] != want) {
                fprintf(stderr, "affine_apply wrong at %u: got %f want %f\n",
                        i, out[i], want);
                return 1;
            }
        }
        cuMemFree(dx); cuMemFree(dout); free(x); free(out);
        printf("affine_apply ok\n");
    }

    return 0;
}
