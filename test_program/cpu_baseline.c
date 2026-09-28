/*
 * cpu_baseline32_min.c - Cortex-A9 baseline for the 32x32 @ 32x32 int8
 *                        matmul, for comparison against matmul32_dma.c.
 *
 * One kernel: the accelerator's own arithmetic. Each ARRAY_COLS-sized
 * contraction block is requantized and saturated to int8 BEFORE being
 * accumulated into the previous block's byte, which is then saturated
 * again - so this computes the same function the hardware does, which is
 * what makes a speedup number meaningful.
 *
 * Reports min / median / mean / stddev over many iterations after warmup.
 *
 * CAVEAT for any speedup claim: at 32x32 the working set is 3 KB and stays
 * in L1, so this is the CPU at its best. Real layers stream weights from
 * DDR and are slower per MAC.
 *
 * Build:  gcc -O3 -mcpu=cortex-a9 -mfpu=neon -o cpu_baseline32_min cpu_baseline32_min.c
 * Run:    ./cpu_baseline32_min [-i iters] [-w warmup] [-r seed] [-s scale] [--csv]
 */

#define _POSIX_C_SOURCE 200809L

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define K 32
#define N 32
#define M 32

#define ARRAY_COLS 16                 /* accelerator contraction block */
#define NUM_BLOCKS (N / ARRAY_COLS)   /* = 2 */

#define MACS ((double)K * N * M)

static uint32_t out_scale = 2048;     /* same Q0.16 as matmul32_dma.c */

static inline int8_t sat8(int32_t v)
{
    if (v > 127) return 127;
    if (v < -128) return -128;
    return (int8_t)v;
}

/* saturate_int8( (psum * OUTPUT_SCALE) >>> 16 ) */
static inline int8_t requant(int32_t psum, uint32_t scale)
{
    return sat8((int32_t)(((int64_t)psum * (int64_t)scale) >> 16));
}

static void matmul_blocked(const int8_t *W, const int8_t *X, int8_t *Y,
                           uint32_t scale)
{
    for (int k = 0; k < K; k++) {
        for (int m = 0; m < M; m++) {
            int32_t acc = 0;
            for (int nb = 0; nb < NUM_BLOCKS; nb++) {
                int32_t psum = 0;
                for (int c = 0; c < ARRAY_COLS; c++) {
                    int p = nb * ARRAY_COLS + c;
                    psum += (int32_t)W[k * N + p] * (int32_t)X[p * M + m];
                }
                int8_t s = requant(psum, scale);
                acc = (nb == 0) ? s : sat8(acc + s);
            }
            Y[k * M + m] = (int8_t)acc;
        }
    }
}

/* ------------------------------------------------------------------ */

static uint64_t ns_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static int cmp_u64(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

/* Newton's method - avoids linking -lm for one square root */
static double dsqrt(double x)
{
    if (x <= 0.0) return 0.0;
    double r = x, prev = 0.0;
    for (int i = 0; i < 40 && r != prev; i++) { prev = r; r = 0.5 * (r + x / r); }
    return r;
}

static volatile int8_t sink; /* keeps the result live */

int main(int argc, char **argv)
{
    int      iters = 1000, warmup = 100, csv = 0;
    unsigned seed = 1;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-i") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-w") && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) seed = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) out_scale = (uint32_t)strtoul(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--csv")) csv = 1;
        else {
            printf("usage: %s [-i iters] [-w warmup] [-r seed] [-s scale] [--csv]\n",
                   argv[0]);
            return 1;
        }
    }
    if (iters < 1) iters = 1;

    static int8_t W[K * N], X[N * M], Y[K * M];
    srand(seed);
    for (int i = 0; i < K * N; i++) W[i] = (int8_t)((rand() % 64) - 32);
    for (int i = 0; i < N * M; i++) X[i] = (int8_t)((rand() % 64) - 32);

    for (int i = 0; i < warmup; i++) { matmul_blocked(W, X, Y, out_scale); sink = Y[0]; }

    uint64_t *t = malloc((size_t)iters * sizeof *t);
    if (!t) { perror("malloc"); return 1; }
    for (int i = 0; i < iters; i++) {
        uint64_t a = ns_now();
        matmul_blocked(W, X, Y, out_scale);
        uint64_t b = ns_now();
        sink = Y[i % (K * M)];   /* consume, defeat dead-code elimination */
        t[i] = b - a;
    }

    qsort(t, (size_t)iters, sizeof t[0], cmp_u64);
    double min = (double)t[0];
    double med = (iters & 1) ? (double)t[iters / 2]
                             : 0.5 * ((double)t[iters / 2 - 1] + (double)t[iters / 2]);
    double sum = 0.0;
    for (int i = 0; i < iters; i++) sum += (double)t[i];
    double mean = sum / iters;
    double var = 0.0;
    for (int i = 0; i < iters; i++) { double d = (double)t[i] - mean; var += d * d; }
    double sd = (iters > 1) ? dsqrt(var / (iters - 1)) : 0.0;

    double gops = 2.0 * MACS / med;   /* ns -> GOP/s directly */

    if (csv) {
        printf("variant,min_ns,median_ns,mean_ns,sd_ns,gops\n");
        printf("blocked,%.0f,%.0f,%.0f,%.0f,%.4f\n", min, med, mean, sd, gops);
    } else {
        printf("cpu_baseline32_min: %dx%d @ %dx%d int8, OUTPUT_SCALE=%u (%.5f)\n",
               K, N, N, M, out_scale, out_scale / 65536.0);
        printf("  %.0f MACs per call, %d iterations after %d warmup\n\n",
               MACS, iters, warmup);
        printf("  blocked   min %8.2f us   median %8.2f us   mean %8.2f us   "
               "sd %6.2f us\n", min / 1e3, med / 1e3, mean / 1e3, sd / 1e3);
        printf("            %.3f GOP/s  (%.1f ns per MAC)\n",
               gops, med / MACS);
        printf("\nCompare against matmul32_dma.c: \"start -> done\" for the\n"
               "compute-only speedup, total for end to end.\n");
    }

    free(t);
    return 0;
}