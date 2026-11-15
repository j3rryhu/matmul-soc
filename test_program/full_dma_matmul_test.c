/*
 * matmul32.c - 32x32 @ 32x32 int8 matmul on matmul_accel_0.
 *              128-bit data path, hardware address remap.
 *
 * ---------------------------------------------------------------------------
 * How the buffers are written now
 * ---------------------------------------------------------------------------
 * buf_addr_remap does the banking, so the host writes W and X each as ONE
 * contiguous row-major image with every row padded to a whole 16-byte word:
 *
 *     weights  WBUF_BASE + k*pitch(N) + n        (K rows)
 *     inputs   IBUF_BASE + p*pitch(M) + m        (N rows, p = contraction)
 *     pitch(c) = roundup(c, 16)
 *
 * Hardware scatters row r into bank r % 16 using WEIGHT_COLS and
 * INPUT_MAX_ADDR, so THOSE REGISTERS MUST BE WRITTEN FIRST. Writing a buffer
 * before the dimensions means the remap places the words with stale values.
 *
 * At 32x32 pitch == 32, so both images are dense - 1024 bytes for 1024 live.
 * One DMA descriptor each, no padding to work around.
 *
 * Results still come back banked: output row k is in bank k % 16, at
 * OBUF_BASE + 128*(k%16) + (k/16)*M. output_buffer has no remap. And
 * readback is PIO, because dma_0.read_master only reaches DDR.
 *
 * ---------------------------------------------------------------------------
 * Addressing: SYMBOLS units, so avs_address is the byte offset. Control
 * registers are spaced 16 bytes apart (alignment = 16 in ctrl_reg.rdl),
 * reached with 32-bit accesses. PIO buffer traffic is one NEON vld1/vst1 per
 * 16 aligned bytes.
 *
 * gcc -O2 -Wall -mcpu=cortex-a9 -mfpu=neon -o matmul32 matmul32.c -lm
 *   -i        identity test (W=2I, scale 0.5, Y must equal X) instead of quant
 *   -p        PIO for weights and inputs instead of DMA
 *   -r SEED   default 1
 */

#define _POSIX_C_SOURCE 200809L

#include <fcntl.h>
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#define HAVE_NEON 1
#else
#define HAVE_NEON 0
#endif

#define TWO_PI 6.283185307179586476925286766559

/* ---- accelerator, h2f bridge ------------------------------------- */
#define ACCEL_PHYS 0xC0000000UL
#define ACCEL_SPAN 0x00010000UL

#define CTRL_BASE 0x0000u
#define WBUF_BASE 0x1000u
#define IBUF_BASE 0x5000u
#define OBUF_BASE 0x7000u

/* register index; byte offset is index*16 (16-byte spacing) */
#define REG_CONTROL        0
#define REG_WEIGHT_ROWS    1
#define REG_WEIGHT_COLS    2
#define REG_INPUT_MAX_ADDR 3
#define REG_INPUT_COLS     4
#define REG_STATUS         5
#define REG_OUTPUT_SCALE   6
#define REG_STRIDE         16   /* one 128-bit beat per register, matching the
                                 * regblock on the board (`alignment = 16` in
                                 * rdl/ctrl_reg.rdl). The cocotb TB's 4-byte
                                 * spacing is stale - go by the RTL. */

#define ST_DONE 0x2u
#define BEAT    16u

/* ---- DMA, LW bridge ---------------------------------------------- */
#define LW_BASE 0xFF200000UL
#define LW_SPAN 0x00200000UL

#define DMA_STATUS     0
#define DMA_READ_ADDR  1
#define DMA_WRITE_ADDR 2
#define DMA_LENGTH     3
#define DMA_CONTROL    6

#define DMA_ST_DONE   0x1u
#define CTRL_GO       (1u << 3)
#define CTRL_LEEN     (1u << 7)
#define CTRL_QUADWORD (1u << 11)

#define DMA_ACCEL_BASE 0x00000000u   /* accel base in write_master's space */

#define SRC_PHYS 0x1F800000UL
#define SRC_SPAN 0x00100000UL
#define STAGE_W  0x0000u
#define STAGE_X  0x4000u

/* ---- shape ------------------------------------------------------- */
#define ARRAY_ROWS      16
#define ARRAY_COLS      16
#define OBUF_BANK_DEPTH 128

#define K 32
#define N 32
#define M 32

#define NUM_BANDS (N / ARRAY_COLS)   /* contraction blocks */
#define PITCH(c)  (((c) + BEAT - 1) / BEAT * BEAT)

#define WPITCH PITCH(N)              /* weight row pitch */
#define IPITCH PITCH(M)              /* input  row pitch */
#define WLEN   ((size_t)K * WPITCH)
#define ILEN   ((size_t)N * IPITCH)

static volatile uint8_t  *accel;
static volatile uint32_t *dma;
static volatile uint8_t  *src;

/* ------------------------------------------------------------------ */
/* quantization - mirrors tb/test_accel_top.py                         */
/* ------------------------------------------------------------------ */

static double quantize_sym(const double *x, int8_t *q, int n)
{
    double max_abs = 0.0;
    for (int i = 0; i < n; i++) {
        double a = fabs(x[i]);
        if (a > max_abs) max_abs = a;
    }
    double scale = (max_abs > 0.0) ? max_abs / 127.0 : 1.0;
    for (int i = 0; i < n; i++) {
        double v = rint(x[i] / scale);   /* half-to-even, like np.round */
        if (v > 127.0) v = 127.0;
        if (v < -127.0) v = -127.0;
        q[i] = (int8_t)v;
    }
    return scale;
}

static double randn(void)
{
    double u1, u2;
    do { u1 = (rand() + 1.0) / (RAND_MAX + 2.0); } while (u1 <= 0.0);
    u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
    return sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2);
}

/* ------------------------------------------------------------------ */
/* arithmetic - per contraction block: requantize, saturate, accumulate */
/* ------------------------------------------------------------------ */

static int8_t sat8(int32_t v)
{
    if (v > 127) return 127;
    if (v < -128) return -128;
    return (int8_t)v;
}

static int8_t requant(int32_t psum, uint32_t scale)
{
    return sat8((int32_t)(((int64_t)psum * (int64_t)scale) >> 16));
}

static void golden(const int8_t *W, const int8_t *X, int8_t *Y, uint32_t scale)
{
    for (int k = 0; k < K; k++)
        for (int m = 0; m < M; m++) {
            int32_t acc = 0;
            for (int nb = 0; nb < NUM_BANDS; nb++) {
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

/* ------------------------------------------------------------------ */
/* plumbing                                                            */
/* ------------------------------------------------------------------ */

static void on_bus_error(int sig)
{
    (void)sig;
    const char *m = "\n*** SIGBUS: access rejected, not hung.\n";
    ssize_t r = write(2, m, strlen(m));
    (void)r;
    _exit(2);
}

static uint64_t ns_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void reg_wr(int idx, uint32_t v)
{
    *(volatile uint32_t *)(accel + CTRL_BASE + idx * REG_STRIDE) = v;
}

static uint32_t reg_rd(int idx)
{
    return *(volatile uint32_t *)(accel + CTRL_BASE + idx * REG_STRIDE);
}

/*
 * 16 aligned bytes per access, one NEON vld1/vst1. Inline asm rather than an
 * intrinsic because this is Device memory: exactly one load/store per call,
 * unmerged and unreordered. The ':128' hint makes it a single aligned access
 * and faults rather than silently splitting.
 */
static void beat_wr(volatile uint8_t *p, const int8_t *b)
{
#if HAVE_NEON
    uint8x16_t v = vld1q_u8((const uint8_t *)b);
    __asm__ volatile("vst1.8 {%e0-%f0}, [%1 :128]" : : "w"(v), "r"(p) : "memory");
#else
    for (int j = 0; j < 16; j++) p[j] = (uint8_t)b[j];
#endif
}

static void beat_rd(volatile uint8_t *p, int8_t *b)
{
#if HAVE_NEON
    uint8x16_t v;
    __asm__ volatile("vld1.8 {%e0-%f0}, [%1 :128]" : "=w"(v) : "r"(p) : "memory");
    vst1q_u8((uint8_t *)b, v);
#else
    for (int j = 0; j < 16; j++) b[j] = (int8_t)p[j];
#endif
}

static void pio_image(uint32_t base, const int8_t *img, size_t len)
{
    for (size_t i = 0; i < len; i += BEAT) beat_wr(accel + base + i, &img[i]);
}

static void stage_image(uint32_t off, const int8_t *img, size_t len)
{
    for (size_t i = 0; i < len; i += BEAT) beat_wr(src + off + i, &img[i]);
}

static int dma_go(uint32_t src_phys, uint32_t dst, uint32_t bytes)
{
    dma[DMA_STATUS]     = 0;            /* DONE is sticky */
    dma[DMA_READ_ADDR]  = src_phys;
    dma[DMA_WRITE_ADDR] = dst;
    dma[DMA_LENGTH]     = bytes;
    __sync_synchronize();
    dma[DMA_CONTROL] = CTRL_QUADWORD | CTRL_LEEN | CTRL_GO;   /* WCON off */

    uint64_t t0 = ns_now();
    for (;;) {
        uint32_t st = dma[DMA_STATUS];
        if (st & DMA_ST_DONE) return 0;
        if (ns_now() - t0 > 1000000000ull) {
            printf("  DMA TIMEOUT, STATUS 0x%08x, LENGTH left %u\n",
                   st, dma[DMA_LENGTH]);
            printf("  check: dma_0 width 128? LEEN set? write_master wired?\n");
            return -1;
        }
    }
}

/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    int      pio = 0, identity = 0;
    unsigned seed = 1;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-p")) pio = 1;
        else if (!strcmp(argv[i], "-i")) identity = 1;
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) seed = (unsigned)atoi(argv[++i]);
        else { printf("usage: %s [-i] [-p] [-r seed]\n", argv[0]); return 1; }
    }

    setvbuf(stdout, NULL, _IONBF, 0);
    signal(SIGBUS, on_bus_error);
    signal(SIGSEGV, on_bus_error);

    static int8_t W[K * N], X[N * M], Y[K * M], G[K * M];
    uint32_t out_scale;

    if (identity) {
        /* W = 2I with scale 0.5 makes the datapath an exact identity:
         * requant(2x, 0.5) == x for every int8 x, and the other contraction
         * block contributes a clean zero. Y must equal X byte for byte. */
        memset(W, 0, sizeof W);
        for (int k = 0; k < K; k++) W[k * N + (k % N)] = 2;
        for (int p = 0; p < N; p++)
            for (int m = 0; m < M; m++)
                X[p * M + m] = (int8_t)(((p * M + m) % 251) - 125);
        out_scale = 32768;
        printf("matmul32 IDENTITY: W=2I, scale 0.5, expect Y == X\n");
    } else {
        static double Wr[K * N], Xr[N * M], Yr[K * M];
        static int8_t dummy[K * M];
        srand(seed);
        for (int i = 0; i < K * N; i++) Wr[i] = randn();
        for (int i = 0; i < N * M; i++) Xr[i] = randn();
        double ws = quantize_sym(Wr, W, K * N);
        double xs = quantize_sym(Xr, X, N * M);
        for (int k = 0; k < K; k++)
            for (int m = 0; m < M; m++) {
                double a = 0.0;
                for (int p = 0; p < N; p++) a += Wr[k * N + p] * Xr[p * M + m];
                Yr[k * M + m] = a;
            }
        double ys = quantize_sym(Yr, dummy, K * M);
        double combined = ws * xs / ys;
        if (!(combined >= 0.0 && combined < 1.0)) {
            fprintf(stderr, "combined scale %g does not fit Q0.16\n", combined);
            return 1;
        }
        out_scale = (uint32_t)lrint(combined * 65536.0);
        printf("matmul32 QUANT: w=%.6g x=%.6g y=%.6g combined=%.6g\n",
               ws, xs, ys, combined);
    }

    golden(W, X, G, out_scale);
    int sat = 0;
    for (int i = 0; i < K * M; i++) if (G[i] == 127 || G[i] == -128) sat++;
    printf("  %dx%d @ %dx%d  OUTPUT_SCALE=%u  %d/%d saturated  transfer=%s\n",
           K, N, N, M, out_scale, sat, K * M, pio ? "PIO" : "DMA");
    printf("  images: W %zu B (pitch %d), X %zu B (pitch %d) - %s\n",
           WLEN, WPITCH, ILEN, IPITCH,
           (WPITCH == N && IPITCH == M) ? "dense, no padding"
                                        : "rows padded to the word");

    /* ---- contiguous row-major images, rows padded to the word pitch ---- */
    static int8_t wimg[K * PITCH(N)], ximg[N * PITCH(M)];
    memset(wimg, 0, sizeof wimg);
    memset(ximg, 0, sizeof ximg);
    for (int k = 0; k < K; k++)
        memcpy(&wimg[k * WPITCH], &W[k * N], N);
    for (int p = 0; p < N; p++)
        memcpy(&ximg[p * IPITCH], &X[p * M], M);

    /* ---- map ---- */
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/mem (run as root)"); return 1; }
    void *amap = mmap(NULL, ACCEL_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)ACCEL_PHYS);
    void *lmap = mmap(NULL, LW_SPAN,    PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)LW_BASE);
    void *smap = mmap(NULL, SRC_SPAN,   PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)SRC_PHYS);
    if (amap == MAP_FAILED || lmap == MAP_FAILED || smap == MAP_FAILED) {
        perror("mmap"); return 1;
    }
    accel = (volatile uint8_t *)amap;
    dma   = (volatile uint32_t *)lmap;
    src   = (volatile uint8_t *)smap;

    uint32_t st0 = reg_rd(REG_STATUS);
    printf("  STATUS = 0x%08x%s\n", st0,
           (st0 & ~0x3u) ? "   <- unexpected bits: register spacing?" : "");

    /* ---- dimensions FIRST: the remap uses WEIGHT_COLS and INPUT_MAX_ADDR
     * to place each word in its bank, so a buffer written before these
     * would be scattered with stale values ---- */
    reg_wr(REG_WEIGHT_ROWS, K);
    reg_wr(REG_WEIGHT_COLS, N);
    reg_wr(REG_INPUT_MAX_ADDR, M);
    reg_wr(REG_INPUT_COLS, M);
    reg_wr(REG_OUTPUT_SCALE, out_scale);

    /* ---- weights and inputs ---- */
    uint64_t ts0 = ns_now();
    if (!pio) {
        stage_image(STAGE_W, wimg, WLEN);
        stage_image(STAGE_X, ximg, ILEN);
        __sync_synchronize();
    }
    uint64_t t0 = ns_now();

    if (pio) {
        pio_image(WBUF_BASE, wimg, WLEN);
    } else if (dma_go((uint32_t)(SRC_PHYS + STAGE_W),
                      DMA_ACCEL_BASE + WBUF_BASE, (uint32_t)WLEN) != 0) {
        return 1;
    }
    uint64_t t1 = ns_now();

    if (pio) {
        pio_image(IBUF_BASE, ximg, ILEN);
    } else if (dma_go((uint32_t)(SRC_PHYS + STAGE_X),
                      DMA_ACCEL_BASE + IBUF_BASE, (uint32_t)ILEN) != 0) {
        return 1;
    }
    uint64_t t2 = ns_now();

    /* ---- run ---- */
    (void)reg_rd(REG_STATUS);           /* drain any stale completion */
    reg_wr(REG_CONTROL, 1);

    uint32_t st = 0;
    int      polls = 0;
    uint64_t tp = ns_now();
    while (polls < 1000000) {
        st = reg_rd(REG_STATUS);
        polls++;
        if (st & ST_DONE) break;
        if (ns_now() - tp > 1000000000ull) break;
    }
    uint64_t t3 = ns_now();
    if (!(st & ST_DONE)) {
        printf("  TIMEOUT after %d polls, STATUS = 0x%08x\n", polls, st);
        printf("  busy=%u done=%u - if busy never set, CONTROL.start never"
               " pulsed; if it never clears, weight_ctrl is stuck\n",
               st & 1u, (st >> 1) & 1u);
        return 1;
    }

    /* ---- readback: output row k in bank k%16, segment (k/16)*M ---- */
    for (int k = 0; k < K; k++) {
        uint32_t base = OBUF_BASE
                      + (uint32_t)(k % ARRAY_ROWS) * OBUF_BANK_DEPTH
                      + (uint32_t)(k / ARRAY_ROWS) * M;
        for (int m = 0; m < M; m += BEAT)
            beat_rd(accel + base + m, &Y[k * M + m]);
    }
    uint64_t t4 = ns_now();

    /* ---- verify ---- */
    int bad = 0, maxerr = 0;
    int row_bad[K];
    memset(row_bad, 0, sizeof row_bad);
    for (int k = 0; k < K; k++)
        for (int m = 0; m < M; m++) {
            int g = Y[k * M + m], e = G[k * M + m];
            if (g == e) continue;
            int d = abs(g - e);
            if (d > maxerr) maxerr = d;
            row_bad[k]++;
            if (bad < 10)
                printf("  Y[%2d][%2d]: got %4d want %4d  (bank %2d seg %d)\n",
                       k, m, g, e, k % ARRAY_ROWS, k / ARRAY_ROWS);
            bad++;
        }

    if (bad) {
        printf("  FAIL: %d/%d wrong, max |err| = %d\n", bad, K * M, maxerr);
        /* one bad row means one bank; all rows bad means the image layout
         * or the dims-before-buffers ordering */
        printf("  rows wrong:");
        for (int k = 0; k < K; k++) if (row_bad[k]) printf(" %d", k);
        printf("\n");
    } else {
        printf("  PASS: all %d elements match\n", K * M);
    }

    printf("\n  stage   %7.2f us   weights %7.2f us   inputs %7.2f us\n",
           (double)(t0 - ts0) / 1e3, (double)(t1 - t0) / 1e3,
           (double)(t2 - t1) / 1e3);
    printf("  compute %7.2f us   readback %6.2f us   total %8.2f us\n",
           (double)(t3 - t2) / 1e3, (double)(t4 - t3) / 1e3,
           (double)(t4 - t0) / 1e3);
    printf("  %d MACs: %.3f GOP/s compute, %.3f GOP/s end to end\n",
           K * N * M, 2.0 * K * N * M / (double)(t3 - t2),
           2.0 * K * N * M / (double)(t4 - t0));

    munmap(amap, ACCEL_SPAN);
    munmap(lmap, LW_SPAN);
    munmap(smap, SRC_SPAN);
    close(fd);
    return bad ? 1 : 0;
}