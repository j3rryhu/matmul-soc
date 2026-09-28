/*
 * matmul32_quant.c - 32x32 @ 32x32 int8 matmul on matmul_accel_0, using the
 *                    same per-tensor symmetric quantization scheme as
 *                    tb/test_accel_top.py. Weights move by DMA.
 *
 * Quantization (matching the testbench, which matches a real quantized
 * inference flow):
 *
 *   W_real, X_real ~ N(0, 1)
 *   W_int8, w_scale = quantize_symmetric_int8(W_real)   scale = max|.|/127
 *   X_int8, x_scale = quantize_symmetric_int8(X_real)
 *   Y_real          = W_real @ X_real
 *   _,      y_scale = quantize_symmetric_int8(Y_real)   calibration stand-in
 *
 *   OUTPUT_SCALE = round(w_scale * x_scale / y_scale * 65536)   must be < 65536
 *
 * The device only ever sees that one combined Q0.16 factor. Note the scale
 * is derived for the FULL contraction, while the hardware requantizes each
 * ARRAY_COLS-sized block separately and accumulates the saturated results -
 * the golden model below reproduces that ordering, so the requantization
 * error is measured rather than assumed away.
 *
 * np.round is round-half-to-even, so rint() is used here rather than
 * round() (which rounds half away from zero) - otherwise the quantized
 * values would differ from the testbench's on exact .5 boundaries.
 *
 * --dump FILE writes W_int8, X_int8 and OUTPUT_SCALE so the testbench can
 * load the identical matrices: C and numpy cannot share a normal generator,
 * so this is the only way to get exact data equality between sim and
 * hardware.
 *
 * Addressing assumes SYMBOLS address units (avs_address == CPU byte offset).
 *
 * gcc -O2 -Wall -o matmul32_quant matmul32_quant.c -lm
 *   -p         PIO weights instead of DMA (isolates the DMA path)
 *   -s         prompt between phases
 *   -r SEED    default 1
 *   --dump F   write matrices + scale to F
 */

#define _POSIX_C_SOURCE 200809L

#include <fcntl.h>
#include <math.h>

#define TWO_PI 6.283185307179586476925286766559
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

/* ---- accelerator, h2f bridge ------------------------------------- */
#define ACCEL_PHYS 0xC0000000UL
#define ACCEL_SPAN 0x00010000UL   /* SYMBOLS units: 2^16 BYTES */

#define CTRL_BASE 0x0000u
#define WBUF_BASE 0x1000u
#define IBUF_BASE 0x5000u
#define OBUF_BASE 0x7000u

#define REG_CONTROL        0x00u
#define REG_WEIGHT_ROWS    0x04u
#define REG_WEIGHT_COLS    0x08u
#define REG_INPUT_MAX_ADDR 0x0Cu
#define REG_INPUT_COLS     0x10u
#define REG_STATUS         0x14u
#define REG_OUTPUT_SCALE   0x18u

#define ST_DONE 0x2u
#define AVS(a) ((uint32_t)(a))

/* ---- DMA, LW bridge ---------------------------------------------- */
#define LW_BASE 0xFF200000UL
#define LW_SPAN 0x00200000UL
#define DMA_OFF 0x00000000UL

#define DMA_STATUS     0
#define DMA_READ_ADDR  1
#define DMA_WRITE_ADDR 2
#define DMA_LENGTH     3
#define DMA_CONTROL    6

#define DMA_ST_DONE 0x1u

#define CTRL_WORD (1u << 2)
#define CTRL_GO   (1u << 3)
#define CTRL_LEEN (1u << 7)

#define DMA_ACCEL_BASE 0x00000000u   /* verified: Address Map, write_master */

#define SRC_PHYS 0x1F800000UL
#define SRC_SPAN 0x00100000UL

/* ---- shape -------------------------------------------------------- */
#define ARRAY_ROWS      16
#define ARRAY_COLS      16
#define IBUF_BANK_DEPTH 256
#define OBUF_BANK_DEPTH 128

#define K 32
#define N 32
#define M 32

#define NUM_BANDS (N / ARRAY_COLS)
#define NUM_SEGS  (K / ARRAY_ROWS)

static volatile uint8_t  *accel;
static volatile uint32_t *dma;
static volatile uint8_t  *src;
static int step_mode = 0;

/* ------------------------------------------------------------------ */
/* quantization - mirrors tb/test_accel_top.py                         */
/* ------------------------------------------------------------------ */

/*
 * Per-tensor symmetric int8: zero point 0, range [-127,127]. 127 rather
 * than 128 keeps the range symmetric about zero, exactly as the testbench's
 * quantize_symmetric_int8 does.
 */
static double quantize_sym(const double *x, int8_t *q, int n)
{
    double max_abs = 0.0;
    for (int i = 0; i < n; i++) {
        double a = fabs(x[i]);
        if (a > max_abs) max_abs = a;
    }
    double scale = (max_abs > 0.0) ? max_abs / 127.0 : 1.0;
    for (int i = 0; i < n; i++) {
        double v = rint(x[i] / scale);   /* rint: half-to-even, like np.round */
        if (v > 127.0) v = 127.0;
        if (v < -127.0) v = -127.0;
        q[i] = (int8_t)v;
    }
    return scale;
}

/* Box-Muller from rand(); the scheme matches the testbench, the values
 * cannot (numpy uses PCG64 + ziggurat). Use --dump for exact equality. */
static double randn(void)
{
    double u1, u2;
    do { u1 = (rand() + 1.0) / (RAND_MAX + 2.0); } while (u1 <= 0.0);
    u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
    return sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2);
}

/* ------------------------------------------------------------------ */
/* arithmetic                                                          */
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

/* per-block requantize, saturate, accumulate - the hardware's ordering */
static void golden(const int8_t *W, const int8_t *X, int8_t *Y, uint32_t scale)
{
    for (int k = 0; k < K; k++) {
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

static void phase(const char *what)
{
    printf("\n--- %s\n", what);
    if (step_mode) {
        printf("    press enter: ");
        int c;
        while ((c = getchar()) != '\n' && c != EOF) { }
        if (c == EOF) { printf("\n"); exit(0); }
    }
}

static void wr32(uint32_t a, uint32_t v)
{
    if (a & 3u) { fprintf(stderr, "\nBUG: unaligned write avs 0x%04x\n", a); exit(3); }
    *(volatile uint32_t *)(accel + AVS(a)) = v;
}

static uint32_t rd32(uint32_t a)
{
    if (a & 3u) { fprintf(stderr, "\nBUG: unaligned read avs 0x%04x\n", a); exit(3); }
    return *(volatile uint32_t *)(accel + AVS(a));
}

static uint64_t ns_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static uint32_t pack(const int8_t *b)
{
    return ((uint32_t)(uint8_t)b[0])
         | ((uint32_t)(uint8_t)b[1] << 8)
         | ((uint32_t)(uint8_t)b[2] << 16)
         | ((uint32_t)(uint8_t)b[3] << 24);
}

static int dma_go(uint32_t src_phys, uint32_t dst, uint32_t bytes)
{
    dma[DMA_STATUS]     = 0;       /* DONE is sticky */
    dma[DMA_READ_ADDR]  = src_phys;
    dma[DMA_WRITE_ADDR] = dst;
    dma[DMA_LENGTH]     = bytes;
    __sync_synchronize();
    dma[DMA_CONTROL] = CTRL_WORD | CTRL_LEEN | CTRL_GO;  /* WCON off */

    uint64_t t0 = ns_now();
    for (;;) {
        uint32_t st = dma[DMA_STATUS];
        if (st & DMA_ST_DONE) return 0;
        if (ns_now() - t0 > 1000000000ull) {
            printf("    DMA TIMEOUT, STATUS = 0x%08x\n", st);
            return -1;
        }
    }
}

/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    int         pio_weights = 0;
    unsigned    seed = 1;
    const char *dumpfile = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-p")) pio_weights = 1;
        else if (!strcmp(argv[i], "-s")) step_mode = 1;
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) seed = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dump") && i + 1 < argc) dumpfile = argv[++i];
        else { printf("usage: %s [-p] [-s] [-r seed] [--dump file]\n", argv[0]); return 1; }
    }

    setvbuf(stdout, NULL, _IONBF, 0);
    signal(SIGBUS, on_bus_error);
    signal(SIGSEGV, on_bus_error);

    /* ---- quantization flow, mirroring the testbench ---- */
    static double Wr[K * N], Xr[N * M], Yr[K * M];
    static int8_t W[K * N], X[N * M], Y[K * M], G[K * M];

    srand(seed);
    for (int i = 0; i < K * N; i++) Wr[i] = randn();
    for (int i = 0; i < N * M; i++) Xr[i] = randn();

    double w_scale = quantize_sym(Wr, W, K * N);
    double x_scale = quantize_sym(Xr, X, N * M);

    for (int k = 0; k < K; k++)
        for (int m = 0; m < M; m++) {
            double acc = 0.0;
            for (int p = 0; p < N; p++) acc += Wr[k * N + p] * Xr[p * M + m];
            Yr[k * M + m] = acc;
        }

    int8_t dummy[K * M];
    double y_scale = quantize_sym(Yr, dummy, K * M);

    double combined = w_scale * x_scale / y_scale;
    if (!(combined >= 0.0 && combined < 1.0)) {
        fprintf(stderr, "combined_scale = %g does not fit Q0.16 [0,1)\n", combined);
        return 1;
    }
    uint32_t out_scale = (uint32_t)lrint(combined * 65536.0);

    golden(W, X, G, out_scale);

    int sat = 0;
    for (int i = 0; i < K * M; i++) if (G[i] == 127 || G[i] == -128) sat++;

    printf("matmul32_quant: %dx%d @ %dx%d, per-tensor symmetric int8\n",
           K, N, N, M);
    printf("  w_scale=%.8g  x_scale=%.8g  y_scale=%.8g\n",
           w_scale, x_scale, y_scale);
    printf("  combined=%.8g -> OUTPUT_SCALE=%u (0x%04x)\n",
           combined, out_scale, out_scale);
    printf("  golden: %d/%d elements saturated, weights via %s\n",
           sat, K * M, pio_weights ? "PIO" : "DMA");

    if (dumpfile) {
        FILE *f = fopen(dumpfile, "w");
        if (!f) { perror("dump"); return 1; }
        fprintf(f, "# K N M OUTPUT_SCALE\n%d %d %d %u\n", K, N, M, out_scale);
        fprintf(f, "# W_int8 (K rows of N)\n");
        for (int k = 0; k < K; k++) {
            for (int n = 0; n < N; n++) fprintf(f, "%d ", W[k * N + n]);
            fprintf(f, "\n");
        }
        fprintf(f, "# X_int8 (N rows of M)\n");
        for (int p = 0; p < N; p++) {
            for (int m = 0; m < M; m++) fprintf(f, "%d ", X[p * M + m]);
            fprintf(f, "\n");
        }
        fclose(f);
        printf("  matrices written to %s\n", dumpfile);
    }

    /* ---- map ---- */
    phase("map /dev/mem regions");
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/mem (run as root)"); return 1; }

    void *amap = mmap(NULL, ACCEL_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)ACCEL_PHYS);
    void *lmap = mmap(NULL, LW_SPAN,    PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)LW_BASE);
    void *smap = mmap(NULL, SRC_SPAN,   PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)SRC_PHYS);
    if (amap == MAP_FAILED || lmap == MAP_FAILED || smap == MAP_FAILED) {
        perror("mmap"); return 1;
    }
    accel = (volatile uint8_t *)amap;
    dma   = (volatile uint32_t *)((char *)lmap + DMA_OFF);
    src   = (volatile uint8_t *)smap;
    printf("    accel %p  dma %p  src %p\n", amap, (void *)dma, smap);

    phase("read STATUS");
    printf("    STATUS = 0x%08x\n", rd32(CTRL_BASE + REG_STATUS));

    phase("program dimension registers");
    wr32(CTRL_BASE + REG_WEIGHT_ROWS, K);
    wr32(CTRL_BASE + REG_WEIGHT_COLS, N);
    wr32(CTRL_BASE + REG_INPUT_MAX_ADDR, M);
    wr32(CTRL_BASE + REG_INPUT_COLS, M);
    wr32(CTRL_BASE + REG_OUTPUT_SCALE, out_scale);

    uint64_t t0 = ns_now();

    if (pio_weights) {
        phase("write weight buffer by PIO");
        for (int i = 0; i < K * N; i += 4)
            wr32(WBUF_BASE + (uint32_t)i, pack(&W[i]));
    } else {
        phase("stage weights in DDR, DMA into WBUF");
        for (int i = 0; i < K * N; i++) src[i] = (uint8_t)W[i];
        __sync_synchronize();
        if (dma_go((uint32_t)SRC_PHYS, DMA_ACCEL_BASE + WBUF_BASE, K * N) != 0)
            return 1;
    }
    printf("    %d weight bytes delivered\n", K * N);

    uint64_t t1 = ns_now();

    /* inputs: bank = p % ARRAY_COLS (contraction axis), bands appended */
    phase("write input buffer by PIO");
    for (int bank = 0; bank < ARRAY_COLS; bank++) {
        int8_t img[NUM_BANDS * M];
        size_t len = 0;
        for (int nb = 0; nb < NUM_BANDS; nb++) {
            int p = nb * ARRAY_COLS + bank;
            for (int m = 0; m < M; m++) img[len++] = X[p * M + m];
        }
        uint32_t base = IBUF_BASE + (uint32_t)bank * IBUF_BANK_DEPTH;
        for (size_t i = 0; i < len; i += 4) wr32(base + (uint32_t)i, pack(&img[i]));
    }
    printf("    %d input bytes written\n", N * M);

    uint64_t t2 = ns_now();

    phase("start and poll");
    (void)rd32(CTRL_BASE + REG_STATUS);
    wr32(CTRL_BASE + REG_CONTROL, 1);

    uint32_t st = 0;
    int      polls = 0;
    uint64_t tp = ns_now();
    while (polls < 1000000) {
        st = rd32(CTRL_BASE + REG_STATUS);
        polls++;
        if (st & ST_DONE) break;
        if (ns_now() - tp > 1000000000ull) break;
    }
    uint64_t t3 = ns_now();
    if (!(st & ST_DONE)) { printf("    TIMEOUT, STATUS=0x%08x\n", st); return 1; }
    printf("    done after %d polls\n", polls);

    phase("read output buffer");
    for (int k = 0; k < K; k++) {
        uint32_t base = OBUF_BASE
                      + (uint32_t)(k % ARRAY_ROWS) * OBUF_BANK_DEPTH
                      + (uint32_t)(k / ARRAY_ROWS) * M;
        for (int m = 0; m < M; m += 4) {
            uint32_t w = rd32(base + (uint32_t)m);
            for (int j = 0; j < 4; j++)
                Y[k * M + m + j] = (int8_t)(w >> (8 * j));
        }
    }
    uint64_t t4 = ns_now();

    phase("verify");
    int bad = 0, maxerr = 0;
    for (int k = 0; k < K; k++) {
        for (int m = 0; m < M; m++) {
            int g = Y[k * M + m], e = G[k * M + m];
            if (g != e) {
                int d = abs(g - e);
                if (d > maxerr) maxerr = d;
                if (bad < 10) {
                    uint32_t a = OBUF_BASE
                               + (uint32_t)(k % ARRAY_ROWS) * OBUF_BANK_DEPTH
                               + (uint32_t)(k / ARRAY_ROWS) * M + (uint32_t)m;
                    printf("    Y[%2d][%2d]: got %4d want %4d  "
                           "(bank %2d seg %d, avs 0x%04x)\n",
                           k, m, g, e, k % ARRAY_ROWS, k / ARRAY_ROWS, a);
                }
                bad++;
            }
        }
    }
    if (bad) printf("    FAIL: %d/%d mismatches, max |err| = %d\n", bad, K * M, maxerr);
    else     printf("    PASS: all %d elements match\n", K * M);

    printf("\nTiming:\n");
    printf("  weights (%s)  %8.2f us   %5d B\n",
           pio_weights ? "PIO" : "DMA", (double)(t1 - t0) / 1e3, K * N);
    printf("  inputs  (PIO)  %8.2f us   %5d B\n", (double)(t2 - t1) / 1e3, N * M);
    printf("  start -> done  %8.2f us\n", (double)(t3 - t2) / 1e3);
    printf("  readback (PIO) %8.2f us   %5d B\n", (double)(t4 - t3) / 1e3, K * M);
    printf("  total          %8.2f us\n", (double)(t4 - t0) / 1e3);
    printf("\n  %d MACs -> %.3f GOP/s compute-only\n",
           K * N * M, 2.0 * K * N * M / (double)(t3 - t2));

    munmap(amap, ACCEL_SPAN);
    munmap(lmap, LW_SPAN);
    munmap(smap, SRC_SPAN);
    close(fd);
    return bad ? 1 : 0;
}