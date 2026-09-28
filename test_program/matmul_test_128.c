/*
 * matmul32_128.c - 32x32 @ 32x32 int8 matmul on matmul_accel_0 with the
 *                  128-bit data path. Weights AND inputs move by DMA in
 *                  QUADWORD (128-bit) beats; results are read back by PIO.
 *
 * ---------------------------------------------------------------------------
 * ASSUMPTIONS - each one has a flag or a diagnostic, because every
 * addressing assumption on this board has cost a rebuild at some point
 * ---------------------------------------------------------------------------
 *
 * A. Address units are SYMBOLS: avs_address == CPU/DMA byte offset. With
 *    128-bit data the interconnect aligns avs_address to 16 and conveys the
 *    sub-word position through avs_byteenable.
 *
 * B. Control registers keep their 4-byte offsets (0x00..0x18) and are
 *    reached by 32-bit CPU accesses, accel_top picking the register from
 *    avs_byteenable within the 128-bit word. If you instead put one register
 *    per 128-bit word, run with --reg16 (offsets become 0x00,0x10,...,0x60).
 *    Symptom of getting this wrong: the STATUS poll times out rather than
 *    hanging, and the timeout diagnostic prints the lanes around STATUS.
 *
 * C. dma_0 was built with a 128-bit data width and write_master reaches the
 *    accelerator at DMA_ACCEL_BASE. QUADWORD mode needs source, destination
 *    and length all multiples of 16 - checked before every transfer.
 *
 * D. Buffer byte layouts are unchanged in BYTE terms: weights flat row-major
 *    at WBUF_BASE; input bank b at IBUF_BASE + 256*b with bands appended;
 *    output row k in bank k%16 at OBUF_BASE + 128*(k%16) + (k/16)*M.
 *
 * E. -p writes by PIO at the -w width. At -w 128 each store is a full,
 *    aligned 16-byte beat; at -w 32 or -w 64 it is a partial write into a
 *    128-bit port, correct only if the port honours avs_byteenable. So
 *    "-p -w 32" doubles as a test of partial-word write handling.
 *
 * Results come back by PIO at the -w width (default 128, one NEON vld1 per
 * 16 bytes): dma_0.read_master reaches only DDR, not the accelerator, so
 * OBUF -> DDR by DMA would need another Qsys connection.
 *
 * ---------------------------------------------------------------------------
 * Test modes
 * ---------------------------------------------------------------------------
 *   -t identity  (run this FIRST on a new bus width)
 *       W = 2I, OUTPUT_SCALE = 32768, X = near-unique values. Y == X exactly
 *       - no rounding, no saturation - so a wrong value names the byte it
 *       actually came from, which is what makes lane and packing errors
 *       legible.
 *   -t quant     (default)
 *       N(0,1) floats, per-tensor symmetric int8, OUTPUT_SCALE derived as
 *       w_scale*x_scale/y_scale - the testbench's scheme.
 *
 * gcc -O2 -Wall -mcpu=cortex-a9 -mfpu=neon -o matmul32_128 matmul32_128.c -lm
 *   (-mfpu=neon is REQUIRED for -w 128, the default)
 *   -t MODE    identity | quant
 *   -p         PIO for weights and inputs (see assumption E)
 *   -w BITS    PIO access width for readback and -p writes: 32 | 64 | 128
 *   --reg16    one control register per 128-bit word (assumption B)
 *   -r SEED    default 1
 *   -s         prompt between phases
 *   --dump F   write matrices + scale for the testbench
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

#define REG_CONTROL        0x00u
#define REG_WEIGHT_ROWS    0x04u
#define REG_WEIGHT_COLS    0x08u
#define REG_INPUT_MAX_ADDR 0x0Cu
#define REG_INPUT_COLS     0x10u
#define REG_STATUS         0x14u
#define REG_OUTPUT_SCALE   0x18u

#define ST_DONE 0x2u

#define BEAT 16u                  /* 128-bit data path, bytes per beat */

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

#define CTRL_GO       (1u << 3)
#define CTRL_LEEN     (1u << 7)
#define CTRL_QUADWORD (1u << 11)

#define DMA_ACCEL_BASE 0x00000000u

/* staging in reserved DDR; both offsets 16-byte aligned */
#define SRC_PHYS  0x1F800000UL
#define SRC_SPAN  0x00100000UL
#define STAGE_W   0x0000u
#define STAGE_X   0x2000u

/* ---- shape -------------------------------------------------------- */
#define ARRAY_ROWS      16
#define ARRAY_COLS      16
#define IBUF_BANK_DEPTH 256
#define OBUF_BANK_DEPTH 128

#define K 32
#define N 32
#define M 32

#define NUM_BANDS (N / ARRAY_COLS)
#define NUM_SEGS  ((K + ARRAY_ROWS - 1) / ARRAY_ROWS)

#define ROUND_UP(x, a) (((x) + (a) - 1) / (a) * (a))

static volatile uint8_t  *accel;
static volatile uint32_t *dma;
static volatile uint8_t  *src;
static int      step_mode = 0;
static uint32_t reg_mult  = 1;   /* 4 with --reg16 */

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

/* ---- control registers: 32-bit accesses, stride per assumption B ---- */

static void reg_wr(uint32_t off, uint32_t v)
{
    uint32_t a = CTRL_BASE + off * reg_mult;
    *(volatile uint32_t *)(accel + a) = v;
}

static uint32_t reg_rd(uint32_t off)
{
    uint32_t a = CTRL_BASE + off * reg_mult;
    return *(volatile uint32_t *)(accel + a);
}

/* ---- buffer PIO at 32, 64 or 128 bits -------------------------------
 *
 * 128-bit uses one NEON vld1/vst1 on 16 aligned bytes. The A9's port to
 * the L3 is 64-bit, so that single access leaves the core as a 2-beat
 * burst; whether it arrives at the slave as ONE 128-bit beat depends on the
 * h2f bridge upsizing it. The ':128' alignment hint makes it a single
 * aligned access (and faults, rather than silently splitting, if the
 * address is wrong).
 *
 * Full-width accesses also bypass the burst adapter's narrow-transfer
 * logic - the int_byte_cnt_narrow_reg path that was the worst timing path.
 * 32- and 64-bit accesses into a 128-bit port go through it every time.
 *
 * Inline asm rather than intrinsics: this is Device memory, and the
 * compiler must emit exactly one load/store per call, unmerged and
 * unreordered. */

static int acc_bytes = 16;   /* -w 32|64|128 */

static void check_align(uint32_t a, unsigned n, const char *what)
{
    if (a % n) {
        fprintf(stderr, "\nBUG: %s at 0x%04x not aligned to %u bytes\n", what, a, n);
        exit(3);
    }
}

/* write acc_bytes bytes from b[] to buffer byte address a */
static void buf_wr(uint32_t a, const int8_t *b)
{
    check_align(a, (unsigned)acc_bytes, "write");
    volatile uint8_t *p = accel + a;
    switch (acc_bytes) {
    case 4:
        *(volatile uint32_t *)p = pack(b);
        break;
    case 8: {
        uint64_t v;
        memcpy(&v, b, 8);
        *(volatile uint64_t *)p = v;
        break;
    }
    default:
#if HAVE_NEON
    {
        uint8x16_t v = vld1q_u8((const uint8_t *)b);
        __asm__ volatile("vst1.8 {%e0-%f0}, [%1 :128]"
                         : : "w"(v), "r"(p) : "memory");
    }
#else
        fprintf(stderr, "128-bit access needs NEON (-mfpu=neon)\n");
        exit(3);
#endif
    }
}

/* read acc_bytes bytes at buffer byte address a into b[] */
static void buf_rd(uint32_t a, int8_t *b)
{
    check_align(a, (unsigned)acc_bytes, "read");
    volatile uint8_t *p = accel + a;
    switch (acc_bytes) {
    case 4: {
        uint32_t v = *(volatile uint32_t *)p;
        for (int j = 0; j < 4; j++) b[j] = (int8_t)(v >> (8 * j));
        break;
    }
    case 8: {
        uint64_t v = *(volatile uint64_t *)p;
        memcpy(b, &v, 8);
        break;
    }
    default:
#if HAVE_NEON
    {
        uint8x16_t v;
        __asm__ volatile("vld1.8 {%e0-%f0}, [%1 :128]"
                         : "=w"(v) : "r"(p) : "memory");
        vst1q_u8((uint8_t *)b, v);
    }
#else
        fprintf(stderr, "128-bit access needs NEON (-mfpu=neon)\n");
        exit(3);
#endif
    }
}

/* ---- DMA, QUADWORD ------------------------------------------------ */

static int dma_go(uint32_t src_phys, uint32_t dst, uint32_t bytes)
{
    if ((src_phys % BEAT) || (dst % BEAT) || (bytes % BEAT) || bytes == 0) {
        fprintf(stderr, "\nBUG: QUADWORD DMA needs 16-byte alignment: "
                        "src 0x%08x dst 0x%08x len %u\n", src_phys, dst, bytes);
        exit(3);
    }
    dma[DMA_STATUS]     = 0;           /* DONE is sticky */
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
            printf("    DMA TIMEOUT, STATUS = 0x%08x, LENGTH left = %u\n",
                   st, dma[DMA_LENGTH]);
            printf("    check: dma_0 data width is 128? LEEN? write_master "
                   "connected to the accel?\n");
            return -1;
        }
    }
}

/* copy a cached image into uncached staging DDR with 32-bit stores - 4x
 * fewer bus transactions than byte stores; len must be a multiple of 4 */
static void stage_copy(uint32_t off, const int8_t *img, size_t len)
{
    volatile uint32_t *d = (volatile uint32_t *)(src + off);
    for (size_t i = 0; i < len; i += 4) d[i / 4] = pack(&img[i]);
}

/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    int         pio = 0, identity = 0;
    unsigned    seed = 1;
    const char *dumpfile = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-p")) pio = 1;
        else if (!strcmp(argv[i], "-s")) step_mode = 1;
        else if (!strcmp(argv[i], "--reg16")) reg_mult = 4;
        else if (!strcmp(argv[i], "-w") && i + 1 < argc) {
            int b = atoi(argv[++i]);
            if (b != 32 && b != 64 && b != 128) {
                fprintf(stderr, "-w must be 32, 64 or 128\n"); return 1; }
            acc_bytes = b / 8;
        }
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) seed = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dump") && i + 1 < argc) dumpfile = argv[++i];
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) {
            const char *t = argv[++i];
            if (!strcmp(t, "identity")) identity = 1;
            else if (!strcmp(t, "quant")) identity = 0;
            else { fprintf(stderr, "unknown -t %s\n", t); return 1; }
        }
        else {
            printf("usage: %s [-t identity|quant] [-p] [-w 32|64|128] "
                   "[--reg16] [-r seed] [-s] [--dump file]\n", argv[0]);
            return 1;
        }
    }

    if (acc_bytes == 16 && !HAVE_NEON) {
        fprintf(stderr, "-w 128 needs NEON: rebuild with -mfpu=neon, "
                        "or pass -w 64\n");
        return 1;
    }
    /* every row segment and bank must start on an access boundary */
    if (M % acc_bytes) {
        fprintf(stderr, "M=%d is not a multiple of the %d-byte access width; "
                        "use a narrower -w\n", M, acc_bytes);
        return 1;
    }

    setvbuf(stdout, NULL, _IONBF, 0);
    signal(SIGBUS, on_bus_error);
    signal(SIGSEGV, on_bus_error);

    /* ---- test data ---- */
    static int8_t W[K * N], X[N * M], Y[K * M], G[K * M];
    uint32_t out_scale;

    if (identity) {
        /* W = 2I, scale 0.5 -> Y == X exactly; X near-unique per element */
        memset(W, 0, sizeof W);
        for (int k = 0; k < K; k++) W[k * N + (k % N)] = 2;
        for (int p = 0; p < N; p++)
            for (int m = 0; m < M; m++)
                X[p * M + m] = (int8_t)(((p * M + m) % 251) - 125);
        out_scale = 32768;
        golden(W, X, G, out_scale);
        printf("matmul32_128: IDENTITY  W=2I, scale 0.5, expect Y == X\n");
    } else {
        static double Wr[K * N], Xr[N * M], Yr[K * M];
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
        static int8_t dummy[K * M];
        double y_scale  = quantize_sym(Yr, dummy, K * M);
        double combined = w_scale * x_scale / y_scale;
        if (!(combined >= 0.0 && combined < 1.0)) {
            fprintf(stderr, "combined_scale = %g does not fit Q0.16\n", combined);
            return 1;
        }
        out_scale = (uint32_t)lrint(combined * 65536.0);
        golden(W, X, G, out_scale);
        printf("matmul32_128: QUANT  w=%.6g x=%.6g y=%.6g combined=%.6g\n",
               w_scale, x_scale, y_scale, combined);
    }

    int sat = 0;
    for (int i = 0; i < K * M; i++) if (G[i] == 127 || G[i] == -128) sat++;
    printf("  %dx%d @ %dx%d, OUTPUT_SCALE=%u, %d/%d golden saturated\n",
           K, N, N, M, out_scale, sat, K * M);
    printf("  data path: weights+inputs %s%s, readback %d-bit PIO, "
           "registers %s stride\n",
           pio ? "PIO" : "DMA QUADWORD",
           pio ? (acc_bytes == 16 ? " (128-bit)" : acc_bytes == 8 ? " (64-bit)"
                                                                : " (32-bit)") : "",
           acc_bytes * 8, reg_mult == 1 ? "4-byte" : "16-byte");

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

    /* ---- buffer images, built once in cached memory ----
     * weights: flat row-major, byte-for-byte what WBUF holds.
     * inputs:  the IBUF window itself - bank b at 256*b, its bands
     *          concatenated, dead bytes zero - so ONE DMA descriptor fills
     *          all 16 banks. Padding is harmless: nothing past
     *          NUM_BANDS*M in a bank is ever read. The tail is trimmed to
     *          the last bank's live bytes rounded up to a beat. */
    static int8_t wimg[ROUND_UP(K * N, BEAT)];
    static int8_t ximg[ARRAY_COLS * IBUF_BANK_DEPTH];

    size_t wlen = ROUND_UP(K * N, BEAT);
    memcpy(wimg, W, (size_t)K * N);

    size_t bank_live = (size_t)NUM_BANDS * M;
    size_t xlen = (size_t)(ARRAY_COLS - 1) * IBUF_BANK_DEPTH
                + ROUND_UP(bank_live, BEAT);
    for (int bank = 0; bank < ARRAY_COLS; bank++)
        for (int nb = 0; nb < NUM_BANDS; nb++) {
            int p = nb * ARRAY_COLS + bank;
            for (int m = 0; m < M; m++)
                ximg[bank * IBUF_BANK_DEPTH + nb * M + m] = X[p * M + m];
        }

    printf("  weight image %zu B, input image %zu B "
           "(%zu live, %.0f%% efficient)\n",
           wlen, xlen, (size_t)N * M, 100.0 * N * M / (double)xlen);

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

    phase("read STATUS");
    uint32_t st0 = reg_rd(REG_STATUS);
    printf("    STATUS = 0x%08x", st0);
    if (st0 & ~0x3u)
        printf("   <- bits above [1:0] set: register lane or stride looks "
               "wrong, try --reg16");
    printf("\n");

    phase("program dimension registers");
    reg_wr(REG_WEIGHT_ROWS, K);
    reg_wr(REG_WEIGHT_COLS, N);
    reg_wr(REG_INPUT_MAX_ADDR, M);
    reg_wr(REG_INPUT_COLS, M);
    reg_wr(REG_OUTPUT_SCALE, out_scale);

    /* ---- staging: CPU -> uncached DDR. Reported separately, since in a
     * real pipeline activations would be produced straight into a DMA
     * buffer and this cost would not exist. ---- */
    uint64_t ts0 = ns_now();
    if (!pio) {
        phase("stage images in DDR");
        stage_copy(STAGE_W, wimg, wlen);
        stage_copy(STAGE_X, ximg, xlen);
        __sync_synchronize();
    }
    uint64_t ts1 = ns_now();

    /* ---- weights ---- */
    uint64_t t0 = ns_now();
    if (pio) {
        phase("weights by PIO");
        for (size_t i = 0; i < wlen; i += (size_t)acc_bytes)
            buf_wr(WBUF_BASE + (uint32_t)i, &wimg[i]);
    } else {
        phase("weights by DMA QUADWORD");
        if (dma_go((uint32_t)(SRC_PHYS + STAGE_W), DMA_ACCEL_BASE + WBUF_BASE,
                   (uint32_t)wlen) != 0) return 1;
    }
    uint64_t t1 = ns_now();

    /* ---- inputs ---- */
    if (pio) {
        phase("inputs by PIO (live bytes only)");
        for (int bank = 0; bank < ARRAY_COLS; bank++) {
            uint32_t base = IBUF_BASE + (uint32_t)bank * IBUF_BANK_DEPTH;
            for (size_t i = 0; i < ROUND_UP(bank_live, (size_t)acc_bytes);
                 i += (size_t)acc_bytes)
                buf_wr(base + (uint32_t)i, &ximg[bank * IBUF_BANK_DEPTH + i]);
        }
    } else {
        phase("inputs by DMA QUADWORD (one descriptor, whole IBUF window)");
        if (dma_go((uint32_t)(SRC_PHYS + STAGE_X), DMA_ACCEL_BASE + IBUF_BASE,
                   (uint32_t)xlen) != 0) return 1;
    }
    uint64_t t2 = ns_now();

    /* ---- compute ---- */
    phase("start and poll");
    (void)reg_rd(REG_STATUS);             /* drain stale completion */
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
        printf("    TIMEOUT after %d polls, STATUS = 0x%08x\n", polls, st);
        printf("    STATUS may be in a different lane (assumption B). The\n"
               "    registers either side of it read back as:\n");
        printf("      +0x10 = 0x%08x   +0x14 = 0x%08x   +0x18 = 0x%08x\n",
               *(volatile uint32_t *)(accel + CTRL_BASE + 0x10),
               *(volatile uint32_t *)(accel + CTRL_BASE + 0x14),
               *(volatile uint32_t *)(accel + CTRL_BASE + 0x18));
        printf("    If one of them shows done (bit 1), re-run with --reg16 or\n"
               "    check accel_top's register lane selection.\n");
        return 1;
    }
    printf("    done after %d polls\n", polls);

    /* ---- readback ---- */
    phase("read output buffer");
    for (int k = 0; k < K; k++) {
        uint32_t base = OBUF_BASE
                      + (uint32_t)(k % ARRAY_ROWS) * OBUF_BANK_DEPTH
                      + (uint32_t)(k / ARRAY_ROWS) * M;
        for (int m = 0; m < M; m += acc_bytes)
            buf_rd(base + (uint32_t)m, &Y[k * M + m]);
    }
    uint64_t t4 = ns_now();

    /* ---- verify ---- */
    phase("verify");
    int bad = 0, maxerr = 0;
    int lane_bad[BEAT];
    memset(lane_bad, 0, sizeof lane_bad);
    for (int k = 0; k < K; k++) {
        for (int m = 0; m < M; m++) {
            int g = Y[k * M + m], e = G[k * M + m];
            if (g == e) continue;
            uint32_t a = OBUF_BASE
                       + (uint32_t)(k % ARRAY_ROWS) * OBUF_BANK_DEPTH
                       + (uint32_t)(k / ARRAY_ROWS) * M + (uint32_t)m;
            lane_bad[a % BEAT]++;
            int d = abs(g - e);
            if (d > maxerr) maxerr = d;
            if (bad < 10)
                printf("    Y[%2d][%2d]: got %4d want %4d  "
                       "(bank %2d seg %d, avs 0x%04x, beat 0x%04x lane %2u)\n",
                       k, m, g, e, k % ARRAY_ROWS, k / ARRAY_ROWS,
                       a, a & ~(BEAT - 1), a % BEAT);
            bad++;
        }
    }
    if (bad) {
        printf("    FAIL: %d/%d mismatches, max |err| = %d\n", bad, K * M, maxerr);
        /* on a new bus width the lane distribution is the first thing to
         * read: errors confined to some lanes mean packing, spread evenly
         * across all 16 means something upstream of the port */
        printf("    errors by byte lane within the 128-bit beat:\n     ");
        for (unsigned j = 0; j < BEAT; j++) printf(" %2u:%-3d", j, lane_bad[j]);
        printf("\n");
    } else {
        printf("    PASS: all %d elements match\n", K * M);
    }

    /* ---- timing ---- */
    double us_w = (double)(t1 - t0) / 1e3, us_x = (double)(t2 - t1) / 1e3;
    double us_c = (double)(t3 - t2) / 1e3, us_y = (double)(t4 - t3) / 1e3;
    double us_s = (double)(ts1 - ts0) / 1e3;

    printf("\nTiming:\n");
    if (!pio)
        printf("  staging  (CPU->DDR)   %8.2f us   %5zu B   (excluded below)\n",
               us_s, wlen + xlen);
    char wtag[16];
    snprintf(wtag, sizeof wtag, pio ? "%d-bit PIO" : "DMA", acc_bytes * 8);
    printf("  weights  (%-11s) %8.2f us   %5zu B\n", wtag, us_w, wlen);
    printf("  inputs   (%-11s) %8.2f us   %5zu B%s\n",
           wtag, us_x, pio ? (size_t)N * M : xlen,
           pio ? "" : "  (incl. bank padding)");
    printf("  start -> done         %8.2f us\n", us_c);
    printf("  readback (%d-bit PIO) %8.2f us   %5d B   %.1f ns per access\n",
           acc_bytes * 8, us_y, K * M, (double)(t4 - t3) / (K * M / acc_bytes));
    printf("  ----------------------------------------------\n");
    printf("  accelerator total     %8.2f us\n", us_w + us_x + us_c + us_y);
    printf("\n  %d MACs: %.3f GOP/s compute-only, %.3f GOP/s end to end\n",
           K * N * M, 2.0 * K * N * M / (double)(t3 - t2),
           2.0 * K * N * M / (double)(t4 - t0));

    munmap(amap, ACCEL_SPAN);
    munmap(lmap, LW_SPAN);
    munmap(smap, SRC_SPAN);
    close(fd);
    return bad ? 1 : 0;
}