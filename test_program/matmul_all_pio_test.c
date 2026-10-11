/*
 * matmul32_128.c - 32x32 @ 32x32 int8 matmul on matmul_accel_0.
 *                  128-bit data path, BANKED weight buffer.
 *
 * ---------------------------------------------------------------------------
 * What changed for the banked weight buffer
 * ---------------------------------------------------------------------------
 * weight_buffer_32_bank gives every PE row its own bank so all ARRAY_ROWS
 * rows prefetch in parallel. W is no longer flat row-major:
 *
 *     W[k][n]  ->  bank (k % ARRAY_ROWS), byte offset (k / ARRAY_ROWS)*N + n
 *     bank stride = 1 << WBUF_BANK_ADDR_WIDTH = 1024 bytes
 *
 * so each bank holds ITS OWN rows of W back to back, and block (k_blk, n_blk)
 * sits at the same offset k_blk*N + n_blk*ARRAY_COLS in every bank.
 *
 * The padding consequence is severe at small K. Live bytes per bank are
 * ceil(K/ARRAY_ROWS)*N = 64 at 32x32, out of a 1024-byte bank - 6.25%. A
 * single contiguous DMA over the whole WBUF window would move 15424 bytes to
 * deliver 1024. That is why weights default to PIO here and inputs do not:
 *
 *     weights  ceil(K/16)*N  of 1024 per bank ->  6.25% at 32x32
 *     inputs   ceil(N/16)*M  of  256 per bank -> 25%    at 32x32
 *
 * Weight DMA only starts winning when the banks are nearly full, i.e.
 * ceil(K/ARRAY_ROWS)*N approaching 1024 - large K with large N. Both DMA
 * shapes are available (-W dma, -W dmabank) so the crossover can be measured
 * rather than argued about.
 *
 * ---------------------------------------------------------------------------
 * ASSUMPTIONS
 * ---------------------------------------------------------------------------
 * A. SYMBOLS address units: avs_address == CPU/DMA byte offset. With 128-bit
 *    data the interconnect aligns it to 16 and conveys the sub-word position
 *    through avs_byteenable.
 *
 * B. Control registers are spaced 16 bytes apart, one per 128-bit beat
 *    (alignment = 16 in ctrl_reg.rdl), reached with 32-bit accesses. Use
 *    --reg4 if you went back to 4-byte spacing. Getting this wrong times the
 *    STATUS poll out rather than hanging, and the timeout prints the lanes.
 *
 * C. dma_0 has a 128-bit data width. QUADWORD needs source, destination and
 *    length all multiples of 16 - checked before every transfer. If lanes
 *    0-3 of each beat land and the rest are stale, dma_0 is still 32-bit.
 *
 * D. Buffer layouts, in BYTES:
 *      weights  WBUF_BASE + 1024*(k % 16) + (k / 16)*N + n
 *      inputs   IBUF_BASE +  256*(p % 16) + (p / 16)*M + m
 *      outputs  OBUF_BASE +  128*(k % 16) + (k / 16)*M + m
 *
 * E. PIO at --bits 128 writes whole aligned beats; at 32 or 64 it is a
 *    partial write into a 128-bit port, correct only if that port honours
 *    avs_byteenable. So "--bits 32 -p" is the byte-enable test.
 *
 * Results always come back by PIO: dma_0.read_master reaches only DDR.
 *
 * ---------------------------------------------------------------------------
 * gcc -O2 -Wall -mcpu=cortex-a9 -mfpu=neon -o matmul32_128 matmul32_128.c -lm
 *   -t identity|quant   identity first on any layout change (default quant)
 *   -W pio|dma|dmabank  weight transfer (default pio - see padding above)
 *   -X pio|dma          input transfer  (default dma)
 *   -p                  shorthand for -W pio -X pio
 *   --bits 32|64|128    PIO access width (default 128)
 *   --reg4              4-byte control register spacing
 *   -r SEED  -s  --dump FILE
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

/* ---- accelerator ------------------------------------------------- */
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
#define BEAT 16u

/* ---- DMA --------------------------------------------------------- */
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

#define SRC_PHYS 0x1F800000UL
#define SRC_SPAN 0x00100000UL
#define STAGE_W  0x00000u        /* up to 16 KB */
#define STAGE_X  0x08000u        /* up to  4 KB */

/* ---- shape ------------------------------------------------------- */
#define ARRAY_ROWS      16
#define ARRAY_COLS      16
#define WBUF_BANK_DEPTH 1024     /* 1 << WBUF_BANK_ADDR_WIDTH */
#define IBUF_BANK_DEPTH 256
#define OBUF_BANK_DEPTH 128

#define K 32
#define N 32
#define M 32

#define NUM_WROWS  ((K + ARRAY_ROWS - 1) / ARRAY_ROWS)  /* W rows per bank  */
#define NUM_BANDS  ((N + ARRAY_COLS - 1) / ARRAY_COLS)  /* X bands per bank */
#define NUM_SEGS   ((K + ARRAY_ROWS - 1) / ARRAY_ROWS)  /* Y segs  per bank */

#define ROUND_UP(x, a) (((x) + (a) - 1) / (a) * (a))

enum xfer { XF_PIO, XF_DMA, XF_DMABANK };
static const char *xfer_name[] = { "PIO", "DMA", "DMA/bank" };

static volatile uint8_t  *accel;
static volatile uint32_t *dma;
static volatile uint8_t  *src;
static int      step_mode = 0;
static uint32_t reg_mult  = 4;    /* 16-byte spacing; --reg4 sets 1 */
static int      acc_bytes = 16;

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

static void reg_wr(uint32_t off, uint32_t v)
{
    *(volatile uint32_t *)(accel + CTRL_BASE + off * reg_mult) = v;
}

static uint32_t reg_rd(uint32_t off)
{
    return *(volatile uint32_t *)(accel + CTRL_BASE + off * reg_mult);
}

static void check_align(uint32_t a, unsigned n, const char *what)
{
    if (a % n) {
        fprintf(stderr, "\nBUG: %s at 0x%04x not aligned to %u\n", what, a, n);
        exit(3);
    }
}

/*
 * 128-bit accesses are one NEON vld1/vst1 on 16 aligned bytes, written as
 * inline asm because this is Device memory: exactly one load/store per call,
 * unmerged and unreordered. The ':128' hint makes it a single aligned access
 * and faults rather than silently splitting.
 *
 * Full-width accesses also bypass the burst adapter's narrow-transfer logic.
 */
static void buf_wr(uint32_t a, const int8_t *b)
{
    check_align(a, (unsigned)acc_bytes, "write");
    volatile uint8_t *p = accel + a;
    switch (acc_bytes) {
    case 4: *(volatile uint32_t *)p = pack(b); break;
    case 8: { uint64_t v; memcpy(&v, b, 8); *(volatile uint64_t *)p = v; break; }
    default:
#if HAVE_NEON
    { uint8x16_t v = vld1q_u8((const uint8_t *)b);
      __asm__ volatile("vst1.8 {%e0-%f0}, [%1 :128]" : : "w"(v), "r"(p) : "memory"); }
#else
        fprintf(stderr, "128-bit needs NEON (-mfpu=neon)\n"); exit(3);
#endif
    }
}

static void buf_rd(uint32_t a, int8_t *b)
{
    check_align(a, (unsigned)acc_bytes, "read");
    volatile uint8_t *p = accel + a;
    switch (acc_bytes) {
    case 4: { uint32_t v = *(volatile uint32_t *)p;
              for (int j = 0; j < 4; j++) b[j] = (int8_t)(v >> (8 * j));
              break; }
    case 8: { uint64_t v = *(volatile uint64_t *)p; memcpy(b, &v, 8); break; }
    default:
#if HAVE_NEON
    { uint8x16_t v;
      __asm__ volatile("vld1.8 {%e0-%f0}, [%1 :128]" : "=w"(v) : "r"(p) : "memory");
      vst1q_u8((uint8_t *)b, v); }
#else
        fprintf(stderr, "128-bit needs NEON (-mfpu=neon)\n"); exit(3);
#endif
    }
}

static int dma_go(uint32_t src_phys, uint32_t dst, uint32_t bytes)
{
    if ((src_phys % BEAT) || (dst % BEAT) || (bytes % BEAT) || bytes == 0) {
        fprintf(stderr, "\nBUG: QUADWORD needs 16-byte alignment: "
                        "src 0x%08x dst 0x%08x len %u\n", src_phys, dst, bytes);
        exit(3);
    }
    dma[DMA_STATUS]     = 0;
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
            printf("    DMA TIMEOUT, STATUS 0x%08x, LENGTH left %u\n",
                   st, dma[DMA_LENGTH]);
            printf("    check: dma_0 width 128? LEEN? write_master wired?\n");
            return -1;
        }
    }
}

/* cached image -> uncached staging DDR, 16 bytes per store where possible */
static void stage_copy(uint32_t off, const int8_t *img, size_t len)
{
    volatile uint8_t *d = src + off;
#if HAVE_NEON
    size_t i = 0;
    for (; i + 16 <= len; i += 16) {
        uint8x16_t v = vld1q_u8((const uint8_t *)img + i);
        volatile uint8_t *p = d + i;
        __asm__ volatile("vst1.8 {%e0-%f0}, [%1 :128]" : : "w"(v), "r"(p) : "memory");
    }
    for (; i < len; i++) d[i] = (uint8_t)img[i];
#else
    for (size_t i = 0; i + 4 <= len; i += 4)
        *(volatile uint32_t *)(d + i) = pack(&img[i]);
#endif
}

/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    enum xfer wmode = XF_PIO, xmode = XF_DMA;
    int         identity = 0;
    unsigned    seed = 1;
    const char *dumpfile = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-p")) { wmode = XF_PIO; xmode = XF_PIO; }
        else if (!strcmp(argv[i], "-s")) step_mode = 1;
        else if (!strcmp(argv[i], "--reg4")) reg_mult = 1;
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) seed = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dump") && i + 1 < argc) dumpfile = argv[++i];
        else if (!strcmp(argv[i], "--bits") && i + 1 < argc) {
            int b = atoi(argv[++i]);
            if (b != 32 && b != 64 && b != 128) {
                fprintf(stderr, "--bits must be 32, 64 or 128\n"); return 1; }
            acc_bytes = b / 8;
        }
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) {
            const char *t = argv[++i];
            if (!strcmp(t, "identity")) identity = 1;
            else if (!strcmp(t, "quant")) identity = 0;
            else { fprintf(stderr, "unknown -t %s\n", t); return 1; }
        }
        else if (!strcmp(argv[i], "-W") && i + 1 < argc) {
            const char *m = argv[++i];
            if (!strcmp(m, "pio")) wmode = XF_PIO;
            else if (!strcmp(m, "dma")) wmode = XF_DMA;
            else if (!strcmp(m, "dmabank")) wmode = XF_DMABANK;
            else { fprintf(stderr, "-W pio|dma|dmabank\n"); return 1; }
        }
        else if (!strcmp(argv[i], "-X") && i + 1 < argc) {
            const char *m = argv[++i];
            if (!strcmp(m, "pio")) xmode = XF_PIO;
            else if (!strcmp(m, "dma")) xmode = XF_DMA;
            else { fprintf(stderr, "-X pio|dma\n"); return 1; }
        }
        else {
            printf("usage: %s [-t identity|quant] [-W pio|dma|dmabank] "
                   "[-X pio|dma] [-p] [--bits 32|64|128] [--reg4] "
                   "[-r seed] [-s] [--dump file]\n", argv[0]);
            return 1;
        }
    }

    if (acc_bytes == 16 && !HAVE_NEON) {
        fprintf(stderr, "--bits 128 needs NEON: rebuild with -mfpu=neon\n");
        return 1;
    }
    if (M % acc_bytes || N % acc_bytes) {
        fprintf(stderr, "M=%d and N=%d must be multiples of the %d-byte "
                        "access width\n", M, N, acc_bytes);
        return 1;
    }

    setvbuf(stdout, NULL, _IONBF, 0);
    signal(SIGBUS, on_bus_error);
    signal(SIGSEGV, on_bus_error);

    /* ---- capacity ---- */
    if ((size_t)NUM_WROWS * N > WBUF_BANK_DEPTH) {
        fprintf(stderr, "ceil(K/%d)*N = %d exceeds weight bank depth %d\n",
                ARRAY_ROWS, NUM_WROWS * N, WBUF_BANK_DEPTH); return 1; }
    if ((size_t)NUM_BANDS * M > IBUF_BANK_DEPTH) {
        fprintf(stderr, "input bank overflow\n"); return 1; }
    if ((size_t)NUM_SEGS * M > OBUF_BANK_DEPTH) {
        fprintf(stderr, "output bank overflow\n"); return 1; }

    /* ---- test data ---- */
    static int8_t W[K * N], X[N * M], Y[K * M], G[K * M];
    uint32_t out_scale;

    if (identity) {
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
    printf("  weights %s, inputs %s, readback %d-bit PIO, registers %s\n",
           xfer_name[wmode], xfer_name[xmode], acc_bytes * 8,
           reg_mult == 4 ? "16-byte spaced" : "4-byte spaced");

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

    /* ---- bank-strided images, built once in cached memory ----
     *
     * WEIGHTS: bank b holds rows b, b+16, b+32 ... of W, each N bytes, back
     * to back from the start of the bank. Bank stride 1024.
     * INPUTS:  bank b holds contraction indices b, b+16 ... each M bytes.
     * Both images are byte-for-byte copies of the buffer windows, so one
     * contiguous DMA can fill every bank - at the cost of the dead space. */
    static int8_t wimg[ARRAY_ROWS * WBUF_BANK_DEPTH];
    static int8_t ximg[ARRAY_COLS * IBUF_BANK_DEPTH];

    size_t wlive = (size_t)NUM_WROWS * N;          /* per bank */
    size_t xlive = (size_t)NUM_BANDS * M;

    memset(wimg, 0, sizeof wimg);
    for (int bank = 0; bank < ARRAY_ROWS; bank++)
        for (int b = 0; b < NUM_WROWS; b++) {
            int k = b * ARRAY_ROWS + bank;
            if (k >= K) break;
            for (int n = 0; n < N; n++)
                wimg[bank * WBUF_BANK_DEPTH + b * N + n] = W[k * N + n];
        }

    memset(ximg, 0, sizeof ximg);
    for (int bank = 0; bank < ARRAY_COLS; bank++)
        for (int nb = 0; nb < NUM_BANDS; nb++) {
            int p = nb * ARRAY_COLS + bank;
            if (p >= N) break;
            for (int m = 0; m < M; m++)
                ximg[bank * IBUF_BANK_DEPTH + nb * M + m] = X[p * M + m];
        }

    /* contiguous-DMA lengths: through the last bank's live bytes */
    size_t wlen_contig = (size_t)(ARRAY_ROWS - 1) * WBUF_BANK_DEPTH
                       + ROUND_UP(wlive, BEAT);
    size_t xlen_contig = (size_t)(ARRAY_COLS - 1) * IBUF_BANK_DEPTH
                       + ROUND_UP(xlive, BEAT);

    printf("  weight banks: %zu live of %d per bank (%.1f%%), "
           "contiguous DMA would move %zu B for %d\n",
           wlive, WBUF_BANK_DEPTH, 100.0 * wlive / WBUF_BANK_DEPTH,
           wlen_contig, K * N);
    printf("  input  banks: %zu live of %d per bank (%.1f%%), "
           "contiguous DMA moves %zu B for %d\n",
           xlive, IBUF_BANK_DEPTH, 100.0 * xlive / IBUF_BANK_DEPTH,
           xlen_contig, N * M);

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
    if (st0 & ~0x3u) printf("   <- bits above [1:0] set: register spacing "
                            "looks wrong, try --reg4");
    printf("\n");

    phase("program dimension registers");
    reg_wr(REG_WEIGHT_ROWS, K);
    reg_wr(REG_WEIGHT_COLS, N);
    reg_wr(REG_INPUT_MAX_ADDR, M);
    reg_wr(REG_INPUT_COLS, M);
    reg_wr(REG_OUTPUT_SCALE, out_scale);

    /* ---- staging ---- */
    uint64_t ts0 = ns_now();
    size_t staged = 0;
    if (wmode != XF_PIO || xmode != XF_PIO) {
        phase("stage images in DDR");
        if (wmode != XF_PIO) {
            size_t n = (wmode == XF_DMA) ? wlen_contig
                                         : (size_t)ARRAY_ROWS * WBUF_BANK_DEPTH;
            stage_copy(STAGE_W, wimg, n); staged += n;
        }
        if (xmode != XF_PIO) { stage_copy(STAGE_X, ximg, xlen_contig);
                               staged += xlen_contig; }
        __sync_synchronize();
    }
    uint64_t ts1 = ns_now();

    /* ---- weights ---- */
    size_t   wmoved = 0;
    uint64_t t0 = ns_now();
    switch (wmode) {
    case XF_PIO:
        phase("weights by PIO (live bytes per bank)");
        for (int bank = 0; bank < ARRAY_ROWS; bank++) {
            uint32_t base = WBUF_BASE + (uint32_t)bank * WBUF_BANK_DEPTH;
            for (size_t i = 0; i < ROUND_UP(wlive, (size_t)acc_bytes);
                 i += (size_t)acc_bytes)
                buf_wr(base + (uint32_t)i, &wimg[bank * WBUF_BANK_DEPTH + i]);
        }
        wmoved = (size_t)ARRAY_ROWS * ROUND_UP(wlive, (size_t)acc_bytes);
        break;
    case XF_DMA:
        phase("weights by one contiguous DMA over the whole WBUF window");
        if (dma_go((uint32_t)(SRC_PHYS + STAGE_W), DMA_ACCEL_BASE + WBUF_BASE,
                   (uint32_t)wlen_contig) != 0) return 1;
        wmoved = wlen_contig;
        break;
    default:
        phase("weights by one DMA descriptor per bank");
        for (int bank = 0; bank < ARRAY_ROWS; bank++) {
            uint32_t off = (uint32_t)bank * WBUF_BANK_DEPTH;
            if (dma_go((uint32_t)(SRC_PHYS + STAGE_W + off),
                       DMA_ACCEL_BASE + WBUF_BASE + off,
                       (uint32_t)ROUND_UP(wlive, BEAT)) != 0) return 1;
        }
        wmoved = (size_t)ARRAY_ROWS * ROUND_UP(wlive, BEAT);
        break;
    }
    uint64_t t1 = ns_now();

    /* ---- inputs ---- */
    size_t xmoved = 0;
    if (xmode == XF_PIO) {
        phase("inputs by PIO (live bytes per bank)");
        for (int bank = 0; bank < ARRAY_COLS; bank++) {
            uint32_t base = IBUF_BASE + (uint32_t)bank * IBUF_BANK_DEPTH;
            for (size_t i = 0; i < ROUND_UP(xlive, (size_t)acc_bytes);
                 i += (size_t)acc_bytes)
                buf_wr(base + (uint32_t)i, &ximg[bank * IBUF_BANK_DEPTH + i]);
        }
        xmoved = (size_t)ARRAY_COLS * ROUND_UP(xlive, (size_t)acc_bytes);
    } else {
        phase("inputs by one contiguous DMA over the whole IBUF window");
        if (dma_go((uint32_t)(SRC_PHYS + STAGE_X), DMA_ACCEL_BASE + IBUF_BASE,
                   (uint32_t)xlen_contig) != 0) return 1;
        xmoved = xlen_contig;
    }
    uint64_t t2 = ns_now();

    /* ---- compute ---- */
    phase("start and poll");
    (void)reg_rd(REG_STATUS);
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
        printf("    neighbouring register words read back as:\n"
               "      +0x%02x = 0x%08x  +0x%02x = 0x%08x  +0x%02x = 0x%08x\n",
               REG_INPUT_COLS * reg_mult, reg_rd(REG_INPUT_COLS),
               REG_STATUS * reg_mult, reg_rd(REG_STATUS),
               REG_OUTPUT_SCALE * reg_mult, reg_rd(REG_OUTPUT_SCALE));
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
    int bank_bad[ARRAY_ROWS];
    memset(lane_bad, 0, sizeof lane_bad);
    memset(bank_bad, 0, sizeof bank_bad);
    for (int k = 0; k < K; k++) {
        for (int m = 0; m < M; m++) {
            int g = Y[k * M + m], e = G[k * M + m];
            if (g == e) continue;
            uint32_t a = OBUF_BASE
                       + (uint32_t)(k % ARRAY_ROWS) * OBUF_BANK_DEPTH
                       + (uint32_t)(k / ARRAY_ROWS) * M + (uint32_t)m;
            lane_bad[a % BEAT]++;
            bank_bad[k % ARRAY_ROWS]++;
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
        printf("    by byte lane in the beat:\n     ");
        for (unsigned j = 0; j < BEAT; j++) printf(" %2u:%-3d", j, lane_bad[j]);
        /* one bad PE row means one weight bank; all rows bad means the
         * layout or the transfer, not a single bank */
        printf("\n    by PE row (weight bank):\n     ");
        for (int j = 0; j < ARRAY_ROWS; j++) printf(" %2d:%-3d", j, bank_bad[j]);
        printf("\n");
    } else {
        printf("    PASS: all %d elements match\n", K * M);
    }

    /* ---- timing ---- */
    double us_w = (double)(t1 - t0) / 1e3, us_x = (double)(t2 - t1) / 1e3;
    double us_c = (double)(t3 - t2) / 1e3, us_y = (double)(t4 - t3) / 1e3;
    double us_s = (double)(ts1 - ts0) / 1e3;

    printf("\nTiming:\n");
    if (staged)
        printf("  staging  (CPU->DDR)  %8.2f us   %6zu B   (excluded below)\n",
               us_s, staged);
    printf("  weights  (%-8s)   %8.2f us   %6zu B moved for %5d live\n",
           xfer_name[wmode], us_w, wmoved, K * N);
    printf("  inputs   (%-8s)   %8.2f us   %6zu B moved for %5d live\n",
           xfer_name[xmode], us_x, xmoved, N * M);
    printf("  start -> done        %8.2f us\n", us_c);
    printf("  readback (%3d-bit)   %8.2f us   %6d B   %.1f ns per access\n",
           acc_bytes * 8, us_y, K * M, (double)(t4 - t3) / (K * M / acc_bytes));
    printf("  ---------------------------------------------------------\n");
    printf("  accelerator total    %8.2f us\n", us_w + us_x + us_c + us_y);
    printf("\n  %d MACs: %.3f GOP/s compute-only, %.3f GOP/s end to end\n",
           K * N * M, 2.0 * K * N * M / (double)(t3 - t2),
           2.0 * K * N * M / (double)(t4 - t0));

    munmap(amap, ACCEL_SPAN);
    munmap(lmap, LW_SPAN);
    munmap(smap, SRC_SPAN);
    close(fd);
    return bad ? 1 : 0;
}