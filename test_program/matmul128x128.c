/*
 * matmul128.c - 128x128 @ 128x128 int8 matmul on matmul_accel_0.
 *               128-bit data path, hardware address remap, expanded buffers.
 *
 * ---------------------------------------------------------------------------
 * What changed from the 32x32 program
 * ---------------------------------------------------------------------------
 * All three buffers are now 16 banks x 1024 B, and OBUF moved to 0x9000 with
 * a 1024-byte bank stride. 128x128x128 lands exactly at capacity on all
 * three at once:
 *
 *     WBUF  ceil(K/16)*pitch(N) = 8*128 = 1024   of 1024
 *     IBUF  ceil(N/16)*pitch(M) = 8*128 = 1024   of 1024
 *     OBUF  ceil(K/16)*M        = 8*128 = 1024   of 1024
 *
 * So this is the biggest square shape the buffers hold - raising any one of
 * K, N, M overflows two of the three. The static asserts below say which.
 *
 * ---------------------------------------------------------------------------
 * Buffer layout (unchanged - buf_addr_remap does the banking)
 * ---------------------------------------------------------------------------
 *     weights  WBUF_BASE + k*pitch(N) + n        (K rows)
 *     inputs   IBUF_BASE + p*pitch(M) + m        (N rows, p = contraction)
 *     pitch(c) = roundup(c, 16)
 *
 * Hardware scatters row r into bank r % 16 using WEIGHT_COLS and
 * INPUT_MAX_ADDR, so THOSE REGISTERS MUST BE WRITTEN FIRST. At 128 the pitch
 * is exactly 128, so both images are dense: 16384 B moved for 16384 live.
 *
 * Outputs now have a read remap too, so the host sees ONE contiguous
 * row-major image like the other two:
 *
 *     outputs  OBUF_BASE + k*pitch(M) + m        (K rows)
 *
 * The banking is still there underneath - row k lives in bank k % 16 at
 * segment (k/16)*pitch(M) - but the remap hides it, so dma_1 reads
 * K*pitch(M) bytes in one descriptor and they land in DDR already in
 * order. No de-interleaving on the host, and the transfer is only as long
 * as the live data rather than the whole bank window.
 *
 * ---------------------------------------------------------------------------
 * Accuracy note
 * ---------------------------------------------------------------------------
 * golden() reproduces output_loader's truncating requantize bit for bit, so
 * PASS means the datapath is right, NOT that the result is accurate. The
 * flooring shift biases every contraction band by -0.5 LSB and the biases
 * add: at 8 bands expect about -4 LSB. The program reports that separately,
 * against an ideally-quantized reference, so the bias is visible without
 * affecting pass/fail.
 *
 * gcc -O2 -Wall -mcpu=cortex-a9 -mfpu=neon -o matmul128 matmul128.c -lm
 *   -i        identity test (W=2I, scale 0.5, Y must equal X) instead of quant
 *   -p        PIO everywhere: CPU stores in, CPU loads out, neither DMA used
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
#define OBUF_BASE 0x9000u

/* register index; byte offset is index*16 (one register per 128-bit beat,
 * `alignment = 16` in rdl/ctrl_reg.rdl) */
#define REG_CONTROL        0
#define REG_WEIGHT_ROWS    1
#define REG_WEIGHT_COLS    2
#define REG_INPUT_MAX_ADDR 3
#define REG_INPUT_COLS     4
#define REG_STATUS         5
#define REG_OUTPUT_SCALE   6
#define REG_STRIDE         16

#define ST_DONE 0x2u
#define BEAT    16u

/* ---- DMA, LW bridge ----------------------------------------------
 *
 * Two controllers, control ports adjacent in the LW window:
 *   dma_0 at +0x00 - read_master DDR, write_master accel   (W and X in)
 *   dma_1 at +0x20 - read_master accel, write_master DDR   (Y out)
 * They point opposite ways, so dma_1 cannot help with the input transfers;
 * parallel W/X loading needs a third controller or a second write port.
 */
#define LW_BASE 0xFF200000UL
#define LW_SPAN 0x00200000UL
#define DMA0_OFF 0x00u
#define DMA1_OFF 0x20u

#define DMA_STATUS     0
#define DMA_READ_ADDR  1
#define DMA_WRITE_ADDR 2
#define DMA_LENGTH     3
#define DMA_CONTROL    6

#define DMA_ST_DONE   0x1u
#define DMA_ST_BUSY   0x2u
#define CTRL_GO       (1u << 3)
#define CTRL_LEEN     (1u << 7)
#define CTRL_QUADWORD (1u << 11)

#define WATCHDOG_S 15   /* wall-clock backstop; a good run is well under 1 s */

/* Accelerator base as each DMA's own master sees it - these are separate
 * address spaces, so check BOTH the dma_0.write_master and the
 * dma_1.read_master columns of the Platform Designer address map. They
 * need not agree with each other or with the HPS view. */
#define DMA_ACCEL_BASE  0x00000000u   /* dma_0.write_master -> accel */
#define DMA1_ACCEL_BASE 0x00000000u   /* dma_1.read_master  -> accel */

#define SRC_PHYS 0x30000000UL
#define SRC_SPAN 0x00100000UL
#define STAGE_W  0x00000u
#define STAGE_X  0x08000u            /* 32 KB apart: W is 16 KB now */
#define STAGE_Y  0x10000u            /* output landing zone, 16 KB */

/* ---- shape ------------------------------------------------------- */
#define ARRAY_ROWS      16
#define ARRAY_COLS      16
#define BANK_DEPTH      1024         /* all three buffers, per bank */
#define OBUF_BANK_DEPTH BANK_DEPTH

#define K 128
#define N 128
#define M 128

#define NUM_BANDS (N / ARRAY_COLS)   /* contraction blocks */
#define CEILDIV(a, b) (((a) + (b) - 1) / (b))
#define PITCH(c)  (CEILDIV(c, BEAT) * BEAT)

#define WPITCH PITCH(N)              /* weight row pitch */
#define IPITCH PITCH(M)              /* input  row pitch */
#define OPITCH PITCH(M)              /* output row pitch, after the remap */
#define WLEN   ((size_t)K * WPITCH)
#define ILEN   ((size_t)N * IPITCH)
#define OLEN   ((size_t)K * OPITCH)  /* exactly what dma_1 has to move */

/* The testbench's three capacity assertions, checked at compile time so a
 * shape that cannot fit fails here instead of scribbling across banks. */
_Static_assert(CEILDIV(K, ARRAY_ROWS) * PITCH(N) <= BANK_DEPTH,
               "weight buffer: ceil(K/16)*pitch(N) exceeds the bank depth");
_Static_assert(CEILDIV(N, ARRAY_ROWS) * PITCH(M) <= BANK_DEPTH,
               "input buffer: ceil(N/16)*pitch(M) exceeds the bank depth");
_Static_assert(CEILDIV(K, ARRAY_ROWS) * OPITCH <= OBUF_BANK_DEPTH,
               "output buffer: ceil(K/16)*pitch(M) exceeds the bank depth");
_Static_assert(N % ARRAY_COLS == 0,
               "contraction axis cannot be masked in hardware; pad N");
/* With M a multiple of 16, pitch(M) == M, so the row-major output image is
 * dense and every row base is beat aligned. If you ever run an unaligned M,
 * check that obuf_row_addr() in the testbench still means k*pitch(M) and
 * give beat_rd a tail path - the two disagree about nothing while this
 * holds, and about every row once it does not. */
_Static_assert(M % BEAT == 0,
               "M must be a multiple of 16 so output rows stay beat aligned");
_Static_assert(STAGE_X >= WLEN && STAGE_Y >= STAGE_X + ILEN
                   && STAGE_Y + OLEN <= SRC_SPAN,
               "staging regions overlap or run past the staging window");
_Static_assert(OBUF_BASE + (size_t)ARRAY_ROWS * OBUF_BANK_DEPTH <= ACCEL_SPAN,
               "output buffer runs past the mapped accelerator span");

static volatile uint8_t  *accel;
static volatile uint32_t *dma0;   /* DDR   -> accel */
static volatile uint32_t *dma1;   /* accel -> DDR   */
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

/* >> 16 floors, exactly like output_loader's `>>>`. That is where the
 * per-band -0.5 LSB bias comes from; see the accuracy note at the top. */
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

/* ------------------------------------------------------------------ */
/* teardown                                                            */
/*                                                                     */
/* A DMA still holding GO keeps driving the fabric after this process  */
/* is gone, and nothing else can use the bridge until something clears */
/* it. So every exit path runs through here: normal return, early      */
/* failure, Ctrl-C, bus error, and the wall-clock watchdog.            */
/* ------------------------------------------------------------------ */

static void dma_stop(volatile uint32_t *d)
{
    if (!d) return;
    d[DMA_CONTROL] = 0;                 /* drop GO before anything else */
    /* Bounded, and small: each read is an LW-bridge round trip of a few
     * hundred ns, and GO is already gone, so this only lets an in-flight
     * burst retire. Never spin here - this is the path that has to work
     * when everything else is wedged. */
    for (int i = 0; i < 10000; i++)
        if (!(d[DMA_STATUS] & DMA_ST_BUSY)) break;
    d[DMA_LENGTH] = 0;
    d[DMA_STATUS] = 0;                      /* DONE and the EOP bits stick */
}

static void dma_cleanup(void)
{
    dma_stop(dma0);
    dma_stop(dma1);
}

/* async-signal-safe: MMIO stores, write(2), _exit */
static void on_fatal(int sig)
{
    dma_cleanup();
    const char *m =
        (sig == SIGALRM) ? "\n*** watchdog expired: both DMAs stopped, exiting.\n"
      : (sig == SIGBUS)  ? "\n*** SIGBUS: access rejected. Both DMAs stopped.\n"
      : (sig == SIGSEGV) ? "\n*** SIGSEGV. Both DMAs stopped.\n"
                         : "\n*** interrupted: both DMAs stopped.\n";
    ssize_t r = write(2, m, strlen(m));
    (void)r;
    _exit(sig == SIGALRM ? 3 : 2);
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

/* One descriptor on `d`, both addresses incrementing, 128-bit beats. */
static int dma_go(volatile uint32_t *d, const char *who,
                  uint32_t rd_addr, uint32_t wr_addr, uint32_t bytes)
{
    d[DMA_STATUS]     = 0;              /* DONE is sticky */
    d[DMA_READ_ADDR]  = rd_addr;
    d[DMA_WRITE_ADDR] = wr_addr;
    d[DMA_LENGTH]     = bytes;
    __sync_synchronize();
    d[DMA_CONTROL] = CTRL_QUADWORD | CTRL_LEEN | CTRL_GO;  /* RCON/WCON off */

    uint64_t t0 = ns_now();
    for (;;) {
        uint32_t st = d[DMA_STATUS];
        if (st & DMA_ST_DONE) return 0;
        if (ns_now() - t0 > 1000000000ull) {
            printf("  %s TIMEOUT: STATUS 0x%08x (busy=%u done=%u),"
                   " LENGTH left %u of %u\n",
                   who, st, (st & DMA_ST_BUSY) ? 1u : 0u,
                   (st & DMA_ST_DONE) ? 1u : 0u, d[DMA_LENGTH], bytes);
            printf("  rd 0x%08x -> wr 0x%08x. Check that base against the\n"
                   "  matching master's column of the address map, and that\n"
                   "  LENGTH is wide enough for %u.\n", rd_addr, wr_addr, bytes);
            dma_stop(d);
            printf("  stopped: CONTROL cleared, STATUS now 0x%08x\n",
                   d[DMA_STATUS]);
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
    signal(SIGBUS, on_fatal);
    signal(SIGSEGV, on_fatal);

    static int8_t W[K * N], X[N * M], Y[K * M], G[K * M];
    static int8_t ideal[K * M];        /* ideally-quantized reference */
    uint32_t out_scale;
    int have_ideal = 0;

    if (identity) {
        /* W = 2I with scale 0.5 makes the datapath an exact identity:
         * requant(2x, 0.5) == x for every int8 x, and the seven other
         * contraction bands contribute a clean zero. Y must equal X. */
        memset(W, 0, sizeof W);
        for (int k = 0; k < K; k++) W[k * N + (k % N)] = 2;
        for (int p = 0; p < N; p++)
            for (int m = 0; m < M; m++)
                X[p * M + m] = (int8_t)(((p * M + m) % 251) - 125);
        out_scale = 32768;
        printf("matmul128 IDENTITY: W=2I, scale 0.5, expect Y == X\n");
    } else {
        static double Wr[K * N], Xr[N * M], Yr[K * M];
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
        /* quantizing the float product gives both the output scale and the
         * ideal int8 result the hardware is trying to approximate */
        double ys = quantize_sym(Yr, ideal, K * M);
        have_ideal = 1;
        double combined = ws * xs / ys;
        if (!(combined >= 0.0 && combined < 1.0)) {
            fprintf(stderr, "combined scale %g does not fit Q0.16\n", combined);
            return 1;
        }
        out_scale = (uint32_t)lrint(combined * 65536.0);
        printf("matmul128 QUANT: w=%.6g x=%.6g y=%.6g combined=%.6g\n",
               ws, xs, ys, combined);
    }

    golden(W, X, G, out_scale);
    int sat = 0;
    for (int i = 0; i < K * M; i++) if (G[i] == 127 || G[i] == -128) sat++;
    printf("  %dx%d @ %dx%d  OUTPUT_SCALE=%u  %d bands  %d/%d saturated"
           "  transfer=%s\n",
           K, N, N, M, out_scale, NUM_BANDS, sat, K * M, pio ? "PIO" : "DMA");
    printf("  images: W %zu B, X %zu B, Y %zu B (pitch %d/%d/%d, %s)\n",
           WLEN, ILEN, OLEN, WPITCH, IPITCH, OPITCH,
           (WPITCH == N && IPITCH == M && OPITCH == M) ? "dense"
                                                       : "rows padded");
    printf("  bank use: W %d, X %d, Y %d of %d per bank\n",
           CEILDIV(K, ARRAY_ROWS) * WPITCH, CEILDIV(N, ARRAY_ROWS) * IPITCH,
           CEILDIV(K, ARRAY_ROWS) * OPITCH, BANK_DEPTH);

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
    dma0  = (volatile uint32_t *)((volatile uint8_t *)lmap + DMA0_OFF);
    dma1  = (volatile uint32_t *)((volatile uint8_t *)lmap + DMA1_OFF);
    src   = (volatile uint8_t *)smap;

    /* Recover from a previous run that died mid-transfer, before issuing
     * anything of our own - otherwise this run inherits its mess. Then make
     * sure we cannot leave the same mess behind, whichever way we exit. */
    dma_cleanup();
    atexit(dma_cleanup);
    signal(SIGINT,  on_fatal);
    signal(SIGTERM, on_fatal);
    signal(SIGHUP,  on_fatal);
    signal(SIGQUIT, on_fatal);
    signal(SIGALRM, on_fatal);
    alarm(WATCHDOG_S);

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
    } else if (dma_go(dma0, "dma_0 weights", (uint32_t)(SRC_PHYS + STAGE_W),
                      DMA_ACCEL_BASE + WBUF_BASE, (uint32_t)WLEN) != 0) {
        return 1;
    }
    uint64_t t1 = ns_now();

    if (pio) {
        pio_image(IBUF_BASE, ximg, ILEN);
    } else if (dma_go(dma0, "dma_0 inputs", (uint32_t)(SRC_PHYS + STAGE_X),
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
    while (polls < 10000000) {
        st = reg_rd(REG_STATUS);
        polls++;
        if (st & ST_DONE) break;
        if (ns_now() - tp > 2000000000ull) break;
    }
    uint64_t t3 = ns_now();
    if (!(st & ST_DONE)) {
        printf("  TIMEOUT after %d polls, STATUS = 0x%08x\n", polls, st);
        printf("  busy=%u done=%u - if busy never set, CONTROL.start never"
               " pulsed; if it never clears, weight_ctrl is stuck\n",
               st & 1u, (st >> 1) & 1u);
        printf("  %d k-blocks x %d n-blocks = %d passes expected\n",
               CEILDIV(K, ARRAY_ROWS), NUM_BANDS,
               CEILDIV(K, ARRAY_ROWS) * NUM_BANDS);
        return 1;
    }

    /* ---- readback ----
     *
     * The read remap makes OBUF a contiguous row-major image, so this is
     * one descriptor of K*pitch(M) bytes and the rows arrive in order -
     * no bank arithmetic, and only the live bytes move rather than the
     * whole bank window. Underneath, row k is still bank k%16 segment
     * (k/16)*pitch(M); a failure confined to rows sharing k%16 still
     * points at one bank, which is why the mismatch lines print it.
     *
     * The two phases are timed apart on purpose. The DMA is the number that
     * matters in deployment, where the next layer's DMA would read this
     * straight out of DDR and the CPU would never touch it. The host read
     * after it exists only so this program can verify. */
    if (!pio && dma_go(dma1, "dma_1 readback", DMA1_ACCEL_BASE + OBUF_BASE,
                       (uint32_t)(SRC_PHYS + STAGE_Y),
                       (uint32_t)OLEN) != 0) {
        return 1;
    }
    uint64_t t4 = ns_now();

    volatile uint8_t *obuf = pio ? (accel + OBUF_BASE) : (src + STAGE_Y);
    for (int k = 0; k < K; k++) {
        size_t row = (size_t)k * OPITCH;
        for (int m = 0; m < M; m += BEAT)
            beat_rd(obuf + row + m, &Y[k * M + m]);
    }
    uint64_t t5 = ns_now();

    /* ---- verify against the bit-exact model ---- */
    int bad = 0, maxerr = 0;
    static int row_bad[K];
    memset(row_bad, 0, sizeof row_bad);
    for (int k = 0; k < K; k++)
        for (int m = 0; m < M; m++) {
            int g = Y[k * M + m], e = G[k * M + m];
            if (g == e) continue;
            int d = abs(g - e);
            if (d > maxerr) maxerr = d;
            row_bad[k]++;
            if (bad < 10)
                printf("  Y[%3d][%3d]: got %4d want %4d  (obuf 0x%04zx,"
                       " bank %2d seg %d)\n",
                       k, m, g, e, (size_t)(OBUF_BASE + k * OPITCH + m),
                       k % ARRAY_ROWS, k / ARRAY_ROWS);
            bad++;
        }

    if (bad) {
        printf("  FAIL: %d/%d wrong, max |err| = %d\n", bad, K * M, maxerr);
        int nrows = 0, allrows = 1;
        for (int k = 0; k < K; k++) {
            if (row_bad[k]) nrows++;
            if (row_bad[k] != M) allrows = 0;
        }
        printf("  %d/%d rows affected%s\n", nrows, K,
               allrows ? ", every row entirely wrong - layout or dims-before-"
                         "buffers ordering"
                       : nrows <= ARRAY_ROWS ? ", few rows - suspect one bank"
                                             : "");
        printf("  rows wrong:");
        for (int k = 0; k < K; k++) if (row_bad[k]) printf(" %d", k);
        printf("\n");
    } else {
        printf("  PASS: all %d elements match the model\n", K * M);
    }

    /* ---- and separately, how far the model itself sits from ideal ----
     * A PASS above only says the hardware matches its own truncating
     * requantize. This is the part that says whether the arithmetic is
     * worth trusting at this many bands. */
    if (have_ideal) {
        double sum = 0.0, sumsq = 0.0;
        int    worst = 0;
        for (int i = 0; i < K * M; i++) {
            int d = (int)Y[i] - (int)ideal[i];
            sum += d; sumsq += (double)d * d;
            if (abs(d) > worst) worst = abs(d);
        }
        double mean = sum / (K * M);
        printf("\n  vs ideal quantization: mean %+.2f  rms %.2f  worst %d LSB\n",
               mean, sqrt(sumsq / (K * M)), worst);
        printf("  (%d bands x -0.5 LSB from the flooring shift predicts %+.1f;"
               " round-half-up in output_loader removes it)\n",
               NUM_BANDS, -0.5 * NUM_BANDS);
    }

    /* ---- timing ----
     * `total` runs from the first byte leaving DDR to the last result
     * landing back in DDR. The host read that follows is verification
     * scaffolding, so it is reported but not counted. */
    uint64_t rb    = pio ? (t5 - t4) : (t4 - t3);
    uint64_t total = (pio ? t5 : t4) - t0;
    printf("\n  stage    %8.2f us   weights  %8.2f us   inputs %8.2f us\n",
           (double)(t0 - ts0) / 1e3, (double)(t1 - t0) / 1e3,
           (double)(t2 - t1) / 1e3);
    printf("  compute  %8.2f us   readback %8.2f us (%s)\n",
           (double)(t3 - t2) / 1e3, (double)rb / 1e3,
           pio ? "PIO, accel -> host" : "dma_1, accel -> DDR");
    if (!pio)
        printf("  host rd  %8.2f us   %zu B from staging DDR"
               "   (verify only, not in total)\n",
               (double)(t5 - t4) / 1e3, OLEN);
    printf("  --------------------------------------------------------\n");
    printf("  total    %8.2f us\n", (double)total / 1e3);
    printf("  %d MACs: %.3f GOP/s compute, %.3f GOP/s end to end\n",
           K * N * M, 2.0 * K * N * M / (double)(t3 - t2),
           2.0 * K * N * M / (double)total);
    printf("  readback is %.0f%% of total\n",
           100.0 * (double)rb / (double)total);

    alarm(0);
    dma_cleanup();
    dma0 = dma1 = NULL;      /* so the atexit copy cannot touch a dead map */
    munmap(amap, ACCEL_SPAN);
    munmap(lmap, LW_SPAN);
    munmap(smap, SRC_SPAN);
    close(fd);
    return bad ? 1 : 0;
}