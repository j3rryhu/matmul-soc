/*
 * matmul_tiled.c - host twin of tb's test_tiled_canvas_use_prev_out and
 *                  test_tiled_unaligned_col_offset.
 *
 * ---------------------------------------------------------------------------
 * What the new hardware adds
 * ---------------------------------------------------------------------------
 * TILE_CFG (0x70) places each matmul's K x M result inside a canvas:
 *
 *     canvas row    = tile_r * WEIGHT_ROWS      (= tile_r * K_tile)
 *     canvas column = tile_c * INPUT_COLS       (= tile_c * M_tile)
 *
 * TILED_MAT_DIM (0x80) is the FULL result's column count, which sets how
 * wide that canvas is. The buffer then reads back as one dense row-major
 * matrix with rows padded to a 16-byte word, so
 *
 *     host address = OBUF_BASE + row * pitch(TILED_MAT_DIM) + col
 *
 * and the whole thing is K * pitch(TILED_MAT_DIM) contiguous bytes - only
 * the live rows, not the whole 64 KB buffer. Phase 3 runs a 40-wide canvas
 * so the pitch is 48: padded, and three words rather than a power of two,
 * which is the case where a shift-based pitch would pass and a real
 * multiply/divide is needed.
 *
 * use_prev_out makes the first contraction block accumulate onto whatever
 * the canvas already holds instead of overwriting it, so a long contraction
 * can be split across several matmuls. It continues the same saturating
 * int8 chain the hardware runs between contraction blocks of one matmul, so
 * the end result is exactly the banded golden over the full N.
 *
 * Note on WEIGHT_ROWS / WEIGHT_COLS: rdl/ctrl_reg.rdl's header says
 * WEIGHT_ROWS is the contraction axis. The testbench's program_dims and
 * test_partial_row_block_matmul say the opposite - that WEIGHT_ROWS is the
 * output-row axis and weight_ctrl.v's header "describes the wrong axis" -
 * and every run that has ever passed on hardware used that convention. This
 * program follows the testbench: WEIGHT_ROWS = K, WEIGHT_COLS = N. It is
 * also the only reading under which tile_r*WEIGHT_ROWS is a row offset.
 *
 * ---------------------------------------------------------------------------
 * Phases
 * ---------------------------------------------------------------------------
 *   1  256x256 canvas, 2x2 tiles of 128x128, contraction 32 split into two
 *      halves of 16. Half-outer, tile-inner: four tiles with use_prev_out=0,
 *      then the same four with use_prev_out=1. Exercises all four quadrants
 *      and cross-matmul accumulation with three other tiles written between
 *      a tile's two halves.
 *   2  the same thing again with a different seed. Phase 1's results are
 *      still in the canvas, so this is the test that a use_prev_out=0 matmul
 *      really does overwrite rather than accumulate.
 *   3  2x2 tiles of 16x20 on a 32x40 result, canvas pitch 48. Column tiles
 *      start at 20, mid-word, so tile (r,1) shares the word holding columns
 *      16..31 with tile (r,0) and must not disturb its columns 16..19. This
 *      phase also snapshots the buffer first and afterwards checks that the
 *      row padding (columns 40..47) and the rows past 32 are untouched.
 *
 * gcc -O2 -Wall -Wextra -mcpu=cortex-a9 -mfpu=neon -o matmul_tiled matmul_tiled.c -lm
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
#define ACCEL_SPAN 0x00020000UL      /* 128 KB: the canvas alone is 64 KB */

#define CTRL_BASE 0x0000u
#define WBUF_BASE 0x1000u
#define IBUF_BASE 0x5000u
#define OBUF_BASE 0x9000u            /* canvas: 0x9000 .. 0x19000 */

/* register index; byte offset is index*16 */
#define REG_CONTROL        0
#define REG_WEIGHT_ROWS    1
#define REG_WEIGHT_COLS    2
#define REG_INPUT_MAX_ADDR 3
#define REG_INPUT_COLS     4
#define REG_STATUS         5
#define REG_OUTPUT_SCALE   6
#define REG_TILE_CFG       7
#define REG_TILED_MAT_DIM  8
#define REG_STRIDE         16

#define ST_DONE 0x2u
#define BEAT    16u

/* TILE_CFG field layout, from rdl/ctrl_reg.rdl */
#define TILE_R_SHIFT   0
#define TILE_C_SHIFT   15
#define TILE_FIELD_MSK 0x7FFFu
#define USE_PREV_BIT   (1u << 30)

/* ---- output buffer ------------------------------------------------ */
#define OBUF_BANK_BYTES 4096                     /* per bank */
#define OBUF_BYTES ((size_t)ARRAY_ROWS * OBUF_BANK_BYTES)   /* 64 KB */

/* ---- DMA, LW bridge ----------------------------------------------- */
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

#define WATCHDOG_S 30   /* 20 matmuls plus three 64 KB canvas reads */

#define DMA_ACCEL_BASE  0x00000000u   /* dma_0.write_master -> accel */
#define DMA1_ACCEL_BASE 0x00000000u   /* dma_1.read_master  -> accel */

#define SRC_PHYS 0x30000000UL
#define SRC_SPAN 0x00100000UL
#define STAGE_W  0x00000u
#define STAGE_X  0x08000u
#define STAGE_Y  0x10000u             /* 64 KB canvas landing zone */

/* ---- array and limits --------------------------------------------- */
#define ARRAY_ROWS 16
#define ARRAY_COLS 16
#define BANK_DEPTH 1024               /* WBUF and IBUF, per bank */

#define CEILDIV(a, b) (((a) + (b) - 1) / (b))
#define PITCH(c)      (CEILDIV(c, (int)BEAT) * (int)BEAT)

/* biggest shape any phase uses */
#define MAX_K 256
#define MAX_N 32
#define MAX_M 256

_Static_assert(STAGE_X >= 2048 && STAGE_Y >= STAGE_X + 2048
                   && STAGE_Y + OBUF_BYTES <= SRC_SPAN,
               "staging regions overlap or run past the staging window");
_Static_assert(OBUF_BASE + OBUF_BYTES <= ACCEL_SPAN,
               "output buffer runs past the mapped accelerator span");
/* the testbench's capacity rule: ceil(K/ARRAY_ROWS) * pitch(M) per bank */
_Static_assert(CEILDIV(MAX_K, ARRAY_ROWS) * PITCH(MAX_M) <= OBUF_BANK_BYTES,
               "biggest phase overflows the output bank depth");

static volatile uint8_t  *accel;
static volatile uint32_t *dma0;
static volatile uint32_t *dma1;
static volatile uint8_t  *src;

/* ------------------------------------------------------------------ */
/* quantization - mirrors tb's quantized_problem                       */
/* ------------------------------------------------------------------ */

static double quantize_sym(const double *x, int8_t *q, int n)
{
    double max_abs = 0.0;
    for (int i = 0; i < n; i++) {
        double a = fabs(x[i]);
        if (a > max_abs) max_abs = a;
    }
    double scale;
    if (max_abs > 0.0) scale = max_abs / 127.0;
    else               scale = 1.0;

    for (int i = 0; i < n; i++) {
        double v = rint(x[i] / scale);   /* half-to-even, like np.round */
        if (v >  127.0) v =  127.0;
        if (v < -127.0) v = -127.0;
        q[i] = (int8_t)v;
    }
    return scale;
}

static double randn(void)
{
    double u1;
    double u2;
    do {
        u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
    } while (u1 <= 0.0);
    u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
    return sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2);
}

/* ------------------------------------------------------------------ */
/* golden - per contraction block: requantize, saturate, accumulate    */
/* ------------------------------------------------------------------ */

static int8_t sat8(int32_t v)
{
    if (v >  127) return  127;
    if (v < -128) return -128;
    return (int8_t)v;
}

/* >> 16 floors, exactly like output_loader's `>>>` */
static int8_t requant(int32_t psum, uint32_t scale)
{
    int64_t wide = (int64_t)psum * (int64_t)scale;
    return sat8((int32_t)(wide >> 16));
}

/*
 * The whole result, over the full N. Splitting the contraction across
 * matmuls with use_prev_out continues this same chain, so the expected
 * canvas is this regardless of how the work was tiled.
 */
static void golden_blocked(const int8_t *W, const int8_t *X, int8_t *Y,
                           int K, int N, int M, uint32_t scale)
{
    int bands = N / ARRAY_COLS;
    for (int k = 0; k < K; k++) {
        for (int m = 0; m < M; m++) {
            int32_t acc = 0;
            for (int nb = 0; nb < bands; nb++) {
                int32_t psum = 0;
                for (int c = 0; c < ARRAY_COLS; c++) {
                    int p = nb * ARRAY_COLS + c;
                    psum += (int32_t)W[k * N + p] * (int32_t)X[p * M + m];
                }
                int8_t s = requant(psum, scale);
                if (nb == 0) acc = s;
                else         acc = sat8(acc + s);
            }
            Y[k * M + m] = (int8_t)acc;
        }
    }
}

/* ------------------------------------------------------------------ */
/* teardown - every exit path stops both DMAs                          */
/* ------------------------------------------------------------------ */

static void dma_stop(volatile uint32_t *d)
{
    if (!d) return;
    d[DMA_CONTROL] = 0;                 /* drop GO before anything else */
    for (int i = 0; i < 10000; i++) {
        if (!(d[DMA_STATUS] & DMA_ST_BUSY)) break;
    }
    d[DMA_LENGTH] = 0;
    d[DMA_STATUS] = 0;
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
    const char *m;
    if      (sig == SIGALRM) m = "\n*** watchdog expired: both DMAs stopped.\n";
    else if (sig == SIGBUS)  m = "\n*** SIGBUS: access rejected. DMAs stopped.\n";
    else if (sig == SIGSEGV) m = "\n*** SIGSEGV. DMAs stopped.\n";
    else                     m = "\n*** interrupted: both DMAs stopped.\n";
    ssize_t r = write(2, m, strlen(m));
    (void)r;
    _exit(sig == SIGALRM ? 3 : 2);
}

/* ------------------------------------------------------------------ */
/* plumbing                                                            */
/* ------------------------------------------------------------------ */

static uint64_t ns_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void reg_wr(int idx, uint32_t v)
{
    volatile uint32_t *p = (volatile uint32_t *)(accel + CTRL_BASE + idx * REG_STRIDE);
    *p = v;
}

static uint32_t reg_rd(int idx)
{
    volatile uint32_t *p = (volatile uint32_t *)(accel + CTRL_BASE + idx * REG_STRIDE);
    return *p;
}

static uint32_t tile_cfg(int tile_r, int tile_c, int use_prev)
{
    uint32_t v = 0;
    v |= ((uint32_t)tile_r & TILE_FIELD_MSK) << TILE_R_SHIFT;
    v |= ((uint32_t)tile_c & TILE_FIELD_MSK) << TILE_C_SHIFT;
    if (use_prev) v |= USE_PREV_BIT;
    return v;
}

/* 16 aligned bytes per access, one NEON vld1/vst1. Inline asm because this
 * is Device memory: exactly one load/store per call, unmerged. */
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
    for (size_t i = 0; i < len; i += BEAT) {
        beat_wr(accel + base + i, &img[i]);
    }
}

static void stage_image(uint32_t off, const int8_t *img, size_t len)
{
    for (size_t i = 0; i < len; i += BEAT) {
        beat_wr(src + off + i, &img[i]);
    }
}

/*
 * Largest descriptor we will issue, set from probe_dma_len_mask() below.
 *
 * The LENGTH register is narrower than 32 bits. A value that does not fit
 * truncates silently, and the first beat then decrements past zero: 65536
 * into a 16-bit LENGTH stores 0, one 16-byte beat leaves 0xFFF0 = 65520,
 * and with LEEN the transfer waits for a zero it has already gone past.
 * A 64 KB canvas read produced exactly that. Anything at or above the
 * register's range has to be split.
 */
/*
 * Largest descriptor, in bytes. 8192 = 512 beats.
 *
 * Why not larger: a 12800 B read (800 beats) completes, while 32768 and
 * 65536 both move exactly one beat and then stall with busy high. The DMA
 * bursts, so those longer transfers are issued as several maximum-length
 * bursts rather than one short one. If the slave's beat counter is 10 bits
 * while avs_burstcount is 11, a 1024-beat burst wraps to zero and the slave
 * waits forever for a count it has already passed - one beat out, then
 * nothing, which is what both failures look like.
 *
 * To test that: 16368 (1023 beats) should pass and 16384 (1024) should
 * fail. If both pass, the limit is elsewhere and this can go back up.
 */
#define DMA_CHUNK_BYTES 8192

static uint32_t dma_chunk = DMA_CHUNK_BYTES;

/* Write all-ones and read back: the bits that stick are the ones the
 * register implements. Safe while GO is clear. */
static uint32_t probe_dma_len_mask(volatile uint32_t *d)
{
    d[DMA_CONTROL] = 0;
    d[DMA_LENGTH]  = 0xFFFFFFFFu;
    uint32_t mask = d[DMA_LENGTH];
    d[DMA_LENGTH]  = 0;
    return mask;
}

static int dma_one(volatile uint32_t *d, const char *who,
                   uint32_t rd_addr, uint32_t wr_addr, uint32_t bytes)
{
    d[DMA_STATUS]     = 0;              /* DONE is sticky */
    d[DMA_READ_ADDR]  = rd_addr;
    d[DMA_WRITE_ADDR] = wr_addr;
    d[DMA_LENGTH]     = bytes;
    __sync_synchronize();
    d[DMA_CONTROL] = CTRL_QUADWORD | CTRL_LEEN | CTRL_GO;

    uint64_t t0 = ns_now();
    for (;;) {
        uint32_t st = d[DMA_STATUS];
        if (st & DMA_ST_DONE) return 0;
        if (ns_now() - t0 > 1000000000ull) {
            uint32_t busy = (st & DMA_ST_BUSY) ? 1u : 0u;
            uint32_t done = (st & DMA_ST_DONE) ? 1u : 0u;
            printf("  %s TIMEOUT: STATUS 0x%08x (busy=%u done=%u), "
                   "LENGTH left %u of %u\n",
                   who, st, busy, done, d[DMA_LENGTH], bytes);
            printf("  rd 0x%08x -> wr 0x%08x\n", rd_addr, wr_addr);
            if (d[DMA_LENGTH] > bytes) {
                printf("  LENGTH is larger than what was requested - it has"
                       " wrapped past zero. %u does not fit the register;"
                       " lower dma_chunk.\n", bytes);
            }
            dma_stop(d);
            return -1;
        }
    }
}

/* Split anything too big for the LENGTH register into back-to-back
 * descriptors, advancing both addresses. */
static int dma_go(volatile uint32_t *d, const char *who,
                  uint32_t rd_addr, uint32_t wr_addr, uint32_t bytes)
{
    uint32_t done = 0;
    int piece = 0;
    while (done < bytes) {
        uint32_t n = bytes - done;
        if (n > dma_chunk) n = dma_chunk;

        int rc = dma_one(d, who, rd_addr + done, wr_addr + done, n);
        if (rc != 0) {
            uint32_t lo = rd_addr + done;
            uint32_t hi = lo + n;
            printf("  failed on piece %d of %u: source 0x%05x..0x%05x"
                   " (%u beats), %u of %u bytes already moved\n",
                   piece, CEILDIV(bytes, dma_chunk), lo, hi,
                   n / BEAT, done, bytes);
            if (piece == 0) {
                printf("    the FIRST piece failed, so it is the descriptor"
                       " itself - try lowering DMA_CHUNK_BYTES\n");
            } else {
                printf("    pieces 0..%d completed at this size, so the"
                       " size is fine and something about this address or"
                       " this point in the sequence is not\n", piece - 1);
            }
            return -1;
        }
        done += n;
        piece++;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* one tiled matmul                                                    */
/* ------------------------------------------------------------------ */

struct shape {
    const char *name;
    int K, N, M;        /* the whole result, and the full contraction */
    int tk, tm;         /* tile edge: output rows, output columns */
    int tn;             /* contraction per matmul */
};

static int8_t wimg[MAX_K * MAX_N];
static int8_t ximg[MAX_N * MAX_M];

/*
 * One matmul: the (tr, tc) tile's slice of the h'th contraction chunk.
 * Dimensions and TILE_CFG go in before the buffers - the host-side remap
 * uses WEIGHT_COLS and INPUT_MAX_ADDR to place each word in its bank, so a
 * buffer written first would be scattered with stale values.
 */
static int run_tile(const int8_t *W, const int8_t *X, const struct shape *s,
                    int tr, int tc, int h, uint32_t q16, int pio)
{
    int tk = s->tk;
    int tm = s->tm;
    int tn = s->tn;
    int wpitch = PITCH(tn);
    int ipitch = PITCH(tm);
    size_t wlen = (size_t)tk * wpitch;
    size_t ilen = (size_t)tn * ipitch;

    memset(wimg, 0, wlen);
    memset(ximg, 0, ilen);

    for (int i = 0; i < tk; i++) {
        const int8_t *srcrow = &W[(tr * tk + i) * s->N + h * tn];
        memcpy(&wimg[i * wpitch], srcrow, (size_t)tn);
    }
    for (int p = 0; p < tn; p++) {
        const int8_t *srcrow = &X[(h * tn + p) * s->M + tc * tm];
        memcpy(&ximg[p * ipitch], srcrow, (size_t)tm);
    }

    reg_wr(REG_WEIGHT_ROWS, (uint32_t)tk);      /* output rows  - see header */
    reg_wr(REG_WEIGHT_COLS, (uint32_t)tn);      /* contraction  - see header */
    reg_wr(REG_INPUT_MAX_ADDR, (uint32_t)tm);
    reg_wr(REG_INPUT_COLS, (uint32_t)tm);
    reg_wr(REG_OUTPUT_SCALE, q16);
    reg_wr(REG_TILE_CFG, tile_cfg(tr, tc, h > 0));
    /* the FULL result's width, not this tile's - it sets the canvas pitch,
     * and must not change until the canvas has been read */
    reg_wr(REG_TILED_MAT_DIM, (uint32_t)s->M);

    if (pio) {
        pio_image(WBUF_BASE, wimg, wlen);
        pio_image(IBUF_BASE, ximg, ilen);
    } else {
        stage_image(STAGE_W, wimg, wlen);
        stage_image(STAGE_X, ximg, ilen);
        __sync_synchronize();

        int rc = dma_go(dma0, "dma_0 weights",
                        (uint32_t)(SRC_PHYS + STAGE_W),
                        DMA_ACCEL_BASE + WBUF_BASE, (uint32_t)wlen);
        if (rc != 0) return -1;

        rc = dma_go(dma0, "dma_0 inputs",
                    (uint32_t)(SRC_PHYS + STAGE_X),
                    DMA_ACCEL_BASE + IBUF_BASE, (uint32_t)ilen);
        if (rc != 0) return -1;
    }

    (void)reg_rd(REG_STATUS);           /* drain any stale completion */
    reg_wr(REG_CONTROL, 1);

    uint32_t st = 0;
    uint64_t tp = ns_now();
    for (;;) {
        st = reg_rd(REG_STATUS);
        if (st & ST_DONE) break;
        if (ns_now() - tp > 2000000000ull) break;
    }
    if (!(st & ST_DONE)) {
        printf("  TIMEOUT on tile (%d,%d) half %d, STATUS = 0x%08x\n",
               tr, tc, h, st);
        return -1;
    }
    return 0;
}

/* `len` bytes from the start of the canvas - K*pitch(M), plus any margin
 * rows the caller wants to inspect. Must be a multiple of the bus word. */
static int read_canvas(uint8_t *out, size_t len, int pio)
{
    if (!pio) {
        int rc = dma_go(dma1, "dma_1 canvas",
                        DMA1_ACCEL_BASE + OBUF_BASE,
                        (uint32_t)(SRC_PHYS + STAGE_Y),
                        (uint32_t)len);
        if (rc != 0) return -1;
    }

    volatile uint8_t *base;
    if (pio) base = accel + OBUF_BASE;
    else     base = src + STAGE_Y;

    for (size_t i = 0; i < len; i += BEAT) {
        beat_rd(base + i, (int8_t *)&out[i]);
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* verification                                                        */
/* ------------------------------------------------------------------ */

static int count_bad(const uint8_t *canvas, const int8_t *G,
                     int K, int M, int stride)
{
    int bad = 0;
    for (int k = 0; k < K; k++) {
        for (int m = 0; m < M; m++) {
            int8_t got = (int8_t)canvas[(size_t)k * stride + m];
            int8_t exp = G[k * M + m];
            if (got != exp) bad++;
        }
    }
    return bad;
}

static void report_tiles(const uint8_t *canvas, const int8_t *G,
                         const struct shape *s, int stride)
{
    int ntr = s->K / s->tk;
    int ntc = s->M / s->tm;
    printf("  per tile:");
    for (int tr = 0; tr < ntr; tr++) {
        for (int tc = 0; tc < ntc; tc++) {
            int bad = 0;
            for (int i = 0; i < s->tk; i++) {
                for (int j = 0; j < s->tm; j++) {
                    int k = tr * s->tk + i;
                    int m = tc * s->tm + j;
                    int8_t got = (int8_t)canvas[(size_t)k * stride + m];
                    int8_t exp = G[k * s->M + m];
                    if (got != exp) bad++;
                }
            }
            printf("  (%d,%d):%d/%d", tr, tc, bad, s->tk * s->tm);
        }
    }
    printf("\n");
}

/* ------------------------------------------------------------------ */

static int8_t W_q[MAX_K * MAX_N];
static int8_t X_q[MAX_N * MAX_M];
static int8_t G_q[MAX_K * MAX_M];
static double W_r[MAX_K * MAX_N];
static double X_r[MAX_N * MAX_M];
static double Y_r[MAX_K * MAX_M];
static int8_t ideal[MAX_K * MAX_M];
static uint8_t canvas[OBUF_BYTES];
static uint8_t before[OBUF_BYTES];

/*
 * Returns 0 on pass. `snapshot` also reads the canvas up front and reports
 * bytes changed outside the result region - the check that a tile's byte
 * writes do not disturb its neighbours.
 */
static int run_phase(const struct shape *s, unsigned seed, int pio, int snapshot)
{
    int K = s->K;
    int N = s->N;
    int M = s->M;

    srand(seed);
    for (int i = 0; i < K * N; i++) W_r[i] = randn();
    for (int i = 0; i < N * M; i++) X_r[i] = randn();

    double ws = quantize_sym(W_r, W_q, K * N);
    double xs = quantize_sym(X_r, X_q, N * M);

    for (int k = 0; k < K; k++) {
        for (int m = 0; m < M; m++) {
            double a = 0.0;
            for (int p = 0; p < N; p++) {
                a += W_r[k * N + p] * X_r[p * M + m];
            }
            Y_r[k * M + m] = a;
        }
    }
    double ys = quantize_sym(Y_r, ideal, K * M);
    double combined = ws * xs / ys;
    if (!(combined >= 0.0 && combined < 1.0)) {
        fprintf(stderr, "combined scale %g does not fit Q0.16\n", combined);
        return -1;
    }
    uint32_t q16 = (uint32_t)lrint(combined * 65536.0);

    golden_blocked(W_q, X_q, G_q, K, N, M, q16);

    int ntr = K / s->tk;
    int ntc = M / s->tm;
    int nh  = N / s->tn;

    printf("\n%s  seed %u\n", s->name, seed);
    printf("  %dx%d @ %dx%d -> %dx%d canvas region, OUTPUT_SCALE=%u\n",
           K, N, N, M, K, M, q16);
    printf("  %d tiles of %dx%d x %d contraction chunk%s = %d matmuls\n",
           ntr * ntc, s->tk, s->tm, nh, nh == 1 ? "" : "s", ntr * ntc * nh);

    int stride = PITCH(M);
    size_t live = (size_t)K * stride;

    /* One bank-row block past the result, when the buffer has room. Lets
     * the spill check see a tile that wrote past its last row. */
    int margin_rows = 0;
    if ((CEILDIV(K, ARRAY_ROWS) + 1) * stride <= OBUF_BANK_BYTES) {
        margin_rows = ARRAY_ROWS;
    }
    size_t readlen = live + (size_t)margin_rows * stride;

    printf("  canvas: TILED_MAT_DIM=%d, pitch %d, %zu live bytes%s\n",
           M, stride, live,
           margin_rows ? ", plus a margin block for the spill check" : "");

    if (snapshot) {
        if (read_canvas(before, readlen, pio) != 0) return -1;
    }

    uint64_t t0 = ns_now();
    for (int h = 0; h < nh; h++) {
        for (int tr = 0; tr < ntr; tr++) {
            for (int tc = 0; tc < ntc; tc++) {
                if (run_tile(W_q, X_q, s, tr, tc, h, q16, pio) != 0) return -1;
            }
        }
    }
    uint64_t t1 = ns_now();

    if (read_canvas(canvas, readlen, pio) != 0) return -1;
    uint64_t t2 = ns_now();

    int used = stride;
    int bad = count_bad(canvas, G_q, K, M, used);

    if (bad == 0) {
        printf("  PASS: all %d elements match\n", K * M);
    } else {
        printf("  FAIL: %d/%d wrong\n", bad, K * M);
        int shown = 0;
        for (int k = 0; k < K && shown < 8; k++) {
            for (int m = 0; m < M && shown < 8; m++) {
                int8_t got = (int8_t)canvas[(size_t)k * used + m];
                int8_t exp = G_q[k * M + m];
                if (got == exp) continue;
                printf("    Y[%3d][%3d]: got %4d want %4d  (tile %d,%d)\n",
                       k, m, got, exp, k / s->tk, m / s->tm);
                shown++;
            }
        }
    }
    report_tiles(canvas, G_q, s, used);

    if (snapshot) {
        int pad_spill = 0;      /* columns M..pitch-1 of a live row */
        int row_spill = 0;      /* rows past the result */
        int first_r = -1;
        int first_c = -1;

        int total_rows = K + margin_rows;
        for (int r = 0; r < total_rows; r++) {
            for (int c = 0; c < used; c++) {
                int inside_result = (r < K) && (c < M);
                if (inside_result) continue;
                size_t off = (size_t)r * used + c;
                if (canvas[off] == before[off]) continue;
                if (first_r < 0) { first_r = r; first_c = c; }
                if (r < K) pad_spill++;
                else       row_spill++;
            }
        }

        if (pad_spill == 0 && row_spill == 0) {
            printf("  spill: none - row padding (cols %d..%d) and the %d rows"
                   " past %d are untouched\n", M, used - 1, margin_rows, K - 1);
        } else {
            printf("  spill: %d bytes in the row padding, %d in rows past %d,"
                   " first at (%d,%d)\n",
                   pad_spill, row_spill, K - 1, first_r, first_c);
            if (pad_spill) {
                printf("    a tile wrote past column %d - the right edge's"
                       " byte enables\n", M - 1);
            }
            if (row_spill) {
                printf("    a tile wrote past row %d - tile_r placement or"
                       " the row-live mask\n", K - 1);
            }
            bad += pad_spill + row_spill;
        }
    }

    double us_compute = (double)(t1 - t0) / 1e3;
    double us_read    = (double)(t2 - t1) / 1e3;
    printf("  %8.2f us for %d matmuls (%.2f us each), canvas read %.2f us\n",
           us_compute, ntr * ntc * nh, us_compute / (ntr * ntc * nh), us_read);

    return bad ? -1 : 0;
}

/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    int pio = 0;
    unsigned seed = 1;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-p")) {
            pio = 1;
        } else if (!strcmp(argv[i], "-r") && i + 1 < argc) {
            seed = (unsigned)atoi(argv[++i]);
        } else {
            printf("usage: %s [-p] [-r seed]\n", argv[0]);
            return 1;
        }
    }

    setvbuf(stdout, NULL, _IONBF, 0);
    signal(SIGBUS, on_fatal);
    signal(SIGSEGV, on_fatal);

    static const struct shape canvas_fill = {
        "phase 1/2: 2x2 tiles of 128x128, contraction 32 in two halves",
        256, 32, 256,   /* K, N, M */
        128, 128,       /* tile rows, tile cols */
        16              /* contraction per matmul */
    };
    static const struct shape unaligned = {
        "phase 3: 2x2 tiles of 16x20 on a 40-wide canvas (pitch 48)",
        32, 16, 40,
        16, 20,
        16
    };

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/mem (run as root)"); return 1; }

    void *amap = mmap(NULL, ACCEL_SPAN, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd, (off_t)ACCEL_PHYS);
    void *lmap = mmap(NULL, LW_SPAN, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd, (off_t)LW_BASE);
    void *smap = mmap(NULL, SRC_SPAN, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd, (off_t)SRC_PHYS);
    if (amap == MAP_FAILED || lmap == MAP_FAILED || smap == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    accel = (volatile uint8_t *)amap;
    dma0  = (volatile uint32_t *)((volatile uint8_t *)lmap + DMA0_OFF);
    dma1  = (volatile uint32_t *)((volatile uint8_t *)lmap + DMA1_OFF);
    src   = (volatile uint8_t *)smap;

    dma_cleanup();
    atexit(dma_cleanup);
    signal(SIGINT,  on_fatal);
    signal(SIGTERM, on_fatal);
    signal(SIGHUP,  on_fatal);
    signal(SIGQUIT, on_fatal);
    signal(SIGALRM, on_fatal);
    alarm(WATCHDOG_S);

    uint32_t m0 = probe_dma_len_mask(dma0);
    uint32_t m1 = probe_dma_len_mask(dma1);
    uint32_t mask = (m0 < m1) ? m0 : m1;
    if (mask == 0) {
        printf("  dma LENGTH probe read back 0 - leaving the chunk at %u B\n",
               dma_chunk);
    } else {
        int bits = 0;
        uint32_t t = mask;
        while (t) { bits++; t >>= 1; }
        uint32_t maxlen = mask & ~(uint32_t)(BEAT - 1);
        /* the register can hold more than we are willing to ask for; the
         * burst-length limit above is the binding one */
        if (maxlen < dma_chunk) dma_chunk = maxlen;
        printf("  dma LENGTH: %d bits (mask 0x%08x), register max %u B,"
               " using %u B chunks (%u beats)\n",
               bits, mask, maxlen, dma_chunk, dma_chunk / BEAT);
        if (m0 != m1) {
            printf("  note: dma_0 and dma_1 report different LENGTH widths"
                   " (0x%08x / 0x%08x)\n", m0, m1);
        }
    }

    if (!pio && (size_t)OBUF_BASE + OBUF_BYTES > 0x10000u) {
        printf("  WARNING: the canvas ends at 0x%05zx. The Platform Designer\n"
               "  window for matmul_accel_0.avalon_slave_0 is 0x0000-0xffff,\n"
               "  and dma_1's read_address port is 16 bits, so anything above\n"
               "  0x10000 is unreachable by DMA. -p reads it over the h2f\n"
               "  bridge instead, which is not address-limited the same way.\n",
               (size_t)OBUF_BASE + OBUF_BYTES);
    }

    uint32_t st0 = reg_rd(REG_STATUS);
    printf("matmul_tiled: output buffer %zu B at 0x%04x (%d banks x %d), "
           "transfer=%s\n",
           OBUF_BYTES, OBUF_BASE, ARRAY_ROWS, OBUF_BANK_BYTES,
           pio ? "PIO" : "DMA");
    printf("  STATUS = 0x%08x%s\n", st0,
           (st0 & ~0x3u) ? "   <- unexpected bits: register spacing?" : "");

    int failed = 0;

    if (run_phase(&canvas_fill, seed, pio, 0) != 0) failed++;

    /* Phase 1's results are still sitting in the canvas, so a use_prev_out=0
     * matmul that accumulated instead of overwriting would fail here and
     * pass above. */
    if (run_phase(&canvas_fill, seed + 1, pio, 0) != 0) failed++;

    if (run_phase(&unaligned, seed + 2, pio, 1) != 0) failed++;

    printf("\n%s\n", failed ? "FAILED" : "all phases passed");

    alarm(0);
    dma_cleanup();
    dma0 = NULL;
    dma1 = NULL;
    munmap(amap, ACCEL_SPAN);
    munmap(lmap, LW_SPAN);
    munmap(smap, SRC_SPAN);
    close(fd);
    return failed ? 1 : 0;
}