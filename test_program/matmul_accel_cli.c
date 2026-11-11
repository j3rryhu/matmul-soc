/*
 * matmul_accel.c - parametrized systolic-array run with column panels.
 *
 *   ./matmul_accel N [options]
 *
 * Computes an N x N @ N x N int8 matmul, tiled onto the 16x16 array, and
 * reports us and GOP/s in the same shape matmul_baseline prints so the two
 * line up directly.
 *
 * ---------------------------------------------------------------------------
 * Conventions (from the testbench, not rdl/ctrl_reg.rdl's header)
 * ---------------------------------------------------------------------------
 *   WEIGHT_ROWS = tk  output rows of this matmul
 *   WEIGHT_COLS = tn  contraction length of this matmul
 *   INPUT_COLS  = tm  output columns of this matmul
 *
 * TILE_CFG places the tk x tm result inside a canvas whose width is
 * TILED_MAT_DIM, so
 *
 *     canvas row = tile_r * tk,  canvas col = tile_c * tm
 *     host addr  = OBUF_BASE + row * pitch(TILED_MAT_DIM) + col
 *
 * use_prev_out accumulates onto what the canvas already holds, which is how
 * a contraction longer than one matmul is chained.
 *
 * ---------------------------------------------------------------------------
 * Column panels, and why they are the whole point
 * ---------------------------------------------------------------------------
 * TILED_MAT_DIM does NOT have to be the full result width. The output can be
 * cut into vertical panels, each its own canvas pass with TILED_MAT_DIM set
 * to that panel's width. This matters because transfer cost per unit of work
 * is set entirely by the tile shape:
 *
 *     one matmul: MACs = tk*tn*tm,  bytes = tk*tn + tn*tm = tn*(tk+tm)
 *     intensity  = tk*tm/(tk+tm)  MAC per byte - independent of tn and of N
 *
 * so throughput depends on the tile, not on how big the problem is. The
 * ceiling on tk*tm is the canvas:
 *
 *     ceil(tk/16) * pitch(panel) <= OBUF_BANK_BYTES   ->   tk*panel <= 64 KB
 *
 * and for a fixed product tk+tm is least when tk == tm, so the best tile
 * this hardware can run is tk = tm = 256, tn = 64, at 128 MAC/byte. That
 * saturates wbuf, ibuf and the canvas bank simultaneously. tk = tm = 128 is
 * 64 MAC/byte - half - and leaves three quarters of the canvas idle.
 *
 * Without panels, a 512-wide result forces pitch 512, which caps the band at
 * 128 rows and so caps tk at 128, giving 102 MAC/byte. With two panels of
 * 256 the same problem runs the optimal 256x256 tile. Nothing about the
 * hardware changes; only TILED_MAT_DIM does.
 *
 * ---------------------------------------------------------------------------
 * What limits the tile shape
 * ---------------------------------------------------------------------------
 *   wbuf     ceil(tk/16) * pitch(tn)    <= 1024 B per bank
 *   ibuf     ceil(tn/16) * pitch(tm)    <= 1024 B per bank
 *   canvas   ceil(band/16) * pitch(panel) <= 4096 B per bank
 *   dma out  band * pitch(panel) <= (1 << DMA1_ADDR_BITS) - OBUF_BASE, the
 *            part of the canvas dma_1 can address. At 17 bits that is slack
 *            and the canvas bank rule binds instead. PIO readout goes over
 *            the h2f bridge and is never capped.
 *
 * The program searches panel/tk/tm/tn over the divisors of N, minimising
 * bytes pushed across the bridge and then matmul count. -P/-k/-m/-n pin any
 * of them.
 *
 * ---------------------------------------------------------------------------
 * Which number to compare against matmul_baseline
 * ---------------------------------------------------------------------------
 * "compute + transfers" is the one. matmul_baseline has A, Bt and C in
 * ordinary malloc'd memory and times only the triple loop: no copy in, no
 * copy out, because it computes where the data already sits. The costs with
 * a real counterpart on this side are the ones that exist because the array
 * is a separate device, so dma_in + compute + dma_out is like for like.
 *
 * Two stages are marked [harness] and excluded from that number:
 *
 *   pack       rebuilding each tile's padded image with memset/memcpy. A
 *              real pipeline lays W and X out in tile order once.
 *   stage,     DRAM->DRAM hops into and out of the DMA window. They exist
 *   host read  only because this program shuffles through a buffer that came
 *              from the same O_SYNC /dev/mem open as the register windows,
 *              so it is Device-nGnRnE: uncached, unbuffered, non-gathering,
 *              about 140 ns for every 16-byte access, which is why they
 *              measure ~100 MB/s. A DMA source in DRAM does not need those
 *              semantics; a cacheable coherent buffer (CMA, udmabuf, or a
 *              small module) would be an order faster. The CPU baseline has
 *              no equivalent of either stage.
 *
 * A PIO path carries the operands itself, so under -p/-pw/-pr that access is
 * the transfer and counts as one, not as harness.
 *
 * "compute only" is the CONTROL=1 -> STATUS.DONE sum: the ceiling a better
 * data path approaches. Beware at small N - each STATUS read is an uncached
 * Device access over h2f, so the poll cannot resolve finer than one round
 * trip and this window includes some of the poll loop itself.
 *
 * Both engines run the same N^3 MACs so GOP/s compares directly, even though
 * the array requantizes per 16-wide band and saturates between bands while
 * the CPU accumulates the whole contraction in int32 - the two results are
 * not bit-identical by construction. Correctness here is checked against
 * golden_blocked(), which models the banded arithmetic.
 *
 * gcc -O3 -Wall -Wextra -mcpu=cortex-a9 -mfpu=neon -o matmul_accel_cli matmul_accel_cli.c -lm
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

/* ---- accelerator, h2f bridge ------------------------------------- */
#define ACCEL_PHYS 0xC0000000UL
#define ACCEL_SPAN 0x00020000UL

#define CTRL_BASE 0x0000u
#define WBUF_BASE 0x1000u
#define IBUF_BASE 0x5000u
#define OBUF_BASE 0x9000u

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

#define TILE_R_SHIFT   0
#define TILE_C_SHIFT   15
#define TILE_FIELD_MSK 0x7FFFu
#define USE_PREV_BIT   (1u << 30)

/* ---- array and buffers -------------------------------------------- */
#define ARRAY_ROWS 16
#define ARRAY_COLS 16
#define BANK_DEPTH 1024

#define OBUF_BANK_BYTES 4096
#define OBUF_BYTES ((size_t)ARRAY_ROWS * OBUF_BANK_BYTES)

/*
 * How much of the canvas dma_1 can address. At 17 bits the top is 0x1FFFF
 * and the canvas ends at OBUF_BASE + 64 KB = 0x19000, so the whole thing is
 * reachable and the per-bank canvas rule is the binding limit.
 */
#define DMA1_ADDR_BITS    17
#define DMA_CANVAS_WINDOW ((1u << DMA1_ADDR_BITS) - OBUF_BASE)

_Static_assert(DMA_CANVAS_WINDOW >= OBUF_BYTES,
               "dma_1 cannot reach the whole canvas; bands will be capped");

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

/* 512 beats. Above ~1024 beats the slave's beat counter wraps and the
 * transfer moves one beat then stalls with busy high. */
#define DMA_CHUNK_BYTES 8192

#define DMA_ACCEL_BASE  0x00000000u
#define DMA1_ACCEL_BASE 0x00000000u

#define SRC_PHYS 0x30000000UL
#define SRC_SPAN 0x00100000UL
#define STAGE_W  0x00000u
#define STAGE_X  0x08000u
#define STAGE_Y  0x10000u

#define WATCHDOG_S 300

#define CEILDIV(a, b) (((a) + (b) - 1) / (b))
#define PITCH(c)      (CEILDIV(c, (int)BEAT) * (int)BEAT)
#define MAX_N         512
#define MAX_REPS      64

static volatile uint8_t  *accel;
static volatile uint32_t *dma0;
static volatile uint32_t *dma1;
static volatile uint8_t  *src;

static uint32_t dma_chunk = DMA_CHUNK_BYTES;

/* ------------------------------------------------------------------ */
/* teardown - every exit path drops GO on both channels                */
/* ------------------------------------------------------------------ */

static void dma_stop(volatile uint32_t *d)
{
    if (!d) return;
    d[DMA_CONTROL] = 0;
    d[DMA_STATUS]  = 0;
    d[DMA_LENGTH]  = 0;
}

static void dma_cleanup(void)
{
    dma_stop(dma0);
    dma_stop(dma1);
}

static void on_fatal(int sig)
{
    dma_cleanup();
    if (sig == SIGALRM) {
        static const char m[] = "\nWATCHDOG: hung, DMA stopped, exiting\n";
        ssize_t n = write(2, m, sizeof(m) - 1);
        (void)n;
    }
    _exit(sig == SIGALRM ? 2 : 128 + sig);
}

/* ------------------------------------------------------------------ */
/* register and buffer access                                          */
/* ------------------------------------------------------------------ */

static uint64_t ns_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void reg_wr(int idx, uint32_t v)
{
    volatile uint32_t *p;
    p = (volatile uint32_t *)(accel + CTRL_BASE + idx * REG_STRIDE);
    *p = v;
}

static uint32_t reg_rd(int idx)
{
    volatile uint32_t *p;
    p = (volatile uint32_t *)(accel + CTRL_BASE + idx * REG_STRIDE);
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

/* Device memory: exactly one 128-bit access per call, unmerged. */
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

static void blk_wr(volatile uint8_t *dst, const int8_t *s, size_t len)
{
    for (size_t i = 0; i < len; i += BEAT) {
        beat_wr(dst + i, &s[i]);
    }
}

static void blk_rd(volatile uint8_t *s, int8_t *dst, size_t len)
{
    for (size_t i = 0; i < len; i += BEAT) {
        beat_rd(s + i, &dst[i]);
    }
}

/* ------------------------------------------------------------------ */
/* DMA                                                                 */
/* ------------------------------------------------------------------ */

static int dma_one(volatile uint32_t *d, const char *who,
                   uint32_t rd_addr, uint32_t wr_addr, uint32_t bytes)
{
    d[DMA_STATUS]     = 0;
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
            printf("\n  %s TIMEOUT: STATUS 0x%08x (busy=%u), LENGTH left %u of %u\n",
                   who, st, (st & DMA_ST_BUSY) ? 1u : 0u, d[DMA_LENGTH], bytes);
            printf("  rd 0x%08x -> wr 0x%08x\n", rd_addr, wr_addr);
            dma_stop(d);
            return -1;
        }
    }
}

static int dma_go(volatile uint32_t *d, const char *who,
                  uint32_t rd_addr, uint32_t wr_addr, uint32_t bytes)
{
    uint32_t done = 0;
    while (done < bytes) {
        uint32_t n = bytes - done;
        if (n > dma_chunk) n = dma_chunk;

        int rc = dma_one(d, who, rd_addr + done, wr_addr + done, n);
        if (rc != 0) return -1;
        done += n;
    }
    return 0;
}

static uint32_t probe_dma_len_mask(volatile uint32_t *d)
{
    d[DMA_CONTROL] = 0;
    d[DMA_LENGTH]  = 0xFFFFFFFFu;
    uint32_t mask = d[DMA_LENGTH];
    d[DMA_LENGTH]  = 0;
    return mask;
}

/* ------------------------------------------------------------------ */
/* quantization and the banded golden                                  */
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

static void golden_blocked(const int8_t *W, const int8_t *X, int8_t *Y,
                           int N, uint32_t scale)
{
    int bands = N / ARRAY_COLS;
    for (int k = 0; k < N; k++) {
        for (int m = 0; m < N; m++) {
            int32_t acc = 0;
            for (int nb = 0; nb < bands; nb++) {
                int32_t psum = 0;
                for (int c = 0; c < ARRAY_COLS; c++) {
                    int p = nb * ARRAY_COLS + c;
                    psum += (int32_t)W[(size_t)k * N + p]
                          * (int32_t)X[(size_t)p * N + m];
                }
                int8_t s = requant(psum, scale);
                if (nb == 0) acc = s;
                else         acc = sat8(acc + s);
            }
            Y[(size_t)k * N + m] = (int8_t)acc;
        }
    }
}

/*
 * Pick OUTPUT_SCALE from the data rather than guessing. Each 16-wide band
 * requantizes to int8 and the bands accumulate with saturation, so the final
 * magnitude grows as sqrt(N/16). A scale that suits N=32 pins most of the
 * canvas at +-127 by N=256, and a saturated canvas compares equal to the
 * golden even with a broken remap - the check would pass while proving
 * nothing. Aim the accumulated result at about +-40.
 */
static uint32_t pick_scale(const int8_t *W, const int8_t *X, int N)
{
    int bands = N / ARRAY_COLS;
    int step = (N >= 16) ? N / 16 : 1;

    double sum2 = 0.0;
    long n = 0;

    for (int k = 0; k < N; k += step) {
        for (int m = 0; m < N; m += step) {
            for (int nb = 0; nb < bands; nb++) {
                int32_t psum = 0;
                for (int c = 0; c < ARRAY_COLS; c++) {
                    int p = nb * ARRAY_COLS + c;
                    psum += (int32_t)W[(size_t)k * N + p]
                          * (int32_t)X[(size_t)p * N + m];
                }
                sum2 += (double)psum * (double)psum;
                n++;
            }
        }
    }
    if (n == 0) return 1;

    double rms = sqrt(sum2 / (double)n);
    if (rms < 1.0) return 65535u;

    double q = 40.0 * 65536.0 / (rms * sqrt((double)bands));
    if (q < 1.0)     q = 1.0;
    if (q > 65535.0) q = 65535.0;
    return (uint32_t)q;
}

static double sat_fraction(const int8_t *G, size_t n)
{
    size_t pinned = 0;
    for (size_t i = 0; i < n; i++) {
        if (G[i] == 127 || G[i] == -128) pinned++;
    }
    return 100.0 * (double)pinned / (double)n;
}

/* ------------------------------------------------------------------ */
/* tile shape search                                                   */
/* ------------------------------------------------------------------ */

struct plan {
    int panel;            /* canvas width = TILED_MAT_DIM */
    int npanels;
    int tk, tm, tn;       /* per-matmul output rows, output cols, contraction */
    int band;             /* output rows per canvas pass */
    int bands;            /* canvas passes per panel */
    int nmm;              /* total matmuls */
    size_t wlen, ilen;    /* bytes per matmul */
    size_t bytes_in;      /* total pushed over the bridge */
    size_t canvas_bytes;  /* read back per canvas pass */
    double intensity;     /* MAC per byte of operand traffic */
};

static int fits_wbuf(int tk, int tn)
{
    return CEILDIV(tk, ARRAY_ROWS) * PITCH(tn) <= BANK_DEPTH;
}

static int fits_ibuf(int tn, int tm)
{
    return CEILDIV(tn, ARRAY_ROWS) * PITCH(tm) <= BANK_DEPTH;
}

/* largest band of output rows one canvas pass can hold, for this panel */
static int band_limit(int panel, int N, int pio_out)
{
    int pitch = PITCH(panel);

    int by_bank = ARRAY_ROWS * (OBUF_BANK_BYTES / pitch);
    int lim = (by_bank < N) ? by_bank : N;

    if (!pio_out) {
        int by_dma = (int)(DMA_CANVAS_WINDOW / (unsigned)pitch);
        if (by_dma < lim) lim = by_dma;
    }
    return lim;
}

/* largest d <= cap with d | of_what and d % 16 == 0 */
static int best_divisor(int of_what, int cap)
{
    for (int d = cap - (cap % ARRAY_ROWS); d >= ARRAY_ROWS; d -= ARRAY_ROWS) {
        if (of_what % d == 0) return d;
    }
    return 0;
}

static void fill_plan(struct plan *p, int N)
{
    p->wlen = (size_t)p->tk * PITCH(p->tn);
    p->ilen = (size_t)p->tn * PITCH(p->tm);
    p->npanels = N / p->panel;
    p->bands = N / p->band;
    p->nmm = p->npanels * p->bands * (p->band / p->tk)
           * (p->panel / p->tm) * (N / p->tn);
    p->bytes_in = (size_t)p->nmm * (p->wlen + p->ilen);
    p->canvas_bytes = (size_t)p->band * PITCH(p->panel);
    p->intensity = (double)p->tk * (double)p->tm
                 / ((double)p->tk + (double)p->tm);
}

/*
 * Minimise bytes across the bridge, then matmul count. Minimising total
 * bytes is the same as maximising tk*tm/(tk+tm), but it also accounts for
 * the pitch padding that a pure intensity figure hides.
 */
static int choose_plan(struct plan *out, int N, int pio_out,
                       int fp, int fk, int fm, int fn)
{
    struct plan best;
    int have = 0;

    for (int panel = ARRAY_ROWS; panel <= N; panel += ARRAY_ROWS) {
        if (N % panel) continue;
        if (fp && panel != fp) continue;

        int blim = band_limit(panel, N, pio_out);
        if (blim < ARRAY_ROWS) continue;

        for (int tn = ARRAY_ROWS; tn <= N; tn += ARRAY_ROWS) {
            if (N % tn) continue;
            if (fn && tn != fn) continue;

            for (int tm = ARRAY_ROWS; tm <= panel; tm += ARRAY_ROWS) {
                if (panel % tm) continue;
                if (fm && tm != fm) continue;
                if (!fits_ibuf(tn, tm)) continue;

                for (int tk = ARRAY_ROWS; tk <= N && tk <= blim;
                     tk += ARRAY_ROWS) {
                    if (N % tk) continue;
                    if (fk && tk != fk) continue;
                    if (!fits_wbuf(tk, tn)) continue;

                    /* band: a divisor of N, a multiple of tk, within blim */
                    int band = best_divisor(N, blim);
                    while (band && band % tk) {
                        band = best_divisor(N, band - 1);
                    }
                    if (!band) continue;

                    struct plan c;
                    c.panel = panel;
                    c.tk = tk;
                    c.tm = tm;
                    c.tn = tn;
                    c.band = band;
                    fill_plan(&c, N);

                    if (!have
                        || c.bytes_in < best.bytes_in
                        || (c.bytes_in == best.bytes_in && c.nmm < best.nmm)) {
                        best = c;
                        have = 1;
                    }
                }
            }
        }
    }

    if (!have) return -1;
    *out = best;
    return 0;
}

/* ------------------------------------------------------------------ */
/* timing accumulators, one set per repetition                         */
/* ------------------------------------------------------------------ */

struct times {
    uint64_t pack, xfer, dma_in, compute, dma_out, read;
};

static uint64_t t_total(const struct times *t)
{
    return t->pack + t->xfer + t->dma_in + t->compute + t->dma_out + t->read;
}

static int cmp_u64(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a;
    uint64_t y = *(const uint64_t *)b;
    if (x < y) return -1;
    if (x > y) return  1;
    return 0;
}

static double median_us(uint64_t *v, int n)
{
    qsort(v, (size_t)n, sizeof(uint64_t), cmp_u64);
    return (double)v[n / 2] / 1e3;
}

/* ------------------------------------------------------------------ */
/* one matmul                                                          */
/* ------------------------------------------------------------------ */

static int8_t *wimg;
static int8_t *ximg;

/*
 * The (tr, tc) tile of column panel `pc0`, row band `row0`, contraction
 * chunk `h`. Dimensions and TILE_CFG go in before the buffers - the
 * host-side remap uses WEIGHT_COLS and INPUT_MAX_ADDR to place each word in
 * its bank, so a buffer written first would be scattered with stale values.
 */
static int run_tile(const int8_t *W, const int8_t *X, int N,
                    const struct plan *p, int pc0, int row0,
                    int tr, int tc, int h, uint32_t q16, int pio_in,
                    struct times *t)
{
    int tk = p->tk;
    int tm = p->tm;
    int tn = p->tn;
    int wpitch = PITCH(tn);
    int ipitch = PITCH(tm);

    uint64_t a = ns_now();

    memset(wimg, 0, p->wlen);
    memset(ximg, 0, p->ilen);

    for (int i = 0; i < tk; i++) {
        const int8_t *srcrow = &W[(size_t)(row0 + tr * tk + i) * N + h * tn];
        memcpy(&wimg[(size_t)i * wpitch], srcrow, (size_t)tn);
    }
    for (int q = 0; q < tn; q++) {
        const int8_t *srcrow = &X[(size_t)(h * tn + q) * N + pc0 + tc * tm];
        memcpy(&ximg[(size_t)q * ipitch], srcrow, (size_t)tm);
    }

    uint64_t b = ns_now();
    t->pack += b - a;

    reg_wr(REG_WEIGHT_ROWS, (uint32_t)tk);
    reg_wr(REG_WEIGHT_COLS, (uint32_t)tn);
    reg_wr(REG_INPUT_MAX_ADDR, (uint32_t)tm);
    reg_wr(REG_INPUT_COLS, (uint32_t)tm);
    reg_wr(REG_OUTPUT_SCALE, q16);
    reg_wr(REG_TILE_CFG, tile_cfg(tr, tc, h > 0));
    /* this panel's width, not the full result's - it sets the canvas pitch,
     * and must not change until this canvas pass has been read */
    reg_wr(REG_TILED_MAT_DIM, (uint32_t)p->panel);

    if (pio_in) {
        blk_wr(accel + WBUF_BASE, wimg, p->wlen);
        blk_wr(accel + IBUF_BASE, ximg, p->ilen);
        t->xfer += ns_now() - b;
    } else {
        blk_wr(src + STAGE_W, wimg, p->wlen);
        blk_wr(src + STAGE_X, ximg, p->ilen);
        __sync_synchronize();

        uint64_t c = ns_now();
        t->xfer += c - b;

        int rc = dma_go(dma0, "dma_0 weights",
                        (uint32_t)(SRC_PHYS + STAGE_W),
                        DMA_ACCEL_BASE + WBUF_BASE, (uint32_t)p->wlen);
        if (rc == 0) {
            rc = dma_go(dma0, "dma_0 inputs",
                        (uint32_t)(SRC_PHYS + STAGE_X),
                        DMA_ACCEL_BASE + IBUF_BASE, (uint32_t)p->ilen);
        }
        t->dma_in += ns_now() - c;
        if (rc != 0) return -1;
    }

    (void)reg_rd(REG_STATUS);

    uint64_t d = ns_now();
    reg_wr(REG_CONTROL, 1);

    uint32_t st = 0;
    for (;;) {
        st = reg_rd(REG_STATUS);
        if (st & ST_DONE) break;
        if (ns_now() - d > 2000000000ull) break;
    }
    t->compute += ns_now() - d;

    if (!(st & ST_DONE)) {
        printf("\n  TIMEOUT on panel %d band %d tile (%d,%d) chunk %d, "
               "STATUS = 0x%08x\n", pc0, row0, tr, tc, h, st);
        return -1;
    }
    return 0;
}

static int read_canvas(int8_t *out, size_t len, int pio_out, struct times *t)
{
    uint64_t a = ns_now();

    if (!pio_out) {
        int rc = dma_go(dma1, "dma_1 canvas",
                        DMA1_ACCEL_BASE + OBUF_BASE,
                        (uint32_t)(SRC_PHYS + STAGE_Y), (uint32_t)len);
        uint64_t b = ns_now();
        t->dma_out += b - a;
        if (rc != 0) return -1;

        blk_rd(src + STAGE_Y, out, len);
        t->read += ns_now() - b;
    } else {
        blk_rd(accel + OBUF_BASE, out, len);
        t->read += ns_now() - a;
    }
    return 0;
}

/*
 * The whole N x N result, one repetition.
 *
 * Panel outermost, then row band, then the tiles inside that canvas, then
 * the contraction chunks. All chunks of one tile run consecutively so the
 * use_prev_out chain is unbroken, and the canvas is read before the next
 * band or panel reuses it.
 */
static int run_once(const int8_t *W, const int8_t *X, int8_t *Y, int N,
                    const struct plan *p, uint32_t q16,
                    int pio_in, int pio_out, int8_t *cbuf, struct times *t)
{
    int cpitch = PITCH(p->panel);

    for (int pc0 = 0; pc0 < N; pc0 += p->panel) {
        for (int row0 = 0; row0 < N; row0 += p->band) {
            int tiles_r = p->band / p->tk;
            int tiles_c = p->panel / p->tm;
            int chunks  = N / p->tn;

            for (int tr = 0; tr < tiles_r; tr++) {
                for (int tc = 0; tc < tiles_c; tc++) {
                    for (int h = 0; h < chunks; h++) {
                        int rc = run_tile(W, X, N, p, pc0, row0, tr, tc, h,
                                          q16, pio_in, t);
                        if (rc != 0) return -1;
                    }
                }
            }

            int rc = read_canvas(cbuf, p->canvas_bytes, pio_out, t);
            if (rc != 0) return -1;

            for (int r = 0; r < p->band; r++) {
                memcpy(&Y[(size_t)(row0 + r) * N + pc0],
                       &cbuf[(size_t)r * cpitch], (size_t)p->panel);
            }
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* staging safety: refuse to DMA into kernel-managed RAM               */
/* ------------------------------------------------------------------ */

static int staging_is_free(void)
{
    FILE *f = fopen("/proc/iomem", "r");
    if (!f) return 1;               /* cannot tell; let it run */

    unsigned long lo = SRC_PHYS;
    unsigned long hi = SRC_PHYS + SRC_SPAN;
    int clash = 0;
    char line[256];

    while (fgets(line, sizeof(line), f)) {
        unsigned long a, b;
        char name[128];
        if (sscanf(line, "%lx-%lx : %127[^\n]", &a, &b, name) != 3) continue;
        if (!strstr(name, "System RAM")) continue;
        if (lo < b + 1 && a < hi) {
            printf("  FATAL: staging 0x%08lx..0x%08lx overlaps System RAM "
                   "0x%08lx-%08lx\n", lo, hi, a, b);
            clash = 1;
        }
    }
    fclose(f);
    return !clash;
}

/* ------------------------------------------------------------------ */

static void usage(const char *p)
{
    printf("usage: %s N [options]\n", p);
    printf("  N            matrix edge, multiple of %d, up to %d\n",
           ARRAY_ROWS, MAX_N);
    printf("  -P PANEL     canvas width (TILED_MAT_DIM)  (default: chosen)\n");
    printf("  -k TK        output rows per matmul        (default: chosen)\n");
    printf("  -m TM        output cols per matmul        (default: chosen)\n");
    printf("  -n TN        contraction per matmul        (default: chosen)\n");
    printf("  -q REPS      timed repetitions             (default 5)\n");
    printf("  -p           PIO both ways\n");
    printf("  -pw          PIO in, dma_1 out\n");
    printf("  -pr          dma_0 in, PIO out\n");
    printf("  -r SEED      default 1\n");
    printf("\n  all of -P/-k/-m/-n must divide N (and -m must divide -P),\n");
    printf("  and be multiples of %d.\n", ARRAY_ROWS);
}

int main(int argc, char **argv)
{
    int N = 128;
    int fp = 0, fk = 0, fm = 0, fn = 0;
    int reps = 5;
    int pio_in = 0, pio_out = 0;
    unsigned seed = 1;

    if (argc < 2) { usage(argv[0]); return 1; }
    N = atoi(argv[1]);

    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "-P") && i + 1 < argc)      fp = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-k") && i + 1 < argc) fk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-m") && i + 1 < argc) fm = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) fn = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-q") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) seed = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "-p"))  { pio_in = 1; pio_out = 1; }
        else if (!strcmp(argv[i], "-pw")) { pio_in = 1; }
        else if (!strcmp(argv[i], "-pr")) { pio_out = 1; }
        else { usage(argv[0]); return 1; }
    }

    if (N < ARRAY_ROWS || N > MAX_N || N % ARRAY_ROWS) {
        printf("N must be a multiple of %d, between %d and %d\n",
               ARRAY_ROWS, ARRAY_ROWS, MAX_N);
        return 1;
    }
    if (reps < 1) reps = 1;
    if (reps > MAX_REPS) reps = MAX_REPS;

    setvbuf(stdout, NULL, _IONBF, 0);
    signal(SIGBUS, on_fatal);
    signal(SIGSEGV, on_fatal);

    struct plan p;
    if (choose_plan(&p, N, pio_out, fp, fk, fm, fn) != 0) {
        printf("no legal tile shape for N=%d", N);
        if (fp || fk || fm || fn) printf(" with -P/-k/-m/-n as given");
        printf(".\n  wbuf/ibuf allow ceil(tk/16)*pitch(tn) <= %d and "
               "ceil(tn/16)*pitch(tm) <= %d\n", BANK_DEPTH, BANK_DEPTH);
        printf("  canvas allows ceil(band/16)*pitch(panel) <= %d",
               OBUF_BANK_BYTES);
        if (!pio_out) {
            printf(", dma_1 window %u B", DMA_CANVAS_WINDOW);
        }
        printf("\n");
        return 1;
    }

    if (!pio_in || !pio_out) {
        if (!staging_is_free()) return 1;
    }

    /* ---- host buffers -------------------------------------------- */
    size_t n2 = (size_t)N * N;
    int8_t *W = malloc(n2);
    int8_t *X = malloc(n2);
    int8_t *G = malloc(n2);
    int8_t *Y = malloc(n2);
    int8_t *cbuf = malloc(p.canvas_bytes);
    wimg = malloc(p.wlen);
    ximg = malloc(p.ilen);

    if (!W || !X || !G || !Y || !cbuf || !wimg || !ximg) {
        printf("out of memory\n");
        return 1;
    }

    srand(seed);
    for (size_t i = 0; i < n2; i++) W[i] = (int8_t)(rand() % 255 - 127);
    for (size_t i = 0; i < n2; i++) X[i] = (int8_t)(rand() % 255 - 127);

    uint32_t q16 = pick_scale(W, X, N);

    /* ---- map ------------------------------------------------------ */
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
    if (mask) {
        uint32_t maxlen = mask & ~(uint32_t)(BEAT - 1);
        if (maxlen < dma_chunk) dma_chunk = maxlen;
    }

    /* ---- report the plan ------------------------------------------ */
    double macs = (double)N * N * N;
    double n2d  = (double)N * (double)N;

    printf("matmul_accel: %dx%d @ %dx%d, %.2f MMAC, scale 0x%04x\n",
           N, N, N, N, macs / 1e6, q16);
    printf("  panels  %d of %d columns (TILED_MAT_DIM = %d, pitch %d)\n",
           p.npanels, p.panel, p.panel, PITCH(p.panel));
    printf("  tiles   %d x %d per panel, %dx%d each, contraction %d in "
           "%d chunk(s) of %d\n",
           p.band / p.tk * p.bands, p.panel / p.tm, p.tk, p.tm,
           N, N / p.tn, p.tn);
    printf("          %d matmul(s), wbuf %zu B + ibuf %zu B each, "
           "%.1f KB over the bridge\n",
           p.nmm, p.wlen, p.ilen, (double)p.bytes_in / 1024.0);
    printf("  canvas  %d band(s) of %d rows per panel, %zu B read back each\n",
           p.bands, p.band, p.canvas_bytes);
    printf("  path    %s, dma chunk %u B (%u beats)\n",
           (pio_in && pio_out) ? "PIO both ways"
           : pio_in            ? "PIO in, dma_1 out"
           : pio_out           ? "dma_0 in, PIO out"
                               : "DMA both ways",
           dma_chunk, dma_chunk / BEAT);

    /*
     * Transfer cost per unit of work is tk*tm/(tk+tm) MAC per byte, set by
     * the tile alone - not by N. The ceiling is tk*tm <= 64 KB from the
     * canvas, so tk == tm == 256 at 128 MAC/byte is the best this hardware
     * can run. Anything less means operands cross the bridge more than once.
     */
    /* best for THIS N: tk == tm == min(N, 256), since tk*tm <= 64 KB */
    int opt_edge = (N < 256) ? N : 256;
    double opt_inten = (double)opt_edge / 2.0;

    printf("  density %.1f MAC/byte of %.1f achievable at N=%d, W sent %.1fx,"
           " X sent %.1fx\n",
           p.intensity, opt_inten, N,
           (double)p.nmm * (double)p.wlen / n2d,
           (double)p.nmm * (double)p.ilen / n2d);

    if (p.intensity < 0.75 * opt_inten) {
        printf("          <- well short. This program keeps every tile the"
               " same shape, so\n"
               "             panel/tk/tn must divide %d and tm must divide the"
               " panel.\n", N);
        printf("             %d's only divisors that are multiples of %d are:",
               N, ARRAY_ROWS);
        for (int d = ARRAY_ROWS; d <= N; d += ARRAY_ROWS) {
            if (N % d == 0) printf(" %d", d);
        }
        printf("\n             too few to reach %dx%d. The hardware itself"
               " handles partial\n"
               "             tiles (matmul_partial ran K=100, matmul_tiled ran"
               " tm=20 on a\n"
               "             40-wide canvas); this search just does not use"
               " them.\n", opt_edge, opt_edge);
    }

    uint32_t st0 = reg_rd(REG_STATUS);
    if (st0 & ~0x3u) {
        printf("  STATUS = 0x%08x   <- unexpected bits: register spacing?\n", st0);
    }

    /* ---- correctness, untimed ------------------------------------- */
    golden_blocked(W, X, G, N, q16);

    struct times warm;
    memset(&warm, 0, sizeof warm);
    if (run_once(W, X, Y, N, &p, q16, pio_in, pio_out, cbuf, &warm) != 0) {
        printf("\nFAILED during the warm-up run\n");
        return 1;
    }

    size_t bad = 0;
    for (size_t i = 0; i < n2; i++) {
        if (Y[i] != G[i]) bad++;
    }
    double sat = sat_fraction(G, n2);
    printf("\n  verify  accel vs golden_blocked: %s (%zu of %zu bytes differ)\n",
           bad ? "MISMATCH" : "PASS", bad, n2);
    printf("          %.1f%% of the expected result is at a saturation limit\n",
           sat);
    if (sat > 25.0) {
        printf("          <- high: a pinned canvas makes the check weak\n");
    }
    if (bad) {
        printf("  timings below are of a run that does not compute the right "
               "answer - fix this first\n");
    }

    /* ---- timed repetitions ---------------------------------------- */
    uint64_t v_pack[MAX_REPS], v_xfer[MAX_REPS], v_din[MAX_REPS];
    uint64_t v_comp[MAX_REPS], v_dout[MAX_REPS], v_read[MAX_REPS];
    uint64_t v_tot[MAX_REPS];

    for (int r = 0; r < reps; r++) {
        struct times t;
        memset(&t, 0, sizeof t);

        if (run_once(W, X, Y, N, &p, q16, pio_in, pio_out, cbuf, &t) != 0) {
            printf("\nFAILED on repetition %d\n", r);
            return 1;
        }

        v_pack[r] = t.pack;
        v_xfer[r] = t.xfer;
        v_din[r]  = t.dma_in;
        v_comp[r] = t.compute;
        v_dout[r] = t.dma_out;
        v_read[r] = t.read;
        v_tot[r]  = t_total(&t);
    }

    /* ---- report --------------------------------------------------- */
    double us_pack = median_us(v_pack, reps);
    double us_xfer = median_us(v_xfer, reps);
    double us_din  = median_us(v_din,  reps);
    double us_comp = median_us(v_comp, reps);
    double us_dout = median_us(v_dout, reps);
    double us_read = median_us(v_read, reps);
    double us_tot  = median_us(v_tot,  reps);

    /* whichever path carries the operands is intrinsic; in PIO mode the PIO
     * access IS the transfer and there is no staging hop at all */
    double us_dev = us_comp
                  + (pio_in  ? us_xfer : us_din)
                  + (pio_out ? us_read : us_dout);

    double gops_comp = 2.0 * macs / (us_comp * 1e3);
    double gops_dev  = 2.0 * macs / (us_dev  * 1e3);
    double gops_tot  = 2.0 * macs / (us_tot  * 1e3);

    double b_in  = (double)p.bytes_in;
    double b_out = (double)p.npanels * (double)p.bands
                 * (double)p.canvas_bytes;

    printf("\n  %d repetition(s), median\n\n", reps);
    printf("  %-28s %10s %12s\n", "", "us", "GOP/s");
    printf("  %-28s %10s %12s\n",
           "----------------------------", "----------", "------------");
    printf("  %-28s %10.2f %12.3f\n", "compute only", us_comp, gops_comp);
    printf("  %-28s %10.2f %12.3f   <- vs matmul_baseline\n",
           "compute + transfers", us_dev, gops_dev);
    printf("  %-28s %10.2f %12.3f\n\n",
           "all stages incl. harness", us_tot, gops_tot);

    printf("  %-28s %10s %9s %7s\n", "breakdown", "us", "MB/s", "share");
    printf("  %-28s %10s %9s %7s\n",
           "----------------------------", "----------", "---------",
           "-------");

    printf("    %-26s %10.2f %9s %6.1f%%\n", "[harness] pack images",
           us_pack, "-", 100.0 * us_pack / us_tot);
    printf("    %-26s %10.2f %9.1f %6.1f%%\n",
           pio_in ? "pio store to accel" : "[harness] stage to DDR",
           us_xfer, b_in / us_xfer, 100.0 * us_xfer / us_tot);
    if (!pio_in) {
        printf("    %-26s %10.2f %9.1f %6.1f%%\n", "dma_0 in",
               us_din, b_in / us_din, 100.0 * us_din / us_tot);
    }
    printf("    %-26s %10.2f %9s %6.1f%%\n", "array compute",
           us_comp, "-", 100.0 * us_comp / us_tot);
    if (!pio_out) {
        printf("    %-26s %10.2f %9.1f %6.1f%%\n", "dma_1 out",
               us_dout, b_out / us_dout, 100.0 * us_dout / us_tot);
    }
    printf("    %-26s %10.2f %9.1f %6.1f%%\n",
           pio_out ? "pio load from canvas" : "[harness] host read of DDR",
           us_read, b_out / us_read, 100.0 * us_read / us_tot);

    /* where the headroom is: the input path against the array */
    double in_rate = b_in / (pio_in ? us_xfer : us_din);
    printf("\n  at %.0f MB/s in and %.1f MAC/byte, the input path caps this "
           "run at\n  %.1f GOP/s", in_rate, p.intensity,
           2.0 * p.intensity * in_rate / 1e3);
    if (gops_comp < 2.0 * p.intensity * in_rate / 1e3) {
        printf("; the array's %.1f GOP/s is the tighter limit.\n", gops_comp);
    } else {
        printf(", below the array's %.1f GOP/s - transfer bound.\n", gops_comp);
    }

    printf("\n  [harness] stages are excluded from `compute + transfers`:"
           " pack is work\n  a real pipeline does once, and the DDR hops are"
           " slow only because the\n  staging window inherited Device-nGnRnE"
           " from the O_SYNC mapping.\n  matmul_baseline has no counterpart"
           " to either. See the file header.\n");

    /* one line in matmul_baseline's shape, to line up by hand */
    printf("\n%dx%d accel P%d tk%d tm%d tn%d: %.2f us, %.3f GOP/s"
           "   [compute %.2f us, %.3f GOP/s]\n",
           N, N, p.panel, p.tk, p.tm, p.tn, us_dev, gops_dev,
           us_comp, gops_comp);

    alarm(0);
    dma_cleanup();
    dma0 = NULL;
    dma1 = NULL;
    munmap(amap, ACCEL_SPAN);
    munmap(lmap, LW_SPAN);
    munmap(smap, SRC_SPAN);
    close(fd);
    return bad ? 1 : 0;
}