/*
 * bandwidth_test.c - what the DMA path moves per second, what that costs per
 *                    unit of work for each tile shape, and what the bias adds.
 *
 *   ./bandwidth_test [-k TK] [-m TM] [-n TN] [-q REPS] [-a] [-p] [-pb] [-r SEED]
 *
 * Every shape is measured twice: plain Y = W@X, then Y = W@X + b with
 * ACT_CFG.bias_en. DMA by default (-p adds PIO rows for comparison).
 *
 * ---------------------------------------------------------------------------
 * What is timed, and what is not
 * ---------------------------------------------------------------------------
 * The operand images are staged into the DMA window ONCE, outside the timed
 * loop - they do not change between repetitions, and a real pipeline would
 * have its activations in DMA-able memory already. Each repetition times
 * only:
 *
 *   dma_0 in    staging DDR -> weight buffer, input buffer, bias buffer
 *   compute     CONTROL=1 -> STATUS.DONE
 *   dma_1 out   canvas -> staging DDR
 *
 * The one-off staging cost is reported separately at the end, because it is
 * an artifact of the staging buffer inheriting Device-nGnRnE from the same
 * O_SYNC /dev/mem open as the register windows - about 110 MB/s, where a
 * cacheable coherent buffer (CMA, udmabuf) would be an order faster.
 *
 * ---------------------------------------------------------------------------
 * The tile limit, and why it is 128x128
 * ---------------------------------------------------------------------------
 *   psum   16 banks x 1024 slots x 32 bit = 16384 int32 = 64 KB
 *   canvas 16 banks x 4096 bytes          = 65536 int8  = 64 KB
 *
 * Same bytes, four times fewer elements, so the psum buffer caps the output
 * tile at tk*tm <= 16384, not the canvas's 65536. Transfer cost per unit of
 * work is set by the tile alone:
 *
 *   one matmul: MACs = tk*tn*tm,  bytes in = tk*tn + tn*tm = tn*(tk+tm)
 *   density    = tk*tm / (tk + tm)   MAC per byte, independent of tn
 *
 * which is least when tk == tm, so the best tile is tk = tm = 128 at
 * 64 MAC/byte - and at tn = 128 wbuf, ibuf and psum all fill at once.
 *
 * ---------------------------------------------------------------------------
 * Columns
 * ---------------------------------------------------------------------------
 *   p             transfer path: D = DMA, P = PIO over h2f
 *   b             whether ACT_CFG.bias_en was set
 *   in B / out B  bytes moved in (weights + inputs + bias) and out
 *   in MB/s       dma_0 (or PIO) for all the operands together
 *   bias us       the bias leg alone, 4 bytes per output row
 *   out MB/s      dma_1 (or PIO) for the canvas
 *   GOP/s dev     2*MACs / (in + compute + out) - the honest device cost
 *
 * Each shape is checked against the golden before it is timed, and the bias
 * runs additionally check the bias moved most of the tile, so a row is never
 * a timing of a broken or inert run.
 *
 * gcc -O2 -Wall -Wextra -mcpu=cortex-a9 -mfpu=neon -o bandwidth_test bandwidth_test.c -lm
 */

#include "accel_tb.h"

/* ---- DMA, on the LW bridge -------------------------------------- */
#define LW_BASE  0xFF200000UL
#define LW_SPAN  0x00200000UL
#define DMA0_OFF 0x00u
#define DMA1_OFF 0x20u

#define DMA_STATUS  0
#define DMA_READ    1
#define DMA_WRITE   2
#define DMA_LENGTH  3
#define DMA_CONTROL 6

#define DMA_DONE   0x1u
#define DMA_BUSY   0x2u
#define CTRL_GO    (1u << 3)
#define CTRL_LEEN  (1u << 7)
#define CTRL_QUAD  (1u << 11)

/* 512 beats. Above ~1024 the slave's beat counter wraps: the transfer moves
 * one beat and then stalls with busy high. */
#define CHUNK 8192u

#define SRC_PHYS 0x30000000UL
#define SRC_SPAN 0x00100000UL
#define STAGE_W  0x00000u
#define STAGE_X  0x08000u
#define STAGE_B  0x10000u
#define STAGE_Y  0x11000u

#define MAX_TILE 128
#define MAX_REPS 33

static volatile uint32_t *dma0, *dma1;
static volatile uint8_t  *src;

static int8_t  W[MAX_TILE * MAX_TILE], X[MAX_TILE * MAX_TILE];
static int8_t  Y[MAX_TILE * MAX_TILE], G[MAX_TILE * MAX_TILE];
static int8_t  Gnb[MAX_TILE * MAX_TILE];
static uint8_t wimg[MAX_TILE * MAX_TILE], ximg[MAX_TILE * MAX_TILE];
static uint8_t bimg[MAX_TILE * 4 + 16];
static int32_t bias[BIAS_ROWS];

static double stage_us, stage_mb;   /* the one-off staging cost */

/* ---- teardown: every exit path drops GO on both channels --------- */
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

static void on_dma_fatal(int sig)
{
    dma_cleanup();
    if (sig == SIGALRM) {
        static const char m[] = "\nWATCHDOG: hung, DMA stopped, exiting\n";
        ssize_t n = write(2, m, sizeof m - 1);
        (void)n;
    }
    _exit(sig == SIGALRM ? 2 : 128 + sig);
}

#ifndef TB_SIM
static int dma_one(volatile uint32_t *d, const char *who,
                   uint32_t rd, uint32_t wr, uint32_t bytes)
{
    d[DMA_STATUS] = 0;
    d[DMA_READ]   = rd;
    d[DMA_WRITE]  = wr;
    d[DMA_LENGTH] = bytes;
    __sync_synchronize();
    d[DMA_CONTROL] = CTRL_QUAD | CTRL_LEEN | CTRL_GO;

    uint64_t t0 = ns_now();
    for (;;) {
        uint32_t st = d[DMA_STATUS];
        if (st & DMA_DONE) return 0;
        if (ns_now() - t0 > 500000000ull) {
            printf("\n  %s TIMEOUT: STATUS 0x%08x (busy=%u), %u of %u left, "
                   "rd 0x%08x -> wr 0x%08x\n", who, st, (st & DMA_BUSY) ? 1u : 0u,
                   d[DMA_LENGTH], bytes, rd, wr);
            dma_stop(d);
            return -1;
        }
    }
}
#endif /* TB_SIM */

static int dma_go(volatile uint32_t *d, const char *who,
                  uint32_t rd, uint32_t wr, uint32_t bytes)
{
#ifdef TB_SIM
    /* the model has no DMA engine; move the bytes anyway so the staging
     * layout, direction and address arithmetic still get checked */
    (void)d; (void)who;
    if (rd >= SRC_PHYS) {
        memcpy((void *)(accel + wr), (const void *)(src + (rd - SRC_PHYS)),
               bytes);
    } else {
        memcpy((void *)(src + (wr - SRC_PHYS)), (const void *)(accel + rd),
               bytes);
    }
    return 0;
#else
    uint32_t done = 0;

    while (done < bytes) {
        uint32_t n = (bytes - done > CHUNK) ? CHUNK : bytes - done;
        if (dma_one(d, who, rd + done, wr + done, n) != 0) return -1;
        done += n;
    }
    return 0;
#endif
}

/* refuse to DMA into kernel-managed RAM */
static int staging_free(void)
{
    FILE *f;
    char line[256];
    int bad = 0;

#ifdef TB_SIM
    return 1;                   /* staging is a malloc'd buffer, not DDR */
#endif
    f = fopen("/proc/iomem", "r");
    if (!f) return 1;
    while (fgets(line, sizeof line, f)) {
        unsigned long a, b;
        char nm[128];
        if (sscanf(line, "%lx-%lx : %127[^\n]", &a, &b, nm) != 3) continue;
        if (strstr(nm, "System RAM") && SRC_PHYS <= b && a < SRC_PHYS + SRC_SPAN) {
            printf("FATAL: staging 0x%08lx overlaps System RAM %08lx-%08lx\n",
                   SRC_PHYS, a, b);
            bad = 1;
        }
    }
    fclose(f);
    return !bad;
}

/* ---- helpers ----------------------------------------------------- */
static int cmp_u64(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

static double med_us(uint64_t *v, int n)
{
    qsort(v, (size_t)n, sizeof *v, cmp_u64);
    return (double)v[n / 2] / 1e3;
}

/* only this tile's rows: bias[R] at byte 4*R, rounded up to a whole word
 * because an entry commits only when all its byte enables are set */
static size_t bias_bytes(int rows)
{
    return (size_t)CEILDIV(rows, 4) * 4 * 4;
}

static void build_bias_image(int rows)
{
    memset(bimg, 0, bias_bytes(rows));
    for (int r = 0; r < rows; r++) {
        uint32_t v = (uint32_t)bias[r];
        bimg[4 * r + 0] = (uint8_t)(v >>  0);
        bimg[4 * r + 1] = (uint8_t)(v >>  8);
        bimg[4 * r + 2] = (uint8_t)(v >> 16);
        bimg[4 * r + 3] = (uint8_t)(v >> 24);
    }
}

static int tile_legal(int tk, int tm, int tn)
{
    if (tk < 1 || tm < 1 || tn < 1) return 0;
    if (tk > MAX_TILE || tm > MAX_TILE) return 0;
    if (CEILDIV(tk, ARRAY_ROWS) * row_pitch(tn) > WBUF_BANK_BYTES) return 0;
    if (CEILDIV(tn, ARRAY_ROWS) * row_pitch(tm) > IBUF_BANK_BYTES) return 0;
    if (CEILDIV(tk, ARRAY_ROWS) * row_pitch(tm) > OBUF_BANK_BYTES) return 0;
    return CEILDIV(tk, ARRAY_ROWS) * row_pitch(tm) <= PSUM_BANK_SLOTS;
}

/*
 * A bias big enough to move the result by about 30 LSBs at this scale. The
 * bias is in accumulator units, so its useful magnitude depends on
 * OUTPUT_SCALE - at a scale pick_scale returns, +-600 would shift the result
 * by under one LSB and the bias runs would exercise nothing.
 */
static void fill_bias_for_scale(uint32_t q16)
{
    double lim = 30.0 * 65536.0 / (double)q16;

    if (lim < 8.0)  lim = 8.0;
    if (lim > 1e6)  lim = 1e6;
    for (int r = 0; r < BIAS_ROWS; r++) {
        bias[r] = (int32_t)(rand() % (2 * (int)lim + 1) - (int)lim);
    }
}

/* ---- one shape, one path, one bias mode -------------------------- */
static int measure(int tk, int tm, int tn, uint32_t q16, int reps,
                   int bias_en, int use_pio, int bias_pio,
                   double *dev_out, double *bias_leg)
{
    size_t wlen = (size_t)tk * row_pitch(tn);
    size_t ilen = (size_t)tn * row_pitch(tm);
    size_t olen = (size_t)tk * row_pitch(tm);
    size_t blen = bias_en ? bias_bytes(tk) : 0;
    double macs = (double)tk * tn * tm;
    char ch = use_pio ? 'P' : 'D';

    uint64_t vi[MAX_REPS], vb[MAX_REPS], vc[MAX_REPS], vo[MAX_REPS];

    /* ---- golden, and the setup checks ---- */
    golden_act(G, W, X, tk, tn, tm, q16, bias_en ? bias : NULL, NULL, NULL);
    if (bias_en) {
        size_t n = (size_t)tk * tm, moved = 0;

        golden_act(Gnb, W, X, tk, tn, tm, q16, NULL, NULL, NULL);
        for (size_t i = 0; i < n; i++) if (G[i] != Gnb[i]) moved++;
        /* the rescale floors, so even a +-1 bias flips a few elements -
         * require it to move most of the tile or the timing is of a run
         * that never exercises the adder's width */
        if (moved * 100 < n * 50) {
            printf("  %3d %4d %4d  %c %d   bias moves only %zu of %zu "
                   "elements - not timing it\n", tk, tm, tn, ch, bias_en,
                   moved, n);
            return 1;
        }
    }

    /* ---- stage the images once, outside the timed loop ---- */
    padded_image(wimg, W, tk, tn);
    padded_image(ximg, X, tn, tm);
    if (bias_en) build_bias_image(tk);

    uint64_t s0 = ns_now();
    if (!use_pio) {
        write_bytes_at(src + STAGE_W, wimg, wlen);
        write_bytes_at(src + STAGE_X, ximg, ilen);
        if (bias_en) write_bytes_at(src + STAGE_B, bimg, blen);
        __sync_synchronize();
    }
    uint64_t s1 = ns_now();
    if (!use_pio) {
        stage_us = (double)(s1 - s0) / 1e3;
        stage_mb = (double)(wlen + ilen + blen) / stage_us;
    }

    /* ---- timed ---- */
    for (int r = 0; r < reps; r++) {
        program_dims(tk, tn, tm, q16);
        program_tile(0, 0, 0, tm, 0, 0);
        program_act(bias_en ? ACT_BIAS_EN : 0);

        uint64_t t0 = ns_now();
        if (use_pio) {
            write_bytes(WBUF_BASE, wimg, wlen);
            write_bytes(IBUF_BASE, ximg, ilen);
        } else {
            if (dma_go(dma0, "dma_0 weights", SRC_PHYS + STAGE_W,
                       WBUF_BASE, (uint32_t)wlen)) return 1;
            if (dma_go(dma0, "dma_0 inputs", SRC_PHYS + STAGE_X,
                       IBUF_BASE, (uint32_t)ilen)) return 1;
        }
        uint64_t t1 = ns_now();

        if (bias_en) {
            if (use_pio || bias_pio) {
                write_bytes(BBUF_BASE, bimg, blen);
            } else if (dma_go(dma0, "dma_0 bias", SRC_PHYS + STAGE_B,
                              BBUF_BASE, (uint32_t)blen)) {
                printf("  the bias buffer may not be on dma_0's write port - "
                       "retry with -pb to send the bias by PIO\n");
                return 1;
            }
        }
        uint64_t t2 = ns_now();

        (void)reg_rd(STATUS_ADDR);
        reg_wr(CONTROL_ADDR, 1u << MATMUL_START_BIT);
        if (wait_matmul_done()) return 1;
        uint64_t t3 = ns_now();

        if (use_pio) {
            read_bytes(OBUF_BASE, (uint8_t *)Y, olen);
        } else if (dma_go(dma1, "dma_1 canvas", OBUF_BASE,
                          SRC_PHYS + STAGE_Y, (uint32_t)olen)) {
            return 1;
        }
        uint64_t t4 = ns_now();

        vi[r] = t1 - t0;
        vb[r] = t2 - t1;
        vc[r] = t3 - t2;
        vo[r] = t4 - t3;
    }

    /* ---- verify, untimed ---- */
    if (use_pio) {
        read_canvas(Y, tk, tm, tm);
    } else {
        uint8_t *raw = malloc(olen);
        int pitch = row_pitch(tm);
        read_bytes_at(src + STAGE_Y, raw, olen);
        for (int k = 0; k < tk; k++) {
            for (int m = 0; m < tm; m++) {
                Y[(size_t)k * tm + m] = (int8_t)raw[(size_t)k * pitch + m];
            }
        }
        free(raw);
    }
    if (memcmp(Y, G, (size_t)tk * tm) != 0) {
        size_t bad = 0;
        for (size_t i = 0; i < (size_t)tk * tm; i++) if (Y[i] != G[i]) bad++;
        printf("  %3d %4d %4d  %c %d   WRONG RESULT (%zu of %d) - not timing\n",
               tk, tm, tn, ch, bias_en, bad, tk * tm);
        return 1;
    }

    double ui = med_us(vi, reps), ub = med_us(vb, reps);
    double uc = med_us(vc, reps), uo = med_us(vo, reps);
    double dev = ui + ub + uc + uo;
    double dens = (double)tk * tm / (double)(tk + tm);

    /* in B / in MB/s cover the operands only; the bias has its own column */
    printf("  %3d %4d %4d  %c %d %7zu %7zu | %8.1f %7.1f %8.1f %8.1f | "
           "%6.1f %7.2f %8.2f\n",
           tk, tm, tn, ch, bias_en, wlen + ilen, olen,
           (double)(wlen + ilen) / ui, ub, (double)olen / uo, uc,
           dens, 2.0 * macs / (uc * 1e3), 2.0 * macs / (dev * 1e3));

    if (dev_out)  *dev_out = dev;
    if (bias_leg) *bias_leg = ub;
    return 0;
}

static void usage(const char *p)
{
    printf("usage: %s [-k TK] [-m TM] [-n TN] [-q REPS] [-a] [-p] [-pb] [-r SEED]\n", p);
    printf("  -k/-m/-n   one tile shape instead of the sweep\n");
    printf("  -a         sweep every legal combination of 16/32/64/128\n");
    printf("  -p         also measure the PIO path, for comparison\n");
    printf("  -pb        send the bias by PIO even on the DMA path\n");
    printf("  -q REPS    timed repetitions per shape (default 9, max %d)\n",
           MAX_REPS - 1);
    printf("  tile is capped at tk*tm <= %d elements by the psum buffer\n",
           ARRAY_ROWS * PSUM_BANK_SLOTS);
}

int main(int argc, char **argv)
{
    int fk = 0, fm = 0, fn = 0, all = 0, pio = 0, bias_pio = 0, reps = 9;
    unsigned seed = 71;
    int sizes[] = { 16, 32, 64, 128 };
    int fail = 0, nshape = 0;
    double sum_share = 0.0, sum_bias_us = 0.0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-k") && i + 1 < argc)      fk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-m") && i + 1 < argc) fm = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) fn = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-q") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) seed = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "-a"))  all = 1;
        else if (!strcmp(argv[i], "-pb")) bias_pio = 1;
        else if (!strcmp(argv[i], "-p"))  pio = 1;
        else { usage(argv[0]); return 1; }
    }
    if (reps < 1) reps = 1;
    if (reps >= MAX_REPS) reps = MAX_REPS - 1;

    if (!staging_free()) return 1;
    accel_open();
    lw_and_staging_open(LW_BASE, LW_SPAN, SRC_PHYS, SRC_SPAN);
    dma0 = (volatile uint32_t *)((volatile uint8_t *)lw_map + DMA0_OFF);
    dma1 = (volatile uint32_t *)((volatile uint8_t *)lw_map + DMA1_OFF);
    src  = staging_map;

    dma_cleanup();
    atexit(dma_cleanup);
    signal(SIGINT,  on_dma_fatal);
    signal(SIGTERM, on_dma_fatal);
    signal(SIGALRM, on_dma_fatal);

    srand(seed);
    fill_int8(W, sizeof W, 127);
    fill_int8(X, sizeof X, 127);

    printf("bandwidth_test: median of %d, tile capped at tk*tm <= %d (psum)\n",
           reps, ARRAY_ROWS * PSUM_BANK_SLOTS);
    printf("  p: D = DMA (dma_0 in, dma_1 out), P = PIO over h2f\n");
    printf("  b: 0 = Y = W@X, 1 = Y = W@X + bias\n");
    printf("  operands staged into the DMA window once, outside the timed "
           "loop\n\n");

    printf("  %3s %4s %4s %2s %1s %7s %7s | %8s %7s %8s %8s | %6s %7s %8s\n",
           "tk", "tm", "tn", "p", "b", "in B", "out B",
           "in MB/s", "bias us", "out MB/s", "comp us",
           "MAC/B", "GOP/s c", "GOP/s dev");
    printf("  -----------------------------------------------------"
           "-------------------------------------------------\n");

    #define RUN_SHAPE(tk, tm, tn) do {                                        \
        double d0 = 0.0, d1 = 0.0, bleg = 0.0;                                \
        uint32_t q16 = pick_scale(W, X, (tk), (tn), (tm));                    \
        fill_bias_for_scale(q16);                                             \
        fail |= measure((tk),(tm),(tn),q16,reps,0,0,bias_pio,&d0,NULL);       \
        fail |= measure((tk),(tm),(tn),q16,reps,1,0,bias_pio,&d1,&bleg);      \
        if (pio) {                                                            \
            fail |= measure((tk),(tm),(tn),q16,reps,0,1,bias_pio,NULL,NULL);  \
            fail |= measure((tk),(tm),(tn),q16,reps,1,1,bias_pio,NULL,NULL);  \
        }                                                                     \
        if (d0 > 0.0 && d1 > 0.0) {                                           \
            sum_share += 100.0 * (d1 - d0) / d0;                              \
            sum_bias_us += bleg;                                              \
            nshape++;                                                         \
        }                                                                     \
    } while (0)

    if (fk || fm || fn) {
        int tk = fk ? fk : MAX_TILE;
        int tm = fm ? fm : MAX_TILE;
        int tn = fn ? fn : MAX_TILE;

        if (!tile_legal(tk, tm, tn)) {
            printf("  tk=%d tm=%d tn=%d is not legal", tk, tm, tn);
            if (tk > MAX_TILE || tm > MAX_TILE) {
                printf(" (tk and tm are capped at %d)", MAX_TILE);
            }
            printf(":\n");
            printf("    wbuf   %5d / %d\n",
                   CEILDIV(tk, ARRAY_ROWS) * row_pitch(tn), WBUF_BANK_BYTES);
            printf("    ibuf   %5d / %d\n",
                   CEILDIV(tn, ARRAY_ROWS) * row_pitch(tm), IBUF_BANK_BYTES);
            printf("    psum   %5d / %d\n",
                   CEILDIV(tk, ARRAY_ROWS) * row_pitch(tm), PSUM_BANK_SLOTS);
            printf("    canvas %5d / %d\n",
                   CEILDIV(tk, ARRAY_ROWS) * row_pitch(tm), OBUF_BANK_BYTES);
            return 1;
        }
        RUN_SHAPE(tk, tm, tn);
    } else {
        for (unsigned a = 0; a < sizeof sizes / sizeof sizes[0]; a++) {
            for (unsigned b = 0; b < sizeof sizes / sizeof sizes[0]; b++) {
                for (unsigned c = 0; c < sizeof sizes / sizeof sizes[0]; c++) {
                    int tk = sizes[a], tm = sizes[b], tn = sizes[c];

                    if (!all && !(tk == tm && (tk == tn || tk == MAX_TILE)))
                        continue;
                    if (!tile_legal(tk, tm, tn)) continue;
                    RUN_SHAPE(tk, tm, tn);
                }
            }
        }
    }
    #undef RUN_SHAPE

    printf("\n");
    if (nshape) {
        printf("  bias cost over %d shape(s): the leg is %.2f us, and enabling "
               "it moves\n  the device time by %+.1f%%. It is 4 bytes per output "
               "row - %d B at\n  tk=128 - so it is per-access overhead, not "
               "size.\n\n",
               nshape, sum_bias_us / nshape, sum_share / nshape,
               (int)bias_bytes(128));
    }
    if (stage_us > 0.0) {
        printf("  staging the operands into the DMA window cost %.1f us "
               "(%.1f MB/s), once,\n  outside the timed loop. That rate is the "
               "staging buffer inheriting\n  Device-nGnRnE from the O_SYNC "
               "mapping, not the DMA.\n\n", stage_us, stage_mb);
    }
    printf("  density is tk*tm/(tk+tm), independent of tn: only a square tile\n"
           "  filling the psum buffer reaches the %d MAC/byte ceiling.\n",
           MAX_TILE / 2);
    printf("  the tn rows are NOT one problem chunked differently - each is a\n"
           "  standalone matmul, so a smaller tn is proportionally less work\n"
           "  against the same %d B readback. A contraction split across several\n"
           "  use_prev_out matmuls pays that readback once, so subtract `out B`\n"
           "  from all but the last to model it.\n", MAX_TILE * MAX_TILE);

    alarm(0);
    dma_cleanup();
    dma0 = NULL;
    dma1 = NULL;
    return fail;
}