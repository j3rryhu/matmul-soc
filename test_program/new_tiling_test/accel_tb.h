/*
 * accel_tb.h - shared harness for the C ports of the cocotb tests.
 *
 * One test per .c file; this holds what they all need: the register map,
 * the buffer image layouts, canvas readback, and the golden models. All
 * access is PIO over the h2f bridge, the same single-word accesses the
 * cocotb avalon_write/avalon_read do - no DMA anywhere.
 *
 * Build any test with:
 *   gcc -O2 -Wall -Wextra -mcpu=cortex-a9 -mfpu=neon -o <t> <t>.c -lm
 * and run it as root (needs /dev/mem).
 *
 * ---------------------------------------------------------------------------
 * Convention, from rdl/ctrl_reg.rdl and the TB's program_dims()
 * ---------------------------------------------------------------------------
 *   WEIGHT_ROWS = K   output rows of this matmul
 *   WEIGHT_COLS = N   contraction length
 *   INPUT_COLS  = M   output columns   (INPUT_MAX_ADDR gets M too)
 *
 * The RDL's own comment at WEIGHT_ROWS/WEIGHT_COLS still describes the
 * opposite axes; the TILE_CFG comment, program_dims() and every passing run
 * agree with the above. W is K x N, X is N x M, Y is K x M.
 *
 * Images are written as ONE contiguous row-major byte image per buffer with
 * every row padded to a 16-byte word; the hardware remap scatters row r to
 * bank r % 16 (rtl/buf_addr_remap.v). Dimensions must be programmed BEFORE
 * the buffers, because the remap uses WEIGHT_COLS and INPUT_MAX_ADDR to
 * place each word.
 *
 * ---------------------------------------------------------------------------
 * Accumulation and activation model
 * ---------------------------------------------------------------------------
 * Raw 32-bit sums accumulate across contraction blocks in the psum buffer.
 * The activation stage runs only on the LAST contraction block of an output
 * row block:
 *
 *      acc32  = sum over the whole contraction of W*X
 *      if bias_en:   acc32 += bias[canvas_row % BIAS_ROWS]
 *      y8     = sat8((acc32 * OUTPUT_SCALE) >> 16)
 *      if prev_int8: y8 = sat8(y8 + canvas[row][col])
 *      if gelu_en:   y8 = gelu_lut[y8]
 *
 * out_raw stops before all of that and leaves the raw sums in the psum
 * buffer for a later use_prev_out matmul on the same tile.
 */

#ifndef ACCEL_TB_H
#define ACCEL_TB_H

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

/* ---- memory map (must match rtl/accel_top.sv) --------------------- */
#define ACCEL_PHYS 0xC0000000UL
#define ACCEL_SPAN 0x00020000UL

#define CTRL_BASE 0x0000u
#define BBUF_BASE 0x0800u   /* bias_buffer: int32 bias of canvas row R at 4*R */
#define WBUF_BASE 0x1000u
#define IBUF_BASE 0x5000u
#define OBUF_BASE 0x9000u

/* one register per 128-bit bus word, value in byte lanes 0..3 */
#define CONTROL_ADDR        (CTRL_BASE + 0x00u)
#define WEIGHT_ROWS_ADDR    (CTRL_BASE + 0x10u)
#define WEIGHT_COLS_ADDR    (CTRL_BASE + 0x20u)
#define INPUT_MAX_ADDR_ADDR (CTRL_BASE + 0x30u)
#define INPUT_COLS_ADDR     (CTRL_BASE + 0x40u)
#define STATUS_ADDR         (CTRL_BASE + 0x50u)
#define OUTPUT_SCALE_ADDR   (CTRL_BASE + 0x60u)
#define TILE_CFG_ADDR       (CTRL_BASE + 0x70u)
#define CANVAS_WIDTH_ADDR   (CTRL_BASE + 0x80u)
#define ACT_CFG_ADDR        (CTRL_BASE + 0x90u)
#define TILE_ROWS_ADDR      (CTRL_BASE + 0xA0u)
#define TILE_COLS_ADDR      (CTRL_BASE + 0xB0u)

#define MATMUL_START_BIT 0

/* field bits, assigned by PeakRDL in declaration order */
#define STATUS_BUSY_BIT 0
#define STATUS_DONE_BIT 1

#define ACT_BIAS_EN   (1u << 0)
#define ACT_GELU_EN   (1u << 1)
#define ACT_PREV_INT8 (1u << 2)
#define ACT_OUT_RAW   (1u << 3)

#define TILE_R_SHIFT   0
#define TILE_C_SHIFT   15
#define TILE_FIELD_MSK 0x7FFFu
#define USE_PREV_BIT   (1u << 30)

/* ---- array and buffer geometry ------------------------------------ */
#define ARRAY_ROWS 16
#define ARRAY_COLS 16
#define BYTES_PER_WORD 16

#define WBUF_BANK_BYTES 1024
#define IBUF_BANK_BYTES 1024
#define OBUF_BANK_BYTES 4096
#define OBUF_BYTES ((size_t)ARRAY_ROWS * OBUF_BANK_BYTES)

/* psum buffer holds one tile's raw 32-bit sums, up to 128x128 */
#define PSUM_BANK_SLOTS 1024

/* bias buffer entry = canvas row mod BIAS_ROWS (rdl ACT_CFG.bias_en) */
#define BIAS_ROWS 256

#define DONE_TIMEOUT_NS 2000000000ull
#define WATCHDOG_S      120

#define CEILDIV(a, b) (((a) + (b) - 1) / (b))

/* each test uses a subset of the helpers below; don't warn about the rest */
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

static volatile uint8_t *accel;

/* ---- host row pitch: every row padded to a whole word -------------- */
static int row_pitch(int ncols)
{
    return CEILDIV(ncols, BYTES_PER_WORD) * BYTES_PER_WORD;
}

static int8_t sat8(int32_t v)
{
    if (v >  127) return  127;
    if (v < -128) return -128;
    return (int8_t)v;
}

/*
 * (acc * q16) >> 16 with an arithmetic shift, matching output_loader's
 * requantize bit-for-bit. The shift floors, so a negative value rounds away
 * from zero - same as numpy's >> in the cocotb golden.
 *
 * Unsaturated on purpose: there is exactly ONE saturation on the path, and
 * it comes after the prev_int8 add. Saturating here as well would clamp a
 * large rescaled value to 127 before a negative prev could bring it back
 * into range, which reads low - 17 of 672 elements in test5 run 3.
 */
static int32_t rescale(int64_t acc, uint32_t q16)
{
    return (int32_t)((acc * (int64_t)q16) >> 16);
}

/* rescale and saturate, for the paths with no prev_int8 to add */
static int8_t requant(int64_t acc, uint32_t q16)
{
    return sat8(rescale(acc, q16));
}

/* ---- timing and teardown ------------------------------------------ */
static uint64_t ns_now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

static void on_fatal(int sig)
{
    if (sig == SIGALRM) {
        static const char m[] = "\nWATCHDOG: no progress, exiting\n";
        ssize_t n = write(2, m, sizeof m - 1);
        (void)n;
    }
    _exit(sig == SIGALRM ? 2 : 128 + sig);
}

/* ---- 128-bit Device-memory accesses, one per call ----------------- */
static void beat_wr(volatile uint8_t *p, const uint8_t *b)
{
#if HAVE_NEON
    uint8x16_t v = vld1q_u8(b);
    __asm__ volatile("vst1.8 {%e0-%f0}, [%1 :128]" : : "w"(v), "r"(p) : "memory");
#else
    for (int j = 0; j < BYTES_PER_WORD; j++) p[j] = b[j];
#endif
}

static void beat_rd(volatile uint8_t *p, uint8_t *b)
{
#if HAVE_NEON
    uint8x16_t v;
    __asm__ volatile("vld1.8 {%e0-%f0}, [%1 :128]" : "=w"(v) : "r"(p) : "memory");
    vst1q_u8(b, v);
#else
    for (int j = 0; j < BYTES_PER_WORD; j++) b[j] = p[j];
#endif
}

/*
 * TB_SIM builds the test against a software model of the device instead of
 * /dev/mem (see sim_model.c), so the image layouts, tile placement and
 * activation ordering can be checked off the board. The only hooks it needs
 * are the matmul_start pulse and STATUS's read-to-clear.
 */
#ifdef TB_SIM
static void sim_matmul_start(void);
static void sim_open(void);
#endif

static void reg_wr(uint32_t off, uint32_t v)
{
    *(volatile uint32_t *)(accel + off) = v;
#ifdef TB_SIM
    if (off == CONTROL_ADDR && (v & (1u << MATMUL_START_BIT))) {
        sim_matmul_start();
    }
#endif
}

/* plain read with no read-to-clear side effect; the model uses this */
static uint32_t reg_rd_raw(uint32_t off)
{
    return *(volatile uint32_t *)(accel + off);
}

static uint32_t reg_rd(uint32_t off)
{
    uint32_t v = *(volatile uint32_t *)(accel + off);
#ifdef TB_SIM
    if (off == STATUS_ADDR) {       /* done is rclr */
        *(volatile uint32_t *)(accel + off) = v & ~(1u << STATUS_DONE_BIT);
    }
#endif
    return v;
}

/*
 * Write a byte image at a word-aligned offset. The buffers' host ports move
 * a whole 16-byte word per access, so a short tail is zero-padded - callers
 * hand over a whole contiguous image, never a slice that could straddle
 * live data.
 */
static void write_bytes_at(volatile uint8_t *p, const uint8_t *vals, size_t len)
{
    uint8_t w[BYTES_PER_WORD];
    size_t i = 0;

    for (; i + BYTES_PER_WORD <= len; i += BYTES_PER_WORD) {
        beat_wr(p + i, vals + i);
    }
    if (i < len) {
        memset(w, 0, sizeof w);
        memcpy(w, vals + i, len - i);
        beat_wr(p + i, w);
    }
}

static void read_bytes_at(volatile uint8_t *p, uint8_t *dst, size_t len)
{
    uint8_t w[BYTES_PER_WORD];
    size_t i = 0;

    for (; i + BYTES_PER_WORD <= len; i += BYTES_PER_WORD) {
        beat_rd(p + i, dst + i);
    }
    if (i < len) {
        beat_rd(p + i, w);
        memcpy(dst + i, w, len - i);
    }
}

static void write_bytes(uint32_t off, const uint8_t *vals, size_t len)
{
    write_bytes_at(accel + off, vals, len);
}

static void read_bytes(uint32_t off, uint8_t *dst, size_t len)
{
    read_bytes_at(accel + off, dst, len);
}

/* ---- row-major byte image with rows padded to row_pitch ----------- */
static void padded_image(uint8_t *img, const int8_t *A, int rows, int cols)
{
    int p = row_pitch(cols);

    memset(img, 0, (size_t)rows * p);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            img[(size_t)r * p + c] = (uint8_t)A[(size_t)r * cols + c];
        }
    }
}

/* ---- register programming ----------------------------------------- */
static void program_dims(int K, int N, int M, uint32_t q16)
{
    reg_wr(WEIGHT_ROWS_ADDR, (uint32_t)K);
    reg_wr(WEIGHT_COLS_ADDR, (uint32_t)N);
    reg_wr(INPUT_MAX_ADDR_ADDR, (uint32_t)M);
    reg_wr(INPUT_COLS_ADDR, (uint32_t)M);
    reg_wr(OUTPUT_SCALE_ADDR, q16);
}

static void program_act(uint32_t bits)
{
    reg_wr(ACT_CFG_ADDR, bits);
}

static void program_tile(int tile_r, int tile_c, int use_prev,
                         int canvas_width, int tile_rows, int tile_cols)
{
    uint32_t v = 0;
    v |= ((uint32_t)tile_r & TILE_FIELD_MSK) << TILE_R_SHIFT;
    v |= ((uint32_t)tile_c & TILE_FIELD_MSK) << TILE_C_SHIFT;
    if (use_prev) v |= USE_PREV_BIT;

    reg_wr(TILE_CFG_ADDR, v);
    reg_wr(CANVAS_WIDTH_ADDR, (uint32_t)canvas_width);
    reg_wr(TILE_ROWS_ADDR, (uint32_t)tile_rows);
    reg_wr(TILE_COLS_ADDR, (uint32_t)tile_cols);
}

static void write_weights(const int8_t *W, int K, int N)
{
    size_t len = (size_t)K * row_pitch(N);
    uint8_t *img = malloc(len);

    if (CEILDIV(K, ARRAY_ROWS) * row_pitch(N) > WBUF_BANK_BYTES) {
        printf("  FATAL: weight bank needs %d B of %d\n",
               CEILDIV(K, ARRAY_ROWS) * row_pitch(N), WBUF_BANK_BYTES);
        exit(1);
    }
    padded_image(img, W, K, N);
    write_bytes(WBUF_BASE, img, len);
    free(img);
}

static void write_inputs(const int8_t *X, int N, int M)
{
    size_t len = (size_t)N * row_pitch(M);
    uint8_t *img = malloc(len);

    if (CEILDIV(N, ARRAY_ROWS) * row_pitch(M) > IBUF_BANK_BYTES) {
        printf("  FATAL: input bank needs %d B of %d\n",
               CEILDIV(N, ARRAY_ROWS) * row_pitch(M), IBUF_BANK_BYTES);
        exit(1);
    }
    padded_image(img, X, N, M);
    write_bytes(IBUF_BASE, img, len);
    free(img);
}

/*
 * The whole bias vector as one contiguous image of little-endian int32s,
 * bias[R] at byte 4*R, which is what bias_buffer_32_bank's wide write port
 * expects - it splits each host word into ROWS_PER_WORD neighbouring banks
 * itself. Write-only from the host. A bias entry is only committed when all
 * of its byte enables are set, and BIAS_ROWS*4 is a whole number of words,
 * so writing the vector entire always commits every entry.
 */
static void write_bias(const int32_t *bias)
{
    uint8_t img[BIAS_ROWS * 4];

    for (int r = 0; r < BIAS_ROWS; r++) {
        uint32_t v = (uint32_t)bias[r];
        img[4 * r + 0] = (uint8_t)(v >>  0);
        img[4 * r + 1] = (uint8_t)(v >>  8);
        img[4 * r + 2] = (uint8_t)(v >> 16);
        img[4 * r + 3] = (uint8_t)(v >> 24);
    }
    write_bytes(BBUF_BASE, img, sizeof img);
}

static void fill_int32(int32_t *A, size_t n, int lim)
{
    for (size_t i = 0; i < n; i++) {
        A[i] = rand() % (2 * lim + 1) - lim;
    }
}

static int wait_matmul_done(void)
{
    uint64_t t0 = ns_now();

    for (;;) {
        uint32_t st = reg_rd(STATUS_ADDR);
        if (st & (1u << STATUS_DONE_BIT)) return 0;
        if (ns_now() - t0 > DONE_TIMEOUT_NS) {
            printf("\n  TIMEOUT waiting for done, STATUS = 0x%08x\n", st);
            return -1;
        }
    }
}

/*
 * One complete matmul: dimensions and placement first, then the buffers,
 * then start. W is Ktile x N, X is N x Mtile.
 */
static int run_tile(const int8_t *W, const int8_t *X,
                    int Ktile, int N, int Mtile, uint32_t q16,
                    int tile_r, int tile_c, int canvas_width,
                    int use_prev, int tile_rows, int tile_cols)
{
    program_dims(Ktile, N, Mtile, q16);
    program_tile(tile_r, tile_c, use_prev, canvas_width, tile_rows, tile_cols);
    write_weights(W, Ktile, N);
    write_inputs(X, N, Mtile);

    (void)reg_rd(STATUS_ADDR);              /* done is rclr - clear it */
    reg_wr(CONTROL_ADDR, 1u << MATMUL_START_BIT);
    return wait_matmul_done();
}

/*
 * Read a K x M window out of the canvas. The output buffer holds an int8
 * canvas whose rows are padded to row_pitch(canvas_width), so host address
 * = OBUF_BASE + row*pitch + col.
 */
static void read_canvas(int8_t *Y, int K, int M, int canvas_width)
{
    int pitch = row_pitch(canvas_width);
    size_t len = (size_t)K * pitch;
    uint8_t *raw = malloc(len);

    read_bytes(OBUF_BASE, raw, len);
    for (int k = 0; k < K; k++) {
        for (int m = 0; m < M; m++) {
            Y[(size_t)k * M + m] = (int8_t)raw[(size_t)k * pitch + m];
        }
    }
    free(raw);
}

static void assert_fits_canvas(int K, int M)
{
    int need = CEILDIV(K, ARRAY_ROWS) * row_pitch(M);

    if (need > OBUF_BANK_BYTES) {
        printf("  FATAL: canvas needs %d B per bank of %d (K=%d, M=%d)\n",
               need, OBUF_BANK_BYTES, K, M);
        exit(1);
    }
}

/* ---- golden models ------------------------------------------------- */

/* raw int32 sums over the whole contraction - what the psum buffer holds */
static void golden_raw(int32_t *acc, const int8_t *W, const int8_t *X,
                       int K, int N, int M)
{
    for (int k = 0; k < K; k++) {
        for (int m = 0; m < M; m++) {
            int32_t s = 0;
            for (int p = 0; p < N; p++) {
                s += (int32_t)W[(size_t)k * N + p] * (int32_t)X[(size_t)p * M + m];
            }
            acc[(size_t)k * M + m] = s;
        }
    }
}

/*
 * The full activation path. bias / lut / prev are optional (NULL to skip),
 * matching the cocotb golden_act(W, X, q16, bias, lut, prev).
 *
 *   acc32 -> +bias -> rescale -> +prev -> saturate (once) -> GELU
 *
 * Note the single saturation: prev is added to the UNSATURATED rescaled
 * value. bias is indexed by output row, which the caller offsets to the
 * canvas row when the tile is not at tile_r = 0.
 */
static void golden_act(int8_t *Y, const int8_t *W, const int8_t *X,
                       int K, int N, int M, uint32_t q16,
                       const int32_t *bias, const int8_t *lut,
                       const int8_t *prev)
{
    int32_t *acc = malloc((size_t)K * M * sizeof *acc);

    golden_raw(acc, W, X, K, N, M);
    for (int k = 0; k < K; k++) {
        for (int m = 0; m < M; m++) {
            size_t i = (size_t)k * M + m;
            int64_t a = acc[i];

            if (bias) a += bias[k];

            int32_t r = rescale(a, q16);
            int8_t y = prev ? sat8(r + prev[i]) : sat8(r);

            if (lut) y = lut[(uint8_t)y];
            Y[i] = y;
        }
    }
    free(acc);
}

/* ---- comparison ---------------------------------------------------- */
static int check_canvas(const int8_t *got, const int8_t *exp,
                        int K, int M, const char *what)
{
    size_t bad = 0;
    int shown = 0;

    for (int k = 0; k < K; k++) {
        for (int m = 0; m < M; m++) {
            size_t i = (size_t)k * M + m;
            if (got[i] == exp[i]) continue;
            bad++;
            if (shown < 8) {
                printf("    y[%d,%d]: expected %4d, got %4d\n",
                       k, m, exp[i], got[i]);
                shown++;
            }
        }
    }
    printf("  %-6s %s  (%zu of %d mismatched)\n",
           bad ? "FAIL" : "PASS", what, bad, K * M);
    return bad != 0;
}

/* ---- random int8 data and a scale that lands mid-range ------------- */
static void fill_int8(int8_t *A, size_t n, int lim)
{
    for (size_t i = 0; i < n; i++) {
        A[i] = (int8_t)(rand() % (2 * lim + 1) - lim);
    }
}

/*
 * Pick OUTPUT_SCALE so the rescaled result lands near +-40 rather than
 * pinned at the saturation limits: a canvas of mostly +-127 would compare
 * equal to the golden even with a broken remap.
 */
static uint32_t pick_scale(const int8_t *W, const int8_t *X,
                           int K, int N, int M)
{
    int ks = (K >= 8) ? K / 8 : 1;
    int ms = (M >= 8) ? M / 8 : 1;
    double sum2 = 0.0;
    long cnt = 0;

    for (int k = 0; k < K; k += ks) {
        for (int m = 0; m < M; m += ms) {
            int32_t s = 0;
            for (int p = 0; p < N; p++) {
                s += (int32_t)W[(size_t)k * N + p] * (int32_t)X[(size_t)p * M + m];
            }
            sum2 += (double)s * (double)s;
            cnt++;
        }
    }
    if (cnt == 0) return 1;

    double rms = sqrt(sum2 / (double)cnt);
    if (rms < 1.0) return 65535u;

    double q = 40.0 * 65536.0 / rms;
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

/* ---- setup --------------------------------------------------------- */
static void accel_open(void)
{
#ifdef TB_SIM
    setvbuf(stdout, NULL, _IONBF, 0);
    sim_open();
    return;
#else
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    void *m;

    if (fd < 0) { perror("open /dev/mem (run as root)"); exit(1); }
    m = mmap(NULL, ACCEL_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
             (off_t)ACCEL_PHYS);
    if (m == MAP_FAILED) { perror("mmap"); exit(1); }
    close(fd);
    accel = (volatile uint8_t *)m;

    setvbuf(stdout, NULL, _IONBF, 0);
    signal(SIGINT, on_fatal);
    signal(SIGTERM, on_fatal);
    signal(SIGBUS, on_fatal);
    signal(SIGSEGV, on_fatal);
    signal(SIGALRM, on_fatal);
    alarm(WATCHDOG_S);

    uint32_t st = reg_rd(STATUS_ADDR);
    if (st & ~0x3u) {
        printf("  STATUS = 0x%08x  <- unexpected bits: register spacing?\n", st);
    }
#endif /* TB_SIM */
}

/*
 * The LW bridge (DMA registers) and a staging window in DDR. Only the
 * bandwidth test needs these; the functional tests are PIO only and never
 * call this. Under TB_SIM they are plain buffers, which is enough to check
 * the staging layout and address arithmetic.
 */
static volatile uint8_t *lw_map;
static volatile uint8_t *staging_map;

static void lw_and_staging_open(unsigned long lw_base, size_t lw_span,
                                unsigned long src_base, size_t src_span)
{
#ifdef TB_SIM
    (void)lw_base; (void)src_base;
    lw_map = calloc(1, lw_span);
    staging_map = calloc(1, src_span);
    if (!lw_map || !staging_map) { printf("sim: out of memory\n"); exit(1); }
#else
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    void *l, *s;

    if (fd < 0) { perror("open /dev/mem (run as root)"); exit(1); }
    l = mmap(NULL, lw_span, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
             (off_t)lw_base);
    s = mmap(NULL, src_span, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
             (off_t)src_base);
    if (l == MAP_FAILED || s == MAP_FAILED) { perror("mmap"); exit(1); }
    close(fd);
    lw_map = (volatile uint8_t *)l;
    staging_map = (volatile uint8_t *)s;
#endif
}

#ifdef TB_SIM
#include "sim_model.h"
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#endif /* ACCEL_TB_H */