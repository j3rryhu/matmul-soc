/*
 * test1_tiled_canvas_use_prev_out.c
 *
 * Y (256x256) = W (256x32) @ X (32x256), twice, as eight matmuls filling the
 * whole output buffer: a 256-wide canvas (CANVAS_WIDTH = 256) of 2x2 output
 * tiles of 128x128, each tile's contraction split into two halves of 16
 * joined with TILE_CFG.use_prev_out. Each 128x128 tile fills the whole psum
 * buffer by itself.
 *
 * Part 1, int8 continuation, half-outer, tile-inner: all four tiles' first
 * halves, then all four second halves with ACT_CFG.prev_int8, adding their
 * rescaled sum to the int8 the first half left in the output buffer. Checks
 * that a second half adds onto ITS OWN tile's int8, with three other tiles
 * written in between.
 *
 * Part 2, raw continuation, tile-outer, half-inner - the psum buffer holds
 * one tile, so a raw chain has to finish its tile before the next starts.
 * The first half sets ACT_CFG.out_raw and leaves raw 32-bit sums; the second
 * (prev_int8 = 0) adds onto them and rescales once, exactly as the hardware
 * does between contraction blocks of one matmul - so the expected result is
 * a single rescale over the full N = 32.
 *
 * Both check that tile (tr, tc) lands at canvas rows 128*tr.., columns
 * 128*tc.. - every byte of every output bank, read back as one 256x256
 * image - and that the first halves overwrite whatever the canvas held
 * before. Unlike the cocotb version, which relies on earlier tests leaving
 * data in the canvas, this one scribbles the canvas itself before each part
 * so the overwrite check holds when the test runs alone.
 *
 * gcc -O2 -Wall -Wextra -mcpu=cortex-a9 -mfpu=neon -o test1 test1_tiled_canvas_use_prev_out.c -lm
 */

#include "accel_tb.h"

#define T 128                  /* tile edge: Ktile = Mtile = T per matmul */
#define H ARRAY_COLS           /* contraction half: N = H per matmul */
#define K (2 * T)
#define M (2 * T)
#define N (2 * H)

static int8_t W[K * N], X[N * M];
static int8_t Wt[T * H], Xt[H * T];     /* one tile's slices */
static int8_t Y[K * M], G[K * M], first[K * M];

static const int tiles[4][2] = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };

/* W[tr*T .. , half*H ..] and X[half*H .. , tc*T ..] for one matmul */
static int run_half(uint32_t q16, int half, int tr, int tc, int use_prev)
{
    for (int i = 0; i < T; i++) {
        memcpy(&Wt[(size_t)i * H], &W[(size_t)(tr * T + i) * N + half * H],
               (size_t)H);
    }
    for (int p = 0; p < H; p++) {
        memcpy(&Xt[(size_t)p * T], &X[(size_t)(half * H + p) * M + tc * T],
               (size_t)T);
    }
    /* tile_rows / tile_cols = 0: equal-size tiles, grid is this matmul's own
     * WEIGHT_ROWS x INPUT_COLS = 128 x 128 */
    return run_tile(Wt, Xt, T, H, T, q16, tr, tc, M, use_prev, 0, 0);
}

/* fill the canvas with a value no correct result can leave behind */
static void scribble_canvas(void)
{
    size_t len = (size_t)K * row_pitch(M);
    uint8_t *junk = malloc(len);

    memset(junk, 0x5A, len);
    write_bytes(OBUF_BASE, junk, len);
    free(junk);
}

/* one half of the contraction, as a K x M problem over N = H */
static void golden_half(int8_t *out, int half, uint32_t q16,
                        const int8_t *prev)
{
    static int8_t Wh[K * H], Xh[H * M];

    for (int k = 0; k < K; k++) {
        memcpy(&Wh[(size_t)k * H], &W[(size_t)k * N + half * H], (size_t)H);
    }
    memcpy(Xh, &X[(size_t)half * H * M], (size_t)H * M);
    golden_act(out, Wh, Xh, K, H, M, q16, NULL, NULL, prev);
}

int main(void)
{
    int fail = 0;

    /* the asserts the cocotb test makes about its own shape */
    if (CEILDIV(K, ARRAY_ROWS) * row_pitch(M) != OBUF_BANK_BYTES) {
        printf("test is meant to fill the whole output buffer\n");
        return 1;
    }
    if (CEILDIV(T, ARRAY_ROWS) * T != PSUM_BANK_SLOTS) {
        printf("each tile is meant to fill the whole psum buffer\n");
        return 1;
    }

    accel_open();
    printf("test1: %dx%d canvas of 2x2 tiles of %dx%d, contraction %d "
           "in halves of %d\n", K, M, T, T, N, H);

    /* ---- part 1: prev_int8, half-outer ---- */
    srand(41);
    fill_int8(W, sizeof W, 127);
    fill_int8(X, sizeof X, 127);
    uint32_t q16 = pick_scale(W, X, K, N, M);

    golden_half(first, 0, q16, NULL);
    golden_half(G, 1, q16, first);

    printf("\n  part 1: int8 continuation (prev_int8), scale 0x%04x, "
           "%.1f%% saturated\n", q16, sat_fraction(G, (size_t)K * M));

    scribble_canvas();
    for (int half = 0; half < 2; half++) {
        program_act(half == 1 ? ACT_PREV_INT8 : 0);
        for (int t = 0; t < 4; t++) {
            if (run_half(q16, half, tiles[t][0], tiles[t][1], half == 1)) {
                printf("  FAIL   part 1 timed out on half %d tile (%d,%d)\n",
                       half, tiles[t][0], tiles[t][1]);
                return 1;
            }
        }
    }
    read_canvas(Y, K, M, M);
    fail |= check_canvas(Y, G, K, M,
                         "part 1: 2x2 tiles of 128x128, halves joined with prev_int8");

    /* ---- part 2: raw sums (out_raw), tile-outer ---- */
    srand(42);
    fill_int8(W, sizeof W, 127);
    fill_int8(X, sizeof X, 127);
    q16 = pick_scale(W, X, K, N, M);

    /* raw continuation rescales once over the whole contraction */
    golden_act(G, W, X, K, N, M, q16, NULL, NULL, NULL);

    printf("\n  part 2: raw continuation (out_raw), scale 0x%04x, "
           "%.1f%% saturated\n", q16, sat_fraction(G, (size_t)K * M));

    scribble_canvas();
    for (int t = 0; t < 4; t++) {
        for (int half = 0; half < 2; half++) {
            program_act(half == 0 ? ACT_OUT_RAW : 0);
            if (run_half(q16, half, tiles[t][0], tiles[t][1], half == 1)) {
                printf("  FAIL   part 2 timed out on tile (%d,%d) half %d\n",
                       tiles[t][0], tiles[t][1], half);
                return 1;
            }
        }
    }
    read_canvas(Y, K, M, M);
    fail |= check_canvas(Y, G, K, M,
                         "part 2: 2x2 tiles of 128x128, halves joined through raw sums");

    alarm(0);
    printf("\n%s\n", fail ? "TEST1 FAILED" : "test1 passed");
    return fail;
}
