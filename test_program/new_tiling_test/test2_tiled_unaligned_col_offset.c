/*
 * test2_tiled_unaligned_col_offset.c
 *
 * 2x2 tiles of 16 x 37 (Ktile = 16, N = 32, Mtile = 37), each a complete
 * matmul, on a canvas CANVAS_WIDTH = 74 wide:
 *   - int8 canvas rows are 74 bytes, padded to 80 - five words, not a power
 *     of two, so the write side's segment multiply and the read remap's
 *     divide both have to agree on a real pitch;
 *   - the right-hand tiles start at canvas column 37, mid-word, so tile
 *     (r, 1) shares the bus word holding columns 32..47 with tile (r, 0):
 *     its int8 byte writes must not disturb the left tile's columns 32..36;
 *   - N = 32 is two contraction blocks, so the right-hand tiles' raw sums go
 *     through the psum buffer from column 37 on too;
 *   - the bottom tiles start at canvas row 16, one bank row block down.
 * Together the four tiles must hold exactly W @ X, rescaled once.
 *
 * gcc -O2 -Wall -Wextra -mcpu=cortex-a9 -mfpu=neon -o test2 test2_tiled_unaligned_col_offset.c -lm
 */

#include "accel_tb.h"

#define T_K ARRAY_ROWS
#define T_M 37
#define K (2 * T_K)              /* 32 */
#define N (2 * ARRAY_COLS)       /* 32, two contraction blocks */
#define M (2 * T_M)              /* 74, the canvas width */

static int8_t W[K * N], X[N * M];
static int8_t Wt[T_K * N], Xt[N * T_M];
static int8_t Y[K * M], G[K * M];

int main(void)
{
    int fail = 0;
    int pitch_words;

    /* the asserts the cocotb test makes about its own shape */
    if (T_M % BYTES_PER_WORD == 0) {
        printf("test is pointless unless column tiles start mid-word\n");
        return 1;
    }
    pitch_words = row_pitch(M) / BYTES_PER_WORD;
    if (row_pitch(M) == M || !(pitch_words & (pitch_words - 1))) {
        printf("test is pointless unless the canvas pitch is padded and "
               "not a power of two\n");
        return 1;
    }

    accel_open();
    assert_fits_canvas(K, M);

    printf("test2: 2x2 tiles of %dx%d on a %d-wide canvas "
           "(pitch %d = %d words)\n", T_K, T_M, M, row_pitch(M), pitch_words);
    printf("  right tiles start at canvas column %d, mid-word; bottom tiles "
           "at row %d\n", T_M, T_K);

    srand(43);
    fill_int8(W, sizeof W, 127);
    fill_int8(X, sizeof X, 127);
    uint32_t q16 = pick_scale(W, X, K, N, M);

    golden_act(G, W, X, K, N, M, q16, NULL, NULL, NULL);
    printf("  scale 0x%04x, %.1f%% of the expected result saturated\n",
           q16, sat_fraction(G, (size_t)K * M));

    /* scribble first: a correct run must overwrite every byte it owns */
    {
        size_t len = (size_t)K * row_pitch(M);
        uint8_t *junk = malloc(len);
        memset(junk, 0x5A, len);
        write_bytes(OBUF_BASE, junk, len);
        free(junk);
    }

    for (int tr = 0; tr < 2; tr++) {
        for (int tc = 0; tc < 2; tc++) {
            for (int i = 0; i < T_K; i++) {
                memcpy(&Wt[(size_t)i * N], &W[(size_t)(tr * T_K + i) * N],
                       (size_t)N);
            }
            for (int p = 0; p < N; p++) {
                memcpy(&Xt[(size_t)p * T_M], &X[(size_t)p * M + tc * T_M],
                       (size_t)T_M);
            }
            /* tile_rows / tile_cols = 0: the grid is this matmul's own
             * 16 x 37, so tile (tr,tc) lands at row 16*tr, column 37*tc */
            if (run_tile(Wt, Xt, T_K, N, T_M, q16, tr, tc, M, 0, 0, 0)) {
                printf("  FAIL   timed out on tile (%d,%d)\n", tr, tc);
                return 1;
            }
        }
    }

    read_canvas(Y, K, M, M);
    fail |= check_canvas(Y, G, K, M, "2x2 tiles of 16x37 on a 74-wide canvas");

    alarm(0);
    printf("\n%s\n", fail ? "TEST2 FAILED" : "test2 passed");
    return fail;
}
