/*
 * test3_tiled_edge_tiles.c
 *
 * A 40 x 37 result on a TILE_ROWS = 32 x TILE_COLS = 22 tile grid, so the
 * bottom and right tiles are smaller than the grid:
 *
 *     tile (0,0): 32 x 22    tile (0,1): 32 x 15
 *     tile (1,0):  8 x 22    tile (1,1):  8 x 15
 *
 * Every tile's position comes from the grid (first row tile_r*32, first
 * column tile_c*22), not from its own size: with the old tile_r*WEIGHT_ROWS
 * rule the 8-row bottom tiles would start at canvas row 8 instead of 32, and
 * the 15-wide right tiles at column 15 instead of 22. Also checks:
 *   - the bottom tiles are a single partial row block (8 of 16 rows);
 *   - the right tiles start mid-word (column 22 of the int8 canvas) and
 *     write 15 int8s next to the left tiles' 22 in the same bus words;
 *   - N = 32 is two contraction blocks, so the edge tiles accumulate too.
 *
 * gcc -O2 -Wall -Wextra -mcpu=cortex-a9 -mfpu=neon -o test3 test3_tiled_edge_tiles.c -lm
 */

#include "accel_tb.h"

#define TR_GRID 32               /* TILE_ROWS */
#define TC_GRID 22               /* TILE_COLS */
#define K 40
#define N (2 * ARRAY_COLS)       /* 32, two contraction blocks */
#define M 37

static int8_t W[K * N], X[N * M];
static int8_t Wt[TR_GRID * N], Xt[N * TC_GRID];
static int8_t Y[K * M], G[K * M];

int main(void)
{
    int fail = 0;

    accel_open();
    assert_fits_canvas(K, M);

    printf("test3: %dx%d result on a %dx%d tile grid (smaller edge tiles)\n",
           K, M, TR_GRID, TC_GRID);

    srand(59);
    fill_int8(W, sizeof W, 127);
    fill_int8(X, sizeof X, 127);
    uint32_t q16 = pick_scale(W, X, K, N, M);

    golden_act(G, W, X, K, N, M, q16, NULL, NULL, NULL);
    printf("  scale 0x%04x, %.1f%% of the expected result saturated\n",
           q16, sat_fraction(G, (size_t)K * M));

    {
        size_t len = (size_t)K * row_pitch(M);
        uint8_t *junk = malloc(len);
        memset(junk, 0x5A, len);
        write_bytes(OBUF_BASE, junk, len);
        free(junk);
    }

    for (int tr = 0; tr < 2; tr++) {
        int r0 = tr * TR_GRID;
        int kt = K - r0;                        /* 32 then 8 */
        if (kt > TR_GRID) kt = TR_GRID;

        for (int tc = 0; tc < 2; tc++) {
            int c0 = tc * TC_GRID;
            int mt = M - c0;                    /* 22 then 15 */
            if (mt > TC_GRID) mt = TC_GRID;

            for (int i = 0; i < kt; i++) {
                memcpy(&Wt[(size_t)i * N], &W[(size_t)(r0 + i) * N],
                       (size_t)N);
            }
            for (int p = 0; p < N; p++) {
                memcpy(&Xt[(size_t)p * mt], &X[(size_t)p * M + c0],
                       (size_t)mt);
            }

            printf("    tile (%d,%d): %2d x %2d at canvas row %2d, column %2d\n",
                   tr, tc, kt, mt, r0, c0);

            /* the grid, not this tile's own size, sets where it lands */
            if (run_tile(Wt, Xt, kt, N, mt, q16, tr, tc, M, 0,
                         TR_GRID, TC_GRID)) {
                printf("  FAIL   timed out on tile (%d,%d)\n", tr, tc);
                return 1;
            }
        }
    }

    read_canvas(Y, K, M, M);
    fail |= check_canvas(Y, G, K, M,
                         "40x37 result on a 32x22 tile grid (smaller edge tiles)");

    alarm(0);
    printf("\n%s\n", fail ? "TEST3 FAILED" : "test3 passed");
    return fail;
}
