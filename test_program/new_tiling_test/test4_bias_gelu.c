/*
 * test4_bias_gelu.c
 *
 * ACT_CFG bias + GELU, applied on each output element's last contraction
 * pass.
 *
 * Run 1: K = 36 (three output-row blocks, the last partial), N = 48 (three
 * contraction blocks), bias + GELU. weight_ctrl runs the output-row blocks
 * inside each contraction block, so every output-row block has its own last
 * pass (n_blk = 2) and must get the bias and GELU there exactly once, while
 * the earlier passes keep raw 32-bit sums for the next block to add to.
 *
 * Run 2: the first 32 rows of the same W as tile row 1 (canvas rows 32..63),
 * bias only: checks the non-GELU final write path, and that the bias is
 * indexed by *canvas* row (bias[32..63]), not the matmul's own output row.
 *
 * Values are kept small (|w|, |x| <= 7, OUTPUT_SCALE = 1/4) so the rescaled
 * values land in the GELU table's nonlinear range. The int32 biases
 * (accumulator units) span +-600, past int8, so their upper bytes and sign
 * matter; they move a result by up to 150 LSBs.
 *
 * gcc -O2 -Wall -Wextra -mcpu=cortex-a9 -mfpu=neon -o test4 test4_bias_gelu.c -lm
 */

#include "accel_tb.h"

#define K  (2 * ARRAY_ROWS + 4)      /* 36: three row blocks, last partial */
#define N  (3 * ARRAY_COLS)          /* 48: three contraction blocks */
#define M  20
#define K2 (2 * ARRAY_ROWS)          /* 32: run 2's tile */

#define Q16 (1u << 14)               /* 0.25 */

static int8_t  W[K * N], X[N * M];
static int32_t bias[BIAS_ROWS];
static int8_t  Y[2 * K2 * M], G[K * M], Gnb[K * M], Gng[K * M];

/* the GELU table the RTL holds, indexed by (uint8_t) of the rescaled int8 */
static const int8_t gelu_lut[256] = {
       0,    1,    1,    2,    2,    3,    3,    4,    5,    5,    6,    7,    8,    9,    9,   10,
      11,   12,   13,   14,   15,   16,   17,   18,   19,   20,   21,   22,   23,   24,   25,   26,
      27,   28,   29,   30,   31,   32,   34,   35,   36,   37,   38,   39,   40,   41,   43,   44,
      45,   46,   47,   48,   49,   50,   52,   53,   54,   55,   56,   57,   58,   59,   60,   61,
      63,   64,   65,   66,   67,   68,   69,   70,   71,   72,   73,   74,   75,   76,   77,   78,
      80,   81,   82,   83,   84,   85,   86,   87,   88,   89,   90,   91,   92,   93,   94,   95,
      96,   97,   98,   99,  100,  101,  102,  103,  104,  105,  106,  107,  108,  109,  110,  111,
     112,  113,  114,  115,  116,  117,  118,  119,  120,  121,  122,  123,  124,  125,  126,  127,
       0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
       0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
       0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
       0,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
      -1,   -2,   -2,   -2,   -2,   -2,   -2,   -2,   -2,   -2,   -2,   -3,   -3,   -3,   -3,   -3,
      -3,   -3,   -3,   -4,   -4,   -4,   -4,   -4,   -4,   -4,   -4,   -5,   -5,   -5,   -5,   -5,
      -5,   -5,   -5,   -5,   -5,   -5,   -5,   -5,   -5,   -5,   -5,   -5,   -5,   -5,   -5,   -5,
      -5,   -5,   -5,   -4,   -4,   -4,   -4,   -4,   -3,   -3,   -3,   -2,   -2,   -1,   -1,    0,
};

static int same(const int8_t *a, const int8_t *b, size_t n)
{
    return memcmp(a, b, n) == 0;
}

int main(void)
{
    int fail = 0;

    accel_open();
#ifdef TB_SIM
    sim_gelu_lut = gelu_lut;
#endif
    assert_fits_canvas(K, M);
    assert_fits_canvas(2 * K2, M);

    printf("test4: bias + GELU, %dx%d @ %dx%d, scale 0x%04x\n",
           K, N, N, M, Q16);
    printf("  %d output-row blocks (last partial: %d of %d rows), "
           "%d contraction blocks\n",
           CEILDIV(K, ARRAY_ROWS), K % ARRAY_ROWS, ARRAY_ROWS,
           CEILDIV(N, ARRAY_COLS));

    srand(47);
    fill_int8(W, sizeof W, 7);
    fill_int8(X, sizeof X, 7);
    fill_int32(bias, BIAS_ROWS, 600);

    write_bias(bias);

    /* ---- run 1: bias + GELU, tile (0, 0) ---- */
    golden_act(G,   W, X, K, N, M, Q16, bias, gelu_lut, NULL);
    golden_act(Gnb, W, X, K, N, M, Q16, NULL, gelu_lut, NULL);  /* no bias */
    golden_act(Gng, W, X, K, N, M, Q16, bias, NULL,     NULL);  /* no GELU */

    if (same(G, Gnb, (size_t)K * M)) {
        printf("  test setup: the bias changes nothing\n");
        return 1;
    }
    if (same(G, Gng, (size_t)K * M)) {
        printf("  test setup: GELU changes nothing\n");
        return 1;
    }

    program_dims(K, N, M, Q16);
    program_tile(0, 0, 0, M, 0, 0);
    program_act(ACT_BIAS_EN | ACT_GELU_EN);
    write_weights(W, K, N);
    write_inputs(X, N, M);
    (void)reg_rd(STATUS_ADDR);
    reg_wr(CONTROL_ADDR, 1u << MATMUL_START_BIT);
    if (wait_matmul_done()) {
        printf("  FAIL   run 1 timed out\n");
        return 1;
    }

    read_canvas(Y, K, M, M);
    fail |= check_canvas(Y, G, K, M,
                         "run 1: bias + GELU, 36x48 @ 48x20 "
                         "(3 output-row x 3 contraction blocks)");

    /* ---- run 2: bias only, tile row 1 (canvas rows 32..63) ---- */
    program_act(ACT_BIAS_EN);
    if (run_tile(W, X, K2, N, M, Q16, 1, 0, M, 0, 0, 0)) {
        printf("  FAIL   run 2 timed out\n");
        return 1;
    }

    /* bias is indexed by canvas row, so rows 32..63 take bias[32..63] */
    golden_act(G, W, X, K2, N, M, Q16, &bias[K2], NULL, NULL);

    read_canvas(Y, 2 * K2, M, M);
    fail |= check_canvas(&Y[(size_t)K2 * M], G, K2, M,
                         "run 2: bias only, tile row 1 (canvas rows 32..63)");

    /* the point of run 2: bias[0..31] would give a different answer */
    golden_act(Gng, W, X, K2, N, M, Q16, bias, NULL, NULL);
    if (same(G, Gng, (size_t)K2 * M)) {
        printf("  test setup: bias[0..31] and bias[32..63] give the same "
               "result, so run 2 proves nothing\n");
        return 1;
    }

    alarm(0);
    printf("\n%s\n", fail ? "TEST4 FAILED" : "test4 passed");
    return fail;
}
