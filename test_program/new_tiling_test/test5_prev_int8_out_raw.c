/*
 * test5_prev_int8_out_raw.c
 *
 * ACT_CFG.out_raw and ACT_CFG.prev_int8 on one tile: K = 32, M = 21.
 *
 * Run 1 (out_raw, bias_en also set; N = 16): leaves raw 32-bit sums in the
 * psum buffer - no bias, rescale or int8 result. They are not host-readable,
 * so run 2 checks them.
 * Run 2 (use_prev_out, plain; N = 16): adds onto run 1's raw sums and
 * produces the int8 result of the whole 32-long contraction - which only
 * matches if run 1 really left the UNBIASED raw sums.
 * Run 3 (use_prev_out + prev_int8, bias + GELU; N = 32, two contraction
 * blocks): starts its raw sums fresh, and on its last block reads run 2's
 * int8 result back from the output buffer and adds it to its own rescaled
 * (biased) result with one saturation, applies GELU, and writes the result
 * over it. The previous int8 lives in a different buffer from the raw sums,
 * so the multi-block contraction can't overwrite it first.
 *
 * Small values (|w|, |x| <= 7, OUTPUT_SCALE = 1/4) as in test4.
 *
 * gcc -O2 -Wall -Wextra -mcpu=cortex-a9 -mfpu=neon -o test5 test5_prev_int8_out_raw.c -lm
 */

#include "accel_tb.h"

#define K  (2 * ARRAY_ROWS)      /* 32 */
#define N  ARRAY_COLS            /* 16, one contraction block */
#define M  21

#define Q16 (1u << 14)           /* 0.25 */

static int8_t  W1[K * N], W2[K * N], X1[N * M], X2[N * M];
static int8_t  W3[K * 2 * N], X3[2 * N * M];
static int8_t  W12[K * 2 * N], X12[2 * N * M];
static int32_t bias[BIAS_ROWS];
static int8_t  Y[K * M], y2[K * M], G3[K * M], Gnp[K * M];

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

int main(void)
{
    int fail = 0;

    accel_open();
#ifdef TB_SIM
    sim_gelu_lut = gelu_lut;
#endif
    assert_fits_canvas(K, M);

    printf("test5: out_raw / use_prev_out / prev_int8 on one %dx%d tile, "
           "scale 0x%04x\n", K, M, Q16);

    srand(53);
    fill_int8(W1, sizeof W1, 7);
    fill_int8(W2, sizeof W2, 7);
    fill_int8(X1, sizeof X1, 7);
    fill_int8(X2, sizeof X2, 7);
    fill_int8(W3, sizeof W3, 7);
    fill_int8(X3, sizeof X3, 7);
    fill_int32(bias, BIAS_ROWS, 600);

    write_bias(bias);

    /* hstack([W1, W2]) and vstack([X1, X2]) - the 32-long contraction */
    for (int k = 0; k < K; k++) {
        memcpy(&W12[(size_t)k * 2 * N],     &W1[(size_t)k * N], (size_t)N);
        memcpy(&W12[(size_t)k * 2 * N + N], &W2[(size_t)k * N], (size_t)N);
    }
    memcpy(X12, X1, sizeof X1);
    memcpy(&X12[(size_t)N * M], X2, sizeof X2);

    /* ---- run 1: out_raw - raw 32-bit sums, bias_en must be ignored ---- */
    program_act(ACT_BIAS_EN | ACT_OUT_RAW);
    if (run_tile(W1, X1, K, N, M, Q16, 0, 0, M, 0, 0, 0)) {
        printf("  FAIL   run 1 timed out\n");
        return 1;
    }
    printf("  run 1: out_raw with bias_en set - no int8 result to check yet\n");

    /* ---- run 2: use_prev_out onto run 1's raw sums ---- */
    program_act(0);
    if (run_tile(W2, X2, K, N, M, Q16, 0, 0, M, 1, 0, 0)) {
        printf("  FAIL   run 2 timed out\n");
        return 1;
    }
    /* no bias: run 1's out_raw must not have applied it */
    golden_act(y2, W12, X12, K, 2 * N, M, Q16, NULL, NULL, NULL);

    read_canvas(Y, K, M, M);
    fail |= check_canvas(Y, y2, K, M,
                         "run 2: out_raw then use_prev_out, int8 of the "
                         "32-long contraction");

    /* had run 1 applied its bias, the answer would differ - confirm that */
    golden_act(Gnp, W12, X12, K, 2 * N, M, Q16, bias, NULL, NULL);
    if (memcmp(y2, Gnp, (size_t)K * M) == 0) {
        printf("  test setup: bias changes nothing here, so run 2 does not "
               "prove out_raw skipped it\n");
        return 1;
    }

    /* ---- run 3: use_prev_out + prev_int8, bias + GELU, two blocks ---- */
    golden_act(G3,  W3, X3, K, 2 * N, M, Q16, bias, gelu_lut, y2);
    golden_act(Gnp, W3, X3, K, 2 * N, M, Q16, bias, gelu_lut, NULL);
    if (memcmp(G3, Gnp, (size_t)K * M) == 0) {
        printf("  test setup: the previous result changes nothing\n");
        return 1;
    }

    program_act(ACT_BIAS_EN | ACT_GELU_EN | ACT_PREV_INT8);
    if (run_tile(W3, X3, K, 2 * N, M, Q16, 0, 0, M, 1, 0, 0)) {
        printf("  FAIL   run 3 timed out\n");
        return 1;
    }

    read_canvas(Y, K, M, M);
    fail |= check_canvas(Y, G3, K, M,
                         "run 3: use_prev_out + prev_int8 (+ bias, GELU, "
                         "2 contraction blocks) onto the int8 result");

    alarm(0);
    printf("\n%s\n", fail ? "TEST5 FAILED" : "test5 passed");
    return fail;
}
