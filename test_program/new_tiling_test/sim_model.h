/*
 * sim_model.h - software model of accel_top, for TB_SIM builds.
 *
 * Included at the bottom of accel_tb.h so it shares the same `accel`
 * pointer. It models the device as the host sees it: contiguous row-major
 * images at WBUF_BASE / IBUF_BASE with rows padded to row_pitch, an int8
 * canvas at OBUF_BASE with rows padded to row_pitch(CANVAS_WIDTH), and an
 * internal psum buffer that is not host-visible.
 *
 * On a matmul_start pulse it reproduces the RDL's semantics:
 *
 *   r0 = tile_r * (TILE_ROWS ? TILE_ROWS : WEIGHT_ROWS)
 *   c0 = tile_c * (TILE_COLS ? TILE_COLS : INPUT_COLS)
 *   cw = CANVAS_WIDTH ? CANVAS_WIDTH : INPUT_COLS
 *
 *   acc = W @ X                                  (raw 32-bit)
 *   if use_prev_out and not prev_int8:  acc += psum      (raw continuation)
 *   if out_raw:                         psum = acc; stop (no canvas write)
 *   else:
 *       if bias_en:                     acc += bias[(r0+k) % BIAS_ROWS]
 *       r = (acc * OUTPUT_SCALE) >> 16                 (not yet saturated)
 *       if use_prev_out and prev_int8:  r += canvas[r0+k][c0+m]
 *       y = sat8(r)                                    (the one saturation)
 *       if gelu_en:                     y = gelu_lut[y]
 *       canvas[r0+k][c0+m] = y
 *
 * WHAT THIS DOES AND DOES NOT CHECK. It validates the test's own
 * consistency: tile placement arithmetic, slicing, the golden model, the
 * activation ordering and that every canvas byte a result owns gets written.
 * It cannot validate the test against the real hardware, because it reads
 * the images back with the same pitch formula the test writes them with - a
 * shared misunderstanding of the layout would cancel out. Only a run on the
 * board settles that.
 */

#ifndef SIM_MODEL_H
#define SIM_MODEL_H

#define SIM_PSUM_MAX (128 * 128)

static int32_t sim_psum[SIM_PSUM_MAX];
static int sim_psum_valid;          /* psum holds raw sums from an out_raw */
static int sim_psum_tr, sim_psum_tc;

/*
 * The GELU table lives in the RTL, not in a host-writable buffer, so a test
 * that sets gelu_en has to hand the model the same 256 entries. The bias,
 * by contrast, the model decodes straight out of the device's bias buffer -
 * so it checks the test's bias indexing, though not the layout itself.
 */
static const int8_t *sim_gelu_lut;  /* 256 entries, indexed by (uint8_t)y */

static int32_t sim_bias_entry(int row)
{
    volatile uint8_t *p = accel + BBUF_BASE + 4 * (row % BIAS_ROWS);

    return (int32_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8)
                   | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24));
}

static long sim_matmuls;            /* how many matmuls the test ran */

static void sim_open(void)
{
    void *m = calloc(1, ACCEL_SPAN);

    if (!m) { printf("sim: out of memory\n"); exit(1); }
    accel = (volatile uint8_t *)m;
    sim_psum_valid = 0;
    sim_matmuls = 0;
}

static void sim_matmul_start(void)
{
    int K  = (int)(reg_rd_raw(WEIGHT_ROWS_ADDR) & 0xFFFF);
    int N  = (int)(reg_rd_raw(WEIGHT_COLS_ADDR) & 0xFFFF);
    int M  = (int)(reg_rd_raw(INPUT_COLS_ADDR) & 0xFFFF);
    uint32_t q16 = reg_rd_raw(OUTPUT_SCALE_ADDR) & 0xFFFF;
    uint32_t tc_reg = reg_rd_raw(TILE_CFG_ADDR);
    uint32_t act = reg_rd_raw(ACT_CFG_ADDR);

    int tile_r = (int)((tc_reg >> TILE_R_SHIFT) & TILE_FIELD_MSK);
    int tile_c = (int)((tc_reg >> TILE_C_SHIFT) & TILE_FIELD_MSK);
    int use_prev = (tc_reg & USE_PREV_BIT) != 0;

    int cw = (int)(reg_rd_raw(CANVAS_WIDTH_ADDR) & 0xFFFF);
    int tr_grid = (int)(reg_rd_raw(TILE_ROWS_ADDR) & 0xFFFF);
    int tc_grid = (int)(reg_rd_raw(TILE_COLS_ADDR) & 0xFFFF);

    if (cw == 0) cw = M;
    if (tr_grid == 0) tr_grid = K;
    if (tc_grid == 0) tc_grid = M;

    int r0 = tile_r * tr_grid;
    int c0 = tile_c * tc_grid;
    int wpitch = row_pitch(N);
    int xpitch = row_pitch(M);
    int cpitch = row_pitch(cw);

    volatile uint8_t *Wimg = accel + WBUF_BASE;
    volatile uint8_t *Ximg = accel + IBUF_BASE;
    volatile uint8_t *canvas = accel + OBUF_BASE;

    if ((size_t)K * M > SIM_PSUM_MAX) {
        printf("sim: tile %dx%d exceeds the psum buffer (%d slots)\n",
               K, M, SIM_PSUM_MAX);
        exit(1);
    }
    if (r0 % ARRAY_ROWS) {
        printf("sim: tile first row %d is not a multiple of %d\n",
               r0, ARRAY_ROWS);
        exit(1);
    }
    if ((size_t)(r0 + K) * cpitch > OBUF_BYTES) {
        printf("sim: tile overruns the canvas (row %d..%d, pitch %d)\n",
               r0, r0 + K, cpitch);
        exit(1);
    }
    if (use_prev && !(act & ACT_PREV_INT8)) {
        if (!sim_psum_valid) {
            printf("sim: raw continuation with no out_raw matmul before it\n");
            exit(1);
        }
        if (sim_psum_tr != tile_r || sim_psum_tc != tile_c) {
            printf("sim: raw continuation on tile (%d,%d) but the psum "
                   "buffer holds tile (%d,%d)\n",
                   tile_r, tile_c, sim_psum_tr, sim_psum_tc);
            exit(1);
        }
    }

    for (int k = 0; k < K; k++) {
        for (int m = 0; m < M; m++) {
            int64_t acc = 0;

            for (int p = 0; p < N; p++) {
                int8_t w = (int8_t)Wimg[(size_t)k * wpitch + p];
                int8_t x = (int8_t)Ximg[(size_t)p * xpitch + m];
                acc += (int32_t)w * (int32_t)x;
            }
            if (use_prev && !(act & ACT_PREV_INT8)) {
                acc += sim_psum[(size_t)k * M + m];
            }

            if (act & ACT_OUT_RAW) {
                sim_psum[(size_t)k * M + m] = (int32_t)acc;
                continue;
            }

            if (act & ACT_BIAS_EN) {
                acc += sim_bias_entry(r0 + k);
            }

            /* one saturation, after the prev_int8 add */
            int32_t r = rescale(acc, q16);
            int8_t y;

            if (use_prev && (act & ACT_PREV_INT8)) {
                int8_t prev = (int8_t)canvas[(size_t)(r0 + k) * cpitch + c0 + m];
                y = sat8(r + prev);
            } else {
                y = sat8(r);
            }
            if (act & ACT_GELU_EN) {
                if (!sim_gelu_lut) {
                    printf("sim: gelu_en set but the test did not install a "
                           "GELU table\n");
                    exit(1);
                }
                y = sim_gelu_lut[(uint8_t)y];
            }
            canvas[(size_t)(r0 + k) * cpitch + c0 + m] = (uint8_t)y;
        }
    }

    if (act & ACT_OUT_RAW) {
        sim_psum_valid = 1;
        sim_psum_tr = tile_r;
        sim_psum_tc = tile_c;
    } else {
        sim_psum_valid = 0;
    }

    sim_matmuls++;
    *(volatile uint32_t *)(accel + STATUS_ADDR) = 1u << STATUS_DONE_BIT;
}

#endif /* SIM_MODEL_H */