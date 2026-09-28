/*
 * matmul_ladder.c - vary one dimension at a time from the known-good
 *                   16x16 @ 16x16 case, with data whose correct result is
 *                   exact and obvious.
 *
 * All PIO, no DMA - one variable at a time.
 *
 * ---------------------------------------------------------------------------
 * The trick
 * ---------------------------------------------------------------------------
 * W is a "selector": W[k][p] = 2 when p == k % N, else 0.
 * With OUTPUT_SCALE = 32768 (Q0.16 0.5):
 *
 *     psum for the block containing p = k%N  =  2 * X[k%N][m]
 *     every other block                      =  0
 *     requant(2*x, 0.5) = (2*x * 32768) >> 16 = x   exactly, no saturation
 *     accumulate: sat8(x + 0 + ...)           = x
 *
 * so  Y[k][m] == X[k % N][m]  exactly, for ANY K, N, M, any number of
 * contraction blocks and output segments. Every output element is a live
 * check - nothing clips, nothing rounds.
 *
 * Three X patterns, because which one fails says where to look:
 *
 *   ramp_p   X[p][m] = p - 16          constant along m.
 *            Y[k][m] should be (k%N)-16 for every m. If values are right
 *            but wrong m's, the fault is in column streaming / INPUT_COLS.
 *
 *   ramp_m   X[p][m] = (m % 64) - 32   constant along p.
 *            Y[k][m] should be the same for every k. If rows differ, the
 *            fault is in the contraction or the weight load.
 *
 *   index    X[p][m] = ((p*M + m) % 251) - 125
 *            every element nearly unique, so a mismatch value names the
 *            element it actually came from.
 *
 * gcc -O2 -Wall -o matmul_ladder matmul_ladder.c && sudo ./matmul_ladder
 *   -v   print the first bad row in full
 *   -k/-n/-m  run one shape instead of the ladder
 */

#define _POSIX_C_SOURCE 200809L

#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define ACCEL_PHYS 0xC0000000UL
#define ACCEL_SPAN 0x00010000UL

#define CTRL_BASE 0x0000u
#define WBUF_BASE 0x1000u
#define IBUF_BASE 0x5000u
#define OBUF_BASE 0x7000u

#define REG_CONTROL        0x00u
#define REG_WEIGHT_ROWS    0x04u
#define REG_WEIGHT_COLS    0x08u
#define REG_INPUT_MAX_ADDR 0x0Cu
#define REG_INPUT_COLS     0x10u
#define REG_STATUS         0x14u
#define REG_OUTPUT_SCALE   0x18u

#define ST_DONE 0x2u
#define AVS(a) ((uint32_t)(a))

#define ARRAY_ROWS      16
#define ARRAY_COLS      16
#define IBUF_BANK_DEPTH 256
#define OBUF_BANK_DEPTH 128

#define SCALE_HALF 32768u

#define MAXK 64
#define MAXN 64
#define MAXM 128

static volatile uint8_t *accel;
static int verbose = 0;

static void on_bus_error(int sig)
{
    (void)sig;
    const char *m = "\n*** SIGBUS\n";
    ssize_t r = write(2, m, strlen(m));
    (void)r;
    _exit(2);
}

static void wr32(uint32_t a, uint32_t v)
{
    if (a & 3u) { fprintf(stderr, "unaligned write 0x%04x\n", a); exit(3); }
    *(volatile uint32_t *)(accel + AVS(a)) = v;
}

static uint32_t rd32(uint32_t a)
{
    if (a & 3u) { fprintf(stderr, "unaligned read 0x%04x\n", a); exit(3); }
    return *(volatile uint32_t *)(accel + AVS(a));
}

static uint32_t pack(const int8_t *b)
{
    return ((uint32_t)(uint8_t)b[0]) | ((uint32_t)(uint8_t)b[1] << 8)
         | ((uint32_t)(uint8_t)b[2] << 16) | ((uint32_t)(uint8_t)b[3] << 24);
}

enum pattern { PAT_RAMP_P, PAT_RAMP_M, PAT_INDEX };
static const char *pat_name[] = { "ramp_p", "ramp_m", "index " };

static int8_t xval(enum pattern pat, int p, int m, int M)
{
    switch (pat) {
    case PAT_RAMP_P: return (int8_t)(p - 16);
    case PAT_RAMP_M: return (int8_t)((m % 64) - 32);
    default:         return (int8_t)(((p * M + m) % 251) - 125);
    }
}

/*
 * Returns the mismatch count. Y[k][m] must equal X[k % N][m].
 */
static int run_shape(int K, int N, int M, enum pattern pat)
{
    static int8_t W[MAXK * MAXN], X[MAXN * MAXM], Y[MAXK * MAXM];
    int num_bands = (N + ARRAY_COLS - 1) / ARRAY_COLS;

    /* capacity - the design cannot mask a partial contraction block */
    if (N % ARRAY_COLS) { printf("    N must be a multiple of %d\n", ARRAY_COLS); return -1; }
    if (num_bands * M > IBUF_BANK_DEPTH) { printf("    input bank overflow\n"); return -1; }
    if (((K + ARRAY_ROWS - 1) / ARRAY_ROWS) * M > OBUF_BANK_DEPTH) {
        printf("    output bank overflow\n"); return -1; }

    memset(W, 0, (size_t)K * N);
    for (int k = 0; k < K; k++) W[k * N + (k % N)] = 2;
    for (int p = 0; p < N; p++)
        for (int m = 0; m < M; m++) X[p * M + m] = xval(pat, p, m, M);

    wr32(CTRL_BASE + REG_WEIGHT_ROWS, (uint32_t)K);
    wr32(CTRL_BASE + REG_WEIGHT_COLS, (uint32_t)N);
    wr32(CTRL_BASE + REG_INPUT_MAX_ADDR, (uint32_t)M);
    wr32(CTRL_BASE + REG_INPUT_COLS, (uint32_t)M);
    wr32(CTRL_BASE + REG_OUTPUT_SCALE, SCALE_HALF);

    for (int i = 0; i < K * N; i += 4)
        wr32(WBUF_BASE + (uint32_t)i, pack(&W[i]));

    for (int bank = 0; bank < ARRAY_COLS && bank < N; bank++) {
        int8_t img[MAXN / ARRAY_COLS * MAXM + 4];
        size_t len = 0;
        for (int nb = 0; nb < num_bands; nb++) {
            int p = nb * ARRAY_COLS + bank;
            for (int m = 0; m < M; m++) img[len++] = (p < N) ? X[p * M + m] : 0;
        }
        while (len % 4) img[len++] = 0;
        uint32_t base = IBUF_BASE + (uint32_t)bank * IBUF_BANK_DEPTH;
        for (size_t i = 0; i < len; i += 4)
            wr32(base + (uint32_t)i, pack(&img[i]));
    }

    (void)rd32(CTRL_BASE + REG_STATUS);
    wr32(CTRL_BASE + REG_CONTROL, 1);

    uint32_t st = 0;
    for (int i = 0; i < 200000; i++) {
        st = rd32(CTRL_BASE + REG_STATUS);
        if (st & ST_DONE) break;
    }
    if (!(st & ST_DONE)) { printf("    TIMEOUT (STATUS=0x%08x)\n", st); return -1; }

    for (int k = 0; k < K; k++) {
        uint32_t base = OBUF_BASE
                      + (uint32_t)(k % ARRAY_ROWS) * OBUF_BANK_DEPTH
                      + (uint32_t)(k / ARRAY_ROWS) * M;
        for (int m = 0; m < M; m += 4) {
            uint32_t w = rd32(base + (uint32_t)m);
            for (int j = 0; j < 4 && m + j < M; j++)
                Y[k * M + m + j] = (int8_t)(w >> (8 * j));
        }
    }

    /* ---- verify: Y[k][m] == X[k % N][m] ---- */
    int bad = 0, first_bad_m = -1, first_bad_k = -1;
    int col_bad[MAXM]; memset(col_bad, 0, sizeof col_bad);
    int row_bad[MAXK]; memset(row_bad, 0, sizeof row_bad);

    for (int k = 0; k < K; k++) {
        for (int m = 0; m < M; m++) {
            int got = Y[k * M + m];
            int exp = X[(k % N) * M + m];
            if (got != exp) {
                if (bad == 0) { first_bad_k = k; first_bad_m = m; }
                col_bad[m]++;
                row_bad[k]++;
                bad++;
            }
        }
    }

    if (!bad) { printf("    PASS  (%d elements)\n", K * M); return 0; }

    printf("    FAIL  %d/%d wrong, first at Y[%d][%d]\n",
           bad, K * M, first_bad_k, first_bad_m);

    /* lowest column index that is wrong anywhere - if the first columns are
     * clean and it breaks at a fixed m, that names a streaming limit */
    int first_col = -1, clean_cols = 0;
    for (int m = 0; m < M; m++) {
        if (col_bad[m] && first_col < 0) first_col = m;
        if (!col_bad[m]) clean_cols++;
    }
    int full_rows = 0, clean_rows = 0;
    for (int k = 0; k < K; k++) {
        if (row_bad[k] == M) full_rows++;
        if (!row_bad[k]) clean_rows++;
    }
    printf("      columns: first bad m = %d, %d/%d columns fully clean\n",
           first_col, clean_cols, M);
    printf("      rows:    %d/%d rows fully clean, %d rows entirely wrong\n",
           clean_rows, K, full_rows);

    if (verbose) {
        int k = first_bad_k;
        printf("      row %d got: ", k);
        for (int m = 0; m < M; m++) printf("%4d", Y[k * M + m]);
        printf("\n      row %d exp: ", k);
        for (int m = 0; m < M; m++) printf("%4d", X[(k % N) * M + m]);
        printf("\n");
    }
    return bad;
}

int main(int argc, char **argv)
{
    int one_k = 0, one_n = 0, one_m = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-v")) verbose = 1;
        else if (!strcmp(argv[i], "-k") && i + 1 < argc) one_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) one_n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-m") && i + 1 < argc) one_m = atoi(argv[++i]);
        else { printf("usage: %s [-v] [-k K -n N -m M]\n", argv[0]); return 1; }
    }

    setvbuf(stdout, NULL, _IONBF, 0);
    signal(SIGBUS, on_bus_error);
    signal(SIGSEGV, on_bus_error);

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/mem (run as root)"); return 1; }
    void *map = mmap(NULL, ACCEL_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                     (off_t)ACCEL_PHYS);
    if (map == MAP_FAILED) { perror("mmap"); return 1; }
    accel = (volatile uint8_t *)map;

    printf("matmul_ladder: W = 2*selector, OUTPUT_SCALE = 32768 (0.5)\n");
    printf("  expected result is exactly Y[k][m] = X[k %% N][m] - no rounding,\n"
           "  no saturation, every element a live check.\n");

    struct { int K, N, M; const char *note; } shapes[] = {
        { 16, 16,  16, "known good baseline" },
        { 16, 16,  20, "M just past ARRAY_ROWS" },
        { 16, 16,  32, "M doubled, still ONE contraction block" },
        { 16, 16,  64, "M quadrupled" },
        { 32, 16,  16, "K doubled: 2 output segments, one block" },
        { 16, 32,  16, "N doubled: 2 contraction blocks, M small" },
        { 32, 32,  32, "the failing case" },
    };

    int failures = 0;

    if (one_k && one_n && one_m) {
        for (int p = 0; p < 3; p++) {
            printf("\n%dx%d @ %dx%d  pattern %s\n", one_k, one_n, one_n, one_m,
                   pat_name[p]);
            if (run_shape(one_k, one_n, one_m, (enum pattern)p)) failures++;
        }
    } else {
        for (size_t i = 0; i < sizeof shapes / sizeof shapes[0]; i++) {
            printf("\n=== K=%d N=%d M=%d   (%s) ===\n",
                   shapes[i].K, shapes[i].N, shapes[i].M, shapes[i].note);
            for (int p = 0; p < 3; p++) {
                printf("  %s:\n", pat_name[p]);
                if (run_shape(shapes[i].K, shapes[i].N, shapes[i].M,
                              (enum pattern)p)) failures++;
            }
        }
    }

    printf("\n%s\n", failures ? "Some shapes failed - the first failing row of "
                                "the ladder localizes the fault."
                              : "All shapes passed.");
    munmap(map, ACCEL_SPAN);
    close(fd);
    return failures ? 1 : 0;
}