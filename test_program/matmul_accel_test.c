/*
 * matmul_min.c - smallest possible working matmul test for matmul_accel_0.
 *
 * Built only from operations status_probe.c already proved on hardware:
 *   - mmap /dev/mem at 0xC0000000, span 0x40000
 *   - 32-bit write at AVS(a)
 *   - 32-bit read  at AVS(a)
 * Nothing else. No benchmark loop, no random data, no bridge probe.
 *
 * Fixed 16x16 @ 16x16 identity test: W = 2I with OUTPUT_SCALE = 32768 (0.5)
 * gives Y = X exactly for every int8 value.  X carries 256 distinct values,
 * so any packing or banking error is legible from the mismatch addresses.
 *
 * Addressing: avs_address is a byte address stepping by 4 (one 32-bit word),
 * and the interconnect drives avs_address = cpu_offset >> 2, so
 * cpu_offset = 4*a.  Every `a` MUST be a multiple of 4 - a misaligned
 * avs_address hangs the bus and takes the HPS down.
 *
 * stdout is unbuffered and every phase announces itself, so the last line
 * printed is the phase that hung.  -s prompts between phases.
 *
 * gcc -O2 -Wall -o matmul_min matmul_min.c && sudo ./matmul_min
 */

#define _POSIX_C_SOURCE 200809L

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define ACCEL_PHYS 0xC0000000UL
#define ACCEL_SPAN 0x00010000UL   /* was 0x40000: 2^16 bytes, not 2^16 words */

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

#define ROWS            16
#define COLS            16
#define IBUF_BANK_DEPTH 256
#define OBUF_BANK_DEPTH 128

#define K 16
#define N 16
#define M 16

#define AVS(a) ((uint32_t)(a))

static volatile uint8_t *accel;
static int step_mode = 0;

/* ---- the three verified primitives ------------------------------- */

static void wr32(uint32_t a, uint32_t v)
{
    if (a & 3u) {
        fprintf(stderr, "\nBUG: misaligned write at avs 0x%04x - refusing "
                        "(would hang the bridge)\n", a);
        exit(3);
    }
    *(volatile uint32_t *)(accel + AVS(a)) = v;
}

static uint32_t rd32(uint32_t a)
{
    if (a & 3u) {
        fprintf(stderr, "\nBUG: misaligned read at avs 0x%04x - refusing "
                        "(would hang the bridge)\n", a);
        exit(3);
    }
    return *(volatile uint32_t *)(accel + AVS(a));
}

/* ---- helpers ------------------------------------------------------ */

static void phase(const char *what)
{
    printf("\n--- %s\n", what);
    if (step_mode) {
        printf("    press enter: ");
        int c;
        while ((c = getchar()) != '\n' && c != EOF) { }
        if (c == EOF) { printf("\n"); exit(0); }
    }
}

static uint64_t ns_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* pack 4 bytes little-endian into one word */
static uint32_t pack(const int8_t *b)
{
    return ((uint32_t)(uint8_t)b[0])
         | ((uint32_t)(uint8_t)b[1] << 8)
         | ((uint32_t)(uint8_t)b[2] << 16)
         | ((uint32_t)(uint8_t)b[3] << 24);
}

int main(int argc, char **argv)
{
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "-s")) step_mode = 1;

    /* W = 2I, X = 256 distinct values, expect Y == X */
    static int8_t W[K * N], X[N * M], Y[K * M];
    memset(W, 0, sizeof W);
    for (int i = 0; i < K; i++) W[i * N + i] = 2;
    for (int n = 0; n < N; n++)
        for (int m = 0; m < M; m++) X[n * M + m] = (int8_t)(n * 16 + m - 128);

    printf("matmul_min: %dx%d @ %dx%d, W=2I, scale=0.5, expect Y==X\n",
           K, N, N, M);

    phase("map /dev/mem");
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/mem (run as root)"); return 1; }
    void *map = mmap(NULL, ACCEL_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                     (off_t)ACCEL_PHYS);
    if (map == MAP_FAILED) { perror("mmap"); return 1; }
    accel = (volatile uint8_t *)map;
    printf("    mapped at %p\n", map);

    phase("read STATUS");
    printf("    STATUS = 0x%08x\n", rd32(CTRL_BASE + REG_STATUS));

    phase("program dimension registers");
    wr32(CTRL_BASE + REG_WEIGHT_ROWS, K);
    wr32(CTRL_BASE + REG_WEIGHT_COLS, N);
    wr32(CTRL_BASE + REG_INPUT_MAX_ADDR, M);
    wr32(CTRL_BASE + REG_INPUT_COLS, M);
    wr32(CTRL_BASE + REG_OUTPUT_SCALE, 32768);
    printf("    5 registers written\n");

    uint64_t t0 = ns_now();

    /* weights: flat row-major, K*N = 256 bytes = 64 words at 0x1000 */
    phase("write weight buffer (64 words)");
    for (int i = 0; i < K * N; i += 4)
        wr32(WBUF_BASE + (uint32_t)i, pack(&W[i]));
    printf("    done\n");

    /* inputs: bank p holds X[p][0..M-1]; one band only, since N == COLS.
     * 16 bytes = 4 words per bank, bank stride 256. */
    phase("write input buffer (16 banks x 4 words)");
    for (int p = 0; p < N; p++) {
        uint32_t base = IBUF_BASE + (uint32_t)p * IBUF_BANK_DEPTH;
        for (int m = 0; m < M; m += 4)
            wr32(base + (uint32_t)m, pack(&X[p * M + m]));
    }
    printf("    done\n");

    uint64_t t1 = ns_now();

    phase("pulse CONTROL.matmul_start");
    wr32(CTRL_BASE + REG_CONTROL, 1);
    printf("    started\n");

    phase("poll STATUS.done");
    uint32_t st = 0;
    int      polls = 0;
    uint64_t tp = ns_now();
    while (polls < 1000000) {
        st = rd32(CTRL_BASE + REG_STATUS);
        polls++;
        if (st & ST_DONE) break;
        if (ns_now() - tp > 1000000000ull) break; /* 1 s */
    }
    uint64_t t2 = ns_now();
    if (!(st & ST_DONE)) {
        printf("    TIMEOUT after %d polls, last STATUS = 0x%08x\n", polls, st);
        printf("    (not a bus hang - reads returned; the matmul never "
               "reported done)\n");
        return 1;
    }
    printf("    done after %d polls, STATUS = 0x%08x\n", polls, st);

    /* outputs: bank k holds Y[k][0..M-1], 4 words per bank, stride 128 */
    phase("read output buffer (16 banks x 4 words)");
    for (int k = 0; k < K; k++) {
        uint32_t base = OBUF_BASE + (uint32_t)k * OBUF_BANK_DEPTH;
        for (int m = 0; m < M; m += 4) {
            uint32_t w = rd32(base + (uint32_t)m);
            for (int j = 0; j < 4; j++)
                Y[k * M + m + j] = (int8_t)(w >> (8 * j));
        }
    }
    uint64_t t3 = ns_now();
    printf("    done\n");

    phase("verify");
    int bad = 0;
    for (int k = 0; k < K; k++) {
        for (int m = 0; m < M; m++) {
            int g = Y[k * M + m], e = X[k * M + m];
            if (g != e) {
                if (bad < 10)
                    printf("    Y[%2d][%2d]: got %4d want %4d  (avs 0x%04x)\n",
                           k, m, g, e,
                           OBUF_BASE + k * OBUF_BANK_DEPTH + m);
                bad++;
            }
        }
    }
    if (bad) printf("    FAIL: %d/%d mismatches\n", bad, K * M);
    else     printf("    PASS: all %d elements match\n", K * M);

    printf("\nTiming:\n");
    printf("  buffer writes  %8.2f us\n", (double)(t1 - t0) / 1e3);
    printf("  start -> done  %8.2f us\n", (double)(t2 - t1) / 1e3);
    printf("  readback       %8.2f us\n", (double)(t3 - t2) / 1e3);
    printf("  total          %8.2f us\n", (double)(t3 - t0) / 1e3);

    munmap(map, ACCEL_SPAN);
    close(fd);
    return bad ? 1 : 0;
}