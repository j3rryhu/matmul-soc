/* HPS identity matmul + bias/GELU readout test. No DMA or NEON required.
 * gcc -std=gnu99 -O2 -Wall -o matmul_gelu_test matmul_gelu_test.c -lrt
 * ./matmul_gelu_test gelu_int8.hex
 * ./matmul_gelu_test --self-test gelu_int8.hex  (no hardware access)
 * Use the EXACT HEX compiled into the FPGA, not a newly calibrated table.
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

#define K 32
#define N 32
#define M 32
#define STATUS 0x50u
#define ENABLE 0x8100u
#define BIAS 0x8000u
#define WBUF 0x1000u
#define IBUF 0x5000u
#define OBUF 0x7000u
static volatile uint8_t *accel;
static int8_t lut[256];

static int8_t saturate(int value)
{
    return (int8_t)(value > 127 ? 127 : (value < -128 ? -128 : value));
}
static int8_t reference(int8_t value, int8_t bias)
{
    return lut[(uint8_t)saturate((int)value + (int)bias)];
}
static uint32_t pack(const int8_t *p)
{
    return (uint32_t)(uint8_t)p[0] | ((uint32_t)(uint8_t)p[1] << 8)
        | ((uint32_t)(uint8_t)p[2] << 16) | ((uint32_t)(uint8_t)p[3] << 24);
}
static void wr(uint32_t address, uint32_t value)
{
    *(volatile uint32_t *)(accel + address) = value;
    __sync_synchronize();
}
static uint32_t rd(uint32_t address)
{
    uint32_t value = *(volatile uint32_t *)(accel + address);
    __sync_synchronize();
    return value;
}
static uint64_t now_ns(void)
{
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) { perror("clock_gettime"); exit(1); }
    return (uint64_t)t.tv_sec * 1000000000ull + t.tv_nsec;
}
static void read_output(int8_t *values)
{
    for (int k = 0; k < K; k++) {
        uint32_t base = OBUF + (k % 16)*128u + (k / 16)*M;
        for (int m = 0; m < M; m += 4) {
            uint32_t word = rd(base + m);
            for (int j = 0; j < 4; j++) values[k*M+m+j] = (int8_t)(word >> (8*j));
        }
    }
}
static int compare(const char *name, const int8_t *got, const int8_t *expected)
{
    int bad = 0;
    for (int i = 0; i < K*M; i++) if (got[i] != expected[i]) {
        if (bad < 10) printf("  [%d][%d] got %d want %d\n", i/M, i%M, got[i], expected[i]);
        bad++;
    }
    printf("%s: %s (%d/%d mismatches)\n", name, bad ? "FAIL" : "PASS", bad, K*M);
    return bad;
}
static int load_lut(const char *path)
{
    FILE *file = fopen(path, "r");
    if (!file) { perror(path); return -1; }
    unsigned value;
    for (int i = 0; i < 256; i++) {
        if (fscanf(file, "%x", &value) != 1 || value > 255) {
            fprintf(stderr, "Invalid LUT entry %d (expected 256 hexadecimal bytes)\n", i);
            fclose(file); return -1;
        }
        lut[i] = (int8_t)(value < 128 ? (int)value : (int)value - 256);
    }
    char extra[2];
    int trailing = fscanf(file, "%1s", extra);
    fclose(file);
    if (trailing == 1) { fprintf(stderr, "Extra data after LUT entry 255\n"); return -1; }
    return 0;
}
int main(int argc, char **argv)
{
    int offline = argc == 3 && !strcmp(argv[1], "--self-test");
    if (!(argc == 2 || offline)) {
        fprintf(stderr, "usage: %s [--self-test] FPGA_LUT.hex\n", argv[0]); return 1;
    }
    if (load_lut(argv[offline ? 2 : 1])) return 1;
    if (offline) {
        if (saturate(140) != 127 || saturate(-140) != -128 || saturate(30) != 30)
            return 1;
        for (int a = -128; a <= 127; a++) for (int b = -128; b <= 127; b++) {
            int s = a+b, c = s > 127 ? 127 : (s < -128 ? -128 : s);
            if (reference((int8_t)a, (int8_t)b) != lut[(uint8_t)c]) return 1;
        }
        puts("PASS: LUT parsing and 65536 bias reference combinations (no FPGA access)");
        return 0;
    }
    setvbuf(stdout, NULL, _IONBF, 0);
    static int8_t weights[K*N], inputs[N*M], bias[K], raw[K*M], got[K*M], expected[K*M];
    for (int k = 0; k < K; k++) {
        weights[k*N+k] = 2;
        bias[k] = (int8_t)(k-16); /* different bias per row, including both segments */
    }
    for (int i = 0; i < N*M; i++) inputs[i] = (int8_t)((i % 256)-128);
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("/dev/mem (run as root)"); return 1; }
    void *mapping = mmap(NULL, 0x10000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, (off_t)0xC0000000UL);
    if (mapping == MAP_FAILED) { perror("mmap"); close(fd); return 1; }
    accel = (volatile uint8_t *)mapping;
    int failed = 1;
    uint32_t status = rd(STATUS);
    printf("Initial STATUS = 0x%08x\n", status);
    if (status & 1u) { puts("Hardware busy: reset FPGA and retry; no configuration written."); goto cleanup; }
    wr(ENABLE, 0);
    wr(0x10, K); wr(0x20, N); wr(0x30, M); wr(0x40, M); wr(0x60, 32768);
    if (rd(0x10) != K || rd(0x20) != N || rd(0x40) != M || rd(0x60) != 32768) {
        puts("FAIL: dimension/scale register readback; check programmed FPGA version."); goto cleanup;
    }
    puts("Writing identity weights and banked inputs (32-bit PIO)...");
    for (int i = 0; i < K*N; i += 4) wr(WBUF+i, pack(&weights[i]));
    for (int bank = 0; bank < 16; bank++) for (int band = 0; band < 2; band++)
        for (int m = 0; m < M; m += 4)
            wr(IBUF+bank*256u+band*M+m, pack(&inputs[(band*16+bank)*M+m]));
    (void)rd(STATUS); /* clear stale done */
    wr(0, 1);
    uint64_t start = now_ns();
    do {
        status = rd(STATUS);
        if (status & 2u) break;
    } while (now_ns()-start < 1000000000ull);
    if (!(status & 2u) || (status & 1u)) {
        printf("FAIL: completion timeout/busy, STATUS=0x%08x\n", status); goto cleanup;
    }
    read_output(raw);
    if (compare("Raw identity matmul", raw, inputs)) goto cleanup;
    puts("Loading row biases and enabling GELU readout...");
    for (int k = 0; k < K; k += 4) wr(BIAS+k, pack(&bias[k]));
    wr(ENABLE, 1);
    for (int k = 0; k < K; k++) for (int m = 0; m < M; m++)
        expected[k*M+m] = reference(inputs[k*M+m], bias[k]);
    read_output(got);
    int bad = compare("Bias + GELU", got, expected);
    wr(ENABLE, 0);
    read_output(got);
    bad += compare("Bypass preserves raw output", got, raw);
    failed = bad != 0;
    if (!failed) puts("PASS: HPS matmul + bias + GELU readout integration");
cleanup:
    /* Only write disable when idle: a failed, still-busy operation needs reset. */
    if (!(rd(STATUS) & 1u)) wr(ENABLE, 0);
    munmap(mapping, 0x10000); close(fd);
    return failed;
}
