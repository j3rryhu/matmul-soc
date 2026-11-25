#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64
#include "softmax_mmio.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#endif
#ifdef SOFTMAX_BOARD
#include "../test_program/new_tiling_test/accel_tb.h"
#endif

static volatile unsigned checksum;

static void fail(const char *message)
{
    fprintf(stderr, "%s\n", message);
    exit(EXIT_FAILURE);
}

static uint64_t benchmark_ns(void)
{
#ifdef _WIN32
    LARGE_INTEGER counter, frequency;
    QueryPerformanceCounter(&counter);
    QueryPerformanceFrequency(&frequency);
    return (uint64_t)(1e9 * (double)counter.QuadPart / frequency.QuadPart);
#else
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return 1000000000ull * now.tv_sec + now.tv_nsec;
#endif
}

static void int8_softmax(const int8_t *scores, int8_t *probabilities, int length,
                         const uint16_t lut[256])
{
    int maximum = -128;
    for (int key = 0; key < length; ++key)
        if (scores[key] > maximum)
            maximum = scores[key];

    uint32_t total = 0;
    for (int key = 0; key < length; ++key)
        total += lut[maximum - scores[key]];

    for (int key = 0; key < length; ++key) {
        uint32_t numerator = 127u * lut[maximum - scores[key]];
        uint32_t code = numerator / total;
        uint32_t twice_remainder = 2 * (numerator % total);
        if (twice_remainder > total || (twice_remainder == total && (code & 1u)))
            ++code;
        probabilities[key] = (int8_t)code;
    }
}

static void fp32_softmax(const int8_t *scores, float *probabilities, int length)
{
    int maximum = -128;
    for (int key = 0; key < length; ++key)
        if (scores[key] > maximum)
            maximum = scores[key];

    float total = 0.0f;
    for (int key = 0; key < length; ++key) {
        probabilities[key] = expf((scores[key] - maximum) * (float)SOFTMAX_SCORE_SCALE);
        total += probabilities[key];
    }
    for (int key = 0; key < length; ++key)
        probabilities[key] /= total;
}

#ifdef SOFTMAX_BOARD
static void measure_board(const int8_t *scores, const int8_t *expected,
                          int rows, int length)
{
    int8_t weights[ARRAY_ROWS * ARRAY_ROWS] = {0};
    int8_t inputs[ARRAY_ROWS * SOFTMAX_MAX_LENGTH];
    int8_t canvas[ARRAY_ROWS * SOFTMAX_MAX_LENGTH];
    int8_t probabilities[ARRAY_ROWS * SOFTMAX_MAX_LENGTH];
    uint64_t elapsed = 0;
    size_t disagreements = 0;

    accel_open();
    program_act(0);
    // 2I @ X followed by a 1/2 multiplier stages the exact INT8 scores in OBUF.
    for (int i = 0; i < ARRAY_ROWS; ++i)
        weights[i * ARRAY_ROWS + i] = 2;

    for (int base = 0; base < rows; base += ARRAY_ROWS) {
        int count = rows - base < ARRAY_ROWS ? rows - base : ARRAY_ROWS;
        memset(inputs, 0, sizeof(inputs));
        memcpy(inputs, scores + base * length, (size_t)count * length);
        if (run_tile(weights, inputs, ARRAY_ROWS, ARRAY_ROWS, length,
                     32768, 0, 0, 0, 0, ARRAY_ROWS, length))
            fail("Matmul failed to stage the benchmark scores.");
        read_canvas(canvas, ARRAY_ROWS, length, length);
        if (memcmp(canvas, inputs, (size_t)ARRAY_ROWS * length))
            fail("Staged OBUF scores differ from the CPU inputs.");

        uint64_t start = benchmark_ns();
        for (int i = 0; i < count; ++i) {
#ifdef SOFTMAX_PACKED_BOARD
            int status = softmax_obuf_wait_packed(accel, (size_t)i * row_pitch(length),
                                                  length, 1000000);
            if (!status)
                read_bytes(SOFTMAX_PACKED_OFFSET, (uint8_t *)(probabilities + i * length), length);
#else
            int status = softmax_obuf_row_mmio(accel, (size_t)i * row_pitch(length),
                                               probabilities + i * length, length, 1000000);
#endif
            if (status)
                fail("Direct softmax command failed. Check the loaded bitstream.");
        }
        elapsed += benchmark_ns() - start;
        for (int sample = 0; sample < count * length; ++sample)
            disagreements += probabilities[sample] != expected[base * length + sample];
    }
#ifdef SOFTMAX_PACKED_BOARD
    printf("Fabric with packed HPS MMIO: %.3f us/row, %zu mismatches / %zu codes\n",
#else
    printf("Fabric with HPS MMIO: %.3f us/row, %zu mismatches / %zu codes\n",
#endif
           (double)elapsed / rows / 1000, disagreements, (size_t)rows * length);
    if (disagreements)
        exit(EXIT_FAILURE);
}
#endif

int main(int argc, char **argv)
{
    if (argc != 4)
        fail("Usage: benchmark_hps scores.txt exp_lut.hex cpu_vectors.txt");

    FILE *input = fopen(argv[1], "r");
    FILE *rom = fopen(argv[2], "r");
    int rows, length;
    if (!input || !rom || fscanf(input, "%d%d", &rows, &length) != 2 ||
        rows <= 0 || length <= 0 || length > (int)SOFTMAX_MAX_LENGTH ||
        rows > INT32_MAX / length)
        fail("Invalid scores file or missing ROM.");

    uint16_t lut[256];
    for (int i = 0; i < 256; ++i) {
        unsigned value;
        if (fscanf(rom, "%x", &value) != 1 || value > UINT16_MAX)
            fail("Invalid exponential ROM.");
        lut[i] = (uint16_t)value;
    }
    if (lut[0] != UINT16_MAX)
        fail("ROM must contain a nonzero maximum exponential.");
    fclose(rom);

    size_t samples = (size_t)rows * length;
    int8_t *scores = malloc(samples);
    int8_t *codes = malloc(samples);
    if (!scores || !codes)
        fail("Could not allocate benchmark vectors.");
    for (size_t i = 0; i < samples; ++i) {
        int score;
        if (fscanf(input, "%d", &score) != 1 || score < -128 || score > 127)
            fail("Invalid INT8 score.");
        scores[i] = (int8_t)score;
    }
    fclose(input);

    const int repetitions = 20;
    float probabilities[SOFTMAX_MAX_LENGTH];
    for (int row = 0; row < rows; ++row) {
        int8_softmax(scores + row * length, codes + row * length, length, lut);
        fp32_softmax(scores + row * length, probabilities, length);
        checksum += (unsigned)(128.0f * probabilities[row % length]);
    }

    uint64_t start = benchmark_ns();
    for (int rep = 0; rep < repetitions; ++rep)
        for (int row = 0; row < rows; ++row) {
            int offset = row * length;
            int8_softmax(scores + offset, codes + offset, length, lut);
            checksum += (unsigned)codes[offset + rep % length];
        }
    uint64_t int8_ns = benchmark_ns() - start;

    start = benchmark_ns();
    for (int rep = 0; rep < repetitions; ++rep)
        for (int row = 0; row < rows; ++row) {
            fp32_softmax(scores + row * length, probabilities, length);
            checksum += (unsigned)(128.0f * probabilities[rep % length]);
        }
    uint64_t fp32_ns = benchmark_ns() - start;
    double timed_rows = (double)rows * repetitions;
    printf("CPU INT8: %.3f us/row\n", int8_ns / timed_rows / 1000);
    printf("CPU FP32: %.3f us/row\n", fp32_ns / timed_rows / 1000);
    printf("Fixture: %d rows x %d keys, %d repetitions, checksum=%u\n",
           rows, length, repetitions, checksum);

    FILE *output = fopen(argv[3], "w");
    if (!output)
        fail("Could not write CPU vectors.");
    fprintf(output, "%d %zu\n", rows, samples);
    for (int row = 0; row < rows; ++row) {
        fprintf(output, "%d\n", length);
        for (int key = 0; key < length; ++key) {
            int offset = row * length + key;
            fprintf(output, "%d %d\n", scores[offset], codes[offset]);
        }
    }
    if (fclose(output))
        fail("Could not finish writing CPU vectors.");

#ifdef SOFTMAX_BOARD
    measure_board(scores, codes, rows, length);
#endif
    free(scores);
    free(codes);
    return EXIT_SUCCESS;
}
