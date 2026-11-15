/*
 * dma_addr_probe.c - where does dma_1.read_master think the accelerator is?
 *
 * Establishes ground truth with PIO, then has dma_1 read 16 bytes from a
 * series of candidate addresses and reports which one returns it.
 *
 *   1. run an identity matmul over PIO (W = 2I, OUTPUT_SCALE = 32768), so
 *      the canvas holds known data
 *   2. read its first word by PIO                          <- truth
 *   3. dma_1-read one word from each candidate address and compare
 *
 * A hit at 0x9000 means the base is 0 and the problem is elsewhere. A hit
 * at some other address A means dma_1 sees the accelerator at A - 0x9000,
 * and DMA1_ACCEL_BASE should be that. No hit anywhere means dma_1 is not
 * reaching the output buffer at all.
 *
 * Probes run smallest-first and stop at the first timeout, because a stalled
 * read leaves the slave mid-transaction and everything after it is noise.
 *
 * gcc -O2 -Wall -Wextra -mcpu=cortex-a9 -mfpu=neon -o dma_addr_probe dma_addr_probe.c
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

#ifdef __ARM_NEON
#include <arm_neon.h>
#define HAVE_NEON 1
#else
#define HAVE_NEON 0
#endif

#define ACCEL_PHYS 0xC0000000UL
#define ACCEL_SPAN 0x00020000UL
#define WBUF_BASE  0x1000u
#define IBUF_BASE  0x5000u
#define OBUF_BASE  0x9000u

#define REG_CONTROL        0
#define REG_WEIGHT_ROWS    1
#define REG_WEIGHT_COLS    2
#define REG_INPUT_MAX_ADDR 3
#define REG_INPUT_COLS     4
#define REG_STATUS         5
#define REG_OUTPUT_SCALE   6
#define REG_TILE_CFG       7
#define REG_TILED_MAT_DIM  8
#define REG_STRIDE         16

#define ST_DONE 0x2u
#define BEAT    16

#define LW_BASE  0xFF200000UL
#define LW_SPAN  0x00200000UL
#define DMA1_OFF 0x20u

#define DMA_STATUS     0
#define DMA_READ_ADDR  1
#define DMA_WRITE_ADDR 2
#define DMA_LENGTH     3
#define DMA_CONTROL    6

#define DMA_ST_DONE   0x1u
#define DMA_ST_BUSY   0x2u
#define CTRL_GO       (1u << 3)
#define CTRL_LEEN     (1u << 7)
#define CTRL_QUADWORD (1u << 11)

#define SRC_PHYS 0x30000000UL
#define SRC_SPAN 0x00100000UL
#define STAGE_Y  0x10000u

#define DIM 16                    /* 16x16 @ 16x16, one block, one tile */

static volatile uint8_t  *accel;
static volatile uint32_t *dma1;
static volatile uint8_t  *src;

static void on_fatal(int sig)
{
    if (dma1) {
        dma1[DMA_CONTROL] = 0;
        dma1[DMA_STATUS]  = 0;
    }
    const char *m = (sig == SIGALRM) ? "\n*** watchdog: dma_1 stopped.\n"
                                     : "\n*** signal: dma_1 stopped.\n";
    ssize_t r = write(2, m, strlen(m));
    (void)r;
    _exit(2);
}

static uint64_t ns_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void reg_wr(int idx, uint32_t v)
{
    volatile uint32_t *p = (volatile uint32_t *)(accel + idx * REG_STRIDE);
    *p = v;
}

static uint32_t reg_rd(int idx)
{
    volatile uint32_t *p = (volatile uint32_t *)(accel + idx * REG_STRIDE);
    return *p;
}

static void beat_wr(volatile uint8_t *p, const int8_t *b)
{
#if HAVE_NEON
    uint8x16_t v = vld1q_u8((const uint8_t *)b);
    __asm__ volatile("vst1.8 {%e0-%f0}, [%1 :128]" : : "w"(v), "r"(p) : "memory");
#else
    for (int j = 0; j < BEAT; j++) p[j] = (uint8_t)b[j];
#endif
}

static void beat_rd(volatile uint8_t *p, int8_t *b)
{
#if HAVE_NEON
    uint8x16_t v;
    __asm__ volatile("vld1.8 {%e0-%f0}, [%1 :128]" : "=w"(v) : "r"(p) : "memory");
    vst1q_u8((uint8_t *)b, v);
#else
    for (int j = 0; j < BEAT; j++) b[j] = (int8_t)p[j];
#endif
}

/* 0 on completion, -1 on timeout */
static int dma1_read(uint32_t rd_addr, uint32_t bytes)
{
    dma1[DMA_STATUS]     = 0;
    dma1[DMA_READ_ADDR]  = rd_addr;
    dma1[DMA_WRITE_ADDR] = (uint32_t)(SRC_PHYS + STAGE_Y);
    dma1[DMA_LENGTH]     = bytes;
    __sync_synchronize();
    dma1[DMA_CONTROL] = CTRL_QUADWORD | CTRL_LEEN | CTRL_GO;

    uint64_t t0 = ns_now();
    for (;;) {
        uint32_t st = dma1[DMA_STATUS];
        if (st & DMA_ST_DONE) break;
        if (ns_now() - t0 > 200000000ull) {     /* 200 ms */
            dma1[DMA_CONTROL] = 0;
            for (int i = 0; i < 10000; i++) {
                if (!(dma1[DMA_STATUS] & DMA_ST_BUSY)) break;
            }
            dma1[DMA_LENGTH] = 0;
            dma1[DMA_STATUS] = 0;
            return -1;
        }
    }
    dma1[DMA_CONTROL] = 0;
    dma1[DMA_STATUS]  = 0;
    return 0;
}

static int8_t wimg[DIM * DIM];
static int8_t ximg[DIM * DIM];
static int8_t truth[DIM * DIM];
static int8_t got[BEAT];

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    signal(SIGBUS, on_fatal);
    signal(SIGSEGV, on_fatal);
    signal(SIGINT, on_fatal);
    signal(SIGALRM, on_fatal);
    alarm(30);

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/mem (run as root)"); return 1; }
    void *amap = mmap(NULL, ACCEL_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)ACCEL_PHYS);
    void *lmap = mmap(NULL, LW_SPAN,    PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)LW_BASE);
    void *smap = mmap(NULL, SRC_SPAN,   PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)SRC_PHYS);
    if (amap == MAP_FAILED || lmap == MAP_FAILED || smap == MAP_FAILED) {
        perror("mmap"); return 1;
    }
    accel = (volatile uint8_t *)amap;
    dma1  = (volatile uint32_t *)((volatile uint8_t *)lmap + DMA1_OFF);
    src   = (volatile uint8_t *)smap;

    dma1[DMA_CONTROL] = 0;
    dma1[DMA_STATUS]  = 0;

    printf("dma_addr_probe: STATUS = 0x%08x\n", reg_rd(REG_STATUS));

    /* ---- 1. identity matmul over PIO, so the canvas is known ---- */
    memset(wimg, 0, sizeof wimg);
    for (int i = 0; i < DIM; i++) wimg[i * DIM + i] = 2;
    for (int p = 0; p < DIM; p++) {
        for (int m = 0; m < DIM; m++) {
            ximg[p * DIM + m] = (int8_t)(((p * DIM + m) % 251) - 125);
        }
    }

    reg_wr(REG_WEIGHT_ROWS, DIM);
    reg_wr(REG_WEIGHT_COLS, DIM);
    reg_wr(REG_INPUT_MAX_ADDR, DIM);
    reg_wr(REG_INPUT_COLS, DIM);
    reg_wr(REG_OUTPUT_SCALE, 32768u);
    reg_wr(REG_TILE_CFG, 0);
    reg_wr(REG_TILED_MAT_DIM, DIM);

    for (int i = 0; i < DIM * DIM; i += BEAT) beat_wr(accel + WBUF_BASE + i, &wimg[i]);
    for (int i = 0; i < DIM * DIM; i += BEAT) beat_wr(accel + IBUF_BASE + i, &ximg[i]);

    (void)reg_rd(REG_STATUS);
    reg_wr(REG_CONTROL, 1);

    uint32_t st = 0;
    uint64_t t0 = ns_now();
    for (;;) {
        st = reg_rd(REG_STATUS);
        if (st & ST_DONE) break;
        if (ns_now() - t0 > 1000000000ull) {
            printf("  matmul TIMEOUT, STATUS 0x%08x - reconfigure and retry\n", st);
            return 1;
        }
    }

    /* ---- 2. ground truth, by PIO ---- */
    for (int i = 0; i < DIM * DIM; i += BEAT) beat_rd(accel + OBUF_BASE + i, &truth[i]);
    printf("  canvas first word by PIO:");
    for (int i = 0; i < BEAT; i++) printf(" %4d", truth[i]);
    printf("\n");

    int identity_ok = (memcmp(truth, ximg, BEAT) == 0);
    printf("  matches X (identity held): %s\n", identity_ok ? "yes" : "NO");
    if (!identity_ok) {
        printf("  the PIO path itself is not returning X - fix that before"
               " chasing the DMA address\n");
        return 1;
    }

    /* ---- 3. single beat: is the ADDRESS right for dma_1? ----
     *
     * PIO already proved the remap is right for an address a master
     * presents. One beat is the only transaction shape where dma_1 presents
     * every address itself, so it isolates the address from the burst. */
    printf("\n  dma_1, ONE beat from 0x%04x: ", OBUF_BASE);
    if (dma1_read(OBUF_BASE, BEAT) != 0) {
        printf("TIMEOUT\n");
        return 1;
    }
    beat_rd(src + STAGE_Y, got);
    int single_ok = (memcmp(got, truth, BEAT) == 0);
    printf("%s\n", single_ok ? "matches PIO" : "DIFFERENT");
    if (!single_ok) {
        printf("    got  ");
        for (int i = 0; i < 8; i++) printf(" %4d", got[i]);
        printf(" ...\n    want ");
        for (int i = 0; i < 8; i++) printf(" %4d", truth[i]);
        printf(" ...\n");
    }

    if (!single_ok) {
        /* Only now is an address hunt worth doing. */
        static const uint32_t cand[] = {
            0x00000000u, 0x00001000u, 0x00005000u, 0x0000D000u,
            0x00011000u, 0x00019000u, 0x00029000u, 0x00049000u,
            0x00089000u, 0x00109000u, 0x00209000u, 0x00409000u,
        };
        int nc = (int)(sizeof cand / sizeof cand[0]);
        int hit = -1;
        printf("\n  one beat does not match, so sweeping for the base:\n");
        for (int i = 0; i < nc; i++) {
            printf("    0x%08x : ", cand[i]);
            if (dma1_read(cand[i], BEAT) != 0) {
                printf("TIMEOUT - stopping, the slave may now be stuck\n");
                break;
            }
            beat_rd(src + STAGE_Y, got);
            if (memcmp(got, truth, BEAT) == 0) {
                printf("MATCH\n");
                if (hit < 0) hit = i;
            } else {
                printf("%4d %4d %4d %4d ...\n", got[0], got[1], got[2], got[3]);
            }
        }
        printf("\n");
        if (hit < 0) {
            printf("  nothing returned the canvas. dma_1 is not reaching the\n"
                   "  output buffer - check read_master's connection and its\n"
                   "  address width.\n");
        } else {
            printf("  found at 0x%08x: set DMA1_ACCEL_BASE = 0x%08x\n",
                   cand[hit], cand[hit] - OBUF_BASE);
        }
        return 1;
    }

    /* ---- 4. the address is right. Does a BURST stay right? ----
     *
     * The slave generates every address after the first itself. If that
     * increment walks the banked layout instead of the canvas, beat 0 is
     * correct and the rest are not - invisible to PIO, which never bursts. */
    printf("\n  dma_1, bursts from the same address:\n");
    int nbeats = (DIM * DIM) / BEAT;     /* the whole 16x16 canvas */
    for (int count = 2; count <= nbeats; count *= 2) {
        if (dma1_read(OBUF_BASE, (uint32_t)(count * BEAT)) != 0) {
            printf("    %2d beats: TIMEOUT\n", count);
            break;
        }
        printf("    %2d beats:", count);
        for (int b = 0; b < count; b++) {
            beat_rd(src + STAGE_Y + b * BEAT, got);
            if (memcmp(got, &truth[0] + b * BEAT, BEAT) == 0) {
                printf(" %d:ok", b);
                continue;
            }
            /* where in the canvas does this beat's data actually live? */
            int from = -1;
            for (int k = 0; k < nbeats; k++) {
                if (memcmp(got, &truth[0] + k * BEAT, BEAT) != 0) continue;
                from = k;
                break;
            }
            if (from >= 0) printf(" %d:<-%d", b, from);
            else           printf(" %d:?", b);
        }
        printf("\n");
    }

    printf("\n  Reading: 'b:ok' is a correct beat, 'b:<-k' means beat b holds\n"
           "  the data that belongs in beat k, and a query mark is data not in the\n"
           "  canvas at all. If beat 0 is always ok and the rest are not, the\n"
           "  slave's burst address increment is bypassing the read remap.\n");

    alarm(0);
    dma1[DMA_CONTROL] = 0;
    dma1[DMA_STATUS]  = 0;
    dma1 = NULL;
    munmap(amap, ACCEL_SPAN);
    munmap(lmap, LW_SPAN);
    munmap(smap, SRC_SPAN);
    close(fd);
    return 0;
}