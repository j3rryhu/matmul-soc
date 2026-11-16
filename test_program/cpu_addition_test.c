/*
 * bias_test.c - CPU bias-add time vs the DMA time to push the bias to fabric.
 *
 *   ./bias_test [K] [M] [threads]      default 512 784 2
 *
 * CPU side: add bias[k] to every element of int32 accumulator row k.
 *           NEON, 4x unrolled, split across pthreads. The buffer is ordinary
 *           cacheable memory, which is where accumulators land after readback.
 *
 * DMA side: dma_0 moves K int32 bias values from staging DDR into the weight
 *           buffer at 90 MHz. No accelerator wake-up - no CONTROL, no STATUS.
 *
 * The two measure different things on purpose: do it on the CPU and you pay
 * the add over K*M elements; do it in fabric and you pay the transfer of K
 * values. That is the trade.
 *
 * gcc -O3 -Wall -mcpu=cortex-a9 -mfpu=neon -pthread -o bias_test bias_test.c
 */

#define _POSIX_C_SOURCE 200809L
#include <arm_neon.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define ACCEL_PHYS 0xC0000000UL
#define ACCEL_SPAN 0x00020000UL
#define WBUF_BASE  0x1000u

#define LW_BASE  0xFF200000UL
#define LW_SPAN  0x00200000UL
#define DMA0_OFF 0x00u

#define DMA_STATUS 0
#define DMA_READ   1
#define DMA_WRITE  2
#define DMA_LENGTH 3
#define DMA_CONTROL 6
#define DMA_DONE   0x1u
#define CTRL_GO    (1u << 3)
#define CTRL_LEEN  (1u << 7)
#define CTRL_QUAD  (1u << 11)

#define SRC_PHYS 0x30000000UL
#define SRC_SPAN 0x00100000UL
#define CHUNK    8192u          /* 512 beats; above ~1024 the counter wraps */

#define REPS 9

static volatile uint32_t *dma0;

static void dma_cleanup(void)
{
    if (!dma0) return;
    dma0[DMA_CONTROL] = 0;
    dma0[DMA_STATUS]  = 0;
    dma0[DMA_LENGTH]  = 0;
}

static void on_fatal(int sig)
{
    dma_cleanup();
    _exit(sig == SIGALRM ? 2 : 128 + sig);
}

static uint64_t ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

static int cmp(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

/* ---- CPU: bias[k] broadcast over row k ---------------------------- */

struct job {
    int32_t *out;
    const int32_t *bias;
    int k0, k1, M;
};

static void *worker(void *arg)
{
    struct job *j = arg;

    for (int k = j->k0; k < j->k1; k++) {
        int32_t *row = j->out + (size_t)k * j->M;
        int32x4_t vb = vdupq_n_s32(j->bias[k]);
        int m = 0;

        for (; m + 16 <= j->M; m += 16) {
            int32x4_t a0 = vld1q_s32(row + m);
            int32x4_t a1 = vld1q_s32(row + m + 4);
            int32x4_t a2 = vld1q_s32(row + m + 8);
            int32x4_t a3 = vld1q_s32(row + m + 12);
            vst1q_s32(row + m,      vaddq_s32(a0, vb));
            vst1q_s32(row + m + 4,  vaddq_s32(a1, vb));
            vst1q_s32(row + m + 8,  vaddq_s32(a2, vb));
            vst1q_s32(row + m + 12, vaddq_s32(a3, vb));
        }
        for (; m < j->M; m++) row[m] += j->bias[k];
    }
    return NULL;
}

static uint64_t cpu_add(int32_t *out, const int32_t *bias, int K, int M, int T)
{
    pthread_t th[16];
    struct job jb[16];
    int per = (K + T - 1) / T;

    uint64_t t0 = ns();
    for (int i = 0; i < T; i++) {
        jb[i].out = out;
        jb[i].bias = bias;
        jb[i].M = M;
        jb[i].k0 = i * per;
        jb[i].k1 = (jb[i].k0 + per < K) ? jb[i].k0 + per : K;
        if (jb[i].k0 >= K) jb[i].k0 = jb[i].k1 = K;
        pthread_create(&th[i], NULL, worker, &jb[i]);
    }
    for (int i = 0; i < T; i++) pthread_join(th[i], NULL);
    return ns() - t0;
}

/* ---- DMA: staging DDR -> weight buffer ---------------------------- */

static int dma_push(uint32_t bytes, int *descs)
{
    uint32_t done = 0;
    *descs = 0;

    while (done < bytes) {
        uint32_t n = (bytes - done > CHUNK) ? CHUNK : bytes - done;

        dma0[DMA_STATUS] = 0;
        dma0[DMA_READ]   = (uint32_t)(SRC_PHYS + done);
        dma0[DMA_WRITE]  = WBUF_BASE + done;
        dma0[DMA_LENGTH] = n;
        __sync_synchronize();
        dma0[DMA_CONTROL] = CTRL_QUAD | CTRL_LEEN | CTRL_GO;

        uint64_t t0 = ns();
        while (!(dma0[DMA_STATUS] & DMA_DONE)) {
            if (ns() - t0 > 500000000ull) {
                printf("  dma_0 TIMEOUT: STATUS 0x%08x, %u of %u left\n",
                       dma0[DMA_STATUS], dma0[DMA_LENGTH], n);
                dma_cleanup();
                return -1;
            }
        }
        done += n;
        (*descs)++;
    }
    return 0;
}

/* staging must not overlap kernel-managed RAM */
static int staging_free(void)
{
    FILE *f = fopen("/proc/iomem", "r");
    char line[256];
    int bad = 0;
    if (!f) return 1;
    while (fgets(line, sizeof line, f)) {
        unsigned long a, b;
        char nm[128];
        if (sscanf(line, "%lx-%lx : %127[^\n]", &a, &b, nm) != 3) continue;
        if (strstr(nm, "System RAM") && SRC_PHYS <= b && a < SRC_PHYS + SRC_SPAN) {
            printf("FATAL: staging 0x%08lx overlaps System RAM %08lx-%08lx\n",
                   SRC_PHYS, a, b);
            bad = 1;
        }
    }
    fclose(f);
    return !bad;
}

int main(int argc, char **argv)
{
    int K = (argc > 1) ? atoi(argv[1]) : 512;
    int M = (argc > 2) ? atoi(argv[2]) : 784;
    int T = (argc > 3) ? atoi(argv[3]) : 2;

    if (K < 1 || M < 1 || T < 1 || T > 16) {
        printf("usage: %s [K] [M] [threads<=16]\n", argv[0]);
        return 1;
    }
    setvbuf(stdout, NULL, _IONBF, 0);
    if (!staging_free()) return 1;

    size_t n_out = (size_t)K * M;
    int32_t *out  = malloc(n_out * 4);
    int32_t *bias = malloc((size_t)K * 4);
    if (!out || !bias) { printf("out of memory\n"); return 1; }

    for (size_t i = 0; i < n_out; i++) out[i]  = (int32_t)(i * 2654435761u);
    for (int k = 0; k < K; k++)        bias[k] = (int32_t)(k * 7 - 1000);

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/mem (run as root)"); return 1; }

    void *amap = mmap(NULL, ACCEL_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED,
                      fd, (off_t)ACCEL_PHYS);
    void *lmap = mmap(NULL, LW_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED,
                      fd, (off_t)LW_BASE);
    void *smap = mmap(NULL, SRC_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED,
                      fd, (off_t)SRC_PHYS);
    if (amap == MAP_FAILED || lmap == MAP_FAILED || smap == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    dma0 = (volatile uint32_t *)((volatile uint8_t *)lmap + DMA0_OFF);

    dma_cleanup();
    atexit(dma_cleanup);
    signal(SIGINT, on_fatal);
    signal(SIGTERM, on_fatal);
    signal(SIGALRM, on_fatal);
    alarm(60);

    /* bias into the staging window, padded to a 16-byte beat */
    uint32_t bbytes = ((uint32_t)K * 4 + 15u) & ~15u;
    memcpy((void *)smap, bias, (size_t)K * 4);
    __sync_synchronize();

    uint64_t vc[REPS], vd[REPS];
    int descs = 0;

    for (int r = 0; r < REPS; r++) {
        vc[r] = cpu_add(out, bias, K, M, T);

        uint64_t t0 = ns();
        if (dma_push(bbytes, &descs) != 0) return 1;
        vd[r] = ns() - t0;
    }

    /* keep the adds from being optimised away */
    int64_t sum = 0;
    for (size_t i = 0; i < n_out; i += 997) sum += out[i];

    qsort(vc, REPS, sizeof *vc, cmp);
    qsort(vd, REPS, sizeof *vd, cmp);
    double uc = vc[REPS / 2] / 1e3, ud = vd[REPS / 2] / 1e3;

    double touched = (double)n_out * 4 * 2;      /* read + write */

    printf("\nbias_test: K=%d rows, M=%d cols, %d thread(s), median of %d\n",
           K, M, T, REPS);
    printf("  output accumulators  %zu int32 = %.1f KB\n", n_out, n_out * 4 / 1024.0);
    printf("  bias vector          %d int32 = %u B (%d descriptor(s))\n\n",
           K, bbytes, descs);

    printf("  %-26s %10s %12s\n", "", "us", "MB/s");
    printf("  %-26s %10s %12s\n", "--------------------------",
           "----------", "------------");
    printf("  %-26s %10.2f %12.1f\n", "cpu add (neon, threaded)", uc,
           touched / uc);
    printf("  %-26s %10.2f %12.1f\n\n", "dma_0 bias -> wbuf", ud,
           (double)bbytes / ud);

    printf("  cpu add is %.2fx the dma transfer\n", uc / ud);
    printf("  (cpu figure is read+write over %.1f KB of accumulators;\n"
           "   dma figure moves only the %u-byte bias vector)\n",
           n_out * 4 / 1024.0, bbytes);
    printf("  checksum %lld\n", (long long)sum);

    alarm(0);
    dma_cleanup();
    dma0 = NULL;
    return 0;
}