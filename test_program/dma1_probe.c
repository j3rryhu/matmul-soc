/*
 * dma1_probe.c - why does dma_1 never return a beat?
 *
 * Three things could explain "LENGTH left 16384 of 16384":
 *
 *   A. dma_1.read_master cannot reach the accelerator at all - wrong base in
 *      its address map, or read_master not wired to the slave.
 *   B. it reaches the accelerator, but the OBUF read path wedges - the
 *      waitrequest / readdatavalid handshake that only the HPS has ever
 *      driven.
 *   C. the transfer is genuinely too long.
 *
 * The ladder below tells them apart. Each step is one DMA descriptor, with a
 * short timeout, a line printed BEFORE it starts so a hard hang still names
 * the step, and a stop-and-clear afterwards.
 *
 *   step 1   16 B from CTRL+0x50 (STATUS)   reaches the accel at all?
 *   step 2   16 B from OBUF_BASE            one beat of the OBUF read path?
 *   step 3    1 KB from OBUF_BASE           a short run of beats?
 *   step 4   16 KB from OBUF_BASE           the full window
 *
 *   step 1 fails            -> A. address map or master connection.
 *   1 ok, 2 fails           -> B. the OBUF read path. Not size.
 *   1,2 ok, 3 or 4 fails    -> C. size or burst length after all.
 *
 * STATUS is the probe target rather than CONTROL because CONTROL is
 * singlepulse and a read of a write-only register can stall on its own.
 * No matmul is run: OBUF holds whatever it held, and this asks only whether
 * the transfer completes, not what the bytes are.
 *
 * The first failure is the informative one. After it the slave may be stuck,
 * so later steps are reported but should not be trusted.
 *
 * Reload the FPGA before every run of this:
 *     sudo dd if=soc_system.rbf of=/dev/fpga0 bs=1M
 *
 * gcc -O2 -Wall -o dma1_probe dma1_probe.c
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
#define OBUF_BASE  0x9000u
#define REG_STRIDE 16
#define REG_STATUS 5

#define LW_BASE  0xFF200000UL
#define LW_SPAN  0x00200000UL
#define DMA0_OFF 0x00u
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

#define DMA1_ACCEL_BASE 0x00000000u   /* dma_1.read_master -> accel */

#define SRC_PHYS 0x30000000UL
#define SRC_SPAN 0x00100000UL
#define STAGE_Y  0x10000u

#define STEP_TIMEOUT_NS 200000000ull   /* 200 ms per step */

static volatile uint8_t  *accel;
static volatile uint32_t *dma0, *dma1;

static uint64_t ns_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void dma_stop(volatile uint32_t *d)
{
    if (!d) return;
    d[DMA_CONTROL] = 0;
    for (int i = 0; i < 10000; i++)
        if (!(d[DMA_STATUS] & DMA_ST_BUSY)) break;
    d[DMA_LENGTH] = 0;
    d[DMA_STATUS] = 0;
}

static void cleanup(void) { dma_stop(dma0); dma_stop(dma1); }

static void on_fatal(int sig)
{
    cleanup();
    const char *m = (sig == SIGALRM)
        ? "\n*** watchdog: DMAs stopped. The step printed last is where it hung.\n"
        : "\n*** signal: DMAs stopped.\n";
    ssize_t r = write(2, m, strlen(m));
    (void)r;
    _exit(3);
}

/* Returns 0 on completion, -1 on timeout. Prints the verdict either way. */
static int step(int n, const char *what, uint32_t rd, uint32_t bytes)
{
    printf("  step %d: %-34s rd 0x%08x  %6u B ... ", n, what, rd, bytes);

    dma1[DMA_STATUS]     = 0;
    dma1[DMA_READ_ADDR]  = rd;
    dma1[DMA_WRITE_ADDR] = (uint32_t)(SRC_PHYS + STAGE_Y);
    dma1[DMA_LENGTH]     = bytes;
    __sync_synchronize();
    dma1[DMA_CONTROL] = CTRL_QUADWORD | CTRL_LEEN | CTRL_GO;

    uint64_t t0 = ns_now();
    for (;;) {
        uint32_t st = dma1[DMA_STATUS];
        if (st & DMA_ST_DONE) {
            printf("ok  (%.1f us)\n", (double)(ns_now() - t0) / 1e3);
            dma_stop(dma1);
            return 0;
        }
        if (ns_now() - t0 > STEP_TIMEOUT_NS) {
            uint32_t left = dma1[DMA_LENGTH];
            printf("TIMEOUT  busy=%u done=%u  moved %u of %u B\n",
                   (st & DMA_ST_BUSY) ? 1u : 0u, (st & DMA_ST_DONE) ? 1u : 0u,
                   bytes - left, bytes);
            dma_stop(dma1);
            return -1;
        }
    }
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    signal(SIGBUS, on_fatal);
    signal(SIGSEGV, on_fatal);
    signal(SIGINT, on_fatal);
    signal(SIGTERM, on_fatal);
    signal(SIGALRM, on_fatal);

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/mem (run as root)"); return 1; }
    void *amap = mmap(NULL, ACCEL_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)ACCEL_PHYS);
    void *lmap = mmap(NULL, LW_SPAN,    PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)LW_BASE);
    void *smap = mmap(NULL, SRC_SPAN,   PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)SRC_PHYS);
    if (amap == MAP_FAILED || lmap == MAP_FAILED || smap == MAP_FAILED) {
        perror("mmap"); return 1;
    }
    accel = (volatile uint8_t *)amap;
    dma0  = (volatile uint32_t *)((volatile uint8_t *)lmap + DMA0_OFF);
    dma1  = (volatile uint32_t *)((volatile uint8_t *)lmap + DMA1_OFF);

    cleanup();
    atexit(cleanup);
    alarm(10);

    /* step 0 - is the slave even alive? If this hangs, the last run left it
     * wedged and nothing below means anything until you reconfigure. */
    printf("  step 0: HPS reads STATUS ... ");
    uint32_t st0 = *(volatile uint32_t *)(accel + REG_STATUS * REG_STRIDE);
    printf("0x%08x  (slave responding)\n", st0);

    int r1 = step(1, "dma_1 <- CTRL STATUS beat",
                  DMA1_ACCEL_BASE + REG_STATUS * REG_STRIDE, 16);
    int r2 = step(2, "dma_1 <- OBUF, one beat",   DMA1_ACCEL_BASE + OBUF_BASE, 16);
    int r3 = step(3, "dma_1 <- OBUF, 1 KB",       DMA1_ACCEL_BASE + OBUF_BASE, 1024);
    int r4 = step(4, "dma_1 <- OBUF, full 16 KB", DMA1_ACCEL_BASE + OBUF_BASE, 16384);

    printf("\n  verdict: ");
    if (r1 != 0)
        printf("dma_1 cannot reach the accelerator at all.\n"
               "  Not a size problem and not the OBUF path. Check the\n"
               "  dma_1.read_master column of the address map, and that\n"
               "  read_master is actually connected to the accel slave.\n");
    else if (r2 != 0)
        printf("dma_1 reaches the accelerator, but one OBUF beat wedges.\n"
               "  The read handshake, not the size - a single beat is the\n"
               "  smallest possible transfer. Look at avs_waitrequest /\n"
               "  avs_readdatavalid for OBUF reads with a second master.\n");
    else if (r3 != 0 || r4 != 0)
        printf("single beats work, longer runs do not - size or burst\n"
               "  length after all. Largest good size was %s.\n",
               r3 == 0 ? "1 KB" : "16 B");
    else
        printf("every step passed. dma_1 reads OBUF fine in isolation,\n"
               "  so the failure needs the matmul to have run first -\n"
               "  suspect state left by the compute pass.\n");

    printf("\n  (first failure is the real one; after it the slave may be\n"
           "   stuck, so later steps are unreliable. Reconfigure before\n"
           "   the next run: sudo dd if=soc_system.rbf of=/dev/fpga0 bs=1M)\n");

    alarm(0);
    cleanup();
    dma0 = dma1 = NULL;
    munmap(amap, ACCEL_SPAN); munmap(lmap, LW_SPAN); munmap(smap, SRC_SPAN);
    close(fd);
    return 0;
}