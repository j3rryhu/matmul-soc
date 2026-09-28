/*
 * bisect_probe.c - find which operation wedges the h2f bus.
 *
 * status_probe.c passes; matmul_min.c hangs. The operations matmul_min does
 * that status_probe never did are, in order of suspicion:
 *
 *   - writes to WBUF (0x1000) and IBUF (0x5000)   [untested regions]
 *   - a 32-bit read of OBUF (byteenable 0xF)      [probe used a BYTE read]
 *   - a write to CONTROL, which starts the matmul [hands buffers to the core]
 *
 * This walks them one at a time. After every new operation it reads STATUS
 * as a canary: if the bus is wedged, the canary read is what hangs, so the
 * last line printed names the operation that wedged it.
 *
 * Prompts before each step by default (-y to skip) because the later steps
 * can take the HPS down and need a power cycle.
 *
 * gcc -O2 -Wall -o bisect_probe bisect_probe.c && sudo ./bisect_probe
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
#define ACCEL_SPAN 0x00040000UL

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

#define AVS(a) (((uint32_t)(a)) << 2)

static volatile uint8_t *accel;
static int auto_yes = 0;

static void on_bus_error(int sig)
{
    (void)sig;
    const char *m = "\n*** SIGBUS: rejected, not hung.\n";
    ssize_t r = write(2, m, strlen(m));
    (void)r;
    _exit(2);
}

static void step(const char *what)
{
    printf("\n--- %s\n", what);
    if (!auto_yes) {
        printf("    press enter (ctrl-c to stop): ");
        int c;
        while ((c = getchar()) != '\n' && c != EOF) { }
        if (c == EOF) { printf("\n"); exit(0); }
    }
}

static void wr32(uint32_t a, uint32_t v)
{
    *(volatile uint32_t *)(accel + AVS(a)) = v;
}
static uint32_t rd32(uint32_t a)
{
    return *(volatile uint32_t *)(accel + AVS(a));
}

/* read STATUS to prove the bus still responds after the previous step */
static void canary(const char *after)
{
    printf("    canary: reading STATUS after %s ...\n", after);
    uint32_t st = rd32(CTRL_BASE + REG_STATUS);
    printf("    canary OK, STATUS = 0x%08x (busy=%u done=%u)\n",
           st, st & 1u, (st >> 1) & 1u);
}

int main(int argc, char **argv)
{
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "-y")) auto_yes = 1;

    setvbuf(stdout, NULL, _IONBF, 0);
    signal(SIGBUS, on_bus_error);
    signal(SIGSEGV, on_bus_error);

    printf("bisect_probe: accel at 0x%08lx\n", ACCEL_PHYS);

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/mem (run as root)"); return 1; }
    void *map = mmap(NULL, ACCEL_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                     (off_t)ACCEL_PHYS);
    if (map == MAP_FAILED) { perror("mmap"); return 1; }
    accel = (volatile uint8_t *)map;
    printf("mapped at %p\n", map);

    step("A: baseline STATUS read");
    canary("nothing");

    step("B: ONE 32-bit write to WBUF (avs 0x1000)");
    wr32(WBUF_BASE, 0x04030201u);
    printf("    write returned\n");
    canary("one WBUF write");

    step("C: 64 words to WBUF (the whole weight image)");
    for (uint32_t i = 0; i < 256; i += 4) wr32(WBUF_BASE + i, 0x02020202u);
    printf("    64 writes returned\n");
    canary("full WBUF image");

    step("D: ONE 32-bit write to IBUF (avs 0x5000)");
    wr32(IBUF_BASE, 0x11223344u);
    printf("    write returned\n");
    canary("one IBUF write");

    step("E: all 16 IBUF banks (4 words each, stride 256)");
    for (uint32_t p = 0; p < 16; p++)
        for (uint32_t m = 0; m < 16; m += 4)
            wr32(IBUF_BASE + p * 256u + m, 0x01010101u);
    printf("    64 writes returned\n");
    canary("full IBUF image");

    step("F: BYTE read of OBUF (what status_probe did - known good)");
    {
        uint8_t b = accel[AVS(OBUF_BASE)];
        printf("    OBUF[0] byte = 0x%02x\n", b);
    }
    canary("OBUF byte read");

    step("G: 32-bit read of OBUF (byteenable 0xF - NOT previously tested)");
    printf("    reading avs 0x7000 as a word ...\n");
    {
        uint32_t w = rd32(OBUF_BASE);
        printf("    OBUF word = 0x%08x\n", w);
    }
    canary("OBUF word read");

    step("H: program the 5 dimension registers");
    wr32(CTRL_BASE + REG_WEIGHT_ROWS, 16);
    wr32(CTRL_BASE + REG_WEIGHT_COLS, 16);
    wr32(CTRL_BASE + REG_INPUT_MAX_ADDR, 16);
    wr32(CTRL_BASE + REG_INPUT_COLS, 16);
    wr32(CTRL_BASE + REG_OUTPUT_SCALE, 32768);
    printf("    5 registers written\n");
    canary("dimension registers");

    step("I: write CONTROL.matmul_start = 1  <-- PRIME SUSPECT\n"
         "    (hands the buffer ports to weight_ctrl/output_loader;\n"
         "     if the slave stalls while busy, the canary read hangs)");
    wr32(CTRL_BASE + REG_CONTROL, 1u);
    printf("    start pulse written, returned\n");

    printf("    sleeping 100 ms so the matmul finishes before we read ...\n");
    {
        struct timespec ts = { 0, 100 * 1000 * 1000 };
        nanosleep(&ts, NULL);
    }
    canary("matmul_start + 100 ms");

    step("J: poll STATUS 10 times");
    for (int i = 0; i < 10; i++) {
        uint32_t st = rd32(CTRL_BASE + REG_STATUS);
        printf("    poll %2d: STATUS = 0x%08x (busy=%u done=%u)\n",
               i, st, st & 1u, (st >> 1) & 1u);
    }

    step("K: read the whole output buffer (16 banks x 4 words)");
    for (uint32_t k = 0; k < 16; k++) {
        for (uint32_t m = 0; m < 16; m += 4) {
            uint32_t w = rd32(OBUF_BASE + k * 128u + m);
            if (k < 2) printf("    OBUF[%u][%u..] = 0x%08x\n", k, m, w);
        }
    }
    printf("    all 64 reads returned\n");

    printf("\nAll steps completed - no operation wedged the bus.\n");

    munmap(map, ACCEL_SPAN);
    close(fd);
    return 0;
}