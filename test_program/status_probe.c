/*
 * status_probe.c - minimal bus-path diagnostic for matmul_accel_0.
 *
 * Does as little as possible, in increasing order of risk, printing before
 * every access so the last line you see is the step that hung:
 *
 *   1. mmap only                  - no bus traffic at all
 *   2. 32-bit write to WEIGHT_ROWS - write path (harmless register)
 *   3. byte read from OBUF         - read path the cocotb suite DOES cover
 *   4. 32-bit read of STATUS       - read path the cocotb suite does NOT cover
 *
 * If 3 works and 4 hangs, the fault is specific to the ctrl_rf read return
 * (most likely avs_readdatavalid never asserted for a CTRL-region read, which
 * a testbench whose avalon_read only waits on avs_waitrequest cannot detect).
 *
 * A hung AXI transaction on the h2f bridge has no timeout - it takes the HPS
 * down with it and needs a power cycle.  Each step prompts first, so you can
 * stop before the one you don't want to risk.  -y skips the prompts.
 *
 * Build:  gcc -O2 -Wall -o status_probe status_probe.c
 * Run:    sudo ./status_probe
 */

#define _POSIX_C_SOURCE 200809L

#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define ACCEL_PHYS 0xC0000000UL
#define ACCEL_SPAN 0x00040000UL

#define CTRL_BASE 0x0000u
#define OBUF_BASE 0x7000u

#define REG_WEIGHT_ROWS 0x04u
#define REG_STATUS      0x14u

/* avs_address is 16-bit with WORD (32-bit) address units, so the CPU byte
 * offset is 4*a.  Register offsets are multiples of 4, so this is exact for
 * them regardless of the byte-lane convention. */
#define AVS(a) (((uint32_t)(a)) << 2)

static volatile uint8_t *accel;
static int auto_yes = 0;

static void on_bus_error(int sig)
{
    (void)sig;
    /* write(2) is async-signal-safe; printf is not */
    const char *m = "\n*** SIGBUS: the access was rejected rather than hung.\n"
                    "*** Usually means the h2f bridge is disabled or the FPGA\n"
                    "*** is not configured. Check /sys/class/fpga_bridge/.\n";
    ssize_t r = write(2, m, strlen(m));
    (void)r;
    _exit(2);
}

static void step(const char *what)
{
    printf("\n--- %s\n", what);
    if (!auto_yes) {
        printf("    press enter to proceed (ctrl-c to stop): ");
        fflush(stdout);
        int c;
        while ((c = getchar()) != '\n' && c != EOF) { }
        if (c == EOF) { printf("\n"); exit(0); }
    }
    fflush(stdout);
}

int main(int argc, char **argv)
{
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "-y")) auto_yes = 1;

    /* unbuffered: the last line printed must be the step that hung */
    setvbuf(stdout, NULL, _IONBF, 0);
    signal(SIGBUS, on_bus_error);
    signal(SIGSEGV, on_bus_error);

    printf("status_probe: accel at phys 0x%08lx, span 0x%lx\n", ACCEL_PHYS,
           ACCEL_SPAN);
    printf("  WEIGHT_ROWS avs 0x%02x -> cpu +0x%05x\n", REG_WEIGHT_ROWS,
           AVS(REG_WEIGHT_ROWS));
    printf("  STATUS      avs 0x%02x -> cpu +0x%05x\n", REG_STATUS,
           AVS(REG_STATUS));
    printf("  OBUF[0]     avs 0x%04x -> cpu +0x%05x\n", OBUF_BASE,
           AVS(OBUF_BASE));

    step("step 1: open /dev/mem and mmap (no bus traffic)");
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/mem (run as root)"); return 1; }
    void *map = mmap(NULL, ACCEL_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                     (off_t)ACCEL_PHYS);
    if (map == MAP_FAILED) { perror("mmap"); close(fd); return 1; }
    accel = (volatile uint8_t *)map;
    printf("    mapped at %p - OK (nothing has touched the bus yet)\n", map);

    step("step 2: 32-bit WRITE to WEIGHT_ROWS (write path)");
    *(volatile uint32_t *)(accel + AVS(CTRL_BASE + REG_WEIGHT_ROWS + 4)) = 16;
    printf("    write returned - OK\n");
    printf("    (posted write; a stuck avs_waitrequest would have hung here)\n");

    uint8_t readback = *(volatile uint32_t *)(accel + AVS(CTRL_BASE + REG_WEIGHT_ROWS + 4));
    printf("    data read back is: %d", readback);

    step("step 3: byte READ from OBUF (read path the TB covers)");
    uint8_t b = accel[AVS(OBUF_BASE)];
    printf("    OBUF[0] = 0x%02x - OK, read path returns\n", b);

    step("step 4: 32-bit READ of STATUS (read path the TB does NOT cover)");
    uint32_t st = *(volatile uint32_t *)(accel + AVS(CTRL_BASE + REG_STATUS));
    printf("    STATUS = 0x%08x  (busy=%u done=%u) - OK\n", st, st & 1u,
           (st >> 1) & 1u);

    if (st == 0xFFFFFFFFu)
        printf("    NB: all-ones usually means nothing is driving readdata.\n");

    step("step 5: three more STATUS reads (done is read-to-clear)");
    for (int i = 0; i < 3; i++) {
        uint32_t s = *(volatile uint32_t *)(accel + AVS(CTRL_BASE + REG_STATUS));
        printf("    STATUS = 0x%08x  (busy=%u done=%u)\n", s, s & 1u,
               (s >> 1) & 1u);
    }
    printf("\nAll steps completed. The ctrl_rf read path works.\n");

    munmap(map, ACCEL_SPAN);
    close(fd);
    return 0;
}