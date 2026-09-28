/*
 * ocram_probe.c - does the h2f bridge work at all?
 *
 * Touches ONLY onchip_memory at 0x40000 in hps_0.h2f_axi_master's address
 * space, i.e. physical 0xC0040000.  Deliberately does not go near
 * matmul_accel_0 (0x0..0x3ffff in that same space).
 *
 * This is a control experiment.  Both slaves sit behind the same bridge,
 * the same clock and the same decode path, so:
 *
 *   RAM works, accel hangs  -> bridge/clock/HPS config are fine; the fault
 *                              is specific to the accel's AXI-to-Avalon
 *                              adaptation or its interface declaration.
 *   RAM hangs too           -> the accel is exonerated; the problem is the
 *                              h2f bridge itself, its clock, or HPS setup.
 *                              An AXI rewrite of the accel would be wasted.
 *
 * A hung AXI transaction has no timeout and takes the HPS with it - each
 * step prompts first.  -y skips the prompts.
 *
 * gcc -O2 -Wall -o ocram_probe ocram_probe.c && sudo ./ocram_probe
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

#define H2F_BASE   0xC0000000UL
#define OCRAM_OFF  0x00040000UL          /* base in h2f_axi_master's space */
#define OCRAM_PHYS (H2F_BASE + OCRAM_OFF)
#define MAP_LEN    0x00001000UL          /* one page is plenty */

static int auto_yes = 0;

static void on_bus_error(int sig)
{
    const char *m = (sig == SIGBUS)
        ? "\n*** SIGBUS: access rejected, not hung.\n"
          "*** Bridge disabled, FPGA unconfigured, or nothing at this address.\n"
        : "\n*** SIGSEGV during access.\n";
    ssize_t r = write(2, m, strlen(m));
    (void)r;
    _exit(2);
}

static void step(const char *what)
{
    printf("\n--- %s\n", what);
    if (!auto_yes) {
        printf("    press enter (ctrl-c to stop): ");
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

    setvbuf(stdout, NULL, _IONBF, 0); /* last line printed == step that hung */
    signal(SIGBUS, on_bus_error);
    signal(SIGSEGV, on_bus_error);

    printf("ocram_probe: on-chip RAM at phys 0x%08lx\n", OCRAM_PHYS);
    printf("  (h2f bridge 0x%08lx + qsys base 0x%05lx)\n", H2F_BASE, OCRAM_OFF);

    step("step 1: open /dev/mem and mmap (no bus traffic)");
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/mem (run as root)"); return 1; }
    void *map = mmap(NULL, MAP_LEN, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                     (off_t)OCRAM_PHYS);
    if (map == MAP_FAILED) { perror("mmap"); close(fd); return 1; }
    volatile uint32_t *ram = (volatile uint32_t *)map;
    printf("    mapped at %p - OK (bus untouched so far)\n", map);

    step("step 2: single 32-bit WRITE to offset 0");
    ram[0] = 0xDEADBEEFu;
    printf("    write returned - OK\n");

    step("step 3: single 32-bit READ of offset 0");
    uint32_t v = ram[0];
    printf("    read 0x%08x %s\n", v,
           v == 0xDEADBEEFu ? "- MATCH, the h2f bridge works" : "- MISMATCH");

    if (v == 0xFFFFFFFFu)
        printf("    NB: all-ones usually means nothing is driving readdata.\n");
    else if (v == 0x00000000u)
        printf("    NB: all-zeros can mean the write never landed.\n");

    step("step 4: walking pattern over 16 words");
    int bad = 0;
    for (int i = 0; i < 16; i++) ram[i] = 0x100u + (uint32_t)i;
    for (int i = 0; i < 16; i++) {
        uint32_t got = ram[i], want = 0x100u + (uint32_t)i;
        if (got != want) {
            if (bad < 8)
                printf("    word %2d: got 0x%08x want 0x%08x\n", i, got, want);
            bad++;
        }
    }
    printf("    %s (%d/16 bad)\n", bad ? "FAIL" : "PASS - reads and writes both work",
           bad);

    step("step 5: byte-granular access (what the accel's buffers need)");
    volatile uint8_t *b = (volatile uint8_t *)map;
    for (int i = 0; i < 8; i++) b[i] = (uint8_t)(0xA0 + i);
    int bbad = 0;
    for (int i = 0; i < 8; i++) {
        uint8_t got = b[i], want = (uint8_t)(0xA0 + i);
        if (got != want) {
            printf("    byte %d: got 0x%02x want 0x%02x\n", i, got, want);
            bbad++;
        }
    }
    printf("    %s\n", bbad ? "byte access FAILED - byteenables not reaching the slave"
                            : "PASS - byte writes land individually");

    printf("\nIf all steps passed, the h2f bridge, its clock and the HPS-side\n"
           "config are all fine, and the matmul_accel hang is specific to that\n"
           "component's path.\n");

    munmap(map, MAP_LEN);
    close(fd);
    return bad || bbad;
}