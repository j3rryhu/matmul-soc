/*
 * ctrl_probe.c - which CTRL write wedges the bus?
 *
 * bisect_probe got through WBUF writes, IBUF writes, OBUF byte and word
 * reads, and STATUS reads - so the h2f bridge, the interconnect and the
 * buffer paths are all fine. It wedged on the block that writes the five
 * dimension registers.
 *
 * Two candidates:
 *   (a) one specific register address never acks
 *   (b) back-to-back CTRL writes wedge the regblock. The cocotb TB's
 *       avalon_write always drops avs_write for at least one cycle between
 *       transactions, so consecutive writes with no idle cycle - which the
 *       AXI-to-Avalon agent will happily issue - are never exercised in sim.
 *       status_probe only ever did ONE write, which is why it passed.
 *
 * Part 1 writes each register alone with a STATUS canary in between, and a
 * deliberate delay so the transactions cannot coalesce. If one address is
 * bad, that step hangs and names it.
 *
 * Part 2 writes them with no gap at all. If part 1 passes and part 2 hangs,
 * it is back-to-back writes, not any single address.
 *
 * gcc -O2 -Wall -o ctrl_probe ctrl_probe.c && sudo ./ctrl_probe
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
#define REG_STATUS 0x14u

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

static void gap(void)
{
    struct timespec ts = { 0, 10 * 1000 * 1000 }; /* 10 ms */
    nanosleep(&ts, NULL);
}

static void canary(const char *after)
{
    printf("    canary: STATUS read after %s ...\n", after);
    uint32_t st = rd32(CTRL_BASE + REG_STATUS);
    printf("    canary OK, STATUS = 0x%08x\n", st);
}

/* one register, alone, with idle time either side */
static void one(uint32_t off, uint32_t val, const char *name)
{
    char buf[96];
    snprintf(buf, sizeof buf, "write %s (avs 0x%02x) = %u, alone",
             name, off, val);
    step(buf);
    gap();
    wr32(CTRL_BASE + off, val);
    printf("    write returned\n");
    gap();
    canary(name);
}

int main(int argc, char **argv)
{
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "-y")) auto_yes = 1;

    setvbuf(stdout, NULL, _IONBF, 0);
    signal(SIGBUS, on_bus_error);
    signal(SIGSEGV, on_bus_error);

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/mem (run as root)"); return 1; }
    void *map = mmap(NULL, ACCEL_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                     (off_t)ACCEL_PHYS);
    if (map == MAP_FAILED) { perror("mmap"); return 1; }
    accel = (volatile uint8_t *)map;
    printf("ctrl_probe: accel at 0x%08lx, mapped at %p\n", ACCEL_PHYS, map);

    step("baseline STATUS read");
    canary("nothing");

    /* ---- part 1: each register alone ---- */
    printf("\n=== PART 1: one register at a time, 10 ms idle either side ===\n");

    one(0x08u, 16, "WEIGHT_COLS");    /* status_probe already proved this one */
    one(0x04u, 16, "WEIGHT_ROWS");
    one(0x0Cu, 16, "INPUT_MAX_ADDR");
    one(0x10u, 16, "INPUT_COLS");
    one(0x18u, 32768, "OUTPUT_SCALE");

    printf("\n=== PART 1 PASSED: every register is individually writable ===\n");

    /* ---- part 2: back to back ---- */
    printf("\n=== PART 2: consecutive writes, no gap ===\n");

    step("two back-to-back writes (WEIGHT_ROWS, WEIGHT_COLS)");
    wr32(CTRL_BASE + 0x04u, 16);
    wr32(CTRL_BASE + 0x08u, 16);
    printf("    2 writes returned\n");
    canary("2 back-to-back writes");

    step("three back-to-back writes");
    wr32(CTRL_BASE + 0x04u, 16);
    wr32(CTRL_BASE + 0x08u, 16);
    wr32(CTRL_BASE + 0x0Cu, 16);
    printf("    3 writes returned\n");
    canary("3 back-to-back writes");

    step("all five back-to-back (what matmul_min does)");
    wr32(CTRL_BASE + 0x04u, 16);
    wr32(CTRL_BASE + 0x08u, 16);
    wr32(CTRL_BASE + 0x0Cu, 16);
    wr32(CTRL_BASE + 0x10u, 16);
    wr32(CTRL_BASE + 0x18u, 32768);
    printf("    5 writes returned\n");
    canary("5 back-to-back writes");

    printf("\n=== PART 2 PASSED: back-to-back CTRL writes are fine too ===\n");
    printf("\nNeither candidate reproduced it. The difference must be\n"
           "elsewhere - re-run bisect_probe and note whether the hang is on\n"
           "a write or on the canary read that follows.\n");

    munmap(map, ACCEL_SPAN);
    close(fd);
    return 0;
}