/*
 * dma_probe.c - is the fabric alive at all?
 *
 * Touches only dma_0.control_port_slave on the LIGHTWEIGHT bridge
 * (0xFF200000), which is a path this project has used successfully before.
 * dma_0 and matmul_accel_0 share pll_0_outclk0, so if this responds:
 *   - the FPGA is configured
 *   - the PLL is locked and the fabric clock is running
 *   - the LW bridge is released
 * which narrows a matmul_accel hang to the h2f bridge or the component itself.
 *
 * gcc -O2 -Wall -o dma_probe dma_probe.c && sudo ./dma_probe
 */
#define _POSIX_C_SOURCE 200809L
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>

#define LW_BASE 0xFF200000UL
#define LW_SPAN 0x00200000UL
#define DMA_OFF 0x00000000UL   /* dma_0.control_port_slave, 0x0..0x1f */

#define DMA_STATUS     0
#define DMA_READ_ADDR  1
#define DMA_WRITE_ADDR 2
#define DMA_LENGTH     3
#define DMA_CONTROL    6

static void on_bus_error(int sig)
{
    (void)sig;
    const char *m = "\n*** SIGBUS/SIGSEGV - access rejected, not hung.\n";
    ssize_t r = write(2, m, strlen(m)); (void)r;
    _exit(2);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    signal(SIGBUS, on_bus_error);
    signal(SIGSEGV, on_bus_error);

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/mem (run as root)"); return 1; }

    void *lw = mmap(NULL, LW_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                    (off_t)LW_BASE);
    if (lw == MAP_FAILED) { perror("mmap"); return 1; }
    volatile uint32_t *dma = (volatile uint32_t *)((char *)lw + DMA_OFF);

    printf("reading dma_0 STATUS at 0xFF200000 ...\n");
    uint32_t st = dma[DMA_STATUS];
    printf("  STATUS = 0x%08x  (done=%u busy=%u)\n", st, st & 1u, (st >> 1) & 1u);

    /* read/write registers: write a pattern and read it back - proves both
     * directions of the fabric path, not just that a read returns something */
    printf("writing 0x12345678 to READ_ADDR ...\n");
    dma[DMA_READ_ADDR] = 0x12345678u;
    uint32_t rb = dma[DMA_READ_ADDR];
    printf("  READ_ADDR reads back 0x%08x  %s\n", rb,
           rb == 0x12345678u ? "- MATCH, fabric is alive" : "- MISMATCH");

    dma[DMA_LENGTH] = 0x00000040u;
    printf("  LENGTH    reads back 0x%08x\n", dma[DMA_LENGTH]);

    if (st == 0xFFFFFFFFu && rb == 0xFFFFFFFFu)
        printf("\nAll-ones everywhere: nothing is driving readdata.\n");

    munmap(lw, LW_SPAN);
    close(fd);
    return 0;
}