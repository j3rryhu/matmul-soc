/* Tiled int8 matmul, N x N @ N x N.  ./matmul_baseline N [tile]
 * gcc -O3 -Wall -mcpu=cortex-a9 -mfpu=neon -o matmul_baseline matmul_baseline.c */

#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv)
{
    int N = (argc > 1) ? atoi(argv[1]) : 256;
    int T = (argc > 2) ? atoi(argv[2]) : 32;
    if (N <= 0) { printf("usage: %s N [tile]\n", argv[0]); return 1; }
    if (T <= 0 || T > N) T = N;

    size_t n2 = (size_t)N * N;
    int8_t *A  = malloc(n2);
    int8_t *Bt = malloc(n2);          /* B transposed: contiguous inner loop */
    int8_t *C  = malloc(n2);
    if (!A || !Bt || !C) { printf("out of memory\n"); return 1; }

    for (size_t i = 0; i < n2; i++) A[i]  = (int8_t)(rand() % 255 - 127);
    for (size_t i = 0; i < n2; i++) Bt[i] = (int8_t)(rand() % 255 - 127);

    double macs = (double)N * N * N;
    int reps = (int)(4e8 / macs);
    if (reps < 1) reps = 1;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int r = 0; r < reps; r++) {
        for (int ii = 0; ii < N; ii += T) {
            int imax = (ii + T < N) ? ii + T : N;
            for (int jj = 0; jj < N; jj += T) {
                int jmax = (jj + T < N) ? jj + T : N;
                for (int i = ii; i < imax; i++) {
                    const int8_t *ar = A + (size_t)i * N;
                    for (int j = jj; j < jmax; j++) {
                        const int8_t *br = Bt + (size_t)j * N;
                        int32_t s = 0;
                        for (int k = 0; k < N; k++)
                            s += (int32_t)ar[k] * (int32_t)br[k];
                        C[(size_t)i * N + j] = (int8_t)(s >> 12);
                    }
                }
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double us = ((t1.tv_sec - t0.tv_sec) * 1e9
               + (t1.tv_nsec - t0.tv_nsec)) / 1e3 / reps;

    printf("%dx%d tile %d: %.2f us, %.3f GOP/s\n",
           N, N, T, us, 2.0 * macs / (us * 1e3));

    free(A); free(Bt); free(C);
    return 0;
}