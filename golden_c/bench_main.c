/* Timing driver for the golden model: median wall time of a hard-reset module run.
 * Usage: snn_golden_bench module.bin image.u8 reps delay_mode dt_us [query_stream]
 * module.bin: "SNNM", int32 version(1), uint32 delay_seed, int32 inc[784*400], int32 thr[400]
 * image.u8  : 784 bytes. Prints per-run times and the median; counts checksum for sanity. */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "snn_golden.h"

static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + 1e-9 * ts.tv_nsec; }
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

int main(int argc, char **argv) {
    FILE *f; char magic[4]; int32_t ver; uint32_t seed;
    static int32_t inc[SG_NIN * SG_NE], thr[SG_NE]; uint8_t img[SG_NIN];
    int reps, dmode, r, j; double dt; uint32_t stream = 12345;
    sg_consts k; sg_module m; sg_engine e; int32_t counts[SG_NE]; double *t;
    if (argc < 6) { fprintf(stderr, "usage: %s module.bin image.u8 reps delay_mode dt_us [stream]\n", argv[0]); return 2; }
    f = fopen(argv[1], "rb"); if (!f) { perror("module"); return 1; }
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "SNNM", 4) || fread(&ver, 4, 1, f) != 1 || fread(&seed, 4, 1, f) != 1 ||
        fread(inc, 4, SG_NIN * SG_NE, f) != SG_NIN * SG_NE || fread(thr, 4, SG_NE, f) != SG_NE) { fprintf(stderr, "bad module\n"); return 1; }
    fclose(f);
    f = fopen(argv[2], "rb"); if (!f || fread(img, 1, SG_NIN, f) != SG_NIN) { fprintf(stderr, "bad image\n"); return 1; }
    fclose(f);
    reps = atoi(argv[3]); dmode = atoi(argv[4]); dt = atof(argv[5]) * 1e-6;
    if (argc > 6) stream = (uint32_t)strtoul(argv[6], NULL, 10);
    sg_consts_init(&k, dt);
    m.inc = inc; m.thr = thr; m.delay_seed = seed;
    if (sg_engine_init(&e, &m, &k, dmode)) { fprintf(stderr, "alloc\n"); return 1; }
    t = (double *)malloc(sizeof(double) * reps);
    for (r = 0; r < reps; r++) {
        sg_stats st = {0, 0, 0, 0}; long sum = 0; double t0 = now();
        sg_run_hard(&e, img, stream, counts, &st);
        t[r] = now() - t0;
        for (j = 0; j < SG_NE; j++) sum += counts[j];
        printf("run %d time_s %.6f e_spikes %ld in_spikes %lld i_spikes %lld\n", r, t[r], sum, (long long)st.in_spikes, (long long)st.i_spikes);
    }
    qsort(t, reps, sizeof(double), cmpd);
    printf("median_s %.6f min_s %.6f max_s %.6f reps %d dt_us %s delay_mode %d\n", t[reps / 2], t[0], t[reps - 1], reps, argv[5], dmode);
    sg_engine_free(&e); free(t);
    return 0;
}
