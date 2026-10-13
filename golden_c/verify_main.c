/* Run the exported golden vectors (golden_vectors/) and check bit-exactness + time each run.
 * Usage: snn_golden_verify <golden_vectors_dir> [reps]
 * Intended for any target with a C99 compiler (DGX Spark, Icicle Kit U54 Linux, Jetson).
 * Prints PASS/FAIL per module and the median wall time of a hard-reset module run. */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "snn_golden.h"
#define NMOD 4
#define NQ 20
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + 1e-9 * ts.tv_nsec; }
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
static int rd(const char *dir, const char *name, void *buf, size_t sz, size_t n) {
    char p[1024]; FILE *f; size_t r; snprintf(p, sizeof p, "%s/%s", dir, name);
    f = fopen(p, "rb"); if (!f) { perror(p); return -1; }
    r = fread(buf, sz, n, f); fclose(f); return r == n ? 0 : -1;
}
int main(int argc, char **argv) {
    static int32_t inc[SG_NIN * SG_NE], thr[SG_NE], expv[NMOD * NQ * SG_NE];
    static uint8_t imgs[NQ * SG_NIN]; uint32_t streams[NQ];
    int reps = argc > 2 ? atoi(argv[2]) : 3, k, q, r, fails = 0, nt = 0;
    double *t; sg_consts kc;
    if (argc < 2) { fprintf(stderr, "usage: %s golden_vectors_dir [reps]\n", argv[0]); return 2; }
    if (rd(argv[1], "queries.u8", imgs, 1, sizeof imgs) || rd(argv[1], "streams.u32", streams, 4, NQ) ||
        rd(argv[1], "expected.i32", expv, 4, NMOD * NQ * SG_NE)) return 1;
    t = (double *)malloc(sizeof(double) * NMOD * NQ * reps);
    sg_consts_init(&kc, 100e-6);
    for (k = 0; k < NMOD; k++) {
        char name[64], magic[4]; int32_t ver; uint32_t seed; FILE *f; char p[1024];
        sg_module m; sg_engine e; int bad = 0;
        snprintf(name, sizeof name, "module_%d.bin", k); snprintf(p, sizeof p, "%s/%s", argv[1], name);
        f = fopen(p, "rb"); if (!f) { perror(p); return 1; }
        if (fread(magic, 1, 4, f) != 4 || fread(&ver, 4, 1, f) != 1 || fread(&seed, 4, 1, f) != 1 ||
            fread(inc, 4, SG_NIN * SG_NE, f) != SG_NIN * SG_NE || fread(thr, 4, SG_NE, f) != SG_NE) { fprintf(stderr, "bad %s\n", p); return 1; }
        fclose(f);
        m.inc = inc; m.thr = thr; m.delay_seed = seed;
        if (sg_engine_init(&e, &m, &kc, SG_DELAY_1MS_BINNED)) return 1;
        for (q = 0; q < NQ; q++) for (r = 0; r < reps; r++) {
            int32_t c[SG_NE]; double t0 = now();
            sg_run_hard(&e, imgs + q * SG_NIN, streams[q], c, NULL);
            t[nt++] = now() - t0;
            if (r == 0 && memcmp(c, expv + (k * NQ + q) * SG_NE, sizeof c)) bad++;
        }
        printf("module %d: %s (%d/%d queries bit-exact)\n", k, bad ? "FAIL" : "PASS", NQ - bad, NQ);
        fails += bad; sg_engine_free(&e);
    }
    qsort(t, nt, sizeof(double), cmpd);
    printf("%s; module runs timed: %d; median_s %.6f min_s %.6f max_s %.6f\n", fails ? "FAIL" : "ALL PASS", nt, t[nt / 2], t[0], t[nt - 1]);
    free(t); return fails ? 1 : 0;
}
