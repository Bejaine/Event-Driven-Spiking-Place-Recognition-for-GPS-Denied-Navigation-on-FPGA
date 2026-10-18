/*
 * SpikeNav fixed-point golden model on CUDA (Jetson Orin Nano, sm_87): batched hard-reset
 * module runs, bit-exact with golden_c/snn_golden.c (checked against golden_vectors/ and,
 * for other dt, against the C model run on the host).
 *
 * Mapping: one thread block per module run, one thread per E/I neuron pair (400 threads).
 *  - neuron state lives in registers (int64, same arithmetic as sg_step);
 *  - the 16 xorshift32 input generators run on threads 0-15 and append this step's input
 *    events to a shared list (order differs from the C model's sorted list, which does not
 *    matter: the delay ring holds exact integer sums);
 *  - delay ring: in the hardware delay mode (SG_DELAY_1MS_BINNED) every arrival lands on a
 *    whole-ms step at most 11 ms ahead, so the C model's (max_delay+spm+1)-slot ring is held
 *    as 12 ms bins in shared memory. Each thread only touches its own column: no atomics;
 *  - shared-count inhibition n_i = __syncthreads_count(spk_i).
 * Only delay mode 3 (the proposed hardware config) is implemented.
 *
 * Usage:
 *   snn_cuda verify <golden_vectors_dir>                         bit-exact check (dt 0.1 ms)
 *   snn_cuda bench  <golden_vectors_dir> <dt_us> <reps> [out.json]  latency + throughput
 *   snn_cuda loop   <golden_vectors_dir> <dt_us> <batch> <seconds>  sustained load (power)
 */
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
extern "C" {
#include "../golden_c/snn_golden.h"
}

#define NB 12  /* ms bins: arrivals are 1..11 ms ahead of the current ms */
#define NMOD 4
#define NQ 20
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

__device__ __forceinline__ long long asr(long long x, int s) { return x >= 0 ? (x >> s) : ~((~x) >> s); }
__device__ __forceinline__ long long rshr(long long x, int s) { return asr(x + (1LL << (s - 1)), s); }
__device__ __forceinline__ long long clamp64(long long x, long long lo, long long hi) { return x < lo ? lo : (x > hi ? hi : x); }
__device__ __forceinline__ uint32_t xorshift32(uint32_t x) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x; }

struct Run { int mod; int img; };   /* module index, image/stream index */

__global__ void __launch_bounds__(SG_NE)
run_hard_kernel(const int16_t *__restrict__ inc_all, const uint8_t *__restrict__ dly_all,
                const int32_t *__restrict__ thr_all, const int32_t *__restrict__ pthr_all,
                const uint32_t *__restrict__ seed_all, const Run *__restrict__ runs, sg_consts k,
                int32_t *__restrict__ counts_out, int *__restrict__ in_spikes_out) {
    __shared__ int32_t sbuf[NB][SG_NE];
    __shared__ int32_t pthr[SG_NIN];
    __shared__ int16_t evl[3][SG_NIN];
    __shared__ int evn[3];
    const int j = threadIdx.x;
    const Run rr = runs[blockIdx.x];
    const int16_t *inc = inc_all + (size_t)rr.mod * SG_NIN * SG_NE;
    const uint8_t *dly = dly_all + (size_t)rr.mod * SG_NIN * SG_NE;
    const long long thr = thr_all[rr.mod * SG_NE + j];
    for (int i = j; i < SG_NIN; i += SG_NE) pthr[i] = pthr_all[rr.img * SG_NIN + i];
    for (int b = 0; b < NB; b++) sbuf[b][j] = 0;
    if (j < 3) evn[j] = 0;
    uint32_t gst = j < SG_NGEN ? seed_all[rr.img * SG_NGEN + j] : 0;
    long long ve = k.V_REST_E, vi = k.V_REST_I, gee = 0, gie = 0, gei = 0;
    int since_e = SG_BIG, since_i = SG_BIG, timer = SG_BIG, cnt = 0, n_in = 0;
    const int spm = k.steps_per_ms;
    __syncthreads();
    for (int t = 0; t < k.n_present; t++) {
        const int b = t % 3;
        /* input generators (threads 0-15): same sequence as lfsr_step */
        if (j < SG_NGEN) {
            uint32_t s = gst;
            for (int kk = 0; kk < SG_NIN / SG_NGEN; kk++) {
                s = xorshift32(s);
                if ((int32_t)(s >> (32 - SG_THR_BITS)) < pthr[kk * SG_NGEN + j]) {
                    int p = atomicAdd(&evn[b], 1);
                    evl[b][p] = (int16_t)(kk * SG_NGEN + j);
                }
            }
            gst = s;
        }
        if (j == 0) evn[(t + 1) % 3] = 0;   /* last read in step t-2, before the barrier of t-1 */
        /* 1. refractoriness and Euler update (sg_step part 1) */
        if (since_e < SG_BIG) since_e++;
        if (since_i < SG_BIG) since_i++;
        if (timer < SG_BIG) timer++;
        const int nre = since_e >= k.REF_E, nri = since_i >= k.REF_I;
        if (nre) {
            long long C = gie * asr(k.E_INH_E - ve, SG_VS) - gee * asr(ve, SG_VS);
            long long acc = k.KV_E * (k.V_REST_E - ve) + k.KC_E * C;
            ve = clamp64(ve + rshr(acc, SG_SH), SG_V_MIN, SG_V_MAX);
        }
        gee = asr(gee * k.DGE, SG_GSH);
        gie = asr(gie * k.DGI, SG_GSH);
        if (nri) {
            long long acc = k.KV_I * (k.V_REST_I - vi) - k.KC_I * (gei * asr(vi, SG_VS));
            vi = clamp64(vi + rshr(acc, SG_SH), SG_V_MIN, SG_V_MAX);
        }
        gei = asr(gei * k.DGE, SG_GSH);
        /* 2. thresholds */
        const int spk_i = (vi > k.V_TH_I) && nri;
        const int spk_e = (ve > thr) && (timer >= k.T_LOCK) && nre;
        const int n_i = __syncthreads_count(spk_i);   /* also publishes evl[b] */
        /* 3. synapses: push this step's events into the ms bins of my column, deliver */
        const int nev = evn[b];
        const int tb = t / spm;
        for (int q = 0; q < nev; q++) {
            const int i = evl[b][q];
            const int d = dly[i * SG_NE + j];
            sbuf[(tb + 1 + d) % NB][j] += inc[i * SG_NE + j];
        }
        if (t % spm == 0) { gee += sbuf[tb % NB][j]; sbuf[tb % NB][j] = 0; }
        gei += spk_e ? k.W_EI : 0;
        if (gei > SG_G_MAX) gei = SG_G_MAX;
        gie += k.W_IE * (long long)(n_i - spk_i);
        if (gie > SG_G_MAX) gie = SG_G_MAX;
        if (gee > SG_G_MAX) gee = SG_G_MAX;
        /* 4. resets */
        if (spk_e) { ve = k.V_RESET_E; timer = 0; since_e = 0; cnt++; }
        if (spk_i) { vi = k.V_RESET_I; since_i = 0; }
        if (j == 0) n_in += nev;
    }
    counts_out[(size_t)blockIdx.x * SG_NE + j] = cnt;
    if (j == 0) in_spikes_out[blockIdx.x] = n_in;
}

/* ------------------------------------------------------------------ host */
struct Data {
    int nimg;
    std::vector<int16_t> inc;     /* NMOD x 784 x 400 */
    std::vector<uint8_t> dly;     /* NMOD x 784 x 400 */
    std::vector<int32_t> thr;     /* NMOD x 400 */
    std::vector<int32_t> pthr;    /* nimg x 784 */
    std::vector<uint32_t> seeds;  /* nimg x 16 */
    std::vector<uint8_t> imgs;    /* nimg x 784 */
    std::vector<uint32_t> streams;
    std::vector<int32_t> inc32[NMOD], thr32[NMOD];
    uint32_t dseed[NMOD];
    sg_consts k;
};

static void rd(const char *dir, const char *name, void *buf, size_t sz, size_t n) {
    char p[1024]; snprintf(p, sizeof p, "%s/%s", dir, name);
    FILE *f = fopen(p, "rb");
    if (!f || fread(buf, sz, n, f) != n) { fprintf(stderr, "cannot read %s\n", p); exit(1); }
    fclose(f);
}

/* images: the 20 golden queries plus extra entries (DGX bench workload: query 0, stream 12345) */
static void load(Data &D, const char *gv, double dt) {
    sg_consts_init(&D.k, dt);
    D.nimg = NQ + 1;
    D.imgs.resize(D.nimg * SG_NIN); D.streams.resize(D.nimg);
    rd(gv, "queries.u8", D.imgs.data(), 1, NQ * SG_NIN);
    rd(gv, "streams.u32", D.streams.data(), 4, NQ);
    memcpy(&D.imgs[NQ * SG_NIN], &D.imgs[0], SG_NIN); D.streams[NQ] = 12345;
    D.inc.resize((size_t)NMOD * SG_NIN * SG_NE); D.dly.resize((size_t)NMOD * SG_NIN * SG_NE); D.thr.resize(NMOD * SG_NE);
    for (int m = 0; m < NMOD; m++) {
        char name[64], p[1024], magic[4]; int32_t ver;
        snprintf(name, sizeof name, "module_%d.bin", m); snprintf(p, sizeof p, "%s/%s", gv, name);
        FILE *f = fopen(p, "rb");
        D.inc32[m].resize(SG_NIN * SG_NE); D.thr32[m].resize(SG_NE);
        if (!f || fread(magic, 1, 4, f) != 4 || fread(&ver, 4, 1, f) != 1 || fread(&D.dseed[m], 4, 1, f) != 1 ||
            fread(D.inc32[m].data(), 4, SG_NIN * SG_NE, f) != SG_NIN * SG_NE || fread(D.thr32[m].data(), 4, SG_NE, f) != SG_NE) {
            fprintf(stderr, "bad %s\n", p); exit(1);
        }
        fclose(f);
        for (int x = 0; x < SG_NIN * SG_NE; x++) {
            int32_t v = D.inc32[m][x];
            if (v < -32768 || v > 32767) { fprintf(stderr, "inc does not fit int16\n"); exit(1); }
            D.inc[(size_t)m * SG_NIN * SG_NE + x] = (int16_t)v;
        }
        memcpy(&D.thr[m * SG_NE], D.thr32[m].data(), SG_NE * 4);
        /* delays from the C model's own initialisation (same hash, same mode) */
        sg_module mod = {D.inc32[m].data(), D.thr32[m].data(), D.dseed[m]};
        sg_engine e;
        if (sg_engine_init(&e, &mod, &D.k, SG_DELAY_1MS_BINNED)) exit(1);
        memcpy(&D.dly[(size_t)m * SG_NIN * SG_NE], e.delay, SG_NIN * SG_NE);
        sg_engine_free(&e);
    }
    D.pthr.resize(D.nimg * SG_NIN); D.seeds.resize(D.nimg * SG_NGEN);
    for (int q = 0; q < D.nimg; q++) {
        sg_pixel_thresholds(&D.imgs[q * SG_NIN], dt, &D.pthr[q * SG_NIN]);
        sg_lfsr_seed(D.streams[q], &D.seeds[q * SG_NGEN]);
    }
}

struct Dev {
    int16_t *inc; uint8_t *dly; int32_t *thr, *pthr; uint32_t *seeds; Run *runs; int32_t *counts; int *nin; int cap;
};

static void upload(Dev &G, const Data &D, int cap) {
    G.cap = cap;
    CK(cudaMalloc(&G.inc, D.inc.size() * 2)); CK(cudaMemcpy(G.inc, D.inc.data(), D.inc.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&G.dly, D.dly.size())); CK(cudaMemcpy(G.dly, D.dly.data(), D.dly.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&G.thr, D.thr.size() * 4)); CK(cudaMemcpy(G.thr, D.thr.data(), D.thr.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&G.pthr, D.pthr.size() * 4)); CK(cudaMemcpy(G.pthr, D.pthr.data(), D.pthr.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&G.seeds, D.seeds.size() * 4)); CK(cudaMemcpy(G.seeds, D.seeds.data(), D.seeds.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&G.runs, cap * sizeof(Run)));
    CK(cudaMalloc(&G.counts, (size_t)cap * SG_NE * 4));
    CK(cudaMalloc(&G.nin, cap * 4));
}

static void launch(Dev &G, const Data &D, int n) {
    run_hard_kernel<<<n, SG_NE>>>(G.inc, G.dly, G.thr, G.pthr, G.seeds, G.runs, D.k, G.counts, G.nin);
}

/* batch of n runs cycling over (module, golden query) */
static std::vector<Run> batch_runs(int n) {
    std::vector<Run> r(n);
    for (int x = 0; x < n; x++) r[x] = {x % NMOD, (x / NMOD) % NQ};
    return r;
}

/* C reference counts for every (module, image) */
static std::vector<int32_t> cpu_reference(const Data &D) {
    std::vector<int32_t> ref((size_t)NMOD * D.nimg * SG_NE);
    for (int m = 0; m < NMOD; m++) {
        sg_module mod = {D.inc32[m].data(), D.thr32[m].data(), D.dseed[m]};
        sg_engine e;
        if (sg_engine_init(&e, &mod, &D.k, SG_DELAY_1MS_BINNED)) exit(1);
        for (int q = 0; q < D.nimg; q++)
            sg_run_hard(&e, &D.imgs[q * SG_NIN], D.streams[q], &ref[((size_t)m * D.nimg + q) * SG_NE], NULL);
        sg_engine_free(&e);
    }
    return ref;
}

static int check(const Data &D, const std::vector<Run> &runs, const int32_t *counts, const std::vector<int32_t> &ref) {
    int bad = 0;
    for (size_t x = 0; x < runs.size(); x++)
        if (memcmp(counts + x * SG_NE, &ref[((size_t)runs[x].mod * D.nimg + runs[x].img) * SG_NE], SG_NE * 4)) bad++;
    return bad;
}

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

/* time n-run batches: kernel time (events) and end-to-end (upload run list, launch, copy counts) */
static void time_batch(Dev &G, const Data &D, const std::vector<Run> &runs, int reps, std::vector<int32_t> &counts,
                       double &kern_med, double &e2e_med, double &kern_min) {
    int n = runs.size();
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    std::vector<double> tk, te;
    counts.resize((size_t)n * SG_NE);
    for (int r = 0; r < reps + 2; r++) {
        double t0 = now_s();
        CK(cudaMemcpy(G.runs, runs.data(), n * sizeof(Run), cudaMemcpyHostToDevice));
        CK(cudaEventRecord(a));
        launch(G, D, n);
        CK(cudaEventRecord(b));
        CK(cudaMemcpy(counts.data(), G.counts, (size_t)n * SG_NE * 4, cudaMemcpyDeviceToHost));
        double t1 = now_s();
        CK(cudaGetLastError());
        float ms; CK(cudaEventElapsedTime(&ms, a, b));
        if (r >= 2) { tk.push_back(ms * 1e-3); te.push_back(t1 - t0); }   /* 2 warm-up launches */
    }
    kern_med = median(tk); e2e_med = median(te); kern_min = *std::min_element(tk.begin(), tk.end());
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s verify|bench|loop <golden_vectors_dir> ...\n", argv[0]); return 2; }
    const char *mode = argv[1], *gv = argv[2];
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    if (!strcmp(mode, "verify")) {
        Data D; load(D, gv, 100e-6);
        std::vector<int32_t> expv(NMOD * NQ * SG_NE);
        rd(gv, "expected.i32", expv.data(), 4, expv.size());
        Dev G; upload(G, D, NMOD * NQ);
        std::vector<Run> runs(NMOD * NQ);
        for (int m = 0; m < NMOD; m++) for (int q = 0; q < NQ; q++) runs[m * NQ + q] = {m, q};
        CK(cudaMemcpy(G.runs, runs.data(), runs.size() * sizeof(Run), cudaMemcpyHostToDevice));
        launch(G, D, runs.size()); CK(cudaGetLastError());
        std::vector<int32_t> c(runs.size() * SG_NE);
        CK(cudaMemcpy(c.data(), G.counts, c.size() * 4, cudaMemcpyDeviceToHost));
        int fails = 0;
        for (int m = 0; m < NMOD; m++) {
            int bad = 0;
            for (int q = 0; q < NQ; q++)
                if (memcmp(&c[(m * NQ + q) * SG_NE], &expv[(m * NQ + q) * SG_NE], SG_NE * 4)) bad++;
            printf("module %d: %s (%d/%d queries bit-exact)\n", m, bad ? "FAIL" : "PASS", NQ - bad, NQ);
            fails += bad;
        }
        printf("%s on %s (sm_%d%d)\n", fails ? "FAIL" : "ALL PASS", prop.name, prop.major, prop.minor);
        return fails ? 1 : 0;
    }
    if (!strcmp(mode, "bench") && argc >= 5) {
        double dt = atof(argv[3]) * 1e-6; int reps = atoi(argv[4]);
        const char *out = argc > 5 ? argv[5] : NULL;
        Data D; load(D, gv, dt);
        std::vector<int32_t> ref = cpu_reference(D);
        const int sizes[] = {1, 4, 8, 16, 24, 48, 96, 192, 480, 960, 1920, 4800};
        Dev G; upload(G, D, 4800);
        FILE *f = out ? fopen(out, "w") : stdout;
        fprintf(f, "{\n \"device\": \"%s\", \"sm\": \"%d%d\", \"sm_count\": %d, \"dt_us\": %s, \"reps\": %d,\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount, argv[3], reps);
        /* DGX bench workload: module 0, query 0, stream 12345 */
        std::vector<Run> one = {{0, NQ}};
        std::vector<int32_t> c; double km, em, kmin;
        time_batch(G, D, one, reps, c, km, em, kmin);
        long tot = 0; for (int j = 0; j < SG_NE; j++) tot += c[j];
        int bad = check(D, one, c.data(), ref);
        fprintf(f, " \"single_run\": {\"workload\": \"module_0, query 0, stream 12345\", \"kernel_median_s\": %.6f, \"kernel_min_s\": %.6f, \"e2e_median_s\": %.6f, \"e_spikes\": %ld, \"bit_exact_vs_c\": %s},\n",
                km, kmin, em, tot, bad ? "false" : "true");
        /* one 4-module query: modules 0-3 on query 0 in parallel */
        std::vector<Run> four = {{0, 0}, {1, 0}, {2, 0}, {3, 0}};
        time_batch(G, D, four, reps, c, km, em, kmin);
        bad = check(D, four, c.data(), ref);
        fprintf(f, " \"query_4_modules\": {\"kernel_median_s\": %.6f, \"kernel_min_s\": %.6f, \"e2e_median_s\": %.6f, \"bit_exact_vs_c\": %s},\n",
                km, kmin, em, bad ? "false" : "true");
        fprintf(f, " \"batches\": [\n");
        int nsz = sizeof(sizes) / sizeof(sizes[0]);
        for (int s = 0; s < nsz; s++) {
            std::vector<Run> runs = batch_runs(sizes[s]);
            int r = sizes[s] >= 960 ? std::max(3, reps / 5) : reps;
            time_batch(G, D, runs, r, c, km, em, kmin);
            bad = check(D, runs, c.data(), ref);
            fprintf(f, "  {\"batch\": %d, \"reps\": %d, \"kernel_median_s\": %.6f, \"e2e_median_s\": %.6f, \"module_runs_per_s\": %.2f, \"bit_exact_runs\": %d, \"mismatched_runs\": %d}%s\n",
                    sizes[s], r, km, em, sizes[s] / km, sizes[s] - bad, bad, s + 1 < nsz ? "," : "");
            fflush(f);
        }
        fprintf(f, " ]\n}\n");
        if (out) fclose(f);
        return 0;
    }
    if (!strcmp(mode, "loop") && argc >= 6) {
        double dt = atof(argv[3]) * 1e-6; int n = atoi(argv[4]); double secs = atof(argv[5]);
        Data D; load(D, gv, dt);
        Dev G; upload(G, D, n);
        std::vector<Run> runs = batch_runs(n);
        CK(cudaMemcpy(G.runs, runs.data(), n * sizeof(Run), cudaMemcpyHostToDevice));
        launch(G, D, n); CK(cudaDeviceSynchronize());   /* warm-up */
        double t0 = now_s(), t1; long launches = 0;
        do { launch(G, D, n); CK(cudaDeviceSynchronize()); launches++; t1 = now_s(); } while (t1 - t0 < secs);
        printf("{\"batch\": %d, \"launches\": %ld, \"module_runs\": %ld, \"wall_s\": %.4f, \"module_runs_per_s\": %.2f, \"t_start\": %.4f, \"t_end\": %.4f}\n",
               n, launches, launches * n, t1 - t0, launches * n / (t1 - t0),
               std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count() - (t1 - t0),
               std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count());
        return 0;
    }
    fprintf(stderr, "bad arguments\n");
    return 2;
}
