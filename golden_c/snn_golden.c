/*
 * SpikeNav fixed-point golden model of one VPRSNN Modular-SNN module (portable C99).
 *
 * Bit-exact reference for the FPGA module engine. Arithmetic, hash, spike generator and
 * event ordering are specified in reports/golden_model_spec.md and mirrored by the
 * Python model snnsim/engine.py (arith='fixed'); tests/test_golden_c.py checks that the
 * two produce identical per-neuron spike counts.
 *
 * Network: VPRSNN test-mode network (https://github.com/QVPR/VPRSNN, MIT License,
 * (c) 2023 Somayeh Hussaini, Michael Milford and Tobias Fischer).
 *
 * Single-threaded, no SIMD intrinsics, no floating point inside the simulation loop
 * (doubles are only used once to derive the integer constants from dt).
 */
#include "snn_golden.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ helpers */
/* arithmetic shift right that is well defined for negative values in C99 */
static inline int64_t asr(int64_t x, int s) { return x >= 0 ? (x >> s) : ~((~x) >> s); }
static inline int64_t rshr(int64_t x, int s) { return asr(x + ((int64_t)1 << (s - 1)), s); }
static inline int64_t clamp64(int64_t x, int64_t lo, int64_t hi) { return x < lo ? lo : (x > hi ? hi : x); }
static inline int64_t rnd_pos(double x) { return (int64_t)floor(x + 0.5); }   /* round half up */

uint32_t sg_mix32(uint32_t h) {
    h += h << 10; h ^= h >> 6; h += h << 3; h ^= h >> 11; h += h << 15;
    return h;
}

uint32_t sg_delay_hash(uint32_t i, uint32_t j, uint32_t seed) {
    return sg_mix32(sg_mix32(((i << 9) | j) ^ seed));
}

static inline uint32_t xorshift32(uint32_t x) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return x;
}

/* ------------------------------------------------------------------ constants */
static int timer_lockout_steps(double dt) {
    double t = 0.0; int k = 0;
    for (;;) { k++; t = t + dt * 0.1; if (t > 5e-3) return k; }
}

void sg_consts_init(sg_consts *k, double dt) {
    const double tc_e = 100e-3, tc_i = 10e-3, tc_ge = 1e-3, tc_gi = 0.5e-3;
    double ke = dt / tc_e, ki = dt / tc_i, dgi;
    memset(k, 0, sizeof(*k));
    k->dt = dt;
    k->V_REST_E = -65 * 65536; k->V_RESET_E = -65 * 65536;
    k->V_REST_I = -60 * 65536; k->V_RESET_I = -45 * 65536;
    k->V_TH_I = -40 * 65536;   k->E_INH_E = -100 * 65536;
    k->V_INIT_E = -105 * 65536; k->V_INIT_I = -100 * 65536;
    k->KV_E = rnd_pos(ke * (double)(1 << SG_SH));
    k->KV_I = rnd_pos(ki * (double)(1 << SG_SH));
    k->KC_E = rnd_pos(ke * (double)(1 << SG_VS) / (double)SG_GU * (double)(1 << SG_SH));
    k->KC_I = rnd_pos(ki * (double)(1 << SG_VS) / (double)SG_GU * (double)(1 << SG_SH));
    k->DGE = rnd_pos((1.0 - dt / tc_ge) * (double)(1 << SG_GSH));
    dgi = 1.0 - dt / tc_gi; if (dgi < 0) dgi = 0;
    k->DGI = rnd_pos(dgi * (double)(1 << SG_GSH));
    k->W_EI = rnd_pos(10.4 * SG_GU);
    k->W_IE = rnd_pos(17.0 * SG_GU);
    k->REF_E = (int)((5e-3 + 1e-3 * dt) / dt);
    k->REF_I = (int)((2e-3 + 1e-3 * dt) / dt);
    k->T_LOCK = timer_lockout_steps(dt);
    k->steps_per_ms = (int)floor(1e-3 / dt + 0.5);
    k->max_delay = (int)floor(10e-3 / dt + 0.5);
    k->n_present = (int)floor(0.35 / dt + 0.5);
    k->n_rest = (int)floor(0.15 / dt + 0.5);
    k->n_init_rest = (int)floor(0.5 / dt + 0.5);
}

/* ------------------------------------------------------------------ engine */
int sg_engine_init(sg_engine *e, const sg_module *m, const sg_consts *k, int delay_mode) {
    int i, j;
    memset(e, 0, sizeof(*e));
    e->m = m; e->k = k; e->delay_mode = delay_mode;
    if (delay_mode == SG_DELAY_1MS_BINNED) e->S = k->max_delay + k->steps_per_ms + 1;
    else e->S = k->max_delay + 1;
    e->buf = (int64_t *)calloc((size_t)e->S * SG_NE, sizeof(int64_t));
    e->delay = (uint8_t *)malloc((size_t)SG_NIN * SG_NE);
    if (!e->buf || !e->delay) return -1;
    /* per-synapse delay parameter, regenerated from the hash (hardware computes it on the fly) */
    for (i = 0; i < SG_NIN; i++)
        for (j = 0; j < SG_NE; j++) {
            uint32_t h16 = sg_delay_hash((uint32_t)i, (uint32_t)j, m->delay_seed) >> 16;
            int d;
            switch (delay_mode) {
            case SG_DELAY_NONE: d = 0; break;
            case SG_DELAY_HASH: d = (int)((h16 * (uint32_t)(k->max_delay + 1)) >> 16); break;
            case SG_DELAY_1MS: d = (int)((h16 * 11u) >> 16) * k->steps_per_ms; break;
            default: d = (int)((h16 * 11u) >> 16); break;   /* binned: whole ms */
            }
            e->delay[i * SG_NE + j] = (uint8_t)d;
        }
    return 0;
}

void sg_engine_free(sg_engine *e) { free(e->buf); free(e->delay); e->buf = NULL; e->delay = NULL; }

void sg_reset(sg_engine *e, int initial) {
    const sg_consts *k = e->k; int j;
    for (j = 0; j < SG_NE; j++) {
        e->ve[j] = initial ? k->V_INIT_E : k->V_REST_E;
        e->vi[j] = initial ? k->V_INIT_I : k->V_REST_I;
        e->gee[j] = e->gie[j] = e->gei[j] = 0;
        e->since_e[j] = e->since_i[j] = SG_BIG;
        e->timer[j] = initial ? 0 : SG_BIG;
    }
    memset(e->buf, 0, (size_t)e->S * SG_NE * sizeof(int64_t));
    e->t = 0;
}

/* One dt step. ev: input pixels spiking in this step (n_ev of them). Adds E spikes to cnt. */
static void sg_step(sg_engine *e, const int *ev, int n_ev, int32_t *cnt, sg_stats *st) {
    const sg_consts *k = e->k; const sg_module *m = e->m;
    int j, q, n_i = 0;
    uint8_t spk_e[SG_NE], spk_i[SG_NE];
    int64_t *slot;
    /* 1. groups: refractoriness, Euler update from old values */
    for (j = 0; j < SG_NE; j++) {
        int64_t ve = e->ve[j], gee = e->gee[j], gie = e->gie[j];
        int64_t vi = e->vi[j], gei = e->gei[j];
        int nre, nri;
        if (e->since_e[j] < SG_BIG) e->since_e[j]++;
        if (e->since_i[j] < SG_BIG) e->since_i[j]++;
        if (e->timer[j] < SG_BIG) e->timer[j]++;
        nre = e->since_e[j] >= k->REF_E;
        nri = e->since_i[j] >= k->REF_I;
        if (nre) {
            int64_t C = gie * asr(k->E_INH_E - ve, SG_VS) - gee * asr(ve, SG_VS);
            int64_t acc = k->KV_E * (k->V_REST_E - ve) + k->KC_E * C;
            e->ve[j] = clamp64(ve + rshr(acc, SG_SH), SG_V_MIN, SG_V_MAX);
        }
        e->gee[j] = asr(gee * k->DGE, SG_GSH);
        e->gie[j] = asr(gie * k->DGI, SG_GSH);
        if (nri) {
            int64_t acc = k->KV_I * (k->V_REST_I - vi) - k->KC_I * (gei * asr(vi, SG_VS));
            e->vi[j] = clamp64(vi + rshr(acc, SG_SH), SG_V_MIN, SG_V_MAX);
        }
        e->gei[j] = asr(gei * k->DGE, SG_GSH);
        /* 2. thresholds */
        spk_i[j] = (uint8_t)((e->vi[j] > k->V_TH_I) && nri);
        spk_e[j] = (uint8_t)((e->ve[j] > m->thr[j]) && (e->timer[j] >= k->T_LOCK) && nre);
        n_i += spk_i[j];
    }
    /* 3. synapses: push input events into the delay ring, deliver the due slot */
    for (q = 0; q < n_ev; q++) {
        int i = ev[q];
        const int32_t *w = &m->inc[i * SG_NE];
        const uint8_t *d = &e->delay[i * SG_NE];
        for (j = 0; j < SG_NE; j++) {
            long arrive;
            if (e->delay_mode == SG_DELAY_1MS_BINNED)
                arrive = (e->t / k->steps_per_ms + 1 + d[j]) * (long)k->steps_per_ms;
            else
                arrive = e->t + d[j];
            e->buf[(arrive % e->S) * SG_NE + j] += w[j];
        }
    }
    slot = &e->buf[(e->t % e->S) * SG_NE];
    for (j = 0; j < SG_NE; j++) {
        e->gee[j] = e->gee[j] + slot[j];
        slot[j] = 0;
        e->gei[j] = e->gei[j] + (spk_e[j] ? k->W_EI : 0);
        if (e->gei[j] > SG_G_MAX) e->gei[j] = SG_G_MAX;
        e->gie[j] = e->gie[j] + k->W_IE * (int64_t)(n_i - spk_i[j]);   /* shared-count inhibition */
        if (e->gie[j] > SG_G_MAX) e->gie[j] = SG_G_MAX;
        if (e->gee[j] > SG_G_MAX) e->gee[j] = SG_G_MAX;
        /* 4. resets */
        if (spk_e[j]) { e->ve[j] = k->V_RESET_E; e->timer[j] = 0; e->since_e[j] = 0; cnt[j]++; }
        if (spk_i[j]) { e->vi[j] = k->V_RESET_I; e->since_i[j] = 0; }
    }
    if (st) {
        for (j = 0; j < SG_NE; j++) { st->e_spikes += spk_e[j]; st->i_spikes += spk_i[j]; }
        st->in_spikes += n_ev; st->steps++;
    }
    e->t++;
}

/* Generate this step's input spikes with the 16 xorshift32 generators. */
static int lfsr_step(uint32_t *state, const int32_t *thr, int *ev) {
    int n = 0, kk, g;
    for (kk = 0; kk < SG_NIN / SG_NGEN; kk++)
        for (g = 0; g < SG_NGEN; g++) {
            state[g] = xorshift32(state[g]);
            if ((int32_t)(state[g] >> (32 - SG_THR_BITS)) < thr[kk * SG_NGEN + g]) ev[n++] = kk * SG_NGEN + g;
        }
    /* events come out generator-major within a pixel block; sort ascending (insertion) */
    for (kk = 1; kk < n; kk++) { int v = ev[kk], p = kk - 1; while (p >= 0 && ev[p] > v) { ev[p + 1] = ev[p]; p--; } ev[p + 1] = v; }
    return n;
}

void sg_lfsr_seed(uint32_t stream, uint32_t *state) {
    uint32_t g;
    for (g = 0; g < SG_NGEN; g++) {
        uint32_t s = sg_mix32(sg_mix32(stream * SG_NGEN + g) ^ 0xA5A5A5A5u);
        state[g] = s ? s : 1u;
    }
}

void sg_pixel_thresholds(const uint8_t *img, double dt, int32_t *thr) {
    int i;
    for (i = 0; i < SG_NIN; i++) {
        double p = (double)img[i] / 4.0 * dt;
        thr[i] = (int32_t)rnd_pos(p * (double)(1 << SG_THR_BITS));
    }
}

/* Present one image for n_present steps (LFSR input) */
static void present(sg_engine *e, const uint8_t *img, uint32_t stream, int32_t *cnt, sg_stats *st) {
    uint32_t state[SG_NGEN]; int32_t thr[SG_NIN]; int ev[SG_NIN]; int s;
    sg_lfsr_seed(stream, state);
    sg_pixel_thresholds(img, e->k->dt, thr);
    for (s = 0; s < e->k->n_present; s++) {
        int n = lfsr_step(state, thr, ev);
        sg_step(e, ev, n, cnt, st);
    }
}

void sg_run_hard(sg_engine *e, const uint8_t *img, uint32_t stream, int32_t *counts, sg_stats *st) {
    memset(counts, 0, SG_NE * sizeof(int32_t));
    sg_reset(e, 0);
    present(e, img, stream, counts, st);
}

void sg_run_carry(sg_engine *e, const uint8_t *imgs, const uint32_t *streams, int nq, int32_t *counts, sg_stats *st) {
    int32_t scratch[SG_NE]; int32_t acc[SG_NE]; int q, s;
    sg_reset(e, 1);
    memset(scratch, 0, sizeof(scratch));
    for (s = 0; s < e->k->n_init_rest; s++) sg_step(e, NULL, 0, scratch, st);
    memset(acc, 0, sizeof(acc));
    for (q = 0; q < nq; q++) {
        present(e, imgs + (size_t)q * SG_NIN, streams[q], acc, st);
        memcpy(counts + (size_t)q * SG_NE, acc, sizeof(acc));
        memset(acc, 0, sizeof(acc));
        for (s = 0; s < e->k->n_rest; s++) sg_step(e, NULL, 0, acc, st);
    }
}

/* Event-list driven presentation (for externally generated trains): ev_step sorted. */
void sg_run_hard_events(sg_engine *e, const int32_t *ev_step, const int32_t *ev_pix, int n_events,
                        int32_t *counts, sg_stats *st) {
    int s, a = 0; int ev[SG_NIN];
    memset(counts, 0, SG_NE * sizeof(int32_t));
    sg_reset(e, 0);
    for (s = 0; s < e->k->n_present; s++) {
        int n = 0;
        while (a < n_events && ev_step[a] == s) ev[n++] = ev_pix[a++];
        sg_step(e, ev, n, counts, st);
    }
}
