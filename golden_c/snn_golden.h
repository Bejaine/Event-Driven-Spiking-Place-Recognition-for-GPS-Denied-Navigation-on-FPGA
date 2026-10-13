/* SpikeNav fixed-point golden model — public interface. See snn_golden.c. */
#ifndef SNN_GOLDEN_H
#define SNN_GOLDEN_H
#include <stdint.h>

#define SG_NIN 784
#define SG_NE 400
#define SG_GU 4080            /* conductance units per 1.0 */
#define SG_VS 7               /* operand shift for g x v products */
#define SG_SH 30              /* Euler accumulator shift */
#define SG_GSH 16             /* decay factor fractional bits */
#define SG_V_MIN (-(1LL << 24))
#define SG_V_MAX ((1LL << 24) - 1)
#define SG_G_MAX ((1LL << 23) - 1)
#define SG_BIG (1 << 30)      /* "long ago" counter value (saturating) */
#define SG_NGEN 16            /* xorshift32 generators */
#define SG_THR_BITS 24

enum { SG_DELAY_NONE = 0, SG_DELAY_HASH = 1, SG_DELAY_1MS = 2, SG_DELAY_1MS_BINNED = 3 };

typedef struct {
    double dt;
    int64_t V_REST_E, V_RESET_E, V_REST_I, V_RESET_I, V_TH_I, E_INH_E, V_INIT_E, V_INIT_I;
    int64_t KV_E, KV_I, KC_E, KC_I, DGE, DGI, W_EI, W_IE;
    int REF_E, REF_I, T_LOCK, steps_per_ms, max_delay, n_present, n_rest, n_init_rest;
} sg_consts;

typedef struct {
    const int32_t *inc;       /* [784*400] ge increment per synapse (LSB 1/4080), = code*16 for 8-bit */
    const int32_t *thr;       /* [400] E threshold in v LSBs (2^-16 mV) */
    uint32_t delay_seed;
} sg_module;

typedef struct {
    const sg_module *m; const sg_consts *k; int delay_mode; int S;
    int64_t ve[SG_NE], vi[SG_NE], gee[SG_NE], gie[SG_NE], gei[SG_NE];
    int32_t since_e[SG_NE], since_i[SG_NE], timer[SG_NE];
    int64_t *buf; uint8_t *delay; long t;
} sg_engine;

typedef struct { int64_t e_spikes, i_spikes, in_spikes, steps; } sg_stats;

uint32_t sg_mix32(uint32_t h);
uint32_t sg_delay_hash(uint32_t i, uint32_t j, uint32_t seed);
void sg_consts_init(sg_consts *k, double dt);
int sg_engine_init(sg_engine *e, const sg_module *m, const sg_consts *k, int delay_mode);
void sg_engine_free(sg_engine *e);
void sg_reset(sg_engine *e, int initial);
void sg_lfsr_seed(uint32_t stream, uint32_t *state);
void sg_pixel_thresholds(const uint8_t *img, double dt, int32_t *thr);
void sg_run_hard(sg_engine *e, const uint8_t *img, uint32_t stream, int32_t *counts, sg_stats *st);
void sg_run_carry(sg_engine *e, const uint8_t *imgs, const uint32_t *streams, int nq, int32_t *counts, sg_stats *st);
void sg_run_hard_events(sg_engine *e, const int32_t *ev_step, const int32_t *ev_pix, int n_events,
                        int32_t *counts, sg_stats *st);
#endif
