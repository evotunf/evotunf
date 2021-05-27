#ifndef EVOTUNF_EVOLUTIONARY_TUNE_H
#define EVOTUNF_EVOLUTIONARY_TUNE_H

#define IMPL(a, b) LUKASZEWICZ_IMPL(a, b)

typedef struct GaussParams {
    float mu, sigma;
} GaussParams;

#ifdef __cplusplus
extern "C" {
#endif

void predict_cpu_impl(
											const unsigned *fset_lens, const GaussParams *fsets, unsigned n,
											const signed char *rules, unsigned rules_len,
											const GaussParams *uxxs, unsigned *ys, unsigned N);

void tune_lfs_cpu_impl(
											 const unsigned *fset_lens, unsigned n, unsigned rules_len,
											 const GaussParams *uxxs, const unsigned *ys, unsigned N,
											 unsigned population_power, unsigned iterations_number,
											 GaussParams *fsets, signed char *rules);

void predict_gpu_impl(const unsigned *fset_lens, const GaussParams *fsets, unsigned n,
											const signed char *rules, unsigned rules_len,
											const GaussParams *uxxs, unsigned *ys, unsigned N);

void tune_lfs_gpu_impl(
											 const unsigned *fset_lens, unsigned n, unsigned rules_len,
											 const GaussParams *uxxs, const unsigned *ys, unsigned N,
											 unsigned population_power, unsigned iterations_number,
											 GaussParams *fsets, signed char *rules);

#ifdef __cplusplus
}
#endif

#endif // EVOTUNF_EVOLUTIONARY_TUNE_H
