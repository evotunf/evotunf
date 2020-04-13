#undef NDEBUG
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include "evotunf_ext.h"
#include "common.h"
#include "random.h"


#define LUKASZEWICZ_IMPL(a, b) fmin(1.f, 1.f - (a) + (b))
#define IMPL(...) LUKASZEWICZ_IMPL(__VA_ARGS__)
#define GAUSS(x, mu, sigma) (expf(-pow(x - mu, 2)/sigma) / sqrtf(2 * 3.14 * sigma))

float minf(float x, float y) { return (x < y) ? x : y; }

static float evaluate(
        const unsigned *fsets_offsets, const GaussParams *gauss_params,
        const unsigned char *rules, unsigned rules_len, unsigned n, const GaussParams *xx)
{
    unsigned i, j, k;
    float numerator = 0.f, denominator = 0.f;

    for (k = 0; k < rules_len; ++k) {
        float y_center = gauss_params[fsets_offsets[n] + rules[k * (n+1) + n]].mu;
        float cross = 1.f;

        for (j = 0; j < rules_len; ++j) {
            float max = 0.f;

            for (i = 0; i < n; ++i) {
                const GaussParams ux = xx[i];
                const GaussParams ua = gauss_params[fsets_offsets[i] + rules[j * (n+1) + i]];
                const GaussParams ub = gauss_params[fsets_offsets[n] + rules[j * (n+1) + n]];
                float x, sup = 0.f;

                for (x = 0.f; x <= 1.f; x += 0.1f) {
                    float impl = IMPL(GAUSS(ua.mu, ua.sigma, x), GAUSS(ub.mu, ub.sigma, y_center));
                    float t_norm = fminf(GAUSS(ux.mu, ux.sigma, x), impl);
                    if (t_norm > sup) sup = t_norm;
                }

                if (sup > max) max = sup;
            }

            if (max < cross) cross = max;
        }

        numerator += y_center * cross;
        denominator += cross;
    }

    return (denominator) ? numerator / denominator : 0.f;
}

void predict_cpu_impl(const unsigned *fsets_lens, const GaussParams *gauss_params, const unsigned char *rules, unsigned rules_len, unsigned n, const GaussParams *xxs, unsigned *ys, unsigned N)
{
    unsigned i, k;
    unsigned fsets_offsets[n+1], offset = 0;

    for (i = 0; i < n+1; ++i) {
        fsets_offsets[i] = offset;
        offset += fsets_lens[i];
    }

    for (k = 0; k < N; ++k) {
        ys[k] = evaluate(fsets_offsets, gauss_params, rules, rules_len, n, xxs + k);
    }
}

typedef struct Chromosome {
    GaussParams *gauss_params;
    EvolutionaryParams *evolutionary_params;
    unsigned char *rules;
} Chromosome;

typedef struct Population {
    GaussParams *gauss_params_buf;
    EvolutionaryParams *evolutionary_params_buf;
    unsigned char *rules_buf;
    Chromosome chromosomes[0];
} Population;

static Chromosome* allocate_population(unsigned population_power, unsigned fsets_total_len, unsigned rules_len, unsigned n)
{
    unsigned j;

    Population *population = (Population*)malloc(sizeof(Population) + sizeof(Chromosome[population_power]));
    GaussParams *gauss_params_buf = population->gauss_params_buf
        = (GaussParams*)malloc(population_power * sizeof(GaussParams[fsets_total_len]));
    EvolutionaryParams *evolutionary_params_buf = population->evolutionary_params_buf
        = (EvolutionaryParams*)malloc(population_power * sizeof(EvolutionaryParams[fsets_total_len]));
    unsigned char *rules_buf = population->rules_buf
        = (unsigned char*)malloc(population_power * sizeof(unsigned char[rules_len * (n+1)]));

    for (j = 0; j < population_power; ++j) {
        Chromosome *ch = population->chromosomes + j;
        ch->gauss_params = gauss_params_buf + j * fsets_total_len;
        ch->evolutionary_params = evolutionary_params_buf + j * fsets_total_len;
        ch->rules = rules_buf + j * (rules_len * (n+1));
    }

    return population->chromosomes;
}

static void destroy_population(Chromosome *chromosomes)
{
    Population *population = (Population*)((char*)chromosomes - offsetof(Population, chromosomes));

    free(population->evolutionary_params_buf);
    free(population->gauss_params_buf);
    free(population);
}

static void initialize_population(
        DataBounds *data_bounds, const unsigned *fsets_lens, Chromosome *population, unsigned population_power,
        unsigned rules_len, unsigned n, float h)
{
    unsigned i, j, k, offset;

    for (j = 0; j < population_power; ++j) {
        Chromosome *ch = population + j;

        offset = 0;
        for (i = 0; i < n+1; ++i) {
            unsigned fsets_len = fsets_lens[i];

            for (k = 0; k < fsets_len; ++k) {
                GaussParams *gp = ch->gauss_params + offset + k;
                gp->mu = (rnd() % fsets_len + 0.5f) / fsets_len; // data_bounds[i].min + (rnd() % fsets_lens[i]) * data_bounds[i].a + data_bounds[i].a / 2.f;
                gp->sigma = 0.5f / fsets_len; // data_bounds[i].a / 2.f;

                EvolutionaryParams *ep = ch->evolutionary_params + offset + k;
                ep->sigma1 = ep->sigma2 = h / fsets_len; // data_bounds[i].a;
            }
            offset += fsets_len;
        }

        for (k = 0; k < rules_len; ++k) {
            for (i = 0; i < n+1; ++i) {
                ch->rules[k * (n+1) + i] = rnd() % fsets_lens[i];
            }
        }
    }
}

static float compute_mse(
        const unsigned *fsets_offsets, const Chromosome *chromosome, unsigned rules_len, unsigned n,
        const GaussParams *xxs, const unsigned *ys, unsigned N)
{
    unsigned k;
    float score = 0.f;

    for (k = 0; k < N; ++k) {
        score += powf(ys[k] - evaluate(fsets_offsets, chromosome->gauss_params, chromosome->rules, rules_len, n, xxs + k * n), 2.f);
    }

    return sqrtf(score / N);
}

static void compute_scores(
        const unsigned *fsets_offsets, const Chromosome *population, unsigned population_power, unsigned rules_len, unsigned n,
        const GaussParams *xxs, const unsigned *ys, unsigned N, float *scores)
{
    unsigned j;

#pragma omp parallel for
    for (j = 0; j < population_power; ++j) {
        scores[j] = -compute_mse(fsets_offsets, population + j, rules_len, n, xxs, ys, N);
    }
}

static void perform_reproduction(
        Chromosome *new_population, const Chromosome *population, unsigned new_population_power, unsigned population_power,
        unsigned fsets_total_len, unsigned rules_len, unsigned n)
{
    unsigned j;

    for (j = 0; j < new_population_power; ++j) {
        Chromosome *new_ch = new_population + j;
        Chromosome *ch = population + (rnd() % population_power);

        memcpy(new_ch->gauss_params, ch->gauss_params, sizeof(GaussParams[fsets_total_len]));
        memcpy(new_ch->evolutionary_params, ch->evolutionary_params, sizeof(EvolutionaryParams[fsets_total_len]));
        memcpy(new_ch->rules, ch->rules, sizeof(unsigned char[rules_len * (n+1)]));
    }
}

static unsigned get_best_chromosome_idx(const float *scores, unsigned population_power)
{
    unsigned j, max_index = 0;
    float max_value = scores[0];

    for (j = 1; j < population_power; ++j) {
        if (scores[j] > max_value) {
            max_value = scores[j];
            max_index = j;
        }
    }
    return max_index;
}

static void perform_selection(
        float *scores, float *new_scores,
        Chromosome *population, const Chromosome *new_population, unsigned population_power, unsigned new_population_power,
        unsigned fsets_total_len, unsigned rules_len, unsigned n)
{
    unsigned j;

    for (j = 0; j < population_power; ++j) {
        unsigned best_new_chromosome_idx = get_best_chromosome_idx(new_scores, new_population_power);

        scores[j] = new_scores[best_new_chromosome_idx];
        new_scores[best_new_chromosome_idx] = -INFINITY;

        Chromosome *new_ch = new_population + best_new_chromosome_idx;
        Chromosome *ch = population + j;
        memcpy(ch->gauss_params, new_ch->gauss_params, sizeof(GaussParams[fsets_total_len]));
        memcpy(ch->evolutionary_params, new_ch->evolutionary_params, sizeof(EvolutionaryParams[fsets_total_len]));
        memcpy(ch->rules, new_ch->rules, sizeof(unsigned char[rules_len * (n+1)]));
        { unsigned i; for (i = 0; i < fsets_total_len; ++i) printf("%f %f; ", ch->gauss_params[i].mu, ch->gauss_params[i].sigma); printf("\n"); }
        { unsigned i; for (i = 0; i < n+1; ++i) printf("%u; ", ch->rules[i]); printf("\n"); }
    }
}

static void perform_mutation(
        const unsigned *fsets_lens,
        Chromosome *new_population, unsigned new_population_power, unsigned fsets_total_len, unsigned rules_len, unsigned n, float pm)
{
    unsigned i, j, k;

    for (j = 0; j < new_population_power; ++j) {
        Chromosome *ch = new_population + j;

        {
            unsigned chromosome_len = fsets_total_len * 2;
            float tau1 = 1.f / sqrtf(2.f * chromosome_len);
            float tau = 1.f / sqrtf(2.f * sqrtf(chromosome_len));
            float p = gauss_noise(0.f, 1.f);

            for (i = 0; i < fsets_total_len; ++i) {
                EvolutionaryParams *ep = ch->evolutionary_params + i;
                float sigma1 = ep->sigma1 *= expf(tau1 * p + tau * gauss_noise(0.f, 1.f));
                float sigma2 = ep->sigma2 *= expf(tau1 * p + tau * gauss_noise(0.f, 1.f));

                GaussParams *gp = ch->gauss_params + i;
                gp->mu += sigma1 * gauss_noise(0.f, 1.f);
                gp->sigma += sigma2 * gauss_noise(0.f, 1.f);
            }
        }

        {
            for (k = 0; k < rules_len; ++k) {
                for (i = 0; i < n+1; ++i) {
                    if (rnd_prob() <= pm) ch->rules[k * (n+1) + i] = rnd() % fsets_lens[i];
                }
            }
        }
    }
}

void tune_lfs_cpu_impl(
        const unsigned *fsets_lens, unsigned rules_len, unsigned n,
        const GaussParams *xxs, const unsigned *ys, unsigned N,
        unsigned mu, unsigned lamda, unsigned it_number, GaussParams *fsets_table, unsigned char *rules)
{
    unsigned i;
    unsigned fsets_offsets[n+1], fsets_total_len = 0, offset = 0;
    DataBounds data_bounds[n+1];

    compute_data_bounds(xxs, ys, data_bounds, N, n);
    for (i = 0; i < n+1; ++i) {
        assert(fsets_lens[i] > 0);
        fsets_offsets[i] = offset;
        offset += fsets_lens[i];
        fsets_total_len += fsets_lens[i];
        data_bounds[i].a = (data_bounds[i].max - data_bounds[i].min) / fsets_lens[i];
    }

    unsigned population_power = mu;
    unsigned new_population_power = lamda;
    unsigned it;
    Chromosome *population = allocate_population(population_power, fsets_total_len, rules_len, n);
    Chromosome *new_population = allocate_population(new_population_power, fsets_total_len, rules_len, n);
    float scores[population_power], new_scores[new_population_power];

    initialize_population(data_bounds, fsets_lens, population, population_power, rules_len, n, 0.1f);

    FILE *f = fopen("report_cpu.txt", "w");
    for (it = 0; it < it_number; ++it) {
        perform_reproduction(new_population, population, new_population_power, population_power, fsets_total_len, rules_len, n);
        perform_mutation(fsets_lens, new_population, new_population_power, fsets_total_len, rules_len, n, 0.25f / (rules_len * (n+1)));
        compute_scores(fsets_offsets, new_population, new_population_power, rules_len, n, xxs, ys, N, new_scores);
        perform_selection(scores, new_scores, population, new_population, population_power, new_population_power, fsets_total_len, rules_len, n);

        {
            unsigned j;
            float avg = 0.f;

            printf("It [%3d] ", it);
            for (j = 0; j < population_power; ++j) {
                printf("%f ", scores[j]);
                avg += scores[j];
            }
            avg /= population_power;
            printf("Avg: %f\n", avg);
            fprintf(f, "%f\n", avg);
        }
    }
    fclose(f);

    memcpy(fsets_table, population->gauss_params, sizeof(GaussParams[fsets_total_len]));
    memcpy(rules, population->rules, sizeof(unsigned char[rules_len * (n+1)]));

    destroy_population(new_population);
    destroy_population(population);
}
