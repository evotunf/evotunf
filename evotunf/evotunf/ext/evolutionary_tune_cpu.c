#undef NDEBUG
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <assert.h>
#include "evotunf_ext.h"
#include "common.h"

#define KLEENE_DIENES_IMPL(a, b) fmax(1.f - a, b)
#define LUKASZEWICZ_IMPL(a, b) fmin(1.f, 1.f - a + b)
#define REICHENBACH_IMPL(a, b) (1.f - a + a * b)
#define FODOR_IMPL(a, b) ((a > b) ? fmax(1.f - a, b) : 1.f)
#define GOGUEN_IMPL(a, b)  ((a > 0.f) ? fmin(1.f, b / a) : 1.f)
#define GODEL_IMPL(a, b) ((a > b) ? b : 1.f)
#define ZADEH_IMPL(a, b) fmax(fmin(a, b), 1.f - a)
#define RESCHER_IMPL(a, b) ((float)(a <= b))
#define YAGER_IMPL(a, b) powf(b, a)
#define WILLMOTT_IMPL(a, b) fmin(fmax(1.f - a, b), fmax(fmax(a, 1 - b), fmin(1 - a, b)))
#define DUBOIS_PRADE_IMPL(a, b) ((b == 0.f) ? (1.f - a) : (a == 1.f) ? (b) : 1.f)

#define U(center, sigma, x) expf(-pow((x) - (center), 2)/pow((sigma), 2))

typedef struct GaussParams {
    float center, sigma;
} GaussParams;

typedef struct EvolutionaryParams {
    float sigma1, sigma2;
} EvolutionaryParams;

typedef struct Chromosome {
    enum t_norm t_outer;
    enum t_norm t_inner;
    enum implication impl;
    GaussParams *gauss_params;
    EvolutionaryParams *evolutionary_params;
} Chromosome;

typedef struct Population {
    GaussParams *gauss_params_buf;
    EvolutionaryParams *evolutionary_params_buf;
    Chromosome chromosomes[0];
} Population;

static float evaluate(enum t_norm t_outer_kind, enum t_norm t_inner_kind, enum implication impl_kind, GaussParams *gauss_params,
        unsigned n, unsigned rules_number, const float *xx)
{
    size_t i, j, k;
    double numerator = 0., denominator = 0.;

    for (j = 0; j < rules_number; ++j) {
        GaussParams *jth_rule_params = gauss_params + j * (n+1);
        float t_norm = 1.f;

        for (k = 0; k < rules_number; ++k) {
            GaussParams *kth_rule_params = gauss_params + k * (n+1);
            float antecedent = 1.f;

            switch (t_inner_kind) {
                case PRODUCT: for (i = 0; i < n; ++i) antecedent *= U(kth_rule_params[i].center, kth_rule_params[i].sigma, xx[i]); break;
                case MIN: for (i = 0; i < n; ++i) antecedent = fmin(antecedent, U(kth_rule_params[i].center, kth_rule_params[i].sigma, xx[i])); break;
            }

            float consequent = U(kth_rule_params[n].center, kth_rule_params[n].sigma, jth_rule_params[n].center);

            float impl;
            switch (impl_kind) {
                case KLEENE_DIENES: impl = KLEENE_DIENES_IMPL(antecedent, consequent); break;
                case LUKASZEWICZ: impl = LUKASZEWICZ_IMPL(antecedent, consequent); break;
                case REICHENBACH: impl = REICHENBACH_IMPL(antecedent, consequent); break;
                case FODOR: impl = FODOR_IMPL(antecedent, consequent); break;
                case GOGUEN: impl = GOGUEN_IMPL(antecedent, consequent); break;
                case GODEL: impl = GODEL_IMPL(antecedent, consequent); break;
                case ZADEH: impl = ZADEH_IMPL(antecedent, consequent); break;
                case RESCHER: impl = RESCHER_IMPL(antecedent, consequent); break;
                case YAGER: impl = YAGER_IMPL(antecedent, consequent); break;
                case WILLMOTT: impl = WILLMOTT_IMPL(antecedent, consequent); break;
                case DUBOIS_PRADE: impl = DUBOIS_PRADE_IMPL(antecedent, consequent); break;
            }

            ASSERT_EX(0.f <= impl && impl <= 1.f, printf("%f %u %u\n", impl, j, k));

            switch (t_outer_kind) {
                case PRODUCT: t_norm *= impl; break;
                case MIN: t_norm = fmin(t_norm, impl); break;
            }

            ASSERT_EX(0.f <= t_norm && t_norm <= 1.f, printf("%f %u %u\n", t_norm, j, k));
        }

        numerator += jth_rule_params[n].center * t_norm;
        denominator += t_norm;
    }
    // ASSERT_EX(denominator > 0, printf("%d %d %d\n", t_outer_kind, t_inner_kind, impl_kind));
    return (denominator > 0.f) ? (float) (numerator / denominator) : 0.f;
}

float mse(const Chromosome *chromosome, const float *xxs, const float *ys, unsigned rule_base_len, unsigned n, unsigned N)
{
    size_t i;
    float error = 0.f;

    for (i = 0; i < N; ++i) {
        error += pow(evaluate(chromosome->t_outer, chromosome->t_inner, chromosome->impl, chromosome->gauss_params, n, rule_base_len, xxs + i * n) - ys[i], 2);
        ASSERT_EX(isfinite(error), printf("%f %d %d %d\n", error, chromosome->t_outer, chromosome->t_inner, chromosome->impl));
        // break;
    }

    return -sqrtf(error / N);
}

static Chromosome* allocate_population(unsigned population_power, unsigned rule_base_len, unsigned n)
{
    size_t j;

    Population *population = malloc(sizeof(Population) + sizeof(Chromosome[population_power]));
    population->gauss_params_buf = malloc(sizeof(GaussParams[population_power * rule_base_len * (n+1)]));
    population->evolutionary_params_buf = malloc(sizeof(EvolutionaryParams[population_power * rule_base_len * (n+1)]));

    for (j = 0; j < population_power; ++j) {
        population->chromosomes[j].gauss_params = population->gauss_params_buf + j * rule_base_len * (n+1);
        population->chromosomes[j].evolutionary_params = population->evolutionary_params_buf + j * rule_base_len * (n+1);
    }
    return population->chromosomes;
}

static void destroy_population(Chromosome *chromosomes, unsigned population_power)
{
    Population *population = (Population*) ((void*)chromosomes - offsetof(Population, chromosomes));

    free(population->evolutionary_params_buf);
    free(population->gauss_params_buf);
    free(population);
}

static void initialize_population_with_data_bounds_accounting(
        Chromosome *population, unsigned population_power, unsigned rule_base_len, unsigned n,
        const float *xxs, const float *ys, unsigned N, float h)
{
    size_t i, j, k;

    struct data_bounds {
        float min;
        union { float max, diff, a; }
    } data_bounds[n+1];

    for (i = 0; i < n; ++i) data_bounds[i].min = data_bounds[i].max = xxs[i];
    data_bounds[n].min = data_bounds[n].max = ys[0];

    for (j = 1; j < N; ++j) {
        for (i = 0; i < n; ++i) {
            float val = xxs[j * (n+1) + i];

            if (val < data_bounds[i].min) {
                data_bounds[i].min = val;
            }
            if (val > data_bounds[i].max) {
                data_bounds[i].max = val;
            }
        }

        {
            float val = ys[j];

            if (val < data_bounds[n].min) {
                data_bounds[n].min = val;
            }
            if (val > data_bounds[n].max) {
                data_bounds[n].max = val;
            }
        }
    }

    for (i = 0; i < n+1; ++i) printf("%d = %f, ", i, data_bounds[i].max);
    printf("\n");
    for (i = 0; i < n+1; ++i) data_bounds[i].a = (data_bounds[i].max - data_bounds[i].min) / rule_base_len;

    for (j = 0; j < population_power; ++j) {
        Chromosome *chromosome = population + j;
        chromosome->t_outer = rnd() & 0x1;
        chromosome->t_inner = rnd() & 0x1;
        chromosome->impl = rnd() % 11;
        for (k = 0; k < rule_base_len; ++k) {
            GaussParams *gauss_params = chromosome->gauss_params + k * (n+1);

            for (i = 0; i < n+1; ++i) {
                gauss_params[i].center = data_bounds[i].min + (rnd() % rule_base_len) * data_bounds[i].a + data_bounds[i].a / 2;
                gauss_params[i].sigma = data_bounds[i].a / 2;
            }

            EvolutionaryParams *evolutionary_params = chromosome->evolutionary_params + k * (n+1);

            for (i = 0; i < n+1; ++i) {
                evolutionary_params[i].sigma1 = evolutionary_params[i].sigma2 = h * data_bounds[i].a;
            }
        }
    }
}

static void compute_scores(const Chromosome *population, float *scores, unsigned population_power, unsigned rule_base_len,
        const float *xxs, const float *ys, unsigned n, unsigned N)
{
    size_t j;

#pragma omp parallel for default(shared) schedule(dynamic)
    for (j = 0; j < population_power; ++j) {
        scores[j] = mse(population + j, xxs, ys, rule_base_len, n, N);
    }
}

static size_t get_best_chromosome_idx(const float *scores, unsigned population_power)
{
    size_t j, max_score_idx = 0;
    float max_value = scores[0];

    for (j = 1; j < population_power; ++j) {
        if (scores[j] > max_value) {
            max_value = scores[j];
            max_score_idx = j;
        }
    }
    return max_score_idx;
}

static size_t get_worst_chromosome_idx(const float *scores, unsigned population_power)
{
    size_t j, min_score_idx = 0;
    float min_value = scores[0];

    for (j = 1; j < population_power; ++j) {
        if (scores[j] < min_value) {
            min_value = scores[j];
            min_score_idx = j;
        }
    }
    return min_score_idx;
}

static void perform_reproduction(const Chromosome *population, Chromosome *new_population, const float *scores, unsigned population_power, unsigned new_population_power, unsigned rule_base_len, unsigned n)
{
    size_t i, j;
    float score_sum, min_score;

    min_score = scores[get_worst_chromosome_idx(scores, population_power)];
    float pp_log = logf(population_power);
    for (j = 0; j < population_power; ++j) score_sum += expf(-j / pp_log);
    for (j = 0; j < new_population_power; ++j) {
        // float rnd_score = rnd_prob() * score_sum;
        // i = 0;
        // while (i < population_power) { rnd_score -= expf(-i / pp_log); if (rnd_score > 0.f) ++i; else break; }
        size_t chromosome_idx = (size_t) (rnd_prob() * population_power);
        Chromosome *chromosome = population + chromosome_idx;
        Chromosome *new_chromosome = new_population + j;

        new_chromosome->t_outer = chromosome->t_outer;
        new_chromosome->t_inner = chromosome->t_inner;
        new_chromosome->impl = chromosome->impl;
        memcpy(new_chromosome->gauss_params, chromosome->gauss_params, sizeof(GaussParams[rule_base_len * (n+1)]));
        memcpy(new_chromosome->evolutionary_params, chromosome->evolutionary_params, sizeof(EvolutionaryParams[rule_base_len * (n+1)]));
    }
}

static void perform_mutation(Chromosome *population, unsigned population_power, unsigned rule_base_len, unsigned n, float pm)
{
    const float tau1 = 1.f / sqrtf(2 * (rule_base_len * (n+1) * 2));
    const float tau = 1.f / sqrtf(2 * sqrtf(rule_base_len * (n+1) * 2));
    size_t i, j, k;

    printf("sigma1 = %f, sigma2 = %f\n", population->evolutionary_params->sigma1, population->evolutionary_params->sigma2);
    for (j = 0; j < population_power; ++j) {
        // if (rnd_prob() > pm) continue;

        Chromosome *chromosome = population + j;

        float p = gauss_noise(0.f, 1.f);
        if (rnd_prob() <= pm) chromosome->t_outer = rnd() & 0x1;
        if (rnd_prob() <= pm) chromosome->t_inner = rnd() & 0x1;
        if (rnd_prob() <= pm) chromosome->impl = (chromosome->impl + 1) % 11;
        for (k = 0; k < rule_base_len; ++k) {
            for (i = 0; i < n+1; ++i) {
                EvolutionaryParams *evolutionary_params = chromosome->evolutionary_params + k * (n+1) + i;
                evolutionary_params->sigma1 *= expf(tau1 * p + tau * gauss_noise(0.f, 1.f));
                evolutionary_params->sigma2 *= expf(tau1 * p + tau * gauss_noise(0.f, 1.f));

                GaussParams *gauss_params = chromosome->gauss_params + k * (n+1) + i;
                gauss_params->center += evolutionary_params->sigma1 * gauss_noise(0.f, 1.f);
                gauss_params->sigma += evolutionary_params->sigma2 * gauss_noise(0.f, 1.f);
            }
        }
    }
#undef RND_11
}

static void perform_selection(/* In[new_population,new_scores,etc], Out[population,scores] */
        Chromosome *population, Chromosome *new_population, float *scores, float *new_scores,
        unsigned population_power, unsigned new_population_power, unsigned rule_base_len, unsigned n)
{
    size_t i, j;

    for (j = 0; j < population_power; ++j) {
        size_t best_new_chromosome_idx = get_best_chromosome_idx(new_scores, new_population_power);

        // SWAP(scores[j], new_scores[best_new_chromosome_idx]);
        scores[j] = new_scores[best_new_chromosome_idx];
        new_scores[best_new_chromosome_idx] = -INFINITY;

        Chromosome *chromosome = population + j;
        Chromosome *new_chromosome = new_population + best_new_chromosome_idx;
        chromosome->t_outer = new_chromosome->t_outer;
        chromosome->t_inner = new_chromosome->t_inner;
        chromosome->impl = new_chromosome->impl;
        memcpy(chromosome->gauss_params, new_chromosome->gauss_params, sizeof(GaussParams[rule_base_len * (n+1)]));
        memcpy(chromosome->evolutionary_params, new_chromosome->evolutionary_params, sizeof(EvolutionaryParams[rule_base_len * (n+1)]));
    }
}

void perform_evolutionary_tune_lfs(const float *xxs, const float *ys, unsigned n, unsigned N, unsigned rule_base_len, float mu, float lambda, LfsConfig *lfs_config)
{
    assert(lambda >= mu);

    size_t i, j, k;

    unsigned population_power = mu, new_population_power = lambda;
    Chromosome *population = allocate_population(population_power, rule_base_len, n);
    Chromosome *new_population = allocate_population(new_population_power, rule_base_len, n);
    float scores[population_power], new_scores[new_population_power];

    initialize_population_with_data_bounds_accounting(population, population_power, rule_base_len, n, xxs, ys, N, 1.f);
    compute_scores(population, scores, population_power, rule_base_len, xxs, ys, n, N);

    float avg;
    unsigned it;
    FILE* f_report = fopen("report.txt", "w");
    for (it = 0; it < 100; ++it) {
        printf("It [%3u] ", it);
        avg = 0.f;
        for (j = 0; j < population_power; ++j) {
            avg += scores[j];
            printf("%.0f ", scores[j]);
        }
        avg /= population_power;
        fprintf(f_report, "%u,%f\n", it, avg);
        printf("Avg: %.0f", avg);
        printf("\n");
        if (0)
        for (j = 0; j < population_power; ++j) {
            Chromosome *chromosome = population + j;
            printf("[Chromosome %3u] t_outer = %d, t_inner = %d, impl = %d\n", j, chromosome->t_outer, chromosome->t_inner, chromosome->impl);
            if (0)
            for (k = 0; k < rule_base_len; ++k) {
                for (i = 0; i < n+1; ++i) {
                    printf("{%f %.2f} ", chromosome->gauss_params[k * (n+1) + i].center, chromosome->gauss_params[k * (n+1) + i].sigma);
                }
                printf("\n");
            }
        }

        // Check for stop condition.
        for (j = 0; j < population_power; ++j) {
            if (scores[j] >= 0.95f) break;
        }

        perform_reproduction(population, new_population, scores, population_power, new_population_power, rule_base_len, n);
        perform_mutation(new_population, new_population_power, rule_base_len, n, 0.05);
        compute_scores(new_population, new_scores, new_population_power, rule_base_len, xxs, ys, n, N);
        if (0) {
            printf("NewIt [%3u] ", it);
            for (j = 0; j < population_power; ++j) {
                printf("%.1f ", new_scores[j]);
            }
            printf("\n");
        }
        perform_selection(population, new_population, scores, new_scores, population_power, new_population_power, rule_base_len, n);
    }
    fclose(f_report);

    size_t best_chromosome_idx = get_best_chromosome_idx(scores, population_power);
    Chromosome *best_chromosome = population + best_chromosome_idx;

    lfs_config->t_outer = best_chromosome->t_outer;
    lfs_config->t_inner = best_chromosome->t_inner;
    lfs_config->impl = best_chromosome->impl;
    memcpy(lfs_config->gauss_params, best_chromosome->gauss_params, sizeof(GaussParams[rule_base_len * (n+1)]));

    destroy_population(population, population_power);
    destroy_population(new_population, population_power);
}

LfsConfig train_lfs_cpu_impl(float *xx_train, float *y_train, unsigned n, unsigned N, float *rule_base, unsigned rule_base_len)
{
    LfsConfig lfs_config;

    lfs_config.gauss_params = rule_base;
    // perform_evolutionary_tune_lfs(xx_train, y_train, n, N, rule_base_len, 300, 1000, &lfs_config);
    perform_evolutionary_tune_lfs(xx_train, y_train, n, N, rule_base_len, 100, 500, &lfs_config);
    return lfs_config;
}

void infer_lfs_cpu_impl(LfsConfig lfs_config, const float *xxs, float *ys, unsigned n, unsigned N)
{
    size_t j;

    for (j = 0; j < N; ++j) {
        ys[j] = evaluate(lfs_config.t_outer, lfs_config.t_inner, lfs_config.impl, (GaussParams*)(void*)lfs_config.gauss_params,
                    n, lfs_config.rule_base_len, xxs + j * n);
    }
}
