#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <assert.h>
#include "evotunf_ext.h"
#include "common.h"

#define WARP_SIZE 32

// #ifndef NDEBUG
#define CUDA_CALL(x) do { cudaError_t e; if ((e = (x)) != cudaSuccess) { \
    fprintf(stderr, "CUDA Error at %s:%d %s { %s }\n", __FILE__, __LINE__, cudaGetErrorString(e), #x); \
    switch (e) { \
        case cudaErrorInvalidValue: fprintf(stderr, "InvalidValue\n"); break; \
        case cudaErrorMemoryAllocation: fprintf(stderr, "MemoryAllocation\n"); break; \
        case cudaErrorHostMemoryAlreadyRegistered: fprintf(stderr, "HostMemoryAlreadyRegistered\n"); break; \
        case cudaErrorNotSupported: fprintf(stderr, "NotSupported\n"); break; \
    } \
    exit(EXIT_FAILURE); } } while (0)
// #else
// #define CUDA_CALL(x) (x)
// #endif

static cudaDeviceProp device_props;

static void init_device_props()
{
    static int inited;
    int device;

    if (inited) return;

    inited = 1;
    CUDA_CALL(cudaGetDevice(&device));
    CUDA_CALL(cudaGetDeviceProperties(&device_props, device));
}

#define KLEENE_DIENES_IMPL(a, b) fmax(1.f - (a), (b))
#define LUKASZEWICZ_IMPL(a, b) fmin(1.f, 1.f - (a) + (b))
#define REICHENBACH_IMPL(a, b) (1.f - (a) + (a) * (b))
#define FODOR_IMPL(a, b) ((a > b) ? fmax(1.f - (a), (b)) : 1.f)
#define GOGUEN_IMPL(a, b)  ((a > 0.f) ? fmin(1.f, (b) / (a)) : 1.f)
#define GODEL_IMPL(a, b) (((a) > (b)) ? (b) : 1.f)
#define ZADEH_IMPL(a, b) fmax(fmin((a), (b)), 1.f - (a))
#define RESCHER_IMPL(a, b) ((float)((a) <= (b)))
#define YAGER_IMPL(a, b) powf(b, a)
#define WILLMOTT_IMPL(a, b) fmin(fmax(1.f - (a), (b)), fmax(fmax((a), 1 - (b)), fmin(1 - (a), (b))))
#define DUBOIS_PRADE_IMPL(a, b) (((b) == 0.f) ? (1.f - (a)) : ((a) == 1.f) ? (b) : 1.f)
#define ALIEW_IMPL(a, b) (((a) > (b)) ? fmin(1.f - (a), (b)) : 1.f)

#define IMPL(a, b) ALIEW_IMPL(a, b)
#define GAUSS(mu, sigma, x) expf(-powf(((x) - (mu)) / (sigma), 2))

typedef struct {
    float numerator;
    float denominator;
} Fraction;

__device__ float evaluate_kernel(
        const unsigned *fsets_offsets, unsigned y_fsets_len, const GaussParams *gauss_params,
        const unsigned char *rules, const GaussParams *xx)
{
    unsigned step, i, j;
    unsigned rules_len = blockDim.y;
    unsigned n = blockDim.x;
    unsigned rule_idx = threadIdx.y;
    unsigned attr_idx = threadIdx.x;
    unsigned executor_idx = threadIdx.z;

    extern __shared__ char cache[];
    const size_t cache_sz_per_executor = ALIGN_UP(rules_len * sizeof(Fraction) + rules_len * n * sizeof(float), sizeof(float[WARP_SIZE]));
    Fraction *res_cache = (Fraction*)(cache + executor_idx * cache_sz_per_executor);
    float *impl_cache = (float*)(cache + executor_idx * cache_sz_per_executor + rules_len * sizeof(Fraction));


    {
        GaussParams ux = xx[attr_idx];
        float y_center = gauss_params[fsets_offsets[n] + rules[rule_idx * (n+1) + n]].mu;

        float cross = 1.f;

        for (j = 0; j < rules_len; ++j) {
            GaussParams ua = gauss_params[fsets_offsets[attr_idx] + rules[j * (n+1) + attr_idx]];
            GaussParams ub = gauss_params[fsets_offsets[n] + rules[j * (n+1) + n]];
            float x, sup = 0.f;

            for (x = 0.f; x <= 1.01f; x += 0.1f) {
                float impl = IMPL(GAUSS(ua.mu, ua.sigma, x), GAUSS(ub.mu, ub.sigma, y_center));
                float t_norm = fmin(GAUSS(ux.mu, ux.sigma, x), impl);
                if (t_norm > sup) sup = t_norm;
            }
            impl_cache[rule_idx * n + attr_idx] = sup;

            float *impl_cache_row = impl_cache + rule_idx * n;
            for (step = 1; step < n; step <<= 1) {
                if (!(attr_idx & ((step<<1)-1)) && attr_idx + step < n) {
                    if (impl_cache_row[attr_idx + step] > impl_cache_row[attr_idx]) {
                        impl_cache_row[attr_idx] = impl_cache_row[attr_idx + step];
                    }
                }
                __syncthreads();
            }

            if (threadIdx.x == 0) {
                if (impl_cache[rule_idx * n] < cross) cross = impl_cache[rule_idx * n];
                // cross *= impl_cache[j * n];
            }
        }

        if (threadIdx.x == 0) {
            res_cache[rule_idx].numerator = y_center * cross;
            res_cache[rule_idx].denominator = cross;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (step = 1; step < rules_len; step <<= 1) {
            if (!(rule_idx & ((step<<1)-1)) && rule_idx + step < rules_len) {
                res_cache[rule_idx].numerator += res_cache[rule_idx + step].numerator;
                res_cache[rule_idx].denominator += res_cache[rule_idx + step].denominator;
            }
            __syncthreads();
        }

        float y = (res_cache[0].denominator) ? res_cache[0].numerator / res_cache[0].denominator : 0.f;
        float res = 0.f;

        for (i = 0; i < y_fsets_len; ++i) {
            GaussParams uy = gauss_params[fsets_offsets[n] + i];
            res += i * GAUSS(uy.mu, uy.sigma, y);
        }
        return res;
    }
}


__global__ void predict_kernel(
        // enum t_norm t_outer, enum t_norm t_inner, enum impl impl,
        const unsigned *fsets_offsets, unsigned y_fsets_len, const GaussParams *gauss_params, const unsigned char *rules,
        const GaussParams *xxs, size_t xxs_pitch, unsigned *ys)
{
    unsigned idx = blockIdx.x;
    float pred = evaluate_kernel(fsets_offsets, y_fsets_len, gauss_params, rules, (GaussParams*)((char*)xxs + idx * xxs_pitch));
    // if (threadIdx.x == 0 && threadIdx.y == 0) printf("%f\n", pred);
    ys[idx] = (unsigned)lrintf(pred);
}


extern "C"
void predict_gpu_impl(
        // enum t_norm t_outer, enum t_norm t_inner, enum impl impl,
        const unsigned *fsets_lens, const GaussParams *gauss_params, const unsigned char *rules, unsigned rules_len, unsigned n,
        const GaussParams *xxs, unsigned *ys, unsigned N)
{
    /*
     * GPU memory layout
     * +--------------------------
     * | fsets_offsets
     * +--------------------------
     * | GaussParams array [input * fsets_len[input]]
     * +--------------------------
     * | Rules array
     * +--------------------------
     * | Input array
     * +--------------------------
     * | Output buffer
     * +--------------------------
     */

    unsigned i, fsets_total_len = 0, offset = 0;
    unsigned fsets_offsets[n+1];

    for (i = 0; i < n+1; ++i) {
        fsets_offsets[i] = offset;
        offset += fsets_lens[i];
        fsets_total_len += fsets_lens[i];
    }

    size_t xxs_d_pitch;
    unsigned *fsets_offsets_d;
    GaussParams *gauss_params_d, *xxs_d;
    unsigned char *rules_d;
    unsigned *ys_d;

    CUDA_CALL(cudaMalloc(&fsets_offsets_d, sizeof(unsigned[n+1])));
    CUDA_CALL(cudaMalloc(&gauss_params_d, sizeof(GaussParams[fsets_total_len])));
    CUDA_CALL(cudaMalloc(&rules_d, sizeof(unsigned char[rules_len][n+1])));
    CUDA_CALL(cudaMallocPitch(&xxs_d, &xxs_d_pitch, sizeof(GaussParams[n]), N));
    CUDA_CALL(cudaMalloc(&ys_d, sizeof(unsigned[N])));

    cudaMemcpy(fsets_offsets_d, fsets_offsets, sizeof(unsigned[n+1]), cudaMemcpyHostToDevice);
    cudaMemcpy(gauss_params_d, gauss_params, sizeof(GaussParams[fsets_total_len]), cudaMemcpyHostToDevice);
    cudaMemcpy(rules_d, rules, sizeof(unsigned char[rules_len][n+1]), cudaMemcpyHostToDevice);
    cudaMemcpy2D(xxs_d, xxs_d_pitch, xxs, sizeof(GaussParams[n]), sizeof(GaussParams[n]), N, cudaMemcpyHostToDevice);

    {
        size_t shared_sz = sizeof(Fraction[rules_len]) + sizeof(float[rules_len][n]);
        predict_kernel<<<N, dim3(n, rules_len), shared_sz>>>(fsets_offsets_d, fsets_lens[n], gauss_params_d, rules_d, xxs_d, xxs_d_pitch, ys_d);
        CUDA_CALL(cudaPeekAtLastError());
    }

    cudaMemcpy(ys, ys_d, sizeof(unsigned[N]), cudaMemcpyDeviceToHost);

    cudaFree(ys_d);
    cudaFree(xxs_d);
    cudaFree(rules_d);
    cudaFree(gauss_params_d);
    cudaFree(fsets_offsets_d);
}

typedef curandStatePhilox4_32_10_t RandomState;

__global__ void initialize_random_states_kernel(RandomState *states, size_t states_pitch)
{
    unsigned chromosome_idx = blockIdx.x;
    unsigned param_idx = threadIdx.x;
    unsigned state_idx = chromosome_idx * blockDim.x + param_idx;

    RandomState *state = (RandomState*)((char*)states + chromosome_idx * states_pitch) + param_idx;
    curand_init(1234, state_idx, 0, state);
}

static
void initialize_random_states(
        RandomState *rules_random_states_d, size_t rules_random_states_d_pitch,
        unsigned new_population_power, unsigned fsets_total_len, unsigned rules_len)
{
    // initialize_random_states_kernel<<<new_population_power, fsets_total_len>>>(params_random_states_d, params_random_states_d_pitch);
    // CUDA_CALL(cudaDeviceSynchronize());
    // CUDA_CALL(cudaPeekAtLastError());
    initialize_random_states_kernel<<<new_population_power, rules_len>>>(rules_random_states_d, rules_random_states_d_pitch);
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaPeekAtLastError());
}

__global__ void initialize_params_kernel(
        RandomState *states, size_t states_pitch, unsigned *fsets_lens_expanded,
        GaussParams *gauss_params, size_t gauss_params_pitch, EvolutionaryParams *evolutionary_params, size_t evolutionary_params_pitch, float h)
{
    unsigned chromosome_idx = blockIdx.x;
    unsigned param_idx = threadIdx.x;

    unsigned fsets_len = fsets_lens_expanded[param_idx];

    GaussParams *gp = (GaussParams*)((char*)gauss_params + chromosome_idx * gauss_params_pitch) + param_idx;
    RandomState *state = (RandomState*)((char*)states + chromosome_idx * states_pitch) + param_idx;
    gp->mu = (curand(state) % fsets_len + 0.5f) / fsets_len;
    gp->sigma = 0.5f / fsets_len;

    EvolutionaryParams *ep = (EvolutionaryParams*)((char*)evolutionary_params + chromosome_idx * evolutionary_params_pitch) + param_idx;
    ep->sigma1 = ep->sigma2 = h / fsets_len;
}

__global__ void initialize_rules_kernel(
        RandomState *states, size_t states_pitch, const unsigned *fsets_lens,
        unsigned char *rules, size_t rules_pitch, unsigned n)
{
    unsigned i;
    unsigned chromosome_idx = blockIdx.x;
    unsigned rule_idx = threadIdx.x;
    size_t state_offset = chromosome_idx * states_pitch + rule_idx;

    RandomState local_state = ((RandomState*)((char*)states + chromosome_idx * states_pitch))[rule_idx];

    for (i = 0; i < n+1; ++i) {
        rules[chromosome_idx * rules_pitch + rule_idx * (n+1) + i] = curand(&local_state) % fsets_lens[i];
    }

    ((RandomState*)((char*)states + chromosome_idx * states_pitch))[rule_idx] = local_state;
}

static
void initialize_rules_population(
        RandomState *rules_random_states_d, size_t rules_random_states_d_pitch,
        const unsigned *fsets_lens, const unsigned *fsets_lens_d, unsigned fsets_total_len, unsigned population_power,
        GaussParams *fsets_d, unsigned char *rules_d, size_t rules_d_pitch, unsigned rules_len, unsigned n, float h)
{
    size_t i, k, offset = 0;
    GaussParams fsets[fsets_total_len];

    for (i = 0; i < n+1; ++i) {
        unsigned fsets_len = fsets_lens[i];
        for (k = 0; k < fsets_len; ++k) {
            fsets[offset + k].mu = (k + 0.5f) / fsets_len;
            fsets[offset + k].sigma = 0.5f / fsets_len;
        }
        offset += fsets_len;
    }

    cudaMemcpy(fsets_d, fsets, sizeof(GaussParams[fsets_total_len]), cudaMemcpyHostToDevice);

    initialize_rules_kernel<<<population_power, rules_len>>>(
            rules_random_states_d, rules_random_states_d_pitch, fsets_lens_d,
            rules_d, rules_d_pitch, n);
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaPeekAtLastError());
}

#define BLOCK_SIZE 64

__device__ double atomic_add(double* address, double val)
{
    unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void accumulate_scores_kernel(
        const unsigned *fsets_offsets, unsigned y_fsets_len,
        const GaussParams *fsets, const unsigned char *rules, size_t rules_pitch,
        const GaussParams *xxs, size_t xxs_pitch, const unsigned *ys, double *scores, unsigned N)
{
    unsigned chromosome_idx = blockIdx.y;
    unsigned executor_idx = threadIdx.z;
    unsigned executors_number = blockDim.z;
    unsigned data_idx = executor_idx + blockIdx.x * executors_number;

    if (data_idx < N) {
        float pred = evaluate_kernel(
                fsets_offsets, y_fsets_len, fsets, rules + chromosome_idx * rules_pitch,
                (GaussParams*)((char*)xxs + data_idx * xxs_pitch));
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            atomic_add(scores + chromosome_idx, pow((double)ys[data_idx] - pred, 2.));
            // scores[chromosome_idx] += powf(ys[data_idx] - pred, 2);
        }
    }
}

__global__ void normalize_scores_kernel(double *scores, unsigned population_power, unsigned N)
{
    unsigned idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (idx < population_power) scores[idx] = -sqrt(scores[idx] / N);
}

void compute_scores(
        const unsigned *fsets_offsets_d, const unsigned *fsets_lens, unsigned population_power,
        const GaussParams *fsets_d, const unsigned char *rules_d, size_t rules_d_pitch, unsigned rules_len, unsigned n,
        const GaussParams *xxs_d, size_t xxs_d_pitch, const unsigned *ys_d, unsigned N, double *scores_d)
{
    cudaMemset(scores_d, 0, sizeof(double[population_power]));

    {
        unsigned executors_number = device_props.maxThreadsPerBlock / ALIGN_UP(rules_len * n, WARP_SIZE);
        dim3 blocks = dim3((N + executors_number + 1) / executors_number, population_power);
        dim3 threads = dim3(n, rules_len, executors_number);
        size_t shared_sz = executors_number * ALIGN_UP(sizeof(Fraction[rules_len]) + sizeof(float[rules_len][n]), sizeof(float[WARP_SIZE]));
        accumulate_scores_kernel<<<blocks, threads, shared_sz>>>(
                fsets_offsets_d, fsets_lens[n], fsets_d, rules_d, rules_d_pitch, xxs_d, xxs_d_pitch, ys_d, scores_d, N);
    }
    CUDA_CALL(cudaDeviceSynchronize());
    normalize_scores_kernel<<<(population_power + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(scores_d, population_power, N);
    CUDA_CALL(cudaDeviceSynchronize());
}

template <typename T>
__global__ void find_min_kernel(const T *values, unsigned total_len, T *mins)
{
    unsigned step;
    unsigned idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    extern __shared__ char cache[];
    T *values_cache = (T*)cache;

    if (idx < total_len) values_cache[threadIdx.x] = values[idx];

    for (step = 1; threadIdx.x + step < BLOCK_SIZE && idx + step < total_len; step <<= 1) {
        if ((threadIdx.x & ((step<<1)-1)) == 0) {
            if (values_cache[threadIdx.x + step] < values_cache[threadIdx.x]) {
                values_cache[threadIdx.x] = values_cache[threadIdx.x + step];
            }
        }
    }

    mins[blockIdx.x] = values_cache[0];
}

template <typename T>
__global__ void compute_sum_kernel(const T *values, unsigned total_len, T min, T *sums)
{
    unsigned step;
    unsigned idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    extern __shared__ char cache[];
    T *values_cache = (T*)cache;

    if (idx < total_len) values_cache[threadIdx.x] = pow(values[idx] - min, 2.);

    for (step = 1; threadIdx.x + step < BLOCK_SIZE && idx + step < total_len; step <<= 1) {
        if ((threadIdx.x & ((step<<1)-1)) == 0) {
            values_cache[threadIdx.x] += values_cache[threadIdx.x + step];
        }
    }

    sums[blockIdx.x] = values_cache[0];
}

__global__ void select_rules_indices_kernel(
        RandomState *states, size_t states_pitch,
        const double *scores, unsigned *indices, unsigned total_len, double min, double sum)
{
    unsigned i;
    unsigned offset = blockIdx.x * BLOCK_SIZE;
    unsigned idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (idx < total_len) {
        RandomState *state = (RandomState*)((char*)states + idx * states_pitch);
        double p = curand_uniform(state) * sum;
        // double x = 0.;
        for (i = 0; i < total_len; ++i) {
            if (p > 0) {
                p -= pow(scores[i] - min, 2.);
                indices[idx] = i;
            }
            else
                break;
            // x += scores[i] - min;
        }
        // indices[idx] = x;
    }
}

static
void perform_rules_selection(
        RandomState *rules_random_states_d, size_t rules_random_states_d_pitch,
        const double *chromosome_scores_d, unsigned *chromosome_indices_d, unsigned population_power)
{
    unsigned i;
    double min, sum;

    {
        unsigned threads = BLOCK_SIZE;
        unsigned blocks = (population_power + BLOCK_SIZE - 1) / BLOCK_SIZE;
        double mins[blocks], *mins_d;

        unsigned shared_sz = sizeof(double[BLOCK_SIZE]);
        cudaMalloc(&mins_d, sizeof(double[blocks]));
        find_min_kernel<<<blocks, threads, shared_sz>>>(chromosome_scores_d, population_power, mins_d);
        cudaMemcpy(mins, mins_d, sizeof(double[blocks]), cudaMemcpyDeviceToHost);
        cudaFree(mins_d);

        min = mins[0];
        for (i = 1; i < blocks; ++i) if (mins[i] < min) min = mins[i];
    }

    {
        unsigned threads = BLOCK_SIZE;
        unsigned blocks = (population_power + BLOCK_SIZE - 1) / BLOCK_SIZE;
        double sums[blocks], *sums_d;

        unsigned shared_sz = sizeof(double[BLOCK_SIZE]);
        cudaMalloc(&sums_d, sizeof(double[blocks]));
        compute_sum_kernel<<<blocks, threads, shared_sz>>>(chromosome_scores_d, population_power, min, sums_d);
        cudaMemcpy(sums, sums_d, sizeof(double[blocks]), cudaMemcpyDeviceToHost);
        cudaFree(sums_d);

        sum = 0;
        for (i = 0; i < blocks; ++i) sum = sums[i];
        // sum -= min * population_power;
    }

    // printf("min = %f, sum = %f\n", min, sum);
    select_rules_indices_kernel<<<(population_power+BLOCK_SIZE-1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            rules_random_states_d, rules_random_states_d_pitch,
            chromosome_scores_d, chromosome_indices_d, population_power, min, sum);
}

__global__
void generate_random_rules_offsets_kernel(
        RandomState *states, size_t states_pitch,
        unsigned population_power, unsigned rules_len, unsigned n, unsigned *offsets)
{
    unsigned chromosome_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (chromosome_idx < population_power / 2) {
        RandomState *state = (RandomState*)((char*)states + chromosome_idx * states_pitch);
        offsets[chromosome_idx] = (curand(state) % (rules_len * (n+1)));
    }
}

__global__
void perform_crossingover_kernel(
        RandomState *states, size_t states_pitch,
        const unsigned *rules_offsets, const unsigned *indices,
        unsigned char *new_rules, unsigned char *rules, size_t rules_pitch, unsigned rules_len, unsigned n, float pc)
{
    unsigned chromosome_pair_idx = blockIdx.x;
    unsigned i;

    RandomState *state = (RandomState*)((char*)states + chromosome_pair_idx * states_pitch);

    if (curand_uniform(state) < pc) {

        unsigned rules_offset = rules_offsets[chromosome_pair_idx];
        unsigned char *rules_a = rules + rules_pitch * indices[2 * chromosome_pair_idx];
        unsigned char *rules_b = rules + rules_pitch * indices[2 * chromosome_pair_idx + 1];
        unsigned char *new_rules_a = new_rules + rules_pitch * (2 * chromosome_pair_idx);
        unsigned char *new_rules_b = new_rules + rules_pitch * (2 * chromosome_pair_idx + 1);

        extern __shared__ unsigned char rules_cache[];
        unsigned char *rules_a_cache = rules_cache;
        unsigned char *rules_b_cache = rules_cache + rules_len * (n+1);

        for (i = threadIdx.x; i < rules_len * (n+1); ++i) {
            rules_a_cache[i] = rules_a[i];
            rules_b_cache[i] = rules_b[i];
        }

        for (i = threadIdx.x; i < rules_len * (n+1); ++i) {
            if (i < rules_offset) {
                new_rules_a[i] = rules_a_cache[i];
                new_rules_b[i] = rules_b_cache[i];
            } else {
                new_rules_a[i] = rules_b_cache[i];
                new_rules_b[i] = rules_a_cache[i];
            }
        }
    }
}

static
void perform_rules_crossingover(
        RandomState *rules_random_states_d, size_t rules_random_states_d_pitch,
        unsigned *rules_offsets_d, const unsigned *chromosome_indices_d, unsigned population_power,
        unsigned char *new_rules_d, unsigned char *rules_d, size_t rules_d_pitch, unsigned rules_len, unsigned n, float pc)
{
    generate_random_rules_offsets_kernel<<<(population_power / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            rules_random_states_d, rules_random_states_d_pitch,
            population_power, rules_len, n, rules_offsets_d);
    CUDA_CALL(cudaDeviceSynchronize());
    {
        size_t shared_sz = sizeof(unsigned char[2][rules_len * (n+1)]);
        perform_crossingover_kernel<<<population_power / 2, BLOCK_SIZE, shared_sz>>>(
                rules_random_states_d, rules_random_states_d_pitch,
                rules_offsets_d, chromosome_indices_d,
                new_rules_d, rules_d, rules_d_pitch, rules_len, n, pc);
        CUDA_CALL(cudaDeviceSynchronize());
    }
}

__global__ void perform_rules_mutation_kernel(
        RandomState *states, size_t states_pitch, const unsigned *fsets_lens,
        unsigned char *rules, size_t rules_pitch, unsigned n, float pm)
{
    unsigned i;
    unsigned chromosome_idx = blockIdx.x;
    unsigned rule_idx = threadIdx.x;

    unsigned char *rule_row = rules + chromosome_idx * rules_pitch + rule_idx * (n+1);
    RandomState local_state = ((RandomState*)((char*)states + chromosome_idx * states_pitch))[rule_idx];
    for (i = 0; i < n+1; ++i) {
        if (curand_uniform(&local_state) <= pm) {
            rule_row[i] = curand(&local_state) % fsets_lens[i];
            // printf("%u %u %u: %u\n", chromosome_idx, rule_idx, i, rule_row[i]);
        }
        __syncthreads();
    }
    ((RandomState*)((char*)states + chromosome_idx * states_pitch))[rule_idx] = local_state;
}

static
void perform_rules_mutation(
        RandomState *rules_random_states_d, size_t rules_random_states_d_pitch,
        const unsigned *fsets_lens_d, unsigned population_power,
        unsigned char *rules_d, size_t rules_d_pitch, unsigned rules_len, unsigned n, float pm)
{
    perform_rules_mutation_kernel<<<population_power, rules_len>>>(
            rules_random_states_d, rules_random_states_d_pitch, fsets_lens_d,
            rules_d, rules_d_pitch, n, pm);
    CUDA_CALL(cudaDeviceSynchronize());
}

/*
__global__ void generate_random_indices_kernel(RandomState *states, size_t states_pitch, unsigned *indices, unsigned population_power, unsigned k)
{
    unsigned idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (idx < population_power) {
        RandomState *state = (RandomState*)((char*)states + blockIdx.x * states_pitch) + threadIdx.x;
        indices[idx] = curand(state) % k;
    }
}

__global__ void copy_params_for_given_chromosomes_kernel(
        const unsigned *chromosome_indices,
        GaussParams *new_gauss_params, const GaussParams *gauss_params, size_t gauss_params_pitch,
        EvolutionaryParams *new_evolutionary_params, const EvolutionaryParams *evolutionary_params, size_t evolutionary_params_pitch)
{
    unsigned new_chromosome_idx = blockIdx.x;
    unsigned param_idx = threadIdx.x;

    unsigned chromosome_idx = chromosome_indices[new_chromosome_idx];

    GaussParams *ngps = (GaussParams*)((char*)new_gauss_params + new_chromosome_idx * gauss_params_pitch);
    GaussParams *gps = (GaussParams*)((char*)gauss_params + chromosome_idx * gauss_params_pitch);
    ngps[param_idx] = gps[param_idx];

    EvolutionaryParams *neps = (EvolutionaryParams*)((char*)new_evolutionary_params + new_chromosome_idx * evolutionary_params_pitch);
    EvolutionaryParams *eps = (EvolutionaryParams*)((char*)evolutionary_params + chromosome_idx * evolutionary_params_pitch);
    neps[param_idx] = eps[param_idx];
}

__global__ void copy_rules_for_given_chromosomes_kernel(
        const unsigned *chromosome_indices,
        unsigned char *new_rules, const unsigned char *rules, size_t rules_pitch,
        unsigned n)
{
    unsigned i;
    unsigned new_chromosome_idx = blockIdx.x;
    unsigned rules_len = blockDim.x;
    unsigned rule_idx = threadIdx.x;

    unsigned chromosome_idx = chromosome_indices[new_chromosome_idx];
    for (i = 0; i < n+1; ++i) {
        new_rules[new_chromosome_idx * rules_pitch + rule_idx * (n+1) + i] = rules[chromosome_idx * rules_pitch + rule_idx * (n+1) + i];
    }
}

static
void perform_reproduction(
        RandomState *random_states_d, size_t random_states_d_pitch,
        unsigned *chromosome_indices_d, unsigned population_power, unsigned new_population_power,
        GaussParams *new_gauss_params_d, const GaussParams *gauss_params_d, size_t gauss_params_d_pitch,
        EvolutionaryParams *new_evolutionary_params_d, const EvolutionaryParams *evolutionary_params_d, size_t evolutionary_params_d_pitch,
        unsigned char *new_rules_d, const unsigned char *rules_d, size_t rules_d_pitch, unsigned fsets_total_len, unsigned rules_len, unsigned n)
{
    generate_random_indices_kernel<<<(new_population_power + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            random_states_d, random_states_d_pitch, chromosome_indices_d, new_population_power, population_power);
    copy_params_for_given_chromosomes_kernel<<<new_population_power, fsets_total_len>>>(
            chromosome_indices_d, new_gauss_params_d, gauss_params_d, gauss_params_d_pitch,
            new_evolutionary_params_d, evolutionary_params_d, evolutionary_params_d_pitch);
    copy_rules_for_given_chromosomes_kernel<<<new_population_power, rules_len>>>(
            chromosome_indices_d, new_rules_d, rules_d, rules_d_pitch, n);
}

__global__ void find_max_score(double *scores, unsigned *indices, unsigned total_len, unsigned offset)
{
    unsigned step;
    unsigned chromosome_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    extern __shared__ char cache[];
    double *scores_cache = (double*)cache;
    unsigned *indices_cache = (unsigned*)(cache + BLOCK_SIZE * sizeof(double));

    if (chromosome_idx < total_len) {
        scores_cache[threadIdx.x] = 1.; // scores[offset + chromosome_idx];
        indices_cache[threadIdx.x] = offset + chromosome_idx;
    }

    for (step = 1; step < BLOCK_SIZE; step <<= 1) {
        if ((threadIdx.x & ((step<<1)-1)) == 0 && chromosome_idx + step < total_len) {
            if (!isnan(scores_cache[threadIdx.x + step]) && scores_cache[threadIdx.x + step] > scores_cache[threadIdx.x]
                    || isnan(scores_cache[threadIdx.x])) {
                scores_cache[threadIdx.x] = scores_cache[threadIdx.x + step];
                indices_cache[threadIdx.x] = indices_cache[threadIdx.x + step];
            }
        }
    }

    if (threadIdx.x == 0) {
        indices[offset + blockIdx.x] = indices_cache[0];
    }
}

__global__ void find_max_score_from_indices(double *scores, unsigned *indices, unsigned offset)
{
    unsigned step;
    unsigned idx = threadIdx.x;
    unsigned dim = blockDim.x;

    extern __shared__ char cache[];
    double *scores_cache = (double*)cache;
    unsigned *indices_cache = (unsigned*)(cache + dim * sizeof(double));

    {
        unsigned score_idx = indices_cache[idx] = indices[offset + idx];
        scores_cache[idx] = scores[score_idx];
    }
    for (step = 1; step < dim; step <<= 1) {
        if ((idx & ((step<<1)-1)) == 0 && idx + step < dim) {
            if (scores_cache[idx + step] > scores_cache[idx]) {
                scores_cache[idx] = scores_cache[idx + step];
                indices_cache[idx] = indices_cache[idx + step];
            }
        }
    }

    if (threadIdx.x == 0) {
        scores[indices_cache[0]] = -INFINITY;
        scores[offset] = scores_cache[0];
        indices[offset] = indices_cache[0];
    }
}

static
void perform_selection(
        double *chromosome_scores_d, unsigned *chromosome_indices_d, unsigned population_power, unsigned new_population_power,
        GaussParams *gauss_params_d, const GaussParams *new_gauss_params_d, size_t gauss_params_d_pitch,
        EvolutionaryParams *evolutionary_params_d, const EvolutionaryParams *new_evolutionary_params_d, size_t evolutionary_params_d_pitch,
        unsigned char *rules_d, const unsigned char *new_rules_d, size_t rules_d_pitch, unsigned fsets_total_len, unsigned rules_len, unsigned n)
{
    unsigned i;

    for (i = 0; i < population_power; ++i) {
        unsigned len = new_population_power - i;
        unsigned dim = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;
        {
            size_t shared_sz = sizeof(double[BLOCK_SIZE]) + sizeof(unsigned[BLOCK_SIZE]);
            find_max_score<<<dim, BLOCK_SIZE, shared_sz>>>(chromosome_scores_d, chromosome_indices_d, len, i);
            CUDA_CALL(cudaDeviceSynchronize());
        }
        {
            size_t shared_sz = sizeof(double[dim]) + sizeof(unsigned[dim]);
            find_max_score_from_indices<<<1, dim, shared_sz>>>(chromosome_scores_d, chromosome_indices_d, i);
            CUDA_CALL(cudaDeviceSynchronize());
        }
    }

    copy_params_for_given_chromosomes_kernel<<<population_power, fsets_total_len>>>(
            chromosome_indices_d, gauss_params_d, new_gauss_params_d, gauss_params_d_pitch,
            evolutionary_params_d, new_evolutionary_params_d, evolutionary_params_d_pitch);
    copy_rules_for_given_chromosomes_kernel<<<population_power, rules_len>>>(
            chromosome_indices_d, rules_d, new_rules_d, rules_d_pitch, n);
}

// static void perform_reproduction_and_selection(
//         float *chromosome_scores_d, unsigned *chromosome_indices_d, unsigned population_power, unsigned new_population_power,
//         GaussParams *gauss_params_d, size_t gauss_params_d_pitch,
//         EvolutionaryParams *evolutionary_params_d, size_t evolutionary_params_d_pitch,
//         unsigned char *rules_d, size_t rules_d_pitch, unsigned fsets_total_len, unsigned rules_len, unsigned n)
// {

// }

__global__ void generate_random_normal_numbers_kernel(RandomState *states, size_t states_pitch, unsigned population_power, float *numbers)
{
    unsigned idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (idx < population_power) {
        RandomState *state = (RandomState*)((char*)states + idx * states_pitch);
        numbers[idx] = curand_normal(state);
    }
}

__global__ void perform_params_mutation_kernel(
        RandomState *states, size_t states_pitch, const float *ps,
        GaussParams *gauss_params, size_t gauss_params_pitch, EvolutionaryParams *evolutionary_params, size_t evolutionary_params_pitch)
{
    unsigned chromosome_idx = blockIdx.x;
    unsigned fsets_total_len = blockDim.x;
    unsigned param_idx = threadIdx.x;

    unsigned chromosome_len = fsets_total_len * 2;
    float tau1 = 1.f / sqrtf(2.f * chromosome_len);
    float tau = 1.f / sqrtf(2.f * sqrtf(chromosome_len));

    RandomState local_state = ((RandomState*)((char*)states + chromosome_idx * states_pitch))[param_idx];
    float p = ps[chromosome_idx];

    EvolutionaryParams *ep = (EvolutionaryParams*)((char*)evolutionary_params + chromosome_idx * evolutionary_params_pitch) + param_idx;
    float sigma1 = ep->sigma1 *= expf(tau1 * p + tau * curand_normal(&local_state));
    float sigma2 = ep->sigma2 *= expf(tau1 * p + tau * curand_normal(&local_state));

    GaussParams *gp = (GaussParams*)((char*)gauss_params + chromosome_idx * gauss_params_pitch) + param_idx;
    gp->mu += sigma1 * curand_normal(&local_state);
    gp->sigma += sigma2 * curand_normal(&local_state);

    ((RandomState*)((char*)states + chromosome_idx * states_pitch))[param_idx] = local_state;
}

static
void perform_mutation(
        RandomState *params_random_states_d, size_t params_random_states_d_pitch, RandomState *rules_random_states_d, size_t rules_random_states_d_pitch,
        float *ps_d, unsigned new_population_power, const unsigned *fsets_lens_d, unsigned fsets_total_len,
        GaussParams *gauss_params_d, size_t gauss_params_d_pitch, EvolutionaryParams *evolutionary_params_d, size_t evolutionary_params_d_pitch,
        unsigned char *rules_d, size_t rules_d_pitch, unsigned rules_len, unsigned n, float pm)
{
    generate_random_normal_numbers_kernel<<<(new_population_power + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            params_random_states_d, params_random_states_d_pitch, new_population_power, ps_d);
    CUDA_CALL(cudaDeviceSynchronize());
    perform_params_mutation_kernel<<<new_population_power, fsets_total_len>>>(
            params_random_states_d, params_random_states_d_pitch, ps_d,
            gauss_params_d, gauss_params_d_pitch, evolutionary_params_d, evolutionary_params_d_pitch);
    CUDA_CALL(cudaDeviceSynchronize());
}

static
void perform_crossingover(
        RandomState *rules_random_states_d, size_t rules_random_states_d_pitch, unsigned *indices_d, unsigned population_power,
        unsigned char *rules_d, size_t rules_d_pitch, unsigned rules_len, unsigned n)
{
    generate_random_normal_indices_kernel<<<(population_power / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            rules_random_states_d, rules_random_states_d_pitch, population_power / 2, rules_len, n, indices_d);
    CUDA_CALL(cudaDeviceSynchronize());
    perform_crossingover_kernel<<<population_power / 2, BLOCK_SIZE, sizeof(unsigned char[2][rules_len * (n+1)])>>>(
            indices_d, rules_d, rules_d_pitch, rules_len, n);
    CUDA_CALL(cudaDeviceSynchronize());
}
*/

extern "C"
void tune_lfs_gpu_impl(const unsigned *fsets_lens, unsigned rules_len, unsigned n, const GaussParams *xxs, const unsigned *ys, unsigned N, unsigned mu, unsigned lambda, unsigned it_number, GaussParams *gauss_params, unsigned char *rules)
{
    /*
     * GPU memory layout:
     * +==============================
     * | fsets_offsets, fsets_lens buffer
     * +------------------------------
     * | RandomState array [new_population_power, fsets_total_len]
     * +------------------------------
     * | GaussParams array [population_power, fsets_total_len]
     * +------------------------------
     * | NEW GaussParams array [new_population_power, fsets_total_len]
     * +------------------------------
     * | EvolutionaryParams array [population_power, fsets_total_len]
     * +------------------------------
     * | NEW EvolutionaryParams array [new_population_power, fsets_total_len]
     * +==============================
     * | RandomState array [new_population_power, rules_len]
     * +------------------------------
     * | rules array [population_power, rules_len, n+1]
     * +------------------------------
     * | NEW rules array [new_population_power, rules_len, n+1]
     * +==============================
     * | Input array [N, n]
     * +------------------------------
     * | Output buffer [N]
     * +==============================
     * | Union[scores, ps] buffer [new_population_power]
     * +------------------------------
     * | Indices buffer [new_population_power]
     * +==============================
     */

    init_device_props();

    unsigned population_power = mu;
    unsigned new_population_power = lambda;

    assert(new_population_power >= population_power);

    size_t i, it;
    size_t fsets_total_len = 0;

    unsigned fsets_offsets[n+1];

    {
        unsigned offset = 0;
        for (i = 0; i < n+1; ++i) {
            fsets_offsets[i] = offset;
            offset += fsets_lens[i];
            fsets_total_len += fsets_lens[i];
        }
    }

    size_t params_random_states_d_pitch, rules_random_states_d_pitch;
    size_t gauss_params_d_pitch, evolutionary_params_d_pitch;
    size_t rules_d_pitch, xxs_d_pitch;

    unsigned *fsets_offsets_d=0, *fsets_lens_d, *indices_d, *rules_offsets_d;
    RandomState *params_random_states_d, *rules_random_states_d;
    GaussParams *gauss_params_d, *new_gauss_params_d, *fsets_d, *xxs_d;
    EvolutionaryParams *evolutionary_params_d, *new_evolutionary_params_d;
    unsigned char *rules_d, *new_rules_d;
    unsigned *ys_d;
    double scores[new_population_power], *scores_d;
    float *ps_d;

    CUDA_CALL(cudaMalloc(&fsets_offsets_d, sizeof(unsigned[n+1])));
    CUDA_CALL(cudaMalloc(&fsets_lens_d, sizeof(unsigned[n+1])));
    CUDA_CALL(cudaMalloc(&fsets_d, sizeof(GaussParams[fsets_total_len])));
    // CUDA_CALL(cudaMallocPitch(&params_random_states_d, &params_random_states_d_pitch, sizeof(RandomState[fsets_total_len]), new_population_power));
    // CUDA_CALL(cudaMallocPitch(&gauss_params_d, &gauss_params_d_pitch, sizeof(GaussParams[fsets_total_len]), population_power));
    // CUDA_CALL(cudaMallocPitch(&new_gauss_params_d, &gauss_params_d_pitch, sizeof(GaussParams[fsets_total_len]), new_population_power));
    // CUDA_CALL(cudaMallocPitch(&evolutionary_params_d, &evolutionary_params_d_pitch, sizeof(EvolutionaryParams[fsets_total_len]), population_power));
    // CUDA_CALL(cudaMallocPitch(&new_evolutionary_params_d, &evolutionary_params_d_pitch, sizeof(EvolutionaryParams[fsets_total_len]), new_population_power));
    CUDA_CALL(cudaMallocPitch(&rules_random_states_d, &rules_random_states_d_pitch, sizeof(RandomState[rules_len]), new_population_power));
    CUDA_CALL(cudaMallocPitch(&rules_d, &rules_d_pitch, sizeof(unsigned char[rules_len][n+1]), new_population_power));
    CUDA_CALL(cudaMallocPitch(&new_rules_d, &rules_d_pitch, sizeof(unsigned char[rules_len][n+1]), new_population_power));
    CUDA_CALL(cudaMallocPitch(&xxs_d, &xxs_d_pitch, sizeof(GaussParams[n]), N));
    CUDA_CALL(cudaMalloc(&ys_d, sizeof(unsigned[N])));
    CUDA_CALL(cudaMalloc(&scores_d, sizeof(double[new_population_power])));
    CUDA_CALL(cudaMalloc(&ps_d, sizeof(float[new_population_power])));
    CUDA_CALL(cudaMalloc(&indices_d, sizeof(unsigned[new_population_power])));
    CUDA_CALL(cudaMalloc(&rules_offsets_d, sizeof(unsigned[new_population_power / 2])));

    initialize_random_states(
            rules_random_states_d, rules_random_states_d_pitch,
            new_population_power, fsets_total_len, rules_len);

    CUDA_CALL(cudaMemcpy(fsets_offsets_d, fsets_offsets, sizeof(unsigned[n+1]), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaMemcpy(fsets_lens_d, fsets_lens, sizeof(unsigned[n+1]), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaPeekAtLastError());

    initialize_rules_population(
            rules_random_states_d, rules_random_states_d_pitch,
            fsets_lens, fsets_lens_d, fsets_total_len, new_population_power,
            fsets_d, rules_d, rules_d_pitch, rules_len, n, 0.001);

    CUDA_CALL(cudaMemcpy2D(xxs_d, xxs_d_pitch, xxs, sizeof(GaussParams[n]), sizeof(GaussParams[n]), N, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaMemcpy(ys_d, ys, sizeof(unsigned[N]), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaPeekAtLastError());

    FILE *f = fopen("report_gpu.txt", "w");

    // NOTE(sergey): For now we thoughtless initialize a rules table.
    // But we set gauss params of the membership functions to identically
    // describe each of term for each input.
    printf("it_number = %u\n", it_number);
    for (it = 0; it < it_number; ++it) {
        compute_scores(
                fsets_offsets_d, fsets_lens, new_population_power,
                fsets_d, rules_d, rules_d_pitch, rules_len, n,
                xxs_d, xxs_d_pitch, ys_d, N, scores_d);

        perform_rules_selection(
                rules_random_states_d, rules_random_states_d_pitch,
                scores_d, indices_d, new_population_power);

        perform_rules_crossingover(
                rules_random_states_d, rules_random_states_d_pitch,
                rules_offsets_d, indices_d, new_population_power,
                new_rules_d, rules_d, rules_d_pitch, rules_len, n, 0.5f);

        // {
        //     unsigned indices[new_population_power];
        //     cudaMemcpy(indices, indices_d, sizeof(indices), cudaMemcpyDeviceToHost);
        //     for (i = 0; i < sizeof(indices) / sizeof(indices[0]); ++i) printf("%u ", indices[i]);
        //     printf("\n");
        // }

        perform_rules_mutation(
                rules_random_states_d, rules_random_states_d_pitch,
                fsets_lens_d, new_population_power, new_rules_d, rules_d_pitch, rules_len, n, 0.001f);

        cudaMemcpy2D(rules_d, rules_d_pitch, new_rules_d, rules_d_pitch, sizeof(unsigned char[rules_len][n+1]), new_population_power, cudaMemcpyDeviceToDevice);

        // Debug purpose

        cudaMemcpy(scores, scores_d, sizeof(double[new_population_power]), cudaMemcpyDeviceToHost);
        CUDA_CALL(cudaPeekAtLastError());

        {
            double avg = 0.f;
            printf("It [%3d] ", it);
            for (i = 0; i < new_population_power; ++i) {
                printf("%f ", scores[i]);
                avg += scores[i];
            }
            avg /= new_population_power;
            printf("Avg: %f\n", avg);
            fprintf(f, "%f\n", avg);
        }
    }

    /*
    for (it = 0; it < it_number; ++it) {
        perform_reproduction(
                params_random_states_d, params_random_states_d_pitch, indices_d, population_power, new_population_power,
                new_gauss_params_d, gauss_params_d, gauss_params_d_pitch,
                new_evolutionary_params_d, evolutionary_params_d, evolutionary_params_d_pitch,
                new_rules_d, rules_d, rules_d_pitch, fsets_total_len, rules_len, n);

        perform_mutation(
                params_random_states_d, params_random_states_d_pitch, rules_random_states_d, rules_random_states_d_pitch,
                ps_d, new_population_power, fsets_lens_d, fsets_total_len,
                new_gauss_params_d, gauss_params_d_pitch, new_evolutionary_params_d, evolutionary_params_d_pitch,
                new_rules_d, rules_d_pitch, rules_len, n, 0.1f / (rules_len*(n+1)));

        compute_scores(
                new_population_power, fsets_offsets_d,
                new_gauss_params_d, gauss_params_d_pitch, new_rules_d, rules_d_pitch, rules_len, n,
                xxs_d, xxs_d_pitch, ys_d, N, scores_d);

        perform_selection(
                scores_d, indices_d, population_power, new_population_power,
                gauss_params_d, new_gauss_params_d, gauss_params_d_pitch,
                evolutionary_params_d, new_evolutionary_params_d, evolutionary_params_d_pitch,
                rules_d, new_rules_d, rules_d_pitch, fsets_total_len, rules_len, n);

        // Debug purpose

        cudaMemcpy(scores, scores_d, sizeof(double[population_power]), cudaMemcpyDeviceToHost);

        {
            double avg = 0.f;
            printf("It [%3d] ", it);
            for (i = 0; i < population_power; ++i) {
                printf("%f ", scores[i]);
                avg += scores[i];
            }
            avg /= population_power;
            printf("Avg: %f\n", avg);
            fprintf(f, "%f\n", avg);
        }
    }
    */
    fclose(f);

    {
        cudaMemcpy(scores, scores_d, sizeof(scores), cudaMemcpyDeviceToHost);

        double max_value = scores[0];
        unsigned max_index = 0;

        for (i = 1; i < new_population_power; ++i) {
            if (scores[i] > max_value) {
                max_value = scores[i];
                max_index = i;
            }
        }

        cudaMemcpy(gauss_params, fsets_d, sizeof(GaussParams[fsets_total_len]), cudaMemcpyDeviceToHost);
        cudaMemcpy(rules, rules_d + max_index * rules_d_pitch, sizeof(unsigned char[rules_len][n+1]), cudaMemcpyDeviceToHost);
    }

    cudaFree(indices_d);
    cudaFree(ps_d);
    cudaFree(scores_d);
    cudaFree(ys_d);
    cudaFree(xxs_d);
    cudaFree(new_rules_d);
    cudaFree(rules_d);
    cudaFree(rules_random_states_d);
    // cudaFree(new_evolutionary_params_d);
    // cudaFree(evolutionary_params_d);
    // cudaFree(new_gauss_params_d);
    // cudaFree(gauss_params_d);
    // cudaFree(params_random_states_d);
    cudaFree(fsets_d);
    cudaFree(fsets_lens_d);
    cudaFree(fsets_offsets_d);
    CUDA_CALL(cudaDeviceSynchronize());
}
