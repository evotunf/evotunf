#include <utility>
#include <stdlib.h>
#include <stdio.h>
// #include <math.h>
#include <assert.h>
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>
#include "common.h"
// #include "random.h"
#include "evolutionary_tune.h"


#define CUDA_CALL(...)																									\
	do {																																	\
		cudaDeviceSynchronize();																						\
		cudaError_t ret = (__VA_ARGS__);																		\
		if (ret != cudaSuccess) {																						\
			fprintf(stderr, "CUDA: Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(ret)); \
			exit(-1);																													\
		}																																		\
	} while (0)

#define CURAND_CALL(...)																								\
	do {																																	\
		curandStatus_t ret = (__VA_ARGS__);																	\
		if (ret != CURAND_STATUS_SUCCESS) fprintf(stderr, "CUDA: Error at %s:%d: code #%d\n", __FILE__, __LINE__, ret); \
	} while (0)

#define SHARED_MEM_BANK_SIZE 4
#define SHARED_MEM_ALIGNMENT (32*SHARED_MEM_BANK_SIZE)

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

#define SIGN(x) ((typeof(x))(((x) < 0) ? -1 : 1))
#define GAUSS(mu, sigma, x) exp(-pow(((x) - (mu)) / (sigma), 2))
#define TNORM(a, b) fminf(a, b)
#define LUKASZEWICZ_IMPL(a, b) fminf(1.f, 1.f - (a) + (b))

typedef struct GaussEvoParams {
	char s_mu, s_sigma;
} GaussEvoParams;

typedef struct __align__(8) Fraction {
	float numerator, denominator;
} Fraction;

texture<float2> uxxs_tex;
texture<unsigned> ys_tex;


template <typename GaussParamsT>
__host__ __device__ static GaussParams make_gauss_params(GaussParamsT gp, unsigned i = 0, unsigned n = 1)
{
	return gp;
}


template <>
__host__ __device__ GaussParams make_gauss_params<GaussEvoParams>(GaussEvoParams gep, unsigned i, unsigned n)
{
	return {((i+1) + (float)(gep.s_mu-10)/20) / (n+1), (float)gep.s_sigma / 10 / (n+1)};
}


template <typename GaussParamsT>
__device__ unsigned classify_fuzzy(unsigned fsets_total_len, const unsigned *fset_lens, const unsigned *fset_offsets,
																	 const GaussParamsT *fsets,
																	 const signed char *rules,
																	 const GaussParams *uxx_batch)
{
	unsigned batch_len = blockDim.z;
	unsigned thread_group_idx = threadIdx.z;
	const unsigned k = threadIdx.y;
	const unsigned i = threadIdx.x;
	const unsigned n = blockDim.x;
	const unsigned rules_len = blockDim.y;
	const unsigned idx = threadIdx.y * blockDim.x + threadIdx.x;
	const unsigned a_fset_len = fset_lens[i];
	const unsigned b_fset_len = fset_lens[n];
	const unsigned a_fset_offset = fset_offsets[i];
	const unsigned b_fset_offset = fset_offsets[n];

	extern __shared__ char cache[];
	char *cache_base = cache, *cache_top = cache;

	GaussParams *fsets_cache = (GaussParams*) (cache_top = cache_base + ALIGN_UP(fsets_total_len * sizeof(GaussParams), SHARED_MEM_ALIGNMENT),
																						 cache_base);
	signed char *rules_cache = (signed char*) (cache_top = (cache_base = cache_top) + ALIGN_UP(rules_len * (n+1), SHARED_MEM_ALIGNMENT),
																						 cache_base);
	GaussParams *uxx_cache = (GaussParams*) (cache_top = (cache_base = cache_top) + ALIGN_UP(batch_len * n * sizeof(GaussParams), SHARED_MEM_ALIGNMENT),
																					 cache_base + thread_group_idx * n * sizeof(GaussParams));
	float *max_tnorm_cache = (float*) (cache_top = (cache_base = cache_top) + ALIGN_UP(batch_len * rules_len * n * sizeof(float), SHARED_MEM_ALIGNMENT),
																		 cache_base + (thread_group_idx * rules_len + k) * n * sizeof(float));
	Fraction *fractions_cache = (Fraction*) (cache_top = (cache_base = cache_top) + ALIGN_UP(batch_len * rules_len * sizeof(Fraction), SHARED_MEM_ALIGNMENT),
																					 cache_base + thread_group_idx * rules_len * sizeof(Fraction));

	if (threadIdx.z == 0) {
		if (idx < n+1) {
			for (unsigned j = 0; j < fset_lens[idx]; ++j) {
				fsets_cache[fset_offsets[idx] + j] = make_gauss_params(fsets[fset_offsets[idx] + j], j, fset_lens[idx]);
			}
		}
		rules_cache[k * (n+1) + i] = rules[k * (n+1) + i];
		if (threadIdx.x == 0) rules_cache[k * (n+1) + n] = rules[k * (n+1) + n];
	}
	if (threadIdx.y == 0) uxx_cache[i] = uxx_batch[thread_group_idx * n + i];
	// if (threadIdx.y == 0) ((float2*)uxx_cache)[i] = tex1Dfetch(uxxs_tex, blockIdx.x * n + i);

	__syncthreads();

	signed char y = rules_cache[k * (n+1) + n];
	GaussParams ux =  uxx_cache[i];
	float uy_center = fsets_cache[b_fset_offset + (y-1)].mu; //make_gauss_params(fsets_cache[b_fset_offset + (y-1)], y-1, b_fset_len).mu;

	float cross = 1.f;
	
	for (unsigned j = 0; j < rules_len; ++j) {
		signed char a = rules_cache[j * (n+1) + i];
		signed char b = rules_cache[j * (n+1) + n];

		float max_tnorm = -INFINITY;

		if (a) {
			GaussParams ua = fsets_cache[a_fset_offset + (a-1)];//make_gauss_params(fsets_cache[a_fset_offset + (a-1)], a-1, a_fset_len);
			GaussParams ub = fsets_cache[b_fset_offset + (b-1)];//make_gauss_params(fsets_cache[b_fset_offset + (b-1)], b-1, b_fset_len);
			float ub_value = GAUSS(ub.mu, ub.sigma, uy_center);

			for (float t = 0.0f; t < 1.01f; t += 0.05f)
			{
				float tnorm = TNORM(GAUSS(ux.mu, ux.sigma, t), IMPL(GAUSS(ua.mu, ua.sigma, t), ub_value));
				max_tnorm = fmaxf(max_tnorm, tnorm);
			}
		}
		max_tnorm_cache[i] = max_tnorm;
		__syncthreads();

		{
			unsigned step = ROUND_UP_TO_POW2(n)/2;
			if (i < (n-step)) {
				for (; step > 0; step >>= 1) {
					if (i < step) {
						max_tnorm_cache[i] = fmaxf(max_tnorm_cache[i], max_tnorm_cache[i + step]);
					}
					// __syncthreads();
				}
			}
		}
		__syncthreads();
		if (threadIdx.x == 0 && max_tnorm_cache[0] != -INFINITY) cross = fminf(cross, max_tnorm_cache[0]);
	}

	if (threadIdx.x == 0) {
		fractions_cache[k].numerator = uy_center * cross;
		fractions_cache[k].denominator = cross;
	}

	__syncthreads();

	{
		unsigned step = ROUND_UP_TO_POW2(rules_len)/2;
		if (idx < (rules_len-step)) {
			for (; step > 0; step >>= 1) {
				if (idx < step) {
					fractions_cache[idx].numerator += fractions_cache[idx + step].numerator;
					fractions_cache[idx].denominator += fractions_cache[idx + step].denominator;
				}
				__syncthreads();
			}
		}
	}

	if (threadIdx.x == 0 && threadIdx.y == 0) {
		float y = fractions_cache[0].numerator / fractions_cache[0].denominator;
		float max_value = 0.f;
		unsigned max_index = 0;

		for (unsigned j = 0; j < b_fset_len; ++j) {
			GaussParams ub = fsets_cache[b_fset_offset + j];
			float uy_value = GAUSS(ub.mu, ub.sigma, y);

			if (uy_value > max_value) {
				max_value = uy_value;
				max_index = j;
			}
		}
		return max_index + 1;
	}
	return 0;
}


__global__ void perform_inference_kernel(unsigned fsets_total_len, const unsigned *fset_lens, const unsigned *fset_offsets,
																				 const GaussParams *fsets, unsigned n,
																				 const signed char *rules, unsigned rules_len,
																				 const GaussParams *uxx_table, size_t uxx_table_pitch, unsigned batch_len, unsigned *ys, unsigned N)
{
	unsigned batch_idx = blockIdx.x;
	unsigned thread_group_idx = threadIdx.z;
	unsigned idx = batch_idx * batch_len + thread_group_idx;

	if (idx < N) {
		const GaussParams *uxx_batch = (GaussParams*) ((char*)uxx_table + batch_idx * uxx_table_pitch);
		unsigned pred = classify_fuzzy(fsets_total_len, fset_lens, fset_offsets,
																	 fsets, rules, uxx_batch);
		if (threadIdx.y == 0 && threadIdx.x == 0) {
			ys[idx] = pred;
		}
	}
}


void predict_gpu_impl(const unsigned *fset_lens, const GaussParams *fsets, unsigned n,
											const signed char *rules, unsigned rules_len,
											const GaussParams *uxxs, unsigned *ys, unsigned N)
{
	init_device_props();

	unsigned i;
	unsigned fset_offsets[n+1], fsets_total_len = 0;

	for (i = 0; i < n+1; ++i) {
		fset_offsets[i] = fsets_total_len;
		fsets_total_len += fset_lens[i];
	}

	unsigned *fset_lens_d, *fset_offsets_d;
	GaussParams *fsets_d, *uxxs_d;
	signed char *rules_d;
	unsigned *ys_d;
	size_t uxx_table_pitch;
	unsigned batch_len = 1;

	CUDA_CALL(cudaPeekAtLastError());
	CUDA_CALL(cudaMalloc(&fset_lens_d, sizeof(unsigned[n+1])));
	CUDA_CALL(cudaMemcpy(fset_lens_d, fset_lens, sizeof(unsigned[n+1]), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMalloc(&fset_offsets_d, sizeof(unsigned[n+1])));
	CUDA_CALL(cudaMemcpy(fset_offsets_d, fset_offsets, sizeof(unsigned[n+1]), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMalloc(&fsets_d, sizeof(GaussParams[fsets_total_len])));
	CUDA_CALL(cudaMemcpy(fsets_d, fsets, sizeof(GaussParams[fsets_total_len]), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMalloc(&rules_d, sizeof(signed char[rules_len][n+1])));
	CUDA_CALL(cudaMemcpy(rules_d, rules, sizeof(signed char[rules_len][n+1]), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMallocPitch(&uxxs_d, &uxx_table_pitch, sizeof(GaussParams[batch_len][n]), (N + batch_len - 1) / batch_len));
	{
		unsigned last_batch_len = N % batch_len, last_batch_idx = N / batch_len;
		CUDA_CALL(cudaMemcpy2D(uxxs_d, uxx_table_pitch, uxxs, sizeof(GaussParams[batch_len][n]),
													 sizeof(GaussParams[batch_len][n]), N / batch_len, cudaMemcpyHostToDevice));
		if (last_batch_len > 0) {
			CUDA_CALL(cudaMemcpy(uxxs_d + last_batch_idx * uxx_table_pitch,
													 uxxs + sizeof(GaussParams[N / batch_len][batch_len][n]),
													 sizeof(GaussParams[last_batch_len][n]), cudaMemcpyHostToDevice));
		}
	}
	CUDA_CALL(cudaMalloc(&ys_d, sizeof(unsigned[N])));

	{
		dim3 blocks((N + batch_len - 1) / batch_len);
		dim3 threads(n, rules_len, batch_len);
		size_t shared_sz = (/* fsets */ ALIGN_UP(sizeof(GaussParams[fsets_total_len]), SHARED_MEM_ALIGNMENT) +
												/* rules */ ALIGN_UP(sizeof(signed char[rules_len][n+1]), SHARED_MEM_ALIGNMENT) +
												/* uxx batch */ ALIGN_UP(sizeof(GaussParams[batch_len][n]), SHARED_MEM_ALIGNMENT) +
												/* max t-norm */ ALIGN_UP(sizeof(float[batch_len][rules_len][n]), SHARED_MEM_ALIGNMENT) +
												/* fractions */ ALIGN_UP(sizeof(Fraction[batch_len][rules_len]), SHARED_MEM_ALIGNMENT));
		perform_inference_kernel<<<blocks, threads, shared_sz>>>(fsets_total_len, fset_lens_d, fset_offsets_d,
																														 fsets_d, n, rules_d, rules_len,
																														 uxxs_d, uxx_table_pitch, batch_len, ys_d, N);
		cudaDeviceSynchronize();
		CUDA_CALL(cudaPeekAtLastError());
	}

	CUDA_CALL(cudaMemcpy(ys, ys_d, sizeof(unsigned[N]), cudaMemcpyDeviceToHost));

	CUDA_CALL(cudaFree(ys_d));
	CUDA_CALL(cudaFree(uxxs_d));
	CUDA_CALL(cudaFree(rules_d));
	CUDA_CALL(cudaFree(fsets_d));
	CUDA_CALL(cudaFree(fset_offsets_d));
	CUDA_CALL(cudaFree(fset_lens_d));
}


__global__ void compute_scores_kernel(const unsigned *fset_lens, const unsigned *fset_offsets, unsigned n,
																			const GaussEvoParams *fsets_table, size_t fsets_table_pitch, unsigned fsets_total_len,
																			const signed char *rules_table, size_t rules_table_pitch, unsigned rules_len,
																			const GaussParams *uxx_table, size_t uxx_table_pitch, const unsigned *ys,
																			unsigned N, float *scores)
{
	unsigned thread_group_idx = threadIdx.z;
	unsigned batch_len = blockDim.z;
	unsigned batch_idx = blockIdx.x;
	unsigned chromosome_idx = blockIdx.y;
	unsigned idx = batch_idx * batch_len + thread_group_idx;
	unsigned *equals_number = (unsigned*)(void*)scores;

	if (idx < N) {
		const GaussEvoParams *fsets = (const GaussEvoParams*)((const char*)fsets_table + chromosome_idx * fsets_table_pitch);
		const signed char *rules = rules_table + chromosome_idx * rules_table_pitch;
		const GaussParams *uxx_batch = (const GaussParams*)((const char*)uxx_table + batch_idx * uxx_table_pitch);
		unsigned pred = classify_fuzzy(fsets_total_len, fset_lens, fset_offsets,
																	 fsets, rules, uxx_batch);
		// if (threadIdx.x == 0 && threadIdx.y == 0) printf("%03d %03d\n", blockIdx.y, blockIdx.x * blockDim.z + threadIdx.z);
		if (threadIdx.x == 0 && threadIdx.y == 0) {
			atomicAdd(equals_number + chromosome_idx, pred == ys[idx]);
		}
	}
}


__global__ void normalize_scores_kernel(unsigned population_power, float *scores, unsigned N)
{
	unsigned chromosome_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned *equals_number = (unsigned*)(void*)scores;

	if (chromosome_idx < population_power) scores[chromosome_idx] = (float)equals_number[chromosome_idx] / N;
}


static void compute_scores(unsigned population_power,
													 const unsigned *fset_lens_d, const unsigned *fset_offsets_d, unsigned n,
													 const GaussEvoParams *fsets_table_d, size_t fsets_table_pitch, unsigned fsets_total_len,
													 const signed char *rules_table_d, size_t rules_table_pitch, unsigned rules_len,
													 const GaussParams *uxxs_d, size_t uxx_table_pitch, const unsigned *ys_d,
													 unsigned batch_len, unsigned N, float *scores_d)
{
	// static_assert(sizeof(unsigned) == sizeof(float));

	CUDA_CALL(cudaMemset(scores_d, 0, sizeof(float[population_power])));

	{
		dim3 blocks((N + batch_len - 1) / batch_len, population_power);
		dim3 threads(n, rules_len, batch_len);
		size_t shared_sz = (/* fsets */ ALIGN_UP(sizeof(GaussParams[fsets_total_len]), SHARED_MEM_ALIGNMENT) +
												/* rules */ ALIGN_UP(sizeof(signed char[rules_len][n+1]), SHARED_MEM_ALIGNMENT) +
												/* uxx batch */ ALIGN_UP(sizeof(GaussParams[batch_len][n]), SHARED_MEM_ALIGNMENT) +
												/* max t-norm */ ALIGN_UP(sizeof(float[batch_len][rules_len][n]), SHARED_MEM_ALIGNMENT) +
												/* fractions */ ALIGN_UP(sizeof(Fraction[batch_len][rules_len]), SHARED_MEM_ALIGNMENT));
		// printf("compute_score.shared_sz = %d\n", shared_sz);
		compute_scores_kernel<<<blocks, threads, shared_sz>>>(fset_lens_d, fset_offsets_d, n,
																													fsets_table_d, fsets_table_pitch, fsets_total_len,
																													rules_table_d, rules_table_pitch, rules_len,
																													uxxs_d, uxx_table_pitch, ys_d,
																													N, scores_d);
		CUDA_CALL(cudaPeekAtLastError());
	}

	cudaDeviceSynchronize();

	{
		normalize_scores_kernel<<<(N + 255) / 255, 255>>>(population_power, scores_d, N);
		CUDA_CALL(cudaPeekAtLastError());
	}
}


__global__ void init_population_rules_kernel(curandStateMtgp32 *states, unsigned population_power,
																						 const unsigned *fset_lens, unsigned n,
																						 signed char *rules_table, size_t rules_table_pitch, unsigned rules_len)
{
	unsigned i, j, k;
	curandStateMtgp32 *state = states + blockIdx.x;

	for (k = blockIdx.x; k < population_power; k += gridDim.x) {
		signed char *rules = rules_table + k * rules_table_pitch;

		for (j = threadIdx.x; j < rules_len; j += blockDim.x) {
			for (i = 0; i < n; ++i) {
				rules[j * (n+1) + i] = curand(state) % (fset_lens[i] + 1);
			}
			rules[j * (n+1) + n] = curand(state) % fset_lens[n] + 1;
		}
	}
}


static void init_population(curandStateMtgp32 *states_d, unsigned population_power,
														const unsigned *fset_lens, const unsigned *fset_offsets, const unsigned *fset_lens_d, unsigned n,
														GaussEvoParams *fsets_table_d, size_t fsets_table_pitch, unsigned fsets_total_len,
														signed char *rules_table_d, size_t rules_table_pitch, unsigned rules_len)
{
	unsigned i, j;
	GaussEvoParams fsets[fsets_total_len];

	for (i = 0; i < fsets_total_len; ++i) {
		fsets[i].s_mu = 10;
		fsets[i].s_sigma = 10;
	}

	for (i = 0; i < population_power; ++i) {
		CUDA_CALL(cudaMemcpy((GaussEvoParams*)((char*)fsets_table_d + i * fsets_table_pitch), fsets,
												 sizeof(GaussEvoParams[fsets_total_len]), cudaMemcpyHostToDevice));
	}

	{
		unsigned blocks = MIN(population_power, 200);
		unsigned threads = MIN(rules_len, 256);
		init_population_rules_kernel<<<blocks, threads>>>(states_d, population_power,
																											fset_lens_d, n,
																											rules_table_d, rules_table_pitch, rules_len);
		CUDA_CALL(cudaPeekAtLastError());
	}
}


__global__ void perform_selection_kernel(curandStateMtgp32 *states, unsigned population_power,
																				 const float *scores, unsigned *indices, unsigned k = 3)
{
	unsigned idx, i;
	curandStateMtgp32 *state = states + blockIdx.x;

	for (idx = blockIdx.x * blockDim.x + threadIdx.x;
			 idx < population_power;
			 idx += gridDim.x * blockDim.x)
	{
		unsigned max_index = curand(state) % population_power;
		float max_score = scores[max_index];

		for (i = 1; i < k; ++i) {
			unsigned index = curand(state) % population_power;
			float score = scores[index];

			if (score > max_score) {
				max_score = score;
				max_index = index;
			}
		}

		indices[idx] = max_index;
	}
}


static __forceinline void perform_selection(curandStateMtgp32 *states_d, unsigned population_power,
																						const float *scores_d, unsigned *indices_d)
{
	{
		unsigned blocks = MIN((population_power + 255) / 256, 200);
		unsigned threads = 256;
		perform_selection_kernel<<<blocks, threads>>>(states_d, population_power, scores_d, indices_d);
	}
}


__global__ void generate_random_values_kernel(curandStateMtgp32 *states, float *values, unsigned n)
{
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
	curandStateMtgp32 *state = states + blockIdx.x;

	if (idx < n) values[idx] = curand_uniform(state);
}


__global__ void perform_crossingover_kernel(const float *random_values, const unsigned *indices,
																						const GaussEvoParams *fsets_table, GaussEvoParams *new_fsets_table,
																						size_t fsets_table_pitch, unsigned fsets_total_len,
																						const signed char *rules_table, signed char *new_rules_table,
																						size_t rules_table_pitch, unsigned rules_len, unsigned n, float pc)
{
	unsigned i, k = blockIdx.x;

	const GaussEvoParams *fsets_a = (const GaussEvoParams*)((const char*)fsets_table + indices[2*k] * fsets_table_pitch);
	const GaussEvoParams *fsets_b = (const GaussEvoParams*)((const char*)fsets_table + indices[2*k+1] * fsets_table_pitch);
	GaussEvoParams *new_fsets_a = (GaussEvoParams*)((char*)new_fsets_table + (2*k) * fsets_table_pitch);
	GaussEvoParams *new_fsets_b = (GaussEvoParams*)((char*)new_fsets_table + (2*k+1) * fsets_table_pitch);
	const signed char *rules_a = rules_table + indices[2*k] * rules_table_pitch;
	const signed char *rules_b = rules_table + indices[2*k+1] * rules_table_pitch;
	signed char *new_rules_a = new_rules_table + (2*k) * rules_table_pitch;
	signed char *new_rules_b = new_rules_table + (2*k+1) * rules_table_pitch;

	unsigned total_len = fsets_total_len + rules_len * (n+1);

	if (random_values[2*k] < pc) {
		unsigned pos = (unsigned)(random_values[2*k+1] * (total_len-2)) + 1;

		for (i = threadIdx.x; i < fsets_total_len; i += blockDim.x) {
			if (i < pos) {
				new_fsets_a[i] = fsets_a[i];
				new_fsets_b[i] = fsets_b[i];
			} else {
				new_fsets_a[i] = fsets_b[i];
				new_fsets_b[i] = fsets_a[i];
			}
		}
		for (; i < total_len; i += blockDim.x) {
			unsigned j = i - fsets_total_len;
			if (i < pos) {
				new_rules_a[j] = rules_a[j];
				new_rules_b[j] = rules_b[j];
			} else {
				new_rules_a[j] = rules_b[j];
				new_rules_b[j] = rules_a[j];
			}
		}
	} else {
		for (i = threadIdx.x; i < fsets_total_len; i += blockDim.x) {
			new_fsets_a[i] = fsets_a[i];
			new_fsets_b[i] = fsets_b[i];
		}
		for (; i < total_len; i += blockDim.x) {
			unsigned j = i - fsets_total_len;
			new_rules_a[j] = rules_a[j];
			new_rules_b[j] = rules_b[j];
		}
	}
}


static __forceinline void perform_crossingover(curandStateMtgp32 *states_d, float *random_values_d,
																							 const unsigned *indices_d, unsigned population_power,
																							 const GaussEvoParams *fsets_table_d, GaussEvoParams *new_fsets_table_d,
																							 size_t fsets_table_pitch, unsigned fsets_total_len,
																							 const signed char *rules_table_d, signed char *new_rules_table_d, size_t rules_table_pitch,
																							 unsigned rules_len, unsigned n, float pc)
{
	{
		generate_random_values_kernel<<<MIN((population_power + 255) / 256, 200), 256>>>(states_d, random_values_d, population_power);
		CUDA_CALL(cudaPeekAtLastError());
	}

	{
		unsigned total_len = fsets_total_len + rules_len * (n+1);
		perform_crossingover_kernel<<<population_power / 2, MIN(total_len, 1024)>>>(random_values_d, indices_d,
																																								fsets_table_d, new_fsets_table_d,
																																								fsets_table_pitch, fsets_total_len,
																																								rules_table_d, new_rules_table_d,
																																								rules_table_pitch, rules_len, n, pc);
		CUDA_CALL(cudaPeekAtLastError());
	}
}


template <typename T>
__device__ T check_bound(T x, T a, T b, T y)
{
	return (a <= x && x < b) ? x : y;
}


__global__ void perform_fsets_mutation_kernel(curandStateMtgp32 *states, unsigned population_power,
																							GaussEvoParams *fsets_table, size_t fsets_table_pitch, unsigned fsets_total_len,
																							float pm)
{
	unsigned j, k;
	curandStateMtgp32 *state = states + blockIdx.x;

	for (k = blockIdx.x; k < population_power; k += gridDim.x) {
		GaussEvoParams *fsets = (GaussEvoParams*)((char*)fsets_table + k * fsets_table_pitch);

		for (j = threadIdx.x; j < fsets_total_len; j += blockDim.x) {
			float x = check_bound((int)((curand_normal(state)+1.f) * 10), 0, 20, 10);
			float y = check_bound((int)((curand_normal(state)+1.f) * 10), 0, 20, 10);
			if (curand_uniform(state) < pm) fsets[j].s_mu = x;
			if (curand_uniform(state) < pm) fsets[j].s_sigma = y;
		}
	}
}


__global__ void perform_rules_mutation_kernel(curandStateMtgp32 *states, unsigned population_power,
																				unsigned *fset_lens, unsigned n,
																				signed char *rules_table, size_t rules_table_pitch, unsigned rules_len,
																				float pm)
{
	int i, j, k;
	curandStateMtgp32 *state = states + blockIdx.x;

	for (k = blockIdx.x; k < population_power; k += gridDim.x) {
		signed char *rules = rules_table + k * rules_table_pitch;

		for (j = threadIdx.x; j < rules_len; j += blockDim.x) {
			for (i = 0; i < n; ++i) {
				int x = curand(state);
				if (curand_uniform(state) < pm)
					rules[j * (n+1) + i] = x % (fset_lens[i] + 1);
			}
			int y = curand(state);
			if (curand_uniform(state) < pm)
				rules[j * (n+1) + n] = y % fset_lens[n] + 1;
		}
	}
}


static __forceinline void perform_mutation(curandStateMtgp32 *states_d, unsigned population_power,
																					 unsigned *fset_lens_d, unsigned n,
																					 GaussEvoParams *fsets_table_d, size_t fsets_table_pitch, unsigned fsets_total_len,
																					 signed char *rules_table_d, size_t rules_table_pitch, unsigned rules_len,
																					 float pm_fsets, float pm_rules)
{
	{
		unsigned blocks = MIN(population_power, 200);
		unsigned threads = MIN(fsets_total_len, 256);
		perform_fsets_mutation_kernel<<<blocks, threads>>>(states_d, population_power,
																											 fsets_table_d, fsets_table_pitch, fsets_total_len,
																											 pm_fsets);
		CUDA_CALL(cudaPeekAtLastError());
	}

	{
		unsigned blocks = MIN(population_power, 200);
		unsigned threads = MIN(rules_len, 256);
		perform_rules_mutation_kernel<<<blocks, threads>>>(states_d, population_power,
																											 fset_lens_d, n,
																											 rules_table_d, rules_table_pitch, rules_len,
																											 pm_rules);
		CUDA_CALL(cudaPeekAtLastError());
	}
}


typedef struct BestValueLoc {
	float value;
	unsigned index;
} BestValueLoc;


__global__ void find_best_chromosome_kernel(float *scores, unsigned *indices, unsigned population_power,
																						BestValueLoc *best_chromosome_loc)
{
	unsigned step, idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < population_power) {
		indices[idx] = idx;
	}
	for (step = 1; step < population_power; step <<= 1) {
		if (!(idx & ((step<<1)-1)) && idx + step < population_power) {
			if (scores[idx+step] > scores[idx]) {
				scores[idx] = scores[idx+step];
				indices[idx] = indices[idx+step];
			}
		}
		__threadfence();
	}
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		best_chromosome_loc->value = scores[0];
		best_chromosome_loc->index = indices[0];
	}
}


static __forceinline void dump_best_chromosome(float *scores_d, unsigned *indices_d, unsigned population_power,
																							 const GaussEvoParams *fsets_table_d, size_t fsets_table_pitch, unsigned fsets_total_len,
																							 const signed char *rules_table_d, size_t rules_table_pitch, unsigned rules_len, unsigned n,
																							 BestValueLoc *best_chromosome_loc_d, float *last_best_score_ptr,
																							 GaussEvoParams *res_fsets_d, signed char *res_rules_d)
{
	BestValueLoc best_chromosome_loc;

	find_best_chromosome_kernel<<<(population_power + 255) / 256, 256>>>(scores_d, indices_d, population_power, best_chromosome_loc_d);
	CUDA_CALL(cudaMemcpy(&best_chromosome_loc, best_chromosome_loc_d, sizeof(BestValueLoc), cudaMemcpyDeviceToHost));

	if (best_chromosome_loc.value > *last_best_score_ptr) {
		// printf(">Found best chromosome with score = %f\n", best_chromosome_loc.value);
		CUDA_CALL(cudaMemcpy(res_fsets_d, (const GaussEvoParams*)((const char*)fsets_table_d + best_chromosome_loc.index * fsets_table_pitch),
												 sizeof(GaussEvoParams[fsets_total_len]), cudaMemcpyDeviceToDevice));
		CUDA_CALL(cudaMemcpy(res_rules_d, rules_table_d + best_chromosome_loc.index * rules_table_pitch,
												 sizeof(signed char[rules_len][n+1]), cudaMemcpyDeviceToDevice));
		*last_best_score_ptr = best_chromosome_loc.value;
	}
}


static void convert_to_gauss_params(unsigned fsets_total_len, const unsigned *fset_lens, const unsigned *fset_offsets, unsigned n,
																		const GaussEvoParams *gauss_evo_params_d, GaussParams *gauss_params)
{
	unsigned i, j;
	GaussEvoParams gauss_evo_params[fsets_total_len];

	CUDA_CALL(cudaMemcpy(gauss_evo_params, gauss_evo_params_d, sizeof(GaussEvoParams[fsets_total_len]), cudaMemcpyDeviceToHost));

	for (i = 0; i < n+1; ++i) {
		for (j = 0; j < fset_lens[i]; ++j) {
			gauss_params[fset_offsets[i] + j] = make_gauss_params(gauss_evo_params[fset_offsets[i] + j], j, fset_lens[i]);
		}
	}
}


extern "C"
void tune_lfs_gpu_impl(
											 const unsigned *fset_lens, unsigned n, unsigned rules_len,
											 const GaussParams *uxxs, const unsigned *ys, unsigned N,
											 unsigned population_power, unsigned iterations_number,
											 GaussParams *fsets, signed char *rules)
{
	init_device_props();

	unsigned i, k, it;
	unsigned fsets_total_len = 0, fset_offsets[n+1];

	for (i = 0; i < n+1; ++i) {
		fset_offsets[i] = fsets_total_len;
		fsets_total_len += fset_lens[i];
	}

	/* Device memory map:
	 *
	 * +----------------------------------
   * |
	 * +----------------------------------
	 */

	unsigned *fset_lens_d, *fset_offsets_d;

	CUDA_CALL(cudaMalloc(&fset_lens_d, sizeof(unsigned[n+1])));
	CUDA_CALL(cudaMemcpy(fset_lens_d, fset_lens, sizeof(unsigned[n+1]), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMalloc(&fset_offsets_d, sizeof(unsigned[n+1])));
	CUDA_CALL(cudaMemcpy(fset_offsets_d, fset_offsets, sizeof(unsigned[n+1]), cudaMemcpyHostToDevice));

	curandStateMtgp32 *states_d;
	mtgp32_kernel_params *mtgp_kernel_params_d;

	CUDA_CALL(cudaMalloc(&mtgp_kernel_params_d, sizeof(mtgp32_kernel_params)));
	CUDA_CALL(cudaMalloc(&states_d, sizeof(curandStateMtgp32[200])));
	CURAND_CALL(curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, mtgp_kernel_params_d));
	CURAND_CALL(curandMakeMTGP32KernelState(states_d, mtgp32dc_params_fast_11213, mtgp_kernel_params_d, 200, 42)); // time(0)));

	GaussEvoParams *fsets_table_d, *new_fsets_table_d;
	signed char *rules_table_d, *new_rules_table_d;
	GaussParams *uxxs_d;
	unsigned *ys_d, *indices_d;
	float *scores_d;
	size_t fsets_table_pitch, rules_table_pitch, uxx_table_pitch;
	unsigned batch_len = MIN(N, 1);

	CUDA_CALL(cudaMallocPitch(&fsets_table_d, &fsets_table_pitch, sizeof(GaussEvoParams[fsets_total_len]), population_power));
	CUDA_CALL(cudaMallocPitch(&new_fsets_table_d, &fsets_table_pitch, sizeof(GaussEvoParams[fsets_total_len]), population_power));
	CUDA_CALL(cudaMallocPitch(&rules_table_d, &rules_table_pitch, sizeof(signed char[rules_len*(n+1)]), population_power));
	CUDA_CALL(cudaMallocPitch(&new_rules_table_d, &rules_table_pitch, sizeof(signed char[rules_len*(n+1)]), population_power));
	// {
	// 	CUDA_CALL(cudaMalloc(&uxxs_d, sizeof(GaussParams[N][n])));
	// 	CUDA_CALL(cudaMemcpy(uxxs_d, uxxs, sizeof(GaussParams[N][n]), cudaMemcpyHostToDevice));
	// 	CUDA_CALL(cudaMalloc(&ys_d, sizeof(unsigned[N])));
	// 	CUDA_CALL(cudaMemcpy(ys_d, ys, sizeof(unsigned[N]), cudaMemcpyHostToDevice));
	// 	uxx_table_pitch = sizeof(GaussParams[batch_len][n]);
	// 	CUDA_CALL(cudaBindTexture(NULL, uxxs_tex, uxxs_d, sizeof(GaussParams[N][n])));
	// 	CUDA_CALL(cudaBindTexture(NULL, ys_tex, ys_d, sizeof(unsigned[N])));
	// }
	{
		CUDA_CALL(cudaMallocPitch(&uxxs_d, &uxx_table_pitch, sizeof(GaussParams[batch_len][n]), (N + batch_len - 1) / batch_len));
		{
			unsigned last_batch_len = N % batch_len, last_batch_idx = N / batch_len;
			CUDA_CALL(cudaMemcpy2D(uxxs_d, uxx_table_pitch, uxxs, sizeof(GaussParams[batch_len][n]),
														 sizeof(GaussParams[batch_len][n]), N / batch_len, cudaMemcpyHostToDevice));
			if (last_batch_len > 0) {
				CUDA_CALL(cudaMemcpy(uxxs_d + last_batch_idx * uxx_table_pitch,
														 uxxs + sizeof(GaussParams[N / batch_len][batch_len][n]),
														 sizeof(GaussParams[last_batch_len][n]), cudaMemcpyHostToDevice));
			}
		}
		CUDA_CALL(cudaMalloc(&ys_d, sizeof(unsigned[N])));
		CUDA_CALL(cudaMemcpy(ys_d, ys, sizeof(unsigned[N]), cudaMemcpyHostToDevice));
	}
	CUDA_CALL(cudaMalloc(&scores_d, sizeof(float[population_power])));
	CUDA_CALL(cudaMalloc(&indices_d, sizeof(unsigned[population_power])));

	float last_best_score = 0.f;
	BestValueLoc *best_chromosome_loc_d;
	float *scores_copy_d;
	GaussEvoParams *res_fsets_d;
	signed char *res_rules_d;

	CUDA_CALL(cudaMalloc(&best_chromosome_loc_d, sizeof(BestValueLoc)));
	CUDA_CALL(cudaMalloc(&scores_copy_d, sizeof(float[population_power])));
	CUDA_CALL(cudaMalloc(&res_fsets_d, sizeof(GaussEvoParams[fsets_total_len])));
	CUDA_CALL(cudaMalloc(&res_rules_d, sizeof(signed char[rules_len][n+1])));

	init_population(states_d, population_power,
									fset_lens, fset_offsets, fset_lens_d, n,
									fsets_table_d, fsets_table_pitch, fsets_total_len,
									rules_table_d, rules_table_pitch, rules_len);

	float pc = 0.9f, pm_fsets = 1.f / (fsets_total_len * 20), pm_rules = 1.f / (rules_len * n);

	for (it = 0; it < iterations_number; ++it)
	{
		compute_scores(population_power,
									 fset_lens_d, fset_offsets_d, n,
									 fsets_table_d, fsets_table_pitch, fsets_total_len,
									 rules_table_d, rules_table_pitch, rules_len,
									 uxxs_d, uxx_table_pitch, ys_d, batch_len, N, scores_d);

		CUDA_CALL(cudaMemcpy(scores_copy_d, scores_d, sizeof(float[population_power]), cudaMemcpyDeviceToDevice));

		dump_best_chromosome(scores_copy_d, indices_d, population_power,
												 fsets_table_d, fsets_table_pitch, fsets_total_len,
												 rules_table_d, rules_table_pitch, rules_len, n,
												 best_chromosome_loc_d, &last_best_score, res_fsets_d, res_rules_d);

		CUDA_CALL(cudaDeviceSynchronize());

		// {
		// 	printf("[%3u] ", it);
		// 	float scores[population_power];
		// 	CUDA_CALL(cudaMemcpy(scores, scores_d, sizeof(float[population_power]), cudaMemcpyDeviceToHost));

		// 	float min = 1.f, max = 0.f;
		// 	double sum = 0.0;
		// 	for (i = 0; i < population_power; ++i) {
		// 		min = MIN(min, scores[i]);
		// 		max = MAX(max, scores[i]);
		// 		sum += scores[i];
		// 		// printf("%.2f ", scores[i]);
		// 	}
		// 	float avg = sum / population_power;
		// 	printf("Min score: %f Max score: %f Avg score: %f", min, max, avg);
		// 	printf("\n");
		// }

		perform_selection(states_d, population_power, scores_d, indices_d);

		perform_crossingover(states_d, scores_d, indices_d, population_power,
												 fsets_table_d, new_fsets_table_d, fsets_table_pitch, fsets_total_len,
												 rules_table_d, new_rules_table_d, rules_table_pitch, rules_len,
												 n, pc);

		perform_mutation(states_d, population_power, fset_lens_d, n,
										 new_fsets_table_d, fsets_table_pitch, fsets_total_len,
										 new_rules_table_d, rules_table_pitch, rules_len, pm_fsets, pm_rules);

		SWAP(fsets_table_d, new_fsets_table_d);
		SWAP(rules_table_d, new_rules_table_d);
	}

	convert_to_gauss_params(fsets_total_len, fset_lens, fset_offsets, n, res_fsets_d, fsets);
	CUDA_CALL(cudaMemcpy(rules, res_rules_d, sizeof(signed char[rules_len][n+1]), cudaMemcpyDeviceToHost));

	CUDA_CALL(cudaFree(res_rules_d));
	CUDA_CALL(cudaFree(res_fsets_d));
	CUDA_CALL(cudaFree(best_chromosome_loc_d));
	CUDA_CALL(cudaFree(indices_d));
	CUDA_CALL(cudaFree(scores_copy_d));
	CUDA_CALL(cudaFree(scores_d));
	CUDA_CALL(cudaFree(ys_d));
  CUDA_CALL(cudaFree(uxxs_d));
	CUDA_CALL(cudaFree(new_rules_table_d));
	CUDA_CALL(cudaFree(rules_table_d));
	CUDA_CALL(cudaFree(new_fsets_table_d));
	CUDA_CALL(cudaFree(fsets_table_d));
	CUDA_CALL(cudaFree(mtgp_kernel_params_d));
	CUDA_CALL(cudaFree(states_d));
	CUDA_CALL(cudaFree(fset_offsets_d));
	CUDA_CALL(cudaFree(fset_lens_d));
}

