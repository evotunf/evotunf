#include <utility>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>
#include "common.h"
// #include "random.h"
#include "evolutionary_tune.h"


#define CUDA_CALL(...)																									\
	do {																																	\
		cudaError_t ret = (__VA_ARGS__);																		\
		if (ret != cudaSuccess) fprintf(stderr, "CUDA: Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(ret)); \
	} while (0)

#define CURAND_CALL(...)																								\
	do {																																	\
		curandStatus_t ret = (__VA_ARGS__);																	\
		if (ret != CURAND_STATUS_SUCCESS) fprintf(stderr, "CUDA: Error at %s:%d: code #%d\n", __FILE__, __LINE__, ret); \
	} while (0)

#define SHARED_MEM_ALIGNMENT 256

#define SIGN(x) ((typeof(x))(((x) < 0) ? -1 : 1))
#define GAUSS(mu, sigma, x) exp(-pow(((x) - (mu)) / (sigma), 2))
#define TNORM(a, b) fminf(a, b)
#define LUKASZEWICZ_IMPL(a, b) fminf(1.f, 1.f - (a) + (b))

typedef struct __align__(16) Fraction {
	double numerator, denominator;
} Fraction;


__device__ unsigned classify_fuzzy_kernel(unsigned fsets_total_len, unsigned b_fset_len, const unsigned *fset_offsets,
																					const GaussParams *fsets, unsigned n,
																					const signed char *rules, unsigned rules_len,
																					const GaussParams *uxx_batch)
{
	unsigned j, step;
	unsigned batch_len = blockDim.z;
	unsigned thread_group_idx = threadIdx.z;
	const unsigned k = threadIdx.y;
	const unsigned i = threadIdx.x;

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
		unsigned idx = threadIdx.y * blockDim.x + threadIdx.x;
		if (idx < fsets_total_len) fsets_cache[idx] = fsets[idx];
		rules_cache[k * (n+1) + i] = rules[k * (n+1) + i];
		if (threadIdx.x == 0) rules_cache[k * (n+1) + n] = rules[k * (n+1) + n];
	}
	if (threadIdx.y == 0) uxx_cache[i] = uxx_batch[thread_group_idx * n + i];

	__syncthreads();

	signed char y = rules_cache[k * (n+1) + n];
	float uy_center = fsets_cache[fset_offsets[n] + (y-1)].mu;
	GaussParams ux = uxx_cache[i];

	float cross = 1.f;

	for (j = 0; j < rules_len; ++j) {
		// if (k == 0) printf("%d\n", rules[j * (n+1) + i]);
		signed char a = rules_cache[j * (n+1) + i];
		signed char b = rules_cache[j * (n+1) + n];

		float max_tnorm = 0.0f;

		if (a) {
			GaussParams ua = fsets_cache[fset_offsets[i] + (a-1)];
			GaussParams ub = fsets_cache[fset_offsets[n] + (b-1)];
			float ub_value = GAUSS(ub.mu, ub.sigma, uy_center);

			for (float t = 0.0f; t < 1.01f; t += 0.05f)
			{
				float tnorm = TNORM(GAUSS(ux.mu, ux.sigma, t), IMPL(GAUSS(ua.mu, ua.sigma, t), ub_value));
				max_tnorm = fmaxf(max_tnorm, tnorm);
			}
		}
		// if (blockIdx.y == 7) printf("%d %d %d %d %f\n", blockIdx.x * batch_len + threadIdx.z, threadIdx.y, j, threadIdx.x, max_tnorm);
		max_tnorm_cache[i] = max_tnorm;
		__syncthreads();
		for (step = 1; step < n; step <<= 1) {
			if (!(i & ((step<<1)-1)) && i + step < n) {
				max_tnorm_cache[i] = fmaxf(max_tnorm_cache[i], max_tnorm_cache[i + step]);
			}
			__syncthreads();
		}
		if (threadIdx.x == 0 && max_tnorm_cache[0] > 0.f) cross = fminf(cross, max_tnorm_cache[0]);
	}

	if (threadIdx.x == 0) {
		// printf("%f\n", cross);
		fractions_cache[k].numerator = uy_center * cross;
		fractions_cache[k].denominator = cross;
	}

	__syncthreads();

	if (threadIdx.x == 0) {
		for (step = 1; step < rules_len; step <<= 1) {
			if (!(k & ((step<<1)-1)) && k + step < rules_len) {
				fractions_cache[k].numerator += fractions_cache[k + step].numerator;
				fractions_cache[k].denominator += fractions_cache[k + step].denominator;
			}
			__syncthreads();
		}
	}

	if (threadIdx.x == 0 && threadIdx.y == 0) {
		float y = fractions_cache[0].numerator / fractions_cache[0].denominator;
		float max_value = 0.f;
		unsigned max_index = 0;

		for (j = 0; j < b_fset_len; ++j) {
			GaussParams ub = fsets_cache[fset_offsets[n] + j];
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


__global__ void perform_inference_kernel(unsigned fsets_total_len, unsigned b_fset_len, const unsigned *fset_offsets,
																				 const GaussParams *fsets, unsigned n,
																				 const signed char *rules, unsigned rules_len,
																				 const GaussParams *uxx_table, size_t uxx_table_pitch, unsigned batch_len, unsigned *ys, unsigned N)
{
	unsigned batch_idx = blockIdx.x;
	unsigned thread_group_idx = threadIdx.z;
	unsigned idx = batch_idx * batch_len + thread_group_idx;

	if (idx < N) {
		const GaussParams *uxx_batch = (GaussParams*) ((char*)uxx_table + batch_idx * uxx_table_pitch);
		unsigned pred = classify_fuzzy_kernel(fsets_total_len, b_fset_len, fset_offsets,
																					fsets, n, rules, rules_len, uxx_batch);
		if (threadIdx.y == 0 && threadIdx.x == 0) {
			ys[idx] = pred;
		}
	}
}


void predict_gpu_impl(const unsigned *fset_lens, const GaussParams *fsets, unsigned n,
											const signed char *rules, unsigned rules_len,
											const GaussParams *uxxs, unsigned *ys, unsigned N)
{
	unsigned i;
	unsigned fset_offsets[n+1], fsets_total_len = 0;

	for (i = 0; i < n+1; ++i) {
		fset_offsets[i] = fsets_total_len;
		fsets_total_len += fset_lens[i];
	}
	
	unsigned *fset_offsets_d;
	GaussParams *fsets_d, *uxx_table_d;
	signed char *rules_d;
	unsigned *ys_d;
	size_t uxx_table_pitch;
	unsigned batch_len = 2;

	CUDA_CALL(cudaPeekAtLastError());
	CUDA_CALL(cudaMalloc(&fset_offsets_d, sizeof(unsigned[n+1])));
	CUDA_CALL(cudaMemcpy(fset_offsets_d, fset_offsets, sizeof(unsigned[n+1]), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMalloc(&fsets_d, sizeof(GaussParams[fsets_total_len])));
	CUDA_CALL(cudaMemcpy(fsets_d, fsets, sizeof(GaussParams[fsets_total_len]), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMalloc(&rules_d, sizeof(signed char[rules_len][n+1])));
	CUDA_CALL(cudaMemcpy(rules_d, rules, sizeof(signed char[rules_len][n+1]), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMallocPitch(&uxx_table_d, &uxx_table_pitch, sizeof(GaussParams[batch_len][n]), (N + batch_len - 1) / batch_len));
	{
		unsigned last_batch_len = N % batch_len, last_batch_idx = N / batch_len;
		CUDA_CALL(cudaMemcpy2D(uxx_table_d, uxx_table_pitch, uxxs, sizeof(GaussParams[batch_len][n]),
													 sizeof(GaussParams[batch_len][n]), N / batch_len, cudaMemcpyHostToDevice));
		if (last_batch_len > 0) {
			CUDA_CALL(cudaMemcpy(uxx_table_d + last_batch_idx * uxx_table_pitch,
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
		perform_inference_kernel<<<blocks, threads, shared_sz>>>(fsets_total_len, fset_lens[n], fset_offsets_d,
																														 fsets_d, n, rules_d, rules_len,
																														 uxx_table_d, uxx_table_pitch, batch_len, ys_d, N);
		cudaDeviceSynchronize();
		CUDA_CALL(cudaPeekAtLastError());

	}

	CUDA_CALL(cudaMemcpy(ys, ys_d, sizeof(unsigned[N]), cudaMemcpyDeviceToHost));

	CUDA_CALL(cudaFree(ys_d));
	CUDA_CALL(cudaFree(uxx_table_d));
	CUDA_CALL(cudaFree(rules_d));
	CUDA_CALL(cudaFree(fsets_d));
	CUDA_CALL(cudaFree(fset_offsets_d));
}


__global__ void compute_scores_kernel(unsigned fsets_total_len, unsigned b_fset_len, const unsigned *fset_offsets,
																			const GaussParams *fsets, unsigned n,
																			const signed char *rules_table, size_t rules_table_pitch, unsigned rules_len,
																			const GaussParams *uxx_table, size_t uxx_table_pitch, const unsigned *ys,
																			unsigned batch_len, unsigned N, float *scores)
{
	unsigned chromosome_idx = blockIdx.y;
	unsigned batch_idx = blockIdx.x;
	unsigned thread_group_idx = threadIdx.z;
	unsigned idx = batch_idx * batch_len + thread_group_idx;
	unsigned *equals_number = (unsigned*)(void*)scores;

	if (idx < N) {
		const signed char *rules = (signed char*)(rules_table + chromosome_idx * rules_table_pitch);
		const GaussParams *uxx_batch = (GaussParams*)((char*)uxx_table + batch_idx * uxx_table_pitch);
		unsigned pred = classify_fuzzy_kernel(fsets_total_len, b_fset_len, fset_offsets,
																					fsets, n, rules, rules_len, uxx_batch);
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
													 unsigned fsets_total_len, const unsigned *fset_lens, const unsigned *fset_offsets_d,
													 const GaussParams *fsets_d, unsigned n,
													 const signed char *rules_table_d, size_t rules_table_pitch, unsigned rules_len,
													 const GaussParams *uxx_table_d, size_t uxx_table_pitch, const unsigned *ys_d,
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
		compute_scores_kernel<<<blocks, threads, shared_sz>>>(fsets_total_len, fset_lens[n], fset_offsets_d,
																													fsets_d, n,
																													rules_table_d, rules_table_pitch, rules_len,
																													uxx_table_d, uxx_table_pitch, ys_d,
																													batch_len, N, scores_d);
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
		signed char *rules = (signed char*) (rules_table + k * rules_table_pitch);

		for (j = threadIdx.x; j < rules_len; j += blockDim.x) {
			for (i = 0; i < n; ++i) {
				rules[j * (n+1) + i] = curand(state) % (fset_lens[i] + 1);
			}
			rules[j * (n+1) + n] = curand(state) % fset_lens[n] + 1;
		}
	}
}


static void init_population(curandStateMtgp32 *states_d, unsigned population_power,
														unsigned fsets_total_len, const unsigned *fset_lens, const unsigned *fset_offsets, const unsigned *fset_lens_d,
														GaussParams *fsets_d, unsigned n,
														signed char *rules_table_d, size_t rules_table_pitch, unsigned rules_len)
{
	unsigned i, j;
	GaussParams fsets[fsets_total_len];

	for (i = 0; i < n+1; ++i) {
		for (j = 0; j < fset_lens[i]; ++j) {
			fsets[fset_offsets[i] + j].mu = (float)(j+1) / (fset_lens[i]+1);
			fsets[fset_offsets[i] + j].sigma = 1.f / (fset_lens[i]+1);
		}
	}

	CUDA_CALL(cudaMemcpy(fsets_d, fsets, sizeof(GaussParams[fsets_total_len]), cudaMemcpyHostToDevice));

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


__global__ void perform_rules_crossingover_kernel(const float *random_values, const unsigned *indices,
																									const signed char *rules_table, signed char *new_rules_table,
																									size_t rules_table_pitch, unsigned rules_len, unsigned n, float pc)
{
	unsigned i, k = blockIdx.x;

	const signed char *rules_a = (signed char*)rules_table + indices[2*k] * rules_table_pitch;
	const signed char *rules_b = (signed char*)rules_table + indices[2*k+1] * rules_table_pitch;
	signed char *new_rules_a = (signed char*)new_rules_table + (2*k) * rules_table_pitch;
	signed char *new_rules_b = (signed char*)new_rules_table + (2*k+1) * rules_table_pitch;

	unsigned total_len = rules_len * (n+1);
	
	if (random_values[2*k] < pc) {
		unsigned pos = (unsigned)(random_values[2*k+1] * (total_len-2)) + 1;

		for (i = threadIdx.x; i < total_len; i += blockDim.x) {
			if (i < pos) {
				new_rules_a[i] = rules_a[i];
				new_rules_b[i] = rules_b[i];
			} else {
				new_rules_a[i] = rules_b[i];
				new_rules_b[i] = rules_a[i];
			}
		}
	} else {
		for (i = threadIdx.x; i < total_len; i += blockDim.x) {
			new_rules_a[i] = rules_a[i];
			new_rules_b[i] = rules_b[i];
		}
	}
}


static __forceinline void perform_crossingover(curandStateMtgp32 *states_d, float *random_values_d, const unsigned *indices_d,
																							 unsigned population_power, const signed char *rules_table_d, signed char *new_rules_table_d,
																							 size_t rules_table_pitch, unsigned rules_len, unsigned n, float pc)
{
	{
		generate_random_values_kernel<<<MIN((population_power + 255) / 256, 200), 256>>>(states_d, random_values_d, population_power);
		CUDA_CALL(cudaPeekAtLastError());
	}

	{
		perform_rules_crossingover_kernel<<<population_power / 2, MIN(rules_len*(n+1), 1024)>>>(random_values_d, indices_d,
																																														rules_table_d, new_rules_table_d,
																																														rules_table_pitch, rules_len, n, pc);
		CUDA_CALL(cudaPeekAtLastError());
	}
}


__global__ void perform_rules_mutation_kernel(curandStateMtgp32 *states, unsigned population_power,
																							unsigned *fset_lens, unsigned n,
																							signed char *rules_table, size_t rules_table_pitch, unsigned rules_len, float pm)
{
	unsigned i, j, k;
	curandStateMtgp32 *state = states + blockIdx.x;

	for (k = blockIdx.x; k < population_power; k += gridDim.x) {
		signed char *rules = (signed char*)(rules_table + k * rules_table_pitch);

		for (j = threadIdx.x; j < rules_len; j += blockDim.x) {
			for (i = 0; i < n; ++i) {
				if (curand_uniform(state) < pm) rules[j * (n+1) + i] = curand(state) % (fset_lens[i] + 1);
			}
			if (curand_uniform(state) < pm) rules[j * (n+1) + n] = curand(state) % fset_lens[n] + 1;
		}
	}
}


static __forceinline void perform_mutation(curandStateMtgp32 *states_d, unsigned population_power,
																					 unsigned *fset_lens_d, unsigned n,
																					 signed char *rules_table_d, size_t rules_table_pitch, unsigned rules_len, float pm)
{
	{
		unsigned blocks = MIN(population_power, 200);
		unsigned threads = MIN(ALIGN_UP(rules_len, 32), 256);
		perform_rules_mutation_kernel<<<blocks, threads>>>(states_d, population_power,
																											 fset_lens_d, n,
																											 rules_table_d, rules_table_pitch, rules_len, pm);
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
	}
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		best_chromosome_loc->value = scores[0];
		best_chromosome_loc->index = indices[0];
	}
}
	

static __forceinline void dump_best_chromosome(float *scores_d, unsigned *indices_d, unsigned population_power,
																							 const GaussParams *fsets_d, unsigned fsets_total_len,
																							 const signed char *rules_table_d, size_t rules_table_pitch, unsigned rules_len, unsigned n,
																							 BestValueLoc *best_chromosome_loc_d, float *last_best_score_ptr,
																							 GaussParams *res_fsets_d, signed char *res_rules_d)
{
	BestValueLoc best_chromosome_loc;
	
	find_best_chromosome_kernel<<<(population_power + 255) / 256, 256>>>(scores_d, indices_d, population_power, best_chromosome_loc_d);
	CUDA_CALL(cudaMemcpy(&best_chromosome_loc, best_chromosome_loc_d, sizeof(BestValueLoc), cudaMemcpyDeviceToHost));
	
	if (best_chromosome_loc.value > *last_best_score_ptr) {
		CUDA_CALL(cudaMemcpy(res_fsets_d, fsets_d, sizeof(GaussParams[fsets_total_len]), cudaMemcpyDeviceToDevice));
		CUDA_CALL(cudaMemcpy(res_rules_d, rules_table_d + best_chromosome_loc.index * rules_table_pitch,
												 sizeof(signed char[rules_len][n+1]), cudaMemcpyDeviceToDevice));
		*last_best_score_ptr = best_chromosome_loc.value;
	}
}


extern "C"
void tune_lfs_gpu_impl(
											 const unsigned *fset_lens, unsigned n, unsigned rules_len,
											 const GaussParams *uxxs, const unsigned *ys, unsigned N,
											 unsigned population_power, unsigned iterations_number,
											 GaussParams *fsets, signed char *rules)
{
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
	CUDA_CALL(cudaMalloc(&fset_offsets_d, sizeof(unsigned[n+1])));
	CUDA_CALL(cudaMemcpy(fset_lens_d, fset_lens, sizeof(unsigned[n+1]), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(fset_offsets_d, fset_offsets, sizeof(unsigned[n+1]), cudaMemcpyHostToDevice));

	curandStateMtgp32 *states_d;
	mtgp32_kernel_params *mtgp_kernel_params_d;

	CUDA_CALL(cudaMalloc(&mtgp_kernel_params_d, sizeof(mtgp32_kernel_params)));
	CUDA_CALL(cudaMalloc(&states_d, sizeof(curandStateMtgp32[64])));
	CURAND_CALL(curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, mtgp_kernel_params_d));
	CURAND_CALL(curandMakeMTGP32KernelState(states_d, mtgp32dc_params_fast_11213, mtgp_kernel_params_d, 64, 42)); // time(0)));

	GaussParams *fsets_d, *uxx_table_d;
	signed char *rules_table_d, *new_rules_table_d;
	unsigned *ys_d, *indices_d;
	float *scores_d;
	size_t rules_table_pitch, uxx_table_pitch;
	unsigned batch_len = MIN(N, 2);

	CUDA_CALL(cudaMalloc(&fsets_d, sizeof(GaussParams[fsets_total_len])));
	CUDA_CALL(cudaMallocPitch(&rules_table_d, &rules_table_pitch, sizeof(signed char[rules_len*(n+1)]), population_power));
	CUDA_CALL(cudaMallocPitch(&new_rules_table_d, &rules_table_pitch, sizeof(signed char[rules_len*(n+1)]), population_power));
	CUDA_CALL(cudaMallocPitch(&uxx_table_d, &uxx_table_pitch, sizeof(GaussParams[batch_len][n]), (N + batch_len - 1) / batch_len));
	{
		unsigned last_batch_len = N % batch_len, last_batch_idx = N / batch_len;
		CUDA_CALL(cudaMemcpy2D(uxx_table_d, uxx_table_pitch, uxxs, sizeof(GaussParams[batch_len][n]),
													 sizeof(GaussParams[batch_len][n]), N / batch_len, cudaMemcpyHostToDevice));
		if (last_batch_len > 0) {
			CUDA_CALL(cudaMemcpy(uxx_table_d + last_batch_idx * uxx_table_pitch,
													 uxxs + sizeof(GaussParams[N / batch_len][batch_len][n]),
													 sizeof(GaussParams[last_batch_len][n]), cudaMemcpyHostToDevice));
		}
	}
	CUDA_CALL(cudaMalloc(&ys_d, sizeof(unsigned[N])));
	CUDA_CALL(cudaMemcpy(ys_d, ys, sizeof(unsigned[N]), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMalloc(&scores_d, sizeof(float[population_power])));
	CUDA_CALL(cudaMalloc(&indices_d, sizeof(unsigned[population_power])));

	float last_best_score = 0.f;
	BestValueLoc *best_chromosome_loc_d;
	float *scores_copy_d;
	GaussParams *res_fsets_d;
	signed char *res_rules_d;

	CUDA_CALL(cudaMalloc(&best_chromosome_loc_d, sizeof(BestValueLoc)));
	CUDA_CALL(cudaMalloc(&scores_copy_d, sizeof(float[population_power])));
	CUDA_CALL(cudaMalloc(&res_fsets_d, sizeof(GaussParams[fsets_total_len])));
	CUDA_CALL(cudaMalloc(&res_rules_d, sizeof(signed char[rules_len][n+1])));

	init_population(states_d, population_power,
									fsets_total_len, fset_lens, fset_offsets, fset_lens_d,
									fsets_d, n,
									rules_table_d, rules_table_pitch, rules_len);

	float pc = 0.9f, pm = 1.f / (rules_len * n);

	for (it = 0; it < iterations_number; ++it)
	{
		compute_scores(population_power,
									 fsets_total_len, fset_lens, fset_offsets_d,
									 fsets_d, n,
									 rules_table_d, rules_table_pitch, rules_len,
									 uxx_table_d, uxx_table_pitch, ys_d, batch_len, N, scores_d);

		CUDA_CALL(cudaMemcpy(scores_copy_d, scores_d, sizeof(float[population_power]), cudaMemcpyDeviceToDevice));

		dump_best_chromosome(scores_copy_d, indices_d, population_power,
												 fsets_d, fsets_total_len,
												 rules_table_d, rules_table_pitch, rules_len, n,
												 best_chromosome_loc_d, &last_best_score, res_fsets_d, res_rules_d);

		CUDA_CALL(cudaDeviceSynchronize());

		{
			printf("[%3u] ", it);
			float scores[population_power];
			CUDA_CALL(cudaMemcpy(scores, scores_d, sizeof(float[population_power]), cudaMemcpyDeviceToHost));

			float min = 1.f, max = 0.f;
			double sum = 0.0;
			for (i = 0; i < population_power; ++i) {
				min = MIN(min, scores[i]);
				max = MAX(max, scores[i]);
				sum += scores[i];
				// printf("%f ", scores[i]);
			}
			float avg = sum / population_power;
			printf("Min score: %f Max score: %f Avg score: %f", min, max, avg);
			printf("\n");
		}
		
		perform_selection(states_d, population_power, scores_d, indices_d);

		// printf("----676\n");
		perform_crossingover(states_d, scores_d, indices_d, population_power,
												 rules_table_d, new_rules_table_d, rules_table_pitch,
												 rules_len, n, pc);

		// printf("-------\n");
		perform_mutation(states_d, population_power, fset_lens_d, n,
										 new_rules_table_d, rules_table_pitch, rules_len, pm);

		std::swap(rules_table_d, new_rules_table_d);
	}
	
	CUDA_CALL(cudaMemcpy(fsets, res_fsets_d, sizeof(GaussParams[fsets_total_len]), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(rules, res_rules_d, sizeof(signed char[rules_len][n+1]), cudaMemcpyDeviceToHost));

	CUDA_CALL(cudaFree(res_rules_d));
	CUDA_CALL(cudaFree(res_fsets_d));
	CUDA_CALL(cudaFree(best_chromosome_loc_d));
	CUDA_CALL(cudaFree(indices_d));
	CUDA_CALL(cudaFree(scores_copy_d));
	CUDA_CALL(cudaFree(scores_d));
	CUDA_CALL(cudaFree(ys_d));
	CUDA_CALL(cudaFree(uxx_table_d));
	CUDA_CALL(cudaFree(new_rules_table_d));
	CUDA_CALL(cudaFree(rules_table_d));
	CUDA_CALL(cudaFree(fsets_d));
	CUDA_CALL(cudaFree(mtgp_kernel_params_d));
	CUDA_CALL(cudaFree(states_d));
	CUDA_CALL(cudaFree(fset_offsets_d));
	CUDA_CALL(cudaFree(fset_lens_d));
}

