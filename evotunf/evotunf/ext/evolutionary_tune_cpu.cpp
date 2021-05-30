#include <utility>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "random.h"
#include "ga_params.h"
#include "evolutionary_tune.h"


#define SIGN(x) ((typeof(x))(((x) < 0) ? -1 : 1))
#define GAUSS(mu, sigma, x) exp(-pow(((x) - (mu)) / (sigma), 2))
#define INVGAUSS_X1(mu, sigma, y) ((mu) - (sigma)*sqrt(-log(y)))
#define INVGAUSS_X2(mu, sigma, y) ((mu) + (sigma)*sqrt(-log(y)))
#define LUKASZEWICZ_IMPL(a, b) fmin(1.f, 1.f - (a) + (b))
#define ALIEW_IMPL(a, b) (((a) > (b)) ? fmin(1.f - (a), (b)) : 1.f)

typedef struct GaussEvoParams {
	char s_mu, s_sigma;
} GaussEvoParams;

typedef struct Chromosome {
  GaussEvoParams* fsets;
  signed char* rules;
} Chromosome;


#define S_MU(i, n, s) (((i + 1) + (float)(s-10)/20) / (n+1))
#define S_SIGMA(n, s) ((float)s / 10 / (n+1))

static void init_population(const unsigned *fset_offsets, const unsigned *fset_lens,
														unsigned fsets_total_len, unsigned n, unsigned rules_len,
														Chromosome *population, unsigned population_power)
{
  unsigned i, j, k;

  /* static signed char true_rules[][7] = */
  /*   { */
  /*    {0,0,0,0,0,2,2}, */
  /*     {2,0,0,0,0,1,1}, */
  /*     {1,2,0,0,0,1,1}, */
  /*     {1,1,0,0,0,1,1}, */
  /*     {1,3,2,2,0,1,1}, */
  /*     {0,0,0,0,4,1,1}, */
  /*     {1,4,0,0,1,1,2}, */
  /*     {1,4,0,0,2,1,2}, */
  /*     {1,4,0,0,3,1,2}, */
  /*     {1,3,1,1,1,1,2}, */
  /*     {1,3,1,1,2,1,2}, */
  /*     {1,3,1,2,1,1,2}, */
  /*     {1,3,1,2,2,1,2}, */
  /*     {1,3,1,1,3,1,1}, */
  /*     {1,3,1,2,3,1,2} */
  /*    }; */

  for (k = 0; k < population_power; ++k) {
    GaussEvoParams *fsets = population[k].fsets = (GaussEvoParams*)malloc(sizeof(GaussEvoParams[fsets_total_len]));
    signed char *rules = population[k].rules = (signed char*)malloc(ALIGN_UP(sizeof(signed char[rules_len*(n+1)]), 8));

    for (i = 0; i < n+1; ++i) {
      for (j = 0; j < fset_lens[i]; ++j) {
				/* fsets[fset_offsets[i] + j].mu = (float)(j+1) / (fset_lens[i]+1); // rnd_prob(); */
				/* fsets[fset_offsets[i] + j].sigma = 1.f / (fset_lens[i]+1); */
				fsets[fset_offsets[i] + j].s_mu = 10;
				fsets[fset_offsets[i] + j].s_sigma = 10;
      }
    }

    for (j = 0; j < rules_len; ++j) {
      for (i = 0; i < n; ++i) {
				/* rules[j * (n+1) + i] = true_rules[j][i]; */
				rules[j * (n+1) + i] = (signed char) rnd() % (fset_lens[i]+1);
      }
      /* rules[j * (n+1) + n] = true_rules[j][n]; */
      rules[j * (n+1) + n] = (signed char) rnd() % fset_lens[n] + 1;
    }
  }
}


static void destroy_population(Chromosome *population, unsigned population_power)
{
  unsigned k;

  for (k = 0; k < population_power; ++k) {
    free((void*)population[k].fsets);
    free((void*)population[k].rules);
  }
}
  

#define TAU_DISCRETIZATION_DIM 20
#define TAU_DISCRETIZATION_STEP (1.f / TAU_DISCRETIZATION_DIM)


template <typename GaussParamsT>
static GaussParams make_gauss_params(GaussParamsT gp, unsigned i = 0, unsigned n = 0)
{
	return gp;
}

template <>
GaussParams make_gauss_params<GaussEvoParams>(GaussEvoParams gep, unsigned i, unsigned n)
{
	return {((i+1) + (float)(gep.s_mu-10)/20) / (n+1), (float)gep.s_sigma / 10 / (n+1)};
}


template <typename GaussParamsT>
static unsigned classify_fuzzy(
															 const unsigned *fset_offsets, const unsigned *fset_lens, unsigned fsets_total_len,
															 const GaussParamsT *fsets, unsigned n,
															 const signed char *rules, unsigned rules_len,
															 const GaussParams *uxx)
{
  unsigned i, j, k, l;
  double numerator = 0., denominator = 0.;

  for (k = 0; k < rules_len; ++k) {
    // NOTE(sergey): Here we make assumption that b's cannot be zero.
    signed char b = rules[k * (n+1) + n];
    float uy_center = make_gauss_params(fsets[fset_offsets[n] + (b-1)], b-1, fset_lens[n]).mu;
    float cross = 1.f;

    for (j = 0; j < rules_len; ++j) {
      signed char b = rules[j * (n+1) + n];
      GaussParams ub = make_gauss_params(fsets[fset_offsets[n] + (b-1)], b-1, fset_lens[n]);
      float ub_value = GAUSS(ub.mu, ub.sigma, uy_center); // (b<0) + SIGN(b) * GAUSS(ub.mu, ub.sigma, ub_center);
      float max_tnorm = 0.f;

      for (i = 0; i < n; ++i) {
				signed char a = rules[j * (n+1) + i];

				if (a == 0) continue;
	
				GaussParams ua = make_gauss_params(fsets[fset_offsets[i] + (a-1)], a-1, fset_lens[i]);
				GaussParams ux = uxx[i];
				float t;

				for (t = 0.f; t <= 1.01f; t += 0.05f) {
					max_tnorm = fmax(max_tnorm, fmin(GAUSS(ux.mu, ux.sigma, t),
																					 IMPL(GAUSS(ua.mu, ua.sigma, t), ub_value)));
				}
      }
      if (max_tnorm > 0.f) cross = fmin(cross, max_tnorm);
    }
    numerator += uy_center * cross;
    denominator += cross;
  }

  {
    float uy_center = numerator / denominator;
    float max_uy_value = 0.f;
    float max_uy_index = 0;

    for (j = 0; j < fset_lens[n]; ++j) {
      GaussParams ub = make_gauss_params(fsets[fset_offsets[n] + j], j, fset_lens[n]);
      float uy_value = GAUSS(ub.mu, ub.sigma, uy_center);
      
      if (uy_value > max_uy_value) {
				max_uy_value = uy_value;
				max_uy_index = j;
      }
    }
    return max_uy_index + 1;
  }
}


extern "C"
void predict_cpu_impl(
											const unsigned *fset_lens, const GaussParams *fsets, unsigned n,
											const signed char *rules, unsigned rules_len,
											const GaussParams *uxxs, unsigned *ys, unsigned N)
{
  unsigned i, k;
  unsigned fsets_total_len = 0, fset_offsets[n+1];

  for (i = 0; i < n+1; ++i) {
    fset_offsets[i] = fsets_total_len;
    fsets_total_len += fset_lens[i];
  }

  /* for (i = 0; i < fsets_total_len; ++i) printf("%f %f\n", fsets[i].mu, fsets[i].sigma); */
  /* for (k = 0; k < rules_len; ++k) { */
  /*   for (i = 0; i < n+1; ++i) printf("%d ", rules[k * (n+1) + i]); */
  /*   printf("\n"); */
  /* } */

  /* printf("fsets_total_len=%d, n=%d, rules_len=%d, N=%d\n", fsets_total_len, n, rules_len, N); */

#pragma omp parallel for ordered schedule(dynamic, 5)
  for (k = 0; k < N; ++k) {
#pragma omp ordered
    {
      ys[k] = classify_fuzzy(fset_offsets, fset_lens, fsets_total_len,
														 fsets, n, rules, rules_len, uxxs + k * n);
      /* printf("%d: ", k); */
      /* for (unsigned j = 0; j < n; ++j) printf("(%.3f %.3f) ", uxxs[k*n + j].mu, uxxs[k*n + j].sigma); */
      /* printf("%d\n", ys[k]); */
    }
  }
}


static float compute_score(const unsigned *fset_offsets, const unsigned *fset_lens,
													 unsigned fsets_total_len, unsigned n, unsigned rules_len,
													 const Chromosome *chromosome, const GaussParams *uxxs,
													 const unsigned *ys, unsigned N)
{
  unsigned i;
  unsigned error = 0;

  for (i = 0; i < N; ++i) {
    unsigned pred = classify_fuzzy(fset_offsets, fset_lens, fsets_total_len,
																	 chromosome->fsets, n, chromosome->rules, rules_len,
																	 uxxs + i * n);
    error += pred != ys[i];
  }
  return (float)(N - error) / N; // -(error / N);
}


static void print_hist(const float *scores, const unsigned *indices, unsigned population_power)
{
	unsigned hist[population_power];
	memset(hist, 0, sizeof(unsigned[population_power]));
	for (unsigned i = 0; i < population_power; ++i) ++hist[indices[i]];

	for (unsigned i = 0; i < population_power; ++i) {
		if (hist[i] > 5) printf("IDX: % 3d COUNT: % 2d SCORE: %f\n", i, hist[i], scores[i]);
	}
}



static void perform_selection(const float *scores, unsigned *indices, unsigned population_power)
{
  unsigned i, j;

  double min = +INFINITY, max = -INFINITY, sum = 0.f;
  unsigned best_idx;

#define OFFSET 0.01f

  /* for (i = 0; i < population_power; ++i) { */
  /*   if (scores[i] < min) min = scores[i]; */
  /* } */

	/*
	for (i = 0; i < population_power; ++i) {
    if (scores[i] > max) {
			max = scores[i];
			best_idx = i;
    }
    // Add 0.001 for cases when all scores are equal
    sum += pow(scores[i], 1) + OFFSET;
	}
	indices[0] = best_idx;

	for (i = 0; i < population_power; ++i) {
    double x = sum * rnd_prob();
    for (j = 0; j < population_power; ++j) {
			if (x < (pow(scores[j], 1) + OFFSET)) {
				indices[i] = j;
				break;
			}
			x -= pow(scores[j], 1) + OFFSET;
    }
	}
	*/

  /* print_hist(scores, indices, population_power); */

#undef OFFSET

  for (i = 0; i < population_power; ++i) {
    unsigned max_index = rnd() % population_power;
    float max_score = scores[max_index];

    for (j = 1; j < 3; ++j) {
      unsigned index = rnd() % population_power;
      float score = scores[index];

      if (score > max_score) {
				max_score = score;
				max_index = index;
      }
    }
    indices[i] = max_index;
  }
}


static void copy_chromosome(unsigned fsets_total_len, unsigned n, unsigned rules_len,
														const Chromosome *chs, Chromosome *chd)
{
  memcpy(chd->fsets, chs->fsets, sizeof(GaussEvoParams[fsets_total_len]));
  memcpy(chd->rules, chs->rules, sizeof(rules_len*(n+1)));
}


static void perform_crossingover(unsigned fsets_total_len, unsigned n, unsigned rules_len,
																 const Chromosome *cha, const Chromosome *chb,
																 Chromosome *chx, Chromosome *chy)
{
  unsigned i, pos = rnd() % (fsets_total_len + rules_len * (n+1));
  // unsigned i, pos = rnd() % (rules_len * (n+1)) + fsets_total_len; // NOTE(sergey): Temporary learn only rules
	// TODO(sergey): Now we decide to only perform crossingover on rules.
  
  /* printf("fsets_total_len=%d cha->fsets=%p chb->fsets=%p\n", */
  /* 	 fsets_total_len, cha->fsets, chb->fsets); */
  if (pos < fsets_total_len) {
		memcpy(chx->fsets, cha->fsets, sizeof(GaussEvoParams[pos]));
		memcpy(chy->fsets, chb->fsets, sizeof(GaussEvoParams[pos]));
		memcpy(chx->fsets+pos, chb->fsets+pos, sizeof(GaussEvoParams[fsets_total_len - pos]));
		memcpy(chx->fsets+pos, cha->fsets+pos, sizeof(GaussEvoParams[fsets_total_len - pos]));
		memcpy(chx->rules, chb->rules, rules_len*(n+1));
		memcpy(chy->rules, cha->rules, rules_len*(n+1));
	} else {
    memcpy(chx->fsets, cha->fsets, sizeof(GaussEvoParams[fsets_total_len]));
    memcpy(chy->fsets, chb->fsets, sizeof(GaussEvoParams[fsets_total_len]));
    pos -= fsets_total_len;
    memcpy(chx->rules, cha->rules, pos);
    memcpy(chy->rules, chb->rules, pos);
    memcpy(chx->rules + pos, chb->rules + pos, rules_len*(n+1) - pos);
    memcpy(chy->rules + pos, cha->rules + pos, rules_len*(n+1) - pos);
  }
}


static void perform_mutation(const unsigned *fset_lens, unsigned fsets_total_len, unsigned n,
														 unsigned rules_len, Chromosome *ch, float pm_fsets, float pm_rules)
{
  {
    unsigned i;

    for (i = 0; i < fsets_total_len; ++i) {
      if (rnd_prob() < pm_fsets) ch->fsets[i].s_mu = (int)(gauss_noise(0.5, 0.5) * 20) % 20;
      if (rnd_prob() < pm_fsets) ch->fsets[i].s_sigma = (int)(gauss_noise(0.5, 0.5) * 20) % 20;
    }
  }

  {
    unsigned k, i;

    for (k = 0; k < rules_len; ++k) {
      for (i = 0; i < n; ++i) {
				if (rnd_prob() < pm_rules) {
					ch->rules[k*(n+1)+i] = (ch->rules[k*(n+1)+i]+1) % (fset_lens[i]+1); // rnd() % (fset_lens[i]+1);
				}
      }
      if (rnd_prob() < pm_rules) {
				ch->rules[k*(n+1)+n] = (ch->rules[k*(n+1)+n]+1) % fset_lens[n] + 1; // rnd() % fset_lens[n] + 1;
      }
    }
  }
}


static void convert_to_gauss_params(const unsigned *fset_offsets, const unsigned *fset_lens, unsigned n,
																		const GaussEvoParams *gauss_evo_params, GaussParams *gauss_params)
{
	unsigned i, j;

	for (i = 0; i < n+1; ++i) {
		for (j = 0; j < fset_lens[i]; ++j) {
			unsigned idx = fset_offsets[i] + j;
			gauss_params[idx] = make_gauss_params(gauss_evo_params[idx], j, fset_lens[i]);
		}
	}
}


extern "C"
void tune_lfs_cpu_impl(
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

  printf("fsets_total_len=%d, n=%d, rules_len=%d, population_power=%d, iterations_number=%d, N=%d\n",
				 fsets_total_len, n, rules_len, population_power, iterations_number, N);

  Chromosome* population = (Chromosome*)malloc(sizeof(Chromosome[population_power]));
  Chromosome* new_population = (Chromosome*)malloc(sizeof(Chromosome[population_power]));
  float* scores = (float*)malloc(sizeof(float[population_power]));
  unsigned* indices = (unsigned*)malloc(sizeof(unsigned[population_power]));

  for (k = 0; k < population_power; ++k) {
    scores[k] = 0.f;
    indices[k] = k;
  }

  float last_best_score = -INFINITY;
  unsigned last_best_update_it = 0;
	double mean_variance = 0.0;

  init_population(fset_offsets, fset_lens, fsets_total_len, n, rules_len, population, population_power);
  init_population(fset_offsets, fset_lens, fsets_total_len, n, rules_len, new_population, population_power);

  /* float pc0 = 0.2f, pc1 = 0.6f; */
  /* float pm0 = 0.15, pm1 = 0.3f; */
  float pc = 0.9f, pm_fsets = 1.f / (fsets_total_len * 20), pm_rules = 1.f / (rules_len * n);

  FILE *f = fopen("report_cpu.txt", "w");
  FILE *fs = fopen("report_cpu_scores.txt", "w");
  FILE *fpp = fopen("report_ga_params.txt", "w");
  printf("Evolution started...\n");
#pragma omp parallel default(none)																			\
  shared(fsets_total_len, fset_offsets, fset_lens, n, rules_len, N,			\
				 population_power, iterations_number, uxxs, ys,									\
				 last_best_score, last_best_update_it, mean_variance)						\
  private(it, k)																												\
  firstprivate(fsets, rules, pc, pm_fsets, pm_rules, f, fs, fpp,				\
							 population, new_population, scores, indices) // Create threads team
  for (it = 0; it < iterations_number; ++it) {
    /* float a = exp(-(float) iterations_number / (it + 1.f)); */
    /* float pc = (1.f - a) * pc0 + a * pc1; */
    /* float pm = (1.f - a) * pm0 + a * pm1; */
    
#pragma omp for schedule(dynamic)
    for (k = 0; k < population_power; ++k) {
      scores[k] = compute_score(fset_offsets, fset_lens, fsets_total_len,
																n, rules_len, population + k, uxxs, ys, N);
    }

#pragma omp single nowait
    {
      double sum = 0.0f, min = +INFINITY, max = -INFINITY;
      unsigned max_index = 0;
      
      printf("[Iteration %3d] ", it);
      for (k = 0; k < population_power; ++k) {
				sum += scores[k];
				if (scores[k] > max) {
					max = scores[k];
					max_index = k;
				}
				min = MIN(min, scores[k]);
				// printf("%.3f ", scores[k]);
				fprintf(fs, "%f ", scores[k]);
      }
      fprintf(fs, "\n");

      double avg_score = sum / population_power;
      printf("Min score: %f Max score: %f(%d) Avg score: %f\n",
						 min, max, max_index, avg_score);
      fprintf(f, "%f,%f,%f\n", min, max, avg_score);

			{
				double variance = 0.0;

				for (k = 0; k < population_power; ++k) {
					variance += pow(scores[k] - avg_score, 2);
				}
				variance /= population_power;
				mean_variance += variance;
			}

      if (max > last_best_score) {
				last_best_score = max;
				last_best_update_it = it;

#pragma omp flush(last_best_score,last_best_update_it)
	
				printf(">[%3d] Found new best chromosome with score: %f\n", it, max);

				convert_to_gauss_params(fset_offsets, fset_lens, n, population[max_index].fsets, fsets);
				memcpy(rules, population[max_index].rules, sizeof(signed char[rules_len * (n+1)]));

				unsigned local_ys[N], correct = 0;
				predict_cpu_impl(fset_lens, fsets, n, rules, rules_len, uxxs, local_ys, N);
				for (unsigned i = 0; i < N; ++i) correct += local_ys[i] == ys[i];
				printf("Score: %f\n", (float)correct / N);
      }

      printf("pc = %f, pm_fsets = %f, pm_rules = %f\n", pc, pm_fsets, pm_rules);
    }
    
#pragma omp single
		perform_selection(scores, indices, population_power);

#pragma omp for schedule(dynamic)
    for (k = 0; k < population_power; k += 2) {
      if (rnd_prob() < pc) {
				perform_crossingover(fsets_total_len, n, rules_len,
														 population + indices[k], population + indices[k + 1],
														 new_population + k, new_population + k + 1);
      } else {
				copy_chromosome(fsets_total_len, n, rules_len, population + indices[k], new_population + k);
				copy_chromosome(fsets_total_len, n, rules_len, population + indices[k+1], new_population + k+1);
      }
    }

// #pragma omp barrier
		
#pragma omp for schedule(dynamic, 4)
    for (k = 0; k < population_power; ++k) {
      perform_mutation(fset_lens, fsets_total_len, n, rules_len, new_population + k, pm_fsets, pm_rules);
    }

    SWAP(population, new_population);
  }
// #pragma omp barrier
	printf("Mean variance: %f\n", mean_variance);
  printf("Evolution finished...\n");
  fclose(f);
  fclose(fs);
  fclose(fpp);

  destroy_population(new_population, population_power);
  destroy_population(population, population_power);
  free(indices);
  free(scores);
  free(new_population);
  free(population);
}
