#include "evolutionary_tune.h"

const unsigned n = 6;
const unsigned rules_len = 15;
const unsigned N = 22;
const unsigned fset_lens[7] = {2, 4, 2, 2, 4, 2, 2};
const signed char samples[][7] = {
    {1,1,1,1,1,2,2},
    {1,2,2,1,4,1,2},
    {1,3,1,2,3,2,2},
    {2,4,2,2,4,2,2},
    {2,1,1,1,1,1,1},
    {2,3,1,2,3,1,1},
    {2,4,2,2,4,1,1},
    {1,2,1,1,1,1,1},
    {1,1,1,1,1,1,1},
    {1,3,2,2,1,1,1},
    {1,2,2,2,3,1,1},
    {1,2,2,2,3,3,1},
    {1,1,1,1,4,1,1},
    {1,4,1,1,1,1,2},
    {1,4,1,1,2,1,2},
    {1,4,1,1,3,1,2},
    {1,3,1,1,1,1,2},
    {1,3,1,1,2,1,2},
    {1,3,1,2,1,1,2},
    {1,3,1,2,2,1,2},
    {1,3,1,1,3,1,1},
    {1,3,1,2,3,1,2},
};

const unsigned population_power = 100;
const unsigned iterations_number = 150;


static void build_uxxs(const unsigned fset_lens[], const signed char samples[][7], GaussParams uxxs[][6], unsigned n, unsigned N)
{
	for (unsigned j = 0; j < N; ++j) {
		for (unsigned i = 0; i < n; ++i) {
			uxxs[j][i].mu = (float)samples[j][i] / (fset_lens[i]+1);
			uxxs[j][i].sigma = 1.f / (fset_lens[i]+1);
		}
	}
}


static void build_ys(const signed char samples[][7], unsigned ys[], unsigned n, unsigned N)
{
	for (unsigned j = 0; j < N; ++j) ys[j] = samples[j][n];
}


int main(int argc, const char *argv[])
{
	unsigned i, fsets_total_len = 0;
	for (i = 0; i < n+1; ++i) fsets_total_len += fset_lens[i];
	
	GaussParams uxxs[N][n];
	unsigned ys[N];

	build_uxxs(fset_lens, samples, uxxs, n, N);
	build_ys(samples, ys, n, N);

	GaussParams fsets[fsets_total_len];
	signed char rules[rules_len][n+1];

	tune_lfs_gpu_impl(fset_lens, n, rules_len,
										uxxs, ys, N,
										population_power, iterations_number,
										fsets, rules);
	return 0;
}
