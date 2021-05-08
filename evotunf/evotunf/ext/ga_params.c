#include <math.h>

#include "common.h"
#include "ga_params.h"


#define LUKASZEWICZ_IMPL(a, b) fmin(1.f, 1.f - (a) + (b))
#define IMPL(a, b) LUKASZEWICZ_IMPL(a, b)


typedef struct TriangleParams { float x1, x2; } TriangleParams;
typedef float (*TriangleMembershipFunction)(TriangleParams, float x);


static float left_triangle(TriangleParams tp, float x)
{
	return  (x <= tp.x1) ? 1 : (x > tp.x2) ? 0 : (tp.x2-x) / (tp.x2-tp.x1);
}
  
static float right_triangle(TriangleParams tp, float x)
{
	return  (x <= tp.x1) ? 1 : (x > tp.x2) ? 0 : (x-tp.x1) / (tp.x2-tp.x1);
}
  
static float triangle(TriangleParams tp, float x) {
	return  (x < tp.x1) ? 0 : (x > tp.x2) ? 0 : (x <= ((tp.x1+tp.x2)/2)) ? 2*(x-tp.x1)/(tp.x2-tp.x1) : 2*(tp.x2-x)/(tp.x2-tp.x1);
}

static float infer(const TriangleMembershipFunction memfuncs[],
		   const TriangleParams fsets[][3], unsigned n, const signed char rules[][5], unsigned rules_len,
		   float inputs[], unsigned outidx)
{
	unsigned i, j, k;
	float numerator = 0.f, denominator = 0.f;

	for (k = 0; k < rules_len; ++k) {
		unsigned b = rules[k][n+outidx];
		TriangleParams uy = fsets[n+outidx][b-1];
		float uy_center;

		switch (b) {
		case 0: uy_center = uy.x1; break;
		case 1: uy_center = (uy.x1 + uy.x2) / 2; break;
		case 2: uy_center = uy.x2; break;
		}

		float cross = 1.f;
      
		for (j = 0; j < rules_len; ++j) {
			unsigned b = rules[j][n+outidx];
			TriangleParams ub = fsets[n+outidx][b-1];
			TriangleMembershipFunction b_memfunc = memfuncs[b-1];
			float ub_value = b_memfunc(ub, uy_center);
			float max_tnorm = 0.f;

			for (i = 0; i < n; ++i) {
				unsigned a = rules[j][i];

				if (!a) continue;
	  
				TriangleParams ua = fsets[i][a-1];
				TriangleMembershipFunction a_memfunc = memfuncs[a-1];

				max_tnorm = MAX(max_tnorm, IMPL(a_memfunc(ua, inputs[i]), ub_value));
				/* printf("%f %f | ", inputs[i], a_memfunc(ua, inputs[i])); */
			}

			cross = MIN(cross, max_tnorm);
		}

		/* printf("%f ", cross); */
		numerator += cross * uy_center;
		denominator += cross;
		/* printf("(%f %f)\n", cross * uy_center, uy_center); */
	}
	/* printf("\n"); */

	return numerator / denominator;
}


static TriangleMembershipFunction triangle_memfuncs[3] = {left_triangle, triangle, right_triangle};

static const unsigned n = 3;
static TriangleParams fsets[][3] =
{
	{{0.0f, 0.7f}, {0.5f, 0.9f}, {0.7f, 1.0f}},
	{{0, 10}, {5, 15}, {10, 20}},
	{{0.0f, 0.12f}, {0.1f, 0.14f}, {0.12f, 0.2f}},
	{{0.001f, 0.01f}, {0.005f, 0.015f}, {0.010f, 0.1f}},
	{{0.48f, 0.65f}, {0.55f, 0.75f}, {0.65f, 0.9f}},
};

static signed char rules[][5] =
{
        {1, 0, 0, 1, 3},
	{2, 1, 0, 1, 3},
	{2, 2, 0, 3, 2},
	{3, 1, 0, 1, 3},
	{3, 2, 0, 2, 2},
	{3, 3, 0, 3, 1},
	{0, 3, 1, 3, 1},
	{0, 3, 2, 2, 2},
	{0, 3, 3, 1, 3},

        /* {1, 0, 0, 1, 3}, */
	/* {2, 1, 0, 1, 3}, */
	/* {2, 2, 0, 3, 2}, */
	/* {3, 1, 0, 1, 3}, */
	/* {3, 2, 0, 2, 2}, */
	/* {0, 3, 1, 3, 1}, */
	/* {0, 3, 2, 3, 1}, */
	/* {0, 3, 3, 1, 3}, */
};
static const unsigned rules_len = sizeof(rules) / sizeof(signed char[5]);


void compute_ga_params(float bf, unsigned un, float vf, float *pcr, float *pmr)
{
	float inputs[] = {bf, un, 50*vf};

	printf("%f %2d %f\n", bf, un, inputs[2]);
	*pcr = infer(triangle_memfuncs, fsets, n, rules, rules_len, inputs, 1);
	*pmr = infer(triangle_memfuncs, fsets, n, rules, rules_len, inputs, 0);

	/* static unsigned count = 0; */

	/* if (!count--) scanf("%d %f %f", &count, pcr, pmr); */
}
    

