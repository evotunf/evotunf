enum t_norm {
    PRODUCT,
    MIN,
    T_NORM_NUMBER
};

enum implication {
    KLEENE_DIENES,
    LUKASZEWICZ,
    REICHENBACH,
    FODOR,
    GOGUEN,
    GODEL,
    ZADEH,
    RESCHER,
    YAGER,
    WILLMOTT,
    DUBOIS_PRADE,
    IMPLICATION_NUMBER
};

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

typedef struct DataBounds {
    float min;
    union { float max, a; };
} DataBounds;

typedef struct GaussParams {
    float mu, sigma;
} GaussParams;

typedef struct EvolutionaryParams {
    float sigma1, sigma2;
} EvolutionaryParams;

static
void compute_data_bounds(const GaussParams *xxs, const unsigned *ys, DataBounds *data_bounds, unsigned N, unsigned n)
{
    size_t i, j;

    for (i = 0; i < n; ++i) {
        data_bounds[i].min = xxs[i].mu - xxs[i].sigma;
        data_bounds[i].max = xxs[i].mu + xxs[i].sigma;
    }
    data_bounds[n].min = ys[0] - 0.5f;
    data_bounds[n].max = ys[0] + 0.5f;

    for (j = 1; j < N; ++j) {
        for (i = 0; i < n; ++i) {
            GaussParams gp = xxs[j * n + i];

            if (gp.mu - gp.sigma < data_bounds[i].min) {
                data_bounds[i].min = gp.mu - gp.sigma;
            }

            if (gp.mu + gp.sigma > data_bounds[i].max) {
                data_bounds[i].max = gp.mu + gp.sigma;
            }
        }

        {
            float val = ys[j];

            if (val - 0.5f < data_bounds[n].min) {
                data_bounds[n].min = val - 0.5f;
            }

            if (val + 0.5f > data_bounds[n].max) {
                data_bounds[n].max = val + 0.5f;
            }
        }
    }
}
