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

typedef struct GaussParams {
    float mu, sigma;
} GaussParams;

typedef struct EvolutionaryParams {
    float sigma1, sigma2;
} EvolutionaryParams;
