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

typedef struct {
    int t_outer;
    int t_inner;
    int impl;
    unsigned rule_base_len;
    float *gauss_params;
} LfsConfig;

LfsConfig train_lfs_cpu_impl(float *xx_train, float *y_train, unsigned n, unsigned N, float *rule_base, unsigned rule_base_len);
void infer_lfs_cpu_impl(LfsConfig lfs_config, const float *xxs, float *ys, unsigned n, unsigned N);
