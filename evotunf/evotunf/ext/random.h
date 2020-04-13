#ifndef EVOTUNF_RANDOM_H
#define EVOTUNF_RANDOM_H

#include "common.h"

#define MT19937_RND_MAX 0xFFFFFFFF // ((1 << 32) - 1)

static unsigned mt19937_rnd()
{
    struct mersene_twister_generator {
        unsigned w; // word size
        unsigned n; // degree of recurrence
        unsigned m; // middle word, 1 <= m < n
        unsigned r; // separation point
        unsigned a; // coefficients of the rational normal form twist matrix
        unsigned b, c; // TGFSR(R) tempering bitmask
        unsigned s, t; // TGFSR(R) tempering bitshift
        unsigned u, d, l; // additional Mersene Twister tempering bit shift/mask
        unsigned f;
    };

    static const struct mersene_twister_generator mtg = {
        .w = 32, .n = 624, .m = 397, .r = 31,
        .a = 0x9908B0DF,
        .u = 11, .d = 0xFFFFFFFF,
        .s = 7, .b = 0x9D2C5680,
        .t = 15, .c = 0xEFC60000,
        .l = 18,
        .f = 1812433253
    };

    static unsigned mt[624];
    static unsigned index = 625; // mtg.n + 1;
    static const unsigned lower_mask = 0xFFFFFFFF; // (1 << mtg.r) - 1; // the binary number of r 1's
    static const unsigned upper_mask = 0; // ~lower_mask;

    if (index >= mtg.n) {
        size_t i;

        if (index > mtg.n) {
            mt[0] = time(0);
            for (i = 1; i < mtg.n; ++i) {
                mt[i] = (mtg.f * (mt[i-1] ^ (mt[i-1] >> (mtg.w-2))) + i);
            }
        }

        for (i = 0; i < mtg.n; ++i) {
            unsigned x = (mt[i] & upper_mask) + (mt[(i+1) % mtg.n] & lower_mask);
            unsigned xa = x >> 1;
            if (x & 0x1) {
                xa ^= mtg.a;
            }
            mt[i] = mt[(i + mtg.m) % mtg.n] ^ xa;
        }
        index = 0;
    }

    unsigned y = mt[index];
    y ^= (y >> mtg.u) & mtg.d;
    y ^= (y << mtg.s) & mtg.b;
    y ^= (y << mtg.t) & mtg.c;
    y ^= (y >> mtg.l);

    ++index;
    return y;
}

#define RND_MAX MT19937_RND_MAX

static unsigned rnd()
{
    return mt19937_rnd();
}

static float rnd_prob()
{
    return (float) rnd() / ((float)RND_MAX);
}

static float gauss_noise(float mu, float sigma)
{
    static const float epsilon = FLT_MIN;
    static const float two_pi = 2.f * 3.14159265358979323846;

    static float z0, z1;
    static int generate;

    generate = !generate;

    if (!generate)
        return z1 * sigma + mu;

    float u1, u2;
    do {
        u1 = rnd_prob();
        u2 = rnd_prob();
    } while (u1 <= epsilon);

    z0 = sqrt(-2.f * log(u1)) * cos(two_pi * u2);
    z1 = sqrt(-2.f * log(u1)) * sin(two_pi * u2);

    return z0 * sigma + mu;
}

#endif // EVOTUNF_RANDOM_H
