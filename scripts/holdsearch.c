// Hold-register search: F_i = x[j(i)], j(i) = last index <= i (variant 0) or < i (variant 1) with s == 1.
// For each (s, x) pair test exact linearity of target on rows {F=0} and {F=1} separately (first NSUB rows each).
// Input: u32 N, F(features incl const), K(candidates), NSUB; N rows of W u64 (features, target bit at F); N rows of KW u64 candidate bits.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
static inline int getbit(const uint64_t *r, uint32_t c) { return (r[c >> 6] >> (c & 63)) & 1; }
static uint32_t elim(uint64_t *A, uint32_t n, uint32_t W, uint32_t F, uint32_t *bad) {
    uint32_t rank = 0;
    for (uint32_t col = 0; col < F && rank < n; col++) {
        uint32_t w = col >> 6; uint64_t bit = 1ULL << (col & 63); uint32_t r = rank;
        while (r < n && !(A[(size_t)r*W+w] & bit)) r++;
        if (r == n) continue;
        if (r != rank) for (uint32_t k = 0; k < W; k++) { uint64_t t = A[(size_t)r*W+k]; A[(size_t)r*W+k] = A[(size_t)rank*W+k]; A[(size_t)rank*W+k] = t; }
        uint64_t *prow = A + (size_t)rank * W;
        for (uint32_t i = rank + 1; i < n; i++) { uint64_t *row = A + (size_t)i * W; if (row[w] & bit) for (uint32_t k = w; k < W; k++) row[k] ^= prow[k]; }
        rank++;
    }
    uint32_t tw = F >> 6; uint64_t tbit = 1ULL << (F & 63); *bad = 0;
    for (uint32_t i = rank; i < n; i++) if (A[(size_t)i*W + tw] & tbit) (*bad)++;
    return rank;
}
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb"); uint32_t N, F, K, NSUB; fread(&N,4,1,f); fread(&F,4,1,f); fread(&K,4,1,f); fread(&NSUB,4,1,f);
    uint32_t W = (F + 1 + 63) / 64, KW = (K + 63) / 64;
    uint64_t *X = malloc((size_t)N * W * 8); fread(X, 8, (size_t)N * W, f);
    uint64_t *C = malloc((size_t)N * KW * 8); fread(C, 8, (size_t)N * KW, f); fclose(f);
    uint64_t *A0 = malloc((size_t)NSUB * W * 8), *A1 = malloc((size_t)NSUB * W * 8);
    uint8_t *Fv = malloc(N); int32_t *lastj = malloc((size_t)N * 4);
    long tests = 0; double best = 1.0;
    for (uint32_t s = 0; s < K; s++) {
        for (int variant = 0; variant < 2; variant++) {
            int32_t lj = -1;
            for (uint32_t i = 0; i < N; i++) {
                int sv = getbit(C + (size_t)i * KW, s);
                if (variant == 0 && sv) lj = i;
                lastj[i] = lj;
                if (variant == 1 && sv) lj = i;
            }
            for (uint32_t x = 0; x < K; x++) {
                if (x == s) continue;
                uint32_t n0 = 0, n1 = 0;
                for (uint32_t i = 0; i < N && (n0 < NSUB || n1 < NSUB); i++) {
                    if (lastj[i] < 0) continue;
                    int fv = getbit(C + (size_t)lastj[i] * KW, x);
                    if (fv == 0 && n0 < NSUB) { memcpy(A0 + (size_t)n0 * W, X + (size_t)i * W, W * 8); n0++; }
                    else if (fv == 1 && n1 < NSUB) { memcpy(A1 + (size_t)n1 * W, X + (size_t)i * W, W * 8); n1++; }
                }
                if (n0 < 2000 || n1 < 2000) continue;
                uint32_t bad0, bad1; uint32_t r0 = elim(A0, n0, W, F, &bad0); uint32_t r1 = elim(A1, n1, W, F, &bad1);
                double frac = ((double)bad0 / (n0 - r0) + (double)bad1 / (n1 - r1)) / 2; tests++;
                if (bad0 == 0 || bad1 == 0 || frac < 0.35) printf("HIT s=%u x=%u var=%d: n0=%u bad0=%u/%u n1=%u bad1=%u/%u\n", s, x, variant, n0, bad0, n0 - r0, n1, bad1, n1 - r1), fflush(stdout);
                if (frac < best) best = frac;
            }
        }
    }
    printf("tests %ld best mean inconsistent fraction %.4f\n", tests, best);
    return 0;
}
