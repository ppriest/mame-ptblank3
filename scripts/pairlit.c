// For each XOR-pair of candidate raw bits (and polarity), test if target is linear in features on the subset where the pair == pol.
// Input file: u32 N, F(features incl const), K(candidates), NSUB ; then N rows: W=ceil((F+1)/64) u64 (features, target at bit F) ; then N rows of KW=ceil(K/64) u64 candidate bits.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
static inline int getbit(const uint64_t *r, uint32_t c) { return (r[c >> 6] >> (c & 63)) & 1; }
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb"); uint32_t N, F, K, NSUB; fread(&N,4,1,f); fread(&F,4,1,f); fread(&K,4,1,f); fread(&NSUB,4,1,f);
    uint32_t W = (F + 1 + 63) / 64, KW = (K + 63) / 64;
    uint64_t *X = malloc((size_t)N * W * 8); fread(X, 8, (size_t)N * W, f);
    uint64_t *C = malloc((size_t)N * KW * 8); fread(C, 8, (size_t)N * KW, f); fclose(f);
    int single = argc > 2 && strcmp(argv[2], "single") == 0;
    uint64_t *A = malloc((size_t)NSUB * W * 8); uint32_t tw = F >> 6; uint64_t tbit = 1ULL << (F & 63);
    long tests = 0; double best = 1.0;
    for (uint32_t a = 0; a < K; a++) for (uint32_t b = single ? a : a + 1; b < (single ? a + 1 : K); b++) for (int pol = 0; pol < 2; pol++) {
        // gather subset rows
        uint32_t n = 0;
        for (uint32_t i = 0; i < N && n < NSUB; i++) {
            int v = getbit(C + (size_t)i * KW, a) ^ (single ? 0 : getbit(C + (size_t)i * KW, b));
            if (v == pol) { memcpy(A + (size_t)n * W, X + (size_t)i * W, W * 8); n++; }
        }
        if (n < 1500) continue;
        // elimination
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
        uint32_t bad = 0; for (uint32_t i = rank; i < n; i++) if (A[(size_t)i*W + tw] & tbit) bad++;
        uint32_t free_ = n - rank; double frac = free_ ? (double)bad / free_ : 1.0;
        tests++;
        if (bad == 0 || frac < 0.30) printf("HIT a=%u b=%u pol=%d rows=%u rank=%u free=%u bad=%u frac=%.3f\n", a, b, pol, n, rank, free_, bad, frac);
        if (frac < best) best = frac;
    }
    printf("tests %ld best frac %.4f\n", tests, best);
    return 0;
}
