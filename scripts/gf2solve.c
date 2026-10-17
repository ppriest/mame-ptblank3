// GF(2) linear system solver. Input: u32 N, u32 F, then N rows of W=ceil((F+1)/64) u64 words.
// Bit F of a row is the target. Output (stdout): rank, inconsistent-row count, pivot columns, solution columns.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb"); if (!f) { perror("open"); return 1; }
    uint32_t N, F; fread(&N, 4, 1, f); fread(&F, 4, 1, f);
    uint32_t W = (F + 1 + 63) / 64;
    uint64_t *A = malloc((size_t)N * W * 8); fread(A, 8, (size_t)N * W, f); fclose(f);
    uint32_t *pivcol = malloc(F * sizeof(uint32_t)); uint32_t rank = 0;
    for (uint32_t col = 0; col < F && rank < N; col++) {
        uint32_t w = col >> 6; uint64_t bit = 1ULL << (col & 63);
        uint32_t r = rank; while (r < N && !(A[(size_t)r * W + w] & bit)) r++;
        if (r == N) continue;
        if (r != rank) { for (uint32_t k = 0; k < W; k++) { uint64_t t = A[(size_t)r*W+k]; A[(size_t)r*W+k] = A[(size_t)rank*W+k]; A[(size_t)rank*W+k] = t; } }
        uint64_t *prow = A + (size_t)rank * W;
        for (uint32_t i = rank + 1; i < N; i++) {
            uint64_t *row = A + (size_t)i * W;
            if (row[w] & bit) for (uint32_t k = w; k < W; k++) row[k] ^= prow[k];
        }
        pivcol[rank++] = col;
    }
    // rows below rank: features all zero (by construction); check target
    uint32_t tw = F >> 6; uint64_t tbit = 1ULL << (F & 63); uint32_t bad = 0;
    for (uint32_t i = rank; i < N; i++) if (A[(size_t)i*W + tw] & tbit) bad++;
    // back substitution, free vars = 0
    uint8_t *x = calloc(F, 1);
    for (int32_t r = rank - 1; r >= 0; r--) {
        uint64_t *row = A + (size_t)r * W; uint32_t pc = pivcol[r];
        uint8_t v = (row[tw] & tbit) ? 1 : 0;
        for (uint32_t col = pc + 1; col < F; col++) if ((row[col>>6] >> (col&63)) & 1) v ^= x[col];
        x[pc] = v;
    }
    printf("N %u F %u rank %u inconsistent %u\n", N, F, rank, bad);
    printf("pivots"); for (uint32_t r = 0; r < rank; r++) printf(" %u", pivcol[r]); printf("\n");
    printf("solution"); for (uint32_t c = 0; c < F; c++) if (x[c]) printf(" %u", c); printf("\n");
    return 0;
}
