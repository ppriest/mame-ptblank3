// Forbidden 3-literal conjunctions: for all triples (x,y,z) of candidate bits and all 8 polarities, count rows and events where the conjunction holds.
// Input: u32 N, K ; N x KW u64 candidate bits ; N/64 u64 event bits (packed by row). Transposed layout: we store columns as bitsets over rows.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb"); uint32_t N, K; fread(&N, 4, 1, f); fread(&K, 4, 1, f);
    uint32_t RW = (N + 63) / 64;
    uint64_t *col = malloc((size_t)K * RW * 8); fread(col, 8, (size_t)K * RW, f);   // column-major bitsets
    uint64_t *ev = malloc((size_t)RW * 8); fread(ev, 8, RW, f); fclose(f);
    uint64_t *tmp = malloc(RW * 8), *tmp2 = malloc(RW * 8);
    long found = 0;
    for (uint32_t x = 0; x < K; x++) for (int px = 0; px < 2; px++) {
        for (uint32_t w = 0; w < RW; w++) tmp[w] = px ? col[(size_t)x*RW+w] : ~col[(size_t)x*RW+w];
        for (uint32_t y = x + 1; y < K; y++) for (int py = 0; py < 2; py++) {
            for (uint32_t w = 0; w < RW; w++) tmp2[w] = tmp[w] & (py ? col[(size_t)y*RW+w] : ~col[(size_t)y*RW+w]);
            for (uint32_t z = y + 1; z < K; z++) for (int pz = 0; pz < 2; pz++) {
                uint64_t sup = 0, e = 0;
                const uint64_t *cz = col + (size_t)z * RW;
                for (uint32_t w = 0; w < RW; w++) { uint64_t v = tmp2[w] & (pz ? cz[w] : ~cz[w]); sup += __builtin_popcountll(v); e += __builtin_popcountll(v & ev[w]); }
                if (e == 0 && sup >= 1500) { printf("FORBIDDEN x=%u/%d y=%u/%d z=%u/%d support=%llu\n", x, px, y, py, z, pz, (unsigned long long)sup); found++; fflush(stdout); }
            }
        }
    }
    printf("done, forbidden triples: %ld\n", found);
    return 0;
}
