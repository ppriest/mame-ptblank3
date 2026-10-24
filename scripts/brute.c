/* Exhaustive search of the z3nlfsr2 register template with a, c restricted to a plaintext bit mask.
   new bit = a.u ^ l.R ^ (d.R & f.R) ^ (b.R & c.u), u = p[i-4] bits in mask; output = XOR_k R[k] & W_k.m[i].
   For each (a, c, l, d, f, b, init) the register is simulated and the output is linear in W (n*16 unknowns):
   online GF(2) elimination rejects the candidate at the first inconsistent gated word.
   Usage: brute in.bin n mask lo hi   (candidate indices lo..hi-1; layout a | c | l | d | f | b | init) */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
typedef unsigned __int128 u128;
#define PAR(v) (__builtin_popcount(v) & 1)

static int hb(u128 v) {
    uint64_t h = (uint64_t)(v >> 64);
    return h ? 127 - __builtin_clzll(h) : 63 - __builtin_clzll((uint64_t)v);
}

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: brute in.bin n mask lo hi\n"); return 1; }
    FILE *fi = fopen(argv[1], "rb"); int32_t TH[2];
    if (!fi || fread(TH, 4, 2, fi) != 2) { fprintf(stderr, "bad input\n"); return 1; }
    int T = TH[0], H = TH[1], L = T + H, n = atoi(argv[2]); unsigned mask = strtoul(argv[3], 0, 0);
    uint64_t lo = strtoull(argv[4], 0, 0), hi = strtoull(argv[5], 0, 0);
    uint16_t *uc = malloc(2 * L), *x = malloc(2 * L); uint8_t *e = malloc(L), *g = malloc(L);
    int mb[16], nm = 0;
    for (int k = 0; k < 16; k++) if (mask >> k & 1) mb[nm++] = k;
    for (int q = 0; q < L; q++) {
        uint8_t r[6]; if (fread(r, 1, 6, fi) != 6) { fprintf(stderr, "short input\n"); return 1; }
        unsigned u = r[0] | r[1] << 8, v = 0;
        for (int j = 0; j < nm; j++) v |= ((u >> mb[j]) & 1) << j;
        uc[q] = v; x[q] = r[2] | r[3] << 8; e[q] = r[4]; g[q] = r[5];
    }
    unsigned NM = (1u << nm) - 1, NN = (1u << n) - 1;
    int gated = 0; for (int q = 0; q < T; q++) gated += g[q];
    u128 basis[128]; uint8_t brhs[128], have[128]; int used[128];
    memset(have, 0, sizeof have);
    uint64_t passes = 0; int best = 0; uint64_t bestidx = 0;
    for (uint64_t idx = lo; idx < hi; idx++) {
        uint64_t t = idx;
        unsigned a = t & NM; t >>= nm; unsigned c = t & NM; t >>= nm;
        unsigned l = t & NN; t >>= n; unsigned d = t & NN; t >>= n; unsigned f = t & NN; t >>= n;
        unsigned b = t & NN; t >>= n; unsigned R = t & NN, R0 = R;
        int nu = 0, ok = 1, ng = 0;
        for (int q = 0; q < T; q++) {
            if (g[q]) {
                if (!R) { if (e[q]) { ok = 0; break; } }
                else {
                    u128 row = 0; uint8_t rhs = e[q];
                    for (int k = 0; k < n; k++) if (R >> k & 1) row |= (u128)x[q] << (16 * k);
                    while (row) {
                        int p = hb(row);
                        if (!have[p]) { basis[p] = row; brhs[p] = rhs; have[p] = 1; used[nu++] = p; break; }
                        row ^= basis[p]; rhs ^= brhs[p];
                    }
                    if (!row && rhs) { ok = 0; break; }
                }
                ng++;
            }
            unsigned nb = PAR(a & uc[q]) ^ PAR(l & R) ^ (PAR(d & R) & PAR(f & R)) ^ (PAR(b & R) & PAR(c & uc[q]));
            R = ((R << 1) | nb) & NN;
        }
        if (!ok && ng > best) { best = ng; bestidx = idx; }
        if (ok) {
            /* back-substitute W (free variables 0), then simulate the held-out words */
            u128 w = 0;
            for (int p = 0; p < 16 * n; p++) if (have[p]) {
                u128 rest = basis[p] & ~((u128)1 << p) & w;
                int s = brhs[p] ^ (PAR((uint32_t)rest) ^ PAR((uint32_t)(rest >> 32)) ^ PAR((uint32_t)(rest >> 64)) ^ PAR((uint32_t)(rest >> 96)));
                if (s) w |= (u128)1 << p;
            }
            int hit = 0, tot = 0, ones = 0;
            for (int q = T; q < L; q++) {
                if (g[q]) {
                    int o = 0;
                    for (int k = 0; k < n; k++) if (R >> k & 1) o ^= PAR((unsigned)(w >> (16 * k)) & 0xffff & x[q]);
                    hit += o == e[q]; tot++; ones += e[q];
                }
                unsigned nb = PAR(a & uc[q]) ^ PAR(l & R) ^ (PAR(d & R) & PAR(f & R)) ^ (PAR(b & R) & PAR(c & uc[q]));
                R = ((R << 1) | nb) & NN;
            }
            passes++;
            if (passes <= 50)
                printf("FIT idx %llu a=%#x c=%#x l=%#x d=%#x f=%#x b=%#x init=%#x rank %d; held-out %d words acc %.4f (base %.4f)\n",
                       (unsigned long long)idx, a, c, l, d, f, b, R0, nu, tot, tot ? (double)hit / tot : 0,
                       tot ? 1 - (double)ones / tot : 0);
        }
        for (int k = 0; k < nu; k++) have[used[k]] = 0;
    }
    printf("range %llu..%llu: %llu fit all %d gated words; deepest reject %d gated words (idx %llu)\n",
           (unsigned long long)lo, (unsigned long long)hi, (unsigned long long)passes, gated, best, (unsigned long long)bestidx);
    return 0;
}
