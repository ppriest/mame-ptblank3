import sys, numpy as np
from ns10 import load_dump
import pages
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
pg = pages.page_sequence(sys.argv[2]); page_addr = np.repeat(pg, 256); pidx = i // 256; pos = i % 256
idx = np.arange(60, N - 3); dense = G[idx].copy()
for w in range(0, 6): dense &= np.bitwise_count(p[idx - w]) >= 4
idx = idx[dense]; ee = e[idx].astype(np.float64); print("dense gate-open rows", len(idx), "rate", ee.mean())
def chi(v, M, label):
    r = v % M; s = np.bincount(r, weights=ee, minlength=M); n = np.bincount(r, minlength=M); rate = s / np.maximum(n, 1)
    exp = n * ee.mean(); chi2 = (((s - exp) ** 2) / np.maximum(exp * (1 - ee.mean()), 1e-9)).sum()
    return chi2, rate.min(), rate.max()
print("event rate vs counter residues (chi2 vs dof; min/max rate):")
for label, v in (("word index", i[idx]), ("pos in page", pos[idx]), ("page index", pidx[idx]), ("NAND page addr", page_addr[idx]),
                 ("bus cycles 516/page", 516 * pidx[idx] + 2 * pos[idx]), ("bus cycles 514/page", 514 * pidx[idx] + 2 * pos[idx]), ("bus cycles 520/page", 520 * pidx[idx] + 2 * pos[idx]),
                 ("bus cycles 528/page", 528 * pidx[idx] + 2 * pos[idx]), ("words 258/page", 258 * pidx[idx] + pos[idx]), ("words 260/page", 260 * pidx[idx] + pos[idx]), ("bus 516*addr", 516 * page_addr[idx] + 2 * pos[idx])):
    out = []
    for M in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        c2, lo, hi = chi(v.astype(np.int64), M, label); out.append(f"{M}:{c2/(M-1):.1f}")
    print(f"  {label:20s} chi2/dof: " + ' '.join(out))
# per-page event rate variance vs binomial (dense rows only, pages with >= 64 dense rows)
pp = pidx[idx]; s = np.bincount(pp, weights=ee); n = np.bincount(pp); ok = n >= 64
rates = s[ok] / n[ok]; exp_var = (ee.mean() * (1 - ee.mean()) / n[ok]).mean()
print(f"per-page rate: var {rates.var():.5f} vs binomial {exp_var:.5f}; pages {ok.sum()}")
# page address bits
pa = page_addr[idx]
print("rate by NAND page-address bit (bit: rate1/rate0):", ' '.join(f"{j}:{ee[(pa>>j)&1==1].mean():.3f}/{ee[(pa>>j)&1==0].mean():.3f}" for j in range(13)))
