import sys, numpy as np, time
from ns10 import load_dump
import gf2, features, pages
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c)
pg = pages.page_sequence(sys.argv[2]); page_addr = np.repeat(pg, 256); pos = np.arange(N) % 256; pidx = np.arange(N) // 256
sel = np.arange(16, N - 1); NF = 20000
L, names, y = features.window(c, p, t, 0, N, w=3); Ls = L[sel]; ys = y[sel]
r, piv = gf2.rank_and_pivots(Ls[:NF]); S = Ls[:, piv]
def test(counter, label, nbits=22):
    cb = ((counter[sel][:, None] >> np.arange(nbits)) & 1).astype(np.uint8)
    X = np.concatenate([S, cb], 1)[:NF]
    rk, pv = gf2.rank_and_pivots(X)
    Q, qn = gf2.quad_features(X, pv); nrows = min(NF, Q.shape[1] + 2500)
    res = gf2.solve(Q[:nrows], ys[:nrows], "cnt")
    print(f"  {label:40s}: rank {rk}, quad inconsistent {res['bad']} of {nrows - res['rank']}", flush=True)
for K in (257, 258, 259, 260, 264, 272, 288, 320):
    test(K * pidx + pos, f"K={K} words per page (index-based)")
for K in (516, 514, 520, 528, 544):
    test(K * pidx + 2 * pos, f"K={K} bytes per page (index-based)")
test(page_addr * 256 + pos, "NAND page address*256 + pos")
test(page_addr * 258 + pos, "NAND page address*258 + pos")
test(page_addr * 516 + 2 * pos, "NAND page address*516 + 2pos")
test(page_addr, "NAND page address only (16 bits)", 16)
