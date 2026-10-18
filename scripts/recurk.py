import sys, numpy as np, itertools
from ns10 import load_dump
import features, gf2
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
W = 7
def S_of(ix):
    cl = [np.ones((len(ix), 1), np.uint8)]
    for w in range(W): cl += [features.bits16(c[ix - w]), features.bits16(p[ix - w])]
    cl += [features.bits16(c[ix + 1])]
    return np.concatenate(cl, 1)
idx = np.arange(80, N - 3)
def test(lagset, label):
    ok = G[idx].copy()
    for k in lagset: ok &= G[idx - k]
    rows = idx[ok]; rows = rows[::max(1, len(rows) // 30000)][:30000]
    S = S_of(rows); r, piv = gf2.rank_and_pivots(S); Sp = S[:, piv]
    H = [e[rows - k].astype(np.uint8) for k in lagset]
    hm = []
    for dg in range(1, len(lagset) + 1):
        for comb in itertools.combinations(range(len(lagset)), dg):
            v = np.ones(len(rows), np.uint8)
            for q in comb: v &= H[q]
            hm.append(v)
    X = np.concatenate([Sp] + [(hv[:, None] & Sp) for hv in hm] + [hv[:, None] for hv in hm], 1)
    res = gf2.solve(X, e[rows], "rk"); free = len(rows) - res['rank']
    print(f"  {label:22s}: rows {len(rows)}, cols {X.shape[1]}, inconsistent {res['bad']} of {free} ({res['bad']/free:.3f})", flush=True)
for k in range(1, 13): test((k,), f"lag {k}")
for ls in ((4, 5), (4, 8), (3, 4), (4, 5, 6), (2, 4), (5, 6), (6, 7), (4, 6)): test(ls, f"lags {ls}")
