import sys, numpy as np
from ns10 import load_dump
import features, gf2
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
W = 6
def S_of(ix):
    cl = [np.ones((len(ix), 1), np.uint8)]
    for w in range(W): cl += [features.bits16(c[ix - w]), features.bits16(p[ix - w])]
    cl += [features.bits16(c[ix + 1])]
    return np.concatenate(cl, 1)
idx = np.arange(60, N - 3)
for lags in (1, 2, 3):
    ok = G[idx].copy()
    for k in range(1, lags + 1): ok &= G[idx - k]
    rows = idx[ok]; rows = rows[::max(1, len(rows) // 30000)][:30000]
    S = S_of(rows); r, piv = gf2.rank_and_pivots(S); Sp = S[:, piv]
    H = np.stack([e[rows - k] for k in range(1, lags + 1)], 1).astype(np.uint8)
    # all monomials in the hidden lags (2^lags - 1 nontrivial) times (1 + state)
    import itertools
    hm = [np.ones(len(rows), np.uint8)]
    for dg in range(1, lags + 1):
        for comb in itertools.combinations(range(lags), dg):
            v = np.ones(len(rows), np.uint8)
            for q in comb: v &= H[:, q]
            hm.append(v)
    X = np.concatenate([Sp] + [(hv[:, None] & Sp) for hv in hm[1:]] + [hv[:, None] for hv in hm[1:]], 1)
    y = e[rows]
    res = gf2.solve(X, y, "rc")
    print(f"h_i = f(h_i-1..h_i-{lags}, state) with gates open at all steps: rows {len(rows)}, cols {X.shape[1]}, rank {res['rank']}, inconsistent {res['bad']} of {len(rows) - res['rank']}", flush=True)
