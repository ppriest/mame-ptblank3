import sys, numpy as np, itertools, time
from ns10 import load_dump
import gf2
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
L = 8
idx = np.arange(200, N - 3); ok = G[idx].copy()
for k in range(1, L + 1): ok &= G[idx - k]
rows = idx[ok]; sub = rows[::max(1, len(rows) // 40000)][:40000]
cols, names = [], []
for k in range(0, 10): cols.append(((m[sub - k] >> 0) & 1).astype(np.uint8)); names.append(f"m{k}.0")
for k in range(3, 10):
    for bb in (11, 14, 15): cols.append(((p[sub - k] >> bb) & 1).astype(np.uint8)); names.append(f"p{k}.{bb}")
for k in range(1, L + 1): cols.append(e[sub - k].astype(np.uint8)); names.append(f"h{k}")
V = np.stack(cols, 1); n = V.shape[1]; y = e[sub]
r, piv = gf2.rank_and_pivots(np.concatenate([np.ones((len(sub), 1), np.uint8), V], 1)); print(f"vocabulary {n} bits, rank {r}, rows {len(sub)}")
for deg in (2, 3):
    mono = [np.ones(len(sub), np.uint8)] + [V[:, k] for k in range(n)]
    for dg in range(2, deg + 1):
        for comb in itertools.combinations(range(n), dg):
            v = V[:, comb[0]].copy()
            for q in comb[1:]: v &= V[:, q]
            mono.append(v)
    X = np.stack(mono, 1); t0 = time.time(); res = gf2.solve(X, y, "vc")
    print(f"degree {deg}: {X.shape[1]} monomials, rank {res['rank']}, inconsistent {res['bad']} of {len(sub)-res['rank']} ({time.time()-t0:.0f}s)", flush=True)
