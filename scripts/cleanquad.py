import sys, numpy as np, itertools, time
from ns10 import load_dump
import features, gf2
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
L = 8
idx = np.arange(200, N - 3); ok = G[idx].copy()
for k in range(1, L + 1): ok &= G[idx - k]
rows = idx[ok]; print("rows with gate open at i and lags 1..%d: %d" % (L, len(rows)))
sub = rows[::max(1, len(rows) // 30000)][:30000]
W = 6
cl = [np.ones((len(sub), 1), np.uint8)]
for w in range(W): cl += [features.bits16(c[sub - w]), features.bits16(p[sub - w])]
cl += [features.bits16(c[sub + 1])] + [e[sub - k][:, None].astype(np.uint8) for k in range(1, L + 1)]
X = np.concatenate(cl, 1); y = e[sub]
r, piv = gf2.rank_and_pivots(X); res = gf2.solve(X, y, "cql"); print(f"linear (state + clean h-lags 1..{L}): rank {r}, inconsistent {res['bad']} of {len(sub)-res['rank']}")
t0 = time.time(); Q, qn = gf2.quad_features(X, piv); res = gf2.solve(Q, y, "cqq")
print(f"full quadratic: {Q.shape[1]} monomials, inconsistent {res['bad']} of {len(sub)-res['rank']} ({time.time()-t0:.0f}s)", flush=True)
