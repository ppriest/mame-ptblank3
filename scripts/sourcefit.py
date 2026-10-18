import sys, numpy as np, time
from ns10 import load_dump
import features, gf2
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
M = 0xc980; D = np.zeros(N, bool)
for w in range(4, 21): D |= (p[np.maximum(i - w, 0)] & M) != 0
obs = G & D                          # rows where the source K = e is observed
print("observed-source rows:", obs[100:].sum(), "K rate", e[obs].mean())
# (i) linear / quadratic fits of K on the 6-word state, observed rows only
idx = np.arange(100, N - 3); rows = idx[obs[idx]]; rows = rows[::max(1, len(rows) // 30000)][:30000]
W = 6
cl = [np.ones((len(rows), 1), np.uint8)]
for w in range(W): cl += [features.bits16(c[rows - w]), features.bits16(p[rows - w])]
cl += [t[rows - k][:, None].astype(np.uint8) for k in range(1, 9)] + [features.bits16(c[rows + 1])]
X = np.concatenate(cl, 1); y = e[rows]
r, piv = gf2.rank_and_pivots(X); res = gf2.solve(X, y, "sl"); print(f"(i) linear on observed rows: rank {r}, inconsistent {res['bad']} of {len(rows)-res['rank']}")
Q, qn = gf2.quad_features(X, piv); res = gf2.solve(Q, y, "sq"); print(f"    quadratic: {Q.shape[1]} monomials, inconsistent {res['bad']} of {len(rows)-res['rank']}", flush=True)
# (ii) LFSR-type recurrence of K on its own past over runs of consecutive observed rows
for order in (8, 12, 16, 20, 24, 32):
    ok = obs[idx].copy()
    for k in range(1, order + 1): ok &= obs[idx - k]
    rr = idx[ok]
    if len(rr) < order + 200: print(f"(ii) order {order}: only {len(rr)} rows"); continue
    H = np.stack([e[rr - k] for k in range(1, order + 1)], 1).astype(np.uint8)
    X2 = np.concatenate([np.ones((len(rr), 1), np.uint8), H], 1)
    res = gf2.solve(X2[:20000], e[rr][:20000], "lf"); free = min(20000, len(rr)) - res['rank']
    # with data terms too
    X3 = np.concatenate([X2] + [features.bits16(c[rr - w]) for w in range(0, 4)] + [features.bits16(p[rr - w]) for w in range(0, 4)], 1)
    res3 = gf2.solve(X3[:20000], e[rr][:20000], "lf3"); free3 = min(20000, len(rr)) - res3['rank']
    print(f"(ii) order {order}: rows {len(rr)}; pure recurrence inconsistent {res['bad']}/{free}; with 4-word data terms {res3['bad']}/{free3}", flush=True)
# (iii) autocorrelation of K on consecutive observed pairs, and MI with single words
pair = idx[obs[idx] & obs[idx - 1]]
print(f"(iii) P(K_i | K_i-1=1) = {e[pair][e[pair-1]==1].mean():.3f}, P(K_i | K_i-1=0) = {e[pair][e[pair-1]==0].mean():.3f}")
