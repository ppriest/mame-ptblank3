import sys, numpy as np, time
from ns10 import load_dump
import features, gf2
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
def words_since(reset, offset):
    # cnt_i = number of words since the most recent j <= i-offset with reset[j], i.e. (i-offset) - j ; large if none
    last = np.full(N, -10**6, np.int64); lj = -10**6
    for k in range(N):
        if reset[k]: lj = k
        last[k] = lj
    sh = np.maximum(i - offset, 0)
    return np.minimum(i - offset - last[sh], 1023)
idx = np.arange(200, N - 3); idx = idx[G[idx]]; ee = e[idx]
for M in (0xff00, 0xc980):
    cnt_ = words_since((p & M) != 0, 4)[idx]
    rates = [(k, float(ee[cnt_ == k].mean()), int((cnt_ == k).sum())) for k in range(0, 96) if (cnt_ == k).sum() > 200]
    print(f"reset mask 0x{M:04x}: P(e | words since reset) for 0..95:")
    print("   " + ' '.join(f"{k}:{r:.2f}" for k, r, n in rates))
    print("   P(e | cnt >= 64) = %.4f (n=%d);  P(e | cnt >= 96) = %.4f" % (ee[cnt_ >= 64].mean(), (cnt_ >= 64).sum(), ee[cnt_ >= 96].mean()))
    # quadratic fit with counter bits added, gate-open rows, cnt < 128 (7 bits)
    rows = idx[cnt_ < 128]; rows = rows[::max(1, len(rows) // 30000)][:30000]; cb = words_since((p & M) != 0, 4)[rows]
    W = 6
    cl = [np.ones((len(rows), 1), np.uint8)]
    for w in range(W): cl += [features.bits16(c[rows - w]), features.bits16(p[rows - w])]
    cl += [t[rows - k][:, None].astype(np.uint8) for k in range(1, 9)] + [features.bits16(c[rows + 1])]
    cl.append(((cb[:, None] >> np.arange(7)) & 1).astype(np.uint8))
    X = np.concatenate(cl, 1); y = e[rows]
    r, piv = gf2.rank_and_pivots(X); res = gf2.solve(X, y, "cl"); print(f"   linear with counter bits: rank {r}, inconsistent {res['bad']} of {len(rows)-res['rank']}")
    t0 = time.time(); Q, qn = gf2.quad_features(X, piv); res = gf2.solve(Q, y, "cq")
    print(f"   quadratic with counter bits: {Q.shape[1]} monomials, inconsistent {res['bad']} of {len(rows)-res['rank']} ({time.time()-t0:.0f}s)", flush=True)
