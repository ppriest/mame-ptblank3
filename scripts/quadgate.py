import sys, numpy as np, time
from ns10 import load_dump
import features, gf2
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); gate = ~((a == 0) & (bq == 1))
W = int(sys.argv[2]) if len(sys.argv) > 2 else 6
def X_of(ix, cnext=True, lags=8):
    cl = [np.ones((len(ix), 1), np.uint8)]
    for w in range(W): cl += [features.bits16(c[ix - w]), features.bits16(p[ix - w])]
    cl += [t[ix - k][:, None].astype(np.uint8) for k in range(1, lags + 1)]
    if cnext: cl.append(features.bits16(c[ix + 1]))
    return np.concatenate(cl, 1)
def run(rows, label, nfit=30000):
    rows = rows[::max(1, len(rows) // nfit)][:nfit]
    X = X_of(rows); y = t[rows]
    r, piv = gf2.rank_and_pivots(X)
    Q, qn = gf2.quad_features(X, piv)
    t0 = time.time(); res = gf2.solve(Q, y, "qg")
    print(f"[{label}] rows {len(rows)}, lin rank {r}, quad {Q.shape[1]} monomials: rank {res['rank']}, inconsistent {res['bad']} of {len(rows) - res['rank']} ({time.time()-t0:.0f}s)", flush=True)
idx = np.arange(60, N - 3)
run(idx[gate[idx]], f"gate-open rows, W={W}")
for ca, cb in ((0, 0), (1, 0), (1, 1)):
    run(idx[(a[idx] == ca) & (bq[idx] == cb)], f"cell ({ca},{cb}), W={W}")
