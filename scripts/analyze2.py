import sys, numpy as np, time
from ns10 import load_dump
import gf2, features
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c)
pos = np.arange(N)
def fit(w, extra_cols=None, extra_names=(), label="", nfit=24000, first=8):
    sel = np.arange(first, N - 1)
    L, names, y = features.window(c, p, t, 0, N, w=w)
    L = L[sel]; y = y[sel]
    if extra_cols is not None:
        L = np.concatenate([L, extra_cols[sel]], 1); names = names + list(extra_names)
    r, piv = gf2.rank_and_pivots(L[:nfit])
    Q, qn = gf2.quad_features(L[:nfit], piv)
    nrows = min(nfit, Q.shape[1] + 3000)
    t0 = time.time(); res = gf2.solve(Q[:nrows], y[:nrows], "quad")
    free = nrows - res['rank']
    print(f"[{label}] w={w} +{len(extra_names)} extra: lin rank {r}/{L.shape[1]}; quad {Q.shape[1]} monomials, {nrows} rows: rank {res['rank']}, inconsistent {res['bad']} of {free} free rows ({time.time()-t0:.1f}s)", flush=True)
    return res
posbits = ((pos[:, None] >> np.arange(21)) & 1).astype(np.uint8)
for w in (4, 5, 6, 8):
    fit(w, label=f"W{w}", first=w+2)
fit(3, posbits, [f"pos.{k}" for k in range(21)], label="POS21")
fit(3, posbits[:, :8], [f"pos.{k}" for k in range(8)], label="POS8")
