import sys, numpy as np, time
from ns10 import load_dump
import features, gf2
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("nl_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
sel = np.arange(41, N - 3)
def build(extra_e, cnext, W=6):
    cl = [np.ones((len(sel), 1), np.uint8)]
    for w in range(W): cl += [features.bits16(c[sel - w]), features.bits16(p[sel - w])]
    cl += [t[sel - k][:, None].astype(np.uint8) for k in range(1, 9)]
    if cnext: cl.append(features.bits16(c[sel + 1]))
    if extra_e: cl += [e[sel - k][:, None].astype(np.uint8) for k in range(1, 5)]
    return np.concatenate(cl, 1)
y = t[sel]
for label, X in (("6w+lags", build(False, False)), ("6w+lags+cnext", build(False, True)), ("6w+lags+cnext+elags", build(True, True))):
    r, piv = gf2.rank_and_pivots(X[:30000])
    Q, qn = gf2.quad_features(X[:30000], piv); nrows = min(30000, Q.shape[1] + 3000)
    t0 = time.time(); res = gf2.solve(Q[:nrows], y[:nrows], "q6")
    print(f"[{label}] lin rank {r}; quad {Q.shape[1]} monomials, {nrows} rows: rank {res['rank']}, inconsistent {res['bad']} of {nrows - res['rank']} ({time.time()-t0:.0f}s)", flush=True)
