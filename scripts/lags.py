import sys, numpy as np, time
from ns10 import load_dump
import gf2, features
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c)
NL = 32
def tlags(idx):
    return np.stack([t[idx - k] for k in range(1, NL + 1)], 1).astype(np.uint8), [f"t-{k}" for k in range(1, NL + 1)]
def report(label, X, y, names, nfit=30000, quad=True):
    r, piv = gf2.rank_and_pivots(X[:nfit])
    res = gf2.solve(X[:nfit], y[:nfit], "lin")
    pred = np.zeros(len(y), np.uint8)
    for k in res['solution']: pred ^= X[:, k]
    print(f"[{label}] {X.shape[1]} cols rank {r}; linear: inconsistent {res['bad']}/{nfit - res['rank']} ; mismatches on all {len(y)}: {int((pred != y).sum())}", flush=True)
    if res['bad'] == 0: print("   linear solution:", [names[k] for k in res['solution']])
    if quad:
        Q, qn = gf2.quad_features(X[:nfit], piv); nrows = min(nfit, Q.shape[1] + 4000)
        t0 = time.time(); res = gf2.solve(Q[:nrows], y[:nrows], "quad")
        print(f"   quad {Q.shape[1]} monomials {nrows} rows: rank {res['rank']} inconsistent {res['bad']}/{nrows - res['rank']} ({time.time()-t0:.1f}s)", flush=True)
        if res['bad'] == 0:
            pred = np.zeros(len(y), np.uint8)
            for k in res['solution']:
                mono = qn[k]
                if len(mono) == 0: pred ^= 1
                elif len(mono) == 1: pred ^= X[:, piv[mono[0]]]
                else: pred ^= X[:, piv[mono[0]]] & X[:, piv[mono[1]]]
            print(f"   quad mismatches on all: {int((pred != y).sum())}; terms: {[tuple(names[piv[q]] for q in qn[k]) for k in res['solution']]}")
# A. zero regions
z = (p == 0); before = np.zeros(N, np.int64)
for i in range(1, N): before[i] = before[i-1] + 1 if z[i] else 0
idx = np.where((before >= 40) & (np.arange(N) >= 40) & (np.arange(N) < N - 1))[0]
TL, tn = tlags(idx)
X = np.concatenate([features.bits16(c[idx]), features.bits16(c[idx-1]), features.bits16(c[idx-2]), TL], 1)
names = [f"c0.{k}" for k in range(16)] + [f"c1.{k}" for k in range(16)] + [f"c2.{k}" for k in range(16)] + tn
report("ZERO: c 3 words + t lags", X, t[idx], names, quad=False)
# B. global
sel = np.arange(40, N - 1)
L3, n3, y = features.window(c, p, t, 0, N, w=3); L3 = L3[sel]; y = y[sel]
TL, tn = tlags(sel)
report("GLOBAL: (c,p) 3 words + t lags", np.concatenate([L3, TL], 1), y, n3 + tn)
cn = features.bits16(c[sel + 1]); cnn = [f"cn.{k}" for k in range(16)]
report("GLOBAL: (c,p) 3 words + t lags + c next", np.concatenate([L3, TL, cn], 1), y, n3 + tn + cnn)
L4, n4, _ = features.window(c, p, t, 0, N, w=4); L4 = L4[sel]
report("GLOBAL: (c,p) 4 words + t lags", np.concatenate([L4, TL], 1), y, n4 + tn)
