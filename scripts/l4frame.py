import sys, numpy as np
from ns10 import load_dump
import features, gf2
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N)
a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); clear = (a == 0) & (bq == 1)
z = (p == 0); before = np.zeros(N, np.int64)
for k in range(1, N): before[k] = before[k-1] + 1 if z[k] else 0
print("L4'-event rate in deep zero (>=40):", e[(before >= 40) & (i > 41)].mean(), " in zero >= 6:", e[(before >= 6) & (i > 41)].mean())
idx = np.arange(41, N - 3); ee = e[idx]; ncl = ~clear[idx]
print("P(e_i) on non-clear rows:", ee[ncl].mean())
for v1 in (0, 1):
    for v2 in (0, 1):
        sel = ncl & (e[idx - 1] == v1) & (e[idx - 2] == v2)
        print(f"  P(e_i | e_i-1={v1}, e_i-2={v2}) = {ee[sel].mean():.3f} (n={sel.sum()})")
for v1 in (0, 1):
    for v2 in (0, 1):
        for v3 in (0, 1):
            sel = ncl & (e[idx - 1] == v1) & (e[idx - 2] == v2) & (e[idx - 3] == v3)
            print(f"  P(e_i | e-1={v1}, e-2={v2}, e-3={v3}) = {ee[sel].mean():.3f} (n={sel.sum()})")
# exact linearity on sub-cells by (e-1, e-2) among non-clear rows
W = 6
def X_of(ix):
    cl = [np.ones((len(ix), 1), np.uint8)]
    for w in range(W): cl += [features.bits16(c[ix - w]), features.bits16(p[ix - w])]
    cl += [t[ix - k][:, None].astype(np.uint8) for k in range(1, 9)]
    return np.concatenate(cl, 1)
names = ['1'] + [f"{nm}{w}.{bb}" for w in range(W) for nm in ("c", "p") for bb in range(16)] + [f"t-{k}" for k in range(1, 9)]
for v1 in (0, 1):
    for v2 in (0, 1):
        rows = idx[ncl & (e[idx - 1] == v1) & (e[idx - 2] == v2)]
        X = X_of(rows); y = t[rows]; res = gf2.solve(X[:30000], y[:30000], "l4")
        print(f"  exact-linear test on non-clear rows with e-1={v1}, e-2={v2}: rows {len(rows)} rank {res['rank']} inconsistent {res['bad']} of {min(30000,len(rows))-res['rank']}")
        if res['bad'] == 0: print("     solution:", [names[k] for k in res['solution']])
