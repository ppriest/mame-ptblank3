import sys, numpy as np
from ns10 import load_dump
import features, gf2
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
lo, hi = 41, N - 3; idx = np.arange(lo, hi)
a = ((m[idx - 2] >> 0) & 1).astype(np.uint8); bq = ((m[idx - 3] >> 0) & 1).astype(np.uint8)
W = int(sys.argv[2]) if len(sys.argv) > 2 else 5
def X_of(ix):
    cl = [np.ones((len(ix), 1), np.uint8)]
    for w in range(W): cl += [features.bits16(c[ix - w]), features.bits16(p[ix - w])]
    cl += [t[ix - k][:, None].astype(np.uint8) for k in range(1, 9)]
    return np.concatenate(cl, 1)
names = ['1'] + [f"{nm}{w}.{bb}" for w in range(W) for nm in ("c", "p") for bb in range(16)] + [f"t-{k}" for k in range(1, 9)]
for ca in (0, 1):
    for cb in (0, 1):
        rows = idx[(a == ca) & (bq == cb)]
        X = X_of(rows); y = t[rows]
        res = gf2.solve(X[:40000], y[:40000], "cell")
        pred = np.zeros(len(rows), np.uint8)
        for k in res['solution']: pred ^= X[:, k]
        mis = int((pred != y).sum())
        print(f"cell a={ca} b={cb}: rows {len(rows)}, rank {res['rank']}, inconsistent {res['bad']} of {40000-res['rank']}, mismatches on all rows {mis}")
        if res['bad'] == 0: print("    solution:", [names[k] for k in res['solution']])
