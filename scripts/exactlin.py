import sys, numpy as np
from ns10 import load_dump
import features, gf2
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
lo, hi = 41, N - 3; idx = np.arange(lo, hi)
a0 = ((m[idx - 2] >> 0) & 1) == 1; b0 = ((m[idx - 3] >> 0) & 1) == 1
cell = idx[(~a0) & b0]; print("clean cell rows:", len(cell))
W = 7
def X_of(ix):
    cl = [np.ones((len(ix), 1), np.uint8)]
    for w in range(W): cl += [features.bits16(c[ix - w]), features.bits16(p[ix - w])]
    cl += [t[ix - k][:, None].astype(np.uint8) for k in range(1, 13)]
    cl += [features.bits16(c[ix + 1])]
    return np.concatenate(cl, 1)
names = ['1'] + [f"{nm}{w}.{bb}" for w in range(W) for nm in ("c", "p") for bb in range(16)] + [f"t-{k}" for k in range(1, 13)] + [f"cn.{bb}" for bb in range(16)]
Xc = X_of(cell); yc = t[cell]
res = gf2.solve(Xc[:40000], yc[:40000], "cell"); print(f"exact linear fit on clean cell (40k rows): rank {res['rank']}, inconsistent {res['bad']} of {40000 - res['rank']}")
pred = np.zeros(len(cell), np.uint8)
for k in res['solution']: pred ^= Xc[:, k]
print("  mismatches on all cell rows:", int((pred != yc).sum()), "of", len(cell))
print("  solution:", [names[k] for k in res['solution']])
# apply this exact linear model to ALL rows -> events
Xa = X_of(idx); pa = np.zeros(len(idx), np.uint8)
for k in res['solution']: pa ^= Xa[:, k]
e = (pa != t[idx]).astype(np.uint8); print("event rate on all rows with exact model:", e.mean(), " in cell:", e[(~a0) & b0].mean(), " outside cell:", e[a0 | ~b0].mean())
np.save("nl_events2.npy", idx[e == 1]); np.save("exact_lin_solution.npy", np.array(res['solution']))
# compare with FWHT model events
ev1 = np.load("nl_events.npy"); e1 = np.zeros(N, np.uint8); e1[ev1] = 1
print("agreement between FWHT-model events and exact-model events:", float((e1[idx] == e).mean()))
