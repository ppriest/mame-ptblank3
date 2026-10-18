import sys, numpy as np
from ns10 import load_dump
import features, gf2
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
idx = np.arange(80, N - 3); idx = idx[G[idx]]; E = idx[e[idx] == 1]
W = 11
def state(ix):
    cl = [np.ones((len(ix), 1), np.uint8)]
    for w in range(W): cl += [features.bits16(c[ix - w]), features.bits16(p[ix - w])]
    cl += [t[ix - k][:, None].astype(np.uint8) for k in range(1, 9)] + [features.bits16(c[ix + 1])]
    return np.concatenate(cl, 1)
XE = state(E[::2]); XA = state(idx[::12])
r, piv = gf2.rank_and_pivots(XA[:40000]); XEp = XE[:, piv]; XAp = XA[:, piv]
re_, _ = gf2.rank_and_pivots(XEp[:60000]); print(f"span words i..i-10: pivots {len(piv)}; exact hull rank events {re_} vs all {len(piv)}")
rng = np.random.default_rng(11); found = 0
XN = state(idx[e[idx] == 0][rng.choice((e[idx] == 0).sum(), 400, replace=False)])[:, piv]
for trial in range(400):
    smp = rng.choice(len(XEp), size=len(piv) + 25, replace=False)
    Xs = np.concatenate([XEp[smp], XN[trial % 400][None, :]], 0); ys = np.concatenate([np.ones(len(smp), np.uint8), [0]]).astype(np.uint8)
    res = gf2.solve(Xs, ys, "isdw")
    if res['bad'] > 0: continue
    g = np.zeros(len(piv), np.uint8); g[res['solution']] = 1
    fe = float(((XEp.astype(np.int64) @ g.astype(np.int64)) % 2).mean())
    if fe > 0.9: found += 1; print(f"  trial {trial}: form =1 on {fe:.4f} of events; support {int(g.sum())}")
print("robust affine factors found:", found, "(of 400 trials)")
