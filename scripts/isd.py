import sys, numpy as np
from ns10 import load_dump
import features, gf2
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); clear = (a == 0) & (bq == 1)
idx = np.arange(45, N - 3); E = idx[e[idx] == 1]; A_ = idx[~clear[idx]]
W = 6
def state(ix):
    cl = [np.ones((len(ix), 1), np.uint8)]
    for w in range(W): cl += [features.bits16(c[ix - w]), features.bits16(p[ix - w])]
    cl += [t[ix - k][:, None].astype(np.uint8) for k in range(1, 9)] + [features.bits16(c[ix + 1])]
    return np.concatenate(cl, 1)
names = ['1'] + [f"{nm}{w}.{bb}" for w in range(W) for nm in ("c", "p") for bb in range(16)] + [f"t-{k}" for k in range(1, 9)] + [f"cn.{bb}" for bb in range(16)]
XE = state(E); XA = state(A_[::5])
r, piv = gf2.rank_and_pivots(XA[:30000]); XEp = XE[:, piv]; XAp = XA[:, piv]; print("pivots", len(piv), "events", len(E))
rng = np.random.default_rng(7); found = {}; tried = 0
nonev = idx[(e[idx] == 0) & ~clear[idx]]; XN = state(nonev[rng.choice(len(nonev), 2000, replace=False)])[:, piv]
for trial in range(300):
    smp = rng.choice(len(E), size=len(piv) + 20, replace=False)
    Xs = np.concatenate([XEp[smp], XN[trial % 2000][None, :]], 0); ys = np.concatenate([np.ones(len(smp), np.uint8), [0]]).astype(np.uint8)
    res = gf2.solve(Xs, ys, "isd"); tried += 1
    if res['bad'] > 0: continue
    g = np.zeros(len(piv), np.uint8); g[res['solution']] = 1
    frac_e = float(((XEp.astype(np.int64) @ g.astype(np.int64)) % 2).mean()); frac_a = float(((XAp.astype(np.int64) @ g.astype(np.int64)) % 2).mean())
    key = tuple(np.where(g)[0])
    if frac_e > 0.9 and key not in found:
        found[key] = (frac_e, frac_a); print(f"trial {trial}: G=1 on {frac_e:.4f} of events, on {frac_a:.4f} of all non-clear rows; support {len(key)}: {[names[piv[k]] for k in key][:40]}", flush=True)
print("trials", tried, "consistent", sum(1 for _ in found))
print("consistent samples found:", len(found))
