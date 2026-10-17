import sys, numpy as np
from ns10 import load_dump
import features, gf2
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("nl_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
lo, hi = 41, N - 3; idx = np.arange(lo, hi); ee = e[idx]
a = ((m[idx - 2] >> 0) & 1); bq = ((m[idx - 3] >> 0) & 1)
cols, names = [], []
for w in range(-1, 7):
    for nm, arr in (("c", c), ("p", p), ("m", m)):
        if nm == "p" and w == -1: continue
        cols.append(features.bits16(arr[idx - w])); names += [f"{nm}[i-{w}].{bb}" for bb in range(16)]
cols.append(np.stack([t[idx - k] for k in range(1, 13)], 1).astype(np.uint8)); names += [f"t-{k}" for k in range(1, 13)]
cols.append(np.stack([e[idx - k] for k in range(1, 5)], 1).astype(np.uint8)); names += [f"e-{k}" for k in range(1, 5)]
B = np.concatenate(cols, 1)
def analyze(mask, label, minsup=1500):
    Bf = B[mask].astype(np.float32); ef = ee[mask].astype(np.float32); n = Bf.shape[0]; ne = ef.sum()
    # singles
    s1 = Bf.sum(0); e1 = (Bf * ef[:, None]).sum(0); r1 = e1 / np.maximum(s1, 1); r0 = (ne - e1) / np.maximum(n - s1, 1)
    k1 = int(np.argmin(np.where(s1 >= minsup, r1, 1))); k0 = int(np.argmin(np.where(n - s1 >= minsup, r0, 1)))
    print(f"[{label}] rows {n} events {int(ne)} rate {ne/n:.4f}")
    print(f"    lowest single-literal rates: {names[k1]}=1 -> {r1[k1]:.4f} (n={int(s1[k1])}); {names[k0]}=0 -> {r0[k0]:.4f} (n={int(n-s1[k0])})")
    C11 = Bf.T @ Bf; E11 = (Bf * ef[:, None]).T @ Bf; sx = np.diag(C11).copy(); ex = np.diag(E11).copy()
    found = []
    for pa in (1, 0):
        for pb in (1, 0):
            if (pa, pb) == (1, 1): Cab, Eab = C11, E11
            elif (pa, pb) == (1, 0): Cab, Eab = sx[:, None] - C11, ex[:, None] - E11
            elif (pa, pb) == (0, 1): Cab, Eab = sx[None, :] - C11, ex[None, :] - E11
            else: Cab, Eab = n - sx[:, None] - sx[None, :] + C11, ne - ex[:, None] - ex[None, :] + E11
            rate = np.where(Cab >= minsup, Eab / np.maximum(Cab, 1), 1.0); np.fill_diagonal(rate, 1.0)
            k = np.unravel_index(np.argmin(rate), rate.shape)
            found.append((float(rate[k]), names[k[0]], pa, names[k[1]], pb, int(Cab[k])))
    found.sort(); print("    lowest pair-conjunction rates:", found[:3])
    # hull test within cell (affine factor): events vs all
    def state(ix):
        cl = [np.ones((len(ix), 1), np.uint8)]
        for w in range(7): cl += [features.bits16(c[ix - w]), features.bits16(p[ix - w])]
        cl += [t[ix - k][:, None].astype(np.uint8) for k in range(1, 13)] + [features.bits16(c[ix + 1])]
        return np.concatenate(cl, 1)
    E = idx[mask & (ee == 1)]; A_ = idx[mask]
    re_, _ = gf2.rank_and_pivots(state(E[::max(1, len(E)//40000)])); ra, _ = gf2.rank_and_pivots(state(A_[::10]))
    print(f"    affine hull: events rank {re_}, all rank {ra}")
for ca, cb in ((0, 0), (1, 0), (1, 1)):
    analyze((a == ca) & (bq == cb), f"cell a={ca} b={cb}")
