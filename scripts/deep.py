import sys, numpy as np
from ns10 import load_dump
import features, gf2
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); gate = ~((a == 0) & (bq == 1))
idx = np.arange(60, N - 3); idx = idx[gate[idx]]; ee = e[idx]
# (a) forbidden single literals and pairs over words i-2..i-12 (c,p,m)
cols, names = [], []
for w in range(2, 13):
    for nm, arr in (("c", c), ("p", p), ("m", m)):
        cols.append(features.bits16(arr[idx - w])); names += [f"{nm}[i-{w}].{bb}" for bb in range(16)]
B = np.concatenate(cols, 1)
Bf = B[::2].astype(np.float32); ef = ee[::2].astype(np.float32); n = Bf.shape[0]; ne = ef.sum()
s1 = Bf.sum(0); e1 = (Bf * ef[:, None]).sum(0); r1 = e1 / np.maximum(s1, 1); r0 = (ne - e1) / np.maximum(n - s1, 1)
print("(a) single literals: min rate(x=1) %.4f at %s ; min rate(x=0) %.4f at %s" % (r1.min(), names[int(np.argmin(r1))], r0.min(), names[int(np.argmin(r0))]))
C11 = Bf.T @ Bf; E11 = (Bf * ef[:, None]).T @ Bf; sx = np.diag(C11).copy(); ex = np.diag(E11).copy()
out = []
for pa in (1, 0):
    for pb in (1, 0):
        if (pa, pb) == (1, 1): Cab, Eab = C11, E11
        elif (pa, pb) == (1, 0): Cab, Eab = sx[:, None] - C11, ex[:, None] - E11
        elif (pa, pb) == (0, 1): Cab, Eab = sx[None, :] - C11, ex[None, :] - E11
        else: Cab, Eab = n - sx[:, None] - sx[None, :] + C11, ne - ex[:, None] - ex[None, :] + E11
        rate = np.where(Cab >= 1500, Eab / np.maximum(Cab, 1), 1.0); np.fill_diagonal(rate, 1.0)
        ii, jj = np.where(rate == 0)
        for i_, j_ in zip(ii, jj):
            if i_ < j_: out.append((int(Cab[i_, j_]), names[i_], pa, names[j_], pb))
        k = np.unravel_index(np.argmin(rate), rate.shape); out.append((-1, f"min {rate[k]:.4f}", names[k[0]], pa, names[k[1]], pb, int(Cab[k])))
out.sort(key=lambda r: -r[0]); print("    forbidden pairs (0 events):", [o for o in out if o[0] > 0][:10]); print("    lowest:", [o for o in out if o[0] < 0][:4])
# (c) hull test over (c,p) words i-4..i-10 + const
def state(ix):
    cl = [np.ones((len(ix), 1), np.uint8)]
    for w in range(4, 11): cl += [features.bits16(c[ix - w]), features.bits16(p[ix - w])]
    return np.concatenate(cl, 1)
E = idx[ee == 1]; re_, _ = gf2.rank_and_pivots(state(E[::3])); ra, _ = gf2.rank_and_pivots(state(idx[::15]))
print(f"(c) affine hull over words i-4..i-10: events rank {re_}, all rank {ra}")
