import sys, numpy as np, time
from ns10 import load_dump
import features, gf2
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
b11 = ((p >> 11) & 1).astype(bool)
idx = np.arange(200, N - 3); idx = idx[G[idx]]
S = idx[b11[idx - 4]]; y = e[S]; print("rows with p[i-4].11 = 1 (enable certainly set):", len(S), "K rate", y.mean())
def allbits(ix):
    cols, names = [], []
    for w in range(-1, 9):
        for nm, arr in (("c", c), ("p", p), ("m", m)):
            if w == -1 and nm != "c": continue
            cols.append(features.bits16(arr[ix - w])); names += [f"{nm}[i-{w}].{bb}" for bb in range(16)]
    cols.append(np.stack([t[ix - k] for k in range(1, 9)], 1).astype(np.uint8)); names += [f"t-{k}" for k in range(1, 9)]
    return np.concatenate(cols, 1), names
B, names = allbits(S)
r1 = np.array([y[B[:, k] == 1].mean() if (B[:, k] == 1).sum() > 300 else 0.5 for k in range(B.shape[1])]); r0 = np.array([y[B[:, k] == 0].mean() if (B[:, k] == 0).sum() > 300 else 0.5 for k in range(B.shape[1])])
dev = np.abs(r1 - r0); o = np.argsort(-dev)
print("K per-bit strongest:", [(names[k], round(float(r1[k]), 3), round(float(r0[k]), 3)) for k in o[:10]])
print("K per-bit minima: rate1 %.4f (%s), rate0 %.4f (%s)" % (r1.min(), names[int(np.argmin(r1))], r0.min(), names[int(np.argmin(r0))]))
W = 6
def state(ix):
    cl = [np.ones((len(ix), 1), np.uint8)]
    for w in range(W): cl += [features.bits16(c[ix - w]), features.bits16(p[ix - w])]
    cl += [t[ix - k][:, None].astype(np.uint8) for k in range(1, 9)] + [features.bits16(c[ix + 1])]
    return np.concatenate(cl, 1)
XA = state(S[::max(1, len(S) // 30000)]); ra, piv = gf2.rank_and_pivots(XA)
E = S[y == 1]; XE = state(E[::max(1, len(E) // 30000)])[:, piv]; re_, _ = gf2.rank_and_pivots(XE); print(f"affine hull: all {ra}, events {re_}")
rows = S[::max(1, len(S) // 30000)][:30000]; X = state(rows); yy = e[rows]
res = gf2.solve(X, yy, "b11l"); print(f"linear: inconsistent {res['bad']} of {len(rows)-res['rank']}")
Q, qn = gf2.quad_features(X, piv); res = gf2.solve(Q, yy, "b11q"); print(f"quadratic: inconsistent {res['bad']} of {len(rows)-res['rank']}")
Bf = B.astype(np.float32); ef = y.astype(np.float32); n = len(S); ne = ef.sum()
C11 = Bf.T @ Bf; E11 = (Bf * ef[:, None]).T @ Bf; sx = np.diag(C11).copy(); ex = np.diag(E11).copy(); best = (1, None)
for pa in (1, 0):
    for pb in (1, 0):
        if (pa, pb) == (1, 1): Cab, Eab = C11, E11
        elif (pa, pb) == (1, 0): Cab, Eab = sx[:, None] - C11, ex[:, None] - E11
        elif (pa, pb) == (0, 1): Cab, Eab = sx[None, :] - C11, ex[None, :] - E11
        else: Cab, Eab = n - sx[:, None] - sx[None, :] + C11, ne - ex[:, None] - ex[None, :] + E11
        rate = np.where(Cab >= 1000, Eab / np.maximum(Cab, 1), 1.0); np.fill_diagonal(rate, 1.0)
        k = np.unravel_index(np.argmin(rate), rate.shape)
        if rate[k] < best[0]: best = (float(rate[k]), (names[k[0]], pa, names[k[1]], pb, int(Cab[k])))
print("lowest pair-conjunction K rate:", best)
# 2. clear condition: rows set at i-1 (p[i-5].11=1), not re-set at i (p[i-4].11=0)
Cr = idx[b11[idx - 5] & ~b11[idx - 4]]; yc = e[Cr]; print(f"\nrows set at i-1, not re-set: {len(Cr)}, rate {yc.mean():.3f} (survival ~ {yc.mean()/y.mean():.2f})")
Bc, _ = allbits(Cr)
c1 = np.array([yc[Bc[:, k] == 1].mean() if (Bc[:, k] == 1).sum() > 300 else 0.5 for k in range(Bc.shape[1])]); c0 = np.array([yc[Bc[:, k] == 0].mean() if (Bc[:, k] == 0).sum() > 300 else 0.5 for k in range(Bc.shape[1])])
o2 = np.argsort(np.minimum(c1, c0))
print("clear-condition candidates (lowest rates):", [(names[k], round(float(c1[k]), 3), round(float(c0[k]), 3)) for k in o2[:10]])
