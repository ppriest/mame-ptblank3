import sys, numpy as np, time
from ns10 import load_dump
import features, gf2
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
hb = (p & 0xff00) != 0
idx = np.arange(200, N - 3); idx = idx[G[idx]]
# 1. fresh set: p[i-4] has high byte, p[i-5..i-44] have none
fresh = hb[idx - 4].copy()
for w in range(5, 45): fresh &= ~hb[idx - w]
F = idx[fresh]; print("fresh-set rows:", len(F), "event rate", e[F].mean())
v = p[F - 4]
print("P(e | bit j of p[i-4] set / clear) on fresh-set rows:")
print("   " + ' '.join(f"{j}:{e[F][(v>>j)&1==1].mean():.2f}/{e[F][(v>>j)&1==0].mean():.2f}" for j in range(16)))
# rows where only ONE high-byte bit is set in p[i-4]
for j in range(8, 16):
    sel = ((v & 0xff00) == (1 << j))
    if sel.sum() > 30: print(f"   only bit {j} set in high byte: rate {e[F][sel].mean():.3f} (n={sel.sum()})")
# 2. K on certain-set rows (cnt==0): p[i-4] high byte != 0
S = idx[hb[idx - 4]]; print("\ncertain-set rows:", len(S), "K rate", e[S].mean())
cols, names = [], []
for w in range(-1, 9):
    for nm, arr in (("c", c), ("p", p), ("m", m)):
        if w == -1 and nm != "c": continue
        cols.append(features.bits16(arr[S - w])); names += [f"{nm}[i-{w}].{bb}" for bb in range(16)]
cols.append(np.stack([t[S - k] for k in range(1, 9)], 1).astype(np.uint8)); names += [f"t-{k}" for k in range(1, 9)]
B = np.concatenate(cols, 1); y = e[S]
r1 = np.array([y[B[:, k] == 1].mean() if (B[:, k] == 1).sum() > 500 else 0.5 for k in range(B.shape[1])]); r0 = np.array([y[B[:, k] == 0].mean() if (B[:, k] == 0).sum() > 500 else 0.5 for k in range(B.shape[1])])
dev = np.abs(r1 - r0); o = np.argsort(-dev)
print("K per-bit: strongest modulators (bit, rate1, rate0):", [(names[k], round(float(r1[k]), 3), round(float(r0[k]), 3)) for k in o[:8]])
print("K per-bit: minimum rates: rate1 min %.4f at %s ; rate0 min %.4f at %s" % (r1.min(), names[int(np.argmin(r1))], r0.min(), names[int(np.argmin(r0))]))
# hull / quadratic on certain-set rows
W = 6
def state(ix):
    cl = [np.ones((len(ix), 1), np.uint8)]
    for w in range(W): cl += [features.bits16(c[ix - w]), features.bits16(p[ix - w])]
    cl += [t[ix - k][:, None].astype(np.uint8) for k in range(1, 9)] + [features.bits16(c[ix + 1])]
    return np.concatenate(cl, 1)
XA = state(S[::max(1, len(S) // 30000)]); ra, piv = gf2.rank_and_pivots(XA)
E = S[y == 1]; XE = state(E[::max(1, len(E) // 30000)])[:, piv]; re_, _ = gf2.rank_and_pivots(XE)
print(f"affine hull on certain-set rows: all {ra}, events {re_}")
rows = S[::max(1, len(S) // 30000)][:30000]; X = state(rows); yy = e[rows]
res = gf2.solve(X, yy, "fl"); print(f"linear on certain-set rows: inconsistent {res['bad']} of {len(rows)-res['rank']}")
Q, qn = gf2.quad_features(X, piv); res = gf2.solve(Q, yy, "fq"); print(f"quadratic on certain-set rows: {Q.shape[1]} monomials, inconsistent {res['bad']} of {len(rows)-res['rank']}")
# forbidden pairs on certain-set rows
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
print("lowest pair-conjunction event rate on certain-set rows:", best)
