import sys, numpy as np, itertools
from ns10 import load_dump
import features
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("nl_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
lo, hi = 41, N - 2; idx = np.arange(lo, hi); ee = e[idx]; E = idx[ee == 1]
# (A) P(x=1 | event) for all raw bits
rows = []
for w in range(-1, 7):
    for nm, arr in (("c", c), ("p", p), ("m", m)):
        bits = features.bits16(arr[E - w]); pe = bits.mean(0)
        bits_all = features.bits16(arr[idx - w]); pa = bits_all.mean(0)
        for bb in range(16): rows.append((abs(pe[bb] - 0.5), f"{nm}[i-{w}].{bb}", pe[bb], pa[bb]))
tl = np.stack([t[E - k] for k in range(1, 13)], 1).mean(0)
for k in range(12): rows.append((abs(tl[k] - 0.5), f"t-{k+1}", tl[k], 0.5))
rows.sort(reverse=True)
print("(A) most skewed raw bits given event (P(x=1|event), P(x=1|all)):"); [print("   ", r[1], "%.3f" % r[2], "%.3f" % r[3]) for r in rows[:12]]
# (B) XOR pairs over p,m,c bits of words i..i-5 : P(x^y=1 | event)
cols, names = [], []
for w in range(6):
    for nm, arr in (("c", c), ("p", p), ("m", m)):
        cols.append(features.bits16(arr[E - w])); names += [f"{nm}[i-{w}].{bb}" for bb in range(16)]
B = np.concatenate(cols, 1).astype(np.uint8); nE = B.shape[0]
best = (0, None)
s = B.sum(0).astype(np.int64)
# P(x^y=1) = (s_x + s_y - 2*s_xy)/n ; compute s_xy via matrix product on chunks
Bi = B.astype(np.int32)
G = Bi.T @ Bi   # (288,288) co-occurrence
sx = np.diag(G)
Pxor = (sx[:, None] + sx[None, :] - 2 * G) / nE
np.fill_diagonal(Pxor, 0.5)
k = np.unravel_index(np.argmax(np.abs(Pxor - 0.5)), Pxor.shape)
print(f"(B) most skewed XOR pair given event: {names[k[0]]} ^ {names[k[1]]}: P(=1|event) = {Pxor[k]:.3f}")
order = np.argsort(-np.abs(Pxor - 0.5).ravel())[:10]
for q in order: 
    a, bq = np.unravel_index(q, Pxor.shape)
    if a < bq: print(f"    {names[a]} ^ {names[bq]}: {Pxor[a,bq]:.3f}")
# (C) event rate by number of nonzero words among p_i..p_{i-5}, and by OR-popcount
nz = sum((p[idx - w] != 0).astype(np.int64) for w in range(6))
print("(C) rate by #nonzero words in p[i..i-5]:", [round(float(ee[nz == q].mean()), 4) for q in range(7)], " counts:", [int((nz == q).sum()) for q in range(7)])
orp = np.zeros(len(idx), np.uint16)
for w in range(6): orp |= p[idx - w]
opc = np.bitwise_count(orp).astype(np.int64)
print("    rate by popcount(OR of p[i..i-5]):", [round(float(ee[opc == q].mean()), 3) if (opc == q).sum() > 300 else None for q in range(17)])
# dense region only: all 6 words nonzero -> per-bit skew
D = idx[(nz == 6)]; eD = e[D]; print("    dense-region rate:", eD.mean(), "n", len(D))
# (D) event rate by offset into zero runs (new model)
z = (p == 0); before = np.zeros(N, np.int64)
for i in range(1, N): before[i] = before[i-1] + 1 if z[i] else 0
print("(D) event rate by zeros-before (0..12):", [round(float(e[(before == q) & (np.arange(N) > 41)].mean()), 3) for q in range(13)])
# (E) clustering
print("(E) P(e_i | e_{i-1}, e_{i-2}):", {f"{a}{bq}": round(float(ee[(e[idx-1]==a) & (e[idx-2]==bq)].mean()), 3) for a in (0,1) for bq in (0,1)})
