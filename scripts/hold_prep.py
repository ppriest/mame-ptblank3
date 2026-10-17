import sys, numpy as np
from ns10 import load_dump
import features, gf2
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
NROWS = 400000; sel = np.arange(41, 41 + NROWS); W = 6
cols = [np.ones((len(sel), 1), np.uint8)]
for w in range(W): cols += [features.bits16(c[sel - w]), features.bits16(p[sel - w])]
cols += [t[sel - k][:, None].astype(np.uint8) for k in range(1, 9)]
X = np.concatenate(cols, 1); y = t[sel]
r, piv = gf2.rank_and_pivots(X[:30000]); Xp = X[:, piv]; F = Xp.shape[1]
cand, cn = [], []
for w in range(5):
    for nm, arr in (("c", c), ("p", p), ("m", m)):
        bits = features.bits16(arr[sel - w])
        for bb in range(16): cand.append(bits[:, bb]); cn.append(f"{nm}[i-{w}].{bb}")
for k in range(1, 5): cand.append(t[sel - k]); cn.append(f"t-{k}")
for k in range(1, 3): cand.append(e[sel - k]); cn.append(f"e-{k}")
C = np.stack(cand, 1).astype(np.uint8); K = C.shape[1]
Wd = (F + 1 + 63) // 64; A = np.zeros((len(sel), Wd * 64), np.uint8); A[:, :F] = Xp; A[:, F] = y
KW = (K + 63) // 64; Cp = np.zeros((len(sel), KW * 64), np.uint8); Cp[:, :K] = C
with open("hold.bin", "wb") as f:
    f.write(np.array([len(sel), F, K, 4000], np.uint32).tobytes())
    f.write(np.packbits(A, axis=1, bitorder='little').tobytes()); f.write(np.packbits(Cp, axis=1, bitorder='little').tobytes())
open("hold_names.txt", "w").write("\n".join(cn)); print("features", F, "candidates", K)
