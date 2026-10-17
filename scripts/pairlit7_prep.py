import sys, numpy as np
from ns10 import load_dump
import gf2, features
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
W = 7; NLAG = 12; NSUB = 3000; NROWS = 300000
sel = np.arange(16, 16 + NROWS)
cols = []
for k in range(W): cols += [features.bits16(c[sel - k]), features.bits16(p[sel - k])]
for k in range(1, NLAG + 1): cols.append(t[sel - k][:, None].astype(np.uint8))
X = np.concatenate([np.ones((len(sel), 1), np.uint8)] + cols, 1); y = t[sel]
r, piv = gf2.rank_and_pivots(X[:30000]); Xp = X[:, piv]; F = Xp.shape[1]
cand = []; cnames = []
for k in range(W):
    for bb in range(16):
        cand.append(features.bits16(c[sel - k])[:, bb]); cnames.append(f"c{k}.{bb}")
        cand.append(features.bits16(p[sel - k])[:, bb]); cnames.append(f"p{k}.{bb}")
        cand.append(features.bits16(m[sel - k])[:, bb]); cnames.append(f"m{k}.{bb}")
for bb in range(16): cand.append(features.bits16(c[sel + 1])[:, bb]); cnames.append(f"cn.{bb}")
for k in range(1, NLAG + 1): cand.append(t[sel - k]); cnames.append(f"t-{k}")
C = np.stack(cand, 1).astype(np.uint8); K = C.shape[1]
Wd = (F + 1 + 63) // 64; A = np.zeros((len(sel), Wd * 64), np.uint8); A[:, :F] = Xp; A[:, F] = y
KW = (K + 63) // 64; Cp = np.zeros((len(sel), KW * 64), np.uint8); Cp[:, :K] = C
with open("pairlit7.bin", "wb") as f:
    f.write(np.array([len(sel), F, K, NSUB], np.uint32).tobytes())
    f.write(np.packbits(A, axis=1, bitorder='little').tobytes()); f.write(np.packbits(Cp, axis=1, bitorder='little').tobytes())
open("pairlit7_names.txt", "w").write("\n".join(cnames))
print("features", F, "candidates", K, "rows", len(sel))
