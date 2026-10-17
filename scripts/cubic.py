import sys, numpy as np, itertools, time
from ns10 import load_dump
import features, gf2
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
offs = [int(x) for x in sys.argv[2].split(",")]   # plaintext word offsets for the cubic part
deg = int(sys.argv[3]) if len(sys.argv) > 3 else 3
NROWS = int(sys.argv[4]) if len(sys.argv) > 4 else 24000
sel = np.arange(45, 45 + NROWS * 3, 3)[:NROWS]
W = 6
lin = [np.ones((len(sel), 1), np.uint8)]
for w in range(W): lin += [features.bits16(c[sel - w]), features.bits16(p[sel - w])]
lin += [t[sel - k][:, None].astype(np.uint8) for k in range(1, 9)]
Lm = np.concatenate(lin, 1); r, piv = gf2.rank_and_pivots(Lm[:12000]); Lp = Lm[:, piv]
P = np.concatenate([features.bits16(p[sel - w]) for w in offs], 1); n = P.shape[1]
cols = [Lp]
for dg in range(2, deg + 1):
    for comb in itertools.combinations(range(n), dg):
        v = P[:, comb[0]].copy()
        for q in comb[1:]: v &= P[:, q]
        cols.append(v[:, None])
X = np.concatenate(cols, 1); y = t[sel]
print(f"plaintext words {offs}, degree {deg}: {n} bits -> {X.shape[1]} columns, {len(sel)} rows", flush=True)
t0 = time.time(); res = gf2.solve(X, y, "cubic")
print(f"  rank {res['rank']}, inconsistent {res['bad']} of {len(sel) - res['rank']}  ({time.time()-t0:.0f}s)", flush=True)
