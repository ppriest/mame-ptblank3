import sys, numpy as np, itertools, time
from ns10 import load_dump
import features, gf2
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
lo, hi = [int(x) for x in sys.argv[2].split(",")]; which = sys.argv[3]   # word offsets lo..hi for the quadratic part; which: p / pm / cp
NROWS = 30000; sel = np.arange(60, N - 3)[::40][:NROWS]
lin = [np.ones((len(sel), 1), np.uint8)]
for w in range(0, 4): lin += [features.bits16(c[sel - w]), features.bits16(p[sel - w])]
for w in range(4, hi + 1): lin += [features.bits16(c[sel - w]), features.bits16(p[sel - w])]
lin += [t[sel - k][:, None].astype(np.uint8) for k in range(1, 9)]
Lm = np.concatenate(lin, 1); r, piv = gf2.rank_and_pivots(Lm[:12000]); Lp = Lm[:, piv]
qc = []
for w in range(lo, hi + 1):
    if "p" in which: qc.append(features.bits16(p[sel - w]))
    if "m" in which: qc.append(features.bits16(m[sel - w]))
    if "c" in which: qc.append(features.bits16(c[sel - w]))
# include the two gate bits so products with them are allowed
qc.append(((m[sel - 2] >> 0) & 1)[:, None].astype(np.uint8)); qc.append(((m[sel - 3] >> 0) & 1)[:, None].astype(np.uint8))
Q = np.concatenate(qc, 1); n = Q.shape[1]
prods = [(Q[:, i] & Q[:, j])[:, None] for i, j in itertools.combinations(range(n), 2)]
X = np.concatenate([Lp] + prods, 1); y = t[sel]
print(f"quadratic over {which} words i-{lo}..i-{hi} (+gate bits): {n} bits, {X.shape[1]} cols, lin rank {r}", flush=True)
t0 = time.time(); res = gf2.solve(X, y, "qd"); print(f"  rank {res['rank']} inconsistent {res['bad']} of {len(sel) - res['rank']} ({time.time()-t0:.0f}s)", flush=True)
