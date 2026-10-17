import sys, numpy as np, itertools, time
from ns10 import load_dump
import features, gf2
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); gate = ~((a == 0) & (bq == 1))
spec = sys.argv[2]   # e.g. "m2,m3,p4,p5"
only_gate = len(sys.argv) > 3 and sys.argv[3] == "gate"
sel = np.arange(60, N - 3)
if only_gate: sel = sel[gate[sel]]
sel = sel[::max(1, len(sel) // 30000)][:30000]
lin = [np.ones((len(sel), 1), np.uint8)]
for w in range(0, 7): lin += [features.bits16(c[sel - w]), features.bits16(p[sel - w])]
lin += [t[sel - k][:, None].astype(np.uint8) for k in range(1, 9)]
Lm = np.concatenate(lin, 1); r, piv = gf2.rank_and_pivots(Lm[:12000]); Lp = Lm[:, piv]
qc = []
for item in spec.split(","):
    arr = {"m": m, "p": p, "c": c}[item[0]]; w = int(item[1:]); qc.append(features.bits16(arr[sel - w]))
Q = np.concatenate(qc, 1); n = Q.shape[1]
prods = [(Q[:, x] & Q[:, y])[:, None] for x, y in itertools.combinations(range(n), 2)]
X = np.concatenate([Lp] + prods, 1); y = t[sel]
t0 = time.time(); res = gf2.solve(X, y, "qm")
print(f"quadratic over {spec}{' (gate-open rows)' if only_gate else ''}: {n} bits, {X.shape[1]} cols: rank {res['rank']}, inconsistent {res['bad']} of {len(sel) - res['rank']} ({time.time()-t0:.0f}s)", flush=True)
