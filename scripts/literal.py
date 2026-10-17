import sys, numpy as np, time
from ns10 import load_dump
import gf2, features
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
W = 5; NLAG = 8; NSUB = 7000
sel = np.arange(16, N - 1)
cols, names = [], []
for k in range(W):
    cols.append(features.bits16(c[sel - k])); names += [f"c{k}.{bb}" for bb in range(16)]
    cols.append(features.bits16(p[sel - k])); names += [f"p{k}.{bb}" for bb in range(16)]
for k in range(1, NLAG + 1):
    cols.append(t[sel - k][:, None].astype(np.uint8)); names.append(f"t-{k}")
X = np.concatenate([np.ones((len(sel), 1), np.uint8)] + cols, 1); names = ['1'] + names
y = t[sel]
r, piv = gf2.rank_and_pivots(X[:30000]); print("linear feature rank", r, "of", X.shape[1]); Xp = X[:, piv]
# candidate literals
cands = {}
for k in range(W):
    for bb in range(16):
        cands[f"c{k}.{bb}"] = features.bits16(c[sel - k])[:, bb]
        cands[f"p{k}.{bb}"] = features.bits16(p[sel - k])[:, bb]
        cands[f"m{k}.{bb}"] = features.bits16(m[sel - k])[:, bb]
for bb in range(16): cands[f"cnext.{bb}"] = features.bits16(c[sel + 1])[:, bb]
for k in range(1, NLAG + 1): cands[f"t-{k}"] = t[sel - k]
results = []; t0 = time.time()
for name, col in cands.items():
    for pol in (0, 1):
        rows = np.where(col == pol)[0][:NSUB]
        if len(rows) < 2000: continue
        res = gf2.solve(Xp[rows], y[rows], "lit")
        free = len(rows) - res['rank']
        results.append((res['bad'] / max(free, 1), name, pol, res['bad'], free))
        if res['bad'] == 0: print(f"  LINEAR on subset {name}=={pol}: rows {len(rows)} rank {res['rank']} free {free}", flush=True)
results.sort()
print(f"done {len(results)} tests in {time.time()-t0:.0f}s; best 15:")
for q in results[:15]: print("  ", q)
