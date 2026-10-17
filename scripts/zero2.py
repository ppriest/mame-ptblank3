import sys, numpy as np, itertools, time
from ns10 import load_dump
import gf2, features
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c)
W = int(sys.argv[2]) if len(sys.argv) > 2 else 8
zz = np.ones(N, bool)
for k in range(W + 2): zz &= np.roll(p == 0, k)
zz[:16] = False; zz[-1] = False
idx = np.where(zz)[0]; print(f"samples with {W+2} zero plain words: {len(idx)}")
for w in (3, 4, 6, 8, 10):
    if w > W: break
    key = np.zeros(len(idx), np.uint64)
    for k in range(w): key = key * np.uint64(0x9E3779B97F4A7C15) + c[idx - k].astype(np.uint64)
    print(f"  distinct {w}-word cipher states: {len(np.unique(key))}")
cols = np.concatenate([features.bits16(c[idx - k]) for k in range(W)], 1); names = [f"c{k}.{bb}" for k in range(W) for bb in range(16)]
y = t[idx]
r, piv = gf2.rank_and_pivots(cols); print(f"rank of {W}-word cipher history: {r}; pivots: {[names[q] for q in piv]}")
P = cols[:, piv]; n = len(piv)
def monomials(P, deg):
    N_, n_ = P.shape
    cols_ = [np.ones(N_, np.uint8)]; nm = [()]
    for dg in range(1, deg + 1):
        for comb in itertools.combinations(range(n_), dg):
            v = P[:, comb[0]].copy()
            for q in comb[1:]: v &= P[:, q]
            cols_.append(v); nm.append(comb)
    return np.stack(cols_, 1), nm
for deg in (1, 2, 3, 4):
    nmono = sum(len(list(itertools.combinations(range(n), dg))) for dg in range(deg + 1))
    if nmono > 80000: print(f"degree {deg}: {nmono} monomials, skipped"); break
    sub = np.arange(0, len(idx), max(1, len(idx) // (nmono + 4000)))[:nmono + 4000]
    t0 = time.time(); Q, qn = monomials(P[sub], deg); res = gf2.solve(Q, y[sub], "zdeg")
    print(f"degree {deg}: {Q.shape[1]} monomials, {len(sub)} rows, rank {res['rank']}, inconsistent {res['bad']} of {len(sub) - res['rank']}  ({time.time()-t0:.1f}s)", flush=True)
    if res['bad'] == 0:
        # verify on all zero samples
        pred = np.zeros(len(idx), np.uint8)
        for k in res['solution']:
            mono = qn[k]; v = np.ones(len(idx), np.uint8)
            for q in mono: v &= P[:, q]
            pred ^= v
        print(f"   verified on all {len(idx)} zero-region samples: mismatches {int((pred != y).sum())}; terms {len(res['solution'])}")
        print("   terms:", [tuple(names[piv[q]] for q in qn[k]) for k in res['solution']][:60])
        break
