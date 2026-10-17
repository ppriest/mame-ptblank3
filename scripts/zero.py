import sys, numpy as np
from ns10 import load_dump
import gf2, features
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
# A. depth-3 collision pairs: which bits of word i-3 differ?
def key3():
    key = np.zeros(N, np.uint64)
    for k in range(3):
        idx = np.maximum(np.arange(N) - k, 0)
        key = key * np.uint64(0x9E3779B97F4A7C15) + c[idx].astype(np.uint64) + (p[idx].astype(np.uint64) << np.uint64(16)) + np.uint64(k)
    return key
key = key3(); sel = np.arange(8, N - 1)
order = sel[np.argsort(key[sel], kind='stable')]; ks = key[order]; same = ks[1:] == ks[:-1]
i = order[:-1][same]; j = order[1:][same]; conf = t[i] != t[j]
d3 = (c[i-3] != c[j-3]) | (p[i-3] != p[j-3])
print("depth-3 pairs (4th word differs):", d3.sum(), "conflicts", (conf & d3).sum())
dc = (c[i-3] ^ c[j-3])[d3]; dp = (p[i-3] ^ p[j-3])[d3]; cf = conf[d3]
print(" bits of c[i-3] differing (count over pairs):", [int(((dc >> k) & 1).sum()) for k in range(16)])
print(" bits of p[i-3] differing:", [int(((dp >> k) & 1).sum()) for k in range(16)])
print(" t[i-1] differs in those pairs:", int((t[i-3+2][d3] != t[j-3+2][d3]).sum()), "; t[i-2] differs:", int((t[i-2][d3] != t[j-2][d3]).sum()), "; t[i-3] differs:", int((t[i-3][d3] != t[j-3][d3]).sum()))
print(" non-conflicting depth-3 pairs:", int((~cf).sum()), " their c diffs:", [hex(v) for v in dc[~cf][:10]])
# B. longest zero-plaintext runs
z = (p == 0).astype(np.int8); dz = np.diff(np.concatenate([[0], z, [0]]))
rs, re_ = np.where(dz == 1)[0], np.where(dz == -1)[0]; L = re_ - rs; o = np.argsort(-L)
print("zero runs: count", len(L), "longest", [(int(rs[k]), int(L[k])) for k in o[:8]])
s0 = rs[o[0]]
print("longest run start", s0, "c/t for 40 words:")
for k in range(s0 - 4, s0 + 40): print(f"  {k:7d} c={c[k]:04x} p={p[k]:04x} m={m[k]:04x} t={t[k]}")
# period detection inside the longest run
run_c = c[s0:s0 + L[o[0]]]
for per in range(1, 2000):
    if np.array_equal(run_c[per:4*per+per], run_c[:4*per]): print("cipher period in zero run:", per); break
else: print("no cipher period < 2000 found in zero run")
# C. zero-region fit: samples with 6 zero plain words
zz = np.ones(N, bool)
for k in range(6): zz &= np.roll(p == 0, k)
zz[:8] = False; zz[-1] = False
idx = np.where(zz)[0]; print("zero-region samples:", len(idx))
cols = np.concatenate([features.bits16(c[idx - k]) for k in range(6)], 1); names = [f"c{k}.{bb}" for k in range(6) for bb in range(16)]
y = t[idx]
r, piv = gf2.rank_and_pivots(cols); print(" zero-region cipher-history rank (6 words):", r)
res = gf2.solve(cols, y, "zl"); print(" linear fit over 6 cipher words: inconsistent", res['bad'])
Q, qn = gf2.quad_features(cols, piv); res = gf2.solve(Q, y, "zq"); print(" quadratic fit: monomials", Q.shape[1], "rank", res['rank'], "inconsistent", res['bad'], "of", len(y) - res['rank'])
