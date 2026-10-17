import sys, numpy as np
from ns10 import load_dump
import features, gf2
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
W = 4
S = np.concatenate([np.concatenate([features.bits16(c[np.maximum(np.arange(N) - w, 0)]), features.bits16(p[np.maximum(np.arange(N) - w, 0)])], 1) for w in range(W)], 1).astype(np.uint8)  # (N, 128)
CS = np.bitwise_xor.accumulate(S, axis=0)   # prefix parity, so gap-sum over (j, i] = CS[i] ^ CS[j]
names = [f"{nm}{w}.{bb}" for w in range(W) for nm in ("c", "p") for bb in range(16)]
def chain_test(sel, label, nfit=30000, maxgap=200):
    for v in (0, 1):
        obs = np.where(sel == v)[0]; obs = obs[obs > 40]
        i = obs[1:]; j = obs[:-1]; ok = (i - j) <= maxgap; i = i[ok]; j = j[ok]
        X = np.concatenate([np.ones((len(i), 1), np.uint8), S[i], S[j], CS[i] ^ CS[j], CS[i - 1] ^ CS[j - 1]], 1)
        y = (t[i] ^ t[j]).astype(np.uint8)
        res = gf2.solve(X[:nfit], y[:nfit], "br")
        pred = np.zeros(len(i), np.uint8)
        for k in res['solution']: pred ^= X[:, k]
        mis = int((pred != y).sum())
        print(f"[{label} = {v}] pairs {len(i)} (mean gap {np.mean(i-j):.2f}); rank {res['rank']}; inconsistent {res['bad']} of {nfit - res['rank']}; mismatches all {mis}", flush=True)
        if res['bad'] == 0:
            nm = ['1'] + [f"i:{q}" for q in names] + [f"j:{q}" for q in names] + [f"gap(j,i]:{q}" for q in names] + [f"gap[j,i):{q}" for q in names]
            print("    solution:", [nm[k] for k in res['solution']])
a = ((m[np.maximum(np.arange(N) - 2, 0)] >> 0) & 1).astype(np.uint8)
bq = ((m[np.maximum(np.arange(N) - 3, 0)] >> 0) & 1).astype(np.uint8)
chain_test(a, "a = m[i-2].0")
chain_test(bq, "b = m[i-3].0")
# also: consecutive observations within the same (a,b) cell
cell = a * 2 + bq
for cv in range(4):
    chain_test((cell == cv).astype(np.uint8), f"cell=={cv} (a={cv>>1},b={cv&1})")
