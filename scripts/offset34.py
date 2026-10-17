import sys, numpy as np
from ns10 import load_dump
import gf2, features
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
def par(x, mask): return (np.bitwise_count((x & mask).astype(np.uint16)) & 1).astype(np.uint8)
LC0 = (1<<0)|(1<<2)|(1<<3)|(1<<7)|(1<<8)|(1<<9)|(1<<10); LC1 = (1<<0)|(1<<2)
r = t.copy(); r[2:] ^= par(c[2:], LC0) ^ par(c[1:-1], LC1) ^ t[:-2]
z = (p == 0); before = np.zeros(N, np.int64)
for i in range(1, N): before[i] = before[i-1] + 1 if z[i] else 0
for off in (6, 5, 4, 3, 2):
    # samples at offset 'off' into a zero run: before == off+1 (p_i..p_{i-off} zero, p_{i-off-1} nonzero); require the preceding run of nonzero... no constraint
    idx = np.where((before == off + 1) & (np.arange(N) > 40) & (np.arange(N) < N - 1))[0]
    X = np.concatenate([np.ones((len(idx),1),np.uint8), features.bits16(p[idx - off - 1]), features.bits16(p[idx - off - 2]), features.bits16(p[idx - off - 3]),
                        features.bits16(c[idx]), features.bits16(c[idx-1]), features.bits16(c[idx-2]), features.bits16(c[idx-3]), features.bits16(c[idx-4]), features.bits16(c[idx-5]),
                        np.stack([t[idx-k] for k in range(1,9)],1).astype(np.uint8)], 1)
    names = ['1'] + [f"p-{off+1}.{k}" for k in range(16)] + [f"p-{off+2}.{k}" for k in range(16)] + [f"p-{off+3}.{k}" for k in range(16)] + [f"c{j}.{k}" for j in range(6) for k in range(16)] + [f"t-{k}" for k in range(1,9)]
    y = r[idx]; rk, piv = gf2.rank_and_pivots(X)
    res = gf2.solve(X, y, "o34")
    print(f"offset {off}: samples {len(idx)}, P(r=1)={y.mean():.3f}, rank {rk}, linear inconsistent {res['bad']} of {len(idx)-res['rank']}")
    if res['bad'] == 0: print("   solution:", [names[k] for k in res['solution']])
    else:
        # quadratic over pivots if enough samples
        Q, qn = gf2.quad_features(X, piv)
        if Q.shape[1] + 500 < len(idx):
            res2 = gf2.solve(Q, y, "o34q"); print(f"   quad {Q.shape[1]} monomials: inconsistent {res2['bad']} of {len(idx)-res2['rank']}")
