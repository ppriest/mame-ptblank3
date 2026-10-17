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
after = np.zeros(N, np.int64)
for i in range(N - 2, -1, -1): after[i] = (after[i+1] + 1) if z[i] else 0
print("global residual r rate:", r.mean(), " in deep zero (before>=40):", r[before >= 40].mean())
# residual vs offset from zero-run start, for runs >= 80 long
starts = np.where(z & ~np.roll(z, 1))[0]; starts = starts[(starts > 100) & (starts < N - 100)]
runlen = after[starts]
longs = starts[runlen >= 80]; print("zero runs >= 80 words:", len(longs))
prof = np.array([[r[s + k] for k in range(-4, 72)] for s in longs])
print("P(r=1) by offset from run start (-4..71):")
print(' '.join(f"{v:.2f}" for v in prof.mean(0)))
print("sample runs (offset -4..71), '|' marks run start:")
for s in longs[:12]:
    print("  ", ''.join(str(r[s+k]) for k in range(-4, 0)) + '|' + ''.join(str(r[s+k]) for k in range(0, 72)), "  last nonzero p:", f"{p[s-1]:04x}", f"{p[s-2]:04x}")
# true-linear hypothesis in zero regions: m_{i+1}[11] ~ linear(c_i, c_{i-1}, c_{i-2}) + const
idx = np.where((before >= 40) & (np.arange(N) < N - 1))[0]
X = np.concatenate([np.ones((len(idx),1),np.uint8), features.bits16(c[idx]), features.bits16(c[idx-1]), features.bits16(c[idx-2])], 1)
y = ((m[idx + 1] >> 11) & 1).astype(np.uint8)
res = gf2.solve(X[:40000], y[:40000], "tl"); pred = np.zeros(len(idx), np.uint8)
for k in res['solution']: pred ^= X[:, k]
print(f"ZERO: true mask bit 11 ~ linear(c_i,c_i-1,c_i-2): rank {res['rank']} inconsistent {res['bad']} mismatches all {int((pred!=y).sum())} of {len(idx)}")
X4 = np.concatenate([X, features.bits16(c[idx-3]), features.bits16(c[idx-4])], 1)
res = gf2.solve(X4[:40000], y[:40000], "tl4"); pred = np.zeros(len(idx), np.uint8)
for k in res['solution']: pred ^= X4[:, k]
print(f"ZERO: true mask bit 11 ~ linear(5 cipher words): rank {res['rank']} inconsistent {res['bad']} mismatches all {int((pred!=y).sum())}")
names = ['1'] + [f"c{k}.{bb}" for k in range(5) for bb in range(16)]
if res['bad'] == 0: print("   solution:", [names[k] for k in res['solution']])
