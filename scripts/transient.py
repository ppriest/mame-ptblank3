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
# steady-state affine hull V of the 3-mask state in deep zero regions: find annihilator (linear relations incl. const)
deep = np.where(before >= 60)[0]
S3 = lambda idx: np.concatenate([np.ones((len(idx),1),np.uint8), features.bits16(m[idx]), features.bits16(m[idx-1]), features.bits16(m[idx-2])], 1)
Xd = S3(deep[::7][:40000])
# relations: vectors v with Xd v = 0 -> nullspace. Compute via solving each non-pivot column in terms of pivots.
rk, piv = gf2.rank_and_pivots(Xd); print("deep-zero 3-mask state (+const) rank", rk, "of 49 -> annihilator dim", 49 - rk)
nonpiv = [k for k in range(49) if k not in piv]
rels = []
for k in nonpiv:
    res = gf2.solve(Xd[:, piv], Xd[:, k], "rel"); assert res['bad'] == 0
    v = np.zeros(49, np.uint8); v[k] = 1
    for q in res['solution']: v[piv[q]] ^= 1
    rels.append(v)
R = np.array(rels, np.uint8)  # (dim, 49) each row is a relation vanishing on V
def inV(idx):
    return ((S3(idx) @ R.T.astype(np.int64)) % 2 == 0).all(1)
starts = np.where(z & ~np.roll(z, 1))[0]; starts = starts[(starts > 100) & (starts < N - 100)]
longs = starts[after[starts] >= 48]
print("run: entry offset into V (first offset from which state stays in V for 8 words) vs last NL event offset")
agree = 0
for s in longs:
    offs = np.arange(0, 48); iv = inV(s + offs)
    entry = next((k for k in range(0, 40) if iv[k:k+8].all()), None)
    rr = r[s:s+48]; w = np.where(rr == 1)[0]; last = int(w.max()) if len(w) else -1
    print(f"  run {s}: enterV at {entry}, last NL event at {last}, V membership: {''.join('1' if v else '0' for v in iv[:40])}")
    if entry is not None and last < entry: agree += 1
print("runs where last event precedes V entry:", agree, "of", len(longs))
