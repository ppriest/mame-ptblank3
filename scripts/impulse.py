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
# isolated nonzero word: p[i] != 0, >= 40 zeros before, >= 40 zeros after
iso = np.where((~z) & (np.roll(before, 1) >= 40) & (np.roll(after, -1) >= 40))[0]
print("isolated nonzero words inside zero regions:", len(iso))
prof = np.array([[r[i + k] for k in range(-2, 40)] for i in iso])
print("P(r=1) by offset from isolated word (-2..39):"); print(' '.join(f"{v:.2f}" for v in prof.mean(0)))
print("examples: X, then r for offsets -2..24")
for i in iso[:16]: print(f"  X={p[i]:04x} c={c[i]:04x}  r: " + ''.join(str(r[i+k]) for k in range(-2, 0)) + '|' + ''.join(str(r[i+k]) for k in range(0, 25)))
# bit-correlation: for offsets 0..6, P(r=1 | bit b of X set) vs not
for k in range(0, 7):
    rr = r[iso + k]
    pe = [float(rr[((p[iso] >> bb) & 1) == 1].mean()) if (((p[iso] >> bb) & 1) == 1).any() else -1 for bb in range(16)]
    print(f" offset {k}: P(r=1)={rr.mean():.2f}; by X bit set: " + ' '.join(f"{v:.2f}" for v in pe))
# number of distinct X values and the residual pattern per X
vals, inv = np.unique(p[iso], return_inverse=True); print("distinct X values:", len(vals))
