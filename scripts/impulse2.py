import sys, numpy as np
from ns10 import load_dump
import features
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); gate = ~((a == 0) & (bq == 1))
z = (p == 0); before = np.zeros(N, np.int64)
for k in range(1, N): before[k] = before[k-1] + 1 if z[k] else 0
# impulse: a nonzero word at position s with >= 12 zeros before it; e at s+d for d=1..25 while p[s+1..s+d] all zero
S = np.where((~z) & (np.roll(before, 1) >= 12))[0]; S = S[(S > 100) & (S < N - 40)]
print("nonzero words preceded by >= 12 zeros:", len(S))
print(" d : P(e_{s+d} | p[s+1..s+d]=0, gate open)   n")
for dd in range(0, 26):
    rows = S + dd; ok = gate[rows] & (before[rows] >= dd) if dd > 0 else gate[rows]
    if dd > 0: ok &= (before[rows] == dd)   # exactly d zeros after s (p[s] nonzero)
    print(f" {dd:2d}: {e[rows][ok].mean():.4f}   {ok.sum()}")
# which bits of the impulse word matter at d=4,5,6? P(e | bit of p[s] set)
for dd in (4, 5, 6):
    rows = S + dd; ok = gate[rows] & (before[rows] == dd); X = p[S][ok]; y = e[rows][ok]
    r1 = [float(y[((X >> bb) & 1) == 1].mean()) if (((X >> bb) & 1) == 1).sum() > 30 else -1 for bb in range(16)]
    r0 = [float(y[((X >> bb) & 1) == 0].mean()) if (((X >> bb) & 1) == 0).sum() > 30 else -1 for bb in range(16)]
    print(f" d={dd}: rate by bit of impulse word set:", ' '.join(f"{v:.2f}" for v in r1)); print(f"       rate by bit clear:", ' '.join(f"{v:.2f}" for v in r0))
