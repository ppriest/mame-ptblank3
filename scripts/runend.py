import sys, numpy as np
from ns10 import load_dump
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); gate = ~((a == 0) & (bq == 1))
z = (p == 0); before = np.zeros(N, np.int64)
for k in range(1, N): before[k] = before[k-1] + 1 if z[k] else 0
# s = first nonzero word after >= 30 zeros
S = np.where((~z) & (np.roll(before, 1) >= 30))[0]; S = S[(S > 100) & (S < N - 20)]
print("zero-run ends (>=30 zeros then nonzero):", len(S))
print("offset k relative to first nonzero word s: P(e_{s+k}) [gate-open rows only], n")
for k in range(-4, 10):
    rows = S + k; g = gate[rows]
    print(f"  k={k:+d}: P(e)={e[rows][g].mean():.4f} (n={g.sum()}),  nonzero-plain-count among p[s+k..]: p[s+k]!=0 frac {float((p[rows]!=0).mean()):.2f}")
# same at zero-run starts for comparison (first zero word s0 after nonzero), gate-open rows
S0 = np.where(z & ~np.roll(z, 1))[0]; S0 = S0[(S0 > 100) & (S0 < N - 100)]
after = np.zeros(N, np.int64)
for k in range(N - 2, -1, -1): after[k] = (after[k+1] + 1) if z[k] else 0
S0 = S0[after[S0] >= 30]
print("zero-run starts (>=30 zeros follow):", len(S0), " P(e_{s0+k}) gate-open:")
print("  " + ' '.join(f"{k:+d}:{e[S0+k][gate[S0+k]].mean():.3f}" for k in range(-3, 14)))
