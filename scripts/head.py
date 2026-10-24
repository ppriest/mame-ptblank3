import sys, numpy as np
from ns10 import load_dump
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
# recompute residual from scratch for the first words (L4_events was cut at index 41)
def par(x, mask): return bin(int(x) & mask).count('1') & 1
C0 = sum(1 << k for k in (0,1,3,4,7,12,15)); P0 = sum(1 << k for k in (1,2,3,6,8,12,15)); C1 = sum(1 << k for k in (4,8,9,10,12,13,15)); P1 = sum(1 << k for k in (0,2,4,8,10,11))
C2 = sum(1 << k for k in (0,4,5,6,8,9,10,11,14)); P2 = sum(1 << k for k in (0,2)); C3 = sum(1 << k for k in (0,1,2,6,8,11,13,15)); P3 = sum(1 << k for k in (0,2))
def L4(k):
    g = lambda arr, j: arr[j] if j >= 0 else 0
    return par(g(c,k),C0)^par(g(p,k),P0)^par(g(c,k-1),C1)^par(g(p,k-1),P1)^par(g(c,k-2),C2)^par(g(p,k-2),P2)^par(g(c,k-3),C3)^par(g(p,k-3),P3)
E = np.array([t[k] ^ L4(k) for k in range(400)])
print(" idx    c     p     m   t L4 e gate  (e must be 0 when gate closed)")
for k in range(0, 120):
    print(f"{k:4d}  {c[k]:04x}  {p[k]:04x}  {m[k]:04x}  {t[k]} {L4(k)} {E[k]} {'open' if G[k] else 'CLOSED'}{'  <-- p&0xc980' if p[k] & 0xc980 else ''}")
print("first index with residual 1:", int(np.argmax(E == 1)), "; residual ones in first 400:", int(E.sum()), "; violations (e=1 with gate closed) in first 400:", int(((E == 1) & ~G[:400]).sum()))
