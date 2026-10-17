import sys, numpy as np
from ns10 import load_dump
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); clear = (a == 0) & (bq == 1)
idx = np.arange(45, N - 3); ee = e[idx]; ncl = ~clear[idx]
print("P(e | not clear) =", ee[ncl].mean())
for k in range(1, 6): print(f"  P(e_i | clear at i-{k}, not clear at i) = {ee[ncl & clear[idx - k]].mean():.4f}   P(e_i | no clear at i-{k}, not clear at i) = {ee[ncl & ~clear[idx - k]].mean():.4f}")
# distance since last clear
last = np.full(N, -1, np.int64); lc = -1
for k in range(N):
    if clear[k]: lc = k
    last[k] = lc
dist = idx - last[idx]
print("P(e | distance since last clear = k):", [round(float(ee[ncl & (dist == k)].mean()), 3) for k in range(1, 13)])
# P(e | a, b) on non-clear rows and jointly with e-1
for ca, cb in ((0, 0), (1, 0), (1, 1)):
    s = (a[idx] == ca) & (bq[idx] == cb)
    print(f"  cell ({ca},{cb}): P(e)={ee[s].mean():.3f}  P(e|e-1=1)={ee[s & (e[idx-1]==1)].mean():.3f}  P(e|e-1=0,e-2=0,e-3=0)={ee[s & (e[idx-1]==0)&(e[idx-2]==0)&(e[idx-3]==0)].mean():.3f}")
# does an event at i-1 that was in a particular cell matter? P(e_i | e_{i-1}=1, cell_{i-1})
for ca, cb in ((0, 0), (1, 0), (1, 1)):
    s = ncl & (e[idx-1] == 1) & (a[idx-1] == ca) & (bq[idx-1] == cb)
    print(f"  P(e_i | e_i-1=1 in cell ({ca},{cb}), i not clear) = {ee[s].mean():.3f} (n={s.sum()})")
