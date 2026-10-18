import sys, numpy as np, itertools
from ns10 import load_dump
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
idx = np.arange(80, N - 3); idx = idx[G[idx]]; ee = e[idx]
print("gate-open rows", len(idx), "events", int(ee.sum()))
res = []
bitsets = []
for r in range(1, 6):
    for comb in itertools.combinations((7, 8, 11, 14, 15), r): bitsets.append(sum(1 << q for q in comb))
bitsets += [0xff00, 0x8000, 0x0080, 0x0800, 0x8880, 0x8980, 0xc980, 0x0980, 0x8900, 0x0180, 0x0880]
for M in sorted(set(bitsets)):
    for lo in (4, 3, 5):
        for hi in (4, 5, 6, 7, 8, 9, 10, 12, 16, 20):
            if hi < lo: continue
            D = np.zeros(len(idx), bool)
            for w in range(lo, hi + 1): D |= (p[idx - w] & M) != 0
            n0 = (~D).sum(); e0 = int(ee[~D].sum()); n1 = D.sum(); e1 = int(ee[D].sum())
            res.append((e0 / max(n0, 1), n0 / len(idx), e1 / max(n1, 1), M, lo, hi, e0))
res.sort(key=lambda r: (r[0], -r[1]))
print("best masks/ranges by leakage P(e|D=0)  [leak, frac rows D=0, P(e|D=1), mask, lo, hi, events with D=0]:")
for r in res[:25]: print(f"   {r[0]:.4f}  {r[1]:.3f}  {r[2]:.3f}  0x{r[3]:04x}  {r[4]}..{r[5]}  {r[6]}")
