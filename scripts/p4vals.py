import sys, numpy as np
from ns10 import load_dump
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
idx = np.arange(80, N - 3); idx = idx[G[idx]]; ee = e[idx]
for w in (4, 5):
    v = p[idx - w]; vals, cnts = np.unique(v, return_counts=True); order = np.argsort(-cnts)[:40]
    print(f"p[i-{w}]: most common values -> event rate (count):")
    print("   " + '  '.join(f"{vals[q]:04x}:{ee[v == vals[q]].mean():.2f}({cnts[q]})" for q in order))
    # rate by bit of p[i-w] over all gate-open rows
    print(f"   rate by bit=1/bit=0: " + ' '.join(f"{j}:{ee[(v>>j)&1==1].mean():.2f}/{ee[(v>>j)&1==0].mean():.2f}" for j in range(16)))
