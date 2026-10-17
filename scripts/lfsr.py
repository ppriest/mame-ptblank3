import sys, numpy as np
from ns10 import load_dump
import features, gf2
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); gate = ~((a == 0) & (bq == 1))
for r in (4, 8, 12, 16):
    idx = np.arange(80, N - 3); ok = gate[idx].copy()
    for k in range(1, r + 1): ok &= gate[idx - k]
    rows = idx[ok]
    cl = [np.ones((len(rows), 1), np.uint8)] + [e[rows - k][:, None].astype(np.uint8) for k in range(1, r + 1)]
    for w in range(0, r + 3): cl += [features.bits16(c[rows - w]), features.bits16(p[rows - w])]
    X = np.concatenate(cl, 1); y = e[rows]
    res = gf2.solve(X[:30000], y[:30000], "lf")
    print(f"order {r}: rows {len(rows)}, cols {X.shape[1]}, rank {res['rank']}, inconsistent {res['bad']} of {30000 - res['rank']}")
