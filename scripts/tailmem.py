import sys, numpy as np
from ns10 import load_dump
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); gate = ~((a == 0) & (bq == 1))
z = (p == 0); before = np.zeros(N, np.int64)
for k in range(1, N): before[k] = before[k-1] + 1 if z[k] else 0
for lo, hi in ((4, 25), (6, 25), (10, 30), (0, 3)):
    sel = np.where((before >= lo) & (before <= hi) & gate & (i > 60))[0]
    print(f"tail rows with {lo} <= zeros-before <= {hi}, gate open: {len(sel)}, event rate {e[sel].mean():.4f}")
    for w in (4, 6, 8, 10, 12, 16, 20, 24, 32):
        key = np.zeros(len(sel), np.uint64)
        for k in range(w):
            key = key * np.uint64(0x9E3779B97F4A7C15) + c[sel - k].astype(np.uint64) + (p[sel - k].astype(np.uint64) << np.uint64(16)) + np.uint64(k)
        order = np.argsort(key, kind='stable'); ks = key[order]; es = e[sel][order]
        same = ks[1:] == ks[:-1]; conf = same & (es[1:] != es[:-1])
        print(f"   w={w:2d}: identical pairs {int(same.sum()):6d}, conflicts {int(conf.sum()):5d} ({conf.sum()/max(same.sum(),1):.3f})")
