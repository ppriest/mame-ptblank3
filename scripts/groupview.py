import sys, numpy as np
from ns10 import load_dump
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
idx = np.arange(80, N - 3); idx = idx[G[idx]]
lo, hi = (int(x) for x in sys.argv[2].split(",")) if len(sys.argv) > 2 else (0, 11)
MIN = int(sys.argv[3]) if len(sys.argv) > 3 else 30
key = np.zeros(len(idx), np.uint64)
for w in range(lo, hi + 1): key = key * np.uint64(0x9E3779B97F4A7C15) + p[idx - w].astype(np.uint64) + np.uint64(w)
order = np.argsort(key, kind='stable'); ks = key[order]; rows = idx[order]
bounds = np.concatenate([[0], np.where(ks[1:] != ks[:-1])[0] + 1, [len(ks)]]); sizes = np.diff(bounds)
big = np.where(sizes >= MIN)[0]
recs = []
for g in big:
    s, en = bounds[g], bounds[g + 1]; r0 = rows[s]; win = [int(p[r0 - w]) for w in range(lo, hi + 1)]
    if sum(1 for v in win if v != 0) < 4: continue   # skip near-zero windows
    rate = e[rows[s:en]].mean(); spread = int(rows[en - 1] - rows[s]); npos = len(np.unique(rows[s:en] // 256))
    recs.append((rate, en - s, spread, npos, win))
recs.sort(key=lambda r: r[0])
print(f"window p[i-{lo}..i-{hi}] (newest first), groups >= {MIN} with >= 4 nonzero words: {len(recs)}")
print("rate  size  pages  words i-%d .. i-%d" % (lo, hi))
for r in recs[:22] + [None] + recs[-22:]:
    if r is None: print("   ..."); continue
    print(f"{r[0]:.2f}  {r[1]:4d}  {r[3]:4d}   " + ' '.join(f"{v:04x}" for v in r[4]))
