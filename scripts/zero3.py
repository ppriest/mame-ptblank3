import sys, numpy as np
from ns10 import load_dump
import gf2, features
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c)
z = (p == 0)
# run-length of zeros ending at i (inclusive) and starting at i (how many zero words from i onward)
before = np.zeros(N, np.int64); after = np.zeros(N, np.int64)
for i in range(N):
    before[i] = before[i-1] + 1 if (z[i] and i > 0) else int(z[i])
for i in range(N - 1, -1, -1):
    after[i] = (after[i+1] + 1 if i + 1 < N else 1) if z[i] else 0
def linfit(W, need):
    idx = np.where((before >= need) & (np.arange(N) >= 32) & (np.arange(N) < N - 1))[0]
    cols = np.concatenate([features.bits16(c[idx - k]) for k in range(W)], 1)
    y = t[idx]
    r, piv = gf2.rank_and_pivots(cols[:30000])
    res = gf2.solve(cols[:30000], y[:30000], "zl")
    pred = np.zeros(len(idx), np.uint8)
    for k in res['solution']: pred ^= cols[:, k]
    mis = pred != y
    print(f"cipher window {W:2d}, >= {need:2d} zero words: samples {len(idx)}, rank {r}, inconsistent(30k) {res['bad']}, mismatches all {int(mis.sum())}", flush=True)
    return idx, mis
for W, need in ((8, 10), (10, 12), (12, 14), (14, 16), (16, 18), (20, 22), (24, 26)):
    idx, mis = linfit(W, need)
# where are the mismatches of the 10-word fit? distance to end of zero run (words of zero plaintext after i)
idx, mis = linfit(10, 12)
a = after[idx]; print("zeros remaining after i (incl. i): all samples quantiles", np.percentile(a, [5, 25, 50, 75]), " mismatches:", np.percentile(a[mis], [5, 25, 50, 75, 95]))
print("mismatch count by zeros-after (1..8):", [int(((a == k) & mis).sum()) for k in range(1, 9)], " of total samples with that count:", [int((a == k).sum()) for k in range(1, 9)])
bf = before[idx]; print("mismatch count by zeros-before (12..20):", [int(((bf == k) & mis).sum()) for k in range(12, 21)], "of", [int((bf == k).sum()) for k in range(12, 21)])
