import sys, numpy as np, itertools, time
from ns10 import load_dump
import gf2
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
LAGS = 8
idx = np.arange(200, N - 3); ok = G[idx].copy()
for k in range(1, LAGS + 1): ok &= G[idx - k]
rows = idx[ok]
cols, names = [], []
for k in range(0, 12): cols.append(((m[rows - k] >> 0) & 1).astype(np.uint8)); names.append(f"m{k}.0")
for k in range(2, 11):
    for bb in (11, 14, 15): cols.append(((p[rows - k] >> bb) & 1).astype(np.uint8)); names.append(f"p{k}.{bb}")
for k in range(1, LAGS + 1): cols.append(e[rows - k].astype(np.uint8)); names.append(f"h{k}")
V = np.stack(cols, 1); y = e[rows]; n = V.shape[1]
print("rows", len(rows), "vars", n)
def fit(keep):
    W = V[:, keep]; out = [np.ones(len(rows), np.uint8)] + [W[:, k] for k in range(W.shape[1])]
    for comb in itertools.combinations(range(W.shape[1]), 2): out.append(W[:, comb[0]] & W[:, comb[1]])
    for comb in itertools.combinations(range(W.shape[1]), 3): out.append(W[:, comb[0]] & W[:, comb[1]] & W[:, comb[2]])
    X = np.stack(out, 1); res = gf2.solve(X, y, "d1"); return res['bad'], len(rows) - res['rank']
full = list(range(n)); bad, free = fit(full); print(f"all vars: inconsistent {bad} of {free}", flush=True)
for k in range(n):
    keep = [q for q in full if q != k]; bad, free = fit(keep)
    print(f"  drop {names[k]:7s}: inconsistent {bad:5d} of {free}", flush=True)
