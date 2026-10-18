import sys, numpy as np
from ns10 import load_dump
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
idx = np.arange(200, N - 3); idx = idx[G[idx]]
def feats(ix, mk=range(0, 10), pk=range(3, 10), ek=range(1, 9)):
    cols, names = [], []
    for k in mk: cols.append(((m[ix - k] >> 0) & 1).astype(np.uint8)[:, None]); names.append(f"m{k}.0")
    for k in pk:
        for bb in (11, 14, 15): cols.append(((p[ix - k] >> bb) & 1).astype(np.uint8)[:, None]); names.append(f"p{k}.{bb}")
    for k in ek:
        cols.append((G[ix - k] & (e[ix - k] == 1)).astype(np.uint8)[:, None]); names.append(f"e{k}=1")
        cols.append((G[ix - k] & (e[ix - k] == 0)).astype(np.uint8)[:, None]); names.append(f"e{k}=0")
    return np.concatenate(cols, 1), names
X, names = feats(idx); y = e[idx].astype(np.float64); print("rows", len(idx), "rate", y.mean(), "features", X.shape[1])
def H(pr):
    pr = np.clip(pr, 1e-9, 1 - 1e-9); return -(pr * np.log2(pr) + (1 - pr) * np.log2(1 - pr))
leaves = []
def grow(rows, depth, path):
    yy = y[rows]; n = len(rows); rate = yy.mean()
    if depth == 0 or n < 200 or rate < 0.01 or rate > 0.99: leaves.append((n, rate, path)); return
    Xr = X[rows].astype(np.float32); s1 = Xr.sum(0); e1 = (Xr * yy[:, None].astype(np.float32)).sum(0)
    p1 = e1 / np.maximum(s1, 1); p0 = (yy.sum() - e1) / np.maximum(n - s1, 1)
    gain = H(rate) - (s1 / n) * H(p1) - ((n - s1) / n) * H(p0); gain[(s1 < 100) | (n - s1 < 100)] = -1
    k = int(np.argmax(gain))
    if gain[k] < 0.001: leaves.append((n, rate, path)); return
    grow(rows[X[rows, k] == 0], depth - 1, path + [f"{names[k]}=0"]); grow(rows[X[rows, k] == 1], depth - 1, path + [f"{names[k]}=1"])
grow(np.arange(len(idx)), 14, [])
tot = sum(l[0] for l in leaves); hleaf = sum(l[0] * H(l[1]) for l in leaves) / tot
pure = sum(l[0] for l in leaves if l[1] < 0.02 or l[1] > 0.98)
print(f"leaves {len(leaves)}; H(y) {H(y.mean()):.3f} -> mean leaf entropy {hleaf:.3f}; rows in >98%-pure leaves: {pure} ({pure/tot:.2f})")
leaves.sort(key=lambda l: -l[0] * abs(l[1] - 0.5))
print("largest confident leaves:"); [print(f"   n={l[0]:6d} rate={l[1]:.3f}  {' & '.join(l[2])}") for l in leaves[:14]]
# determinism by key on vocabulary windows
def coll(cols_idx, label):
    key = np.zeros(len(idx), np.uint64)
    for k in cols_idx: key = key * np.uint64(2) + X[:, k].astype(np.uint64)
    order = np.argsort(key, kind='stable'); ks = key[order]; es = e[idx][order]
    same = ks[1:] == ks[:-1]; conf = same & (es[1:] != es[:-1])
    print(f"   {label:45s}: {len(np.unique(key))} keys, pairs {int(same.sum())}, conflicts {conf.sum()/max(same.sum(),1):.3f}")
nm = {n: k for k, n in enumerate(names)}
print("determinism given vocabulary subsets (conflict rate; 0 = deterministic):")
coll([nm[f"m{k}.0"] for k in range(2, 8)] + [nm[f"p{k}.{b}"] for k in (4, 5) for b in (11, 14, 15)] + [nm[f"e{k}={v}"] for k in (1, 2, 3) for v in (1, 0)], "m.0[2..7] p[4,5].{11,14,15} e[1..3]")
coll([nm[f"m{k}.0"] for k in range(0, 10)] + [nm[f"p{k}.{b}"] for k in range(3, 10) for b in (11, 14, 15)] + [nm[f"e{k}={v}"] for k in range(1, 9) for v in (1, 0)], "all vocabulary (rare collisions)")
coll([nm[f"m{k}.0"] for k in range(2, 8)] + [nm[f"p{k}.{b}"] for k in (4, 5, 6) for b in (11, 14, 15)], "m.0[2..7] p[4..6].{11,14,15} (no e lags)")
