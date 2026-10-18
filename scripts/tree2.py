import sys, numpy as np
from ns10 import load_dump
import features
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
b11 = ((p >> 11) & 1).astype(bool)
idx = np.arange(200, N - 3); idx = idx[G[idx] & b11[idx - 4]]
cols, names = [], []
for w in range(-1, 13):
    for nm, arr in (("c", c), ("p", p), ("m", m)):
        if w == -1 and nm != "c": continue
        cols.append(features.bits16(arr[idx - w])); names += [f"{nm}[i-{w}].{bb}" for bb in range(16)]
cols.append(np.stack([t[idx - k] for k in range(1, 13)], 1).astype(np.uint8)); names += [f"t-{k}" for k in range(1, 13)]
# gate-aware event lags: value only meaningful where the gate was open; encode as two bits (open&event, open&noevent)
for k in range(1, 9):
    cols.append(np.stack([(G[idx - k] & (e[idx - k] == 1)), (G[idx - k] & (e[idx - k] == 0))], 1).astype(np.uint8)); names += [f"e-{k}=1", f"e-{k}=0"]
# words since last bit-11 word (depth>=5) as 6 bits
last = np.full(N, -10**6, np.int64); lj = -10**6
for k in range(N):
    if b11[k]: lj = k
    last[k] = lj
since = np.minimum(idx - 5 - last[idx - 5], 63); cols.append(((since[:, None] >> np.arange(6)) & 1).astype(np.uint8)); names += [f"since11.{q}" for q in range(6)]
X = np.concatenate(cols, 1).astype(np.uint8); y = e[idx].astype(np.float64); print("rows", len(idx), "rate", y.mean(), "features", X.shape[1])
def H(pr):
    pr = np.clip(pr, 1e-9, 1 - 1e-9); return -(pr * np.log2(pr) + (1 - pr) * np.log2(1 - pr))
leaves = []
def grow(rows, depth, path):
    yy = y[rows]; n = len(rows); rate = yy.mean()
    if depth == 0 or n < 600 or rate < 0.02 or rate > 0.98:
        leaves.append((n, rate, path)); return
    Xr = X[rows].astype(np.float32); s1 = Xr.sum(0); e1 = (Xr * yy[:, None].astype(np.float32)).sum(0)
    p1 = e1 / np.maximum(s1, 1); p0 = (yy.sum() - e1) / np.maximum(n - s1, 1)
    gain = H(rate) - (s1 / n) * H(p1) - ((n - s1) / n) * H(p0); gain[(s1 < 300) | (n - s1 < 300)] = -1
    k = int(np.argmax(gain))
    if gain[k] < 0.002: leaves.append((n, rate, path)); return
    grow(rows[X[rows, k] == 0], depth - 1, path + [f"{names[k]}=0"]); grow(rows[X[rows, k] == 1], depth - 1, path + [f"{names[k]}=1"])
grow(np.arange(len(idx)), 8, [])
leaves.sort(key=lambda l: l[1])
tot = sum(l[0] for l in leaves); hleaf = sum(l[0] * H(l[1]) for l in leaves) / tot
print(f"leaves {len(leaves)}; entropy of y {H(y.mean()):.3f} -> mean leaf entropy {hleaf:.3f} bits")
print("purest low leaves:"); [print(f"   n={l[0]:6d} rate={l[1]:.3f}  {' & '.join(l[2])}") for l in leaves[:6]]
print("purest high leaves:"); [print(f"   n={l[0]:6d} rate={l[1]:.3f}  {' & '.join(l[2])}") for l in leaves[-6:]]
