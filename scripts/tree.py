import sys, numpy as np
from ns10 import load_dump
import features
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("nl_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
idx = np.arange(41, N - 3)[::4]
a = ((m[idx - 2] >> 0) & 1); bq = ((m[idx - 3] >> 0) & 1)
cols, names = [], []
for w in range(0, 7):
    for nm, arr in (("c", c), ("p", p), ("m", m)):
        cols.append(features.bits16(arr[idx - w])); names += [f"{nm}[i-{w}].{bb}" for bb in range(16)]
cols.append(features.bits16(c[idx + 1])); names += [f"c[i+1].{bb}" for bb in range(16)]
cols.append(np.stack([t[idx - k] for k in range(1, 9)], 1).astype(np.uint8)); names += [f"t-{k}" for k in range(1, 9)]
cols.append(np.stack([e[idx - k] for k in range(1, 5)], 1).astype(np.uint8)); names += [f"e-{k}" for k in range(1, 5)]
X = np.concatenate(cols, 1).astype(np.uint8); y = e[idx]
def H(pr): 
    pr = np.clip(pr, 1e-9, 1 - 1e-9); return -(pr * np.log2(pr) + (1 - pr) * np.log2(1 - pr))
def grow(rows, depth, prefix=""):
    yy = y[rows]; n = len(rows); rate = yy.mean()
    if depth == 0 or n < 3000 or rate == 0 or rate == 1:
        print(f"{prefix}leaf n={n} rate={rate:.3f}"); return
    Xr = X[rows].astype(np.float32); s1 = Xr.sum(0); e1 = (Xr * yy[:, None].astype(np.float32)).sum(0)
    p1 = e1 / np.maximum(s1, 1); p0 = (yy.sum() - e1) / np.maximum(n - s1, 1)
    gain = H(rate) - (s1 / n) * H(p1) - ((n - s1) / n) * H(p0); gain[(s1 < 500) | (n - s1 < 500)] = -1
    k = int(np.argmax(gain))
    print(f"{prefix}split {names[k]} gain={gain[k]:.4f} n={n} rate={rate:.3f} -> rate(0)={p0[k]:.3f} rate(1)={p1[k]:.3f}")
    grow(rows[X[rows, k] == 0], depth - 1, prefix + "  0:"); grow(rows[X[rows, k] == 1], depth - 1, prefix + "  1:")
grow(np.arange(len(idx)), 5)
