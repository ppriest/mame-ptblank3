import sys, numpy as np, time
from ns10 import load_dump
import features
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); gate = ~((a == 0) & (bq == 1))
mode = sys.argv[2] if len(sys.argv) > 2 else "all"
idx = np.arange(60, N - 3); idx = idx[gate[idx] & gate[idx - 1]]
if mode == "on": idx = idx[e[idx - 1] == 0]
if mode == "off": idx = idx[e[idx - 1] == 1]
y = e[idx].astype(np.int64); print(f"mode {mode}: transitions {len(idx)}, P(h_i=1) = {y.mean():.4f}")
cols, names = [], []
for w in range(-1, 10):
    for nm, arr in (("c", c), ("p", p), ("m", m)):
        if w == -1 and nm != "c": continue
        bits = features.bits16(arr[idx - w]); cols.append(bits); names += [f"{nm}[i-{w}].{bb}" for bb in range(16)]
cols.append(np.stack([t[idx - k] for k in range(1, 9)], 1).astype(np.uint8)); names += [f"t-{k}" for k in range(1, 9)]
cols.append(np.stack([e[idx - k] for k in range(1, 5)], 1).astype(np.uint8)); names += [f"e-{k}" for k in range(1, 5)]
X = np.concatenate(cols, 1).astype(np.int64)
def cond_entropy(key):
    # H(y | key) with key integer-coded
    kk = key * 2 + y; cnt_ = np.bincount(kk); cnt_ = cnt_.reshape(-1, 2).astype(np.float64)
    tot = cnt_.sum(1); pr = cnt_ / np.maximum(tot[:, None], 1); 
    with np.errstate(divide='ignore', invalid='ignore'):
        h = -(np.where(pr > 0, pr * np.log2(pr), 0)).sum(1)
    return float((tot * h).sum() / tot.sum())
base = np.zeros(len(idx), np.int64)
H0 = cond_entropy(base); print(f"H(h_i) = {H0:.4f} bits")
chosen = []
for step in range(10):
    best = (H0 + 1, None)
    for k in range(X.shape[1]):
        if k in chosen: continue
        key = base * 2 + X[:, k]
        hk = cond_entropy(key)
        if hk < best[0]: best = (hk, k)
    chosen.append(best[1]); base = base * 2 + X[:, best[1]]
    print(f"step {step+1}: add {names[best[1]]:14s} -> H(h_i | ...) = {best[0]:.4f}", flush=True)
# show the resulting table for the first 6 chosen bits
key = np.zeros(len(idx), np.int64)
for k in chosen[:6]: key = key * 2 + X[:, k]
cnt_ = np.bincount(key * 2 + y, minlength=2 ** 7).reshape(-1, 2)
print("table over first 6 chosen bits (pattern -> n0, n1, rate):")
for q in range(2 ** 6):
    if cnt_[q].sum() > 0: print(f"   {q:06b}: {cnt_[q][0]:7d} {cnt_[q][1]:7d}  {cnt_[q][1] / cnt_[q].sum():.3f}")
