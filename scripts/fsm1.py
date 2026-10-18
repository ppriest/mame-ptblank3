import sys, numpy as np
from ns10 import load_dump
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
idx = np.arange(200, N - 3); idx = idx[G[idx]]; y = e[idx].astype(np.int64)
cols, names = [], []
for k in range(0, 13): cols.append(((m[idx - k] >> 0) & 1).astype(np.int64)); names.append(f"m{k}.0")
for k in range(0, 13):
    for bb in (11, 14, 15): cols.append(((p[idx - k] >> bb) & 1).astype(np.int64)); names.append(f"p{k}.{bb}")
for k in range(1, 11):
    cols.append((G[idx - k] & (e[idx - k] == 1)).astype(np.int64)); names.append(f"e{k}=1")
    cols.append((G[idx - k] & (e[idx - k] == 0)).astype(np.int64)); names.append(f"e{k}=0")
X = np.stack(cols, 1)
def condH(key):
    kk = key * 2 + y; cnt_ = np.bincount(kk).astype(np.float64)
    if len(cnt_) % 2: cnt_ = np.append(cnt_, 0.0)
    cnt_ = cnt_.reshape(-1, 2); tot = cnt_.sum(1); ok = tot > 0
    pr = cnt_[ok] / tot[ok][:, None]
    with np.errstate(divide='ignore', invalid='ignore'):
        h = -(np.where(pr > 0, pr * np.log2(pr), 0)).sum(1)
    return float((tot[ok] * h).sum() / tot.sum())
base = np.zeros(len(idx), np.int64); chosen = []
H0 = condH(base); print(f"H(h) = {H0:.4f}; rows {len(idx)}")
for step in range(22):
    best = (9, None)
    for k in range(X.shape[1]):
        if k in chosen: continue
        hk = condH(base * 2 + X[:, k])
        if hk < best[0]: best = (hk, k)
    chosen.append(best[1]); base = base * 2 + X[:, best[1]]
    nkeys = len(np.unique(base))
    print(f"step {step+1:2d}: + {names[best[1]]:8s} -> H(h|...) = {best[0]:.4f}   (distinct keys {nkeys})", flush=True)
    if best[0] < 0.003: break
np.save("../data/fsm_chosen.npy", np.array([names[k] for k in chosen]))
