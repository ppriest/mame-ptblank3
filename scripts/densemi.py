import sys, numpy as np
from ns10 import load_dump
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
idx = np.arange(60, N - 3); dense = G[idx].copy()
for w in range(0, 8): dense &= np.bitwise_count(p[idx - w]) >= 4
idx = idx[dense]; ee = e[idx].astype(np.int64); print("dense gate-open rows:", len(idx), "event rate", ee.mean())
def mi(x, y):
    n = len(x); jx = x.astype(np.int64) * 2 + y
    cj = np.bincount(jx, minlength=1 << 17).astype(np.float64); cx = cj.reshape(-1, 2).sum(1); cy = np.array([n - y.sum(), y.sum()], np.float64)
    pj = cj / n; px = cx / n; py = cy / n; nz = pj > 0
    return float((pj[nz] * np.log2(pj[nz] / (np.repeat(px, 2)[nz] * np.tile(py, 1 << 16)[nz]))).sum())
rng = np.random.default_rng(3); esh = rng.permutation(ee)
print("MI(event; word) within dense regions, minus shuffled baseline:")
for w in range(-1, 8):
    row = []
    for nm, arr in (("c", c), ("p", p), ("m", m)):
        x = arr[idx - w].copy()
        if w == -1 and nm != "c": x = x & 0xf7ff
        row.append(f"{nm}:{mi(x, ee) - mi(x, esh):+.4f}")
    print(f"  word i-{w:2d}: " + "  ".join(row))
for k in (1, 2, 3):
    x = e[idx - k].astype(np.uint16); print(f"  MI(event; e_i-{k}) = {mi(x, ee) - mi(x, esh):+.4f}   (P(e|e-{k}=1)={ee[e[idx-k]==1].mean():.3f}, P(e|e-{k}=0)={ee[e[idx-k]==0].mean():.3f})")
