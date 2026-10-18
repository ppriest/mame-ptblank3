import sys, numpy as np, itertools, time
from ns10 import load_dump
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
idx = np.arange(80, N - 3); idx = idx[G[idx]]
dense = np.ones(len(idx), bool)
for w in range(0, 6): dense &= np.bitwise_count(p[idx - w]) >= 3
mode = sys.argv[2] if len(sys.argv) > 2 else "dense"
if mode == "dense": idx = idx[dense]
ee = e[idx].astype(np.int64); n = len(idx); print("rows", n, "rate", ee.mean())
def mi(x):
    jx = x.astype(np.int64) * 2 + ee
    cj = np.bincount(jx, minlength=1 << 17).astype(np.float64); cx = cj.reshape(-1, 2).sum(1); cy = np.array([n - ee.sum(), ee.sum()], np.float64)
    pj = cj / n; px = cx / n; py = cy / n; nz = pj > 0
    return float((pj[nz] * np.log2(pj[nz] / (np.repeat(px, 2)[nz] * np.tile(py, 1 << 16)[nz]))).sum())
rng = np.random.default_rng(5); perm = rng.permutation(n)
def mi_adj(x): 
    v = mi(x); jx = x.astype(np.int64) * 2 + ee[perm]
    cj = np.bincount(jx, minlength=1 << 17).astype(np.float64); cx = cj.reshape(-1, 2).sum(1); cy = np.array([n - ee.sum(), ee.sum()], np.float64)
    pj = cj / n; px = cx / n; py = cy / n; nz = pj > 0
    return v - float((pj[nz] * np.log2(pj[nz] / (np.repeat(px, 2)[nz] * np.tile(py, 1 << 16)[nz]))).sum())
words = {}
for w in range(0, 8):
    for nm, arr in (("c", c), ("p", p), ("m", m)): words[f"{nm}{w}"] = arr[idx - w].astype(np.uint32)
words["c-1"] = c[idx + 1].astype(np.uint32)
base = {k: mi_adj(v.astype(np.uint16)) for k, v in words.items()}
print("single-word MI (adj) max:", max(base.items(), key=lambda kv: kv[1]))
res = []; t0 = time.time()
keys = list(words.keys())
for ka, kb in itertools.combinations(keys, 2):
    A, B = words[ka], words[kb]
    for op, f in (("+", (A + B) & 0xffff), ("-", (A - B) & 0xffff), ("^", A ^ B), ("&", A & B), ("|", A | B)):
        res.append((mi_adj(f.astype(np.uint16)), f"{ka}{op}{kb}"))
res.sort(reverse=True)
print(f"{len(res)} pair functions in {time.time()-t0:.0f}s; top 15 by MI(event; f(A,B)) minus shuffled:")
for v, nm in res[:15]: print(f"   {v:+.4f}  {nm}")
