import sys, numpy as np, time
from ns10 import load_dump
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
idx = np.arange(80, N - 3); idx = idx[G[idx]]; ee = e[idx]
words = {}
for w in range(-1, 9):
    for nm, arr in (("c", c), ("p", p), ("m", m)):
        if w == -1 and nm != "c": continue
        words[f"{nm}{w}"] = arr[idx - w]
MINROWS = 400
res = []; t0 = time.time()
for sname in [k for k in words if k.startswith("p")]:
    S = words[sname]; vals, inv, cnts = np.unique(S, return_inverse=True, return_counts=True)
    keep = np.where(cnts >= MINROWS)[0]
    if len(keep) < 5: continue
    for dname, Dw in words.items():
        if dname == sname: continue
        bits = ((Dw[:, None] >> np.arange(16, dtype=np.uint16)) & 1).astype(np.float32)
        score = 0; wsum = 0
        for v in keep:
            rows = inv == v; n = rows.sum(); y = ee[rows].astype(np.float32); B = bits[rows]
            s1 = B.sum(0); e1 = (B * y[:, None]).sum(0); r1 = e1 / np.maximum(s1, 1); r0 = (y.sum() - e1) / np.maximum(n - s1, 1)
            okb = (s1 >= 20) & (n - s1 >= 20)
            if not okb.any(): continue
            sep = np.abs(r1 - r0)[okb].max(); score += sep * n; wsum += n
        if wsum: res.append((score / wsum, sname, dname, len(keep), wsum))
res.sort(reverse=True)
print(f"{len(res)} (select, data) pairs in {time.time()-t0:.0f}s; mean best-bit separation given select value (mux -> ~1.0):")
for r in res[:12]: print(f"   {r[0]:.3f}  select={r[1]:4s} data={r[2]:4s} values {r[3]} rows {r[4]}")
