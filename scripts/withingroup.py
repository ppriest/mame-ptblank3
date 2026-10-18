import sys, numpy as np, time
from ns10 import load_dump
import features
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
idx = np.arange(80, N - 3); idx = idx[G[idx]]
lo, hi = (int(x) for x in sys.argv[2].split(",")) if len(sys.argv) > 2 else (0, 11)
MIN = int(sys.argv[3]) if len(sys.argv) > 3 else 40
key = np.zeros(len(idx), np.uint64)
for w in range(lo, hi + 1): key = key * np.uint64(0x9E3779B97F4A7C15) + p[idx - w].astype(np.uint64) + np.uint64(w)
order = np.argsort(key, kind='stable'); ks = key[order]; rows = idx[order]
bounds = np.concatenate([[0], np.where(ks[1:] != ks[:-1])[0] + 1, [len(ks)]]); sizes = np.diff(bounds)
big = np.where(sizes >= MIN)[0]
# keep only groups whose plaintext window is dense (>= 8 of the window's words nonzero, popcount >= 3)
dense_ok = []
for g in big:
    r0 = rows[bounds[g]]
    nzw = sum(1 for w in range(lo, hi + 1) if np.bitwise_count(p[r0 - w]) >= 3)
    if nzw >= (hi - lo + 1) - 2: dense_ok.append(g)
big = np.array(dense_ok, dtype=np.int64)
# skip all-zero plaintext groups
print(f"plaintext window i-{lo}..i-{hi}: groups >= {MIN}: {len(big)}, rows {sizes[big].sum()}")
cols, names = [], []
for w in range(-1, 8):
    for nm, arr in (("c", c), ("m", m)):
        if w == -1 and nm != "c": continue
        cols.append(features.bits16(arr[rows - w])); names += [f"{nm}[i-{w}].{bb}" for bb in range(16)]
cols.append(np.stack([t[rows - k] for k in range(1, 9)], 1).astype(np.uint8)); names += [f"t-{k}" for k in range(1, 9)]
B = np.concatenate(cols, 1).astype(np.float32); y = e[rows].astype(np.float32)
sep_sum = np.zeros(B.shape[1]); wsum = 0; hvar = 0; nvar = 0
for g in big:
    s, en = bounds[g], bounds[g + 1]; Bg = B[s:en]; yg = y[s:en]; n = en - s
    if yg.mean() == 0 or yg.mean() == 1: continue
    s1 = Bg.sum(0); e1 = (Bg * yg[:, None]).sum(0)
    ok = (s1 >= 8) & (n - s1 >= 8)
    r1 = e1 / np.maximum(s1, 1); r0 = (yg.sum() - e1) / np.maximum(n - s1, 1)
    sep = np.where(ok, np.abs(r1 - r0), 0); sep_sum += sep * n; wsum += n
    hvar += n * yg.mean() * (1 - yg.mean()); nvar += n
sep = sep_sum / wsum
order_b = np.argsort(-sep)
print(f"dense groups used: {len(big)}, rows {int(wsum)}; mean within-group event variance {hvar/max(nvar,1):.3f} (marginal p(1-p) for rate 0.27 = 0.197)")
print("top within-group single-bit separations (a mux select would give ~0.5-1.0; noise ~0.1-0.15):")
for k in order_b[:15]: print(f"   {sep[k]:.3f}  {names[k]}")
print("median separation over all bits: %.3f" % np.median(sep))
