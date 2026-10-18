import sys, numpy as np, itertools
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
cols = []
for k in range(0, 12): cols.append(((m[rows - k] >> 0) & 1).astype(np.uint8))
for k in range(2, 11):
    for bb in (11, 14, 15): cols.append(((p[rows - k] >> bb) & 1).astype(np.uint8))
for k in range(1, LAGS + 1): cols.append(e[rows - k].astype(np.uint8))
V = np.stack(cols, 1); y = e[rows]
key = np.packbits(V, axis=1); keyv = key.view(np.uint8).reshape(len(rows), -1)
kk = np.array([int.from_bytes(r.tobytes(), 'little') for r in keyv], dtype=object)
uniq = {}
for q, k_ in enumerate(kk): uniq.setdefault(k_, []).append(q)
conf = sum(1 for v in uniq.values() if len(set(y[v])) > 1)
print(f"rows {len(rows)}, distinct vocabulary patterns {len(uniq)}, patterns with conflicting outputs {conf}")
trivial = (V[:, 12:39].sum(1) == 0) & (V[:, 39:].sum(1) == 0)
print(f"rows with all p bits and h lags zero: {trivial.sum()} (event rate there {y[trivial].mean():.4f})")
# split by pattern: half of distinct patterns for fit, rest for test
rng = np.random.default_rng(0); keys = list(uniq.keys()); rng.shuffle(keys)
fitk = set(keys[:len(keys) // 2])
fit_rows = np.array([q for k_ in keys[:len(keys)//2] for q in uniq[k_]]); test_rows = np.array([uniq[k_][0] for k_ in keys[len(keys)//2:]])
def monos(W):
    out = [np.ones(len(W), np.uint8)] + [W[:, k] for k in range(W.shape[1])]
    for cb in itertools.combinations(range(W.shape[1]), 2): out.append(W[:, cb[0]] & W[:, cb[1]])
    for cb in itertools.combinations(range(W.shape[1]), 3): out.append(W[:, cb[0]] & W[:, cb[1]] & W[:, cb[2]])
    return np.stack(out, 1)
Xf = monos(V[fit_rows]); res = gf2.solve(Xf, y[fit_rows], "ho")
print(f"fit on {len(fit_rows)} rows ({len(keys)//2} patterns): inconsistent {res['bad']}")
Xt = monos(V[test_rows]); pred = np.zeros(len(test_rows), np.uint8)
for k_ in res['solution']: pred ^= Xt[:, k_]
yt = y[test_rows]; nt = ~trivial[test_rows]
print(f"held-out novel patterns: {len(test_rows)}; accuracy {np.mean(pred == yt):.3f}; on non-trivial ones ({nt.sum()}): {np.mean(pred[nt] == yt[nt]):.3f}; base rate of 1s there {yt[nt].mean():.3f}")
