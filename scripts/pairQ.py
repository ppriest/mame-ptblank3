import sys, numpy as np, itertools
from ns10 import load_dump
import gf2, heldfit
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
EN = np.zeros(N, bool)
for w in range(4, 97): EN[100:] |= (p[100 - w:N - w] & 0xc980) != 0
W = int(sys.argv[2]); MW = int(sys.argv[3]); DEG = int(sys.argv[4])
P = p.astype(np.uint64) + np.uint64(1); B = np.uint64(0x100000001B3); powW = np.uint64(1)
with np.errstate(over='ignore'):
    for _ in range(W): powW = powW * B
    h = np.zeros(N, np.uint64); acc = np.uint64(0)
    for k in range(N):
        acc = acc * B + P[k]
        if k >= W: acc = acc - P[k - W] * powW
        h[k] = acc
idx = np.arange(200, N - 3); idx = idx[G[idx] & EN[idx]]
hh = h[idx]; order = np.argsort(hh, kind='stable'); hs = hh[order]; rs = idx[order]
same = hs[1:] == hs[:-1]; I = rs[:-1][same]; J = rs[1:][same]
far = np.abs(I.astype(np.int64) - J.astype(np.int64)) > 4 * W; I, J = I[far], J[far]
act = (e[I] | e[J]) == 1; I, J = I[act], J[act]
print(f"W={W}: pairs with >=1 event: {len(I)}; disagreement {np.mean(e[I]!=e[J]):.3f}")
def bits(x): return ((x[:, None] >> np.arange(16, dtype=np.uint16)) & 1).astype(np.uint8)
def V(r):
    cols = [bits(m[r - w]) for w in range(MW)] + [np.delete(bits(c[r + 1]), 11, axis=1)] + [np.stack([t[r - k] for k in range(1, 5)], 1).astype(np.uint8)]
    return np.concatenate(cols, 1)
def mono(Vm):
    out = [Vm]
    if DEG >= 2: out.append(np.stack([Vm[:, x] & Vm[:, y] for x, y in itertools.combinations(range(Vm.shape[1]), 2)], 1))
    return np.concatenate(out, 1)
X = mono(V(I)) ^ mono(V(J)); y = (e[I] ^ e[J]).astype(np.uint8)
# split by region: pairs sorted by I position, hold out the last 30% of positions
o = np.argsort(I); X, y, I2 = X[o], y[o], I[o]; ns = int(0.7 * len(y))
res, acc, base = heldfit.fit_test(X[:ns], y[:ns], X[ns:], y[ns:], "pq")
print(f"   mask words {MW}, degree {DEG}: {X.shape[1]} columns, fit rows {ns}, rank {res['rank']}, inconsistent {res['bad']}; held-out acc {acc:.3f} (predict-0 {base:.3f}, predict-1 {1-base:.3f})")
