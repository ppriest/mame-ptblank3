import sys, numpy as np
from ns10 import load_dump
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
EN = np.zeros(N, bool)
for w in range(4, 97): EN[100:] |= (p[100 - w:N - w] & 0xc980) != 0
# rolling hash of plaintext window [i-W+1 .. i] via polynomial hash
P = p.astype(np.uint64) + np.uint64(1)
def window_hash(W):
    B = np.uint64(0x100000001B3); h = np.zeros(N, np.uint64); powW = np.uint64(1)
    for _ in range(W): powW = powW * B
    acc = np.uint64(0)
    with np.errstate(over='ignore'):
        for k in range(N):
            acc = acc * B + P[k]
            if k >= W: acc = acc - P[k - W] * powW
            h[k] = acc
    return h
idx = np.arange(200, N - 3); idx = idx[G[idx] & EN[idx]]
for W in [int(x) for x in sys.argv[2].split(",")]:
    h = window_hash(W)[idx]
    order = np.argsort(h, kind='stable'); hs = h[order]; rs = idx[order]
    same = hs[1:] == hs[:-1]; I = rs[:-1][same]; J = rs[1:][same]
    far = np.abs(I.astype(np.int64) - J.astype(np.int64)) > 4 * W
    I, J = I[far], J[far]
    # verify windows really identical (hash collisions)
    ok = np.array([np.array_equal(p[x - W + 1:x + 1], p[y - W + 1:y + 1]) for x, y in zip(I[:2000], J[:2000])])
    diffc = c[I] != c[J]
    ag = (e[I] == e[J]); pe = e[idx].mean()
    sel = diffc
    print(f"W={W:4d}: far pairs {len(I):6d} (verified {ok.mean():.3f}), different-cipher {sel.sum():6d}; agreement {ag[sel].mean():.3f}; both-event {int((e[I]&e[J])[sel].sum())}, one-event {int((e[I]^e[J])[sel].sum())}; chance agreement {1-2*pe*(1-pe):.3f}", flush=True)
