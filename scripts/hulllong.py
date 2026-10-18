import sys, numpy as np, time
from ns10 import load_dump
import features, gf2
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
L = int(sys.argv[2]) if len(sys.argv) > 2 else 32
idx = np.arange(L + 4, N - 3); idx = idx[G[idx]]; E = idx[e[idx] == 1]
def state(ix):
    cl = [np.ones((len(ix), 1), np.uint8)]
    for w in range(L): cl += [features.bits16(c[ix - w]), features.bits16(p[ix - w])]
    return np.concatenate(cl, 1)
t0 = time.time()
XA = state(idx[::max(1, len(idx) // 12000)]); ra, piv = gf2.rank_and_pivots(XA)
XE = state(E[::max(1, len(E) // 12000)])[:, piv]; re_, _ = gf2.rank_and_pivots(XE)
print(f"L={L} words ({32*L+1} cols): rank all {ra}, rank events {re_} -> affine factors: {ra - re_}  ({time.time()-t0:.0f}s)")
