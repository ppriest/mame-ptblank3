import sys, numpy as np
from ns10 import load_dump
import features
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("nl_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
lo, hi = 41, N - 2; idx = np.arange(lo, hi)[::3]; ee = e[idx]
cols, names = [], []
for w in range(-1, 6):
    for nm, arr in (("c", c), ("p", p), ("m", m)):
        cols.append(features.bits16(arr[idx - w])); names += [f"{nm}[i-{w}].{bb}" for bb in range(16)]
cols.append(np.stack([t[idx - k] for k in range(1, 9)], 1).astype(np.uint8)); names += [f"t-{k}" for k in range(1, 9)]
B = np.concatenate(cols, 1).astype(np.int32); n = B.shape[0]
# co-occurrence counts for (x=1,y=1) overall and among events
def cooc(M): return M.T @ M
C11 = cooc(B); E11 = cooc(B * ee[:, None])
sx = np.diag(C11); ex = np.diag(E11); ne = ee.sum()
# four polarity combos: counts of rows with x=a,y=b
res = []
for a in (1, 0):
    for bq in (1, 0):
        if a == 1 and bq == 1: Cab, Eab = C11, E11
        elif a == 1 and bq == 0: Cab, Eab = sx[:, None] - C11, ex[:, None] - E11
        elif a == 0 and bq == 1: Cab, Eab = sx[None, :] - C11, ex[None, :] - E11
        else: Cab, Eab = n - sx[:, None] - sx[None, :] + C11, ne - ex[:, None] - ex[None, :] + E11
        rate = Eab / np.maximum(Cab, 1); rate[Cab < 2000] = 0; np.fill_diagonal(rate, 0)
        k = np.unravel_index(np.argmax(rate), rate.shape)
        res.append((rate[k], names[k[0]], a, names[k[1]], bq, int(Cab[k])))
res.sort(reverse=True)
print("2. highest event rate on a conjunction of two literals (rate, x=a, y=b, support):")
for r in res: print("   ", r)
