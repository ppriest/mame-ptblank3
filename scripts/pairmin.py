import sys, numpy as np
from ns10 import load_dump
import features, gf2
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("nl_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
lo, hi = 41, N - 3; idx = np.arange(lo, hi); ee = e[idx]
cols, names = [], []
for w in range(-1, 7):
    for nm, arr in (("c", c), ("p", p), ("m", m)):
        cols.append(features.bits16(arr[idx - w])); names += [f"{nm}[i-{w}].{bb}" for bb in range(16)]
cols.append(np.stack([t[idx - k] for k in range(1, 13)], 1).astype(np.uint8)); names += [f"t-{k}" for k in range(1, 13)]
B = np.concatenate(cols, 1)
def forbidden(mask, label, minsup=1500, top=12):
    Bf = B[mask].astype(np.float32); ef = ee[mask].astype(np.float32); n = Bf.shape[0]
    C11 = Bf.T @ Bf; E11 = (Bf * ef[:, None]).T @ Bf
    sx = np.diag(C11).copy(); ex = np.diag(E11).copy(); ne = ef.sum()
    out = []
    for a in (1, 0):
        for bq in (1, 0):
            if (a, bq) == (1, 1): Cab, Eab = C11, E11
            elif (a, bq) == (1, 0): Cab, Eab = sx[:, None] - C11, ex[:, None] - E11
            elif (a, bq) == (0, 1): Cab, Eab = sx[None, :] - C11, ex[None, :] - E11
            else: Cab, Eab = n - sx[:, None] - sx[None, :] + C11, ne - ex[:, None] - ex[None, :] + E11
            rate = np.where(Cab >= minsup, Eab / np.maximum(Cab, 1), 1.0); np.fill_diagonal(rate, 1.0)
            ii, jj = np.where(rate == 0)
            for i_, j_ in zip(ii, jj):
                if i_ < j_: out.append((int(Cab[i_, j_]), names[i_], a, names[j_], bq))
            # also lowest nonzero
            k = np.unravel_index(np.argmin(rate), rate.shape)
            out.append((-1, f"min-rate {rate[k]:.4f}", names[k[0]], a, names[k[1]], bq, int(Cab[k])))
    out.sort(key=lambda r: -r[0])
    print(f"[{label}] rows {n}, events {int(ne)}, rate {ne/n:.4f}; forbidden pairs (0 events, support>=minsup):")
    for r in out[:top]: print("    ", r)
    return out
all_ = np.ones(len(idx), bool)
forbidden(all_, "ALL")
a0 = ((m[idx - 2] >> 0) & 1) == 1; b0 = ((m[idx - 3] >> 0) & 1) == 1
sub = a0 | ~b0
forbidden(sub, "SUBSET (m2.0 | ~m3.0)")
# affine hull test for R on the subset
E = idx[sub & (ee == 1)]; A_ = idx[sub]
def state(ix):
    cl = [np.ones((len(ix), 1), np.uint8)]
    for w in range(-1, 7): cl += [features.bits16(c[ix - w]), features.bits16(p[ix - w])]
    cl += [t[ix - k][:, None].astype(np.uint8) for k in range(1, 13)]
    return np.concatenate(cl, 1)
re_, _ = gf2.rank_and_pivots(state(E[::max(1, len(E)//50000)])); ra, _ = gf2.rank_and_pivots(state(A_[::30]))
print(f"affine hull on subset: events rank {re_}, all rank {ra}")
