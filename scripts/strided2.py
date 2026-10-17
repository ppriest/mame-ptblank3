import sys, numpy as np, time
from ns10 import load_dump
import gf2, features
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c)
NF = 26000; sel = np.arange(40, N - 1)
L, names, y = features.window(c, p, t, 0, N, w=4)
tl = np.stack([t[np.arange(N) - k] for k in range(1, 5)], 1).astype(np.uint8)
L = np.concatenate([L, tl], 1)
r, piv = gf2.rank_and_pivots(L[sel][:20000]); S = L[:, piv]
bitsC = features.bits16(c); bitsP = features.bits16(p)
def acc_stream(stream, P, exclude_current):
    """Per residue class r: value at i = parity of stream[k] for k<=i (k<i if exclude_current) with k = r mod P; forward filled."""
    nb = stream.shape[1]; H = np.zeros((N, nb * P), np.uint8)
    for rr in range(P):
        acc = np.bitwise_xor.accumulate(stream[rr::P], 0)
        if exclude_current: acc = np.concatenate([np.zeros((1, nb), np.uint8), acc[:-1]], 0)
        H[rr::P, rr*nb:(rr+1)*nb] = acc
    for rr in range(P):
        blk = H[:, rr*nb:(rr+1)*nb]
        for shift in range(1, P):
            rows = np.arange(rr + shift, N, P); blk[rows] = blk[rows - shift]
    return H
def bilinear(H, label):
    idx = sel[:NF]; Si = S[idx]; Hi = H[idx]
    both = np.concatenate([Si, Hi], 1); rh, hp = gf2.rank_and_pivots(both[:12000])
    Hi = both[:, [q for q in hp if q >= Si.shape[1]]]
    prods = (Hi[:, :, None] & Si[:, None, :]).reshape(len(idx), -1)
    X = np.concatenate([np.ones((len(idx), 1), np.uint8), Si, Hi, prods], 1)
    res = gf2.solve(X, t[idx], "bil"); free = len(idx) - res['rank']
    print(f"  [{label}] hidden cols {Hi.shape[1]}, X cols {X.shape[1]}, rank {res['rank']}, inconsistent {res['bad']} of {free}", flush=True)
    return res, Hi, X
cp = np.concatenate([bitsC, bitsP], 1); tt = t[:, None].astype(np.uint8)
bilinear(acc_stream(cp, 2, False), "P=2, c/p only")
bilinear(acc_stream(tt, 2, False), "P=2, t only, incl current")
bilinear(acc_stream(tt, 2, True), "P=2, t only, excl current")
bilinear(acc_stream(tt, 1, True), "P=1, t only, excl current")
bilinear(np.concatenate([acc_stream(cp, 2, False), acc_stream(tt, 2, True)], 1), "P=2, c/p + t(excl current)")
bilinear(np.concatenate([acc_stream(cp, 1, False), acc_stream(tt, 1, True)], 1), "P=1, c/p + t(excl current)")
