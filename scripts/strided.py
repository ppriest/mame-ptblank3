import sys, numpy as np, time
from ns10 import load_dump
import gf2, features
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c)
NF = 26000
sel = np.arange(40, N - 1)
L, names, y = features.window(c, p, t, 0, N, w=4)
tl = np.stack([t[np.arange(N) - k] for k in range(1, 5)], 1).astype(np.uint8)
L = np.concatenate([L, tl], 1)
r, piv = gf2.rank_and_pivots(L[sel][:20000]); S = L[:, piv]; print("state pivots (4 words + 4 t lags):", len(piv))
bitsC = features.bits16(c); bitsP = features.bits16(p); stream = np.concatenate([bitsC, bitsP, t[:, None].astype(np.uint8)], 1)  # 33 stream bits
def strided_prefix(P):
    H = np.zeros((N, 33 * P), np.uint8)
    for rr in range(P):
        acc = np.bitwise_xor.accumulate(stream[rr::P], 0)
        H[rr::P, rr*33:(rr+1)*33] = acc
        # make the register value at i = parity over indices <= i with same residue; positions with other residues hold the latest value
    # fill: for each residue column block, forward-fill to all positions (value = last update at index <= i)
    for rr in range(P):
        blk = H[:, rr*33:(rr+1)*33]
        for shift in range(1, P):
            rows = np.arange(rr + shift, N, P)
            blk[rows] = blk[rows - shift]
    return H
def bilinear(H, label):
    idx = sel[:NF]; Si = S[idx]; Hi = H[idx]
    rh, hp = gf2.rank_and_pivots(np.concatenate([Si, Hi], 1)[:12000]); Hi2 = np.concatenate([Si, Hi], 1)[:, hp][:, len(piv):] if rh > len(piv) else Hi[:, :0]
    # use independent hidden columns only
    Hi = Hi2
    prods = (Hi[:, :, None] & Si[:, None, :]).reshape(len(idx), -1)
    X = np.concatenate([np.ones((len(idx), 1), np.uint8), Si, Hi, prods], 1)
    t0 = time.time(); res = gf2.solve(X, t[idx], "bil"); free = len(idx) - res['rank']
    print(f"  [{label}] hidden cols {Hi.shape[1]}, X cols {X.shape[1]}, rank {res['rank']}, inconsistent {res['bad']} of {free} ({time.time()-t0:.1f}s)", flush=True)
for P in (2, 3, 4, 5, 6, 7, 8):
    bilinear(strided_prefix(P), f"period-{P} accumulators")
