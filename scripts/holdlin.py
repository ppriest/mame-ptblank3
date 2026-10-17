import sys, numpy as np, time
from ns10 import load_dump
import features, gf2
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
NF = 16000; sel = np.arange(45, N - 3)
W = 6
cols = []
for w in range(W): cols += [features.bits16(c[np.maximum(np.arange(N) - w, 0)]), features.bits16(p[np.maximum(np.arange(N) - w, 0)])]
cols += [t[np.maximum(np.arange(N) - k, 0)][:, None].astype(np.uint8) for k in range(1, 9)]
L = np.concatenate(cols, 1)
r, piv = gf2.rank_and_pivots(L[sel][:20000]); S = L[:, piv]; print("state pivots", len(piv), flush=True)
H64 = np.concatenate([features.bits16(c), features.bits16(p), features.bits16(c[np.maximum(np.arange(N) - 1, 0)]), features.bits16(p[np.maximum(np.arange(N) - 1, 0)])], 1)  # 64 bits: word j and j-1
cands = {}
for w in range(4):
    for nm, arr in (("c", c), ("p", p), ("m", m)):
        bits = features.bits16(arr[np.maximum(np.arange(N) - w, 0)])
        for bb in range(16): cands[f"{nm}[i-{w}].{bb}"] = bits[:, bb]
for k in range(1, 4): cands[f"t-{k}"] = t[np.maximum(np.arange(N) - k, 0)]
cands["e-1"] = e[np.maximum(np.arange(N) - 1, 0)]
results = []; t0 = time.time()
idx = sel[:NF]
for name, s in cands.items():
    fired = np.where(s == 1)[0]
    for variant in ("<=", "<"):
        jj = np.searchsorted(fired, idx, side='right' if variant == "<=" else 'left') - 1
        if (jj < 0).any(): continue
        jidx = fired[jj]
        Si = S[idx]; Hi = H64[jidx]
        both = np.concatenate([Si, Hi], 1); rh, hp = gf2.rank_and_pivots(both[:8000]); Hi = both[:, [q for q in hp if q >= Si.shape[1]]]
        prods = (Hi[:, :, None] & Si[:, None, :]).reshape(len(idx), -1)
        X = np.concatenate([np.ones((len(idx), 1), np.uint8), Si, Hi, prods], 1)
        res = gf2.solve(X, t[idx], "hl"); free = len(idx) - res['rank']
        results.append((res['bad'] / max(free, 1), name, variant, res['rank'], res['bad'], free))
        if res['bad'] == 0: print(f"  CONSISTENT: select {name} {variant}: rank {res['rank']} free {free}", flush=True)
results.sort()
print(f"done {len(results)} in {time.time()-t0:.0f}s; best:", results[:6])
