import sys, numpy as np, time
from ns10 import load_dump
import gf2, features
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
NF = 9000
sel = np.arange(8, N - 1)
L3, names, y = features.window(c, p, t, 0, N, w=3)
r, piv = gf2.rank_and_pivots(L3[sel][:20000]); print("3-word pivots", len(piv))
S = L3[:, piv]                      # (N,75) pivot state bits for every position
def bilinear_fit(H, hnames, label, rows=NF):
    """t_i = L(S_i) ^ L'(H_i) ^ sum_{k,m} a_km H_i[k] S_i[m]  (+ const). H: (N, nH) hidden-candidate features."""
    idx = sel[:rows]
    Si = S[idx]; Hi = H[idx]
    prods = (Hi[:, :, None] & Si[:, None, :]).reshape(len(idx), -1)
    X = np.concatenate([np.ones((len(idx), 1), np.uint8), Si, Hi, prods], 1)
    res = gf2.solve(X, t[idx], "bil")
    free = len(idx) - res['rank']
    print(f"  [{label}] cols {X.shape[1]} rank {res['rank']} inconsistent {res['bad']} of {free}", flush=True)
    return res
# D. prefix parities of the 32 stream bit columns (c bits, p bits), evaluated up to and including word i
bitsC = features.bits16(c); bitsP = features.bits16(p)
pref = np.concatenate([np.bitwise_xor.accumulate(bitsC, 0), np.bitwise_xor.accumulate(bitsP, 0)], 1).astype(np.uint8)
bilinear_fit(pref, None, "prefix-parity accumulator (32 cols)")
pref1 = np.roll(pref, 1, 0); pref1[0] = 0
bilinear_fit(pref1, None, "prefix-parity up to word i-1")
# E. hold register: F_i = x(state_{j(i)}), j(i) = last index <= i (variant A) or < i (variant B) where select bit fired
cands = {}
for k in range(3):
    for bb in range(16):
        cands[f"c{k}.{bb}"] = bitsC[np.maximum(np.arange(N) - k, 0), bb]
        cands[f"p{k}.{bb}"] = bitsP[np.maximum(np.arange(N) - k, 0), bb]
        cands[f"m{k}.{bb}"] = bitsC[np.maximum(np.arange(N) - k, 0), bb] ^ bitsP[np.maximum(np.arange(N) - k, 0), bb]
best = []
t0 = time.time()
for name, s in cands.items():
    fired = np.where(s == 1)[0]
    for variant in ("<=", "<"):
        pos = np.arange(N)
        jj = np.searchsorted(fired, pos, side='right' if variant == "<=" else 'left') - 1
        jidx = fired[np.maximum(jj, 0)]
        H = S[jidx]                     # state at last fire (75 cols)
        idx = sel[:NF]
        Si = S[idx]; Hi = H[idx]
        prods = (Hi[:, :, None] & Si[:, None, :]).reshape(len(idx), -1)
        X = np.concatenate([np.ones((len(idx), 1), np.uint8), Si, Hi, prods], 1)
        res = gf2.solve(X, t[idx], "hold")
        free = len(idx) - res['rank']
        best.append((res['bad'] / max(free, 1), name, variant, res['rank'], res['bad'], free))
        if res['bad'] == 0: print(f"  CONSISTENT: select {name} variant {variant}: rank {res['rank']} free {free}", flush=True)
best.sort()
print("hold-register search done in %.0fs; best 8 (inconsistent fraction):" % (time.time() - t0))
for bst in best[:8]: print("  ", bst)
