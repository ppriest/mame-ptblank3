"""Template search: R = nonlinear plaintext-driven register (rule enumerated), output h = XOR_k R[k] & L_k(x), x = current words.
Solve L exactly in GF(2) on the first 70% of the stream, score on the last 30%."""
import sys, time, numpy as np
from ns10 import load_dump
import gf2, heldfit
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
EN = np.zeros(N, bool)
for w in range(4, 97): EN[100:] |= (p[100 - w:N - w] & 0xc980) != 0
idx = np.arange(200, N - 3); idx = idx[G[idx]]
split = int(0.7 * N); rng = np.random.default_rng(0)
tr = rng.choice(idx[idx < split], 30000, replace=False); te = rng.choice(idx[idx >= split + 100], 60000, replace=False)
def bits(x): return ((x[:, None] >> np.arange(16, dtype=np.uint16)) & 1).astype(np.uint8)
XW = int(sys.argv[2]) if len(sys.argv) > 2 else 2
def xfeat(r): return np.concatenate([np.ones((len(r), 1), np.uint8)] + [bits(c[r - w]) for w in range(XW)] + [bits(p[r - w]) for w in range(XW)], 1)
Xtr, Xte = xfeat(tr), xfeat(te)
def run(R, label):
    """R: (N, n) uint8 register trajectory."""
    Rf, Rt = R[tr], R[te]
    Ff = np.concatenate([Xtr, (Rf[:, :, None] & Xtr[:, None, :]).reshape(len(tr), -1)], 1)
    Ft = np.concatenate([Xte, (Rt[:, :, None] & Xte[:, None, :]).reshape(len(te), -1)], 1)
    res, acc, base = heldfit.fit_test(Ff, e[tr], Ft, e[te], "rt")
    pr = heldfit.predict(Ft, res['solution']); en = EN[te]
    print(f"  {label:44s} n={R.shape[1]:2d} cols {Ff.shape[1]:5d} inconsistent {res['bad']:5d}/{len(tr)-res['rank']:5d}  held-out {acc:.4f} (base {base:.4f})  enabled {(pr[en]==e[te][en]).mean():.4f} (base {1-e[te][en].mean():.4f})", flush=True)
def enable_bits(M): return ((p & M) != 0).astype(np.uint8)
def shifted(v, d0, d1): return np.stack([np.concatenate([np.zeros(dd, np.uint8), v[:N - dd]]) for dd in range(d0, d1 + 1)], 1)
def since_last(v, cap=63):
    out = np.zeros(N, np.int64); last = -10**6
    for k in range(N):
        if v[k]: last = k
        out[k] = min(k - last, cap)
    return out
def binbits(x, nb): return ((x[:, None] >> np.arange(nb)) & 1).astype(np.uint8)
for M in (0xc980, 0x0800, 0x4800, 0xc800):
    E = enable_bits(M)
    run(shifted(E, 4, 19), f"OR-enable shift reg 0x{M:04x}, depths 4..19")
    s = since_last(E); s4 = np.concatenate([np.full(4, 63), s[:N - 4]])
    run(binbits(s4, 6), f"words since enable 0x{M:04x} (binary, from depth 4)")
    oh = np.stack([(s4 == k) for k in range(16)] + [(s4 >= 16)], 1).astype(np.uint8); run(oh, f"words since enable 0x{M:04x} (one-hot 0..16)")
# saturating charge counter: C += k on enabling word at depth 4, C -= 1 per word, clamp [0, 63]
for M, k_, cap in ((0xc980, 8, 63), (0x0800, 16, 63), (0xc980, 4, 31)):
    E = enable_bits(M); C = np.zeros(N, np.int64); cc = 0
    for q in range(N):
        cc = max(cc - 1, 0)
        if q >= 4 and E[q - 4]: cc = min(cc + k_, cap)
        C[q] = cc
    run(binbits(C, 6), f"charge counter 0x{M:04x} +{k_}/-1 cap {cap}")
