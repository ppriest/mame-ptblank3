"""Template: FIFO of words latched when an enabling plaintext word passes depth D; h linear in FIFO contents (+ current words)."""
import sys, numpy as np
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
def fifo(M, D, depth, src):
    """F[q, k] = src word latched at the k-th most recent position j <= q-D with p[j] & M != 0 (0 if none)."""
    fire = np.where((p & M) != 0)[0]
    pos = np.searchsorted(fire, np.arange(N) - D, side='right') - 1     # index into fire of the most recent fire <= q-D
    out = np.zeros((N, depth), np.uint16)
    for k in range(depth):
        j = pos - k; ok = j >= 0
        out[ok, k] = src[fire[j[ok]]]
    return out
def run(F, label, XW=2):
    def feats(r):
        cols = [np.ones((len(r), 1), np.uint8)] + [bits(c[r - w]) for w in range(XW)] + [bits(p[r - w]) for w in range(XW)]
        cols += [bits(F[r, k]) for k in range(F.shape[1])]
        return np.concatenate(cols, 1)
    Ff, Ft = feats(tr), feats(te)
    res, acc, base = heldfit.fit_test(Ff, e[tr], Ft, e[te], "ff")
    pr = heldfit.predict(Ft, res['solution']); en = EN[te]
    print(f"  {label:46s} cols {Ff.shape[1]:4d} inconsistent {res['bad']:5d}/{len(tr)-res['rank']:5d} held-out {acc:.4f} (base {base:.4f}) enabled {(pr[en]==e[te][en]).mean():.4f} (base {1-e[te][en].mean():.4f})", flush=True)
for M in (0xc980, 0x0800, 0x4800):
    for srcname, src in (("m", m), ("c", c), ("p", p)):
        for depth in (1, 4, 8):
            run(fifo(M, 4, depth, src), f"enable 0x{M:04x} latch {srcname}, FIFO depth {depth}")

print("--- gated: one-hot(words since enable at depth 4) x FIFO bits ---")
def since(M):
    v = (p & M) != 0; out = np.zeros(N, np.int64); last = -10**6
    for k in range(N):
        if k >= 4 and v[k - 4]: last = k
        out[k] = min(k - last, 40)
    return out
def run_g(F, S, label):
    def feats(r):
        Fb = np.concatenate([np.ones((len(r), 1), np.uint8)] + [bits(F[r, k]) for k in range(F.shape[1])] + [bits(c[r]), bits(p[r])], 1)
        oh = np.stack([(S[r] == k) for k in range(12)] + [(S[r] >= 12) & (S[r] < 40)], 1).astype(np.uint8)
        return (oh[:, :, None] & Fb[:, None, :]).reshape(len(r), -1)
    Ff, Ft = feats(tr), feats(te)
    res, acc, base = heldfit.fit_test(Ff, e[tr], Ft, e[te], "fg")
    pr = heldfit.predict(Ft, res['solution']); en = EN[te]
    print(f"  {label:46s} cols {Ff.shape[1]:4d} inconsistent {res['bad']:5d}/{len(tr)-res['rank']:5d} held-out {acc:.4f} (base {base:.4f}) enabled {(pr[en]==e[te][en]).mean():.4f} (base {1-e[te][en].mean():.4f})", flush=True)
for M in (0xc980, 0x0800):
    S = since(M)
    for srcname, src in (("m", m), ("c", c)):
        for depth in (1, 2, 4):
            run_g(fifo(M, 4, depth, src), S, f"gated enable 0x{M:04x} latch {srcname}, depth {depth}")
