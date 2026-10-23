import sys, numpy as np
from ns10 import load_dump
import gf2, heldfit
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
PW, MW, MIN = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
idx = np.arange(100, N - 3); idx = idx[G[idx]]
key = np.zeros(len(idx), np.uint64)
for w in range(PW): key = key * np.uint64(0x9E3779B97F4A7C15) + p[idx - w].astype(np.uint64) + np.uint64(w)
order = np.argsort(key, kind='stable'); ks = key[order]; rows = idx[order]
bounds = np.concatenate([[0], np.where(ks[1:] != ks[:-1])[0] + 1, [len(ks)]]); sizes = np.diff(bounds)
def bits(x): return ((x[:, None] >> np.arange(16, dtype=np.uint16)) & 1).astype(np.uint8)
tot_t = tot_ok = tot_base = 0; out = []
for g in np.where(sizes >= MIN)[0]:
    r = rows[bounds[g]:bounds[g + 1]]
    if sum(1 for w in range(PW) if p[r[0] - w] != 0) < PW // 2: continue
    y = e[r]
    if y.mean() in (0.0, 1.0): continue
    X = np.concatenate([np.ones((len(r), 1), np.uint8), bits(c[r + 1])] + [bits(m[r - w]) for w in range(MW)] + [np.stack([t[r - k] for k in range(1, 5)], 1).astype(np.uint8)], 1)
    rng = np.random.default_rng(int(g)); perm = rng.permutation(len(r)); nf = int(len(r) * 0.7)
    f, te = perm[:nf], perm[nf:]
    res, acc, base = heldfit.fit_test(X[f], y[f], X[te], y[te], "gl")
    out.append((len(r), y.mean(), res['rank'], res['bad'], acc, base)); tot_t += len(te); tot_ok += acc * len(te); tot_base += base * len(te)
print(f"plain window {PW} words, mask words {MW}, groups >= {MIN}: {len(out)} mixed dense groups")
for o in sorted(out, key=lambda o: -o[0])[:15]: print(f"   size {o[0]:5d} rate {o[1]:.2f} rank {o[2]:3d} fit-inconsistent {o[3]:4d}  held-out acc {o[4]:.3f} (base {o[5]:.3f})")
if tot_t: print(f"pooled held-out acc {tot_ok/tot_t:.4f} vs base {tot_base/tot_t:.4f}")
