import sys, time, numpy as np
from ns10 import load_dump
import gf2, heldfit
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
PBITS = [int(x) for x in sys.argv[2].split(",")]; D0, D1 = (int(x) for x in sys.argv[3].split(",")); SW = int(sys.argv[4]); SKIND = sys.argv[5]; NFIT = int(sys.argv[6])
EN = np.zeros(N, bool)
for w in range(4, 97): EN[100:] |= (p[100 - w:N - w] & 0xc980) != 0
idx = np.arange(max(D1, 100) + 2, N - 3); idx = idx[G[idx]]
split = int(0.7 * N); tr = idx[idx < split]; te = idx[idx >= split + D1]
rng = np.random.default_rng(0); tr = rng.choice(tr, min(NFIT, len(tr)), replace=False); te = rng.choice(te, min(60000, len(te)), replace=False)
def bits(x): return ((x[:, None] >> np.arange(16, dtype=np.uint16)) & 1).astype(np.uint8)
def X_of(r):
    S = [np.ones((len(r), 1), np.uint8)]
    for w in range(SW):
        if "c" in SKIND: S.append(bits(c[r - w]))
        if "p" in SKIND: S.append(bits(p[r - w]))
    S = np.concatenate(S, 1)
    P = np.stack([((p[r - dd] >> bb) & 1).astype(np.uint8) for dd in range(D0, D1 + 1) for bb in PBITS], 1)
    prods = (P[:, :, None] & S[:, None, :]).reshape(len(r), -1)
    return np.concatenate([S, prods], 1)
t0 = time.time(); Xf = X_of(tr); Xt = X_of(te)
print(f"pbits {PBITS} depths {D0}..{D1}, state {SKIND} x {SW} words: {Xf.shape[1]} columns, fit rows {len(tr)}, test rows {len(te)} (build {time.time()-t0:.0f}s)", flush=True)
t0 = time.time(); res, acc, base = heldfit.fit_test(Xf, e[tr], Xt, e[te], "bl")
pr = heldfit.predict(Xt, res['solution']); en = EN[te]
print(f"   fit rank {res['rank']} inconsistent {res['bad']} of {len(tr)-res['rank']} ({time.time()-t0:.0f}s); held-out acc {acc:.4f} (base {base:.4f}); enabled rows acc {(pr[en]==e[te][en]).mean():.4f} (base {1-e[te][en].mean():.4f})", flush=True)
