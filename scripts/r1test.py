import sys, numpy as np
from ns10 import load_dump
import features, gf2, heldfit
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
idx = np.arange(200, N - 3)
W = int(sys.argv[2]) if len(sys.argv) > 2 else 6
def S_of(ix):
    cl = [np.ones((len(ix), 1), np.uint8)]
    for w in range(W): cl += [features.bits16(c[ix - w]), features.bits16(p[ix - w])]
    cl += [features.bits16(c[ix + 1])]
    return np.concatenate(cl, 1)
for label, cond in (("gate_i,gate_i-1 open, h_i-1=0", G[idx] & G[idx - 1] & (e[idx - 1] == 0)),
                    ("gate_i,gate_i-1 open, h_i-1=1", G[idx] & G[idx - 1] & (e[idx - 1] == 1)),
                    ("all gate-open rows", G[idx])):
    rows = idx[cond]; rng = np.random.default_rng(1); rng.shuffle(rows)
    fit, test = rows[:14000], rows[14000:44000]
    Sf = S_of(fit); r, piv = gf2.rank_and_pivots(Sf); Sf = Sf[:, piv]; St = S_of(test)[:, piv]
    res, acc, base = heldfit.fit_test(Sf, e[fit], St, e[test], "r1l")
    print(f"[{label}] rows {len(rows)}; linear: held-out acc {acc:.3f} (predict-0 {base:.3f})", flush=True)
    fit = rows[:16000]; Qf, qn = gf2.quad_features(S_of(fit)[:, piv], list(range(len(piv)))); Qt, _ = gf2.quad_features(St, list(range(len(piv))))
    res, acc, base = heldfit.fit_test(Qf, e[fit], Qt, e[test], "r1q")
    print(f"     quadratic ({Qf.shape[1]} monos, fit rows {len(fit)}, inconsistent {res['bad']}): held-out acc {acc:.3f} (predict-0 {base:.3f})", flush=True)
