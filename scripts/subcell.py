import sys, numpy as np, time
from ns10 import load_dump
import features, gf2
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("nl_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
lo, hi = 41, N - 3; idx = np.arange(lo, hi)
a = ((m[idx - 2] >> 0) & 1); bq = ((m[idx - 3] >> 0) & 1)
W = 5
def X_of(ix):
    cl = [np.ones((len(ix), 1), np.uint8)]
    for w in range(W): cl += [features.bits16(c[ix - w]), features.bits16(p[ix - w])]
    cl += [t[ix - k][:, None].astype(np.uint8) for k in range(1, 9)]
    return np.concatenate(cl, 1)
xnames = ['1'] + [f"{nm}{w}.{bb}" for w in range(W) for nm in ("c", "p") for bb in range(16)] + [f"t-{k}" for k in range(1, 9)]
def candidates(ix):
    cols, names = [], []
    for w in range(-1, 7):
        for nm, arr in (("c", c), ("p", p), ("m", m)):
            if nm == "p" and w == -1: continue
            cols.append(features.bits16(arr[ix - w])); names += [f"{nm}[i-{w}].{bb}" for bb in range(16)]
    cols.append(np.stack([t[ix - k] for k in range(1, 9)], 1).astype(np.uint8)); names += [f"t-{k}" for k in range(1, 9)]
    cols.append(np.stack([e[ix - k] for k in range(1, 5)], 1).astype(np.uint8)); names += [f"e-{k}" for k in range(1, 5)]
    return np.concatenate(cols, 1), names
NFIT = 16000
for ca, cb in ((1, 0), (0, 0), (1, 1)):
    rows = idx[(a == ca) & (bq == cb)]
    X = X_of(rows); y = t[rows]; C, cn = candidates(rows)
    t0 = time.time(); best = []
    for k in range(C.shape[1]):
        for v in (0, 1):
            sub = np.where(C[:, k] == v)[0]
            if len(sub) < 5000: continue
            res = gf2.solve(X[sub[:NFIT]], y[sub[:NFIT]], "sub")
            free = min(NFIT, len(sub)) - res['rank']
            best.append((res['bad'] / max(free, 1), cn[k], v, res['bad'], free, len(sub)))
            if res['bad'] == 0:
                pred = np.zeros(len(sub), np.uint8)
                for q in res['solution']: pred ^= X[sub][:, q]
                print(f"  cell a={ca} b={cb}: EXACT LINEAR on {cn[k]}=={v}: rows {len(sub)}, mismatches all {int((pred != y[sub]).sum())}; solution {[xnames[q] for q in res['solution']]}", flush=True)
    best.sort()
    print(f"cell a={ca} b={cb}: {len(best)} tests in {time.time()-t0:.0f}s; best 5:", [(round(x[0], 3), x[1], x[2]) for x in best[:5]], flush=True)
