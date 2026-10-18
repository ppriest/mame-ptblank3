import sys, numpy as np, itertools, time
from ns10 import load_dump
import gf2
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
def vocab(rows, mk, pk, pbits, ek):
    cols, names = [], []
    for k in mk: cols.append(((m[rows - k] >> 0) & 1).astype(np.uint8)); names.append(f"m{k}.0")
    for k in pk:
        for bb in pbits: cols.append(((p[rows - k] >> bb) & 1).astype(np.uint8)); names.append(f"p{k}.{bb}")
    for k in ek: cols.append(e[rows - k].astype(np.uint8)); names.append(f"e{k}")
    return np.stack(cols, 1), names
def monos(V, deg):
    n = V.shape[1]; out = [np.ones(len(V), np.uint8)]; nm = [()]
    for k in range(n): out.append(V[:, k]); nm.append((k,))
    for dg in range(2, deg + 1):
        for comb in itertools.combinations(range(n), dg):
            v = V[:, comb[0]].copy()
            for q in comb[1:]: v &= V[:, q]
            out.append(v); nm.append(comb)
    return np.stack(out, 1), nm
idx = np.arange(200, N - 3); idx = idx[G[idx]]
def run(mk, pk, pbits, ek, deg, nfit, label):
    fit = idx[::max(1, len(idx) // nfit)][:nfit]
    V, names = vocab(fit, mk, pk, pbits, ek); X, nm = monos(V, deg)
    t0 = time.time(); res = gf2.solve(X, e[fit], "fr"); free = len(fit) - res['rank']
    print(f"{label:44s} bits {V.shape[1]} monos {X.shape[1]} rows {len(fit)} free {free} inconsistent {res['bad']} ({res['bad']/max(free,1):.4f}) {time.time()-t0:.0f}s", flush=True)
    return res, nm, names
run(range(0, 14), range(2, 11), (11, 14, 15), range(1, 11), 2, 40000, "deg2 m[0..13] p[2..10] e[1..10]")
run(range(0, 14), range(2, 11), (11, 14, 15), range(1, 11), 3, 50000, "deg3 m[0..13] p[2..10] e[1..10]")
