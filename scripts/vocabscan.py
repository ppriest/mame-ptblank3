import sys, numpy as np, itertools, time
from ns10 import load_dump
import gf2
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
idx_all = np.arange(200, N - 3)
def run(mk, pk, pbits, lags, deg, nrows, label):
    ok = G[idx_all].copy()
    for k in lags: ok &= G[idx_all - k]
    rows = idx_all[ok]; sub = rows[::max(1, len(rows) // nrows)][:nrows]
    cols = []
    for k in mk: cols.append(((m[sub - k] >> 0) & 1).astype(np.uint8))
    for k in pk:
        for bb in pbits: cols.append(((p[sub - k] >> bb) & 1).astype(np.uint8))
    for k in lags: cols.append(e[sub - k].astype(np.uint8))
    V = np.stack(cols, 1); n = V.shape[1]; y = e[sub]
    mono = [np.ones(len(sub), np.uint8)] + [V[:, k] for k in range(n)]
    for dg in range(2, deg + 1):
        for comb in itertools.combinations(range(n), dg):
            v = V[:, comb[0]].copy()
            for q in comb[1:]: v &= V[:, q]
            mono.append(v)
    X = np.stack(mono, 1); t0 = time.time(); res = gf2.solve(X, y, "vs")
    free = len(sub) - res['rank']
    print(f"  {label:48s} bits {n:3d} deg {deg} monos {X.shape[1]:6d} rows {len(sub):6d} free {free:6d} inconsistent {res['bad']:5d} ({res['bad']/max(free,1):.4f}) {time.time()-t0:.0f}s", flush=True)
    return res, V, mono
run(range(0, 10), range(3, 10), (11, 14, 15), range(1, 9), 3, 40000, "baseline: m.0[0..9] p[3..9].{11,14,15} h[1..8]")
run(range(0, 10), range(3, 10), (11, 14, 15), range(1, 5), 3, 60000, "h-lags 1..4 only")
run(range(0, 10), range(3, 10), (11, 14, 15), range(1, 7), 3, 60000, "h-lags 1..6")
run(range(0, 12), range(2, 11), (11, 14, 15), range(1, 9), 3, 40000, "extended depths m[0..11] p[2..10]")
run(range(0, 10), range(3, 10), (11,), range(1, 9), 3, 40000, "p.11 only")
run(range(0, 10), range(3, 10), (11, 15), range(1, 9), 3, 40000, "p.11,15")
run(range(2, 8), range(4, 8), (11, 14, 15), range(1, 7), 3, 60000, "reduced m[2..7] p[4..7] h[1..6]")
run(range(2, 8), range(4, 8), (11, 14, 15), range(1, 7), 4, 60000, "reduced set, degree 4")
run(range(0, 10), range(3, 10), (11, 14, 15), range(1, 11), 3, 30000, "h-lags 1..10")
