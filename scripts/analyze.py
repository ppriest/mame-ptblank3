import sys, numpy as np, time
from ns10 import load_dump
import gf2, features
typ, b, c, p, cnt = load_dump(sys.argv[1])
d = typ == b'D'
# segment id: increments at each 'I'
seg = np.cumsum(typ == b'I')[d]
c = c[d]; p = p[d]; t = b[d]; cnt = cnt[d]; N = len(c)
pos = np.zeros(N, np.int64)   # position since init
starts = np.where(np.diff(np.concatenate([[-1], seg])) != 0)[0]
for s0, s1 in zip(starts, list(starts[1:]) + [N]): pos[s0:s1] = np.arange(s1 - s0)
print("words", N, "segments", len(starts), "segment lengths", [int(x) for x in np.diff(np.concatenate([starts, [N]]))][:20])
ok = pos >= 4                          # drop the first 4 words of each segment (IV influence)
ok[-1] = False                         # cnext needs i+1
sel = np.where(ok)[0]
m = (c ^ p).astype(np.uint16)

def predict(L, piv, qn, solution):
    pred = np.zeros(L.shape[0], np.uint8)
    for k in solution:
        mono = qn[k]
        if len(mono) == 0: pred ^= 1
        elif len(mono) == 1: pred ^= L[:, piv[mono[0]]]
        else: pred ^= L[:, piv[mono[0]]] & L[:, piv[mono[1]]]
    return pred

def run(w, extra, rows=None, label="", nfit=20000):
    L, names, y = features.window(c, p, t, 0, N, w=w, extra=extra)
    L = L[sel]; y = y[sel]
    r, piv = gf2.rank_and_pivots(L[:nfit])
    rall, _ = gf2.rank_and_pivots(L[::50])
    res = gf2.solve(L[::10], y[::10], "lin")
    print(f"[{label}] w={w} extra={extra}: {L.shape[1]} cols, rank {r} (strided-all {rall}); linear fit inconsistent rows {res['bad']}/{len(y[::10])}")
    Q, qn = gf2.quad_features(L[:nfit], piv)
    nrows = rows or min(nfit, Q.shape[1] + 3000)
    t0 = time.time(); res = gf2.solve(Q[:nrows], y[:nrows], "quad")
    print(f"   quad: {Q.shape[1]} monomials over {len(piv)} pivots, {nrows} rows -> rank {res['rank']}, inconsistent {res['bad']}  ({time.time()-t0:.1f}s)")
    if res['bad'] == 0:
        pred = predict(L, piv, qn, res['solution'])
        mism = int((pred != y).sum())
        print(f"   quad solution: {len(res['solution'])} terms, mismatches on all {len(y)} rows: {mism}")
        if mism == 0:
            print("   TERMS:", [tuple(names[piv[i]] for i in qn[k]) for k in res['solution']])
    return res

# collision / determinism test on raw windows
def collisions(w, extra_next=False):
    key = np.zeros(N, np.uint64)
    for k in range(w):
        key = key * np.uint64(1000003) + c[np.arange(N) - k].astype(np.uint64) + (p[np.arange(N) - k].astype(np.uint64) << np.uint64(16))
    if extra_next: key = key * np.uint64(1000003) + c[np.minimum(np.arange(N) + 1, N - 1)].astype(np.uint64)
    key = key[sel]; y = t[sel]
    order = np.argsort(key, kind='stable'); ks = key[order]; ys = y[order]
    same = ks[1:] == ks[:-1]
    conflict = same & (ys[1:] != ys[:-1])
    print(f"collisions w={w} next={extra_next}: identical-state pairs {int(same.sum())}, with different t: {int(conflict.sum())}")
run(3, ('cnext',), label="B")
run(4, ('cnext',), label="D")
run(4, ('cnext', 'tprev'), label="E")
