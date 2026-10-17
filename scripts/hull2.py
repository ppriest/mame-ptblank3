import sys, numpy as np
from ns10 import load_dump
import gf2, features
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("nl_events.npy"); ev = ev[ev > 40]
allidx = np.arange(41, N - 1)
W = 7
def state(idx):
    cols = [np.ones((len(idx), 1), np.uint8)]
    for k in range(W): cols += [features.bits16(c[idx - k]), features.bits16(p[idx - k])]
    cols += [t[idx - k][:, None].astype(np.uint8) for k in range(1, 13)]
    cols += [features.bits16(c[idx + 1])]
    return np.concatenate(cols, 1)
names = ['1'] + [f"{w}{k}.{bb}" for k in range(W) for w in ("c", "p") for bb in range(16)] + [f"t-{k}" for k in range(1, 13)] + [f"cn.{bb}" for bb in range(16)]
Xe = state(ev[::max(1, len(ev)//60000)]); Xa = state(allidx[::25])
re_, pe = gf2.rank_and_pivots(Xe); ra, pa = gf2.rank_and_pivots(Xa)
print(f"events {len(ev)} (rate {len(ev)/len(allidx):.4f}); hull rank: events {re_}, all {ra}  (cols {Xe.shape[1]})")
nonpiv = [k for k in range(Xe.shape[1]) if k not in pe]
rels = []
for k in nonpiv:
    res = gf2.solve(Xe[:, pe], Xe[:, k], "rel"); assert res['bad'] == 0
    v = np.zeros(Xe.shape[1], np.uint8); v[k] = 1
    for q in res['solution']: v[pe[q]] ^= 1
    rels.append(v)
R = np.array(rels, np.uint8)
val = (Xa.astype(np.int64) @ R.T.astype(np.int64)) % 2
holds_all = (val == 0).all(0)
cand = R[~holds_all]
print(f"relations on events: {len(R)}; hold on all data: {holds_all.sum()}; candidate factor forms (=0 on all events, not on all data): {len(cand)}")
# reduce candidates modulo relations that hold on all data, to get sparse representatives: row-reduce [cand] against [R_all]
Rall = R[holds_all]
def reduce_sparse(v, Rall):
    # greedy: try XOR with each all-data relation if it reduces weight
    best = v.copy(); improved = True
    while improved:
        improved = False
        for u in Rall:
            w = best ^ u
            if w.sum() < best.sum(): best = w; improved = True
    return best
for v in cand:
    vs = reduce_sparse(v, Rall)
    frac_all = float((((Xa.astype(np.int64) @ vs.astype(np.int64)) % 2) == 0).mean())
    print("  factor form (==0 on events):", [names[k] for k in np.where(vs)[0]], f" | fraction ==0 on all data: {frac_all:.3f}")
np.save("factor_forms.npy", cand)
