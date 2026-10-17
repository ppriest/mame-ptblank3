import sys, numpy as np
from ns10 import load_dump
import features, gf2
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("nl_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
lo, hi = 41, N - 3; idx = np.arange(lo, hi); ee = e[idx]
a0 = ((m[idx - 2] >> 0) & 1) == 1; b0 = ((m[idx - 3] >> 0) & 1) == 1; sub = a0 | ~b0
E = idx[sub & (ee == 1)]; A_ = idx[sub]
W = 7
def state(ix):
    cl = [np.ones((len(ix), 1), np.uint8)]
    for w in range(-1, W): cl += [features.bits16(c[ix - w]), features.bits16(p[ix - w])]
    cl += [t[ix - k][:, None].astype(np.uint8) for k in range(1, 13)]
    return np.concatenate(cl, 1)
names = ['1'] + [f"{nm}[i-{w}].{bb}" for w in range(-1, W) for nm in ("c", "p") for bb in range(16)] + [f"t-{k}" for k in range(1, 13)]
Xe = state(E[::max(1, len(E)//60000)]); Xa = state(A_[::20])
re_, pe = gf2.rank_and_pivots(Xe); ra, pa = gf2.rank_and_pivots(Xa); print("ranks: events", re_, "all(subset)", ra)
nonpiv = [k for k in range(Xe.shape[1]) if k not in pe]
rels = []
for k in nonpiv:
    res = gf2.solve(Xe[:, pe], Xe[:, k], "rel"); assert res['bad'] == 0
    v = np.zeros(Xe.shape[1], np.uint8); v[k] = 1
    for q in res['solution']: v[pe[q]] ^= 1
    rels.append(v)
R = np.array(rels, np.uint8)
val = (Xa.astype(np.int64) @ R.T.astype(np.int64)) % 2; holds_all = (val == 0).all(0)
Rall = R[holds_all]; cand = R[~holds_all]; print("relations on events:", len(R), "hold on all:", holds_all.sum(), "candidates:", len(cand))
def reduce_sparse(v, Rall, rounds=4):
    best = v.copy()
    for _ in range(rounds):
        improved = False
        for u in Rall:
            w = best ^ u
            if w.sum() < best.sum(): best = w; improved = True
        if not improved: break
    return best
for v in cand:
    vs = reduce_sparse(v, Rall)
    fa = float((((Xa.astype(np.int64) @ vs.astype(np.int64)) % 2) == 0).mean())
    print("G-form (==0 on all events):", [names[k] for k in np.where(vs)[0]], f"| fraction ==0 on all subset states: {fa:.3f}")
    np.save("G_form.npy", vs)
# express G in terms of masks too: substitute p = c ^ m where helpful -- try reducing with m-bits: build alternative state with c,m and see sparsity
def state_cm(ix):
    cl = [np.ones((len(ix), 1), np.uint8)]
    for w in range(-1, W): cl += [features.bits16(c[ix - w]), features.bits16(m[ix - w])]
    cl += [t[ix - k][:, None].astype(np.uint8) for k in range(1, 13)]
    return np.concatenate(cl, 1)
names_cm = ['1'] + [f"{nm}[i-{w}].{bb}" for w in range(-1, W) for nm in ("c", "m") for bb in range(16)] + [f"t-{k}" for k in range(1, 13)]
Xe2 = state_cm(E[::max(1, len(E)//60000)]); Xa2 = state_cm(A_[::20])
re2, pe2 = gf2.rank_and_pivots(Xe2)
nonpiv = [k for k in range(Xe2.shape[1]) if k not in pe2]; rels = []
for k in nonpiv:
    res = gf2.solve(Xe2[:, pe2], Xe2[:, k], "rel"); v = np.zeros(Xe2.shape[1], np.uint8); v[k] = 1
    for q in res['solution']: v[pe2[q]] ^= 1
    rels.append(v)
R2 = np.array(rels, np.uint8); val2 = (Xa2.astype(np.int64) @ R2.T.astype(np.int64)) % 2; h2 = (val2 == 0).all(0)
for v in R2[~h2]:
    vs = reduce_sparse(v, R2[h2]); print("G-form in (c,m) basis:", [names_cm[k] for k in np.where(vs)[0]])
