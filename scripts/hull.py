import sys, numpy as np
from ns10 import load_dump
import gf2, features
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
def par(x, mask): return (np.bitwise_count((x & mask).astype(np.uint16)) & 1).astype(np.uint8)
LC0 = (1<<0)|(1<<2)|(1<<3)|(1<<7)|(1<<8)|(1<<9)|(1<<10); LC1 = (1<<0)|(1<<2)
r = t.copy(); r[2:] ^= par(c[2:], LC0) ^ par(c[1:-1], LC1) ^ t[:-2]
z = (p == 0); before = np.zeros(N, np.int64)
for i in range(1, N): before[i] = before[i-1] + 1 if z[i] else 0
W = int(sys.argv[2]) if len(sys.argv) > 2 else 6
def state(idx):  # const + masks of W words + t lags 1..6
    cols = [np.ones((len(idx), 1), np.uint8)] + [features.bits16(m[idx - k]) for k in range(W)] + [t[idx - k][:, None].astype(np.uint8) for k in range(1, 7)]
    return np.concatenate(cols, 1)
names = ['1'] + [f"m{k}.{bb}" for k in range(W) for bb in range(16)] + [f"t-{k}" for k in range(1, 7)]
for need in (6, 8, 10):
    ev = np.where((before >= need) & (r == 1) & (np.arange(N) > 40))[0]
    allz = np.where((before >= need) & (np.arange(N) > 40))[0]
    Xe = state(ev); Xa = state(allz[::max(1, len(allz)//60000)])
    re_, pe = gf2.rank_and_pivots(Xe); ra, pa = gf2.rank_and_pivots(Xa)
    print(f"zeros-before >= {need}: events {len(ev)} (rate {len(ev)/len(allz):.5f}); hull rank: events {re_}, all {ra}  (cols {Xe.shape[1]})")
    if re_ < ra:
        # forms constant on events: nullspace of events matrix restricted to... compute relations: columns dependent on pivots
        nonpiv = [k for k in range(Xe.shape[1]) if k not in pe]
        rels = []
        for k in nonpiv:
            res = gf2.solve(Xe[:, pe], Xe[:, k], "rel"); assert res['bad'] == 0
            v = np.zeros(Xe.shape[1], np.uint8); v[k] = 1
            for q in res['solution']: v[pe[q]] ^= 1
            rels.append(v)
        R = np.array(rels, np.uint8)
        # which relations also hold on all steady-state states? those are V-relations; the others are candidate factors
        val = (Xa.astype(np.int64) @ R.T.astype(np.int64)) % 2
        holds_all = (val == 0).all(0)
        cand = R[~holds_all]
        print(f"   relations on events: {len(R)}; also hold on all deep-zero states: {holds_all.sum()}; candidate factor forms: {len(cand)}")
        # reduce candidate forms modulo V-relations: show sparse representatives (rank of cand modulo the V relations)
        Rv = R[holds_all]
        for v in cand[:10]:
            print("     form (=0 on all events):", [names[k] for k in np.where(v)[0]], " fraction =0 on all deep-zero:", float((((Xa.astype(np.int64) @ v.astype(np.int64)) % 2) == 0).mean()))
