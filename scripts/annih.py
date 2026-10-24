"""Low-degree annihilators of the residual's event set (or non-event set) on enabled, gate-open rows.
usage: annih.py dump VARSET DEG TARGET(ev|non) [NSAMP]"""
import sys, time, itertools, numpy as np
from ns10 import load_dump
import gf2
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
EN = np.zeros(N, bool)
for w in range(4, 97): EN[100:] |= (p[100 - w:N - w] & 0xc980) != 0
VARSET, DEG, TARGET = sys.argv[2], int(sys.argv[3]), sys.argv[4]; NSAMP = int(sys.argv[5]) if len(sys.argv) > 5 else 0
idx = np.arange(120, N - 3); idx = idx[G[idx] & EN[idx]]
def bits(x): return ((x[:, None] >> np.arange(16, dtype=np.uint16)) & 1).astype(np.uint8)
def V_of(r):
    cols, names = [], []
    def add(arr, nm):
        cols.append(arr); names.extend(nm)
    spec = VARSET.split("+")
    for s in spec:
        kind, rng_ = s[0], s[1:]
        if kind in "mcp":
            lo, hi = (int(x) for x in rng_.split("-"))
            arr = {"m": m, "c": c, "p": p}[kind]
            for w in range(lo, hi + 1): add(bits(arr[r - w]), [f"{kind}{w}.{k}" for k in range(16)])
        elif kind == "t":
            lo, hi = (int(x) for x in rng_.split("-"))
            add(np.stack([t[r - k] for k in range(lo, hi + 1)], 1).astype(np.uint8), [f"t-{k}" for k in range(lo, hi + 1)])
        elif kind == "h":   # high plaintext bits 11,14,15 at depths
            lo, hi = (int(x) for x in rng_.split("-"))
            for w in range(lo, hi + 1):
                for bb in (11, 14, 15): add(((p[r - w] >> bb) & 1).astype(np.uint8)[:, None], [f"p{w}.{bb}"])
    return np.concatenate(cols, 1), names
def mono(Vm, deg):
    n = Vm.shape[1]; out = [np.ones((len(Vm), 1), np.uint8), Vm]; nm = [()] + [(k,) for k in range(n)]
    for dg in range(2, deg + 1):
        combs = list(itertools.combinations(range(n), dg))
        blk = np.ones((len(Vm), len(combs)), np.uint8)
        for q in range(dg): blk &= Vm[:, [cb[q] for cb in combs]]
        out.append(blk); nm += combs
    return np.concatenate(out, 1), nm
Vtmp, names = V_of(idx[:10]); nvar = Vtmp.shape[1]
ncols = sum(len(list(itertools.combinations(range(nvar), k))) for k in range(DEG + 1))
NS = NSAMP or int(1.6 * ncols + 4000)
rng = np.random.default_rng(1)
tgt_rows = idx[e[idx] == (1 if TARGET == "ev" else 0)]
S = rng.choice(tgt_rows, min(NS, len(tgt_rows)), replace=False)
A = rng.choice(idx, min(NS, len(idx)), replace=False)
t0 = time.time()
VA, _ = V_of(A); MA, nm = mono(VA, DEG)
rA, pivA = gf2.rank_and_pivots(MA)
VS, _ = V_of(S); MS, _ = mono(VS, DEG); MSp = MS[:, pivA]
rS, pivS = gf2.rank_and_pivots(MSp)
print(f"vars {VARSET} ({nvar}), degree {DEG}, target {TARGET}: {MA.shape[1]} monomials; rank all-rows {rA}, rank {TARGET}-rows {rS} (sample {len(S)} of {len(tgt_rows)}) -> deficiency {rA - rS}  ({time.time()-t0:.0f}s)", flush=True)
if rA > rS:
    nonpiv = [k for k in range(len(pivA)) if k not in set(pivS)][:12]
    def evalf(rows, f):
        out = []
        for q in range(0, len(rows), 20000):
            Vq, _ = V_of(rows[q:q + 20000]); Mq, _ = mono(Vq, DEG); Mq = Mq[:, pivA]
            out.append((Mq[:, f == 1].sum(1) % 2).astype(np.uint8))
        return np.concatenate(out)
    other = idx[e[idx] == (0 if TARGET == "ev" else 1)][:200000]
    for k in nonpiv:
        res = gf2.solve(MSp[:, pivS], MSp[:, k], "an")
        f = np.zeros(len(pivA), np.uint8); f[k] = 1
        for q in res['solution']: f[pivS[q]] ^= 1
        val_t = evalf(tgt_rows, f); val_o = evalf(other, f)
        terms = [nm[pivA[q]] for q in np.where(f)[0]]
        tstr = ' + '.join('*'.join(names[v] for v in tm) if tm else '1' for tm in terms[:24])
        print(f"   annihilator: nonzero on {int(val_t.sum())} of {len(val_t)} {TARGET} rows (must be 0); on other rows nonzero {val_o.mean():.3f}; {len(terms)} terms: {tstr}{' ...' if len(terms) > 24 else ''}", flush=True)
