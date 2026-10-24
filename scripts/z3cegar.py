"""Incremental z3 on the z3nlfsr2 template: register transitions are asserted over the whole segment, output
constraints only on a growing set of words. Each candidate model is simulated over the segment; words it gets wrong
are added and the solver re-checks (learned clauses kept). UNSAT on any subset is UNSAT on the segment.
Args: dump S0 T TIMEOUT n seed [planted]. Env NX (16: m[i]; 32: m[i], m[i-1])."""
import os, sys, time, numpy as np, z3
from ns10 import load_dump
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); G = ~((((m[np.maximum(i - 2, 0)]) & 1) == 0) & ((m[np.maximum(i - 3, 0)] & 1) == 1))
S0, T, TL, n, seed = (int(v) for v in sys.argv[2:7]); NX = int(os.environ.get("NX", "16"))
PL = len(sys.argv) > 7 and sys.argv[7] == "planted"
X = [[(int(m[k]) >> j) & 1 for j in range(16)] + [(int(m[k - 1]) >> j) & 1 for j in range(16)] for k in range(S0, S0 + T + 20000)]
U = [[(int(p[k - 4]) >> j) & 1 for j in range(16)] for k in range(S0, S0 + T + 20000)]

def sim(P, R, q0, q1):
    """P = (a, b, c, l, d, f, W); R[0] newest. Returns outputs for steps q0..q1-1 (relative to S0)."""
    a, bb, cc, ll, dd, ff, Wm = P; out = []; R = list(R)
    par = lambda v, x, k: sum(v[j] & x[j] for j in range(k)) & 1
    for q in range(q0, q1):
        x = X[q]; u = U[q]
        out.append(sum(R[k] & par(Wm[k], x, NX) for k in range(n)) & 1)
        nb = par(a, u, 16) ^ par(ll, R, n) ^ (par(dd, R, n) & par(ff, R, n)) ^ (par(bb, R, n) & par(cc, u, 16))
        R = [nb] + R[:-1]
    return out, R

if PL:
    rg = np.random.default_rng(7)
    P0 = (rg.integers(0, 2, 16), rg.integers(0, 2, n), rg.integers(0, 2, 16), rg.integers(0, 2, n),
          rg.integers(0, 2, n), rg.integers(0, 2, n), rg.integers(0, 2, (n, NX)))
    o, _ = sim(P0, list(rg.integers(0, 2, n)), 0, T + 20000); e = e.copy(); e[S0:S0 + T + 20000] = o
    print("planted instance generated", flush=True)

def xor_all(terms):
    terms = [q for q in terms if q is not False]
    if not terms: return z3.BoolVal(False)
    acc = terms[0]
    for q in terms[1:]: acc = z3.Xor(acc, q)
    return acc
A = [z3.Bool(f"a{k}") for k in range(16)]; Bv = [z3.Bool(f"b{k}") for k in range(n)]; Cv = [z3.Bool(f"c{k}") for k in range(16)]
Lv = [z3.Bool(f"l{k}") for k in range(n)]; Dv = [z3.Bool(f"dd{k}") for k in range(n)]; Fv = [z3.Bool(f"ff{k}") for k in range(n)]
W = [[z3.Bool(f"w{k}_{j}") for j in range(NX)] for k in range(n)]
r = [z3.Bool(f"r{q}") for q in range(T + n)]          # R at step q = (r[q+n-1], ..., r[q])
z3.set_param("sat.random_seed", seed); z3.set_param("smt.random_seed", seed)
s = z3.Solver()
lin = lambda vs, x: xor_all([vs[k] for k in range(len(x)) if x[k]])
lr = lambda vs, Rq: xor_all([z3.And(vs[k], Rq[k]) for k in range(n)])
for q in range(T):
    Rq = [r[q + n - 1 - k] for k in range(n)]; u = U[q]
    s.add(r[q + n] == xor_all([lin(A, u), lr(Lv, Rq), z3.And(lr(Dv, Rq), lr(Fv, Rq)), z3.And(lr(Bv, Rq), lin(Cv, u))]))
def add_out(q):
    Rq = [r[q + n - 1 - k] for k in range(n)]; x = X[q][:NX]
    s.add(xor_all([z3.And(Rq[k], lin(W[k], x)) for k in range(n)]) == bool(e[S0 + q]))
gate = [q for q in range(T) if G[S0 + q]]
rng = np.random.default_rng(seed); used = set(rng.choice(gate, min(40, len(gate)), replace=False).tolist())
for q in used: add_out(q)
t0 = time.time(); it = 0
while True:
    left = TL - (time.time() - t0)
    if left <= 0: print(f"n={n} seed={seed}: unknown (budget) after {it} iterations, {len(used)} constraints", flush=True); break
    s.set("timeout", int(left * 1000)); res = s.check(); it += 1
    if res != z3.sat:
        print(f"n={n} seed={seed}: z3 {res} after {it} iterations, {len(used)} of {len(gate)} constraints, {time.time()-t0:.0f}s", flush=True); break
    mod = s.model(); ev_ = lambda v: int(z3.is_true(mod.eval(v, model_completion=True)))
    P = ([ev_(v) for v in A], [ev_(v) for v in Bv], [ev_(v) for v in Cv], [ev_(v) for v in Lv], [ev_(v) for v in Dv],
         [ev_(v) for v in Fv], [[ev_(v) for v in row] for row in W])
    o, Rend = sim(P, [ev_(r[n - 1 - k]) for k in range(n)], 0, T)
    bad = [q for q in gate if o[q] != e[S0 + q] and q not in used]
    if not bad:
        o2, _ = sim(P, Rend, T, T + 20000)
        g2 = [q for q in range(T, T + 20000) if G[S0 + q]]
        acc = np.mean([o2[q - T] == e[S0 + q] for q in g2]); base = 1 - np.mean([e[S0 + q] for q in g2])
        print(f"n={n} seed={seed}: SAT on all {len(gate)} gated words after {it} iterations, {len(used)} constraints, "
              f"{time.time()-t0:.0f}s; held-out next 20000: acc {acc:.4f} (base {base:.4f})", flush=True)
        print("   params", P, flush=True); break
    pick = [bad[k] for k in np.linspace(0, len(bad) - 1, min(8, len(bad))).astype(int)]
    for q in pick: add_out(q); used.add(q)
    if it % 10 == 0: print(f"   n={n} seed={seed} it {it}: {len(bad)} wrong, {len(used)} constraints, {time.time()-t0:.0f}s", flush=True)
