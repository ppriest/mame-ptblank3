"""The z3nlfsr2 template in CryptoMiniSat, parities as native XOR clauses, products as AND gates.
new bit = a.p[i-4] ^ l.R ^ (d.R & f.R) ^ (b.R & c.p[i-4]); output on gated words = XOR_k R[k] & W_k.x.
Args: dump S0 T TIMEOUT n threads [real|planted|pin|pinreal]. Env NX (16: m[i]; 32: m[i], m[i-1]).
pin: planted data, planted parameter groups in env PIN fixed (default all; must be SAT); pinreal: real data, same pin (expected UNSAT)."""
import os, sys, time, numpy as np, pycryptosat
from ns10 import load_dump
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); G = ~(((m[np.maximum(i - 2, 0)] & 1) == 0) & ((m[np.maximum(i - 3, 0)] & 1) == 1))
S0, T, TL, n, TH = (int(v) for v in sys.argv[2:7]); NX = int(os.environ.get("NX", "16"))
mode = sys.argv[7] if len(sys.argv) > 7 else "real"
X = [[(int(m[k]) >> j) & 1 for j in range(16)] + [(int(m[k - 1]) >> j) & 1 for j in range(16)] for k in range(S0, S0 + T + 20000)]
U = [[(int(p[k - 4]) >> j) & 1 for j in range(16)] for k in range(S0, S0 + T + 20000)]

def sim(P, R, q0, q1):
    a, bb, cc, ll, dd, ff, Wm = P; out = []; R = list(R)
    par = lambda v, x, k: sum(v[j] & x[j] for j in range(k)) & 1
    for q in range(q0, q1):
        x = X[q]; u = U[q]
        out.append(sum(R[k] & par(Wm[k], x, NX) for k in range(n)) & 1)
        nb = par(a, u, 16) ^ par(ll, R, n) ^ (par(dd, R, n) & par(ff, R, n)) ^ (par(bb, R, n) & par(cc, u, 16))
        R = [nb] + R[:-1]
    return out, R

if mode != "real":
    rg = np.random.default_rng(7)
    P0 = (rg.integers(0, 2, 16), rg.integers(0, 2, n), rg.integers(0, 2, 16), rg.integers(0, 2, n),
          rg.integers(0, 2, n), rg.integers(0, 2, n), rg.integers(0, 2, (n, NX)))
    if mode != "pinreal":
        o, _ = sim(P0, list(rg.integers(0, 2, n)), 0, T + 20000); e = e.copy(); e[S0:S0 + T + 20000] = o

s = pycryptosat.Solver(threads=TH, time_limit=TL); nv = [0]
def new(): nv[0] += 1; return nv[0]
def AND(x, y):
    if x is None or y is None: return None
    z = new(); s.add_clause([-z, x]); s.add_clause([-z, y]); s.add_clause([z, -x, -y]); return z
def XOR(vs):
    vs = [v for v in vs if v is not None]
    if not vs: return None
    if len(vs) == 1: return vs[0]
    z = new(); s.add_xor_clause(vs + [z], False); return z
vec = lambda k: [new() for _ in range(k)]
A, Bv, Cv, Lv, Dv, Fv = vec(16), vec(n), vec(16), vec(n), vec(n), vec(n); W = [vec(NX) for _ in range(n)]
r = vec(T + n)                                        # R at step q = (r[q+n-1], ..., r[q])
unsat_trivial = False
for q in range(T):
    R = [r[q + n - 1 - k] for k in range(n)]; u = U[q]
    terms = [A[j] for j in range(16) if u[j]] + [AND(Lv[k], R[k]) for k in range(n)]
    terms.append(AND(XOR([AND(Dv[k], R[k]) for k in range(n)]), XOR([AND(Fv[k], R[k]) for k in range(n)])))
    terms.append(AND(XOR([AND(Bv[k], R[k]) for k in range(n)]), XOR([Cv[j] for j in range(16) if u[j]])))
    s.add_xor_clause([r[q + n]] + [v for v in terms if v is not None], False)
    if G[S0 + q]:
        x = X[q][:NX]; o = [AND(R[k], XOR([W[k][j] for j in range(NX) if x[j]])) for k in range(n)]
        o = [v for v in o if v is not None]
        if o: s.add_xor_clause(o, bool(e[S0 + q]))
        elif e[S0 + q]: unsat_trivial = True
if mode in ("pin", "pinreal"):
    pin = os.environ.get("PIN", "abcldfW")             # parameter groups to fix
    for name, vs, vals in zip("abcldf", (A, Bv, Cv, Lv, Dv, Fv), P0[:6]):
        if name in pin:
            for v, x in zip(vs, vals): s.add_clause([v if x else -v])
    if "W" in pin:
        wc = int(os.environ.get("PINWC", NX))          # pin only the first PINWC columns of each W row
        for k in range(n):
            for v, x in list(zip(W[k], P0[6][k]))[:wc]: s.add_clause([v if x else -v])
ng = int(G[S0:S0 + T].sum())
t0 = time.time(); res, sol = (False, None) if unsat_trivial else s.solve()
print(f"n={n} NX={NX} {mode} {S0}..{S0+T} ({ng} gated, {nv[0]} vars): {'SAT' if res else 'UNSAT' if res is False else 'unknown'} "
      f"in {time.time()-t0:.0f}s", flush=True)
if res:
    g = lambda vs: [int(bool(sol[v])) for v in vs]
    P = (g(A), g(Bv), g(Cv), g(Lv), g(Dv), g(Fv), [g(row) for row in W])
    o, Rend = sim(P, [int(bool(sol[r[n - 1 - k]])) for k in range(n)], 0, T)
    fit = np.mean([o[q] == e[S0 + q] for q in range(T) if G[S0 + q]])
    o2, _ = sim(P, Rend, T, T + 20000); g2 = [q for q in range(T, T + 20000) if G[S0 + q]]
    acc = np.mean([o2[q - T] == e[S0 + q] for q in g2]); base = 1 - np.mean([e[S0 + q] for q in g2])
    print(f"   fit {fit:.4f}; held-out next 20000: acc {acc:.4f} (base {base:.4f})", flush=True)
    print("   params", P, flush=True)
