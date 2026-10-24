"""z3 template: 6-bit NLFSR fed by plaintext at depth 4; output = XOR_k R[k] & parity(W_k . x), x = masks of words i, i-1.
Fit on a segment, then simulate on the following segment."""
import sys, time, numpy as np, z3
from ns10 import load_dump
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a_ = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a_ == 0) & (bq == 1))
n = int(sys.argv[5]) if len(sys.argv) > 5 else 6; S0, T = int(sys.argv[2]), int(sys.argv[3]); TL = int(sys.argv[4])
PL = len(sys.argv) > 6 and sys.argv[6] in ("planted", "pinreal")
if PL:
    rg = np.random.default_rng(7); pa = rg.integers(0, 2, 16); pb = rg.integers(0, 2, n); pc = rg.integers(0, 2, 16); pW = rg.integers(0, 2, (n, int(__import__('os').environ.get('NX', '32')))); pl = rg.integers(0, 2, n); pd = rg.integers(0, 2, n); pf = rg.integers(0, 2, n); R = list(rg.integers(0, 2, n))
    e_real = e; e = e.copy()
    for i_ in range(S0, S0 + T + 20000):
        u = p[i_ - 4]; x = [(int(m[i_]) >> k) & 1 for k in range(16)] + [(int(m[i_ - 1]) >> k) & 1 for k in range(16)]
        e[i_] = sum(R[k] & (sum(pW[k][j] & x[j] for j in range(len(pW[k]))) & 1) for k in range(n)) & 1
        fbv = sum(pb[k] & R[k] for k in range(n)) & 1; ub = [(int(u) >> k) & 1 for k in range(16)]
        par = lambda v: sum(v[k] & R[k] for k in range(n)) & 1
        R = [(sum(pa[k] & ub[k] for k in range(16)) & 1) ^ par(pl) ^ (par(pd) & par(pf)) ^ (fbv & (sum(pc[k] & ub[k] for k in range(16)) & 1))] + R[:-1]
    print("planted instance generated")
    if sys.argv[6] == "pinreal": e = e_real; print("using REAL data with planted params pinned")
def bitsv(x, k): return (int(x) >> k) & 1
def xor_all(terms):
    terms = [q for q in terms if q is not False]
    if not terms: return z3.BoolVal(False)
    acc = terms[0]
    for q in terms[1:]: acc = z3.Xor(acc, q)
    return acc
Lv = [z3.Bool(f"l{k}") for k in range(n)]; Dv = [z3.Bool(f"dd{k}") for k in range(n)]; Fv = [z3.Bool(f"ff{k}") for k in range(n)]
A = [z3.Bool(f"a{k}") for k in range(16)]; Bv = [z3.Bool(f"b{k}") for k in range(n)]; Cv = [z3.Bool(f"c{k}") for k in range(16)]
NX = int(__import__("os").environ.get("NX", "32"))
W = [[z3.Bool(f"w{k}_{j}") for j in range(NX)] for k in range(n)]
r = [z3.Bool(f"r{q}") for q in range(T + n)]          # r[q] = new bit entering at step S0+q-n ; R_i = (r[i], r[i-1], ..., r[i-n+1])
s = z3.Solver(); s.set("timeout", TL * 1000)
for q in range(T):
    i_ = S0 + q; u = p[i_ - 4]
    Rq = [r[q + n - 1 - k] for k in range(n)]           # R at step i_: R[0] newest
    lin_u = xor_all([A[k] for k in range(16) if bitsv(u, k)])
    fb = xor_all([z3.And(Bv[k], Rq[k]) for k in range(n)])
    cu = xor_all([Cv[k] for k in range(16) if bitsv(u, k)])
    lf = xor_all([z3.And(Lv[k], Rq[k]) for k in range(n)]); dR = xor_all([z3.And(Dv[k], Rq[k]) for k in range(n)]); fR = xor_all([z3.And(Fv[k], Rq[k]) for k in range(n)])
    s.add(r[q + n] == xor_all([lin_u, lf, z3.And(dR, fR), z3.And(fb, cu)]))    # next new bit
    if G[i_]:
        x = [bitsv(m[i_], k) for k in range(16)] + [bitsv(m[i_ - 1], k) for k in range(16)]
        out = xor_all([z3.And(Rq[k], xor_all([W[k][j] for j in range(NX) if x[j]])) for k in range(n)])
        s.add(out == bool(e[i_]))
if len(sys.argv) > 7 and sys.argv[7] == "pin":
    for k in range(16): s.add(A[k] == bool(pa[k])); s.add(Cv[k] == bool(pc[k]))
    for k in range(n): s.add(Bv[k] == bool(pb[k])); s.add(Lv[k] == bool(pl[k])); s.add(Dv[k] == bool(pd[k])); s.add(Fv[k] == bool(pf[k]))
    for k in range(n):
        for j in range(NX): s.add(W[k][j] == bool(pW[k][j]))
t0 = time.time(); res = s.check(); print(f"segment {S0}..{S0+T}: z3 {res} in {time.time()-t0:.0f}s", flush=True)
if res == z3.sat:
    mod = s.model(); ev_ = lambda v: z3.is_true(mod.eval(v, model_completion=True))
    a = [ev_(v) for v in A]; bb = [ev_(v) for v in Bv]; ll = [ev_(v) for v in Lv]; dd = [ev_(v) for v in Dv]; fff = [ev_(v) for v in Fv]; cc = [ev_(v) for v in Cv]; Wm = [[ev_(v) for v in row] for row in W]
    R = [ev_(r[T + n - 1 - k]) for k in range(n)]; ok = tot = 0
    for i_ in range(S0 + T, S0 + T + 20000):
        u = p[i_ - 4]
        if G[i_]:
            x = [bitsv(m[i_], k) for k in range(16)] + [bitsv(m[i_ - 1], k) for k in range(16)]
            out = 0
            for k in range(n): out ^= R[k] & (sum(Wm[k][j] & x[j] for j in range(NX)) & 1)
            ok += out == e[i_]; tot += 1
        fbv = sum(bb[k] & R[k] for k in range(n)) & 1
        pr_ = lambda v: sum(v[k] & R[k] for k in range(n)) & 1
        nb = (sum(a[k] & bitsv(u, k) for k in range(16)) & 1) ^ pr_(ll) ^ (pr_(dd) & pr_(fff)) ^ (fbv & (sum(cc[k] & bitsv(u, k) for k in range(16)) & 1))
        R = [bool(nb)] + R[:-1]
    print(f"   held-out next 20000 words: acc {ok/tot:.4f} (base {1-e[S0+T:S0+T+20000][G[S0+T:S0+T+20000]].mean():.4f})")
