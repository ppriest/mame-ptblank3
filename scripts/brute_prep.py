"""Per-word input for brute.c: segment S0..S0+T+H as records (u = p[i-4], x = m[i], e, gate), 6 bytes each.
Args: dump S0 T H out [planted n mask seed]. Planted: e replaced by the template with a, c inside mask;
env XSRC picks the output word (default m0 = m[i]); prints the planted candidate index in brute.c's layout (a | c | l | d | f | b | init, low to high)."""
import sys, numpy as np
from ns10 import load_dump
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); G = ~(((m[np.maximum(i - 2, 0)] & 1) == 0) & ((m[np.maximum(i - 3, 0)] & 1) == 1))
S0, T, H = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); L = T + H; sl = slice(S0, S0 + L)
XS = __import__("os").environ.get("XSRC", "m0")              # output inputs: word (m, c, p) at depth k, e.g. m2
xd = int(XS[1:]); x = {"m": m, "c": c, "p": p}[XS[0]][S0 - xd:S0 - xd + L].astype(np.uint16)
u = p[S0 - 4:S0 - 4 + L].astype(np.uint16); ee = e[sl].copy(); g = G[sl].astype(np.uint8)
if len(sys.argv) > 6 and sys.argv[6] == "planted":
    n, mask, seed = int(sys.argv[7]), int(sys.argv[8], 0), int(sys.argv[9])
    mb = [k for k in range(16) if mask >> k & 1]; nm = len(mb); rg = np.random.default_rng(seed)
    a, cc = int(rg.integers(0, 1 << nm)), int(rg.integers(0, 1 << nm))
    l, dd, f, bb, R = (int(rg.integers(0, 1 << n)) for _ in range(5)); W = [int(rg.integers(0, 1 << 16)) for _ in range(n)]
    idx = a | cc << nm | l << 2 * nm | dd << (2 * nm + n) | f << (2 * nm + 2 * n) | bb << (2 * nm + 3 * n) | R << (2 * nm + 4 * n)
    par = lambda v: bin(v).count("1") & 1
    comp = lambda v: sum(((v >> mb[j]) & 1) << j for j in range(nm))
    for q in range(L):
        uc = comp(int(u[q])); xq = int(x[q])
        ee[q] = sum(((R >> k) & 1) & par(W[k] & xq) for k in range(n)) & 1
        nb = par(a & uc) ^ par(l & R) ^ (par(dd & R) & par(f & R)) ^ (par(bb & R) & par(cc & uc))
        R = ((R << 1) | nb) & ((1 << n) - 1)
    print(f"planted n={n} mask={mask:#06x} index {idx} of {1 << (2 * nm + 5 * n)}; W {[hex(w) for w in W]}")
rec = np.zeros(L, dtype=[("u", "<u2"), ("x", "<u2"), ("e", "u1"), ("g", "u1")])
rec["u"], rec["x"], rec["e"], rec["g"] = u, x, ee, g
with open(sys.argv[5], "wb") as fo: np.array([T, H], "<i4").tofile(fo); rec.tofile(fo)
print(f"wrote {sys.argv[5]}: T={T} H={H}, {int(g[:T].sum())} gated in fit, e rate on gated {ee[:T][g[:T] == 1].mean():.3f}")
