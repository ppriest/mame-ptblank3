import sys, numpy as np, time
from ns10 import load_dump
import gf2, features
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c)
P = int(sys.argv[2]) if len(sys.argv) > 2 else 2
NF = 26000; sel = np.arange(40, N - 1)
L, lnames, y = features.window(c, p, t, 0, N, w=4)
tl = np.stack([t[np.arange(N) - k] for k in range(1, 5)], 1).astype(np.uint8); lnames += [f"t-{k}" for k in range(1, 5)]
L = np.concatenate([L, tl], 1)
r, piv = gf2.rank_and_pivots(L[sel][:20000]); S = L[:, piv]; snames = [lnames[q] for q in piv]
bitsC = features.bits16(c); bitsP = features.bits16(p)
def acc_stream(stream, snm, P, exclude_current):
    nb = stream.shape[1]; H = np.zeros((N, nb * P), np.uint8); hn = []
    for rr in range(P):
        acc = np.bitwise_xor.accumulate(stream[rr::P], 0)
        if exclude_current: acc = np.concatenate([np.zeros((1, nb), np.uint8), acc[:-1]], 0)
        H[rr::P, rr*nb:(rr+1)*nb] = acc; hn += [f"acc{P}[{rr}]{'<' if exclude_current else '<='}.{s}" for s in snm]
    for rr in range(P):
        blk = H[:, rr*nb:(rr+1)*nb]
        for shift in range(1, P):
            rows = np.arange(rr + shift, N, P); blk[rows] = blk[rows - shift]
    return H, hn
cpn = [f"c{k}" for k in range(16)] + [f"p{k}" for k in range(16)]
Hcp, hcpn = acc_stream(np.concatenate([bitsC, bitsP], 1), cpn, P, False)
Ht, htn = acc_stream(t[:, None].astype(np.uint8), ["t"], P, False)
H = np.concatenate([Hcp, Ht], 1); hn = hcpn + htn
idx = sel[:NF]; Si = S[idx]; Hi = H[idx]
both = np.concatenate([Si, Hi], 1); rh, hp = gf2.rank_and_pivots(both[:12000])
hsel = [q - Si.shape[1] for q in hp if q >= Si.shape[1]]; Hi = Hi[:, hsel]; hn = [hn[q] for q in hsel]
nS, nH = Si.shape[1], Hi.shape[1]
prods = (Hi[:, :, None] & Si[:, None, :]).reshape(len(idx), -1)
X = np.concatenate([np.ones((len(idx), 1), np.uint8), Si, Hi, prods], 1)
res = gf2.solve(X, t[idx], "imp"); print(f"P={P}: X cols {X.shape[1]} rank {res['rank']} inconsistent {res['bad']} of {len(idx)-res['rank']}")
def colname(k):
    if k == 0: return "1"
    k -= 1
    if k < nS: return snames[k]
    k -= nS
    if k < nH: return hn[k]
    k -= nH; return hn[k // nS] + "*" + snames[k % nS]
sol = res['solution']; print("solution terms:", len(sol))
# locate terms involving the own-residue t accumulator at i (which contains t_i): any H column whose name has 'acc..t' -- the own residue block differs per row, so instead compute g numerically:
# g(S) = 1 ^ alpha0 ^ sum beta_m s_m, where alpha0 = coefficient of a_t, beta_m = coeff of a_t*s_m, a_t = the H column equal to the accumulator containing t_i.
# The column containing t_i at row i is the block of residue (i mod P); columns differ by row, so derive implicitly: predict with t_i=0 and t_i=1 and see which rows give a consistent/inconsistent equation.
def predict(Xrows): 
    pr = np.zeros(Xrows.shape[0], np.uint8)
    for k in sol: pr ^= Xrows[:, k]
    return pr
# rebuild features with t_i flipped: flip t at idx positions, recompute accumulator columns for those rows
def build_rows(tt_override):
    Ht2, _ = acc_stream(tt_override[:, None].astype(np.uint8), ["t"], P, False)
    H2 = np.concatenate([Hcp, Ht2], 1)[idx][:, hsel]
    pr2 = (H2[:, :, None] & Si[:, None, :]).reshape(len(idx), -1)
    return np.concatenate([np.ones((len(idx), 1), np.uint8), Si, H2, pr2], 1)
t_flip = t.copy(); t_flip[idx] ^= 1
pred0 = predict(X); pred1 = predict(build_rows(t_flip))   # pred with actual t_i, and with t_i flipped
g = pred0 ^ pred1                        # g=1: equation depends on t_i (determines it); g=0: equation is a constraint
print("P(g=1) =", g.mean(), "; on g=1 rows, pred==t:", (pred0[g==1] == t[idx][g==1]).mean(), "; on g=0 rows, pred==t:", (pred0[g==0] == t[idx][g==0]).mean())
# express g as linear form over S: fit
rg = gf2.solve(np.concatenate([np.ones((len(idx),1),np.uint8), Si], 1), g, "g")
print("g linear in S? inconsistent", rg['bad'], "; support:", ["1" if k == 0 else snames[k-1] for k in rg['solution']])
# save g's definition for later use
np.save("g_rows.npy", np.array([0] + [k for k in rg['solution']]))
print("example solution terms:", [colname(k) for k in sol[:40]])

print("\nALL terms:", [colname(k) for k in sol])
par_i = (idx % 2).astype(np.uint8)
print("g vs parity of i: P(g=1 | i even) =", g[par_i == 0].mean(), " P(g=1 | i odd) =", g[par_i == 1].mean())
# verify the formula on ALL rows of the non-degenerate parity, over the whole stream
allidx = sel
Sa = S[allidx]; Ha = H[allidx][:, hsel]
def predict_rows(Sa, Ha):
    pr = np.zeros(Sa.shape[0], np.uint8)
    for k in sol:
        if k == 0: pr ^= 1; continue
        kk = k - 1
        if kk < nS: pr ^= Sa[:, kk]; continue
        kk -= nS
        if kk < nH: pr ^= Ha[:, kk]; continue
        kk -= nH; pr ^= Ha[:, kk // nS] & Sa[:, kk % nS]
    return pr
pa = predict_rows(Sa, Ha); pr_i = allidx % 2
for q in (0, 1):
    mk = pr_i == q
    print(f"whole stream, i%2=={q}: rows {mk.sum()}, formula matches t: {(pa[mk] == t[allidx][mk]).mean():.6f}")
