import numpy as np
from ns10 import *
import gf2, features
# synthetic: encrypt random plaintext with ptblank3 linear part plus a made-up quadratic bit-11 term
rng = np.random.default_rng(1)
N = 60000
plain = rng.integers(0, 65536, N, dtype=np.uint16)
def nl(pc, pp):  # masks = pc^pp ; term = m1[3] & (c2[7] ^ p1[2])
    m = pc ^ pp
    return ((m >> 19) & 1) & (((pc >> 39) ^ (pp >> 18)) & 1)
dec = Decrypter(); dec.init(5)
c = np.zeros(N, np.uint16); t = np.zeros(N, np.uint8)
for i in range(N):
    ci = int(plain[i]) ^ dec.mask
    tb = nl(((dec.pc << 16) ^ ci) & 0xffffffffffffffff, ((dec.pp << 16) ^ int(plain[i])) & 0xffffffffffffffff)
    dec.decrypt(ci, tb); c[i] = ci; t[i] = tb
p = plain
L, names, y = features.window(c, p, t, 4, N, w=3)
r, piv = gf2.rank_and_pivots(L); print("linear rank", r, "of", L.shape[1])
res = gf2.solve(L, y); print("linear fit: inconsistent", res["bad"])
Q, qn = gf2.quad_features(L, piv)
sub = slice(0, 8000)
res = gf2.solve(Q[sub], y[sub], "quad"); print("quad fit (8000 rows): rank", res["rank"], "inconsistent", res["bad"], "terms", len(res["solution"]))
# verify on all
pred = np.zeros(len(y), np.uint8)
for k in res["solution"]: pred ^= Q[:, k]
print("quad solution verified on all rows:", bool(np.all(pred == y)))
