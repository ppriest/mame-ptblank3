import sys, numpy as np, itertools
from ns10 import load_dump
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
idx = np.arange(200, N - 3); idx = idx[G[idx]]; y = e[idx].astype(np.float64); ym = y.mean()
def corr(x):
    x = x.astype(np.float64); mx = x.mean()
    if mx == 0 or mx == 1: return 0.0
    return float(((x - mx) * (y - ym)).mean() / np.sqrt(mx * (1 - mx) * ym * (1 - ym)))
words = {}
for w in range(-1, 9):
    for nm, arr in (("c", c), ("p", p), ("m", m)):
        if w == -1 and nm != "c": continue
        words[f"{nm}{w}"] = arr[idx - w].astype(np.int64)
res = []
for (ka, A), (kb, B) in itertools.combinations(words.items(), 2):
    for op in ("+", "-"):
        S = (A + B) if op == "+" else (A - B)
        R = S & 0x1ffff
        carries = (R ^ A ^ B)            # carry-in vector into each bit (for +), borrow for -
        for k in range(1, 17):
            res.append((abs(corr((carries >> k) & 1)), f"carry{'' if op=='+' else '(borrow)'} into bit {k} of {ka}{op}{kb}"))
        for k in range(0, 16):
            res.append((abs(corr((R >> k) & 1)), f"sum bit {k} of {ka}{op}{kb}"))
# running sums of plaintext / cipher / mask from depth D
for nm, arr in (("p", p), ("c", c), ("m", m)):
    for D in (0, 3, 4, 5):
        cs = np.cumsum(arr.astype(np.int64)); S = cs[idx - D] & 0xffff; S1 = cs[idx - D - 1] & 0xffff; x = arr[idx - D].astype(np.int64)
        cv = ((S1 + x) & 0x1ffff) ^ S1 ^ x
        for k in range(1, 17): res.append((abs(corr((cv >> k) & 1)), f"carry into bit {k} of running sum of {nm} at depth {D}"))
        for k in range(16): res.append((abs(corr((S >> k) & 1)), f"bit {k} of running sum of {nm} to depth {D}"))
res.sort(reverse=True)
print(f"{len(res)} candidate bits; |correlation| with residual (gate-open rows, rate {ym:.3f}):")
for r_ in res[:20]: print(f"   {r_[0]:.4f}  {r_[1]}")
# reference: strongest single raw bit
best = max((abs(corr((v >> k) & 1)), f"{kk}.{k}") for kk, v in words.items() for k in range(16))
print("reference: strongest raw bit", best)
