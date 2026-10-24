import sys, itertools, numpy as np
from ns10 import load_dump
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
EN = np.zeros(N, bool)
for w in range(4, 97): EN[100:] |= (p[100 - w:N - w] & 0xc980) != 0
idx = np.arange(120, N - 3); idx = idx[G[idx] & EN[idx]]; y = e[idx].astype(np.float64); ym = y.mean()
def corr(x):
    x = x.astype(np.float64); mx = x.mean()
    if mx <= 0 or mx >= 1: return 0.0
    return float(((x - mx) * (y - ym)).mean() / np.sqrt(mx * (1 - mx) * ym * (1 - ym)))
words = {}
for w in range(0, 9):
    for nm, arr in (("c", c), ("p", p), ("m", m)): words[f"{nm}{w}"] = arr[idx - w].astype(np.int64)
res = []
for (ka, A), (kb, B) in itertools.permutations(words.items(), 2):
    res.append((abs(corr(A > B)), f"{ka} > {kb}"))
    for lowbits in (4, 8, 11, 12, 16):
        msk = (1 << lowbits) - 1; res.append((abs(corr((A & msk) > (B & msk))), f"{ka}&0x{msk:x} > {kb}&0x{msk:x}"))
    if ka < kb:
        for op in ("+", "-"):
            S = (A + B) if op == "+" else (A - B); cv = (S & 0x1ffff) ^ A ^ B
            for k in range(1, 17): res.append((abs(corr((cv >> k) & 1)), f"carry into {k} of {ka}{op}{kb}"))
res.sort(reverse=True)
ref = max((abs(corr((v >> k) & 1)), f"{kk}.{k}") for kk, v in words.items() for k in range(16))
print(f"enabled gate-open rows {len(idx)}, rate {ym:.3f}; strongest raw bit {ref}")
for r_ in res[:12]: print(f"   {r_[0]:.4f}  {r_[1]}")
