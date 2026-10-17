import sys, numpy as np
from ns10 import load_dump
from fwht import best_form, parity16
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
LC0 = (1<<0)|(1<<2)|(1<<3)|(1<<7)|(1<<8)|(1<<9)|(1<<10); LC1 = (1<<0)|(1<<2)
idx_all = np.arange(16, N - 1)
a = ((m[idx_all - 2] >> 0) & 1); bq = ((m[idx_all - 3] >> 0) & 1)
glob = np.load("linmodel.npy")
for ca, cb in ((0, 1), (0, 0), (1, 0), (1, 1)):
    idx = idx_all[(a == ca) & (bq == cb)]
    A = {k: int(glob[k]) for k in range(8)}; B = {k: int(glob[8 + k]) for k in range(8)}; const = int(glob[16]); T2 = 1
    B[0] ^= LC0; B[1] ^= LC1
    def predict(ix):
        pr = np.full(len(ix), const, np.uint8)
        if T2: pr ^= t[ix - 2]
        for k in range(8):
            if A[k]: pr ^= parity16(p[ix - k], A[k])
            if B[k]: pr ^= parity16(c[ix - k], B[k])
        return pr
    before = float((predict(idx) == t[idx]).mean())
    for rnd in range(3):
        for k in range(7):
            for which in ("A", "B"):
                cur = (A if which == "A" else B)[k]; words = (p if which == "A" else c)[idx - k]
                y = predict(idx) ^ t[idx] ^ parity16(words, cur)
                aa, cst, agr = best_form(words, y, exclude_zero=False)
                if which == "A": A[k] = aa
                else: B[k] = aa
                const ^= cst
        y = predict(idx) ^ t[idx] ^ (t[idx - 2] if T2 else 0); T2 = 1 if float((y == t[idx - 2]).mean()) > float((y == 0).mean()) else 0
    after = float((predict(idx) == t[idx]).mean())
    print(f"cell a={ca} b={cb}: rows {len(idx)}; agreement global-model {before:.4f} -> refined {after:.4f}; A={[hex(A[k]) for k in range(7)]} B={[hex(B[k]) for k in range(7)]} t-2:{T2} const {const}", flush=True)
