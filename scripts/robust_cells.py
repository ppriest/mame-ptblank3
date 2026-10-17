import sys, numpy as np
from ns10 import load_dump
from fwht import best_form, parity16
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
idx_all = np.arange(16, N - 1)
a = ((m[idx_all - 2] >> 0) & 1); bq = ((m[idx_all - 3] >> 0) & 1)
for ca, cb in ((0, 0), (0, 1), (1, 0), (1, 1)):
    idx = idx_all[(a == ca) & (bq == cb)]
    A = {k: 0 for k in range(8)}; B = {k: 0 for k in range(8)}; T = {k: 0 for k in range(1, 5)}; const = 0
    def predict(ix):
        pr = np.full(len(ix), const, np.uint8)
        for k in range(8):
            if A[k]: pr ^= parity16(p[ix - k], A[k])
            if B[k]: pr ^= parity16(c[ix - k], B[k])
        for k in range(1, 5):
            if T[k]: pr ^= t[ix - k]
        return pr
    for rnd in range(4):
        for k in range(7):
            for which in ("A", "B"):
                cur = (A if which == "A" else B)[k]; words = (p if which == "A" else c)[idx - k]
                y = predict(idx) ^ t[idx] ^ parity16(words, cur)
                aa, cst, agr = best_form(words, y, exclude_zero=False)
                if which == "A": A[k] = aa
                else: B[k] = aa
                const ^= cst
        for k in range(1, 5):   # t lags: include if it improves agreement
            y = predict(idx) ^ t[idx] ^ (t[idx - k] if T[k] else 0)
            agr_with = float((y == t[idx - k]).mean()); agr_without = float((y == 0).mean())
            T[k] = 1 if agr_with > agr_without else 0
    agr = float((predict(idx) == t[idx]).mean())
    print(f"cell a={ca} b={cb}: rows {len(idx)}, robust linear agreement {agr:.4f} (residual {1-agr:.4f}); A={[hex(A[k]) for k in range(6)]} B={[hex(B[k]) for k in range(6)]} T={[T[k] for k in range(1,5)]} const {const}", flush=True)
