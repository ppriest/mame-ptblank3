import sys, numpy as np
from ns10 import load_dump
from fwht import best_form, parity16
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
LC0 = (1<<0)|(1<<2)|(1<<3)|(1<<7)|(1<<8)|(1<<9)|(1<<10); LC1 = (1<<0)|(1<<2)
z = (p == 0); before = np.zeros(N, np.int64)
for i in range(1, N): before[i] = before[i-1] + 1 if z[i] else 0
idx_all = np.arange(16, N - 1)
# model: t_i = L_c(c_i,c_{i-1}) ^ t_{i-2} ^ XOR_k A_k(p_{i-k}) ^ XOR_k B_k(c_{i-k}) (extra c terms) ^ const
A = {k: 0 for k in range(0, 8)}; B = {k: 0 for k in range(0, 8)}; const = 0
def predict(idx):
    pr = parity16(c[idx], LC0) ^ parity16(c[idx - 1], LC1) ^ t[idx - 2] ^ const
    for k in range(8):
        if A[k]: pr ^= parity16(p[idx - k], A[k])
        if B[k]: pr ^= parity16(c[idx - k], B[k])
    return pr
def agreement(idx): return float((predict(idx) == t[idx]).mean())
print("initial agreement (deep-zero law only):", agreement(idx_all))
# sequential: at offset off (before == off+1) the last nonzero plaintext is p_{i-off-1}
for off in (4, 3, 2, 1, 0):
    k = off + 1
    idx = np.where((before == off + 1) & (np.arange(N) > 40) & (np.arange(N) < N - 1))[0]
    y = predict(idx) ^ t[idx]        # residual to explain
    a, cst, agr = best_form(p[idx - k], y)
    print(f"offset {off}: samples {len(idx)}, residual rate {y.mean():.3f}; best A{k}=0x{a:04x} const {cst} agreement {agr:.3f}")
    A[k] = a; const ^= cst
# current word p_i on general samples
idx = np.where((before == 0) & (np.arange(N) > 40) & (np.arange(N) < N - 1))[0]
y = predict(idx) ^ t[idx]; a, cst, agr = best_form(p[idx], y)
print(f"general (p_i != 0): samples {len(idx)}, residual rate {y.mean():.3f}; best A0=0x{a:04x} const {cst} agreement {agr:.3f}")
A[0] = a; const ^= cst
print("agreement on all:", agreement(idx_all))
# refinement rounds: refit each A_k, B_k on all data given the others
for rnd in range(3):
    for k in range(0, 7):
        for which in ("A", "B"):
            cur = (A if which == "A" else B)[k]
            words = (p if which == "A" else c)[idx_all - k]
            y = predict(idx_all) ^ t[idx_all] ^ parity16(words, cur)   # residual with this term removed
            a, cst, agr = best_form(words, y, exclude_zero=False)
            if which == "A": A[k] = a
            else: B[k] = a
            const ^= cst
    print(f"round {rnd}: agreement {agreement(idx_all):.4f}; A={[hex(A[k]) for k in range(7)]} B={[hex(B[k]) for k in range(7)]} const {const}")
np.save("linmodel.npy", np.array([A[k] for k in range(8)] + [B[k] for k in range(8)] + [const]))
res = predict(idx_all) ^ t[idx_all]
print("final residual (NL event) rate:", res.mean(), "events:", int(res.sum()))
np.save("nl_events.npy", idx_all[res == 1])
