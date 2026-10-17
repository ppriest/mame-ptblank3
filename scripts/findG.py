import sys, numpy as np
from ns10 import load_dump
from fwht import best_form, parity16, fwht
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); clear = (a == 0) & (bq == 1)
idx = np.arange(45, N - 3)
rows = idx[(e[idx - 1] == 1) & ~clear[idx]]
print("rows with e_{i-1}=1 and not clear:", len(rows), "P(e_i)=", e[rows].mean())
y = e[rows]
print("best single-word affine form for e_i on these rows (agreement; 0.5 = none):")
for w in range(0, 6):
    for nm, arr in (("c", c), ("p", p), ("m", m)):
        f, cst, agr = best_form(arr[rows - w], y)
        print(f"   {nm}[i-{w}]: form 0x{f:04x} const {cst} agreement {agr:.3f}")
f, cst, agr = best_form(c[rows + 1], y); print(f"   c[i+1]: form 0x{f:04x} const {cst} agreement {agr:.3f}")
# two-word search: word A (16 bits) x low/high byte of word B (8 bits) -> 24-bit FWHT
def best_form24(wA, wB8, y):
    v = wA.astype(np.int64) | (wB8.astype(np.int64) << 16)
    s = np.bincount(v, weights=(1 - 2 * y.astype(np.int64)), minlength=1 << 24); W = fwht(s); W[0] = 0
    k = int(np.argmax(np.abs(W))); return k & 0xffff, k >> 16, (len(y) + abs(W[k])) / (2 * len(y))
best = []
words = {f"{nm}[i-{w}]": arr[rows - w] for w in range(0, 4) for nm, arr in (("c", c), ("p", p), ("m", m))}
keys = list(words.keys())
for ia in range(len(keys)):
    for ib in range(len(keys)):
        if ia == ib: continue
        for half in (0, 1):
            fa, fb, agr = best_form24(words[keys[ia]], (words[keys[ib]] >> (8 * half)) & 0xff, y)
            best.append((agr, keys[ia], fa, keys[ib], half, fb))
best.sort(reverse=True)
print("best two-word (16+8 bit) forms:"); [print("   ", f"{r[0]:.3f}", r[1], f"0x{r[2]:04x}", r[3], "byte", r[4], f"0x{r[5]:02x}") for r in best[:8]]
