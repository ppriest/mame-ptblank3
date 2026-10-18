import sys, numpy as np
from ns10 import load_dump
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
idx = np.arange(100, N - 3); idx = idx[G[idx]]; ee = e[idx]
for M in (0xc980, 0xff00, 0xc880, 0x8080, 0x0080, 0x8000, 0x0800, 0x0100, 0xc000):
    row = []
    for hi in (12, 16, 20, 24, 32, 40, 48, 64, 96):
        D = np.zeros(len(idx), bool)
        for w in range(4, hi + 1): D |= (p[idx - w] & M) != 0
        row.append(f"{hi}:{int(ee[~D].sum())}/{(~D).sum()//1000}k")
    print(f"mask 0x{M:04x} leaked events / rows(D=0) by depth: " + ' '.join(row))
# context of leaked events for M=0xc980, depth 4..20: distance back to the last word with (p & 0xff00) != 0, and to last nonzero word
M = 0xc980; D = np.zeros(len(idx), bool)
for w in range(4, 21): D |= (p[idx - w] & M) != 0
leak = idx[(~D) & (ee == 1)]
hb = (p & 0xff00) != 0; nzw = p != 0
def dist_back(pos, flag):
    out = []
    for q in pos:
        k = 1
        while k < 400 and not flag[q - k]: k += 1
        out.append(k)
    return np.array(out)
dh = dist_back(leak, hb); dn = dist_back(leak, nzw)
print(f"leaked events: {len(leak)}; distance to last high-byte word: median {np.median(dh):.0f}, min {dh.min()}, max {dh.max()}; hist(<=4,5-20,21-40,41-100,>100):", [(dh <= 4).sum(), ((dh > 4) & (dh <= 20)).sum(), ((dh > 20) & (dh <= 40)).sum(), ((dh > 40) & (dh <= 100)).sum(), (dh > 100).sum()])
print(f"   distance to last nonzero word: median {np.median(dn):.0f}; hist(1-4,5-20,>20):", [(dn <= 4).sum(), ((dn > 4) & (dn <= 20)).sum(), (dn > 20).sum()])
print("   sample leaked contexts (p[i-8..i+1]):")
for q in leak[:8]: print("     ", ' '.join(f"{p[q-w]:04x}" for w in range(8, -2, -1)), f" dist_hb={dist_back([q], hb)[0]}")
