import sys, numpy as np
from ns10 import load_dump
import features, pages
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("nl_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
lo, hi = 41, N - 2; ee = e[lo:hi]; idx = np.arange(lo, hi)
print("event rate", ee.mean())
print("P(e_i+1 | e_i)=%.3f  P(e_i+2 | e_i)=%.3f  P(e_i+1 | not e_i)=%.3f" % ((e[ev+1]).mean(), (e[ev+2]).mean(), e[idx[ee==0]+1].mean()))
print("autocorr of e at lags 1..8:", ["%.3f" % np.corrcoef(ee[:-k], ee[k:])[0,1] for k in range(1, 9)])
pos = idx % 256; pr = np.bincount(pos, weights=ee, minlength=256) / np.bincount(pos, minlength=256)
print("rate by pos in page: min %.3f max %.3f; first 16: %s" % (pr.min(), pr.max(), np.round(pr[:16], 3)))
print("rate by pos parity:", [float(ee[pos % 2 == q].mean()) for q in (0, 1)], " by pos mod 4:", [round(float(ee[pos % 4 == q].mean()), 4) for q in range(4)])
pg = pages.page_sequence(sys.argv[2]); page_addr = np.repeat(pg, 256)[lo:hi]
print("rate by page parity:", [round(float(ee[(page_addr % 2) == q].mean()), 4) for q in (0, 1)], " by page in block (first 8):", [round(float(ee[(page_addr % 32) == q].mean()), 3) for q in range(8)])
# per-bit conditioning
def cond(bits, label):
    r1 = [float(ee[bits[:, k] == 1].mean()) for k in range(bits.shape[1])]; r0 = [float(ee[bits[:, k] == 0].mean()) for k in range(bits.shape[1])]
    dev = np.abs(np.array(r1) - np.array(r0)); k = int(np.argmax(dev))
    print(f"  {label}: max |rate(1)-rate(0)| = {dev[k]:.3f} at bit {k} (rate1 {r1[k]:.3f}, rate0 {r0[k]:.3f})")
for w in range(6):
    cond(features.bits16(c[idx - w]), f"c[i-{w}]"); cond(features.bits16(p[idx - w]), f"p[i-{w}]"); cond(features.bits16(m[idx - w]), f"m[i-{w}]")
cond(features.bits16(c[idx + 1]), "c[i+1]"); cond(features.bits16(p[idx + 1]), "p[i+1]")
tl = np.stack([t[idx - k] for k in range(1, 9)], 1).astype(np.uint8); cond(tl, "t lags 1..8")
# zero plaintext conditioning
for w in range(7):
    zz = p[idx - w] == 0
    print(f"  P(event | p[i-{w}]==0) = {ee[zz].mean():.4f}  (n={zz.sum()});  P(event | p[i-{w}]!=0) = {ee[~zz].mean():.4f}")
allz = np.ones(len(idx), bool)
for w in range(6): allz &= p[idx - w] == 0
print(f"  P(event | p[i..i-5] all zero) = {ee[allz].mean():.5f} (n={allz.sum()})")
# popcount dependence
for w in (0, 1, 2):
    pc = np.bitwise_count(p[idx - w]).astype(np.int64)
    print(f"  rate by popcount(p[i-{w}]):", [round(float(ee[pc == q].mean()), 3) if (pc == q).sum() > 500 else None for q in range(17)])
