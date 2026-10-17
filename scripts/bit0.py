import sys, numpy as np
from ns10 import load_dump
import features
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("nl_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
lo, hi = 41, N - 2; idx = np.arange(lo, hi); ee = e[idx]
dense = np.ones(len(idx), bool)
for w in range(6): dense &= p[idx - w] != 0
b0 = lambda w: ((m[idx - w] >> 0) & 1).astype(np.int64)
print("2x2 on (m[i-2].0, m[i-3].0): rate", {f"{a}{bq}": round(float(ee[(b0(2)==a)&(b0(3)==bq)].mean()),3) for a in (0,1) for bq in (0,1)})
print("   same, dense regions only:", {f"{a}{bq}": round(float(ee[dense & (b0(2)==a)&(b0(3)==bq)].mean()),3) for a in (0,1) for bq in (0,1)})
# pattern of mask bit 0 over words i..i-5 (6 bits) -> rate, dense only
pat = sum(b0(w) << w for w in range(6))
rates = np.array([ee[dense & (pat == q)].mean() for q in range(64)]); cnts = np.array([(dense & (pat == q)).sum() for q in range(64)])
order = np.argsort(-rates)
print("top patterns of m.0 bits (w0..w5 = words i..i-5), dense regions:")
for q in order[:10]: print(f"   pattern {q:06b} (w5..w0): rate {rates[q]:.3f} n {cnts[q]}")
print("bottom:"); [print(f"   pattern {q:06b}: rate {rates[q]:.3f} n {cnts[q]}") for q in order[-5:]]
# other mask bits: which single mask bits of words i-2, i-3 modulate? table of rate(bit=1)/rate(bit=0) in dense regions
for w in (1, 2, 3, 4):
    bits = features.bits16(m[idx - w]); r1 = np.array([ee[dense & (bits[:, k] == 1)].mean() for k in range(16)]); r0 = np.array([ee[dense & (bits[:, k] == 0)].mean() for k in range(16)])
    print(f"m[i-{w}] dense: rate1/rate0 per bit:", ' '.join(f"{a:.2f}/{bq:.2f}" for a, bq in zip(r1, r0)))
# joint with previous event
for a in (0, 1):
    for bq in (0, 1):
        sel = dense & (b0(2) == a) & (b0(3) == bq)
        print(f"  m2.0={a} m3.0={bq}: P(e | e_prev=1) = {ee[sel & (e[idx-1]==1)].mean():.3f}  P(e | e_prev=0) = {ee[sel & (e[idx-1]==0)].mean():.3f}")
