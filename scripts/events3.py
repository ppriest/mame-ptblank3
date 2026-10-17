import sys, numpy as np
from ns10 import load_dump
from fwht import fwht
import features
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("nl_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
lo, hi = 41, N - 4; idx = np.arange(lo, hi); ee = e[idx]; E = idx[ee == 1]
# 1. per-word FWHT on events: max_D P(D(word)=1 | event)
print("1. best one-word affine factor given events: P(D=1|event) [and P(D=1|all)]")
for w in range(-2, 7):
    for nm, arr in (("c", c), ("p", p), ("m", m)):
        s = np.bincount(arr[E - w].astype(np.int64), minlength=65536); W = fwht(s); W[0] = 0
        a = int(np.argmax(np.abs(W))); pe = 0.5 + W[a] / (2 * len(E))     # P(D=0)-P(D=1) = W/n  -> P(D=1) = (n - W)/2n
        pe1 = (len(E) - W[a]) / (2 * len(E))
        sa = np.bincount(arr[idx - w].astype(np.int64), minlength=65536); Wa = fwht(sa); pa1 = (len(idx) - Wa[a]) / (2 * len(idx))
        flag = " <==" if max(pe1, 1 - pe1) > 0.9 else ""
        print(f"   {nm}[i-{w}]: D=0x{a:04x} P(D=1|event)={pe1:.3f} P(D=1|all)={pa1:.3f}{flag}")
# 3. missing linear terms in words ahead
for w in (1, 2, 3):
    bits = features.bits16(c[idx + w]); r1 = np.array([ee[bits[:, k] == 1].mean() for k in range(16)]); r0 = np.array([ee[bits[:, k] == 0].mean() for k in range(16)])
    print(f"3. c[i+{w}] max |rate1-rate0| = {np.abs(r1 - r0).max():.3f}")
# 4. event strings in long zero runs, and decay curve to 40
z = (p == 0); before = np.zeros(N, np.int64)
for i in range(1, N): before[i] = before[i-1] + 1 if z[i] else 0
after = np.zeros(N, np.int64)
for i in range(N - 2, -1, -1): after[i] = (after[i+1] + 1) if z[i] else 0
print("4. event rate by zeros-before 0..40:", ' '.join(f"{float(e[(before == q) & (np.arange(N) > 41)].mean()):.3f}" for q in range(41)))
starts = np.where(z & ~np.roll(z, 1))[0]; starts = starts[(starts > 100) & (starts < N - 100)]; longs = starts[after[starts] >= 60]
print("   event strings (offsets -8..51, '|' = run start) for 16 long runs:")
for s in longs[:16]: print("    " + ''.join(str(e[s+k]) for k in range(-8, 0)) + '|' + ''.join(str(e[s+k]) for k in range(0, 52)))
