import sys, numpy as np
from ns10 import load_dump
import features
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); gate = ~((a == 0) & (bq == 1))
idx = np.arange(60, N - 3); idx = idx[gate[idx] & gate[idx - 1]]
on = idx[e[idx - 1] == 1]; off = idx[(e[idx - 1] == 0) & (e[idx - 2] == 0) & (e[idx - 3] == 0) & gate[idx - 2] & gate[idx - 3]]
print(f"rows after an event (gate open both): {len(on)}, P(e)={e[on].mean():.3f};  rows after 3 quiet (gates open): {len(off)}, P(e)={e[off].mean():.3f}")
def scan(rows, label):
    y = e[rows]; out = []
    for w in range(-1, 13):
        for nm, arr in (("c", c), ("p", p), ("m", m)):
            if w == -1 and nm != "c": continue
            bits = features.bits16(arr[rows - w])
            for bb in range(16):
                s1 = bits[:, bb] == 1
                if s1.sum() > 500 and (~s1).sum() > 500:
                    r1 = y[s1].mean(); r0 = y[~s1].mean(); out.append((abs(r1 - r0), f"{nm}[i-{w}].{bb}", r1, r0))
    for k in range(1, 9):
        s1 = t[rows - k] == 1; r1 = y[s1].mean(); r0 = y[~s1].mean(); out.append((abs(r1 - r0), f"t-{k}", r1, r0))
    out.sort(reverse=True)
    print(f"[{label}] strongest single-bit modulators of e_i (|rate1-rate0|, bit, rate1, rate0):")
    for o in out[:10]: print(f"    {o[0]:.3f}  {o[1]:12s} {o[2]:.3f} {o[3]:.3f}")
scan(on, "after event: stay-on vs turn-off"); scan(off, "after quiet: turn-on")
