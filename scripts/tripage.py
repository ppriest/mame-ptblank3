import sys, numpy as np, time
from ns10 import load_dump
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); E = np.zeros(N, bool); E[ev] = True
Ep = E.reshape(-1, 256); rate = E.mean(); print("pages", Ep.shape[0], "event rate", rate)
t0 = time.time(); zeros = []; low = []
for a in range(1, 255):
    A = Ep[:, :256 - a] & Ep[:, a:]
    na = A.sum()
    for bb in range(a + 1, 256):
        cntb = (A[:, :256 - bb] & Ep[:, bb:]).sum()
        exp = A[:, :256 - bb].sum() * rate
        if exp > 30:
            if cntb == 0: zeros.append((a, bb, int(exp)))
            elif cntb < exp * 0.25: low.append((a, bb, int(cntb), int(exp)))
    if a % 50 == 0: print(f"  a={a} ({time.time()-t0:.0f}s) zeros {len(zeros)} low {len(low)}", flush=True)
print("within-page zero triple pairs (a, b, expected):", zeros[:40], "total", len(zeros))
print("strongly depleted (count < 25% expected):", low[:20], "total", len(low))
