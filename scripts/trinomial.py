import sys, numpy as np, time
from ns10 import load_dump
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c)
ev = np.load("../data/L4_events.npy"); E = np.zeros(N, np.float64); E[ev] = 1
B = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
n = 1 << 22; FE = np.fft.rfft(E, n)
# baseline: pair counts P2[a] = sum_i E_i E_{i+a}
P2 = np.fft.irfft(FE * np.conj(FE), n)[:N]
print("events", int(E.sum()), "; pair counts at small lags:", np.round(P2[1:6]).astype(int)); t0 = time.time()
zeros = []
for a in range(1, B + 1):
    Fa = E[:-a] * E[a:]                       # positions i with events at i and i+a
    if Fa.sum() < 50: continue
    Ffa = np.fft.rfft(Fa, n)
    cnt3 = np.fft.irfft(Ffa * np.conj(FE), n)   # cnt3[k] = sum_i Fa_i E_{i+k}  for k >= 0 (positive lags)
    cnt3 = np.round(cnt3[:B + 1])
    exp = Fa.sum() * E.mean()
    for bb in range(a + 1, B + 1):
        if cnt3[bb] == 0 and exp > 25: zeros.append((a, bb, int(Fa.sum()), exp))
    if a % 500 == 0: print(f"  a={a} ({time.time()-t0:.0f}s) zero-triples so far: {len(zeros)}", flush=True)
print("zero triple-coincidence lag pairs (a, b, pairs(a), expected triples):")
for z in zeros[:40]: print("   ", z)
print("total:", len(zeros))
