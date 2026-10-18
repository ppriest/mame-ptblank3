import sys, numpy as np
from ns10 import load_dump
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.float64); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
# restrict to gate-open rows (set others to the mean so they do not contribute)
x = e.copy(); mu = e[G].mean(); x[~G] = mu; x -= mu
n = 1 << 22
F = np.fft.rfft(x, n); ac = np.fft.irfft(F * np.conj(F), n)[:N] / N
ac /= ac[0]
lags = np.arange(N)
print("autocorrelation (normalised): lag1..8:", np.round(ac[1:9], 4))
# peaks beyond lag 64
cand = np.argsort(-ac[64:])[:25] + 64
print("top lags > 64 by autocorrelation:")
for L in sorted(cand, key=lambda q: -ac[q]): print(f"   lag {L:8d}: {ac[L]:+.5f}")
print("noise level ~ %.5f" % (1 / np.sqrt(N)))
# also check per-byte-period hypotheses: lags that are odd periods P with ac peaks at P, 2P
