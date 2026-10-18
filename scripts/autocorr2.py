import sys, numpy as np
from ns10 import load_dump
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.float64); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
# detrend: subtract a local event rate (moving average over 129 words), zero out gate-closed rows
k = 129; ker = np.ones(k) / k
loc = np.convolve(e, ker, mode='same'); x = (e - loc); x[~G] = 0
n = 1 << 22; F = np.fft.rfft(x, n); ac = np.fft.irfft(F * np.conj(F), n)[:N] / N; ac /= ac[0]
sm = np.convolve(ac, np.ones(51) / 51, mode='same'); spike = ac - sm
print("detrended autocorrelation lag1..8:", np.round(ac[1:9], 4), " noise ~ %.5f" % (1 / np.sqrt(N)))
cand = np.argsort(-spike[16:N // 2])[:20] + 16
print("largest spikes above the smooth background (lag: spike, ac):")
for L in sorted(cand, key=lambda q: -spike[q]): print(f"   {L:8d}: {spike[L]:+.5f}  ac {ac[L]:+.5f}")
# also negative spikes
cand = np.argsort(spike[16:N // 2])[:8] + 16
print("largest negative spikes:"); [print(f"   {L:8d}: {spike[L]:+.5f}") for L in cand]
# power spectrum peaks (periodicity): top frequencies
P = np.abs(F) ** 2; P[:8] = 0
top = np.argsort(-P[:n // 2])[:12]
print("top spectral peaks (period in words, power/median):")
med = np.median(P[8:n // 2])
for f in top: print(f"   period {n / f:12.2f}  power/median {P[f] / med:8.1f}")
