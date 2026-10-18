import sys, numpy as np
from ns10 import load_dump
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.float64); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
loc = np.convolve(e, np.ones(129) / 129, mode='same'); x = e - loc; x[~G] = 0
n = 1 << 22; F = np.fft.rfft(x, n); ac = np.fft.irfft(F * np.conj(F), n)[:N] / N; ac /= ac[0]
sm = np.convolve(ac, np.ones(51) / 51, mode='same')
print("detrended ac minus background at structural lags:")
for L in [2,4,8,16,32,64,128,255,256,257,258,512,516,768,1024,1032,2048,4096,8192,16384,32768,65536,131072,262144,524288]:
    print(f"  lag {L:7d}: {ac[L]-sm[L]:+.5f}   (ac {ac[L]:+.5f})")
