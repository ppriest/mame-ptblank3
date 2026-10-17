import sys, numpy as np
from ns10 import load_dump
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); clear = (a == 0) & (bq == 1)
idx = np.arange(45, N - 3); idx = idx[~clear[idx]]; ee = e[idx].astype(np.int64)
def mi(x, y):
    n = len(x); jx = x.astype(np.int64) * 2 + y
    cj = np.bincount(jx, minlength=1 << 17).astype(np.float64); cx = cj.reshape(-1, 2).sum(1); cy = np.array([n - y.sum(), y.sum()], np.float64)
    pj = cj / n; px = cx / n; py = cy / n
    nz = pj > 0; pxj = np.repeat(px, 2)[nz]; pyj = np.tile(py, 1 << 16)[nz]
    return float((pj[nz] * np.log2(pj[nz] / (pxj * pyj))).sum())
rng = np.random.default_rng(1); esh = rng.permutation(ee)
print("MI(event; word) in bits, minus shuffled baseline; H(event) = %.3f" % (-(ee.mean()*np.log2(ee.mean()) + (1-ee.mean())*np.log2(1-ee.mean()))))
for w in range(-1, 7):
    row = []
    for nm, arr in (("c", c), ("p", p), ("m", m)):
        x = arr[idx - w]
        if nm == "p" and w == -1: x = x & 0xf7ff
        if nm == "m" and w == -1: x = x & 0xf7ff
        row.append(f"{nm}:{mi(x, ee) - mi(x, esh):.4f}")
    print(f"  word i-{w:2d}: " + "  ".join(row))
# also: MI with low byte / high byte separately for p and m
print("MI with (m_{i-2}, m_{i-3}) low bytes jointly:", mi(((m[idx-2] & 0xff) | ((m[idx-3] & 0xff) << 8)).astype(np.uint16), ee) - mi(((m[idx-2] & 0xff) | ((m[idx-3] & 0xff) << 8)).astype(np.uint16), esh))
