import sys, numpy as np
from ns10 import load_dump
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
idx = np.arange(80, N - 3); idx = idx[G[idx]]
def groups(lo, hi, minsize=8):
    key = np.zeros(len(idx), np.uint64)
    for w in range(lo, hi + 1): key = key * np.uint64(0x9E3779B97F4A7C15) + p[idx - w].astype(np.uint64) + np.uint64(w)
    order = np.argsort(key, kind='stable'); ks = key[order]; es = e[idx][order]
    bounds = np.concatenate([[0], np.where(ks[1:] != ks[:-1])[0] + 1, [len(ks)]])
    sizes = np.diff(bounds); sums = np.add.reduceat(es.astype(np.int64), bounds[:-1])
    big = sizes >= minsize
    rates = sums[big] / sizes[big]
    hist = np.histogram(rates, bins=np.linspace(0, 1, 21))[0]
    nz_groups = (sums[big] > 0).mean()
    # exclude the all-zero-plaintext group(s)
    print(f"plain words i-{lo}..i-{hi}: groups>= {minsize}: {big.sum()}, rows covered {sizes[big].sum()}; mean rate {rates.mean():.3f}; groups with 0 events: {(sums[big]==0).sum()} ({(sums[big]==0).mean():.2f})")
    print("   rate histogram (0..1 in 20 bins):", hist.tolist())
    # among groups with >= 30 members: distribution of rates
    big30 = sizes >= 30; r30 = sums[big30] / sizes[big30]
    print(f"   groups>=30: {big30.sum()}; histogram:", np.histogram(r30, bins=np.linspace(0, 1, 21))[0].tolist())
    return order, bounds, sizes, sums
for lo, hi in ((4, 5), (4, 7), (4, 9), (2, 9), (0, 9), (0, 15)):
    groups(lo, hi)
