import sys, numpy as np
from ns10 import load_dump
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
idx = np.arange(80, N - 3); idx = idx[G[idx]]
nz = np.ones(len(idx), bool)
for w in range(0, 6): nz &= p[idx - w] != 0
idx_d = idx[nz]; print("gate-open rows with p[i..i-5] nonzero:", len(idx_d), "event rate", e[idx_d].mean())
def coll(rows, lo, hi, label):
    key = np.zeros(len(rows), np.uint64)
    for w in range(lo, hi + 1): key = key * np.uint64(0x9E3779B97F4A7C15) + p[rows - w].astype(np.uint64) + np.uint64(w)
    order = np.argsort(key, kind='stable'); ks = key[order]; es = e[rows][order]; rs = rows[order]
    same = ks[1:] == ks[:-1]; agree = same & (es[1:] == es[:-1])
    # exclude trivially adjacent duplicates (same region, overlapping): require |i-j| > 64
    far = same & (np.abs(rs[1:].astype(np.int64) - rs[:-1].astype(np.int64)) > 64)
    agree_f = far & (es[1:] == es[:-1]); both1 = far & (es[1:] == 1) & (es[:-1] == 1)
    pe = es.mean(); chance = 1 - 2 * pe * (1 - pe)
    print(f"  {label:28s}: pairs {int(same.sum()):7d} (far {int(far.sum()):6d}); P(agree|far) = {agree_f.sum()/max(far.sum(),1):.3f} (chance {chance:.3f}); both events {int(both1.sum())} of {int(far.sum())}")
for lo, hi in ((0, 7), (0, 11), (0, 15), (0, 23), (4, 11), (4, 15), (1, 12), (2, 17)):
    coll(idx_d, lo, hi, f"plain words i-{lo}..i-{hi}")
# the same with cipher words (control: should be near chance unless state repeats)
def collc(rows, lo, hi, label):
    key = np.zeros(len(rows), np.uint64)
    for w in range(lo, hi + 1): key = key * np.uint64(0x9E3779B97F4A7C15) + c[rows - w].astype(np.uint64) + np.uint64(w)
    order = np.argsort(key, kind='stable'); ks = key[order]; es = e[rows][order]; rs = rows[order]
    same = ks[1:] == ks[:-1]; far = same & (np.abs(rs[1:].astype(np.int64) - rs[:-1].astype(np.int64)) > 64)
    print(f"  {label:28s}: pairs {int(same.sum()):7d} (far {int(far.sum()):6d}); P(agree|far) = {(far & (es[1:]==es[:-1])).sum()/max(far.sum(),1):.3f}")
collc(idx_d, 0, 7, "cipher words i..i-7 (control)")
