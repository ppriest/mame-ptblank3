import sys, numpy as np
from ns10 import load_dump
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); gate = ~((a == 0) & (bq == 1))
z = (p == 0); before = np.zeros(N, np.int64)
for k in range(1, N): before[k] = before[k-1] + 1 if z[k] else 0
# lone impulse words
S = np.where((~z) & (np.roll(before, 1) >= 12))[0]; S = S[(S > 100) & (S < N - 40)]
rows = S + 4; ok = gate[rows] & (before[rows] == 4)
print("lone words (>=12 zeros before, 4 zeros after) and e at depth 4:")
print("  ", sorted([(f"{p[s]:04x}", int(e[s+4])) for s in S[ok]]))
# determinism of transitions
idx = np.arange(60, N - 3); idx = idx[gate[idx] & gate[idx - 1]]
on = idx[e[idx - 1] == 1]
off = idx[(e[idx - 1] == 0) & (e[idx - 2] == 0) & (e[idx - 3] == 0) & gate[idx - 2] & gate[idx - 3]]
def coll(rows, keyfn, label):
    key = keyfn(rows); order = np.argsort(key, kind='stable'); ks = key[order]; es = e[rows][order]
    same = ks[1:] == ks[:-1]; conf = same & (es[1:] != es[:-1])
    # also: fraction of rows in groups of size>=2 and the "majority-consistent" rate
    print(f"   {label:34s}: pairs {int(same.sum()):7d}, conflicts {int(conf.sum()):6d} ({conf.sum()/max(same.sum(),1):.3f})")
def K(*cols):
    def f(rows):
        key = np.zeros(len(rows), np.uint64)
        for arr, w in cols: key = key * np.uint64(0x9E3779B97F4A7C15) + arr[rows - w].astype(np.uint64)
        return key
    return f
for label, rows in (("TURN-ON (after 3 quiet)", off), ("AFTER EVENT", on)):
    print(f"[{label}] rows {len(rows)} P(e)={e[rows].mean():.3f}")
    coll(rows, K((p, 4)), "p4"); coll(rows, K((p, 4), (p, 5)), "p4,p5"); coll(rows, K((p, 4), (p, 5), (p, 6)), "p4,p5,p6"); coll(rows, K((p, 4), (p, 5), (p, 6), (p, 7)), "p4..p7")
    coll(rows, K((p, 3), (p, 4), (p, 5)), "p3,p4,p5"); coll(rows, K((m, 3)), "m3"); coll(rows, K((m, 3), (m, 4)), "m3,m4"); coll(rows, K((m, 2), (m, 3), (m, 4)), "m2,m3,m4")
    coll(rows, K((p, 4), (m, 3)), "p4,m3"); coll(rows, K((p, 4), (p, 5), (m, 3), (m, 4)), "p4,p5,m3,m4"); coll(rows, K((c, 4), (c, 5)), "c4,c5"); coll(rows, K((p, 4), (p, 5), (c, 4), (c, 5)), "p4,p5,c4,c5")
