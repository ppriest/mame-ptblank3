import sys, numpy as np
from ns10 import load_dump
import features, gf2
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
def par(x, mask): return (np.bitwise_count((x & mask).astype(np.uint16)) & 1).astype(np.uint8)
# L4' from the reset-state fit
C0 = sum(1 << k for k in (0,1,3,4,7,12,15)); P0 = sum(1 << k for k in (1,2,3,6,8,12,15))
C1 = sum(1 << k for k in (4,8,9,10,12,13,15)); P1 = sum(1 << k for k in (0,2,4,8,10,11))
C2 = sum(1 << k for k in (0,4,5,6,8,9,10,11,14)); P2 = sum(1 << k for k in (0,2))
C3 = sum(1 << k for k in (0,1,2,6,8,11,13,15)); P3 = sum(1 << k for k in (0,2))
i = np.arange(N); ii = np.maximum(i, 3)
L4 = par(c[ii], C0) ^ par(p[ii], P0) ^ par(c[ii-1], C1) ^ par(p[ii-1], P1) ^ par(c[ii-2], C2) ^ par(p[ii-2], P2) ^ par(c[ii-3], C3) ^ par(p[ii-3], P3)
e = (t ^ L4).astype(np.uint8); e[:41] = 0
print("L4'-event rate overall:", e[41:].mean())
a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1)
clear = (a == 0) & (bq == 1); clear[:41] = True
print("events on clear rows:", int(e[clear].sum()))
ci = np.where(clear)[0]
rows_set = []; rows_reset = []; lens = []; strings = []
for s, en in zip(ci[:-1], ci[1:]):
    seg = np.arange(s + 1, en)
    if len(seg) == 0: continue
    evs = seg[e[seg] == 1]
    if len(evs):
        f = evs[0]; rows_set.append(np.arange(f, en)); rows_reset.append(np.arange(s + 1, f)); lens.append(en - f)
        if len(strings) < 40 and en - f >= 6: strings.append(''.join(str(v) for v in e[f:en]))
    else: rows_reset.append(seg)
rows_set = np.concatenate(rows_set); rows_reset = np.concatenate(rows_reset)
print("reset-state rows:", len(rows_reset), "events there:", int(e[rows_reset].sum()), "| set-state rows:", len(rows_set), "event rate there:", e[rows_set].mean())
print("set-state lengths: mean %.2f, hist(1..12):" % np.mean(lens), np.bincount(np.array(lens), minlength=13)[1:13])
print("event strings inside set state (from first event to clear):")
for s in strings[:30]: print("   ", s)
# within set state: event rate by position since first event
pos = np.concatenate([np.arange(en - f) for f, en in [(r[0], r[-1] + 1) for r in np.split(rows_set, np.where(np.diff(rows_set) != 1)[0] + 1)]])
es = e[rows_set]
print("event rate by position since first event (0..10):", [round(float(es[pos == k].mean()), 3) for k in range(11)], "counts", [int((pos == k).sum()) for k in range(11)])
# exact linear test on set-state rows excluding position 0
W = 6
def X_of(ix):
    cl = [np.ones((len(ix), 1), np.uint8)]
    for w in range(W): cl += [features.bits16(c[ix - w]), features.bits16(p[ix - w])]
    cl += [t[ix - k][:, None].astype(np.uint8) for k in range(1, 9)]
    return np.concatenate(cl, 1)
for label, rows in (("set-state all", rows_set), ("set-state pos>=1", rows_set[pos >= 1]), ("set-state pos==1", rows_set[pos == 1])):
    X = X_of(rows); y = t[rows]; res = gf2.solve(X[:30000], y[:30000], "ss")
    print(f"[{label}] rows {len(rows)} rank {res['rank']} inconsistent {res['bad']} of {min(30000, len(rows)) - res['rank']}")
np.save("L4_events.npy", np.where(e == 1)[0])
