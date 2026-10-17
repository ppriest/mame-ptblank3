import sys, numpy as np
from ns10 import load_dump
import features, gf2
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("nl_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
a = ((m[np.maximum(np.arange(N) - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(np.arange(N) - 3, 0)] >> 0) & 1)
clear = (a == 0) & (bq == 1); clear[:41] = True
W = 6
def X_of(ix):
    cl = [np.ones((len(ix), 1), np.uint8)]
    for w in range(W): cl += [features.bits16(c[ix - w]), features.bits16(p[ix - w])]
    cl += [t[ix - k][:, None].astype(np.uint8) for k in range(1, 9)]
    return np.concatenate(cl, 1)
names = ['1'] + [f"{nm}{w}.{bb}" for w in range(W) for nm in ("c", "p") for bb in range(16)] + [f"t-{k}" for k in range(1, 9)]
# intervals between clears
ci = np.where(clear)[0]
sure1 = []; before_first = []
for s, en in zip(ci[:-1], ci[1:]):
    seg = np.arange(s + 1, en)
    if len(seg) == 0: continue
    evs = seg[e[seg] == 1]
    if len(evs): 
        f = evs[0]; sure1.append(np.arange(f, en)); before_first.append(np.arange(s + 1, f))
    else: before_first.append(seg)
sure1 = np.concatenate(sure1); before_first = np.concatenate(before_first)
print("rows surely h=1:", len(sure1), " rows before first event (h=0 or set-but-quiet):", len(before_first), " clear rows:", clear.sum())
for label, rows in (("h=1 (from first event to clear)", sure1), ("before first event", before_first)):
    X = X_of(rows); y = t[rows]
    res = gf2.solve(X[:40000], y[:40000], "sr")
    pred = np.zeros(len(rows), np.uint8)
    for k in res['solution']: pred ^= X[:, k]
    print(f"[{label}] rank {res['rank']} inconsistent {res['bad']} of {40000 - res['rank']}; mismatches on all rows {int((pred != y).sum())} of {len(rows)}; event rate on these rows {e[rows].mean():.3f}")
    if res['bad'] == 0: print("   solution:", [names[k] for k in res['solution']])
# event rate as function of position within interval relative to first event
