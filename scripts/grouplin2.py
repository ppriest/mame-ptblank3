import sys, numpy as np
from ns10 import load_dump
import gf2, heldfit
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
PW, MW, MIN = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
idx = np.arange(100, N - 3); idx = idx[G[idx]]
key = np.zeros(len(idx), np.uint64)
for w in range(PW): key = key * np.uint64(0x9E3779B97F4A7C15) + p[idx - w].astype(np.uint64) + np.uint64(w)
order = np.argsort(key, kind='stable'); ks = key[order]; rows = idx[order]
bounds = np.concatenate([[0], np.where(ks[1:] != ks[:-1])[0] + 1, [len(ks)]]); sizes = np.diff(bounds)
def bits(x): return ((x[:, None] >> np.arange(16, dtype=np.uint16)) & 1).astype(np.uint8)
names = ['1'] + [f"m{w}.{k}" for w in range(MW) for k in range(16)]
def X_of(r): return np.concatenate([np.ones((len(r), 1), np.uint8), np.delete(bits(c[r + 1]), 11, axis=1), np.stack([t[r - k] for k in range(1, 5)], 1).astype(np.uint8)] + [bits(m[r - w]) for w in range(MW)], 1)
tot = ok = base_ok = 0; groups = []
for g in np.where(sizes >= MIN)[0]:
    r = rows[bounds[g]:bounds[g + 1]]
    if sum(1 for w in range(PW) if p[r[0] - w] != 0) < PW // 2: continue
    y = e[r]
    if y.mean() < 0.05 or y.mean() > 0.95: continue
    X = X_of(r)
    # distinct mask states
    keyr = np.packbits(X, axis=1); distinct = len({row.tobytes() for row in keyr})
    # split by distinct states (no state in both halves)
    st = [row.tobytes() for row in keyr]; uniq = list(dict.fromkeys(st)); rng = np.random.default_rng(int(g)); rng.shuffle(uniq)
    fitset = set(uniq[:int(0.7 * len(uniq))]); f = np.array([q for q, s in enumerate(st) if s in fitset]); te = np.array([q for q, s in enumerate(st) if s not in fitset])
    res, acc, base = heldfit.fit_test(X[f], y[f], X[te], y[te], "gl2")
    groups.append((len(r), distinct, y.mean(), res['rank'], acc, base, res['solution'], r[0]))
    tot += len(te); ok += acc * len(te); base_ok += base * len(te)
print(f"plain window {PW}, mask words 0..{MW-1} only (no cnext, no t lags), groups >= {MIN} with rate 0.05..0.95: {len(groups)}")
for gr in sorted(groups, key=lambda x: -x[0])[:20]:
    print(f"   size {gr[0]:5d} distinct mask states {gr[1]:5d} rate {gr[2]:.2f} rank {gr[3]:3d}  held-out (novel states) acc {gr[4]:.3f} base {gr[5]:.3f}  terms: {[names[k] for k in gr[6]][:14]}")
if tot: print(f"pooled held-out acc on novel mask states: {ok/tot:.4f} vs base {base_ok/tot:.4f} over {tot} rows")
