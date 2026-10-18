import sys, numpy as np
from ns10 import load_dump
import features
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
idx = np.arange(80, N - 3); idx = idx[G[idx]]
lo, hi = (int(x) for x in sys.argv[2].split(",")) if len(sys.argv) > 2 else (0, 11)
dense = np.ones(len(idx), bool)
for w in range(lo, hi + 1): dense &= np.bitwise_count(p[idx - w]) >= 2
idx = idx[dense]
key = np.zeros(len(idx), np.uint64)
for w in range(lo, hi + 1): key = key * np.uint64(0x9E3779B97F4A7C15) + p[idx - w].astype(np.uint64) + np.uint64(w)
order = np.argsort(key, kind='stable'); ks = key[order]; rows = idx[order]
same = ks[1:] == ks[:-1]
I = rows[:-1][same]; J = rows[1:][same]
far = (np.abs(I.astype(np.int64) - J.astype(np.int64)) > 256) & (c[I] != c[J])
I = I[far]; J = J[far]
agree = (e[I] == e[J]); pe = e[idx].mean()
print(f"window i-{lo}..i-{hi}: far, different-cipher pairs: {len(I)}; agreement {agree.mean():.3f} (chance {1-2*pe*(1-pe):.3f}); both-event {int((e[I]&e[J]).sum())} vs independent {len(I)*pe*pe:.0f}")
cols, names = [], []
for w in range(-1, 9):
    for nm, arr in (("c", c), ("m", m), ("p", p)):
        if w == -1 and nm != "c": continue
        cols.append((features.bits16(arr[I - w]) == features.bits16(arr[J - w])).astype(np.uint8)); names += [f"{nm}[i-{w}].{bb}" for bb in range(16)]
cols.append(np.stack([(t[I - k] == t[J - k]) for k in range(1, 9)], 1).astype(np.uint8)); names += [f"t-{k}" for k in range(1, 9)]
EQ = np.concatenate(cols, 1)
out = []
for k in range(EQ.shape[1]):
    eq = EQ[:, k] == 1
    if eq.sum() > 200 and (~eq).sum() > 200: out.append((agree[eq].mean() - agree[~eq].mean(), names[k], agree[eq].mean(), agree[~eq].mean(), int(eq.sum())))
out.sort(reverse=True)
print("bits whose equality across the pair most raises agreement (diff, bit, agree|equal, agree|differ, n_equal):")
for o in out[:14]: print(f"   {o[0]:+.3f}  {o[1]:12s} {o[2]:.3f} {o[3]:.3f} {o[4]}")
print("most negative:", [(round(o[0], 3), o[1]) for o in out[-4:]])
# agreement vs number of matching mask bits in m[i-2] and m[i-3]
for w in (2, 3, 1, 4):
    nm_ = np.bitwise_count((m[I - w] ^ m[J - w]).astype(np.uint16))
    print(f"  agreement vs #differing bits in m[i-{w}]: " + ' '.join(f"{q}:{agree[nm_==q].mean():.2f}({(nm_==q).sum()})" for q in range(0, 17) if (nm_ == q).sum() > 50))
