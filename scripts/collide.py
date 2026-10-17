import sys, numpy as np
from ns10 import load_dump
import pages
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c)
pg = pages.page_sequence(sys.argv[2]); print("pages", len(pg), "first", [hex(x) for x in pg[:3]], "last", [hex(x) for x in pg[-3:]])
assert len(pg) * 256 == N
page_of = np.repeat(pg, 256); posinpage = np.arange(N) % 256
# how many words back does history match? compute for pairs with identical 3-word state
def state_key(w):
    key = np.zeros(N, np.uint64)
    for k in range(w):
        idx = np.maximum(np.arange(N) - k, 0)
        key = key * np.uint64(0x9E3779B97F4A7C15) + c[idx].astype(np.uint64) + (p[idx].astype(np.uint64) << np.uint64(16)) + np.uint64(k)
    return key
key = state_key(3)
sel = np.arange(8, N - 1)
order = sel[np.argsort(key[sel], kind='stable')]; ks = key[order]
same = ks[1:] == ks[:-1]
i = order[:-1][same]; j = order[1:][same]
conf = t[i] != t[j]
print("w=3 collision pairs", len(i), "conflicts", conf.sum())
# depth of matching history for each pair
depth = np.full(len(i), 3)
for k in range(3, 40):
    ok = (c[i - k] == c[j - k]) & (p[i - k] == p[j - k]) & (depth == k)
    depth[ok] = k + 1
for k in range(3, 20):
    m = depth >= k
    if m.sum(): print(f"  history identical >= {k:2d} words: pairs {m.sum():5d}, conflict rate {conf[m].mean():.3f}")
m = depth >= 40; print(f"  history identical >= 40 words: pairs {m.sum()}, conflicts {conf[m].sum()}")
# zero-plaintext context?
zero = (p[i] == 0) & (p[i-1] == 0) & (p[i-2] == 0)
print("  pairs in zero plaintext:", zero.sum(), "conflict rate", conf[zero].mean() if zero.sum() else None, "| nonzero:", (~zero).sum(), conf[~zero].mean() if (~zero).sum() else None)
delta = (j - i)
print("conflict rate by (j-i) mod 2^k == 0:")
for k in range(1, 14):
    m = (delta % (1 << k)) == 0
    if m.sum(): print(f"  mod {1<<k:5d}: pairs {m.sum():5d} conflict {conf[m].mean():.3f}   | not: pairs {(~m).sum():5d} conflict {conf[~m].mean():.3f}")
m = posinpage[i] == posinpage[j]; print(f"same pos in page: pairs {m.sum()} conflict {conf[m].mean() if m.sum() else None}; diff pos: {(~m).sum()} conflict {conf[~m].mean():.3f}")
m = page_of[i] == page_of[j]; print(f"same NAND page: pairs {m.sum()} conflict {conf[m].mean() if m.sum() else None}")
m = (page_of[i] & 0xff) == (page_of[j] & 0xff); print(f"same row byte: pairs {m.sum()} conflict {conf[m].mean() if m.sum() else None}")
m = (page_of[i] >> 8) == (page_of[j] >> 8); print(f"same page byte: pairs {m.sum()} conflict {conf[m].mean() if m.sum() else None}")
m = (page_of[i] >> 5) == (page_of[j] >> 5); print(f"same block: pairs {m.sum()} conflict {conf[m].mean() if m.sum() else None}")
# deep-history conflicts in detail
m = np.where((depth >= 12) & conf)[0][:6]
for q in m:
    a, bb = i[q], j[q]
    print(f"\nconflict pair i={a} j={bb} depth {depth[q]} pos {posinpage[a]},{posinpage[bb]} page {page_of[a]:04x},{page_of[bb]:04x}")
    for x in (a, bb):
        print("   c:", ' '.join('%04x' % v for v in c[x-5:x+3]), " p:", ' '.join('%04x' % v for v in p[x-5:x+3]), " t:", ''.join(str(v) for v in t[x-5:x+3]))
