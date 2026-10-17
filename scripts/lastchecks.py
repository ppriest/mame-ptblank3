import sys, numpy as np
from ns10 import load_dump
import features
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); gate = ~((a == 0) & (bq == 1))
z = (p == 0); before = np.zeros(N, np.int64)
for k in range(1, N): before[k] = before[k-1] + 1 if z[k] else 0
idx = np.arange(60, N - 3); idx = idx[gate[idx] & gate[idx - 1]]
off = idx[(e[idx - 1] == 0) & (e[idx - 2] == 0) & (e[idx - 3] == 0) & gate[idx - 2] & gate[idx - 3]]
def coll(rows, keyfn, label):
    key = keyfn(rows); order = np.argsort(key, kind='stable'); ks = key[order]; es = e[rows][order]
    same = ks[1:] == ks[:-1]; conf = same & (es[1:] != es[:-1])
    # how many of the identical pairs also have identical m4,m5 ?
    print(f"   {label:18s}: pairs {int(same.sum()):7d}, conflicts {int(conf.sum()):5d} ({conf.sum()/max(same.sum(),1):.3f})")
def K(*cols):
    def f(rows):
        key = np.zeros(len(rows), np.uint64)
        for arr, w in cols: key = key * np.uint64(0x9E3779B97F4A7C15) + arr[rows - w].astype(np.uint64)
        return key
    return f
print("turn-on determinism, state-repetition check:")
coll(off, K((c, 4)), "c4"); coll(off, K((c, 4), (c, 5)), "c4,c5"); coll(off, K((m, 4), (m, 5)), "m4,m5"); coll(off, K((m, 4)), "m4"); coll(off, K((c, 4), (p, 4)), "c4,p4 (=m4,p4)"); coll(off, K((m, 3), (m, 4), (m, 5)), "m3,m4,m5"); coll(off, K((c, 1), (c, 2)), "c1,c2 (control)")
# lone ffff impulses: what differs between firing and non-firing?
S = np.where((~z) & (np.roll(before, 1) >= 12))[0]; S = S[(S > 100) & (S < N - 40)]
rows = S + 4; ok = gate[rows] & (before[rows] == 4) & (p[S] == 0xffff)
print(f"\nlone 0xffff impulses: {ok.sum()}; firing at depth 4: {int(e[rows][ok].sum())}")
for s in S[ok]:
    i4 = s + 4
    print(f"  s={s:7d} e={e[i4]}  c_s={c[s]:04x} m_s={m[s]:04x} | m_{{s+1}}={m[s+1]:04x} m_{{s+2}}={m[s+2]:04x} m_{{s+3}}={m[s+3]:04x} m_{{s+4}}={m[s+4]:04x} | t[s-2..s+4]={''.join(str(v) for v in t[s-2:s+5])}  pre-run p: {p[s-1]:04x} {p[s-2]:04x} (zeros before s: {before[s-1]})")
