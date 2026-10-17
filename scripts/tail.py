import sys, numpy as np
from ns10 import load_dump
import features, gf2
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
def par(x, mask): return (np.bitwise_count((x & mask).astype(np.uint16)) & 1).astype(np.uint8)
LC0 = (1<<0)|(1<<2)|(1<<3)|(1<<7)|(1<<8)|(1<<9)|(1<<10); LC1 = (1<<0)|(1<<2)
r = t.copy(); r[2:] ^= par(c[2:], LC0) ^ par(c[1:-1], LC1) ^ t[:-2]
z = (p == 0)
after = np.zeros(N, np.int64)
for i in range(N - 2, -1, -1): after[i] = (after[i+1] + 1) if z[i] else 0
starts = np.where(z & ~np.roll(z, 1))[0]; starts = starts[(starts > 100) & (starts < N - 100)]
longs = starts[after[starts] >= 48]
print("runs >= 48:", len(longs))
last1 = []
for s in longs:
    rr = r[s:s+48]; w = np.where(rr == 1)[0]; last1.append(w.max() if len(w) else -1)
last1 = np.array(last1); print("last residual offset per run: hist", np.bincount(last1 + 1, minlength=49)[:49])
# which bits are set at tail events (offset 6..40) vs. all positions at those offsets
ev = []; base = []
for s in longs:
    for k in range(6, 48):
        base.append(s + k)
        if r[s + k]: ev.append(s + k)
ev = np.array(ev); base = np.array(base); print("tail events:", len(ev), "of", len(base))
for lab, arr in (("m_i", m), ("m_i-1", np.roll(m, 1)), ("m_i-2", np.roll(m, 2)), ("m_i-3", np.roll(m, 3))):
    pe = [(float(((arr[ev] >> k) & 1).mean())) for k in range(16)]
    pb = [(float(((arr[base] >> k) & 1).mean())) for k in range(16)]
    print(f" {lab:5s} P(bit=1|event): " + ' '.join(f"{v:.2f}" for v in pe))
    print(f"       P(bit=1|all)  : " + ' '.join(f"{v:.2f}" for v in pb))
# detailed view of a bursty run: the 12th long run from before (>=80)
longs80 = starts[after[starts] >= 80]; s = longs80[11]
print(f"\nrun at {s}: offsets -6..24")
for k in range(-6, 25):
    i = s + k; print(f"  {k:3d} c={c[i]:04x} p={p[i]:04x} m={m[i]:04x} t={t[i]} r={r[i]}")
# residual right at run start as function of pre-run plaintext: fit r_{s+3} ~ linear/quad over (c,p) of s-1..s-3 and m of s..s+3
S = starts[after[starts] >= 6]
y = r[S + 3]; print("\nruns >= 6:", len(S), "P(r_{s+3}=1) =", y.mean(), " P(r_{s+4}=1) =", r[S+4].mean(), " P(r_{s+5}=1) =", r[S+5].mean())
cols = [features.bits16(c[S - k]) for k in (1, 2, 3)] + [features.bits16(p[S - k]) for k in (1, 2, 3)] + [features.bits16(m[S + k]) for k in (0, 1, 2, 3)]
names = [f"c-{k}.{bb}" for k in (1,2,3) for bb in range(16)] + [f"p-{k}.{bb}" for k in (1,2,3) for bb in range(16)] + [f"m+{k}.{bb}" for k in (0,1,2,3) for bb in range(16)]
X = np.concatenate(cols, 1); rk, piv = gf2.rank_and_pivots(X)
res = gf2.solve(np.concatenate([np.ones((len(S),1),np.uint8), X], 1), y, "rs"); print(f" r_(s+3) linear fit: rank {rk}, inconsistent {res['bad']} of {len(S)-res['rank']}")
Q, qn = gf2.quad_features(X, piv); res = gf2.solve(Q, y, "rsq"); print(f" r_(s+3) quad fit: {Q.shape[1]} monomials, inconsistent {res['bad']} of {len(S)-res['rank']}")
