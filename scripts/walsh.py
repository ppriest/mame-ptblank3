import sys, numpy as np
from ns10 import load_dump
import features
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
def par(x, mask): return (np.bitwise_count((x & mask).astype(np.uint16)) & 1).astype(np.uint8)
LC0 = (1<<0)|(1<<2)|(1<<3)|(1<<7)|(1<<8)|(1<<9)|(1<<10); LC1 = (1<<0)|(1<<2)
r = t.copy(); r[2:] ^= par(c[2:], LC0) ^ par(c[1:-1], LC1) ^ t[:-2]
z = (p == 0); before = np.zeros(N, np.int64)
for i in range(1, N): before[i] = before[i-1] + 1 if z[i] else 0
forms = np.arange(1, 65536, dtype=np.uint16)
def search(words, y, label):
    """words: (n,) uint16 values of the candidate word; y: residual. For every nonzero linear form A over the 16 bits: events among A==0 and A==1."""
    # parity of (word & form) for all forms: use byte lookup
    pc = np.bitwise_count(np.arange(65536, dtype=np.uint16)).astype(np.uint8) & 1   # parity table of 16-bit values
    best = []
    A = pc[(words[:, None] & forms[None, :])]          # (n, 65535) parities -- memory: n*65535 bytes
    n1 = A.sum(0); e1 = (A & y[:, None]).sum(0); e0 = y.sum() - e1; n0 = len(y) - n1
    # candidate factor: zero events when A==0 (or when A==1 -> complement form)
    z0 = np.where(e0 == 0)[0]; z1 = np.where(e1 == 0)[0]
    print(f"[{label}] samples {len(y)}, events {int(y.sum())}; forms with 0 events on A==0: {len(z0)} ; on A==1: {len(z1)}")
    for q in z0[:8]: print(f"    A=0x{forms[q]:04x} bits {[bb for bb in range(16) if forms[q]>>bb & 1]}: n0={n0[q]} e0=0 | n1={n1[q]} e1={e1[q]} (rate {e1[q]/max(n1[q],1):.2f})")
    for q in z1[:8]: print(f"    A=0x{forms[q]:04x}+1 bits {[bb for bb in range(16) if forms[q]>>bb & 1]}: n1={n1[q]} e1=0 | n0={n0[q]} e0={e0[q]} (rate {e0[q]/max(n0[q],1):.2f})")
    # also the most imbalanced forms
    rate0 = e0 / np.maximum(n0, 1); rate1 = e1 / np.maximum(n1, 1)
    q = np.argmax(np.abs(rate1 - rate0)); print(f"    most separating form 0x{forms[q]:04x}: rate(A=0)={rate0[q]:.3f} rate(A=1)={rate1[q]:.3f}")
for off in (2, 3, 4, 5):
    idx = np.where((before == off + 1) & (np.arange(N) > 40) & (np.arange(N) < N - 1))[0]
    idx = idx[:6000]
    for back in (off + 1, off + 2):
        search(p[idx - back], r[idx], f"offset {off}, form over p[i-{back}]")
