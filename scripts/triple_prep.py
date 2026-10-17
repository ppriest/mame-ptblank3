import sys, numpy as np
from ns10 import load_dump
import features
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); gate = ~((a == 0) & (bq == 1))
idx = np.arange(60, N - 3); idx = idx[gate[idx]]
cols, names = [], []
for w in range(1, 6): cols.append(features.bits16(m[idx - w])); names += [f"m[i-{w}].{bb}" for bb in range(16)]
for w in range(4, 7): cols.append(features.bits16(p[idx - w])); names += [f"p[i-{w}].{bb}" for bb in range(16)]
C = np.concatenate(cols, 1).astype(np.uint8); n, K = C.shape
RW = (n + 63) // 64; pad = RW * 64 - n
Cp = np.concatenate([C, np.zeros((pad, K), np.uint8)], 0)
colbits = np.packbits(Cp.T, axis=1, bitorder='little')   # (K, RW*8 bytes)
ep = np.concatenate([e[idx], np.zeros(pad, np.uint8)]); evbits = np.packbits(ep, bitorder='little')
with open("triple.bin", "wb") as f:
    f.write(np.array([n, K], np.uint32).tobytes()); f.write(colbits.tobytes()); f.write(evbits.tobytes())
open("triple_names.txt", "w").write("\n".join(names)); print("rows", n, "bits", K)
