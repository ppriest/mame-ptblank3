import sys, numpy as np
from ns10 import *
typ, b, c, p, cnt = load_dump(sys.argv[1])
print("records", len(typ), "D", (typ==b'D').sum(), "I", (typ==b'I').sum(), "X", (typ==b'X').sum())
ii = np.where(typ==b'I')[0]
print("inits at record idx / count / iv:", [(int(i), int(cnt[i]), int(b[i])) for i in ii[:40]])
print("deactivates at:", np.where(typ==b'X')[0][:40])
d = np.where(typ==b'D')[0]
print("count range", cnt[d].min(), cnt[d].max(), "monotonic:", bool(np.all(np.diff(cnt[d])==1)))
# replay
dec = Decrypter(); ok = 0; bad = 0; firstbad = None
for k in range(len(typ)):
    t = typ[k]
    if t == b'I': dec.init(int(b[k]))
    elif t == b'D':
        pp = dec.decrypt(int(c[k]), int(b[k]))
        if pp == p[k]: ok += 1
        else:
            bad += 1
            if firstbad is None: firstbad = (k, hex(c[k]), hex(p[k]), hex(pp))
print("replay ok", ok, "bad", bad, "first bad", firstbad)
# table cross-check
tab = np.unpackbits(np.fromfile(sys.argv[2], dtype=np.uint8))
print("table bits match dump bits:", bool(np.all(tab[cnt[d]] == b[d])))
