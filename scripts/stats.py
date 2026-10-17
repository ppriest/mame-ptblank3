import numpy as np, sys
raw = np.fromfile(sys.argv[1], dtype=np.uint8)
bits = np.unpackbits(raw)          # MSB first, matches MAME's 7-(count%8)
n = len(bits); print("words", n, "ones", bits.sum(), "bias", bits.mean())
b = bits.astype(np.int8)*2-1
print("autocorr lag: value (random ~ +/-%.4f)" % (2/np.sqrt(n)))
for lag in list(range(1,17))+[32,64,128,255,256,257,512,1024,4096,8191,8192,8193]:
    print("  %5d %+.4f" % (lag, (b[:-lag]*b[lag:]).mean()))
print("mean of bit by index mod 2^k (min/max over residues, chi2):")
for k in range(1,14):
    m = 1<<k; idx = np.arange(n) % m
    s = np.bincount(idx, weights=bits, minlength=m); c = np.bincount(idx, minlength=m)
    p = s/c; e = c*bits.mean()
    chi2 = (((s-e)**2)/(e*(1-bits.mean()))).sum()
    print("  k=%2d  min %.4f max %.4f  chi2 %.1f (dof %d)" % (k, p.min(), p.max(), chi2, m-1))
# per-position within page profile for k=8
idx = np.arange(n) % 256
p = np.bincount(idx, weights=bits, minlength=256)/np.bincount(idx, minlength=256)
print("pos-in-page profile (first 32):", np.round(p[:32],3))
# runs
d = np.diff(np.concatenate([[0],bits,[0]]))
starts = np.where(d==1)[0]; ends = np.where(d==-1)[0]
runs = ends-starts
print("runs of 1: count", len(runs), "max", runs.max(), "hist", np.bincount(runs)[:12])
zruns = starts[1:]-ends[:-1]
print("runs of 0: max", zruns.max(), "hist", np.bincount(zruns)[:12])
print("first 64 bits:", ''.join(map(str,bits[:64])))
# per-block (8192 words) ones count
blk = bits[:n//8192*8192].reshape(-1,8192).sum(1)
print("per-block ones: min %d max %d mean %.1f std %.1f (binomial std %.1f)" % (blk.min(), blk.max(), blk.mean(), blk.std(), np.sqrt(8192*bits.mean()*(1-bits.mean()))))
print("blocks with lowest ones:", np.argsort(blk)[:10], blk[np.argsort(blk)[:10]])
