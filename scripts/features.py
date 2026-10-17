import numpy as np
def bits16(a):  # (N,) uint16 -> (N,16) uint8, bit 0 first
    a = np.asarray(a, np.uint16)
    return ((a[:, None] >> np.arange(16, dtype=np.uint16)) & 1).astype(np.uint8)
def window(c, p, t, i0, i1, w=3, extra=()):
    """Samples i in [i0,i1): features = c_{i-k}, p_{i-k} for k in 0..w-1 (word i is the most recent decrypted word,
    t_i is the bit folded into the mask of word i+1). extra may contain 'cnext' (c_{i+1}), 'tprev' (t_{i-1..i-3})."""
    idx = np.arange(i0, i1)
    cols, names = [], []
    for k in range(w):
        cols.append(bits16(c[idx - k])); names += [f"c{k}.{b}" for b in range(16)]
        cols.append(bits16(p[idx - k])); names += [f"p{k}.{b}" for b in range(16)]
    if 'cnext' in extra:
        cols.append(bits16(c[np.minimum(idx + 1, len(c) - 1)])); names += [f"cn.{b}" for b in range(16)]
    if 'tprev' in extra:
        for k in (1, 2, 3):
            cols.append(t[idx - k][:, None].astype(np.uint8)); names.append(f"t-{k}")
    return np.concatenate(cols, 1), names, t[idx].astype(np.uint8)

def extra_cols(idx, iv=None, pos=None, nposbits=0):
    """Optional per-sample side inputs: 4 IV bits of the segment, low nposbits of the position since init."""
    cols, names = [], []
    if iv is not None:
        v = iv[idx].astype(np.uint16)
        cols.append(((v[:, None] >> np.arange(4, dtype=np.uint16)) & 1).astype(np.uint8)); names += [f"iv.{b}" for b in range(4)]
    if pos is not None and nposbits:
        v = pos[idx].astype(np.uint64)
        cols.append(((v[:, None] >> np.arange(nposbits, dtype=np.uint64)) & 1).astype(np.uint8)); names += [f"pos.{b}" for b in range(nposbits)]
    return (np.concatenate(cols, 1) if cols else np.zeros((len(idx), 0), np.uint8)), names
