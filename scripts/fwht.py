import numpy as np
def fwht(a):
    a = a.astype(np.int64).copy(); h = 1; n = len(a)
    while h < n:
        a = a.reshape(-1, 2, h); x, y = a[:, 0, :].copy(), a[:, 1, :].copy(); a[:, 0, :] = x + y; a[:, 1, :] = x - y; a = a.reshape(n); h *= 2
    return a
def best_form(words, y, exclude_zero=True):
    """Best linear form A (16-bit mask) and constant maximizing agreement of A(word)^const with y. Returns (mask, const, agreement)."""
    words = np.asarray(words, np.uint16).astype(np.int64); y = np.asarray(y).astype(np.int64)
    s = np.bincount(words, weights=(1 - 2 * y), minlength=65536)   # sum of (+1 for y=0, -1 for y=1)
    W = fwht(s)   # W[A] = sum_v s[v] * (-1)^{A.v} = #(A(v)=y) - #(A(v)!=y)
    n = len(y)
    if exclude_zero: W0 = W.copy(); W0[0] = 0
    else: W0 = W
    a = int(np.argmax(np.abs(W0))); const = 1 if W0[a] < 0 else 0
    return a, const, (n + abs(W0[a])) / (2 * n)
def parity16(words, mask):
    return (np.bitwise_count((np.asarray(words, np.uint16) & np.uint16(mask))) & 1).astype(np.uint8)
