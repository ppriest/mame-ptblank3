import numpy as np, subprocess, os, itertools
HERE = os.path.dirname(os.path.abspath(__file__))
def solve(X, y, tag="sys"):
    """X: (N,F) uint8 0/1; y: (N,) 0/1. Returns dict(rank, bad, pivots, solution(list of col idx))."""
    X = np.asarray(X, np.uint8); y = np.asarray(y, np.uint8)
    N, F = X.shape; W = (F + 1 + 63) // 64
    A = np.zeros((N, W * 64), np.uint8); A[:, :F] = X; A[:, F] = y
    packed = np.packbits(A, axis=1, bitorder='little').view(np.uint64)
    path = os.path.join(HERE, tag + ".bin")
    with open(path, "wb") as f:
        f.write(np.array([N, F], np.uint32).tobytes()); f.write(packed.tobytes())
    out = subprocess.run([os.path.join(HERE, "gf2solve.exe"), path], capture_output=True, text=True, check=True).stdout.split("\n")
    hdr = out[0].split(); piv = list(map(int, out[1].split()[1:])); sol = list(map(int, out[2].split()[1:]))
    return dict(rank=int(hdr[5]), bad=int(hdr[7]), pivots=piv, solution=sol)
def quad_features(L, pivots=None):
    """L: (N,n) linear features. Returns (N, 1+n+n(n-1)/2) [const, linear, products] over chosen columns and the monomial list."""
    if pivots is not None: L = L[:, pivots]
    N, n = L.shape
    cols = [np.ones(N, np.uint8)] + [L[:, i] for i in range(n)]
    names = [()] + [(i,) for i in range(n)]
    for i, j in itertools.combinations(range(n), 2):
        cols.append(L[:, i] & L[:, j]); names.append((i, j))
    return np.stack(cols, 1), names
def rank_and_pivots(L):
    r = solve(L, np.zeros(L.shape[0], np.uint8), "rank")
    return r["rank"], r["pivots"]
