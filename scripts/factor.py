"""Turn a quadratic GF(2) solution (list of monomials over pivot columns) into linear part + products."""
import numpy as np
def gf2_rank(M):
    M = M.copy() % 2; r = 0; rows, cols = M.shape
    for cidx in range(cols):
        piv = np.where(M[r:, cidx] == 1)[0]
        if len(piv) == 0: continue
        pr = r + piv[0]; M[[r, pr]] = M[[pr, r]]
        for i in range(rows):
            if i != r and M[i, cidx]: M[i] ^= M[r]
        r += 1
        if r == rows: break
    return r
def describe(solution, qnames, n):
    const = 0; lin = np.zeros(n, np.uint8); Mq = np.zeros((n, n), np.uint8)
    for k in solution:
        mono = qnames[k]
        if len(mono) == 0: const ^= 1
        elif len(mono) == 1: lin[mono[0]] ^= 1
        else: Mq[mono[0], mono[1]] ^= 1
    alt = (Mq ^ Mq.T) % 2
    return const, lin, Mq, gf2_rank(alt)
def factor_rank2(Mq):
    """If the alternating form has rank 2, return linear forms (A,B) over the n variables with sum_{i<j} Mq[i,j] x_i x_j == A*B (mod linear terms)."""
    n = Mq.shape[0]; alt = (Mq ^ Mq.T) % 2
    rows = [i for i in range(n) if alt[i].any()]
    if not rows: return None
    a = alt[rows[0]].copy()          # a = row of first variable with nonzero entries: this is "B" when x_i is in A
    # the alternating form of rank 2 equals u ∧ v; pick u = e_i-row relation: find v s.t. alt = u v^T + v u^T
    i = rows[0]; u = np.zeros(n, np.uint8); 
    # alt[i] = v (since u[i]=1 and v[i]=0 w.l.o.g.) ; then u = alt[j] for some j with v[j]=1
    v = a; j = int(np.where(v == 1)[0][0]); u = alt[j].copy()
    chk = (np.outer(u, v) ^ np.outer(v, u)) % 2
    assert np.array_equal(chk, alt), "not a pure rank-2 product"
    return u, v
