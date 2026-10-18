import numpy as np, itertools
import gf2
def predict(X, sol):
    pr = np.zeros(X.shape[0], np.uint8)
    for k in sol: pr ^= X[:, k]
    return pr
def fit_test(Xfit, yfit, Xtest, ytest, tag="ht"):
    res = gf2.solve(Xfit, yfit, tag)
    pr = predict(Xtest, res['solution'])
    return res, float((pr == ytest).mean()), float((ytest == 0).mean())
