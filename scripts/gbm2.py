import sys, time, numpy as np, lightgbm as lgb
from ns10 import load_dump
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
W = int(sys.argv[2]); lags = sys.argv[3] == "lags"; NTR = int(sys.argv[4]) if len(sys.argv) > 4 else 400000
idx = np.arange(max(W, 100) + 2, N - 3); idx = idx[G[idx]]
EN = np.zeros(N, bool)
for w in range(4, 97): EN[100:] |= (p[100 - w:N - w] & 0xc980) != 0
if len(sys.argv) > 5 and sys.argv[5] == "enabled": idx = idx[EN[idx]]
split = int(0.7 * N); tr = idx[idx < split]; te = idx[idx >= split + W]
rng = np.random.default_rng(0); tr = rng.choice(tr, min(NTR, len(tr)), replace=False); te = rng.choice(te, min(200000, len(te)), replace=False)
def bits(x): return ((x[:, None] >> np.arange(16, dtype=np.uint16)) & 1).astype(np.uint8)
def feats(ix):
    cols = [np.delete(bits(c[ix + 1]), 11, axis=1)]
    for w in range(W): cols += [bits(c[ix - w]), bits(p[ix - w]), bits(m[ix - w])]
    if lags:
        cols.append(np.stack([t[ix - k] for k in range(1, W + 1)], 1).astype(np.uint8))
        cols.append(np.stack([e[ix - k] for k in range(1, W + 1)], 1).astype(np.uint8))
        cols.append(np.stack([G[ix - k] for k in range(1, W + 1)], 1).astype(np.uint8))
    return np.concatenate(cols, 1)
t0 = time.time(); Xtr = feats(tr); Xte = feats(te); ytr = e[tr]; yte = e[te]
print(f"W={W} lags={lags}: features {Xtr.shape[1]}, train {len(tr)} (rate {ytr.mean():.3f}), test {len(te)} (rate {yte.mean():.3f}); build {time.time()-t0:.0f}s", flush=True)
params = dict(objective="binary", learning_rate=0.08, num_leaves=255, min_data_in_leaf=40, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=18, max_bin=2)
dtr = lgb.Dataset(Xtr, ytr, free_raw_data=True); dte = lgb.Dataset(Xte, yte, reference=dtr)
t0 = time.time()
bst = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dte], callbacks=[lgb.early_stopping(100, verbose=False)])
pr = bst.predict(Xte, num_iteration=bst.best_iteration)
acc = float(((pr > 0.5) == yte).mean()); base = float(max(yte.mean(), 1 - yte.mean()))
eps = 1e-9; ll = float(-(yte * np.log(pr + eps) + (1 - yte) * np.log(1 - pr + eps)).mean()); h0 = float(-(yte.mean() * np.log(yte.mean()) + (1 - yte.mean()) * np.log(1 - yte.mean())))
conf = (pr < 0.02) | (pr > 0.98)
print(f"   rounds {bst.best_iteration}, {time.time()-t0:.0f}s: held-out acc {acc:.4f} (base {base:.4f}); logloss {ll:.4f} nats vs entropy {h0:.4f} -> explains {1-ll/h0:.1%}; confident rows {conf.mean():.2f} with acc {((pr[conf]>0.5)==yte[conf]).mean():.4f}", flush=True)
en = EN[te]
for lab, sel in (("enabled", en), ("not enabled", ~en)):
    if sel.sum(): print(f"   {lab}: rows {sel.sum()}, rate {yte[sel].mean():.3f}, acc {((pr[sel]>0.5)==yte[sel]).mean():.4f} (base {max(yte[sel].mean(),1-yte[sel].mean()):.4f}), logloss {float(-(yte[sel]*np.log(pr[sel]+eps)+(1-yte[sel])*np.log(1-pr[sel]+eps)).mean()):.4f} vs entropy {float(-(yte[sel].mean()*np.log(yte[sel].mean()+eps)+(1-yte[sel].mean())*np.log(1-yte[sel].mean()+eps))):.4f}")
imp = bst.feature_importance("gain"); names = ["cn"] * 15
for w in range(W): names += [f"c{w}"] * 16 + [f"p{w}"] * 16 + [f"m{w}"] * 16
if lags: names += [f"t-{k}" for k in range(1, W + 1)] + [f"e-{k}" for k in range(1, W + 1)] + [f"g-{k}" for k in range(1, W + 1)]
agg = {}
for n_, v in zip(names, imp): agg[n_] = agg.get(n_, 0) + v
tot = sum(agg.values()); top = sorted(agg.items(), key=lambda kv: -kv[1])[:14]
print("   importance by word:", ' '.join(f"{k}:{v/tot:.2f}" for k, v in top))
