import sys, time, numpy as np, torch, torch.nn as nn
from ns10 import load_dump
torch.set_num_threads(18); torch.manual_seed(0)
typ, b, c, p, cnt = load_dump(sys.argv[1]); d = typ == b'D'
c = c[d]; p = p[d]; t = b[d]; N = len(c); m = (c ^ p).astype(np.uint16)
ev = np.load("../data/L4_events.npy"); e = np.zeros(N, np.uint8); e[ev] = 1
i = np.arange(N); a = ((m[np.maximum(i - 2, 0)] >> 0) & 1); bq = ((m[np.maximum(i - 3, 0)] >> 0) & 1); G = ~((a == 0) & (bq == 1))
EN = np.zeros(N, bool)
for w in range(4, 97): EN[100:] |= (p[100 - w:N - w] & 0xc980) != 0
H = int(sys.argv[2]); EPOCHS = int(sys.argv[3])
def bits(x): return ((x[:, None] >> np.arange(16, dtype=np.uint16)) & 1).astype(np.float32)
tp = np.concatenate([[0], t[:-1]]).astype(np.float32); ep = np.concatenate([[0], e[:-1]]).astype(np.float32)
X = np.concatenate([bits(c), bits(p), bits(m), tp[:, None], ep[:, None], G[:, None].astype(np.float32)], 1) * 2 - 1
Y = e.astype(np.float32); M = G.astype(np.float32)
L, WARM = 640, 128
split = int(0.7 * N)
def chunks(lo, hi, step):
    starts = np.arange(lo, hi - L, step); return starts
tr_starts = chunks(0, split, L - WARM); te_starts = chunks(split, N, L - WARM)
Xt = torch.from_numpy(X); Yt = torch.from_numpy(Y); Mt = torch.from_numpy(M)
def batch(starts):
    ix = torch.from_numpy(starts[:, None] + np.arange(L)[None, :])
    return Xt[ix], Yt[ix], Mt[ix]
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.inp = nn.Linear(X.shape[1], 128); s.gru = nn.GRU(128, H, num_layers=2, batch_first=True); s.out = nn.Sequential(nn.Linear(H + 128, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(s, x):
        z = torch.relu(s.inp(x)); h, _ = s.gru(z); return s.out(torch.cat([h, z], -1)).squeeze(-1)
net = Net(); opt = torch.optim.Adam(net.parameters(), lr=2e-3); lossf = nn.BCEWithLogitsLoss(reduction='none')
def evaluate():
    net.eval(); preds = []; ys = []; ms = []; ens = []
    with torch.no_grad():
        for k in range(0, len(te_starts), 256):
            st = te_starts[k:k + 256]; x, y, mk = batch(st); pr = torch.sigmoid(net(x))[:, WARM:]
            preds.append(pr.numpy().ravel()); ys.append(y[:, WARM:].numpy().ravel()); ms.append(mk[:, WARM:].numpy().ravel())
            ens.append(EN[st[:, None] + np.arange(WARM, L)[None, :]].ravel())
    net.train(); pr = np.concatenate(preds); y = np.concatenate(ys); mk = np.concatenate(ms) > 0; en = np.concatenate(ens)
    out = {}
    for lab, sel in (("gate-open", mk), ("enabled", mk & en)):
        yy = y[sel]; pp = pr[sel]; out[lab] = (float(((pp > 0.5) == yy).mean()), float(1 - yy.mean()), float(-(yy * np.log(pp + 1e-9) + (1 - yy) * np.log(1 - pp + 1e-9)).mean()), float(-(yy.mean() * np.log(yy.mean()) + (1 - yy.mean()) * np.log(1 - yy.mean()))))
    return out
t0 = time.time()
for ep_ in range(EPOCHS):
    perm = np.random.default_rng(ep_).permutation(tr_starts); tot = 0; nb = 0
    for k in range(0, len(perm), 64):
        x, y, mk = batch(perm[k:k + 64]); lo = net(x)
        l = (lossf(lo, y) * mk)[:, WARM:].sum() / mk[:, WARM:].sum().clamp(min=1)
        opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); tot += float(l.detach()); nb += 1
    r = evaluate()
    print(f"epoch {ep_+1} ({time.time()-t0:.0f}s) train loss {tot/nb:.4f} | held-out gate-open acc {r['gate-open'][0]:.4f} (base {r['gate-open'][1]:.4f}) | enabled acc {r['enabled'][0]:.4f} (base {r['enabled'][1]:.4f}) logloss {r['enabled'][2]:.4f} vs entropy {r['enabled'][3]:.4f}", flush=True)
