#!/usr/bin/env python3
"""후보 채택 규칙(로지스틱 회귀) 학습: python3 train_accept.py --train core:tag,wild:tag --test val:tag
후보 특징량 → 정답 여부(TP). 결과: 가중치(Kotlin 에 옮길 값)와 임계값별 정밀도/재현."""
import sys, os, json, math, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from cand_stats import load

FEATS = ['ls', 'lh', 'dens', 'dom', 'meanR', 'pres', 'lw', 'lh2', 'asp']
def feat(r):
    w, h = max(r['box'][2], 1), max(r['box'][3], 1)
    return [math.log1p(r['score']), math.log1p(r['hits']), r['dens'], r['dom'], r['meanR'], r['pres'] / r['K'], math.log(w), math.log(h), math.log(w / h)]

def build(spec, only_ok=True):
    X = []; y = []; meta = []
    for a in spec.split(','):
        t, g = a.split(':')
        for r in load(t, g):
            if r['why'] != '-' or r['exp'] == 'ambiguous': continue
            X.append(feat(r)); y.append(1 if r['tp'] else 0); meta.append(r)
    return np.array(X), np.array(y), meta

def fit(X, y, lam=1.0, it=50):
    mu = X.mean(0); sd = X.std(0) + 1e-9; Z = (X - mu) / sd; Z1 = np.hstack([Z, np.ones((len(Z), 1))])
    w = np.zeros(Z1.shape[1])
    cw = np.where(y == 1, len(y) / (2 * max(1, y.sum())), len(y) / (2 * max(1, (1 - y).sum())))   # 클래스 균형
    for _ in range(it):
        p = 1 / (1 + np.exp(-Z1 @ w)); g = Z1.T @ (cw * (p - y)) + lam * np.r_[w[:-1], 0]
        Hm = (Z1 * (cw * p * (1 - p))[:, None]).T @ Z1 + lam * np.eye(len(w)); w -= np.linalg.solve(Hm, g)
    return w, mu, sd

def predict(X, m):
    w, mu, sd = m; Z = (X - mu) / sd; return 1 / (1 + np.exp(-(np.hstack([Z, np.ones((len(Z), 1))]) @ w)))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--train'); ap.add_argument('--test'); a = ap.parse_args()
    X, y, meta = build(a.train); print('학습 후보 %d (TP %d)' % (len(y), y.sum()))
    m = fit(X, y)
    print('특징 가중치(표준화):', {f: round(float(v), 2) for f, v in zip(FEATS + ['bias'], m[0])})
    for name, spec in (('학습', a.train), ('시험', a.test)):
        if not spec: continue
        Xt, yt, mt = build(spec); p = predict(Xt, m)
        print('[%s] 후보 %d TP %d' % (name, len(yt), yt.sum()))
        for th in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
            sel = p >= th; tp = int((sel & (yt == 1)).sum()); fp = int((sel & (yt == 0)).sum()); neg = sum(1 for i in np.where(sel & (yt == 0))[0] if mt[i]['exp'] == 'none')
            print('  p>=%.2f: 채택 %d (TP %d, FP %d, 음성사이트 FP %d)' % (th, sel.sum(), tp, fp, neg))
    json.dump(dict(feats=FEATS, w=m[0].tolist(), mu=m[1].tolist(), sd=m[2].tolist()), open(os.path.expanduser('~/wm2-work/accept_model.json'), 'w'))
