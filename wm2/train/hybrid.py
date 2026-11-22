#!/usr/bin/env python3
"""하이브리드 인식: 학습형 확률 지도 + 장 사이 반복 확인.
사용: python3 hybrid.py <maps폴더> <tier폴더> <출력.tsv> [--th 0.5] [--ncc 0.6] [--sup 0.5]
후보 = 각 장의 확률 지도 덩어리. 다른 장의 지도에서 같은 모양(정규화 상관)이 얼마나 나오는지 세어, 절반 이상에서 나오면 워터마크로 인정."""
import os, sys, math, time
import numpy as np
from scipy import ndimage, signal

def comps(p, th, merge=3, min_area=30):
    m = p > th; d = ndimage.binary_dilation(m, iterations=merge); lab, n = ndimage.label(d); out = []
    for i, sl in enumerate(ndimage.find_objects(lab)):
        c = (lab[sl] == i + 1) & m[sl]
        if c.sum() < min_area: continue
        ys, xs = np.where(c); y0, y1, x0, x1 = sl[0].start + ys.min(), sl[0].start + ys.max() + 1, sl[1].start + xs.min(), sl[1].start + xs.max() + 1
        out.append((x0, y0, x1 - x0, y1 - y0, float(p[sl][c].mean())))
    return out

def ncc_search(tpl, p):
    th, tw = tpl.shape
    if th > p.shape[0] or tw > p.shape[1]: return 0., (0, 0)
    t = tpl - tpl.mean(); tn = math.sqrt((t * t).sum()) + 1e-6
    corr = signal.fftconvolve(p, t[::-1, ::-1], mode='valid')
    ones = np.ones_like(tpl); s1 = signal.fftconvolve(p, ones[::-1, ::-1], mode='valid'); s2 = signal.fftconvolve(p * p, ones[::-1, ::-1], mode='valid')
    n = th * tw; var = np.maximum(s2 - s1 * s1 / n, 1e-6); ncc = corr / (tn * np.sqrt(var))
    iy, ix = np.unravel_index(np.argmax(ncc), ncc.shape); return float(ncc[iy, ix]), (int(ix), int(iy))

def site(maps, th, ncc_t, sup_f, sc=.5):
    files = sorted(k for k in maps if not k.endswith('.size')); P = {f: maps[f].astype(np.float32) for f in files}; S = {f: maps[f + '.size'] for f in files}
    res = {f: [] for f in files}
    byw = {}
    for f in files: byw.setdefault(int(S[f][0]), []).append(f)
    for w, grp in byw.items():
        K = len(grp)
        if K < 3: continue
        need = max(2, math.ceil(sup_f * (K - 1)))
        cands = []
        for f in grp:
            for (x, y, bw, bh, sc_) in comps(P[f], th):
                if bw * bh > 0.12 * P[f].size: continue
                tpl = P[f][y:y + bh, x:x + bw]; sup = []; 
                for g in grp:
                    if g == f: continue
                    v, (px, py) = ncc_search(tpl, P[g])
                    if v >= ncc_t: sup.append((g, px, py, v))
                if len(sup) >= need: cands.append((sum(s[3] for s in sup) / len(sup) * sc_, f, x, y, bw, bh, sup))
        cands.sort(key=lambda c: -c[0]); taken = []
        for sc_, f, x, y, bw, bh, sup in cands:
            if any(t[0] == f and abs(t[1] - x) < max(8, bw // 2) and abs(t[2] - y) < max(8, bh // 2) for t in taken): continue
            taken.append((f, x, y)); res[f].append((int(x / sc), int(y / sc), int(bw / sc), int(bh / sc), 'mark', sc_))
            for g, px, py, v in sup: taken.append((g, px, py)); res[g].append((int(px / sc), int(py / sc), int(bw / sc), int(bh / sc), 'mark', sc_))
    return res

if __name__ == '__main__':
    md, root, outp = sys.argv[1:4]
    arg = lambda n, d: float(sys.argv[sys.argv.index(n) + 1]) if n in sys.argv else d
    th, ncc_t, sup_f = arg('--th', .5), arg('--ncc', .6), arg('--sup', .5)
    lines = []; t0 = time.time()
    for fn in sorted(os.listdir(md)):
        if not fn.endswith('.npz'): continue
        s = fn[:-4]; maps = dict(np.load(os.path.join(md, fn))); res = site(maps, th, ncc_t, sup_f)
        for f in sorted(res):
            if not res[f]: lines.append('%s\t%s\tNONE' % (s, f))
            for b in res[f]: lines.append('%s\t%s\t%d\t%d\t%d\t%d\t%s\t%.3f' % (s, f, *b))
    open(outp, 'w').write('\n'.join(lines) + '\n'); print('완료 %.0fs' % (time.time() - t0))
