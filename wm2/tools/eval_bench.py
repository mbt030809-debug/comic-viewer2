#!/usr/bin/env python3
"""wm2 평가: python3 eval_bench.py <truth.json> <예측.tsv> [--fail 실패목록.txt] [--cats a,b]
맞음 = 정답 상자와 IoU>=0.5. 안 보이는 워터마크(대비<3)는 인식률 계산에서 뺀다.
집계: 카테고리별 인식률(맞음/보이는 정답), 엉뚱한 상자(정답 이미지에서 안 맞은 예측), 음성 이미지 오인식률."""
import sys, json, argparse
from collections import defaultdict


def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx)); iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    i = ix * iy; u = aw * ah + bw * bh - i
    return i / u if u > 0 else 0


def load_pred(path):
    P = defaultdict(list); seen = set()
    for l in open(path, encoding='utf-8'):
        l = l.rstrip('\n')
        if not l or l[0] in '#@': continue
        f = l.split('\t')
        seen.add((f[0], f[1]))
        if f[2] == 'NONE': continue
        P[(f[0], f[1])].append((int(f[2]), int(f[3]), int(f[4]), int(f[5]), f[6], float(f[7])))
    return P, seen


def evaluate(truth, P, seen, cats=None, thr=0.5):
    R = defaultdict(lambda: defaultdict(float)); fails = []
    for s in truth['sites']:
        if cats and s['cat'] not in cats: continue
        c = s['cat']; ex = s['expected']; r = R[c]; r['sites'] += 1
        site_tp = 0; site_vis = 0; site_pred = 0
        for im in s['images']:
            key = (s['id'], im['file'])
            if key not in seen: continue
            r['images'] += 1
            preds = [p[:4] for p in P.get(key, [])]
            tr = [m for m in im['marks'] if m['visible']]
            used = set(); tp = 0
            for m in tr:
                best = -1; bi = 0
                for j, p in enumerate(preds):
                    if j in used: continue
                    v = iou(m['box'], p)
                    if v > bi: bi = v; best = j
                if best >= 0 and bi >= thr: used.add(best); tp += 1
            r['truth'] += len(tr); r['tp'] += tp; r['pred'] += len(preds)
            extra = len(preds) - len(used)
            if ex == 'mark':
                r['extra'] += extra
                if im['marks'] and not tr: pass
            elif ex == 'none':
                r['neg_images'] += 1; r['neg_fp'] += 1 if preds else 0
            else:
                r['amb_images'] += 1; r['amb_fp'] += 1 if preds else 0
            site_tp += tp; site_vis += len(tr); site_pred += len(preds)
        if ex == 'mark' and site_vis and site_tp < site_vis:
            fails.append((s['id'], c, 'recall %d/%d extra %d' % (site_tp, site_vis, site_pred - site_tp)))
        if ex == 'none' and site_pred: fails.append((s['id'], c, 'false positive %d preds' % site_pred))
    return R, fails


def report(R):
    tot = defaultdict(float)
    print('%-12s %5s %6s %8s %7s %7s' % ('category', 'sites', 'truth', 'recall', 'extra', 'negFP'))
    for c in sorted(R):
        r = R[c]
        if not r['images']: continue
        rec = ('%5.1f%%' % (100 * r['tp'] / r['truth'])) if r['truth'] else '   -  '
        neg = ('%2d/%-3d' % (r['neg_fp'], r['neg_images'])) if r['neg_images'] else (('amb %d/%d' % (r['amb_fp'], r['amb_images'])) if r['amb_images'] else '  -  ')
        print('%-12s %5d %6d %8s %7d %7s' % (c, r['sites'], r['truth'], rec, r['extra'], neg))
        for k in ('truth', 'tp', 'extra', 'neg_images', 'neg_fp', 'pred'): tot[k] += r[k]
    rec = 100 * tot['tp'] / max(1, tot['truth']); negfp = 100 * tot['neg_fp'] / max(1, tot['neg_images'])
    prec = 100 * tot['tp'] / max(1, tot['tp'] + tot['extra'] + tot['neg_fp'])
    print('--- 전체: 인식률 %.1f%% (%d/%d) · 엉뚱한 상자 %d · 음성 이미지 오인식 %.1f%% (%d/%d) · 정밀도 %.1f%%' % (rec, tot['tp'], tot['truth'], tot['extra'], negfp, tot['neg_fp'], tot['neg_images'], prec))
    return rec, negfp, prec


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('truth'); ap.add_argument('pred'); ap.add_argument('--fail'); ap.add_argument('--cats')
    a = ap.parse_args()
    truth = json.load(open(a.truth)); P, seen = load_pred(a.pred)
    R, fails = evaluate(truth, P, seen, set(a.cats.split(',')) if a.cats else None)
    report(R)
    if a.fail: open(a.fail, 'w').write('\n'.join('%s\t%s\t%s' % f for f in sorted(fails, key=lambda f: (f[1], f[0]))))
