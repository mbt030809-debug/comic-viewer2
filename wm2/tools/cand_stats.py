#!/usr/bin/env python3
"""후보 특징량 분석: python3 cand_stats.py <tier>:<tag> ... → 정답 후보(TP)와 가짜 후보(FP)를 가르는 값을 찾는다."""
import sys, json, os, collections
sys.path.insert(0, os.path.dirname(__file__))
from eval_bench import iou
H = os.path.expanduser('~')
def load(tier, tag):
    T = {s['id']: s for s in json.load(open(f'{H}/wm2-bench/{tier}/truth.json'))['sites']}
    rows = []
    for l in open(f'{H}/wm2-work/res/{tier}_{tag}.tsv', encoding='utf-8'):
        if not l.startswith('@'): continue
        f = l.rstrip('\n').split('\t'); sid = f[0][1:]; s = T[sid]
        im = next(i for i in s['images'] if i['file'] == f[1]); box = [int(f[3]), int(f[4]), int(f[5]), int(f[6])]
        vis = [m for m in im['marks'] if m['visible']]
        best = max([iou(m['box'], box) for m in vis] or [0])
        rows.append(dict(site=sid, cat=s['cat'], exp=s['expected'], anchor=float(f[2]), box=box, score=float(f[7]), hits=int(f[8]), pres=int(f[9]), K=int(f[10]), dom=float(f[11]), meanR=float(f[12]), why=f[13], iou=best,
                         tp=best >= .5, dens=int(f[8]) / max(1, (box[2] // 2) * (box[3] // 2))))
    return rows
if __name__ == '__main__':
    rows = []
    for a in sys.argv[1:]: t, g = a.split(':'); rows += load(t, g)
    ok = [r for r in rows if r['why'] == '-']
    tp = [r for r in ok if r['tp']]; fp = [r for r in ok if not r['tp'] and r['exp'] != 'ambiguous']
    print('shape-ok 후보 %d: TP %d, FP %d (음성 사이트 FP %d)' % (len(ok), len(tp), len(fp), sum(1 for r in fp if r['exp'] == 'none')))
    for key in ('score', 'hits', 'meanR', 'dom', 'dens'):
        def q(v, p): v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))] if v else 0
        print('%-6s TP p10/50/90: %8.2f %8.2f %8.2f | FP p10/50/90/99: %8.2f %8.2f %8.2f %8.2f' % (key, q([r[key] for r in tp], .1), q([r[key] for r in tp], .5), q([r[key] for r in tp], .9), q([r[key] for r in fp], .1), q([r[key] for r in fp], .5), q([r[key] for r in fp], .9), q([r[key] for r in fp], .99)))
    # 위치 정확하지만 존재 수 모자란 후보 등
    print('TP 후보의 존재 비율 분포', collections.Counter(round(r['pres'] / r['K'], 1) for r in tp))
    print('FP 후보의 존재 비율 분포', collections.Counter(round(r['pres'] / r['K'], 1) for r in fp))
