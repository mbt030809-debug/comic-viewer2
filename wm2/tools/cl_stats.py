#!/usr/bin/env python3
"""군집 특징량 분석: python3 cl_stats.py tier:tag ..."""
import sys, json, os, collections
sys.path.insert(0, os.path.dirname(__file__))
from eval_bench import iou
H = os.path.expanduser('~')
KEYS = ['file', 'x', 'y', 'w', 'h', 'support', 'score', 'k', 'nprops', 'meanz', 'dom', 'dens', 'spread', 'items']
def load(tier, tag):
    T = {s['id']: s for s in json.load(open(f'{H}/wm2-bench/{tier}/truth.json'))['sites']}
    rows = []
    for l in open(f'{H}/wm2-work/res/{tier}_{tag}.tsv', encoding='utf-8'):
        if not l.startswith('@') or '\t@cl' not in l and not l.rstrip().endswith('@cl'): continue
        f = l.rstrip('\n').split('\t'); sid = f[0][1:]; s = T[sid]
        r = dict(zip(KEYS, f[1:15])); 
        for k in KEYS[1:]: r[k] = float(r[k])
        im = next((i for i in s['images'] if i['file'] == r['file']), None)
        vis = [m for m in (im['marks'] if im else []) if m['visible']]
        r['iou'] = max([iou(m['box'], [r['x'], r['y'], r['w'], r['h']]) for m in vis] or [0])
        r['tp'] = r['iou'] >= .5; r['site'] = sid; r['cat'] = s['cat']; r['exp'] = s['expected']; r['sup'] = r['support'] / r['k']
        rows.append(r)
    return rows
if __name__ == '__main__':
    rows = []
    for a in sys.argv[1:]: t, g = a.split(':'); rows += load(t, g)
    rows = [r for r in rows if r['exp'] != 'ambiguous']
    tp = [r for r in rows if r['tp']]; fp = [r for r in rows if not r['tp']]
    print('군집 %d: TP %d FP %d' % (len(rows), len(tp), len(fp)))
    def q(v, p): v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))] if v else 0
    for key in ('support', 'sup', 'score', 'meanz', 'dom', 'dens', 'spread', 'nprops'):
        print('%-7s TP p10/50/90: %8.2f %8.2f %8.2f | FP p10/50/90/99: %8.2f %8.2f %8.2f %8.2f' % (key, q([r[key] for r in tp], .1), q([r[key] for r in tp], .5), q([r[key] for r in tp], .9), q([r[key] for r in fp], .1), q([r[key] for r in fp], .5), q([r[key] for r in fp], .9), q([r[key] for r in fp], .99)))
    # 사이트별 최고 점수 군집이 TP인 비율
    by = collections.defaultdict(list)
    for r in rows: by[r['site']].append(r)
    print('사이트별 군집 수 중앙값', sorted(len(v) for v in by.values())[len(by) // 2])
