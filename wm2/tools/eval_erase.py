#!/usr/bin/env python3
"""제거 평가: python3 eval_erase.py <truth.json> <벤치폴더> <지운결과폴더> [--cats a,b]
이미지마다 (정답 상자 안에서) 워터마크 입은 상태와 깨끗한 원본의 평균 절대 차이(전)와, 지운 결과와 원본의 차이(후)를 비교한다.
'범위 밖 변경' = 정답 상자(+20px) 밖에서 지운 결과가 입력과 달라진 정도(멀쩡한 부분을 건드리면 큼)."""
import sys, os, json, collections
import numpy as np
from PIL import Image
truth, bench, outd = sys.argv[1:4]
cats = set(sys.argv[sys.argv.index('--cats') + 1].split(',')) if '--cats' in sys.argv else None
T = {s['id']: s for s in json.load(open(truth))['sites']}
R = collections.defaultdict(lambda: collections.defaultdict(float))
for sid in sorted(os.listdir(outd)):
    s = T.get(sid)
    if not s or (cats and s['cat'] not in cats): continue
    for i, im in enumerate(s['images']):
        ep = os.path.join(outd, sid, os.path.splitext(im['file'])[0] + '.png')
        cp = os.path.join(bench, sid, 'clean_%02d.jpg' % (i + 1)); mp = os.path.join(bench, sid, im['file'])
        r = R[s['cat']]; r['images'] += 1
        if not os.path.exists(ep): r['untouched'] += 1; continue
        er = np.asarray(Image.open(ep).convert('RGB'), np.float32); mk = np.asarray(Image.open(mp).convert('RGB'), np.float32); cl = np.asarray(Image.open(cp).convert('RGB'), np.float32)
        box = np.zeros(er.shape[:2], bool); near = np.zeros(er.shape[:2], bool)
        for m in im['marks']:
            if not m['visible']: continue
            x, y, w, h = m['box']; box[max(0, y - 4):y + h + 4, max(0, x - 4):x + w + 4] = True; near[max(0, y - 20):y + h + 20, max(0, x - 20):x + w + 20] = True
        if s['expected'] != 'mark' or not box.any():
            r['neg_changed'] += 1; r['neg_delta'] += float(np.abs(er - mk).mean()); continue
        before = float(np.abs(mk - cl)[box].mean()); after = float(np.abs(er - cl)[box].mean())
        r['n'] += 1; r['before'] += before; r['after'] += after; r['outside'] += float(np.abs(er - mk)[~near].mean())
print('%-12s %6s %7s %7s %8s %9s' % ('category', 'images', 'before', 'after', 'reduce%', 'outside'))
tb = ta = tn = 0
for c in sorted(R):
    r = R[c]
    if r['n']:
        print('%-12s %6d %7.1f %7.1f %7.0f%% %9.2f%s' % (c, r['images'], r['before'] / r['n'], r['after'] / r['n'], 100 * (1 - r['after'] / r['before']), r['outside'] / r['n'], ('  (미처리 %d장)' % r['untouched']) if r['untouched'] else ''))
        tb += r['before']; ta += r['after']; tn += r['n']
    else: print('%-12s %6d  음성/기타: 바뀐 %d장 (평균 변화 %.2f), 미처리 %d' % (c, r['images'], r['neg_changed'], r['neg_delta'] / max(1, r['neg_changed']), r['untouched']))
if tn: print('--- 전체: 워터마크 자리 오차 %.1f → %.1f (%.0f%% 감소)' % (tb / tn, ta / tn, 100 * (1 - ta / tb)))
