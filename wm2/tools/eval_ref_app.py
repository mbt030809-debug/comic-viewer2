#!/usr/bin/env python3
"""참조형 벤치 앱 경로 결과(Wm2AppBench TSV)를 정답(truth.json)과 비교.
사용: eval_ref_app.py <truth.json> <결과.tsv> [--fail 파일]
맞음 = IoU>=0.5 (기본은 예시 번호 무시, --idx 면 번호도 일치). 번호 일치 재현율도 함께 낸다."""
import sys, json, collections
IGN = '--idx' not in sys.argv   # 기본: 예시 번호는 따지지 않음(지우기는 번호를 안 씀). --idx 를 주면 번호도 일치해야 맞음
def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx)); iy = max(0, min(ay + ah, by + bh) - max(ay, by)); i = ix * iy
    return i / (aw * ah + bw * bh - i) if i else 0
t = json.load(open(sys.argv[1])); sets = {s['id']: s for s in t['sets']}
pred = collections.defaultdict(list); refs = {}; seen = set()
for l in open(sys.argv[2], encoding='utf-8'):
    p = l.rstrip('\n').split('\t')
    if p[1] == 'gallery': refs[p[0]] = int(p[3]); continue
    seen.add((p[0], p[1]))
    if p[2] != 'NONE': pred[(p[0], p[1])].append((int(p[2]), [int(v) for v in p[3:7]]))
tp = tp_any = truth = extra = noimg = noimg_fp = nset = gal_ok = 0; fails = []
for sid, s in sets.items():
    if not any(k[0] == sid for k in seen): continue
    nset += 1; gal_ok += (refs.get(sid) == len(s['marks']))
    for im in s['images']:
        P = list(pred.get((sid, im['file']), [])); used = set()
        for m in im['marks']:
            truth += 1; best = None
            for j, (r, b) in enumerate(P):
                if j in used: continue
                v = iou(b, m['box'])
                if v >= .5 and (best is None or v > best[0]): best = (v, j, r)
            if best: tp_any += 1
            hit = [j for j, (r, b) in enumerate(P) if j not in used and (IGN or r == m['idx']) and iou(b, m['box']) >= .5]
            if hit: tp += 1; used.add(hit[0])
            else: fails.append('%s %s 못찾음 예시%d %s' % (sid, im['file'], m['idx'], m['box']))
        ex = [(r, b) for j, (r, b) in enumerate(P) if j not in used]; extra += len(ex)
        if not im['marks']: noimg += 1; noimg_fp += bool(P)
print('세트 %d · 갤러리 예시 개수 맞음 %d/%d' % (nset, gal_ok, nset))
print('인식률 %.1f%% (%d/%d) · 엉뚱한 상자 %d · 정밀도 %.1f%% · 마크 없는 이미지 오인식 %d/%d' % (100 * tp / max(1, truth), tp, truth, extra, 100 * tp / max(1, tp + extra), noimg_fp, noimg))
if '--fail' in sys.argv: open(sys.argv[sys.argv.index('--fail') + 1], 'w', encoding='utf-8').write('\n'.join(fails))
