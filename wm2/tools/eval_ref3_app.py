#!/usr/bin/env python3
"""2단계 파이프라인 벤치(v3: 예시는 한 장의 단색 여백, 마크는 다른 장의 그림 위)의 앱 경로 결과(Wm2AppBench TSV)를 정답(truth.json)과 비교.
사용: eval_ref3_app.py <truth.json> <결과.tsv> [--fail 파일]
맞음 = 그림 위 정답 마크와 IoU>=0.5 (예시 번호는 따지지 않음). 예시가 있는 장(example_marks)·캡션 여백 장의 검출은 엉뚱한 상자로 센다."""
import sys, json, collections
def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx)); iy = max(0, min(ay + ah, by + bh) - max(ay, by)); i = ix * iy
    return i / (aw * ah + bw * bh - i) if i else 0
t = json.load(open(sys.argv[1])); sets = {s['id']: s for s in t['sets']}
pred = collections.defaultdict(list); seen = set()
for l in open(sys.argv[2], encoding='utf-8'):
    p = l.rstrip('\n').split('\t')
    if p[1] == 'gallery': continue
    seen.add((p[0], p[1]))
    if p[2] != 'NONE': pred[(p[0], p[1])].append([int(v) for v in p[3:7]])
tp = truth = extra = noimg = noimg_fp = nset = 0; tpo = tru_o = 0; fails = []
for sid, s in sets.items():
    if not any(k[0] == sid for k in seen): continue
    nset += 1
    for im in s['images']:
        P = list(pred.get((sid, im['file']), [])); used = set()
        for m in im['marks']:
            truth += 1; opaque = m['alpha'] >= .95; tru_o += opaque
            hit = [j for j, b in enumerate(P) if j not in used and iou(b, m['box']) >= .5]
            if hit: tp += 1; tpo += opaque; used.add(hit[0])
            else: fails.append('%s %s 못찾음 알파%.2f %s' % (sid, im['file'], m['alpha'], m['box']))
        extra += len(P) - len(used)
        if not im['marks']: noimg += 1; noimg_fp += bool(P)
print('세트 %d · 인식률 %.1f%% (%d/%d, 불투명 %.1f%% · 반투명 %.1f%%) · 엉뚱한 상자 %d · 정밀도 %.1f%% · 마크 없는 이미지 오인식 %d/%d' % (nset, 100 * tp / max(1, truth), tp, truth, 100 * tpo / max(1, tru_o), 100 * (tp - tpo) / max(1, truth - tru_o), extra, 100 * tp / max(1, tp + extra), noimg_fp, noimg))
if '--fail' in sys.argv: open(sys.argv[sys.argv.index('--fail') + 1], 'w', encoding='utf-8').write('\n'.join(fails))
