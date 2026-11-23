#!/usr/bin/env python3
"""2단계 파이프라인 종단 평가 (테스트툰 v3: 예시 마크가 '어느 한 장'의 단색 여백에 있고, 나머지 그림 위에 같은 마크가 박힘).
 1단계: 모든 이미지에서 단색 여백의 예시 마크 후보를 찾는다(--s1 rule:N | learned:<ckpt> | oracle).
 2단계: 후보마다 참조형 모델(--s2 ckpt)로 모든 이미지에서 찾는다. 후보 자신이 있던 여백 자리는 제외.
 반복 확인: 후보가 다른 이미지 --vmin 장 이상에서 발견될 때만 채택(캡션·쪽번호 같은 오후보 제거).
정답 비교: 채택 후보를 정답 예시와 IoU≥0.5 로 연결해 예시 번호를 얻고, (번호, 상자 IoU≥0.5) 가 같으면 맞음.
지표: 종단 재현율/정밀도, 채택된 참조 수(진짜/가짜), 마크 없는 이미지 오검출, 마크가 불투명/반투명일 때 재현율, 종류별 재현율.
사용: python eval_pipeline.py --bench <refdata3> --tiers core,val --s1 rule:5 --s2 <ref.pt> --tag X --out rows_pipe.jsonl"""
import os, sys, json, time, argparse
from collections import defaultdict
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from eval_s1 import load_method, iou, overlap_frac
from eval_ref import make_ref, target_tensor
from ref_model import RefUNet
from infer_eval import boxes_from_mask
from train import get_device


def page_bg(arr, b):
    x, y, w, h = b[:4]; H, W = arr.shape[:2]; ys = [max(0, y - 6), min(H - 1, y + h + 6)]; xs = [max(0, x - 6), min(W - 1, x + w + 6)]
    px = np.concatenate([arr[ys[0], xs[0]:xs[1]], arr[ys[1], xs[0]:xs[1]], arr[ys[0]:ys[1], xs[0]], arr[ys[0]:ys[1], xs[1]]], 0); return float(np.median(px))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--bench', required=True); ap.add_argument('--tiers', default='core,val'); ap.add_argument('--s1', required=True); ap.add_argument('--s2', required=True); ap.add_argument('--tag', default='pipe')
    ap.add_argument('--out'); ap.add_argument('--device', default='dml'); ap.add_argument('--vmin', type=int, default=1); ap.add_argument('--th', type=float, default=.5); ap.add_argument('--max-sets', type=int, default=0)
    a = ap.parse_args(); dev = get_device(a.device)
    d = torch.load(a.s2, map_location='cpu', weights_only=True); net = RefUNet(tuple(d['ch'])); net.load_state_dict(d['sd']); net.to(dev).eval()
    s1 = None if a.s1 == 'oracle' else load_method(a.s1, a.device)
    for tier in a.tiers.split(','):
        root = os.path.join(a.bench, tier); truth = json.load(open(os.path.join(root, 'truth.json'), encoding='utf-8')); sets = truth['sets'][:a.max_sets] if a.max_sets else truth['sets']
        T = defaultdict(float); K = defaultdict(lambda: [0, 0]); AL = defaultdict(lambda: [0, 0]); t0 = time.time()
        for s in sets:
            sd = os.path.join(root, s['id']); pages = [Image.open(os.path.join(sd, im['file'])).convert('RGB') for im in s['images']]; arrs = [np.asarray(p) for p in pages]
            cands = []   # (page, box, true_idx)
            for pi, (im, pil) in enumerate(zip(s['images'], pages)):
                boxes = [tuple(e['box']) + (1.0,) for e in im['example_marks']] if s1 is None else s1(pil)
                for b in boxes:
                    tidx = -1
                    for e in im['example_marks']:
                        if iou(b, e['box']) >= .5: tidx = e['idx']
                    cands.append((pi, [int(v) for v in b[:4]], tidx))
            T['cands'] += len(cands); T['cands_true'] += sum(1 for c in cands if c[2] >= 0)
            tens = [target_tensor(os.path.join(sd, im['file'])) for im in s['images']]
            hits = []   # 후보별 [(page, box, score)]
            for ci, (pi, b, tidx) in enumerate(cands):
                arr = arrs[pi]; pad = 4; x0, y0 = max(0, b[0] - pad), max(0, b[1] - pad); w, h = min(arr.shape[1] - x0, b[2] + 2 * pad), min(arr.shape[0] - y0, b[3] + 2 * pad)
                rf = make_ref(arr, (x0, y0, w, h, page_bg(arr, b))).to(dev); res = []
                for pj, (x, W, H) in enumerate(tens):
                    with torch.no_grad(): p = torch.sigmoid(net(x.to(dev), rf))[0, 0].cpu().numpy()
                    for bb in boxes_from_mask(p, W, H, th=a.th, tile_frac=2.0):
                        if bb[2] * bb[3] > .2 * W * H: continue
                        if pj == pi and iou(bb, b) > .3: continue   # 후보 자신
                        if any(pj == cj and iou(bb, cb) > .3 for cj, cb, _ in cands): continue   # 다른 예시 자리
                        res.append((pj, list(bb[:4]), bb[5]))
                hits.append(res)
            keep = [ci for ci, res in enumerate(hits) if len({r[0] for r in res if r[0] != cands[ci][0]}) >= a.vmin]
            T['accepted'] += len(keep); T['accepted_true'] += sum(1 for ci in keep if cands[ci][2] >= 0)
            truth_idx = {e['idx'] for im in s['images'] for e in im['example_marks']}; T['true_refs'] += len(truth_idx)
            T['true_refs_accepted'] += len({cands[ci][2] for ci in keep if cands[ci][2] >= 0})
            for pj, im in enumerate(s['images']):
                preds = [(cands[ci][2], r[1]) for ci in keep for r in hits[ci] if r[0] == pj]; used = set(); T['images'] += 1
                for m in im['marks']:
                    best, bj = 0, -1
                    for j, (ti, pb) in enumerate(preds):
                        if j in used or ti != m['idx']: continue
                        v = iou(pb, m['box'])
                        if v > best: best, bj = v, j
                    hit = best >= .5; T['truth'] += 1; T['tp'] += hit
                    if hit: used.add(bj)
                    k = s['marks'][m['idx']]['kind']; K[k][0] += hit; K[k][1] += 1; ak = 'opaque' if m['alpha'] >= .99 else 'translucent'; AL[ak][0] += hit; AL[ak][1] += 1
                T['fp'] += len(preds) - len(used)
                if not im['marks']: T['empty_imgs'] += 1; T['empty_fp'] += int(len(preds) > 0)
        el = time.time() - t0
        res = dict(recall=100 * T['tp'] / max(1, T['truth']), precision=100 * T['tp'] / max(1, T['tp'] + T['fp']), tp=int(T['tp']), truth=int(T['truth']), fp=int(T['fp']),
                   refs_accepted=int(T['accepted']), refs_accepted_true=int(T['accepted_true']), true_refs=int(T['true_refs']), true_refs_accepted=int(T['true_refs_accepted']), cands=int(T['cands']), cands_true=int(T['cands_true']),
                   empty_img_fp=100 * T['empty_fp'] / max(1, T['empty_imgs']), sec=el)
        print('[%s %s] 종단 재현율 %.1f%% (%d/%d) · 정밀도 %.1f%% · 마크 없는 이미지 오검출 %.1f%% · 참조 후보 %d(진짜 %d) → 채택 %d(진짜 %d, 정답 예시 %d/%d 채택) · %.0fs' % (
            a.tag, tier, res['recall'], res['tp'], res['truth'], res['precision'], res['empty_img_fp'], res['cands'], res['cands_true'], res['refs_accepted'], res['refs_accepted_true'], res['true_refs_accepted'], res['true_refs'], el))
        print('  종류별: ' + ' · '.join('%s %.0f' % (k, 100 * v[0] / max(1, v[1])) for k, v in sorted(K.items())) + ' | 투명도별: ' + ' · '.join('%s %.0f(%d)' % (k, 100 * v[0] / max(1, v[1]), v[1]) for k, v in sorted(AL.items())))
        if a.out: open(a.out, 'a', encoding='utf-8').write(json.dumps(dict(tag=a.tag, tier=tier, s1=a.s1, s2=os.path.basename(a.s2), vmin=a.vmin, bench=a.bench, **res, kinds={k: 100 * v[0] / max(1, v[1]) for k, v in K.items()}, alpha={k: 100 * v[0] / max(1, v[1]) for k, v in AL.items()}), ensure_ascii=False) + '\n')


if __name__ == '__main__': main()
