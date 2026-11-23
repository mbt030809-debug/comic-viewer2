#!/usr/bin/env python3
"""2단계(참조형 신경망) 실패 원인 진단(오라클 1단계). 사용: python diag_s2.py <ref.pt> <bench/tier> [세트수]
미검출: 같은 예시 번호로 낸 상자와의 최대 IoU 분포(없음 / 0~0.3 / 0.3~0.5), 불투명·반투명 따로.
오검출: 상자가 (다른 번호의 정답과 겹침 / 같은 번호 정답과 겹침(IoU<0.5 또는 중복) / 어떤 정답과도 안 겹침(말풍선·머리·눈 등))."""
import os, sys, json
from collections import Counter
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from eval_s1 import iou, overlap_frac
from eval_ref import make_ref, target_tensor
from eval_pipeline import page_bg
from ref_model import RefUNet
from infer_eval import boxes_from_mask
from train import get_device

ck, root = sys.argv[1:3]; ns = int(sys.argv[3]) if len(sys.argv) > 3 else 30; dev = get_device('dml')
d = torch.load(ck, map_location='cpu', weights_only=True); net = RefUNet(tuple(d['ch'])); net.load_state_dict(d['sd']); net.to(dev).eval()
sets = json.load(open(os.path.join(root, 'truth.json'), encoding='utf-8'))['sets'][:ns]; miss = {'opaque': Counter(), 'translucent': Counter()}; fp = Counter(); fpsz = []
for s in sets:
    sd = os.path.join(root, s['id']); pages = [np.asarray(Image.open(os.path.join(sd, im['file'])).convert('RGB')) for im in s['images']]; tens = [target_tensor(os.path.join(sd, im['file'])) for im in s['images']]
    refs = {}
    for pi, im in enumerate(s['images']):
        for e in im['example_marks']:
            x, y, w, h = e['box']; arr = pages[pi]; x0, y0 = max(0, x - 4), max(0, y - 4); refs[e['idx']] = make_ref(arr, (x0, y0, min(arr.shape[1] - x0, w + 8), min(arr.shape[0] - y0, h + 8), page_bg(arr, e['box']))).to(dev)
    ex_boxes = {pi: [e['box'] for e in im['example_marks']] for pi, im in enumerate(s['images'])}
    for pj, im in enumerate(s['images']):
        x, W, H = tens[pj]
        for idx, rf in refs.items():
            with torch.no_grad(): p = torch.sigmoid(net(x.to(dev), rf))[0, 0].cpu().numpy()
            preds = [b[:4] for b in boxes_from_mask(p, W, H, th=.5, tile_frac=2.0) if b[2] * b[3] <= .2 * W * H and not any(iou(b, eb) > .3 for eb in ex_boxes[pj])]
            mine = [m for m in im['marks'] if m['idx'] == idx]; others = [m for m in im['marks'] if m['idx'] != idx]
            for m in mine:
                best = max([iou(b, m['box']) for b in preds], default=0); k = 'opaque' if m['alpha'] >= .99 else 'translucent'
                miss[k]['hit>=.5' if best >= .5 else 'iou .3-.5' if best >= .3 else 'iou 0-.3' if best > 0 else 'no box'] += 1
            for b in preds:
                if any(iou(b, m['box']) >= .5 for m in mine): continue
                if any(overlap_frac(b, m['box']) > .3 or overlap_frac(m['box'], b) > .3 for m in others): fp['다른 마크와 겹침(예시 혼동)'] += 1
                elif any(overlap_frac(b, m['box']) > .3 or overlap_frac(m['box'], b) > .3 for m in mine): fp['같은 마크인데 IoU<0.5·중복']  += 1
                else: fp['정답 마크 없는 곳(그림·말풍선·머리·눈 등)'] += 1; fpsz.append(b[2] * b[3] / (W * H))
for k, c in miss.items(): print('%s 마크:' % k, dict(c), '(합 %d)' % sum(c.values()))
print('오검출 구성:', dict(fp), '(합 %d)' % sum(fp.values()))
if fpsz: print('정답 없는 곳 오검출 크기(화면 대비 면적) 중앙값 %.4f' % np.median(fpsz))
