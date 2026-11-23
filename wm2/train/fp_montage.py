#!/usr/bin/env python3
"""참조형 모델의 오검출 상자(정답과 안 맞는 상자)를 잘라 한 장으로 모은다. 사용: python fp_montage.py <ckpt> <refdata/core> <out.png> [세트수] [장수]"""
import os, sys, json, random
import numpy as np, torch
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(__file__))
from ref_model import RefUNet
from eval_ref import split_gallery, make_ref, target_tensor, iou
from infer_eval import boxes_from_mask
ck, root, outp = sys.argv[1:4]; ns = int(sys.argv[4]) if len(sys.argv) > 4 else 40; nshow = int(sys.argv[5]) if len(sys.argv) > 5 else 60
d = torch.load(ck, map_location='cpu', weights_only=True); net = RefUNet(tuple(d['ch'])); net.load_state_dict(d['sd']); net.eval()
truth = json.load(open(os.path.join(root, 'truth.json'), encoding='utf-8'))['sets'][:ns]; fps = []; tps = []
for s in truth:
    sd = os.path.join(root, s['id']); g = np.asarray(Image.open(os.path.join(sd, 'gallery.png')).convert('RGB')); comps = split_gallery(g)
    refs = []
    for cb in comps:
        best, bi = 0, -1
        for m in s['marks']:
            v = iou(list(cb[:4]), m['gallery']['box'])
            if v > best: best, bi = v, m['idx']
        refs.append((bi, make_ref(g, cb)))
    for im in s['images']:
        x, W, H = target_tensor(os.path.join(sd, im['file'])); full = Image.open(os.path.join(sd, im['file'])).convert('RGB')
        for idx, rf in refs:
            with torch.no_grad(): p = torch.sigmoid(net(x, rf))[0, 0].numpy()
            for b in boxes_from_mask(p, W, H, th=.5, tile_frac=2.0):
                if b[2] * b[3] > .2 * W * H: continue
                ok = any(m['idx'] == idx and iou(b, m['box']) >= .5 for m in im['marks'])
                crop = full.crop((max(0, b[0] - 20), max(0, b[1] - 20), min(W, b[0] + b[2] + 20), min(H, b[1] + b[3] + 20)))
                (tps if ok else fps).append((crop, '%s#%d %s' % (s['id'][-3:], idx, s['marks'][idx]['kind']), b[5]))
print('TP 상자 %d · FP 상자 %d' % (len(tps), len(fps)))
random.Random(1).shuffle(fps); show = fps[:nshow]; cw, ch = 150, 110; cols = 10; rows = (len(show) + cols - 1) // cols
M = Image.new('RGB', (cols * cw, rows * (ch + 14)), (40, 40, 40)); dr = ImageDraw.Draw(M)
for i, (c, lab, sc) in enumerate(show):
    c.thumbnail((cw - 4, ch - 4)); x, y = (i % cols) * cw, (i // cols) * (ch + 14); M.paste(c, (x + 2, y + 14)); dr.text((x + 2, y + 1), '%s %.2f' % (lab, sc), fill=(255, 255, 0))
M.save(outp); print('저장', outp, M.size)
