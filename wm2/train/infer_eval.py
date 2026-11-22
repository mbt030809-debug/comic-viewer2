#!/usr/bin/env python3
"""벤치마크 평가: python3 infer_eval.py <ckpt> <tier폴더> <출력.tsv> [사이트필터] [--th 0.5]
이미지를 0.5배로 줄여 U-Net 으로 마스크를 예측하고 덩어리 상자를 원본 좌표로 되돌려 eval_bench.py 형식으로 쓴다."""
import os, sys, time
import numpy as np, torch
from PIL import Image
from scipy import ndimage
sys.path.insert(0, os.path.dirname(__file__))
from model import UNet

def predict_mask(net, img, sc=.5):
    w, h = int(img.width * sc), int(img.height * sc); a = np.asarray(img.resize((w, h), Image.BOX), np.float32) / 255
    ph, pw = (-h) % 16, (-w) % 16; a = np.pad(a, ((0, ph), (0, pw), (0, 0)), mode='reflect')
    with torch.no_grad(): p = torch.sigmoid(net(torch.from_numpy(a.transpose(2, 0, 1)[None].copy())))[0, 0].numpy()
    return p[:h, :w]

def boxes_from_mask(p, W, H, th=.5, sc=.5, merge=6, min_area=30, tile_frac=.12):
    m = p > th
    if m.mean() > tile_frac: return [(0, 0, W, H, 'tile', float(p[m].mean()))]   # 전면 무늬
    d = ndimage.binary_dilation(m, iterations=merge); lab, n = ndimage.label(d); out = []
    for i, sl in enumerate(ndimage.find_objects(lab)):
        comp = (lab[sl] == i + 1) & m[sl]
        if comp.sum() < min_area: continue
        ys, xs = np.where(comp); x0, y0, x1, y1 = sl[1].start + xs.min(), sl[0].start + ys.min(), sl[1].start + xs.max() + 1, sl[0].start + ys.max() + 1
        out.append((int(x0 / sc), int(y0 / sc), int((x1 - x0) / sc), int((y1 - y0) / sc), 'mark', float(p[sl][comp].mean())))
    return out

if __name__ == '__main__':
    ck, root, outp = sys.argv[1:4]; filt = sys.argv[4] if len(sys.argv) > 4 and not sys.argv[4].startswith('--') else ''
    th = float(sys.argv[sys.argv.index('--th') + 1]) if '--th' in sys.argv else .5
    torch.set_num_threads(4); d = torch.load(ck, map_location='cpu', weights_only=True); net = UNet(tuple(d['ch'])); net.load_state_dict(d['sd']); net.eval()
    lines = []; t0 = time.time()
    for s in sorted(os.listdir(root)):
        d = os.path.join(root, s)
        if not os.path.isdir(d) or filt not in s: continue
        for f in sorted(x for x in os.listdir(d) if x.startswith('img_')):
            im = Image.open(os.path.join(d, f)).convert('RGB'); bs = boxes_from_mask(predict_mask(net, im), im.width, im.height, th)
            if not bs: lines.append('%s\t%s\tNONE' % (s, f))
            for b in bs: lines.append('%s\t%s\t%d\t%d\t%d\t%d\t%s\t%.3f' % (s, f, *b))
    open(outp, 'w').write('\n'.join(lines) + '\n'); print('완료 %.0fs → %s' % (time.time() - t0, outp))
