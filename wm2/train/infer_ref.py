#!/usr/bin/env python3
"""참조형 워터마크 인식 — 직접 써 보기 (onnxruntime 만 있으면 됨: pip install onnxruntime numpy pillow scipy)
사용:  python infer_ref.py <model.onnx> <gallery.png> <img1> [img2 ...] [--out 결과폴더] [--th 0.5]
 - gallery.png : 맨 첫 장. 예시 마크들이 흰 또는 검은 배경 위에 있고, 마크 사이가 배경색으로 충분히 떨어져 있어야 한다.
                 (가로 띠마다 배경색이 달라도 됨. 가장 왼쪽 열 픽셀 색을 그 행의 배경으로 본다.)
 - img*        : 워터마크가 박힌 이미지들(원본 해상도).
출력: 결과폴더/갤러리_분리.png (분리된 예시 마크에 번호), 결과폴더/<이미지이름>_det.png (찾은 상자와 예시 번호), 터미널에 상자 목록.
모델 입력: 이미지·예시 모두 0.5배 BOX 축소. 예시는 112x192 캔버스(배경색)에 가운데 놓음. 상자 면적이 이미지의 20%를 넘으면 버린다."""
import os, sys, argparse, time
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import onnxruntime as ort

RH, RW = 112, 192
MAX_AREA = .20
COLORS = [(255, 0, 255), (0, 200, 255), (255, 160, 0), (0, 220, 80), (255, 60, 60), (160, 100, 255)]


def split_gallery(g):
    """갤러리 → [(x,y,w,h,배경값)] 위에서 아래 순서"""
    bg = g[:, 0:1, :].astype(np.int16); nb = (np.abs(g.astype(np.int16) - bg).max(2) > 40)
    dil = ndimage.binary_dilation(nb, structure=np.ones((3, 3), bool), iterations=20)
    lab, n = ndimage.label(dil); out = []
    for i in range(1, n + 1):
        ys, xs = np.where((lab == i) & nb)
        if len(xs) < 30: continue
        out.append((int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1), float(bg[int(ys.min()), 0].mean())))
    out.sort(key=lambda b: b[1]); return out


def make_ref(g, box):
    x, y, w, h, bg = box; crop = Image.fromarray(g[y:y + h, x:x + w]); nw, nh = max(4, w // 2), max(4, h // 2)
    a = np.asarray(crop.resize((nw, nh), Image.BOX), np.float32) / 255
    f = min(1., (RW - 8) / nw, (RH - 8) / nh)
    if f < 1: nh, nw = max(4, int(nh * f)), max(4, int(nw * f)); a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((nw, nh), Image.BOX), np.float32) / 255
    can = np.full((RH, RW, 3), bg / 255, np.float32); y0, x0 = (RH - nh) // 2, (RW - nw) // 2; can[y0:y0 + nh, x0:x0 + nw] = a
    return can.transpose(2, 0, 1)[None].copy()


def target_array(img):
    W, H = img.size; w, h = W // 2, H // 2
    a = np.asarray(img.resize((w, h), Image.BOX), np.float32) / 255
    a = np.pad(a, ((0, (-h) % 16), (0, (-w) % 16), (0, 0)), mode='reflect').transpose(2, 0, 1)[None].copy()
    return a


def boxes_from_prob(p, th=.5, sc=.5, merge=6, min_area=30):
    m = p > th; d = ndimage.binary_dilation(m, iterations=merge); lab, n = ndimage.label(d); out = []
    for i, sl in enumerate(ndimage.find_objects(lab)):
        comp = (lab[sl] == i + 1) & m[sl]
        if comp.sum() < min_area: continue
        ys, xs = np.where(comp); x0, y0, x1, y1 = sl[1].start + xs.min(), sl[0].start + ys.min(), sl[1].start + xs.max() + 1, sl[0].start + ys.max() + 1
        out.append((int(x0 / sc), int(y0 / sc), int((x1 - x0) / sc), int((y1 - y0) / sc), float(p[sl][comp].mean())))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('model'); ap.add_argument('gallery'); ap.add_argument('images', nargs='+'); ap.add_argument('--out', default='ref_out'); ap.add_argument('--th', type=float, default=.5)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    sess = ort.InferenceSession(a.model, providers=['CPUExecutionProvider'])
    g = np.asarray(Image.open(a.gallery).convert('RGB')); comps = split_gallery(g)
    print('예시 마크 %d개 분리됨' % len(comps))
    gi = Image.fromarray(g).copy(); dr = ImageDraw.Draw(gi)
    for i, (x, y, w, h, bg) in enumerate(comps):
        c = COLORS[i % len(COLORS)]; dr.rectangle([x - 4, y - 4, x + w + 4, y + h + 4], outline=c, width=3); dr.text((x, max(0, y - 16)), '#%d' % i, fill=c)
    gi.save(os.path.join(a.out, '갤러리_분리.png')); refs = [make_ref(g, b) for b in comps]
    for path in a.images:
        img = Image.open(path).convert('RGB'); W, H = img.size; x = target_array(img); res = []; t0 = time.time()
        for i, rf in enumerate(refs):
            p = 1 / (1 + np.exp(-sess.run(None, {'image': x, 'ref': rf})[0][0, 0]))
            for b in boxes_from_prob(p, a.th):
                if b[2] * b[3] <= MAX_AREA * W * H: res.append((i,) + b)
        im2 = img.copy(); dr = ImageDraw.Draw(im2)
        for i, bx, by, bw, bh, sc in res:
            c = COLORS[i % len(COLORS)]; dr.rectangle([bx, by, bx + bw, by + bh], outline=c, width=3); dr.text((bx, max(0, by - 14)), '#%d %.2f' % (i, sc), fill=c)
        name = os.path.splitext(os.path.basename(path))[0]; im2.save(os.path.join(a.out, name + '_det.png'))
        print('%s (%dx%d, %.1fs): %s' % (os.path.basename(path), W, H, time.time() - t0, ', '.join('#%d [%d,%d,%d,%d]' % (i, bx, by, bw, bh) for i, bx, by, bw, bh, _ in res) or '없음'))


if __name__ == '__main__': main()
