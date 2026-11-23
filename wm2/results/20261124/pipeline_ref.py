#!/usr/bin/env python3
"""wm2 인식 파이프라인 기준 구현 (PC 시험·앱 이식용).
기본 설정은 **신경망 없이** 규칙 + SIFT 특징점 매칭만 쓴다(빠름): pip install numpy pillow scipy opencv-python-headless
 python pipeline_ref.py <이미지 폴더 또는 이미지들...> [--out 결과폴더]
입력: 같은 사이트 이미지 3~8장(순서 무관). 그중 어느 한 장의 단색(흰/검은) 여백에 '예시 워터마크'들이 있고, 나머지 이미지의 그림 위에 같은 마크가 박혀 있다.
1단계(예시 마크 찾기, 모든 이미지에서): rule = s1_rule.py(규칙 v5), neural = 1단계 U-Net(--s1net), hybrid = 규칙 후보 중 U-Net 마스크가 상자의 25% 이상을 예시로 본 것만.
2단계(그림 위에서 찾기, 예시마다 모든 이미지에서):
  kp(기본) = s2_kp.py: 예시의 SIFT 특징점을 그림 위와 맞춰 RANSAC 닮음변환(크기·회전·위치)을 구함. 인라이어 --kp-min(기본 6) 이상이면 검출.
  nn = 참조형 U-Net(--s2net) 상자. --consensus N: 신경망 상자 중 같은 자리에서 특징점 매칭(인라이어 ≥ N)도 확인한 것만 남김, --also-kp M: 신경망이 못 본 곳을 특징점 매칭(인라이어 ≥ M)이 보강.
예시와 크기·가로세로비(0.6~1.75배)가 다른 상자, 면적 > 화면 20% 상자, 예시 자리 자체는 제외. 반복 확인(--vmin, 기본 1): 예시가 자기 페이지 밖의 이미지 vmin 장 이상에서 발견될 때만 채택.
출력: 결과폴더/result.json (examples, detections), 결과폴더/<이미지>_det.png."""
import os, sys, json, glob, argparse, time
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s1_rule, s2_kp

RH, RW = 112, 192
MAX_AREA = .20
COLORS = [(255, 0, 255), (0, 200, 255), (255, 160, 0), (0, 220, 80), (255, 60, 60), (160, 100, 255)]


def iou(a, b):
    ax, ay, aw, ah = a[:4]; bx, by, bw, bh = b[:4]
    ix, iy = max(0, min(ax + aw, bx + bw) - max(ax, bx)), max(0, min(ay + ah, by + bh) - max(ay, by)); i = ix * iy
    return i / max(1, aw * ah + bw * bh - i)


def half_pad(img):
    W, H = img.size; w, h = W // 2, H // 2
    a = np.asarray(img.resize((w, h), Image.BOX), np.float32) / 255
    return np.pad(a, ((0, (-h) % 16), (0, (-w) % 16), (0, 0)), mode='reflect').transpose(2, 0, 1)[None].copy()


def sigmoid(x): return 1 / (1 + np.exp(-x))


def boxes_from_prob(p, th=.5, sc=.5, merge=6, min_area=30):
    m = p > th; d = ndimage.binary_dilation(m, iterations=merge); lab, n = ndimage.label(d); out = []
    for i, sl in enumerate(ndimage.find_objects(lab)):
        comp = (lab[sl] == i + 1) & m[sl]
        if comp.sum() < min_area: continue
        ys, xs = np.where(comp); x0, y0, x1, y1 = sl[1].start + xs.min(), sl[0].start + ys.min(), sl[1].start + xs.max() + 1, sl[0].start + ys.max() + 1
        out.append((int(x0 / sc), int(y0 / sc), int((x1 - x0) / sc), int((y1 - y0) / sc), float(p[sl][comp].mean())))
    return out


def stage1(img, mode, sess1):
    W, H = img.size
    if mode == 'rule': return s1_rule.detect(np.asarray(img), s1_rule.RULE_VERSION)
    p = sigmoid(sess1.run(None, {'image': half_pad(img)})[0][0, 0])
    if mode == 'neural':
        return [(b[0], b[1], b[2], b[3], b[4]) for b in boxes_from_prob(p) if b[2] * b[3] <= MAX_AREA * W * H and b[2] >= .06 * W and b[3] >= 18]
    out = []   # hybrid
    for b in s1_rule.detect(np.asarray(img), s1_rule.RULE_VERSION):
        reg = p[b[1] // 2:(b[1] + b[3]) // 2 + 1, b[0] // 2:(b[0] + b[2]) // 2 + 1]
        if reg.size and (reg > .5).mean() >= .25: out.append((b[0], b[1], b[2], b[3], float(reg.mean())))
    return out


def page_bg(arr, b):
    x, y, w, h = b[:4]; H, W = arr.shape[:2]; ys = [max(0, y - 6), min(H - 1, y + h + 6)]; xs = [max(0, x - 6), min(W - 1, x + w + 6)]
    px = np.concatenate([arr[ys[0], xs[0]:xs[1]], arr[ys[1], xs[0]:xs[1]], arr[ys[0]:ys[1], xs[0]], arr[ys[0]:ys[1], xs[1]]], 0); return float(np.median(px))


def make_ref(arr, box, bg):
    x, y, w, h = box; crop = Image.fromarray(arr[y:y + h, x:x + w]); nw, nh = max(4, w // 2), max(4, h // 2)
    a = np.asarray(crop.resize((nw, nh), Image.BOX), np.float32) / 255; f = min(1., (RW - 8) / nw, (RH - 8) / nh)
    if f < 1: nh, nw = max(4, int(nh * f)), max(4, int(nw * f)); a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((nw, nh), Image.BOX), np.float32) / 255
    can = np.full((RH, RW, 3), bg / 255, np.float32); y0, x0 = (RH - nh) // 2, (RW - nw) // 2; can[y0:y0 + nh, x0:x0 + nw] = a
    return can.transpose(2, 0, 1)[None].copy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('inputs', nargs='+'); ap.add_argument('--s1net'); ap.add_argument('--s2net'); ap.add_argument('--s1', default='rule', choices=['hybrid', 'rule', 'neural']); ap.add_argument('--s2', default='kp', choices=['kp', 'nn'])
    ap.add_argument('--kp-min', type=int, default=6); ap.add_argument('--consensus', type=int, default=0); ap.add_argument('--also-kp', type=int, default=0); ap.add_argument('--vmin', type=int, default=1); ap.add_argument('--th', type=float, default=.5); ap.add_argument('--out', default='pipe_out')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    files = []
    for p in a.inputs: files += sorted(glob.glob(os.path.join(p, '*'))) if os.path.isdir(p) else [p]
    files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')) and not os.path.basename(f).lower().startswith('gallery')]
    ort = None
    if a.s1 != 'rule' or a.s2 == 'nn': import onnxruntime as ort
    sess1 = ort.InferenceSession(a.s1net, providers=['CPUExecutionProvider']) if a.s1 != 'rule' else None; sess2 = ort.InferenceSession(a.s2net, providers=['CPUExecutionProvider']) if a.s2 == 'nn' else None
    pages = [Image.open(f).convert('RGB') for f in files]; arrs = [np.asarray(p) for p in pages]; reps = [s2_kp.page_repr(p) for p in pages]; halves = [half_pad(p) for p in pages] if a.s2 == 'nn' else None
    t0 = time.time(); cands = []
    for pi, pil in enumerate(pages):
        for b in stage1(pil, a.s1, sess1): cands.append((pi, [int(v) for v in b[:4]]))
    print('1단계: 예시 후보 %d개 (%.1fs)' % (len(cands), time.time() - t0))
    dets = []; accepted = []
    for ci, (pi, b) in enumerate(cands):
        arr = arrs[pi]; pad = 4; x0, y0 = max(0, b[0] - pad), max(0, b[1] - pad); w, h = min(arr.shape[1] - x0, b[2] + 2 * pad), min(arr.shape[0] - y0, b[3] + 2 * pad); bgv = page_bg(arr, b)
        tpl, msk = s2_kp.prepare(arr[y0:y0 + h, x0:x0 + w], bgv); rf = make_ref(arr, (x0, y0, w, h), bgv) if a.s2 == 'nn' else None; res = []
        for pj, pil in enumerate(pages):
            W, H = pil.size
            if a.s2 == 'kp': found = [bb + ('kp',) for bb in s2_kp.match(reps[pj], tpl, msk, thr=a.kp_min)]
            else:
                p = sigmoid(sess2.run(None, {'image': halves[pj], 'ref': rf})[0][0, 0]); found = [bb + ('nn',) for bb in boxes_from_prob(p, a.th)]
                if a.consensus > 0:
                    kpk = s2_kp.match(reps[pj], tpl, msk, thr=a.consensus); found = [f for f in found if any(iou(f, t) >= .2 for t in kpk)]
                if a.also_kp > 0:
                    for tb in s2_kp.match(reps[pj], tpl, msk, thr=a.also_kp):
                        if all(iou(tb, f) < .3 for f in found): found.append((tb[0], tb[1], tb[2], tb[3], tb[4], 'kp'))
            for bb in found:
                if bb[2] * bb[3] > MAX_AREA * W * H: continue
                if not (.6 * b[2] <= bb[2] <= 1.75 * b[2] and .6 * b[3] <= bb[3] <= 1.75 * b[3]): continue
                if pj == pi and iou(bb, b) > .3: continue
                if any(pj == cj and iou(bb, cb) > .3 for cj, cb in cands): continue
                res.append(dict(image=os.path.basename(files[pj]), example=ci, box=[int(v) for v in bb[:4]], score=round(float(bb[4]), 3), by=bb[5], page=pj))
        if len({r['page'] for r in res if r['page'] != pi}) >= a.vmin: accepted.append(ci); dets += res
    print('2단계: 채택 예시 %d/%d개, 검출 %d개 (%.1fs)' % (len(accepted), len(cands), len(dets), time.time() - t0))
    ex = [dict(id=ci, image=os.path.basename(files[cands[ci][0]]), box=cands[ci][1]) for ci in accepted]
    json.dump(dict(settings=vars(a), examples=ex, detections=[{k: v for k, v in d.items() if k != 'page'} for d in dets]), open(os.path.join(a.out, 'result.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    for pj, pil in enumerate(pages):
        im = pil.copy(); dr = ImageDraw.Draw(im)
        for e in ex:
            if e['image'] == os.path.basename(files[pj]): dr.rectangle([e['box'][0], e['box'][1], e['box'][0] + e['box'][2], e['box'][1] + e['box'][3]], outline=(255, 255, 0), width=3); dr.text((e['box'][0], max(0, e['box'][1] - 14)), 'example #%d' % e['id'], fill=(255, 255, 0))
        for d in dets:
            if d['page'] == pj:
                c = COLORS[d['example'] % len(COLORS)]; x, y, w, h = d['box']; dr.rectangle([x, y, x + w, y + h], outline=c, width=3); dr.text((x, max(0, y - 14)), '#%d %.2f' % (d['example'], d['score']), fill=c)
        im.save(os.path.join(a.out, os.path.splitext(os.path.basename(files[pj]))[0] + '_det.png'))


if __name__ == '__main__': main()
