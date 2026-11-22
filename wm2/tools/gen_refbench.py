#!/usr/bin/env python3
"""참조형 인식 벤치(테스트툰): 세트 하나 = 맨 첫 장(gallery.png: 예시 마크들을 흰/검은 배경에 둠) + 뒤 이미지들(img_NN).
뒤 이미지에는 예시 마크가 위치·크기 매번 다르게, 반투명(알파 낮음), 한 이미지에 여러 개(같은 마크 반복 포함), 일부 이미지에는 없음으로 박힌다.
첫 장의 예시 마크는 뒤 이미지에 박힌 마크 전부이다(모든 예시 마크가 최소 한 장에는 나온다).
정답(truth.json): 예시 마크별 갤러리 상자와, 이미지별 (예시 마크 번호, 상자[x,y,w,h]).
사용: python gen_refbench.py --tier core|val --out <dir> --n 120 --jobs 10"""
import os, sys, json, math, random, argparse
import numpy as np
from PIL import Image
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
import gen_bench as G

KINDS = ['text', 'text', 'badge', 'badge', 'emblem', 'icon', 'url', 'outline', 'qr']
ALPHAS = [.15, .2, .25, .3, .4, .5]
MIN_CONTRAST = 5.0
MAX_AREA = .20   # 워터마크 상자 면적 / 화면 면적 상한(전면 무늬·화면 전체 인식 방지)


def lum(spr):
    sel = spr.A > .5
    if not sel.any(): return 128.
    c = spr.P[sel] / np.maximum(spr.A[sel][:, None], 1e-3)
    return float(np.mean(c @ np.array([.299, .587, .114])))


def gallery(sprs, rng, W=720, pad=30):
    """예시 마크마다 가로 띠(흰 또는 검은 배경) 하나. 반환 (이미지, [(상자, 배경)])"""
    bands = []; H = 0
    for s in sprs:
        l = lum(s); bg = 0 if l > 170 else 255 if l < 85 else rng.choice([0, 255])
        sw = min(s.w, W - 20)
        if sw != s.w: s = s.scaled(sw / s.w)
        h = s.h + 2 * pad; bands.append((s, bg, h)); H += h
    img = np.zeros((H, W, 3), np.float32); y = 0; boxes = []
    for s, bg, h in bands:
        img[y:y + h] = bg; x0 = (W - s.w) // 2; y0 = y + pad
        reg = img[y0:y0 + s.h, x0:x0 + s.w]; img[y0:y0 + s.h, x0:x0 + s.w] = reg * (1 - s.A[..., None]) + s.P
        boxes.append(dict(box=[x0, y0, s.w, s.h], bg=int(bg))); y += h
    return np.clip(img, 0, 255).astype(np.uint8), boxes


def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix, iy = max(0, min(ax + aw, bx + bw) - max(ax, bx)), max(0, min(ay + ah, by + bh) - max(ay, by)); i = ix * iy
    return i / max(1, aw * ah + bw * bh - i)


def gen_set(args):
    sid, seed, tier, root = args
    rng = random.Random(seed); d = os.path.join(root, sid); os.makedirs(d, exist_ok=True)
    M = rng.choice([1, 2, 2, 3, 3, 4, 5]); marks = []; sprs = []
    for _ in range(M):
        kind = rng.choice(KINDS); P = G.new_mark(rng, kind, 'random', rng.choice(ALPHAS), tier, 720); marks.append(P); sprs.append(G.build_sprite(P, 720, 1000, rng))
    gimg, gboxes = gallery(sprs, rng); Image.fromarray(gimg).save(os.path.join(d, 'gallery.png'))
    K = rng.randint(4, 8); style = rng.choice(G.STYLES); codec = rng.choice(['png', 'jpg95', 'jpg85']); hue = rng.randint(0, 10 ** 6)
    pres = [[rng.random() < .7 for _ in range(M)] for _ in range(K)]
    for k in range(K):
        if rng.random() < .15: pres[k] = [False] * M
    for m in range(M):
        if not any(p[m] for p in pres): pres[rng.randrange(K)][m] = True
    images = []
    for k in range(K):
        H = rng.randint(1100, 2200); img = G.make_comic(seed * 100 + k, 720, H, style, hue); placed = []; ms = []
        for m in range(M):
            if not pres[k][m]: continue
            for _ in range(1 + (rng.random() < .35)):
                for _try in range(12):
                    f = rng.uniform(.7, 1.3); s = sprs[m].scaled(f)
                    if rng.random() < .3: s = G.spr_rotated(s, rng.uniform(-12, 12))
                    if s.w >= 700 or s.h >= H - 20: continue
                    x = rng.randint(0, 720 - s.w); y = rng.randint(0, H - s.h); a = float(np.clip(marks[m]['alpha'] * rng.uniform(.7, 1.3), .1, .9))
                    if s.w * s.h > MAX_AREA * 720 * H: continue   # 워터마크는 화면의 20%를 넘지 않는다
                    if any(iou([x, y, s.w, s.h], p) > 0 for p in placed): continue
                    patch = img[y:y + s.h, x:x + s.w].copy(); _, con = G.bake(patch, s, a, 0, 0)
                    if con < MIN_CONTRAST: continue
                    box, con = G.bake(img, s, a, x, y); placed.append([x, y, s.w, s.h]); ms.append(dict(idx=m, box=box, alpha=round(a, 3), contrast=round(con, 1), scale=round(f, 2))); break
        ext = 'png' if codec == 'png' else 'jpg'; fn = 'img_%02d.%s' % (k + 1, ext); pim = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        if ext == 'png': pim.save(os.path.join(d, fn))
        else: pim.save(os.path.join(d, fn), 'JPEG', quality=int(codec[3:]))
        images.append(dict(file=fn, w=720, h=H, marks=ms))
    return dict(id=sid, tier=tier, style=style, codec=codec, M=M,
                marks=[dict(idx=i, kind=P['kind'], name=P['name'], font=P['font'], alpha=P['alpha'], gallery=gboxes[i]) for i, P in enumerate(marks)], images=images)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tier', default='core'); ap.add_argument('--out', required=True); ap.add_argument('--n', type=int, default=120); ap.add_argument('--jobs', type=int, default=8); ap.add_argument('--seed', type=int, default=5000)
    a = ap.parse_args(); root = os.path.join(a.out, a.tier); os.makedirs(root, exist_ok=True)
    base = a.seed + (0 if a.tier == 'core' else 100000)
    tasks = [('ref_%s_%03d' % (a.tier, i), base + i, a.tier, root) for i in range(a.n)]
    with Pool(a.jobs) as p: sets = p.map(gen_set, tasks, chunksize=2)
    json.dump(dict(tier=a.tier, sets=sets), open(os.path.join(root, 'truth.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    nm = sum(len(im['marks']) for s in sets for im in s['images']); ni = sum(len(s['images']) for s in sets); ne = sum(1 for s in sets for im in s['images'] if not im['marks'])
    print('세트 %d · 이미지 %d(마크 없는 이미지 %d) · 정답 마크 %d' % (len(sets), ni, ne, nm))


if __name__ == '__main__': main()
