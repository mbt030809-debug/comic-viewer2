#!/usr/bin/env python3
"""참조형 인식 벤치 v2 (사양: wm2/RECOGNITION_SPEC_20261123.md). gen_refbench.py 대비:
 ① 마크 종류에 캐릭터(마스코트) 추가 ② 마크 폭을 사양 범위로 제한(로고류 이미지 폭의 8~30%, 글자류 10~50%, 상자 면적 ≤ 화면 20%)
 ③ 배경에 만화 선화(집중선·머리카락·눈 하이라이트·불꽃)를 일부 영역에 넣음 — 워터마크 아님, 정답에 없음
사용: python gen_refbench2.py --tier core|val --out <dir> --n 120 --jobs 8"""
import os, sys, json, random, argparse
import numpy as np
from PIL import Image
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
import gen_bench as G
from gen_refbench import gallery, iou, lum, MIN_CONTRAST, MAX_AREA
import manga_fx, marks_extra

KINDS = ['text', 'text', 'badge', 'badge', 'emblem', 'icon', 'url', 'outline', 'qr', 'character', 'character']
LOGO = {'badge', 'emblem', 'icon', 'qr', 'character'}
ALPHAS = [.15, .2, .25, .3, .4, .5]
W0 = 720


def char_spr(seed):
    k, A, P = marks_extra.character_sprite(seed, 2); s = G.Spr(A.shape[1], A.shape[0]); s.A = A.astype(np.float32) / 255; s.P = P.astype(np.float32); return s


def add_manga(img, rng):
    """full-res 이미지(H,W,3 float)에 0.5배 세계 기준 조각마다 만화 선화를 넣는다(조각 192x192 → 384x384 원본 크기)."""
    H = img.shape[0]; nr = np.random.default_rng(rng.randint(0, 10 ** 9)); n = max(2, H // 350)
    for _ in range(n):
        if rng.random() > .8: continue
        y = rng.randint(0, max(0, H - 384)); x = rng.randint(0, W0 - 384); patch = img[y:y + 384, x:x + 384]
        small = np.asarray(Image.fromarray(np.clip(patch, 0, 255).astype(np.uint8)).resize((192, 192), Image.BOX), np.float32)
        small, used = manga_fx.apply_fx(small, nr, p=1.0)
        img[y:y + 384, x:x + 384] = np.asarray(Image.fromarray(np.clip(small, 0, 255).astype(np.uint8)).resize((384, 384), Image.BICUBIC), np.float32)
    return img


def scale_for(w, kind, rng):
    lo, hi = (.08 * W0, .30 * W0) if kind in LOGO else (.10 * W0, .50 * W0)
    flo, fhi = max(.75, lo / w), min(1.33, hi / w)
    return rng.uniform(flo, fhi) if flo <= fhi else float(np.clip(1., lo / w, hi / w))


def gen_set(args):
    sid, seed, tier, root = args
    rng = random.Random(seed); d = os.path.join(root, sid); os.makedirs(d, exist_ok=True)
    M = rng.choice([1, 2, 2, 3, 3, 4, 5]); marks = []; sprs = []
    for mi in range(M):
        kind = rng.choice(KINDS)
        if kind == 'character':
            P = dict(kind='character', name='char', font=[], alpha=rng.choice(ALPHAS)); s = char_spr(seed * 10 + mi)
        else:
            P = G.new_mark(rng, kind, 'random', rng.choice(ALPHAS), tier, 720); s = G.build_sprite(P, 720, 1000, rng)
        marks.append(P); sprs.append(s)
    gimg, gboxes = gallery(sprs, rng); Image.fromarray(gimg).save(os.path.join(d, 'gallery.png'))
    K = rng.randint(4, 8); style = rng.choice(G.STYLES); codec = rng.choice(['png', 'jpg95', 'jpg85']); hue = rng.randint(0, 10 ** 6)
    pres = [[rng.random() < .7 for _ in range(M)] for _ in range(K)]
    for k in range(K):
        if rng.random() < .15: pres[k] = [False] * M
    for m in range(M):
        if not any(p[m] for p in pres): pres[rng.randrange(K)][m] = True
    images = []
    for k in range(K):
        H = rng.randint(1100, 2200); img = G.make_comic(seed * 100 + k, 720, H, style, hue); img = add_manga(img, rng); placed = []; ms = []
        for m in range(M):
            if not pres[k][m]: continue
            for _ in range(1 + (rng.random() < .35)):
                for _try in range(12):
                    s = sprs[m].scaled(scale_for(sprs[m].w, marks[m]['kind'], rng))
                    if rng.random() < .3: s = G.spr_rotated(s, rng.uniform(-12, 12))
                    if s.w >= 700 or s.h >= H - 20 or s.w * s.h > MAX_AREA * 720 * H: continue
                    x = rng.randint(0, 720 - s.w); y = rng.randint(0, H - s.h); a = float(np.clip(marks[m]['alpha'] * rng.uniform(.7, 1.3), .1, .9))
                    if any(iou([x, y, s.w, s.h], p) > 0 for p in placed): continue
                    patch = img[y:y + s.h, x:x + s.w].copy(); _, con = G.bake(patch, s, a, 0, 0)
                    if con < MIN_CONTRAST: continue
                    box, con = G.bake(img, s, a, x, y); placed.append([x, y, s.w, s.h]); ms.append(dict(idx=m, box=box, alpha=round(a, 3), contrast=round(con, 1), scale=round(s.w / sprs[m].w, 2))); break
        ext = 'png' if codec == 'png' else 'jpg'; fn = 'img_%02d.%s' % (k + 1, ext); pim = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        if ext == 'png': pim.save(os.path.join(d, fn))
        else: pim.save(os.path.join(d, fn), 'JPEG', quality=int(codec[3:]))
        images.append(dict(file=fn, w=720, h=H, marks=ms))
    return dict(id=sid, tier=tier, style=style, codec=codec, M=M,
                marks=[dict(idx=i, kind=P['kind'], name=P.get('name', ''), font=P.get('font', []), alpha=P['alpha'], gallery=gboxes[i]) for i, P in enumerate(marks)], images=images)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tier', default='core'); ap.add_argument('--out', required=True); ap.add_argument('--n', type=int, default=120); ap.add_argument('--jobs', type=int, default=8); ap.add_argument('--seed', type=int, default=7000)
    a = ap.parse_args(); root = os.path.join(a.out, a.tier); os.makedirs(root, exist_ok=True); base = a.seed + (0 if a.tier == 'core' else 100000)
    tasks = [('ref2_%s_%03d' % (a.tier, i), base + i, a.tier, root) for i in range(a.n)]
    with Pool(a.jobs) as p: sets = p.map(gen_set, tasks, chunksize=2)
    json.dump(dict(tier=a.tier, sets=sets), open(os.path.join(root, 'truth.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    nm = sum(len(im['marks']) for s in sets for im in s['images']); ni = sum(len(s['images']) for s in sets); ne = sum(1 for s in sets for im in s['images'] if not im['marks'])
    print('세트 %d · 이미지 %d(마크 없는 이미지 %d) · 정답 마크 %d' % (len(sets), ni, ne, nm))


if __name__ == '__main__': main()
