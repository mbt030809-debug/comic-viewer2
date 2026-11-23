#!/usr/bin/env python3
"""2단계 파이프라인 벤치 v3 (사용자 확정 사양, wm2/RECOGNITION_SPEC_20261123.md + 2026-11-23 추가 답변)
 1단계: 사이트 이미지(3~8장) 중 **어느 한 장(위치 모름)**의 **단색(흰/검은) 여백**에 '예시 워터마크'가 놓여 있다. 그 장은 일반 만화 페이지(컷 + 단색 여백 띠).
 2단계: 나머지 이미지의 웹툰 그림 위에 같은 마크가 박혀 있다 — **불투명(알파 1.0) 중심 60%, 반투명(0.15~0.7) 40%**. 위치·크기 매번 다름, 한 이미지에 여러 개, 일부 이미지엔 없음.
 방해물: 다른 장의 단색 여백에는 캡션·쪽번호·선 같은 글자 요소(예시 마크 아님), 그림에는 말풍선·집중선·머리카락·눈·불꽃.
 크기: 로고류 폭 8~30%, 글자류 10~50%, 면적 ≤ 화면 20%.
정답(truth.json): 이미지별 example_marks(예시 번호, 상자: 1단계 정답), marks(예시 번호, 상자, 알파: 2단계 정답), decoy_margins.
사용: python gen_refbench3.py --tier core|val --out <dir> --n 120 --jobs 8"""
import os, sys, json, random, argparse
import numpy as np
from PIL import Image, ImageDraw
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
import gen_bench as G
from gen_refbench import iou, MIN_CONTRAST, MAX_AREA
from gen_refbench2 import KINDS, char_spr, add_manga, scale_for, W0

CAPTIONS = ['To be continued', 'NEXT EPISODE', '다음 화에 계속', 'END', 'page 12', '12', '— 3 —', 'Copyright', '제 12화', 'The end', '다음 이야기', 'Episode 7']


def render_opaque(spr, bgv):
    """스프라이트를 단색 배경(bgv)에 불투명하게 그린 조각(h,w,3 float)"""
    return bgv * (1 - spr.A[..., None]) + spr.P


def make_band(sprs, kinds, rng, W=W0):
    """예시 마크들을 흰/검은 단색 띠에 배치(마크가 잘 보이도록 마크 밝기에 맞춰 배경을 고른다 → 흰 띠·검은 띠 최대 2개를 이어 붙임).
    반환 (띠 이미지 float, [(idx, 상자[x,y,w,h](띠 좌표))], 배경값 목록)"""
    from gen_refbench import lum
    groups = {0.: [], 255.: []}
    for i, s in enumerate(sprs):
        l = lum(s); bgv = 0. if l > 150 else 255. if l < 105 else (0. if rng.random() < .5 else 255.)
        groups[bgv].append((i, s.scaled(scale_for(s.w, kinds[i], rng))))
    bands = []; boxes = []; y = 0; bgs = []
    for bgv, items in groups.items():
        if not items: continue
        b, bx = _layout(items, bgv, rng, W); bands.append(b); boxes += [(i, [x0, y0 + y, w, h]) for i, (x0, y0, w, h) in bx]; y += b.shape[0]; bgs.append(bgv)
    return np.concatenate(bands, 0), boxes, bgs


def _layout(items, bgv, rng, W=W0):
    rows = [[]]; rw = 0; pad = rng.randint(30, 70)
    for it in items:
        if rw + it[1].w + 30 > W - 2 * 20 and rows[-1]: rows.append([]); rw = 0
        rows[-1].append(it); rw += it[1].w + rng.randint(30, 90)
    heights = [max(s.h for _, s in r) for r in rows]; H = sum(heights) + (len(rows) + 1) * pad
    band = np.full((H, W, 3), bgv, np.float32); boxes = []; y = pad
    for r, hh in zip(rows, heights):
        tot = sum(s.w for _, s in r); gap = (W - tot) // (len(r) + 1) + rng.randint(-10, 10); gap = max(20, gap)
        x = max(10, (W - tot - gap * (len(r) - 1)) // 2)
        for i, s in r:
            yy = y + (hh - s.h) // 2; xx = min(x, W - s.w - 2); band[yy:yy + s.h, xx:xx + s.w] = render_opaque(s, bgv)
            ys, xs = np.where(s.A > .03); boxes.append((i, [int(xx + xs.min()), int(yy + ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]))
            x += s.w + gap
        y += hh + pad
    return band, boxes


def decoy_band(rng, W=W0):
    """예시 마크가 아닌 단색 띠(캡션·쪽번호·선). 반환 (띠 이미지, [글자 상자])"""
    bgv = 0. if rng.random() < .5 else 255.; fg = 255 - int(bgv); H = rng.randint(90, 260); im = Image.new('RGB', (W, H), (int(bgv),) * 3); d = ImageDraw.Draw(im); boxes = []
    for _ in range(rng.randint(1, 3)):
        fnt = G.font(rng.choice(G.LAT_FONTS[False] + G.KR_FONTS[False]), rng.randint(18, 40)); txt = rng.choice(CAPTIONS)
        bb = d.textbbox((0, 0), txt, font=fnt); w, h = bb[2] - bb[0], bb[3] - bb[1]
        if w >= W - 40: continue
        x = rng.randint(10, W - w - 10); y = rng.randint(5, max(6, H - h - 10)); d.text((x - bb[0], y - bb[1]), txt, font=fnt, fill=(fg,) * 3); boxes.append([x, y, w, h])
    if rng.random() < .5: yy = rng.randint(10, H - 10); d.line([(rng.randint(0, 200), yy), (rng.randint(400, W), yy)], fill=(fg,) * 3, width=rng.randint(1, 4))
    return np.asarray(im, np.float32), boxes


def insert_band(img, band, y):
    return np.concatenate([img[:y], band, img[y:]], 0), y


def gen_set(args):
    sid, seed, tier, root = args
    rng = random.Random(seed); d = os.path.join(root, sid); os.makedirs(d, exist_ok=True)
    M = rng.choice([1, 2, 2, 3, 3, 4]); marks = []; sprs = []; kinds = []
    for mi in range(M):
        kind = rng.choice(KINDS)
        if kind == 'character': P = dict(kind='character', name='char', font=[], alpha=1.0); s = char_spr(seed * 10 + mi)
        else: P = G.new_mark(rng, kind, 'random', 1.0, tier, 720); s = G.build_sprite(P, 720, 1000, rng)
        marks.append(P); sprs.append(s); kinds.append(kind)
    K = rng.randint(4, 8); ex_page = rng.randrange(K); style = rng.choice(G.STYLES); codec = rng.choice(['png', 'jpg95', 'jpg85']); hue = rng.randint(0, 10 ** 6)
    pres = [[rng.random() < .7 for _ in range(M)] for _ in range(K)]
    for k in range(K):
        if rng.random() < .15: pres[k] = [False] * M
    for m in range(M):
        if not any(p[m] for k, p in enumerate(pres) if k != ex_page): pres[rng.choice([k for k in range(K) if k != ex_page] or [0])][m] = True
    pres[ex_page] = [False] * M   # 예시 페이지에는 그림 위 마크를 두지 않는다(1단계 정답이 모호해지지 않게)
    images = []
    for k in range(K):
        H = rng.randint(1000, 2000); img = G.make_comic(seed * 100 + k, 720, H, style, hue); img = add_manga(img, rng); ex = []; decoys = []; reserved = []
        if k == ex_page:
            band, bx, _bgs = make_band(sprs, kinds, rng); y = rng.randint(150, max(151, H - 150)); img, _ = insert_band(img, band, y); H = img.shape[0]
            ex = [dict(idx=i, box=[b[0], b[1] + y, b[2], b[3]]) for i, b in bx]; reserved.append([0, y, 720, band.shape[0]])
        elif rng.random() < .4:
            band, tb = decoy_band(rng); y = rng.randint(100, max(101, H - 100)); img, _ = insert_band(img, band, y); H = img.shape[0]
            decoys = [dict(box=[b[0], b[1] + y, b[2], b[3]]) for b in tb]; reserved.append([0, y, 720, band.shape[0]])
        placed = list(reserved); ms = []
        for m in range(M):
            if not pres[k][m]: continue
            for _ in range(1 + (rng.random() < .3)):
                for _try in range(14):
                    s = sprs[m].scaled(scale_for(sprs[m].w, kinds[m], rng))
                    if rng.random() < .3: s = G.spr_rotated(s, rng.uniform(-12, 12))
                    if s.w >= 700 or s.h >= H - 20 or s.w * s.h > MAX_AREA * 720 * H: continue
                    x = rng.randint(0, 720 - s.w); y = rng.randint(0, H - s.h); a = 1.0 if rng.random() < .6 else float(rng.uniform(.15, .7))
                    if any(iou([x, y, s.w, s.h], p) > 0 for p in placed): continue
                    patch = img[y:y + s.h, x:x + s.w].copy(); _, con = G.bake(patch, s, a, 0, 0)
                    if con < MIN_CONTRAST: continue
                    box, con = G.bake(img, s, a, x, y); placed.append([x, y, s.w, s.h]); ms.append(dict(idx=m, box=box, alpha=round(a, 3), contrast=round(con, 1), scale=round(s.w / sprs[m].w, 2))); break
        ext = 'png' if codec == 'png' else 'jpg'; fn = 'img_%02d.%s' % (k + 1, ext); pim = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        if ext == 'png': pim.save(os.path.join(d, fn))
        else: pim.save(os.path.join(d, fn), 'JPEG', quality=int(codec[3:]))
        images.append(dict(file=fn, w=720, h=H, example_marks=ex, decoy_margins=decoys, marks=ms))
    return dict(id=sid, tier=tier, style=style, codec=codec, M=M, ex_page=ex_page,
                marks=[dict(idx=i, kind=P['kind'], name=P.get('name', ''), font=P.get('font', [])) for i, P in enumerate(marks)], images=images)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tier', default='core'); ap.add_argument('--out', required=True); ap.add_argument('--n', type=int, default=120); ap.add_argument('--jobs', type=int, default=8); ap.add_argument('--seed', type=int, default=9000)
    a = ap.parse_args(); root = os.path.join(a.out, a.tier); os.makedirs(root, exist_ok=True); base = a.seed + (0 if a.tier == 'core' else 100000)
    tasks = [('ref3_%s_%03d' % (a.tier, i), base + i, a.tier, root) for i in range(a.n)]
    with Pool(a.jobs) as p: sets = p.map(gen_set, tasks, chunksize=2)
    json.dump(dict(tier=a.tier, sets=sets), open(os.path.join(root, 'truth.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    ni = sum(len(s['images']) for s in sets); nex = sum(len(im['example_marks']) for s in sets for im in s['images']); nm = sum(len(im['marks']) for s in sets for im in s['images'])
    nd = sum(1 for s in sets for im in s['images'] if im['decoy_margins']); op = sum(1 for s in sets for im in s['images'] for m in im['marks'] if m['alpha'] >= .99)
    print('세트 %d · 이미지 %d · 예시 마크 %d(1단계 정답) · 그림 위 마크 %d(불투명 %d) · 캡션 여백 이미지 %d' % (len(sets), ni, nex, nm, op, nd))


if __name__ == '__main__': main()
