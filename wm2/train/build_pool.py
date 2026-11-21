#!/usr/bin/env python3
"""학습용 배경 만화 풀과 워터마크 스프라이트 풀을 만든다(벤치마크와 겹치지 않는 seed).
사용: python3 build_pool.py --out ~/wm2-train --nbg 1200 --nspr 3000 --jobs 2"""
import os, sys, argparse, random
import numpy as np
from multiprocessing import Pool
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))
import gen_bench as G

BW, BH = 360, 500   # 0.5배 축소 크기

def make_bg(i):
    rng = random.Random(10 ** 9 + i); style = rng.choice(G.STYLES)
    img = G.make_comic(10 ** 9 + i, 720, 1000, style, rng.randint(0, 10 ** 6))
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).resize((BW, BH), Image.BOX)
    return np.asarray(im, np.uint8)

KINDS = ['text', 'text', 'badge', 'badge', 'emblem', 'icon', 'url', 'outline', 'qr', 'banner', 'tile_text', 'tile_logo']
def make_spr(i):
    rng = random.Random(2 * 10 ** 9 + i); kind = rng.choice(KINDS)
    mode = {'banner': 'band', 'tile_text': 'tiled', 'tile_logo': 'tiled'}.get(kind, 'anchor')
    P = G.new_mark(rng, kind, mode, 1.0, 'dev', 720)
    spr = G.build_sprite(P, 720, 1000, random.Random(i))
    if kind.startswith('tile') : spr = spr.scaled(0.5)        # 전면 무늬는 0.5배 이미지에 맞춤
    elif kind == 'banner': spr = spr.scaled(0.5)
    else: spr = spr.scaled(0.5)
    ys, xs = np.where(spr.A > .03)
    if len(xs) == 0: return None
    A = (spr.A * 255).astype(np.uint8); Pm = np.clip(spr.P, 0, 255).astype(np.uint8)
    return kind, A, Pm

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=os.path.expanduser('~/wm2-train')); ap.add_argument('--nbg', type=int, default=1200); ap.add_argument('--nspr', type=int, default=3000); ap.add_argument('--jobs', type=int, default=2)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    with Pool(a.jobs) as p:
        sp = [x for x in p.map(make_spr, range(a.nspr), chunksize=20) if x]
        import pickle; pickle.dump(sp, open(os.path.join(a.out, 'sprites.pkl'), 'wb')); print('스프라이트', len(sp))
        bgs = np.stack(p.map(make_bg, range(a.nbg), chunksize=4)); np.save(os.path.join(a.out, 'bgs.npy'), bgs); print('배경', bgs.shape)
