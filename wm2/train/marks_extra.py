"""캐릭터(마스코트) 워터마크 스프라이트 합성. 사용자 분류: '캐릭터' = 워터마크가 마스코트 그림일 수 있음(단독 또는 이름 글자와 함께).
반환 형식은 풀 스프라이트와 같다: (kind, A uint8 [h,w], P uint8 [h,w,3] = 색*A). scale=1 은 0.5배 세계(학습 풀 크기), 2 는 원본 해상도(벤치)."""
import os, sys, math, random
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))
import gen_bench as G

TYPES = ['cat', 'bear', 'rabbit', 'blob', 'robot', 'bird']


def _mascot_mask(r, D, style, typ, ss=4):
    """D x D 마스코트 마스크(0..1)"""
    S = D * ss; im = Image.new('L', (S, S), 0); d = ImageDraw.Draw(im); c = S / 2; hr = S * r.uniform(.30, .38); hy = S * .56
    lw = max(2, int(S * r.uniform(.02, .035)))
    def head(fill): d.ellipse([c - hr, hy - hr, c + hr, hy + hr], fill=fill)
    def ear(x, y, rr, fill): d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=fill)
    # 귀·장식(실루엣)
    if typ == 'cat':
        for sx in (-1, 1): d.polygon([(c + sx * hr * .95, hy - hr * .35), (c + sx * hr * .55, hy - hr * 1.25), (c + sx * hr * .1, hy - hr * .85)], fill=255)
    elif typ == 'bear':
        for sx in (-1, 1): ear(c + sx * hr * .8, hy - hr * .85, hr * .32, 255)
    elif typ == 'rabbit':
        for sx in (-1, 1): d.ellipse([c + sx * hr * .45 - hr * .2, hy - hr * 2.0, c + sx * hr * .45 + hr * .2, hy - hr * .7], fill=255)
    elif typ == 'robot':
        d.rectangle([c - hr * .06, hy - hr * 1.5, c + hr * .06, hy - hr * .8], fill=255); ear(c, hy - hr * 1.55, hr * .14, 255)
    elif typ == 'bird':
        d.polygon([(c - hr * .2, hy - hr * .95), (c, hy - hr * 1.5), (c + hr * .2, hy - hr * .95)], fill=255)
    head(255)
    if typ == 'robot': d.rectangle([c - hr, hy - hr * .85, c + hr, hy + hr * .85], fill=255)
    body = r.random() < .5
    if body: d.ellipse([c - hr * .6, hy + hr * .85, c + hr * .6, hy + hr * 1.9], fill=255)
    if style == 'outline':   # 속을 파내 외곽선만 남김
        inner = im.copy(); ImageDraw.Draw(inner)
        from PIL import ImageFilter
        er = im.filter(ImageFilter.MinFilter(2 * lw + 1)); im = Image.fromarray(np.clip(np.asarray(im, np.int16) - np.asarray(er, np.int16), 0, 255).astype(np.uint8)); d = ImageDraw.Draw(im)
        fill_feat = 255
    else: fill_feat = 0
    # 얼굴 요소(눈·입·볼)
    ey = hy - hr * .1; ex = hr * r.uniform(.35, .5); er_ = hr * r.uniform(.09, .15)
    for sx in (-1, 1):
        if typ == 'robot': d.rectangle([c + sx * ex - er_ * 1.4, ey - er_, c + sx * ex + er_ * 1.4, ey + er_], fill=fill_feat)
        else: d.ellipse([c + sx * ex - er_, ey - er_ * 1.2, c + sx * ex + er_, ey + er_ * 1.2], fill=fill_feat)
    my = hy + hr * .35
    if r.random() < .7: d.arc([c - hr * .3, my - hr * .2, c + hr * .3, my + hr * .25], 20, 160, fill=fill_feat, width=lw)
    else: d.ellipse([c - hr * .1, my - hr * .08, c + hr * .1, my + hr * .12], fill=fill_feat)
    if r.random() < .4:
        for sx in (-1, 1): d.ellipse([c + sx * hr * .72 - hr * .1, my - hr * .25, c + sx * hr * .72 + hr * .1, my - hr * .05], fill=fill_feat)
    a = np.asarray(im.resize((D, D), Image.LANCZOS), np.float32) / 255
    return a


def character_sprite(seed, scale=1):
    """캐릭터 스프라이트 하나. 종류: 마스코트 단독 / 마스코트 + 이름 글자 / 원형 틀 안의 마스코트"""
    r = random.Random(seed); nr = np.random.default_rng(seed)
    white = (r.randint(235, 255),) * 3; black = (r.randint(10, 40),) * 3; brand = r.choice(G.BRANDS)
    col = r.choice([white, white, black, brand]); typ = r.choice(TYPES); style = r.choice(['filled', 'filled', 'outline'])
    D = int(r.randint(44, 90) * scale); m = _mascot_mask(nr, D, style, typ); form = r.choice(['solo', 'solo', 'text', 'ring'])
    if form == 'ring':
        Dr = int(D * 1.35); ring = np.zeros((Dr, Dr), np.float32); yy, xx = np.mgrid[0:Dr, 0:Dr]; rr = np.hypot(xx - Dr / 2, yy - Dr / 2); rw = max(2, int(2.2 * scale))
        ring[(rr <= Dr / 2) & (rr >= Dr / 2 - rw)] = 1; o = (Dr - D) // 2; ring[o:o + D, o:o + D] = np.maximum(ring[o:o + D, o:o + D], m); m = ring
    elif form == 'text':
        P = G.new_mark(r, 'text', 'random', 1.0, 'dev', 720); fnt = G.font(tuple(P['font']), max(10, int(D * .32))); f, _ = G.tlayers(P['name'], fnt)
        w = max(m.shape[1], f.shape[1]); out = np.zeros((m.shape[0] + f.shape[0] + 2, w), np.float32); out[:m.shape[0], (w - m.shape[1]) // 2:(w - m.shape[1]) // 2 + m.shape[1]] = m
        out[m.shape[0] + 2:, (w - f.shape[1]) // 2:(w - f.shape[1]) // 2 + f.shape[1]] = f; m = out
    A = (np.clip(m, 0, 1) * 255).astype(np.uint8); Pm = (A[..., None].astype(np.float32) / 255 * np.array(col, np.float32)).astype(np.uint8)
    return ('character', A, Pm)


if __name__ == '__main__':
    sp = [character_sprite(i) for i in range(24)]; W = 8 * 110; H = 3 * 110; can = Image.new('RGB', (W, H), (128, 128, 150))
    for i, (k, A, P) in enumerate(sp):
        im = Image.fromarray(np.concatenate([np.clip(P, 0, 255), A[..., None]], -1).astype(np.uint8), 'RGBA') if False else Image.fromarray(np.dstack([P, A]).astype(np.uint8), 'RGBA')
        im.thumbnail((100, 100)); can.paste(im, ((i % 8) * 110 + 5, (i // 8) * 110 + 5), im)
    can.save(sys.argv[1] if len(sys.argv) > 1 else 'chars.png'); print('ok')
