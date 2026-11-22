#!/usr/bin/env python3
"""wm2 워터마크 인식 벤치마크 생성기 (Pillow + numpy).

만화(웹툰 세로 이미지)를 절차적으로 만들고, 워터마크를 종류·위치·투명도별로 굽고, 정답 상자(truth.json)와
눈으로 확인하는 정답지 페이지(answer.html)를 함께 낸다. 같은 seed면 항상 같은 결과.

사용:  python3 gen_bench.py --tier core|wild|val --out ~/wm2-bench [--only 부분문자열] [--limit N] [--clean]
  core : 축(종류 x 위치 x 투명도)을 체계적으로 훑는 사이트 + 음성(워터마크 없음·미끼) 사이트
  wild : 무작위 조합(분포 측정용, 수정할 때 보는 사이트)
  val  : 다른 seed + 따로 둔 글꼴(수정 후 일반화 확인용. 고칠 때는 보지 않는다)
출력(사이트마다):  img_01.png|jpg ... / th_01.jpg(썸네일) / mark_01.png(굽힌 워터마크 그림) / index.html(앱 시험용 세로 나열 페이지)
"""
import os, sys, json, math, random, argparse, colorsys, shutil
from multiprocessing import Pool
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FD = os.environ.get('WM2_FONT_DIR', '/system/fonts/')   # PC 에서는 WM2_FONT_DIR 로 글꼴 폴더를 지정(없는 글꼴은 아래 대체 글꼴 사용)
_FALLBACK = [os.environ.get('WM2_FALLBACK_FONT', ''), '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
             '/Library/Fonts/Arial Unicode.ttf', 'C:/Windows/Fonts/malgun.ttf', 'C:/Windows/Fonts/arial.ttf']
LAT_FONTS = {False: [('DroidSans-Bold.ttf', 0), ('DroidSans.ttf', 0), ('NotoSerif-Bold.ttf', 0), ('DancingScript-Regular.ttf', 0),
                     ('ComingSoon.ttf', 0), ('CarroisGothicSC-Regular.ttf', 0)],
             True: [('NotoSerif-Italic.ttf', 0), ('CutiveMono.ttf', 0), ('DroidSansMono.ttf', 0)]}
KR_FONTS = {False: [('NotoSansCJK-Regular.ttc', 1)], True: [('NotoSerifCJK-Regular.ttc', 1)]}
KRS = list("가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허고노도로모보소오조초코토포호구누두루무부수우주추쿠투푸후그느드르므브스으즈츠크트프흐기니디리미비시이지치키티피히")
SYL = ['to', 'on', 'ma', 'ng', 'ko', 'ri', 'ta', 'zu', 'bo', 'mi', 'la', 'xe', 'pa', 'ni', 've', 'do', 'ka', 'sy', 'qu', 'ju']
SUF = ['toon', 'manga', 'comics', 'hub', '.com', 'scan', 'webtoon', 'zone', 'world']
BRANDS = [(220, 40, 40), (40, 90, 200), (30, 160, 90), (240, 150, 20), (120, 50, 170), (20, 20, 20), (235, 235, 235), (230, 70, 140), (0, 150, 170)]
_fc = {}
# 폰 글꼴이 없는 PC(Windows)에서 글꼴 종류별로 서로 다른 시스템 글꼴을 쓰도록 하는 대응표(없으면 아래 단일 대체 글꼴).
_WIN = 'C:/Windows/Fonts/'
_SUBST = {'DroidSans-Bold.ttf': ('arialbd.ttf', 0), 'DroidSans.ttf': ('segoeui.ttf', 0), 'NotoSerif-Bold.ttf': ('timesbd.ttf', 0),
          'DancingScript-Regular.ttf': ('segoepr.ttf', 0), 'ComingSoon.ttf': ('comic.ttf', 0), 'CarroisGothicSC-Regular.ttf': ('bahnschrift.ttf', 0),
          'NotoSerif-Italic.ttf': ('timesi.ttf', 0), 'CutiveMono.ttf': ('cour.ttf', 0), 'DroidSansMono.ttf': ('consola.ttf', 0),
          'NotoSansCJK-Regular.ttc': ('malgun.ttf', 0), 'NotoSerifCJK-Regular.ttc': ('batang.ttc', 0)}


def font(spec, size):
    k = (spec, size)
    if k not in _fc:
        try: _fc[k] = ImageFont.truetype(FD + spec[0], size, index=spec[1])
        except OSError:
            s = _SUBST.get(spec[0])
            if s and os.path.exists(_WIN + s[0]):
                try: _fc[k] = ImageFont.truetype(_WIN + s[0], size, index=s[1]); return _fc[k]
                except OSError: pass
            for p in _FALLBACK:
                if p and os.path.exists(p):
                    try: _fc[k] = ImageFont.truetype(p, size); break
                    except OSError: pass
            else: raise
    return _fc[k]


# ============================================================== 만화 배경 생성
def mkpal(rng, style):
    h0 = rng.random()
    def c():
        h = (h0 + rng.choice([0, 0, 0.08, -0.08, 0.5])) % 1
        if style == 'dark': s, v = rng.uniform(.3, .8), rng.uniform(.15, .5)
        else: s, v = rng.uniform(.15, .7), rng.uniform(.6, 1)
        return tuple(int(x * 255) for x in colorsys.hsv_to_rgb(h, s, v))
    return c


def kr_words(rng, n):
    return ' '.join(''.join(rng.choice(KRS) for _ in range(rng.randint(2, 4))) for _ in range(n))


def arr2im(a): return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def bg_vgrad(pw, ph, c0, c1):
    t = np.linspace(0, 1, ph, dtype=np.float32)[:, None, None]
    return np.broadcast_to(np.array(c0, np.float32) * (1 - t) + np.array(c1, np.float32) * t, (ph, pw, 3)).copy()


def bg_radial(pw, ph, c0, c1, cx, cy):
    yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2); d = np.clip(d / (max(pw, ph) * .8), 0, 1)[..., None]
    return np.array(c0, np.float32) * (1 - d) + np.array(c1, np.float32) * d


def bg_noise(nrng, pw, ph, c0, c1):
    out = np.zeros((ph, pw), np.float32)
    for sc, amp in ((56, 1.0), (18, .5), (7, .25)):
        a = nrng.random((ph // sc + 2, pw // sc + 2)).astype(np.float32)
        out += amp * np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((pw, ph), Image.BICUBIC), np.float32) / 255
    out = (out - out.min()) / (out.max() - out.min() + 1e-6)
    return np.array(c0, np.float32) * (1 - out[..., None]) + np.array(c1, np.float32) * out[..., None]


def bg_halftone(pw, ph, base, ink, sp, ang, radial):
    yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
    ca, sa = math.cos(ang), math.sin(ang)
    u = (xx * ca + yy * sa) / sp; v = (-xx * sa + yy * ca) / sp
    fu = u - np.floor(u) - .5; fv = v - np.floor(v) - .5; d = np.sqrt(fu * fu + fv * fv)
    ramp = np.sqrt(((xx - pw / 2) / pw) ** 2 + ((yy - ph / 2) / ph) ** 2) * 1.4 if radial else yy / ph
    m = (d < (.12 + .38 * np.clip(ramp, 0, 1))).astype(np.float32)[..., None]
    return np.array(base, np.float32) * (1 - m) + np.array(ink, np.float32) * m


def make_panel(rng, nrng, pw, ph, style, pal, border):
    kinds = {'toon': ['noise', 'vgrad', 'city', 'room', 'radial', 'flat', 'sky'], 'bw': ['halftone', 'hatch', 'speed', 'flat', 'noise'],
             'scroll': ['flat', 'vgrad', 'flat', 'noise'], 'busy': ['halftone', 'hatch', 'noise', 'speed'], 'dark': ['noise', 'city', 'vgrad', 'radial'], 'flat': ['flat', 'flat', 'vgrad']}[style]
    k = rng.choice(kinds)
    c0, c1 = pal(), pal()
    if k == 'flat': a = np.broadcast_to(np.array(c0 if style != 'bw' else (255, 255, 255), np.float32), (ph, pw, 3)).copy()
    elif k == 'vgrad': a = bg_vgrad(pw, ph, c0, c1)
    elif k == 'radial': a = bg_radial(pw, ph, c0, c1, rng.uniform(0, pw), rng.uniform(0, ph))
    elif k == 'noise': a = bg_noise(nrng, pw, ph, c0, c1)
    elif k == 'halftone': a = bg_halftone(pw, ph, (255, 255, 255), (40, 40, 40), rng.uniform(5, 10), rng.uniform(0, 1.5), rng.random() < .5)
    else: a = bg_vgrad(pw, ph, c0, c1)
    im = arr2im(a); d = ImageDraw.Draw(im)
    if k == 'hatch':
        im = Image.new('RGB', (pw, ph), (255, 255, 255)); d = ImageDraw.Draw(im); sp = rng.randint(5, 9)
        for i in range(-ph, pw, sp): d.line([(i, 0), (i + ph, ph)], fill=(50, 50, 50), width=1)
    if k == 'speed':
        cx, cy = pw * rng.uniform(.3, .7), ph * rng.uniform(.3, .7)
        for _ in range(rng.randint(60, 140)):
            ang = rng.uniform(0, 6.283); r0 = rng.uniform(.25, .6) * min(pw, ph); r1 = max(pw, ph)
            d.line([(cx + r0 * math.cos(ang), cy + r0 * math.sin(ang)), (cx + r1 * math.cos(ang), cy + r1 * math.sin(ang))], fill=(30, 30, 30), width=rng.randint(1, 3))
    if k == 'sky':
        for _ in range(rng.randint(2, 6)):
            x, y = rng.uniform(0, pw), rng.uniform(0, ph * .5); w = rng.uniform(80, 220)
            d.ellipse([x, y, x + w, y + w * .4], fill=(250, 250, 255))
    if k == 'city':
        x = 0
        while x < pw:
            bw = rng.randint(40, 120); bh = rng.randint(int(ph * .2), int(ph * .7)); col = pal()
            col = tuple(int(v * .5) for v in col)
            d.rectangle([x, ph - bh, x + bw, ph], fill=col)
            for wy in range(ph - bh + 12, ph - 10, 26):
                for wx in range(x + 8, x + bw - 14, 22):
                    if rng.random() < .6: d.rectangle([wx, wy, wx + 10, wy + 14], fill=(250, 230, 130))
            x += bw + rng.randint(0, 6)
    if k == 'room':
        fy = int(ph * rng.uniform(.65, .8)); d.rectangle([0, fy, pw, ph], fill=pal())
        wx, wy = int(pw * rng.uniform(.1, .6)), int(ph * .12); ww, wh = int(pw * .28), int(ph * .3)
        d.rectangle([wx, wy, wx + ww, wy + wh], fill=(190, 225, 250), outline=(60, 60, 60), width=4)
        d.line([(wx + ww // 2, wy), (wx + ww // 2, wy + wh)], fill=(60, 60, 60), width=3)
    ncha = 0 if style == 'flat' and rng.random() < .5 else rng.choice([0, 1, 1, 2, 2, 3])
    for _ in range(ncha):
        sc = rng.uniform(.5, 1.3) * (ph / 520.0)
        draw_char(d, rng, rng.uniform(.15, .85) * pw, ph * rng.uniform(.88, 1.0), sc, pal)
    for _ in range(rng.choice([0, 1, 1, 2, 3])):
        draw_bubble(d, rng, pw, ph)
    if border:
        w = rng.choice([2, 3, 4, 5]); d.rectangle([0, 0, pw - 1, ph - 1], outline=(10, 10, 10), width=w)
    return im


def draw_char(d, rng, cx, by, sc, pal):
    lw = max(2, int(3 * sc)); sk = (255, rng.randint(205, 232), rng.randint(180, 215)); hair = tuple(rng.randint(10, 90) for _ in range(3))
    shirt, pants = pal(), pal(); k = lambda v: int(v * sc)
    ty = by - k(90) - k(110); hr = k(32); hy = ty - hr + k(4)
    for dx in (-26, 4): d.rectangle([cx + k(dx), by - k(90), cx + k(dx + 22), by], fill=pants, outline=(0, 0, 0), width=lw)
    d.polygon([(cx - k(40), ty), (cx + k(40), ty), (cx + k(30), ty + k(110)), (cx - k(30), ty + k(110))], fill=shirt, outline=(0, 0, 0), width=lw)
    d.line([(cx - k(40), ty + k(10)), (cx - k(62), ty + k(95))], fill=sk, width=k(12)); d.line([(cx + k(40), ty + k(10)), (cx + k(62), ty + k(95))], fill=sk, width=k(12))
    d.ellipse([cx - hr, hy - hr, cx + hr, hy + hr], fill=sk, outline=(0, 0, 0), width=lw)
    d.pieslice([cx - hr - 3, hy - hr - k(6), cx + hr + 3, hy + hr - k(4)], 180, 360, fill=hair, outline=(0, 0, 0), width=lw)
    for ex in (-12, 12): d.ellipse([cx + k(ex) - 3, hy - 2, cx + k(ex) + 3, hy + k(8)], fill=(20, 20, 20))
    d.arc([cx - k(10), hy + k(10), cx + k(10), hy + k(24)], 20, 160, fill=(120, 30, 30), width=max(1, lw - 1))


def draw_bubble(d, rng, pw, ph):
    n = rng.randint(1, 3); fs = rng.randint(16, 26); fnt = font(KR_FONTS[False][0], fs)
    lines = [kr_words(rng, rng.randint(1, 3)) for _ in range(n)]
    tw = max(d.textlength(l, font=fnt) for l in lines); w = int(tw + 50); h = int(n * fs * 1.5 + 36)
    if w > pw - 20 or h > ph - 20: return
    x = rng.randint(10, pw - w - 10); y = rng.randint(10, max(11, int(ph * .55)))
    d.ellipse([x, y, x + w, y + h], fill=(255, 255, 255), outline=(0, 0, 0), width=3)
    tx, ty = x + w * rng.uniform(.3, .7), y + h
    d.polygon([(tx - 10, y + h - 6), (tx + 10, y + h - 6), (tx + rng.uniform(-30, 30), min(ph - 2, y + h + 30))], fill=(255, 255, 255), outline=(0, 0, 0))
    d.ellipse([x + 4, y + 4, x + w - 4, y + h - 4], fill=(255, 255, 255))
    for i, l in enumerate(lines):
        d.text((x + w / 2 - d.textlength(l, font=fnt) / 2, y + 16 + i * fs * 1.5), l, font=fnt, fill=(15, 15, 15))


def make_comic(seed, W, H, style, hue_seed):
    rng = random.Random(seed); nrng = np.random.default_rng(seed)
    pal = mkpal(random.Random(hue_seed), style)
    dark = style == 'dark'
    im = Image.new('RGB', (W, H), (18, 18, 26) if dark else (255, 255, 255))
    y = rng.randint(0, 40)
    while y < H - 100:
        y += {'scroll': rng.randint(100, 420), 'toon': rng.randint(8, 40), 'bw': rng.randint(10, 30), 'dark': rng.randint(8, 30), 'flat': rng.randint(60, 200), 'busy': rng.randint(4, 12)}[style]
        ph = min(rng.randint(260, 720), H - y)
        if ph < 120: break
        lay = rng.choice(['full', 'full', 'inset', 'split'])
        bord = style != 'scroll' or rng.random() < .4
        if lay == 'split' and W > 500:
            gw = rng.randint(6, 20); pw = (W - gw) // 2
            im.paste(make_panel(rng, nrng, pw, ph, style, pal, bord), (0, y)); im.paste(make_panel(rng, nrng, W - gw - pw, ph, style, pal, bord), (pw + gw, y))
        else:
            m = rng.randint(10, 60) if lay == 'inset' else 0
            im.paste(make_panel(rng, nrng, W - 2 * m, ph, style, pal, bord), (m, y))
        y += ph
    im = im.filter(ImageFilter.GaussianBlur(rng.uniform(.4, .8)))
    a = np.asarray(im, np.float32) + nrng.normal(0, 1.0, (H, W, 1)).astype(np.float32)
    if style == 'bw': a = np.repeat(a.mean(axis=2, keepdims=True), 3, axis=2)
    return np.clip(a, 0, 255)


# ============================================================== 워터마크 스프라이트
class Spr:
    """A: 불투명도 모양(0..1), P: A를 곱한 색(미리 곱함). 합성: out = I*(1-a*A) + a*P"""
    def __init__(s, w, h):
        s.w, s.h = w, h; s.A = np.zeros((h, w), np.float32); s.P = np.zeros((h, w, 3), np.float32)

    def over(s, m, col, x=0, y=0):
        mm = np.zeros((s.h, s.w), np.float32); h, w = m.shape
        x0, y0 = max(0, x), max(0, y); x1, y1 = min(s.w, x + w), min(s.h, y + h)
        if x1 <= x0 or y1 <= y0: return s
        mm[y0:y1, x0:x1] = m[y0 - y:y1 - y, x0 - x:x1 - x]
        s.P = mm[..., None] * np.array(col, np.float32) + (1 - mm[..., None]) * s.P; s.A = mm + (1 - mm) * s.A
        return s

    def scaled(s, f):
        w, h = max(8, int(s.w * f)), max(8, int(s.h * f)); o = Spr(w, h)
        o.A = np.asarray(Image.fromarray(s.A, 'F').resize((w, h), Image.LANCZOS), np.float32).clip(0, 1)
        o.P = np.stack([np.asarray(Image.fromarray(s.P[..., i], 'F').resize((w, h), Image.LANCZOS), np.float32) for i in range(3)], -1).clip(0, 255) * (o.A[..., None] > 0)
        return o


def spr_rotated(s, ang):
    o = Spr(1, 1)
    A = Image.fromarray(s.A, 'F').rotate(ang, resample=Image.BICUBIC, expand=True)
    o.A = np.asarray(A, np.float32).clip(0, 1); o.h, o.w = o.A.shape
    o.P = np.stack([np.asarray(Image.fromarray(s.P[..., i], 'F').rotate(ang, resample=Image.BICUBIC, expand=True), np.float32) for i in range(3)], -1).clip(0, 255) * (o.A[..., None] > 0)
    return o


def smask(w, h, fn, ss=3):
    im = Image.new('L', (w * ss, h * ss), 0); fn(ImageDraw.Draw(im), ss)
    return np.asarray(im.resize((w, h), Image.LANCZOS), np.float32) / 255


def tlayers(txt, fnt, stroke=0, pad=3):
    probe = ImageDraw.Draw(Image.new('L', (8, 8))); bb = probe.textbbox((0, 0), txt, font=fnt, stroke_width=stroke)
    w, h = bb[2] - bb[0] + 2 * pad, bb[3] - bb[1] + 2 * pad
    f = Image.new('L', (w, h), 0); ImageDraw.Draw(f).text((pad - bb[0], pad - bb[1]), txt, font=fnt, fill=255)
    s = Image.new('L', (w, h), 0); ImageDraw.Draw(s).text((pad - bb[0], pad - bb[1]), txt, font=fnt, fill=255, stroke_width=stroke, stroke_fill=255)
    return np.asarray(f, np.float32) / 255, np.asarray(s, np.float32) / 255


def star_pts(cx, cy, r, ri):
    return [(cx + (r if i % 2 == 0 else ri) * math.sin(i * math.pi / 5), cy - (r if i % 2 == 0 else ri) * math.cos(i * math.pi / 5)) for i in range(10)]


def build_sprite(P, W, H, rng):
    """P: 워터마크 사양. 반환 Spr. 타일형은 이미지 크기(W,H)로 만든다."""
    k = P['kind']; name = P['name']; col = tuple(P['color']); con = tuple(P['contrast']); fs = P['size']
    fnt = font(tuple(P['font']), fs) if fs else None
    if k in ('text', 'outline', 'url'):
        txt = name if k != 'url' else 'www.' + name.lower().replace(' ', '') + '.com'
        stroke = P.get('stroke', 0) if k == 'text' else (2 if k == 'outline' else 0)
        f, s = tlayers(txt, fnt, stroke)
        spr = Spr(f.shape[1], f.shape[0])
        if k == 'outline': spr.over(np.clip(s - f, 0, 1), col)
        else:
            if stroke: spr.over(s, con)
            spr.over(f, col)
        return spr
    if k == 'badge':
        f, _ = tlayers(name, fnt); h = int(f.shape[0] * 1.5) + 8; w = f.shape[1] + h + 18
        spr = Spr(w, h); r = h // 3
        spr.over(smask(w, h, lambda d, ss: d.rounded_rectangle([0, 0, w * ss - 1, h * ss - 1], radius=r * ss, fill=255)), col)
        spr.over(smask(h, h, lambda d, ss: d.ellipse([h * ss * .2, h * ss * .2, h * ss * .8, h * ss * .8], fill=255)), con, 4, 0)
        spr.over(smask(h, h, lambda d, ss: d.polygon(star_pts(h * ss / 2, h * ss / 2, h * ss * .22, h * ss * .09), fill=255)), col, 4, 0)
        spr.over(f, con, h + 8, (h - f.shape[0]) // 2); return spr
    if k == 'emblem':
        D = P['dia']; spr = Spr(D, D); rw = max(3, D // 22)
        spr.over(smask(D, D, lambda d, ss: (d.ellipse([0, 0, D * ss - 1, D * ss - 1], fill=255), d.ellipse([rw * ss, rw * ss, (D - rw) * ss, (D - rw) * ss], fill=0))), col)
        r2 = rw * 2.2
        spr.over(smask(D, D, lambda d, ss: (d.ellipse([r2 * ss, r2 * ss, (D - r2) * ss, (D - r2) * ss], fill=255), d.ellipse([(r2 + 2) * ss, (r2 + 2) * ss, (D - r2 - 2) * ss, (D - r2 - 2) * ss], fill=0))), col)
        spr.over(smask(D, D, lambda d, ss: d.polygon(star_pts(D * ss / 2, D * ss * .32, D * ss * .13, D * ss * .055), fill=255)), col)
        f, _ = tlayers(name, font(tuple(P['font']), max(12, D // 7)))
        if f.shape[1] > D * .7: f = np.asarray(Image.fromarray(f, 'F').resize((int(D * .7), max(4, int(f.shape[0] * D * .7 / f.shape[1]))), Image.LANCZOS), np.float32)
        spr.over(f, col, (D - f.shape[1]) // 2, int(D * .56)); return spr
    if k == 'icon':
        D = P['dia']; spr = Spr(D, D); shape = P.get('shape', 'circle')
        def fn(d, ss):
            if shape == 'circle': d.ellipse([0, 0, D * ss - 1, D * ss - 1], fill=255)
            elif shape == 'star': d.polygon(star_pts(D * ss / 2, D * ss / 2 + D * ss * .03, D * ss * .5, D * ss * .2), fill=255)
            else: d.polygon([(D * ss * .1, D * ss * .05), (D * ss * .9, D * ss * .05), (D * ss * .9, D * ss * .55), (D * ss * .5, D * ss * .97), (D * ss * .1, D * ss * .55)], fill=255)
        spr.over(smask(D, D, fn), col)
        f, _ = tlayers(name[0].upper(), font(tuple(P['font']), int(D * .55))); spr.over(f, con, (D - f.shape[1]) // 2, (D - f.shape[0]) // 2); return spr
    if k == 'banner':
        h = P['bh']; f, _ = tlayers(name + ('  ' + name) * 1, font(tuple(P['font']), int(h * .5))); spr = Spr(W, h)
        spr.A[:] = 1; spr.P[:] = np.array(col, np.float32); spr.over(f, con, (W - f.shape[1]) // 2, (h - f.shape[0]) // 2); return spr
    if k == 'qr':
        n, ms = 21, P['ms']; g = np.random.default_rng(P['qseed']).random((n, n)) < .5
        for (ox, oy) in ((0, 0), (n - 7, 0), (0, n - 7)):
            g[oy:oy + 7, ox:ox + 7] = True; g[oy + 1:oy + 6, ox + 1:ox + 6] = False; g[oy + 2:oy + 5, ox + 2:ox + 5] = True
        S = (n + 2) * ms; spr = Spr(S, S); spr.A[:] = 1; spr.P[:] = 255
        m = np.kron(g, np.ones((ms, ms), np.float32)); spr.over(m, (0, 0, 0), ms, ms); return spr
    if k in ('tile_text', 'tile_logo'):
        dg = int(math.hypot(W, H)) + 4; canvas = Image.new('L', (dg, dg), 0)
        if k == 'tile_text': unit, _ = tlayers(name, fnt)
        else:
            sp = build_sprite(dict(P, kind='badge'), W, H, rng); unit = sp.A
        uh, uw = unit.shape; sx, sy = int(uw + P['gx']), int(uh + P['gy'])
        ui = Image.fromarray((unit * 255).astype(np.uint8))
        row = 0
        for y in range(0, dg, sy):
            for x in range(-(sx // 2 if row % 2 else 0), dg, sx): canvas.paste(ui, (x, y), ui)
            row += 1
        canvas = canvas.rotate(P['angle'], resample=Image.BICUBIC)
        m = np.asarray(canvas.crop(((dg - W) // 2, (dg - H) // 2, (dg - W) // 2 + W, (dg - H) // 2 + H)), np.float32) / 255
        spr = Spr(W, H); spr.over(m, col); return spr
    raise ValueError(k)


def new_mark(rng, kind, mode, alpha, tier, W):
    val = tier == 'val'
    lang = rng.choice(['lat', 'lat', 'kr'])
    if kind in ('banner',) and False: lang = 'lat'
    fspec = rng.choice(KR_FONTS[val] if lang == 'kr' else LAT_FONTS[val])
    name = (''.join(rng.choice(KRS) for _ in range(rng.randint(2, 4))) + rng.choice(['툰', '만화', '뷰', ''])) if lang == 'kr' else \
        ''.join(rng.choice(SYL) for _ in range(rng.randint(2, 3))).capitalize() + rng.choice(SUF)
    brand = rng.choice(BRANDS); white = (rng.randint(235, 255),) * 3; black = (rng.randint(10, 40),) * 3
    if kind in ('text', 'outline', 'url', 'tile_text'):
        col = rng.choice([white, white, black, brand]); con = black if sum(col) > 400 else white
    else: col = brand; con = white if sum(brand) < 600 else black
    P = dict(kind=kind, name=name, font=list(fspec), color=list(col), contrast=list(con), alpha=alpha, mode=mode)
    if kind == 'text': P.update(size=rng.randint(30, 64), stroke=rng.choice([0, 0, 2, 3]))
    elif kind == 'outline': P.update(size=rng.randint(36, 70))
    elif kind == 'url': P.update(size=rng.randint(20, 28))
    elif kind == 'badge': P.update(size=rng.randint(22, 36))
    elif kind == 'emblem': P.update(dia=rng.randint(110, 220), size=0)
    elif kind == 'icon': P.update(dia=rng.randint(48, 90), shape=rng.choice(['circle', 'star', 'shield']), size=0)
    elif kind == 'banner': P.update(bh=rng.randint(36, 72), size=0)
    elif kind == 'qr': P.update(ms=rng.randint(3, 5), qseed=rng.randint(0, 10 ** 6), size=0)
    elif kind == 'tile_text': P.update(size=rng.randint(24, 40), gx=rng.randint(60, 200), gy=rng.randint(50, 160), angle=rng.choice([-35, -30, -25, -20, 0, 20]))
    elif kind == 'tile_logo': P.update(size=rng.randint(16, 22), gx=rng.randint(120, 260), gy=rng.randint(100, 220), angle=rng.choice([-30, 0, 0, 25]))
    mx, my = rng.randint(8, 60), rng.randint(8, 90)
    P.update(corner=rng.choice(['TL', 'TR', 'BL', 'BR']), mx=mx, my=my, fy=rng.choice([.25, .5, .75]), xa=rng.choice(['L', 'C', 'R']),
             jit=rng.choice([15, 30, 50]), smin=.6, smax=1.5, band=rng.choice(['top', 'bottom']))
    if kind == 'banner' and mode not in ('band',): P['mode'] = 'band'
    return P


def place_xy(rng, P, sw, sh, W, H):
    m = P['mode']; mx, my = P['mx'], P['my']
    def corner(c, ox=0, oy=0):
        x = mx if c[1] == 'L' else W - mx - sw; y = my if c[0] == 'T' else H - my - sh
        return x + ox, y + oy
    if m == 'anchor': x, y = corner(P['corner'])
    elif m == 'ratio':
        x = {'L': mx, 'R': W - mx - sw, 'C': (W - sw) // 2}[P['xa']]; y = int(P['fy'] * H - sh / 2)
    elif m == 'jitter': x, y = corner(P['corner'], rng.randint(-P['jit'], P['jit']), rng.randint(-P['jit'], P['jit']))
    elif m == 'free_y': x, _ = corner(P['corner']); y = rng.randint(10, max(11, H - sh - 10))
    elif m in ('random', 'scaled'): x, y = rng.randint(10, max(11, W - sw - 10)), rng.randint(10, max(11, H - sh - 10))
    elif m == 'band': x = 0; y = 0 if P['band'] == 'top' else H - sh
    elif m == 'tiled': x = y = 0
    else: raise ValueError(m)
    if P.get('edge'):
        x = -int(sw * rng.uniform(.25, .45)) if P['corner'][1] == 'L' else W - int(sw * rng.uniform(.55, .75)); return int(x), int(max(0, min(H - sh, y)))
    return int(max(0, min(W - sw, x))), int(max(0, min(H - sh, y)))


def bake(img, spr, alpha, x, y):
    """가장자리에서 잘려도 됨. 반환: (상자[x,y,w,h](이미지 안쪽만), 대비값=변화량 95백분위)"""
    H, W = img.shape[:2]; h, w = spr.A.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0: return None, 0
    A = spr.A[y0 - y:y1 - y, x0 - x:x1 - x]; Pm = spr.P[y0 - y:y1 - y, x0 - x:x1 - x]; reg = img[y0:y1, x0:x1].copy()
    new = reg * (1 - alpha * A[..., None]) + alpha * Pm; img[y0:y1, x0:x1] = new
    ys, xs = np.where(A > .03)
    if len(xs) == 0: return None, 0
    diff = np.abs(new - reg).max(axis=2)[A > .03]
    return [int(x0 + xs.min()), int(y0 + ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)], float(np.percentile(diff, 95))


# ============================================================== 사이트 사양
STYLES = ['toon', 'toon', 'bw', 'scroll', 'scroll', 'dark', 'flat', 'busy']
ALPHAS = [.12, .2, .3, .45, .6, .8, 1.0]


NONE_CATS = ('none', 'decoy_frame', 'decoy_dup', 'decoy_bars', 'dom_overlay')
AMBIG_CATS = ('decoy_footer',)


def site_spec(sid, cat, tier, seed, kinds=None, mode=None, alpha=None, K=None, W=None, codec=None, **kw):
    rng = random.Random(seed)
    marks = []
    if cat in NONE_CATS or cat in AMBIG_CATS: pass
    elif cat == 'multi':
        for kd, md in kw['pairs']: marks.append(new_mark(rng, kd, md, alpha or rng.choice(ALPHAS), tier, W or 720))
    else:
        marks.append(new_mark(rng, kinds, mode, alpha if alpha is not None else rng.choice(ALPHAS), tier, W or 720))
    for P in marks:
        if kw.get('rot') and P['mode'] != 'tiled': P['rot'] = kw['rot']
        if kw.get('edge') and P['mode'] != 'tiled': P['edge'] = True
        if kw.get('lowc') and P['kind'] in ('text', 'outline', 'url', 'badge'): P['color'] = [250, 250, 250]; P['contrast'] = [245, 245, 245]
    style = kw.get('style') or rng.choice(STYLES)
    return dict(id=sid, cat=cat, tier=tier, seed=seed, marks=marks, K=K or 6, W=W or 720, style=style, codec=codec or rng.choice(['png', 'jpg95', 'jpg85']),
                partial=kw.get('partial', 0.0), ajit=kw.get('ajit', 0.0), frame_decoy=kw.get('frame_decoy', False), dups=kw.get('dups', 0), hue=rng.randint(0, 10 ** 6),
                mixed_w=kw.get('mixed_w', False), tall=kw.get('tall', False), footer=cat == 'decoy_footer', chrome=cat == 'decoy_bars' and rng.random() < .7, dom=cat == 'dom_overlay')


def core_specs():
    sp = []; n = [0]
    def add(cat, **kw):
        n[0] += 1; k = kw.get('kinds') or kw.get('pairs') or cat; kn = '+'.join(x[0] for x in k) if isinstance(k, list) else k
        a = kw.get('alpha'); sp.append(site_spec('%s_%s_%s%s_%03d' % (cat, kn, kw.get('mode') or '', ('_a%d' % int(a * 100)) if a else '', n[0]), cat, 'core', 1000 + n[0], **kw))
    corners = 0
    for kd in ['text', 'badge', 'emblem', 'icon', 'url', 'outline', 'qr']:
        for a in ([.2, .6, 1.0] if kd not in ('qr', 'icon') else [.5, 1.0]): add('anchor', kinds=kd, mode='anchor', alpha=a)
    for a in (.15, .35, .6, 1.0): add('anchor', kinds='text', mode='anchor', alpha=a)
    for kd in ['text', 'badge', 'emblem']:
        for a in (.3, .8): add('ratio', kinds=kd, mode='ratio', alpha=a)
    for kd in ['badge', 'emblem', 'text', 'outline']:
        for a in (.35, .8): add('jitter', kinds=kd, mode='jitter', alpha=a)
    for kd in ['badge', 'emblem', 'text']:
        for a in (.3, .8): add('free_y', kinds=kd, mode='free_y', alpha=a)
    for kd in ['badge', 'emblem', 'text']:
        for a in (.3, .8): add('random', kinds=kd, mode='random', alpha=a)
    for kd in ['badge', 'emblem', 'outline']:
        for a in (.45, .8): add('scaled', kinds=kd, mode='scaled', alpha=a)
    for a in (.1, .18, .3): add('tiled', kinds='tile_text', mode='tiled', alpha=a)
    for a in (.12, .25): add('tiled', kinds='tile_logo', mode='tiled', alpha=a)
    for a in (.3, .6, .9): add('band', kinds='banner', mode='band', alpha=a)
    for pr in ([('badge', 'anchor'), ('emblem', 'ratio')], [('text', 'anchor'), ('text', 'anchor')], [('badge', 'jitter'), ('tile_text', 'tiled')],
               [('emblem', 'anchor'), ('url', 'anchor')], [('badge', 'anchor'), ('outline', 'random')], [('icon', 'anchor'), ('text', 'free_y')]): add('multi', pairs=pr, alpha=.5)
    for kd, md in [('badge', 'anchor'), ('emblem', 'jitter'), ('text', 'anchor'), ('badge', 'free_y')]: add('partial', kinds=kd, mode=md, alpha=.6, partial=.3)
    for kd, md, ww in [('badge', 'anchor', 800), ('emblem', 'anchor', 690), ('text', 'anchor', 1000), ('badge', 'jitter', 800)]: add('anchor', kinds=kd, mode=md, alpha=.4, W=ww, codec='jpg85')
    for kd in ['badge', 'text']: add('anchor', kinds=kd, mode='anchor', alpha=.5, K=4, codec='jpg85')
    for kd in ['badge', 'text']: add('anchor', kinds=kd, mode='anchor', alpha=.5, K=3, codec='jpg85')
    for kd, md in [('badge', 'anchor'), ('emblem', 'jitter'), ('text', 'ratio')]: add('framed', kinds=kd, mode=md, alpha=.5, frame_decoy=True)
    for kd in ['badge', 'emblem']: add('dups', kinds=kd, mode='anchor', alpha=.5, dups=2)
    for kd, a in [('badge', .3), ('emblem', .6), ('text', .3), ('outline', .6)]: add('busy', kinds=kd, mode='anchor', alpha=a, style='busy')
    for kd, md in [('badge', 'anchor'), ('emblem', 'anchor'), ('text', 'jitter')]: add('mixedw', kinds=kd, mode=md, alpha=.5, mixed_w=True, K=6)
    for kd, md in [('badge', 'anchor'), ('emblem', 'ratio')]: add('tall', kinds=kd, mode=md, alpha=.5, tall=True, K=5)
    for kd, md, r in [('badge', 'anchor', 8), ('text', 'random', -12), ('emblem', 'jitter', 15)]: add('rotated', kinds=kd, mode=md, alpha=.5, rot=r)
    for kd in ['badge', 'emblem']: add('edgecut', kinds=kd, mode='anchor', alpha=.5, edge=True)
    for kd, a in [('text', .6), ('outline', .6)]: add('lowcontrast', kinds=kd, mode='anchor', alpha=a, lowc=True, style='flat')
    for _ in range(3): add('dom_overlay')
    for _ in range(3): add('decoy_footer')
    for _ in range(8): add('none')
    for _ in range(6): add('decoy_frame', frame_decoy=True)
    for _ in range(3): add('decoy_dup', dups=2)
    for _ in range(3): add('decoy_bars')
    return sp


def rand_specs(tier, n, base):
    rng = random.Random(base); sp = []
    cats = [('anchor', .22), ('ratio', .08), ('jitter', .12), ('free_y', .08), ('random', .08), ('scaled', .08), ('tiled', .10), ('band', .04), ('multi', .08), ('partial', .04), ('none', .05), ('decoy_frame', .03), ('busy', .04), ('mixedw', .03), ('tall', .02), ('rotated', .03), ('edgecut', .02), ('lowcontrast', .02), ('dom_overlay', .02), ('decoy_footer', .02), ('decoy_bars', .02)]
    kinds = ['text', 'badge', 'emblem', 'icon', 'url', 'outline', 'qr']
    for i in range(n):
        cat = rng.choices([c for c, _ in cats], [w for _, w in cats])[0]; seed = base * 10 + i; sid = '%s_%s_%03d' % (tier, cat, i)
        kw = dict(K=rng.randint(4, 8), W=rng.choice([720, 720, 720, 800, 690]), codec=rng.choice(['png', 'jpg95', 'jpg85', 'jpg75']), dups=rng.choice([0, 0, 0, 2]),
                  frame_decoy=cat == 'decoy_frame' or rng.random() < .15, ajit=rng.choice([0, 0, .08]))
        a = rng.choice(ALPHAS)
        if cat in ('anchor', 'ratio', 'jitter', 'free_y', 'random', 'scaled'):
            kd = rng.choice(kinds if cat == 'anchor' else ['text', 'badge', 'emblem', 'outline', 'icon'])
            sp.append(site_spec(sid, cat, tier, seed, kinds=kd, mode=cat, alpha=a, **kw))
        elif cat == 'tiled': sp.append(site_spec(sid, cat, tier, seed, kinds=rng.choice(['tile_text', 'tile_text', 'tile_logo']), mode='tiled', alpha=rng.choice([.1, .15, .2, .3]), **kw))
        elif cat in ('busy', 'mixedw', 'tall', 'rotated', 'edgecut', 'lowcontrast'):
            ex = dict(busy=dict(style='busy'), mixedw=dict(mixed_w=True), tall=dict(tall=True, K=5), rotated=dict(rot=rng.choice([-15, -8, 8, 15])), edgecut=dict(edge=True), lowcontrast=dict(lowc=True, style='flat'))[cat]
            kw2 = dict(kw); kw2.update(ex)
            sp.append(site_spec(sid, cat, tier, seed, kinds=rng.choice(['badge', 'emblem', 'text', 'outline']) if cat != 'lowcontrast' else rng.choice(['text', 'outline']), mode=rng.choice(['anchor', 'anchor', 'jitter']), alpha=a, **kw2))
        elif cat == 'band': sp.append(site_spec(sid, cat, tier, seed, kinds='banner', mode='band', alpha=a, **kw))
        elif cat == 'multi':
            pr = [(rng.choice(['badge', 'emblem', 'text', 'icon']), rng.choice(['anchor', 'ratio', 'jitter'])), (rng.choice(['text', 'url', 'outline', 'tile_text']), rng.choice(['anchor', 'random', 'free_y']))]
            pr = [(k, 'tiled' if k.startswith('tile') else m) for k, m in pr]; sp.append(site_spec(sid, cat, tier, seed, pairs=pr, alpha=a, **kw))
        elif cat == 'partial': sp.append(site_spec(sid, cat, tier, seed, kinds=rng.choice(['badge', 'emblem', 'text']), mode=rng.choice(['anchor', 'jitter', 'free_y']), alpha=a, partial=rng.choice([.2, .3, .4]), **kw))
        else: sp.append(site_spec(sid, cat, tier, seed, **kw))
    return sp


# ============================================================== 사이트 생성
def build_site(args):
    sp, out, want_clean = args
    d = os.path.join(out, sp['id']); os.makedirs(d, exist_ok=True)
    rng = random.Random(sp['seed'] * 7 + 1); K = sp['K']; images = []
    sprs = {}
    fr = None
    if sp['frame_decoy'] or sp['cat'] == 'decoy_frame': fr = (rng.randint(8, 24), rng.randint(4, 8), rng.choice([(10, 10, 10), (60, 60, 60)]), rng.random() < .5)
    bars = rng.randint(30, 60) if sp['cat'] == 'decoy_bars' else 0
    dom = []
    if sp['dom']: dom = [dict(text=rng.choice(['Toonix', '무단전재금지', 'manga.example']), x=rng.randint(5, 60), y=rng.randint(5, 80), size=rng.randint(22, 44), opacity=rng.choice([.2, .4, .7]))]
    prev = []
    for i in range(K):
        if sp['dups'] and i >= K - sp['dups'] and prev:
            images.append(dict(file=None, dupOf=prev[rng.randrange(len(prev))])); continue
        W = sp['W'] + (80 if sp['mixed_w'] and i % 2 else 0)
        H = rng.randint(2500, 4500) if sp['tall'] else rng.randint(900, 1900)
        img = make_comic(sp['seed'] * 100 + i, W, H, sp['style'], sp['hue'])
        if sp['footer']: img[H - 260:] = make_comic(sp['seed'] * 100 + 99, W, 260, 'toon', sp['hue'])
        if fr:
            m, w, c, top = fr; dr = Image.fromarray(img.astype(np.uint8)); dd = ImageDraw.Draw(dr); dd.rectangle([m, m, W - m - 1, H - m - 1], outline=c, width=w)
            if top: dd.line([(m, m + 70), (W - m, m + 70)], fill=c, width=3)
            img = np.asarray(dr, np.float32).copy()
        if bars:
            img[:bars] = 25; img[H - bars:] = 25
            if sp['chrome']:
                dr = Image.fromarray(img.astype(np.uint8)); dd = ImageDraw.Draw(dr); f = font(LAT_FONTS[False][0], max(12, bars // 2))
                dd.text((14, bars // 4), '12:30', font=f, fill=(230, 230, 230)); dd.text((W - 120, bars // 4), 'LTE 87%', font=f, fill=(230, 230, 230))
                for ix in (W // 4, W // 2, 3 * W // 4): dd.ellipse([ix - 9, H - bars // 2 - 9, ix + 9, H - bars // 2 + 9], outline=(230, 230, 230), width=2)
                img = np.asarray(dr, np.float32).copy()
        Image.fromarray(np.clip(np.rint(img), 0, 255).astype(np.uint8)).save(os.path.join(d, 'clean_%02d.jpg' % (i + 1)), 'JPEG', quality=92)
        if want_clean: Image.fromarray(np.clip(np.rint(img), 0, 255).astype(np.uint8)).save(os.path.join(d, 'clean_%02d.png' % (i + 1)))
        marks = []
        if rng.random() >= sp['partial']:
            for mi, P in enumerate(sp['marks']):
                k = P['kind']; tile = P['mode'] == 'tiled'
                key = (mi, W, H if (tile or k == 'banner') else 0)
                base = sprs.get(key)
                if base is None:
                    base = build_sprite(P, W, H, random.Random(sp['seed'] + mi))
                    if P.get('rot'): base = spr_rotated(base, P['rot'])
                    sprs[key] = base
                spr = base.scaled(rng.uniform(P['smin'], P['smax'])) if P['mode'] == 'scaled' else base
                a = float(np.clip(P['alpha'] + (rng.uniform(-sp['ajit'], sp['ajit']) if sp['ajit'] else 0), .05, 1))
                x, y = place_xy(rng, P, spr.w, spr.h, W, H); box, con = bake(img, spr, a, x, y)
                if box: marks.append(dict(kind=k, mode=P['mode'], box=box, alpha=round(a, 3), tiled=tile, contrast=round(con, 1), visible=con >= 3.0))
        ext = 'png' if sp['codec'] == 'png' else 'jpg'; fn = 'img_%02d.%s' % (i + 1, ext)
        pim = Image.fromarray(np.clip(np.rint(img), 0, 255).astype(np.uint8))
        if ext == 'png': pim.save(os.path.join(d, fn))
        else: pim.save(os.path.join(d, fn), 'JPEG', quality=int(sp['codec'][3:]))
        pim.resize((200, int(H * 200 / W)), Image.LANCZOS).save(os.path.join(d, 'th_%02d.jpg' % (i + 1)), 'JPEG', quality=70)
        images.append(dict(file=fn, w=W, h=H, marks=marks)); prev.append(len(images) - 1)
    # 중복 이미지는 원본 파일을 복사
    for i, im in enumerate(images):
        if im.get('dupOf') is not None:
            src = images[im['dupOf']]; ext = os.path.splitext(src['file'])[1]; fn = 'img_%02d%s' % (i + 1, ext)
            for a_, b_ in ((src['file'], fn), ('th_%02d.jpg' % (im['dupOf'] + 1), 'th_%02d.jpg' % (i + 1)), ('clean_%02d.jpg' % (im['dupOf'] + 1), 'clean_%02d.jpg' % (i + 1))):
                shutil.copy(os.path.join(d, a_), os.path.join(d, b_))
            im.update(file=fn, w=src['w'], h=src['h'], marks=src['marks'])
    # 굽힌 워터마크 그림(미리보기)
    for mi, P in enumerate(sp['marks']):
        spr = build_sprite(P, sp['W'], 1400, random.Random(sp['seed'] + mi)); a = P['alpha']
        if P.get('rot'): spr = spr_rotated(spr, P['rot'])
        rgb = np.where(spr.A[..., None] > 0, spr.P / np.maximum(spr.A[..., None], 1e-6), 128); al = np.clip(spr.A * a, 0, 1)
        im = Image.fromarray(np.dstack([np.clip(rgb, 0, 255), al * 255]).astype(np.uint8), 'RGBA')
        if im.width > 400: im = im.resize((400, max(1, int(im.height * 400 / im.width))), Image.LANCZOS)
        bgc = Image.new('RGBA', im.size, (130, 130, 130, 255)); bgc.alpha_composite(im); bgc.convert('RGB').save(os.path.join(d, 'mark_%02d.png' % (mi + 1)))
    with open(os.path.join(d, 'index.html'), 'w') as f:   # 테스트 웹툰: 이미지만 세로로 나열(앱 시험용)
        f.write('<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width"><title>%s</title><body style="margin:0;background:#fff;position:relative">' % sp['id'])
        for im in images: f.write('<img src="%s" style="display:block;width:100%%;margin:0 0 8px">' % im['file'])
        for o in dom: f.write('<div style="position:fixed;left:%d%%;top:%d%%;font:bold %dpx sans-serif;color:#fff;text-shadow:0 0 3px #000;opacity:%s;pointer-events:none">%s</div>' % (o['x'], o['y'], o['size'], o['opacity'], o['text']))
    expected = 'none' if sp['cat'] in NONE_CATS else ('ambiguous' if sp['cat'] in AMBIG_CATS else 'mark')
    if sp['cat'] == 'decoy_bars' and sp['chrome']: expected = 'ambiguous'   # 글자가 든 고정 띠는 불투명 배너 워터마크와 원리상 구분 불가
    site = dict(id=sp['id'], cat=sp['cat'], tier=sp['tier'], expected=expected, W=sp['W'], K=K, codec=sp['codec'], style=sp['style'], partial=sp['partial'], frame_decoy=bool(fr), bars=bars,
                chrome=bool(sp['chrome']), dups=sp['dups'], mixed_w=sp['mixed_w'], tall=sp['tall'], footer=sp['footer'], dom=dom, marks=[{k: v for k, v in P.items() if k not in ('smin', 'smax')} for P in sp['marks']], images=images)
    site_answer_page(site, d)
    return site


NOTES = {'none': '워터마크 없음 — 인식하면 오인식', 'decoy_frame': '미끼: 모든 장 같은 자리의 컷 테두리·선 — 인식하면 오인식', 'decoy_dup': '미끼: 같은 이미지가 중복됨 — 인식하면 오인식',
         'decoy_bars': '미끼: 위아래 고정 띠(화면 요소 모양) — 인식하면 오인식', 'dom_overlay': '이미지엔 워터마크 없음(웹페이지 위 CSS 글자만 있음) — 이미지 인식 대상 아님',
         'decoy_footer': '애매함: 모든 장 맨 아래에 같은 그림(푸터)이 들어 있음 — 오인식으로 세지 않고 따로 집계'}


def site_answer_page(s, d):
    """정답지 웹툰: 장을 세로로 나열하고 정답 상자를 그린다. 위 체크로 정답 상자/원본(워터마크 전) 전환."""
    css = ('body{margin:0;background:#eee;font:14px sans-serif}header{position:sticky;top:0;z-index:9;background:#fff;padding:8px 10px;border-bottom:1px solid #bbb}'
           '.im{position:relative;max-width:720px;margin:0 auto 10px;background:#fff}.im img{display:block;width:100%}.b{position:absolute;border:2px solid #f0f;box-sizing:border-box}'
           '.b i{position:absolute;left:0;top:-16px;background:#f0f;color:#fff;font:11px sans-serif;padding:0 3px;white-space:nowrap;font-style:normal}.hide .b{display:none}'
           '.lab{max-width:720px;margin:0 auto;color:#555;font-size:12px;padding:2px 4px}.dom{position:absolute;left:0;top:0;font-weight:bold;color:#fff;text-shadow:0 0 3px #000;outline:2px dashed #0af}')
    ms = ' / '.join('%s·%s·α%.2f' % (m['kind'], m['mode'], m['alpha']) for m in s['marks']) or '없음'
    h = ['<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width"><title>정답지 %s</title><style>%s</style>' % (s['id'], css),
         '<header><b>%s</b> · %s<br>워터마크: %s · 폭 %d · %d장 · %s<br><label><input type=checkbox checked onchange="document.body.classList.toggle(\'hide\',!this.checked)"> 정답 상자</label> '
         '<label><input type=checkbox onchange="for(const e of document.querySelectorAll(\'img[data-c]\')){e.src=this.checked?e.dataset.c:e.dataset.w}"> 원본(워터마크 전)</label></header>' % (s['id'], NOTES.get(s['cat'], s['cat']), ms, s['W'], s['K'], s['codec'])]
    for i, im in enumerate(s['images']):
        h.append('<div class=lab>%02d · %dx%d · 정답 %d개%s</div><div class=im><img src="%s" data-w="%s" data-c="clean_%02d.jpg">' % (i + 1, im['w'], im['h'], len(im['marks']), ' (일부러 비움)' if s['partial'] and not im['marks'] else '', im['file'], im['file'], i + 1))
        for m in im['marks']:
            x, y, w, hh = m['box']
            h.append('<div class=b style="left:%.2f%%;top:%.2f%%;width:%.2f%%;height:%.2f%%"><i>%s α%.2f 대비%.0f%s</i></div>' % (100 * x / im['w'], 100 * y / im['h'], 100 * w / im['w'], 100 * hh / im['h'], m['kind'], m['alpha'], m['contrast'], '' if m['visible'] else ' 안보임'))
        if i == 0:
            for o in s['dom']: h.append('<div class=dom style="left:%d%%;top:%d%%;font-size:%dpx">%s (CSS 글자)</div>' % (o['x'], o['y'], o['size'], o['text']))
        h.append('</div>')
    open(os.path.join(d, 'answer.html'), 'w').write(''.join(h))


def answer_page(tier, sites, out):
    css = ('body{font:14px sans-serif;margin:0;background:#f4f4f6;color:#222}header{position:sticky;top:0;background:#fff;padding:8px 12px;border-bottom:1px solid #ccc;z-index:9}'
           'h2{margin:18px 12px 6px}.site{background:#fff;margin:8px 12px;border:1px solid #ddd;border-radius:6px;padding:8px}.meta{color:#555;font-size:12px;margin-bottom:6px}'
           '.row{display:flex;gap:6px;overflow-x:auto;align-items:flex-start}.im{position:relative;flex:none;width:200px}.im img{display:block;width:200px}'
           '.b{position:absolute;border:2px solid #f0f;box-sizing:border-box;pointer-events:none}.hide .b{display:none}.neg{color:#06c}.mk{height:80px;flex:none}a{color:#06c}')
    h = ['<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width"><title>wm2 정답지 %s</title><style>%s</style>' % (tier, css),
         '<header><b>wm2 정답지 · %s</b> · 사이트 %d개 · <label><input type=checkbox checked onchange="document.body.classList.toggle(\'hide\',!this.checked)"> 정답 상자(분홍)</label></header>' % (tier, len(sites))]
    cur = None
    for s in sorted(sites, key=lambda s: (s['cat'], s['id'])):
        if s['cat'] != cur: cur = s['cat']; h.append('<h2>%s (%d)</h2>' % (cur, sum(1 for x in sites if x['cat'] == cur)))
        ms = ' / '.join('%s·%s·α%.2f' % (m['kind'], m['mode'], m['alpha']) for m in s['marks']) or '<span class=neg>%s</span>' % NOTES.get(s['cat'], '')
        ex = [t for t, c in (('프레임 미끼', s['frame_decoy']), ('위아래 띠 미끼', s['bars']), ('중복 이미지 %d' % s['dups'], s['dups']), ('폭 섞임', s['mixed_w']), ('아주 긴 이미지', s['tall']),
                              ('일부 이미지엔 없음 %d%%' % int(s['partial'] * 100), s['partial'])) if c]
        h.append('<div class=site><div><b>%s</b> · <a href="%s/answer.html">정답지 웹툰</a> · <a href="%s/index.html">테스트 웹툰</a></div><div class=meta>%s · 폭 %d · %d장 · %s · %s %s</div><div class=row>' % (s['id'], s['id'], s['id'], ms, s['W'], s['K'], s['codec'], s['style'], ' · '.join(ex)))
        for i in range(len(s['marks'])): h.append('<img class=mk src="%s/mark_%02d.png">' % (s['id'], i + 1))
        for i, im in enumerate(s['images']):
            h.append('<div class=im><img loading=lazy src="%s/th_%02d.jpg">' % (s['id'], i + 1))
            for m in im['marks']:
                x, y, w, hh = m['box']; h.append('<div class=b style="left:%.2f%%;top:%.2f%%;width:%.2f%%;height:%.2f%%"></div>' % (100 * x / im['w'], 100 * y / im['h'], 100 * w / im['w'], 100 * hh / im['h']))
            h.append('</div>')
        h.append('</div></div>')
    open(os.path.join(out, 'answer.html'), 'w').write(''.join(h))


def root_index(root):
    h = ['<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width"><title>wm2 벤치마크</title><body style="font:15px sans-serif;margin:12px">', '<h2>wm2 벤치마크</h2>']
    for tier in ('core', 'wild', 'val'):
        tp = os.path.join(root, tier, 'truth.json')
        if not os.path.exists(tp): continue
        t = json.load(open(tp)); cats = {}
        for s in t['sites']: cats[s['cat']] = cats.get(s['cat'], 0) + 1
        h.append('<h3>%s · 사이트 %d개 · 이미지 %d장</h3><p><a href="%s/answer.html">정답지 전체 보기(썸네일+정답 상자)</a></p><p style="color:#555">%s</p><ul>' % (
            tier, len(t['sites']), sum(len(s['images']) for s in t['sites']), tier, ' · '.join('%s %d' % kv for kv in sorted(cats.items()))))
        for s in t['sites']:
            h.append('<li>%s — <a href="%s/%s/index.html">테스트</a> · <a href="%s/%s/answer.html">정답지</a></li>' % (s['id'], tier, s['id'], tier, s['id']))
        h.append('</ul>')
    open(os.path.join(root, 'index.html'), 'w').write(''.join(h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', default='core'); ap.add_argument('--out', default=os.path.expanduser('~/wm2-bench'))
    ap.add_argument('--only', default=''); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--clean', action='store_true'); ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--n', type=int, default=0)
    a = ap.parse_args()
    specs = core_specs() if a.tier == 'core' else rand_specs(a.tier, a.n or (120 if a.tier == 'wild' else 60), 20000 if a.tier == 'wild' else 50000)
    if a.only: specs = [s for s in specs if a.only in s['id']]
    if a.limit: specs = specs[:a.limit]
    out = os.path.join(a.out, a.tier); os.makedirs(out, exist_ok=True)
    with Pool(a.jobs) as p: sites = p.map(build_site, [(s, out, a.clean) for s in specs], chunksize=1)
    tp = os.path.join(out, 'truth.json')
    if (a.only or a.limit) and os.path.exists(tp):   # 일부만 다시 만들 때는 기존 정답과 병합
        old = {x['id']: x for x in json.load(open(tp))['sites']}
        for x in sites: old[x['id']] = x
        sites = list(old.values())
    json.dump(dict(tier=a.tier, sites=sites), open(tp, 'w'), ensure_ascii=False, indent=1)
    answer_page(a.tier, sites, out)
    root_index(a.out)
    print('사이트 %d개, 이미지 %d장 → %s' % (len(sites), sum(len(s['images']) for s in sites), out))


if __name__ == '__main__': main()
