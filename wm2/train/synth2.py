"""synth2: synth.py 에 '하드 네거티브'(말풍선·캡션·효과음 글자·본문 글자·직선, 정답 아님)와 빈 조각 비율 상승을 더한 합성기.
정답 마스크 규칙은 synth.py 와 같다(합성 전후 변화량 95백분위>=5 인 도장만). 비워터마크 요소는 워터마크보다 먼저 그려져 마스크에 들어가지 않는다."""
import os, random, io
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
from synth import Synth

_W = 'C:/Windows/Fonts/'
_CAND = ['malgun.ttf', 'malgunbd.ttf', 'batang.ttc', 'gulim.ttc', 'HANDotum.ttf', 'HANBatang.ttf', 'NGULIM.TTF', 'arial.ttf', 'arialbd.ttf', 'times.ttf', 'timesbd.ttf',
         'comic.ttf', 'comicbd.ttf', 'impact.ttf', 'segoeui.ttf', 'verdana.ttf', 'georgia.ttf', 'tahoma.ttf', 'trebuc.ttf', 'cour.ttf', 'consola.ttf', 'segoepr.ttf', 'bahnschrift.ttf',
         'MalangmalangB.ttf', 'Hancom Gothic Bold.ttf', 'HMKMAMI.TTF', 'ROCKB.TTF', 'FRAHV.TTF', 'GOTHICB.TTF', 'calibrib.ttf']
KR = list('가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허고노도로모보소오조초코토포호구누두루무부수우주추쿠투푸후그느드르므브스으즈츠크트프흐기니디리미비시이지치키티피히')
LAT = ['what', 'no way', 'hey!', 'sure', 'the end', 'wait', 'come on', 'ok', 'I see', 'really?', 'next time', 'oh no', 'later', 'go', 'stop', 'hello', 'yes', 'hmm', 'run!', 'ha ha']
_fcache = {}


def _fonts():
    if 'list' not in _fcache: _fcache['list'] = [p for p in (_W + c for c in _CAND) if os.path.exists(p)] or ['']
    return _fcache['list']


def _font(path, size):
    k = (path, size)
    if k not in _fcache:
        try: _fcache[k] = ImageFont.truetype(path, size) if path else ImageFont.load_default()
        except OSError: _fcache[k] = ImageFont.load_default()
    return _fcache[k]


def _text(r, kr, n):
    return ''.join(KR[int(r.integers(len(KR)))] for _ in range(n)) if kr else LAT[int(r.integers(len(LAT)))]


def distract(img, r, S):
    """img: SxSx3 float32 → 비워터마크 요소(말풍선·캡션·글자·선)를 그린 같은 크기 배열"""
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)); d = ImageDraw.Draw(im); fonts = _fonts()
    for _ in range(int(r.integers(1, 4))):
        k = r.choice(['bubble', 'bubble', 'caption', 'sfx', 'plain', 'plain', 'lines'])
        fp = fonts[int(r.integers(len(fonts)))]; kr = r.random() < .6
        if k == 'lines':
            for _ in range(int(r.integers(1, 4))):
                c = int(r.choice([0, 40, 90, 200, 255])); w = int(r.integers(1, 4)); x0, y0 = int(r.integers(0, S)), int(r.integers(0, S))
                if r.random() < .6: x1, y1 = (int(r.integers(0, S)), y0) if r.random() < .5 else (x0, int(r.integers(0, S)))
                else: x1, y1 = int(r.integers(0, S)), int(r.integers(0, S))
                d.line([x0, y0, x1, y1], fill=(c, c, c), width=w)
            continue
        fs = int(r.integers(30, 64)) if k == 'sfx' else int(r.integers(11, 27)); f = _font(fp, fs)
        lines = [_text(r, kr, int(r.integers(2, 6))) for _ in range(int(r.integers(1, 4)))]; txt = '\n'.join(lines)
        try: bb = d.multiline_textbbox((0, 0), txt, font=f, stroke_width=2 if k == 'sfx' else 0)
        except Exception: continue
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        if tw >= S - 8 or th >= S - 8: continue
        x, y = int(r.integers(2, S - tw - 2)), int(r.integers(2, S - th - 2))
        if k == 'bubble':
            padx, pady = int(tw * .35) + 6, int(th * .35) + 6; bx = [x - padx, y - pady, x + tw + padx, y + th + pady]
            fillc = tuple(int(v) for v in r.choice([255, 250, 235, 30], size=1).repeat(3)); txtc = (0, 0, 0) if fillc[0] > 100 else (255, 255, 255)
            (d.ellipse if r.random() < .6 else d.rounded_rectangle)(bx, fill=fillc, outline=(0, 0, 0) if r.random() < .8 else None, width=int(r.integers(1, 3)))
            d.multiline_text((x, y), txt, font=f, fill=txtc, align='center')
        elif k == 'caption':
            bx = [x - 4, y - 3, x + tw + 4, y + th + 3]; fillc = tuple(int(v) for v in r.choice([255, 255, 240, 20], size=1).repeat(3)); txtc = (0, 0, 0) if fillc[0] > 100 else (255, 255, 255)
            d.rectangle(bx, fill=fillc, outline=(0, 0, 0) if r.random() < .5 else None); d.multiline_text((x, y), txt, font=f, fill=txtc)
        elif k == 'sfx':
            col = tuple(int(v) for v in r.integers(0, 256, 3)); d.multiline_text((x, y), txt, font=f, fill=col, stroke_width=2, stroke_fill=(0, 0, 0) if r.random() < .6 else (255, 255, 255))
        else:
            c = int(r.choice([0, 20, 255])); d.multiline_text((x, y), txt, font=f, fill=(c, c, c))
    return np.asarray(im, np.float32).copy()


class Synth2(Synth):
    """synth.py 의 __getitem__ 을 그대로 두고 (1) 빈 조각 비율 35%, (2) 60% 확률로 비워터마크 요소 추가만 바꾼다."""
    P_DIST = .6
    P_EMPTY = .35

    def __getitem__(self, i):
        # synth.py 의 흐름을 재사용하기 위해 난수·배경 조각 선택까지 같게 하고, 워터마크 개수 분포와 하드 네거티브만 바꿔서 합성한다.
        r = np.random.default_rng(self.seed * 1000003 + i); S = self.S
        N, BH, BW, _ = self.bgs.shape
        bg = self.bgs[r.integers(N)]; x0 = int(r.integers(0, BW - S + 1)); y0 = int(r.integers(0, BH - S + 1))
        img = bg[y0:y0 + S, x0:x0 + S].astype(np.float32).copy(); mask = np.zeros((S, S), np.float32)
        if r.random() < .15:
            c = float(r.choice([10, 60])); m = int(r.integers(4, 30)); w = int(r.integers(1, 4)); img[m:m + w, m:S - m] = c; img[S - m - w:S - m, m:S - m] = c; img[m:S - m, m:m + w] = c
        if r.random() < self.P_DIST: img = distract(img, r, S)
        nm = 0 if r.random() < self.P_EMPTY else int(r.choice([1, 1, 1, 2]))
        for _ in range(nm):
            img, mask = self._apply_one(img, mask, r, S, x0, y0)
        img = np.clip(img, 0, 255)
        if r.random() < .6:
            b = io.BytesIO(); Image.fromarray(img.astype(np.uint8)).save(b, 'JPEG', quality=int(r.integers(65, 96))); img = np.asarray(Image.open(b).convert('RGB'), np.float32)
        return torch.from_numpy(img.transpose(2, 0, 1) / 255.), torch.from_numpy(mask[None])

    def _apply_one(self, img, mask, r, S, x0, y0):
        if r.random() < .25 and self.tile:
            kind, A, P = self.tile[int(r.integers(len(self.tile)))]
            A = A[y0:y0 + S, x0:x0 + S] if A.shape[0] >= y0 + S and A.shape[1] >= x0 + S else None
            if A is None: return img, mask
            P = self._crop(P, y0, x0, S)
            a = float(np.exp(r.uniform(np.log(.1), np.log(.5)))) if kind.startswith('tile') else float(np.exp(r.uniform(np.log(.2), np.log(1))))
            A = A.astype(np.float32) / 255; old = img.copy(); img = img * (1 - a * A[..., None]) + a * P.astype(np.float32) * (A[..., None] > 0); sel = A > .03
            if sel.any() and np.percentile(np.abs(img - old).max(axis=2)[sel], 95) >= 5: mask = np.maximum(mask, sel.astype(np.float32))
            return img, mask
        kind, A, P = self.small[int(r.integers(len(self.small)))]
        A = A.astype(np.float32) / 255; Pf = P.astype(np.float32)
        f = float(r.uniform(.6, 1.5)); h, w = A.shape
        nh, nw = max(8, int(h * f)), max(8, int(w * f))
        if nw > S - 4 or nh > S - 4: f = min((S - 4) / w, (S - 4) / h); nh, nw = max(8, int(h * f)), max(8, int(w * f))
        Aim = Image.fromarray(A, 'F').resize((nw, nh), Image.BILINEAR)
        if r.random() < .3: Aim = Aim.rotate(float(r.uniform(-15, 15)), expand=True, resample=Image.BILINEAR)
        sh, sw = Aim.size[1], Aim.size[0]
        Pim = np.stack([np.asarray(Image.fromarray(Pf[..., c], 'F').resize((nw, nh), Image.BILINEAR)) for c in range(3)], -1)
        Aa = np.asarray(Aim, np.float32).clip(0, 1)
        if Pim.shape[:2] != Aa.shape: Pim = np.stack([np.asarray(Image.fromarray(Pim[..., c].astype(np.float32), 'F').resize((sw, sh), Image.BILINEAR)) for c in range(3)], -1)
        if sw >= S or sh >= S: return img, mask
        px = int(r.integers(-sw // 6, S - sw + sw // 6 + 1)); py = int(r.integers(-sh // 6, S - sh + sh // 6 + 1))
        a = float(np.exp(r.uniform(np.log(.12), np.log(1.0))))
        xa, ya, xb, yb = max(0, px), max(0, py), min(S, px + sw), min(S, py + sh)
        if xb <= xa or yb <= ya: return img, mask
        Ac = Aa[ya - py:yb - py, xa - px:xb - px]; Pc = Pim[ya - py:yb - py, xa - px:xb - px]
        old = img[ya:yb, xa:xb].copy(); img[ya:yb, xa:xb] = old * (1 - a * Ac[..., None]) + a * Pc
        sel = Ac > .03
        if sel.any() and np.percentile(np.abs(img[ya:yb, xa:xb] - old).max(axis=2)[sel], 95) >= 5: mask[ya:yb, xa:xb] = np.maximum(mask[ya:yb, xa:xb], sel.astype(np.float32))
        return img, mask
