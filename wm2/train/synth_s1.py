"""1단계(단색 여백에서 예시 워터마크 찾기) 학습 합성기. 항목: (이미지 3xSxS 0..1, 마스크 1xSxS)
조각 안에 흰/검은 단색 여백 띠(꽉 찬 폭)를 두고 그 안에 예시 마크(불투명, 단색 배경 위에 그린 모양)를 0~3개 놓는다 → 마스크 1.
정답이 아닌 것: 여백의 캡션·쪽번호·선, 그림 영역의 말풍선·글자·집중선·머리카락·눈·불꽃, **그림 위에 박힌 마크**(2단계 대상이라 1단계에서는 찾지 않음).
0.5배 세계(폭 360)에서 학습: 로고 폭 29~108px, 글자 36~180px(synth_ref2.BOUNDS)."""
import io
import numpy as np, torch
from PIL import Image, ImageDraw
from synth2 import distract
from synth_ref2 import SynthRef2, put2, BOUNDS, LOGO
from synth_ref import lum
from manga_fx import apply_fx
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))
import gen_bench as G

CAPS = ['To be continued', 'NEXT EPISODE', '다음 화에 계속', 'END', 'page 12', '12', '— 3 —', 'Copyright', '제 12화', 'The end', '다음 이야기', 'Episode 7', 'www.site.com', 'TOON']


def caption(img, y0, y1, bgv, r):
    S = img.shape[1]; im = Image.fromarray(img.astype(np.uint8)); d = ImageDraw.Draw(im); fg = 255 - int(bgv)
    for _ in range(int(r.integers(1, 3))):
        fnt = G.font(G.LAT_FONTS[False][int(r.integers(len(G.LAT_FONTS[False])))] if r.random() < .6 else G.KR_FONTS[False][0], int(r.integers(9, 22))); txt = CAPS[int(r.integers(len(CAPS)))]
        bb = d.textbbox((0, 0), txt, font=fnt); w, h = bb[2] - bb[0], bb[3] - bb[1]
        if w >= S - 10 or y1 - y0 - h < 4: continue
        x = int(r.integers(4, S - w - 4)); y = int(r.integers(y0 + 2, y1 - h - 2)); d.text((x - bb[0], y - bb[1]), txt, font=fnt, fill=(fg,) * 3)
    if r.random() < .4 and y1 - y0 > 12: yy = int(r.integers(y0 + 4, y1 - 4)); d.line([(int(r.integers(0, 100)), yy), (int(r.integers(180, S)), yy)], fill=(fg,) * 3, width=int(r.integers(1, 3)))
    return np.asarray(im, np.float32)


class SynthS1(SynthRef2):
    P_BAND = .85
    P_MARKS = .7

    def __getitem__(self, i):
        r = np.random.default_rng(self.seed * 1000003 + i); S = self.S
        N, BH, BW, _ = self.bgs.shape
        bg = self.bgs[r.integers(N)]; x0 = int(r.integers(0, BW - S + 1)); y0 = int(r.integers(0, BH - S + 1))
        img = bg[y0:y0 + S, x0:x0 + S].astype(np.float32).copy(); mask = np.zeros((S, S), np.float32)
        if r.random() < self.P_DIST: img = distract(img, r, S)
        img, _ = apply_fx(img, r, p=self.P_MANGA)
        if r.random() < .5:   # 그림 위에 박힌 마크(불투명/반투명) — 1단계 정답 아님
            for _ in range(int(r.integers(1, 3))):
                al = 1.0 if r.random() < .6 else float(r.uniform(.15, .7)); img, _ = put2(img, self.small[int(r.integers(len(self.small)))], r, S, al)
        if r.random() < self.P_BAND:
            hb = int(r.integers(55, 190)) if r.random() < .85 else S; ya = int(r.integers(0, S - hb + 1)) if hb < S else 0; yb = ya + hb; bgv = 0. if r.random() < .5 else 255.
            img[ya:yb] = bgv
            if r.random() < self.P_MARKS:
                placed = []
                for _ in range(int(r.integers(1, 4))):
                    for _try in range(25):
                        spr = self.small[int(r.integers(len(self.small)))]; kind, A, P = spr
                        if abs(lum(A, P) - bgv) < 70: continue   # 단색 배경 위에서 보이는 마크만(흰 배경엔 어두운/색 마크, 검은 배경엔 밝은 마크)
                        A = A.astype(np.float32) / 255; Pf = P.astype(np.float32); lo, hi = BOUNDS[kind in LOGO]
                        f = float(np.clip(r.uniform(.75, 1.33), lo / A.shape[1], hi / A.shape[1])); nh, nw = max(8, int(A.shape[0] * f)), max(8, int(A.shape[1] * f))
                        if nh > hb - 8 or nw > S - 8: continue
                        px = int(r.integers(4, S - nw - 3)); py = int(r.integers(ya + 4, yb - nh - 3)) if yb - nh - 3 > ya + 4 else ya + 4
                        if any(not (px + nw < q[0] or q[0] + q[2] < px or py + nh < q[1] or q[1] + q[3] < py) for q in placed): continue
                        Ar = np.asarray(Image.fromarray(A, 'F').resize((nw, nh), Image.BILINEAR), np.float32).clip(0, 1); Pr = np.stack([np.asarray(Image.fromarray(Pf[..., c], 'F').resize((nw, nh), Image.BILINEAR)) for c in range(3)], -1)
                        img[py:py + nh, px:px + nw] = bgv * (1 - Ar[..., None]) + Pr; mask[py:py + nh, px:px + nw] = np.maximum(mask[py:py + nh, px:px + nw], (Ar > .03).astype(np.float32)); placed.append((px, py, nw, nh)); break
            elif r.random() < .8: img = caption(img, ya, yb, bgv, r)
        img = np.clip(img, 0, 255)
        if r.random() < .6:
            b = io.BytesIO(); Image.fromarray(img.astype(np.uint8)).save(b, 'JPEG', quality=int(r.integers(65, 96))); img = np.asarray(Image.open(b).convert('RGB'), np.float32)
        return torch.from_numpy(img.transpose(2, 0, 1) / 255.).float(), torch.from_numpy(mask[None])
