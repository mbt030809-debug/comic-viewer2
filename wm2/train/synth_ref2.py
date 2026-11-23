"""참조형 인식 학습 합성기 v2 (사양: wm2/RECOGNITION_SPEC_20261123.md)
v1 대비 변경: ① 캐릭터(마스코트) 마크 종류 추가 ② 마크 크기를 사양 범위로 제한(로고류 폭 8~30%, 글자류 10~50% — 0.5배 세계 폭 360px 기준)
③ 만화 선화 하드 네거티브(집중선·머리카락·눈 하이라이트·불꽃)를 배경에 넣음(워터마크 아님, 마스크에 안 들어감)."""
import os, io, pickle
import numpy as np, torch
from PIL import Image
from synth2 import distract
import synth_ref as V1
from synth_ref import SynthRef, make_ref, RH, RW
from manga_fx import apply_fx

IMG_W = 360   # 0.5배 세계의 이미지 폭(학습 풀 배경 폭)
LOGO = {'badge', 'emblem', 'icon', 'qr', 'character'}
BOUNDS = {True: (.08 * IMG_W, .30 * IMG_W), False: (.10 * IMG_W, .50 * IMG_W)}   # 로고류 / 글자류 폭(px)


def ensure_chars(root, n=2500):
    p = os.path.join(root, 'extra_chars.pkl')
    if not os.path.exists(p):
        from marks_extra import character_sprite
        pickle.dump([character_sprite(100000 + i, 1) for i in range(n)], open(p + '.tmp', 'wb')); os.replace(p + '.tmp', p)
    return pickle.load(open(p, 'rb'))


def put2(img, spr, r, S, alpha, fr=(.75, 1.33)):
    """크기 범위를 지키며 합성. 반환 (img, 보이는 영역 마스크 or None)"""
    kind, A, P = spr; w = A.shape[1]; lo, hi = BOUNDS[kind in LOGO]
    flo, fhi = max(fr[0], lo / w), min(fr[1], hi / w)
    f = float(r.uniform(flo, fhi)) if flo <= fhi else float(np.clip(1., lo / w, hi / w))
    return V1.put(img, spr, r, S, alpha, fr=(f, f))


class SynthRef3Mixin:
    pass


class SynthRef2(SynthRef):
    P_MANGA = .6
    P_OPAQUE = 0.0

    def _load(self):
        super()._load(); ex = ensure_chars(self.root); self.small = list(self.small) + ex

    def __getitem__(self, i):
        r = np.random.default_rng(self.seed * 1000003 + i); S = self.S
        N, BH, BW, _ = self.bgs.shape
        bg = self.bgs[r.integers(N)]; x0 = int(r.integers(0, BW - S + 1)); y0 = int(r.integers(0, BH - S + 1))
        img = bg[y0:y0 + S, x0:x0 + S].astype(np.float32).copy(); mask = np.zeros((S, S), np.float32)
        if r.random() < .15:
            c = float(r.choice([10, 60])); m = int(r.integers(4, 30)); w = int(r.integers(1, 4)); img[m:m + w, m:S - m] = c; img[S - m - w:S - m, m:S - m] = c; img[m:S - m, m:m + w] = c
        if r.random() < self.P_DIST: img = distract(img, r, S)
        img, _ = apply_fx(img, r, p=self.P_MANGA)
        while True:
            spr = self.small[int(r.integers(len(self.small)))]; ref, _ = make_ref(spr, r)
            if ref is not None: break
        al = lambda: 1.0 if r.random() < self.P_OPAQUE else float(np.exp(r.uniform(np.log(.12), np.log(.7))))   # 불투명 비율(v3: 60%)
        u = r.random()
        if u < .55:
            for _ in range(1 + (r.random() < .25)):
                img, m = put2(img, spr, r, S, al())
                if m is not None: mask = np.maximum(mask, m)
            if r.random() < .4:
                for _ in range(int(r.integers(1, 3))): img, _ = put2(img, self.small[int(r.integers(len(self.small)))], r, S, al())
        elif u < .9:
            for _ in range(int(r.integers(1, 3))):
                if r.random() < .35:
                    same = [s for s in self.small if s[0] == spr[0] and s is not spr]; o = same[int(r.integers(len(same)))] if same else self.small[int(r.integers(len(self.small)))]
                else: o = self.small[int(r.integers(len(self.small)))]
                if o is spr: continue
                img, _ = put2(img, o, r, S, al())
        img = np.clip(img, 0, 255)
        if r.random() < .6:
            b = io.BytesIO(); Image.fromarray(img.astype(np.uint8)).save(b, 'JPEG', quality=int(r.integers(65, 96))); img = np.asarray(Image.open(b).convert('RGB'), np.float32)
        t = lambda a: torch.from_numpy(a.transpose(2, 0, 1) / 255.).float()
        return t(img), t(ref), torch.from_numpy(mask[None])
