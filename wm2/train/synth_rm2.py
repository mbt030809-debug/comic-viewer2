"""워터마크 제거 학습 합성기 v2 (사양: wm2/RECOGNITION_SPEC_20261123.md). 항목: (합성본, 입력 마스크, 깨끗한 원본, 정답 마스크).
v1(synth_rm.py) 대비: 캐릭터 마크 추가, 마크 크기 범위(로고 8~30%·글자 10~50%), 배경에 만화 선화(집중선·머리카락·눈·불꽃)가 깨끗한 원본에도 들어감,
알파는 낮은 값(0.12~0.7) 위주. 0.5배 세계(폭 360)에서 학습한다."""
import io
import numpy as np, torch
from PIL import Image
from synth2 import distract
from synth_rm import dilate
from synth_ref2 import SynthRef2, put2
from manga_fx import apply_fx


class SynthRm2(SynthRef2):
    P_EMPTY = .05

    def __getitem__(self, i):
        r = np.random.default_rng(self.seed * 1000003 + i); S = self.S
        N, BH, BW, _ = self.bgs.shape
        bg = self.bgs[r.integers(N)]; x0 = int(r.integers(0, BW - S + 1)); y0 = int(r.integers(0, BH - S + 1))
        img = bg[y0:y0 + S, x0:x0 + S].astype(np.float32).copy(); mask = np.zeros((S, S), np.float32)
        if r.random() < .15:
            c = float(r.choice([10, 60])); m = int(r.integers(4, 30)); w = int(r.integers(1, 4)); img[m:m + w, m:S - m] = c; img[S - m - w:S - m, m:S - m] = c; img[m:S - m, m:m + w] = c
        if r.random() < self.P_DIST: img = distract(img, r, S)
        img, _ = apply_fx(img, r, p=self.P_MANGA)
        clean = np.clip(img, 0, 255).copy()
        nm = 0 if r.random() < self.P_EMPTY else int(r.choice([1, 1, 1, 2]))
        for _ in range(nm):
            spr = self.small[int(r.integers(len(self.small)))]; al = float(np.exp(r.uniform(np.log(.12), np.log(.7)))) if r.random() < .85 else float(r.uniform(.7, 1.0))
            img, m = put2(img, spr, r, S, al)
            if m is not None: mask = np.maximum(mask, m)
        img = np.clip(img, 0, 255)
        if r.random() < .6:
            b = io.BytesIO(); Image.fromarray(img.astype(np.uint8)).save(b, 'JPEG', quality=int(r.integers(65, 96))); img = np.asarray(Image.open(b).convert('RGB'), np.float32)
        u = r.random()
        if u < .35 and mask.any():
            ys, xs = np.where(mask > 0); pad = int(r.integers(0, 5)); im = np.zeros_like(mask); im[max(0, ys.min() - pad):ys.max() + 1 + pad, max(0, xs.min() - pad):xs.max() + 1 + pad] = 1
        else: im = dilate(mask, int(r.integers(1, 6)))
        t = lambda a: torch.from_numpy(a.transpose(2, 0, 1) / 255.).float()
        return t(img), torch.from_numpy(im[None]).float(), t(clean), torch.from_numpy(mask[None]).float()
