"""워터마크 제거 학습용 합성기. (합성본 3xSxS, 입력 마스크 1xSxS, 깨끗한 원본 3xSxS, 정답 마스크 1xSxS) 를 돌려준다.
synth2 의 배경·미끼(말풍선·글자)·워터마크 합성을 그대로 쓰되 워터마크가 없는 'clean' 을 같이 기록한다.
입력 마스크는 실제로는 인식기가 낸 상자/마스크이므로 정답 마스크를 흔들어서 쓴다: 팽창 · 상자 채우기 · 일부 누락/과잉."""
import io
import numpy as np, torch
from PIL import Image, ImageFilter
from synth2 import Synth2, distract


def dilate(m, k):
    if k <= 0: return m
    return np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(2 * k + 1)), np.float32) / 255


class SynthRm(Synth2):
    P_DIST = .6
    P_EMPTY = .05

    def __getitem__(self, i):
        r = np.random.default_rng(self.seed * 1000003 + i); S = self.S
        N, BH, BW, _ = self.bgs.shape
        bg = self.bgs[r.integers(N)]; x0 = int(r.integers(0, BW - S + 1)); y0 = int(r.integers(0, BH - S + 1))
        img = bg[y0:y0 + S, x0:x0 + S].astype(np.float32).copy(); mask = np.zeros((S, S), np.float32)
        if r.random() < .15:
            c = float(r.choice([10, 60])); m = int(r.integers(4, 30)); w = int(r.integers(1, 4)); img[m:m + w, m:S - m] = c; img[S - m - w:S - m, m:S - m] = c; img[m:S - m, m:m + w] = c
        if r.random() < self.P_DIST: img = distract(img, r, S)
        clean = np.clip(img, 0, 255).copy()
        nm = 0 if r.random() < self.P_EMPTY else int(r.choice([1, 1, 1, 2]))
        for _ in range(nm):
            img, mask = self._apply_one(img, mask, r, S, x0, y0)
        img = np.clip(img, 0, 255)
        if r.random() < .6:
            q = int(r.integers(65, 96))
            b = io.BytesIO(); Image.fromarray(img.astype(np.uint8)).save(b, 'JPEG', quality=q); img = np.asarray(Image.open(b).convert('RGB'), np.float32)
        # 입력 마스크(인식기 출력 흉내)
        u = r.random()
        if u < .35 and mask.any():   # 상자 마스크
            ys, xs = np.where(mask > 0); pad = int(r.integers(0, 5)); im = np.zeros_like(mask); im[max(0, ys.min() - pad):ys.max() + 1 + pad, max(0, xs.min() - pad):xs.max() + 1 + pad] = 1
        else:
            im = dilate(mask, int(r.integers(1, 6)))
        if mask.any() and r.random() < .1: im = im * 0   # 인식 실패(제거하지 말아야 함) 대비 — 목표는 그대로 두는 것이 아니라 측정용으로 마스크 없음 학습도 포함
        t = lambda a: torch.from_numpy(a.transpose(2, 0, 1) / 255.).float()
        return t(img), torch.from_numpy(im[None]).float(), t(clean), torch.from_numpy(mask[None]).float()
