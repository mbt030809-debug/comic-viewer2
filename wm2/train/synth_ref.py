"""참조형 인식 학습 합성기. 항목: (타깃 3xSxS 0..1, 예시 마크 3xRHxRW 0..1, 마스크 1xSxS)
예시 마크 = 풀 스프라이트를 흰 또는 검은 배경에 불투명하게 그린 것(갤러리 첫 장 모양). 타깃에는 그 마크가 위치·크기·알파가 다르게 박히거나(양성),
다른 마크만 있거나(음성, 같은 종류 마크 35%), 아무것도 없다. 마스크 = 예시 마크가 눈에 보이게 박힌 픽셀만(다른 워터마크는 0)."""
import io
import numpy as np, torch
from PIL import Image
from synth2 import Synth2, distract

RH, RW = 112, 192


def lum(A, P):
    sel = A > 128
    if not sel.any(): return 128.
    c = P[sel].astype(np.float32) / np.maximum(A[sel][:, None].astype(np.float32) / 255, 1e-3)
    return float(np.mean(c.clip(0, 255) @ np.array([.299, .587, .114])))


def make_ref(spr, r):
    """스프라이트 → 예시 캔버스(RH x RW x 3, 0..255) 와 배경값"""
    kind, A, P = spr; A = A.astype(np.float32) / 255; Pf = P.astype(np.float32)
    l = lum(spr[1], spr[2]); bg = 0 if l > 170 else 255 if l < 85 else float(r.choice([0, 255]))
    if r.random() < .08: bg = float(r.integers(0, 256))
    ys, xs = np.where(A > .03)
    if len(xs) == 0: return None, bg
    A = A[ys.min():ys.max() + 1, xs.min():xs.max() + 1]; Pf = Pf[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = A.shape; f = min(1., (RW - 8) / w, (RH - 8) / h)
    if f < 1:
        nh, nw = max(4, int(h * f)), max(4, int(w * f)); A = np.asarray(Image.fromarray(A, 'F').resize((nw, nh), Image.BILINEAR)); Pf = np.stack([np.asarray(Image.fromarray(Pf[..., c], 'F').resize((nw, nh), Image.BILINEAR)) for c in range(3)], -1); h, w = nh, nw
    can = np.full((RH, RW, 3), bg, np.float32); y0 = (RH - h) // 2 + int(r.integers(-3, 4)); x0 = (RW - w) // 2 + int(r.integers(-3, 4)); y0 = max(0, min(RH - h, y0)); x0 = max(0, min(RW - w, x0))
    reg = can[y0:y0 + h, x0:x0 + w]; can[y0:y0 + h, x0:x0 + w] = reg * (1 - A[..., None]) + Pf
    return np.clip(can, 0, 255), bg


def put(img, spr, r, S, alpha, fr=(.7, 1.3)):
    """스프라이트 하나를 img(SxSx3, float)에 합성. 반환 (img, 보이는 영역 마스크 or None)"""
    kind, A, P = spr; A = A.astype(np.float32) / 255; Pf = P.astype(np.float32)
    f = float(r.uniform(*fr)); h, w = A.shape; nh, nw = max(8, int(h * f)), max(8, int(w * f))
    if nw > S - 4 or nh > S - 4: f = min((S - 4) / w, (S - 4) / h); nh, nw = max(8, int(h * f)), max(8, int(w * f))
    Aim = Image.fromarray(A, 'F').resize((nw, nh), Image.BILINEAR)
    Pim = np.stack([np.asarray(Image.fromarray(Pf[..., c], 'F').resize((nw, nh), Image.BILINEAR)) for c in range(3)], -1)
    if r.random() < .3:
        ang = float(r.uniform(-12, 12)); Aim = Aim.rotate(ang, expand=True, resample=Image.BILINEAR)
        Pim = np.stack([np.asarray(Image.fromarray(Pim[..., c].astype(np.float32), 'F').rotate(ang, expand=True, resample=Image.BILINEAR)) for c in range(3)], -1)
    Aa = np.asarray(Aim, np.float32).clip(0, 1); sh, sw = Aa.shape
    if sw >= S or sh >= S: return img, None
    px = int(r.integers(0, S - sw + 1)); py = int(r.integers(0, S - sh + 1))
    old = img[py:py + sh, px:px + sw].copy(); img[py:py + sh, px:px + sw] = old * (1 - alpha * Aa[..., None]) + alpha * Pim
    sel = Aa > .03; m = None
    if sel.any() and np.percentile(np.abs(img[py:py + sh, px:px + sw] - old).max(axis=2)[sel], 95) >= 5:
        m = np.zeros((S, S), np.float32); m[py:py + sh, px:px + sw] = sel.astype(np.float32)
    return img, m


class SynthRef(Synth2):
    P_DIST = .6

    def __getitem__(self, i):
        r = np.random.default_rng(self.seed * 1000003 + i); S = self.S
        N, BH, BW, _ = self.bgs.shape
        bg = self.bgs[r.integers(N)]; x0 = int(r.integers(0, BW - S + 1)); y0 = int(r.integers(0, BH - S + 1))
        img = bg[y0:y0 + S, x0:x0 + S].astype(np.float32).copy(); mask = np.zeros((S, S), np.float32)
        if r.random() < .15:
            c = float(r.choice([10, 60])); m = int(r.integers(4, 30)); w = int(r.integers(1, 4)); img[m:m + w, m:S - m] = c; img[S - m - w:S - m, m:S - m] = c; img[m:S - m, m:m + w] = c
        if r.random() < self.P_DIST: img = distract(img, r, S)
        while True:
            ri = int(r.integers(len(self.small))); spr = self.small[ri]; ref, _ = make_ref(spr, r)
            if ref is not None: break
        al = lambda: float(np.exp(r.uniform(np.log(.12), np.log(.7))))
        u = r.random()
        if u < .55:   # 양성: 예시 마크가 박힘(1~2번), 다른 마크가 같이 있을 수도
            for _ in range(1 + (r.random() < .25)):
                img, m = put(img, spr, r, S, al())
                if m is not None: mask = np.maximum(mask, m)
            if r.random() < .4:
                for _ in range(int(r.integers(1, 3))): img, _ = put(img, self.small[int(r.integers(len(self.small)))], r, S, al())
        elif u < .9:  # 음성: 다른 마크만(같은 종류 35%)
            for _ in range(int(r.integers(1, 3))):
                if r.random() < .35:
                    same = [s for s in self.small if s[0] == spr[0] and s is not spr]; o = same[int(r.integers(len(same)))] if same else self.small[int(r.integers(len(self.small)))]
                else: o = self.small[int(r.integers(len(self.small)))]
                if o is spr: continue
                img, _ = put(img, o, r, S, al())
        img = np.clip(img, 0, 255)
        if r.random() < .6:
            b = io.BytesIO(); Image.fromarray(img.astype(np.uint8)).save(b, 'JPEG', quality=int(r.integers(65, 96))); img = np.asarray(Image.open(b).convert('RGB'), np.float32)
        t = lambda a: torch.from_numpy(a.transpose(2, 0, 1) / 255.).float()
        return t(img), t(ref), torch.from_numpy(mask[None])
