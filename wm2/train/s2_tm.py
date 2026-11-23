"""2단계 규칙 기반(b): 예시 마크를 템플릿으로 그림 위에서 찾는다(다중 크기 마스크 정규화 상관, 학습 없음).
예시 = 1단계가 준 상자(단색 배경 위 마크). 마크 픽셀(배경과 다른 곳)만 마스크로 쓰고, 0.5배 해상도 회색조에서 크기 0.75~1.33배를 돌며 상관을 구한다.
정규화 상관은 밝기 변화·반투명 섞임(알파)에 어느 정도 강하지만, 회전·그림 질감이 심하면 약하다. 반환 [(x,y,w,h,score)] 원본 좌표."""
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.signal import fftconvolve

SCALES = (.72, .82, .92, 1.0, 1.1, 1.22, 1.38)   # 예시 대비 크기 비율 0.75~1.33 (+ 여유)


def _gray(a): return (a[..., 0] * .299 + a[..., 1] * .587 + a[..., 2] * .114).astype(np.float32)


def prepare(crop_rgb, bg):
    """예시 조각(uint8 RGB, 원본 해상도) → (0.5배 회색 템플릿, 마스크)"""
    h, w = crop_rgb.shape[:2]; im = Image.fromarray(crop_rgb); sm = im.resize((max(4, w // 2), max(4, h // 2)), Image.BOX); g = _gray(np.asarray(sm, np.float32))
    m = (np.abs(np.asarray(sm, np.float32) - bg).max(2) > 28); m = ndimage.binary_dilation(m, iterations=1)
    return g, m.astype(np.float32)


def match(page_gray_half, tpl, mask, thr=.55, scales=SCALES, max_area=.2):
    """page_gray_half: 0.5배 회색조 페이지. 반환 원본 좌표 상자 [(x,y,w,h,score)]"""
    H, W = page_gray_half.shape; I = page_gray_half; I2 = I * I; out = []
    for s in scales:
        t = np.asarray(Image.fromarray(tpl).resize((max(4, int(tpl.shape[1] * s)), max(4, int(tpl.shape[0] * s))), Image.BILINEAR), np.float32)
        m = np.asarray(Image.fromarray(mask).resize((t.shape[1], t.shape[0]), Image.BILINEAR), np.float32) > .5
        if m.sum() < 40 or t.shape[0] >= H or t.shape[1] >= W: continue
        w = m.astype(np.float32); w /= w.sum(); tb = (w * t).sum(); tz = (t - tb) * w; sT = np.sqrt(((t - tb) ** 2 * w).sum()) + 1e-6
        k = lambda a: a[::-1, ::-1]
        mu = fftconvolve(I, k(w), 'same'); var = fftconvolve(I2, k(w), 'same') - mu * mu; cov = fftconvolve(I, k(tz), 'same')
        ncc = np.where(var >= 25., cov / (np.sqrt(np.maximum(var, 25.)) * sT), 0.)   # 거의 평평한 곳(표준편차<5)은 상관이 잡음이라 0 처리
        # 지역 최대 + 문턱
        pk = (ncc == ndimage.maximum_filter(ncc, size=(max(5, t.shape[0] // 2), max(5, t.shape[1] // 2)))) & (ncc >= thr)
        for y, x in zip(*np.where(pk)):
            y0, x0 = y - t.shape[0] // 2, x - t.shape[1] // 2
            if y0 < 0 or x0 < 0 or y0 + t.shape[0] > H or x0 + t.shape[1] > W: continue
            ys, xs = np.where(m); bx = (x0 + xs.min(), y0 + ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
            if bx[2] * bx[3] * 4 > max_area * W * H * 4: continue
            out.append((int(bx[0] * 2), int(bx[1] * 2), int(bx[2] * 2), int(bx[3] * 2), float(ncc[y, x])))
    # 크기별 중복 제거(점수 높은 것 우선)
    out.sort(key=lambda b: -b[4]); keep = []
    for b in out:
        ok = True
        for k_ in keep:
            ix, iy = max(0, min(b[0] + b[2], k_[0] + k_[2]) - max(b[0], k_[0])), max(0, min(b[1] + b[3], k_[1] + k_[3]) - max(b[1], k_[1]))
            if ix * iy > .4 * min(b[2] * b[3], k_[2] * k_[3]): ok = False; break
        if ok: keep.append(b)
    return keep
