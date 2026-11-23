"""2단계 고전 특징점 매칭(k): 예시 마크의 SIFT 특징점을 그림 위 특징점과 맞춘 뒤 RANSAC 으로 닮음변환(크기·회전·이동)을 구해 상자를 낸다. 학습 없음, 빠름.
템플릿 매칭(s2_tm)과 같은 인터페이스: page_repr(PIL) / prepare(crop, bg) / match(page_repr, tpl, mask, thr). thr = 최소 인라이어 수.
불투명 마크(그림 위에 그대로 얹힘)에 강하고 크기·회전 변화에 강하다. 반투명·얇은 글자는 특징점이 적어 약하다. 원본 해상도 회색조에서 계산."""
import numpy as np
import cv2
from scipy import ndimage

_sift = None


def _s():
    global _sift
    if _sift is None: _sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.02)
    return _sift


def _gray(a): return (a[..., 0] * .299 + a[..., 1] * .587 + a[..., 2] * .114).astype(np.float32)


def page_repr(pil):
    g = np.asarray(pil.convert('L'), np.uint8); kp, des = _s().detectAndCompute(g, None)
    return (np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32), des, g.shape)


def prepare(crop_rgb, bg):
    g = np.asarray(cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY), np.uint8); m = (np.abs(crop_rgb.astype(np.int16) - bg).max(2) > 28); m = ndimage.binary_dilation(m, iterations=2).astype(np.uint8) * 255
    kp, des = _s().detectAndCompute(g, m); ys, xs = np.where(m > 0)
    if not kp or len(xs) == 0: return None, None
    return (np.float32([k.pt for k in kp]), des, (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)), m


def match(rep, tpl, mask, thr=10, max_inst=4, ratio=.78):
    """반환 [(x,y,w,h,score)] 원본 좌표. thr = 최소 인라이어 수"""
    if tpl is None or rep[1] is None or len(rep[0]) < thr or len(tpl[0]) < 4: return []
    pts_t, des_t, (bx0, by0, bx1, by1) = tpl; pts_p, des_p, shape = rep; H, W = shape
    bf = cv2.BFMatcher(cv2.NORM_L2); knn = bf.knnMatch(des_t, des_p, k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < ratio * m[1].distance]
    out = []
    for _ in range(max_inst):
        if len(good) < max(4, thr): break
        src = pts_t[[g.queryIdx for g in good]].reshape(-1, 1, 2); dst = pts_p[[g.trainIdx for g in good]].reshape(-1, 1, 2)
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=1000, confidence=.99)
        if M is None or inl is None or int(inl.sum()) < thr: break
        s = float(np.hypot(M[0, 0], M[1, 0]))
        keep = [g for g, i in zip(good, inl.ravel()) if not i]
        if .55 <= s <= 1.8:
            c = np.float32([[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]]).reshape(-1, 1, 2); q = cv2.transform(c, M).reshape(-1, 2)
            x0, y0, x1, y1 = max(0, q[:, 0].min()), max(0, q[:, 1].min()), min(W, q[:, 0].max()), min(H, q[:, 1].max())
            if x1 - x0 > 8 and y1 - y0 > 8: out.append((int(x0), int(y0), int(x1 - x0), int(y1 - y0), float(inl.sum())))
        good = keep
    return out
