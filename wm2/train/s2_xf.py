"""2단계 XFeat 특징점 매칭(x): XFeat(CVPR 2024, 가벼운 CNN 특징점·기술자)로 예시와 그림의 특징점을 맞추고 RANSAC 닮음변환으로 상자를 낸다. s2_kp(SIFT)와 같은 인터페이스.
추출부는 xfeat_port.py(원 저장소를 옮겨 적은 코드)를 쓰고, 가중치 xfeat.pt 는 환경변수 XFEAT_WEIGHTS(기본 C:\\wm2work\\ext\\xfeat\\weights\\xfeat.pt)에서 읽는다."""
import os
import numpy as np, torch, cv2
from scipy import ndimage
import xfeat_port

WEIGHTS = os.environ.get('XFEAT_WEIGHTS', r'C:\wm2work\ext\xfeat\weights\xfeat.pt')
_xf = None


def _net():
    global _xf
    if _xf is None: _xf = xfeat_port.load(WEIGHTS); torch.set_num_threads(4)
    return _xf


def _feat(gray_u8, top_k=3000): return _net().detect_and_compute(gray_u8.astype(np.float32) / 255, top_k)


def page_repr(pil):
    g = np.asarray(pil.convert('L'), np.uint8); pts, des = _feat(g); return (pts, des, g.shape)


def prepare(crop_rgb, bg):
    g = np.asarray(cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY), np.uint8); m = (np.abs(crop_rgb.astype(np.int16) - bg).max(2) > 28); m = ndimage.binary_dilation(m, iterations=2); ys, xs = np.where(m)
    if len(xs) == 0: return None, None
    g2 = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)   # 예시가 작아 2배 키워 특징점을 더 얻는다(좌표는 나중에 1/2)
    pts, des = _feat(g2, 1500)
    if len(pts) == 0: return None, None
    pts = pts / 2; keep = m[np.clip(pts[:, 1].astype(int), 0, m.shape[0] - 1), np.clip(pts[:, 0].astype(int), 0, m.shape[1] - 1)]
    return (pts[keep], des[keep], (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)), m


def match(rep, tpl, mask, thr=10, max_inst=4, min_cossim=.82):
    if tpl is None or len(rep[0]) < thr or len(tpl[0]) < 4: return []
    pts_t, des_t, (bx0, by0, bx1, by1) = tpl; pts_p, des_p, shape = rep; H, W = shape
    sim = des_t @ des_p.T; n12 = sim.argmax(1); n21 = sim.argmax(0); idx = np.arange(len(des_t)); ok = (n21[n12] == idx) & (sim[idx, n12] > min_cossim)
    src_all = pts_t[ok]; dst_all = pts_p[n12[ok]]; out = []
    for _ in range(max_inst):
        if len(src_all) < max(4, thr): break
        M, inl = cv2.estimateAffinePartial2D(src_all.reshape(-1, 1, 2), dst_all.reshape(-1, 1, 2), method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=1000, confidence=.99)
        if M is None or inl is None or int(inl.sum()) < thr: break
        s = float(np.hypot(M[0, 0], M[1, 0])); iv = inl.ravel().astype(bool)
        if .55 <= s <= 1.8:
            c = np.float32([[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]]).reshape(-1, 1, 2); q = cv2.transform(c, M).reshape(-1, 2)
            x0, y0, x1, y1 = max(0, q[:, 0].min()), max(0, q[:, 1].min()), min(W, q[:, 0].max()), min(H, q[:, 1].max())
            if x1 - x0 > 8 and y1 - y0 > 8: out.append((int(x0), int(y0), int(x1 - x0), int(y1 - y0), float(iv.sum())))
        src_all, dst_all = src_all[~iv], dst_all[~iv]
    return out
