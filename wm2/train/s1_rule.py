"""1단계 규칙 기반: 한 페이지에서 '단색(흰/검은) 여백'을 찾고 그 안의 예시 워터마크 후보 상자를 돌려준다. 학습 없음.
규칙은 개발 세트(core)의 오검출·미검출을 보며 버전을 올린다(RULE_VERSION). 모든 크기는 원본 해상도 기준(이미지 폭 W 대비).
v1: 단색 영역(열림 연산으로 얇은 선·글자 제거) 안의 비배경 덩어리를 크기로만 거른다.
v2: 여백 조건 강화(폭 전체를 가로지르는 띠, 높이 ≥ 90px) + 가장자리에 닿는 덩어리·너무 가는 글줄 제외.
v3: 글자 한 줄(캡션·쪽번호)과 마크를 가르는 단서 추가 — 색 수(마크는 채워진 면/여러 색), 채움 비율, 선 두께.
이 파일의 함수는 (img uint8 HxWx3) → [(x,y,w,h,score)]."""
import numpy as np
from scipy import ndimage

RULE_VERSION = 5


def _solid_masks(g, tol=14):
    """흰/검은 단색 픽셀 마스크"""
    return (g >= 255 - tol), (g <= tol)


def find_margins(g, v, W):
    """단색 여백 영역 목록 [(y0,y1,bgv)] — 폭 전체를 가로지르는 띠"""
    H = g.shape[0]; out = []
    for bgv, m in zip((255, 0), _solid_masks(g)):
        # 행마다 단색 픽셀 비율. 마크가 있는 행도 통과하도록 낮은 문턱(45%)
        frac = m.mean(1); ok = frac > (.45 if v >= 2 else .3)
        # 연속 구간
        y = 0
        while y < H:
            if ok[y]:
                y1 = y
                while y1 < H and ok[y1]: y1 += 1
                if y1 - y >= (90 if v >= 2 else 40):
                    # 띠의 위·아래 경계는 거의 완전 단색 행이어야 한다(마크가 경계를 넘지 않음)
                    out.append((y, y1, bgv))
                y = y1
            else: y += 1
    return out


def detect(img, v=RULE_VERSION):
    g = img.astype(np.int16); gray = g.mean(2); H, W = gray.shape; boxes = []
    for y0, y1, bgv in find_margins(gray, v, W):
        band = img[y0:y1].astype(np.int16); diff = np.abs(band - bgv).max(2) > 28   # 배경과 다른 픽셀
        dil = ndimage.binary_dilation(diff, structure=np.ones((3, 3), bool), iterations=14 if v >= 1 else 8)
        lab, n = ndimage.label(dil)
        for i, sl in enumerate(ndimage.find_objects(lab)):
            comp = diff[sl] & (lab[sl] == i + 1)
            if comp.sum() < 120: continue
            ys, xs = np.where(comp); x0, yy0, x1_, yy1 = sl[1].start + xs.min(), sl[0].start + ys.min(), sl[1].start + xs.max() + 1, sl[0].start + ys.max() + 1
            w, h = x1_ - x0, yy1 - yy0
            if w > .5 * W or w < .06 * W or h < 18 or w * h > .2 * W * H: continue            # 너무 작거나 크다
            if v >= 2 and (x0 <= 2 or x1_ >= W - 2 or yy0 <= 1 or yy1 >= (y1 - y0) - 1): continue  # 띠 가장자리에 닿음 = 잘린 것
            if v in (2, 3) and w / max(1, h) > 9 and h < 34: continue                           # 가는 글줄(캡션·쪽번호). v4 부터는 재현율을 위해 빼고 2단계 반복 확인에 맡김
            if v >= 4:
                reg = band[yy0:yy1, x0:x1_]; m = comp[yy0 - sl[0].start:yy1 - sl[0].start, x0 - sl[1].start:x1_ - sl[1].start]
                fill = m.mean(); px = (reg[m] // 48).astype(np.int32); key = px[:, 0] * 64 + px[:, 1] * 8 + px[:, 2]; cnt = np.bincount(key, minlength=512); nb = int((cnt > .03 * len(key)).sum())
                closed = ndimage.binary_fill_holes(m).mean() > .35   # 닫힌 윤곽(원형 도장 테두리 등)은 선이 가늘어도 마크일 수 있음
                if fill < .09 and (v < 5 or not closed): continue        # 가는 선 몇 줄(집중선·머리카락 가닥)
                if nb > (6 if v < 5 else 9): continue            # 색이 연속적으로 번지는 것(눈 발광·불꽃 = 그라데이션). 마크는 단색 몇 개로 칠해짐
            if v == 3:
                reg = band[yy0:yy1, x0:x1_]; m = comp[yy0 - sl[0].start:yy1 - sl[0].start, x0 - sl[1].start:x1_ - sl[1].start]
                fill = m.mean(); cols = len({tuple(c) for c in (reg[m][::7] // 40).tolist()})
                if w / max(1, h) > 5 and fill < .35 and cols <= 3: continue                       # 단색 글줄(획만 있는 한 줄 글자)
                if fill < .06: continue                                                          # 선·점 몇 개
            boxes.append((int(x0), int(yy0 + y0), int(w), int(h), 1.0))
    return boxes
