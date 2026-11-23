"""만화 선화 하드 네거티브(합성): 집중선, 머리카락 가닥, 눈의 광원·하이라이트, 불꽃·연기.
실제 웹툰에서 워터마크로 오인식된 요소들이다(사용자 확인: 집중선·머리카락·눈·불꽃). 모두 '워터마크 아님'이며 마스크에 들어가지 않는다.
함수는 (img[H,W,3] float 0..255, rng numpy Generator) → img. 슈퍼샘플로 그려 가장자리를 부드럽게 한다."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _blend(img, layer_rgb, alpha):
    a = alpha[..., None]; return img * (1 - a) + layer_rgb * a


def _canvas(H, W, ss=2):
    return Image.new('RGBA', (W * ss, H * ss), (0, 0, 0, 0))


def _flatten(img, lay, ss=2):
    H, W = img.shape[:2]; lay = lay.resize((W, H), Image.LANCZOS) if ss != 1 else lay
    a = np.asarray(lay, np.float32); al = a[..., 3] / 255.
    return _blend(img, a[..., :3], al)


def speed_lines(img, r):
    """한 점에서 퍼지는 가는 선 다발(흑 또는 백). 화면 일부 영역에만."""
    H, W = img.shape[:2]; ss = 2; lay = _canvas(H, W, ss); d = ImageDraw.Draw(lay)
    cx, cy = r.uniform(-.3, 1.3) * W, r.uniform(-.3, 1.3) * H; n = int(r.integers(40, 170)); R0 = r.uniform(.1, .5) * min(H, W)
    col = (0, 0, 0, 255) if r.random() < .6 else (255, 255, 255, 255)
    for _ in range(n):
        th = r.uniform(0, 2 * math.pi); p1 = R0 * r.uniform(1., 1.4); L = r.uniform(.15, 1.3) * max(H, W); p2 = p1 + L; w = r.uniform(.8, 3.5)
        ux, uy = math.cos(th), math.sin(th); nx, ny = -uy, ux
        a = (cx + ux * p1, cy + uy * p1); b = (cx + ux * p2, cy + uy * p2)
        d.polygon([((a[0] + nx * w) * ss, (a[1] + ny * w) * ss), (b[0] * ss, b[1] * ss), ((a[0] - nx * w) * ss, (a[1] - ny * w) * ss)], fill=col)
    # 일부 영역만(상자/원형 마스크)
    x0, y0 = int(r.uniform(0, .5) * W), int(r.uniform(0, .5) * H); x1, y1 = int(x0 + r.uniform(.4, 1.) * W), int(y0 + r.uniform(.4, 1.) * H)
    m = Image.new('L', lay.size, 0); ImageDraw.Draw(m).rectangle([x0 * ss, y0 * ss, x1 * ss, y1 * ss], fill=255); lay.putalpha(Image.fromarray(np.minimum(np.asarray(lay.getchannel('A')), np.asarray(m))))
    return _flatten(img, lay, ss)


def hair(img, r):
    """머리카락·털 가닥: 곡선 다발(어둡거나 밝거나 갈색)"""
    H, W = img.shape[:2]; ss = 2; lay = _canvas(H, W, ss); d = ImageDraw.Draw(lay)
    for _ in range(int(r.integers(1, 4))):
        sx, sy = r.uniform(0, W), r.uniform(0, H); ang = r.uniform(0, 2 * math.pi); k = int(r.integers(5, 20)); curv = r.uniform(-.03, .03)
        base = r.choice(['dark', 'light', 'brown']);
        for i in range(k):
            x, y = sx + r.normal(0, 6), sy + r.normal(0, 6); a = ang + r.normal(0, .12); L = int(r.integers(25, 90)); w = r.uniform(.8, 2.8)
            c = {'dark': (int(r.integers(10, 60)),) * 3, 'light': (int(r.integers(200, 255)),) * 3, 'brown': (int(r.integers(90, 170)), int(r.integers(50, 110)), int(r.integers(20, 70)))}[str(base)] + (255,)
            pts = []
            for t in range(L):
                a += curv + r.normal(0, .01); x += math.cos(a); y += math.sin(a); pts.append((x * ss, y * ss))
            if len(pts) > 2: d.line(pts, fill=c, width=max(1, int(w * ss)))
    return _flatten(img, lay, ss)


def eye_glow(img, r):
    """눈(아몬드 모양) + 발광 홍채/하이라이트"""
    H, W = img.shape[:2]; ss = 2; lay = _canvas(H, W, ss); d = ImageDraw.Draw(lay)
    cx, cy = r.uniform(.15, .85) * W, r.uniform(.15, .85) * H; w = r.uniform(24, 70); h = w * r.uniform(.3, .55); ang = r.uniform(-.3, .3)
    def pt(u, v): return ((cx + (u * math.cos(ang) - v * math.sin(ang))) * ss, (cy + (u * math.sin(ang) + v * math.cos(ang))) * ss)
    upper = [pt(w * (t / 20 - .5) * 2 / 2, -h * math.sin(math.pi * t / 20)) for t in range(21)]; lower = [pt(w * (t / 20 - .5), h * .7 * math.sin(math.pi * t / 20)) for t in range(20, -1, -1)]
    d.polygon(upper + lower, fill=(240, 240, 240, 255) if r.random() < .5 else (10, 10, 10, 255)); d.line(upper, fill=(0, 0, 0, 255), width=int(2.5 * ss))
    ir = h * r.uniform(.5, .9); d.ellipse([(cx - ir) * ss, (cy - ir) * ss, (cx + ir) * ss, (cy + ir) * ss], fill=(30, 20, 20, 255))
    out = _flatten(img, lay, ss); yy, xx = np.mgrid[0:H, 0:W]; dd = ((xx - cx) ** 2 + (yy - cy) ** 2) / (ir * r.uniform(1.2, 2.6)) ** 2
    g = np.exp(-dd * 2) * r.uniform(.5, 1.); col = np.array(r.choice([(255, 235, 150), (255, 255, 255), (140, 220, 255), (255, 120, 90)]), np.float32)
    out = out * (1 - g[..., None]) + col * g[..., None]
    hd = ((xx - cx - ir * .3) ** 2 + (yy - cy - ir * .3) ** 2) < (ir * .28) ** 2; out[hd] = 255
    return out


def flames(img, r):
    """불꽃·연기: 세로로 늘어난 노이즈를 문턱으로 잘라 만든 반투명 덩어리(오렌지~노랑)"""
    H, W = img.shape[:2]
    n = np.asarray(Image.fromarray((r.random((int(r.integers(3, 7)), int(r.integers(8, 20)))) * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC), np.float32) / 255
    n2 = np.asarray(Image.fromarray((r.random((int(r.integers(6, 14)), int(r.integers(12, 30)))) * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC), np.float32) / 255
    f = .65 * n + .35 * n2; t = r.uniform(.45, .62); a = np.clip((f - t) / .08, 0, 1) * r.uniform(.3, .75)
    yy = np.linspace(0, 1, H)[:, None]; c1 = np.array(r.choice([(255, 140, 30), (255, 90, 20), (140, 200, 255), (255, 210, 80)]), np.float32); c2 = np.array((255, 230, 120), np.float32)
    col = c1 * (1 - yy[..., None]) + c2 * yy[..., None]
    return _blend(img, np.broadcast_to(col, (H, W, 3)), a)


FX = [speed_lines, hair, eye_glow, flames]


def apply_fx(img, r, p=.6, kinds=None):
    """확률 p 로 1~2개 적용. 반환 (img, 적용한 이름 목록)"""
    used = []
    if r.random() < p:
        fs = kinds or FX
        for i in r.choice(len(fs), size=int(r.integers(1, 3)), replace=False): img = fs[int(i)](img, r); used.append(fs[int(i)].__name__)
    return img, used
