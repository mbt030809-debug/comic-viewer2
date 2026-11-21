import os, pickle, random, io
import numpy as np, torch
from PIL import Image

class Synth(torch.utils.data.Dataset):
    """배경 풀 + 스프라이트 풀을 즉석에서 합성. 항목: (이미지 3xSxS 0..1, 마스크 1xSxS)"""
    def __init__(self, root, n, seed, size=192):
        self.bgs = np.load(os.path.join(root, 'bgs.npy'), mmap_mode='r'); self.spr = pickle.load(open(os.path.join(root, 'sprites.pkl'), 'rb'))
        self.n, self.seed, self.S = n, seed, size
        self.tile = [s for s in self.spr if s[0].startswith('tile') or s[0] == 'banner']; self.small = [s for s in self.spr if not (s[0].startswith('tile') or s[0] == 'banner')]
    def __len__(self): return self.n
    def __getitem__(self, i):
        r = np.random.default_rng(self.seed * 1000003 + i); S = self.S
        N, BH, BW, _ = self.bgs.shape
        bg = self.bgs[r.integers(N)]; x0 = int(r.integers(0, BW - S + 1)); y0 = int(r.integers(0, BH - S + 1))
        img = bg[y0:y0 + S, x0:x0 + S].astype(np.float32).copy(); mask = np.zeros((S, S), np.float32)
        if r.random() < .15:   # 미끼: 컷 테두리·선(정답 아님)
            c = float(r.choice([10, 60])); m = int(r.integers(4, 30)); w = int(r.integers(1, 4)); img[m:m + w, m:S - m] = c; img[S - m - w:S - m, m:S - m] = c; img[m:S - m, m:m + w] = c
        nm = int(r.choice([0, 1, 1, 1, 1, 2], p=None))
        for _ in range(nm):
            if r.random() < .25 and self.tile:
                kind, A, P = self.tile[int(r.integers(len(self.tile)))]
                A = A[y0:y0 + S, x0:x0 + S] if A.shape[0] >= y0 + S and A.shape[1] >= x0 + S else None
                if A is None: continue
                P = self._crop(P, y0, x0, S)
                a = float(np.exp(r.uniform(np.log(.1), np.log(.5)))) if kind.startswith('tile') else float(np.exp(r.uniform(np.log(.2), np.log(1))))
                A = A.astype(np.float32) / 255; old = img.copy(); img = img * (1 - a * A[..., None]) + a * P.astype(np.float32) * (A[..., None] > 0); sel = A > .03
                if sel.any() and np.percentile(np.abs(img - old).max(axis=2)[sel], 95) >= 5: mask = np.maximum(mask, sel.astype(np.float32))
                continue
            kind, A, P = self.small[int(r.integers(len(self.small)))]
            A = A.astype(np.float32) / 255; Pf = P.astype(np.float32)
            f = float(r.uniform(.6, 1.5)); h, w = A.shape
            nh, nw = max(8, int(h * f)), max(8, int(w * f))
            if nw > S - 4 or nh > S - 4: f = min((S - 4) / w, (S - 4) / h); nh, nw = max(8, int(h * f)), max(8, int(w * f))
            Aim = Image.fromarray(A, 'F').resize((nw, nh), Image.BILINEAR)
            if r.random() < .3: Aim = Aim.rotate(float(r.uniform(-15, 15)), expand=True, resample=Image.BILINEAR)
            sh, sw = Aim.size[1], Aim.size[0]
            Pim = np.stack([np.asarray(Image.fromarray(Pf[..., c], 'F').resize((nw, nh), Image.BILINEAR)) for c in range(3)], -1)
            if r.random() < .3 and (sw, sh) != (nw, nh): Pim = np.stack([np.asarray(Image.fromarray(Pim[..., c].astype(np.float32), 'F').rotate(0, expand=False)) for c in range(3)], -1)
            Aa = np.asarray(Aim, np.float32).clip(0, 1)
            if Pim.shape[:2] != Aa.shape: Pim = np.stack([np.asarray(Image.fromarray(Pim[..., c].astype(np.float32), 'F').resize((sw, sh), Image.BILINEAR)) for c in range(3)], -1)
            if sw >= S or sh >= S: continue
            px = int(r.integers(-sw // 6, S - sw + sw // 6 + 1)); py = int(r.integers(-sh // 6, S - sh + sh // 6 + 1))
            a = float(np.exp(r.uniform(np.log(.12), np.log(1.0))))
            xa, ya, xb, yb = max(0, px), max(0, py), min(S, px + sw), min(S, py + sh)
            if xb <= xa or yb <= ya: continue
            Ac = Aa[ya - py:yb - py, xa - px:xb - px]; Pc = Pim[ya - py:yb - py, xa - px:xb - px]
            old = img[ya:yb, xa:xb].copy(); img[ya:yb, xa:xb] = old * (1 - a * Ac[..., None]) + a * Pc
            sel = Ac > .03
            if sel.any() and np.percentile(np.abs(img[ya:yb, xa:xb] - old).max(axis=2)[sel], 95) >= 5: mask[ya:yb, xa:xb] = np.maximum(mask[ya:yb, xa:xb], sel.astype(np.float32))
        img = np.clip(img, 0, 255)
        if r.random() < .6:   # JPEG 왕복
            b = io.BytesIO(); Image.fromarray(img.astype(np.uint8)).save(b, 'JPEG', quality=int(r.integers(65, 96))); img = np.asarray(Image.open(b).convert('RGB'), np.float32)
        return torch.from_numpy(img.transpose(2, 0, 1) / 255.), torch.from_numpy(mask[None])
    @staticmethod
    def _crop(P, y0, x0, S): return P[y0:y0 + S, x0:x0 + S]
