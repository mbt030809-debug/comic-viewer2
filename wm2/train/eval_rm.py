#!/usr/bin/env python3
"""워터마크 제거 평가(벤치의 img_NN.png = 워터마크 있음, clean_NN.jpg = 정답 원본).
사용: python eval_rm.py <rm.pt> --bench <dir> --tiers core,wild --tag X --out rows_rm.jsonl [--max-sites N] [--device dml|cpu]
마스크 = |img-clean| 최대 채널 차이 >= 8 인 곳(보이는 워터마크)을 --mdil 픽셀 팽창(인식기가 낸 영역 흉내).
지표(워터마크 영역 기준): 오차 감소율 = 1 - 결과오차/합성오차, 영역 PSNR 전→후, 영역 밖 변화 없음(블렌딩이라 0).
배경 환경별(style, codec, 마크 종류·모드, 알파 구간) 집계를 같이 낸다."""
import os, sys, json, time, argparse
from collections import defaultdict
import numpy as np, torch
from PIL import Image, ImageFilter
sys.path.insert(0, os.path.dirname(__file__))
from rm_model import RmUNet
from train_rm import get_device


def run_tiles(net, dev, img, m, T=512, mg=32):
    H, W, _ = img.shape; out = img.copy()
    for y in range(0, H, T):
        for x in range(0, W, T):
            if not m[y:y + T, x:x + T].any(): continue
            y0, x0, y1, x1 = max(0, y - mg), max(0, x - mg), min(H, y + T + mg), min(W, x + T + mg)
            a = img[y0:y1, x0:x1]; b = m[y0:y1, x0:x1]; h, w = a.shape[:2]; ph, pw = (-h) % 16, (-w) % 16
            a = np.pad(a, ((0, ph), (0, pw), (0, 0)), mode='reflect'); b = np.pad(b, ((0, ph), (0, pw)), mode='constant')
            ta = torch.from_numpy(a.transpose(2, 0, 1)[None].copy()); tb = torch.from_numpy(b[None, None].copy())
            with torch.no_grad(): o = net(ta.to(dev), tb.to(dev)).cpu().numpy()[0].transpose(1, 2, 0)[:h, :w]
            ys, xs = y - y0, x - x0; hh, ww = min(T, H - y), min(T, W - x)
            out[y:y + hh, x:x + ww] = o[ys:ys + hh, xs:xs + ww]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('ckpt'); ap.add_argument('--bench', required=True); ap.add_argument('--tiers', default='core,wild'); ap.add_argument('--tag', default='rm'); ap.add_argument('--out'); ap.add_argument('--device', default='dml')
    ap.add_argument('--max-sites', type=int, default=0); ap.add_argument('--mdil', type=int, default=3); ap.add_argument('--thr', type=int, default=8); ap.add_argument('--scale', type=float, default=1.0); ap.add_argument('--mode', default='native', choices=['native', 'half_resid']); ap.add_argument('--tsvdir')
    a = ap.parse_args(); dev = get_device(a.device)
    d = torch.load(a.ckpt, map_location='cpu', weights_only=True); net = RmUNet(tuple(d['ch'])); net.load_state_dict(d['sd']); net.to(dev).eval()
    params = sum(p.numel() for p in net.parameters())
    for tier in a.tiers.split(','):
        root = os.path.join(a.bench, tier); truth = json.load(open(os.path.join(root, 'truth.json'), encoding='utf-8')); sites = [s for s in truth['sites'] if s.get('marks')]
        if a.max_sites: sites = sites[::max(1, len(sites) // a.max_sites)][:a.max_sites]
        groups = defaultdict(lambda: [0., 0., 0., 0.]); tot = [0., 0., 0., 0.]; rowsout = []; t0 = time.time(); nimg = 0
        for s in sites:
            sd = os.path.join(root, s['id']); st = [0., 0., 0., 0.]   # 합성오차합, 결과오차합, 픽셀수, mse전, mse후는 아래 별도
            e0s = e1s = n = m0 = m1 = 0.
            for f in sorted(os.listdir(sd)):
                if not (f.startswith('img_') and f[-4:] in ('.png', '.jpg')): continue
                k = f[4:-4]; cp = os.path.join(sd, 'clean_%s.jpg' % k)
                if not os.path.exists(cp): continue
                img = np.asarray(Image.open(os.path.join(sd, f)).convert('RGB'), np.float32) / 255; cl = np.asarray(Image.open(cp).convert('RGB'), np.float32) / 255
                if img.shape != cl.shape: continue
                if a.scale != 1:   # 학습이 0.5배 세계(폭 360)라 같은 크기로 줄여서 평가(--scale 0.5)
                    h2, w2 = int(img.shape[0] * a.scale), int(img.shape[1] * a.scale)
                    img = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize((w2, h2), Image.BOX), np.float32) / 255; cl = np.asarray(Image.fromarray((cl * 255).astype(np.uint8)).resize((w2, h2), Image.BOX), np.float32) / 255
                gt = (np.abs(img - cl).max(2) * 255 >= a.thr)
                if gt.sum() < 20: continue
                mi = np.asarray(Image.fromarray(gt.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(2 * a.mdil + 1)), np.float32) / 255
                if a.mode == 'half_resid':   # 0.5배에서 제거 → 변화량(잔차)만 원본 크기로 키워 원본에 더함(원본 해상도 보존)
                    H0, W0 = img.shape[:2]; im8 = lambda x: Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8))
                    ih = np.asarray(im8(img).resize((W0 // 2, H0 // 2), Image.BOX), np.float32) / 255
                    mh = (np.asarray(Image.fromarray((mi * 255).astype(np.uint8)).resize((W0 // 2, H0 // 2), Image.BILINEAR), np.float32) / 255 > .3).astype(np.float32)
                    oh = run_tiles(net, dev, ih, mh); dl = oh - ih
                    df = np.stack([np.asarray(Image.fromarray(dl[..., c].astype(np.float32), 'F').resize((W0, H0), Image.BICUBIC)) for c in range(3)], -1)
                    out = np.clip(img + df * mi[..., None], 0, 1)
                else: out = run_tiles(net, dev, img, mi)
                g = gt[..., None]
                e0s += float((np.abs(img - cl) * g).sum()); e1s += float((np.abs(out - cl) * g).sum()); n += float(g.sum() * 3)
                m0 += float((((img - cl) ** 2) * g).sum()); m1 += float((((out - cl) ** 2) * g).sum()); nimg += 1
            if n == 0: continue
            mk = s['marks'][0]; al = mk.get('alpha', 0) or 0
            keys = ['all', 'style=%s' % s.get('style'), 'codec=%s' % s.get('codec'), 'kind=%s' % mk.get('kind'), 'mode=%s' % mk.get('mode'), 'alpha=%s' % ('<0.2' if al < .2 else '0.2-0.4' if al < .4 else '>=0.4')]
            for kk in keys:
                g_ = groups[kk]; g_[0] += e0s; g_[1] += e1s; g_[2] += n; g_[3] += 1
            rowsout.append((s['id'], 100 * (1 - e1s / max(e0s, 1e-9)), 10 * np.log10(n / max(m0, 1e-12)), 10 * np.log10(n / max(m1, 1e-12))))
        print('[%s %s] %d장 %.0fs' % (a.tag, tier, nimg, time.time() - t0))
        res = {}
        for kk in sorted(groups, key=lambda x: (x != 'all', x)):
            e0, e1, n, ns = groups[kk]; res[kk] = dict(reduction=100 * (1 - e1 / max(e0, 1e-9)), sites=int(ns)); print('  %-22s 영역 오차 감소 %5.1f%% (사이트 %d)' % (kk, res[kk]['reduction'], ns))
        if a.tsvdir:
            os.makedirs(a.tsvdir, exist_ok=True)
            with open(os.path.join(a.tsvdir, '%s_%s_rm.tsv' % (a.tag, tier)), 'w', encoding='utf-8') as fo:
                fo.write('site\treduction_pct\tpsnr_before\tpsnr_after\n')
                for r in rowsout: fo.write('%s\t%.1f\t%.1f\t%.1f\n' % r)
        if a.out: open(a.out, 'a', encoding='utf-8').write(json.dumps(dict(tag=a.tag, ckpt=os.path.basename(a.ckpt), params=params, tier=tier, mdil=a.mdil, groups=res, images=nimg), ensure_ascii=False) + '\n')


if __name__ == '__main__': main()
