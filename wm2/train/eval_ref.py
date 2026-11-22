#!/usr/bin/env python3
"""참조형 인식 평가. 사용: python eval_ref.py <ref.pt> --bench <refdata> --tiers core,val --tag X --out rows_ref.jsonl [--device dml|cpu] [--max-sets N]
파이프라인(벤치 한 세트 = gallery.png + img_NN):
 1) gallery.png 를 예시 마크로 나눈다(행 왼쪽 가장자리 색 = 띠 배경, 배경과 다른 픽셀을 크게 팽창해 덩어리 라벨링 → 덩어리마다 한 예시).
 2) 예시 마크를 0.5배 축소해 112x192 캔버스(배경색)에 놓는다. 타깃 이미지는 0.5배 BOX 축소 + 16배수 reflect 패딩.
 3) 이미지마다 예시 마크별로 모델을 돌려 확률 지도 → 상자(좌표 x2). 면적이 화면의 20%를 넘는 상자는 버린다.
 4) 정답과 비교: 예시 번호가 같고 IoU>=0.5 이면 맞음.
지표: 상자 재현율·정밀도, 그 이미지에 없는 예시(이미지×예시 쌍) 오검출율, 마크가 하나도 없는 이미지의 오검출율, 이미지 단위 '어느 예시가 있나' 정답률, 갤러리 분리 성공률."""
import os, sys, json, time, argparse
from collections import defaultdict
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from ref_model import RefUNet, RH, RW
from infer_eval import boxes_from_mask
from train import get_device
from scipy import ndimage


def iou(a, b):
    ax, ay, aw, ah = a[:4]; bx, by, bw, bh = b[:4]
    ix, iy = max(0, min(ax + aw, bx + bw) - max(ax, bx)), max(0, min(ay + ah, by + bh) - max(ay, by)); i = ix * iy
    return i / max(1, aw * ah + bw * bh - i)

MAX_AREA = .20


def split_gallery(g):
    """갤러리 → [(상자x,y,w,h, 배경값)] 위에서 아래 순서"""
    bg = g[:, 0:1, :].astype(np.int16); nb = (np.abs(g.astype(np.int16) - bg).max(2) > 40)
    k = np.ones((41, 41), bool); dil = ndimage.binary_dilation(nb, structure=np.ones((3, 3), bool), iterations=20)
    lab, n = ndimage.label(dil); out = []
    for i in range(1, n + 1):
        ys, xs = np.where((lab == i) & nb)
        if len(xs) < 30: continue
        out.append((int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1), float(bg[int(ys.min()), 0].mean())))
    out.sort(key=lambda b: b[1]); return out


def make_ref(g, box):
    x, y, w, h, bg = box; crop = Image.fromarray(g[y:y + h, x:x + w]); nw, nh = max(4, w // 2), max(4, h // 2)
    a = np.asarray(crop.resize((nw, nh), Image.BOX), np.float32) / 255
    f = min(1., (RW - 8) / nw, (RH - 8) / nh)
    if f < 1: nh, nw = max(4, int(nh * f)), max(4, int(nw * f)); a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((nw, nh), Image.BOX), np.float32) / 255
    can = np.full((RH, RW, 3), bg / 255, np.float32); y0, x0 = (RH - nh) // 2, (RW - nw) // 2; can[y0:y0 + nh, x0:x0 + nw] = a
    return torch.from_numpy(can.transpose(2, 0, 1)[None].copy())


def target_tensor(path):
    img = Image.open(path).convert('RGB'); W, H = img.size; w, h = W // 2, H // 2
    a = np.asarray(img.resize((w, h), Image.BOX), np.float32) / 255; a = np.pad(a, ((0, (-h) % 16), (0, (-w) % 16), (0, 0)), mode='reflect').transpose(2, 0, 1)[None].copy()
    return torch.from_numpy(a), W, H


def match(preds, truths, thr=.5):
    """preds/truths: [(idx, box)] → (tp, fp, fn, 매칭된 pred 인덱스)"""
    used = set(); tp = 0
    for ti, tb in truths:
        best, bj = 0, -1
        for j, (pi, pb) in enumerate(preds):
            if j in used or pi != ti: continue
            v = iou(pb, tb)
            if v > best: best, bj = v, j
        if best >= thr: tp += 1; used.add(bj)
    return tp, len(preds) - tp, len(truths) - tp


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('ckpt'); ap.add_argument('--bench', required=True); ap.add_argument('--tiers', default='core,val'); ap.add_argument('--tag', default='ref'); ap.add_argument('--out'); ap.add_argument('--device', default='dml')
    ap.add_argument('--max-sets', type=int, default=0); ap.add_argument('--th', type=float, default=.5); a = ap.parse_args(); dev = get_device(a.device)
    d = torch.load(a.ckpt, map_location='cpu', weights_only=True); net = RefUNet(tuple(d['ch'])); net.load_state_dict(d['sd']); net.to(dev).eval(); params = sum(p.numel() for p in net.parameters())
    for tier in a.tiers.split(','):
        root = os.path.join(a.bench, tier); truth = json.load(open(os.path.join(root, 'truth.json'), encoding='utf-8')); sets = truth['sets']
        if a.max_sets: sets = sets[:a.max_sets]
        T = defaultdict(float); G = defaultdict(lambda: defaultdict(float)); t0 = time.time(); nq = 0
        for s in sets:
            sd = os.path.join(root, s['id']); g = np.asarray(Image.open(os.path.join(sd, 'gallery.png')).convert('RGB')); comps = split_gallery(g)
            # 예시 번호 부여: 정답 갤러리 상자와 가장 겹치는 덩어리
            idxmap = {}
            for ci, cb in enumerate(comps):
                best, bi = 0, -1
                for m in s['marks']:
                    v = iou(list(cb[:4]), m['gallery']['box'])
                    if v > best: best, bi = v, m['idx']
                if best > .3: idxmap[ci] = bi
            T['sets'] += 1; T['split_ok'] += int(len(comps) == s['M'] and len(set(idxmap.values())) == s['M'])
            refs = {idx: make_ref(g, comps[ci]).to(dev) for ci, idx in idxmap.items()}
            for im in s['images']:
                x, W, H = target_tensor(os.path.join(sd, im['file'])); x = x.to(dev); truths = [(m['idx'], m['box']) for m in im['marks']]; preds = []
                for idx, rf in refs.items():
                    with torch.no_grad(): p = torch.sigmoid(net(x, rf))[0, 0].cpu().numpy()
                    nq += 1
                    for b in boxes_from_mask(p, W, H, th=a.th, tile_frac=2.0):
                        if b[2] * b[3] <= MAX_AREA * W * H: preds.append((idx, list(b[:4])))
                tp, fp, fn = match(preds, truths)
                T['tp'] += tp; T['fp'] += fp; T['fn'] += fn; T['images'] += 1
                present = {i for i, _ in truths}; predset = {i for i, _ in preds}
                for idx in refs:
                    if idx in present: T['pairs_pos'] += 1; T['pairs_pos_hit'] += int(idx in predset)
                    else: T['pairs_neg'] += 1; T['pairs_neg_fp'] += int(idx in predset)
                if not truths: T['empty_imgs'] += 1; T['empty_fp'] += int(len(preds) > 0)
                for m in im['marks']:
                    kind = s['marks'][m['idx']]['kind']; al = m['alpha']
                    keys = ['kind=%s' % kind, 'alpha=%s' % ('<0.2' if al < .2 else '0.2-0.3' if al < .3 else '0.3-0.4' if al < .4 else '>=0.4'), 'scale=%s' % ('<0.85' if m['scale'] < .85 else '0.85-1.15' if m['scale'] <= 1.15 else '>1.15'), 'style=%s' % s['style']]
                    hit = match([p for p in preds if p[0] == m['idx']], [(m['idx'], m['box'])])[0]
                    for k in keys: G[k]['n'] += 1; G[k]['hit'] += hit
        el = time.time() - t0
        res = dict(recall=100 * T['tp'] / max(1, T['tp'] + T['fn']), precision=100 * T['tp'] / max(1, T['tp'] + T['fp']), absent_pair_fp=100 * T['pairs_neg_fp'] / max(1, T['pairs_neg']), present_pair_hit=100 * T['pairs_pos_hit'] / max(1, T['pairs_pos']),
                   empty_img_fp=100 * T['empty_fp'] / max(1, T['empty_imgs']), gallery_split_ok=100 * T['split_ok'] / max(1, T['sets']), tp=int(T['tp']), fp=int(T['fp']), fn=int(T['fn']), ms_per_query=1000 * el / max(1, nq))
        print('[%s %s th=%.2f] 상자 재현율 %.1f%% · 정밀도 %.1f%% · 없는 예시 오검출 %.1f%% · 마크 없는 이미지 오검출 %.1f%% · 예시 단위 있음 적중 %.1f%% · 갤러리 분리 %.0f%% · %.0fms/쿼리 · %.2fM' % (a.tag, tier, a.th, res['recall'], res['precision'], res['absent_pair_fp'], res['empty_img_fp'], res['present_pair_hit'], res['gallery_split_ok'], res['ms_per_query'], params / 1e6))
        for k in sorted(G): print('  %-18s 재현율 %5.1f%% (%d)' % (k, 100 * G[k]['hit'] / G[k]['n'], G[k]['n']))
        if a.out: open(a.out, 'a', encoding='utf-8').write(json.dumps(dict(tag=a.tag, ckpt=os.path.basename(a.ckpt), params=params, tier=tier, th=a.th, **res, groups={k: dict(recall=100 * v['hit'] / v['n'], n=int(v['n'])) for k, v in G.items()}), ensure_ascii=False) + '\n')


if __name__ == '__main__': main()
