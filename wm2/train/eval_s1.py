#!/usr/bin/env python3
"""1단계 평가(단색 여백의 예시 워터마크 찾기). 사용:
 python eval_s1.py --bench <refdata3 또는 fresh> --tiers core,val --method rule:3 | learned:<ckpt.pt> [--tag X] [--out rows_s1.jsonl] [--device dml]
정답: truth.json 의 image.example_marks(예시 마크 상자). 지표: 예시 재현율(IoU≥0.5), 정밀도, 오검출 구성(캡션 여백 / 그림 위 마크(2단계 대상) / 그 외=말풍선·머리·눈 등), 예시가 없는 이미지의 오검출률, 종류별 재현율."""
import os, sys, json, time, argparse
from collections import defaultdict
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))


def iou(a, b):
    ax, ay, aw, ah = a[:4]; bx, by, bw, bh = b[:4]
    ix, iy = max(0, min(ax + aw, bx + bw) - max(ax, bx)), max(0, min(ay + ah, by + bh) - max(ay, by)); i = ix * iy
    return i / max(1, aw * ah + bw * bh - i)


def overlap_frac(a, b):
    """a 상자가 b 와 겹치는 비율(a 면적 대비)"""
    ax, ay, aw, ah = a[:4]; bx, by, bw, bh = b[:4]
    ix, iy = max(0, min(ax + aw, bx + bw) - max(ax, bx)), max(0, min(ay + ah, by + bh) - max(ay, by)); return ix * iy / max(1, aw * ah)

MERGE = int(os.environ.get('S1_MERGE', '6'))   # 마크 조각을 한 상자로 묶는 팽창 반경(0.5배 픽셀). 여백 안의 마크는 서로 멀어서 크게 잡아도 된다
TILE = int(os.environ.get('S1_TILE', '0'))   # 0 이면 페이지 전체를 한 번에. 학습 조각(256)과 같은 크기로 잘라 넣어 GroupNorm 통계 차이를 없앤다.


def infer_prob(net, dev, a):
    """a: [1,3,h,w] float(16배수). 겹치는 타일(TILE, 보폭 TILE*3/4)로 돌려 확률을 평균한다."""
    import torch
    _, _, h, w = a.shape
    if TILE <= 0 or (h <= TILE and w <= TILE):
        with torch.no_grad(): return torch.sigmoid(net(torch.from_numpy(a).to(dev)))[0, 0].cpu().numpy()
    st = TILE * 3 // 4; acc = np.zeros((h, w), np.float32); cnt = np.zeros((h, w), np.float32)
    ys = list(range(0, max(1, h - TILE + 1), st)); xs = list(range(0, max(1, w - TILE + 1), st))
    if ys[-1] + TILE < h: ys.append(h - TILE)
    if xs[-1] + TILE < w: xs.append(w - TILE)
    for y in ys:
        for x in xs:
            y1, x1 = min(h, y + TILE), min(w, x + TILE)
            with torch.no_grad(): p = torch.sigmoid(net(torch.from_numpy(a[:, :, y:y1, x:x1].copy()).to(dev)))[0, 0].cpu().numpy()
            acc[y:y1, x:x1] += p; cnt[y:y1, x:x1] += 1
    return acc / np.maximum(cnt, 1)

def load_method(spec, device):
    kind, _, arg = spec.partition(':')
    if kind == 'rule':
        import s1_rule; v = int(arg or s1_rule.RULE_VERSION); return lambda img: s1_rule.detect(np.asarray(img), v)
    if kind == 'learned':
        import torch
        from model import UNet
        from train import get_device
        from infer_eval import boxes_from_mask
        dev = get_device(device); d = torch.load(arg, map_location='cpu', weights_only=True); net = UNet(tuple(d['ch'])); net.load_state_dict(d['sd']); net.to(dev).eval()
        def run(img):
            W, H = img.size; w, h = W // 2, H // 2; a = np.asarray(img.resize((w, h), Image.BOX), np.float32) / 255; a = np.pad(a, ((0, (-h) % 16), (0, (-w) % 16), (0, 0)), mode='reflect').transpose(2, 0, 1)[None].copy()
            p = infer_prob(net, dev, a)
            out = []
            for b in boxes_from_mask(p, W, H, th=.5, tile_frac=2.0, merge=MERGE):
                if b[2] * b[3] <= .2 * W * H and b[2] >= .06 * W and b[3] >= 18: out.append((b[0], b[1], b[2], b[3], b[5]))   # 크기 사양(너무 작은 것·배경 전체 제외)
            return out
        return run
    if kind == 'hybrid':   # c: 규칙이 후보를 내고(rule:N), 신경망 마스크가 상자 안을 충분히 '예시 마크'로 보면 채택
        import torch
        import s1_rule
        from model import UNet
        from train import get_device
        ver, _, ck = arg.partition(':'); dev = get_device(device); d = torch.load(ck, map_location='cpu', weights_only=True); net = UNet(tuple(d['ch'])); net.load_state_dict(d['sd']); net.to(dev).eval()
        def run(img):
            W, H = img.size; w, h = W // 2, H // 2; a = np.asarray(img.resize((w, h), Image.BOX), np.float32) / 255; a = np.pad(a, ((0, (-h) % 16), (0, (-w) % 16), (0, 0)), mode='reflect').transpose(2, 0, 1)[None].copy()
            p = infer_prob(net, dev, a)
            out = []
            for b in s1_rule.detect(np.asarray(img), int(ver)):
                x0, y0, x1, y1 = b[0] // 2, b[1] // 2, (b[0] + b[2]) // 2 + 1, (b[1] + b[3]) // 2 + 1; reg = p[y0:y1, x0:x1]
                if reg.size and (reg > .5).mean() >= .25: out.append((b[0], b[1], b[2], b[3], float(reg.mean())))
            return out
        return run
    raise SystemExit('method: rule:N | learned:<ckpt> | hybrid:<ruleN>:<ckpt>')


def evaluate(run, root, max_sets=0):
    truth = json.load(open(os.path.join(root, 'truth.json'), encoding='utf-8')); sets = truth['sets'][:max_sets] if max_sets else truth['sets']
    T = defaultdict(float); K = defaultdict(lambda: [0, 0]); t0 = time.time(); fps_info = []
    for s in sets:
        for im in s['images']:
            pil = Image.open(os.path.join(root, s['id'], im['file'])).convert('RGB'); dets = run(pil); ex = im['example_marks']; used = set(); T['images'] += 1
            for e in ex:
                best, bj = 0, -1
                for j, b in enumerate(dets):
                    if j in used: continue
                    v = iou(b, e['box'])
                    if v > best: best, bj = v, j
                hit = best >= .5; T['ex'] += 1; T['tp'] += hit
                if hit: used.add(bj)
                k = s['marks'][e['idx']]['kind']; K[k][0] += hit; K[k][1] += 1
            for j, b in enumerate(dets):
                if j in used: continue
                T['fp'] += 1
                if any(overlap_frac(b, dd['box']) > .3 or overlap_frac(dd['box'], b) > .3 for dd in im['decoy_margins']): T['fp_caption'] += 1
                elif any(overlap_frac(b, m['box']) > .3 for m in im['marks']): T['fp_art_mark'] += 1
                elif ex and any(overlap_frac(b, e['box']) > .3 for e in ex): T['fp_dup_or_part'] += 1
                else: T['fp_other'] += 1
            if not ex:
                T['noex_images'] += 1; T['noex_fp_images'] += int(len(dets) > 0)
    res = dict(recall=100 * T['tp'] / max(1, T['ex']), precision=100 * T['tp'] / max(1, T['tp'] + T['fp']), tp=int(T['tp']), ex=int(T['ex']), fp=int(T['fp']),
               fp_caption=int(T['fp_caption']), fp_art_mark=int(T['fp_art_mark']), fp_dup_or_part=int(T['fp_dup_or_part']), fp_other=int(T['fp_other']),
               noexample_image_fp=100 * T['noex_fp_images'] / max(1, T['noex_images']), images=int(T['images']), sec=time.time() - t0)
    return res, {k: 100 * v[0] / max(1, v[1]) for k, v in K.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--bench', required=True); ap.add_argument('--tiers', default='core,val'); ap.add_argument('--method', required=True); ap.add_argument('--tag'); ap.add_argument('--out'); ap.add_argument('--device', default='dml'); ap.add_argument('--max-sets', type=int, default=0)
    a = ap.parse_args(); run = load_method(a.method, a.device); tag = a.tag or a.method
    for tier in a.tiers.split(','):
        res, kinds = evaluate(run, os.path.join(a.bench, tier), a.max_sets)
        print('[%s %s] 예시 재현율 %.1f%% (%d/%d) · 정밀도 %.1f%% · 오검출 %d (캡션 여백 %d · 그림 위 마크 %d · 중복/일부 %d · 그 외(말풍선·머리·눈 등) %d) · 예시 없는 이미지의 오검출 %.1f%% · %.0fs' % (
            tag, tier, res['recall'], res['tp'], res['ex'], res['precision'], res['fp'], res['fp_caption'], res['fp_art_mark'], res['fp_dup_or_part'], res['fp_other'], res['noexample_image_fp'], res['sec']))
        print('  종류별 재현율: ' + ' · '.join('%s %.0f' % (k, v) for k, v in sorted(kinds.items())))
        if a.out: open(a.out, 'a', encoding='utf-8').write(json.dumps(dict(tag=tag, tier=tier, bench=a.bench, **res, kinds=kinds), ensure_ascii=False) + '\n')


if __name__ == '__main__': main()


