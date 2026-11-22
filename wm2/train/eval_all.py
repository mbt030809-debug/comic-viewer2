#!/usr/bin/env python3
"""여러 티어·여러 임계값을 한 번에 평가(확률 지도는 이미지당 한 번만 계산).
사용: python3 eval_all.py <ckpt.pt> --bench ~/wm2-bench --tiers core,wild,val --ths 0.5,0.7,0.9 --tag base --out results.jsonl [--device dml|cpu]
출력: 티어×임계값별 인식률·음성 이미지 오인식률(ambiguous 는 따로)·정밀도·추론 시간, tsv 는 --tsvdir 에 저장."""
import os, sys, json, time, argparse
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))
from model import UNet
from infer_eval import boxes_from_mask
import eval_bench as E
from collections import defaultdict


def load_net(ck, dev):
    d = torch.load(ck, map_location='cpu', weights_only=True); net = UNet(tuple(d['ch'])); net.load_state_dict(d['sd']); return net.to(dev).eval(), d


def prob_map(net, img, dev, sc=.5):
    w, h = int(img.width * sc), int(img.height * sc); a = np.asarray(img.resize((w, h), Image.BOX), np.float32) / 255
    ph, pw = (-h) % 16, (-w) % 16; a = np.pad(a, ((0, ph), (0, pw), (0, 0)), mode='reflect')
    with torch.no_grad(): p = torch.sigmoid(net(torch.from_numpy(a.transpose(2, 0, 1)[None].copy()).to(dev)))[0, 0].cpu().numpy()
    return p[:h, :w]


def totals(R):
    t = defaultdict(float)
    for r in R.values():
        for k in ('truth', 'tp', 'extra', 'neg_images', 'neg_fp', 'amb_images', 'amb_fp'): t[k] += r[k]
    return dict(recall=100 * t['tp'] / max(1, t['truth']), neg_fp=100 * t['neg_fp'] / max(1, t['neg_images']), amb_fp=100 * t['amb_fp'] / max(1, t['amb_images']),
                precision=100 * t['tp'] / max(1, t['tp'] + t['extra'] + t['neg_fp']), extra=int(t['extra']), tp=int(t['tp']), truth=int(t['truth']),
                neg_n=int(t['neg_images']), neg_fp_n=int(t['neg_fp']))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('ckpt'); ap.add_argument('--bench', required=True); ap.add_argument('--tiers', default='core')
    ap.add_argument('--ths', default='0.5'); ap.add_argument('--tag', default='run'); ap.add_argument('--out'); ap.add_argument('--tsvdir'); ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=6); ap.add_argument('--bytype', action='store_true')
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    if a.device == 'dml':
        import torch_directml; dev = torch_directml.device()
    else: dev = torch.device(a.device)
    net, d = load_net(a.ckpt, dev); npar = sum(p.numel() for p in net.parameters()); ths = [float(x) for x in a.ths.split(',')]
    for tier in a.tiers.split(','):
        root = os.path.join(a.bench, tier); truth = json.load(open(os.path.join(root, 'truth.json'), encoding='utf-8'))
        maps = {}; t0 = time.time(); n = 0
        for s in truth['sites']:
            for im in s['images']:
                img = Image.open(os.path.join(root, s['id'], im['file'])).convert('RGB'); maps[(s['id'], im['file'])] = (prob_map(net, img, dev), img.width, img.height); n += 1
        ms = 1000 * (time.time() - t0) / max(1, n)
        for th in ths:
            P = defaultdict(list); seen = set(); lines = []
            for key, (p, W, H) in maps.items():
                seen.add(key); bs = boxes_from_mask(p, W, H, th)
                if not bs: lines.append('%s\t%s\tNONE' % key)
                for b in bs: P[key].append(b); lines.append('%s\t%s\t%d\t%d\t%d\t%d\t%s\t%.3f' % (*key, *b))
            R, fails = E.evaluate(truth, P, seen); tt = totals(R)
            row = dict(tag=a.tag, ckpt=os.path.basename(a.ckpt), params=npar, ch=d['ch'], tier=tier, th=th, ms_per_img=round(ms, 1), device=str(a.device), **{k: round(v, 1) if isinstance(v, float) else v for k, v in tt.items()})
            print('[%s %s th=%.2f] 인식률 %.1f%% · 음성 오인식 %.1f%% (%d/%d) · ambiguous %.1f%% · 정밀도 %.1f%% · %.0fms/장 · %.2fM' % (a.tag, tier, th, tt['recall'], tt['neg_fp'], tt['neg_fp_n'], tt['neg_n'], tt['amb_fp'], tt['precision'], ms, npar / 1e6), flush=True)
            if a.out: open(a.out, 'a', encoding='utf-8').write(json.dumps(row, ensure_ascii=False) + '\n')
            if a.tsvdir:
                os.makedirs(a.tsvdir, exist_ok=True); tag = '%s_%s_th%02d' % (a.tag, tier, int(th * 100))
                open(os.path.join(a.tsvdir, tag + '.tsv'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
                open(os.path.join(a.tsvdir, tag + '_fail.txt'), 'w', encoding='utf-8').write('\n'.join('%s\t%s\t%s' % f for f in sorted(fails, key=lambda f: (f[1], f[0]))))
            if a.bytype and th == ths[0]: E.report(R)
