#!/usr/bin/env python3
"""1단계 학습형(단색 여백의 예시 워터마크 찾기): python train_s1.py --pool <풀> --steps 10000 --bs 12 --size 256 --device dml --out <ckpt>
모델은 기존 U-Net(model.py, 0.5배 이미지 입력 → 마스크 로짓). 검증: 양성 마스크 IoU, 마스크 없는 조각 오검출율."""
import os, sys, time, argparse
import torch
sys.path.insert(0, os.path.dirname(__file__))
from model import UNet
from synth_s1 import SynthS1
from train import loss_fn, pix_iou, get_device

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--pool', required=True); ap.add_argument('--steps', type=int, default=10000); ap.add_argument('--bs', type=int, default=12); ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--out', required=True); ap.add_argument('--workers', type=int, default=4); ap.add_argument('--ch', default='16,32,64,96,128'); ap.add_argument('--device', default='dml')
    ap.add_argument('--save-every', type=int, default=1000); ap.add_argument('--seed', type=int, default=61); ap.add_argument('--resume')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); torch.manual_seed(0); dev = get_device(a.device)
    ch = tuple(int(c) for c in a.ch.split(',')); net = UNet(ch)
    if a.resume: net.load_state_dict(torch.load(a.resume, map_location='cpu', weights_only=True)['sd'])
    net.to(dev); print('파라미터 %.3fM · 장치 %s' % (sum(p.numel() for p in net.parameters()) / 1e6, dev), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=.1)
    ds = SynthS1(a.pool, n=a.steps * a.bs, seed=a.seed, size=a.size)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, num_workers=a.workers, shuffle=False, persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers > 0 else None)
    val = SynthS1(a.pool, n=96, seed=777, size=a.size); vb = [val[i] for i in range(len(val))]; vx = torch.stack([v[0] for v in vb]); vm = torch.stack([v[1] for v in vb])
    t0 = time.time(); ema = None; net.train()
    for step, (x, m) in enumerate(dl, 1):
        x, m = x.to(dev), m.to(dev); lo = net(x); l = loss_fn(lo, m); opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step(); sched.step()
        ema = l.item() if ema is None else .98 * ema + .02 * l.item()
        if step % 50 == 0: print('step %d loss %.3f iou %.3f %.2fs/step 경과 %.0fs' % (step, ema, pix_iou(lo, m), (time.time() - t0) / step, time.time() - t0), flush=True)
        if step % a.save_every == 0 or step == a.steps:
            net.eval()
            with torch.no_grad(): vl = torch.cat([net(vx[i:i + 32].to(dev)).cpu() for i in range(0, len(vx), 32)])
            has = vm.sum((1, 2, 3)) > 0; fp = ((torch.sigmoid(vl[~has]) > .5).float().sum((1, 2, 3)) > 20).float().mean().item() if (~has).any() else 0
            print('  [검증] 양성 마스크 IoU %.3f · 음성 조각 오검출율 %.2f (양성 %d / 음성 %d)' % (pix_iou(vl[has], vm[has]), fp, int(has.sum()), int((~has).sum())), flush=True); net.train()
            torch.save({'ch': list(ch), 'sd': {k: v.detach().cpu() for k, v in net.state_dict().items()}}, os.path.join(a.out, 's1_%d.pt' % step))
