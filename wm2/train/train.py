#!/usr/bin/env python3
"""학습: python3 train.py --pool ~/wm2-train --steps 3000 --bs 12 --out ~/wm2-train/ckpt  (폰 CPU)"""
import os, sys, time, argparse, math
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from model import UNet
from synth import Synth

def loss_fn(lo, m):
    bce = F.binary_cross_entropy_with_logits(lo, m, pos_weight=torch.tensor(2.0))
    p = torch.sigmoid(lo); inter = (p * m).sum((1, 2, 3)); dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + m.sum((1, 2, 3)) + 1)
    return bce + dice.mean()

def pix_iou(lo, m, th=.5):
    p = (torch.sigmoid(lo) > th).float(); i = (p * m).sum().item(); u = ((p + m) > 0).float().sum().item(); return i / max(1, u)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--pool', default=os.path.expanduser('~/wm2-train')); ap.add_argument('--steps', type=int, default=3000); ap.add_argument('--bs', type=int, default=12)
    ap.add_argument('--size', type=int, default=192); ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--out', default=os.path.expanduser('~/wm2-train/ckpt')); ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--resume'); ap.add_argument('--ch', default='8,16,32,48,64'); ap.add_argument('--threads', type=int, default=3)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); torch.set_num_threads(a.threads); torch.manual_seed(0)
    ch = tuple(int(c) for c in a.ch.split(',')); net = UNet(ch)
    if a.resume: net.load_state_dict(torch.load(a.resume, map_location='cpu', weights_only=True)['sd'])
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=.1)
    ds = Synth(a.pool, n=a.steps * a.bs, seed=1, size=a.size); dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, num_workers=a.workers, shuffle=False)
    val = Synth(a.pool, n=96, seed=777, size=a.size); vb = [val[i] for i in range(len(val))]
    vx = torch.stack([v[0] for v in vb]); vm = torch.stack([v[1] for v in vb])
    t0 = time.time(); ema = None; net.train()
    for step, (x, m) in enumerate(dl, 1):
        lo = net(x); l = loss_fn(lo, m); opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step(); sched.step()
        ema = l.item() if ema is None else .98 * ema + .02 * l.item()
        if step % 50 == 0:
            print('step %d loss %.3f iou %.3f %.1fs/step 경과 %.0fs' % (step, ema, pix_iou(lo, m), (time.time() - t0) / step, time.time() - t0), flush=True)
        if step % 250 == 0 or step == a.steps:
            net.eval()
            with torch.no_grad(): vl = net(vx)
            has = vm.sum((1, 2, 3)) > 0; fp = ((torch.sigmoid(vl[~has]) > .5).float().sum((1, 2, 3)) > 20).float().mean().item() if (~has).any() else 0
            print('  [검증] 마스크 IoU %.3f · 빈 조각 오검출율 %.2f' % (pix_iou(vl[has], vm[has]), fp), flush=True); net.train()
            torch.save({'ch': list(ch), 'sd': net.state_dict()}, os.path.join(a.out, 'unet_%d.pt' % step))
