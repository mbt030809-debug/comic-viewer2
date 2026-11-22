#!/usr/bin/env python3
"""학습: python3 train.py --pool ~/wm2-train --steps 3000 --bs 12 --out ~/wm2-train/ckpt  (폰 CPU)
PC(AMD GPU/Windows): --device dml (torch-directml). 진단: --overfit 200 (샘플 200개 반복 학습)"""
import os, sys, time, argparse, math
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from model import UNet
from synth import Synth

def loss_fn(lo, m, pw=2.0):
    bce = (pw * m * F.softplus(-lo) + (1 - m) * F.softplus(lo)).mean()   # = binary_cross_entropy_with_logits(pos_weight=pw), DML 에서 CPU 대체 연산을 피함
    p = torch.sigmoid(lo); inter = (p * m).sum((1, 2, 3)); dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + m.sum((1, 2, 3)) + 1)
    return bce + dice.mean()

def pix_iou(lo, m, th=.5):
    p = (torch.sigmoid(lo) > th).float(); i = (p * m).sum().item(); u = ((p + m) > 0).float().sum().item(); return i / max(1, u)

class Rep(torch.utils.data.Dataset):
    """앞 k개 샘플만 반복(용량 진단용)"""
    def __init__(self, ds, k, n): self.ds, self.k, self.n = ds, k, n
    def __len__(self): return self.n
    def __getitem__(self, i): return self.ds[i % self.k]

def get_device(name):
    if name == 'dml':
        import torch_directml; return torch_directml.device()
    return torch.device(name)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--pool', default=os.path.expanduser('~/wm2-train')); ap.add_argument('--steps', type=int, default=3000); ap.add_argument('--bs', type=int, default=12)
    ap.add_argument('--size', type=int, default=192); ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--out', default=os.path.expanduser('~/wm2-train/ckpt')); ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--resume'); ap.add_argument('--ch', default='8,16,32,48,64'); ap.add_argument('--threads', type=int, default=3)
    ap.add_argument('--device', default='cpu'); ap.add_argument('--overfit', type=int, default=0); ap.add_argument('--save-every', type=int, default=250); ap.add_argument('--seed', type=int, default=1); ap.add_argument('--synth', default='synth', choices=['synth', 'synth2'])
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); torch.set_num_threads(a.threads); torch.manual_seed(0)
    dev = get_device(a.device)
    ch = tuple(int(c) for c in a.ch.split(',')); net = UNet(ch)
    if a.resume: net.load_state_dict(torch.load(a.resume, map_location='cpu', weights_only=True)['sd'])
    net.to(dev); print('파라미터 %.3fM · 장치 %s' % (sum(p.numel() for p in net.parameters()) / 1e6, dev), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=.1)
    if a.synth == 'synth2':
        from synth2 import Synth2 as Synth   # 하드 네거티브(말풍선·글자·직선) + 빈 조각 35%
    ds = Synth(a.pool, n=a.steps * a.bs, seed=a.seed, size=a.size)
    if a.overfit: ds = Rep(ds, a.overfit, a.steps * a.bs)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, num_workers=a.workers, shuffle=False, persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers > 0 else None)
    val = Synth(a.pool, n=96, seed=777, size=a.size); vb = [val[i] for i in range(len(val))]
    vx = torch.stack([v[0] for v in vb]); vm = torch.stack([v[1] for v in vb])
    t0 = time.time(); ema = None; net.train()
    for step, (x, m) in enumerate(dl, 1):
        x, m = x.to(dev), m.to(dev)
        lo = net(x); l = loss_fn(lo, m); opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step(); sched.step()
        ema = l.item() if ema is None else .98 * ema + .02 * l.item()
        if step % 50 == 0:
            print('step %d loss %.3f iou %.3f %.2fs/step 경과 %.0fs' % (step, ema, pix_iou(lo, m), (time.time() - t0) / step, time.time() - t0), flush=True)
        if step % a.save_every == 0 or step == a.steps:
            net.eval()
            with torch.no_grad(): vl = torch.cat([net(vx[i:i + 32].to(dev)).cpu() for i in range(0, len(vx), 32)])
            has = vm.sum((1, 2, 3)) > 0; fp = ((torch.sigmoid(vl[~has]) > .5).float().sum((1, 2, 3)) > 20).float().mean().item() if (~has).any() else 0
            print('  [검증] 마스크 IoU %.3f · 빈 조각 오검출율 %.2f' % (pix_iou(vl[has], vm[has]), fp), flush=True); net.train()
            torch.save({'ch': list(ch), 'sd': {k: v.detach().cpu() for k, v in net.state_dict().items()}}, os.path.join(a.out, 'unet_%d.pt' % step))
