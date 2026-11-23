#!/usr/bin/env python3
"""참조형 인식 학습: python train_ref.py --pool <풀> --steps 20000 --bs 12 --device dml --out <ckpt>
검증(합성 96개): 양성 픽셀 IoU, 음성(마스크 없음) 조각 오검출율(같은 종류 마크 포함)."""
import os, sys, time, argparse
import torch
sys.path.insert(0, os.path.dirname(__file__))
from ref_model import RefUNet
from synth_ref import SynthRef
from train import loss_fn, pix_iou, get_device

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--pool', required=True); ap.add_argument('--steps', type=int, default=20000); ap.add_argument('--bs', type=int, default=12)
    ap.add_argument('--size', type=int, default=192); ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--out', required=True); ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--resume'); ap.add_argument('--ch', default='16,32,64,96,128'); ap.add_argument('--device', default='dml'); ap.add_argument('--save-every', type=int, default=2000); ap.add_argument('--seed', type=int, default=31); ap.add_argument('--synth', default='v1', choices=['v1', 'v2', 'v3'])
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); torch.manual_seed(0); dev = get_device(a.device)
    ch = tuple(int(c) for c in a.ch.split(',')); net = RefUNet(ch)
    if a.resume: net.load_state_dict(torch.load(a.resume, map_location='cpu', weights_only=True)['sd'])
    net.to(dev); print('파라미터 %.3fM · 장치 %s' % (sum(p.numel() for p in net.parameters()) / 1e6, dev), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=.1)
    if a.synth in ('v2', 'v3'):
        from synth_ref2 import SynthRef2 as SynthRef   # 캐릭터·크기 범위·만화 선화 하드 네거티브
        if a.synth == 'v3': SynthRef.P_OPAQUE = .6     # v3: 불투명(알파 1.0) 60% + 반투명 40%
    ds = SynthRef(a.pool, n=a.steps * a.bs, seed=a.seed, size=a.size)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, num_workers=a.workers, shuffle=False, persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers > 0 else None)
    val = SynthRef(a.pool, n=96, seed=777, size=a.size); vb = [val[i] for i in range(len(val))]
    vx, vr, vm = [torch.stack([v[k] for v in vb]) for k in range(3)]
    t0 = time.time(); ema = None; net.train()
    for step, (x, rf, m) in enumerate(dl, 1):
        x, rf, m = x.to(dev), rf.to(dev), m.to(dev)
        lo = net(x, rf); l = loss_fn(lo, m); opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step(); sched.step()
        ema = l.item() if ema is None else .98 * ema + .02 * l.item()
        if step % 50 == 0: print('step %d loss %.3f iou %.3f %.2fs/step 경과 %.0fs' % (step, ema, pix_iou(lo, m), (time.time() - t0) / step, time.time() - t0), flush=True)
        if step % a.save_every == 0 or step == a.steps:
            net.eval()
            with torch.no_grad(): vl = torch.cat([net(vx[i:i + 32].to(dev), vr[i:i + 32].to(dev)).cpu() for i in range(0, len(vx), 32)])
            has = vm.sum((1, 2, 3)) > 0; fp = ((torch.sigmoid(vl[~has]) > .5).float().sum((1, 2, 3)) > 20).float().mean().item() if (~has).any() else 0
            print('  [검증] 양성 마스크 IoU %.3f · 음성 조각 오검출율 %.2f (양성 %d / 음성 %d)' % (pix_iou(vl[has], vm[has]), fp, int(has.sum()), int((~has).sum())), flush=True); net.train()
            torch.save({'ch': list(ch), 'sd': {k: v.detach().cpu() for k, v in net.state_dict().items()}}, os.path.join(a.out, 'ref_%d.pt' % step))
