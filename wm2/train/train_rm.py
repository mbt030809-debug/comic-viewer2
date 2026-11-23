#!/usr/bin/env python3
"""워터마크 제거 학습: python train_rm.py --pool <풀> --steps 20000 --bs 12 --device dml --ch 16,32,64,96,128 --out <ckpt>
손실 = 마스크 영역 L1(가중 10) + 영역 기울기 L1(선명도). 검증: 합성 96개의 '영역 오차 감소율'."""
import os, sys, time, argparse
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from rm_model import RmUNet
from synth_rm import SynthRm


def grad(a): return a[..., :, 1:] - a[..., :, :-1], a[..., 1:, :] - a[..., :-1, :]


def loss_fn(out, clean, m_in, m_gt):
    w = 1 + 9 * torch.clamp(m_in + m_gt, 0, 1)
    l1 = ((out - clean).abs() * w).mean()
    gx, gy = grad(out); cx, cy = grad(clean)
    lg = (gx - cx).abs().mean() + (gy - cy).abs().mean()
    return l1 + .5 * lg


def reduction(out, comp, clean, m_gt):
    """워터마크 영역(정답 마스크)에서 |합성-원본| 대비 |결과-원본| 감소율(%)과 영역 PSNR 전/후"""
    mg = m_gt.expand_as(out); n = mg.sum().clamp(min=1)
    e0 = ((comp - clean).abs() * mg).sum() / n; e1 = ((out - clean).abs() * mg).sum() / n
    mse = lambda a: (((a - clean) ** 2) * mg).sum() / n
    psnr = lambda a: (10 * torch.log10(1 / mse(a).clamp(min=1e-8))).item()
    return (100 * (1 - e1 / e0.clamp(min=1e-8))).item(), psnr(comp), psnr(out)


def get_device(name):
    if name == 'dml':
        import torch_directml; return torch_directml.device()
    return torch.device(name)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--pool', required=True); ap.add_argument('--steps', type=int, default=20000); ap.add_argument('--bs', type=int, default=12)
    ap.add_argument('--size', type=int, default=192); ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--out', required=True); ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--resume'); ap.add_argument('--ch', default='16,32,64,96,128'); ap.add_argument('--device', default='dml'); ap.add_argument('--save-every', type=int, default=2000); ap.add_argument('--seed', type=int, default=21); ap.add_argument('--synth', default='v1', choices=['v1', 'v2'])
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); torch.manual_seed(0); dev = get_device(a.device)
    ch = tuple(int(c) for c in a.ch.split(',')); net = RmUNet(ch)
    if a.resume: net.load_state_dict(torch.load(a.resume, map_location='cpu', weights_only=True)['sd'])
    net.to(dev); print('파라미터 %.3fM · 장치 %s' % (sum(p.numel() for p in net.parameters()) / 1e6, dev), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=.1)
    if a.synth == 'v2':
        from synth_rm2 import SynthRm2 as SynthRm; val_cls = SynthRm   # 캐릭터·크기 범위·만화 선화 배경
    ds = SynthRm(a.pool, n=a.steps * a.bs, seed=a.seed, size=a.size)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, num_workers=a.workers, shuffle=False, persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers > 0 else None)
    val = SynthRm(a.pool, n=96, seed=777, size=a.size); vb = [val[i] for i in range(len(val))]
    vx, vm, vc, vg = [torch.stack([v[k] for v in vb]) for k in range(4)]
    t0 = time.time(); ema = None; net.train()
    for step, (x, mi, c, mg) in enumerate(dl, 1):
        x, mi, c, mg = x.to(dev), mi.to(dev), c.to(dev), mg.to(dev)
        out = net(x, mi); l = loss_fn(out, c, mi, mg); opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step(); sched.step()
        ema = l.item() if ema is None else .98 * ema + .02 * l.item()
        if step % 50 == 0: print('step %d loss %.4f %.2fs/step 경과 %.0fs' % (step, ema, (time.time() - t0) / step, time.time() - t0), flush=True)
        if step % a.save_every == 0 or step == a.steps:
            net.eval()
            with torch.no_grad(): vo = torch.cat([net(vx[i:i + 32].to(dev), vm[i:i + 32].to(dev)).cpu() for i in range(0, len(vx), 32)])
            red, p0, p1 = reduction(vo, vx, vc, vg); print('  [검증] 영역 오차 감소 %.1f%% · 영역 PSNR %.1f→%.1f dB' % (red, p0, p1), flush=True); net.train()
            torch.save({'ch': list(ch), 'sd': {k: v.detach().cpu() for k, v in net.state_dict().items()}}, os.path.join(a.out, 'rm_%d.pt' % step))
