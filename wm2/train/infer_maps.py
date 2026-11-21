#!/usr/bin/env python3
"""확률 지도 저장: python3 infer_maps.py <ckpt> <tier폴더> <출력폴더> [사이트필터]  (사이트마다 maps.npz: 파일명 → float16 확률 지도(0.5배))"""
import os, sys, time
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from model import UNet
from infer_eval import predict_mask
if __name__ == '__main__':
    ck, root, outd = sys.argv[1:4]; filt = sys.argv[4] if len(sys.argv) > 4 else ''
    torch.set_num_threads(3); d = torch.load(ck, map_location='cpu', weights_only=True); net = UNet(tuple(d['ch'])); net.load_state_dict(d['sd']); net.eval()
    os.makedirs(outd, exist_ok=True); t0 = time.time()
    for s in sorted(os.listdir(root)):
        dd = os.path.join(root, s)
        if not os.path.isdir(dd) or filt not in s: continue
        maps = {}
        for f in sorted(x for x in os.listdir(dd) if x.startswith('img_')):
            im = Image.open(os.path.join(dd, f)).convert('RGB'); maps[f] = predict_mask(net, im).astype(np.float16); maps[f + '.size'] = np.array([im.width, im.height])
        np.savez_compressed(os.path.join(outd, s + '.npz'), **maps)
    print('완료 %.0fs' % (time.time() - t0))
