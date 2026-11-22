#!/usr/bin/env python3
"""참조형 인식 모델 → ONNX(opset 17). 입력 image[1,3,h,w] (0.5배 축소 타깃, RGB 0..1, h,w 16의 배수), ref[1,3,112,192] (예시 마크, 0.5배 축소, 배경색 캔버스) → 출력 logits[1,1,h,w].
사용: python export_ref_onnx.py <ref.pt> <out.onnx>"""
import os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from ref_model import RefUNet, RH, RW

ck, out = sys.argv[1:3]
d = torch.load(ck, map_location='cpu', weights_only=True); net = RefUNet(tuple(d['ch'])); net.load_state_dict(d['sd']); net.eval()
x = torch.rand(1, 3, 256, 192); r = torch.rand(1, 3, RH, RW)
torch.onnx.export(net, (x, r), out, opset_version=17, input_names=['image', 'ref'], output_names=['logits'],
                  dynamic_axes={'image': {2: 'h', 3: 'w'}, 'logits': {2: 'h', 3: 'w'}}, dynamo=False)
print('onnx %.2fMB · 파라미터 %d' % (os.path.getsize(out) / 1e6, sum(p.numel() for p in net.parameters())))
import onnxruntime as ort
s = ort.InferenceSession(out, providers=['CPUExecutionProvider'])
x = torch.rand(1, 3, 512, 368); r = torch.rand(1, 3, RH, RW)
with torch.no_grad(): t = net(x, r).numpy()
o = s.run(None, {'image': x.numpy(), 'ref': r.numpy()})[0]; print('입력', tuple(x.shape), '로짓 최대 차이', float(np.abs(t - o).max()))
x2 = torch.rand(1, 3, 800, 368)
t0 = time.time(); [s.run(None, {'image': x2.numpy(), 'ref': r.numpy()}) for _ in range(3)]; print('ORT CPU(이 PC) %.0f ms/쿼리 (800x368)' % ((time.time() - t0) * 333))
