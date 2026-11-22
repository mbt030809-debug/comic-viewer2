#!/usr/bin/env python3
"""제거 모델 체크포인트 → ONNX(opset 17). 입력 image[1,3,h,w] RGB 0..1, mask[1,1,h,w] 0..1 (h,w 는 16의 배수) → 출력 result[1,3,h,w] 0..1.
사용: python export_rm_onnx.py <rm.pt> <out.onnx>"""
import os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from rm_model import RmUNet

ck, out = sys.argv[1:3]
d = torch.load(ck, map_location='cpu', weights_only=True); net = RmUNet(tuple(d['ch'])); net.load_state_dict(d['sd']); net.eval()
x = torch.rand(1, 3, 256, 192); m = (torch.rand(1, 1, 256, 192) > .8).float()
torch.onnx.export(net, (x, m), out, opset_version=17, input_names=['image', 'mask'], output_names=['result'],
                  dynamic_axes={'image': {2: 'h', 3: 'w'}, 'mask': {2: 'h', 3: 'w'}, 'result': {2: 'h', 3: 'w'}}, dynamo=False)
print('onnx %.2fMB · 파라미터 %d' % (os.path.getsize(out) / 1e6, sum(p.numel() for p in net.parameters())))
import onnxruntime as ort
s = ort.InferenceSession(out, providers=['CPUExecutionProvider'])
x = torch.rand(1, 3, 512, 368); m = (torch.rand(1, 1, 512, 368) > .8).float()
with torch.no_grad(): t = net(x, m).numpy()
o = s.run(None, {'image': x.numpy(), 'mask': m.numpy()})[0]; print('최대 차이', float(np.abs(t - o).max()))
t0 = time.time(); [s.run(None, {'image': x.numpy(), 'mask': m.numpy()}) for _ in range(3)]; print('ORT CPU(이 PC) %.0f ms/장 (512x368)' % ((time.time() - t0) * 333))
