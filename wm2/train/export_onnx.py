#!/usr/bin/env python3
"""체크포인트 → ONNX(opset 17, 입력 image[1,3,h,w] RGB 0..1, 출력 logits[1,1,h,w]; h,w 는 16의 배수). 검증용으로 onnxruntime 과 로짓 차이를 출력한다.
사용: python3 export_onnx.py <ckpt.pt> <out.onnx> [검증용 이미지]"""
import os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from model import UNet

ck, out = sys.argv[1:3]; img_path = sys.argv[3] if len(sys.argv) > 3 else None
d = torch.load(ck, map_location='cpu', weights_only=True); net = UNet(tuple(d['ch'])); net.load_state_dict(d['sd']); net.eval()
torch.onnx.export(net, torch.rand(1, 3, 256, 192), out, opset_version=17, input_names=['image'], output_names=['logits'],
                  dynamic_axes={'image': {2: 'h', 3: 'w'}, 'logits': {2: 'h', 3: 'w'}}, dynamo=False)
print('onnx %.2fMB · 파라미터 %d' % (os.path.getsize(out) / 1e6, sum(p.numel() for p in net.parameters())))
import onnxruntime as ort
s = ort.InferenceSession(out, providers=['CPUExecutionProvider'])
if img_path:
    from PIL import Image
    img = Image.open(img_path).convert('RGB'); w, h = img.width // 2, img.height // 2; a = np.asarray(img.resize((w, h), Image.BOX), np.float32) / 255
    a = np.pad(a, ((0, (-h) % 16), (0, (-w) % 16), (0, 0)), mode='reflect').transpose(2, 0, 1)[None].copy()
    with torch.no_grad(): t = net(torch.from_numpy(a)).numpy()
    o = s.run(None, {'image': a})[0]; print('입력', a.shape, '로짓 최대 차이', float(np.abs(t - o).max()))
    t0 = time.time(); [s.run(None, {'image': a}) for _ in range(5)]; print('ORT CPU(이 PC) %.0f ms/장' % ((time.time() - t0) * 200))
