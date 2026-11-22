#!/usr/bin/env python3
"""ONNX 모델로 벤치 추론: python3 infer_onnx.py <model.onnx> <tier폴더> <출력.tsv> [사이트필터] [--th 0.5]
(앱과 같은 전처리: 0.5배 BOX 축소 → 16배수 reflect 패딩 → 로짓 → sigmoid → 상자)"""
import os, sys, time
import numpy as np
import onnxruntime as ort
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from infer_eval import boxes_from_mask

def predict(sess, img, sc=.5):
    w, h = int(img.width * sc), int(img.height * sc); a = np.asarray(img.resize((w, h), Image.BOX), np.float32) / 255
    ph, pw = (-h) % 16, (-w) % 16; a = np.pad(a, ((0, ph), (0, pw), (0, 0)), mode='reflect')
    lo = sess.run(None, {'image': a.transpose(2, 0, 1)[None].copy()})[0][0, 0]
    return (1 / (1 + np.exp(-lo)))[:h, :w]

if __name__ == '__main__':
    model, root, outp = sys.argv[1:4]; filt = sys.argv[4] if len(sys.argv) > 4 and not sys.argv[4].startswith('--') else ''
    th = float(sys.argv[sys.argv.index('--th') + 1]) if '--th' in sys.argv else .5
    so = ort.SessionOptions(); so.intra_op_num_threads = 3; sess = ort.InferenceSession(model, so, providers=['CPUExecutionProvider'])
    print('입력', [(i.name, i.shape) for i in sess.get_inputs()], '출력', [(o.name, o.shape) for o in sess.get_outputs()])
    lines = []; t0 = time.time(); n = 0
    for s in sorted(os.listdir(root)):
        d = os.path.join(root, s)
        if not os.path.isdir(d) or filt not in s: continue
        for f in sorted(x for x in os.listdir(d) if x.startswith('img_')):
            im = Image.open(os.path.join(d, f)).convert('RGB'); bs = boxes_from_mask(predict(sess, im), im.width, im.height, th); n += 1
            if not bs: lines.append('%s\t%s\tNONE' % (s, f))
            for b in bs: lines.append('%s\t%s\t%d\t%d\t%d\t%d\t%s\t%.3f' % (s, f, *b))
    open(outp, 'w').write('\n'.join(lines) + '\n'); print('완료 %d장 %.0fs (%.2fs/장) → %s' % (n, time.time() - t0, (time.time() - t0) / max(1, n), outp))
