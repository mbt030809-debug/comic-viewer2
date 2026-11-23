#!/usr/bin/env python3
"""1단계 오검출/미검출 확인용 몽타주. 사용: python s1_debug.py <method> <bench/tier> <out.png> [세트수] [fp|miss]"""
import os, sys, json, random
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(__file__))
from eval_s1 import load_method, iou, overlap_frac

method, root, outp = sys.argv[1:4]; ns = int(sys.argv[4]) if len(sys.argv) > 4 else 40; mode = sys.argv[5] if len(sys.argv) > 5 else 'fp'
run = load_method(method, 'dml'); truth = json.load(open(os.path.join(root, 'truth.json'), encoding='utf-8'))['sets'][:ns]; items = []
for s in truth:
    for im in s['images']:
        pil = Image.open(os.path.join(root, s['id'], im['file'])).convert('RGB'); dets = run(pil); ex = im['example_marks']; used = set()
        for e in ex:
            best, bj = 0, -1
            for j, b in enumerate(dets):
                v = iou(b, e['box'])
                if v > best: best, bj = v, j
            if best >= .5: used.add(bj)
            elif mode == 'miss': items.append((pil.crop(tuple(np.clip([e['box'][0] - 30, e['box'][1] - 30, e['box'][0] + e['box'][2] + 30, e['box'][1] + e['box'][3] + 30], 0, [pil.width, pil.height, pil.width, pil.height]))), 'miss %s' % s['marks'][e['idx']]['kind']))
        if mode == 'fp':
            for j, b in enumerate(dets):
                if j in used: continue
                cat = 'cap' if any(overlap_frac(b, dd['box']) > .3 for dd in im['decoy_margins']) else 'art' if any(overlap_frac(b, m['box']) > .3 for m in im['marks']) else 'other'
                items.append((pil.crop((max(0, b[0] - 30), max(0, b[1] - 30), min(pil.width, b[0] + b[2] + 30), min(pil.height, b[1] + b[3] + 30))), '%s %dx%d' % (cat, b[2], b[3])))
random.Random(1).shuffle(items); items = items[:60]; cw, ch, cols = 190, 120, 8; rows = (len(items) + cols - 1) // cols
M = Image.new('RGB', (cols * cw, max(1, rows) * (ch + 14)), (60, 60, 60)); dr = ImageDraw.Draw(M)
for i, (c, lab) in enumerate(items):
    c.thumbnail((cw - 4, ch - 4)); x, y = (i % cols) * cw, (i // cols) * (ch + 14); M.paste(c, (x + 2, y + 14)); dr.text((x + 2, y + 1), lab, fill=(255, 255, 0))
M.save(outp); print(len(items), outp)
