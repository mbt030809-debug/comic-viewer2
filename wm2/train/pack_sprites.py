#!/usr/bin/env python3
"""sprites.pkl → sprites_flat/ (메모리맵용). 사용: python3 pack_sprites.py <풀폴더>"""
import os, sys, json, pickle
import numpy as np

root = sys.argv[1]
sp = pickle.load(open(os.path.join(root, 'sprites.pkl'), 'rb'))
kinds = sorted({s[0] for s in sp}); kid = {k: i for i, k in enumerate(kinds)}
na = sum(s[1].size for s in sp); npp = sum(s[2].size for s in sp)
out = os.path.join(root, 'sprites_flat'); os.makedirs(out, exist_ok=True)
A = np.lib.format.open_memmap(os.path.join(out, 'A.npy'), mode='w+', dtype=np.uint8, shape=(na,))
P = np.lib.format.open_memmap(os.path.join(out, 'P.npy'), mode='w+', dtype=np.uint8, shape=(npp,))
meta = []; oa = op = 0
for kind, a, p in sp:
    h, w = a.shape; A[oa:oa + h * w] = a.ravel(); P[op:op + h * w * 3] = p.reshape(-1); meta.append((kid[kind], h, w, oa, op)); oa += h * w; op += h * w * 3
A.flush(); P.flush(); np.save(os.path.join(out, 'meta.npy'), np.array(meta, np.int64)); json.dump(kinds, open(os.path.join(out, 'kinds.json'), 'w'))
print('스프라이트 %d개, A %.0fMB P %.0fMB' % (len(sp), na / 1e6, npp / 1e6))
