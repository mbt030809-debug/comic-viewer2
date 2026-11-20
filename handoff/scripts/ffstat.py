import sys,glob,re,numpy as np
from PIL import Image
d=sys.argv[1]; ok=bad=none=0
for p in glob.glob(d+'/*.bin'):
    a=np.array(Image.open(p).convert('RGB')).astype(int); r,g,b=a[...,0],a[...,1],a[...,2]
    m=(r>=214)&(r<=234)&(g<14)&(b<14); ys,xs=np.where(m)
    if len(ys)<=60: none+=1; continue
    cx,cy=xs.mean(),ys.mean()
    if 900<=cx<=1140 and 720<=cy<=1010: ok+=1
    else: bad+=1
print('맞게',ok,'엉뚱한 곳',bad,'인식 안 함',none)
