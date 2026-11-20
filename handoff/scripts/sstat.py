import sys,glob,re,numpy as np
from PIL import Image
d=sys.argv[1]; kind=sys.argv[2]
ok=bad=none=0
for p in glob.glob(d+'/*.bin'):
    k=int(re.search(r'_(\d+)_[CN]\.bin',p).group(1)); hh=900+(k*137)%700 if kind=='S' else 0
    a=np.array(Image.open(p).convert('RGB')).astype(int); r,g,b=a[...,0],a[...,1],a[...,2]
    m=(r>=214)&(r<=234)&(g<14)&(b<14); ys,xs=np.where(m)
    if len(ys)<=60: none+=1; continue
    cx,cy=xs.mean(),ys.mean()
    # S: faint stamp 240x290, x = 965(even)/85(odd) fixed-x only when wm.fixedx; y0 = 80+(k*197+53)%(hh-290-160)
    y0=80+(k*197+53)%(hh-290-160); sx=965 if k%2==0 else 85
    if sx<=cx<=sx+240 and y0<=cy<=y0+290: ok+=1
    else: bad+=1
print('맞게',ok,'엉뚱한 곳',bad,'인식 안 함',none)
