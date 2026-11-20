import sys,glob,re,numpy as np
from PIL import Image
d=sys.argv[1]; fixed=sys.argv[2]=='1'
def y0(k,hh): return 700 if fixed else 80+(k*197+53)%(hh-360-160)
ok=bad=none=0
for p in glob.glob(d+'/*.bin'):
    k=int(re.search(r'_(\d+)_[CN]\.bin',p).group(1)); hh=1100+(k*137)%700
    a=np.array(Image.open(p).convert('RGB')).astype(int); r,g,b=a[...,0],a[...,1],a[...,2]
    m=(r>=214)&(r<=234)&(g<14)&(b<14); ys,xs=np.where(m)
    if len(ys)<=60: none+=1; continue
    cx,cy=xs.mean(),ys.mean(); y=y0(k,hh)
    if 840<=cx<=1230 and y<=cy<=y+360: ok+=1
    else: bad+=1
print('맞게',ok,'엉뚱한 곳',bad,'인식 안 함',none)
