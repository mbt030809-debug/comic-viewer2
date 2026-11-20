import sys,glob,os,numpy as np
from PIL import Image, ImageDraw, ImageFont
din,dout,outp=sys.argv[1:4]
FONT='/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc'; fs=ImageFont.truetype(FONT,14)
SC=0.2; cols=7
fl=sorted(glob.glob(din+'/*.jpg')); tiles=[]
def red(a): r,g,b=a[...,0],a[...,1],a[...,2]; return (r>=200)&(g<40)&(b<40)&(r-g>150)
for i,p in enumerate(fl):
    o=Image.open(dout+'/'+os.path.basename(p)).convert('RGB'); ia=np.array(Image.open(p).convert('RGB')).astype(int); oa=np.array(o).astype(int)
    new=red(oa)&~red(ia); ys,xs=np.where(new)
    t=o.copy(); d=ImageDraw.Draw(t); pos=None
    if len(ys)>60:
        cx,cy=int(xs.mean()),int(ys.mean()); pos=(cx,cy); d.ellipse([cx-45,cy-45,cx+45,cy+45],outline=(0,255,255),width=10)
    t=t.resize((int(1248*SC),int(1972*SC))); ImageDraw.Draw(t).text((3,3),str(i+1)+(' (%d,%d)'%pos if pos else ' -'),font=fs,fill=(255,255,0)); tiles.append(t)
rows=(len(tiles)+cols-1)//cols; W=cols*tiles[0].width; H=rows*tiles[0].height
S=Image.new('RGB',(W,H),(0,0,0))
for i,t in enumerate(tiles): S.paste(t,((i%cols)*t.width,(i//cols)*t.height))
S.save(outp); print(S.size)
