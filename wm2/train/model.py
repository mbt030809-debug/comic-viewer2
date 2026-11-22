import torch, torch.nn as nn, torch.nn.functional as F

def cb(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.GroupNorm(min(8, o // 2), o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.GroupNorm(min(8, o // 2), o), nn.ReLU(inplace=True))

class UNet(nn.Module):
    """워터마크 마스크 예측용 작은 U-Net(0.5배 축소 이미지 입력, 출력 = 마스크 로짓)"""
    def __init__(self, ch=(16, 32, 64, 96, 128)):
        super().__init__()
        k = torch.tensor([[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]]) / 8; sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8
        self.register_buffer('hp', k[None, None]); self.register_buffer('gx', sx[None, None]); self.register_buffer('gy', sx.t()[None, None])
        self.enc = nn.ModuleList([cb(5 if i == 0 else ch[i - 1], ch[i]) for i in range(len(ch))])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in range(len(ch) - 1)])
        self.dec = nn.ModuleList([cb(ch[i] * 2, ch[i]) for i in range(len(ch) - 1)])
        self.out = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        g = (x[:, 0:1] * .299 + x[:, 1:2] * .587 + x[:, 2:3] * .114) * 255; gp = F.pad(g, (1, 1, 1, 1), mode='replicate')
        hp = F.conv2d(gp, self.hp); gx = F.conv2d(gp, self.gx); gy = F.conv2d(gp, self.gy)
        x = torch.cat([x - 0.5, hp / 16, torch.sqrt(gx * gx + gy * gy + 1e-3) / 16], 1); sk = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else F.max_pool2d(x, 2)); sk.append(x)
        for i in reversed(range(len(self.up))):
            x = self.up[i](x); x = self.dec[i](torch.cat([x, sk[i]], 1))
        return self.out(x)

if __name__ == '__main__':
    m = UNet(); print('파라미터 %.2fM' % (sum(p.numel() for p in m.parameters()) / 1e6))
