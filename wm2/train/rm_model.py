import torch, torch.nn as nn, torch.nn.functional as F


def cb(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.GroupNorm(min(8, o // 2), o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.GroupNorm(min(8, o // 2), o), nn.ReLU(inplace=True))


class RmUNet(nn.Module):
    """워터마크 제거 U-Net. 입력: image[N,3,H,W] 0..1, mask[N,1,H,W] 0..1 (인식기가 낸 영역). 출력: 제거된 이미지 [N,3,H,W] 0..1.
    마스크 밖은 입력 그대로(블렌딩), 마스크 안은 image+delta. H,W 는 16의 배수."""
    def __init__(self, ch=(16, 32, 64, 96, 128)):
        super().__init__()
        self.enc = nn.ModuleList([cb(4 if i == 0 else ch[i - 1], ch[i]) for i in range(len(ch))])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in range(len(ch) - 1)])
        self.dec = nn.ModuleList([cb(ch[i] * 2, ch[i]) for i in range(len(ch) - 1)])
        self.out = nn.Conv2d(ch[0], 3, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, img, m):
        x = torch.cat([img - 0.5, m], 1); sk = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else F.max_pool2d(x, 2)); sk.append(x)
        for i in reversed(range(len(self.up))):
            x = self.up[i](x); x = self.dec[i](torch.cat([x, sk[i]], 1))
        pred = (img + self.out(x)).clamp(0, 1)
        return img * (1 - m) + pred * m
