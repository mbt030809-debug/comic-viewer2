import torch, torch.nn as nn, torch.nn.functional as F

RH, RW = 112, 192


def cb(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.GroupNorm(min(8, o // 2), o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.GroupNorm(min(8, o // 2), o), nn.ReLU(inplace=True))


class RefUNet(nn.Module):
    """참조형 워터마크 인식. 입력: image[N,3,H,W] (0.5배 축소 타깃, H,W 16의 배수), ref[N,3,112,192] (예시 마크: 흰/검은 배경 위, 0.5배 축소 크기).
    출력: logits[N,1,H,W] = '예시 마크가 박힌 픽셀' 확률 지도. 인코더(앞 4단)를 타깃·예시가 공유하고, 1/8 해상도에서 예시 특징(마크 영역만)을
    커널로 한 채널별 상관을 구해 디코더 1/8 단계에 넣는다."""
    def __init__(self, ch=(16, 32, 64, 96, 128), cc=32):
        super().__init__()
        k = torch.tensor([[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]]) / 8; sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8
        self.register_buffer('hp', k[None, None]); self.register_buffer('gx', sx[None, None]); self.register_buffer('gy', sx.t()[None, None])
        self.enc = nn.ModuleList([cb(5 if i == 0 else ch[i - 1], ch[i]) for i in range(len(ch))])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in range(len(ch) - 1)])
        self.dec = nn.ModuleList([cb(ch[i] * 2 + (cc if i == 3 else 0), ch[i]) for i in range(len(ch) - 1)])
        self.pt = nn.Sequential(nn.Conv2d(ch[3], cc, 1), nn.GroupNorm(8, cc)); self.pr = nn.Conv2d(ch[3], cc, 1)
        self.G = 8; self.cf = nn.Sequential(nn.Conv2d(self.G, cc, 3, padding=1, bias=False), nn.GroupNorm(8, cc), nn.ReLU(inplace=True))
        self.out = nn.Conv2d(ch[0], 1, 1); self.cc = cc

    def prep(self, x):
        g = (x[:, 0:1] * .299 + x[:, 1:2] * .587 + x[:, 2:3] * .114) * 255; gp = F.pad(g, (1, 1, 1, 1), mode='replicate')
        hp = F.conv2d(gp, self.hp); gx = F.conv2d(gp, self.gx); gy = F.conv2d(gp, self.gy)
        return torch.cat([x - 0.5, hp / 16, torch.sqrt(gx * gx + gy * gy + 1e-3) / 16], 1)

    def encode(self, x, levels):
        x = self.prep(x); sk = []
        for i in range(levels):
            x = self.enc[i](x if i == 0 else F.max_pool2d(x, 2)); sk.append(x)
        return sk

    def forward(self, img, ref):
        sk = self.encode(img, 4); R = self.encode(ref, 4)[3]                      # R: [N,ch3,14,24]
        bgc = ref[:, :, :2, :2].mean((2, 3), keepdim=True)
        nb = ((ref - bgc).abs().amax(1, keepdim=True) > 0.12).float(); nbm = F.max_pool2d(nb, 8)   # 마크가 있는 칸만
        pr = self.pr(R) * nbm; pr = pr / (torch.sqrt((pr * pr).sum((1, 2, 3), keepdim=True) / nbm.sum((1, 2, 3), keepdim=True).clamp(min=1) / self.cc + 1e-6))
        pt = self.pt(sk[3]); N, C, H8, W8 = pt.shape; kh, kw = pr.shape[2], pr.shape[3]
        if kh % 2 == 0: pr = F.pad(pr, (0, 0, 0, 1))       # 커널을 홀수 크기로(0 채움) → 합성곱 자체 패딩 사용
        if kw % 2 == 0: pr = F.pad(pr, (0, 1))
        kh, kw = pr.shape[2], pr.shape[3]; G = self.G
        # 샘플마다 서로 다른 커널 → 표본별 합성곱(채널을 G묶음으로 나눠 G장의 상관 지도). DML 의 배치-그룹 합성곱 역전파 오류를 피한다.
        corr = torch.cat([F.conv2d(pt[n:n + 1], pr[n].reshape(G, C // G, kh, kw), padding=(kh // 2, kw // 2), groups=G) for n in range(N)], 0)
        corr = corr / nbm.sum((1, 2, 3), keepdim=True).clamp(min=1)
        cf = self.cf(corr)
        x = self.enc[4](F.max_pool2d(sk[3], 2))
        for i in reversed(range(len(self.up))):
            x = self.up[i](x); x = self.dec[i](torch.cat([x, sk[i], cf] if i == 3 else [x, sk[i]], 1))
        return self.out(x)


if __name__ == '__main__':
    m = RefUNet(); print('파라미터 %.3fM' % (sum(p.numel() for p in m.parameters()) / 1e6))
    print(m(torch.rand(2, 3, 192, 192), torch.rand(2, 3, RH, RW)).shape)
