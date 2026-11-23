"""XFeat(CVPR 2024, Potje et al., https://github.com/verlab/accelerated_features, Apache-2.0)의 희소 특징점 추출부를 이 저장소 코드로 옮긴 것.
원 저장소 코드를 실행하지 않고, 모델 정의·전처리·NMS·보간만 옮겨 쓴다. 가중치(xfeat.pt)는 텐서만 weights_only=True 로 읽는다.
함수: load(weights_path) → XFeatPort; XFeatPort.detect_and_compute(gray float32 [H,W] 0..1, top_k) → (keypoints [N,2] x,y, descriptors [N,64] L2정규화)."""
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F


class BasicLayer(nn.Module):
    def __init__(self, i, o, k=3, stride=1, padding=1, dilation=1, bias=False):
        super().__init__()
        self.layer = nn.Sequential(nn.Conv2d(i, o, k, padding=padding, stride=stride, dilation=dilation, bias=bias), nn.BatchNorm2d(o, affine=False), nn.ReLU(inplace=True))

    def forward(self, x): return self.layer(x)


class XFeatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.InstanceNorm2d(1)
        self.skip1 = nn.Sequential(nn.AvgPool2d(4, stride=4), nn.Conv2d(1, 24, 1, stride=1, padding=0))
        self.block1 = nn.Sequential(BasicLayer(1, 4, stride=1), BasicLayer(4, 8, stride=2), BasicLayer(8, 8, stride=1), BasicLayer(8, 24, stride=2))
        self.block2 = nn.Sequential(BasicLayer(24, 24, stride=1), BasicLayer(24, 24, stride=1))
        self.block3 = nn.Sequential(BasicLayer(24, 64, stride=2), BasicLayer(64, 64, stride=1), BasicLayer(64, 64, 1, padding=0))
        self.block4 = nn.Sequential(BasicLayer(64, 64, stride=2), BasicLayer(64, 64, stride=1), BasicLayer(64, 64, stride=1))
        self.block5 = nn.Sequential(BasicLayer(64, 128, stride=2), BasicLayer(128, 128, stride=1), BasicLayer(128, 128, stride=1), BasicLayer(128, 64, 1, padding=0))
        self.block_fusion = nn.Sequential(BasicLayer(64, 64, stride=1), BasicLayer(64, 64, stride=1), nn.Conv2d(64, 64, 1, padding=0))
        self.heatmap_head = nn.Sequential(BasicLayer(64, 64, 1, padding=0), BasicLayer(64, 64, 1, padding=0), nn.Conv2d(64, 1, 1), nn.Sigmoid())
        self.keypoint_head = nn.Sequential(BasicLayer(64, 64, 1, padding=0), BasicLayer(64, 64, 1, padding=0), BasicLayer(64, 64, 1, padding=0), nn.Conv2d(64, 65, 1))
        self.fine_matcher = nn.Sequential(nn.Linear(128, 512), nn.BatchNorm1d(512, affine=False), nn.ReLU(inplace=True), nn.Linear(512, 512), nn.BatchNorm1d(512, affine=False), nn.ReLU(inplace=True),
                                          nn.Linear(512, 512), nn.BatchNorm1d(512, affine=False), nn.ReLU(inplace=True), nn.Linear(512, 512), nn.BatchNorm1d(512, affine=False), nn.ReLU(inplace=True), nn.Linear(512, 64))

    def _unfold2d(self, x, ws=2):
        B, C, H, W = x.shape
        x = x.unfold(2, ws, ws).unfold(3, ws, ws).reshape(B, C, H // ws, W // ws, ws ** 2)
        return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H // ws, W // ws)

    def forward(self, x):
        with torch.no_grad():
            x = x.mean(dim=1, keepdim=True); x = self.norm(x)
        x1 = self.block1(x); x2 = self.block2(x1 + self.skip1(x)); x3 = self.block3(x2); x4 = self.block4(x3); x5 = self.block5(x4)
        x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear'); x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        feats = self.block_fusion(x3 + x4 + x5)
        return feats, self.keypoint_head(self._unfold2d(x, ws=8)), self.heatmap_head(feats)


def _kpts_heatmap(kpts):
    scores = F.softmax(kpts, 1)[:, :64]; B, _, H, W = scores.shape
    return scores.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8).permute(0, 1, 3, 2, 4).reshape(B, 1, H * 8, W * 8)


def _sample(x, pos, H, W, mode):
    grid = (2. * (pos / torch.tensor([W - 1, H - 1], dtype=pos.dtype)) - 1.).unsqueeze(-2).to(x.dtype)
    return F.grid_sample(x, grid, mode=mode, align_corners=False).permute(0, 2, 3, 1).squeeze(-2)


class XFeatPort:
    def __init__(self, net): self.net = net.eval()

    @torch.inference_mode()
    def detect_and_compute(self, gray, top_k=3000, thr=0.05):
        H, W = gray.shape; _H, _W = (H // 32) * 32, (W // 32) * 32
        x = F.interpolate(torch.from_numpy(gray)[None, None].float(), (_H, _W), mode='bilinear', align_corners=False); rh, rw = H / _H, W / _W
        M1, K1, H1 = self.net(x); M1 = F.normalize(M1, dim=1)
        K1h = _kpts_heatmap(K1); local = nn.MaxPool2d(5, stride=1, padding=2)(K1h); pos = ((K1h == local) & (K1h > thr))[0, 0].nonzero().flip(-1)   # (x,y)
        if len(pos) == 0: return np.zeros((0, 2), np.float32), np.zeros((0, 64), np.float32)
        p = pos[None].float(); sc = (_sample(K1h, p, _H, _W, 'nearest') * _sample(H1, p, _H, _W, 'bilinear')).squeeze(-1)[0]
        idx = torch.argsort(-sc)[:top_k]; pos = pos[idx]; sc = sc[idx]; keep = sc > 0; pos = pos[keep]
        feats = F.normalize(_sample(M1, pos[None].float(), _H, _W, 'bicubic')[0], dim=-1)
        return (pos.float() * torch.tensor([rw, rh])).numpy().astype(np.float32), feats.numpy().astype(np.float32)


def load(weights_path):
    net = XFeatModel(); net.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True)); return XFeatPort(net)
