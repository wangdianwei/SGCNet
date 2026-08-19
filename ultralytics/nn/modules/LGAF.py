import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 强制导入 Mamba
# ==========================================

from mamba_ssm import Mamba

def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    """Standard convolution with BN and SiLU."""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class VisualGlobalMamba(nn.Module):
    raise RuntimeError()
class CoordAtt(nn.Module):
    """Coordinate Attention: 关注空间位置信息"""

    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        return identity * a_h * a_w


class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class AdaptiveFeatureFusion(nn.Module):
    raise RuntimeError
class DualAdaptiveBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, se_reduction=16):
        super().__init__()

        # --- Local Branch: 3x3 Conv ---
        # 擅长捕捉高频细节、边缘、纹理
        self.local_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            CoordAtt(out_channels, out_channels, reduction=se_reduction)
        )


        self.global_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),  # Channel mixing

            # Mamba 核心层 (必须安装 mamba-ssm)
            VisualGlobalMamba(in_channels, d_state=16, expand=2),

            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),  # Projection
            nn.BatchNorm2d(out_channels),
            SELayer(out_channels, reduction=se_reduction)
        )

        # 降采样处理 (Mamba本身不改变尺寸)
        if stride > 1:
            self.global_branch.add_module('pool', nn.AvgPool2d(kernel_size=stride, stride=stride))

        # --- Fusion ---
        self.fusion = AdaptiveFeatureFusion(out_channels, num_branches=2)

        # --- Shortcut ---
        self.shortcut = nn.Identity()
        if stride > 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.final_act = nn.SiLU(inplace=True)

    def forward(self, x):
        feat_local = self.local_branch(x)

        feat_global = self.global_branch(x)

        # 3. 自适应融合
        feat_fused = self.fusion([feat_local, feat_global])

        # 4. 残差连接
        out = feat_fused + self.shortcut(x)

        return self.final_act(out)

class LGAF(nn.Module):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """
        C3k2 with Mamba enhancement.
        Args:
            c1: input channels
            c2: output channels
            n: number of blocks
            e: expansion ratio
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        # 堆叠 n 个 DualAdaptiveBlock
        self.m = nn.ModuleList(
            DualAdaptiveBlock(
                in_channels=self.c,
                out_channels=self.c,
                stride=1,
                se_reduction=16
            ) for _ in range(n)
        )

    def forward(self, x):
        # Split -> Process -> Concat All
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


