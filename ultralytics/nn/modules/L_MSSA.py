import torch
import torch.nn as nn
import torch.nn.functional as F


class L_MSSA(nn.Module):
    """
    Final Stable Version: ASFF-like Lightweight Multi-Scale Morphological Aligner
    """

    def __init__(self, c_list, c2, level="P4", k_strip=5, alpha_init=-2.2, beta_init=-1.7):
        super().__init__()
        c3, c4, c5 = c_list
        self.level = level

        if level == "P3":
            self.real_c2 = c3
        elif level == "P4":
            self.real_c2 = c4
        else:
            self.real_c2 = c5

        self.mid = max(16, min(24, self.real_c2 // 8))

        self.proj3 = nn.Conv2d(c3, self.mid, kernel_size=1, bias=False)
        self.proj4 = nn.Conv2d(c4, self.mid, kernel_size=1, bias=False)
        self.proj5 = nn.Conv2d(c5, self.mid, kernel_size=1, bias=False)

        self.target_norm = nn.Sequential(

        )

        hidden = max(8, self.mid // 2)
        self.aux_gate = nn.Sequential(

        )

        self.axial_strip = nn.Sequential(
        )

        self.out_conv = nn.Conv2d(self.mid, self.real_c2, kernel_size=1, bias=False)

        # constrained strengths
        self.alpha_param = nn.Parameter()
        self.beta_param = nn.Parameter()

    def forward(self, x):
        p3, p4, p5 = x

        if self.level == "P3":
            target = p3
            target_size = p3.shape[2:]
            t = self._resample(self.proj3(p3), target_size)
            a1 = self._resample(self.proj4(p4), target_size)
            a2 = self._resample(self.proj5(p5), target_size)

        elif self.level == "P4":
            target = p4
            target_size = p4.shape[2:]
            t = self._resample(self.proj4(p4), target_size)
            a1 = self._resample(self.proj3(p3), target_size)
            a2 = self._resample(self.proj5(p5), target_size)

        else:
            target = p5
            target_size = p5.shape[2:]
            t = self._resample(self.proj5(p5), target_size)
            a1 = self._resample(self.proj3(p3), target_size)
            a2 = self._resample(self.proj4(p4), target_size)

        t = self.target_norm(t)

        gate_in = torch.cat([t, a1, a2], dim=1)
        w_aux = self.aux_gate(gate_in)
        w_aux = torch.softmax(w_aux, dim=1)

        aux = w_aux[:, 0:1] * a1 + w_aux[:, 1:2] * a2

        beta = torch.sigmoid()     # ~0.15 init
        alpha = torch.sigmoid()   # ~0.10 init
        fused = t + beta * aux
        refined = self.axial_strip()

        enh = self.out_conv(refined)
        out = target + alpha * enh
        return out

    def _resample(self, x, size):
        if x.shape[2:] == size:
            return x
        if x.shape[2] < size[0]:
            return F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return F.adaptive_avg_pool2d()
