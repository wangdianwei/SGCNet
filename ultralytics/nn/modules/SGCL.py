from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.ops import roi_align
except Exception:
    roi_align = None


def _to_gray(img: torch.Tensor) -> torch.Tensor:
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


def _xywhn_to_xyxy_abs(xywhn: torch.Tensor, img_wh: Tuple[int, int]) -> torch.Tensor:
    W, H = img_wh
    xc, yc, w, h = xywhn.unbind(-1)
    x1 = (xc - w / 2) * W
    y1 = (yc - h / 2) * H
    x2 = (xc + w / 2) * W
    y2 = (yc + h / 2) * H
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _clip_boxes_xyxy(boxes: torch.Tensor, W: int, H: int) -> torch.Tensor:
    boxes = boxes.clone()
    boxes[:, 0].clamp_(0, W - 1)
    boxes[:, 2].clamp_(0, W - 1)
    boxes[:, 1].clamp_(0, H - 1)
    boxes[:, 3].clamp_(0, H - 1)
    boxes[:, 2] = torch.max(boxes[:, 2], boxes[:, 0] + 1.0)
    boxes[:, 3] = torch.max(boxes[:, 3], boxes[:, 1] + 1.0)
    return boxes


def _focal_bce_with_logits(logits, targets, alpha=0.25, gamma=2.0):
    if targets.dim() == logits.dim() - 1:
        targets = targets.unsqueeze(1)
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = targets * p + (1 - targets) * (1 - p)
    loss = (targets * alpha + (1 - targets) * (1 - alpha)) * (1 - p_t + 1e-7).pow(gamma) * ce
    return loss.mean()


def _dice_loss_from_logits(logits, targets, eps=1e-6):
    if targets.dim() == logits.dim() - 1:
        targets = targets.unsqueeze(1)
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(1, 2, 3))
    den = (probs + targets).sum(dim=(1, 2, 3)) + eps
    return (1 - num / den).mean()


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv1())
        return x * scale


class StripHead(nn.Module):

    def __init__(self, c: int, hidden: int = 64):
        super().__init__()
        hidden = min(128, max(32, c // 2))

        k_size = 11
        pad = k_size // 2
        self.conv1 = nn.Conv2d(c, hidden, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=k_size, padding=pad, groups=hidden, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.sa = SpatialAttention(kernel_size=7)
        self.conv3 = nn.Conv2d(hidden, hidden, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(hidden)
        self.out = nn.Conv2d(hidden, 1, 1, bias=True)

    def forward(self, x):
        x = F.silu(self.bn1(self.conv1(x)))
        res = x
        x = F.silu(self.bn2(self.conv2(x)))
        x = self.sa(x)
        x = F.silu(self.bn3(self.conv3(x)))
        return self.out(x + res)


@dataclass
class SGCLStripConfig:
    enable: bool = True
    cls_vertical: int = 3
    cls_horizontal: int = 0
    cls_alligator: int = 2

    roi_out: int = 20
    mask_out: int = 56
    blackhat_k: int = 11
    connect_k: int = 9

    beta_aux: float = 0.12

class SGCLStrip(nn.Module):
    def __init__(self, in_channels: int, cfg: SGCLStripConfig = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else SGCLStripConfig()
        self.head = StripHead(in_channels)
        self._iter = 0
        self.temp_feat = None

    def forward(self, x):
        if self.training:
            self.temp_feat = x
        return x

    @torch.no_grad()
    def _morph_blackhat_torch(self, img: torch.Tensor, kernel_size: int) -> torch.Tensor:
        pad = kernel_size // 2
        dilated = F.max_pool2d()
        closed = -F.max_pool2d()
        return (closed - img).clamp(min=0)

    @torch.no_grad()
    def _apply_directional_processing(self, binary_proxy: torch.Tensor, is_vertical: bool) -> torch.Tensor:
        connected = F.max_pool2d()
        return connected

    @torch.no_grad()
    def _build_adaptive_mask(self, roi_gray: torch.Tensor, cls_id: int) -> torch.Tensor:
        raise RuntimeError()

    def forward_loss(self, feat: torch.Tensor, imgs: torch.Tensor, batch: Dict, stride: float):
        cfg = self.cfg
        self._iter += 1
        if (not cfg.enable) or roi_align is None:
            return feat.new_zeros(()), {"sgcl_n": 0.0}
        if cfg.aux_warmup_iters > 0 and self._iter < cfg.aux_warmup_iters:
            return feat.new_zeros(()), {"sgcl_n": 0.0}
        bboxes = batch.get("bboxes", None)
        cls = batch.get("cls", None)
        bidx = batch.get("batch_idx", None)
        if bboxes is None or cls is None or bidx is None:
            return feat.new_zeros(()), {"sgcl_n": 0.0}

        cls = cls.view(-1).detach()
        bidx = bidx.view(-1).detach().to(torch.int64)
        bboxes = bboxes.detach()
        keep = (cls == cfg.cls_vertical) | (cls == cfg.cls_horizontal) | (cls == cfg.cls_alligator)
        if keep.sum() == 0:
            return feat.new_zeros(()), {"sgcl_n": 0.0}

        cls_filtered = cls[keep]
        bidx_filtered = bidx[keep]
        bboxes_filtered = bboxes[keep]
        _, _, H, W = imgs.shape
        imgs01 = imgs
        if imgs01.max() > 1.5:
            imgs01 = (imgs01 / 255.0).clamp(0, 1)

        boxes_xyxy = _xywhn_to_xyxy_abs(bboxes_filtered, (W, H))
        boxes_xyxy = _clip_boxes_xyxy(boxes_xyxy, W, H)
        rois = torch.cat([bidx_filtered.float().unsqueeze(1), boxes_xyxy], dim=1)

        roi_feat = roi_align(feat, rois, output_size=(cfg.roi_out, cfg.roi_out), spatial_scale=1.0 / float(stride),
                             sampling_ratio=2, aligned=True)
        strip_logits = self.head(roi_feat)
        gray = _to_gray(imgs01).clamp(0, 1)
        roi_gray = roi_align(gray, rois, output_size=(cfg.mask_out, cfg.mask_out), spatial_scale=1.0, sampling_ratio=2,
                             aligned=True)

        masks = []
        for i in range(roi_gray.shape[0]):
            masks.append(self._build_adaptive_mask(roi_gray[i:i + 1], int(cls_filtered[i].item())))

        pseudo = torch.cat(masks, dim=0)
        pseudo_ds = F.interpolate(pseudo, size=strip_logits.shape[-2:], mode="bilinear", align_corners=False)
        l_focal = _focal_bce_with_logits(strip_logits, pseudo_ds, alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
        l_dice = _dice_loss_from_logits(strip_logits, pseudo_ds)
        aux = l_focal + cfg.lambda_dice * l_dice
        return aux * cfg.beta_aux, {"sgcl_n": float(pseudo_ds.shape[0]), "sgcl_focal": float(l_focal.detach().cpu()),
                                    "sgcl_dice": float(l_dice.detach().cpu())}