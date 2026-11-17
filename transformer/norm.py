"""LayerNorm / BatchNorm：同一个张量，归一化的轴不一样。

    x: (B, L, d)      B = batch，L = 序列长度，d = d_model

    LayerNorm  在最后一维 d 上算 mean/std  -> 一个 token 自己归一化，跟别人无关
    BatchNorm  在 (B, L) 上算             -> 同一个维度跨所有样本、所有位置归一化

这一条差异，决定了 Transformer 为什么用 LN 不用 BN。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.b = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        del mask                                    # LN 用不上 mask，留着是为了调用一致
        m = x.mean(dim=-1, keepdim=True)
        s = x.std(dim=-1, unbiased=False, keepdim=True)
        return self.g * (x - m) / (s + self.eps) + self.b


class BatchNormSeq(nn.Module):
    """BatchNorm 在序列上的语义：对每个 d，在 (B, L) 上归一化。

    masked=True 时只用非 padding 的位置统计（PowerNorm, arXiv 2003.07845 的做法）。
    """

    def __init__(self, d: int, eps: float = 1e-5, masked: bool = False) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.b = nn.Parameter(torch.zeros(d))
        self.eps = eps
        self.masked = masked

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.masked and mask is not None:
            w = mask.unsqueeze(-1).to(x.dtype)                    # (B, L, 1)
            cnt = w.sum().clamp(min=1.0)
            m = (x * w).sum(dim=(0, 1), keepdim=True) / cnt
            v = ((x - m) ** 2 * w).sum(dim=(0, 1), keepdim=True) / cnt
            s = torch.sqrt(v + 1e-8)
        else:
            m = x.mean(dim=(0, 1), keepdim=True)
            s = x.std(dim=(0, 1), unbiased=False, keepdim=True)
        return self.g * (x - m) / (s + self.eps) + self.b
