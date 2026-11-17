"""Transformer 的 block：attention + FFN，外面套残差和 LayerNorm。

    post-norm（原论文）: x = LN(x + Sublayer(x))
    pre-norm（后来的主流）: x = x + Sublayer(LN(x))

一个 block 就两个子层；encoder 只有这两个，decoder 在前面多一个 cross-attention。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .norm import LayerNorm


class FFN(nn.Module):
    """两层全连接 + 激活，对每个位置独立作用。d_model -> d_ff -> d_model。"""

    def __init__(self, d: int, d_ff: int | None = None) -> None:
        super().__init__()
        self.d_ff = d_ff or 4 * d
        self.fc1 = nn.Linear(d, self.d_ff)
        self.fc2 = nn.Linear(self.d_ff, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class MultiHeadAttention(nn.Module):
    """多头注意力，一个类兼容两种用法：

        self-attention   forward(x)           Q/K/V 都来自 x
        cross-attention  forward(y, memory)   Q 来自 y，K/V 来自 memory

    causal=True 时加因果掩码（decoder 的 masked self-attention）。
    """

    def __init__(self, d: int, n_heads: int = 8, causal: bool = False) -> None:
        super().__init__()
        assert d % n_heads == 0, "d_model 要能被头数整除"
        self.d, self.h, self.causal = d, n_heads, causal
        self.hd = d // n_heads
        self.w_q = nn.Linear(d, d)
        self.w_k = nn.Linear(d, d)
        self.w_v = nn.Linear(d, d)
        self.w_o = nn.Linear(d, d)
        self.last_attn: torch.Tensor | None = None      # 留一份给 demo 看

    def _split(self, t: torch.Tensor) -> torch.Tensor:
        b, length, _ = t.shape
        return t.view(b, length, self.h, self.hd).transpose(1, 2)   # (B, h, L, hd)

    def forward(self, x: torch.Tensor, memory: torch.Tensor | None = None) -> torch.Tensor:
        src = x if memory is None else memory
        q = self._split(self.w_q(x))
        k = self._split(self.w_k(src))
        v = self._split(self.w_v(src))

        s = q @ k.transpose(-1, -2) / math.sqrt(self.hd)            # (B, h, Lq, Lk)
        if self.causal:
            lq, lk = s.shape[-2], s.shape[-1]
            mask = torch.triu(torch.ones(lq, lk, dtype=torch.bool), diagonal=1)
            s = s.masked_fill(mask, float("-inf"))

        a = torch.softmax(s, dim=-1)
        self.last_attn = a.detach()

        out = a @ v                                                 # (B, h, Lq, hd)
        b, _, lq, _ = out.shape
        return self.w_o(out.transpose(1, 2).reshape(b, lq, self.d))


class EncoderBlock(nn.Module):
    """两个子层：self-attention -> FFN，各自 residual + norm。"""

    def __init__(self, d: int, n_heads: int = 8, d_ff: int | None = None,
                 pre_norm: bool = False) -> None:
        super().__init__()
        self.pre_norm = pre_norm
        self.attn = MultiHeadAttention(d, n_heads)
        self.ffn = FFN(d, d_ff)
        self.ln1 = LayerNorm(d)
        self.ln2 = LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        else:
            x = self.ln1(x + self.attn(x))
            x = self.ln2(x + self.ffn(x))
        return x


class DecoderBlock(nn.Module):
    """三个子层：masked self-attention -> cross-attention -> FFN。"""

    def __init__(self, d: int, n_heads: int = 8, d_ff: int | None = None,
                 pre_norm: bool = False) -> None:
        super().__init__()
        self.pre_norm = pre_norm
        self.self_attn = MultiHeadAttention(d, n_heads, causal=True)
        self.cross_attn = MultiHeadAttention(d, n_heads)
        self.ffn = FFN(d, d_ff)
        self.ln1 = LayerNorm(d)
        self.ln2 = LayerNorm(d)
        self.ln3 = LayerNorm(d)

    def forward(self, y: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            y = y + self.self_attn(self.ln1(y))
            y = y + self.cross_attn(self.ln2(y), memory)
            y = y + self.ffn(self.ln3(y))
        else:
            y = self.ln1(y + self.self_attn(y))
            y = self.ln2(y + self.cross_attn(y, memory))
            y = self.ln3(y + self.ffn(y))
        return y
