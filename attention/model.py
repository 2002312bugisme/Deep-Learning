"""Self-attention 的核心实现，形状跟课件一致（一个 token 一列）。

    Q = W^q I,  K = W^k I,  V = W^v I        d_k x N
    A[j][i] = k^j · q^i                      N x N，行是 key、列是 query
    A' = softmax(A, 每列归一化)
    O = V A'                                 d_v x N
"""

from __future__ import annotations

import torch


class SelfAttention(torch.nn.Module):
    """单头 self-attention。参数只有 W^q / W^k / W^v 三个矩阵。"""

    def __init__(self, w_q: torch.Tensor, w_k: torch.Tensor, w_v: torch.Tensor) -> None:
        super().__init__()
        self.W_q = torch.nn.Parameter(w_q.clone())
        self.W_k = torch.nn.Parameter(w_k.clone())
        self.W_v = torch.nn.Parameter(w_v.clone())

    def forward(self, x: torch.Tensor, pe: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if pe is not None:
            x = x + pe                       # 位置编码是"加"进去，不是拼接
        q = self.W_q @ x
        k = self.W_k @ x
        v = self.W_v @ x
        a = k.T @ q                          # A[j][i] = k^j · q^i
        a_p = torch.softmax(a, dim=0)        # 同一列（同一个 query）对所有 key 归一化
        return {"I": x, "Q": q, "K": k, "V": v, "A": a, "A'": a_p, "O": v @ a_p}


def sinusoidal_pe(d_model: int, n: int) -> torch.Tensor:
    """手搓版位置编码（原论文的 sin/cos 方案），返回 d_model x N。"""
    pos = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(0, d_model, 2, dtype=torch.float32)
    div = torch.exp(-torch.log(torch.tensor(10000.0)) * i / d_model)
    pe = torch.zeros(n, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.T
