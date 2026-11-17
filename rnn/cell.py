"""最原始的 Elman RNN 单元。

课件《Elman Network / Jordan Network》那一页的三个权重，这里一一对应：

    a^t = tanh(W_i · x^t + W_h · a^{t-1} + b)
          ↑输入       ↑上一步的隐状态（"store" 存下来的东西）

    W_i : 输入  -> 隐层
    W_h : 隐层  -> 隐层（recurrent，同一个矩阵反复用，这就是 "the same
          network is used again and again"）
    W_o : 隐层  -> 输出（放在模型里，因为双向版本要拼接两个方向再乘）

Jordan Network 只差一处：recurrent 的来源从 a^{t-1} 换成上一步的输出 y^{t-1}。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ElmanCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.W_i = nn.Linear(input_dim, hidden_dim)                  # 含 bias
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)      # 纯 recurrent

    def forward(self, x_t: torch.Tensor, a_prev: torch.Tensor) -> torch.Tensor:
        """单步：吃一个输入向量和上一步隐状态，吐出这一步的隐状态。"""
        return torch.tanh(self.W_i(x_t) + self.W_h(a_prev))
