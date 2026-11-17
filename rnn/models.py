"""单向 / 双向 RNN 的最小实现，重点全在"把中间结果留下来"。

两个模型都按时间步手写循环，而不是直接 nn.RNN，因为 demo 要的正是
每一步的 a^t —— 用现成模块就只能拿到最后一步。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .cell import ElmanCell


class UniRNN(nn.Module):
    """单向 RNN（课件左图）。

    同一个 cell 反复使用，a^t 只看得到 x^1..x^t。forward 把每一步的隐状态
    都收集下来返回，这就是课件里 "store" 那条蓝色箭头存的东西，
    也是 y^t 唯一的来源。
    """

    def __init__(self, vocab_size: int, embed_dim: int = 6, hidden_dim: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.cell = ElmanCell(embed_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, vocab_size)   # a^t -> y^t
        self.hidden_dim = hidden_dim

    def forward(self, ids: list[int]) -> dict[str, torch.Tensor]:
        a = torch.zeros(self.hidden_dim)               # a^0 = 0
        states: list[torch.Tensor] = []
        probs: list[torch.Tensor] = []

        for tid in torch.as_tensor(ids, dtype=torch.long):
            a = self.cell(self.embed(tid), a)          # 存下这一步的中间结果
            states.append(a)
            probs.append(torch.softmax(self.W_o(a), dim=-1))

        return {"a": torch.stack(states), "p": torch.stack(probs)}


class BiRNN(nn.Module):
    """双向 RNN（课件右图）。

    正向通道从 x^1 读到 x^T，反向通道从 x^T 读回 x^1；两条通道各有独立的
    参数，各自攒出一串中间结果。位置 t 的输出同时读两个方向：

        y^t = W_o([ a_fwd^t ; a_bwd^t ])

    所以 t 处既知道它左边（包括自己）的上下文，也知道它右边（包括自己）的上下文。
    """

    def __init__(self, vocab_size: int, embed_dim: int = 6, hidden_dim: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.cell_fwd = ElmanCell(embed_dim, hidden_dim)
        self.cell_bwd = ElmanCell(embed_dim, hidden_dim)   # 反向是另一套参数
        self.W_o = nn.Linear(hidden_dim * 2, vocab_size)   # 两路拼接 -> y^t
        self.hidden_dim = hidden_dim

    def forward(self, ids: list[int]) -> dict[str, torch.Tensor]:
        inputs = [self.embed(t) for t in torch.as_tensor(ids, dtype=torch.long)]

        a = torch.zeros(self.hidden_dim)
        fwd: list[torch.Tensor] = []
        for x_t in inputs:                             # 正向：x^1 -> x^T
            a = self.cell_fwd(x_t, a)
            fwd.append(a)

        a = torch.zeros(self.hidden_dim)
        bwd: list[torch.Tensor] = []
        for x_t in reversed(inputs):                   # 反向：x^T -> x^1
            a = self.cell_bwd(x_t, a)
            bwd.append(a)
        bwd.reverse()                                  # 让下标对齐到时间步

        a_fwd = torch.stack(fwd)
        a_bwd = torch.stack(bwd)
        joined = torch.cat([a_fwd, a_bwd], dim=-1)      # (T, 2H)
        return {"a_fwd": a_fwd, "a_bwd": a_bwd,
                "p": torch.softmax(self.W_o(joined), dim=-1)}
