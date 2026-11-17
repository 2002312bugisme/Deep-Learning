"""手写 LSTM cell / layer，对照课件那两页图。

    z^f = σ(W_f [x^t ; h^{t-1}] + b_f)      忘记门：c^{t-1} 留多少
    z^i = σ(W_i [x^t ; h^{t-1}] + b_i)      输入门：这次写多少
    z   = tanh(W_z [x^t ; h^{t-1}] + b_z)   候选：写什么
    z^o = σ(W_o [x^t ; h^{t-1}] + b_o)      输出门：露多少

    c^t = z^f ⊙ c^{t-1} + z^i ⊙ z          ← 加法通道，记忆本体
    h^t = z^o ⊙ tanh(c^t)                  ← 下一步的输入 + 这一步的输出

和课件的一处差别：课件第一版把门的输入简化成只画 x^t，标准 LSTM 是
[x^t ; h^{t-1}] 拼起来再乘。第二版标的 "peephole" 扩展再多接一条 c^{t-1}
（那个红圈），标准 LSTM 没有这条线，这里也没实现。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    """单个 LSTM 单元：4 个门 + 一条 c 通道。

    水平方向反复调用的就是这一个对象。它的参数数量和序列长度无关 ——
    这是整件事的关键。
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        gate_in = input_dim + hidden_dim          # [x^t ; h^{t-1}]
        self.W_f = nn.Linear(gate_in, hidden_dim)  # 课件 Green box: z^f
        self.W_i = nn.Linear(gate_in, hidden_dim)  # 课件 Green box: z^i
        self.W_z = nn.Linear(gate_in, hidden_dim)  # 课件 Green box: z
        self.W_o = nn.Linear(gate_in, hidden_dim)  # 课件 Green box: z^o

    def forward(self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor],
                forced_forget: float | None = None):
        h_prev, c_prev = state
        z_in = torch.cat([x_t, h_prev])            # 拼成一个向量，一次矩阵乘

        z_f = torch.sigmoid(self.W_f(z_in))        # (0,1) 忘多少
        z_i = torch.sigmoid(self.W_i(z_in))        # (0,1) 写多少
        z = torch.tanh(self.W_z(z_in))             # (-1,1) 写什么
        z_o = torch.sigmoid(self.W_o(z_in))        # (0,1) 露多少

        if forced_forget is not None:              # 仅供 demo：把忘记门钉死
            z_f = torch.full_like(z_f, forced_forget)

        c_t = z_f * c_prev + z_i * z               # 加法，不是乘法
        h_t = z_o * torch.tanh(c_t)
        return h_t, c_t, {"f": z_f, "i": z_i, "z": z, "o": z_o}


class LSTMLayer(nn.Module):
    """一层 LSTM = 一份 cell 参数 + 沿时间轴循环 T 次。

    这里手写循环而不是用 nn.LSTM，因为 demo 要看每一步的四个门和 c^t。
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 3) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell = LSTMCell(input_dim, hidden_dim)
        self.W_y = nn.Linear(hidden_dim, hidden_dim)   # y^t（这里只做投影）

    def forward(self, xs, forced_forget: float | None = None) -> dict[str, torch.Tensor]:
        """xs: (T, input_dim) 的时间序列，返回每一步的 h/c/四个门/y。"""
        xs = torch.as_tensor(xs, dtype=torch.float32).reshape(-1, self.input_dim)
        h = torch.zeros(self.hidden_dim)               # h^0 = 0
        c = torch.zeros(self.hidden_dim)               # c^0 = 0

        rec: dict[str, list[torch.Tensor]] = {k: [] for k in ("h", "c", "f", "i", "z", "o", "y")}
        for x_t in xs:
            h, c, gates = self.cell(x_t, (h, c), forced_forget)
            rec["h"].append(h)
            rec["c"].append(c)
            rec["y"].append(self.W_y(h))
            for k in "fizo":
                rec[k].append(gates[k])

        return {k: torch.stack(v) for k, v in rec.items()}
