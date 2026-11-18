"""GAN 的两个网络，加一个 2D toy 分布。

Generator : z ~ N(0, I)  ->  一个样本（这里就是 2D 平面上的一个点）
Discriminator : 样本 -> 一个标量分数（越大越像真的）

损失用 BCEWithLogits，所以 D 输出的是 logit，不自己加 sigmoid。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sample_ring(n: int, radius: float = 1.0, noise: float = 0.05,
                generator: torch.Generator | None = None) -> torch.Tensor:
    """目标分布：半径 radius 的圆环。真实数据长这样。"""
    a = torch.rand(n, 1, generator=generator) * 2 * math.pi
    r = radius + torch.randn(n, 1, generator=generator) * noise
    return torch.cat([r * torch.cos(a), r * torch.sin(a)], dim=1)


def sample_bimodal(n: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """一维双峰：y = ±2 各占一半。用来演示 MSE 会输出均值 0。"""
    sign = torch.where(torch.rand(n, 1, generator=generator) < 0.5, -1.0, 1.0)
    return 2.0 * sign + torch.randn(n, 1, generator=generator) * 0.1


class Generator(nn.Module):
    """把低维 z 变成数据空间里的一个点。"""

    def __init__(self, z_dim: int = 2, out_dim: int = 2, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """给一个样本打分。输出是 logit，越大越真。"""

    def __init__(self, in_dim: int = 2, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def d_score(disc: Discriminator, x: torch.Tensor) -> float:
    """D 对一批样本的平均输出（sigmoid 之后的概率形式，便于理解）。"""
    return torch.sigmoid(disc(x)).mean().item()


def params_snapshot(module: nn.Module) -> list[torch.Tensor]:
    """拷一份参数，用来验证「谁没被更新」。"""
    return [p.detach().clone() for p in module.parameters()]


def params_changed(module: nn.Module, snap: list[torch.Tensor]) -> float:
    """和快照比，参数变化的最大绝对值。"""
    return max((p.detach() - s).abs().max().item()
               for p, s in zip(module.parameters(), snap))


def ascii_plot(real: torch.Tensor, fake: torch.Tensor,
               w: int = 58, h: int = 22, lim: float = 1.7) -> str:
    """把两组 2D 点画成字符图：o = 真数据，x = 生成的，@ = 重叠。"""
    grid = [["." for _ in range(w)] for _ in range(h)]

    def put(pts: torch.Tensor, ch: str) -> None:
        for x, y in pts.tolist():
            c = int((x + lim) / (2 * lim) * (w - 1))
            r = int((lim - y) / (2 * lim) * (h - 1))
            if 0 <= r < h and 0 <= c < w:
                grid[r][c] = ch if grid[r][c] in (".", ch) else "@"

    put(real, "o")
    put(fake, "x")
    return "\n".join("".join(row) for row in grid)
