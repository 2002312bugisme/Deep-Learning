"""几种 divergence 的数值实现（1D 直方图版）+ GAN 的目标函数。

全部在同一个 bin 网格上算，所以可以互相比较。
"""

from __future__ import annotations

import math

import torch

LOG2 = math.log(2)
EPS = 1e-12


def normal_pmf(mu: float, sigma: float, edges: torch.Tensor) -> torch.Tensor:
    """高斯分布落在每个 bin 里的概率。"""
    cdf = 0.5 * (1.0 + torch.erf((edges - mu) / (sigma * math.sqrt(2.0))))
    p = (cdf[1:] - cdf[:-1]).clamp_min(0)
    return p / p.sum()


def kl(p: torch.Tensor, q: torch.Tensor) -> float:
    return (p * (p.clamp_min(EPS) / q.clamp_min(EPS)).log()).sum().item()


def js(p: torch.Tensor, q: torch.Tensor) -> float:
    """Jensen-Shannon：对称版 KL，取值 [0, log2]。"""
    m = 0.5 * (p + q)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def tv(p: torch.Tensor, q: torch.Tensor) -> float:
    """Total Variation：L1 的一半。"""
    return 0.5 * (p - q).abs().sum().item()


def hellinger(p: torch.Tensor, q: torch.Tensor) -> float:
    """Hellinger 距离（取平方和开方的形式）。"""
    m = 0.5 * (p + q)
    return (0.5 * ((p.sqrt() - m.sqrt()) ** 2).sum()
            + 0.5 * ((q.sqrt() - m.sqrt()) ** 2).sum()).sqrt().item()


def w1(p: torch.Tensor, q: torch.Tensor, dx: float) -> float:
    """1-Wasserstein：两条 CDF 之间围出的面积。一维有闭式解，不用解最优传输。"""
    return ((p.cumsum(0) - q.cumsum(0)).abs().sum() * dx).item()


def optimal_d(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """给定 P_data 和 P_G，让 V 最大的 D 是逐点的 p/(p+q)。"""
    return p / (p + q).clamp_min(EPS)


def v_value(p: torch.Tensor, q: torch.Tensor, d: torch.Tensor) -> float:
    """GAN 的目标函数 V = E_pdata[log D] + E_pG[log(1-D)]。"""
    return (p * d.clamp_min(EPS).log()
            + q * (1 - d).clamp_min(EPS).log()).sum().item()


def worst_d(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """反过来：让 V 最小的 D（用来展示「不是随便哪个 D 都行」）。"""
    return q / (p + q).clamp_min(EPS)
