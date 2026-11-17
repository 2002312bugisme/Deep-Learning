"""高效 self-attention 的五条路，全部可以互相对照。

约定：q, k, v 都是 (N, d) —— 一行一个 token。课件画的是转置 (d, N)，数学上等价。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ---------- 复杂度公式（只看乘加次数的大头） ----------

def flops_full(n: int, d: int, dv: int) -> float:
    """标准：A = QKᵀ 花 N²d，A·V 花 N²d'，合计 (d+d')N²。"""
    return (d + dv) * n * n


def flops_linear(n: int, d: int, dv: int) -> float:
    """换顺序：(KᵀV) 花 Ndd'，再乘 Q 花 Ndd'，合计 2dd'N。"""
    return 2 * d * dv * n


def flops_local(n: int, d: int, dv: int, w: int) -> float:
    """局部窗口：每行只看 2w+1 个，合计 (d+d')N(2w+1)。"""
    return (d + dv) * n * (2 * w + 1)


def flops_lowrank(n: int, d: int, dv: int, k: int) -> float:
    """低秩：投影 key/value 各 NKd / NKd'，再算 N×K 的注意力，合计 2NK(d+d')。"""
    return 2 * n * k * (d + dv)


# ---------- 五条路 ----------

def full(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """标准 self-attention，O(N²)。"""
    a = torch.softmax(q @ k.T / math.sqrt(q.shape[-1]), dim=-1)
    return a @ v, a


def dot_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """去掉 softmax 的版本：(QKᵀ)V。"""
    return (q @ k.T) @ v


def linear(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """去掉 softmax 后换顺序：Q(KᵀV)。和上面**数学上严格相等**。"""
    return q @ (k.T @ v)


def linear_softmax(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """想保留 softmax 只能近似：exp(q·k) ≈ φ(q)·φ(k)，φ(x) = elu(x) + 1。"""
    pq = F.elu(q) + 1.0
    pk = F.elu(k) + 1.0
    num = pq @ (pk.T @ v)                    # (N, d')
    den = pq @ pk.sum(dim=0)                 # (N,)
    return num / den.unsqueeze(-1)


def local(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, w: int = 64):
    """截断注意力：每个位置只看前后 w 个位置，其余置 0。"""
    n = q.shape[0]
    idx = torch.arange(n)
    keep = (idx[:, None] - idx[None, :]).abs() <= w
    s = (q @ k.T) / math.sqrt(q.shape[-1])
    s = s.masked_fill(~keep, float("-inf"))
    return torch.softmax(s, dim=-1) @ v, keep


def kmeans(x: torch.Tensor, n_clusters: int, iters: int = 8) -> torch.Tensor:
    """极简 k-means，够 demo 用。用等距点初始化，结果稳定。"""
    n = x.shape[0]
    centers = x[torch.linspace(0, n - 1, n_clusters).long()].clone()
    assign = torch.zeros(n, dtype=torch.long)
    for _ in range(iters):
        assign = torch.cdist(x, centers).argmin(dim=1)
        for c in range(n_clusters):
            m = assign == c
            if m.any():
                centers[c] = x[m].mean(dim=0)
    return assign


def clustered(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, n_clusters: int = 8):
    """聚类注意力：只算同簇的 q-k，跨簇权重恒为 0。"""
    assign = kmeans(torch.cat([q, k], dim=-1), n_clusters)
    keep = assign[:, None] == assign[None, :]
    s = (q @ k.T) / math.sqrt(q.shape[-1])
    s = s.masked_fill(~keep, float("-inf"))
    return torch.softmax(s, dim=-1) @ v, keep


# ---------- 低秩到底成不成立：直接看奇异值谱 ----------

def spectrum(a: torch.Tensor, top: int = 8) -> list[float]:
    """归一化奇异值谱。衰减快 = 低秩，衰减慢 = 满秩。"""
    s = torch.linalg.svdvals(a)
    return (s[:top] / s[0]).tolist()


def reconstruct_error(a: torch.Tensor, kk: int) -> float:
    """只保留前 k 个奇异值重构，返回相对误差。这就是 N×N → N×K 的误差下界。"""
    u, s, vt = torch.linalg.svd(a, full_matrices=False)
    rec = (u[:, :kk] * s[:kk]) @ vt[:kk]
    return (rec - a).norm().item() / a.norm().item()


def sparsity(mask: torch.Tensor) -> float:
    """mask 为 False（被置 0）的比例。"""
    return 1.0 - mask.float().mean().item()
